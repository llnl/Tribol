// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/assembly/MfemPairAssembler.hpp"
#ifdef BUILD_REDECOMP
#include <limits>
#include <stdexcept>
#include <utility>

namespace tribol {
namespace {
/**
 * @brief Check a nonnegative array-size product before allocation.
 * @param count Number of entries or blocks.
 * @param size Values per entry or block.
 * @return Product representable by IndexT.
 * @throws std::length_error The product exceeds the supported index range.
 */
IndexT checkedProduct( IndexT count, IndexT size )
{
  if ( size != 0 && count > std::numeric_limits<IndexT>::max() / size ) {
    throw std::length_error( "Pair assembly storage exceeds the supported index range." );
  }
  return count * size;
}
}  // namespace

MfemPairAssembler::MfemPairAssembler( const MfemMeshData& mesh_data, const MfemJacobianData* jacobian_data,
                                      MeshData::Viewer mesh1, MeshData::Viewer mesh2, IndexT pair_capacity,
                                      bool assemble_residual )
    : mesh_data_( mesh_data ),
      connectivity_{ mesh1.getConnectivity(), mesh2.getConnectivity() },
      dimension_( mesh1.spatialDimension() )
{
  if ( pair_capacity < 0 || dimension_ <= 0 || mesh2.spatialDimension() != dimension_ ||
       mesh_data.GetRedecompResponse().FESpace()->GetVDim() != dimension_ || mesh1.numberOfNodesPerElement() <= 0 ||
       mesh2.numberOfNodesPerElement() <= 0 ) {
    throw std::invalid_argument(
        "Pair assembly requires matching spatial dimensions and positive element node counts." );
  }
  const IndexT dofs1 = checkedProduct( mesh1.numberOfNodesPerElement(), dimension_ );
  const IndexT dofs2 = checkedProduct( mesh2.numberOfNodesPerElement(), dimension_ );
  if ( dofs1 > std::numeric_limits<IndexT>::max() - dofs2 ) {
    throw std::length_error( "Pair DOF count exceeds the supported index range." );
  }
  dof_offsets_ = MdArray<IndexT, 3>{ 0, dofs1, dofs1 + dofs2 };
  const auto& redecomp_fes = *mesh_data.GetRedecompResponse().FESpace();
  if ( assemble_residual ) {
    redecomp_residual_.SetSpace( const_cast<mfem::FiniteElementSpace*>( &redecomp_fes ) );
    redecomp_residual_ = 0.0;
  }
  if ( !jacobian_data ) {
    return;
  }
  const std::array<const Array1D<int>*, 2> element_maps{ &mesh_data.GetElemMap1(), &mesh_data.GetElemMap2() };
  constexpr int pair_size = 2;
  std::vector<MfemMatrixBlock> blocks;
  blocks.reserve( pair_size * pair_size );
  for ( int i = 0; i < pair_size; ++i ) {
    for ( int j = 0; j < pair_size; ++j ) {
      blocks.push_back( { i, j, connectivity_[i].shape()[1], connectivity_[j].shape()[1], dof_offsets_[i],
                          dof_offsets_[j], *element_maps[i], *element_maps[j] } );
    }
  }
  matrix_ = std::make_unique<MfemMatrixAssembler>( MfemMatrixSpace{ jacobian_data->ParentPath(), redecomp_fes },
                                                   MfemMatrixSpace{ jacobian_data->ParentPath(), redecomp_fes },
                                                   std::move( blocks ), pair_capacity );
}

void MfemPairAssembler::addPair( InterfacePair pair, ArrayViewT<const RealT> residual,
                                 ArrayViewT<const RealT, 2> jacobian )
{
  if ( !matrix_ ) {
    throw std::logic_error( "Jacobian assembly was not enabled for this pair assembler." );
  }
  if ( !redecomp_residual_.FESpace() ) {
    throw std::logic_error( "Residual assembly was not enabled for this pair assembler." );
  }
  if ( residual.size() != dof_offsets_[2] || jacobian.shape()[0] != dof_offsets_[2] ||
       jacobian.shape()[1] != dof_offsets_[2] ) {
    throw std::invalid_argument( "Pair residual and Jacobian must match the two elements' displacement DOFs." );
  }
  const MdArray<IndexT, 2> elements{ pair.element_id1_, pair.element_id2_ };
  const int scalar_size = redecomp_residual_.FESpace()->GetNDofs();
  constexpr int pair_size = 2;
  for ( int i = 0; i < pair_size; ++i ) {
    const IndexT row_nodes = connectivity_[i].shape()[1];
    for ( IndexT node = 0; node < row_nodes; ++node ) {
      for ( int component = 0; component < dimension_; ++component ) {
        redecomp_residual_( connectivity_[i]( elements[i], node ) + component * scalar_size ) +=
            residual[dof_offsets_[i] + node * dimension_ + component];
      }
    }
  }
  addPairJacobian( pair, jacobian );
}

void MfemPairAssembler::addPairJacobian( InterfacePair pair, ArrayViewT<const RealT, 2> jacobian )
{
  if ( !matrix_ ) {
    throw std::logic_error( "Jacobian assembly was not enabled for this pair assembler." );
  }
  matrix_->addPair( pair, jacobian );
}

ContactFormulationResult MfemPairAssembler::assemble() const
{
  if ( !redecomp_residual_.FESpace() ) {
    throw std::logic_error( "Residual assembly was not enabled for this pair assembler." );
  }
  auto* parent_fes = mesh_data_.GetParentCoords().ParFESpace();
  shared::ParVector residual( const_cast<mfem::ParFiniteElementSpace*>( parent_fes ) );
  residual.fill( 0.0 );
  mfem::Vector parent_residual( parent_fes->GetVSize() );
  parent_residual = 0.0;
  mesh_data_.GetParentRedecompTransfer().RedecompToParent( redecomp_residual_, parent_residual );
  parent_fes->GetProlongationMatrix()->MultTranspose( parent_residual, residual.get() );
  ContactFormulationResult result;
  result.force.reset( residual.release() );
  if ( matrix_ ) {
    result.df_dx.reset( assembleJacobian().release() );
  }
  return result;
}

shared::ParSparseMat MfemPairAssembler::assembleJacobian() const
{
  if ( !matrix_ ) {
    throw std::logic_error( "Jacobian assembly was not enabled for this pair assembler." );
  }
  return matrix_->assemble();
}

}  // namespace tribol
#endif  // BUILD_REDECOMP
