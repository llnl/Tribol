// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MfemPairAssembler.hpp"
#ifdef BUILD_REDECOMP
#include <algorithm>
#include <limits>
#include <stdexcept>

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

MfemPairAssembler::MfemPairAssembler( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                                      MeshData::Viewer mesh1, MeshData::Viewer mesh2, IndexT pair_capacity )
    : mesh_data_( mesh_data ),
      jacobian_data_( jacobian_data ),
      connectivity_{ mesh1.getConnectivity(), mesh2.getConnectivity() },
      dimension_( mesh1.spatialDimension() ),
      redecomp_residual_( const_cast<mfem::FiniteElementSpace*>( mesh_data.GetRedecompResponse().FESpace() ) )
{
  if ( pair_capacity < 0 || dimension_ <= 0 || mesh2.spatialDimension() != dimension_ ||
       redecomp_residual_.FESpace()->GetVDim() != dimension_ || mesh1.numberOfNodesPerElement() <= 0 ||
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
  const auto& surface_fes = mesh_data.GetLORMesh() ? *mesh_data.GetLORMeshFESpace() : mesh_data.GetSubmeshFESpace();
  const auto& redecomp_fes = *redecomp_residual_.FESpace();
  const std::array<const Array1D<int>*, 2> element_maps{ &mesh_data.GetElemMap1(), &mesh_data.GetElemMap2() };
  constexpr int pair_size = 2;
  jacobian_contributions_.reserve( pair_size * pair_size );
  for ( int i = 0; i < pair_size; ++i ) {
    const IndexT row_dofs = dof_offsets_[i + 1] - dof_offsets_[i];
    for ( int j = 0; j < pair_size; ++j ) {
      const IndexT column_dofs = dof_offsets_[j + 1] - dof_offsets_[j];
      jacobian_contributions_.emplace_back( surface_fes, surface_fes, redecomp_fes, redecomp_fes, *element_maps[i],
                                            *element_maps[j] );
      jacobian_contributions_.back().reserve(
          pair_capacity, checkedProduct( pair_capacity, checkedProduct( row_dofs, column_dofs ) ) );
    }
  }
  const IndexT max_element_dofs = std::max( dofs1, dofs2 );
  block_.resize( checkedProduct( max_element_dofs, max_element_dofs ) );
  redecomp_residual_ = 0.0;
}

void MfemPairAssembler::addPair( InterfacePair pair, ArrayViewT<const RealT> residual,
                                 ArrayViewT<const RealT, 2> jacobian )
{
  if ( residual.size() != dof_offsets_[2] || jacobian.shape()[0] != dof_offsets_[2] ||
       jacobian.shape()[1] != dof_offsets_[2] ) {
    throw std::invalid_argument( "Pair residual and Jacobian must match the two elements' displacement DOFs." );
  }
  const MdArray<IndexT, 2> elements{ pair.element_id1_, pair.element_id2_ };
  const int scalar_size = redecomp_residual_.FESpace()->GetNDofs();
  constexpr int pair_size = 2;
  for ( int i = 0; i < pair_size; ++i ) {
    const IndexT row_nodes = connectivity_[i].shape()[1];
    const IndexT row_dofs = dof_offsets_[i + 1] - dof_offsets_[i];
    for ( IndexT node = 0; node < row_nodes; ++node ) {
      for ( int component = 0; component < dimension_; ++component ) {
        redecomp_residual_( connectivity_[i]( elements[i], node ) + component * scalar_size ) +=
            residual[dof_offsets_[i] + node * dimension_ + component];
      }
    }
    for ( int j = 0; j < pair_size; ++j ) {
      const IndexT column_nodes = connectivity_[j].shape()[1];
      const IndexT column_dofs = dof_offsets_[j + 1] - dof_offsets_[j];
      for ( IndexT column = 0; column < column_dofs; ++column ) {
        const IndexT pair_column = dof_offsets_[j] + ( column % column_nodes ) * dimension_ + column / column_nodes;
        for ( IndexT row = 0; row < row_dofs; ++row ) {
          const IndexT pair_row = dof_offsets_[i] + ( row % row_nodes ) * dimension_ + row / row_nodes;
          block_[row + column * row_dofs] = jacobian( pair_row, pair_column );
        }
      }
      jacobian_contributions_[i * pair_size + j].append( elements[i], elements[j], block_.data(),
                                                         checkedProduct( row_dofs, column_dofs ) );
    }
  }
}

ContactFormulationResult MfemPairAssembler::assemble() const
{
  auto* parent_fes = mesh_data_.GetParentCoords().ParFESpace();
  shared::ParVector residual( const_cast<mfem::ParFiniteElementSpace*>( parent_fes ) );
  residual.fill( 0.0 );
  mfem::Vector parent_residual( parent_fes->GetVSize() );
  parent_residual = 0.0;
  mesh_data_.GetParentRedecompTransfer().RedecompToParent( redecomp_residual_, parent_residual );
  parent_fes->GetProlongationMatrix()->MultTranspose( parent_residual, residual.get() );
  auto jacobian = jacobian_data_.GetMfemJacobian( parent_fes, parent_fes, jacobian_contributions_ );
  ContactFormulationResult result;
  result.force.reset( residual.release() );
  result.df_dx.reset( jacobian.release() );
  return result;
}

}  // namespace tribol
#endif  // BUILD_REDECOMP
