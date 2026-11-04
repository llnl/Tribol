// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MfemNodalAssembler.hpp"
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
    throw std::length_error( "Nodal assembly storage exceeds the supported index range." );
  }
  return count * size;
}
}  // namespace

MfemNodalAssembler::MfemNodalAssembler( const MfemMeshData& mesh_data, const MfemSubmeshData& submesh_data,
                                        const MfemJacobianData& jacobian_data, MeshData::Viewer mesh1,
                                        MeshData::Viewer mesh2, IndexT pair_capacity )
    : mesh_data_( mesh_data ),
      submesh_data_( submesh_data ),
      jacobian_data_( jacobian_data ),
      connectivity_{ mesh1.getConnectivity(), mesh2.getConnectivity() },
      dimension_( mesh1.spatialDimension() ),
      nodes1_( mesh1.numberOfNodesPerElement() ),
      redecomp_gap_( const_cast<mfem::FiniteElementSpace*>( submesh_data.GetRedecompGap().FESpace() ) ),
      redecomp_area_( const_cast<mfem::FiniteElementSpace*>( submesh_data.GetRedecompGap().FESpace() ) ),
      pair_assembler_( mesh_data, jacobian_data, mesh1, mesh2, pair_capacity )
{
  if ( redecomp_gap_.FESpace()->GetVDim() != 1 ) {
    throw std::invalid_argument( "Nodal assembly requires a scalar Redecomp dual space." );
  }
  const IndexT dofs1 = checkedProduct( nodes1_, dimension_ );
  const IndexT dofs2 = checkedProduct( mesh2.numberOfNodesPerElement(), dimension_ );
  dof_offsets_ = MdArray<IndexT, 3>{ 0, dofs1, dofs1 + dofs2 };

  const bool use_lor = ( mesh_data.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes = use_lor ? *mesh_data.GetLORMeshFESpace() : mesh_data.GetSubmeshFESpace();
  const auto& pressure_surface_fes = use_lor ? *submesh_data.GetLORMeshFESpace() : submesh_data.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *mesh_data.GetRedecompResponse().FESpace();
  const auto& pressure_redecomp_fes = *redecomp_gap_.FESpace();
  const std::array<const Array1D<int>*, 2> element_maps{ &mesh_data.GetElemMap1(), &mesh_data.GetElemMap2() };

  constexpr int pair_size = 2;
  gap_gradient_contributions_.reserve( pair_size );
  area_gradient_contributions_.reserve( pair_size );
  for ( int j = 0; j < pair_size; ++j ) {
    const IndexT column_dofs = dof_offsets_[j + 1] - dof_offsets_[j];
    const IndexT block_size = checkedProduct( nodes1_, column_dofs );
    gap_gradient_contributions_.emplace_back( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                              displacement_redecomp_fes, *element_maps[0], *element_maps[j] );
    gap_gradient_contributions_.back().reserve( pair_capacity, checkedProduct( pair_capacity, block_size ) );
    area_gradient_contributions_.emplace_back( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                               displacement_redecomp_fes, *element_maps[0], *element_maps[j] );
    area_gradient_contributions_.back().reserve( pair_capacity, checkedProduct( pair_capacity, block_size ) );
  }

  const IndexT max_column_dofs = std::max( dofs1, dofs2 );
  gradient_block_.resize( checkedProduct( nodes1_, max_column_dofs ) );
  redecomp_gap_ = 0.0;
  redecomp_area_ = 0.0;
}

void MfemNodalAssembler::appendGradientBlocks( InterfacePair pair, ArrayViewT<const RealT, 2> gradients,
                                               std::vector<PackedPairJacobianContribs>& contributions )
{
  const MdArray<IndexT, 2> elements{ pair.element_id1_, pair.element_id2_ };
  constexpr int pair_size = 2;
  for ( int j = 0; j < pair_size; ++j ) {
    const IndexT column_nodes = connectivity_[j].shape()[1];
    const IndexT column_dofs = dof_offsets_[j + 1] - dof_offsets_[j];
    for ( IndexT column = 0; column < column_dofs; ++column ) {
      const IndexT pair_column = dof_offsets_[j] + ( column % column_nodes ) * dimension_ + column / column_nodes;
      for ( IndexT row = 0; row < nodes1_; ++row ) {
        gradient_block_[row + column * nodes1_] = gradients( row, pair_column );
      }
    }
    contributions[j].append( elements[0], elements[j], gradient_block_.data(), checkedProduct( nodes1_, column_dofs ) );
  }
}

void MfemNodalAssembler::addPairQuantities( InterfacePair pair, ArrayViewT<const RealT> weighted_gaps,
                                            ArrayViewT<const RealT> areas, ArrayViewT<const RealT, 2> gap_gradients,
                                            ArrayViewT<const RealT, 2> area_gradients )
{
  if ( weighted_gaps.size() != nodes1_ || areas.size() != nodes1_ || gap_gradients.shape()[0] != nodes1_ ||
       gap_gradients.shape()[1] != dof_offsets_[2] || area_gradients.shape()[0] != nodes1_ ||
       area_gradients.shape()[1] != dof_offsets_[2] ) {
    throw std::invalid_argument(
        "Nodal quantities and gradients must match mesh1's node count and the pair displacement DOFs." );
  }
  for ( IndexT node = 0; node < nodes1_; ++node ) {
    const IndexT redecomp_dof = connectivity_[0]( pair.element_id1_, node );
    redecomp_gap_( redecomp_dof ) += weighted_gaps[node];
    redecomp_area_( redecomp_dof ) += areas[node];
  }
  appendGradientBlocks( pair, gap_gradients, gap_gradient_contributions_ );
  appendGradientBlocks( pair, area_gradients, area_gradient_contributions_ );
}

shared::ParVector MfemNodalAssembler::assembleSubmeshVector( const mfem::GridFunction& redecomp_field ) const
{
  auto* submesh_fes = const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() );
  mfem::ParLinearForm linear_form( submesh_fes );
  submesh_data_.GetPressureTransfer().RedecompToSubmesh( redecomp_field, linear_form );
  shared::ParVector result( submesh_fes );
  result.fill( 0.0 );
  submesh_fes->GetProlongationMatrix()->MultTranspose( linear_form, result.get() );
  return result;
}

MfemNodalQuantityData MfemNodalAssembler::assembleQuantities( bool clamp_positive_gaps, RealT area_tolerance ) const
{
  MfemNodalQuantityData data;
  data.weighted_gap = assembleSubmeshVector( redecomp_gap_ );
  data.area = assembleSubmeshVector( redecomp_area_ );

  mfem::Array<int> rows_to_eliminate;
  if ( clamp_positive_gaps ) {
    rows_to_eliminate.Reserve( data.weighted_gap.size() );
    for ( int i = 0; i < data.weighted_gap.size(); ++i ) {
      if ( data.weighted_gap[i] > 0.0 ) {
        data.weighted_gap[i] = 0.0;
        rows_to_eliminate.push_back( i );
      }
    }
  }

  data.normalized_gap = data.weighted_gap.divide( data.area, area_tolerance );

  const auto* submesh_fes = &submesh_data_.GetSubmeshFESpace();
  const auto* parent_fes = mesh_data_.GetParentCoords().ParFESpace();
  data.weighted_gap_jacobian = jacobian_data_.GetMfemJacobian( submesh_fes, parent_fes, gap_gradient_contributions_ );
  if ( clamp_positive_gaps ) {
    data.weighted_gap_jacobian.eliminateRows( rows_to_eliminate );
  }
  data.area_jacobian = jacobian_data_.GetMfemJacobian( submesh_fes, parent_fes, area_gradient_contributions_ );
  return data;
}

shared::ParVector MfemNodalAssembler::getSubmeshPressureTrueDofs() const
{
  shared::ParVector pressure( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetSubmeshPressure().GetTrueDofs( pressure.get() );
  return pressure;
}

mfem::GridFunction MfemNodalAssembler::transferToRedecomp( const mfem::Vector& submesh_true_dofs ) const
{
  auto* submesh_fes = const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() );
  if ( submesh_true_dofs.Size() != submesh_fes->GetTrueVSize() ) {
    throw std::invalid_argument( "Submesh vector length must match the dual submesh true-DOF count." );
  }
  mfem::GridFunction redecomp_field( submesh_data_.GetRedecompGap() );
  redecomp_field = 0.0;
  mfem::ParGridFunction submesh_field( submesh_fes );
  submesh_field.SetFromTrueDofs( submesh_true_dofs );
  submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_field, redecomp_field );
  return redecomp_field;
}

shared::ParSparseMat MfemNodalAssembler::makeSubmeshDiagonalMatrix( const mfem::Vector& diagonal_true_dofs ) const
{
  const auto& submesh_fes = submesh_data_.GetSubmeshFESpace();
  if ( diagonal_true_dofs.Size() != submesh_fes.GetTrueVSize() ) {
    throw std::invalid_argument( "Diagonal vector length must match the dual submesh true-DOF count." );
  }
  return shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                               submesh_fes.GetTrueDofOffsets(), diagonal_true_dofs );
}

shared::ParSparseMat MfemNodalAssembler::assembleHessians() const { return pair_assembler_.assembleJacobian(); }

}  // namespace tribol
#endif  // BUILD_REDECOMP
