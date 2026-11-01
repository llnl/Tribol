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
  pair_jacobian_ = ArrayT<RealT, 2, MemorySpace::Host>( dof_offsets_[2], dof_offsets_[2] );
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

void MfemNodalAssembler::addPairHessian( InterfacePair pair, ArrayViewT<const RealT, 2> pair_hessian )
{
  pair_assembler_.addPairJacobian( pair, pair_hessian );
}

void MfemNodalAssembler::addPairHessians( InterfacePair pair, ArrayViewT<const RealT> gap_weights,
                                          ArrayViewT<const RealT, 3> gap_hessians, ArrayViewT<const RealT> area_weights,
                                          ArrayViewT<const RealT, 3> area_hessians )
{
  const IndexT pair_dofs = dof_offsets_[2];
  const bool include_area = area_weights.size() > 0;
  if ( gap_weights.size() != nodes1_ || gap_hessians.shape()[0] != nodes1_ || gap_hessians.shape()[1] != pair_dofs ||
       gap_hessians.shape()[2] != pair_dofs ||
       ( include_area && ( area_weights.size() != nodes1_ || area_hessians.shape()[0] != nodes1_ ||
                           area_hessians.shape()[1] != pair_dofs || area_hessians.shape()[2] != pair_dofs ) ) ) {
    throw std::invalid_argument(
        "Nodal Hessian weights and arrays must match mesh1's node count and the pair displacement DOFs." );
  }

  pair_jacobian_.fill( 0.0 );
  for ( IndexT node = 0; node < nodes1_; ++node ) {
    const RealT gap_weight = gap_weights[node];
    for ( IndexT row = 0; row < pair_dofs; ++row ) {
      for ( IndexT column = 0; column < pair_dofs; ++column ) {
        pair_jacobian_( row, column ) += gap_weight * gap_hessians( node, row, column );
      }
    }
  }
  if ( include_area ) {
    for ( IndexT node = 0; node < nodes1_; ++node ) {
      const RealT area_weight = area_weights[node];
      for ( IndexT row = 0; row < pair_dofs; ++row ) {
        for ( IndexT column = 0; column < pair_dofs; ++column ) {
          pair_jacobian_( row, column ) += area_weight * area_hessians( node, row, column );
        }
      }
    }
  }
  pair_assembler_.addPairJacobian( pair, pair_jacobian_.view() );
}

shared::ParSparseMat MfemNodalAssembler::assembleHessians() const { return pair_assembler_.assembleJacobian(); }

MfemNodalDerivativeData MfemNodalAssembler::makeDerivativeData( bool has_area_derivatives,
                                                                bool has_outer_hessian ) const
{
  auto* submesh_fes = const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() );
  MfemNodalDerivativeData data;
  data.has_area_derivatives = has_area_derivatives;
  data.has_outer_hessian = has_outer_hessian;
  data.pressure = shared::ParVector( submesh_fes );
  data.pressure.fill( 0.0 );
  data.d_energy_d_gap = shared::ParVector( submesh_fes );
  data.d_energy_d_gap.fill( 0.0 );
  if ( has_area_derivatives ) {
    data.d_energy_d_area = shared::ParVector( submesh_fes );
    data.d_energy_d_area.fill( 0.0 );
  }
  if ( has_outer_hessian ) {
    data.d2_energy_dg2 = shared::ParVector( submesh_fes );
    data.d2_energy_dg2.fill( 0.0 );
    if ( has_area_derivatives ) {
      data.d2_energy_dg_da = shared::ParVector( submesh_fes );
      data.d2_energy_dg_da.fill( 0.0 );
      data.d2_energy_da2 = shared::ParVector( submesh_fes );
      data.d2_energy_da2.fill( 0.0 );
    }
  }
  return data;
}

shared::ParVector MfemNodalAssembler::assembleForce( const MfemNodalQuantityData& quantities,
                                                     const MfemNodalDerivativeData& derivatives ) const
{
  const int dual_size = submesh_data_.GetSubmeshFESpace().GetTrueVSize();
  if ( derivatives.d_energy_d_gap.size() != dual_size ||
       ( derivatives.has_area_derivatives && derivatives.d_energy_d_area.size() != dual_size ) ) {
    throw std::invalid_argument( "Nodal first-derivative vector lengths must match the dual submesh true-DOF count." );
  }
  auto force = derivatives.d_energy_d_gap * quantities.weighted_gap_jacobian;
  if ( derivatives.has_area_derivatives ) {
    force += derivatives.d_energy_d_area * quantities.area_jacobian;
  }
  return force;
}

shared::ParSparseMat MfemNodalAssembler::assembleJacobian( const MfemNodalQuantityData& quantities,
                                                           const MfemNodalDerivativeData& derivatives ) const
{
  const int dual_size = submesh_data_.GetSubmeshFESpace().GetTrueVSize();
  if ( derivatives.has_outer_hessian &&
       ( derivatives.d2_energy_dg2.size() != dual_size ||
         ( derivatives.has_area_derivatives &&
           ( derivatives.d2_energy_dg_da.size() != dual_size || derivatives.d2_energy_da2.size() != dual_size ) ) ) ) {
    throw std::invalid_argument( "Nodal second-derivative vector lengths must match the dual submesh true-DOF count." );
  }
  auto df_dx = assembleHessians();
  if ( derivatives.has_outer_hessian ) {
    auto d2_dg2_diag = makeSubmeshDiagonalMatrix( derivatives.d2_energy_dg2.get() );
    df_dx += shared::ParSparseMat::rap( d2_dg2_diag, quantities.weighted_gap_jacobian );
    if ( derivatives.has_area_derivatives ) {
      auto d2_dg_da_diag = makeSubmeshDiagonalMatrix( derivatives.d2_energy_dg_da.get() );
      auto d2_da2_diag = makeSubmeshDiagonalMatrix( derivatives.d2_energy_da2.get() );
      df_dx += shared::ParSparseMat::rap( quantities.weighted_gap_jacobian, d2_dg_da_diag, quantities.area_jacobian );
      df_dx += shared::ParSparseMat::rap( quantities.area_jacobian, d2_dg_da_diag, quantities.weighted_gap_jacobian );
      df_dx += shared::ParSparseMat::rap( d2_da2_diag, quantities.area_jacobian );
    }
  }
  return df_dx;
}

}  // namespace tribol
#endif  // BUILD_REDECOMP
