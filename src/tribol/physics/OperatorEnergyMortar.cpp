// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/OperatorEnergyMortar.hpp"

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )

#include <algorithm>

#include "axom/slic.hpp"

#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/MfemContactData.hpp"

namespace tribol {

OperatorEnergyMortar::OperatorEnergyMortar( CouplingScheme& coupling_scheme, double penalty_stiffness,
                                            double smoothing_length, int quadrature_order, bool enzyme_quadrature,
                                            RealT residual_gap )
    : coupling_scheme_{ coupling_scheme }
{
  parameters_.k = penalty_stiffness;
  parameters_.del = smoothing_length;
  parameters_.N = quadrature_order;
  parameters_.enzyme_quadrature = enzyme_quadrature;
  parameters_.residual_gap = residual_gap;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( parameters_ );
}

void OperatorEnergyMortar::setInterfacePairs( ArrayT<InterfacePair>&& pairs, [[maybe_unused]] int check_level )
{
  pairs_ = std::move( pairs );
}

void OperatorEnergyMortar::updateMeshes( MeshData& first_surface, MeshData& second_surface )
{
  // EnergyMortar integrates on the nonmortar surface, which is Tribol's
  // second surface. Keep these names explicit so pair flipping stays visible.
  nonmortar_mesh_ = &second_surface;
  mortar_mesh_ = &first_surface;
}

void OperatorEnergyMortar::updateConstantPenaltyStiffness( double first_penalty, double second_penalty )
{
  parameters_.k = 0.5 * ( first_penalty + second_penalty );
  evaluator_ = std::make_unique<EnergyMortarCalculator>( parameters_ );
}

void OperatorEnergyMortar::setResidualGap( RealT residual_gap )
{
  parameters_.residual_gap = residual_gap;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( parameters_ );
}

void OperatorEnergyMortar::updateNodalForces()
{
  SLIC_ERROR_ROOT_IF( nonmortar_mesh_ == nullptr || mortar_mesh_ == nullptr,
                      "Operator EnergyMortar requires current Tribol patch meshes." );
  auto* contact_data = coupling_scheme_.getMfemContactData();
  auto* method_data = coupling_scheme_.getMethodData();
  SLIC_ERROR_ROOT_IF( contact_data == nullptr || method_data == nullptr,
                      "Operator EnergyMortar requires contact-patch and Jacobian storage." );
  SLIC_ERROR_ROOT_IF( !coupling_scheme_.getEnforcementOptions().penalty_options.kinematic_calc_set,
                      "Operator EnergyMortar requires kinematic penalty parameters before update()." );

  method_data->reserveBlockJ( { BlockSpace::MORTAR, BlockSpace::NONMORTAR }, pairs_.size() );
  auto nonmortar_view = nonmortar_mesh_->getView();
  auto mortar_view = mortar_mesh_->getView();
  auto& mortar_patch = contact_data->SurfacePatch( 0 );
  auto& nonmortar_patch = contact_data->SurfacePatch( 1 );
  constexpr int spatial_dimension = 2;
  constexpr int nodes_per_element = 2;
  constexpr int element_vector_dofs = spatial_dimension * nodes_per_element;
  constexpr int pair_vector_dofs = 2 * element_vector_dofs;
  static_assert( pair_vector_dofs == QuadraturePointPenaltyData::num_force_dofs );
  energy_ = 0.0;

  for ( const auto& pair : pairs_ ) {
    // The EnergyMortar calculator follows the literature convention with the
    // nonmortar element first, opposite Tribol's pair ordering.
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto quadrature_data =
        evaluator_->compute_quadrature_point_penalty_data( flipped_pair, nonmortar_view, mortar_view );
    if ( !quadrature_data.has_active_qp ) continue;

    energy_ += quadrature_data.energy;
    const int mortar_element = static_cast<int>( pair.m_element_id1 );
    const int nonmortar_element = static_cast<int>( pair.m_element_id2 );
    for ( int node = 0; node < nodes_per_element; ++node ) {
      const int nonmortar_node =
          static_cast<int>( nonmortar_patch.connectivity[nonmortar_element * nodes_per_element + node] );
      const int mortar_node = static_cast<int>( mortar_patch.connectivity[mortar_element * nodes_per_element + node] );
      for ( int dimension = 0; dimension < spatial_dimension; ++dimension ) {
        nonmortar_patch.response[dimension][nonmortar_node] +=
            quadrature_data.force[node * spatial_dimension + dimension];
        mortar_patch.response[dimension][mortar_node] +=
            quadrature_data.force[element_vector_dofs + node * spatial_dimension + dimension];
      }
    }

    SurfaceContactElem element_matrix( spatial_dimension, nullptr, nullptr, nullptr, nodes_per_element, 0, nullptr,
                                       nullptr, mortar_element, nonmortar_element );
    element_matrix.allocateBlockJ( PENALTY );
    for ( int row_surface = 0; row_surface < 2; ++row_surface ) {
      for ( int column_surface = 0; column_surface < 2; ++column_surface ) {
        const BlockSpace row_space = row_surface == 0 ? BlockSpace::NONMORTAR : BlockSpace::MORTAR;
        const BlockSpace column_space = column_surface == 0 ? BlockSpace::NONMORTAR : BlockSpace::MORTAR;
        auto& block = element_matrix.blockJ( static_cast<int>( row_space ), static_cast<int>( column_space ) );
        for ( int column = 0; column < element_vector_dofs; ++column ) {
          for ( int row = 0; row < element_vector_dofs; ++row ) {
            // EnergyMortar and MethodData both use node-major element vector
            // DOFs here. The patch assembler converts these blocks to H's
            // component-major nodal layout during global composition.
            const int calculator_row = row + row_surface * element_vector_dofs;
            const int calculator_column = column + column_surface * element_vector_dofs;
            block( row, column ) = quadrature_data.stiffness[calculator_row + calculator_column * pair_vector_dofs];
          }
        }
      }
    }
    method_data->storeElemBlockJ( { mortar_element, nonmortar_element }, element_matrix.blockJ );
  }
}

}  // namespace tribol

#endif
