// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Quadrature-point Energy Mortar penalty energy defined independently of its derivatives.
 */
#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTARENERGY_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTARENERGY_HPP_

#include "tribol/geom/LinearEdgeGeometry.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/** @brief Primal pair energy, activation, and quadrature status. */
struct EnergyMortarEnergyValue {
  RealT energy{ 0.0 };                   ///< Integrated penalty energy; discard when quadrature fails.
  bool has_active_qp{ false };           ///< Whether any nonpositive gap uses the compressive branch.
  QuadratureResult quadrature_result{};  ///< Generation status and point count; valid empty rules give zero energy.
};

/**
 * @brief Define the quadrature-point penalty energy using current edge geometry.
 *
 * Mesh1 is the non-mortar integration side. The gap uses mesh2's current directed normal and the opposing-normal
 * alignment correction. Physical weights come from the pair rule and are applied once. The definition owns only
 * fixed inputs; each call regenerates quadrature from the supplied active coordinates.
 * @tparam QuadratureRule Linear two-dimensional edge-pair rule.
 */
template <typename QuadratureRule>
class EnergyMortarEnergy {
 public:
  static constexpr IndexT dof_count =
      2 * LinearEdgeGeometry::vertex_count * LinearEdgeGeometry::spatial_dimension;  ///< Pair coordinate DOFs.

  /**
   * @brief Copy the integration rule and fixed penalty inputs.
   * @param quadrature_rule Pair quadrature generator.
   * @param penalty Penalty stiffness in the existing Energy Mortar energy convention.
   * @param residual_gap Gap offset held fixed during differentiation.
   */
  TRIBOL_HOST_DEVICE EnergyMortarEnergy( QuadratureRule quadrature_rule, RealT penalty, RealT residual_gap )
      : quadrature_rule_( quadrature_rule ), penalty_( penalty ), residual_gap_( residual_gap )
  {
  }

  /**
   * @brief Calculate energy and activation without consuming failed quadrature.
   * @param coordinates Endpoints ordered by element, node, and spatial component.
   * @return Energy, activation, and status. Failed geometry/storage gives zero energy with failure status; valid
   * empty rules give successful zero energy. At zero gap, energy is zero and the compressive branch remains active.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE EnergyMortarEnergyValue
  evaluate( MdSpan<const RealT, dof_count> coordinates ) const
  {
    constexpr IndexT edge_dofs = dof_count / 2;
    const LinearEdgeGeometry edge1( { coordinates.data(), edge_dofs } );
    const LinearEdgeGeometry edge2( { coordinates.data() + edge_dofs, edge_dofs } );
    const auto generated = quadrature_rule_.generate( edge1, edge2 );
    EnergyMortarEnergyValue result{};
    result.quadrature_result = { generated.result.status, generated.result.required_point_count };
    if ( generated.result.status != QuadratureStatus::Success || generated.result.required_point_count == 0 ) {
      return result;
    }
    const auto normal1 = edge1.normal(), normal2 = edge2.normal();
    const RealT cosine = normal1.dot( normal2 );
    const RealT alignment = cosine < 0.0 ? cosine : 0.0;
    const auto points = generated.getView();
    for ( IndexT q = 0; q < generated.result.required_point_count; ++q ) {
      const auto point1 = edge1.position( points.geometry1_coords( q, 0 ) );
      const auto point2 = edge2.position( points.geometry2_coords( q, 0 ) );
      const RealT gap = LinearEdgeGeometry::Vector( point1, point2 ).dot( normal2 ) * alignment - residual_gap_;
      if ( gap > 0.0 ) continue;
      result.has_active_qp = true;
      result.energy += 0.5 * penalty_ * gap * gap * points.weights[q];
    }
    return result;
  }

  /**
   * @brief Return the scalar energy consumed by an energy pair evaluator.
   * @param coordinates Current pair coordinates.
   * @return Integrated penalty energy, including zero for valid empty or inactive rules.
   * @throws std::invalid_argument Pair geometry is invalid.
   * @throws std::length_error Quadrature capacity is insufficient.
   */
  TRIBOL_ENZYME_INLINE RealT energy( MdSpan<const RealT, dof_count> coordinates ) const
  {
    const auto value = evaluate( coordinates );
    if ( value.quadrature_result.status != QuadratureStatus::Success )
      reportQuadratureFailure( value.quadrature_result.status );
    return value.energy;
  }

 private:
  QuadratureRule quadrature_rule_;  ///< Owned pair quadrature configuration.
  RealT penalty_;                   ///< Inactive penalty stiffness.
  RealT residual_gap_;              ///< Inactive gap offset.
};

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_ENERGYMORTARENERGY_HPP_
