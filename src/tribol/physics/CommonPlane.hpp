// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_

#include "Physics.hpp"
#include "tribol/physics/PenaltyStiffness.hpp"
#include "tribol/physics/PairEvaluation.hpp"
#include "tribol/geom/CommonPlaneEdgeMapping.hpp"
#include "tribol/geom/CommonPlaneGap.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/integ/PairQuadratureRule.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/BasisFunction.hpp"

namespace tribol {

/** @brief Pair quadrature using physical distance on a common line. @tparam BaseRule Reference Gauss rule. */
template <typename BaseRule>
using CommonPlaneQuadratureRule = PairQuadratureRule<CommonPlaneEdgeMapping, BaseRule>;

/** @brief Residual and generation status for a pair of linear two-dimensional edges. */
using CommonPlaneResidualData = PairResidualValue<2 * LinearEdgeGeometry::Coords::size()>;

/** @brief Pair residual and its derivative in node-major pair-coordinate order. */
struct CommonPlanePairData : CommonPlaneResidualData {
  MdArray<RealT, coordinate_count, coordinate_count> jacobian{};  ///< Rows are residual entries, columns coordinates.
};

/**
 * @brief Configure Common Plane with an owned geometric or offset gap.
 * @tparam GapDefinition Gap callable accepting two physical points and their current common-plane frame.
 */
template <typename GapDefinition = CommonPlaneGap>
struct CommonPlaneResidual {
  GapDefinition gap{};  ///< Owned geometric gap and any fixed offset.
};

/**
 * @brief Direct Common Plane residual calculation with the shared local Enzyme derivative interface.
 * @tparam BaseRule Gauss rule on the common-line interval.
 * @tparam GapDefinition Geometric or offset gap type; defaults to the unshifted common-plane gap.
 */
template <typename BaseRule, typename GapDefinition = CommonPlaneGap>
class CommonPlaneForce {
 public:
  using Quadrature = CommonPlaneQuadratureRule<BaseRule>;                   ///< Common-line pair generator.
  using Physics = CommonPlaneResidual<GapDefinition>;                       ///< Fixed gap configuration.
  static constexpr IndexT first_dofs = LinearEdgeGeometry::Coords::size();  ///< First-element coordinate count.
  static constexpr IndexT dof_count = 2 * first_dofs;                       ///< Pair coordinate and residual count.
  using PrimalResult = PairResidualValue<dof_count>;  ///< Residual, quadrature status, and activation.
  /** @brief Store fixed configuration. @param rule Pair generator. @param penalty Effective stiffness.
   * @param physics Owned gap definition. */
  TRIBOL_HOST_DEVICE CommonPlaneForce( Quadrature rule, RealT penalty, Physics physics = {} )
      : quadrature_rule_( rule ), penalty_( penalty ), physics_( physics )
  {
  }
  /**
   * @brief Integrate equal-and-opposite nodal penalty residuals using current common-line geometry.
   * @param coordinates Current coordinates in element/node/component order.
   * @return Residual, activation, and status. Failed or empty rules give zeros; failures must be reported by the
   * caller. Zero gap remains active to retain the compressive tangent.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PrimalResult evaluate( MdSpan<const RealT, dof_count> coordinates ) const
  {
    const LinearEdgeGeometry geometry1( { coordinates.data(), first_dofs } );
    const LinearEdgeGeometry geometry2( { coordinates.data() + first_dofs, first_dofs } );
    const auto points = quadrature_rule_.generate( geometry1, geometry2 );
    PrimalResult result{ {}, points.result, false };
    if ( points.result.status != QuadratureStatus::Success || points.result.required_point_count == 0 ) return result;
    const auto frame = CommonPlaneEdgeMapping::makeFrame( geometry1, geometry2 );
    using Basis = BasisFunction<EDGE, LINEAR_LAGRANGE>;
    for ( IndexT q = 0; q < points.result.required_point_count; ++q ) {
      const RealT gap = physics_.gap( geometry1.position( points.geometry1_coords( q, 0 ) ),
                                      geometry2.position( points.geometry2_coords( q, 0 ) ), frame );
      if ( gap > 0.0 ) continue;
      result.has_active_points = true;
      const RealT pressure = penalty_ * gap;
      MdArray<RealT, Basis::node_count> basis1{}, basis2{};
      Basis::eval( { &points.geometry1_coords( q, 0 ), Basis::parametric_dimension }, basis1.view() );
      Basis::eval( { &points.geometry2_coords( q, 0 ), Basis::parametric_dimension }, basis2.view() );
      for ( IndexT node = 0; node < Basis::node_count; ++node ) {
        for ( IndexT component = 0; component < LinearEdgeGeometry::spatial_dimension; ++component ) {
          const IndexT row = node * LinearEdgeGeometry::spatial_dimension + component;
          const RealT traction = points.weights[q] * pressure * frame.normal[component];
          result.residual[row] += basis1[node] * traction;
          result.residual[first_dofs + row] -= basis2[node] * traction;
        }
      }
    }
    return result;
  }
  /** @brief Residual entry for Enzyme. @param coordinates Current active coordinates.
   * @return Integrated residual. @pre The host has checked primal quadrature status at these coordinates. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdArray<RealT, dof_count> residual(
      MdSpan<const RealT, dof_count> coordinates ) const
  {
    return evaluate( coordinates ).residual;
  }

 private:
  Quadrature quadrature_rule_;  ///< Owned mapping and base rule.
  RealT penalty_;               ///< Fixed effective stiffness.
  Physics physics_;             ///< Owned geometric gap and offset.
};

/**
 * @brief Evaluate Common Plane tractions from separate element coordinate arrays.
 * @tparam BaseRule Gauss rule on the common-line interval.
 * @tparam GapDefinition Geometric or offset common-plane gap type.
 * @param coordinates1 First element endpoints.
 * @param coordinates2 Second element endpoints.
 * @param quadrature_rule Pair generator.
 * @param penalty Fixed effective stiffness.
 * @param gap Owned gap definition; defaults to unshifted common-plane separation.
 * @return Residual and generation status. Failed/empty rules return zero; zero gap remains active.
 */
template <typename BaseRule, typename GapDefinition = CommonPlaneGap>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE CommonPlaneResidualData computeCommonPlaneResidual(
    LinearEdgeGeometry::Coords coordinates1, LinearEdgeGeometry::Coords coordinates2,
    const CommonPlaneQuadratureRule<BaseRule>& quadrature_rule, RealT penalty, GapDefinition gap = {} )
{
  constexpr IndexT first_dofs = LinearEdgeGeometry::Coords::size();
  MdArray<RealT, CommonPlaneResidualData::coordinate_count> coordinates;
  for ( IndexT i = 0; i < first_dofs; ++i ) {
    coordinates[i] = coordinates1[i];
    coordinates[first_dofs + i] = coordinates2[i];
  }
  return CommonPlaneForce<BaseRule, GapDefinition>{ quadrature_rule, penalty, CommonPlaneResidual{ gap } }.evaluate(
      coordinates.view() );
}

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Evaluate a pair residual and differentiate it with respect to both current edges using Enzyme.
 * @tparam BaseRule Reference Gauss rule.
 * @tparam GapDefinition Geometric or offset common-plane gap type.
 * @param coordinates1 First current edge endpoints.
 * @param coordinates2 Second current edge endpoints.
 * @param quadrature_rule Configured generator; not retained.
 * @param penalty Positive effective stiffness per unit length.
 * @param gap Owned gap definition; its geometry and finite-configuration preconditions apply.
 * @return Residual and generally nonsymmetric Jacobian; valid zero-point rules give zeros.
 * @throws std::invalid_argument Geometry or penalty parameters are invalid.
 * @throws std::length_error Quadrature storage is insufficient.
 */
template <typename BaseRule, typename GapDefinition = CommonPlaneGap>
CommonPlanePairData computeCommonPlanePair( LinearEdgeGeometry::Coords coordinates1,
                                            LinearEdgeGeometry::Coords coordinates2,
                                            const CommonPlaneQuadratureRule<BaseRule>& quadrature_rule, RealT penalty,
                                            GapDefinition gap = {} );
#endif

/*!
 *
 * \brief computes penalty stiffness for Common Plane + Penalty
 *
 * \param [in] K1/t1 contact spring stiffness for face 1 (bulk_modulus/element_thickness for face 1)
 * \param [in] K2/t2 contact spring stiffness for face 2 (bulk_modulus/element_thickness for face 2)
 *
 * \return face-pair based, element-wise penalty stiffness per area
 *
 *
 * \pre Bulk modulus and element thickness arrays are registered by host code
 *
 */
TRIBOL_HOST_DEVICE inline RealT ComputePenaltyStiffnessPerArea( const RealT K1_over_t1, const RealT K2_over_t2 )
{
  // compute face-pair specific penalty stiffness per unit area.
  // Note: This assumes that each face has a spring stiffness
  // equal to that side's material Bulk modulus, K, over the
  // thickness of the volume element to which that face belongs,
  // times the overlap area. That is, K1_over_t1 * A and K2_over_t2 * A. We
  // then assume the two springs are in series and compute an
  // equivalent spring stiffness as,
  // k_eq = A*(K1_over_t1)*(K2_over_t2) / ((K1_over_t1)+(K2_over_t2).
  // Note, the host code registers each face's (K/t) as a penalty scale.
  //
  // UNITS: we multiply k_eq above by the overlap area A, to get a
  // stiffness per unit area. This will make the force calculations
  // commensurate with the previous calculations using only the
  // constant registered penalty scale.

  return SeriesPenaltyStiffness{}( K1_over_t1, K2_over_t2 );

}  // end ComputePenaltyStiffnessPerArea

/*!
 *
 * \brief routine to apply interface physics in the direction normal to the interface
 *
 * \param [in] cs pointer to the coupling scheme
 *
 * \return 0 if no error
 *
 */
template <>
int ApplyNormal<COMMON_PLANE, PENALTY>( CouplingScheme* cs );

/*!
 *
 * \brief routine to apply interface physics in the direction tangential to the interface
 *
 * \param [in] cs pointer to the coupling scheme
 *
 * \return 0 if no error
 *
 */
template <>
int ApplyTangential<COMMON_PLANE, PENALTY, VISCOUS_TANGENTIAL>( CouplingScheme* cs );

}  // end namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_ */
