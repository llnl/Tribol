// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_

#include "Physics.hpp"
#include "tribol/physics/PenaltyStiffness.hpp"
#include "tribol/physics/PairDefinition.hpp"
#include "tribol/geom/CommonPlaneEdgeMapping.hpp"
#include "tribol/geom/CommonPlaneGap.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/integ/PairIntegral.hpp"
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
 * @brief Define Common Plane by integrating nodal tractions using an owned gap definition.
 * @tparam GapDefinition Gap callable accepting a pair point and its current common-plane frame.
 */
template <typename GapDefinition = CommonPlaneGap>
struct CommonPlaneResidual {
  GapDefinition gap{};  ///< Owned gap configuration; current geometry is supplied during evaluation.
  /**
   * @brief Integrate penalty traction using the shared pair integration context.
   * @tparam IntegrationRule Current common-line pair rule.
   * @param integration_rule Current geometry and generated quadrature.
   * @param penalty Fixed effective stiffness.
   * @return Pair residual; failed or empty quadrature returns zero. The compressive branch includes zero gap.
   */
  template <typename IntegrationRule>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto operator()( const IntegrationRule& integration_rule,
                                                           RealT penalty ) const
  {
    using Residual = MdArray<RealT, CommonPlaneResidualData::coordinate_count>;
    using Basis = BasisFunction<EDGE, LINEAR_LAGRANGE>;
    // Capture fixed inputs by value to keep the configured offset in Enzyme's geometry derivatives.
    return integral(
        [gap = gap, penalty]( const auto& point ) -> QuadraturePointContribution<Residual> {
          const auto frame = CommonPlaneEdgeMapping::makeFrame( point.geometry1, point.geometry2 );
          const RealT normal_gap = gap( point, frame );
          if ( normal_gap > 0.0 ) return {};
          const RealT pressure = penalty * normal_gap;
          const auto basis1 = point.template evalBasis1<EDGE, LINEAR_LAGRANGE>();
          const auto basis2 = point.template evalBasis2<EDGE, LINEAR_LAGRANGE>();
          Residual integrand{};
          for ( int node = 0; node < Basis::node_count; ++node ) {
            for ( int component = 0; component < LinearEdgeGeometry::spatial_dimension; ++component ) {
              const RealT traction = pressure * frame.normal[component];
              integrand[node * LinearEdgeGeometry::spatial_dimension + component] = basis1[node] * traction;
              integrand[( node + Basis::node_count ) * LinearEdgeGeometry::spatial_dimension + component] =
                  -basis2[node] * traction;
            }
          }
          return { integrand, true };
        },
        integration_rule );
  }
};

/**
 * @brief Common Plane residual expression with the shared pair derivative interface.
 * @tparam BaseRule Gauss rule on the common-line interval.
 * @tparam GapDefinition Geometric or offset gap type; defaults to the unshifted common-plane gap.
 */
template <typename BaseRule, typename GapDefinition = CommonPlaneGap>
using CommonPlaneForce =
    PairResidualDefinition<CommonPlaneQuadratureRule<BaseRule>, CommonPlaneResidual<GapDefinition>>;

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
