// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_

#include "Physics.hpp"
#include "tribol/geom/CommonPlaneEdgeMapping.hpp"
#include "tribol/integ/PairQuadratureRule.hpp"
#include "tribol/integ/EdgeQuadratureRule.hpp"
#include "tribol/integ/BasisFunction.hpp"

namespace tribol {

/** @brief Pair quadrature using physical distance on a common line. @tparam BaseRule Reference Gauss rule. */
template <typename BaseRule>
using CommonPlaneQuadratureRule = PairQuadratureRule<CommonPlaneEdgeMapping, BaseRule>;

/** @brief Residual and generation status for a pair of linear two-dimensional edges. */
struct CommonPlaneResidualData {
  static constexpr int dimension = LinearEdgeGeometry::spatial_dimension;             ///< Physical components.
  static constexpr int nodes_per_element = LinearEdgeGeometry::endpoint_count;        ///< Nodes on each edge.
  static constexpr int pair_size = 2;                                                 ///< Elements in a contact pair.
  static constexpr int coordinate_count = dimension * nodes_per_element * pair_size;  ///< Pair coordinate DOFs.
  MdArray<RealT, coordinate_count> residual{};  ///< Node-major contact residual, first element then second.
  QuadratureResult quadrature_result{};  ///< Status and initialized quadrature count; discard residual on failure.
  bool has_active_points{ false };       ///< Whether any gap is nonpositive, including exactly zero.
};

/** @brief Pair residual and its derivative in node-major pair-coordinate order. */
struct CommonPlanePairData : CommonPlaneResidualData {
  MdArray<RealT, coordinate_count, coordinate_count> jacobian{};  ///< Rows are residual entries, columns coordinates.
};

/**
 * @brief Integrate penalty tractions directly; no derivative of an energy is used.
 * @tparam BaseRule Reference rule on the common-line interval.
 * @param coordinates1 First element's current endpoints.
 * @param coordinates2 Second element's current endpoints.
 * @param quadrature_rule Configured pair generator; not retained.
 * @param penalty Positive effective stiffness per unit overlap length.
 * @param residual_gap Gap offset subtracted before activation.
 * @return Pair residual and status. Failed quadrature returns zero residual that must be discarded; valid empty rules
 * return a successful zero residual. Nonpositive gaps use the active branch, including at zero pressure.
 */
template <typename BaseRule>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE CommonPlaneResidualData computeCommonPlaneResidual(
    LinearEdgeGeometry::Coordinates coordinates1, LinearEdgeGeometry::Coordinates coordinates2,
    const CommonPlaneQuadratureRule<BaseRule>& quadrature_rule, RealT penalty, RealT residual_gap )
{
  const LinearEdgeGeometry edge1( coordinates1 ), edge2( coordinates2 );
  const auto quadrature = quadrature_rule.generate( edge1, edge2 );
  CommonPlaneResidualData result{};
  result.quadrature_result = { quadrature.result.status, quadrature.result.required_point_count };
  if ( quadrature.result.status != QuadratureStatus::Success || quadrature.result.required_point_count == 0 ) {
    return result;
  }
  const auto points = quadrature.getView();
  const auto frame = CommonPlaneEdgeMapping::makeFrame( edge1, edge2 );
  using Basis = BasisFunction<EDGE, LINEAR_LAGRANGE>;
  for ( IndexT q = 0; q < points.capacity(); ++q ) {
    const RealT parent1 = points.geometry1_parent_coordinates( q, 0 );
    const RealT parent2 = points.geometry2_parent_coordinates( q, 0 );
    const auto point1 = edge1.position( parent1 ), point2 = edge2.position( parent2 );
    const RealT gap = LinearEdgeGeometry::Vector( point2, point1 ).dot( frame.normal ) - residual_gap;
    if ( gap > 0.0 ) continue;
    result.has_active_points = true;
    const RealT pressure = penalty * gap;
    MdArray<RealT, Basis::node_count> basis1{}, basis2{};
    Basis::Eval( MdSpan<const RealT, 1>{ &parent1, 1 }, basis1.view() );
    Basis::Eval( MdSpan<const RealT, 1>{ &parent2, 1 }, basis2.view() );
    for ( int node = 0; node < Basis::node_count; ++node ) {
      for ( int component = 0; component < CommonPlaneResidualData::dimension; ++component ) {
        const RealT traction = points.physical_weights[q] * pressure * frame.normal[component];
        result.residual[node * CommonPlaneResidualData::dimension + component] += basis1[node] * traction;
        result.residual[( node + Basis::node_count ) * CommonPlaneResidualData::dimension + component] -=
            basis2[node] * traction;
      }
    }
  }
  return result;
}

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Evaluate a pair residual and differentiate it with respect to both current edges using Enzyme.
 * @tparam BaseRule Reference Gauss rule.
 * @param coordinates1 First current edge endpoints.
 * @param coordinates2 Second current edge endpoints.
 * @param quadrature_rule Configured generator; not retained.
 * @param penalty Positive effective stiffness per unit length.
 * @param residual_gap Gap offset subtracted before activation.
 * @return Residual and generally nonsymmetric Jacobian; valid zero-point rules give zeros.
 * @throws std::invalid_argument Geometry or penalty parameters are invalid.
 * @throws std::length_error Quadrature storage is insufficient.
 */
template <typename BaseRule>
CommonPlanePairData computeCommonPlanePair( LinearEdgeGeometry::Coordinates coordinates1,
                                            LinearEdgeGeometry::Coordinates coordinates2,
                                            const CommonPlaneQuadratureRule<BaseRule>& quadrature_rule, RealT penalty,
                                            RealT residual_gap );
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

  return K1_over_t1 * K2_over_t2 / ( K1_over_t1 + K2_over_t2 );

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
