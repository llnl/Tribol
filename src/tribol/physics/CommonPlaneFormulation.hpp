// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Composed Common Plane contact formulation without changes to the legacy implementation. */
#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_

#include "tribol/config.hpp"
#include "tribol/geom/CommonPlaneEdgeMapping.hpp"
#include "tribol/geom/CommonPlaneGap.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/physics/ContactEnforcement.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/assembly/PairEvaluation.hpp"
#include "tribol/search/PairListBuilder.hpp"

namespace tribol {

/**
 * @brief Fixed-size Common Plane residual function for direct evaluation and optional Enzyme differentiation.
 * @tparam BaseRuleGenerator Reference Gauss rule generator.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 */
template <typename BaseRuleGenerator, typename GapDefinition = CommonPlaneGap>
class CommonPlanePairFunction {
 public:
  using FirstGeometry = LinearEdgeGeometry;                                   ///< Geometry of the first directed edge.
  using SecondGeometry = LinearEdgeGeometry;                                  ///< Geometry of the second directed edge.
  static constexpr IndexT num_geom1_coords = FirstGeometry::Coords::size();   ///< Geometry1 coordinate count.
  static constexpr IndexT num_geom2_coords = SecondGeometry::Coords::size();  ///< Geometry2 coordinate count.
  static constexpr IndexT num_coords = num_geom1_coords + num_geom2_coords;   ///< Combined coordinate count.
  static constexpr IndexT num_values = num_coords;                            ///< Residual entry count.

  /** @brief Store fixed configuration. @param rule_generator Pair quadrature rule generator. @param penalty Stiffness.
   * @param gap Gap callable.
   */
  TRIBOL_HOST_DEVICE CommonPlanePairFunction(
      PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, BaseRuleGenerator> rule_generator, RealT penalty,
      GapDefinition gap = {} )
      : rule_generator_( rule_generator ), penalty_( penalty ), gap_( gap )
  {
  }

  /** @brief Evaluate one pair residual. @param geometry1 First current edge. @param geometry2 Second current edge.
   * @return Residual and status; failed, empty, and open pairs return zero residual and inactive state. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairResidualValue<num_coords> evaluate(
      const FirstGeometry& geometry1, const SecondGeometry& geometry2 ) const
  {
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    PairResidualValue<num_coords> result{ {}, points.result, false };
    if ( points.result.status != QuadratureStatus::Success || points.result.num_required_points == 0 ) {
      return result;
    }
    const auto line = CommonPlaneEdgeMapping::computeCommonPlaneLine( geometry1, geometry2 );
    // The linear coordinate maps make the weighted parent coordinates the overlap centroid for every base rule.
    RealT measure = 0.0;
    RealT center1 = 0.0;
    RealT center2 = 0.0;
    for ( IndexT point = 0; point < points.result.num_required_points; ++point ) {
      measure += points.weights[point];
      center1 += points.weights[point] * points.geometry1_coords( point, 0 );
      center2 += points.weights[point] * points.geometry2_coords( point, 0 );
    }
    const RealT gap = gap_( geometry1.position( center1 / measure ), geometry2.position( center2 / measure ), line );
    if ( gap > 0.0 ) {
      return result;
    }
    result.has_active_points = true;
    const RealT pressure = penalty_ * gap;
    using Basis = BasisFunction<EDGE, LINEAR_LAGRANGE>;
    for ( IndexT point = 0; point < points.result.num_required_points; ++point ) {
      MdArray<RealT, Basis::num_nodes> basis1{}, basis2{};
      Basis::eval( { &points.geometry1_coords( point, 0 ), Basis::parametric_dimension }, basis1.view() );
      Basis::eval( { &points.geometry2_coords( point, 0 ), Basis::parametric_dimension }, basis2.view() );
      for ( IndexT node = 0; node < Basis::num_nodes; ++node ) {
        for ( IndexT component = 0; component < LinearEdgeGeometry::spatial_dimension; ++component ) {
          const IndexT row = node * LinearEdgeGeometry::spatial_dimension + component;
          const RealT traction = points.weights[point] * pressure * line.normal[component];
          result.residual[row] += basis1[node] * traction;
          result.residual[num_geom1_coords + row] -= basis2[node] * traction;
        }
      }
    }
    return result;
  }

 private:
  PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, BaseRuleGenerator>
      rule_generator_;  ///< Owned pair quadrature rule generator.
  RealT penalty_;       ///< Effective pair stiffness.
  GapDefinition gap_;   ///< Owned gap definition.
};

#ifdef BUILD_REDECOMP
/**
 * @brief Common Plane pair search and force assembly, with optional Enzyme Jacobians.
 * @tparam BaseRuleGenerator Reference Gauss rule generator.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 * @tparam Evaluation ResidualForceEvaluation or ResidualForceJacobianEvaluation prerequisites.
 */
template <typename BaseRuleGenerator, ExecutionMode Exec, typename Evaluation>
class CommonPlaneFormulation final : public ContactFormulation {
 public:
  /**
   * @brief Store fixed search and physics settings and references to transfer data.
   * @param evaluation Transfer-data references selecting force-only or force/Jacobian calculation. The referenced
   * data must remain valid for this formulation's lifetime.
   * @param pair_builder Owned pair search composition.
   * @param rule_generator Owned common-line quadrature rule generator.
   * @param gap Owned gap definition.
   * @param penalty Positive, finite effective stiffness fixed for this formulation's lifetime.
   * @throws std::invalid_argument Stiffness is nonpositive or nonfinite; no formulation is constructed.
   */
  CommonPlaneFormulation( Evaluation evaluation, PairListBuilder pair_builder,
                          PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, BaseRuleGenerator> rule_generator,
                          OffsetGap<CommonPlaneGap> gap, RealT penalty )
      : evaluation_( evaluation ),
        pair_builder_( std::move( pair_builder ) ),
        function_( std::move( rule_generator ), detail::checkedPenaltyStiffness( penalty ), std::move( gap ) )
  {
  }

  /** @brief Compute Common Plane force and the optional Jacobian. @param mesh1 First mesh. @param mesh2 Second mesh.
   * @return Owned force; df_dx is absent when Jacobian calculation was disabled at construction. */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

 private:
  Evaluation evaluation_;         ///< References to transfer data used by the selected calculation.
  PairListBuilder pair_builder_;  ///< Owned search composition.
  /// Owned pair quadrature rule generator, fixed stiffness, and gap definition.
  const CommonPlanePairFunction<BaseRuleGenerator, OffsetGap<CommonPlaneGap>> function_;
};
#endif

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
