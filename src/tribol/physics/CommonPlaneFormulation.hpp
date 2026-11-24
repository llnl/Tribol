// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Opt-in composed Common Plane contact formulation, separate from the legacy response path. */
#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_

#include <utility>

#include "tribol/config.hpp"
#include "tribol/geom/CommonPlaneGap.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/assembly/AssemblePairs.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/search/PairListBuilder.hpp"

namespace tribol {

/**
 * @brief Constant-penalty Common Plane traction along the generated rule's integration-surface normal, with activation
 * determined by the configured gap.
 *
 * The rule generator pairs the two element points along the integration-surface normal n and stores n with the point.
 * When the pair is separated exactly along n, as with linear edges or quadratic edges projected by
 * CommonPlaneQuadraticEdgeProjection, the traction is parallel to the separation of its application points and the
 * integrated residual has no net moment.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 */
template <typename GapDefinition = CommonPlaneGap>
struct CommonPlanePenaltyTraction {
  RealT penalty;        ///< Fixed penalty stiffness per area.
  GapDefinition gap{};  ///< Owned gap configuration.

  /** @brief No pair geometry is needed because each point supplies its integration-surface normal. */
  struct PairGeometryData {};

  /**
   * @brief Prepare no pair geometry.
   * @tparam Geometry1 First surface-element geometry type.
   * @tparam Geometry2 Second surface-element geometry type.
   * @param geometry1 First current surface-element geometry, unused.
   * @param geometry2 Second current surface-element geometry, unused.
   * @return Empty pair geometry data.
   */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairGeometryData computePairGeometryData(
      const Geometry1& TRIBOL_UNUSED_PARAM( geometry1 ), const Geometry2& TRIBOL_UNUSED_PARAM( geometry2 ) ) const
  {
    return {};
  }

  /**
   * @brief Compute penalty traction along the integration-surface normal; zero gap retains the compressive-side
   * derivative.
   * @tparam Point Physical point type.
   * @tparam Vector Physical vector type.
   * @tparam ParametricDim1 Number of parametric coordinates on geometry1.
   * @tparam ParametricDim2 Number of parametric coordinates on geometry2.
   * @param point1 Physical point on geometry1.
   * @param point2 Physical point on geometry2.
   * @param parametric_coords1 Element parametric coordinate on geometry1, unused.
   * @param parametric_coords2 Element parametric coordinate on geometry2, unused.
   * @param normal Unit integration-surface normal along which the rule paired the two points.
   * @param pair_geometry_data Empty pair geometry data, unused.
   * @return Active compressive traction penalty * gap * normal; separated points return inactive.
   */
  template <typename Point, typename Vector, IndexT ParametricDim1, IndexT ParametricDim2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadraturePointValue<MdArray<RealT, Point::DIMENSION>>
  computeTractionAtQuadraturePoint( const Point& point1, const Point& point2,
                                    MdSpan<const RealT, ParametricDim1> TRIBOL_UNUSED_PARAM( parametric_coords1 ),
                                    MdSpan<const RealT, ParametricDim2> TRIBOL_UNUSED_PARAM( parametric_coords2 ),
                                    const Vector& normal,
                                    const PairGeometryData& TRIBOL_UNUSED_PARAM( pair_geometry_data ) ) const
  {
    const RealT scalar_gap = gap.computeGap( point1, point2, normal );
    if ( scalar_gap > 0.0 ) {
      return {};
    }
    QuadraturePointValue<MdArray<RealT, Point::DIMENSION>> result;
    result.active = true;
    const RealT pressure = penalty * scalar_gap;
    for ( IndexT component = 0; component < Point::DIMENSION; ++component ) {
      result.value[component] = pressure * normal[component];
    }
    return result;
  }
};

#ifdef BUILD_REDECOMP
/**
 * @brief Common Plane pair search followed by the configured residual evaluation and assembly.
 *
 * Uses PairTractionIntegral with CommonPlanePenaltyTraction and the configured residual basis on both edges.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairRuleGenerator Generates quadrature points, physical weights, and integration-surface normals for the
 * current edge pair.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 * @tparam ResidualOutputs Either ResidualOnly or ResidualAndJacobian in assembly/AssemblePairs.hpp
 * @tparam ResidualBasis Finite-element basis used to weight traction on both edges.
 */
template <ExecutionMode Exec, typename PairRuleGenerator, typename GapDefinition, typename ResidualOutputs,
          typename ResidualBasis = BasisFunction<EDGE, LINEAR_LAGRANGE>>
class CommonPlaneFormulation final : public ContactFormulation {
 public:
  /**
   * @brief Construct the penalty traction integral and own the output policy and pair search.
   * @param rule_generator Configured pair quadrature generator, retained by value.
   * @param penalty Fixed effective penalty stiffness.
   * @param gap Configured common-plane gap, retained by value.
   * @param residual_outputs Configured output policy, retained by value.
   * @param pair_builder Owned pair search composition.
   */
  CommonPlaneFormulation( PairRuleGenerator rule_generator, RealT penalty, GapDefinition gap,
                          ResidualOutputs residual_outputs, PairListBuilder pair_builder )
      : pair_integral_( std::move( rule_generator ), { penalty, std::move( gap ) } ),
        residual_outputs_( std::move( residual_outputs ) ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Compute the Common Plane residual outputs selected by the policy.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @param inputs Unused; this formulation reads no caller-supplied fields.
   * @return Owned residual outputs selected by ResidualOutputs; other outputs are absent.
   */
  ContactFormulationResult computeNodalResiduals(
      MeshData::Viewer mesh1, MeshData::Viewer mesh2,
      [[maybe_unused]] const ContactFormulationInputs& inputs ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return residual_outputs_.template computeResidual<Exec>( pairs, mesh1, mesh2, pair_integral_ );
  }

 private:
  const PairTractionIntegral<PairRuleGenerator, CommonPlanePenaltyTraction<GapDefinition>, ResidualBasis>
      pair_integral_;  ///< Integrates the fixed Common Plane penalty traction with the configured edge basis.
  const ResidualOutputs residual_outputs_;  ///< Owns the selected assembly behavior -- residual or residual + jacobian
  PairListBuilder pair_builder_;            ///< Owned search composition.
};
#endif

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
