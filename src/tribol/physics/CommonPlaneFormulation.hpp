// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Opt-in composed Common Plane contact formulation, separate from the legacy response path. */
#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_

#include <utility>

#include "tribol/config.hpp"
#include "tribol/geom/CommonPlaneEdgeMapping.hpp"
#include "tribol/geom/CommonPlaneGap.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/assembly/AssemblePairs.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/search/PairListBuilder.hpp"

namespace tribol {

/** @brief Constant-penalty Common Plane traction, with activation determined by the configured gap.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 */
template <typename GapDefinition = CommonPlaneGap>
struct CommonPlanePenaltyTraction {
  RealT penalty;        ///< Fixed penalty stiffness per area.
  GapDefinition gap{};  ///< Owned gap configuration.

  /** @brief Compute the common line once for the current pair.
   * @param geometry1 First current edge.
   * @param geometry2 Second current edge.
   * @return Common line used for both gap evaluation and traction direction.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto computePairGeometryData( const LinearEdgeGeometry& geometry1,
                                                                        const LinearEdgeGeometry& geometry2 ) const
  {
    return gap.computePairGeometryData( geometry1, geometry2 );
  }

  /** @brief Compute penalty traction at one quadrature point; zero gap retains the compressive-side derivative.
   * @param point1 Physical point on geometry1.
   * @param point2 Physical point on geometry2.
   * @param parametric_coords1 Geometry1 parametric point, unused by this penalty law.
   * @param parametric_coords2 Geometry2 parametric point, unused by this penalty law.
   * @param line Common line prepared for this pair.
   * @return Penalty traction and active=true for nonpositive gap; zero traction and inactive state for opening.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadraturePointValue<MdArray<RealT, LinearEdgeGeometry::spatial_dim>>
  operator()( const LinearEdgeGeometry::Point& point1, const LinearEdgeGeometry::Point& point2,
              MdSpan<const RealT, LinearEdgeGeometry::parametric_dim> TRIBOL_UNUSED_PARAM( parametric_coords1 ),
              MdSpan<const RealT, LinearEdgeGeometry::parametric_dim> TRIBOL_UNUSED_PARAM( parametric_coords2 ),
              const CommonPlaneEdgeMapping::CommonPlaneLine& line ) const
  {
    const RealT scalar_gap = gap( point1, point2, line );
    if ( scalar_gap > 0.0 ) {
      return {};
    }
    QuadraturePointValue<MdArray<RealT, LinearEdgeGeometry::spatial_dim>> result;
    result.active = true;
    const RealT pressure = penalty * scalar_gap;
    for ( IndexT spatial_dim = 0; spatial_dim < LinearEdgeGeometry::spatial_dim; ++spatial_dim ) {
      result.value[spatial_dim] = pressure * line.normal[spatial_dim];
    }
    return result;
  }
};

#ifdef BUILD_REDECOMP
/**
 * @brief Common Plane pair search followed by the configured residual evaluation and assembly.
 *
 * Uses PairTractionIntegral with CommonPlanePenaltyTraction and linear Lagrange residual bases on both edges.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairRuleGenerator Generates quadrature points and physical weights for the current edge pair.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 * @tparam ResidualOutputs Either ResidualOnly or ResidualAndJacobian in assembly/AssemblePairs.hpp
 */
template <ExecutionMode Exec, typename PairRuleGenerator, typename GapDefinition, typename ResidualOutputs>
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
  const PairTractionIntegral<PairRuleGenerator, CommonPlanePenaltyTraction<GapDefinition>,
                             BasisFunction<EDGE, LINEAR_LAGRANGE>>
      pair_integral_;  ///< Integrates the fixed Common Plane penalty traction with linear edge bases.
  const ResidualOutputs residual_outputs_;  ///< Owns the selected assembly behavior -- residual or residual + jacobian
  PairListBuilder pair_builder_;            ///< Owned search composition.
};
#endif

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
