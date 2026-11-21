// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Opt-in composed Common Plane contact formulation, separate from the legacy response path. */
#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_

#include <cmath>
#include <utility>

#include "tribol/config.hpp"
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

  /** @brief Own both element geometries so local normals can be evaluated at each paired point. */
  template <typename Geometry1, typename Geometry2>
  struct PairGeometryData {
    Geometry1 geometry1;  ///< First current surface-element geometry.
    Geometry2 geometry2;  ///< Second current surface-element geometry.
  };

  /**
   * @brief Retain both geometries for pointwise normal evaluation.
   * @tparam Geometry1 First surface-element geometry type providing unitNormal().
   * @tparam Geometry2 Second surface-element geometry type providing unitNormal().
   * @param geometry1 First current surface-element geometry.
   * @param geometry2 Second current surface-element geometry.
   * @return Owning copies used at each quadrature point.
   */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairGeometryData<Geometry1, Geometry2> computePairGeometryData(
      const Geometry1& geometry1, const Geometry2& geometry2 ) const
  {
    return { geometry1, geometry2 };
  }

  /**
   * @brief Compute penalty traction using physical positions and pointwise averaged normals; zero gap retains the
   * compressive-side derivative.
   * @tparam Geometry1 First surface-element geometry type.
   * @tparam Geometry2 Second surface-element geometry type.
   * @param point1 Physical point on geometry1.
   * @param point2 Physical point on geometry2.
   * @param parametric_coords1 Element parametric coordinate on geometry1.
   * @param parametric_coords2 Element parametric coordinate on geometry2.
   * @param pair_geometry_data Element geometries retained for local normal evaluation.
   * @return Active compressive traction; unusable, nonopposing, or separated points return inactive.
   */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadraturePointValue<MdArray<RealT, Geometry1::spatial_dim>>
  computeTractionAtQuadraturePoint( const typename Geometry1::Point& point1, const typename Geometry2::Point& point2,
                                    MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1,
                                    MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2,
                                    const PairGeometryData<Geometry1, Geometry2>& pair_geometry_data ) const
  {
    static_assert( Geometry1::spatial_dim == Geometry2::spatial_dim );
    typename Geometry1::Vector normal1;
    typename Geometry2::Vector normal2;
    if ( !pair_geometry_data.geometry1.unitNormal( parametric_coords1, normal1 ) ||
         !pair_geometry_data.geometry2.unitNormal( parametric_coords2, normal2 ) || normal1.dot( normal2 ) >= 0.0 ) {
      return {};
    }
    auto normal = normal2 - normal1;
    const RealT normal_length = std::sqrt( normal.dot( normal ) );
    if ( !std::isfinite( normal_length ) || normal_length <= 0.0 ) {
      return {};
    }
    normal /= normal_length;
    const RealT scalar_gap = gap.computeGap( point1, point2, normal );
    if ( scalar_gap > 0.0 ) {
      return {};
    }
    QuadraturePointValue<MdArray<RealT, Geometry1::spatial_dim>> result;
    result.active = true;
    const RealT pressure = penalty * scalar_gap;
    for ( IndexT component = 0; component < Geometry1::spatial_dim; ++component ) {
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
 * @tparam PairRuleGenerator Generates quadrature points and physical weights for the current edge pair.
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
  const PairTractionIntegral<PairRuleGenerator, CommonPlanePenaltyTraction<GapDefinition>,
                             ResidualBasis>
      pair_integral_;  ///< Integrates the fixed Common Plane penalty traction with the configured edge basis.
  const ResidualOutputs residual_outputs_;  ///< Owns the selected assembly behavior -- residual or residual + jacobian
  PairListBuilder pair_builder_;            ///< Owned search composition.
};
#endif

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
