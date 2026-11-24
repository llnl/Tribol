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

/**
 * @brief Own both element geometries so local normals can be evaluated at each paired point.
 * @tparam Geometry1 First surface-element geometry type providing unitNormal().
 * @tparam Geometry2 Second surface-element geometry type providing unitNormal().
 */
template <typename Geometry1, typename Geometry2>
struct CommonPlanePairGeometries {
  Geometry1 geometry1;  ///< First current surface-element geometry.
  Geometry2 geometry2;  ///< Second current surface-element geometry.

  /**
   * @brief Average the local unit normals at a paired point as the unit direction of normal2 minus normal1.
   * @param parametric_coords1 Element parametric coordinate on geometry1.
   * @param parametric_coords2 Element parametric coordinate on geometry2.
   * @param[out] normal Averaged unit normal when true.
   * @return Whether both local normals are usable and oppose each other.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE bool averagedUnitNormal(
      MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1,
      MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2, typename Geometry1::Vector& normal ) const
  {
    static_assert( Geometry1::spatial_dim == Geometry2::spatial_dim );
    typename Geometry1::Vector normal1;
    typename Geometry2::Vector normal2;
    if ( !geometry1.unitNormal( parametric_coords1, normal1 ) || !geometry2.unitNormal( parametric_coords2, normal2 ) ||
         normal1.dot( normal2 ) >= 0.0 ) {
      return false;
    }
    normal = normal2 - normal1;
    const RealT normal_length = std::sqrt( normal.dot( normal ) );
    if ( !std::isfinite( normal_length ) || normal_length <= 0.0 ) {
      return false;
    }
    normal /= normal_length;
    return true;
  }
};

/** @brief Constant-penalty Common Plane traction, with activation determined by the configured gap.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 */
template <typename GapDefinition = CommonPlaneGap>
struct CommonPlanePenaltyTraction {
  RealT penalty;        ///< Fixed penalty stiffness per area.
  GapDefinition gap{};  ///< Owned gap configuration.

  /**
   * @brief Retain both geometries for pointwise normal evaluation.
   * @tparam Geometry1 First surface-element geometry type providing unitNormal().
   * @tparam Geometry2 Second surface-element geometry type providing unitNormal().
   * @param geometry1 First current surface-element geometry.
   * @param geometry2 Second current surface-element geometry.
   * @return Owning copies used at each quadrature point.
   */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE CommonPlanePairGeometries<Geometry1, Geometry2> computePairGeometryData(
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
   * @note On curved elements, the averaged local normal is generally not parallel to the separation of points paired
   * on linear subelements, so this traction does not conserve angular momentum there. Use
   * ProjectedCommonPlanePenaltyTraction with a projection that pairs points on the curved elements.
   */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadraturePointValue<MdArray<RealT, Geometry1::spatial_dim>>
  computeTractionAtQuadraturePoint( const typename Geometry1::Point& point1, const typename Geometry2::Point& point2,
                                    MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1,
                                    MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2,
                                    const CommonPlanePairGeometries<Geometry1, Geometry2>& pair_geometry_data ) const
  {
    typename Geometry1::Vector normal;
    if ( !pair_geometry_data.averagedUnitNormal( parametric_coords1, parametric_coords2, normal ) ) {
      return {};
    }
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

/**
 * @brief Constant-penalty Common Plane traction for points paired along the common-plane normal on curved elements.
 *
 * The rule generator must separate every point pair exactly along the common-plane normal n that paired it, as
 * SubdividedPairQuadratureRuleGenerator does with CommonPlaneQuadraticEdgeProjection. Then point1 - point2 = g n,
 * where g is the Common Plane gap, and the traction penalty * g * n equals penalty * (point1 - point2) without
 * evaluating n at the point. A traction parallel to the separation of its application points produces no net moment,
 * so the integrated residual conserves angular momentum. The averaged local normal only orients the gap sign.
 */
struct ProjectedCommonPlanePenaltyTraction {
  RealT penalty;  ///< Fixed penalty stiffness per area.

  /**
   * @brief Retain both geometries for pointwise normal evaluation.
   * @tparam Geometry1 First surface-element geometry type providing unitNormal().
   * @tparam Geometry2 Second surface-element geometry type providing unitNormal().
   * @param geometry1 First current surface-element geometry.
   * @param geometry2 Second current surface-element geometry.
   * @return Owning copies used at each quadrature point.
   */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE CommonPlanePairGeometries<Geometry1, Geometry2> computePairGeometryData(
      const Geometry1& geometry1, const Geometry2& geometry2 ) const
  {
    return { geometry1, geometry2 };
  }

  /**
   * @brief Penalize the separation of a paired point; zero gap retains the compressive-side derivative.
   * @tparam Geometry1 First surface-element geometry type.
   * @tparam Geometry2 Second surface-element geometry type.
   * @param point1 Physical point on geometry1.
   * @param point2 Physical point on geometry2, separated from point1 along the common-plane normal.
   * @param parametric_coords1 Element parametric coordinate on geometry1.
   * @param parametric_coords2 Element parametric coordinate on geometry2.
   * @param pair_geometry_data Element geometries retained for local normal evaluation.
   * @return Traction penalty * (point1 - point2) when active; unusable or nonopposing local normals and separated
   * points return inactive.
   */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadraturePointValue<MdArray<RealT, Geometry1::spatial_dim>>
  computeTractionAtQuadraturePoint( const typename Geometry1::Point& point1, const typename Geometry2::Point& point2,
                                    MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1,
                                    MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2,
                                    const CommonPlanePairGeometries<Geometry1, Geometry2>& pair_geometry_data ) const
  {
    typename Geometry1::Vector orientation;
    if ( !pair_geometry_data.averagedUnitNormal( parametric_coords1, parametric_coords2, orientation ) ) {
      return {};
    }
    const typename Geometry1::Vector separation( point2, point1 );
    if ( separation.dot( orientation ) > 0.0 ) {
      return {};
    }
    QuadraturePointValue<MdArray<RealT, Geometry1::spatial_dim>> result;
    result.active = true;
    for ( IndexT component = 0; component < Geometry1::spatial_dim; ++component ) {
      result.value[component] = penalty * separation[component];
    }
    return result;
  }
};

#ifdef BUILD_REDECOMP
/**
 * @brief Common Plane pair search followed by the configured residual evaluation and assembly.
 *
 * Uses PairTractionIntegral with the configured penalty traction and residual basis on both edges.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairRuleGenerator Generates quadrature points and physical weights for the current edge pair.
 * @tparam TractionFunction Penalty traction, such as CommonPlanePenaltyTraction or ProjectedCommonPlanePenaltyTraction.
 * @tparam ResidualOutputs Either ResidualOnly or ResidualAndJacobian in assembly/AssemblePairs.hpp
 * @tparam ResidualBasis Finite-element basis used to weight traction on both edges.
 */
template <ExecutionMode Exec, typename PairRuleGenerator, typename TractionFunction, typename ResidualOutputs,
          typename ResidualBasis = BasisFunction<EDGE, LINEAR_LAGRANGE>>
class CommonPlaneFormulation final : public ContactFormulation {
 public:
  /**
   * @brief Construct the penalty traction integral and own the output policy and pair search.
   * @param rule_generator Configured pair quadrature generator, retained by value.
   * @param traction Configured penalty traction, retained by value.
   * @param residual_outputs Configured output policy, retained by value.
   * @param pair_builder Owned pair search composition.
   */
  CommonPlaneFormulation( PairRuleGenerator rule_generator, TractionFunction traction, ResidualOutputs residual_outputs,
                          PairListBuilder pair_builder )
      : pair_integral_( std::move( rule_generator ), std::move( traction ) ),
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
  const PairTractionIntegral<PairRuleGenerator, TractionFunction, ResidualBasis>
      pair_integral_;  ///< Integrates the fixed Common Plane penalty traction with the configured edge basis.
  const ResidualOutputs residual_outputs_;  ///< Owns the selected assembly behavior -- residual or residual + jacobian
  PairListBuilder pair_builder_;            ///< Owned search composition.
};
#endif

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
