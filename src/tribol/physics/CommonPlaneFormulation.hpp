// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Opt-in composed Common Plane contact formulation, separate from the legacy response path. */
#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_

#include <cmath>
#include <optional>
#include <utility>

#include "tribol/config.hpp"
#include "tribol/geom/CommonPlaneGap.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/assembly/AssemblePairs.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/timestep/TimestepCalculator.hpp"
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
    const auto geom = computeNormalAndGap( point1, point2, parametric_coords1, parametric_coords2, pair_geometry_data );
    if ( !geom.valid || geom.scalar_gap > 0.0 ) {
      return {};
    }
    QuadraturePointValue<MdArray<RealT, Geometry1::spatial_dim>> result;
    result.active = true;
    const RealT pressure = penalty * geom.scalar_gap;
    for ( IndexT d = 0; d < Geometry1::spatial_dim; ++d ) {
      result.value[d] = pressure * geom.common_plane_normal[d];
    }
    return result;
  }

  /**
   * @brief Compute the contact normal, contact activation, and penalty stiffness at one quadrature point for
   * TimestepCalculator::computeForPairIntegral().
   *
   * Unlike computeTractionAtQuadraturePoint(), separated points (scalar_gap > 0) still return valid = true with
   * in_contact = false so the velocity-projection criterion can limit dt before approaching surfaces interpenetrate.
   *
   * @tparam Geometry1 First surface-element geometry type.
   * @tparam Geometry2 Second surface-element geometry type.
   * @param point1 Physical point on geometry1.
   * @param point2 Physical point on geometry2.
   * @param parametric_coords1 Element parametric coordinate on geometry1.
   * @param parametric_coords2 Element parametric coordinate on geometry2.
   * @param pair_geometry_data Element geometries retained for local normal evaluation.
   * @return Contact normal, contact activation, and penalty stiffness per area.
   */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE QuadraturePointTimestepData computeTimestepDataAtQuadraturePoint(
      const typename Geometry1::Point& point1, const typename Geometry2::Point& point2,
      MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1,
      MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2,
      const PairGeometryData<Geometry1, Geometry2>& pair_geometry_data ) const
  {
    static_assert( Geometry1::spatial_dim == Geometry2::spatial_dim );
    QuadraturePointTimestepData qp_data;
    const auto geom = computeNormalAndGap( point1, point2, parametric_coords1, parametric_coords2, pair_geometry_data );
    if ( !geom.valid ) {
      return qp_data;
    }
    qp_data.valid = true;
    qp_data.in_contact = geom.scalar_gap <= 0.0;
    qp_data.penalty_stiffness = penalty;
    for ( IndexT d = 0; d < Geometry1::spatial_dim; ++d ) {
      qp_data.contact_normal[d] = geom.common_plane_normal[d];
    }
    return qp_data;
  }

 private:
  /** @brief Averaged common-plane unit normal and scalar gap at one paired quadrature point. */
  template <typename Geometry1>
  struct PointNormalAndGap {
    bool valid{ false };  ///< True when both normals exist, oppose each other, and average cleanly.
    typename Geometry1::Vector common_plane_normal{};  ///< Unit normal in the normal2 - normal1 direction.
    RealT scalar_gap{ 0.0 };  ///< Signed gap along common_plane_normal (<= 0 means in contact).
  };

  /**
   * @brief Compute the averaged common-plane normal and scalar gap at one quadrature point.
   *
   * Shared by computeTractionAtQuadraturePoint() and computeTimestepDataAtQuadraturePoint() so traction evaluation and
   * timestep voting use the exact same opposing-normal check, averaged normal, and gap definition.
   *
   * @tparam Geometry1 First surface-element geometry type.
   * @tparam Geometry2 Second surface-element geometry type.
   * @param point1 Physical point on geometry1.
   * @param point2 Physical point on geometry2.
   * @param parametric_coords1 Element parametric coordinate on geometry1.
   * @param parametric_coords2 Element parametric coordinate on geometry2.
   * @param pair_geometry_data Element geometries retained for local normal evaluation.
   * @return Common-plane normal and gap, with valid = false if surface normals are degenerate or do not oppose each
   * other.
   */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PointNormalAndGap<Geometry1> computeNormalAndGap(
      const typename Geometry1::Point& point1, const typename Geometry2::Point& point2,
      MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1,
      MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2,
      const PairGeometryData<Geometry1, Geometry2>& pair_geometry_data ) const
  {
    PointNormalAndGap<Geometry1> out;
    typename Geometry1::Vector normal1;
    typename Geometry2::Vector normal2;
    if ( !pair_geometry_data.geometry1.unitNormal( parametric_coords1, normal1 ) ||
         !pair_geometry_data.geometry2.unitNormal( parametric_coords2, normal2 ) || normal1.dot( normal2 ) >= 0.0 ) {
      return out;
    }
    out.common_plane_normal = normal2 - normal1;
    const RealT normal_length = std::sqrt( out.common_plane_normal.dot( out.common_plane_normal ) );
    if ( !std::isfinite( normal_length ) || normal_length <= 0.0 ) {
      return out;
    }
    out.common_plane_normal /= normal_length;
    out.scalar_gap = gap.computeGap( point1, point2, out.common_plane_normal );
    out.valid = true;
    return out;
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
   * @brief Construct the penalty traction integral and own the output policy, pair search, and optional timestep
   * settings.
   * @param rule_generator Configured pair quadrature generator, retained by value.
   * @param penalty Fixed effective penalty stiffness.
   * @param gap Configured common-plane gap, retained by value.
   * @param residual_outputs Configured output policy, retained by value.
   * @param pair_builder Owned pair search composition.
   * @param timestep_settings Optional timestep voting settings (std::nullopt disables timestep voting).
   * @param parent_cfl_data MFEM transfer and mass data for nodal penalty CFL with LOR geometry, when needed.
   */
  CommonPlaneFormulation( PairRuleGenerator rule_generator, RealT penalty, GapDefinition gap,
                          ResidualOutputs residual_outputs, PairListBuilder pair_builder,
                          std::optional<TimestepSettings> timestep_settings = std::nullopt,
                          std::optional<detail::MfemParentPenaltyCflData> parent_cfl_data = std::nullopt )
      : pair_integral_( std::move( rule_generator ), { penalty, std::move( gap ) } ),
        residual_outputs_( std::move( residual_outputs ) ),
        pair_builder_( std::move( pair_builder ) ),
        timestep_settings_( timestep_settings ),
        parent_cfl_data_( parent_cfl_data )
  {
  }

  /**
   * @brief Compute the Common Plane residual outputs selected by the policy, plus an allowable timestep when enabled.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @param inputs Optional incoming simulation timestep used by velocity-projection voting.
   * @return Owned residual outputs selected by ResidualOutputs and optional time_step; other outputs are absent.
   */
  ContactFormulationResult computeNodalResiduals( MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                  const ContactFormulationInputs& inputs ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    auto result = residual_outputs_.template computeResidual<Exec>( pairs, mesh1, mesh2, pair_integral_ );
    if ( timestep_settings_.has_value() ) {
      result.time_step = TimestepCalculator( *timestep_settings_ )
                             .template compute<Exec>( pairs, mesh1, mesh2, pair_integral_, inputs.dt, false,
                                                      parent_cfl_data_ ? &*parent_cfl_data_ : nullptr );
    }
    return result;
  }

 private:
  const PairTractionIntegral<PairRuleGenerator, CommonPlanePenaltyTraction<GapDefinition>,
                             ResidualBasis>
      pair_integral_;  ///< Integrates the fixed Common Plane penalty traction with the configured edge basis.
  const ResidualOutputs residual_outputs_;  ///< Owns the selected assembly behavior -- residual or residual + jacobian
  PairListBuilder pair_builder_;            ///< Owned search composition.
  std::optional<TimestepSettings> timestep_settings_;  ///< Present when allowable timestep voting is enabled.
  std::optional<detail::MfemParentPenaltyCflData>
      parent_cfl_data_;  ///< Present for nodal penalty CFL with LOR geometry.
};
#endif

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
