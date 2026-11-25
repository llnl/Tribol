// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Quadrature-point Energy Mortar pair energy and contact formulation. */
#ifndef SRC_TRIBOL_PHYSICS_QUADRATUREPOINTENERGYMORTAR_HPP_
#define SRC_TRIBOL_PHYSICS_QUADRATUREPOINTENERGYMORTAR_HPP_

#include "tribol/config.hpp"
#include "tribol/geom/OpposingNormalGap.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
#include "tribol/assembly/AssemblePairs.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/search/PairListBuilder.hpp"
#include <utility>
#endif

namespace tribol {

/** @brief Quadratic compressive penalty energy density using the configured gap.
 * @tparam GapDefinition Geometric or offset gap supporting per-pair geometry preparation.
 */
template <typename GapDefinition>
struct PenaltyEnergyDensity {
  RealT penalty;        ///< Fixed effective penalty stiffness.
  GapDefinition gap{};  ///< Owned gap configuration.

  /** @brief Compute the gap's geometry data once for the current pair.
   * @tparam Geometry1 First geometry type.
   * @tparam Geometry2 Second geometry type.
   * @param geometry1 First current geometry.
   * @param geometry2 Second current geometry.
   * @return Geometry data used by the configured gap.
   */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto computePairGeometryData( const Geometry1& geometry1,
                                                                        const Geometry2& geometry2 ) const
  {
    return gap.computePairGeometryData( geometry1, geometry2 );
  }

  /** @brief Compute 0.5*penalty*gap^2; zero gap retains the compressive-side derivative.
   * @tparam Point1 Geometry1 physical point type.
   * @tparam Point2 Geometry2 physical point type.
   * @tparam ParametricCoords1 Geometry1 parametric coordinate view type.
   * @tparam ParametricCoords2 Geometry2 parametric coordinate view type.
   * @tparam GeometryData Data prepared by the gap for this pair.
   * @param point1 Physical point on geometry1.
   * @param point2 Physical point on geometry2.
   * @param parametric_coords1 Geometry1 parametric point, unused by this energy law.
   * @param parametric_coords2 Geometry2 parametric point, unused by this energy law.
   * @param pair_geometry_data Current gap geometry data.
   * @return Energy density and active=true for nonpositive gap; zero density and inactive state for opening.
   */
  template <typename Point1, typename Point2, typename ParametricCoords1, typename ParametricCoords2,
            typename GeometryData>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadraturePointValue<RealT> computeEnergyAtQuadraturePoint(
      const Point1& point1, const Point2& point2, ParametricCoords1 TRIBOL_UNUSED_PARAM( parametric_coords1 ),
      ParametricCoords2 TRIBOL_UNUSED_PARAM( parametric_coords2 ), const GeometryData& pair_geometry_data ) const
  {
    const RealT separation = gap.computeGap( point1, point2, pair_geometry_data );
    if ( separation > 0.0 ) {
      return {};
    }
    return { 0.5 * penalty * separation * separation, true };
  }
};

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )

/**
 * @brief Quadrature-point Energy Mortar with energy, residual, and residual-Jacobian assembly.
 *
 * Uses PairIntegral with EnergyIntegrand and PenaltyEnergyDensity; the residual and its coordinate Jacobian are energy
 * derivatives.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairRuleGenerator Generates quadrature points and physical weights for the current edge pair.
 * @tparam GapDefinition Geometric or offset alignment-corrected gap.
 */
template <ExecutionMode Exec, typename PairRuleGenerator, typename GapDefinition = OpposingNormalGap>
class QuadraturePointEnergyMortar final : public ContactFormulation {
 public:
  /**
   * @brief Construct the penalty energy integral, own the search, and reference vector and matrix transfer data.
   * @param mesh_data Displacement transfer data; must outlive this formulation.
   * @param jacobian_data Coordinate-matrix transfer data; must outlive this formulation.
   * @param rule_generator Configured pair quadrature generator, retained by value.
   * @param penalty Fixed effective penalty stiffness.
   * @param gap Configured alignment-corrected gap, retained by value.
   * @param pair_builder Owned pair search.
   */
  QuadraturePointEnergyMortar( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                               PairRuleGenerator rule_generator, RealT penalty, GapDefinition gap,
                               PairListBuilder pair_builder )
      : mesh_data_( mesh_data ),
        jacobian_data_( jacobian_data ),
        pair_integral_( std::move( rule_generator ),
                        EnergyIntegrand<PenaltyEnergyDensity<GapDefinition>>{ { penalty, std::move( gap ) } } ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Evaluate pair energies and assemble energy, residual, and coordinate Jacobian.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @param inputs Unused; this formulation reads no caller-supplied fields.
   * @return Owned energy, residual, and df_dx; other outputs are absent.
   */
  ContactFormulationResult computeNodalResiduals(
      MeshData::Viewer mesh1, MeshData::Viewer mesh2,
      [[maybe_unused]] const ContactFormulationInputs& inputs ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return assembleEnergyResidualAndJacobian<Exec>( pairs, mesh1, mesh2, mesh_data_, jacobian_data_, pair_integral_ );
  }

 private:
  const MfemMeshData& mesh_data_;          ///< Displacement transfer data referenced for this formulation's lifetime.
  const MfemJacobianData& jacobian_data_;  ///< Matrix transfer data referenced for this formulation's lifetime.
  const PairIntegral<PairRuleGenerator, EnergyIntegrand<PenaltyEnergyDensity<GapDefinition>>>
      pair_integral_;             ///< Integrates the fixed compressive penalty energy density.
  PairListBuilder pair_builder_;  ///< Owned pair search.
};

#endif

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_QUADRATUREPOINTENERGYMORTAR_HPP_
