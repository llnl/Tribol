// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Integrate pair-local traction, energy, and nodal gaps using supplied quadrature and point calculations.
 *
 * Each integral constructs current geometry inside evaluate(), so differentiation includes the geometry, mapping,
 * and physical weights. A point calculation computes pair geometry data once with computePairGeometryData(geometry1,
 * geometry2), then receives physical points, both element-parametric coordinate views, and that data at each quadrature
 * point. Traction calculations also receive the generated rule's integration-surface normal at the point. It returns a
 * QuadraturePointValue, whose active flag must retain points with nonzero derivatives even at zero value. The point
 * calculation chooses activation; the traction and energy integrals impose no gap or pressure law. Nodal gap
 * integration retains opening gaps and applies no contact activation.
 *
 * Supplied geometries provide Coords, spatial_dim, parametric_dim, and position(parametric_coords). Basis types supply
 * size_basis, parametric_dim, and eval(parametric_coords, values). Only the configured mappings, bases, and point
 * calculations determine the supported element types and dimensions. These integrals add no MFEM assembly dependency.
 */
#ifndef SRC_TRIBOL_PHYSICS_PAIRINTEGRALS_HPP_
#define SRC_TRIBOL_PHYSICS_PAIRINTEGRALS_HPP_

#include <concepts>
#include <type_traits>
#include <utility>

#include "tribol/assembly/PairFunctionData.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"

namespace tribol {

/** @brief One quadrature-point quantity (usually energy density or traction) and whether its values or derivatives must
 * be integrated.
 * @tparam T Scalar energy density or fixed-size vector traction.
 */
template <typename T>
struct QuadraturePointValue {
  T value{};            ///< Unweighted energy density or traction vector; the integral applies the physical weight.
  bool active = false;  ///< Whether to retain this point, including a zero value with a nonzero tangent.
};

namespace detail {

/** Pair geometry data prepared once and reused at every quadrature point. */
template <typename Function, typename Generator>
using PairGeometryData = std::remove_cvref_t<decltype( std::declval<const Function&>().computePairGeometryData(
    std::declval<const typename Generator::Geometry1&>(), std::declval<const typename Generator::Geometry2&>() ) )>;

}  // namespace detail

/**
 * @brief Requires pair geometry preparation and vector traction evaluation at one paired quadrature point.
 *
 * These requirements check C++ expressions; host/device execution, differentiation, activation, and traction sign must
 * be verified separately.
 * @tparam Function Fixed traction calculation.
 * @tparam Generator Pair quadrature generator defining both element geometries.
 */
template <typename Function, typename Generator>
concept PairTractionFunction =
    PairRuleGenerator<Generator> &&
    requires( const Function& function, const typename Generator::Geometry1& geometry1,
              const typename Generator::Geometry2& geometry2,
              MdSpan<const RealT, Generator::Geometry1::parametric_dim> parametric_coords1,
              MdSpan<const RealT, Generator::Geometry2::parametric_dim> parametric_coords2,
              const typename Generator::Geometry1::Vector& normal,
              const detail::PairGeometryData<Function, Generator>& pair_geometry_data ) {
      {
        function.computeTractionAtQuadraturePoint( geometry1.position( parametric_coords1 ),
                                                   geometry2.position( parametric_coords2 ), parametric_coords1,
                                                   parametric_coords2, normal, pair_geometry_data )
      } -> std::same_as<QuadraturePointValue<MdArray<RealT, Generator::Geometry1::spatial_dim>>>;
    };

/**
 * @brief Requires pair geometry preparation and scalar energy-density evaluation at one paired quadrature point.
 *
 * These requirements check C++ expressions; host/device execution, differentiation, activation, and physical units
 * must be verified separately.
 * @tparam Function Fixed energy-density calculation.
 * @tparam Generator Pair quadrature generator defining both element geometries.
 */
template <typename Function, typename Generator>
concept PairEnergyFunction = PairRuleGenerator<Generator> &&
                             requires( const Function& function, const typename Generator::Geometry1& geometry1,
                                       const typename Generator::Geometry2& geometry2,
                                       MdSpan<const RealT, Generator::Geometry1::parametric_dim> parametric_coords1,
                                       MdSpan<const RealT, Generator::Geometry2::parametric_dim> parametric_coords2,
                                       const detail::PairGeometryData<Function, Generator>& pair_geometry_data ) {
                               {
                                 function.computeEnergyAtQuadraturePoint(
                                     geometry1.position( parametric_coords1 ), geometry2.position( parametric_coords2 ),
                                     parametric_coords1, parametric_coords2, pair_geometry_data )
                               } -> std::same_as<QuadraturePointValue<RealT>>;
                             };

/**
 * @brief Requires pair geometry preparation and scalar gap evaluation at one paired quadrature point.
 *
 * Gap evaluation receives physical points and the prepared pair data but does not receive parametric coordinates.
 * These requirements check C++ expressions; host/device execution, differentiation, and the gap sign convention must
 * be verified separately.
 * @tparam Function Fixed gap calculation.
 * @tparam Generator Pair quadrature generator defining both element geometries.
 */
template <typename Function, typename Generator>
concept PairGapFunction = PairRuleGenerator<Generator> &&
                          requires( const Function& function, const typename Generator::Geometry1& geometry1,
                                    const typename Generator::Geometry2& geometry2,
                                    MdSpan<const RealT, Generator::Geometry1::parametric_dim> parametric_coords1,
                                    MdSpan<const RealT, Generator::Geometry2::parametric_dim> parametric_coords2,
                                    const detail::PairGeometryData<Function, Generator>& pair_geometry_data ) {
                            {
                              function.computeGap( geometry1.position( parametric_coords1 ),
                                                   geometry2.position( parametric_coords2 ), pair_geometry_data )
                            } -> std::same_as<RealT>;
                          };

/**
 * @brief Integrate a supplied vector traction against independent bases on both geometries, with opposite signs.
 * @tparam RuleGenerator Generates paired parametric coordinates and physical weights from current geometries.
 * @tparam TractionFunction Fixed traction calculation supporting computePairGeometryData() and the point-call interface
 * above.
 * @tparam Basis1 Residual basis on geometry1.
 * @tparam Basis2 Residual basis on geometry2; it may differ from Basis1.
 */
template <PairRuleGenerator RuleGenerator, PairTractionFunction<RuleGenerator> TractionFunction,
          FiniteElementBasis Basis1, FiniteElementBasis Basis2 = Basis1>
class PairTractionIntegral {
 public:
  using Geometry1 = typename RuleGenerator::Geometry1;  ///< First element geometry.
  using Geometry2 = typename RuleGenerator::Geometry2;  ///< Second element geometry.
  using Coords1 = typename Geometry1::Coords;           ///< Geometry1 coordinate view.
  using Coords2 = typename Geometry2::Coords;           ///< Geometry2 coordinate view.
  /**
   * @brief Number of scalar coordinates that define geometry1.
   *
   * For example, a two-vertex edge in 2D stores {x0, y0, x1, y1}, so this value is 4.
   */
  static constexpr IndexT size_geom1_coords = Coords1::size();
  /**
   * @brief Number of scalar coordinates that define geometry2.
   *
   * For example, a two-vertex edge in 2D stores {x0, y0, x1, y1}, so this value is 4.
   */
  static constexpr IndexT size_geom2_coords = Coords2::size();
  /**
   * @brief Total number of scalar coordinates that define the geometry pair.
   *
   * For example, two 2D edges each contribute four coordinates, so this value is 8. Geometry1 coordinates come first,
   * followed by geometry2 coordinates.
   */
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;
  using PairResidual =
      PairResidualValue<Basis1::size_basis, Basis2::size_basis, Geometry1::spatial_dim>;  ///< Integrated pair residual.
  static constexpr IndexT size_values = PairResidual::size_values;  ///< Flattened residual extent.
  static_assert( Geometry1::spatial_dim == Geometry2::spatial_dim );
  static_assert( Basis1::parametric_dim == Geometry1::parametric_dim );
  static_assert( Basis2::parametric_dim == Geometry2::parametric_dim );

  /** @brief Store fixed quadrature and traction calculations by value.
   * @param rule_generator Current-geometry quadrature generator.
   * @param traction_function Point traction and activation calculation.
   */
  TRIBOL_HOST_DEVICE PairTractionIntegral( RuleGenerator rule_generator, TractionFunction traction_function )
      : rule_generator_( rule_generator ), traction_function_( traction_function )
  {
  }

  /** @brief Integrate +N1*traction on geometry1 and -N2*traction on geometry2.
   * @param coords1 First current element coordinates.
   * @param coords2 Second current element coordinates.
   * @return Residual and quadrature status. Failed, empty, or wholly inactive rules return zero residuals and inactive
   * state. Failure status must be checked before using the output. The supplied traction sets the residual sign.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairResidual evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const Geometry1 geometry1( coords1 );
    const Geometry2 geometry2( coords2 );
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    PairResidual pair_residual;
    pair_residual.quadrature_metadata = points.metadata;
    if ( points.metadata.status != QuadratureStatus::Success || points.metadata.num_required_points == 0 ) {
      return pair_residual;
    }
    const auto pair_geometry_data = traction_function_.computePairGeometryData( geometry1, geometry2 );
    for ( IndexT point = 0; point < points.metadata.num_required_points; ++point ) {
      const MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1{
          &points.geom1_parametric_coords( point, 0 ), Geometry1::parametric_dim };
      const MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2{
          &points.geom2_parametric_coords( point, 0 ), Geometry2::parametric_dim };
      static_assert( std::remove_cvref_t<decltype( points )>::spatial_dim == Geometry1::spatial_dim );
      const typename Geometry1::Vector normal( &points.integration_surface_normals( point, 0 ),
                                               Geometry1::spatial_dim );
      const auto traction = traction_function_.computeTractionAtQuadraturePoint(
          geometry1.position( parametric_coords1 ), geometry2.position( parametric_coords2 ), parametric_coords1,
          parametric_coords2, normal, pair_geometry_data );
      if ( !traction.active ) {
        continue;
      }
      pair_residual.has_active_points = true;
      MdArray<RealT, Basis1::size_basis> basis1{};
      MdArray<RealT, Basis2::size_basis> basis2{};
      Basis1::eval( parametric_coords1, basis1.view() );
      Basis2::eval( parametric_coords2, basis2.view() );
      for ( IndexT d = 0; d < Geometry1::spatial_dim; ++d ) {
        const RealT weighted_traction = points.weights[point] * traction.value[d];
        for ( IndexT basis_index = 0; basis_index < Basis1::size_basis; ++basis_index ) {
          pair_residual.geometry1( basis_index, d ) += basis1[basis_index] * weighted_traction;
        }
        for ( IndexT basis_index = 0; basis_index < Basis2::size_basis; ++basis_index ) {
          pair_residual.geometry2( basis_index, d ) -= basis2[basis_index] * weighted_traction;
        }
      }
    }
    return pair_residual;
  }

  /** @brief Return the owned pair quadrature rule generator. */
  TRIBOL_HOST_DEVICE const RuleGenerator& ruleGenerator() const { return rule_generator_; }

  /** @brief Return the owned point traction calculation. */
  TRIBOL_HOST_DEVICE const TractionFunction& tractionFunction() const { return traction_function_; }

 private:
  RuleGenerator rule_generator_;        ///< Owned quadrature generator.
  TractionFunction traction_function_;  ///< Owned point traction and activation calculation.
};

/**
 * @brief Integrate a supplied scalar energy density over the current pair overlap.
 * @tparam RuleGenerator Generates paired parametric coordinates and physical weights.
 * @tparam EnergyFunction Fixed energy density calculation supporting computePairGeometryData() and the point-call
 * interface above.
 */
template <PairRuleGenerator RuleGenerator, PairEnergyFunction<RuleGenerator> EnergyFunction>
class PairEnergyIntegral {
 public:
  using Geometry1 = typename RuleGenerator::Geometry1;  ///< First element geometry.
  using Geometry2 = typename RuleGenerator::Geometry2;  ///< Second element geometry.
  using Coords1 = typename Geometry1::Coords;           ///< Geometry1 coordinate view.
  using Coords2 = typename Geometry2::Coords;           ///< Geometry2 coordinate view.
  static constexpr IndexT size_values = 1;              ///< Scalar pair energy.
  /**
   * @brief Number of scalar coordinates that define geometry1.
   *
   * For example, a two-vertex edge in 2D stores {x0, y0, x1, y1}, so this value is 4.
   */
  static constexpr IndexT size_geom1_coords = Coords1::size();
  /**
   * @brief Number of scalar coordinates that define geometry2.
   *
   * For example, a two-vertex edge in 2D stores {x0, y0, x1, y1}, so this value is 4.
   */
  static constexpr IndexT size_geom2_coords = Coords2::size();
  /**
   * @brief Total number of scalar coordinates that define the geometry pair.
   *
   * For example, two 2D edges each contribute four coordinates, so this value is 8. Geometry1 coordinates come first,
   * followed by geometry2 coordinates.
   */
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;
  using PairEnergy = PairEnergyValue;  ///< Integrated pair energy and quadrature state.

  /** @brief Store fixed quadrature and energy density calculations by value.
   * @param rule_generator Current-geometry quadrature generator.
   * @param energy_function Point energy density and activation calculation.
   */
  TRIBOL_HOST_DEVICE PairEnergyIntegral( RuleGenerator rule_generator, EnergyFunction energy_function )
      : rule_generator_( rule_generator ), energy_function_( energy_function )
  {
  }

  /** @brief Integrate scalar energy; coordinate derivatives include the gap, mapping, and physical weights.
   * @param coords1 First current element coordinates.
   * @param coords2 Second current element coordinates.
   * @return Energy and status. Failed, empty, or wholly inactive rules return zero energy and inactive state.
   * Failure status must be checked before using the output.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairEnergy evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const Geometry1 geometry1( coords1 );
    const Geometry2 geometry2( coords2 );
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    PairEnergy pair_energy{ 0.0, false, points.metadata };
    if ( points.metadata.status != QuadratureStatus::Success || points.metadata.num_required_points == 0 ) {
      return pair_energy;
    }
    const auto pair_geometry_data = energy_function_.computePairGeometryData( geometry1, geometry2 );
    for ( IndexT point = 0; point < points.metadata.num_required_points; ++point ) {
      const MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1{
          &points.geom1_parametric_coords( point, 0 ), Geometry1::parametric_dim };
      const MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2{
          &points.geom2_parametric_coords( point, 0 ), Geometry2::parametric_dim };
      const auto energy = energy_function_.computeEnergyAtQuadraturePoint(
          geometry1.position( parametric_coords1 ), geometry2.position( parametric_coords2 ), parametric_coords1,
          parametric_coords2, pair_geometry_data );
      if ( !energy.active ) {
        continue;
      }
      pair_energy.has_active_points = true;
      pair_energy.energy += energy.value * points.weights[point];
    }
    return pair_energy;
  }

 private:
  RuleGenerator rule_generator_;    ///< Owned quadrature generator.
  EnergyFunction energy_function_;  ///< Owned point energy density and activation calculation.
};

/**
 * @brief Integrate weighted gaps and tributary areas against a nodal basis on geometry1, before contact activation.
 * @tparam RuleGenerator Generates paired parametric coordinates and physical weights.
 * @tparam GapDefinition Fixed gap supporting computePairGeometryData(geometry1,geometry2) and
 * (point1,point2,pair_geometry_data).
 * @tparam NodalBasis Scalar nodal basis on geometry1; current nodal assembly requires basis functions to match FE
 * nodes.
 */
template <PairRuleGenerator RuleGenerator, PairGapFunction<RuleGenerator> GapDefinition, FiniteElementBasis NodalBasis>
class PairNodalGapIntegral {
 public:
  using Geometry1 = typename RuleGenerator::Geometry1;         ///< First element geometry.
  using Geometry2 = typename RuleGenerator::Geometry2;         ///< Second element geometry.
  using Coords1 = typename Geometry1::Coords;                  ///< Geometry1 coordinate view.
  using Coords2 = typename Geometry2::Coords;                  ///< Geometry2 coordinate view.
  static constexpr IndexT num_nodes = NodalBasis::size_basis;  ///< Nodes carrying gap and area integrals.
  static constexpr IndexT size_values = 2 * num_nodes;         ///< Weighted gaps followed by areas.
  /**
   * @brief Number of scalar coordinates that define geometry1.
   *
   * For example, a two-vertex edge in 2D stores {x0, y0, x1, y1}, so this value is 4.
   */
  static constexpr IndexT size_geom1_coords = Coords1::size();
  /**
   * @brief Number of scalar coordinates that define geometry2.
   *
   * For example, a two-vertex edge in 2D stores {x0, y0, x1, y1}, so this value is 4.
   */
  static constexpr IndexT size_geom2_coords = Coords2::size();
  /**
   * @brief Total number of scalar coordinates that define the geometry pair.
   *
   * For example, two 2D edges each contribute four coordinates, so this value is 8. Geometry1 coordinates come first,
   * followed by geometry2 coordinates.
   */
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;
  using PairNodalGap = PairNodalGapValue<num_nodes>;  ///< Integrated nodal gaps, areas, and quadrature state.
  static_assert( NodalBasis::parametric_dim == Geometry1::parametric_dim );

  /** @brief Store fixed quadrature and gap configuration by value.
   * @param rule_generator Current-geometry quadrature generator.
   * @param gap Gap calculation and its geometry preparation.
   */
  TRIBOL_HOST_DEVICE PairNodalGapIntegral( RuleGenerator rule_generator, GapDefinition gap = {} )
      : rule_generator_( rule_generator ), gap_( gap )
  {
  }

  /** @brief Integrate N_i*gap and N_i without selecting the contact active set.
   * @param coords1 First current element coordinates.
   * @param coords2 Second current element coordinates.
   * @return Nodal values and status; failed or empty rules give zero values and inactive state. Successful nonempty
   * quadrature is active even for opening gaps. Failure status must be checked before using the output.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairNodalGap evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const Geometry1 geometry1( coords1 );
    const Geometry2 geometry2( coords2 );
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    PairNodalGap pair_nodal_gap;
    pair_nodal_gap.quadrature_metadata = points.metadata;
    if ( points.metadata.status != QuadratureStatus::Success || points.metadata.num_required_points == 0 ) {
      return pair_nodal_gap;
    }
    const auto pair_geometry_data = gap_.computePairGeometryData( geometry1, geometry2 );
    pair_nodal_gap.has_active_points = true;
    for ( IndexT point = 0; point < points.metadata.num_required_points; ++point ) {
      const MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1{
          &points.geom1_parametric_coords( point, 0 ), Geometry1::parametric_dim };
      const MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2{
          &points.geom2_parametric_coords( point, 0 ), Geometry2::parametric_dim };
      MdArray<RealT, num_nodes> basis{};
      NodalBasis::eval( parametric_coords1, basis.view() );
      const RealT gap = gap_.computeGap( geometry1.position( parametric_coords1 ),
                                         geometry2.position( parametric_coords2 ), pair_geometry_data );
      for ( IndexT node = 0; node < num_nodes; ++node ) {
        const RealT area_contribution = points.weights[point] * basis[node];
        pair_nodal_gap.weightedGap( node ) += area_contribution * gap;
        pair_nodal_gap.tributaryArea( node ) += area_contribution;
      }
    }
    return pair_nodal_gap;
  }

 private:
  RuleGenerator rule_generator_;  ///< Owned quadrature generator.
  GapDefinition gap_;             ///< Owned gap calculation.
};

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_PAIRINTEGRALS_HPP_
