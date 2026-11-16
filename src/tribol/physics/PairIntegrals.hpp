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
 * point. It returns a QuadraturePointValue, whose active flag must retain points with nonzero derivatives even at zero
 * value. The point calculation chooses activation; the traction and energy integrals impose no gap or pressure law.
 * Nodal gap integration retains opening gaps and applies no contact activation.
 *
 * Supplied geometries provide Coords, spatial_dim, parametric_dim, and position(parametric_coords). Basis types supply
 * size_basis, parametric_dim, and eval(parametric_coords, values). Only the configured mappings, bases, and point
 * calculations determine the supported element types and dimensions. These integrals add no MFEM assembly dependency.
 */
#ifndef SRC_TRIBOL_PHYSICS_PAIRINTEGRALS_HPP_
#define SRC_TRIBOL_PHYSICS_PAIRINTEGRALS_HPP_

#include "tribol/assembly/PairEvaluation.hpp"

namespace tribol {

/**
 * @brief Integrate a supplied vector traction against independent bases on both geometries, with opposite signs.
 * @tparam PairRuleGenerator Generates paired parametric coordinates and physical weights from current geometries.
 * @tparam TractionFunction Fixed traction calculation supporting computePairGeometryData() and the point-call interface
 * above.
 * @tparam Basis1 Residual basis on geometry1.
 * @tparam Basis2 Residual basis on geometry2; it may differ from Basis1.
 */
template <typename PairRuleGenerator, typename TractionFunction, typename Basis1, typename Basis2 = Basis1>
class PairTractionIntegral {
 public:
  using Geometry1 = typename PairRuleGenerator::Geometry1;                      ///< First element geometry.
  using Geometry2 = typename PairRuleGenerator::Geometry2;                      ///< Second element geometry.
  using Coords1 = typename Geometry1::Coords;                                   ///< Geometry1 coordinate view.
  using Coords2 = typename Geometry2::Coords;                                   ///< Geometry2 coordinate view.
  static constexpr IndexT size_geom1_coords = Coords1::size();                  ///< Geometry1 coordinate extent.
  static constexpr IndexT size_geom2_coords = Coords2::size();                  ///< Geometry2 coordinate extent.
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;  ///< Combined coordinate extent.
  using Result =
      PairResidualValue<Basis1::size_basis, Basis2::size_basis, Geometry1::spatial_dim>;  ///< Pair residual and state.
  static constexpr IndexT size_values = Result::size_values;  ///< Flattened residual extent.
  static_assert( Geometry1::spatial_dim == Geometry2::spatial_dim );
  static_assert( Basis1::parametric_dim == Geometry1::parametric_dim );
  static_assert( Basis2::parametric_dim == Geometry2::parametric_dim );

  /** @brief Store fixed quadrature and traction calculations by value.
   * @param rule_generator Current-geometry quadrature generator.
   * @param traction_function Point traction and activation calculation.
   */
  TRIBOL_HOST_DEVICE PairTractionIntegral( PairRuleGenerator rule_generator, TractionFunction traction_function )
      : rule_generator_( rule_generator ), traction_function_( traction_function )
  {
  }

  /** @brief Integrate +N1*traction on geometry1 and -N2*traction on geometry2.
   * @param coords1 First current element coordinates.
   * @param coords2 Second current element coordinates.
   * @return Residual and quadrature status. Failed, empty, or wholly inactive rules return zero residuals and inactive
   * state. Failure status must be checked before using the output. The supplied traction sets the residual sign.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Result evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const Geometry1 geometry1( coords1 );
    const Geometry2 geometry2( coords2 );
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    Result result;
    result.quadrature_result = points.result;
    if ( points.result.status != QuadratureStatus::Success || points.result.num_required_points == 0 ) {
      return result;
    }
    const auto pair_geometry_data = traction_function_.computePairGeometryData( geometry1, geometry2 );
    for ( IndexT point = 0; point < points.result.num_required_points; ++point ) {
      const MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1{
          &points.geom1_parametric_coords( point, 0 ), Geometry1::parametric_dim };
      const MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2{
          &points.geom2_parametric_coords( point, 0 ), Geometry2::parametric_dim };
      const auto traction =
          traction_function_( geometry1.position( parametric_coords1 ), geometry2.position( parametric_coords2 ),
                              parametric_coords1, parametric_coords2, pair_geometry_data );
      if ( !traction.active ) {
        continue;
      }
      result.has_active_points = true;
      MdArray<RealT, Basis1::size_basis> basis1{};
      MdArray<RealT, Basis2::size_basis> basis2{};
      Basis1::eval( parametric_coords1, basis1.view() );
      Basis2::eval( parametric_coords2, basis2.view() );
      for ( IndexT spatial_dim = 0; spatial_dim < Geometry1::spatial_dim; ++spatial_dim ) {
        const RealT weighted_traction = points.weights[point] * traction.value[spatial_dim];
        for ( IndexT basis_index = 0; basis_index < Basis1::size_basis; ++basis_index ) {
          result.geometry1( basis_index, spatial_dim ) += basis1[basis_index] * weighted_traction;
        }
        for ( IndexT basis_index = 0; basis_index < Basis2::size_basis; ++basis_index ) {
          result.geometry2( basis_index, spatial_dim ) -= basis2[basis_index] * weighted_traction;
        }
      }
    }
    return result;
  }

 private:
  PairRuleGenerator rule_generator_;    ///< Owned quadrature generator.
  TractionFunction traction_function_;  ///< Owned point traction and activation calculation.
};

/**
 * @brief Integrate a supplied scalar energy density over the current pair overlap.
 * @tparam PairRuleGenerator Generates paired parametric coordinates and physical weights.
 * @tparam EnergyFunction Fixed energy density calculation supporting computePairGeometryData() and the point-call
 * interface above.
 */
template <typename PairRuleGenerator, typename EnergyFunction>
class PairEnergyIntegral {
 public:
  using Geometry1 = typename PairRuleGenerator::Geometry1;                      ///< First element geometry.
  using Geometry2 = typename PairRuleGenerator::Geometry2;                      ///< Second element geometry.
  using Coords1 = typename Geometry1::Coords;                                   ///< Geometry1 coordinate view.
  using Coords2 = typename Geometry2::Coords;                                   ///< Geometry2 coordinate view.
  static constexpr IndexT size_values = 1;                                      ///< Scalar pair energy.
  static constexpr IndexT size_geom1_coords = Coords1::size();                  ///< Geometry1 coordinate extent.
  static constexpr IndexT size_geom2_coords = Coords2::size();                  ///< Geometry2 coordinate extent.
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;  ///< Combined coordinate extent.

  /** @brief Store fixed quadrature and energy density calculations by value.
   * @param rule_generator Current-geometry quadrature generator.
   * @param energy_function Point energy density and activation calculation.
   */
  TRIBOL_HOST_DEVICE PairEnergyIntegral( PairRuleGenerator rule_generator, EnergyFunction energy_function )
      : rule_generator_( rule_generator ), energy_function_( energy_function )
  {
  }

  /** @brief Integrate scalar energy; coordinate derivatives include the gap, mapping, and physical weights.
   * @param coords1 First current element coordinates.
   * @param coords2 Second current element coordinates.
   * @return Energy and status. Failed, empty, or wholly inactive rules return zero energy and inactive state.
   * Failure status must be checked before using the output.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairEnergyValue evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const Geometry1 geometry1( coords1 );
    const Geometry2 geometry2( coords2 );
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    PairEnergyValue result{ 0.0, false, points.result };
    if ( points.result.status != QuadratureStatus::Success || points.result.num_required_points == 0 ) {
      return result;
    }
    const auto pair_geometry_data = energy_function_.computePairGeometryData( geometry1, geometry2 );
    for ( IndexT point = 0; point < points.result.num_required_points; ++point ) {
      const MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1{
          &points.geom1_parametric_coords( point, 0 ), Geometry1::parametric_dim };
      const MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2{
          &points.geom2_parametric_coords( point, 0 ), Geometry2::parametric_dim };
      const auto energy =
          energy_function_( geometry1.position( parametric_coords1 ), geometry2.position( parametric_coords2 ),
                            parametric_coords1, parametric_coords2, pair_geometry_data );
      if ( !energy.active ) {
        continue;
      }
      result.has_active_points = true;
      result.energy += energy.value * points.weights[point];
    }
    return result;
  }

 private:
  PairRuleGenerator rule_generator_;  ///< Owned quadrature generator.
  EnergyFunction energy_function_;    ///< Owned point energy density and activation calculation.
};

/** @brief Nodal weighted gaps and tributary areas with activation and quadrature status.
 * @tparam NumNodes Number of FE nodes carrying the nodal integrals. */
template <IndexT NumNodes>
class PairNodalGapValue {
  static_assert( NumNodes > 0 );

 public:
  static constexpr IndexT num_nodes = NumNodes;        ///< Number of FE nodes.
  static constexpr IndexT size_values = 2 * NumNodes;  ///< Weighted gaps followed by tributary areas.
  /** @brief Fields stored for each geometry1 node. */
  enum class Field : IndexT
  {
    WeightedGap,    ///< Basis-weighted gap integral.
    TributaryArea,  ///< Basis-weighted area integral.
  };

  static constexpr IndexT num_fields = 2;  ///< Number of nodal integral fields.

  /** @brief Construct zero nodal integrals with default quadrature state and no active points. */
  TRIBOL_HOST_DEVICE PairNodalGapValue() = default;

  /**
   * @brief Return a field's flattened value index.
   * @param field Nodal integral field.
   * @param node Geometry1 element-local node.
   * @return Index into values().
   */
  TRIBOL_HOST_DEVICE static constexpr IndexT valueIndex( Field field, IndexT node )
  {
    return static_cast<IndexT>( field ) * num_nodes + node;
  }

  /** @brief Return a weighted-gap index. @param node Geometry1 local node. @return Index into values(). */
  TRIBOL_HOST_DEVICE static constexpr IndexT weightedGapIndex( IndexT node )
  {
    return valueIndex( Field::WeightedGap, node );
  }

  /** @brief Return a tributary-area index. @param node Geometry1 local node. @return Index into values(). */
  TRIBOL_HOST_DEVICE static constexpr IndexT tributaryAreaIndex( IndexT node )
  {
    return valueIndex( Field::TributaryArea, node );
  }

  /** @brief Access a weighted gap. @param node Geometry1 local node. @return Mutable nodal integral. */
  TRIBOL_HOST_DEVICE RealT& weightedGap( IndexT node ) { return values_[weightedGapIndex( node )]; }

  /** @brief Access a weighted gap. @param node Geometry1 local node. @return Read-only nodal integral. */
  TRIBOL_HOST_DEVICE const RealT& weightedGap( IndexT node ) const { return values_[weightedGapIndex( node )]; }

  /** @brief Access a tributary area. @param node Geometry1 local node. @return Mutable nodal integral. */
  TRIBOL_HOST_DEVICE RealT& tributaryArea( IndexT node ) { return values_[tributaryAreaIndex( node )]; }

  /** @brief Access a tributary area. @param node Geometry1 local node. @return Read-only nodal integral. */
  TRIBOL_HOST_DEVICE const RealT& tributaryArea( IndexT node ) const { return values_[tributaryAreaIndex( node )]; }

  /** @brief Return the function value vector. @return Weighted gaps followed by tributary areas in node order. */
  TRIBOL_HOST_DEVICE MdArray<RealT, size_values> values() const { return values_; }

  /// Whether successful quadrature is nonempty, including opening gaps; this does not indicate contact activation.
  bool has_active_points = false;
  QuadratureResult quadrature_result{};  ///< Failed output must not be consumed.

 private:
  MdArray<RealT, size_values> values_{};  ///< Weighted gaps followed by tributary areas in local-node order.
};

/**
 * @brief Integrate weighted gaps and tributary areas against a nodal basis on geometry1, before contact activation.
 * @tparam PairRuleGenerator Generates paired parametric coordinates and physical weights.
 * @tparam GapDefinition Fixed gap supporting computePairGeometryData(geometry1,geometry2) and
 * (point1,point2,pair_geometry_data).
 * @tparam NodalBasis Scalar nodal basis on geometry1; current nodal assembly requires basis functions to match FE
 * nodes.
 */
template <typename PairRuleGenerator, typename GapDefinition, typename NodalBasis>
class PairNodalGapIntegral {
 public:
  using Geometry1 = typename PairRuleGenerator::Geometry1;      ///< First element geometry.
  using Geometry2 = typename PairRuleGenerator::Geometry2;      ///< Second element geometry.
  using Coords1 = typename Geometry1::Coords;                   ///< Geometry1 coordinate view.
  using Coords2 = typename Geometry2::Coords;                   ///< Geometry2 coordinate view.
  static constexpr IndexT num_nodes = NodalBasis::size_basis;   ///< Nodes carrying gap and area integrals.
  static constexpr IndexT size_values = 2 * num_nodes;          ///< Weighted gaps followed by areas.
  static constexpr IndexT size_geom1_coords = Coords1::size();  ///< Geometry1 coordinate extent.
  static constexpr IndexT size_geom2_coords = Coords2::size();  ///< Geometry2 coordinate extent.
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;  ///< Combined coordinate extent.
  using Result = PairNodalGapValue<num_nodes>;  ///< Pair-local weighted gaps, areas, and quadrature state.
  static_assert( NodalBasis::parametric_dim == Geometry1::parametric_dim );

  /** @brief Store fixed quadrature and gap configuration by value.
   * @param rule_generator Current-geometry quadrature generator.
   * @param gap Gap calculation and its geometry preparation.
   */
  TRIBOL_HOST_DEVICE PairNodalGapIntegral( PairRuleGenerator rule_generator, GapDefinition gap = {} )
      : rule_generator_( rule_generator ), gap_( gap )
  {
  }

  /** @brief Integrate N_i*gap and N_i without selecting the contact active set.
   * @param coords1 First current element coordinates.
   * @param coords2 Second current element coordinates.
   * @return Nodal values and status; failed or empty rules give zero values and inactive state. Successful nonempty
   * quadrature is active even for opening gaps. Failure status must be checked before using the output.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Result evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const Geometry1 geometry1( coords1 );
    const Geometry2 geometry2( coords2 );
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    Result result;
    result.quadrature_result = points.result;
    if ( points.result.status != QuadratureStatus::Success || points.result.num_required_points == 0 ) {
      return result;
    }
    const auto pair_geometry_data = gap_.computePairGeometryData( geometry1, geometry2 );
    result.has_active_points = true;
    for ( IndexT point = 0; point < points.result.num_required_points; ++point ) {
      const MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1{
          &points.geom1_parametric_coords( point, 0 ), Geometry1::parametric_dim };
      const MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2{
          &points.geom2_parametric_coords( point, 0 ), Geometry2::parametric_dim };
      MdArray<RealT, num_nodes> basis{};
      NodalBasis::eval( parametric_coords1, basis.view() );
      const RealT gap = gap_( geometry1.position( parametric_coords1 ), geometry2.position( parametric_coords2 ),
                              pair_geometry_data );
      for ( IndexT node = 0; node < num_nodes; ++node ) {
        const RealT area_contribution = points.weights[point] * basis[node];
        result.weightedGap( node ) += area_contribution * gap;
        result.tributaryArea( node ) += area_contribution;
      }
    }
    return result;
  }

 private:
  PairRuleGenerator rule_generator_;  ///< Owned quadrature generator.
  GapDefinition gap_;                 ///< Owned gap calculation.
};

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_PAIRINTEGRALS_HPP_
