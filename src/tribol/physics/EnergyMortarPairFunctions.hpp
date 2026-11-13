// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Direct Energy Mortar quadrature and explicit nodal penalty energy partials. */
#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTARPAIRFUNCTIONS_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTARPAIRFUNCTIONS_HPP_

#include "tribol/geom/OpposingNormalGap.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"
#include "tribol/assembly/PairEvaluation.hpp"

namespace tribol {
/** @brief Energy and partials with respect to one assembled weighted gap and tributary area. */
struct EnergyMortarNodalPenaltyPartials {
  RealT energy = 0.0;                                  ///< Nodal penalty energy.
  RealT denergy_dweighted_gap = 0.0;                   ///< First weighted-gap partial, also the nodal pressure.
  RealT denergy_dtributary_area = 0.0;                 ///< First tributary-area partial.
  RealT d2energy_dweighted_gap2 = 0.0;                 ///< Second weighted-gap partial.
  RealT d2energy_dweighted_gap_dtributary_area = 0.0;  ///< Mixed weighted-gap/area partial.
  RealT d2energy_dtributary_area2 = 0.0;               ///< Second tributary-area partial.
  /**
   * @brief Calculate penalty energy and its field partials before coordinate assembly.
   * @param weighted_gap Assembled gap integral before activation.
   * @param tributary_area Assembled tributary area.
   * @param penalty Fixed effective stiffness.
   * @param area_tolerance Maximum unsupported absolute area.
   * @return Zero values and derivatives for unsupported area or open contact. At zero gap the compressive tangent
   * remains active. Division is performed only after checking area support.
   */
  TRIBOL_HOST_DEVICE static EnergyMortarNodalPenaltyPartials evaluate( RealT weighted_gap, RealT tributary_area,
                                                                       RealT penalty, RealT area_tolerance )
  {
    if ( std::abs( tributary_area ) <= area_tolerance || weighted_gap > 0.0 ) {
      return {};
    }
    const RealT inverse_area = 1.0 / tributary_area;
    const RealT pressure = penalty * weighted_gap * inverse_area;
    return { 0.5 * pressure * weighted_gap,
             pressure,
             -0.5 * pressure * weighted_gap * inverse_area,
             penalty * inverse_area,
             -pressure * inverse_area,
             pressure * weighted_gap * inverse_area * inverse_area };
  }
};

/**
 * @brief Integrate quadrature-point penalty energy from two current element coordinate views.
 * @tparam PairRuleGenerator Smoothed projected-edge pair rule generator.
 * @tparam GapDefinition Geometric or offset gap callable.
 */
template <typename PairRuleGenerator, typename GapDefinition = OpposingNormalGap>
class EnergyMortarEnergyPairFunction {
 public:
  using Geometry1 = typename PairRuleGenerator::Geometry1;                      ///< Geometry1 type.
  using Geometry2 = typename PairRuleGenerator::Geometry2;                      ///< Geometry2 type.
  using Coords1 = typename Geometry1::Coords;                                   ///< Geometry1 coordinate view.
  using Coords2 = typename Geometry2::Coords;                                   ///< Geometry2 coordinate view.
  static constexpr IndexT size_values = 1;                                      ///< Scalar pair energy output.
  static constexpr IndexT size_geom1_coords = Coords1::size();                  ///< Geometry1 coordinate count.
  static constexpr IndexT size_geom2_coords = Coords2::size();                  ///< Geometry2 coordinate count.
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;  ///< Combined coordinate count.
  /** @brief Store fixed inputs. @param rule_generator Pair quadrature rule generator. @param penalty Stiffness. @param
   * gap Gap callable. */
  TRIBOL_HOST_DEVICE EnergyMortarEnergyPairFunction( PairRuleGenerator rule_generator, RealT penalty,
                                                     GapDefinition gap = {} )
      : rule_generator_( rule_generator ), penalty_( penalty ), gap_( gap )
  {
  }
  /**
   * @brief Integrate energy, retaining zero-gap activation for differentiation.
   * A nonpositive quadrature-point gap sets has_active_points; zero gap retains the compressive tangent.
   * @param coords1 First current element coordinates.
   * @param coords2 Second current element coordinates.
   * @return Energy and status. Failed, empty, or open rules give zero energy; failure must be reported by the caller.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairEnergyValue evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const Geometry1 geometry1( coords1 );
    const Geometry2 geometry2( coords2 );
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    PairEnergyValue pair_energy_data{ 0.0, false, points.result };
    if ( points.result.status != QuadratureStatus::Success || points.result.num_required_points == 0 ) {
      return pair_energy_data;
    }
    const auto normal1 = geometry1.normal(), normal2 = geometry2.normal();
    for ( IndexT point = 0; point < points.result.num_required_points; ++point ) {
      const RealT gap = gap_( geometry1.position( points.geom1_parametric_coords( point, 0 ) ),
                              geometry2.position( points.geom2_parametric_coords( point, 0 ) ), normal1, normal2 );
      if ( gap > 0.0 ) {
        continue;
      }
      pair_energy_data.has_active_points = true;
      pair_energy_data.energy += 0.5 * penalty_ * gap * gap * points.weights[point];
    }
    return pair_energy_data;
  }

 private:
  PairRuleGenerator rule_generator_;  ///< Owned pair quadrature rule generator.
  RealT penalty_;                     ///< Fixed effective stiffness.
  GapDefinition gap_;                 ///< Owned gap configuration.
};

/**
 * @brief Integrate nodal weighted gaps and areas from two current element coordinate views.
 * @tparam PairRuleGenerator Smoothed projected-edge pair rule generator.
 * @tparam GapDefinition Geometric or offset gap callable.
 */
template <typename PairRuleGenerator, typename GapDefinition = OpposingNormalGap>
class EnergyMortarNodalIntegralPairFunction {
 public:
  using Geometry1 = typename PairRuleGenerator::Geometry1;        ///< Geometry1 type.
  using Geometry2 = typename PairRuleGenerator::Geometry2;        ///< Geometry2 type.
  using Coords1 = typename Geometry1::Coords;                     ///< Geometry1 coordinate view.
  using Coords2 = typename Geometry2::Coords;                     ///< Geometry2 coordinate view.
  using NonmortarBasis = BasisFunction<EDGE, LINEAR_LAGRANGE>;    ///< Basis for non-mortar nodal integrals.
  static constexpr IndexT num_nodes = NonmortarBasis::num_nodes;  ///< Number of non-mortar FE nodes per edge.
  static constexpr IndexT size_values = 2 * num_nodes;            ///< Weighted gaps followed by areas.
  static constexpr IndexT size_geom1_coords = Coords1::size();    ///< Geometry1 coordinate count.
  static constexpr IndexT size_geom2_coords = Coords2::size();    ///< Geometry2 coordinate count.
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;  ///< Combined coordinate count.
  /** @brief Nodal weighted gaps and tributary areas with activation and quadrature status. */
  class NodalIntegralValue {
   public:
    /** @brief Fields stored for each non-mortar node. */
    enum class Field : IndexT
    {
      WeightedGap,    ///< Basis-weighted gap integral.
      TributaryArea,  ///< Basis-weighted area integral.
    };

    static constexpr IndexT num_fields = 2;  ///< Number of nodal integral fields.

    /** @brief Construct zero nodal integrals with default quadrature state and no active points. */
    TRIBOL_HOST_DEVICE NodalIntegralValue() = default;

    /**
     * @brief Return a field's flattened value index.
     * @param field Nodal integral field.
     * @param node Non-mortar element-local node.
     * @return Index into values().
     */
    TRIBOL_HOST_DEVICE static constexpr IndexT valueIndex( Field field, IndexT node )
    {
      return static_cast<IndexT>( field ) * num_nodes + node;
    }

    /** @brief Return a weighted-gap index. @param node Non-mortar local node. @return Index into values(). */
    TRIBOL_HOST_DEVICE static constexpr IndexT weightedGapIndex( IndexT node )
    {
      return valueIndex( Field::WeightedGap, node );
    }

    /** @brief Return a tributary-area index. @param node Non-mortar local node. @return Index into values(). */
    TRIBOL_HOST_DEVICE static constexpr IndexT tributaryAreaIndex( IndexT node )
    {
      return valueIndex( Field::TributaryArea, node );
    }

    /** @brief Access a weighted gap. @param node Non-mortar local node. @return Mutable nodal integral. */
    TRIBOL_HOST_DEVICE RealT& weightedGap( IndexT node ) { return values_[weightedGapIndex( node )]; }

    /** @brief Access a weighted gap. @param node Non-mortar local node. @return Read-only nodal integral. */
    TRIBOL_HOST_DEVICE const RealT& weightedGap( IndexT node ) const { return values_[weightedGapIndex( node )]; }

    /** @brief Access a tributary area. @param node Non-mortar local node. @return Mutable nodal integral. */
    TRIBOL_HOST_DEVICE RealT& tributaryArea( IndexT node ) { return values_[tributaryAreaIndex( node )]; }

    /** @brief Access a tributary area. @param node Non-mortar local node. @return Read-only nodal integral. */
    TRIBOL_HOST_DEVICE const RealT& tributaryArea( IndexT node ) const { return values_[tributaryAreaIndex( node )]; }

    /** @brief Return the function value vector. @return Weighted gaps followed by tributary areas in node order. */
    TRIBOL_HOST_DEVICE MdArray<RealT, size_values> values() const { return values_; }

    /// Whether successful quadrature is nonempty, including opening gaps; this does not indicate contact activation.
    bool has_active_points = false;
    QuadratureResult quadrature_result{};  ///< Failed output must not be consumed.

   private:
    MdArray<RealT, size_values> values_{};  ///< Weighted gaps followed by tributary areas in local-node order.
  };
  using Result = NodalIntegralValue;  ///< Pair-local nodal integrals and state.
  /** @brief Store fixed inputs. @param rule_generator Pair quadrature rule generator. @param gap Gap callable. */
  TRIBOL_HOST_DEVICE EnergyMortarNodalIntegralPairFunction( PairRuleGenerator rule_generator, GapDefinition gap = {} )
      : rule_generator_( rule_generator ), gap_( gap )
  {
  }
  /**
   * @brief Integrate both nodal quantities in the same quadrature loop, before nodal contact activation.
   * Nonempty successful quadrature sets has_active_points even for opening gaps. Nodal enforcement selects the
   * contact active set after assembly.
   * @param coords1 First current element coordinates.
   * @param coords2 Second current element coordinates.
   * @return Nodal values and status; failed or empty rules give zeros. Failure must be reported by the caller.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Result evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const Geometry1 geometry1( coords1 );
    const Geometry2 geometry2( coords2 );
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    Result nodal_integral_data;
    nodal_integral_data.quadrature_result = points.result;
    if ( points.result.status != QuadratureStatus::Success || points.result.num_required_points == 0 ) {
      return nodal_integral_data;
    }
    const auto normal1 = geometry1.normal(), normal2 = geometry2.normal();
    nodal_integral_data.has_active_points = true;
    for ( IndexT point = 0; point < points.result.num_required_points; ++point ) {
      MdArray<RealT, num_nodes> basis{};
      NonmortarBasis::eval( { &points.geom1_parametric_coords( point, 0 ), NonmortarBasis::parametric_dim },
                            basis.view() );
      const RealT gap = gap_( geometry1.position( points.geom1_parametric_coords( point, 0 ) ),
                              geometry2.position( points.geom2_parametric_coords( point, 0 ) ), normal1, normal2 );
      for ( IndexT node = 0; node < num_nodes; ++node ) {
        const RealT area_contribution = points.weights[point] * basis[node];
        nodal_integral_data.weightedGap( node ) += area_contribution * gap;
        nodal_integral_data.tributaryArea( node ) += area_contribution;
      }
    }
    return nodal_integral_data;
  }

 private:
  PairRuleGenerator rule_generator_;  ///< Owned pair quadrature rule generator.
  GapDefinition gap_;                 ///< Owned gap configuration.
};
}  // namespace tribol
#endif
