// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Direct Energy Mortar quadrature and explicit nodal penalty energy partials. */
#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTARENERGY_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTARENERGY_HPP_

#include "tribol/geom/OpposingNormalGap.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"
#include "tribol/assembly/PairEvaluation.hpp"

namespace tribol {
/** @brief Energy and partials with respect to one assembled weighted gap and tributary area. */
struct EnergyMortarNodalPartials {
  RealT energy = 0.0;     ///< Nodal penalty energy.
  RealT gap = 0.0;        ///< First weighted-gap partial, also the nodal pressure.
  RealT area = 0.0;       ///< First tributary-area partial.
  RealT gap_gap = 0.0;    ///< Second weighted-gap partial.
  RealT gap_area = 0.0;   ///< Mixed weighted-gap/area partial.
  RealT area_area = 0.0;  ///< Second tributary-area partial.
  /**
   * @brief Calculate penalty energy and its field partials before coordinate assembly.
   * @param weighted_gap Assembled gap integral before activation.
   * @param tributary_area Assembled tributary area.
   * @param penalty Fixed effective stiffness.
   * @param area_tolerance Maximum unsupported absolute area.
   * @return Zero values and derivatives for unsupported area or open contact. At zero gap the compressive tangent
   * remains active. Division is performed only after checking area support.
   */
  TRIBOL_HOST_DEVICE static EnergyMortarNodalPartials evaluate( RealT weighted_gap, RealT tributary_area, RealT penalty,
                                                                RealT area_tolerance )
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
 * @brief Integrate quadrature-point penalty energy from two current element geometries.
 * @tparam PairRuleGenerator Smoothed projected-edge pair rule generator.
 * @tparam GapDefinition Geometric or offset gap callable.
 */
template <typename PairRuleGenerator, typename GapDefinition = OpposingNormalGap>
class EnergyMortarEnergy {
 public:
  using FirstGeometry = typename PairRuleGenerator::FirstGeometry;    ///< First element geometry.
  using SecondGeometry = typename PairRuleGenerator::SecondGeometry;  ///< Second element geometry.
  using PrimalResult = PairEnergyValue;                               ///< Energy, activation, and quadrature status.
  static constexpr IndexT value_ct = 1;                               ///< Scalar pair energy output.
  static constexpr IndexT geom1_coord_ct = FirstGeometry::Coords::size();   ///< Geometry1 coordinate count.
  static constexpr IndexT geom2_coord_ct = SecondGeometry::Coords::size();  ///< Geometry2 coordinate count.
  static constexpr IndexT coord_ct = geom1_coord_ct + geom2_coord_ct;       ///< Combined coordinate count.
  /** @brief Store fixed inputs. @param rule_generator Pair quadrature rule generator. @param penalty Stiffness. @param
   * gap Gap callable. */
  TRIBOL_HOST_DEVICE EnergyMortarEnergy( PairRuleGenerator rule_generator, RealT penalty, GapDefinition gap = {} )
      : rule_generator_( rule_generator ), penalty_( penalty ), gap_( gap )
  {
  }
  /**
   * @brief Integrate energy, retaining zero-gap activation for differentiation.
   * @param geometry1 First current element geometry.
   * @param geometry2 Second current element geometry.
   * @return Energy and status. Failed, empty, or open rules give zero energy; failure must be reported by the caller.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PrimalResult evaluate( const FirstGeometry& geometry1,
                                                                 const SecondGeometry& geometry2 ) const
  {
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    PrimalResult result{ 0.0, false, points.result };
    if ( points.result.status != QuadratureStatus::Success || points.result.required_point_count == 0 ) {
      return result;
    }
    const auto normal1 = geometry1.normal(), normal2 = geometry2.normal();
    for ( IndexT q = 0; q < points.result.required_point_count; ++q ) {
      const RealT gap = gap_( geometry1.position( points.geometry1_coords( q, 0 ) ),
                              geometry2.position( points.geometry2_coords( q, 0 ) ), normal1, normal2 );
      if ( gap > 0.0 ) {
        continue;
      }
      result.has_active_points = true;
      result.energy += 0.5 * penalty_ * gap * gap * points.weights[q];
    }
    return result;
  }

 private:
  PairRuleGenerator rule_generator_;  ///< Owned pair quadrature rule generator.
  RealT penalty_;                     ///< Fixed effective stiffness.
  GapDefinition gap_;                 ///< Owned gap configuration.
};

/**
 * @brief Integrate nodal weighted gaps and areas from two current element geometries.
 * @tparam PairRuleGenerator Smoothed projected-edge pair rule generator.
 * @tparam GapDefinition Geometric or offset gap callable.
 */
template <typename PairRuleGenerator, typename GapDefinition = OpposingNormalGap>
class EnergyMortarNodalEnergy {
 public:
  using FirstGeometry = typename PairRuleGenerator::FirstGeometry;          ///< First element geometry.
  using SecondGeometry = typename PairRuleGenerator::SecondGeometry;        ///< Second element geometry.
  using Basis = BasisFunction<EDGE, LINEAR_LAGRANGE>;                       ///< Non-mortar nodal basis.
  static constexpr IndexT node_ct = Basis::node_count;                      ///< Non-mortar nodes per edge.
  static constexpr IndexT value_ct = 2 * node_ct;                           ///< Weighted gaps followed by areas.
  static constexpr IndexT geom1_coord_ct = FirstGeometry::Coords::size();   ///< Geometry1 coordinate count.
  static constexpr IndexT geom2_coord_ct = SecondGeometry::Coords::size();  ///< Geometry2 coordinate count.
  static constexpr IndexT coord_ct = geom1_coord_ct + geom2_coord_ct;       ///< Combined coordinate count.
  /** @brief Nodal values with activation and quadrature status. */
  struct PrimalResult {
    MdArray<RealT, value_ct> value{};      ///< Weighted gaps, then areas, in local-node order.
    bool has_active_points = false;        ///< Whether quadrature contains any integration points.
    QuadratureResult quadrature_result{};  ///< Failed output must not be consumed.

    /** @brief Return the function value vector. @return Weighted gaps followed by tributary areas in local-node order.
     */
    TRIBOL_HOST_DEVICE MdArray<RealT, value_ct> values() const { return value; }
  };
  /** @brief Store fixed inputs. @param rule_generator Pair quadrature rule generator. @param gap Gap callable. */
  TRIBOL_HOST_DEVICE EnergyMortarNodalEnergy( PairRuleGenerator rule_generator, GapDefinition gap = {} )
      : rule_generator_( rule_generator ), gap_( gap )
  {
  }
  /**
   * @brief Integrate both nodal quantities in the same quadrature loop, before nodal contact activation.
   * @param geometry1 First current element geometry.
   * @param geometry2 Second current element geometry.
   * @return Nodal values and status; failed or empty rules give zeros. Failure must be reported by the caller.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PrimalResult evaluate( const FirstGeometry& geometry1,
                                                                 const SecondGeometry& geometry2 ) const
  {
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    PrimalResult result{ {}, false, points.result };
    if ( points.result.status != QuadratureStatus::Success || points.result.required_point_count == 0 ) {
      return result;
    }
    const auto normal1 = geometry1.normal(), normal2 = geometry2.normal();
    result.has_active_points = true;
    for ( IndexT q = 0; q < points.result.required_point_count; ++q ) {
      MdArray<RealT, node_ct> basis{};
      Basis::eval( { &points.geometry1_coords( q, 0 ), Basis::parametric_dimension }, basis.view() );
      const RealT gap = gap_( geometry1.position( points.geometry1_coords( q, 0 ) ),
                              geometry2.position( points.geometry2_coords( q, 0 ) ), normal1, normal2 );
      for ( IndexT node = 0; node < node_ct; ++node ) {
        const RealT measure = points.weights[q] * basis[node];
        result.value[node] += measure * gap;
        result.value[node_ct + node] += measure;
      }
    }
    return result;
  }

 private:
  PairRuleGenerator rule_generator_;  ///< Owned pair quadrature rule generator.
  GapDefinition gap_;                 ///< Owned gap configuration.
};
}  // namespace tribol
#endif
