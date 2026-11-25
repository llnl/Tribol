// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Shared pair quadrature integration driver and composable integrands for traction, energy, nodal gap, and
 * mortar weight calculations.
 *
 * PairIntegral constructs current geometry inside evaluate(), so differentiation includes the geometry, mapping, and
 * physical integration weights. Each integrand computes per-pair geometry data once via
 * computePairGeometryData(geometry1, geometry2), declares whether any nonempty quadrature rule is active via
 * active_for_nonempty_rule, and adds each quadrature point's weighted contribution via
 * addQuadraturePointContribution(quadrature_point, pair_geometry_data, result).
 *
 * Supplied geometries provide Coords, spatial_dim, parametric_dim, and position(parametric_coords). Basis types supply
 * size_basis, parametric_dim, and eval(parametric_coords, values). These integrals add no MFEM assembly dependency.
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
  bool active = false;  ///< Whether to retain this point, including a zero value with a nonzero derivative.
};

namespace detail {

/** @brief Pair geometry data prepared once and reused at every quadrature point. */
template <typename Function, typename Generator>
using PairGeometryData = std::remove_cvref_t<decltype( std::declval<const Function&>().computePairGeometryData(
    std::declval<const typename Generator::Geometry1&>(), std::declval<const typename Generator::Geometry2&>() ) )>;

}  // namespace detail

/**
 * @brief Requires pair geometry preparation and vector traction evaluation at one paired quadrature point.
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
 * @brief Referencing view of one generated quadrature point passed to an integrand's addQuadraturePointContribution()
 * method.
 * @tparam Geometry1 First surface element geometry type.
 * @tparam Geometry2 Second surface element geometry type.
 */
template <typename Geometry1, typename Geometry2>
struct PairQuadraturePoint {
  const Geometry1& geometry1;  ///< Current geometry1; remains valid during addQuadraturePointContribution().
  const Geometry2& geometry2;  ///< Current geometry2; remains valid during addQuadraturePointContribution().
  MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1;  ///< Geometry1 element-parametric coordinates.
  MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2;  ///< Geometry2 element-parametric coordinates.
  typename Geometry1::Vector integration_surface_normal;              ///< Integration-surface unit normal at the point.
  RealT physical_weight;                                              ///< Physical integration weight at the point.
};

/**
 * @brief Integrate a supplied integrand over the quadrature points generated for one pair of current geometries.
 *
 * Constructs current geometries inside evaluate(), checks rule status and empty rules, prepares pair-level data once,
 * and accumulates every quadrature point into Result. Quadrature status is always preserved on the returned Result;
 * failed or empty rules return zero values with has_active_points set to false.
 *
 * @tparam RuleGenerator Generates paired element-parametric coordinates, physical weights, and normals.
 * @tparam Integrand Supplies Result<Geometry1, Geometry2>, active_for_nonempty_rule, computePairGeometryData(), and
 * addQuadraturePointContribution().
 */
template <PairRuleGenerator RuleGenerator, typename Integrand>
class PairIntegral {
 public:
  using Geometry1 = typename RuleGenerator::Geometry1;                       ///< First element geometry.
  using Geometry2 = typename RuleGenerator::Geometry2;                       ///< Second element geometry.
  using Coords1 = typename Geometry1::Coords;                                ///< Geometry1 coordinate view.
  using Coords2 = typename Geometry2::Coords;                                ///< Geometry2 coordinate view.
  using Result = typename Integrand::template Result<Geometry1, Geometry2>;  ///< Integrated pair value type.
  /// Number of scalar coordinates defining geometry1.
  static constexpr IndexT size_geom1_coords = Coords1::size();
  /// Number of scalar coordinates defining geometry2.
  static constexpr IndexT size_geom2_coords = Coords2::size();
  /// Total number of scalar coordinates defining the pair (geometry1 followed by geometry2).
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;
  /// Flattened scalar output extent of Result.
  static constexpr IndexT size_values = Result::size_values;
  static_assert( Geometry1::spatial_dim == Geometry2::spatial_dim );

  /**
   * @brief Store the configured rule generator and integrand by value.
   * @param rule_generator Current-geometry quadrature generator.
   * @param integrand Configured integrand calculation.
   */
  TRIBOL_HOST_DEVICE explicit PairIntegral( RuleGenerator rule_generator, Integrand integrand = Integrand{} )
      : rule_generator_( std::move( rule_generator ) ), integrand_( std::move( integrand ) )
  {
  }

  /**
   * @brief Generate quadrature from current coordinates and accumulate the integrand over all points.
   * @param coords1 Current geometry1 coordinates.
   * @param coords2 Current geometry2 coordinates.
   * @return Integrated values, activation flag, and quadrature metadata. Failed or empty rules return zero values.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Result evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const Geometry1 geometry1( coords1 );
    const Geometry2 geometry2( coords2 );
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    Result result{};
    result.quadrature_metadata = points.metadata;
    if ( points.metadata.status != QuadratureStatus::Success || points.metadata.num_required_points == 0 ) {
      return result;
    }
    const auto pair_geometry_data = integrand_.computePairGeometryData( geometry1, geometry2 );
    result.has_active_points = Integrand::active_for_nonempty_rule;
    for ( IndexT point = 0; point < points.metadata.num_required_points; ++point ) {
      const MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1{
          &points.geom1_parametric_coords( point, 0 ), Geometry1::parametric_dim };
      const MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2{
          &points.geom2_parametric_coords( point, 0 ), Geometry2::parametric_dim };
      static_assert( std::remove_cvref_t<decltype( points )>::spatial_dim == Geometry1::spatial_dim );
      const typename Geometry1::Vector normal( &points.integration_surface_normals( point, 0 ),
                                               Geometry1::spatial_dim );
      const PairQuadraturePoint<Geometry1, Geometry2> quadrature_point{
          geometry1, geometry2, parametric_coords1, parametric_coords2, normal, points.weights[point] };
      integrand_.addQuadraturePointContribution( quadrature_point, pair_geometry_data, result );
    }
    return result;
  }

  /** @brief Return the owned pair quadrature rule generator. @return Configured rule generator. */
  TRIBOL_HOST_DEVICE const RuleGenerator& ruleGenerator() const { return rule_generator_; }

  /** @brief Return the owned integrand. @return Configured integrand. */
  TRIBOL_HOST_DEVICE const Integrand& integrand() const { return integrand_; }

 private:
  RuleGenerator rule_generator_;  ///< Owned quadrature rule generator.
  Integrand integrand_;           ///< Owned integrand physics and accumulation logic.
};

/**
 * @brief Integrand accumulating +N1 * traction on geometry1 and -N2 * traction on geometry2.
 * @tparam TractionFunction Point traction and activation calculation.
 * @tparam Basis1 Residual basis on geometry1.
 * @tparam Basis2 Residual basis on geometry2; defaults to Basis1.
 */
template <typename TractionFunction, FiniteElementBasis Basis1, FiniteElementBasis Basis2 = Basis1>
class TractionIntegrand {
 public:
  /// Integrated pair residual on geometry1 and geometry2.
  template <typename Geometry1, typename Geometry2>
  using Result = PairResidualValue<Basis1::size_basis, Basis2::size_basis, Geometry1::spatial_dim>;
  /// Individual active quadrature points mark the pair active.
  static constexpr bool active_for_nonempty_rule = false;

  /** @brief Store the point traction calculation by value. @param traction_function Configured traction calculation. */
  TRIBOL_HOST_DEVICE explicit TractionIntegrand( TractionFunction traction_function = {} )
      : traction_function_( std::move( traction_function ) )
  {
  }

  /** @brief Compute pair-level geometry data once for the traction calculation. */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto computePairGeometryData( const Geometry1& geometry1,
                                                                        const Geometry2& geometry2 ) const
  {
    return traction_function_.computePairGeometryData( geometry1, geometry2 );
  }

  /** @brief Evaluate the traction at one quadrature point and add its weighted nodal residuals when active. */
  template <typename Geometry1, typename Geometry2, typename PairData>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE void addQuadraturePointContribution(
      const PairQuadraturePoint<Geometry1, Geometry2>& qp, const PairData& pair_geometry_data,
      Result<Geometry1, Geometry2>& result ) const
  {
    static_assert( Basis1::parametric_dim == Geometry1::parametric_dim );
    static_assert( Basis2::parametric_dim == Geometry2::parametric_dim );
    const auto traction = traction_function_.computeTractionAtQuadraturePoint(
        qp.geometry1.position( qp.parametric_coords1 ), qp.geometry2.position( qp.parametric_coords2 ),
        qp.parametric_coords1, qp.parametric_coords2, qp.integration_surface_normal, pair_geometry_data );
    if ( !traction.active ) {
      return;
    }
    result.has_active_points = true;
    MdArray<RealT, Basis1::size_basis> basis1{};
    MdArray<RealT, Basis2::size_basis> basis2{};
    Basis1::eval( qp.parametric_coords1, basis1.view() );
    Basis2::eval( qp.parametric_coords2, basis2.view() );
    for ( IndexT d = 0; d < Geometry1::spatial_dim; ++d ) {
      const RealT weighted_traction = qp.physical_weight * traction.value[d];
      for ( IndexT basis_index = 0; basis_index < Basis1::size_basis; ++basis_index ) {
        result.geometry1( basis_index, d ) += basis1[basis_index] * weighted_traction;
      }
      for ( IndexT basis_index = 0; basis_index < Basis2::size_basis; ++basis_index ) {
        result.geometry2( basis_index, d ) -= basis2[basis_index] * weighted_traction;
      }
    }
  }

  /** @brief Return the owned point traction calculation. @return Configured traction function. */
  TRIBOL_HOST_DEVICE const TractionFunction& tractionFunction() const { return traction_function_; }

 private:
  TractionFunction traction_function_;  ///< Owned point traction and activation calculation.
};

/**
 * @brief Integrand accumulating a scalar energy density over active quadrature points.
 * @tparam EnergyFunction Point energy density and activation calculation.
 */
template <typename EnergyFunction>
class EnergyIntegrand {
 public:
  /// Integrated scalar pair energy.
  template <typename Geometry1, typename Geometry2>
  using Result = PairEnergyValue;
  /// Individual active quadrature points mark the pair active.
  static constexpr bool active_for_nonempty_rule = false;

  /** @brief Store the point energy density calculation by value. @param energy_function Configured energy law. */
  TRIBOL_HOST_DEVICE explicit EnergyIntegrand( EnergyFunction energy_function = {} )
      : energy_function_( std::move( energy_function ) )
  {
  }

  /** @brief Compute pair-level geometry data once for the energy density calculation. */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto computePairGeometryData( const Geometry1& geometry1,
                                                                        const Geometry2& geometry2 ) const
  {
    return energy_function_.computePairGeometryData( geometry1, geometry2 );
  }

  /** @brief Evaluate the scalar energy density at one quadrature point and add its weighted value when active. */
  template <typename Geometry1, typename Geometry2, typename PairData>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE void addQuadraturePointContribution(
      const PairQuadraturePoint<Geometry1, Geometry2>& qp, const PairData& pair_geometry_data,
      PairEnergyValue& result ) const
  {
    const auto energy = energy_function_.computeEnergyAtQuadraturePoint(
        qp.geometry1.position( qp.parametric_coords1 ), qp.geometry2.position( qp.parametric_coords2 ),
        qp.parametric_coords1, qp.parametric_coords2, pair_geometry_data );
    if ( !energy.active ) {
      return;
    }
    result.has_active_points = true;
    result.energy += energy.value * qp.physical_weight;
  }

 private:
  EnergyFunction energy_function_;  ///< Owned point energy density and activation calculation.
};

/**
 * @brief Integrand accumulating basis-weighted gaps and tributary areas on geometry1 before contact activation.
 * @tparam GapDefinition Geometric or offset gap calculation.
 * @tparam NodalBasis Scalar nodal basis on geometry1.
 */
template <typename GapDefinition, FiniteElementBasis NodalBasis>
class NodalGapIntegrand {
 public:
  static constexpr IndexT num_nodes = NodalBasis::size_basis;  ///< Number of geometry1 FE nodes carrying integrals.
  /// Weighted gaps followed by tributary areas.
  template <typename Geometry1, typename Geometry2>
  using Result = PairNodalGapValue<num_nodes>;
  /// Any successful nonempty integration stays active so derivative providers retain opening-gap derivatives.
  static constexpr bool active_for_nonempty_rule = true;

  /** @brief Store the gap calculation by value. @param gap Configured gap calculation. */
  TRIBOL_HOST_DEVICE explicit NodalGapIntegrand( GapDefinition gap = {} ) : gap_( std::move( gap ) ) {}

  /** @brief Compute pair-level geometry data once for the gap calculation. */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE auto computePairGeometryData( const Geometry1& geometry1,
                                                                        const Geometry2& geometry2 ) const
  {
    return gap_.computePairGeometryData( geometry1, geometry2 );
  }

  /** @brief Add one quadrature point's weighted gap and tributary area contributions. */
  template <typename Geometry1, typename Geometry2, typename PairData>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE void addQuadraturePointContribution(
      const PairQuadraturePoint<Geometry1, Geometry2>& qp, const PairData& pair_geometry_data,
      PairNodalGapValue<num_nodes>& result ) const
  {
    static_assert( NodalBasis::parametric_dim == Geometry1::parametric_dim );
    const RealT separation = gap_.computeGap( qp.geometry1.position( qp.parametric_coords1 ),
                                              qp.geometry2.position( qp.parametric_coords2 ), pair_geometry_data );
    MdArray<RealT, num_nodes> basis1{};
    NodalBasis::eval( qp.parametric_coords1, basis1.view() );
    for ( IndexT basis_index = 0; basis_index < num_nodes; ++basis_index ) {
      const RealT nodal_weight = qp.physical_weight * basis1[basis_index];
      result.weightedGap( basis_index ) += nodal_weight * separation;
      result.tributaryArea( basis_index ) += nodal_weight;
    }
  }

 private:
  GapDefinition gap_;  ///< Owned gap calculation.
};

/**
 * @brief Integrand accumulating non-mortar and cross-surface mortar weight matrices over the projected overlap.
 *
 * The multiplier basis on geometry1 can differ from the geometry1 displacement basis (for example, to integrate dual
 * multiplier basis functions against standard Lagrange basis functions).
 *
 * @tparam MultiplierBasis Multiplier basis on geometry1.
 * @tparam Basis1 Displacement basis on geometry1; defaults to MultiplierBasis.
 * @tparam Basis2 Displacement basis on geometry2; defaults to Basis1.
 */
template <FiniteElementBasis MultiplierBasis, FiniteElementBasis Basis1 = MultiplierBasis,
          FiniteElementBasis Basis2 = Basis1>
class MortarWeightsIntegrand {
 public:
  /// Contiguous non-mortar and cross-surface mortar weight matrices.
  template <typename Geometry1, typename Geometry2>
  using Result = PairMortarWeightsValue<MultiplierBasis::size_basis, Basis1::size_basis, Basis2::size_basis>;
  /// Any successful nonempty integration stays active so derivative providers retain weight derivatives.
  static constexpr bool active_for_nonempty_rule = true;

  /** @brief Empty per-pair data because mortar weights depend only on quadrature points and basis values. */
  struct EmptyPairData {};

  /** @brief Return empty pair-level data. */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static EmptyPairData computePairGeometryData(
      const Geometry1& TRIBOL_UNUSED_PARAM( geometry1 ), const Geometry2& TRIBOL_UNUSED_PARAM( geometry2 ) )
  {
    return {};
  }

  /** @brief Add multiplier-by-geometry1 and multiplier-by-geometry2 basis products at one quadrature point. */
  template <typename Geometry1, typename Geometry2>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE void addQuadraturePointContribution(
      const PairQuadraturePoint<Geometry1, Geometry2>& qp,
      const EmptyPairData& TRIBOL_UNUSED_PARAM( pair_geometry_data ),
      PairMortarWeightsValue<MultiplierBasis::size_basis, Basis1::size_basis, Basis2::size_basis>& result ) const
  {
    static_assert( MultiplierBasis::parametric_dim == Geometry1::parametric_dim );
    static_assert( Basis1::parametric_dim == Geometry1::parametric_dim );
    static_assert( Basis2::parametric_dim == Geometry2::parametric_dim );
    MdArray<RealT, MultiplierBasis::size_basis> multiplier_basis{};
    MdArray<RealT, Basis2::size_basis> basis2{};
    MultiplierBasis::eval( qp.parametric_coords1, multiplier_basis.view() );
    Basis2::eval( qp.parametric_coords2, basis2.view() );

    if constexpr ( std::is_same_v<MultiplierBasis, Basis1> ) {
      accumulateProducts( qp.physical_weight, multiplier_basis, multiplier_basis, basis2, result );
    } else {
      MdArray<RealT, Basis1::size_basis> basis1{};
      Basis1::eval( qp.parametric_coords1, basis1.view() );
      accumulateProducts( qp.physical_weight, multiplier_basis, basis1, basis2, result );
    }
  }

 private:
  /** @brief Multiply evaluated basis arrays by the physical integration weight and add to both weight matrices. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void accumulateProducts(
      RealT physical_weight, const MdArray<RealT, MultiplierBasis::size_basis>& multiplier_basis,
      const MdArray<RealT, Basis1::size_basis>& basis1, const MdArray<RealT, Basis2::size_basis>& basis2,
      PairMortarWeightsValue<MultiplierBasis::size_basis, Basis1::size_basis, Basis2::size_basis>& result )
  {
    for ( IndexT multiplier_basis_index = 0; multiplier_basis_index < MultiplierBasis::size_basis;
          ++multiplier_basis_index ) {
      const RealT weighted_multiplier = physical_weight * multiplier_basis[multiplier_basis_index];
      for ( IndexT basis_index = 0; basis_index < Basis1::size_basis; ++basis_index ) {
        result.geometry1( multiplier_basis_index, basis_index ) += weighted_multiplier * basis1[basis_index];
      }
      for ( IndexT basis_index = 0; basis_index < Basis2::size_basis; ++basis_index ) {
        result.geometry2( multiplier_basis_index, basis_index ) += weighted_multiplier * basis2[basis_index];
      }
    }
  }
};

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_PAIRINTEGRALS_HPP_
