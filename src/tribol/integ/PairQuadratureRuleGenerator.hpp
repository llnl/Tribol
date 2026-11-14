// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Compose an element-pair integration mapping with an integration-surface quadrature rule generator.
 */
#ifndef SRC_TRIBOL_INTEG_PAIRQUADRATURERULEGENERATOR_HPP_
#define SRC_TRIBOL_INTEG_PAIRQUADRATURERULEGENERATOR_HPP_

#include <concepts>
#include <type_traits>

#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/**
 * @brief Requires element geometry types and integration-surface construction for a pair of elements.
 *
 * Geometry types must be constructible from their coordinate views and provide positive compile-time parametric
 * dimensions. Geometry validity remains a runtime responsibility.
 * @tparam Mapping Type supplying the element geometries and constructing the integration surface.
 */
template <typename Mapping>
concept ElementPairMapping = requires(
    const Mapping& mapping, typename Mapping::Geometry1::Coords coords1, typename Mapping::Geometry2::Coords coords2,
    const typename Mapping::Geometry1& geometry1, const typename Mapping::Geometry2& geometry2 ) {
  typename std::integral_constant<int, Mapping::Geometry1::parametric_dim>;
  typename std::integral_constant<int, Mapping::Geometry2::parametric_dim>;
  requires( Mapping::Geometry1::parametric_dim > 0 && Mapping::Geometry2::parametric_dim > 0 );
  { typename Mapping::Geometry1( coords1 ) } -> std::same_as<typename Mapping::Geometry1>;
  { typename Mapping::Geometry2( coords2 ) } -> std::same_as<typename Mapping::Geometry2>;
  mapping.makeIntegrationSurface( geometry1, geometry2 );
};

/**
 * @brief Requires a mapping and generator that can exchange integration-surface rules and paired element coordinates.
 *
 * The generator accepts the mapping's integration surface. The mapping accepts the generated rule's read-only view
 * and writes paired element coordinates and physical weights. These requirements check types and expressions;
 * coordinate conventions, weight scaling, and host/device execution must be verified separately.
 * @tparam Mapping Type constructing the integration surface and mapping its quadrature points to both elements.
 * @tparam Generator Type generating integration-surface points, weights, and generation status.
 */
template <typename Mapping, typename Generator>
concept CompatibleQuadratureMappingAndGenerator =
    ElementPairMapping<Mapping> &&
    requires {
      typename Generator::IntegrationSurfaceRule;
      typename std::integral_constant<int, Generator::coordinate_dim>;
      typename std::integral_constant<int, Generator::max_num_points>;
      typename std::integral_constant<int, Generator::polynomial_order>;
      requires( Generator::coordinate_dim == 1 || Generator::coordinate_dim == 2 );
      requires( Generator::max_num_points > 0 );
    } &&
    requires( const Generator& generator, const Mapping& mapping, const typename Mapping::Geometry1& geometry1,
              const typename Mapping::Geometry2& geometry2,
              const typename Generator::IntegrationSurfaceRule& integration_surface_rule,
              IntegrationSurfaceRuleView<Generator::coordinate_dim> integration_surface_rule_view,
              PairQuadratureRuleView<Mapping::Geometry1::parametric_dim, Mapping::Geometry2::parametric_dim> output ) {
      {
        generator.generate( mapping.makeIntegrationSurface( geometry1, geometry2 ) )
      } -> std::same_as<typename Generator::IntegrationSurfaceRule>;
      { integration_surface_rule.getView() } -> std::same_as<IntegrationSurfaceRuleView<Generator::coordinate_dim>>;
      QuadratureResult{ integration_surface_rule.result.status, integration_surface_rule.result.num_required_points };
      {
        mapping.mapToElements( geometry1, geometry2, integration_surface_rule_view, output )
      } -> std::same_as<QuadratureResult>;
    };

/**
 * @brief Generate quadrature points in each element's parametric coordinates and physical weights, without retaining
 * geometry or point data.
 *
 * Mapping defines element geometry types constructible from Coords views and the integration surface passed to
 * IntegrationSurfaceRuleGenerator. It constructs the contact integration surface from the element geometries, maps
 * generated integration points back to both elements, and calculates each point's physical integration weight. Each
 * call constructs element geometry objects from the supplied coordinates and discards them before returning. For
 * linear edges, these objects copy the endpoints and cache the edge vector and squared length for
 * reuse by the mapping.
 * @tparam Mapping Integration-surface construction and mapping to both elements.
 * @tparam IntegrationSurfaceRuleGenerator Generates points and weights in the mapping's integration-surface
 * coordinates.
 */
template <ElementPairMapping Mapping, typename IntegrationSurfaceRuleGenerator>
  requires CompatibleQuadratureMappingAndGenerator<Mapping, IntegrationSurfaceRuleGenerator>
class PairQuadratureRuleGenerator {
 public:
  using BaseRuleGenerator = IntegrationSurfaceRuleGenerator;  ///< Integration-surface quadrature generator.
  using Geometry1 = typename Mapping::Geometry1;              ///< Geometry type of the first element.
  using Geometry2 = typename Mapping::Geometry2;              ///< Geometry type of the second element.
  /// Maximum number of points in a successfully generated rule.
  static constexpr int max_num_points = IntegrationSurfaceRuleGenerator::max_num_points;
  /// Maximum polynomial degree integrated exactly in integration-surface coordinates.
  static constexpr int polynomial_order = IntegrationSurfaceRuleGenerator::polynomial_order;

  /** @brief Use a default mapping and integration-surface rule generator. */
  TRIBOL_DEFAULT_HOST_DEVICE PairQuadratureRuleGenerator() = default;
  /**
   * @brief Copy the mapping and integration-surface rule generator.
   * @param mapping Forward and reverse mapping configuration.
   * @param integration_surface_rule_generator Integration-surface generator copied into this pair rule generator.
   */
  TRIBOL_HOST_DEVICE explicit PairQuadratureRuleGenerator(
      Mapping mapping, IntegrationSurfaceRuleGenerator integration_surface_rule_generator = {} )
      : mapping_( mapping ), integration_surface_rule_generator_( integration_surface_rule_generator )
  {
  }

  /** @brief Quadrature points and weights expressed in the integration surface's coordinates. */
  using IntegrationSurfaceRule = typename IntegrationSurfaceRuleGenerator::IntegrationSurfaceRule;
  /** @brief Generated pair quadrature rule containing parametric coordinates, physical weights, and generation status.
   */
  using Rule = PairQuadratureRule<Geometry1::parametric_dim, Geometry2::parametric_dim, max_num_points>;

  /**
   * @brief Generate into caller storage using geometry prepared by the pair calculation.
   * @param geometry1 First current element geometry.
   * @param geometry2 Second current element geometry.
   * @param[out] output Paired parametric coordinates and physical weights.
   * @return Generation status and full count; failed output must not be consumed.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
  generate( const Geometry1& geometry1, const Geometry2& geometry2,
            PairQuadratureRuleView<Geometry1::parametric_dim, Geometry2::parametric_dim> output ) const
  {
    const auto integration_surface_rule =
        integration_surface_rule_generator_.generate( mapping_.makeIntegrationSurface( geometry1, geometry2 ) );
    // Return fields explicitly so Enzyme can identify their integer types.
    if ( integration_surface_rule.result.status != QuadratureStatus::Success ) {
      return { integration_surface_rule.result.status, integration_surface_rule.result.num_required_points };
    }
    return mapping_.mapToElements( geometry1, geometry2, integration_surface_rule.getView(), output );
  }

  /**
   * @brief Generate into local storage using already prepared element geometry.
   * @param geometry1 First current element geometry.
   * @param geometry2 Second current element geometry.
   * @return Local paired quadrature; failed output must not be consumed.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Rule generate( const Geometry1& geometry1, const Geometry2& geometry2 ) const
  {
    Rule output{};
    output.result = generate( geometry1, geometry2, output.getOutputView() );
    return output;
  }

 private:
  Mapping mapping_{};  ///< Owned mapping configuration; no generated surfaces or coordinate views are retained.
  IntegrationSurfaceRuleGenerator
      integration_surface_rule_generator_{};  ///< Owned generator; no generated points or weights are retained.
};

}  // namespace tribol
#endif
