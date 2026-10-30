// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Compose an element-pair integration mapping with an integration-surface quadrature rule.
 */
#ifndef SRC_TRIBOL_INTEG_PAIRQUADRATURERULE_HPP_
#define SRC_TRIBOL_INTEG_PAIRQUADRATURERULE_HPP_

#include <concepts>
#include <type_traits>

#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/**
 * @brief Requires element geometry types and integration-surface construction for a pair of elements.
 *
 * Geometry types must be constructible from their coordinate views. Dimensions and vertex counts must be compile-time
 * constants. Geometry validity remains a runtime responsibility.
 * @tparam Mapping Type supplying the element geometries and constructing the integration surface.
 */
template <typename Mapping>
concept ElementPairMapping =
    requires( const Mapping& mapping, typename Mapping::FirstGeometry::Coords element1_coords,
              typename Mapping::SecondGeometry::Coords element2_coords,
              const typename Mapping::FirstGeometry& geometry1, const typename Mapping::SecondGeometry& geometry2 ) {
      typename std::integral_constant<int, Mapping::FirstGeometry::parent_dimension>;
      typename std::integral_constant<int, Mapping::SecondGeometry::parent_dimension>;
      typename std::integral_constant<int, Mapping::FirstGeometry::spatial_dimension>;
      typename std::integral_constant<int, Mapping::FirstGeometry::vertex_count>;
      requires( Mapping::FirstGeometry::parent_dimension > 0 && Mapping::SecondGeometry::parent_dimension > 0 );
      { typename Mapping::FirstGeometry( element1_coords ) } -> std::same_as<typename Mapping::FirstGeometry>;
      { typename Mapping::SecondGeometry( element2_coords ) } -> std::same_as<typename Mapping::SecondGeometry>;
      mapping.makeIntegrationSurface( geometry1, geometry2 );
    };

/**
 * @brief Requires a generator and mapping that can exchange integration-surface rules and paired element coordinates.
 *
 * The generator accepts the mapping's integration surface. The mapping accepts the generated rule's read-only view
 * and writes paired element coordinates and physical weights. These requirements check types and expressions;
 * coordinate conventions, weight scaling, and host/device execution must be verified separately.
 * @tparam Generator Type generating integration-surface points, weights, and generation status.
 * @tparam Mapping Type constructing the integration surface and mapping its quadrature points to both elements.
 */
template <typename Generator, typename Mapping>
concept IntegrationSurfaceRuleGeneratorFor =
    ElementPairMapping<Mapping> &&
    requires {
      typename Generator::IntegrationSurfaceRule;
      typename std::integral_constant<int, Generator::parent_dimension>;
      typename std::integral_constant<int, Generator::max_points>;
      typename std::integral_constant<int, Generator::polynomial_order>;
      requires( Generator::parent_dimension == 1 || Generator::parent_dimension == 2 );
      requires( Generator::max_points > 0 );
    } &&
    requires( const Generator& generator, const Mapping& mapping, const typename Mapping::FirstGeometry& geometry1,
              const typename Mapping::SecondGeometry& geometry2,
              const typename Generator::IntegrationSurfaceRule& integration_surface_rule,
              IntegrationSurfaceRuleView<Generator::parent_dimension> integration_surface_rule_view,
              PairQuadratureView<Mapping::FirstGeometry::parent_dimension, Mapping::SecondGeometry::parent_dimension>
                  output ) {
      {
        generator.generate( mapping.makeIntegrationSurface( geometry1, geometry2 ) )
      } -> std::same_as<typename Generator::IntegrationSurfaceRule>;
      { integration_surface_rule.getView() } -> std::same_as<IntegrationSurfaceRuleView<Generator::parent_dimension>>;
      QuadratureResult{ integration_surface_rule.result.status, integration_surface_rule.result.required_point_count };
      {
        mapping.mapToElements( geometry1, geometry2, integration_surface_rule_view, output )
      } -> std::same_as<QuadratureResult>;
    };

/**
 * @brief Generate paired parent points and physical weights locally, without retaining geometry or point data.
 *
 * Mapping defines element geometry types constructible from Coords views and the integration surface passed to
 * IntegrationSurfaceRuleGenerator. It owns forward surface construction and reverse point mapping, including the
 * physical measure. Each call constructs element geometry objects from the supplied coordinates and discards them
 * before returning. For linear edges, these objects copy the endpoints and cache the edge vector and squared length for
 * reuse by the mapping.
 * @tparam Mapping Integration-surface construction and mapping to both elements.
 * @tparam IntegrationSurfaceRuleGenerator Generates points and weights in the mapping's integration-surface
 * coordinates.
 */
template <ElementPairMapping Mapping, typename IntegrationSurfaceRuleGenerator>
  requires IntegrationSurfaceRuleGeneratorFor<IntegrationSurfaceRuleGenerator, Mapping>
class PairQuadratureRule {
 public:
  using FirstGeometry = typename Mapping::FirstGeometry;                      ///< Geometry type of the first element.
  using SecondGeometry = typename Mapping::SecondGeometry;                    ///< Geometry type of the second element.
  static constexpr int parent_dimension = FirstGeometry::parent_dimension;    ///< First geometry parent components.
  static constexpr int parent_dimension2 = SecondGeometry::parent_dimension;  ///< Second geometry parent components.
  static constexpr int spatial_dimension = FirstGeometry::spatial_dimension;  ///< Physical components.
  static constexpr int vertex_count = FirstGeometry::vertex_count;  ///< Number of vertices in the first geometry.
  /// Maximum number of points in a successfully generated rule.
  static constexpr int max_points = IntegrationSurfaceRuleGenerator::max_points;
  static constexpr int polynomial_order =
      IntegrationSurfaceRuleGenerator::polynomial_order;  ///< Exact reference degree.

  /** @brief Use a default mapping and integration-surface rule generator. */
  TRIBOL_DEFAULT_HOST_DEVICE PairQuadratureRule() = default;
  /**
   * @brief Copy the mapping and integration-surface rule generator.
   * @param mapping Forward and reverse mapping configuration.
   * @param integration_surface_rule_generator Generator copied into this pair rule.
   */
  TRIBOL_HOST_DEVICE explicit PairQuadratureRule(
      Mapping mapping, IntegrationSurfaceRuleGenerator integration_surface_rule_generator = {} )
      : mapping_( mapping ), integration_surface_rule_generator_( integration_surface_rule_generator )
  {
  }

  /** @brief Quadrature points and weights expressed in the integration surface's coordinates. */
  using IntegrationSurfaceRule = typename IntegrationSurfaceRuleGenerator::IntegrationSurfaceRule;
  /** @brief Local paired parent coordinates and physical weights. */
  using QuadratureData = PairQuadratureData<parent_dimension, parent_dimension2, max_points>;

  /**
   * @brief Construct the integration surface and generate quadrature points and weights in its coordinates.
   *
   * The mapping defines these coordinates. SmoothedProjectedEdgeMapping uses edge1's parent coordinate, with vertices
   * at -1 and 1; its weights do not yet include edge1's physical Jacobian.
   *
   * @param element1_coords Current first-element coordinates, with geometry defined by FirstGeometry.
   * @param element2_coords Current second-element coordinates, with geometry defined by SecondGeometry.
   * @return Integration-surface rule with generation status; invalid geometry has zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurfaceRule generateIntegrationSurfaceRule(
      typename FirstGeometry::Coords element1_coords, typename SecondGeometry::Coords element2_coords ) const
  {
    return integration_surface_rule_generator_.generate(
        mapping_.makeIntegrationSurface( FirstGeometry( element1_coords ), SecondGeometry( element2_coords ) ) );
  }

  /**
   * @brief Map integration-surface quadrature points to both elements using their current geometry.
   *
   * The integration surface and its rule are not recalculated.
   * @param element1_coords Current first-element coordinates.
   * @param element2_coords Current second-element coordinates.
   * @param integration_surface_rule Quadrature points and weights expressed in the integration surface's
   * coordinates.
   * @return Local paired quadrature with generation status; invalid geometry has zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData mapIntegrationSurfaceRuleToElements(
      typename FirstGeometry::Coords element1_coords, typename SecondGeometry::Coords element2_coords,
      IntegrationSurfaceRuleView<IntegrationSurfaceRuleGenerator::parent_dimension> integration_surface_rule ) const
  {
    QuadratureData output{};
    output.result = mapping_.mapToElements( FirstGeometry( element1_coords ), SecondGeometry( element2_coords ),
                                            integration_surface_rule, output.getOutputView() );
    return output;
  }

  /**
   * @brief Generate a complete rule into caller-provided storage.
   * @param element1_coords Current first-element coordinates.
   * @param element2_coords Current second-element coordinates.
   * @param [out] output Parent coordinates on both elements and physical weights.
   * @return Generation status and full point count; failures must not be consumed. No allocation or retry is performed.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
  generate( typename FirstGeometry::Coords element1_coords, typename SecondGeometry::Coords element2_coords,
            PairQuadratureView<parent_dimension, parent_dimension2> output ) const
  {
    const FirstGeometry geometry1( element1_coords );
    const SecondGeometry geometry2( element2_coords );
    const auto integration_surface_rule =
        integration_surface_rule_generator_.generate( mapping_.makeIntegrationSurface( geometry1, geometry2 ) );
    // Return fields explicitly so Enzyme can identify their integer types.
    if ( integration_surface_rule.result.status != QuadratureStatus::Success ) {
      return { integration_surface_rule.result.status, integration_surface_rule.result.required_point_count };
    }
    return mapping_.mapToElements( geometry1, geometry2, integration_surface_rule.getView(), output );
  }

  /**
   * @brief Generate and reverse-map one pair into local fixed-capacity storage.
   * @param element1_coords Current first-element coordinates.
   * @param element2_coords Current second-element coordinates.
   * @return Local paired quadrature with generation status; valid zero measure has zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData
  generate( typename FirstGeometry::Coords element1_coords, typename SecondGeometry::Coords element2_coords ) const
  {
    QuadratureData output{};
    output.result = generate( element1_coords, element2_coords, output.getOutputView() );
    return output;
  }

 private:
  Mapping mapping_{};  ///< Owned mapping configuration; no generated surfaces or coordinate views are retained.
  IntegrationSurfaceRuleGenerator
      integration_surface_rule_generator_{};  ///< Owned generator; no generated points or weights are retained.
};

}  // namespace tribol
#endif
