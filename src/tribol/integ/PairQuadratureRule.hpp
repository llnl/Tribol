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

#include "tribol/integ/Quadrature.hpp"

namespace tribol {

/**
 * @brief Generate paired parent points and physical weights locally, without retaining geometry or point data.
 *
 * Mapping defines element geometry types constructible from Coordinates views and the integration surface passed to
 * BaseQuadratureRule. It owns forward surface construction and reverse point mapping, including the physical measure.
 * Each call constructs element geometry objects from the supplied coordinates and discards them before returning.
 * For linear edges, these objects copy the endpoints and cache the edge vector and squared length for reuse by the
 * mapping.
 * @tparam Mapping Integration-surface construction and mapping to both elements.
 * @tparam BaseQuadratureRule Points and weights expressed in the mapping's integration-surface coordinates.
 */
template <typename Mapping, typename BaseQuadratureRule>
class PairQuadratureRule {
 public:
  using FirstGeometry = typename Mapping::FirstGeometry;                    ///< Geometry type of coordinates1 elements.
  using SecondGeometry = typename Mapping::SecondGeometry;                  ///< Geometry type of coordinates2 elements.
  static constexpr int parent_dimension = FirstGeometry::parent_dimension;  ///< First geometry parent components.
  static constexpr int parent_dimension2 = SecondGeometry::parent_dimension;  ///< Second geometry parent components.
  static constexpr int spatial_dimension = FirstGeometry::spatial_dimension;  ///< Physical components.
  static constexpr int vertex_count = FirstGeometry::vertex_count;  ///< Number of vertices in the first geometry.
  static constexpr int points_per_base_rule = BaseQuadratureRule::points_per_base_rule;  ///< Local point capacity.
  static constexpr int polynomial_order = BaseQuadratureRule::polynomial_order;          ///< Exact reference degree.

  /** @brief Use default mapping and base-rule configuration. */
  TRIBOL_DEFAULT_HOST_DEVICE PairQuadratureRule() = default;
  /**
   * @brief Copy mapping and base-rule configuration.
   * @param mapping Forward and reverse mapping configuration.
   * @param base_rule Integration-surface quadrature configuration.
   */
  TRIBOL_HOST_DEVICE explicit PairQuadratureRule( Mapping mapping, BaseQuadratureRule base_rule = {} )
      : mapping_( mapping ), base_rule_( base_rule )
  {
  }

  /** @brief Quadrature points and weights expressed in the integration surface's coordinates. */
  using IntegrationSurfaceQuadrature = typename BaseQuadratureRule::IntegrationSurfaceQuadrature;
  /** @brief Local paired parent coordinates and physical weights. */
  using QuadratureData = PairQuadratureData<parent_dimension, parent_dimension2, points_per_base_rule>;

  /**
   * @brief Construct the integration surface and generate quadrature points and weights in its coordinates.
   *
   * The mapping defines these coordinates. SmoothedProjectedEdgeMapping uses edge1's parent coordinate, with vertices
   * at -1 and 1; its weights do not yet include edge1's physical Jacobian.
   *
   * @param coordinates1 Current first-element coordinates, with geometry defined by FirstGeometry.
   * @param coordinates2 Current second-element coordinates, with geometry defined by SecondGeometry.
   * @return Integration-surface quadrature with generation status; invalid geometry has zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurfaceQuadrature generateIntegrationSurfaceQuadrature(
      typename FirstGeometry::Coordinates coordinates1, typename SecondGeometry::Coordinates coordinates2 ) const
  {
    return base_rule_.generate(
        mapping_.makeIntegrationSurface( FirstGeometry( coordinates1 ), SecondGeometry( coordinates2 ) ) );
  }

  /**
   * @brief Map integration-surface quadrature points to both elements using their current geometry.
   *
   * The integration surface and its quadrature are not recalculated.
   * @param coordinates1 Current first-element coordinates.
   * @param coordinates2 Current second-element coordinates.
   * @param reference Quadrature points and weights expressed in the integration surface's coordinates.
   * @return Local paired quadrature with generation status; invalid geometry has zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData mapIntegrationSurfaceQuadratureToElements(
      typename FirstGeometry::Coordinates coordinates1, typename SecondGeometry::Coordinates coordinates2,
      IntegrationSurfaceQuadratureView<BaseQuadratureRule::parent_dimension> reference ) const
  {
    QuadratureData output{};
    output.result = mapping_.mapToElements( FirstGeometry( coordinates1 ), SecondGeometry( coordinates2 ), reference,
                                            output.getOutputView() );
    return output;
  }

  /**
   * @brief Generate a complete rule into caller-provided storage.
   * @param coordinates1 Current first-element coordinates.
   * @param coordinates2 Current second-element coordinates.
   * @param [out] output Parent coordinates on both elements and physical weights.
   * @return Generation status and full point count; failures must not be consumed. No allocation or retry is performed.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
  generate( typename FirstGeometry::Coordinates coordinates1, typename SecondGeometry::Coordinates coordinates2,
            PairQuadratureView<parent_dimension, parent_dimension2> output ) const
  {
    const FirstGeometry geometry1( coordinates1 );
    const SecondGeometry geometry2( coordinates2 );
    const auto reference = base_rule_.generate( mapping_.makeIntegrationSurface( geometry1, geometry2 ) );
    // Return fields explicitly so Enzyme can identify their integer types.
    if ( reference.result.status != QuadratureStatus::Success ) {
      return { reference.result.status, reference.result.required_point_count };
    }
    return mapping_.mapToElements( geometry1, geometry2, reference.getView(), output );
  }

  /**
   * @brief Generate and reverse-map one pair into local fixed-capacity storage.
   * @param coordinates1 Current first-element coordinates.
   * @param coordinates2 Current second-element coordinates.
   * @return Local paired quadrature with generation status; valid zero measure has zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData
  generate( typename FirstGeometry::Coordinates coordinates1, typename SecondGeometry::Coordinates coordinates2 ) const
  {
    QuadratureData output{};
    output.result = generate( coordinates1, coordinates2, output.getOutputView() );
    return output;
  }

 private:
  Mapping mapping_{};  ///< Owned mapping configuration; no generated surfaces or coordinate views are retained.
  BaseQuadratureRule base_rule_{};  ///< Owned integration-surface quadrature configuration.
};

}  // namespace tribol
#endif
