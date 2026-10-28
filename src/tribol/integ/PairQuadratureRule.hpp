// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Compose an element-pair integration mapping with a reference quadrature rule.
 */
#ifndef SRC_TRIBOL_INTEG_PAIRQUADRATURERULE_HPP_
#define SRC_TRIBOL_INTEG_PAIRQUADRATURERULE_HPP_

#include "tribol/integ/Quadrature.hpp"

namespace tribol {

/**
 * @brief Generate paired parent points and physical weights locally, without retaining geometry or point data.
 *
 * Mapping defines both element geometry types and the integration surface passed to BaseQuadratureRule. The mapping
 * owns forward surface construction and reverse point mapping, including the physical integration measure.
 * @tparam Mapping Integration-surface construction and mapping to both elements.
 * @tparam BaseQuadratureRule Points and reference weights on the mapping's integration surface.
 */
template <typename Mapping, typename BaseQuadratureRule>
class PairQuadratureRule {
 public:
  using Mesh1Geometry = typename Mapping::Mesh1Geometry;                      ///< Geometry type of mesh1 elements.
  using Mesh2Geometry = typename Mapping::Mesh2Geometry;                      ///< Geometry type of mesh2 elements.
  static constexpr int parent_dimension = Mesh1Geometry::parent_dimension;    ///< Mesh1 parent components.
  static constexpr int parent_dimension2 = Mesh2Geometry::parent_dimension;   ///< Mesh2 parent components.
  static constexpr int spatial_dimension = Mesh1Geometry::spatial_dimension;  ///< Physical components.
  static constexpr int endpoint_count = Mesh1Geometry::endpoint_count;        ///< Mesh1 endpoints per element.
  static constexpr int points_per_base_rule = BaseQuadratureRule::points_per_base_rule;  ///< Local point capacity.
  static constexpr int polynomial_order = BaseQuadratureRule::polynomial_order;          ///< Exact reference degree.

  /** @brief Use default mapping and base-rule configuration. */
  TRIBOL_DEFAULT_HOST_DEVICE PairQuadratureRule() = default;
  /**
   * @brief Copy mapping and base-rule configuration.
   * @param mapping Forward and reverse mapping configuration.
   * @param base_rule Reference point-generation configuration.
   */
  TRIBOL_HOST_DEVICE explicit PairQuadratureRule( Mapping mapping, BaseQuadratureRule base_rule = {} )
      : mapping_( mapping ), base_rule_( base_rule )
  {
  }

  /** @brief Local integration-surface points and weights. */
  using ReferenceQuadrature = typename BaseQuadratureRule::ReferenceQuadrature;
  /** @brief Local paired parent coordinates and physical weights. */
  using QuadratureData = PairQuadratureData<parent_dimension, parent_dimension2, points_per_base_rule>;

  /**
   * @brief Construct the integration surface and place reference points on it.
   * @param mesh1 Current first-element coordinates, with geometry defined by Mesh1Geometry.
   * @param mesh2 Current second-element coordinates, with geometry defined by Mesh2Geometry.
   * @return Local reference quadrature with generation status; invalid geometry has zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ReferenceQuadrature generateReferenceQuadrature(
      typename Mesh1Geometry::Coordinates mesh1, typename Mesh2Geometry::Coordinates mesh2 ) const
  {
    return base_rule_.generate( mapping_.makeIntegrationSurface( mesh1, mesh2 ) );
  }

  /**
   * @brief Reverse-map existing reference points using current geometry, without repeating overlap calculations.
   * @param mesh1 Current first-element coordinates.
   * @param mesh2 Current second-element coordinates.
   * @param reference Integration-surface parent points and reference weights.
   * @return Local paired quadrature with generation status; invalid geometry has zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData
  mapReferenceQuadrature( typename Mesh1Geometry::Coordinates mesh1, typename Mesh2Geometry::Coordinates mesh2,
                          ReferenceQuadratureView<BaseQuadratureRule::parent_dimension> reference ) const
  {
    QuadratureData output{};
    output.result = mapping_.mapToElements( mesh1, mesh2, reference, output.getOutputView() );
    return output;
  }

  /**
   * @brief Generate a complete rule into caller-provided storage.
   * @param mesh1 Current first-element coordinates.
   * @param mesh2 Current second-element coordinates.
   * @param [out] output Parent coordinates on both elements and physical weights.
   * @return Generation status and full point count; failures must not be consumed. No allocation or retry is performed.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
  generate( typename Mesh1Geometry::Coordinates mesh1, typename Mesh2Geometry::Coordinates mesh2,
            PairQuadratureView<parent_dimension, parent_dimension2> output ) const
  {
    const auto reference = generateReferenceQuadrature( mesh1, mesh2 );
    // Return fields explicitly so Enzyme can identify their integer types.
    if ( reference.result.status != QuadratureStatus::Success ) return { QuadratureStatus::InvalidGeometry, 0 };
    return mapping_.mapToElements( mesh1, mesh2, reference.getView(), output );
  }

  /**
   * @brief Generate and reverse-map one pair into local fixed-capacity storage.
   * @param mesh1 Current first-element coordinates.
   * @param mesh2 Current second-element coordinates.
   * @return Local paired quadrature with generation status; valid zero measure has zero points.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureData generate( typename Mesh1Geometry::Coordinates mesh1,
                                                                   typename Mesh2Geometry::Coordinates mesh2 ) const
  {
    QuadratureData output{};
    output.result = generate( mesh1, mesh2, output.getOutputView() );
    return output;
  }

 private:
  Mapping mapping_{};               ///< Owned mapping configuration; no generated surfaces or mesh views are retained.
  BaseQuadratureRule base_rule_{};  ///< Owned reference point-generation configuration.
};

}  // namespace tribol
#endif
