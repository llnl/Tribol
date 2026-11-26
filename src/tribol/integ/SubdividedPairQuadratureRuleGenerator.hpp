// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Compose surface-element subdivision with a linear subelement-pair quadrature generator. */
#ifndef SRC_TRIBOL_INTEG_SUBDIVIDEDPAIRQUADRATURERULEGENERATOR_HPP_
#define SRC_TRIBOL_INTEG_SUBDIVIDEDPAIRQUADRATURERULEGENERATOR_HPP_

#include <cmath>
#include <concepts>
#include <utility>

#include "tribol/geom/LinearEdgeGeometry.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"

namespace tribol {

/** @brief Element parametric coordinates bounding one uniform subelement. */
struct SubelementInterval {
  RealT lower;  ///< Element parametric coordinate at the subelement's first vertex.
  RealT upper;  ///< Element parametric coordinate at the subelement's second vertex.

  /**
   * @brief Map a subelement parametric coordinate affinely into the element.
   * @param subelement_coord Subelement parametric coordinate in [-1, 1].
   * @return Element parametric coordinate in [lower, upper].
   *
   * This does a linear interpolation from a subelement coordinate to a surface element coordinate, which may not be in
   * the same direction as the gap vector.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT mapToElement( RealT subelement_coord ) const
  {
    return 0.5 * ( ( 1.0 - subelement_coord ) * lower + ( 1.0 + subelement_coord ) * upper );
  }
};

/** @brief One subelement quadrature point together with its owning surface element and subelement interval.
 * @tparam ElementGeometry Current surface-element geometry.
 */
template <typename ElementGeometry>
struct SubelementPoint {
  const ElementGeometry& element;        ///< Owning surface-element geometry.
  const LinearEdgeGeometry& subelement;  ///< Linear subelement that produced the point.
  SubelementInterval interval;           ///< Element parametric bounds of the subelement.
  RealT subelement_coord;                ///< Point's parametric coordinate on the subelement in [-1, 1].
};

/** @brief Element parametric coordinates of one paired quadrature point, with a validity flag. */
struct ProjectedElementCoords {
  RealT geom1_parametric_coord = 0.0;  ///< Element parametric coordinate on geometry1.
  RealT geom2_parametric_coord = 0.0;  ///< Element parametric coordinate on geometry2.
  bool valid = false;                  ///< Whether both coordinates are usable element parametric coordinates.
};

/**
 * @brief Requires a projection mapping one paired subelement quadrature point onto both surface elements.
 *
 * The projection receives both subelement points (each bundling its surface element, linear subelement, parametric
 * interval, and subelement coordinate) and the integration-surface normal along which the subelement rule paired the
 * point. It returns the element parametric coordinates at which the integrand is evaluated.
 * @tparam Projection Subelement-to-element projection type.
 * @tparam ElementGeometry Current surface-element geometry.
 */
template <typename Projection, typename ElementGeometry>
concept SubelementPointProjection =
    requires( const Projection& projection, const SubelementPoint<ElementGeometry>& point,
              const LinearEdgeGeometry::Vector& normal ) {
      { projection.mapToElements( point, point, normal ) } -> std::same_as<ProjectedElementCoords>;
    };

/**
 * @brief Reuse each subelement parametric coordinate affinely in its surface element.
 *
 * On curved elements, the evaluated element points are offset from the subelement points that the linear rule paired,
 * so their gap is generally not parallel to the subelement pairing direction (e.g. the common plane normal).
 */
struct AffineSubelementMapping {
  /**
   * @brief Map both subelement coordinates affinely into their element intervals.
   * @tparam ElementGeometry Current surface-element geometry.
   * @param point1 First subelement point.
   * @param point2 Second subelement point.
   * @param normal Integration-surface normal of the subelement pair, unused.
   * @return Affinely mapped element coordinates, always valid.
   */
  template <typename ElementGeometry>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ProjectedElementCoords
  mapToElements( const SubelementPoint<ElementGeometry>& point1, const SubelementPoint<ElementGeometry>& point2,
                 const LinearEdgeGeometry::Vector& TRIBOL_UNUSED_PARAM( normal ) ) const
  {
    return { point1.interval.mapToElement( point1.subelement_coord ),
             point2.interval.mapToElement( point2.subelement_coord ), true };
  }
};

/**
 * @brief Generate element-edge quadrature by applying a linear rule to every pair of uniform linear subelements.
 * @tparam ElementGeometry Current surface-element geometry evaluated in one parametric coordinate.
 * @tparam SubelementPairRuleGenerator Rule generator operating on two linear subelement edges.
 * @tparam MaxNumSubdivisions Largest accepted number of subelements on either surface element.
 * @tparam Projection Maps each paired subelement point onto both surface elements.
 */
template <typename ElementGeometry, PairRuleGenerator SubelementPairRuleGenerator, int MaxNumSubdivisions,
          SubelementPointProjection<ElementGeometry> Projection = AffineSubelementMapping>
  requires( ElementGeometry::parametric_dim == 1 && SubelementPairRuleGenerator::Geometry1::parametric_dim == 1 &&
            SubelementPairRuleGenerator::Geometry2::parametric_dim == 1 &&
            requires { typename std::integral_constant<int, SubelementPairRuleGenerator::max_num_points>; } )
class SubdividedPairQuadratureRuleGenerator {
 public:
  using Geometry1 = ElementGeometry;                               ///< First surface-element geometry.
  using Geometry2 = ElementGeometry;                               ///< Second surface-element geometry.
  static constexpr int max_num_subdivisions = MaxNumSubdivisions;  ///< Fixed upper bound on uniform subelements.
  static constexpr int max_num_points =
      MaxNumSubdivisions * MaxNumSubdivisions * SubelementPairRuleGenerator::max_num_points;  ///< Fixed point capacity.
  static constexpr int polynomial_order =
      SubelementPairRuleGenerator::polynomial_order;  ///< Subelement-rule exactness order.
  using Rule = PairQuadratureRule<Geometry1::parametric_dim, Geometry2::parametric_dim,
                                  max_num_points>;  ///< Fixed-capacity element-coordinate rule.

  static_assert( MaxNumSubdivisions > 0 );
  static_assert( std::same_as<typename SubelementPairRuleGenerator::Geometry1, LinearEdgeGeometry> );
  static_assert( std::same_as<typename SubelementPairRuleGenerator::Geometry2, LinearEdgeGeometry> );

  /**
   * @brief Store the number of uniform subelements, subelement-pair generator, and subelement-to-element projection.
   * @param num_subdivisions Number of subelements on each surface element.
   * @param subelement_pair_generator Linear subelement-pair rule generator.
   * @param projection Maps paired subelement points onto both surface elements.
   */
  TRIBOL_HOST_DEVICE explicit SubdividedPairQuadratureRuleGenerator(
      int num_subdivisions, SubelementPairRuleGenerator subelement_pair_generator = {}, Projection projection = {} )
      : num_subdivisions_( num_subdivisions ),
        subelement_pair_generator_( std::move( subelement_pair_generator ) ),
        projection_( std::move( projection ) )
  {
  }

  /**
   * @brief Generate all subelement-pair rules and express their coordinates in both surface elements.
   * @param geometry1 First current surface-element geometry.
   * @param geometry2 Second current surface-element geometry.
   * @return Element-parametric points, unchanged physical subelement weights and integration-surface normals, and
   * generation status. Points that do not project onto both surface elements are skipped.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Rule generate( const Geometry1& geometry1, const Geometry2& geometry2 ) const
  {
    Rule output{};
    if ( num_subdivisions_ < 1 || num_subdivisions_ > MaxNumSubdivisions ) {
      output.metadata = { QuadratureStatus::InvalidGeometry, 0 };
      return output;
    }

    IndexT output_point = 0;
    for ( int subelement1 = 0; subelement1 < num_subdivisions_; ++subelement1 ) {
      const auto subelement_geometry1 = makeSubelementGeometry( geometry1, subelement1 );
      if ( !subelement_geometry1.valid ) {
        continue;
      }

      for ( int subelement2 = 0; subelement2 < num_subdivisions_; ++subelement2 ) {
        const auto subelement_geometry2 = makeSubelementGeometry( geometry2, subelement2 );
        if ( !subelement_geometry2.valid ) {
          continue;
        }
        const auto subelement_rule =
            subelement_pair_generator_.generate( subelement_geometry1.geometry, subelement_geometry2.geometry );
        if ( subelement_rule.metadata.status != QuadratureStatus::Success ) {
          output.metadata = { subelement_rule.metadata.status, subelement_rule.metadata.num_required_points };
          return output;
        }
        if ( output_point + subelement_rule.metadata.num_required_points > max_num_points ) {
          output.metadata = { QuadratureStatus::InsufficientCapacity,
                              output_point + subelement_rule.metadata.num_required_points };
          return output;
        }
        // Map each subelement-pair quadrature point onto both surface elements; skip points that miss either element.
        for ( IndexT point = 0; point < subelement_rule.metadata.num_required_points; ++point ) {
          const LinearEdgeGeometry::Vector normal( &subelement_rule.integration_surface_normals( point, 0 ),
                                                   LinearEdgeGeometry::spatial_dim );
          const SubelementPoint<ElementGeometry> point1{ geometry1, subelement_geometry1.geometry,
                                                         subelementInterval( subelement1 ),
                                                         subelement_rule.geom1_parametric_coords( point, 0 ) };
          const SubelementPoint<ElementGeometry> point2{ geometry2, subelement_geometry2.geometry,
                                                         subelementInterval( subelement2 ),
                                                         subelement_rule.geom2_parametric_coords( point, 0 ) };
          const auto element_coords = projection_.mapToElements( point1, point2, normal );
          if ( !element_coords.valid ) {
            continue;
          }
          output.geom1_parametric_coords( output_point, 0 ) = element_coords.geom1_parametric_coord;
          output.geom2_parametric_coords( output_point, 0 ) = element_coords.geom2_parametric_coord;
          output.weights[output_point] = subelement_rule.weights[point];
          for ( IndexT d = 0; d < LinearEdgeGeometry::spatial_dim; ++d ) {
            output.integration_surface_normals( output_point, d ) = normal[d];
          }
          ++output_point;
        }
      }
    }
    output.metadata.num_required_points = output_point;
    return output;
  }

 private:
  /**
   * @brief Return one uniform subelement boundary in element parametric coordinates.
   * @param boundary Boundary index from zero through num_subdivisions_.
   * @return Element parametric coordinate in [-1, 1].
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT subdivisionBound( int boundary ) const
  {
    return -1.0 + 2.0 * static_cast<RealT>( boundary ) / static_cast<RealT>( num_subdivisions_ );
  }

  /**
   * @brief Return the element parametric interval of one uniform subelement.
   * @param subelement_index Index of the subelement in [0, num_subdivisions_).
   * @return Element parametric bounds of the subelement.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE SubelementInterval subelementInterval( int subelement_index ) const
  {
    return { subdivisionBound( subelement_index ), subdivisionBound( subelement_index + 1 ) };
  }

  /** @brief Linear subelement geometry together with a validity flag. */
  struct SubelementGeometryResult {
    LinearEdgeGeometry geometry;
    bool valid = false;
  };

  /**
   * @brief Construct a linear subelement for the given subdivision index and check that it is usable.
   * @param geometry Surface-element geometry evaluated at both interval endpoints.
   * @param subelement_index Index of the subelement in [0, num_subdivisions_).
   * @return Linear subelement geometry with valid=true when both endpoints are finite and define a nonzero chord.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE SubelementGeometryResult
  makeSubelementGeometry( const ElementGeometry& geometry, int subelement_index ) const
  {
    MdArray<RealT, LinearEdgeGeometry::num_vertices, LinearEdgeGeometry::spatial_dim> coords{};
    const MdArray<RealT, 1> lower_coordinate{ subdivisionBound( subelement_index ) };
    const MdArray<RealT, 1> upper_coordinate{ subdivisionBound( subelement_index + 1 ) };
    const auto lower_point = geometry.position( lower_coordinate.view() );
    const auto upper_point = geometry.position( upper_coordinate.view() );
    bool finite = true;
    for ( IndexT d = 0; d < LinearEdgeGeometry::spatial_dim; ++d ) {
      coords( 0, d ) = lower_point[d];
      coords( 1, d ) = upper_point[d];
      if ( !std::isfinite( lower_point[d] ) || !std::isfinite( upper_point[d] ) ) {
        finite = false;
      }
    }
    const LinearEdgeGeometry subelement_geometry( coords.view() );
    const RealT squared_length = subelement_geometry.squaredLength();
    const bool valid = finite && std::isfinite( squared_length ) && squared_length > 0.0;
    return { subelement_geometry, valid };
  }

  int num_subdivisions_;                                   ///< Number of uniform subelements on each surface element.
  SubelementPairRuleGenerator subelement_pair_generator_;  ///< Linear subelement-pair quadrature generator.
  Projection projection_;                                  ///< Maps paired subelement points onto both elements.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_INTEG_SUBDIVIDEDPAIRQUADRATURERULEGENERATOR_HPP_
