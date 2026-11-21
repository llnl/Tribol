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

/**
 * @brief Generate element-edge quadrature by applying a linear rule to every pair of uniform linear subelements.
 * @tparam ElementGeometry Current surface-element geometry evaluated in one parametric coordinate.
 * @tparam InnerPairRuleGenerator Rule generator operating on two linear subelement edges.
 * @tparam MaxNumSubdivisions Largest accepted number of subelements on either surface element.
 */
template <typename ElementGeometry, PairRuleGenerator InnerPairRuleGenerator, int MaxNumSubdivisions>
  requires( ElementGeometry::parametric_dim == 1 && InnerPairRuleGenerator::Geometry1::parametric_dim == 1 &&
            InnerPairRuleGenerator::Geometry2::parametric_dim == 1 &&
            requires { typename std::integral_constant<int, InnerPairRuleGenerator::max_num_points>; } )
class SubdividedPairQuadratureRuleGenerator {
 public:
  using Geometry1 = ElementGeometry;                             ///< First surface-element geometry.
  using Geometry2 = ElementGeometry;                             ///< Second surface-element geometry.
  static constexpr int max_num_subdivisions = MaxNumSubdivisions;  ///< Fixed upper bound on uniform subelements.
  static constexpr int max_num_points =
      MaxNumSubdivisions * MaxNumSubdivisions * InnerPairRuleGenerator::max_num_points;  ///< Fixed point capacity.
  static constexpr int polynomial_order = InnerPairRuleGenerator::polynomial_order;  ///< Inner-rule exactness order.
  using Rule = PairQuadratureRule<Geometry1::parametric_dim, Geometry2::parametric_dim,
                                  max_num_points>;  ///< Fixed-capacity element-coordinate rule.

  static_assert( MaxNumSubdivisions > 0 );
  static_assert( std::same_as<typename InnerPairRuleGenerator::Geometry1, LinearEdgeGeometry> );
  static_assert( std::same_as<typename InnerPairRuleGenerator::Geometry2, LinearEdgeGeometry> );

  /**
   * @brief Store the number of uniform subelements and configured inner generator.
   * @param num_subdivisions Number of subelements on each surface element.
   * @param inner_generator Linear subelement-pair rule generator.
   */
  TRIBOL_HOST_DEVICE explicit SubdividedPairQuadratureRuleGenerator( int num_subdivisions,
                                                                     InnerPairRuleGenerator inner_generator = {} )
      : num_subdivisions_( num_subdivisions ), inner_generator_( std::move( inner_generator ) )
  {
  }

  /**
   * @brief Generate all subelement-pair rules and express their coordinates in both surface elements.
   * @param geometry1 First current surface-element geometry.
   * @param geometry2 Second current surface-element geometry.
   * @return Element-parametric points, unchanged physical subelement weights, and generation status.
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
      const RealT lower1 = subdivisionBound( subelement1 );
      const RealT upper1 = subdivisionBound( subelement1 + 1 );
      MdArray<RealT, LinearEdgeGeometry::num_vertices, LinearEdgeGeometry::spatial_dim> subelement_coords1{};
      if ( !makeSubelementCoordinates( geometry1, lower1, upper1, subelement_coords1 ) ) {
        continue;
      }
      const LinearEdgeGeometry subelement_geometry1( subelement_coords1.view() );

      for ( int subelement2 = 0; subelement2 < num_subdivisions_; ++subelement2 ) {
        const RealT lower2 = subdivisionBound( subelement2 );
        const RealT upper2 = subdivisionBound( subelement2 + 1 );
        MdArray<RealT, LinearEdgeGeometry::num_vertices, LinearEdgeGeometry::spatial_dim> subelement_coords2{};
        if ( !makeSubelementCoordinates( geometry2, lower2, upper2, subelement_coords2 ) ) {
          continue;
        }
        const LinearEdgeGeometry subelement_geometry2( subelement_coords2.view() );
        const auto subelement_rule = inner_generator_.generate( subelement_geometry1, subelement_geometry2 );
        if ( subelement_rule.metadata.status != QuadratureStatus::Success ) {
          output.metadata = { subelement_rule.metadata.status, subelement_rule.metadata.num_required_points };
          return output;
        }
        if ( output_point + subelement_rule.metadata.num_required_points > max_num_points ) {
          output.metadata = { QuadratureStatus::InsufficientCapacity,
                              output_point + subelement_rule.metadata.num_required_points };
          return output;
        }
        for ( IndexT point = 0; point < subelement_rule.metadata.num_required_points; ++point ) {
          output.geom1_parametric_coords( output_point, 0 ) =
              mapToElement( subelement_rule.geom1_parametric_coords( point, 0 ), lower1, upper1 );
          output.geom2_parametric_coords( output_point, 0 ) =
              mapToElement( subelement_rule.geom2_parametric_coords( point, 0 ), lower2, upper2 );
          output.weights[output_point] = subelement_rule.weights[point];
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
   * @brief Map one subelement coordinate on [-1, 1] into its surface-element interval.
   * @param subelement_coordinate Coordinate in the linear subelement.
   * @param lower Lower element parametric coordinate of the subelement interval.
   * @param upper Upper element parametric coordinate of the subelement interval.
   * @return Parametric coordinate in the surface element.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static RealT mapToElement( RealT subelement_coordinate, RealT lower,
                                                                     RealT upper )
  {
    return 0.5 * ( ( 1.0 - subelement_coordinate ) * lower + ( 1.0 + subelement_coordinate ) * upper );
  }

  /**
   * @brief Evaluate a surface element at subelement endpoints and reject unusable linear chords.
   * @param geometry Surface-element geometry evaluated at both interval endpoints.
   * @param lower Lower element parametric coordinate of the subelement interval.
   * @param upper Upper element parametric coordinate of the subelement interval.
   * @param[out] subelement_coords Physical endpoints when they are finite.
   * @return Whether both endpoints are finite and define a nonzero chord.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static bool makeSubelementCoordinates(
      const ElementGeometry& geometry, RealT lower, RealT upper,
      MdArray<RealT, LinearEdgeGeometry::num_vertices, LinearEdgeGeometry::spatial_dim>& subelement_coords )
  {
    const MdArray<RealT, 1> lower_coordinate{ lower };
    const MdArray<RealT, 1> upper_coordinate{ upper };
    const auto lower_point = geometry.position( lower_coordinate.view() );
    const auto upper_point = geometry.position( upper_coordinate.view() );
    RealT squared_length = 0.0;
    for ( IndexT component = 0; component < LinearEdgeGeometry::spatial_dim; ++component ) {
      subelement_coords( 0, component ) = lower_point[component];
      subelement_coords( 1, component ) = upper_point[component];
      const RealT difference = upper_point[component] - lower_point[component];
      if ( !std::isfinite( lower_point[component] ) || !std::isfinite( upper_point[component] ) ) {
        return false;
      }
      squared_length += difference * difference;
    }
    return std::isfinite( squared_length ) && squared_length > 0.0;
  }

  int num_subdivisions_;                    ///< Number of uniform subelements on each surface element.
  InnerPairRuleGenerator inner_generator_;  ///< Linear subelement-pair quadrature generator.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_INTEG_SUBDIVIDEDPAIRQUADRATURERULEGENERATOR_HPP_
