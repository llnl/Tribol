// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Compose parent-element subdivision with a linear child-pair quadrature generator. */
#ifndef SRC_TRIBOL_INTEG_SUBDIVIDEDPAIRQUADRATURERULEGENERATOR_HPP_
#define SRC_TRIBOL_INTEG_SUBDIVIDEDPAIRQUADRATURERULEGENERATOR_HPP_

#include <cmath>
#include <concepts>
#include <utility>

#include "tribol/geom/LinearEdgeGeometry.hpp"
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"

namespace tribol {

/**
 * @brief Generate parent-edge quadrature by applying a linear rule to every pair of uniform child chords.
 * @tparam ParentGeometry Current parent geometry evaluated in one parametric coordinate.
 * @tparam InnerPairRuleGenerator Rule generator operating on two linear child edges.
 * @tparam MaxSubdivisionCount Largest accepted number of children on either parent.
 */
template <typename ParentGeometry, PairRuleGenerator InnerPairRuleGenerator, int MaxSubdivisionCount>
  requires( ParentGeometry::parametric_dim == 1 && InnerPairRuleGenerator::Geometry1::parametric_dim == 1 &&
            InnerPairRuleGenerator::Geometry2::parametric_dim == 1 &&
            requires { typename std::integral_constant<int, InnerPairRuleGenerator::max_num_points>; } )
class SubdividedPairQuadratureRuleGenerator {
 public:
  using Geometry1 = ParentGeometry;                                  ///< First parent-element geometry.
  using Geometry2 = ParentGeometry;                                  ///< Second parent-element geometry.
  static constexpr int max_subdivision_count = MaxSubdivisionCount;  ///< Fixed child count bound.
  static constexpr int max_num_points =
      MaxSubdivisionCount * MaxSubdivisionCount * InnerPairRuleGenerator::max_num_points;  ///< Fixed point capacity.
  static constexpr int polynomial_order = InnerPairRuleGenerator::polynomial_order;  ///< Inner-rule exactness order.
  using Rule = PairQuadratureRule<Geometry1::parametric_dim, Geometry2::parametric_dim,
                                  max_num_points>;  ///< Fixed-capacity parent-coordinate rule.

  static_assert( MaxSubdivisionCount > 0 );
  static_assert( std::same_as<typename InnerPairRuleGenerator::Geometry1, LinearEdgeGeometry> );
  static_assert( std::same_as<typename InnerPairRuleGenerator::Geometry2, LinearEdgeGeometry> );

  /**
   * @brief Store the uniform child count and configured inner generator.
   * @param subdivision_count Number of children on each parent edge.
   * @param inner_generator Linear child-pair rule generator.
   */
  TRIBOL_HOST_DEVICE explicit SubdividedPairQuadratureRuleGenerator( int subdivision_count,
                                                                     InnerPairRuleGenerator inner_generator = {} )
      : subdivision_count_( subdivision_count ), inner_generator_( std::move( inner_generator ) )
  {
  }

  /**
   * @brief Generate all child-pair rules and express their coordinates in both parent elements.
   * @param geometry1 First current parent geometry.
   * @param geometry2 Second current parent geometry.
   * @return Parent-parametric points, unchanged physical child weights, and generation status.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Rule generate( const Geometry1& geometry1, const Geometry2& geometry2 ) const
  {
    Rule output{};
    if ( subdivision_count_ < 1 || subdivision_count_ > MaxSubdivisionCount ) {
      output.metadata = { QuadratureStatus::InvalidGeometry, 0 };
      return output;
    }

    IndexT output_point = 0;
    for ( int child1 = 0; child1 < subdivision_count_; ++child1 ) {
      const RealT lower1 = parentBound( child1 );
      const RealT upper1 = parentBound( child1 + 1 );
      MdArray<RealT, LinearEdgeGeometry::num_vertices, LinearEdgeGeometry::spatial_dim> child_coords1{};
      if ( !makeChildCoordinates( geometry1, lower1, upper1, child_coords1 ) ) {
        continue;
      }
      const LinearEdgeGeometry child_geometry1( child_coords1.view() );

      for ( int child2 = 0; child2 < subdivision_count_; ++child2 ) {
        const RealT lower2 = parentBound( child2 );
        const RealT upper2 = parentBound( child2 + 1 );
        MdArray<RealT, LinearEdgeGeometry::num_vertices, LinearEdgeGeometry::spatial_dim> child_coords2{};
        if ( !makeChildCoordinates( geometry2, lower2, upper2, child_coords2 ) ) {
          continue;
        }
        const LinearEdgeGeometry child_geometry2( child_coords2.view() );
        const auto child_rule = inner_generator_.generate( child_geometry1, child_geometry2 );
        if ( child_rule.metadata.status != QuadratureStatus::Success ) {
          output.metadata = { child_rule.metadata.status, child_rule.metadata.num_required_points };
          return output;
        }
        if ( output_point + child_rule.metadata.num_required_points > max_num_points ) {
          output.metadata = { QuadratureStatus::InsufficientCapacity,
                              output_point + child_rule.metadata.num_required_points };
          return output;
        }
        for ( IndexT point = 0; point < child_rule.metadata.num_required_points; ++point ) {
          output.geom1_parametric_coords( output_point, 0 ) =
              mapToParent( child_rule.geom1_parametric_coords( point, 0 ), lower1, upper1 );
          output.geom2_parametric_coords( output_point, 0 ) =
              mapToParent( child_rule.geom2_parametric_coords( point, 0 ), lower2, upper2 );
          output.weights[output_point] = child_rule.weights[point];
          ++output_point;
        }
      }
    }
    output.metadata.num_required_points = output_point;
    return output;
  }

 private:
  /**
   * @brief Return one uniform child boundary in parent coordinates.
   * @param boundary Boundary index from zero through the active subdivision count.
   * @return Parent coordinate in [-1,1].
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT parentBound( int boundary ) const
  {
    return -1.0 + 2.0 * static_cast<RealT>( boundary ) / static_cast<RealT>( subdivision_count_ );
  }

  /**
   * @brief Map one child coordinate on [-1,1] into its parent interval.
   * @param child_coordinate Coordinate in the linear child element.
   * @param lower Lower parent coordinate of the child interval.
   * @param upper Upper parent coordinate of the child interval.
   * @return Coordinate in the parent element.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static RealT mapToParent( RealT child_coordinate, RealT lower, RealT upper )
  {
    return 0.5 * ( ( 1.0 - child_coordinate ) * lower + ( 1.0 + child_coordinate ) * upper );
  }

  /**
   * @brief Evaluate a parent at child endpoints and reject unusable linear chords.
   * @param geometry Parent geometry evaluated at both interval endpoints.
   * @param lower Lower parent coordinate of the child interval.
   * @param upper Upper parent coordinate of the child interval.
   * @param[out] child_coords Physical endpoints when they are finite.
   * @return Whether both endpoints are finite and define a nonzero chord.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static bool makeChildCoordinates(
      const ParentGeometry& geometry, RealT lower, RealT upper,
      MdArray<RealT, LinearEdgeGeometry::num_vertices, LinearEdgeGeometry::spatial_dim>& child_coords )
  {
    const MdArray<RealT, 1> lower_coordinate{ lower };
    const MdArray<RealT, 1> upper_coordinate{ upper };
    const auto lower_point = geometry.position( lower_coordinate.view() );
    const auto upper_point = geometry.position( upper_coordinate.view() );
    RealT squared_length = 0.0;
    for ( IndexT component = 0; component < LinearEdgeGeometry::spatial_dim; ++component ) {
      child_coords( 0, component ) = lower_point[component];
      child_coords( 1, component ) = upper_point[component];
      const RealT difference = upper_point[component] - lower_point[component];
      if ( !std::isfinite( lower_point[component] ) || !std::isfinite( upper_point[component] ) ) {
        return false;
      }
      squared_length += difference * difference;
    }
    return std::isfinite( squared_length ) && squared_length > 0.0;
  }

  int subdivision_count_;                   ///< Number of uniform children on each parent.
  InnerPairRuleGenerator inner_generator_;  ///< Linear child-pair quadrature generator.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_INTEG_SUBDIVIDEDPAIRQUADRATURERULEGENERATOR_HPP_
