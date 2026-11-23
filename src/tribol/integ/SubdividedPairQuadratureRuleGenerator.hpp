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
 * @tparam SubelementPairRuleGenerator Rule generator operating on two linear subelement edges.
 * @tparam MaxNumSubdivisions Largest accepted number of subelements on either surface element.
 */
template <typename ElementGeometry, PairRuleGenerator SubelementPairRuleGenerator, int MaxNumSubdivisions>
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
   * @brief Store the number of uniform subelements and configured subelement-pair generator.
   * @param num_subdivisions Number of subelements on each surface element.
   * @param subelement_pair_generator Linear subelement-pair rule generator.
   */
  TRIBOL_HOST_DEVICE explicit SubdividedPairQuadratureRuleGenerator(
      int num_subdivisions, SubelementPairRuleGenerator subelement_pair_generator = {} )
      : num_subdivisions_( num_subdivisions ), subelement_pair_generator_( std::move( subelement_pair_generator ) )
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
        for ( IndexT point = 0; point < subelement_rule.metadata.num_required_points; ++point ) {
          output.geom1_parametric_coords( output_point, 0 ) =
              mapToElement( subelement_rule.geom1_parametric_coords( point, 0 ), subelement1 );
          output.geom2_parametric_coords( output_point, 0 ) =
              mapToElement( subelement_rule.geom2_parametric_coords( point, 0 ), subelement2 );
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
   * @param subelement_index Index of the subelement in [0, num_subdivisions_).
   * @return Parametric coordinate in the surface element.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT mapToElement( RealT subelement_coordinate, int subelement_index ) const
  {
    const RealT lower = subdivisionBound( subelement_index );
    const RealT upper = subdivisionBound( subelement_index + 1 );
    return 0.5 * ( ( 1.0 - subelement_coordinate ) * lower + ( 1.0 + subelement_coordinate ) * upper );
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
};

}  // namespace tribol

#endif  // SRC_TRIBOL_INTEG_SUBDIVIDEDPAIRQUADRATURERULEGENERATOR_HPP_
