#ifndef TRIBOL_GEOM_SMOOTHEDSEGMENTOVERLAP_HPP_
#define TRIBOL_GEOM_SMOOTHEDSEGMENTOVERLAP_HPP_

#include "tribol/geom/ProjectedOverlap.hpp"

#include <algorithm>
#include <array>

namespace tribol {
namespace smoothed_segment_detail {

template <typename Scalar>
[[nodiscard]] inline Scalar cross( const std::array<Scalar, 2>& left, const std::array<Scalar, 2>& right )
{
  return left[0] * right[1] - left[1] * right[0];
}

template <typename Scalar>
[[nodiscard]] inline Scalar clamp( Scalar value, Real lower, Real upper )
{
  if ( linearization_detail::primal( value ) < lower ) {
    return Scalar{ lower };
  }
  if ( linearization_detail::primal( value ) > upper ) {
    return Scalar{ upper };
  }
  return value;
}

template <typename Scalar>
[[nodiscard]] inline Scalar smoothBound( Scalar value, Real width )
{
  if ( width == 0.0 ) {
    return value;
  }
  const Real primal_value = linearization_detail::primal( value );
  if ( primal_value <= width ) {
    const Scalar shifted = value + width;
    return shifted * shifted / ( 4.0 * width );
  }
  if ( primal_value >= 1.0 - width ) {
    const Scalar shifted = 1.0 + width - value;
    return 1.0 - shifted * shifted / ( 4.0 * width );
  }
  return value;
}

template <typename Scalar>
[[nodiscard]] inline std::array<Scalar, 2> point( const SurfaceMeshViewT<Scalar>& mesh, Index element, int local_node )
{
  const Index node = mesh.connectivity[mesh.element_offsets[element] + local_node];
  return { mesh.coordinates( node, 0 ), mesh.coordinates( node, 1 ) };
}

template <typename Scalar>
[[nodiscard]] inline Scalar projectionParameter( const std::array<Scalar, 2>& point_on_source,
                                                 const std::array<Scalar, 2>& target_first,
                                                 const std::array<Scalar, 2>& target_tangent,
                                                 const std::array<Scalar, 3>& source_normal )
{
  const std::array<Scalar, 2> offset{ point_on_source[0] - target_first[0], point_on_source[1] - target_first[1] };
  const std::array<Scalar, 2> direction{ source_normal[0], source_normal[1] };
  return cross( offset, direction ) / cross( target_tangent, direction );
}

template <typename Scalar>
[[nodiscard]] inline std::array<Scalar, 2> interpolate( const std::array<Scalar, 2>& first,
                                                        const std::array<Scalar, 2>& second, Scalar parameter )
{
  return { first[0] + parameter * ( second[0] - first[0] ), first[1] + parameter * ( second[1] - first[1] ) };
}

template <typename Scalar>
[[nodiscard]] inline std::array<Scalar, 2> projectToLine( const std::array<Scalar, 2>& point_to_project,
                                                          const std::array<Scalar, 2>& line_point,
                                                          const std::array<Scalar, 3>& line_normal )
{
  const Scalar distance = ( point_to_project[0] - line_point[0] ) * line_normal[0] +
                          ( point_to_project[1] - line_point[1] ) * line_normal[1];
  return { point_to_project[0] - distance * line_normal[0], point_to_project[1] - distance * line_normal[1] };
}

}  // namespace smoothed_segment_detail

template <NormalPolicy Normal, typename Integration, typename Scalar>
[[nodiscard]] inline InteractionPatchT<Scalar> smoothedProjectedSegmentPatch(
    const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
    const typename geometry::ProjectedOverlap<Normal>::Parameters& geometry_parameters,
    const typename Integration::Parameters& integration_parameters )
{
  static_assert( detail::is_smoothed_segment_integration_v<Integration> );
  InteractionPatchT<Scalar> result;
  if ( surfaces.mortar.dimension != 2 ||
       !projected_overlap_detail::isLinearSegment( surfaces.mortar, pair.mortar_element ) ||
       !projected_overlap_detail::isLinearSegment( surfaces.nonmortar, pair.nonmortar_element ) ) {
    return result;
  }

  const auto mortar_first = smoothed_segment_detail::point( surfaces.mortar, pair.mortar_element, 0 );
  const auto mortar_second = smoothed_segment_detail::point( surfaces.mortar, pair.mortar_element, 1 );
  const auto nonmortar_first = smoothed_segment_detail::point( surfaces.nonmortar, pair.nonmortar_element, 0 );
  const auto nonmortar_second = smoothed_segment_detail::point( surfaces.nonmortar, pair.nonmortar_element, 1 );
  const std::array<Scalar, 2> mortar_tangent{ mortar_second[0] - mortar_first[0], mortar_second[1] - mortar_first[1] };
  const auto mortar_normal = projected_overlap_detail::segmentNormal( surfaces.mortar, pair.mortar_element );
  const auto nonmortar_normal = projected_overlap_detail::segmentNormal( surfaces.nonmortar, pair.nonmortar_element );
  const Scalar determinant = smoothed_segment_detail::cross(
      mortar_tangent, std::array<Scalar, 2>{ nonmortar_normal[0], nonmortar_normal[1] } );
  if ( linearization_detail::absolute( determinant ) <= 1.0e-28 ) {
    return result;
  }

  Scalar first_projection =
      smoothed_segment_detail::projectionParameter( nonmortar_first, mortar_first, mortar_tangent, nonmortar_normal );
  Scalar second_projection =
      smoothed_segment_detail::projectionParameter( nonmortar_second, mortar_first, mortar_tangent, nonmortar_normal );
  if ( linearization_detail::primal( second_projection ) < linearization_detail::primal( first_projection ) ) {
    std::swap( first_projection, second_projection );
  }
  const Real width = integration_parameters.endpoint_width;
  Scalar lower = smoothed_segment_detail::clamp( first_projection, -width, 1.0 + width );
  Scalar upper = smoothed_segment_detail::clamp( second_projection, -width, 1.0 + width );
  lower = smoothed_segment_detail::smoothBound( lower, width );
  upper = smoothed_segment_detail::smoothBound( upper, width );

  const Scalar mortar_length =
      projected_overlap_detail::norm( std::array<Scalar, 3>{ mortar_tangent[0], mortar_tangent[1], Scalar{} } );
  result.measure = ( upper - lower ) * mortar_length;
  if ( linearization_detail::primal( result.measure ) <= geometry_parameters.minimum_measure ||
       !projected_overlap_detail::meetsOverlapFraction<Scalar, Normal>( surfaces, pair, result.measure,
                                                                        geometry_parameters ) ) {
    return {};
  }

  result.normal = projected_overlap_detail::selectNormal<Normal>( mortar_normal, nonmortar_normal );
  result.vertex_count = 2;
  result.manifold_dimension = 1;
  const Scalar locations[3] = { lower, upper, 0.5 * ( lower + upper ) };
  for ( int location = 0; location < 3; ++location ) {
    const auto mortar_point = smoothed_segment_detail::interpolate( mortar_first, mortar_second, locations[location] );
    const auto nonmortar_point =
        smoothed_segment_detail::projectToLine( mortar_point, nonmortar_first, nonmortar_normal );
    std::array<Scalar, 3>& mortar_output = location < 2 ? result.mortar_vertices[location] : result.mortar_centroid;
    std::array<Scalar, 3>& nonmortar_output =
        location < 2 ? result.nonmortar_vertices[location] : result.nonmortar_centroid;
    for ( int component = 0; component < 2; ++component ) {
      mortar_output[component] = mortar_point[component];
      nonmortar_output[component] = nonmortar_point[component];
      if ( location < 2 ) {
        result.integration_vertices[location][component] =
            0.5 * ( mortar_point[component] + nonmortar_point[component] );
      }
    }
  }
  result.valid = true;
  return result;
}

}  // namespace tribol

#endif
