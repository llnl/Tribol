#ifndef TRIBOL_GEOM_PROJECTEDOVERLAP_HPP_
#define TRIBOL_GEOM_PROJECTEDOVERLAP_HPP_

#include "tribol/core/ExactTangent.hpp"
#include "tribol/core/MeshView.hpp"
#include "tribol/method/Method.hpp"
#include "tribol/search/Search.hpp"

#include <algorithm>
#include <array>
#include <type_traits>

namespace tribol {

template <typename Scalar>
struct InteractionGeometryT {
  std::array<Scalar, 3> normal{};
  std::array<Scalar, 3> mortar_point{};
  std::array<Scalar, 3> nonmortar_point{};
  Scalar measure{};
  Scalar gap{};
  bool valid{};
};

using InteractionGeometry = InteractionGeometryT<Real>;

inline constexpr int maximumInteractionVertices = 8;

template <typename Scalar>
struct InteractionPatchT {
  std::array<Scalar, 3> normal{};
  std::array<std::array<Scalar, 3>, maximumInteractionVertices> integration_vertices{};
  std::array<std::array<Scalar, 3>, maximumInteractionVertices> mortar_vertices{};
  std::array<std::array<Scalar, 3>, maximumInteractionVertices> nonmortar_vertices{};
  std::array<Scalar, 3> mortar_centroid{};
  std::array<Scalar, 3> nonmortar_centroid{};
  Scalar measure{};
  int vertex_count{};
  int manifold_dimension{};
  bool valid{};
};

using InteractionPatch = InteractionPatchT<Real>;

namespace projected_overlap_detail {

template <typename Scalar, std::size_t Size>
[[nodiscard]] inline std::array<Scalar, Size> subtract( const std::array<Scalar, Size>& left,
                                                        const std::array<Scalar, Size>& right )
{
  std::array<Scalar, Size> result{};
  for ( std::size_t component = 0; component < Size; ++component ) {
    result[component] = left[component] - right[component];
  }
  return result;
}

template <typename Scalar>
[[nodiscard]] inline std::array<Scalar, 3> cross( const std::array<Scalar, 3>& left,
                                                  const std::array<Scalar, 3>& right )
{
  return { left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2],
           left[0] * right[1] - left[1] * right[0] };
}

template <typename Scalar, std::size_t Size>
[[nodiscard]] inline Scalar dot( const std::array<Scalar, Size>& left, const std::array<Scalar, Size>& right )
{
  Scalar result{};
  for ( std::size_t component = 0; component < Size; ++component ) {
    result += left[component] * right[component];
  }
  return result;
}

template <typename Scalar>
[[nodiscard]] inline Scalar norm( const std::array<Scalar, 3>& value )
{
  return linearization_detail::squareRoot( dot( value, value ) );
}

template <typename Scalar>
[[nodiscard]] inline std::array<Scalar, 3> normalize( std::array<Scalar, 3> value )
{
  const Scalar length = norm( value );
  if ( length <= 1.0e-28 ) {
    return {};
  }
  for ( Scalar& component : value ) {
    component /= length;
  }
  return value;
}

template <typename Scalar>
[[nodiscard]] inline std::array<Scalar, 3> segmentNormal( const SurfaceMeshViewT<Scalar>& mesh, Index element )
{
  const Index begin = mesh.element_offsets[element];
  const Index first_node = mesh.connectivity[begin];
  const Index second_node = mesh.connectivity[begin + 1];
  const Scalar dx = mesh.coordinates( second_node, 0 ) - mesh.coordinates( first_node, 0 );
  const Scalar dy = mesh.coordinates( second_node, 1 ) - mesh.coordinates( first_node, 1 );
  const Scalar length = linearization_detail::squareRoot( dx * dx + dy * dy );
  if ( length <= 1.0e-28 ) {
    return {};
  }
  return { dy / length, -dx / length, {} };
}

template <typename Scalar>
[[nodiscard]] inline std::array<Scalar, 3> normalizedDifference( const std::array<Scalar, 3>& left,
                                                                 const std::array<Scalar, 3>& right )
{
  auto result = subtract( left, right );
  if ( norm( result ) <= 1.0e-14 ) {
    return left;
  }
  return normalize( result );
}

template <NormalPolicy Normal, typename Scalar>
[[nodiscard]] inline std::array<Scalar, 3> selectNormal( const std::array<Scalar, 3>& mortar,
                                                         const std::array<Scalar, 3>& nonmortar )
{
  if constexpr ( std::same_as<Normal, normal::MeanPlane> ) {
    return normalizedDifference( nonmortar, mortar );
  } else {
    return { -mortar[0], -mortar[1], -mortar[2] };
  }
}

template <typename Scalar>
[[nodiscard]] inline bool isLinearSegment( const SurfaceMeshViewT<Scalar>& mesh, Index element )
{
  return mesh.topologies[element] == ElementTopology::Segment &&
         mesh.element_offsets[element + 1] - mesh.element_offsets[element] == 2;
}

template <typename Scalar>
[[nodiscard]] inline bool isLinearFace( const SurfaceMeshViewT<Scalar>& mesh, Index element )
{
  const Index nodes = mesh.element_offsets[element + 1] - mesh.element_offsets[element];
  return ( mesh.topologies[element] == ElementTopology::Triangle && nodes == 3 ) ||
         ( mesh.topologies[element] == ElementTopology::Quadrilateral && nodes == 4 );
}

template <typename Scalar>
[[nodiscard]] inline std::array<Scalar, 3> facePoint( const SurfaceMeshViewT<Scalar>& mesh, Index element,
                                                      Index local_node )
{
  const Index node = mesh.connectivity[mesh.element_offsets[element] + local_node];
  return { mesh.coordinates( node, 0 ), mesh.coordinates( node, 1 ), mesh.coordinates( node, 2 ) };
}

template <typename Scalar>
[[nodiscard]] inline std::array<Scalar, 3> faceNormal( const SurfaceMeshViewT<Scalar>& mesh, Index element )
{
  const auto first = facePoint( mesh, element, 0 );
  const auto second = facePoint( mesh, element, 1 );
  const auto third = facePoint( mesh, element, 2 );
  auto normal = cross( subtract( second, first ), subtract( third, first ) );
  if ( mesh.topologies[element] == ElementTopology::Quadrilateral ) {
    const auto fourth = facePoint( mesh, element, 3 );
    const auto second_normal = cross( subtract( third, first ), subtract( fourth, first ) );
    for ( int component = 0; component < 3; ++component ) {
      normal[component] += second_normal[component];
    }
  }
  return normalize( normal );
}

template <typename Scalar>
[[nodiscard]] inline Scalar faceMeasure( const SurfaceMeshViewT<Scalar>& mesh, Index element )
{
  if ( mesh.topologies[element] == ElementTopology::Segment ) {
    const Index begin = mesh.element_offsets[element];
    const Index first = mesh.connectivity[begin];
    const Index second = mesh.connectivity[begin + 1];
    const Scalar dx = mesh.coordinates( second, 0 ) - mesh.coordinates( first, 0 );
    const Scalar dy = mesh.coordinates( second, 1 ) - mesh.coordinates( first, 1 );
    return linearization_detail::squareRoot( dx * dx + dy * dy );
  }
  const auto first = facePoint( mesh, element, 0 );
  const auto second = facePoint( mesh, element, 1 );
  const auto third = facePoint( mesh, element, 2 );
  Scalar measure = 0.5 * norm( cross( subtract( second, first ), subtract( third, first ) ) );
  if ( mesh.topologies[element] == ElementTopology::Quadrilateral ) {
    const auto fourth = facePoint( mesh, element, 3 );
    measure += 0.5 * norm( cross( subtract( third, first ), subtract( fourth, first ) ) );
  }
  return measure;
}

template <typename Scalar, NormalPolicy Normal>
[[nodiscard]] inline bool meetsOverlapFraction(
    const SurfacePairViewT<Scalar>& surfaces, ElementPair pair, Scalar overlap_measure,
    const typename geometry::ProjectedOverlap<Normal>::Parameters& parameters )
{
  if ( parameters.minimum_overlap_fraction == 0.0 ) {
    return true;
  }
  const Real mortar_measure = linearization_detail::primal( faceMeasure( surfaces.mortar, pair.mortar_element ) );
  const Real nonmortar_measure =
      linearization_detail::primal( faceMeasure( surfaces.nonmortar, pair.nonmortar_element ) );
  const Real reference_measure = std::max( mortar_measure, nonmortar_measure );
  return reference_measure > 0.0 &&
         linearization_detail::primal( overlap_measure ) >= parameters.minimum_overlap_fraction * reference_measure;
}

template <typename Scalar>
[[nodiscard]] inline Scalar cross2d( const std::array<Scalar, 2>& left, const std::array<Scalar, 2>& right )
{
  return left[0] * right[1] - left[1] * right[0];
}

template <typename Scalar>
[[nodiscard]] inline bool insideHalfPlane( const std::array<Scalar, 2>& point, const std::array<Scalar, 2>& first,
                                           const std::array<Scalar, 2>& second, Real orientation )
{
  return orientation * cross2d( subtract( second, first ), subtract( point, first ) ) >= -1.0e-14;
}

template <typename Scalar>
[[nodiscard]] inline std::array<Scalar, 2> lineIntersection( const std::array<Scalar, 2>& first,
                                                             const std::array<Scalar, 2>& second,
                                                             const std::array<Scalar, 2>& clip_first,
                                                             const std::array<Scalar, 2>& clip_second )
{
  const auto direction = subtract( second, first );
  const auto clip_direction = subtract( clip_second, clip_first );
  const Scalar denominator = cross2d( direction, clip_direction );
  if ( linearization_detail::absolute( denominator ) < 1.0e-28 ) {
    return first;
  }
  const Scalar parameter = cross2d( subtract( clip_first, first ), clip_direction ) / denominator;
  return { first[0] + parameter * direction[0], first[1] + parameter * direction[1] };
}

template <typename Scalar>
struct ProjectedPolygonT {
  std::array<std::array<Scalar, 2>, maximumInteractionVertices> points{};
  int size{};
};

template <typename Scalar>
[[nodiscard]] inline ProjectedPolygonT<Scalar> clipPolygon( ProjectedPolygonT<Scalar> subject,
                                                            const std::array<std::array<Scalar, 2>, 4>& clip,
                                                            int clip_size )
{
  Scalar signed_area{};
  for ( int vertex = 0; vertex < clip_size; ++vertex ) {
    signed_area += cross2d( clip[vertex], clip[( vertex + 1 ) % clip_size] );
  }
  const Real orientation = linearization_detail::primal( signed_area ) >= 0.0 ? 1.0 : -1.0;
  for ( int edge = 0; edge < clip_size && subject.size > 0; ++edge ) {
    ProjectedPolygonT<Scalar> output;
    const auto& clip_first = clip[edge];
    const auto& clip_second = clip[( edge + 1 ) % clip_size];
    auto previous = subject.points[subject.size - 1];
    bool previous_inside = insideHalfPlane( previous, clip_first, clip_second, orientation );
    for ( int vertex = 0; vertex < subject.size; ++vertex ) {
      const auto current = subject.points[vertex];
      const bool current_inside = insideHalfPlane( current, clip_first, clip_second, orientation );
      if ( current_inside != previous_inside && output.size < static_cast<int>( output.points.size() ) ) {
        output.points[output.size++] = lineIntersection( previous, current, clip_first, clip_second );
      }
      if ( current_inside && output.size < static_cast<int>( output.points.size() ) ) {
        output.points[output.size++] = current;
      }
      previous = current;
      previous_inside = current_inside;
    }
    subject = output;
  }
  return subject;
}

template <NormalPolicy Normal, typename Scalar>
[[nodiscard]] inline InteractionPatchT<Scalar> projectedFaceOverlapPatch(
    const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
    const typename geometry::ProjectedOverlap<Normal>::Parameters& parameters )
{
  InteractionPatchT<Scalar> result;
  if ( !isLinearFace( surfaces.mortar, pair.mortar_element ) ||
       !isLinearFace( surfaces.nonmortar, pair.nonmortar_element ) ) {
    return result;
  }

  const auto mortar_normal = faceNormal( surfaces.mortar, pair.mortar_element );
  const auto nonmortar_normal = faceNormal( surfaces.nonmortar, pair.nonmortar_element );
  const auto normal = selectNormal<Normal>( mortar_normal, nonmortar_normal );
  if ( norm( normal ) <= 1.0e-28 ) {
    return result;
  }

  auto first_axis = subtract( facePoint( surfaces.mortar, pair.mortar_element, 1 ),
                              facePoint( surfaces.mortar, pair.mortar_element, 0 ) );
  const Scalar normal_component = dot( first_axis, normal );
  for ( int component = 0; component < 3; ++component ) {
    first_axis[component] -= normal_component * normal[component];
  }
  first_axis = normalize( first_axis );
  if ( norm( first_axis ) <= 1.0e-28 ) {
    return result;
  }
  const auto second_axis = cross( normal, first_axis );
  const auto origin = facePoint( surfaces.nonmortar, pair.nonmortar_element, 0 );
  const SurfaceMeshViewT<Scalar> meshes[2] = { surfaces.mortar, surfaces.nonmortar };
  const Index elements[2] = { pair.mortar_element, pair.nonmortar_element };
  std::array<std::array<std::array<Scalar, 2>, 4>, 2> projected{};
  int counts[2]{};
  for ( int side = 0; side < 2; ++side ) {
    counts[side] = static_cast<int>( meshes[side].element_offsets[elements[side] + 1] -
                                     meshes[side].element_offsets[elements[side]] );
    for ( int node = 0; node < counts[side]; ++node ) {
      const auto relative = subtract( facePoint( meshes[side], elements[side], node ), origin );
      projected[side][node] = { dot( relative, first_axis ), dot( relative, second_axis ) };
    }
  }

  ProjectedPolygonT<Scalar> overlap;
  overlap.size = counts[0];
  for ( int node = 0; node < counts[0]; ++node ) {
    overlap.points[node] = projected[0][node];
  }
  overlap = clipPolygon( overlap, projected[1], counts[1] );
  if ( overlap.size < 3 ) {
    return result;
  }

  Scalar twice_area{};
  std::array<Scalar, 2> projected_centroid{};
  for ( int vertex = 0; vertex < overlap.size; ++vertex ) {
    const auto& first = overlap.points[vertex];
    const auto& second = overlap.points[( vertex + 1 ) % overlap.size];
    const Scalar cross_value = cross2d( first, second );
    twice_area += cross_value;
    projected_centroid[0] += ( first[0] + second[0] ) * cross_value;
    projected_centroid[1] += ( first[1] + second[1] ) * cross_value;
  }
  if ( linearization_detail::absolute( twice_area ) <= 2.0 * parameters.minimum_measure ) {
    return result;
  }
  projected_centroid[0] /= 3.0 * twice_area;
  projected_centroid[1] /= 3.0 * twice_area;
  result.measure = 0.5 * linearization_detail::absolute( twice_area );
  if ( !meetsOverlapFraction<Scalar, Normal>( surfaces, pair, result.measure, parameters ) ) {
    return {};
  }

  std::array<Scalar, 3> projected_point{};
  for ( int component = 0; component < 3; ++component ) {
    projected_point[component] = origin[component] + projected_centroid[0] * first_axis[component] +
                                 projected_centroid[1] * second_axis[component];
  }
  const std::array<std::array<Scalar, 3>, 2> face_normals{ mortar_normal, nonmortar_normal };
  auto lift_to_faces = [&]( const std::array<Scalar, 3>& point, std::array<Scalar, 3>& mortar_point,
                            std::array<Scalar, 3>& nonmortar_point ) {
    std::array<Scalar, 3>* lifted[2] = { &mortar_point, &nonmortar_point };
    for ( int side = 0; side < 2; ++side ) {
      const auto side_plane_point = facePoint( meshes[side], elements[side], 0 );
      const Scalar denominator = dot( face_normals[side], normal );
      if ( linearization_detail::absolute( denominator ) <= 1.0e-28 ) {
        return false;
      }
      const Scalar distance = dot( face_normals[side], subtract( side_plane_point, point ) ) / denominator;
      for ( int component = 0; component < 3; ++component ) {
        ( *lifted[side] )[component] = point[component] + distance * normal[component];
      }
    }
    return true;
  };
  if ( !lift_to_faces( projected_point, result.mortar_centroid, result.nonmortar_centroid ) ) {
    return {};
  }
  result.vertex_count = overlap.size;
  result.manifold_dimension = 2;
  for ( int vertex = 0; vertex < overlap.size; ++vertex ) {
    for ( int component = 0; component < 3; ++component ) {
      result.integration_vertices[vertex][component] = origin[component] +
                                                       overlap.points[vertex][0] * first_axis[component] +
                                                       overlap.points[vertex][1] * second_axis[component];
    }
    if ( !lift_to_faces( result.integration_vertices[vertex], result.mortar_vertices[vertex],
                         result.nonmortar_vertices[vertex] ) ) {
      return {};
    }
  }
  result.normal = normal;
  result.valid = true;
  return result;
}

}  // namespace projected_overlap_detail

template <NormalPolicy Normal, typename Scalar>
[[nodiscard]] inline InteractionPatchT<Scalar> projectedOverlapPatch(
    const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
    const typename geometry::ProjectedOverlap<Normal>::Parameters& parameters )
{
  InteractionPatchT<Scalar> result;
  if ( surfaces.mortar.dimension == 3 ) {
    return projected_overlap_detail::projectedFaceOverlapPatch<Normal>( surfaces, pair, parameters );
  }
  if ( surfaces.mortar.dimension != 2 ||
       !projected_overlap_detail::isLinearSegment( surfaces.mortar, pair.mortar_element ) ||
       !projected_overlap_detail::isLinearSegment( surfaces.nonmortar, pair.nonmortar_element ) ) {
    return result;
  }

  const auto mortar_normal = projected_overlap_detail::segmentNormal( surfaces.mortar, pair.mortar_element );
  const auto nonmortar_normal = projected_overlap_detail::segmentNormal( surfaces.nonmortar, pair.nonmortar_element );
  const auto normal = projected_overlap_detail::selectNormal<Normal>( mortar_normal, nonmortar_normal );
  const std::array<Scalar, 2> tangent{ -normal[1], normal[0] };
  const SurfaceMeshViewT<Scalar> meshes[2] = { surfaces.mortar, surfaces.nonmortar };
  const Index elements[2] = { pair.mortar_element, pair.nonmortar_element };
  Scalar projected[2][2]{};
  std::array<std::array<std::array<Scalar, 2>, 2>, 2> points{};

  for ( int side = 0; side < 2; ++side ) {
    const Index begin = meshes[side].element_offsets[elements[side]];
    for ( int node = 0; node < 2; ++node ) {
      const Index mesh_node = meshes[side].connectivity[begin + node];
      for ( int component = 0; component < 2; ++component ) {
        points[side][node][component] = meshes[side].coordinates( mesh_node, component );
      }
      projected[side][node] = points[side][node][0] * tangent[0] + points[side][node][1] * tangent[1];
    }
  }

  const Scalar mortar_lower = std::min( projected[0][0], projected[0][1] );
  const Scalar mortar_upper = std::max( projected[0][0], projected[0][1] );
  const Scalar nonmortar_lower = std::min( projected[1][0], projected[1][1] );
  const Scalar nonmortar_upper = std::max( projected[1][0], projected[1][1] );
  const Scalar lower = std::max( mortar_lower, nonmortar_lower );
  const Scalar upper = std::min( mortar_upper, nonmortar_upper );
  if ( upper - lower <= parameters.minimum_measure ) {
    return result;
  }

  result.normal = normal;
  result.measure = upper - lower;
  if ( !projected_overlap_detail::meetsOverlapFraction<Scalar, Normal>( surfaces, pair, result.measure, parameters ) ) {
    return {};
  }
  result.vertex_count = 2;
  result.manifold_dimension = 1;
  const Scalar locations[3] = { lower, upper, 0.5 * ( lower + upper ) };
  for ( int location = 0; location < 3; ++location ) {
    std::array<Scalar, 3>* side_points[2] = {
        location < 2 ? &result.mortar_vertices[location] : &result.mortar_centroid,
        location < 2 ? &result.nonmortar_vertices[location] : &result.nonmortar_centroid };
    for ( int side = 0; side < 2; ++side ) {
      const Scalar denominator = projected[side][1] - projected[side][0];
      if ( linearization_detail::absolute( denominator ) <= 1.0e-28 ) {
        return {};
      }
      const Scalar parameter = ( locations[location] - projected[side][0] ) / denominator;
      for ( int component = 0; component < 2; ++component ) {
        ( *side_points[side] )[component] =
            points[side][0][component] + parameter * ( points[side][1][component] - points[side][0][component] );
      }
    }
    if ( location < 2 ) {
      for ( int component = 0; component < 2; ++component ) {
        result.integration_vertices[location][component] =
            0.5 * ( result.mortar_vertices[location][component] + result.nonmortar_vertices[location][component] );
      }
    }
  }
  result.valid = true;
  return result;
}

template <NormalPolicy Normal, typename Scalar>
[[nodiscard]] inline InteractionGeometryT<Scalar> projectedOverlap(
    const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
    const typename geometry::ProjectedOverlap<Normal>::Parameters& parameters )
{
  const auto patch = projectedOverlapPatch<Normal>( surfaces, pair, parameters );
  InteractionGeometryT<Scalar> result;
  result.normal = patch.normal;
  result.mortar_point = patch.mortar_centroid;
  result.nonmortar_point = patch.nonmortar_centroid;
  result.measure = patch.measure;
  result.valid = patch.valid;
  result.gap = projected_overlap_detail::dot(
      projected_overlap_detail::subtract( result.mortar_point, result.nonmortar_point ), result.normal );
  return result;
}

}  // namespace tribol

#endif
