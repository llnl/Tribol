#ifndef TRIBOL_GEOM_CONFORMINGOVERLAP_HPP_
#define TRIBOL_GEOM_CONFORMINGOVERLAP_HPP_

#include "tribol/geom/ProjectedOverlap.hpp"

#include <array>
#include <algorithm>

namespace tribol {

namespace conforming_overlap_detail {

template <typename Scalar>
inline Scalar squaredDistance( const std::array<Scalar, 3>& first, const std::array<Scalar, 3>& second,
                               const std::array<Scalar, 3>& normal )
{
  const auto difference = projected_overlap_detail::subtract( first, second );
  const Scalar normal_component = projected_overlap_detail::dot( difference, normal );
  return projected_overlap_detail::dot( difference, difference ) - normal_component * normal_component;
}

template <typename Scalar>
inline Scalar triangleArea( const std::array<Scalar, 3>& first, const std::array<Scalar, 3>& second,
                            const std::array<Scalar, 3>& third )
{
  return 0.5 * projected_overlap_detail::norm(
                   projected_overlap_detail::cross( projected_overlap_detail::subtract( second, first ),
                                                    projected_overlap_detail::subtract( third, first ) ) );
}

}  // namespace conforming_overlap_detail

template <typename Scalar>
inline InteractionPatchT<Scalar> conformingOverlapPatch( const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
                                                         const geometry::ConformingOverlap::Parameters& parameters )
{
  InteractionPatchT<Scalar> result;
  if ( surfaces.mortar.dimension != 3 ||
       !projected_overlap_detail::isLinearFace( surfaces.mortar, pair.mortar_element ) ||
       !projected_overlap_detail::isLinearFace( surfaces.nonmortar, pair.nonmortar_element ) ||
       surfaces.mortar.topologies[pair.mortar_element] != surfaces.nonmortar.topologies[pair.nonmortar_element] ) {
    return result;
  }
  const Index mortar_begin = surfaces.mortar.element_offsets[pair.mortar_element];
  const Index nonmortar_begin = surfaces.nonmortar.element_offsets[pair.nonmortar_element];
  const int node_count = static_cast<int>( surfaces.mortar.element_offsets[pair.mortar_element + 1] - mortar_begin );
  if ( node_count != surfaces.nonmortar.element_offsets[pair.nonmortar_element + 1] - nonmortar_begin ) {
    return result;
  }
  const auto mortar_normal = projected_overlap_detail::faceNormal( surfaces.mortar, pair.mortar_element );
  result.normal = { -mortar_normal[0], -mortar_normal[1], -mortar_normal[2] };
  std::array<bool, maximumInteractionVertices> matched{};
  for ( int mortar_node = 0; mortar_node < node_count; ++mortar_node ) {
    const auto mortar_point = projected_overlap_detail::facePoint( surfaces.mortar, pair.mortar_element, mortar_node );
    int best_node = -1;
    Scalar best_distance{};
    for ( int nonmortar_node = 0; nonmortar_node < node_count; ++nonmortar_node ) {
      if ( matched[nonmortar_node] ) {
        continue;
      }
      const auto nonmortar_point =
          projected_overlap_detail::facePoint( surfaces.nonmortar, pair.nonmortar_element, nonmortar_node );
      const Scalar distance =
          conforming_overlap_detail::squaredDistance( mortar_point, nonmortar_point, result.normal );
      if ( best_node < 0 || linearization_detail::primal( distance ) < linearization_detail::primal( best_distance ) ) {
        best_node = nonmortar_node;
        best_distance = distance;
      }
    }
    if ( best_node < 0 || linearization_detail::squareRoot( std::max( linearization_detail::primal( best_distance ),
                                                                      0.0 ) ) > parameters.alignment_tolerance ) {
      return InteractionPatchT<Scalar>{};
    }
    matched[best_node] = true;
    result.mortar_vertices[mortar_node] = mortar_point;
    result.nonmortar_vertices[mortar_node] =
        projected_overlap_detail::facePoint( surfaces.nonmortar, pair.nonmortar_element, best_node );
    for ( int component = 0; component < 3; ++component ) {
      result.integration_vertices[mortar_node][component] =
          0.5 * ( result.mortar_vertices[mortar_node][component] + result.nonmortar_vertices[mortar_node][component] );
    }
  }
  Scalar centroid_weight{};
  for ( int triangle = 1; triangle + 1 < node_count; ++triangle ) {
    const Scalar area =
        conforming_overlap_detail::triangleArea( result.integration_vertices[0], result.integration_vertices[triangle],
                                                 result.integration_vertices[triangle + 1] );
    result.measure += area;
    centroid_weight += area;
    for ( int component = 0; component < 3; ++component ) {
      result.mortar_centroid[component] +=
          area *
          ( result.mortar_vertices[0][component] + result.mortar_vertices[triangle][component] +
            result.mortar_vertices[triangle + 1][component] ) /
          3.0;
      result.nonmortar_centroid[component] +=
          area *
          ( result.nonmortar_vertices[0][component] + result.nonmortar_vertices[triangle][component] +
            result.nonmortar_vertices[triangle + 1][component] ) /
          3.0;
    }
  }
  if ( linearization_detail::primal( centroid_weight ) <= 0.0 ) {
    return InteractionPatchT<Scalar>{};
  }
  for ( int component = 0; component < 3; ++component ) {
    result.mortar_centroid[component] /= centroid_weight;
    result.nonmortar_centroid[component] /= centroid_weight;
  }
  result.vertex_count = node_count;
  result.manifold_dimension = 2;
  result.valid = true;
  return result;
}

template <typename Scalar>
inline InteractionGeometryT<Scalar> conformingOverlap( const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
                                                       const geometry::ConformingOverlap::Parameters& parameters )
{
  const auto patch = conformingOverlapPatch( surfaces, pair, parameters );
  InteractionGeometryT<Scalar> result;
  result.normal = patch.normal;
  result.mortar_point = patch.mortar_centroid;
  result.nonmortar_point = patch.nonmortar_centroid;
  result.measure = patch.measure;
  result.gap = projected_overlap_detail::dot(
      projected_overlap_detail::subtract( result.mortar_point, result.nonmortar_point ), result.normal );
  result.valid = patch.valid;
  return result;
}

}  // namespace tribol

#endif
