#ifndef TRIBOL_INTEGRATION_INTERACTIONQUADRATURE_HPP_
#define TRIBOL_INTEGRATION_INTERACTIONQUADRATURE_HPP_

#include "tribol/geom/ProjectedOverlap.hpp"
#include "tribol/integration/Quadrature.hpp"

#include <array>
#include <type_traits>

namespace tribol::integration {

template <typename Scalar>
struct InteractionQuadraturePointT {
  std::array<Scalar, 3> mortar_position{};
  std::array<Scalar, 3> nonmortar_position{};
  std::array<Scalar, 3> normal{};
  Scalar weight{};
};

using InteractionQuadraturePoint = InteractionQuadraturePointT<Real>;

template <typename Scalar>
using InteractionQuadratureRuleT = QuadratureRule<InteractionQuadraturePointT<Scalar>>;

using InteractionQuadratureRule = InteractionQuadratureRuleT<Real>;

namespace detail {

template <typename Scalar>
TRIBOL_HOST_DEVICE inline void interpolatePair( const InteractionPatchT<Scalar>& patch, int first, int second,
                                                int third, basis::ReferencePoint reference,
                                                InteractionQuadraturePointT<Scalar>& point )
{
  const Scalar shape[3] = { 1.0 - reference.first - reference.second, reference.first, reference.second };
  const int vertices[3] = { first, second, third };
  point.normal = patch.normal;
  for ( int node = 0; node < 3; ++node ) {
    for ( int component = 0; component < 3; ++component ) {
      point.mortar_position[component] += shape[node] * patch.mortar_vertices[vertices[node]][component];
      point.nonmortar_position[component] += shape[node] * patch.nonmortar_vertices[vertices[node]][component];
    }
  }
}

}  // namespace detail

template <int Order, typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline InteractionQuadratureRuleT<Scalar> interactionQuadrature(
    const InteractionPatchT<Scalar>& patch, Polygon<Order> )
{
  static_assert( Order == 1 || Order == 2 || Order == 4,
                 "Built-in interaction quadrature supports orders one, two, and four." );
  InteractionQuadratureRuleT<Scalar> rule;
  if ( !patch.valid ) {
    return rule;
  }
  if ( patch.manifold_dimension == 1 && patch.vertex_count == 2 ) {
    constexpr Real location = 0.57735026918962576451;
    constexpr Real references[2] = { -location, location };
    const int point_count = Order == 1 ? 1 : 2;
    rule.size = point_count;
    for ( int point_index = 0; point_index < point_count; ++point_index ) {
      const Real reference = Order == 1 ? 0.0 : references[point_index];
      const Scalar first_shape = 0.5 * ( 1.0 - reference );
      const Scalar second_shape = 0.5 * ( 1.0 + reference );
      auto& point = rule[point_index];
      point.normal = patch.normal;
      point.weight = patch.measure / point_count;
      for ( int component = 0; component < 3; ++component ) {
        point.mortar_position[component] =
            first_shape * patch.mortar_vertices[0][component] + second_shape * patch.mortar_vertices[1][component];
        point.nonmortar_position[component] = first_shape * patch.nonmortar_vertices[0][component] +
                                              second_shape * patch.nonmortar_vertices[1][component];
      }
    }
    return rule;
  }
  constexpr std::array<basis::ReferencePoint, 3> degree_two_points{ basis::ReferencePoint{ 1.0 / 6.0, 1.0 / 6.0 },
                                                                    basis::ReferencePoint{ 2.0 / 3.0, 1.0 / 6.0 },
                                                                    basis::ReferencePoint{ 1.0 / 6.0, 2.0 / 3.0 } };
  constexpr auto triangle_rule = detail::degreeFourTriangleRule();
  static_assert( ( maximumInteractionVertices - 2 ) * triangle_rule.size <= maximumQuadraturePoints );
  for ( int triangle = 1; triangle + 1 < patch.vertex_count; ++triangle ) {
    const auto first_edge = detail::subtract( patch.integration_vertices[triangle], patch.integration_vertices[0] );
    const auto second_edge =
        detail::subtract( patch.integration_vertices[triangle + 1], patch.integration_vertices[0] );
    const Scalar area = 0.5 * detail::norm( detail::cross( first_edge, second_edge ), 3 );
    const int points_per_triangle = Order == 1 ? 1 : Order == 2 ? 3 : triangle_rule.size;
    for ( int point_index = 0; point_index < points_per_triangle; ++point_index ) {
      auto& point = rule[rule.size++];
      const auto reference = Order == 1   ? basis::ReferencePoint{ 1.0 / 3.0, 1.0 / 3.0 }
                             : Order == 2 ? degree_two_points[point_index]
                                          : triangle_rule[point_index].position;
      detail::interpolatePair( patch, 0, triangle, triangle + 1, reference, point );
      point.weight = Order == 1 ? area : Order == 2 ? area / 3.0 : area * triangle_rule[point_index].weight;
    }
  }
  return rule;
}

template <int Points, typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline InteractionQuadratureRuleT<Scalar> interactionQuadrature(
    const InteractionPatchT<Scalar>& patch, SmoothedSegment<Points> )
{
  InteractionQuadratureRuleT<Scalar> rule;
  if ( !patch.valid || patch.manifold_dimension != 1 || patch.vertex_count != 2 ) {
    return rule;
  }
  constexpr Real two_point_location = 0.57735026918962576451;
  constexpr Real three_point_location = 0.77459666924148337704;
  constexpr Real references[3] = { -three_point_location, 0.0, three_point_location };
  constexpr Real weights[3] = { 5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0 };
  rule.size = Points;
  for ( int point_index = 0; point_index < Points; ++point_index ) {
    const Real reference = Points == 1   ? 0.0
                           : Points == 2 ? ( point_index == 0 ? -two_point_location : two_point_location )
                                         : references[point_index];
    const Real reference_weight = Points == 1 ? 2.0 : Points == 2 ? 1.0 : weights[point_index];
    const Scalar first_shape = 0.5 * ( 1.0 - reference );
    const Scalar second_shape = 0.5 * ( 1.0 + reference );
    auto& point = rule[point_index];
    point.normal = patch.normal;
    point.weight = 0.5 * patch.measure * reference_weight;
    for ( int component = 0; component < 3; ++component ) {
      point.mortar_position[component] =
          first_shape * patch.mortar_vertices[0][component] + second_shape * patch.mortar_vertices[1][component];
      point.nonmortar_position[component] =
          first_shape * patch.nonmortar_vertices[0][component] + second_shape * patch.nonmortar_vertices[1][component];
    }
  }
  return rule;
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline InteractionQuadratureRuleT<Scalar> interactionQuadrature(
    const InteractionPatchT<Scalar>& patch, Centroid )
{
  InteractionQuadratureRuleT<Scalar> rule;
  if ( patch.valid ) {
    rule.size = 1;
    rule[0] = { patch.mortar_centroid, patch.nonmortar_centroid, patch.normal, patch.measure };
  }
  return rule;
}

template <int Order, typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline InteractionQuadratureRuleT<Scalar> interactionQuadrature(
    const InteractionPatchT<Scalar>& patch, Face<Order> )
{
  return interactionQuadrature( patch, Polygon<Order>{} );
}

static_assert( std::is_trivially_copyable_v<InteractionQuadratureRule> );

}  // namespace tribol::integration

#endif
