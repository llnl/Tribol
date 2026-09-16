#ifndef TRIBOL_INTEGRATION_QUADRATURE_HPP_
#define TRIBOL_INTEGRATION_QUADRATURE_HPP_

#include "tribol/basis/LinearBasis.hpp"

#include <array>
#include <cmath>
#include <type_traits>

namespace tribol::integration {

inline constexpr int maximumQuadraturePoints = 24;

struct ReferenceQuadraturePoint {
  basis::ReferencePoint position{};
  Real weight{};
};

struct PhysicalQuadraturePoint {
  std::array<Real, 3> position{};
  basis::ReferencePoint reference{};
  Real weight{};
};

template <typename Point>
struct QuadratureRule {
  std::array<Point, maximumQuadraturePoints> points{};
  int size{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Point& operator[]( int point ) { return points[point]; }
  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr const Point& operator[]( int point ) const { return points[point]; }
};

namespace detail {

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline std::array<Scalar, 3> subtract( const std::array<Scalar, 3>& left,
                                                                        const std::array<Scalar, 3>& right )
{
  return { left[0] - right[0], left[1] - right[1], left[2] - right[2] };
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline std::array<Scalar, 3> cross( const std::array<Scalar, 3>& left,
                                                                     const std::array<Scalar, 3>& right )
{
  return { left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2],
           left[0] * right[1] - left[1] * right[0] };
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline Scalar norm( const std::array<Scalar, 3>& value, int dimension )
{
  Scalar squared{};
  for ( int component = 0; component < dimension; ++component ) {
    squared += value[component] * value[component];
  }
  return linearization_detail::squareRoot( squared );
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline Real jacobian( const SurfaceMeshView& mesh, Index element,
                                                       basis::ReferencePoint point )
{
  const auto topology = mesh.topologies[element];
  const Index begin = mesh.element_offsets[element];
  if ( topology == ElementTopology::Segment ) {
    std::array<Real, 3> derivative{};
    for ( int component = 0; component < mesh.dimension; ++component ) {
      derivative[component] = 0.5 * ( mesh.coordinates( mesh.connectivity[begin + 1], component ) -
                                      mesh.coordinates( mesh.connectivity[begin], component ) );
    }
    return norm( derivative, mesh.dimension );
  }
  std::array<Real, 3> derivative_first{};
  std::array<Real, 3> derivative_second{};
  if ( topology == ElementTopology::Triangle ) {
    for ( int component = 0; component < mesh.dimension; ++component ) {
      derivative_first[component] = mesh.coordinates( mesh.connectivity[begin + 1], component ) -
                                    mesh.coordinates( mesh.connectivity[begin], component );
      derivative_second[component] = mesh.coordinates( mesh.connectivity[begin + 2], component ) -
                                     mesh.coordinates( mesh.connectivity[begin], component );
    }
  } else {
    const Real derivative_first_shape[4] = { -0.25 * ( 1.0 - point.second ), 0.25 * ( 1.0 - point.second ),
                                             0.25 * ( 1.0 + point.second ), -0.25 * ( 1.0 + point.second ) };
    const Real derivative_second_shape[4] = { -0.25 * ( 1.0 - point.first ), -0.25 * ( 1.0 + point.first ),
                                              0.25 * ( 1.0 + point.first ), 0.25 * ( 1.0 - point.first ) };
    for ( int node = 0; node < 4; ++node ) {
      for ( int component = 0; component < mesh.dimension; ++component ) {
        const Real coordinate = mesh.coordinates( mesh.connectivity[begin + node], component );
        derivative_first[component] += derivative_first_shape[node] * coordinate;
        derivative_second[component] += derivative_second_shape[node] * coordinate;
      }
    }
  }
  return norm( cross( derivative_first, derivative_second ), 3 );
}

template <int Order>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr QuadratureRule<ReferenceQuadraturePoint> referenceRule(
    ElementTopology topology )
{
  static_assert( Order == 1 || Order == 2, "Built-in linear quadrature supports orders one and two." );
  QuadratureRule<ReferenceQuadraturePoint> rule;
  if constexpr ( Order == 1 ) {
    rule.size = 1;
    if ( topology == ElementTopology::Triangle ) {
      rule[0] = { { 1.0 / 3.0, 1.0 / 3.0 }, 0.5 };
    } else {
      rule[0] = { { 0.0, 0.0 }, topology == ElementTopology::Segment ? 2.0 : 4.0 };
    }
    return rule;
  }
  if ( topology == ElementTopology::Segment ) {
    constexpr Real location = 0.57735026918962576451;
    rule.size = 2;
    rule[0] = { { -location, 0.0 }, 1.0 };
    rule[1] = { { location, 0.0 }, 1.0 };
  } else if ( topology == ElementTopology::Triangle ) {
    rule.size = 3;
    rule[0] = { { 1.0 / 6.0, 1.0 / 6.0 }, 1.0 / 6.0 };
    rule[1] = { { 2.0 / 3.0, 1.0 / 6.0 }, 1.0 / 6.0 };
    rule[2] = { { 1.0 / 6.0, 2.0 / 3.0 }, 1.0 / 6.0 };
  } else {
    constexpr Real location = 0.57735026918962576451;
    rule.size = 4;
    rule[0] = { { -location, -location }, 1.0 };
    rule[1] = { { location, -location }, 1.0 };
    rule[2] = { { location, location }, 1.0 };
    rule[3] = { { -location, location }, 1.0 };
  }
  return rule;
}

}  // namespace detail

template <int Order>
[[nodiscard]] TRIBOL_HOST_DEVICE inline QuadratureRule<PhysicalQuadraturePoint> faceQuadrature(
    const SurfaceMeshView& mesh, Index element, Face<Order> )
{
  const auto reference = detail::referenceRule<Order>( mesh.topologies[element] );
  QuadratureRule<PhysicalQuadraturePoint> rule;
  rule.size = reference.size;
  for ( int point = 0; point < reference.size; ++point ) {
    rule[point] = { basis::mapToPhysical( mesh, element, reference[point].position ), reference[point].position,
                    reference[point].weight * detail::jacobian( mesh, element, reference[point].position ) };
  }
  return rule;
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline QuadratureRule<PhysicalQuadraturePoint> faceQuadrature(
    const SurfaceMeshView& mesh, Index element, Centroid )
{
  const auto reference = detail::referenceRule<1>( mesh.topologies[element] );
  QuadratureRule<PhysicalQuadraturePoint> rule;
  rule.size = 1;
  rule[0] = { basis::mapToPhysical( mesh, element, reference[0].position ), reference[0].position,
              reference[0].weight * detail::jacobian( mesh, element, reference[0].position ) };
  return rule;
}

template <int Order>
[[nodiscard]] TRIBOL_HOST_DEVICE inline QuadratureRule<PhysicalQuadraturePoint> polygonQuadrature(
    ArrayView<const std::array<Real, 3>> vertices, int manifold_dimension, Polygon<Order> )
{
  static_assert( Order == 1 || Order == 2, "Built-in polygon quadrature supports orders one and two." );
  QuadratureRule<PhysicalQuadraturePoint> rule;
  if ( manifold_dimension == 1 ) {
    if ( vertices.size() != 2 ) {
      return rule;
    }
    const auto difference = detail::subtract( vertices[1], vertices[0] );
    const Real length = detail::norm( difference, 3 );
    const auto reference = detail::referenceRule<Order>( ElementTopology::Segment );
    rule.size = reference.size;
    for ( int point = 0; point < reference.size; ++point ) {
      const Real fraction = 0.5 * ( 1.0 + reference[point].position.first );
      for ( int component = 0; component < 3; ++component ) {
        rule[point].position[component] = vertices[0][component] + fraction * difference[component];
      }
      rule[point].reference = reference[point].position;
      rule[point].weight = 0.5 * length * reference[point].weight;
    }
    return rule;
  }
  if ( manifold_dimension != 2 || vertices.size() < 3 ) {
    return rule;
  }
  constexpr std::array<basis::ReferencePoint, 3> triangle_points{ basis::ReferencePoint{ 1.0 / 3.0, 1.0 / 3.0 },
                                                                  basis::ReferencePoint{ 2.0 / 3.0, 1.0 / 6.0 },
                                                                  basis::ReferencePoint{ 1.0 / 6.0, 2.0 / 3.0 } };
  for ( Index triangle = 1; triangle + 1 < vertices.size(); ++triangle ) {
    const auto first_edge = detail::subtract( vertices[triangle], vertices[0] );
    const auto second_edge = detail::subtract( vertices[triangle + 1], vertices[0] );
    const Real area = 0.5 * detail::norm( detail::cross( first_edge, second_edge ), 3 );
    const int points_per_triangle = Order == 1 ? 1 : 3;
    for ( int local_point = 0; local_point < points_per_triangle; ++local_point ) {
      const auto reference = Order == 1 ? triangle_points[0] : triangle_points[local_point];
      auto& point = rule[rule.size++];
      for ( int component = 0; component < 3; ++component ) {
        point.position[component] = vertices[0][component] + reference.first * first_edge[component] +
                                    reference.second * second_edge[component];
      }
      point.reference = reference;
      point.weight = area / points_per_triangle;
    }
  }
  return rule;
}

static_assert( std::is_trivially_copyable_v<QuadratureRule<PhysicalQuadraturePoint>> );

}  // namespace tribol::integration

#endif
