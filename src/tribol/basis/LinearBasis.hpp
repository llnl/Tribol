#ifndef TRIBOL_BASIS_LINEARBASIS_HPP_
#define TRIBOL_BASIS_LINEARBASIS_HPP_

#include "tribol/core/ExactTangent.hpp"
#include "tribol/core/MeshView.hpp"
#include "tribol/method/Method.hpp"

#include <array>
#include <type_traits>

namespace tribol::basis {

inline constexpr int maximumLinearNodes = 4;

template <typename Scalar>
struct ReferencePointT {
  Scalar first{};
  Scalar second{};
};

using ReferencePoint = ReferencePointT<Real>;

template <typename Scalar>
struct ShapeValuesT {
  std::array<Scalar, maximumLinearNodes> values{};
  int size{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Scalar& operator[]( int node ) { return values[node]; }
  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Scalar operator[]( int node ) const { return values[node]; }
};

using ShapeValues = ShapeValuesT<Real>;

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr int numberOfLinearNodes( ElementTopology topology )
{
  switch ( topology ) {
    case ElementTopology::Segment:
      return 2;
    case ElementTopology::Triangle:
      return 3;
    case ElementTopology::Quadrilateral:
      return 4;
  }
  return 0;
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ShapeValuesT<Scalar> primalShape( ElementTopology topology,
                                                                             ReferencePointT<Scalar> point )
{
  ShapeValuesT<Scalar> shape;
  shape.size = numberOfLinearNodes( topology );
  switch ( topology ) {
    case ElementTopology::Segment:
      shape[0] = 0.5 * ( 1.0 - point.first );
      shape[1] = 0.5 * ( 1.0 + point.first );
      break;
    case ElementTopology::Triangle:
      shape[0] = 1.0 - point.first - point.second;
      shape[1] = point.first;
      shape[2] = point.second;
      break;
    case ElementTopology::Quadrilateral:
      shape[0] = 0.25 * ( 1.0 - point.first ) * ( 1.0 - point.second );
      shape[1] = 0.25 * ( 1.0 + point.first ) * ( 1.0 - point.second );
      shape[2] = 0.25 * ( 1.0 + point.first ) * ( 1.0 + point.second );
      shape[3] = 0.25 * ( 1.0 - point.first ) * ( 1.0 + point.second );
      break;
  }
  return shape;
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ShapeValuesT<Scalar> dualShape( ElementTopology topology,
                                                                           ReferencePointT<Scalar> point )
{
  ShapeValuesT<Scalar> shape;
  if ( topology == ElementTopology::Quadrilateral ) {
    const Scalar first[2] = { 0.5 * ( 1.0 - 3.0 * point.first ), 0.5 * ( 1.0 + 3.0 * point.first ) };
    const Scalar second[2] = { 0.5 * ( 1.0 - 3.0 * point.second ), 0.5 * ( 1.0 + 3.0 * point.second ) };
    shape.size = 4;
    shape[0] = first[0] * second[0];
    shape[1] = first[1] * second[0];
    shape[2] = first[1] * second[1];
    shape[3] = first[0] * second[1];
    return shape;
  }
  const auto primal = primalShape( topology, point );
  shape.size = primal.size;
  const Real scale = static_cast<Real>( primal.size + 1 );
  for ( int node = 0; node < primal.size; ++node ) {
    shape[node] = scale * primal[node] - 1.0;
  }
  return shape;
}

template <BasisPolicy Basis, typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ShapeValuesT<Scalar> shape( ElementTopology topology,
                                                                       ReferencePointT<Scalar> point )
{
  if constexpr ( std::same_as<Basis, Primal> ) {
    return primalShape( topology, point );
  } else {
    return dualShape( topology, point );
  }
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline std::array<Scalar, 3> mapToPhysical( const SurfaceMeshViewT<Scalar>& mesh,
                                                                             Index element,
                                                                             ReferencePointT<Scalar> point )
{
  const auto values = primalShape( mesh.topologies[element], point );
  std::array<Scalar, 3> result{};
  const Index begin = mesh.element_offsets[element];
  for ( int local_node = 0; local_node < values.size; ++local_node ) {
    const Index node = mesh.connectivity[begin + local_node];
    for ( int component = 0; component < mesh.dimension; ++component ) {
      result[component] += values[local_node] * mesh.coordinates( node, component );
    }
  }
  return result;
}

namespace detail {

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline Scalar dot( const std::array<Scalar, 3>& left,
                                                    const std::array<Scalar, 3>& right, int dimension )
{
  Scalar value{};
  for ( int component = 0; component < dimension; ++component ) {
    value += left[component] * right[component];
  }
  return value;
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline std::array<Scalar, 3> nodePoint( const SurfaceMeshViewT<Scalar>& mesh,
                                                                         Index element, int local_node )
{
  const Index node = mesh.connectivity[mesh.element_offsets[element] + local_node];
  std::array<Scalar, 3> result{};
  for ( int component = 0; component < mesh.dimension; ++component ) {
    result[component] = mesh.coordinates( node, component );
  }
  return result;
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline std::array<Scalar, 3> subtract( const std::array<Scalar, 3>& left,
                                                                        const std::array<Scalar, 3>& right )
{
  return { left[0] - right[0], left[1] - right[1], left[2] - right[2] };
}

}  // namespace detail

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline ReferencePointT<Scalar> mapToReference( const SurfaceMeshViewT<Scalar>& mesh,
                                                                                Index element,
                                                                                const std::array<Scalar, 3>& point )
{
  using linearization_detail::absolute;
  using linearization_detail::primal;
  const auto topology = mesh.topologies[element];
  const auto first = detail::nodePoint( mesh, element, 0 );
  const auto first_edge = detail::subtract( detail::nodePoint( mesh, element, 1 ), first );
  const auto relative = detail::subtract( point, first );
  if ( topology == ElementTopology::Segment ) {
    const Scalar denominator = detail::dot( first_edge, first_edge, mesh.dimension );
    const Scalar coordinate =
        primal( denominator ) > 0.0 ? detail::dot( relative, first_edge, mesh.dimension ) / denominator : Scalar{ 0.5 };
    return { 2.0 * coordinate - 1.0, {} };
  }
  if ( topology == ElementTopology::Triangle ) {
    const auto second_edge = detail::subtract( detail::nodePoint( mesh, element, 2 ), first );
    const Scalar first_first = detail::dot( first_edge, first_edge, mesh.dimension );
    const Scalar first_second = detail::dot( first_edge, second_edge, mesh.dimension );
    const Scalar second_second = detail::dot( second_edge, second_edge, mesh.dimension );
    const Scalar first_relative = detail::dot( first_edge, relative, mesh.dimension );
    const Scalar second_relative = detail::dot( second_edge, relative, mesh.dimension );
    const Scalar determinant = first_first * second_second - first_second * first_second;
    if ( absolute( determinant ) <= 1.0e-28 ) {
      return {};
    }
    return { ( second_second * first_relative - first_second * second_relative ) / determinant,
             ( first_first * second_relative - first_second * first_relative ) / determinant };
  }

  ReferencePointT<Scalar> reference{};
  for ( int iteration = 0; iteration < 12; ++iteration ) {
    const Scalar first_coordinate = reference.first;
    const Scalar second_coordinate = reference.second;
    const Scalar derivative_first[4] = { -0.25 * ( 1.0 - second_coordinate ), 0.25 * ( 1.0 - second_coordinate ),
                                         0.25 * ( 1.0 + second_coordinate ), -0.25 * ( 1.0 + second_coordinate ) };
    const Scalar derivative_second[4] = { -0.25 * ( 1.0 - first_coordinate ), -0.25 * ( 1.0 + first_coordinate ),
                                          0.25 * ( 1.0 + first_coordinate ), 0.25 * ( 1.0 - first_coordinate ) };
    const auto mapped = mapToPhysical( mesh, element, reference );
    std::array<Scalar, 3> tangent_first{};
    std::array<Scalar, 3> tangent_second{};
    for ( int node = 0; node < 4; ++node ) {
      const auto coordinate = detail::nodePoint( mesh, element, node );
      for ( int component = 0; component < mesh.dimension; ++component ) {
        tangent_first[component] += derivative_first[node] * coordinate[component];
        tangent_second[component] += derivative_second[node] * coordinate[component];
      }
    }
    const auto error = detail::subtract( point, mapped );
    const Scalar first_first = detail::dot( tangent_first, tangent_first, mesh.dimension );
    const Scalar first_second = detail::dot( tangent_first, tangent_second, mesh.dimension );
    const Scalar second_second = detail::dot( tangent_second, tangent_second, mesh.dimension );
    const Scalar determinant = first_first * second_second - first_second * first_second;
    if ( absolute( determinant ) <= 1.0e-28 ) {
      break;
    }
    const Scalar first_error = detail::dot( tangent_first, error, mesh.dimension );
    const Scalar second_error = detail::dot( tangent_second, error, mesh.dimension );
    const Scalar first_update = ( second_second * first_error - first_second * second_error ) / determinant;
    const Scalar second_update = ( first_first * second_error - first_second * first_error ) / determinant;
    reference.first += first_update;
    reference.second += second_update;
    if ( absolute( first_update ) + absolute( second_update ) <= 1.0e-13 ) {
      break;
    }
  }
  return reference;
}

static_assert( std::is_trivially_copyable_v<ReferencePoint> );
static_assert( std::is_trivially_copyable_v<ShapeValues> );

}  // namespace tribol::basis

#endif
