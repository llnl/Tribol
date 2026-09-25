#ifndef TRIBOL_INTEGRATION_AUTOMATICQUADRATURE_HPP_
#define TRIBOL_INTEGRATION_AUTOMATICQUADRATURE_HPP_

#include "tribol/basis/LinearBasis.hpp"
#include "tribol/core/ExactTangent.hpp"
#include "tribol/core/OverlapCellView.hpp"
#include "tribol/core/ParentTraceView.hpp"
#include "tribol/method/Method.hpp"

#include <array>

namespace tribol::integration {

inline constexpr int maximumAutomaticAxisPoints = 6;
inline constexpr int maximumAutomaticExactness = 2 * maximumAutomaticAxisPoints - 1;

struct OverlapQuadraturePoint {
  basis::ReferencePoint mortar_reference{};
  basis::ReferencePoint nonmortar_reference{};
  std::array<Real, 3> normal{};
  Real weight{};
};

namespace automatic_quadrature_detail {

struct AxisRule {
  std::array<Real, maximumAutomaticAxisPoints> points{};
  std::array<Real, maximumAutomaticAxisPoints> weights{};
  int size{};
};

[[nodiscard]] TRIBOL_HOST_DEVICE inline AxisRule gaussLegendreRule( int count )
{
  AxisRule rule;
  rule.size = count;
  if ( count == 1 ) {
    rule.weights[0] = 2.0;
  } else if ( count == 2 ) {
    constexpr Real point = 0.57735026918962576451;
    rule.points = { -point, point };
    rule.weights = { 1.0, 1.0 };
  } else if ( count == 3 ) {
    constexpr Real point = 0.77459666924148337704;
    rule.points = { -point, 0.0, point };
    rule.weights = { 5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0 };
  } else if ( count == 4 ) {
    rule.points = { -0.86113631159405257522, -0.33998104358485626480, 0.33998104358485626480, 0.86113631159405257522 };
    rule.weights = { 0.34785484513745385737, 0.65214515486254614263, 0.65214515486254614263, 0.34785484513745385737 };
  } else if ( count == 5 ) {
    rule.points = { -0.90617984593866399280, -0.53846931010568309104, 0.0, 0.53846931010568309104,
                    0.90617984593866399280 };
    rule.weights = { 0.23692688505618908751, 0.47862867049936646804, 0.56888888888888888889, 0.47862867049936646804,
                     0.23692688505618908751 };
  } else {
    rule.size = 6;
    rule.points = { -0.93246951420315202781, -0.66120938646626451366, -0.23861918608319690863,
                    0.23861918608319690863,  0.66120938646626451366,  0.93246951420315202781 };
    rule.weights = { 0.17132449237917034504, 0.36076157304813860757, 0.46791393457269104739,
                     0.46791393457269104739, 0.36076157304813860757, 0.17132449237917034504 };
  }
  return rule;
}

template <typename Cell>
[[nodiscard]] TRIBOL_HOST_DEVICE inline Real triangleArea( const Cell& cell, Index first, Index second, Index third )
{
  std::array<Real, 3> first_edge{};
  std::array<Real, 3> second_edge{};
  for ( int component = 0; component < 3; ++component ) {
    first_edge[component] = cell.integrationVertex( second, component ) - cell.integrationVertex( first, component );
    second_edge[component] = cell.integrationVertex( third, component ) - cell.integrationVertex( first, component );
  }
  const std::array<Real, 3> cross{ first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
                                   first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
                                   first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0] };
  return 0.5 * linearization_detail::squareRoot( cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2] );
}

template <typename Cell>
[[nodiscard]] TRIBOL_HOST_DEVICE inline basis::ReferencePoint interpolateReference( const Cell& cell, bool mortar,
                                                                                    const int* vertices,
                                                                                    const Real* shape, int count )
{
  basis::ReferencePoint result{};
  for ( int node = 0; node < count; ++node ) {
    result.first += shape[node] * cell.parentReference( mortar, vertices[node], 0 );
    if ( cell.dimension() == 3 ) {
      result.second += shape[node] * cell.parentReference( mortar, vertices[node], 1 );
    }
  }
  return result;
}

struct BatchCellAccessor {
  OverlapCellBatchView cells{};
  Index cell{};

  [[nodiscard]] TRIBOL_HOST_DEVICE int dimension() const { return cells.dimension; }
  [[nodiscard]] TRIBOL_HOST_DEVICE Index vertexCount() const
  {
    return cells.vertex_offsets[cell + 1] - cells.vertex_offsets[cell];
  }
  [[nodiscard]] TRIBOL_HOST_DEVICE Real normal( int component ) const { return cells.normals( cell, component ); }
  [[nodiscard]] TRIBOL_HOST_DEVICE Real measure() const { return cells.measures[cell]; }
  [[nodiscard]] TRIBOL_HOST_DEVICE Real integrationVertex( Index vertex, int component ) const
  {
    return cells.integration_vertices( cells.vertex_offsets[cell] + vertex, component );
  }
  [[nodiscard]] TRIBOL_HOST_DEVICE Real parentReference( bool mortar, Index vertex, int component ) const
  {
    const Index global_vertex = cells.vertex_offsets[cell] + vertex;
    return mortar ? cells.mortar_parent_reference_coordinates( global_vertex, component )
                  : cells.nonmortar_parent_reference_coordinates( global_vertex, component );
  }
};

}  // namespace automatic_quadrature_detail

[[nodiscard]] TRIBOL_HOST_DEVICE inline int automaticQuadratureExactness( int mortar_order, int nonmortar_order,
                                                                          const Automatic::Parameters& parameters )
{
  const int field_order = mortar_order > nonmortar_order ? mortar_order : nonmortar_order;
  const int traction_order = parameters.traction_order < 0 ? field_order : parameters.traction_order;
  const int exactness = field_order + traction_order;
  return exactness > parameters.minimum_exactness ? exactness : parameters.minimum_exactness;
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline int automaticQuadratureExactness( const ParentTracePairView& traces,
                                                                          const OverlapCellBatchView& cells, Index cell,
                                                                          const Automatic::Parameters& parameters )
{
  return automaticQuadratureExactness( traces.mortar.parent_face_orders[cells.mortar_parent_faces[cell]],
                                       traces.nonmortar.parent_face_orders[cells.nonmortar_parent_faces[cell]],
                                       parameters );
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline int automaticAxisPointCount( int exactness )
{
  const int points = ( exactness + 2 ) / 2;
  return points < 1 ? 1 : points > maximumAutomaticAxisPoints ? maximumAutomaticAxisPoints : points;
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline Index automaticQuadraturePointCount( const OverlapCellBatchView& cells,
                                                                             Index cell, int exactness )
{
  if ( !cells.isAccepted( cell ) ) {
    return 0;
  }
  const int axis_points = automaticAxisPointCount( exactness );
  const Index vertices = cells.vertex_offsets[cell + 1] - cells.vertex_offsets[cell];
  return cells.dimension == 2 ? axis_points : ( vertices - 2 ) * axis_points * axis_points;
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline Index automaticQuadraturePointCount( const ParentTracePairView& traces,
                                                                             const OverlapCellBatchView& cells,
                                                                             Index cell,
                                                                             const Automatic::Parameters& parameters )
{
  return automaticQuadraturePointCount( cells, cell, automaticQuadratureExactness( traces, cells, cell, parameters ) );
}

template <typename Cell>
[[nodiscard]] TRIBOL_HOST_DEVICE inline OverlapQuadraturePoint automaticQuadraturePointFromCell( const Cell& cell,
                                                                                                 int exactness,
                                                                                                 Index local_point )
{
  OverlapQuadraturePoint result;
  const int axis_points = automaticAxisPointCount( exactness );
  const auto axis = automatic_quadrature_detail::gaussLegendreRule( axis_points );
  for ( int component = 0; component < 3; ++component ) {
    result.normal[component] = cell.normal( component );
  }
  if ( cell.dimension() == 2 ) {
    const Real coordinate = axis.points[local_point];
    const Real shape[2] = { 0.5 * ( 1.0 - coordinate ), 0.5 * ( 1.0 + coordinate ) };
    const int vertices[2] = { 0, 1 };
    result.mortar_reference = automatic_quadrature_detail::interpolateReference( cell, true, vertices, shape, 2 );
    result.nonmortar_reference = automatic_quadrature_detail::interpolateReference( cell, false, vertices, shape, 2 );
    result.weight = 0.5 * cell.measure() * axis.weights[local_point];
    return result;
  }
  const Index triangle = local_point / ( axis_points * axis_points );
  const int first_axis = static_cast<int>( local_point % ( axis_points * axis_points ) ) / axis_points;
  const int second_axis = static_cast<int>( local_point % axis_points );
  const Real first = 0.5 * ( 1.0 + axis.points[first_axis] );
  const Real second = 0.5 * ( 1.0 + axis.points[second_axis] );
  const Real shape[3] = { ( 1.0 - first ) * ( 1.0 - second ), first, ( 1.0 - first ) * second };
  const int vertices[3] = { 0, static_cast<int>( triangle + 1 ), static_cast<int>( triangle + 2 ) };
  result.mortar_reference = automatic_quadrature_detail::interpolateReference( cell, true, vertices, shape, 3 );
  result.nonmortar_reference = automatic_quadrature_detail::interpolateReference( cell, false, vertices, shape, 3 );
  const Real area = automatic_quadrature_detail::triangleArea( cell, vertices[0], vertices[1], vertices[2] );
  result.weight = 0.5 * area * axis.weights[first_axis] * axis.weights[second_axis] * ( 1.0 - first );
  return result;
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline OverlapQuadraturePoint automaticQuadraturePoint(
    const OverlapCellBatchView& cells, Index cell, Index local_point, int exactness )
{
  return automaticQuadraturePointFromCell( automatic_quadrature_detail::BatchCellAccessor{ cells, cell }, exactness,
                                           local_point );
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline OverlapQuadraturePoint automaticQuadraturePoint(
    const ParentTracePairView& traces, const OverlapCellBatchView& cells, Index cell, Index local_point,
    const Automatic::Parameters& parameters )
{
  return automaticQuadraturePoint( cells, cell, local_point,
                                   automaticQuadratureExactness( traces, cells, cell, parameters ) );
}

}  // namespace tribol::integration

#endif
