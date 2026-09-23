#ifndef TRIBOL_BASIS_PARENTBASIS_HPP_
#define TRIBOL_BASIS_PARENTBASIS_HPP_

#include "tribol/basis/LinearBasis.hpp"
#include "tribol/core/ParentTraceView.hpp"

#include <array>

namespace tribol::basis {

inline constexpr int maximumParentDofs = ( maximumParentTraceOrder + 1 ) * ( maximumParentTraceOrder + 1 );

template <typename Scalar>
struct ParentShapeValuesT {
  std::array<Scalar, maximumParentDofs> values{};
  int size{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Scalar& operator[]( int node ) { return values[node]; }
  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Scalar operator[]( int node ) const { return values[node]; }
};

using ParentShapeValues = ParentShapeValuesT<Real>;

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr int parentDofCount( ElementTopology topology, int order )
{
  return static_cast<int>( parentFaceDofCount( topology, order ) );
}

namespace parent_basis_detail {

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr Scalar lagrangeValue( int node, int order, Scalar coordinate )
{
  Scalar value{ 1.0 };
  for ( int other = 0; other <= order; ++other ) {
    if ( other != node ) {
      value *= ( static_cast<Scalar>( order ) * coordinate - static_cast<Scalar>( other ) ) /
               static_cast<Scalar>( node - other );
    }
  }
  return value;
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr Scalar simplexFactor( int degree, int order, Scalar barycentric )
{
  Scalar value{ 1.0 };
  for ( int factor = 0; factor < degree; ++factor ) {
    value *= ( static_cast<Scalar>( order ) * barycentric - static_cast<Scalar>( factor ) ) /
             static_cast<Scalar>( degree - factor );
  }
  return value;
}

}  // namespace parent_basis_detail

template <typename Scalar = Real>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ReferencePointT<Scalar> parentReferenceNode( ElementTopology topology,
                                                                                        int order, int node )
{
  if ( order < 1 || order > maximumParentTraceOrder || node < 0 || node >= parentDofCount( topology, order ) ) {
    return {};
  }
  if ( topology == ElementTopology::Segment ) {
    return { static_cast<Scalar>( node ) / static_cast<Scalar>( order ), Scalar{} };
  }
  if ( topology == ElementTopology::Quadrilateral ) {
    const int width = order + 1;
    return { static_cast<Scalar>( node % width ) / static_cast<Scalar>( order ),
             static_cast<Scalar>( node / width ) / static_cast<Scalar>( order ) };
  }
  int current{};
  for ( int second = 0; second <= order; ++second ) {
    for ( int first = 0; first <= order - second; ++first ) {
      if ( current++ == node ) {
        return { static_cast<Scalar>( first ) / static_cast<Scalar>( order ),
                 static_cast<Scalar>( second ) / static_cast<Scalar>( order ) };
      }
    }
  }
  return {};
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ParentShapeValuesT<Scalar> parentShape( ElementTopology topology, int order,
                                                                                   ReferencePointT<Scalar> point )
{
  ParentShapeValuesT<Scalar> shape;
  shape.size = parentDofCount( topology, order );
  if ( shape.size == 0 ) {
    return shape;
  }
  if ( topology == ElementTopology::Segment ) {
    for ( int node = 0; node <= order; ++node ) {
      shape[node] = parent_basis_detail::lagrangeValue( node, order, point.first );
    }
    return shape;
  }
  if ( topology == ElementTopology::Quadrilateral ) {
    int node{};
    for ( int second = 0; second <= order; ++second ) {
      const Scalar second_value = parent_basis_detail::lagrangeValue( second, order, point.second );
      for ( int first = 0; first <= order; ++first ) {
        shape[node++] = parent_basis_detail::lagrangeValue( first, order, point.first ) * second_value;
      }
    }
    return shape;
  }
  const Scalar barycentric[3] = { 1.0 - point.first - point.second, point.first, point.second };
  int node{};
  for ( int second = 0; second <= order; ++second ) {
    for ( int first = 0; first <= order - second; ++first ) {
      const int zeroth = order - first - second;
      shape[node++] = parent_basis_detail::simplexFactor( zeroth, order, barycentric[0] ) *
                      parent_basis_detail::simplexFactor( first, order, barycentric[1] ) *
                      parent_basis_detail::simplexFactor( second, order, barycentric[2] );
    }
  }
  return shape;
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE constexpr ParentShapeValuesT<Scalar> parentMonomials( ElementTopology topology,
                                                                                       int order,
                                                                                       ReferencePointT<Scalar> point )
{
  ParentShapeValuesT<Scalar> values;
  values.size = parentDofCount( topology, order );
  if ( values.size == 0 ) {
    return values;
  }
  if ( topology == ElementTopology::Segment ) {
    Scalar power{ 1.0 };
    for ( int exponent = 0; exponent <= order; ++exponent ) {
      values[exponent] = power;
      power *= point.first;
    }
    return values;
  }
  int term{};
  Scalar second_power{ 1.0 };
  for ( int second = 0; second <= order; ++second ) {
    Scalar first_power{ 1.0 };
    const int first_limit = topology == ElementTopology::Triangle ? order - second : order;
    for ( int first = 0; first <= first_limit; ++first ) {
      values[term++] = first_power * second_power;
      first_power *= point.first;
    }
    second_power *= point.second;
  }
  return values;
}

template <typename TraceScalar, typename ReferenceScalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline auto parentShape( const ParentTraceViewT<TraceScalar>& trace, Index parent_face,
                                                          ReferencePointT<ReferenceScalar> reference )
{
  using ResultScalar = decltype( trace.parent_basis_coefficients[Index{}] * reference.first );
  ParentShapeValuesT<ResultScalar> shape;
  const ElementTopology topology = trace.parent_face_topologies[parent_face];
  const int order = trace.parent_face_orders[parent_face];
  const auto monomials = parentMonomials( topology, order, reference );
  shape.size = monomials.size;
  const Index coefficient_begin = trace.parent_basis_offsets[parent_face];
  for ( int dof = 0; dof < shape.size; ++dof ) {
    for ( int term = 0; term < monomials.size; ++term ) {
      shape[dof] += trace.parent_basis_coefficients[coefficient_begin + dof * monomials.size + term] * monomials[term];
    }
  }
  return shape;
}

template <typename TraceScalar, typename FieldScalar, typename ReferenceScalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline auto interpolateParentField( const ParentTraceViewT<TraceScalar>& trace,
                                                                     Index parent_face,
                                                                     const FieldView<FieldScalar>& field,
                                                                     ReferencePointT<ReferenceScalar> reference )
{
  using ResultScalar = decltype( field( Index{}, 0 ) * reference.first );
  std::array<ResultScalar, 3> result{};
  const auto shape = parentShape( trace, parent_face, reference );
  const Index begin = trace.parent_dof_offsets[parent_face];
  for ( int local_dof = 0; local_dof < shape.size; ++local_dof ) {
    const Index dof = trace.parent_dofs[begin + local_dof];
    for ( int component = 0; component < trace.geometry.dimension; ++component ) {
      result[component] += shape[local_dof] * trace.parent_dof_signs[begin + local_dof] * field( dof, component );
    }
  }
  return result;
}

static_assert( std::is_trivially_copyable_v<ParentShapeValues> );

}  // namespace tribol::basis

#endif
