// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Projection of generated source quadrature onto target edges without changing source weights.
 */

#ifndef SRC_TRIBOL_INTEG_EDGEQUADRATUREPROJECTION_HPP_
#define SRC_TRIBOL_INTEG_EDGEQUADRATUREPROJECTION_HPP_

#include "tribol/geom/EdgeProjection.hpp"
#include "tribol/integ/PairQuadrature.hpp"

namespace tribol {

/**
 * @brief Project source physical points onto a target edge with a configured projection.
 *
 * Projection preserves source points and weights. Invalid targets and short capacity leave output untouched. A failed
 * point projection may leave partial output, which must be discarded.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @tparam ParentDimension Number of parent-coordinate components per point.
 * @tparam Projection Value type providing targetIsValid() and projectPoint().
 * @param source Completed source points and physical weights.
 * @param target Target endpoints indexed by endpoint, then physical component.
 * @param [out] target_parent_coordinates Writable target coordinates [point][1]; extra capacity is untouched.
 * @param projection Target validation and point-projection behavior.
 * @return Success and the point count, InsufficientCapacity and the required count, or InvalidGeometry and zero.
 */
template <typename Value, int ParentDimension, typename Projection>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult projectQuadratureToEdge(
    SurfaceQuadratureView<const Value, ParentDimension, 2> source, MdSpan<const Value, 2, 2> target,
    ArrayViewT<Value, 2> target_parent_coordinates, const Projection& projection )
{
  assert( target_parent_coordinates.shape()[1] == 1 );
  if ( !projection.template targetIsValid<Value>( target ) ) return { QuadratureStatus::InvalidGeometry, 0 };
  const IndexT count = source.physical_weights.size();
  if ( target_parent_coordinates.shape()[0] < count ) return { QuadratureStatus::InsufficientCapacity, count };
  for ( IndexT q = 0; q < count; ++q ) {
    const auto point = projection.template projectPoint<Value>(
        MdSpan<const Value, 2>{ source.physical_coordinates.data() + 2 * q, 2 }, target );
    if ( !point.valid ) return { QuadratureStatus::InvalidGeometry, 0 };
    target_parent_coordinates( q, 0 ) = point.parent_coordinate;
  }
  return { QuadratureStatus::Success, count };
}

/**
 * @brief Assemble paired parent coordinates and source weights after projecting source points.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @tparam Capacity Number of points available in the fixed-size backing arrays.
 * @tparam Projection Value type providing target validation and point mapping.
 * @param source Owning source quadrature; a failed source result is propagated.
 * @param target Target endpoint coordinates.
 * @param [out] output Writable paired coordinates and source weights.
 * @param projection Target validation and point-projection behavior.
 * @return Completion status and count; any failed result requires discarding output.
 */
template <typename Value, IndexT Capacity, typename Projection>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
projectEdgeQuadrature( const SurfaceQuadratureData<Value, 1, 2, Capacity>& source, MdSpan<const Value, 2, 2> target,
                       PairQuadratureView<Value, 1> output, const Projection& projection )
{
  if ( source.result.status != QuadratureStatus::Success ) return source.result;
  const auto view = source.getView();
  const auto projected = projectQuadratureToEdge( view, target, output.mesh2_parent_coordinates, projection );
  if ( projected.status != QuadratureStatus::Success ) return projected;
  for ( IndexT q = 0; q < projected.required_point_count; ++q ) {
    output.mesh1_parent_coordinates( q, 0 ) = view.parent_coordinates( q, 0 );
    output.physical_weights[q] = view.physical_weights[q];
  }
  return projected;
}

/**
 * @brief Assemble a paired rule in fixed-capacity local storage.
 *
 * @tparam Value Coordinate and weight scalar type.
 * @tparam Capacity Number of points available in the fixed-size backing arrays.
 * @tparam Projection Value type providing target validation and point mapping.
 * @param source Owning source quadrature.
 * @param target Target endpoint coordinates.
 * @param projection Target validation and point-projection behavior.
 * @return Owned paired quadrature, with completion status and source physical weights.
 */
template <typename Value, IndexT Capacity, typename Projection>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureData<Value, 1, Capacity> projectEdgeQuadrature(
    const SurfaceQuadratureData<Value, 1, 2, Capacity>& source, MdSpan<const Value, 2, 2> target,
    const Projection& projection )
{
  PairQuadratureData<Value, 1, Capacity> output{};
  output.result = projectEdgeQuadrature( source, target, output.getOutputView(), projection );
  return output;
}

}  // namespace tribol

#endif
