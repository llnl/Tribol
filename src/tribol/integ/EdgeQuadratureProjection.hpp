// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_EDGEQUADRATUREPROJECTION_HPP_
#define SRC_TRIBOL_INTEG_EDGEQUADRATUREPROJECTION_HPP_

#include "tribol/integ/EdgeQuadratureRule.hpp"
#include "tribol/integ/PairQuadrature.hpp"
#include "tribol/search/InterfacePairChecks.hpp"

namespace tribol {

/**
 * @brief Find the source-edge parent interval covered by projection of another edge
 *
 * Both edges have two nodes in two dimensions. Invalid geometry returns valid=false and zero bounds; an empty
 * overlap is still a valid calculation. Smoothing and interval clipping belong to the chosen quadrature rule.
 */
template <typename Value>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE EdgeQuadratureInterval<Value> projectEdgeInterval(
    MdSpan<const Value, 2, 2> source, MdSpan<const Value, 2, 2> other )
{
  const auto projection = InterfacePairChecks::computeLinearEdgeProjectionBounds( source.data(), source.data() + 2,
                                                                                  other.data(), other.data() + 2 );
  return { projection.bounds, projection.valid };
}

/**
 * @brief Project source physical points onto a target edge along the target normal
 *
 * Only target parent coordinates are written; the source points and physical weights are unchanged. Call this
 * separately for each target surface. Target coordinates may lie outside [-1, 1]; this operation does not clip them.
 * A degenerate target returns InvalidGeometry with zero points; short output returns InsufficientCapacity with the
 * full count. Neither failure writes output. Extra capacity is untouched.
 */
template <typename Value, int ParentDimension>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
projectQuadratureToEdge( SurfaceQuadratureView<const Value, ParentDimension, 2> source,
                         MdSpan<const Value, 2, 2> target, ArrayViewT<Value, 2> target_parent_coordinates )
{
  assert( target_parent_coordinates.shape()[1] == 1 );
  const Value dx = target( 1, 0 ) - target( 0, 0 );
  const Value dy = target( 1, 1 ) - target( 0, 1 );
  const Value length_squared = dx * dx + dy * dy;
  constexpr double minimum_edge_length = 1.0e-14;
  if ( length_squared < minimum_edge_length * minimum_edge_length ) {
    return { QuadratureStatus::InvalidGeometry, 0 };
  }
  const IndexT count = source.physical_weights.size();
  if ( target_parent_coordinates.shape()[0] < count ) return { QuadratureStatus::InsufficientCapacity, count };
  for ( IndexT q = 0; q < count; ++q ) {
    const Value px = source.physical_coordinates( q, 0 ) - target( 0, 0 );
    const Value py = source.physical_coordinates( q, 1 ) - target( 0, 1 );
    target_parent_coordinates( q, 0 ) = 2.0 * ( px * dx + py * dy ) / length_squared - 1.0;
  }
  return { QuadratureStatus::Success, count };
}

/** @brief Assemble pair coordinates and source weights after projecting a completed source rule onto a target edge */
template <typename Value, IndexT Capacity>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE QuadratureResult
projectEdgeQuadrature( const SurfaceQuadratureData<Value, 1, 2, Capacity>& source, MdSpan<const Value, 2, 2> target,
                       PairQuadratureView<Value, 1> output )
{
  if ( source.result.status != QuadratureStatus::Success ) return source.result;
  const auto view = source.getView();
  const auto projected = projectQuadratureToEdge( view, target, output.mesh2_parent_coordinates );
  if ( projected.status != QuadratureStatus::Success ) return projected;
  for ( IndexT q = 0; q < projected.required_point_count; ++q ) {
    output.mesh1_parent_coordinates( q, 0 ) = view.parent_coordinates( q, 0 );
    output.physical_weights[q] = view.physical_weights[q];
  }
  return projected;
}

/** @brief Assemble a pair rule in fixed-capacity local storage without retaining a view into the source */
template <typename Value, IndexT Capacity>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureData<Value, 1, Capacity> projectEdgeQuadrature(
    const SurfaceQuadratureData<Value, 1, 2, Capacity>& source, MdSpan<const Value, 2, 2> target )
{
  PairQuadratureData<Value, 1, Capacity> output{};
  output.result = projectEdgeQuadrature( source, target, output.getOutputView() );
  return output;
}

}  // namespace tribol

#endif
