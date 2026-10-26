// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_GENERATEPAIRQUADRATURE_HPP_
#define SRC_TRIBOL_INTEG_GENERATEPAIRQUADRATURE_HPP_

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tribol/integ/InterfaceQuadrature.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/EdgeQuadratureProjection.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {
namespace detail {

inline IndexT checkedQuadratureSum( IndexT a, IndexT b )
{
  if ( a < 0 || b < 0 || b > std::numeric_limits<IndexT>::max() - a ) {
    throw std::length_error( "Quadrature storage exceeds the index range." );
  }
  return a + b;
}

inline IndexT checkedQuadratureProduct( IndexT a, IndexT b )
{
  if ( a < 0 || b < 0 || ( b != 0 && a > std::numeric_limits<IndexT>::max() / b ) ) {
    throw std::length_error( "Quadrature storage exceeds the index range." );
  }
  return a * b;
}

/** @brief Own point storage without allocating pair arrays or offsets for temporary slices */
template <int Dimension>
class QuadratureStorage {
 public:
  QuadratureStorage( IndexT count, int allocator )
      : mesh1_( axom::StackArray<axom::IndexType, 2>{ count, Dimension }, allocator ),
        mesh2_( axom::StackArray<axom::IndexType, 2>{ count, Dimension }, allocator ),
        weights_( axom::ArrayOptions::Uninitialized{}, count, count, allocator )
  {
  }

  PairQuadratureView<RealT, Dimension> slice( IndexT begin, IndexT count )
  {
    // Empty buffers may have null data pointers; avoid pointer arithmetic on those pointers.
    auto* coords1 = mesh1_.data();
    auto* coords2 = mesh2_.data();
    auto* weights = weights_.data();
    if ( begin != 0 ) {
      coords1 += begin * Dimension;
      coords2 += begin * Dimension;
      weights += begin;
    }
    const axom::StackArray<axom::IndexType, 2> shape{ count, Dimension };
    return { ArrayViewT<RealT, 2>{ coords1, shape }, ArrayViewT<RealT, 2>{ coords2, shape },
             ArrayViewT<RealT>{ weights, count } };
  }

 private:
  ArrayT<RealT, 2> mesh1_;
  ArrayT<RealT, 2> mesh2_;
  ArrayT<RealT> weights_;
};

/**
 * @brief Generate each host pair once within its capacity limit and pack completed output
 *
 * Insufficient capacity throws length_error identifying the pair, capacity, and required count. No partial output
 * is returned. Pair order and each pair's point order are preserved; zero-point pairs are omitted. The callback keeps
 * allocation separate from the host/device per-pair generator.
 */
template <typename QuadratureRule, typename GeneratePair>
InterfaceQuadratureData generateQuadrature( const QuadratureRule& quadrature_rule,
                                            ArrayViewT<const InterfacePair> pairs, int allocator,
                                            GeneratePair&& generate_pair )
{
#ifdef TRIBOL_USE_UMPIRE
  if ( axom::isDeviceAllocator( allocator ) ) {
    throw std::invalid_argument( "Batch quadrature generation requires host-accessible storage." );
  }
#endif
  constexpr int dimension = QuadratureRule::parent_dimension;
  const IndexT capacity = quadrature_rule.pointCapacity();
  const IndexT scratch_point_count = checkedQuadratureProduct( pairs.size(), capacity );
  checkedQuadratureProduct( scratch_point_count, dimension );
  QuadratureStorage<dimension> scratch( scratch_point_count, allocator );
  std::vector<IndexT> point_counts( pairs.size() );
  IndexT contributing_pairs = 0;
  IndexT point_count = 0;
  for ( IndexT i = 0; i < pairs.size(); ++i ) {
    const auto result = generate_pair( i, scratch.slice( i * capacity, capacity ) );
    if ( result.status == QuadratureStatus::InvalidGeometry ) {
      throw std::invalid_argument( "Cannot generate a quadrature rule for invalid pair geometry." );
    }
    if ( result.required_point_count < 0 ||
         ( result.status == QuadratureStatus::Success && result.required_point_count > capacity ) ||
         ( result.status == QuadratureStatus::InsufficientCapacity && result.required_point_count <= capacity ) ) {
      throw std::runtime_error( "Quadrature generator reported an inconsistent point count." );
    }
    if ( result.status == QuadratureStatus::InsufficientCapacity ) {
      throw std::length_error(
          "Quadrature pair " + std::to_string( i ) + " (elements " + std::to_string( pairs[i].element_id1_ ) + ", " +
          std::to_string( pairs[i].element_id2_ ) + ") requires " + std::to_string( result.required_point_count ) +
          " points; capacity " + std::to_string( capacity ) + "." );
    }
    point_counts[i] = result.required_point_count;
    if ( point_counts[i] > 0 ) {
      contributing_pairs = checkedQuadratureSum( contributing_pairs, 1 );
      point_count = checkedQuadratureSum( point_count, point_counts[i] );
    }
  }
  checkedQuadratureSum( contributing_pairs, 1 );
  checkedQuadratureProduct( point_count, dimension );
  InterfaceQuadratureData output( dimension, contributing_pairs, point_count, allocator );
  auto output_pairs = output.getInterfacePairs();
  auto offsets = output.getPairOffsets();
  auto coords1 = output.getMesh1ParentCoordinates();
  auto coords2 = output.getMesh2ParentCoordinates();
  auto weights = output.getWeights();
  offsets[0] = 0;
  IndexT pair_index = 0;
  IndexT point_index = 0;
  for ( IndexT i = 0; i < pairs.size(); ++i ) {
    const IndexT count = point_counts[i];
    if ( count == 0 ) continue;
    const auto source = scratch.slice( i * capacity, count );
    output_pairs[pair_index] = pairs[i];
    for ( IndexT q = 0; q < count; ++q ) {
      for ( int d = 0; d < dimension; ++d ) {
        coords1( point_index, d ) = source.mesh1_parent_coordinates( q, d );
        coords2( point_index, d ) = source.mesh2_parent_coordinates( q, d );
      }
      weights[point_index++] = source.physical_weights[q];
    }
    offsets[++pair_index] = point_index;
  }
  return output;
}

}  // namespace detail

/**
 * @brief Generate on source-edge projected intervals, then project onto target edges and pack pair data
 *
 * Both meshes must contain two-dimensional linear edges in host-accessible memory. Only supplied pairs receive
 * temporary storage. Capacity errors throw length_error without returning partial results or regenerating any pair.
 */
template <typename QuadratureRule, typename SelectInterval>
InterfaceQuadratureData generatePairQuadrature( const QuadratureRule& quadrature_rule,
                                                const ArrayT<InterfacePair>& pairs, const MeshData::Viewer& mesh1,
                                                const MeshData::Viewer& mesh2, const SelectInterval& select_interval )
{
  using Traits = FaceElementTraits<EDGE, LINEAR_LAGRANGE>;
  if ( mesh1.spatialDimension() != 2 || mesh2.spatialDimension() != 2 || mesh1.getElementType() != EDGE ||
       mesh2.getElementType() != EDGE || mesh1.getFaceOrder() != LINEAR_LAGRANGE ||
       mesh2.getFaceOrder() != LINEAR_LAGRANGE ) {
    throw std::invalid_argument( "Projected edge quadrature requires two-dimensional linear edges." );
  }
#ifdef TRIBOL_USE_UMPIRE
  if ( mesh1.getMemorySpace() == MemorySpace::Device || mesh2.getMemorySpace() == MemorySpace::Device ||
       axom::isDeviceAllocator( mesh1.getAllocatorId() ) || axom::isDeviceAllocator( mesh2.getAllocatorId() ) ) {
    throw std::invalid_argument( "Batch quadrature generation requires host-accessible meshes." );
  }
#endif
  return detail::generateQuadrature<QuadratureRule>(
      quadrature_rule, pairs.view(), mesh1.getAllocatorId(),
      [&]( IndexT index, PairQuadratureView<RealT, QuadratureRule::parent_dimension> output ) {
        MdArray<RealT, Traits::node_count, Traits::spatial_dimension> coords1;
        MdArray<RealT, Traits::node_count, Traits::spatial_dimension> coords2;
        mesh1.getFaceCoords( pairs[index].element_id1_, coords1.data() );
        mesh2.getFaceCoords( pairs[index].element_id2_, coords2.data() );
        const auto interval = select_interval.template operator()<RealT>( coords1.view(), coords2.view() );
        const auto source = quadrature_rule.template generate<RealT>( coords1.view(), interval );
        return projectEdgeQuadrature<RealT>( source, coords2.view(), output );
      } );
}

}  // namespace tribol

#endif
