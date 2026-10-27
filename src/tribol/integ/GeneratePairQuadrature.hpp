// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_GENERATEPAIRQUADRATURE_HPP_
#define SRC_TRIBOL_INTEG_GENERATEPAIRQUADRATURE_HPP_

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "tribol/common/LoopExec.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/EdgeQuadratureProjection.hpp"
#include "tribol/integ/InterfaceQuadrature.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {
namespace detail {

/**
 * @brief Add two quadrature storage counts after checking the IndexT range
 *
 * @param a First nonnegative count
 * @param b Second nonnegative count
 * @return Sum of a and b
 *
 * @throws std::length_error when either count is negative or their sum exceeds the IndexT range
 */
inline IndexT checkedQuadratureSum( IndexT a, IndexT b )
{
  if ( a < 0 || b < 0 || b > std::numeric_limits<IndexT>::max() - a ) {
    throw std::length_error( "Quadrature storage exceeds the index range." );
  }
  return a + b;
}

/**
 * @brief Multiply two quadrature storage counts after checking the IndexT range
 *
 * @param a First nonnegative count
 * @param b Second nonnegative count
 * @return Product of a and b
 *
 * @throws std::length_error when either count is negative or their product exceeds the IndexT range
 */
inline IndexT checkedQuadratureProduct( IndexT a, IndexT b )
{
  if ( a < 0 || b < 0 || ( b != 0 && a > std::numeric_limits<IndexT>::max() / b ) ) {
    throw std::length_error( "Quadrature storage exceeds the index range." );
  }
  return a * b;
}

/**
 * @brief Borrow point storage and create per-pair slices inside host or device kernels
 *
 * @tparam Dimension Number of parent-coordinate components stored for each quadrature point
 */
template <int Dimension>
class QuadratureStorageView {
 public:
  /**
   * @brief Borrow flattened scratch arrays for pair generation and packing
   *
   * @param mesh1 Mesh 1 parent coordinates indexed by scratch point, then parent component
   * @param mesh2 Mesh 2 parent coordinates indexed by scratch point, then parent component
   * @param weights Physical integration weights indexed by scratch point
   */
  TRIBOL_HOST_DEVICE QuadratureStorageView( ArrayViewT<RealT, 2> mesh1, ArrayViewT<RealT, 2> mesh2,
                                            ArrayViewT<RealT> weights )
      : mesh1_( mesh1 ), mesh2_( mesh2 ), weights_( weights )
  {
  }

  /**
   * @brief Borrow one pair's contiguous range from the flattened scratch arrays
   *
   * @param begin First scratch-point index in the pair's range
   * @param count Number of writable points in the pair's range
   * @return Pair quadrature view with count-point coordinate and weight arrays
   */
  TRIBOL_HOST_DEVICE PairQuadratureView<RealT, Dimension> slice( IndexT begin, IndexT count ) const
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
  /** @brief Mesh 1 parent coordinates for every scratch point */
  ArrayViewT<RealT, 2> mesh1_;

  /** @brief Mesh 2 parent coordinates for every scratch point */
  ArrayViewT<RealT, 2> mesh2_;

  /** @brief Physical integration weights for every scratch point */
  ArrayViewT<RealT> weights_;
};

/**
 * @brief Own point storage without allocating pair arrays or offsets for temporary slices
 *
 * @tparam Dimension Number of parent-coordinate components stored for each quadrature point
 */
template <int Dimension>
class QuadratureStorage {
 public:
  /**
   * @brief Allocate flattened scratch storage beside the pair and mesh inputs
   *
   * @param count Total scratch-point capacity across all submitted pairs
   * @param allocator Allocator ID shared by all scratch arrays
   */
  QuadratureStorage( IndexT count, int allocator )
      : mesh1_( axom::StackArray<axom::IndexType, 2>{ count, Dimension }, allocator ),
        mesh2_( axom::StackArray<axom::IndexType, 2>{ count, Dimension }, allocator ),
        weights_( axom::ArrayOptions::Uninitialized{}, count, count, allocator )
  {
  }

  /**
   * @brief Borrow writable scratch storage for execution-space kernels
   *
   * @return View containing all scratch coordinate and weight arrays
   */
  QuadratureStorageView<Dimension> getView() { return { mesh1_.view(), mesh2_.view(), weights_.view() }; }

 private:
  /** @brief Owned mesh 1 parent coordinates indexed by scratch point, then parent component */
  ArrayT<RealT, 2> mesh1_;

  /** @brief Owned mesh 2 parent coordinates indexed by scratch point, then parent component */
  ArrayT<RealT, 2> mesh2_;

  /** @brief Owned physical integration weights indexed by scratch point */
  ArrayT<RealT> weights_;
};

/** @brief Counts exact-size output after host validation of all pair results */
struct QuadratureTotals {
  /** @brief Number of successful input pairs with at least one quadrature point */
  IndexT contributing_pairs{};

  /** @brief Total number of quadrature points from successful input pairs */
  IndexT points{};
};

/**
 * @brief Report the first failed pair and total successful output after parallel generation completes
 *
 * Results are inspected in input order so parallel generation retains the sequential exception order. Pair data is
 * copied to the host only when an insufficient-capacity message needs the element IDs.
 *
 * @param results Host-accessible status and required-point count for each input pair
 * @param pairs Input pairs in the same order as results
 * @param capacity Scratch-point capacity supplied to each pair generator
 * @return Number of contributing pairs and total successful quadrature points
 *
 * @throws std::invalid_argument when the first failed pair has invalid geometry
 * @throws std::runtime_error when a generator reports a status and point count that disagree
 * @throws std::length_error when a pair needs more than capacity points or output totals exceed the IndexT range
 */
inline QuadratureTotals validateQuadratureResults( ArrayViewT<const QuadratureResult, 1, MemorySpace::Host> results,
                                                   ArrayViewT<const InterfacePair> pairs, IndexT capacity )
{
  QuadratureTotals totals;
  for ( IndexT pair_index = 0; pair_index < results.size(); ++pair_index ) {
    const auto result = results[pair_index];
    if ( result.status == QuadratureStatus::InvalidGeometry ) {
      throw std::invalid_argument( "Cannot generate a quadrature rule for invalid pair geometry." );
    }
    if ( result.required_point_count < 0 ||
         ( result.status == QuadratureStatus::Success && result.required_point_count > capacity ) ||
         ( result.status == QuadratureStatus::InsufficientCapacity && result.required_point_count <= capacity ) ) {
      throw std::runtime_error( "Quadrature generator reported an inconsistent point count." );
    }
    if ( result.status == QuadratureStatus::InsufficientCapacity ) {
      const ArrayT<InterfacePair, 1, MemorySpace::Host> host_pairs( pairs );
      throw std::length_error( "Quadrature pair " + std::to_string( pair_index ) + " (elements " +
                               std::to_string( host_pairs[pair_index].element_id1_ ) + ", " +
                               std::to_string( host_pairs[pair_index].element_id2_ ) + ") requires " +
                               std::to_string( result.required_point_count ) + " points; capacity " +
                               std::to_string( capacity ) + "." );
    }
    if ( result.status == QuadratureStatus::Success && result.required_point_count > 0 ) {
      totals.contributing_pairs = checkedQuadratureSum( totals.contributing_pairs, 1 );
      totals.points = checkedQuadratureSum( totals.points, result.required_point_count );
    }
  }
  return totals;
}

/**
 * @brief Generate every pair once, compact successful output in input order, and return exact-size storage
 *
 * The generator runs in the selected execution mode and must be callable from that execution space. Host code checks
 * completed statuses so kernels never throw. Insufficient capacity identifies the first failing input pair. No partial
 * output is returned. Zero-point pairs are omitted, while contributing pairs and their points retain input order.
 *
 * @tparam EXEC Concrete execution mode used by generation, scans, and packing
 * @tparam QuadratureRule Rule type that provides parent_dimension and pointCapacity()
 * @tparam GeneratePair Device-copyable callable that generates one pair into supplied scratch storage
 *
 * @param quadrature_rule Rule object that defines the per-pair scratch capacity
 * @param pairs Input interface pairs in the order retained by output compaction
 * @param allocator Allocator ID used by pairs, scratch storage, and returned data
 * @param generate_pair Callable receiving an input-pair index and its writable PairQuadratureView
 * @return Exact-size owned quadrature data allocated with allocator
 *
 * @throws std::invalid_argument when any generated pair has invalid geometry
 * @throws std::runtime_error when a generator reports an inconsistent point count
 * @throws std::length_error when storage sizes exceed IndexT or any pair exceeds its scratch capacity
 */
template <ExecutionMode EXEC, typename QuadratureRule, typename GeneratePair>
InterfaceQuadratureData generateQuadrature( const QuadratureRule& quadrature_rule,
                                            ArrayViewT<const InterfacePair> pairs, int allocator,
                                            GeneratePair&& generate_pair )
{
  static_assert( EXEC != ExecutionMode::Dynamic, "Quadrature generation requires a concrete execution mode." );
  constexpr int dimension = QuadratureRule::parent_dimension;
  const IndexT pair_count = pairs.size();
  const IndexT capacity = quadrature_rule.pointCapacity();
  const IndexT scratch_point_count = checkedQuadratureProduct( pair_count, capacity );
  checkedQuadratureProduct( scratch_point_count, dimension );
  checkedQuadratureSum( pair_count, 1 );
  if ( pair_count == 0 ) return InterfaceQuadratureData( dimension, allocator );

  QuadratureStorage<dimension> scratch( scratch_point_count, allocator );
  ArrayT<QuadratureResult> results( axom::ArrayOptions::Uninitialized{}, pair_count, pair_count, allocator );
  ArrayT<IndexT> point_offsets( pair_count + 1, pair_count + 1, allocator );
  ArrayT<IndexT> contributing_pair_offsets( pair_count + 1, pair_count + 1, allocator );
  point_offsets.fill( 0 );
  contributing_pair_offsets.fill( 0 );

  const auto scratch_view = scratch.getView();
  auto result_values = results.view();
  auto point_offset_values = point_offsets.view();
  auto contributing_pair_offset_values = contributing_pair_offsets.view();
  using Generator = std::decay_t<GeneratePair>;
  Generator generator( std::forward<GeneratePair>( generate_pair ) );
  forAllExec<EXEC>( pair_count, [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) mutable {
    const auto result = generator( pair_index, scratch_view.slice( pair_index * capacity, capacity ) );
    result_values[pair_index] = result;
    const bool has_valid_count = result.status == QuadratureStatus::Success && result.required_point_count >= 0 &&
                                 result.required_point_count <= capacity;
    const IndexT point_count = has_valid_count ? result.required_point_count : 0;
    point_offset_values[pair_index] = point_count;
    contributing_pair_offset_values[pair_index] = point_count > 0 ? 1 : 0;
  } );

  exclusiveScanInplaceExec<EXEC>( point_offsets.view() );
  exclusiveScanInplaceExec<EXEC>( contributing_pair_offsets.view() );

  const auto scanned_point_offsets = point_offsets.view();
  const auto scanned_pair_offsets = contributing_pair_offsets.view();
  const ArrayT<QuadratureResult, 1, MemorySpace::Host> host_results( results );
  const auto totals = validateQuadratureResults( host_results.view(), pairs, capacity );
  const IndexT contributing_pairs = totals.contributing_pairs;
  const IndexT point_count = totals.points;
  checkedQuadratureSum( contributing_pairs, 1 );
  checkedQuadratureProduct( point_count, dimension );

  InterfaceQuadratureData output( dimension, contributing_pairs, point_count, allocator );
  auto output_pairs = output.getInterfacePairs();
  auto output_offsets = output.getPairOffsets();
  auto coords1 = output.getMesh1ParentCoordinates();
  auto coords2 = output.getMesh2ParentCoordinates();
  auto weights = output.getWeights();
  forAllExec<EXEC>( pair_count, [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
    if ( pair_index == pair_count - 1 ) output_offsets[contributing_pairs] = point_count;
    const IndexT count = scanned_point_offsets[pair_index + 1] - scanned_point_offsets[pair_index];
    if ( count == 0 ) return;
    const IndexT output_pair = scanned_pair_offsets[pair_index];
    const IndexT output_point = scanned_point_offsets[pair_index];
    const auto source = scratch_view.slice( pair_index * capacity, count );
    output_pairs[output_pair] = pairs[pair_index];
    output_offsets[output_pair] = output_point;
    for ( IndexT point = 0; point < count; ++point ) {
      for ( int component = 0; component < dimension; ++component ) {
        coords1( output_point + point, component ) = source.mesh1_parent_coordinates( point, component );
        coords2( output_point + point, component ) = source.mesh2_parent_coordinates( point, component );
      }
      weights[output_point + point] = source.physical_weights[point];
    }
  } );
  return output;
}

/**
 * @brief Dispatch pair generation when the execution mode is selected at run time
 *
 * @tparam QuadratureRule Rule type that provides parent_dimension and pointCapacity()
 * @tparam GeneratePair Device-copyable callable that generates one pair into supplied scratch storage
 *
 * @param execution_mode Concrete execution mode used by generation, scans, and packing
 * @param quadrature_rule Rule object that defines the per-pair scratch capacity
 * @param pairs Input interface pairs in the order retained by output compaction
 * @param allocator Allocator ID used by pairs, scratch storage, and returned data
 * @param generate_pair Callable receiving an input-pair index and its writable PairQuadratureView
 * @return Exact-size owned quadrature data allocated with allocator
 *
 * @throws std::invalid_argument when execution_mode is not concrete or any generated pair has invalid geometry
 * @throws std::runtime_error when a generator reports an inconsistent point count
 * @throws std::length_error when storage sizes exceed IndexT or any pair exceeds its scratch capacity
 */
template <typename QuadratureRule, typename GeneratePair>
InterfaceQuadratureData generateQuadrature( ExecutionMode execution_mode, const QuadratureRule& quadrature_rule,
                                            ArrayViewT<const InterfacePair> pairs, int allocator,
                                            GeneratePair&& generate_pair )
{
  return dispatchExecutionMode( execution_mode, "Quadrature generation requires a concrete execution mode.",
                                [&]<ExecutionMode EXEC>() {
                                  return generateQuadrature<EXEC>( quadrature_rule, pairs, allocator,
                                                                   std::forward<GeneratePair>( generate_pair ) );
                                } );
}

}  // namespace detail

/**
 * @brief Generate projected-edge quadrature in the selected execution mode and pack owned pair data
 *
 * Both meshes must contain two-dimensional linear edges. Pairs and both meshes must use the same allocator, and their
 * memory must be accessible from EXEC. Only supplied pairs receive temporary storage. Capacity errors throw
 * length_error after generation completes, without returning partial results or regenerating any pair.
 *
 * @tparam EXEC Concrete execution mode used by generation, scans, and packing
 * @tparam QuadratureRule Projected-edge rule type with a fixed parent dimension
 * @tparam SelectInterval Device-copyable callable that selects one source-edge parent interval
 *
 * @param quadrature_rule Rule used to generate source-edge points and physical weights
 * @param pairs Input element pairs allocated beside both meshes
 * @param mesh1 Integration-side mesh used to generate source quadrature
 * @param mesh2 Target mesh onto which source quadrature points are projected
 * @param select_interval Callable receiving both edge coordinate arrays and returning the source interval
 * @return Exact-size owned quadrature data allocated beside pairs and both meshes
 *
 * @throws std::invalid_argument when geometry, allocators, or execution-space accessibility are invalid
 * @throws std::runtime_error when the underlying rule reports an inconsistent point count
 * @throws std::length_error when storage sizes exceed IndexT or any pair exceeds its scratch capacity
 */
template <ExecutionMode EXEC, typename QuadratureRule, typename SelectInterval>
InterfaceQuadratureData generatePairQuadrature( const QuadratureRule& quadrature_rule,
                                                const ArrayT<InterfacePair>& pairs, const MeshData::Viewer& mesh1,
                                                const MeshData::Viewer& mesh2, const SelectInterval& select_interval )
{
  static_assert( EXEC != ExecutionMode::Dynamic, "Pair quadrature generation requires a concrete execution mode." );
  using Traits = FaceElementTraits<EDGE, LINEAR_LAGRANGE>;
  if ( mesh1.spatialDimension() != 2 || mesh2.spatialDimension() != 2 || mesh1.getElementType() != EDGE ||
       mesh2.getElementType() != EDGE || mesh1.getFaceOrder() != LINEAR_LAGRANGE ||
       mesh2.getFaceOrder() != LINEAR_LAGRANGE ) {
    throw std::invalid_argument( "Projected edge quadrature requires two-dimensional linear edges." );
  }
  if ( mesh1.getAllocatorId() != mesh2.getAllocatorId() || mesh1.getAllocatorId() != pairs.getAllocatorID() ) {
    throw std::invalid_argument( "Projected edge quadrature requires pairs and meshes to use the same allocator." );
  }
  if ( !executionModeUsesAllocator<EXEC>( mesh1.getAllocatorId() ) ) {
    throw std::invalid_argument(
        "Projected edge quadrature requires an allocator accessible from the execution mode." );
  }
  const auto pair_values = pairs.view();
  return detail::generateQuadrature<EXEC>(
      quadrature_rule, pair_values, mesh1.getAllocatorId(),
      [=] TRIBOL_HOST_DEVICE( IndexT pair_index, PairQuadratureView<RealT, QuadratureRule::parent_dimension> output ) {
        MdArray<RealT, Traits::node_count, Traits::spatial_dimension> coords1;
        MdArray<RealT, Traits::node_count, Traits::spatial_dimension> coords2;
        mesh1.getFaceCoords( pair_values[pair_index].element_id1_, coords1.data() );
        mesh2.getFaceCoords( pair_values[pair_index].element_id2_, coords2.data() );
        const auto interval = select_interval.template operator()<RealT>( coords1.view(), coords2.view() );
        const auto source = quadrature_rule.template generate<RealT>( coords1.view(), interval );
        return projectEdgeQuadrature<RealT>( source, coords2.view(), output );
      } );
}

/**
 * @brief Dispatch projected-edge quadrature generation when the execution mode is selected at run time
 *
 * @tparam QuadratureRule Projected-edge rule type with a fixed parent dimension
 * @tparam SelectInterval Device-copyable callable that selects one source-edge parent interval
 *
 * @param execution_mode Concrete execution mode used by generation, scans, and packing
 * @param quadrature_rule Rule used to generate source-edge points and physical weights
 * @param pairs Input element pairs allocated beside both meshes
 * @param mesh1 Integration-side mesh used to generate source quadrature
 * @param mesh2 Target mesh onto which source quadrature points are projected
 * @param select_interval Callable receiving both edge coordinate arrays and returning the source interval
 * @return Exact-size owned quadrature data allocated beside pairs and both meshes
 *
 * @throws std::invalid_argument when execution_mode, geometry, allocators, or accessibility are invalid
 * @throws std::runtime_error when the underlying rule reports an inconsistent point count
 * @throws std::length_error when storage sizes exceed IndexT or any pair exceeds its scratch capacity
 */
template <typename QuadratureRule, typename SelectInterval>
InterfaceQuadratureData generatePairQuadrature( ExecutionMode execution_mode, const QuadratureRule& quadrature_rule,
                                                const ArrayT<InterfacePair>& pairs, const MeshData::Viewer& mesh1,
                                                const MeshData::Viewer& mesh2, const SelectInterval& select_interval )
{
  return dispatchExecutionMode(
      execution_mode, "Pair quadrature generation requires a concrete execution mode.", [&]<ExecutionMode EXEC>() {
        return generatePairQuadrature<EXEC>( quadrature_rule, pairs, mesh1, mesh2, select_interval );
      } );
}

}  // namespace tribol

#endif
