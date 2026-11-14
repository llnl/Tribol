// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Integration-surface and element-pair quadrature rules, views, generation results, and validation.
 *
 * Integration-surface representations shared by mappings and rule generators belong in this header. Keep
 * representations for additional element topologies or geometries here alongside EdgeQuadratureInterval.
 */

#ifndef SRC_TRIBOL_INTEG_QUADRATURECOMMON_HPP_
#define SRC_TRIBOL_INTEG_QUADRATURECOMMON_HPP_

#include <cassert>
#include <type_traits>

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/common/Reductions.hpp"

namespace tribol {

/**
 * @brief Bounds in a one-dimensional integration surface's coordinates, which need not be normalized element
 * coordinates.
 */
struct EdgeQuadratureInterval {
  MdArray<RealT, 2> bounds{ -1.0, 1.0 };  ///< Lower and upper bounds; default to the whole reference edge.
  bool valid{ true };                     ///< False for invalid geometry; collapsed valid bounds have zero measure.
};

/** @brief Whether a complete rule was generated, needs more storage, or has invalid geometry */
enum class QuadratureStatus
{
  Success,               ///< Every reported point is initialized; zero points means zero measure.
  InsufficientCapacity,  ///< Reported count exceeds capacity; discard partial output.
  InvalidGeometry        ///< Mapping rejected the supplied element geometry; discard output and report zero points.
};

/**
 * @brief Throw a quadrature failure outside differentiated calculations.
 * @param status Failed generation status; must not be Success.
 * @throws std::length_error The rule requires more point storage.
 * @throws std::invalid_argument Pair geometry is invalid.
 */
[[noreturn, gnu::noinline]]
#ifdef TRIBOL_USE_ENZYME
MFEM_ENZYME_INACTIVE
#endif
    void reportQuadratureFailure( QuadratureStatus status );

/**
 * @brief Reports the complete point count, even when the supplied storage is too small
 *
 * Success initializes every point below num_required_points; success with zero points means zero integration
 * measure. InsufficientCapacity may leave partial output, which must be discarded. InvalidGeometry
 * returns zero points and its output must also be discarded. Unused capacity is never part of a completed rule.
 */
struct QuadratureResult {
  QuadratureStatus status{ QuadratureStatus::Success };  ///< Whether the complete rule can be consumed.
  IndexT num_required_points{ 0 };  ///< Complete count on success or overflow; zero for invalid geometry.
};

namespace detail {

/**
 * @brief Find the most severe quadrature error across all pairs using the specified execution policy.
 * @tparam Exec Concrete execution policy.
 * @param results One status per pair in execution-accessible memory.
 * @return Highest-severity status present in the array, or Success when the array is empty.
 */
template <ExecutionMode Exec>
QuadratureStatus reduceQuadratureStatus( ArrayViewT<const QuadratureResult> results )
{
  ReduceMax<Exec, int> status( static_cast<int>( QuadratureStatus::Success ) );
  forAllExec<Exec>( results.size(),
                    [=] TRIBOL_HOST_DEVICE( IndexT pair ) { status.max( static_cast<int>( results[pair].status ) ); } );
  return static_cast<QuadratureStatus>( status.get() );
}

}  // namespace detail

/**
 * @brief Check the quadrature rules produced for all pairs and throw if any rule reports an error.
 * @tparam Exec Concrete execution mode used to populate the results.
 * @param results One preallocated result per pair.
 * Empty results pass the check.
 * @throws std::invalid_argument A rule reports invalid geometry.
 * @throws std::length_error A rule reports insufficient quadrature capacity.
 */
template <ExecutionMode Exec>
void checkPairQuadratureResults( const Array1D<QuadratureResult>& results )
{
  const auto status = detail::reduceQuadratureStatus<Exec>( results.view() );
  if ( status != QuadratureStatus::Success ) {
    reportQuadratureFailure( status );
  }
}

/**
 * @brief Provide read-only access to quadrature points and weights in the integration surface's coordinates.
 *
 * The mapping defines the coordinates and any conversion of weights to physical integration weights. Arrays are
 * indexed by point, then coordinate index. The caller's arrays must remain alive while this view is used.
 *
 * @tparam CoordinateDim Number of coordinates stored for each integration-surface quadrature point.
 */
template <int CoordinateDim>
struct IntegrationSurfaceRuleView {
  static_assert( CoordinateDim == 1 || CoordinateDim == 2 );
  ArrayViewT<const RealT, 2> coords;  ///< Integration-surface coordinates [point][parametric_dim].
  ArrayViewT<const RealT> weights;    ///< Weights in integration-surface coordinates.

  /**
   * @brief Return the number of integration-surface quadrature points.
   *
   * @return Length of weights.
   */
  TRIBOL_HOST_DEVICE IndexT numberOfPoints() const { return weights.size(); }

  /**
   * @brief View matching coordinate and weight arrays without copying their entries.
   *
   * Both arrays must have the same point count and remain alive while the view is used.
   *
   * @param coords Integration-surface coordinate array [point][CoordinateDim].
   * @param weights Weights in the integration surface's coordinates, one per point.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurfaceRuleView( ArrayViewT<const RealT, 2> coords,
                                                                      ArrayViewT<const RealT> weights )
      : coords( coords ), weights( weights )
  {
    assert( coords.shape()[0] == weights.size() && coords.shape()[1] == CoordinateDim );
  }
};

/**
 * @brief Own fixed-capacity integration-surface quadrature points and weights.
 *
 * Views refer to this object's arrays and must not outlive it.
 *
 * @tparam CoordinateDim Number of coordinates stored for each integration-surface quadrature point.
 * @tparam Capacity Number of points available in the fixed-size backing arrays.
 */
template <int CoordinateDim, IndexT Capacity>
struct IntegrationSurfaceRule {
  MdArray<RealT, Capacity, CoordinateDim> coords{};  ///< Integration-surface coordinates [point][parametric_dim].
  MdArray<RealT, Capacity> weights{};                ///< Owned weights in integration-surface coordinates.
  QuadratureResult result{};                         ///< Completion status and number of populated points.

  /**
   * @brief Return a view of the completed integration-surface quadrature points and weights.
   *
   * @pre result.status is Success and num_required_points does not exceed Capacity.
   *
   * @return Read-only views of the initialized points and weights, valid while this object remains alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurfaceRuleView<CoordinateDim> getView() const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.num_required_points >= 0 && result.num_required_points <= Capacity );
    const axom::StackArray<axom::IndexType, 2> shape{ result.num_required_points, CoordinateDim };
    return { ArrayViewT<const RealT, 2>{ coords.data(), shape },
             ArrayViewT<const RealT>{ weights.data(), result.num_required_points } };
  }

  /**
   * @brief Prevent returning a view of arrays in a temporary rule.
   */
  IntegrationSurfaceRuleView<CoordinateDim> getView() const&& = delete;
};

/**
 * @brief Provide views of quadrature points in each element's parametric coordinates and physical integration weights.
 *
 * Parametric quadrature-point coordinates locate each point within an element. Contact calculations use them to find
 * the point's physical position and evaluate the element basis there.
 * LinearEdgeGeometry uses one coordinate, xi, with vertices at -1 and 1. Arrays are indexed by point, then parametric
 * coordinate
 * index. Mapping to the second element does not add another Jacobian. The caller's arrays must remain alive while
 * this view is used.
 * @tparam ParametricDim1 Number of coordinates used to parameterize geometry1: one for an edge, two for a face.
 * @tparam ParametricDim2 Number of coordinates used to parameterize geometry2: one for an edge, two for a face.
 * @tparam ReadOnly Whether array entries are read-only.
 */
template <int ParametricDim1, int ParametricDim2, bool ReadOnly = false>
struct PairQuadratureRuleView {
  using Value = std::conditional_t<ReadOnly, const RealT, RealT>;  ///< RealT with the requested write access.
  ArrayViewT<Value, 2>
      geom1_parametric_coords;  ///< Quadrature-point coordinates in geometry1's element parameterization.
  ArrayViewT<Value, 2>
      geom2_parametric_coords;  ///< Quadrature-point coordinates in geometry2's element parameterization.
  ArrayViewT<Value> weights;    ///< Weights including the mapping's physical integration measure.

  /** @brief Return the available point capacity. @return Number of weight entries. */
  TRIBOL_HOST_DEVICE IndexT capacity() const { return weights.size(); }

  /**
   * @brief View matching coordinate and weight arrays without copying their entries.
   * @param geometry1_points Quadrature-point coordinates in geometry1's element parameterization.
   * @param geometry2_points Quadrature-point coordinates in geometry2's element parameterization.
   * @param weights Physical integration weights.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureRuleView( ArrayViewT<Value, 2> geometry1_points,
                                                                  ArrayViewT<Value, 2> geometry2_points,
                                                                  ArrayViewT<Value> weights )
      : geom1_parametric_coords( geometry1_points ), geom2_parametric_coords( geometry2_points ), weights( weights )
  {
    assert( geometry1_points.shape()[0] == weights.size() && geometry1_points.shape()[1] == ParametricDim1 );
    assert( geometry2_points.shape()[0] == weights.size() && geometry2_points.shape()[1] == ParametricDim2 );
  }
};

/**
 * @brief Own fixed-capacity quadrature points in each element's parametric coordinates and physical integration
 * weights.
 *
 * Parametric quadrature-point coordinates locate each point within an element. Contact calculations use them to find
 * the point's physical position and evaluate the element basis there.
 * LinearEdgeGeometry uses one coordinate, xi, with vertices at -1 and 1.
 * @tparam ParametricDim1 Number of coordinates used to parameterize geometry1: one for an edge, two for a face.
 * @tparam ParametricDim2 Number of coordinates used to parameterize geometry2: one for an edge, two for a face.
 * @tparam Capacity Available points in the local backing arrays.
 */
template <int ParametricDim1, int ParametricDim2, IndexT Capacity>
struct PairQuadratureRule {
  MdArray<RealT, Capacity, ParametricDim1>
      geom1_parametric_coords{};  ///< Quadrature-point coordinates in geometry1's element parameterization.
  MdArray<RealT, Capacity, ParametricDim2>
      geom2_parametric_coords{};       ///< Quadrature-point coordinates in geometry2's element parameterization.
  MdArray<RealT, Capacity> weights{};  ///< Owned physical integration weights.
  QuadratureResult result{};           ///< Status and initialized point count.

  /** @brief Return views of all available output storage. @return Writable views valid while this object is alive. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureRuleView<ParametricDim1, ParametricDim2> getOutputView() &
  {
    return { { geom1_parametric_coords.data(), { Capacity, ParametricDim1 } },
             { geom2_parametric_coords.data(), { Capacity, ParametricDim2 } },
             { weights.data(), Capacity } };
  }

  /**
   * @brief Return read-only views of the initialized quadrature coordinates and weights.
   * @pre result.status is Success and num_required_points does not exceed Capacity.
   * @return Read-only arrays valid while this owner is alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureRuleView<ParametricDim1, ParametricDim2, true> getView() const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.num_required_points >= 0 && result.num_required_points <= Capacity );
    const IndexT num_points = result.num_required_points;
    return { { geom1_parametric_coords.data(), { num_points, ParametricDim1 } },
             { geom2_parametric_coords.data(), { num_points, ParametricDim2 } },
             { weights.data(), num_points } };
  }

  /** @brief Prevent returning output views from a temporary rule. */
  PairQuadratureRuleView<ParametricDim1, ParametricDim2> getOutputView() && = delete;
  /** @brief Prevent returning initialized point views from a temporary rule. */
  PairQuadratureRuleView<ParametricDim1, ParametricDim2, true> getView() const&& = delete;
};

}  // namespace tribol

#endif
