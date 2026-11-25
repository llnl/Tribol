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

/**
 * @brief Convex polygon in a two-dimensional integration surface's coordinates.
 *
 * Vertices are stored in counter-clockwise order in common-plane (e1, e2) coordinates. Clipping two quads and one
 * zero-gap half-plane produces at most 9 vertices; capacity 10 leaves margin for intermediate vertices.
 */
struct PolygonQuadratureSurface {
  static constexpr IndexT max_num_vertices = 10;   ///< Maximum supported polygon vertex count.
  MdArray<RealT, max_num_vertices, 2> vertices{};  ///< CCW polygon vertices [vertex][coord].
  IndexT num_vertices{ 0 };                        ///< Number of populated vertices (0 or >= 3 when valid).
  bool valid{ true };                              ///< False for degenerate or invalid input geometry.
};

/** @brief Whether a complete rule was generated, needs more storage, or has invalid geometry */
enum class QuadratureStatus
{
  Success,               ///< Every reported point is initialized; zero points means zero measure.
  InsufficientCapacity,  ///< Reported count exceeds capacity; discard partial output.
  InvalidGeometry        ///< Mapping rejected the supplied element geometry; discard output and report zero points.
};

/**
 * @brief Report a quadrature failure with a SLIC error outside differentiated calculations.
 * @param status Failed generation status; must not be Success.
 * @note Reports a SLIC error if the rule requires more point storage.
 * @note Reports a SLIC error if pair geometry is invalid.
 */
[[gnu::noinline]]
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
struct QuadratureMetadata {
  QuadratureStatus status{ QuadratureStatus::Success };  ///< Whether the complete rule can be consumed.
  IndexT num_required_points{ 0 };  ///< Complete count on success or overflow; zero for invalid geometry.
};

namespace detail {

/**
 * @brief Find the most severe quadrature error across all pairs using the specified execution policy.
 * @tparam Exec Concrete execution policy.
 * @param metadata One metadata entry per pair in execution-accessible memory.
 * @return Highest-severity status present in the array, or Success when the array is empty.
 */
template <ExecutionMode Exec>
QuadratureStatus reduceQuadratureStatus( ArrayViewT<const QuadratureMetadata> metadata )
{
  ReduceMax<Exec, int> status( static_cast<int>( QuadratureStatus::Success ) );
  forAllExec<Exec>( metadata.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
    status.max( static_cast<int>( metadata[pair].status ) );
  } );
  return static_cast<QuadratureStatus>( status.get() );
}

}  // namespace detail

/**
 * @brief Check the quadrature metadata produced for all pairs and report a SLIC error for any failed rule.
 * @tparam Exec Concrete execution mode used to populate the metadata.
 * @param metadata One preallocated metadata entry per pair.
 * Empty metadata passes the check.
 * @note Reports a SLIC error if a rule reports invalid geometry.
 * @note Reports a SLIC error if a rule reports insufficient quadrature capacity.
 */
template <ExecutionMode Exec>
void checkPairQuadratureMetadata( const Array1D<QuadratureMetadata>& metadata )
{
  const auto status = detail::reduceQuadratureStatus<Exec>( metadata.view() );
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
  QuadratureMetadata metadata{};                     ///< Completion status and number of populated points.

  /**
   * @brief Return a view of the completed integration-surface quadrature points and weights.
   *
   * @pre metadata.status is Success and num_required_points does not exceed Capacity.
   *
   * @return Read-only views of the initialized points and weights, valid while this object remains alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE IntegrationSurfaceRuleView<CoordinateDim> getView() const&
  {
    assert( metadata.status == QuadratureStatus::Success );
    assert( metadata.num_required_points >= 0 && metadata.num_required_points <= Capacity );
    const axom::StackArray<axom::IndexType, 2> shape{ metadata.num_required_points, CoordinateDim };
    return { ArrayViewT<const RealT, 2>{ coords.data(), shape },
             ArrayViewT<const RealT>{ weights.data(), metadata.num_required_points } };
  }

  /**
   * @brief Prevent returning a view of arrays in a temporary rule.
   */
  IntegrationSurfaceRuleView<CoordinateDim> getView() const&& = delete;
};

/**
 * @brief Provide read-only views of quadrature points in each element's parametric coordinates, physical integration
 * weights, and integration-surface normals.
 *
 * Parametric quadrature-point coordinates locate each point within an element. Contact calculations use them to find
 * the point's physical position and evaluate the element basis there.
 * LinearEdgeGeometry uses one coordinate, xi, with vertices at -1 and 1. Arrays are indexed by point, then parametric
 * coordinate index. Mapping to the second element does not add another Jacobian. Surface elements have codimension one,
 * so each normal has ParametricDim1 + 1 spatial components. The viewed rule must remain alive while this view is used.
 * @tparam ParametricDim1 Number of coordinates used to parameterize geometry1: one for an edge, two for a face.
 * @tparam ParametricDim2 Number of coordinates used to parameterize geometry2: one for an edge, two for a face.
 */
template <int ParametricDim1, int ParametricDim2>
struct PairQuadratureRuleView {
  static_assert( ParametricDim1 == ParametricDim2 );
  using Value = const RealT;                              ///< Read-only entry type.
  static constexpr int spatial_dim = ParametricDim1 + 1;  ///< Spatial components of each normal.
  ArrayViewT<Value, 2>
      geom1_parametric_coords;  ///< Quadrature-point coordinates in geometry1's element parameterization.
  ArrayViewT<Value, 2>
      geom2_parametric_coords;  ///< Quadrature-point coordinates in geometry2's element parameterization.
  ArrayViewT<Value> weights;    ///< Weights including the mapping's physical integration measure.
  /// Unit normal of the integration surface at each point, along which the mapping paired the two element points; zero
  /// when the mapping does not pair points along a direction.
  ArrayViewT<Value, 2> integration_surface_normals;

  /**
   * @brief View matching coordinate and weight arrays without copying their entries.
   * @param geometry1_points Quadrature-point coordinates in geometry1's element parameterization.
   * @param geometry2_points Quadrature-point coordinates in geometry2's element parameterization.
   * @param weights Physical integration weights.
   * @param normals Integration-surface unit normals [point][spatial_dim].
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureRuleView( ArrayViewT<Value, 2> geometry1_points,
                                                                  ArrayViewT<Value, 2> geometry2_points,
                                                                  ArrayViewT<Value> weights,
                                                                  ArrayViewT<Value, 2> normals )
      : geom1_parametric_coords( geometry1_points ),
        geom2_parametric_coords( geometry2_points ),
        weights( weights ),
        integration_surface_normals( normals )
  {
    assert( geometry1_points.shape()[0] == weights.size() && geometry1_points.shape()[1] == ParametricDim1 );
    assert( geometry2_points.shape()[0] == weights.size() && geometry2_points.shape()[1] == ParametricDim2 );
    assert( normals.shape()[0] == weights.size() && normals.shape()[1] == spatial_dim );
  }
};

/**
 * @brief Own fixed-capacity quadrature points in each element's parametric coordinates, physical integration weights,
 * and integration-surface normals.
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
  static_assert( ParametricDim1 == ParametricDim2 );
  static constexpr int spatial_dim = ParametricDim1 + 1;  ///< Spatial components of each normal.
  MdArray<RealT, Capacity, ParametricDim1>
      geom1_parametric_coords{};  ///< Quadrature-point coordinates in geometry1's element parameterization.
  MdArray<RealT, Capacity, ParametricDim2>
      geom2_parametric_coords{};  ///< Quadrature-point coordinates in geometry2's element parameterization.
  /// Unit normal of the integration surface at each point, along which the mapping paired the two element points; zero
  /// when the mapping does not pair points along a direction. Declared before the weights so that the weights, not the
  /// strided normals, precede the integer metadata; Enzyme type analysis otherwise extends the normals' floating-point
  /// pattern into the metadata.
  MdArray<RealT, Capacity, spatial_dim> integration_surface_normals{};
  MdArray<RealT, Capacity> weights{};  ///< Owned physical integration weights.
  QuadratureMetadata metadata{};       ///< Status and initialized point count.

  /**
   * @brief Return read-only views of the initialized quadrature coordinates and weights.
   * @pre metadata.status is Success and num_required_points does not exceed Capacity.
   * @return Read-only arrays valid while this owner is alive.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureRuleView<ParametricDim1, ParametricDim2> getView() const&
  {
    assert( metadata.status == QuadratureStatus::Success );
    assert( metadata.num_required_points >= 0 && metadata.num_required_points <= Capacity );
    const IndexT num_points = metadata.num_required_points;
    return { { geom1_parametric_coords.data(), { num_points, ParametricDim1 } },
             { geom2_parametric_coords.data(), { num_points, ParametricDim2 } },
             { weights.data(), num_points },
             { integration_surface_normals.data(), { num_points, spatial_dim } } };
  }

  /** @brief Prevent returning initialized point views from a temporary rule. */
  PairQuadratureRuleView<ParametricDim1, ParametricDim2> getView() const&& = delete;
};

}  // namespace tribol

#endif
