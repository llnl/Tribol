// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_QUADRATURE_HPP_
#define SRC_TRIBOL_INTEG_QUADRATURE_HPP_

#include <cassert>

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"

namespace tribol {

/** @brief Whether a complete rule was generated, needs more storage, or has invalid geometry */
enum class QuadratureStatus
{
  Success,
  InsufficientCapacity,
  InvalidGeometry
};

/**
 * @brief Reports the complete point count, even when the supplied storage is too small
 *
 * Success initializes every point below required_point_count; success with zero points means zero integration
 * measure. InsufficientCapacity may leave partial output, which must be discarded. InvalidGeometry
 * returns zero points and its output must also be discarded. Unused capacity is never part of a completed rule.
 */
struct QuadratureResult {
  QuadratureStatus status{ QuadratureStatus::Success };
  IndexT required_point_count{ 0 };
};

/** @brief Borrows integration-side parent points and weights before applying the physical Jacobian */
template <typename Value, int ParentDimension>
struct ReferenceQuadratureView {
  static_assert( ParentDimension == 1 || ParentDimension == 2 );
  ArrayViewT<Value, 2> parent_coordinates;
  ArrayViewT<Value> reference_weights;

  TRIBOL_HOST_DEVICE IndexT numberOfPoints() const { return reference_weights.size(); }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ReferenceQuadratureView( ArrayViewT<Value, 2> coordinates,
                                                                   ArrayViewT<Value> weights )
      : parent_coordinates( coordinates ), reference_weights( weights )
  {
    assert( coordinates.shape()[0] == weights.size() && coordinates.shape()[1] == ParentDimension );
  }
};

/** @brief Owns parent points and reference weights that may be frozen while physical geometry changes */
template <typename Value, int ParentDimension, IndexT Capacity>
struct ReferenceQuadratureData {
  MdArray<Value, Capacity, ParentDimension> parent_coordinates{};
  MdArray<Value, Capacity> reference_weights{};
  QuadratureResult result{};

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE ReferenceQuadratureView<const Value, ParentDimension> getView() const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.required_point_count >= 0 && result.required_point_count <= Capacity );
    const axom::StackArray<axom::IndexType, 2> shape{ result.required_point_count, ParentDimension };
    return { ArrayViewT<const Value, 2>{ parent_coordinates.data(), shape },
             ArrayViewT<const Value>{ reference_weights.data(), result.required_point_count } };
  }

  ReferenceQuadratureView<const Value, ParentDimension> getView() const&& = delete;
};

/**
 * @brief Borrows quadrature on one source surface, before projection onto any target surface
 *
 * Parent and physical coordinates are [point][component]. Physical weights include only the source surface measure.
 * All arrays belong to the caller. A projection must keep these weights rather than apply a target Jacobian.
 */
template <typename Value, int ParentDimension, int SpatialDimension>
struct SurfaceQuadratureView {
  ArrayViewT<Value, 2> parent_coordinates;
  ArrayViewT<Value, 2> physical_coordinates;
  ArrayViewT<Value> physical_weights;

  TRIBOL_HOST_DEVICE IndexT capacity() const { return physical_weights.size(); }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE SurfaceQuadratureView( ArrayViewT<Value, 2> parent,
                                                                 ArrayViewT<Value, 2> physical,
                                                                 ArrayViewT<Value> weights )
      : parent_coordinates( parent ), physical_coordinates( physical ), physical_weights( weights )
  {
    assert( parent.shape()[0] == weights.size() && physical.shape()[0] == weights.size() );
    assert( parent.shape()[1] == ParentDimension && physical.shape()[1] == SpatialDimension );
  }
};

/** @brief Owns fixed-capacity source quadrature; views remain valid only while this object's arrays remain alive */
template <typename Value, int ParentDimension, int SpatialDimension, IndexT Capacity>
struct SurfaceQuadratureData {
  MdArray<Value, Capacity, ParentDimension> parent_coordinates{};
  MdArray<Value, Capacity, SpatialDimension> physical_coordinates{};
  MdArray<Value, Capacity> physical_weights{};
  QuadratureResult result{};

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE SurfaceQuadratureView<Value, ParentDimension, SpatialDimension>
  getOutputView() &
  {
    return { ArrayViewT<Value, 2>{ parent_coordinates.data(),
                                   axom::StackArray<axom::IndexType, 2>{ Capacity, ParentDimension } },
             ArrayViewT<Value, 2>{ physical_coordinates.data(),
                                   axom::StackArray<axom::IndexType, 2>{ Capacity, SpatialDimension } },
             ArrayViewT<Value>{ physical_weights.data(), Capacity } };
  }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE SurfaceQuadratureView<const Value, ParentDimension, SpatialDimension>
  getView() const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.required_point_count >= 0 && result.required_point_count <= Capacity );
    const IndexT count = result.required_point_count;
    return { ArrayViewT<const Value, 2>{ parent_coordinates.data(),
                                         axom::StackArray<axom::IndexType, 2>{ count, ParentDimension } },
             ArrayViewT<const Value, 2>{ physical_coordinates.data(),
                                         axom::StackArray<axom::IndexType, 2>{ count, SpatialDimension } },
             ArrayViewT<const Value>{ physical_weights.data(), count } };
  }

  SurfaceQuadratureView<Value, ParentDimension, SpatialDimension> getOutputView() && = delete;
  SurfaceQuadratureView<const Value, ParentDimension, SpatialDimension> getView() const&& = delete;
};

}  // namespace tribol

#endif
