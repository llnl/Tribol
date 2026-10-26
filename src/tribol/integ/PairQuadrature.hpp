// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_PAIRQUADRATURE_HPP_
#define SRC_TRIBOL_INTEG_PAIRQUADRATURE_HPP_

#include <cassert>

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/integ/Quadrature.hpp"
#include "tribol/common/MdArray.hpp"

namespace tribol {

/**
 * @brief Borrows runtime-capacity output storage for a pair rule with a fixed parent dimension
 *
 * Coordinates are [point][parent component]. Weights contain the full physical integration measure on mesh1.
 * The caller owns all arrays and must keep them alive throughout generation. Capacity is independent of the number
 * of points per base rule and the number of base rules contributing to this pair.
 */
template <typename Value, int ParentDimension>
struct PairQuadratureView {
  static_assert( ParentDimension == 1 || ParentDimension == 2 );

  ArrayViewT<Value, 2> mesh1_parent_coordinates;
  ArrayViewT<Value, 2> mesh2_parent_coordinates;
  ArrayViewT<Value> physical_weights;

  TRIBOL_HOST_DEVICE IndexT capacity() const { return physical_weights.size(); }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView( ArrayViewT<Value, 2> mesh1, ArrayViewT<Value, 2> mesh2,
                                                              ArrayViewT<Value> weights )
      : mesh1_parent_coordinates( mesh1 ), mesh2_parent_coordinates( mesh2 ), physical_weights( weights )
  {
    assert( mesh1.shape()[0] == weights.size() && mesh2.shape()[0] == weights.size() );
    assert( mesh1.shape()[1] == ParentDimension && mesh2.shape()[1] == ParentDimension );
  }
};

/**
 * @brief Owns fixed-capacity physical quadrature for allocation-free element kernels
 *
 * This is one possible backing store for the runtime-capacity view, not a limit on other generators. Views borrow
 * this object's arrays; copying the object copies its values without retaining pointers into the original.
 */
template <typename Value, int ParentDimension, IndexT Capacity>
struct PairQuadratureData {
  MdArray<Value, Capacity, ParentDimension> mesh1_parent_coordinates{};
  MdArray<Value, Capacity, ParentDimension> mesh2_parent_coordinates{};
  MdArray<Value, Capacity> physical_weights{};
  QuadratureResult result{};

  /** @brief Borrow all available storage for generation */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView<Value, ParentDimension> getOutputView() &
  {
    const axom::StackArray<axom::IndexType, 2> shape{ Capacity, ParentDimension };
    return { ArrayViewT<Value, 2>{ mesh1_parent_coordinates.data(), shape },
             ArrayViewT<Value, 2>{ mesh2_parent_coordinates.data(), shape },
             ArrayViewT<Value>{ physical_weights.data(), Capacity } };
  }

  /** @brief Borrow only the completed points; check result.status before consuming a generated rule */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PairQuadratureView<const Value, ParentDimension> getView() const&
  {
    assert( result.status == QuadratureStatus::Success );
    assert( result.required_point_count >= 0 && result.required_point_count <= Capacity );
    const axom::StackArray<axom::IndexType, 2> shape{ result.required_point_count, ParentDimension };
    return { ArrayViewT<const Value, 2>{ mesh1_parent_coordinates.data(), shape },
             ArrayViewT<const Value, 2>{ mesh2_parent_coordinates.data(), shape },
             ArrayViewT<const Value>{ physical_weights.data(), result.required_point_count } };
  }

  PairQuadratureView<Value, ParentDimension> getOutputView() && = delete;
  PairQuadratureView<const Value, ParentDimension> getView() const&& = delete;
};

}  // namespace tribol

#endif
