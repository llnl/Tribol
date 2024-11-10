// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_COMMON_ARRAYTYPES_HPP_
#define TRIBOL_COMMON_ARRAYTYPES_HPP_

// Tribol includes
#include "tribol/common/ExecModel.hpp"

#include "axom/core/Array.hpp"
#include "axom/core/ArrayView.hpp"

namespace tribol
{

/**
 * @brief Array allocated on the free store, i.e. the heap
 */
template <typename T, int DIM = 1, MemorySpace SPACE = MemorySpace::Dynamic>
using ArrayT = axom::Array<T, DIM, toAxomMemorySpace<SPACE>::value>;

/**
 * @brief View of (i.e. non-owned) array allocated on the free store
 */
template <typename T, int DIM = 1, MemorySpace SPACE = MemorySpace::Dynamic>
using ArrayViewT = axom::ArrayView<T, DIM, toAxomMemorySpace<SPACE>::value>;

/**
 * @brief Array allocated on automatic storage, i.e. the stack
 */
template <typename T, int N>
using StackArrayT = axom::StackArray<T, N>;

/**
 * @brief Convenience definition for a one-dimensional array
 */
template <typename T, MemorySpace SPACE = MemorySpace::Dynamic>
using Array1D = ArrayT<T, 1, SPACE>;

/**
 * @brief Convenience definition for a two-dimensional array
 */
template <typename T, MemorySpace SPACE = MemorySpace::Dynamic>
using Array2D = ArrayT<T, 2, SPACE>;

/**
 * @brief Convenience definition for a one-dimensional array view
 */
template <typename T, MemorySpace SPACE = MemorySpace::Dynamic>
using Array1DView = ArrayViewT<T, 1, SPACE>;

/**
 * @brief Convenience definition for a two-dimensional array view
 */
template <typename T, MemorySpace SPACE = MemorySpace::Dynamic>
using Array2DView = ArrayViewT<T, 2, SPACE>;

/**
 * @brief Convenience definition for an array of array views
 */
template <typename T, MemorySpace SPACE = MemorySpace::Dynamic>
using MultiArrayView = ArrayT<ArrayViewT<T, 1, SPACE>, 1, SPACE>;

/**
 * @brief Convenience definition for an array view of array views
 */
template <typename T, MemorySpace SPACE = MemorySpace::Dynamic>
using MultiViewArrayView = ArrayViewT<const ArrayViewT<T, 1, SPACE>, 1, SPACE>;

} // namespace tribol

#endif /* TRIBOL_COMMON_ARRAYTYPES_HPP_ */
