// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_ATOMICS_HPP_
#define SRC_TRIBOL_COMMON_ATOMICS_HPP_

#include "tribol/common/ExecModel.hpp"
#include "axom/core.hpp"

#ifdef TRIBOL_USE_RAJA
#include "RAJA/RAJA.hpp"
#endif

namespace tribol {

template <typename T, typename U>
TRIBOL_HOST_DEVICE inline T atomicAdd(T* addr, U val) {
#ifdef TRIBOL_USE_RAJA
    return RAJA::atomicAdd<RAJA::auto_atomic>(addr, static_cast<T>(val));
#else
    T old = *addr;
    *addr += static_cast<T>(val);
    return old;
#endif
}

template <typename T, typename U>
TRIBOL_HOST_DEVICE inline T atomicMin(T* addr, U val) {
#ifdef TRIBOL_USE_RAJA
    return RAJA::atomicMin<RAJA::auto_atomic>(addr, static_cast<T>(val));
#else
    T old = *addr;
    *addr = axom::utilities::min(*addr, static_cast<T>(val));
    return old;
#endif
}

template <typename T, typename U>
TRIBOL_HOST_DEVICE inline T atomicMax(T* addr, U val) {
#ifdef TRIBOL_USE_RAJA
    return RAJA::atomicMax<RAJA::auto_atomic>(addr, static_cast<T>(val));
#else
    T old = *addr;
    *addr = axom::utilities::max(*addr, static_cast<T>(val));
    return old;
#endif
}

template <typename T>
TRIBOL_HOST_DEVICE inline T atomicInc(T* addr) {
#ifdef TRIBOL_USE_RAJA
    return RAJA::atomicInc<RAJA::auto_atomic>(addr);
#else
    T old = *addr;
    (*addr)++;
    return old;
#endif
}

} // namespace tribol

#endif /* SRC_TRIBOL_COMMON_ATOMICS_HPP_ */
