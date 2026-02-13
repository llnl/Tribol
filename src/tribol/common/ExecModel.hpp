// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_EXECMODEL_HPP_
#define SRC_TRIBOL_COMMON_EXECMODEL_HPP_

#include "shared/common/ExecModel.hpp"

namespace tribol {

using MemorySpace = shared::MemorySpace;
using ExecutionMode = shared::ExecutionMode;

template <MemorySpace MSPACE>
using toAxomMemorySpace = shared::toAxomMemorySpace<MSPACE>;

#ifdef TRIBOL_USE_UMPIRE
using shared::toUmpireMemoryType;
#endif

using shared::getDefaultAllocatorID;
using shared::getResourceAllocatorID;
using shared::isOnDevice;

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_EXECMODEL_HPP_ */
