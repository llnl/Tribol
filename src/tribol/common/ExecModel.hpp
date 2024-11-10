// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)
 
#ifndef SRC_COMMON_EXECMODEL_HPP_
#define SRC_COMMON_EXECMODEL_HPP_

// Tribol includes
#include "tribol/common/BasicTypes.hpp"

// Axom includes
#include "axom/core/memory_management.hpp"
#include "axom/slic.hpp"

namespace tribol
{

/**
 * @brief A MemorySpace ties a resource to an associated pointer
 */
enum class MemorySpace
{
  // Dynamic can be used for pointers whose resource is defined not using this
  // enum.
  Dynamic,
  Host,
#ifdef TRIBOL_USE_UMPIRE
  Device,
  Unified
#endif
};

/**
 * @brief An ExecutionMode defines what resource should be used to do loop
 * computations
 */
enum class ExecutionMode
{
  Sequential,
#ifdef TRIBOL_USE_OPENMP
  OpenMP,
#endif
#ifdef TRIBOL_USE_CUDA
  Cuda,
#endif
#ifdef TRIBOL_USE_HIP
  Hip,
#endif
  // Dynamic is used to determine the ExecutionMode on the fly.
  Dynamic
};

/**
 * @brief SFINAE struct to deduce axom memory space from a Tribol memory space
 * at compile time
 * 
 * @tparam MSPACE 
 */
template <MemorySpace MSPACE>
struct toAxomMemorySpace
{
  static constexpr axom::MemorySpace value = axom::MemorySpace::Dynamic;
};

#ifdef TRIBOL_USE_UMPIRE

template <>
struct toAxomMemorySpace<MemorySpace::Host>
{
  static constexpr axom::MemorySpace value = axom::MemorySpace::Host;
};

template <>
struct toAxomMemorySpace<MemorySpace::Device>
{
  static constexpr axom::MemorySpace value = axom::MemorySpace::Device;
};

template <>
struct toAxomMemorySpace<MemorySpace::Unified>
{
  static constexpr axom::MemorySpace value = axom::MemorySpace::Unified;
};

#endif

// Waiting for C++ 17...
// template <MemorySpace MSPACE>
// inline constexpr axom::MemorySpace axomMemorySpaceV = toAxomMemorySpace<MSPACE>::value;

#ifdef TRIBOL_USE_UMPIRE

inline umpire::resource::MemoryResourceType toUmpireMemoryType(MemorySpace mem_space)
{
  switch (mem_space)
  {
    case MemorySpace::Host:
      return umpire::resource::MemoryResourceType::Host;
    case MemorySpace::Device:
      return umpire::resource::MemoryResourceType::Device;
    case MemorySpace::Unified:
      return umpire::resource::MemoryResourceType::Unified;
    default:
      return umpire::resource::MemoryResourceType::Unknown;
  }
};

#endif

inline int getResourceAllocatorID(MemorySpace mem_space)
{
  int allocator_id = axom::getDefaultAllocatorID();
#ifdef TRIBOL_USE_UMPIRE
  if (mem_space != MemorySpace::Dynamic)
  {
    allocator_id = axom::getUmpireResourceAllocatorID(toUmpireMemoryType(mem_space));
  }
#else
  TRIBOL_UNUSED_VAR(mem_space);
#endif
  return allocator_id;
}

inline bool isOnDevice(ExecutionMode exec)
{
  switch (exec)
  {
#if defined(TRIBOL_USE_CUDA)
    case ExecutionMode::Cuda:
      return true;
#elif defined(TRIBOL_USE_HIP)
    case ExecutionMode::Hip:
      return true;
#endif
#ifdef TRIBOL_USE_OPENMP
    case ExecutionMode::OpenMP:
      return false;
#endif
    case ExecutionMode::Dynamic:
      SLIC_ERROR_ROOT("Dynamic execution mode does not define a memory space location.");
      return false;
    case ExecutionMode::Sequential:
      return false;
    default:
      SLIC_ERROR_ROOT("Unknown execution mode.");
      return false;
  }
}

} // namespace tribol


#endif /* SRC_COMMON_EXECMODEL_HPP_ */
