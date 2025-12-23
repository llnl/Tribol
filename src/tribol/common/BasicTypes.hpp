// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_BASICTYPES_HPP_
#define SRC_TRIBOL_COMMON_BASICTYPES_HPP_

// Tribol config include
#include "tribol/config.hpp"

// C includes
#include <cstddef>

// C++ includes
#include <type_traits>

// MPI includes
#ifdef TRIBOL_USE_MPI
#include <mpi.h>
#endif

// Axom includes
#include "axom/core/Types.hpp"

// MFEM includes
#include "mfem.hpp"

namespace tribol {

#ifdef TRIBOL_USE_MPI

using CommT = MPI_Comm;
#define TRIBOL_COMM_WORLD MPI_COMM_WORLD
#define TRIBOL_COMM_NULL MPI_COMM_NULL

#else

using CommT = int;
#define TRIBOL_COMM_WORLD 0
#define TRIBOL_COMM_NULL -1

#endif

// match index type used in axom (since data is held in axom data structures)
using IndexT = axom::IndexType;

// size type matching size of addressable memory
using SizeT = size_t;

#ifdef TRIBOL_USE_SINGLE_PRECISION

#error "Tribol does not support single precision."
using RealT = float;

#else

using RealT = double;

#endif

// mfem's real_t should match ours
static_assert( std::is_same_v<RealT, mfem::real_t>, "tribol::RealT and mfem::real_t are required to match" );

#define TRIBOL_UNUSED_VAR AXOM_UNUSED_VAR
#define TRIBOL_UNUSED_PARAM AXOM_UNUSED_PARAM

// Execution space specifiers
#if defined( TRIBOL_USE_CUDA ) || defined( TRIBOL_USE_HIP )
#ifndef __device__
#error "TRIBOL_USE_CUDA or TRIBOL_USE_HIP but __device__ is undefined.  Check include files"
#endif
#define TRIBOL_DEVICE __device__
#define TRIBOL_HOST_DEVICE __host__ __device__
#else
#define TRIBOL_DEVICE
#define TRIBOL_HOST_DEVICE
#endif

// Execution space identifier for defaulted constructors and destructors
#ifdef TRIBOL_USE_HIP
#define TRIBOL_DEFAULT_DEVICE __device__
#define TRIBOL_DEFAULT_HOST_DEVICE __host__ __device__
#else
#define TRIBOL_DEFAULT_DEVICE
#define TRIBOL_DEFAULT_HOST_DEVICE
#endif

// Defined when Tribol doesn't have a device available
#if !( defined( TRIBOL_USE_CUDA ) || defined( TRIBOL_USE_HIP ) )
#define TRIBOL_USE_HOST
#endif

// Define variable when in device code
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
#define TRIBOL_DEVICE_CODE
#endif

// Ignore host code in __host__ __device__ code warning on NVCC
#ifdef TRIBOL_USE_CUDA
#define TRIBOL_NVCC_EXEC_CHECK_DISABLE #pragma nv_exec_check_disable
#else
#define TRIBOL_NVCC_EXEC_CHECK_DISABLE
#endif

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_BASICTYPES_HPP_ */
