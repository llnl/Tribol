// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_ENZYME_HPP_
#define SRC_TRIBOL_COMMON_ENZYME_HPP_

// Tribol config include
#include "tribol/config.hpp"

// Shared includes
#include "tribol/common/BasicTypes.hpp"

#ifdef TRIBOL_USE_ENZYME
#ifdef MFEM_USE_ENZYME
#include "mfem/general/enzyme.hpp"
#else  // MFEM_USE_ENZYME
// NOTE: contents copied from MFEM's general/enzyme.hpp

/*
 * Variables prefixed with enzyme_* or function types prefixed with __enzyme_*,
 * are variables which will get preprocessed in the LLVM intermediate
 * representation when the Enzyme LLVM plugin is loaded. See the Enzyme
 * documentation (https://enzyme.mit.edu) for more information.
 */

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;
extern int enzyme_interleave;

#if defined( MFEM_USE_CUDA ) || defined( MFEM_USE_HIP )
#define MFEM_DEVICE_EXTERN_STMT( name ) extern __device__ int name;
#else
#define MFEM_DEVICE_EXTERN_STMT( name )
#endif

MFEM_DEVICE_EXTERN_STMT( enzyme_dup )
MFEM_DEVICE_EXTERN_STMT( enzyme_dupnoneed )
MFEM_DEVICE_EXTERN_STMT( enzyme_out )
MFEM_DEVICE_EXTERN_STMT( enzyme_const )
MFEM_DEVICE_EXTERN_STMT( enzyme_interleave )

// warning: if inlined, triggers function '__enzyme_autodiff' is not defined
template <typename return_type, typename... Args>
MFEM_HOST_DEVICE return_type __enzyme_autodiff( Args... );

// warning: if inlined, triggers function '__enzyme_fwddiff' is not defined
template <typename return_type, typename... Args>
MFEM_HOST_DEVICE return_type __enzyme_fwddiff( Args... );

#define MFEM_ENZYME_INACTIVENOFREE __attribute__( ( enzyme_inactive, enzyme_nofree ) )
#define MFEM_ENZYME_INACTIVE __attribute__( ( enzyme_inactive ) )
#define MFEM_ENZYME_FN_LIKE( x ) __attribute__( ( enzyme_function_like( #x ) ) )

#endif  // MFEM_USE_ENZYME

#if !defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_DEVICE_CODE )
// When compiling with NVCC or HIPCC, the compiler performs multiple passes.
// In the HOST pass, reading the __device__ or __constant__ globals (like
// enzyme_const) triggers warnings.
//
// To fix this, we declare host-side aliases that map to the same
// underlying assembly symbols required by the Enzyme pass, but without
// the device attributes.
extern "C" {
extern int tribol_host_enzyme_const asm( "enzyme_const" );
extern int tribol_host_enzyme_dup asm( "enzyme_dup" );
extern int tribol_host_enzyme_out asm( "enzyme_out" );
extern int tribol_host_enzyme_dupnoneed asm( "enzyme_dupnoneed" );
}

#define TRIBOL_ENZYME_CONST tribol_host_enzyme_const
#define TRIBOL_ENZYME_DUP tribol_host_enzyme_dup
#define TRIBOL_ENZYME_OUT tribol_host_enzyme_out
#define TRIBOL_ENZYME_DUPNONEED tribol_host_enzyme_dupnoneed

#else  // !defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_DEVICE_CODE )
// We are either:
// 1. In a device compilation pass (__CUDA_ARCH__ or __HIP_DEVICE_COMPILE__ defined).
// 2. Using a standard host compiler (GCC, Clang, etc.).
// In these cases, we use the symbols provided by the Enzyme/MFEM headers.
#define TRIBOL_ENZYME_CONST enzyme_const
#define TRIBOL_ENZYME_DUP enzyme_dup
#define TRIBOL_ENZYME_OUT enzyme_out
#define TRIBOL_ENZYME_DUPNONEED enzyme_dupnoneed
#endif  // !defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_DEVICE_CODE )

#else  // TRIBOL_USE_ENZYME
// Fallback definitions if Enzyme is disabled
#define TRIBOL_ENZYME_CONST 0
#define TRIBOL_ENZYME_DUP 0
#define TRIBOL_ENZYME_OUT 0
#define TRIBOL_ENZYME_DUPNONEED 0
#endif  // TRIBOL_USE_ENZYME

#endif /* SRC_TRIBOL_COMMON_ENZYME_HPP_ */
