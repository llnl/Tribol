// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_ENZYME_HPP_
#define SRC_TRIBOL_COMMON_ENZYME_HPP_

#include "tribol/config.hpp"

#include "tribol/common/BasicTypes.hpp"

#ifdef TRIBOL_USE_ENZYME
#include "mfem/general/enzyme.hpp"

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

#else
// We are either:
// 1. In a device compilation pass (__CUDA_ARCH__ or __HIP_DEVICE_COMPILE__ defined).
// 2. Using a standard host compiler (GCC, Clang, etc.).
// In these cases, we use the symbols provided by the Enzyme/MFEM headers.
#define TRIBOL_ENZYME_CONST enzyme_const
#define TRIBOL_ENZYME_DUP enzyme_dup
#define TRIBOL_ENZYME_OUT enzyme_out
#define TRIBOL_ENZYME_DUPNONEED enzyme_dupnoneed
#endif

#else
// Fallback definitions if Enzyme is disabled
#define TRIBOL_ENZYME_CONST 0
#define TRIBOL_ENZYME_DUP 0
#define TRIBOL_ENZYME_OUT 0
#define TRIBOL_ENZYME_DUPNONEED 0
#endif

#endif /* SRC_TRIBOL_COMMON_ENZYME_HPP_ */