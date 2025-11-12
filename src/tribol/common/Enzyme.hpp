// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_COMMON_ENZYME_HPP_
#define TRIBOL_COMMON_ENZYME_HPP_

// Tribol config include
#include "tribol/config.hpp"

// Tribol includes
#include "tribol/common/BasicTypes.hpp"

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;
extern int enzyme_interleave;

#if defined( TRIBOL_USE_CUDA ) || defined( TRIBOL_USE_HIP )
#define TRIBOL_DEVICE_EXTERN_STMT( name ) extern __device__ int name;
#else
#define TRIBOL_DEVICE_EXTERN_STMT( name )
#endif

TRIBOL_DEVICE_EXTERN_STMT( enzyme_dup )
TRIBOL_DEVICE_EXTERN_STMT( enzyme_dupnoneed )
TRIBOL_DEVICE_EXTERN_STMT( enzyme_out )
TRIBOL_DEVICE_EXTERN_STMT( enzyme_const )
TRIBOL_DEVICE_EXTERN_STMT( enzyme_interleave )

// warning: if inlined, triggers function '__enzyme_autodiff' is not defined
template <typename return_type, typename... Args>
TRIBOL_HOST_DEVICE return_type __enzyme_autodiff( Args... );

// warning: if inlined, triggers function '__enzyme_fwddiff' is not defined
template <typename return_type, typename... Args>
TRIBOL_HOST_DEVICE return_type __enzyme_fwddiff( Args... );

#define TRIBOL_ENZYME_INACTIVENOFREE __attribute__( ( enzyme_inactive, enzyme_nofree ) )
#define TRIBOL_ENZYME_INACTIVE __attribute__( ( enzyme_inactive ) )
#define TRIBOL_ENZYME_FN_LIKE( x ) __attribute__( ( enzyme_function_like( #x ) ) )

#else
#define TRIBOL_ENZYME_INACTIVENOFREE
#define TRIBOL_ENZYME_INACTIVE
#define TRIBOL_ENZYME_FN_LIKE( x )
#endif

#define TRIBOL_ENZYME_FN_LIKE_FREE TRIBOL_ENZYME_FN_LIKE( free )
#define TRIBOL_ENZYME_FN_LIKE_DYNCAST TRIBOL_ENZYME_FN_LIKE( __dynamic_cast )

#endif /* TRIBOL_COMMON_ENZYME_HPP_ */
