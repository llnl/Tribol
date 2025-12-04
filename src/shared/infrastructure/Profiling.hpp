// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_SHARED_INFRASTRUCTURE_PROFILING_HPP_
#define SRC_SHARED_INFRASTRUCTURE_PROFILING_HPP_

/**
 * @file Profiling.hpp
 *
 * @brief Various helper functions and macros for profiling using Caliper
 */

// Shared config include
#include "shared/config.hpp"

#ifdef TRIBOL_USE_CALIPER
#include "caliper/cali.h"
#endif

/**
 * @def TRIBOL_MARK_FUNCTION
 * Marks a function for Caliper profiling. No-op macro when TRIBOL_ENABLE_PROFILING is off.
 */

/**
 * @def TRIBOL_MARK_LOOP_BEGIN(id, name)
 * Marks the beginning of a loop block for Caliper profiling. No-op macro when TRIBOL_ENABLE_PROFILING is off.
 */

/**
 * @def TRIBOL_MARK_LOOP_ITERATION(id, i)
 * Marks the beginning of a loop iteration for Caliper profiling. No-op macro when TRIBOL_ENABLE_PROFILING is off.
 */

/**
 * @def TRIBOL_MARK_LOOP_END(id)
 * Marks the end of a loop block for Caliper profiling. No-op macro when TRIBOL_ENABLE_PROFILING is off.
 */

/**
 * @def TRIBOL_MARK_BEGIN(id)
 * Marks the start of a region Caliper profiling. No-op macro when TRIBOL_ENABLE_PROFILING is off.
 */

/**
 * @def TRIBOL_MARK_END(id)
 * Marks the end of a region Caliper profiling. No-op macro when TRIBOL_ENABLE_PROFILING is off.
 */

/**
 * @def TRIBOL_MARK_SCOPE(name)
 * Marks a particular scope for Caliper profiling. No-op macro when TRIBOL_ENABLE_PROFILING is off.
 */

// NOTE: The motivation behind wrapping Caliper macros to avoid conflicting macro definitions in the no-op case, and
// give downstream users the option to disable profiling Smith if it pollutes their timings.

#ifdef TRIBOL_USE_CALIPER
#define TRIBOL_MARK_FUNCTION CALI_CXX_MARK_FUNCTION
#define TRIBOL_MARK_LOOP_BEGIN( id, name ) CALI_CXX_MARK_LOOP_BEGIN( id, name )
#define TRIBOL_MARK_LOOP_ITERATION( id, i ) CALI_CXX_MARK_LOOP_ITERATION( id, i )
#define TRIBOL_MARK_LOOP_END( id ) CALI_CXX_MARK_LOOP_END( id )
#define TRIBOL_MARK_BEGIN( name ) CALI_MARK_BEGIN( name )
#define TRIBOL_MARK_END( name ) CALI_MARK_END( name )
#define TRIBOL_MARK_SCOPE( name ) CALI_CXX_MARK_SCOPE( name )
#else
// Define no-op macros in case Smith has not been configured with Caliper
#define TRIBOL_MARK_FUNCTION
#define TRIBOL_MARK_LOOP_BEGIN( id, name )
#define TRIBOL_MARK_LOOP_ITERATION( id, i )
#define TRIBOL_MARK_LOOP_END( id )
#define TRIBOL_MARK_BEGIN( name )
#define TRIBOL_MARK_END( name )
#define TRIBOL_MARK_SCOPE( name )
#endif

#endif  // SRC_SHARED_INFRASTRUCTURE_PROFILING_HPP_
