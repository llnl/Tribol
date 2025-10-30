// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_ERROR_HANDLING_HPP_
#define SRC_TRIBOL_COMMON_ERROR_HANDLING_HPP_

// Tribol config include
#include "tribol/config.hpp"

// Axom includes
#include "axom/slic.hpp"

// Simple, enzyme compatible error handling macros (debug-only)
#if defined( TRIBOL_DEBUG_BUILD )
#define BASIC_ASSERT( EXP ) assert( EXP )
#define BASIC_ASSERT_MSG( EXP, msg ) assert( EXP )
#define BASIC_CHECK( EXP ) assert( EXP )
#define BASIC_CHECK_MSG( EXP, msg ) assert( EXP )
#define BASIC_DEBUG( msg )                        \
  do {                                            \
    std::cout << "[DEBUG]: " << msg << std::endl; \
  } while ( false )
#define BASIC_DEBUG_IF( EXP, msg )                  \
  do {                                              \
    if ( EXP ) {                                    \
      std::cout << "[DEBUG]: " << msg << std::endl; \
    }                                               \
  } while ( false )
// No special case root debug messages with enzyme
#define BASIC_DEBUG_ROOT( msg ) BASIC_DEBUG( msg )
// No special case root debug messages with enzyme
#define BASIC_DEBUG_ROOT_IF( EXP, msg ) BASIC_DEBUG_IF( EXP, msg )
#else
#define BASIC_ASSERT( EXP )
#define BASIC_ASSERT_MSG( EXP, msg )
#define BASIC_CHECK( EXP )
#define BASIC_CHECK_MSG( EXP, msg )
#define BASIC_DEBUG( msg )
#define BASIC_DEBUG_IF( EXP, msg )
#define BASIC_DEBUG_ROOT( msg )
#define BASIC_DEBUG_ROOT_IF( EXP, msg )
#endif

#define BASIC_INFO( msg )                        \
  do {                                           \
    std::cout << "[INFO]: " << msg << std::endl; \
  } while ( false )
#define BASIC_INFO_IF( EXP, msg )                  \
  do {                                             \
    if ( EXP ) {                                   \
      std::cout << "[INFO]: " << msg << std::endl; \
    }                                              \
  } while ( false )
// No special case root info messages with enzyme
#define BASIC_INFO_ROOT( msg ) BASIC_INFO( msg )
// No special case root info messages with enzyme
#define BASIC_INFO_ROOT_IF( EXP, msg ) BASIC_INFO_IF( EXP, msg )
#define BASIC_ERROR( msg )                        \
  do {                                            \
    std::cerr << "[ERROR]: " << msg << std::endl; \
    std::terminate();                             \
  } while ( false )
#define BASIC_ERROR_IF( msg, exp )                  \
  do {                                              \
    if ( exp ) {                                    \
      std::cerr << "[ERROR]: " << msg << std::endl; \
      std::terminate();                             \
    }                                               \
  } while ( false )
// No special case root error messages with enzyme
#define BASIC_ERROR_ROOT( msg ) BASIC_ERROR( msg )
// No special case root error messages with enzyme
#define BASIC_ERROR_ROOT_IF( EXP, msg ) BASIC_ERROR_IF( EXP, msg )
#define BASIC_WARNING( msg )                        \
  do {                                              \
    std::cerr << "[WARNING]: " << msg << std::endl; \
  } while ( false )
#define BASIC_WARNING_IF( EXP, msg )                  \
  do {                                                \
    if ( EXP ) {                                      \
      std::cerr << "[WARNING]: " << msg << std::endl; \
    }                                                 \
  } while ( false )
// No special case root warning messages with enzyme
#define BASIC_WARNING_ROOT( msg ) BASIC_WARNING( msg )
// No special case root warning messages with enzyme
#define BASIC_WARNING_ROOT_IF( EXP, msg ) BASIC_WARNING_IF( EXP, msg )

#endif /* SRC_TRIBOL_COMMON_ERROR_HANDLING_HPP_ */
