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

#define IF_ACTIVE( active_or_not, exp ) __IF_ACTIVE_IMPL( active_or_not, exp )
#define __IF_ACTIVE_IMPL( active_or_not, exp ) __##active_or_not##_IMPL( exp )
#define __1_IMPL( exp ) exp
#define __0_IMPL( exp ) ( (void)0 )

#define IF_NOT_ACTIVE( active_or_not, exp ) __IF_NOT_ACTIVE_IMPL( active_or_not, exp )
#define __IF_NOT_ACTIVE_IMPL( active_or_not, exp ) __##active_or_not##_NOT_IMPL( exp )
#define __1_NOT_IMPL( exp ) ( (void)0 )
#define __0_NOT_IMPL( exp ) exp

#define LOG( level, type, ... ) __LOG_IMPL( level, type, __VA_ARGS__ )
#define __LOG_IMPL( level, type, ... ) type##_##level( __VA_ARGS__ )

// Simple, enzyme compatible error handling macros (debug-only)
#ifdef TRIBOL_DEBUG_BUILD
#define BASIC_ASSERT( exp ) assert( exp )
#define BASIC_ASSERT_MSG( exp, msg ) assert( exp )
#define BASIC_CHECK( exp ) assert( exp )
#define BASIC_CHECK_MSG( exp, msg ) assert( exp )
#define BASIC_DEBUG( msg )                        \
  do {                                            \
    std::cout << "[DEBUG]: " << msg << std::endl; \
  } while ( false )
#define BASIC_DEBUG_IF( exp, msg )                  \
  do {                                              \
    if ( exp ) {                                    \
      std::cout << "[DEBUG]: " << msg << std::endl; \
    }                                               \
  } while ( false )
// No special case root debug messages with enzyme
#define BASIC_DEBUG_ROOT( msg ) BASIC_DEBUG( msg )
// No special case root debug messages with enzyme
#define BASIC_DEBUG_ROOT_IF( exp, msg ) BASIC_DEBUG_IF( exp, msg )
#else
#define BASIC_ASSERT( exp )
#define BASIC_ASSERT_MSG( exp, msg )
#define BASIC_CHECK( exp )
#define BASIC_CHECK_MSG( exp, msg )
#define BASIC_DEBUG( msg )
#define BASIC_DEBUG_IF( exp, msg )
#define BASIC_DEBUG_ROOT( msg )
#define BASIC_DEBUG_ROOT_IF( exp, msg )
#endif

#define BASIC_INFO( msg )                                \
  do {                                                   \
    std::ostringstream __oss;                            \
    __oss << msg;                                        \
    std::cout << "[INFO]: " << __oss.str() << std::endl; \
  } while ( false )
#define BASIC_INFO_IF( exp, msg )                          \
  do {                                                     \
    if ( exp ) {                                           \
      std::ostringstream __oss;                            \
      __oss << msg;                                        \
      std::cout << "[INFO]: " << __oss.str() << std::endl; \
    }                                                      \
  } while ( false )
// No special case root info messages with enzyme
#define BASIC_INFO_ROOT( msg ) BASIC_INFO( msg )
// No special case root info messages with enzyme
#define BASIC_INFO_ROOT_IF( exp, msg ) BASIC_INFO_IF( exp, msg )
#define BASIC_ERROR( msg )                                \
  do {                                                    \
    std::ostringstream __oss;                             \
    __oss << msg;                                         \
    std::cerr << "[ERROR]: " << __oss.str() << std::endl; \
    std::terminate();                                     \
  } while ( false )
#define BASIC_ERROR_IF( exp, msg )                          \
  do {                                                      \
    if ( exp ) {                                            \
      std::ostringstream __oss;                             \
      __oss << msg;                                         \
      std::cerr << "[ERROR]: " << __oss.str() << std::endl; \
      std::terminate();                                     \
    }                                                       \
  } while ( false )
// No special case root error messages with enzyme
#define BASIC_ERROR_ROOT( msg ) BASIC_ERROR( msg )
// No special case root error messages with enzyme
#define BASIC_ERROR_ROOT_IF( exp, msg ) BASIC_ERROR_IF( exp, msg )
#define BASIC_WARNING( msg )                                \
  do {                                                      \
    std::ostringstream __oss;                               \
    __oss << msg;                                           \
    std::cerr << "[WARNING]: " << __oss.str() << std::endl; \
  } while ( false )
#define BASIC_WARNING_IF( exp, msg )                          \
  do {                                                        \
    if ( exp ) {                                              \
      std::ostringstream __oss;                               \
      __oss << msg;                                           \
      std::cerr << "[WARNING]: " << __oss.str() << std::endl; \
    }                                                         \
  } while ( false )
// No special case root warning messages with enzyme
#define BASIC_WARNING_ROOT( msg ) BASIC_WARNING( msg )
// No special case root warning messages with enzyme
#define BASIC_WARNING_ROOT_IF( exp, msg ) BASIC_WARNING_IF( exp, msg )

#if ENZYME_ACTIVE == 1
#define ENZYME_ASSERT( exp ) BASIC_ASSERT( exp )
#define ENZYME_ASSERT_MSG( exp, msg ) BASIC_ASSERT_MSG( exp, msg )
#define ENZYME_CHECK( exp ) BASIC_CHECK( exp )
#define ENZYME_CHECK_MSG( exp, msg ) BASIC_CHECK_MSG( exp, msg )
#define ENZYME_DEBUG( msg ) BASIC_DEBUG( msg )
#define ENZYME_DEBUG_IF( exp, msg ) BASIC_DEBUG_IF( exp, msg )
#define ENZYME_DEBUG_ROOT( msg ) BASIC_DEBUG_ROOT( msg )
#define ENZYME_DEBUG_ROOT_IF( exp, msg ) BASIC_DEBUG_ROOT_IF( exp, msg )
#else
#define ENZYME_ASSERT( exp ) SLIC_ASSERT( exp )
#define ENZYME_ASSERT_MSG( exp, msg ) SLIC_ASSERT_MSG( exp, msg )
#define ENZYME_CHECK( exp ) SLIC_CHECK( exp )
#define ENZYME_CHECK_MSG( exp, msg ) SLIC_CHECK_MSG( exp, msg )
#define ENZYME_DEBUG( msg ) SLIC_DEBUG( msg )
#define ENZYME_DEBUG_IF( exp, msg ) SLIC_DEBUG_IF( exp, msg )
#define ENZYME_DEBUG_ROOT( msg ) SLIC_DEBUG_ROOT( msg )
#define ENZYME_DEBUG_ROOT_IF( exp, msg ) SLIC_DEBUG_ROOT_IF( exp, msg )
#endif

#endif /* SRC_TRIBOL_COMMON_ERROR_HANDLING_HPP_ */
