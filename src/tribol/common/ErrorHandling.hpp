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

namespace tribol {

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

/**
 * @brief Error handling utilities for Tribol, wrapping AXOM SLIC functionality
 */
namespace error {
enum class Level
{
  Debug,
  Info,
  Warning,
  Error,
  Critical,
  NumLevels
};

enum class Output
{
  None,
  Log,
  PrintToScreen,
  LogAndPrintToScreen
};

enum class Action
{
  None,
  ThrowException,
  Abort
};

enum class Group
{
  None,
  Minimal,
  Medium,
  Maximum,
  NumGroups
};

enum class Label
{
  General,
  ComputationalGeometry,
  Physics,
  FiniteElement,
  NumLabels
};

enum class Parallel
{
  Root,
  PerRank
};

using ActionMap = std::array<Action, static_cast<size_t>( Level::NumLevels )>;

using ActionMapByLabel = std::array<ActionMap, static_cast<size_t>( Label::NumLabels )>;

constexpr std::array<ActionMap, static_cast<size_t>( Group::NumGroups )> action_map_by_group = {
    // Group::None
    ActionMap{
        Action::None,  // Debug
        Action::None,  // Info
        Action::None,  // Warning
        Action::None,  // Error
        Action::None   // Critical
    },
    // Group::Minimal
    ActionMap{
        Action::None,            // Debug
        Action::None,            // Info
        Action::ThrowException,  // Warning
        Action::ThrowException,  // Error
        Action::Abort            // Critical
    },
    // Group::Medium
    ActionMap{
        Action::None,            // Debug
        Action::None,            // Info
        Action::ThrowException,  // Warning
        Action::Abort,           // Error
        Action::Abort            // Critical
    },
    // Group::Maximum
    ActionMap{
        Action::None,   // Debug
        Action::None,   // Info
        Action::Abort,  // Warning
        Action::Abort,  // Error
        Action::Abort   // Critical
    } };

using OutputMap = std::array<Output, static_cast<size_t>( Level::NumLevels )>;

using OutputMapByLabel = std::array<OutputMap, static_cast<size_t>( Label::NumLabels )>;

constexpr std::array<OutputMap, static_cast<size_t>( Group::NumGroups )> output_map_by_group = {
    // Group::None
    OutputMap{
        Output::None,  // Debug
        Output::None,  // Info
        Output::None,  // Warning
        Output::None,  // Error
        Output::None   // Critical
    },
    // Group::Minimal
    OutputMap{
        Output::None,                 // Debug
        Output::None,                 // Info
        Output::Log,                  // Warning
        Output::LogAndPrintToScreen,  // Error
        Output::LogAndPrintToScreen   // Critical
    },
    // Group::Medium
    OutputMap{
        Output::Log,                  // Debug
        Output::Log,                  // Info
        Output::LogAndPrintToScreen,  // Warning
        Output::LogAndPrintToScreen,  // Error
        Output::LogAndPrintToScreen   // Critical
    },
    // Group::Maximum
    OutputMap{
        Output::LogAndPrintToScreen,  // Debug
        Output::LogAndPrintToScreen,  // Info
        Output::LogAndPrintToScreen,  // Warning
        Output::LogAndPrintToScreen,  // Error
        Output::LogAndPrintToScreen   // Critical
    } };

template <Level level, Action action, Output output, Parallel parallel>
void handle( std::string message, bool test = true )
{
  handleOutput<level, output, parallel>( message, test );
  handleAction<level, action, parallel>( test );
}

template <Level level, Output output, Parallel parallel>
void handleOutput( std::string, bool = true )
{
}

template <>
void handleOutput<Level::Debug, Output::Log, Parallel::Root>( std::string message, bool test )
{
  SLIC_DEBUG_ROOT_IF( test, message );
}

template <>
void handleOutput<Level::Debug, Output::Log, Parallel::PerRank>( std::string message, bool test )
{
  SLIC_DEBUG_IF( test, message );
}

template <>
void handleOutput<Level::Debug, Output::PrintToScreen, Parallel::Root>( std::string message, bool test )
{
  SLIC_DEBUG_ROOT_IF( test, message );
}

template <>
void handleOutput<Level::Debug, Output::PrintToScreen, Parallel::PerRank>( std::string message, bool test )
{
  SLIC_DEBUG_IF( test, message );
}

template <>
void handleOutput<Level::Debug, Output::LogAndPrintToScreen, Parallel::Root>( std::string message, bool test )
{
  SLIC_DEBUG_ROOT_IF( test, message );
}

template <>
void handleOutput<Level::Debug, Output::LogAndPrintToScreen, Parallel::PerRank>( std::string message, bool test )
{
  SLIC_DEBUG_IF( test, message );
}

template <>
void handleOutput<Level::Info, Output::Log, Parallel::Root>( std::string message, bool test )
{
  SLIC_INFO_ROOT_IF( test, message );
}

template <>
void handleOutput<Level::Info, Output::Log, Parallel::PerRank>( std::string message, bool test )
{
  SLIC_INFO_IF( test, message );
}

template <>
void handleOutput<Level::Info, Output::PrintToScreen, Parallel::Root>( std::string message, bool test )
{
  SLIC_INFO_ROOT_IF( test, message );
}

template <>
void handleOutput<Level::Info, Output::PrintToScreen, Parallel::PerRank>( std::string message, bool test )
{
  SLIC_INFO_IF( test, message );
}

template <>
void handleOutput<Level::Info, Output::LogAndPrintToScreen, Parallel::Root>( std::string message, bool test )
{
  SLIC_INFO_ROOT_IF( test, message );
}

template <>
void handleOutput<Level::Info, Output::LogAndPrintToScreen, Parallel::PerRank>( std::string message, bool test )
{
  SLIC_INFO_IF( test, message );
}

template <>
void handleOutput<Level::Warning, Output::Log, Parallel::Root>( std::string message, bool test )
{
  SLIC_DEBUG_ROOT_IF( test, message );
}

template <>
void handleOutput<Level::Debug, Output::Log, Parallel::PerRank>( std::string message, bool test )
{
  SLIC_DEBUG_IF( test, message );
}

template <>
void handleOutput<Level::Debug, Output::PrintToScreen, Parallel::Root>( std::string message, bool test )
{
  SLIC_DEBUG_ROOT_IF( test, message );
}

template <>
void handleOutput<Level::Debug, Output::PrintToScreen, Parallel::PerRank>( std::string message, bool test )
{
  SLIC_DEBUG_IF( test, message );
}

template <>
void handleOutput<Level::Debug, Output::LogAndPrintToScreen, Parallel::Root>( std::string message, bool test )
{
  SLIC_DEBUG_ROOT_IF( test, message );
}

template <>
void handleOutput<Level::Debug, Output::LogAndPrintToScreen, Parallel::PerRank>( std::string message, bool test )
{
  SLIC_DEBUG_IF( test, message );
}

template <Level level, Action action, Output output, Parallel parallel>
class Handler {
 public:
  template <Label label, Level level, Parallel parallel>
  void handle( bool test = true )
  {
    Action action = action_map_.at( static_cast<size_t>( label ) ).at( static_cast<size_t>( level ) );
    Output output = output_map_.at( static_cast<size_t>( label ) ).at( static_cast<size_t>( level ) );

    // Handle output
    if ( output == Output::Log || output == Output::LogAndPrintToScreen ) {
      if ( parallel == Parallel::Root ) {
        SLIC_LOG( static_cast<axom::slic::MessageLevel>( level ), "Root rank logging message." );
      } else {
        SLIC_LOG( static_cast<axom::slic::MessageLevel>( level ), "All ranks logging message." );
      }
    }
    if ( output == Output::PrintToScreen || output == Output::LogAndPrintToScreen ) {
      if ( parallel == Parallel::Root ) {
        SLIC_PRINT( static_cast<axom::slic::MessageLevel>( level ), "Root rank printing message." );
      } else {
        SLIC_PRINT( static_cast<axom::slic::MessageLevel>( level ), "All ranks printing message." );
      }
    }

    // Handle action
    if ( action == Action::ThrowException ) {
      throw std::runtime_error( "Exception thrown by Tribol error handler." );
    } else if ( action == Action::Abort ) {
      SLIC_ABORT( "Aborting execution as directed by Tribol error handler." );
    }
  }

 private:
  ActionMapByLabel action_map_;
  OutputMapByLabel output_map_;
};

}  // namespace error

}  // namespace tribol

#endif /* SRC_TRIBOL_COMMON_ERROR_HANDLING_HPP_ */
