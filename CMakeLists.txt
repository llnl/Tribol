# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------
# Setup Tribol project
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

cmake_policy(SET CMP0057 NEW)

project(tribol LANGUAGES C CXX)

if (ENABLE_FORTRAN)
   enable_language(Fortran)
endif()

#------------------------------------------------------------------------------
# Setup BLT
#------------------------------------------------------------------------------
if (DEFINED BLT_SOURCE_DIR)
    # Support having an external BLT outside of the repository
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Provided 'BLT_SOURCE_DIR' does not contain SetupBLT.cmake")
    endif()
else()
    # Use internal 'blt' submodule path if BLT_SOURCE_DIR not provided
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "Cannot locate BLT. "
            "Either run the following two commands in your git repository: \n"
            "    git submodule init cmake/blt\n"
            "    git submodule update\n"
            "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif()
endif()

# Override a conflicting target name between blt and mfem's build systems
SET(BLT_CODE_CHECK_TARGET_NAME "code_check" CACHE STRING "")

if ("${PROJECT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    # Set some default BLT options before loading BLT only if not included in
    # another project
    if (NOT BLT_CXX_STD)
        set(BLT_CXX_STD "c++17" CACHE STRING "")
    endif()

    # These BLT tools are not used in Tribol, turn them off
    set(_unused_blt_tools
        CLANGQUERY
        VALGRIND
        ASTYLE
        CMAKEFORMAT
        UNCRUSTIFY
        YAPF)
    foreach(_tool ${_unused_blt_tools})
        set(ENABLE_${_tool} OFF CACHE BOOL "")
    endforeach()

    # These BLT tools are only used by Tribol developers, so turn them off
    # unless an explicit executable path is given
    set(_used_blt_tools
        CLANGFORMAT
        CLANGTIDY
        CPPCHECK
        DOXYGEN
        SPHINX)
    foreach(_tool ${_used_blt_tools})
        if(NOT ${_tool}_EXECUTABLE)
            set(ENABLE_${_tool} OFF CACHE BOOL "")
        else()
            set(ENABLE_${_tool} ON CACHE BOOL "")
        endif()
    endforeach()

    set(BLT_REQUIRED_CLANGFORMAT_VERSION  "19" CACHE STRING "")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

#------------------------------------------------------------------------------
# Fortran Configuration
#------------------------------------------------------------------------------
if(ENABLE_FORTRAN)
    # Check C/C++ compiler compatiblity with the Fortran compiler
    include(FortranCInterface)
    FortranCInterface_VERIFY()
    FortranCInterface_VERIFY(CXX)

    # Axom assumes that all Fortran files use free formatting
    set(CMAKE_Fortran_FORMAT FREE)
endif()

#------------------------------------------------------------------------------
# Load tribol macros and options
#------------------------------------------------------------------------------
include(cmake/TribolMacros.cmake)
include(cmake/Options.cmake)
include(cmake/TribolCompilerFlags.cmake)

#------------------------------------------------------------------------------
# Add Code Checks
#------------------------------------------------------------------------------

message(STATUS "Tribol Code Checks: ${TRIBOL_ENABLE_CODE_CHECKS}")

# Add extra file extensions for code styling
# Note: Add them also to tribol_add_code_checks in cmake/TribolMacros.cmake

# CUDA
list(APPEND BLT_C_FILE_EXTS ".cuh")
# Configured C++ files
list(APPEND BLT_C_FILE_EXTS ".cpp.in" ".hpp.in")

if (TRIBOL_STYLE_CI_ONLY OR TRIBOL_ENABLE_CODE_CHECKS)
    tribol_add_code_checks(PREFIX tribol)
endif()

if (TRIBOL_STYLE_CI_ONLY)
    # Exit processing the rest of the build in style only build to avoid any possible
    # CMake configuration issues outside of just enabling `make style`. This build,
    # is not capable of building Tribol and should not be advertised. It is for CI 
    # purposes only.
    return()
endif()

#------------------------------------------------------------------------------
# Add TPLs
#------------------------------------------------------------------------------
include(cmake/SetupThirdParty.cmake)

#------------------------------------------------------------------------------
# Set up TRIBOL_DEBUG compiler define string, as appropriate, based on config
# type. Result is stored in TRIBOL_DEBUG_DEFINE_STRING cache variable
#------------------------------------------------------------------------------
set(TRIBOL_DEBUG_DEFINE_STRING "")

# Handle the three valid values for TRIBOL_DEBUG_DEFINE: {on, off, default}
string(TOUPPER "${TRIBOL_DEBUG_DEFINE}" _tribol_debug_define_upper)
if("${_tribol_debug_define_upper}" MATCHES "ON|TRUE")
  set(TRIBOL_DEBUG_DEFINE_STRING "TRIBOL_DEBUG")
elseif("${_tribol_debug_define_upper}" MATCHES "OFF|FALSE")
  # no-op
elseif("${_tribol_debug_define_upper}" MATCHES "DEFAULT")
  # Default behavior is to be on for Debug and RelWithDebInfo configurations and off otherwise
  if(NOT CMAKE_CONFIGURATION_TYPES) # This case handles single-config generators, e.g. make
      if( CMAKE_BUILD_TYPE MATCHES "(Debug|RelWithDebInfo)" )
        set(TRIBOL_DEBUG_DEFINE_STRING "TRIBOL_DEBUG")
      endif()
  else () # This case handles multi-config generators, e.g. MSVC
      set(TRIBOL_DEBUG_DEFINE_STRING "$<$<CONFIG:Debug,RelWithDebInfo>:TRIBOL_DEBUG>")
  endif()
else()  # Handle bad value for TRIBOL_DEBUG_DEFINE config variable
  message(FATAL_ERROR 
    "Invalid value for TRIBOL_DEBUG_DEFINE. Must be 'DEFAULT', 'ON' or 'OFF'; was '${TRIBOL_DEBUG_DEFINE}'")
endif()

set(TRIBOL_DEBUG_DEFINE_STRING "${TRIBOL_DEBUG_DEFINE_STRING}" CACHE STRING "" FORCE)
# Define AXOM_DEBUG for SLIC macros
string(REPLACE "TRIBOL" "AXOM" AXOM_DEBUG_DEFINE_STRING "${TRIBOL_DEBUG_DEFINE_STRING}")
set(AXOM_DEBUG_DEFINE_STRING "${AXOM_DEBUG_DEFINE_STRING}" CACHE STRING "" FORCE)

#------------------------------------------------------------------------------
# Add source directory
#------------------------------------------------------------------------------
add_subdirectory(src)

if ( TRIBOL_ENABLE_DOCS )
  add_subdirectory(docs)
endif()

#------------------------------------------------------------------------------
# Generate header file with configuration options
#------------------------------------------------------------------------------
include(cmake/TribolVersion.cmake)
include(cmake/TribolConfig.cmake)

