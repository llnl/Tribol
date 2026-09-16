# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------
# Set up tribol target
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# add legacy support libraries
#------------------------------------------------------------------------------
if(TRIBOL_ENABLE_LEGACY)
  add_subdirectory(shared)

  if ( NOT DEFINED BUILD_REDECOMP )
    set(BUILD_REDECOMP ON)
  endif()
  if ( BUILD_REDECOMP )
    if ( TRIBOL_USE_MPI )
      add_subdirectory(redecomp)
    else()
      message(STATUS "Redecomp requested, but MPI not enabled. "
        "Build will proceed with redecomp disabled.")
      set(BUILD_REDECOMP OFF)
    endif()
  endif()
else()
  set(BUILD_REDECOMP OFF)
endif()

#------------------------------------------------------------------------------
# add tribol
#------------------------------------------------------------------------------
add_subdirectory(tribol)

#------------------------------------------------------------------------------
# add tribol's and redecomp's tests
#------------------------------------------------------------------------------
if ( TRIBOL_ENABLE_TESTS )
  add_subdirectory(tests)
endif()

#------------------------------------------------------------------------------
# add tribol's and redecomp's examples
#------------------------------------------------------------------------------
if ( TRIBOL_ENABLE_EXAMPLES AND TRIBOL_ENABLE_LEGACY )
  add_subdirectory(examples)
elseif(TRIBOL_ENABLE_EXAMPLES)
  message(STATUS "Legacy Tribol examples are disabled; use the tested examples in the public API documentation")
endif()

convert_to_native_escaped_file_path(${PROJECT_SOURCE_DIR} TRIBOL_REPO_DIR)
convert_to_native_escaped_file_path(${CMAKE_BINARY_DIR}   TRIBOL_BIN_DIR)

# Generate config headers
set(TRIBOL_DATA_DIR ${PROJECT_SOURCE_DIR}/data)
tribol_configure_file(${PROJECT_SOURCE_DIR}/src/tribol/config.hpp.in
                      ${PROJECT_BINARY_DIR}/include/tribol/config.hpp)

tribol_configure_file(${PROJECT_SOURCE_DIR}/src/shared/config.hpp.in
                      ${PROJECT_BINARY_DIR}/include/shared/config.hpp)
