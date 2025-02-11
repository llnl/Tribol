# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------
# Set up tribol target
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# add shared
#------------------------------------------------------------------------------
add_subdirectory(shared)

#------------------------------------------------------------------------------
# add redecomp
#------------------------------------------------------------------------------
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
if ( TRIBOL_ENABLE_EXAMPLES )
  add_subdirectory(examples)
endif()
