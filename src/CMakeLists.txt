# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------
# Set up tribol target
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# add tribol
#------------------------------------------------------------------------------
add_subdirectory(tribol)

#------------------------------------------------------------------------------
# add Tribol tests
#------------------------------------------------------------------------------
if ( TRIBOL_ENABLE_TESTS )
  add_subdirectory(tests)
endif()
