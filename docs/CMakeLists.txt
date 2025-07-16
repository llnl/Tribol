# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------

if (DOXYGEN_FOUND)
  add_subdirectory(doxygen)
endif()

if (SPHINX_FOUND)
  add_subdirectory(sphinx)
endif()
