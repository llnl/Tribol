# Copyright (c) Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------

if (DOXYGEN_FOUND)
  add_subdirectory(doxygen)
endif()

blt_add_sphinx_target( TARGET tribol_docs )
