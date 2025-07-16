# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

blt_add_sphinx_target( tribol_guide )

if ( DOXYGEN_FOUND )
  add_dependencies( tribol_guide tribol_doxygen )
endif()