# Copyright (c) Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

blt_add_doxygen_target( TARGET tribol_doxygen )

# Required directory for doxygen to generate output to 
add_custom_target( build_doxygen_output_dir 
                   ALL 
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/src/docs/html/doxygen
                 ) 

add_dependencies( tribol_doxygen build_doxygen_output_dir )
