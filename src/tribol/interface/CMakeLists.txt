# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------
# Generate tribol's C and Fortran interfaces
#------------------------------------------------------------------------------

add_shroud(
    YAML_INPUT_FILE      simple_tribol_shroud.yaml
    YAML_OUTPUT_DIR      yaml
    C_FORTRAN_OUTPUT_DIR c_fortran
)

add_shroud(
    YAML_INPUT_FILE      test_mesh_shroud.yaml
    YAML_OUTPUT_DIR      yaml
    C_FORTRAN_OUTPUT_DIR c_fortran
)
