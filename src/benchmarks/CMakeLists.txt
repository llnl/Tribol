# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

if ( BUILD_REDECOMP AND TRIBOL_USE_ENZYME )
  blt_add_executable(
    NAME tribol_nodal_field_benchmark
    SOURCES tribol_nodal_field.cpp
    OUTPUT_DIR ${PROJECT_BINARY_DIR}/benchmarks
    DEPENDS_ON tribol ${tribol_device_depends}
    FOLDER tribol/benchmarks )
else()
  message(STATUS "Nodal-field benchmark requires Enzyme and MFEM/Redecomp; skipping it.")
endif()
