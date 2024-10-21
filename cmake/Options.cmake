# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

# Get value for some options from corresponding blt options
cmake_dependent_option(TRIBOL_USE_CUDA "Enables Tribol with CUDA support" ON "ENABLE_CUDA" OFF)
cmake_dependent_option(TRIBOL_USE_HIP "Enables Tribol with HIP support" ON "ENABLE_HIP" OFF)
cmake_dependent_option(TRIBOL_USE_MPI "Enables MPI in Tribol" ON "ENABLE_MPI" OFF)
cmake_dependent_option(TRIBOL_USE_OPENMP "Enables Tribol with OpenMP support" ON "ENABLE_OPENMP" OFF)
cmake_dependent_option(TRIBOL_ENABLE_TESTS "Enables Tribol Tests" ON "ENABLE_TESTS" OFF)
cmake_dependent_option(TRIBOL_ENABLE_EXAMPLES "Enables Tribol Examples" ON "ENABLE_EXAMPLES" OFF)
cmake_dependent_option(TRIBOL_ENABLE_DOCS "Enables Tribol Docs" ON "ENABLE_DOCS" OFF)

option(TRIBOL_USE_SINGLE_PRECISION "Use single-precision floating point" OFF)
option(TRIBOL_USE_64BIT_INDEXTYPE "Use 64-bit index type" OFF)

option(TRIBOL_ENABLE_ASAN "Enable AddressSanitizer for memory checking (Clang or GCC only)" OFF)
if(TRIBOL_ENABLE_ASAN)
    if(NOT (C_COMPILER_FAMILY_IS_CLANG OR C_COMPILER_FAMILY_IS_GNU))
        message(FATAL_ERROR "ENABLE_ASAN only supports Clang and GCC")
    endif()
endif()
