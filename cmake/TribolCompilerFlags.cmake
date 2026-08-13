# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

if(TRIBOL_ENABLE_ASAN)
    message(STATUS "AddressSanitizer is ON (TRIBOL_ENABLE_ASAN)")
    foreach(_flagvar CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS)
        string(APPEND ${_flagvar} " -fsanitize=address -fno-omit-frame-pointer")
    endforeach()
endif()

# Need to add symbols to dynamic symtab in order to be visible from stacktraces
string(APPEND CMAKE_EXE_LINKER_FLAGS " -rdynamic")

# Apple ld warns about duplicate -l flags when the same library is reachable
# via multiple dependency paths (common with Spack-built CMake targets that use
# raw -l strings instead of imported targets). Suppress the spurious warning.
if(APPLE)
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,-no_warn_duplicate_libraries")
endif()
