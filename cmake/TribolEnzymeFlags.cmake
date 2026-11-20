# Copyright (c) Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------
# GPU compile options required by every translation unit that instantiates Tribol's Enzyme kernels.
#
# Tribol's contact formulations are header-only templates, so downstream CUDA/HIP translation units that instantiate
# them need the same options as Tribol's own sources. This file is installed next to tribol-config.cmake, which
# includes it after finding Enzyme.
#------------------------------------------------------------------------------

## tribol_add_enzyme_gpu_flags( USE_CUDA <bool> USE_HIP <bool> )
##
## Appends Enzyme's loose type analysis options for the enabled GPU language to the ClangEnzymeFlags target. Repeated
## calls do not append the options again.
function(tribol_add_enzyme_gpu_flags)
  cmake_parse_arguments(arg "" "USE_CUDA;USE_HIP" "" ${ARGN})

  if(NOT TARGET ClangEnzymeFlags)
    message(FATAL_ERROR "tribol_add_enzyme_gpu_flags() requires the ClangEnzymeFlags target from Enzyme.")
  endif()

  get_property(already_added TARGET ClangEnzymeFlags PROPERTY TRIBOL_ENZYME_GPU_FLAGS_ADDED)
  if(already_added)
    return()
  endif()

  if(arg_USE_CUDA)
    # Clang-CUDA creates PHI nodes without enough pointer type information in every translation unit that calls Enzyme,
    # including tests and the nodal-normal calculation. Loose type analysis lets Enzyme recover those valid types.
    set_property(TARGET ClangEnzymeFlags APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
      "$<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-mllvm>"
      "$<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-enzyme-loose-types>")
  elseif(arg_USE_HIP)
    # Enzyme also needs loose type analysis for floating-point PHI nodes in AMD GPU contact derivatives.
    # BLT marks device sources as HIP, including files with a .cpp extension.
    # Send this option only to Clang's frontend; HIP forwards driver-level -mllvm options to the device linker,
    # which does not load the Enzyme plugin. SHELL preserves both -Xclang arguments during option deduplication.
    set_property(TARGET ClangEnzymeFlags APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
      "$<$<COMPILE_LANG_AND_ID:HIP,Clang>:SHELL:-Xclang -mllvm -Xclang -enzyme-loose-types>")
  endif()

  set_property(TARGET ClangEnzymeFlags PROPERTY TRIBOL_ENZYME_GPU_FLAGS_ADDED TRUE)
endfunction()
