# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

message(STATUS "Configuring TPLs...\n----------------------")

set(EXPORTED_TPL_DEPS)

if(TRIBOL_USE_CUDA)
  set(tribol_device_depends blt::cuda CACHE STRING "" FORCE)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
endif()

if(TARGET mfem)
  message(STATUS "MFEM support is ON, using existing mfem target")
  get_target_property(_mfem_is_imported mfem IMPORTED)
  if(NOT _mfem_is_imported)
    install(TARGETS mfem EXPORT tribol-targets DESTINATION lib)
  endif()
  set(MFEM_FOUND TRUE CACHE BOOL "" FORCE)
elseif(MFEM_DIR)
  message(STATUS "Setting up external MFEM TPL")
  include(${PROJECT_SOURCE_DIR}/cmake/thirdparty/SetupMFEM.cmake)
  list(APPEND EXPORTED_TPL_DEPS mfem)
else()
  message(FATAL_ERROR "MFEM is required. Configure with MFEM_DIR pointing to an MFEM install.")
endif()

foreach(dependency ${EXPORTED_TPL_DEPS})
  get_target_property(_is_imported ${dependency} IMPORTED)
  if(NOT _is_imported)
    install(TARGETS ${dependency} EXPORT tribol-targets DESTINATION lib)
    set_target_properties(${dependency} PROPERTIES EXPORT_NAME tribol::${dependency})
  endif()
endforeach()

message(STATUS "--------------------------\nFinished configuring TPLs")
