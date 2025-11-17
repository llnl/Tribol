# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------
# Set up tribol's TPLs
#------------------------------------------------------------------------------

message(STATUS "Configuring TPLs...\n"
               "----------------------")

set(EXPORTED_TPL_DEPS)
include(CMakeFindDependencyMacro)

#------------------------------------------------------------------------------
# Create global variable to toggle between GPU targets
#------------------------------------------------------------------------------
if(TRIBOL_USE_CUDA)
  set(tribol_device_depends blt::cuda CACHE STRING "" FORCE)
endif()
if(TRIBOL_USE_HIP)
  set(tribol_device_depends blt::hip CACHE STRING "" FORCE)
endif()

#------------------------------------------------------------------------------
# Enzyme
#------------------------------------------------------------------------------
if (ENZYME_DIR)
  message(STATUS "Setting up external Enzyme TPL...")

  set(Enzyme_DIR ${ENZYME_DIR})
  find_dependency(Enzyme REQUIRED PATHS "${ENZYME_DIR}")

  set(TRIBOL_USE_ENZYME TRUE)
else()
  message(STATUS "Enzyme support is OFF")
endif()


#------------------------------------------------------------------------------
# Umpire
#------------------------------------------------------------------------------

if (UMPIRE_DIR)
  message(STATUS "Setting up external Umpire TPL...")

  set(umpire_DIR ${UMPIRE_DIR})
  find_dependency(umpire REQUIRED PATHS "${UMPIRE_DIR}")

  set(TRIBOL_USE_UMPIRE TRUE)
else()
  message(STATUS "Umpire support is OFF")
endif()


#------------------------------------------------------------------------------
# RAJA
#------------------------------------------------------------------------------

if (RAJA_DIR)
  message(STATUS "Setting up external RAJA TPL...")

  find_dependency(raja REQUIRED PATHS "${RAJA_DIR}")

  set(TRIBOL_USE_RAJA TRUE)
else()
  message(STATUS "RAJA support is OFF")
endif()


#------------------------------------------------------------------------------
# axom
#------------------------------------------------------------------------------
if (TARGET axom)
    # Case: Tribol included in project that also creates an axom target, no need to recreate axom
    message(STATUS "Axom support is ON, using existing axom target")

    # Add components and dependencies of axom to this export set but don't prefix it with tribol::
    # NOTE(chapman39@llnl.gov): Cannot simply install axom, since it is an imported library
    set(_axom_exported_targets ${axom_exported_targets})
    install(TARGETS              ${_axom_exported_targets}
            EXPORT               tribol-targets
            DESTINATION          lib)
    unset(_axom_exported_targets)

    set(AXOM_FOUND TRUE CACHE BOOL "" FORCE)

elseif (AXOM_DIR)
  message(STATUS "Setting up external Axom TPL...")
  tribol_assert_path_exists( ${AXOM_DIR} )
  find_dependency(axom REQUIRED PATHS "${AXOM_DIR}/lib/cmake")
else()
  message(FATAL_ERROR 
     "Axom is a required dependency for tribol. "
     "Please configure tribol with a path to Axom via the AXOM_DIR variable.")
endif()


#------------------------------------------------------------------------------
# mfem
#------------------------------------------------------------------------------

if (TARGET mfem)
    # Case: Tribol included in project that also creates an mfem target, no need to recreate mfem
    # Note - white238: I can't seem to get this to pass install testing due to mfem being included
    # in multiple export sets
    message(STATUS "MFEM support is ON, using existing mfem target")

    # Add it to this export set but don't prefix it with tribol::
    # NOTE: imported targets cannot be part of an export set
    get_target_property(_is_imported mfem IMPORTED)
    if(NOT "${_is_imported}")
        install(TARGETS              mfem
                EXPORT               tribol-targets
                DESTINATION          lib)
    endif()

    set(MFEM_FOUND TRUE CACHE BOOL "" FORCE)
elseif (MFEM_DIR)
  message(STATUS "Setting up external MFEM TPL...")

  include(${PROJECT_SOURCE_DIR}/cmake/thirdparty/SetupMFEM.cmake)

  list(APPEND EXPORTED_TPL_DEPS mfem)
else()
  message(FATAL_ERROR 
     "MFEM is a required dependency for tribol. "
     "Please configure tribol with a path to MFEM via the MFEM_DIR variable.")
endif()


#------------------------------------------------------------------------------
# Shroud - Generates C/Fortran/Python bindings
#------------------------------------------------------------------------------
if(EXISTS ${SHROUD_EXECUTABLE})
    message(STATUS "Setting up shroud TPL...")
    execute_process(COMMAND ${SHROUD_EXECUTABLE}
                    --cmake ${CMAKE_CURRENT_BINARY_DIR}/SetupShroud.cmake
                    ERROR_VARIABLE SHROUD_cmake_error
                    RESULT_VARIABLE SHROUD_cmake_result
                    OUTPUT_STRIP_TRAILING_WHITESPACE )
    if(NOT "${SHROUD_cmake_result}" STREQUAL "0")
        message(FATAL_ERROR "Error code from Shroud: ${SHROUD_cmake_result}\n${SHROUD_cmake_error}")
    endif()

    include(${CMAKE_CURRENT_BINARY_DIR}/SetupShroud.cmake)
else()
    message(STATUS "Shroud support is OFF")
endif()


#---------------------------------------------------------------------------
# Remove non-existant INTERFACE_INCLUDE_DIRECTORIES from imported targets
# to work around CMake error
#---------------------------------------------------------------------------
set(_imported_targets
    axom
    axom::mfem
    conduit
    conduit::conduit_mpi
    conduit::conduit
    conduit_relay_mpi
    conduit_relay_mpi_io
    conduit_blueprint
    conduit_blueprint_mpi)

foreach(_target ${_imported_targets})
    if(TARGET ${_target})
        message(STATUS "Removing non-existant include directories from target[${_target}]")

        get_target_property(_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
        set(_existing_dirs)
        foreach(_dir ${_dirs})
            if (EXISTS "${_dir}")
                list(APPEND _existing_dirs "${_dir}")
            endif()
        endforeach()
        if (_existing_dirs)
            set_target_properties(${_target} PROPERTIES
                                  INTERFACE_INCLUDE_DIRECTORIES "${_existing_dirs}" )
        endif()
    endif()
endforeach()

# export tribol-targets
foreach(dep ${EXPORTED_TPL_DEPS})
  # If the target is EXPORTABLE, add it to the export set
  get_target_property(_is_imported ${dep} IMPORTED)
  if(NOT ${_is_imported})
      install(TARGETS              ${dep}
              EXPORT               tribol-targets
              DESTINATION          lib)
      # Namespace target to avoid conflicts
      set_target_properties(${dep} PROPERTIES EXPORT_NAME tribol::${dep})
  endif()
endforeach()

#------------------------------------------------------------------------------
# Caliper
#------------------------------------------------------------------------------
if(TRIBOL_ENABLE_PROFILING AND NOT CALIPER_DIR)
    message(FATAL_ERROR "TRIBOL_ENABLE_PROFILING cannot be ON without CALIPER_DIR defined. Either specify a host \
                          config with CALIPER_DIR, or rebuild Tribol TPLs with +profiling variant.")
endif()

if(CALIPER_DIR AND TRIBOL_ENABLE_PROFILING)
    tribol_assert_path_exists(${CALIPER_DIR})

    # Should this logic be in the Caliper CMake package?
    # If CMake version doesn't support CUDAToolkit the libraries
    # are just "baked in"
    if(TRIBOL_ENABLE_CUDA)
        if(CMAKE_VERSION VERSION_LESS 3.17)
            message(FATAL_ERROR "Tribol+Caliper+CUDA requires CMake > 3.17.")
        else()
            find_package(CUDAToolkit REQUIRED)
        endif() 
    endif()

    find_dependency(caliper REQUIRED PATHS "${CALIPER_DIR}")
    message(STATUS "Caliper support is ON")
    set(TRIBOL_USE_CALIPER TRUE)
else()
    message(STATUS "Caliper support is OFF")
    set(TRIBOL_USE_CALIPER FALSE)
endif()

message(STATUS "--------------------------\n"
               "Finished configuring TPLs")

