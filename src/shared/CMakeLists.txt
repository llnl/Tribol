# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

## list of headers
set(shared_headers

    infrastructure/Profiling.hpp
    mesh/MeshBuilder.hpp
    )

## list of sources
set(shared_sources

    mesh/MeshBuilder.cpp
    )

## setup the dependency list for tribol shared library
set(shared_depends mfem)
blt_list_append(TO shared_depends ELEMENTS blt::mpi IF TRIBOL_USE_MPI )
blt_list_append(TO shared_depends ELEMENTS caliper IF TRIBOL_USE_CALIPER )
message(STATUS "Tribol shared library dependencies: ${shared_depends}")

## create the library
blt_add_library(
    NAME tribol_shared
    SOURCES ${shared_sources}
    HEADERS ${shared_headers}
    DEPENDS_ON ${shared_depends} ${tribol_device_depends}
    FOLDER shared
    )

# Add separable compilation flag
if(ENABLE_HIP)
    target_compile_options(tribol_shared PRIVATE -fgpu-rdc)
    target_compile_options(tribol_shared PRIVATE
        $<$<CONFIG:Debug>:
            -ggdb
        >
    )
    target_link_options(tribol_shared PRIVATE -fgpu-rdc)
    target_link_options(tribol_shared PRIVATE --hip-link)
endif()

# Don't propagate internal dependencies
target_link_libraries(tribol_shared PRIVATE axom::slic)

target_include_directories(tribol_shared PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )

## mirror the directory structure on the install
foreach( shared_header ${shared_headers} )
    get_filename_component( shared_base_dir ${shared_header} DIRECTORY)
    install( FILES ${shared_header}
             DESTINATION "include/shared/${shared_base_dir}" )
endforeach()

install(TARGETS     tribol_shared
        EXPORT      tribol-targets
        DESTINATION lib
        )

# Add the TRIBOL_DEBUG compile definition, if non-empty
if(TRIBOL_DEBUG_DEFINE_STRING)
  blt_add_target_definitions(TO tribol_shared SCOPE PUBLIC TARGET_DEFINITIONS "${TRIBOL_DEBUG_DEFINE_STRING}")
  # Define AXOM_DEBUG for SLIC macros
  blt_add_target_definitions(TO tribol_shared SCOPE PUBLIC TARGET_DEFINITIONS "${AXOM_DEBUG_DEFINE_STRING}")
endif()

