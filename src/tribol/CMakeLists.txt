# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

# Add targets to generate C and Fortran interfaces
if (SHROUD_FOUND)
     add_subdirectory(interface)
endif()

## list of headers
set(tribol_headers

    common/ArrayTypes.hpp
    common/BasicTypes.hpp
    common/Containers.hpp
    common/Enzyme.hpp
    common/ExecModel.hpp
    common/LoopExec.hpp
    common/Parameters.hpp

    geom/CompGeom.hpp
    geom/ElementNormal.hpp
    geom/GeomUtilities.hpp
    geom/Hyperplane.hpp
    geom/NodalNormal.hpp

    integ/FE.hpp
    integ/Integration.hpp

    interface/mfem_tribol.hpp
    interface/simple_tribol.hpp
    interface/tribol.hpp

    mesh/CouplingScheme.hpp
    mesh/InterfacePairs.hpp
    mesh/MeshData.hpp
    mesh/MethodCouplingData.hpp
    mesh/MfemData.hpp

    physics/AlignedMortar.hpp
    physics/CommonPlane.hpp
    physics/Mortar.hpp
    physics/Physics.hpp

    search/InterfacePairFinder.hpp

    utils/Algorithm.hpp
    utils/ContactPlaneOutput.hpp
    utils/DataManager.hpp
    utils/Math.hpp
    utils/TestUtils.hpp
)

## list of sources
set(tribol_sources
     
    geom/CompGeom.cpp
    geom/ElementNormal.cpp
    geom/GeomUtilities.cpp
    geom/NodalNormal.cpp

    integ/FE.cpp
    integ/Integration.cpp

    interface/mfem_tribol.cpp
    interface/simple_tribol.cpp
    interface/tribol.cpp

    mesh/CouplingScheme.cpp
    mesh/InterfacePairs.cpp
    mesh/MeshData.cpp
    mesh/MethodCouplingData.cpp
    mesh/MfemData.cpp

    physics/AlignedMortar.cpp
    physics/CommonPlane.cpp
    physics/Mortar.cpp
    physics/Physics.cpp
     
    search/InterfacePairFinder.cpp

    utils/ContactPlaneOutput.cpp
    utils/Math.cpp
    utils/TestUtils.cpp
    )

if (ENABLE_FORTRAN)
    ## append shroud-generated headers/sources for tribol_simple interface
    list(APPEND tribol_headers interface/c_fortran/wrapTRIBOL_SIMPLE.h)
    list(APPEND tribol_headers interface/c_fortran/typesTRIBOL_SIMPLE.h)

    list(APPEND tribol_sources interface/c_fortran/wrapTRIBOL_SIMPLE.cpp)
    list(APPEND tribol_sources interface/c_fortran/wrapftribol_simple.F)

    ## append shroud-generated headers/sources for TestMesh interface
    list(APPEND tribol_headers interface/c_fortran/wrapTestMesh.h)
    list(APPEND tribol_headers interface/c_fortran/typesTRIBOL_TEST_MESH.h)

    list(APPEND tribol_sources interface/c_fortran/wrapTestMesh.cpp)
    list(APPEND tribol_sources interface/c_fortran/wrapTRIBOL_TEST_MESH.cpp)
    list(APPEND tribol_sources interface/c_fortran/wrapftribol_test_mesh.F)
endif()

## setup the dependency list for tribol
set(tribol_depends tribol_shared axom::primal axom::slic mfem)
blt_list_append(TO tribol_depends ELEMENTS blt::mpi IF TRIBOL_USE_MPI )
blt_list_append(TO tribol_depends ELEMENTS redecomp IF BUILD_REDECOMP )
blt_list_append(TO tribol_depends ELEMENTS umpire IF TRIBOL_USE_UMPIRE )
blt_list_append(TO tribol_depends ELEMENTS RAJA IF TRIBOL_USE_RAJA )
blt_list_append(TO tribol_depends ELEMENTS ClangEnzymeFlags IF TRIBOL_USE_ENZYME)
message(STATUS "Tribol library dependencies: ${tribol_depends}")

## create the library
blt_add_library(
    NAME tribol
    SOURCES ${tribol_sources}
    HEADERS ${tribol_headers}
    DEPENDS_ON ${tribol_depends} ${tribol_device_depends}
    FOLDER tribol
    )

# Add separable compilation flag
if(ENABLE_HIP)
    target_compile_options(tribol PRIVATE -fgpu-rdc)
    target_compile_options(tribol PRIVATE
        $<$<CONFIG:Debug>:
            -ggdb
        >
    )
    target_link_options(tribol PRIVATE -fgpu-rdc)
    target_link_options(tribol PRIVATE --hip-link)
endif()

# Don't propagate internal dependencies
target_link_libraries(tribol PRIVATE axom::quest)

target_include_directories(tribol PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )

## tribol install target
tribol_install( SOURCE_HEADERS ${tribol_headers} )

if(ENABLE_FORTRAN AND ENABLE_MPI)
    install(FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/tribol_simple.mod DESTINATION lib/fortran)
    install(FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/tribol_test_mesh.mod DESTINATION lib/fortran)
endif()

# Add the TRIBOL_DEBUG compile definition, if non-empty
if(TRIBOL_DEBUG_DEFINE_STRING)
  blt_add_target_definitions(TO tribol SCOPE PUBLIC TARGET_DEFINITIONS "${TRIBOL_DEBUG_DEFINE_STRING}")
  # Define AXOM_DEBUG for SLIC macros
  blt_add_target_definitions(TO tribol SCOPE PUBLIC TARGET_DEFINITIONS "${AXOM_DEBUG_DEFINE_STRING}")
endif()
