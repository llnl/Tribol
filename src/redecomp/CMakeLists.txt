# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

## list of headers
set( redecomp_headers

     redecomp.hpp
     MultiRedecomp.hpp
     RedecompMesh.hpp
     RedecompTransfer.hpp
     common/TypeDefs.hpp
     partition/PartitionElements.hpp
     partition/PartitionEntity.hpp
     partition/Partitioner.hpp
     partition/PartitionMethod.hpp
     partition/RCB.hpp
     transfer/GridFnTransfer.hpp
     transfer/MatrixTransfer.hpp
     transfer/SparseMatrixTransfer.hpp
     transfer/TransferByElements.hpp
     transfer/TransferByNodes.hpp
     utils/ArrayUtility.hpp
     utils/BisecTree.hpp
     utils/MPIArray.hpp
     utils/MPIUtility.hpp
     )

## list of sources
set( redecomp_sources

     MultiRedecomp.cpp
     RedecompMesh.cpp
     RedecompTransfer.cpp
     partition/PartitionElements.cpp
     partition/Partitioner.cpp
     partition/PartitionMethod.cpp
     partition/RCB.cpp
     transfer/MatrixTransfer.cpp
     transfer/SparseMatrixTransfer.cpp
     transfer/TransferByElements.cpp
     transfer/TransferByNodes.cpp
     utils/MPIUtility.cpp
     )

## setup the dependency list for redecomp
set(redecomp_depends tribol_shared axom::primal mfem)
blt_list_append(TO redecomp_depends ELEMENTS blt::mpi IF TRIBOL_USE_MPI )

## create the library
blt_add_library(
    NAME redecomp
    SOURCES ${redecomp_sources}
    HEADERS ${redecomp_headers}
    DEPENDS_ON ${redecomp_depends} ${tribol_device_depends}
    FOLDER redecomp
    )

target_include_directories(redecomp PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    )         

## mirror the directory structure on the install
foreach( redecomp_header ${redecomp_headers} )
  get_filename_component( redecomp_base_dir ${redecomp_header} DIRECTORY)
  install( FILES ${redecomp_header}
            DESTINATION "include/redecomp/${redecomp_base_dir}" )
endforeach()

install(TARGETS     redecomp
        EXPORT      tribol-targets
        DESTINATION lib
        )

# Add the TRIBOL_DEBUG compile definition, if non-empty
if(TRIBOL_DEBUG_DEFINE_STRING)
  blt_add_target_definitions(TO redecomp SCOPE PUBLIC TARGET_DEFINITIONS "${TRIBOL_DEBUG_DEFINE_STRING}")
  # Define AXOM_DEBUG for SLIC macros
  blt_add_target_definitions(TO redecomp SCOPE PUBLIC TARGET_DEFINITIONS "${AXOM_DEBUG_DEFINE_STRING}")
endif()
