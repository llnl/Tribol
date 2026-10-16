# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

## examples
set( contact_examples
     common_plane_gpu.cpp
     common_plane.cpp
     mortar_lm_patch_test.cpp
     )


set(example_depends tribol)

foreach( example ${contact_examples} )

  get_filename_component( example_name ${example} NAME_WE )

  blt_add_executable(
      NAME       ${example_name}_ex
      SOURCES    ${example}
      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
      DEPENDS_ON ${example_depends} ${tribol_device_depends}
      )

  if (ENABLE_HIP)
    target_compile_options(${example_name}_ex PRIVATE
        $<$<CONFIG:Debug>:
            -ggdb
        >
    )
  endif()

endforeach()

## Run examples as tests, when testing enabled
if ( TRIBOL_ENABLE_TESTS AND ENABLE_TESTS )

  foreach( example common_plane mortar_lm_patch_test)
    if(TRIBOL_USE_MPI)
      blt_add_test( NAME    tribol_${example}_ex
                    COMMAND ${example}_ex 
                    NUM_MPI_TASKS 1)
    else()
      blt_add_test( NAME    tribol_${example}_ex
                    COMMAND ${example}_ex )
    endif()
  endforeach() 

endif()

if ( BUILD_REDECOMP AND TRIBOL_USE_MPI )
  set( operator_examples
      element_exchange.cpp
      mfem_common_plane.cpp
      mfem_mortar_lm_patch.cpp
      )

  foreach( example ${operator_examples} )
    get_filename_component( example_name ${example} NAME_WE )
    blt_add_executable(
        NAME       ${example_name}_ex
        SOURCES    ${example}
        OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
        DEPENDS_ON tribol ${tribol_device_depends}
        )
  endforeach()

  if ( TRIBOL_ENABLE_TESTS AND ENABLE_TESTS )
    foreach( example element_exchange mfem_common_plane mfem_mortar_lm_patch )
      blt_add_test( NAME          tribol_${example}_ex
                    COMMAND       ${example}_ex
                    NUM_MPI_TASKS 2 )
    endforeach()
  endif()
endif()

# Define fortran examples, for now, this requires MPI
if(ENABLE_FORTRAN AND TRIBOL_USE_MPI)

  blt_add_executable(
    NAME       simple_tribol_interface_ex
    SOURCES    simple_tribol_interface.F
    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
    DEPENDS_ON tribol ${tribol_device_depends}
    )

  if (ENABLE_HIP)
    target_compile_options(simple_tribol_interface_ex PRIVATE
        $<$<CONFIG:Debug>:
            -ggdb
        >
    )
  endif()

endif()
