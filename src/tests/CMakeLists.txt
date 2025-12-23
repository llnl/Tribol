# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Add single source tests
#------------------------------------------------------------------------------
set( tribol_tests
     tribol_check_tpl.cpp
     tribol_common_plane_penalty.cpp
     tribol_common_plane_gap_rate.cpp
     tribol_common_plane_gpu.cpp
     tribol_comp_geom_2d.cpp
     tribol_comp_geom_3d.cpp
     tribol_coupling_scheme.cpp
     tribol_coupling_scheme_manager.cpp
     tribol_enforcement_options.cpp
     tribol_execution_modes.cpp
     tribol_hex_mesh.cpp
     tribol_inv_iso.cpp
     tribol_iso_integ.cpp
     tribol_math.cpp
     tribol_mortar_data_geom.cpp
     tribol_mortar_data_weights.cpp
     tribol_mortar_force.cpp
     tribol_mortar_gap.cpp
     tribol_mortar_jacobian.cpp
     tribol_mortar_lm_patch_test.cpp
     tribol_mortar_sparse_weights.cpp
     tribol_mortar_wts.cpp
     tribol_nodal_nrmls.cpp
     tribol_quad_integ.cpp
     tribol_tet_mesh.cpp
     tribol_timestep_vote.cpp
     tribol_twb_integ.cpp
     )

set(test_depends tribol gtest)

foreach( test ${tribol_tests} )

  get_filename_component( test_name ${test} NAME_WE )

  blt_add_executable(
    NAME ${test_name}_test
    SOURCES ${test}
    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    DEPENDS_ON ${test_depends} ${tribol_device_depends}
    FOLDER tribol/tests )

  blt_add_test( NAME ${test_name}
                COMMAND ${test_name}_test )

  if (ENABLE_CUDA)
    set_target_properties(${test_name}_test PROPERTIES CUDA_SEPARABLE_COMPILATION On)
  endif()

  if (ENABLE_HIP)
    target_compile_options(${test_name}_test PRIVATE -fgpu-rdc)
    target_compile_options(${test_name}_test PRIVATE
        $<$<CONFIG:Debug>:
            -ggdb
        >
    )
    target_link_options(${test_name}_test PRIVATE -fgpu-rdc)
    target_link_options(${test_name}_test PRIVATE --hip-link)
  endif()

endforeach()

#------------------------------------------------------------------------------
# Add redecomp tests
#------------------------------------------------------------------------------
if ( BUILD_REDECOMP AND TRIBOL_USE_MPI )

  set( redecomp_tests
      redecomp_transfer.cpp
      redecomp_multitransfer.cpp
      redecomp_massmatrix.cpp
      redecomp_rectmatrix.cpp
      redecomp_sparsematrix.cpp
      )

  set(redecomp_test_depends tribol gtest)

  foreach( test ${redecomp_tests} )

    get_filename_component( test_name ${test} NAME_WE )

    blt_add_executable(
      NAME ${test_name}_test
      SOURCES ${test}
      OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
      DEPENDS_ON ${redecomp_test_depends} ${tribol_device_depends}
      FOLDER redecomp/tests )
    
    # NOTE: just do 2 ranks for now until we make the tests smaller
    #foreach ( nranks RANGE 1 3 )
    foreach ( nranks RANGE 2 2 )
      blt_add_test( NAME ${test_name}_${nranks}rank
                    COMMAND ${test_name}_test
                    NUM_MPI_TASKS ${nranks} )
    endforeach()

    if (ENABLE_CUDA)
      set_target_properties(${test_name}_test PROPERTIES CUDA_SEPARABLE_COMPILATION On)
    endif()

  endforeach()

endif()

#------------------------------------------------------------------------------
# Add tribol + redecomp tests
#------------------------------------------------------------------------------
if ( BUILD_REDECOMP AND TRIBOL_USE_MPI )

  set( combined_tests
      tribol_mfem_common_plane.cpp
      tribol_mfem_mortar_lm.cpp
      tribol_proximity_check.cpp
      tribol_redecomp_tol.cpp
      )

  set(combined_test_depends tribol gtest)

  foreach( test ${combined_tests} )

    get_filename_component( test_name ${test} NAME_WE )

    blt_add_executable(
      NAME ${test_name}_test
      SOURCES ${test}
      OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
      DEPENDS_ON ${combined_test_depends} ${tribol_device_depends}
      FOLDER tribol/tests )

    # NOTE: just do 2 ranks for now until we make the tests smaller
    #foreach ( nranks RANGE 1 3 )
    foreach ( nranks RANGE 2 2 )
      blt_add_test( NAME ${test_name}_${nranks}rank
                    COMMAND ${test_name}_test
                    NUM_MPI_TASKS ${nranks} )
    endforeach()

    if (ENABLE_CUDA)
      set_target_properties(${test_name}_test PROPERTIES CUDA_SEPARABLE_COMPILATION On)
    endif()
    
    if (ENABLE_HIP)
      target_compile_options(${test_name}_test PRIVATE -fgpu-rdc)
      target_compile_options(${test_name}_test PRIVATE
          $<$<CONFIG:Debug>:
              -ggdb
          >
      )
      target_link_options(${test_name}_test PRIVATE -fgpu-rdc)
      target_link_options(${test_name}_test PRIVATE --hip-link)
    endif()

  endforeach()

endif()

#------------------------------------------------------------------------------
# Add tribol + enzyme tests
#------------------------------------------------------------------------------

if( TRIBOL_USE_ENZYME )

  set( combined_tests
      tribol_enzyme_element_mortar.cpp
      tribol_enzyme_nodal_normal.cpp
      tribol_enzyme_mortar_assembled.cpp
      tribol_enzyme_poly_intersect.cpp
      )

  set(combined_test_depends tribol gtest)

  foreach( test ${combined_tests} )

    get_filename_component( test_name ${test} NAME_WE )

    blt_add_executable(
      NAME ${test_name}_test
      SOURCES ${test}
      OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
      DEPENDS_ON ${combined_test_depends} ${tribol_device_depends}
      FOLDER tribol/tests )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name}_test )

    if (ENABLE_CUDA)
      set_target_properties(${test_name}_test PROPERTIES CUDA_SEPARABLE_COMPILATION On)
    endif()
    
    if (ENABLE_HIP)
      target_compile_options(${test_name}_test PRIVATE -fgpu-rdc)
      target_compile_options(${test_name}_test PRIVATE
          $<$<CONFIG:Debug>:
              -ggdb
          >
      )
      target_link_options(${test_name}_test PRIVATE -fgpu-rdc)
      target_link_options(${test_name}_test PRIVATE --hip-link)
    endif()

  endforeach()

endif()

#------------------------------------------------------------------------------
# Add smoke tests
#------------------------------------------------------------------------------

if( MFEM_FOUND )

  set(mfem_smoke_depends mfem gtest)
  # MFEM's openmp dependency isn't being picked up by FindMFEM.cmake. Force it to be added for now.
  blt_list_append(TO mfem_smoke_depends ELEMENTS blt::openmp IF TRIBOL_USE_OPENMP )
  blt_add_executable(
    NAME mfem_smoke_test
    SOURCES mfem_smoke.cpp
    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    DEPENDS_ON ${mfem_smoke_depends} ${tribol_device_depends}
    FOLDER tribol/tests )

  blt_add_test( NAME mfem_smoke_test
                COMMAND mfem_smoke_test )

  if (ENABLE_CUDA)
    set_target_properties(mfem_smoke_test PROPERTIES CUDA_SEPARABLE_COMPILATION On)
  endif()

endif()

if( TRIBOL_USE_ENZYME )

  set(enzyme_smoke_depends ClangEnzymeFlags gtest)
  blt_add_executable(
    NAME enzyme_smoke_test
    SOURCES enzyme_smoke.cpp
    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    DEPENDS_ON ${enzyme_smoke_depends} ${tribol_device_depends}
    FOLDER tribol/tests )

  blt_add_test( NAME enzyme_smoke_test
                COMMAND enzyme_smoke_test )

  if (ENABLE_CUDA)
    set_target_properties(enzyme_smoke_test PROPERTIES CUDA_SEPARABLE_COMPILATION On)
  endif()

endif()

if( TRIBOL_USE_CALIPER )

  set(caliper_smoke_depends caliper gtest)
  blt_add_executable(
    NAME caliper_smoke_test
    SOURCES caliper_smoke.cpp
    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    DEPENDS_ON ${caliper_smoke_depends} ${tribol_device_depends}
    FOLDER tribol/tests )

  blt_add_test( NAME caliper_smoke_test
                COMMAND caliper_smoke_test )

  if (ENABLE_CUDA)
    set_target_properties(caliper_smoke_test PROPERTIES CUDA_SEPARABLE_COMPILATION On)
  endif()

endif()

