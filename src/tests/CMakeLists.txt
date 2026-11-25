# Copyright (c) Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Add single source tests
#------------------------------------------------------------------------------
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  # NVCC's device pass reports GoogleTest registration data and host-only concept test types in anonymous namespaces as
  # unused. The host compiler still checks unused test code, so suppress this device-pass false positive for GoogleTest
  # consumers without changing production warning diagnostics.
  target_compile_options(gtest INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>")
endif()

set( tribol_tests
     tribol_energy_mortar_pair_evaluation.cpp
     tribol_basis_function.cpp
     tribol_check_tpl.cpp
     tribol_common_plane_3d.cpp
     tribol_common_plane_pair_evaluation.cpp
     tribol_common_plane_penalty.cpp
     tribol_common_plane_gap_rate.cpp
     tribol_common_plane_gpu.cpp
     tribol_comp_geom_2d.cpp
     tribol_comp_geom_3d.cpp
     tribol_coarse_search.cpp
     tribol_coupling_scheme.cpp
     tribol_coupling_scheme_manager.cpp
     tribol_enforcement_options.cpp
     tribol_execution_modes.cpp
     tribol_hex_mesh.cpp
     tribol_pair_quadrature.cpp
     tribol_refined_search.cpp
     tribol_interface_pair_checks.cpp
     tribol_interface_pair_finder.cpp
     tribol_inv_iso.cpp
     tribol_iso_integ.cpp
     tribol_math.cpp
     tribol_mdarray.cpp
     tribol_mdspan.cpp
     tribol_mesh_data.cpp
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
     tribol_residual_gap.cpp
     tribol_tet_mesh.cpp
     tribol_timestep_calculator.cpp
     tribol_timestep_vote.cpp
     tribol_twb_integ.cpp
     tribol_version.cpp
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
                COMMAND ${test_name}_test
                NUM_MPI_TASKS 1 )

  if (ENABLE_HIP)
    target_compile_options(${test_name}_test PRIVATE
        $<$<CONFIG:Debug>:
            -ggdb
        >
    )
  endif()

endforeach()

if(TRIBOL_USE_ENZYME)
  # Enzyme rejects the aggregate-heavy geometry IR produced at -O0. These tests differentiate pair functions.
  target_compile_options(tribol_energy_mortar_pair_evaluation_test PRIVATE "$<$<CONFIG:Debug>:-O2>")
  target_compile_options(tribol_common_plane_pair_evaluation_test PRIVATE "$<$<CONFIG:Debug>:-O2>")
  target_compile_options(tribol_common_plane_3d_test PRIVATE "$<$<CONFIG:Debug>:-O2>")
endif()

#------------------------------------------------------------------------------
# Add shared tests
#------------------------------------------------------------------------------
if ( TRIBOL_USE_MPI )

  set( shared_tests
       shared_par_sparse_mat.cpp
       shared_par_vector.cpp
     )

  set(mpi_test_depends tribol_shared gtest)

  foreach( test ${shared_tests} )

    get_filename_component( test_name ${test} NAME_WE )

    blt_add_executable(
      NAME ${test_name}_test
      SOURCES ${test}
      OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
      DEPENDS_ON ${mpi_test_depends} ${tribol_device_depends}
      FOLDER tribol/tests )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name}_test
                  NUM_MPI_TASKS 2 )

    if (ENABLE_HIP)
      target_compile_options(${test_name}_test PRIVATE
          $<$<CONFIG:Debug>:
              -ggdb
          >
      )
    endif()

  endforeach()

endif()

#------------------------------------------------------------------------------
# Add redecomp tests
#------------------------------------------------------------------------------
if ( BUILD_REDECOMP AND TRIBOL_USE_MPI )

  set( redecomp_tests
      redecomp_transfer.cpp
      redecomp_multitransfer.cpp
      redecomp_massmatrix.cpp
      redecomp_matrixtransfer_equivalence.cpp
      redecomp_rectmatrix.cpp
      redecomp_sparsematrix.cpp
      redecomp_gpu_mpi.cpp
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
      target_compile_options(${test_name}_test PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
      )
    endif()
  endforeach()

endif()

#------------------------------------------------------------------------------
# Add tribol + redecomp tests
#------------------------------------------------------------------------------
if ( BUILD_REDECOMP AND TRIBOL_USE_MPI )

  set( combined_tests
      tribol_mfem_residual_and_jacobian_assembly.cpp
      tribol_mfem_common_plane_formulation.cpp
      tribol_mfem_data.cpp
      tribol_mfem_common_plane.cpp
      tribol_mfem_mortar_lm.cpp
      tribol_mfem_jacobian.cpp
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
    
    if (ENABLE_HIP)
      target_compile_options(${test_name}_test PRIVATE
          $<$<CONFIG:Debug>:
              -ggdb
          >
      )
    endif()

  endforeach()

  blt_add_test(NAME tribol_mfem_residual_and_jacobian_assembly
      COMMAND tribol_mfem_residual_and_jacobian_assembly_test NUM_MPI_TASKS 1)
  if(TRIBOL_USE_ENZYME)
    # Direct residual evaluators instantiate Enzyme's geometry derivatives in this translation unit.
    target_compile_options(tribol_mfem_common_plane_formulation_test PRIVATE "$<$<CONFIG:Debug>:-O2>")
  endif()

  blt_add_test(NAME tribol_mfem_common_plane_formulation
      COMMAND tribol_mfem_common_plane_formulation_test NUM_MPI_TASKS 1)

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
      tribol_mfem_energy_mortar_formulation.cpp
      tribol_energy_mortar_patch.cpp
      tribol_energy_mortar_patch_LM.cpp
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

    if (ENABLE_HIP)
      target_compile_options(${test_name}_test PRIVATE
          $<$<CONFIG:Debug>:
              -ggdb
          >
      )
    endif()

  endforeach()

  # Enzyme rejects the aggregate-heavy edge geometry IR produced at -O0.
  target_compile_options(tribol_mfem_energy_mortar_formulation_test PRIVATE "$<$<CONFIG:Debug>:-O2>")

endif()

#------------------------------------------------------------------------------
# Add smoke tests
#------------------------------------------------------------------------------

if( MFEM_FOUND )

  set(mfem_smoke_depends mfem gtest)
  blt_add_executable(
    NAME mfem_smoke_test
    SOURCES mfem_smoke.cpp
    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    DEPENDS_ON ${mfem_smoke_depends} ${tribol_device_depends}
    FOLDER tribol/tests )

  blt_add_test( NAME mfem_smoke_test
                COMMAND mfem_smoke_test )

endif()

if( TRIBOL_USE_ENZYME )

  if( TRIBOL_USE_MPI )
    blt_add_test(NAME tribol_mfem_energy_mortar_formulation_2rank
        COMMAND tribol_mfem_energy_mortar_formulation_test NUM_MPI_TASKS 2)
  endif()

  set(enzyme_smoke_depends ClangEnzymeFlags gtest)
  blt_add_executable(
    NAME enzyme_smoke_test
    SOURCES enzyme_smoke.cpp
    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    DEPENDS_ON ${enzyme_smoke_depends} ${tribol_device_depends}
    FOLDER tribol/tests )

  blt_add_test( NAME enzyme_smoke_test
                COMMAND enzyme_smoke_test )

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

endif()
