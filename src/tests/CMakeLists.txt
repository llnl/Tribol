# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Add single source tests
#------------------------------------------------------------------------------
if(TRIBOL_ENABLE_LEGACY)

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
     tribol_residual_gap.cpp
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
      set_target_properties(${test_name}_test PROPERTIES CUDA_SEPARABLE_COMPILATION On)
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
      tribol_energy_mortar_patch.cpp
      tribol_energy_mortar_patch_LM.cpp
      tribol_finite_diff_energy_mortar.cpp
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
      target_compile_options(${test_name}_test PRIVATE
          $<$<CONFIG:Debug>:
              -ggdb
          >
      )
    endif()

  endforeach()

endif()

#------------------------------------------------------------------------------
# End legacy implementation tests
#------------------------------------------------------------------------------
endif()

#------------------------------------------------------------------------------
# Add policy-composed contact tests
#------------------------------------------------------------------------------

blt_add_executable(
  NAME tribol_interface_spec_test
  SOURCES spec/tribol_interface_spec.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/spec)

blt_add_test(
  NAME tribol_interface_spec
  COMMAND tribol_interface_spec_test)

blt_add_executable(
  NAME tribol_projected_overlap_test
  SOURCES components/projected_overlap.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/components)

blt_add_test(
  NAME tribol_projected_overlap
  COMMAND tribol_projected_overlap_test)

blt_add_executable(
  NAME tribol_contact_diagnostics_test
  SOURCES components/contact_diagnostics.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/components)

blt_add_test(
  NAME tribol_contact_diagnostics
  COMMAND tribol_contact_diagnostics_test)

blt_add_executable(
  NAME tribol_timestep_policy_test
  SOURCES components/timestep_vote.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/components)

blt_add_test(
  NAME tribol_timestep_policy
  COMMAND tribol_timestep_policy_test)

blt_add_executable(
  NAME tribol_legacy_common_plane_physics_test
  SOURCES legacy_parity/common_plane_physics.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/legacy-parity)

blt_add_test(
  NAME tribol_legacy_common_plane_physics
  COMMAND tribol_legacy_common_plane_physics_test)

blt_add_executable(
  NAME tribol_legacy_auto_contact_physics_test
  SOURCES legacy_parity/auto_contact_physics.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/legacy-parity)

blt_add_test(
  NAME tribol_legacy_auto_contact_physics
  COMMAND tribol_legacy_auto_contact_physics_test)

blt_add_executable(
  NAME tribol_legacy_mortar_energy_physics_test
  SOURCES legacy_parity/mortar_energy_physics.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/legacy-parity)

blt_add_test(
  NAME tribol_legacy_mortar_energy_physics
  COMMAND tribol_legacy_mortar_energy_physics_test)

blt_add_executable(
  NAME tribol_search_policies_test
  SOURCES components/search_policies.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/components)

blt_add_test(
  NAME tribol_search_policies
  COMMAND tribol_search_policies_test)

blt_add_executable(
  NAME tribol_constraint_stages_test
  SOURCES components/constraint_stages.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/components)

blt_add_test(
  NAME tribol_constraint_stages
  COMMAND tribol_constraint_stages_test)

blt_add_executable(
  NAME tribol_method_conformance_test
  SOURCES components/method_conformance.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/components)

blt_add_test(
  NAME tribol_method_conformance
  COMMAND tribol_method_conformance_test)

blt_add_executable(
  NAME tribol_capability_matrix_test
  SOURCES components/capability_matrix.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/components)

blt_add_test(
  NAME tribol_capability_matrix
  COMMAND tribol_capability_matrix_test)

blt_add_executable(
  NAME tribol_array_adapter_test
  SOURCES adapters/array_contact.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/adapters)

blt_add_test(
  NAME tribol_array_adapter
  COMMAND tribol_array_adapter_test)

blt_add_executable(
  NAME tribol_linearization_test
  SOURCES components/linearization.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/components)

blt_add_test(
  NAME tribol_linearization
  COMMAND tribol_linearization_test)

blt_add_executable(
  NAME tribol_state_linearization_test
  SOURCES components/state_linearization.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/components)

blt_add_test(
  NAME tribol_state_linearization
  COMMAND tribol_state_linearization_test)

blt_add_executable(
  NAME tribol_pointwise_policies_test
  SOURCES components/pointwise_policies.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/components)

blt_add_test(
  NAME tribol_pointwise_policies
  COMMAND tribol_pointwise_policies_test)

blt_add_executable(
  NAME tribol_execution_policies_test
  SOURCES parallel/execution_policies.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/parallel)

blt_add_test(
  NAME tribol_execution_policies
  COMMAND tribol_execution_policies_test)

if(ENABLE_OPENMP)
  blt_add_executable(
    NAME tribol_execution_openmp_test
    SOURCES parallel/execution_openmp.cpp
    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    DEPENDS_ON tribol::core blt::openmp
    FOLDER tribol/parallel)

  blt_add_test(
    NAME tribol_execution_openmp
    COMMAND tribol_execution_openmp_test)
endif()

blt_add_executable(
  NAME tribol_allocation_contract_test
  SOURCES components/allocation_contract.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::core
  FOLDER tribol/components)

blt_add_test(
  NAME tribol_allocation_contract
  COMMAND tribol_allocation_contract_test)

if(ENABLE_CUDA)
  blt_add_executable(
    NAME tribol_execution_cuda_test
    SOURCES parallel/execution_cuda.cu
    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    DEPENDS_ON tribol::cuda_execution ${tribol_device_depends}
    FOLDER tribol/parallel)
  set_target_properties(tribol_execution_cuda_test PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(tribol_execution_cuda_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  blt_add_test(
    NAME tribol_execution_cuda
    COMMAND tribol_execution_cuda_test)
endif()

blt_add_executable(
  NAME tribol_mfem_adapter_test
  SOURCES adapters/mfem_contact.cpp
  OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
  DEPENDS_ON tribol::mfem_adapter ${tribol_device_depends}
  FOLDER tribol/adapters)

blt_add_test(
  NAME tribol_mfem_adapter_smoke
  COMMAND tribol_mfem_adapter_test
  NUM_MPI_TASKS 1)

blt_add_test(
  NAME tribol_mfem_adapter_2rank
  COMMAND tribol_mfem_adapter_test
  NUM_MPI_TASKS 2)

blt_add_test(
  NAME tribol_mfem_adapter_4rank
  COMMAND tribol_mfem_adapter_test
  NUM_MPI_TASKS 4)

configure_file(
  install/RunInstallConsumer.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/RunInstallConsumer.cmake
  @ONLY)

add_test(
  NAME tribol_install_consumer
  COMMAND ${CMAKE_COMMAND}
          -DTRIBOL_TEST_CONFIG=$<CONFIG>
          -P ${CMAKE_CURRENT_BINARY_DIR}/RunInstallConsumer.cmake)

if(TRIBOL_ENABLE_FUTURE)
  blt_add_executable(
    NAME tribol_future_contact_test
    SOURCES tribol_future_contact.cpp
    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    DEPENDS_ON tribol::future gtest ${tribol_device_depends}
    FOLDER tribol/future/tests)

  blt_add_test(
    NAME tribol_future_contact
    COMMAND tribol_future_contact_test
    NUM_MPI_TASKS 1)

  blt_add_test(
    NAME tribol_future_contact_2rank
    COMMAND tribol_future_contact_test
    NUM_MPI_TASKS 2)

  if(ENABLE_HIP AND TRIBOL_USE_ENZYME)
    blt_add_executable(
      NAME tribol_future_enzyme_hip_smoke_test
      SOURCES tribol_future_enzyme_hip_smoke.cpp
      OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
      DEPENDS_ON tribol::future gtest ${tribol_device_depends}
      FOLDER tribol/future/tests)
    blt_add_test(NAME tribol_future_enzyme_hip_smoke
                 COMMAND tribol_future_enzyme_hip_smoke_test)
  endif()
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
