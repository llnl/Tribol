# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)


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

if(ENABLE_HIP)
  set_source_files_properties(parallel/execution_cuda.cu PROPERTIES LANGUAGE CXX)
  blt_add_executable(
    NAME tribol_execution_hip_test
    SOURCES parallel/execution_cuda.cu
    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    DEPENDS_ON tribol::hip_execution ${tribol_device_depends}
    FOLDER tribol/parallel)
  target_compile_features(tribol_execution_hip_test PRIVATE cxx_std_20)
  target_compile_definitions(tribol_execution_hip_test PRIVATE TRIBOL_TEST_USE_HIP)

  blt_add_test(
    NAME tribol_execution_hip
    COMMAND tribol_execution_hip_test)
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
