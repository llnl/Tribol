set(tribol_core_headers
    ArrayView.hpp
    Config.hpp
    ExactTangent.hpp
    MeshView.hpp
    StaticGradient.hpp
    Version.hpp
    ../adapters/array/ArrayContact.hpp
    ../basis/LinearBasis.hpp
    ../contact/Contact.hpp
    ../contact/ContactEvaluation.inl
    ../contact/ContactExactHelpers.inl
    ../contact/ContactLinearization.inl
    ../constraint/NormalConstraint.hpp
    ../constraint/PenetrationLimit.hpp
    ../diagnostics/ContactOutput.hpp
    ../diagnostics/ScheduledOutput.hpp
    ../evaluation/PolicyEvaluator.hpp
    ../evaluation/PointwisePenalty.hpp
    ../evaluation/State.hpp
    ../evaluation/VariationalEvaluator.inl
    ../execution/Execution.hpp
    ../execution/ContactDevice.hpp
    ../execution/ContactCuda.hpp
    ../execution/ContactOpenMP.hpp
    ../execution/CudaContact.hpp
    ../execution/CudaPenalty.hpp
    ../execution/DeviceContact.hpp
    ../execution/DevicePenalty.hpp
    ../execution/HipContact.hpp
    ../execution/HipPenalty.hpp
    ../execution/PenaltyKernel.hpp
    ../geom/ConformingOverlap.hpp
    ../geom/ProjectedOverlap.hpp
    ../geom/SmoothedSegmentOverlap.hpp
    ../integration/InteractionQuadrature.hpp
    ../integration/Quadrature.hpp
    ../method/Method.hpp
    ../method/MethodCompatibility.inl
    ../method/Traits.hpp
    ../method/Validation.hpp
    ../search/Proximity.hpp
    ../search/Search.hpp
    ../timestep/KinematicVote.hpp
    ../Tribol.hpp)

blt_add_library(NAME tribol_core HEADERS ${tribol_core_headers} FOLDER
                tribol/core)

add_library(tribol::core ALIAS tribol_core)
set_target_properties(tribol_core PROPERTIES EXPORT_NAME tribol::core)
target_compile_features(tribol_core INTERFACE cxx_std_20)
target_include_directories(
  tribol_core
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>)

if(ENABLE_OPENMP)
  target_link_libraries(tribol_core INTERFACE blt::openmp)
endif()

install(TARGETS tribol_core EXPORT tribol-targets)
foreach(tribol_core_header ${tribol_core_headers})
  get_filename_component(_header_parent "${tribol_core_header}" DIRECTORY)
  if(_header_parent STREQUAL "..")
    set(_install_subdirectory "")
  elseif(_header_parent MATCHES "^\\.\\./")
    string(REGEX REPLACE "^\\.\\./" "" _install_subdirectory
                         "${_header_parent}")
  else()
    set(_install_subdirectory "core/${_header_parent}")
  endif()
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${tribol_core_header}"
          DESTINATION "include/tribol/${_install_subdirectory}")
endforeach()

if(ENABLE_CUDA OR ENABLE_HIP)
  if(ENABLE_CUDA)
    set(_tribol_device_sources
        ../execution/CudaContact.cu
        ../execution/CudaPenalty.cu)
  else()
    set(_tribol_device_sources
        ../execution/HipContact.cpp
        ../execution/HipPenalty.cpp)
  endif()

  blt_add_library(
    NAME
    tribol_device_execution
    SOURCES
    ${_tribol_device_sources}
    HEADERS
    ../execution/CudaContact.hpp
    ../execution/CudaPenalty.hpp
    ../execution/DeviceContact.hpp
    ../execution/DevicePenalty.hpp
    ../execution/HipContact.hpp
    ../execution/HipPenalty.hpp
    DEPENDS_ON
    tribol_core
    ${tribol_device_depends}
    FOLDER
    tribol/execution)
  add_library(tribol::device_execution ALIAS tribol_device_execution)
  set_target_properties(tribol_device_execution PROPERTIES EXPORT_NAME tribol::device_execution)
  target_compile_features(tribol_device_execution PUBLIC cxx_std_20)
  if(ENABLE_CUDA)
    target_compile_features(tribol_device_execution PUBLIC cuda_std_20)
    set_target_properties(
      tribol_device_execution
      PROPERTIES CUDA_STANDARD 20
                 CUDA_STANDARD_REQUIRED ON
                 CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_options(
      tribol_device_execution
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  endif()
  target_include_directories(
    tribol_device_execution
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
           $<INSTALL_INTERFACE:include>)
  install(TARGETS tribol_device_execution EXPORT tribol-targets)

  if(ENABLE_CUDA)
    add_library(tribol_cuda_execution INTERFACE)
    target_link_libraries(tribol_cuda_execution INTERFACE tribol_device_execution)
    add_library(tribol::cuda_execution ALIAS tribol_cuda_execution)
    set_target_properties(tribol_cuda_execution PROPERTIES EXPORT_NAME tribol::cuda_execution)
    install(TARGETS tribol_cuda_execution EXPORT tribol-targets)
  else()
    add_library(tribol_hip_execution INTERFACE)
    target_link_libraries(tribol_hip_execution INTERFACE tribol_device_execution)
    add_library(tribol::hip_execution ALIAS tribol_hip_execution)
    set_target_properties(tribol_hip_execution PROPERTIES EXPORT_NAME tribol::hip_execution)
    install(TARGETS tribol_hip_execution EXPORT tribol-targets)
  endif()
endif()
