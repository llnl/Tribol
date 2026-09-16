set(tribol_core_headers
    ArrayView.hpp
    Config.hpp
    ExactTangent.hpp
    MeshView.hpp
    Version.hpp
    ../adapters/array/ArrayContact.hpp
    ../basis/LinearBasis.hpp
    ../contact/Contact.hpp
    ../constraint/NormalConstraint.hpp
    ../evaluation/PolicyEvaluator.hpp
    ../evaluation/PointwisePenalty.hpp
    ../evaluation/State.hpp
    ../execution/Execution.hpp
    ../execution/ContactCuda.hpp
    ../execution/CudaPenalty.hpp
    ../execution/PenaltyKernel.hpp
    ../geom/ConformingOverlap.hpp
    ../geom/ProjectedOverlap.hpp
    ../integration/InteractionQuadrature.hpp
    ../integration/Quadrature.hpp
    ../method/Method.hpp
    ../method/Traits.hpp
    ../method/Validation.hpp
    ../search/Search.hpp
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

if(ENABLE_CUDA)
  blt_add_library(
    NAME
    tribol_cuda_execution
    SOURCES
    ../execution/CudaPenalty.cu
    HEADERS
    ../execution/CudaPenalty.hpp
    DEPENDS_ON
    tribol_core
    ${tribol_device_depends}
    FOLDER
    tribol/execution)
  add_library(tribol::cuda_execution ALIAS tribol_cuda_execution)
  set(_cuda tribol_cuda_execution)
  set_target_properties(${_cuda} PROPERTIES EXPORT_NAME tribol::cuda_execution)
  target_compile_features(tribol_cuda_execution PUBLIC cxx_std_20)
  set_target_properties(tribol_cuda_execution
                        PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(
    tribol_cuda_execution
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_include_directories(
    tribol_cuda_execution
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
           $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
           $<INSTALL_INTERFACE:include>)
  install(TARGETS tribol_cuda_execution EXPORT tribol-targets)
endif()
