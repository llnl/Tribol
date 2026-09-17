set(_tribol_mfem_adapter_headers
    MfemContact.hpp
    MfemContactAssembly.inl
    MfemContactMapping.inl
    MfemDistribution.hpp
    MfemSurface.hpp)
set(_tribol_mfem_adapter_depends tribol_core mfem)
blt_list_append(TO _tribol_mfem_adapter_depends ELEMENTS blt::mpi IF
                TRIBOL_USE_MPI)
blt_list_append(TO _tribol_mfem_adapter_depends ELEMENTS tribol_cuda_execution
                IF ENABLE_CUDA)

blt_add_library(
  NAME
  tribol_mfem_adapter
  HEADERS
  ${_tribol_mfem_adapter_headers}
  DEPENDS_ON
  ${_tribol_mfem_adapter_depends}
  FOLDER
  tribol/adapters)

add_library(tribol::mfem_adapter ALIAS tribol_mfem_adapter)
set(_target tribol_mfem_adapter)
set_target_properties(${_target} PROPERTIES EXPORT_NAME tribol::mfem_adapter)
target_compile_features(tribol_mfem_adapter INTERFACE cxx_std_20)
target_include_directories(
  tribol_mfem_adapter
  INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>)

install(TARGETS tribol_mfem_adapter EXPORT tribol-targets)
install(FILES ${_tribol_mfem_adapter_headers}
        DESTINATION include/tribol/adapters/mfem)
