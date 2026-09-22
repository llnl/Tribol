#ifndef TRIBOL_BENCHMARKS_REFERENCE_LEGACYCUDACOMPATIBILITY_HPP_
#define TRIBOL_BENCHMARKS_REFERENCE_LEGACYCUDACOMPATIBILITY_HPP_

#include "mfem/config/config.hpp"

#undef MFEM_USE_ENZYME

#include <cuda_runtime_api.h>

#if CUDART_VERSION >= 13000
inline cudaError_t tribolBenchmarkCudaMemAdvise( const void* pointer, std::size_t bytes, cudaMemoryAdvise advice,
                                                 int device )
{
  cudaMemLocation location{};
  location.type = device == cudaCpuDeviceId ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
  location.id = device == cudaCpuDeviceId ? 0 : device;
  return ::cudaMemAdvise( pointer, bytes, advice, location );
}

#define cudaMemAdvise tribolBenchmarkCudaMemAdvise
#endif

#endif
