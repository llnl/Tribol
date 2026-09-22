#include "LegacyCudaCompatibility.hpp"

#if CUDART_VERSION >= 13000
extern "C" cudaError_t cudaGetDeviceProperties_v2( cudaDeviceProp* properties, int device )
{
  return cudaGetDeviceProperties( properties, device );
}
#endif
