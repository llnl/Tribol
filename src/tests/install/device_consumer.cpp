#include "tribol/Tribol.hpp"

#include <type_traits>

// Requirements: PAR-006, QUALITY-001

int main()
{
#if defined( TRIBOL_CONSUMER_USE_CUDA )
  using Execution = tribol::execution::Cuda;
  const bool available = tribol::execution::cudaDeviceAvailable();
#elif defined( TRIBOL_CONSUMER_USE_HIP )
  using Execution = tribol::execution::Hip;
  const bool available = tribol::execution::hipDeviceAvailable();
#else
#error "The device install consumer requires a configured backend."
#endif
  static_assert( tribol::execution::DevicePolicy<Execution> );
  static_cast<void>( available );
  return 0;
}
