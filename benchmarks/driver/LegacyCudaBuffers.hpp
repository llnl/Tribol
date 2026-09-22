#ifndef TRIBOL_BENCHMARKS_DRIVER_LEGACY_CUDA_BUFFERS_HPP_
#define TRIBOL_BENCHMARKS_DRIVER_LEGACY_CUDA_BUFFERS_HPP_

#include "BenchmarkMesh.hpp"

#include "tribol/common/ArrayTypes.hpp"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace tribol_benchmark::legacy_cuda {

template <typename T>
using DeviceArray = tribol::ArrayT<T, 1, tribol::MemorySpace::Device>;

inline void check( cudaError_t status, const char* operation )
{
  if ( status != cudaSuccess ) {
    throw std::runtime_error( std::string( operation ) + ": " + cudaGetErrorString( status ) );
  }
}

template <typename T>
DeviceArray<T> copyToDevice( const std::vector<T>& source )
{
  DeviceArray<T> result( static_cast<tribol::IndexT>( source.size() ) );
  check( cudaMemcpy( result.data(), source.data(), source.size() * sizeof( T ), cudaMemcpyHostToDevice ),
         "cudaMemcpy to legacy input" );
  return result;
}

template <typename T>
void copyToHost( std::vector<T>& destination, const DeviceArray<T>& source )
{
  check( cudaMemcpy( destination.data(), source.data(), destination.size() * sizeof( T ), cudaMemcpyDeviceToHost ),
         "cudaMemcpy from legacy response" );
}

struct Components {
  DeviceArray<tribol::RealT> x;
  DeviceArray<tribol::RealT> y;
  DeviceArray<tribol::RealT> z;
};

inline Components splitCoordinates( const SurfaceData<tribol::IndexT>& surface )
{
  Components result;
  result.x = copyToDevice( surface.component( 0 ) );
  result.y = copyToDevice( surface.component( 1 ) );
  if ( surface.dimension == 3 ) {
    result.z = copyToDevice( surface.component( 2 ) );
  }
  return result;
}

struct Response {
  explicit Response( int nodes ) : x( nodes ), y( nodes ), z( nodes ) { clear(); }

  DeviceArray<tribol::RealT> x;
  DeviceArray<tribol::RealT> y;
  DeviceArray<tribol::RealT> z;

  void clear()
  {
    const auto clear_array = []( auto& values ) {
      check( cudaMemset( values.data(), 0, static_cast<std::size_t>( values.size() ) * sizeof( tribol::RealT ) ),
             "cudaMemset legacy response" );
    };
    clear_array( x );
    clear_array( y );
    clear_array( z );
  }
};

inline void synchronize() { check( cudaDeviceSynchronize(), "legacy CUDA synchronization" ); }

}  // namespace tribol_benchmark::legacy_cuda

#endif
