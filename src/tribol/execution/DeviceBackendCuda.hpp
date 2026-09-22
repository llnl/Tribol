#ifndef TRIBOL_EXECUTION_DEVICEBACKENDCUDA_HPP_
#define TRIBOL_EXECUTION_DEVICEBACKENDCUDA_HPP_

#include "tribol/core/Config.hpp"

#include <cstddef>

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>

#if CUDART_VERSION >= 13000
inline cudaError_t cudaMemAdvise( const void* pointer, std::size_t bytes, cudaMemoryAdvise advice, int device )
{
  cudaMemLocation location{};
  location.type = device == cudaCpuDeviceId ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
  location.id = device == cudaCpuDeviceId ? 0 : device;
  return ::cudaMemAdvise( pointer, bytes, advice, location );
}
#endif

#include "RAJA/RAJA.hpp"

#include <stdexcept>
#include <string>

namespace tribol::execution::detail {

class CudaDeviceBackend {
 public:
  using ForallPolicy = RAJA::cuda_exec_async<128>;
  using Resource = RAJA::resources::Cuda;

  static Resource resource() { return Resource::CudaFromStream( nullptr ); }

  template <typename T>
  static T* allocate( std::size_t size )
  {
    return resource().template allocate<T>( size );
  }

  static void deallocate( void* pointer )
  {
    if ( pointer != nullptr ) {
      resource().deallocate( pointer, camp::resources::MemoryAccess::Device );
    }
  }

  static void copy( void* destination, const void* source, std::size_t bytes )
  {
    resource().memcpy( destination, source, bytes );
    resource().wait();
  }

  static void clear( void* destination, std::size_t bytes ) { resource().memset( destination, 0, bytes ); }

  static void synchronize() { resource().wait(); }

  static void checkLaunch( const char* operation ) { require( cudaGetLastError(), operation ); }

  [[nodiscard]] static bool available()
  {
    int count{};
    return cudaGetDeviceCount( &count ) == cudaSuccess && count > 0;
  }

  template <typename Input, typename Output, typename Reduction, typename Value>
  static std::size_t reduceTemporaryBytes( Input input, Output output, Index count, Reduction reduction, Value initial )
  {
    std::size_t bytes{};
    require(
        cub::DeviceReduce::Reduce( nullptr, bytes, input, output, count, reduction, initial, resource().get_stream() ),
        "size CUDA device reduction" );
    return bytes;
  }

  template <typename Input, typename Output, typename Reduction, typename Value>
  static void reduce( void* temporary, std::size_t bytes, Input input, Output output, Index count, Reduction reduction,
                      Value initial )
  {
    require( cub::DeviceReduce::Reduce( temporary, bytes, input, output, count, reduction, initial,
                                        resource().get_stream() ),
             "CUDA device reduction" );
  }

  template <typename KeyInput, typename KeyOutput, typename ValueInput, typename ValueOutput>
  static std::size_t sortPairsTemporaryBytes( KeyInput keys, KeyOutput sorted_keys, ValueInput values,
                                              ValueOutput sorted_values, Index count )
  {
    std::size_t bytes{};
    require( cub::DeviceRadixSort::SortPairs( nullptr, bytes, keys, sorted_keys, values, sorted_values, count, 0,
                                              sizeof( *keys ) * 8, resource().get_stream() ),
             "size CUDA device radix sort" );
    return bytes;
  }

  template <typename KeyInput, typename KeyOutput, typename ValueInput, typename ValueOutput>
  static void sortPairs( void* temporary, std::size_t bytes, KeyInput keys, KeyOutput sorted_keys, ValueInput values,
                         ValueOutput sorted_values, Index count )
  {
    require( cub::DeviceRadixSort::SortPairs( temporary, bytes, keys, sorted_keys, values, sorted_values, count, 0,
                                              sizeof( *keys ) * 8, resource().get_stream() ),
             "CUDA device radix sort" );
  }

  template <typename Input, typename Unique, typename Counts, typename NumberRuns>
  static std::size_t runLengthEncodeTemporaryBytes( Input input, Unique unique, Counts counts, NumberRuns number_runs,
                                                    Index count )
  {
    std::size_t bytes{};
    require( cub::DeviceRunLengthEncode::Encode( nullptr, bytes, input, unique, counts, number_runs, count,
                                                 resource().get_stream() ),
             "size CUDA run-length encoding" );
    return bytes;
  }

  template <typename Input, typename Unique, typename Counts, typename NumberRuns>
  static void runLengthEncode( void* temporary, std::size_t bytes, Input input, Unique unique, Counts counts,
                               NumberRuns number_runs, Index count )
  {
    require( cub::DeviceRunLengthEncode::Encode( temporary, bytes, input, unique, counts, number_runs, count,
                                                 resource().get_stream() ),
             "CUDA run-length encoding" );
  }

  template <typename Input, typename Output>
  static std::size_t exclusiveSumTemporaryBytes( Input input, Output output, Index count )
  {
    std::size_t bytes{};
    require( cub::DeviceScan::ExclusiveSum( nullptr, bytes, input, output, count, resource().get_stream() ),
             "size CUDA exclusive scan" );
    return bytes;
  }

  template <typename Input, typename Output>
  static void exclusiveSum( void* temporary, std::size_t bytes, Input input, Output output, Index count )
  {
    require( cub::DeviceScan::ExclusiveSum( temporary, bytes, input, output, count, resource().get_stream() ),
             "CUDA exclusive scan" );
  }

 private:
  static void require( cudaError_t error, const char* operation )
  {
    if ( error != cudaSuccess ) {
      throw std::runtime_error( std::string( operation ) + ": " + cudaGetErrorString( error ) );
    }
  }
};

}  // namespace tribol::execution::detail

#endif
