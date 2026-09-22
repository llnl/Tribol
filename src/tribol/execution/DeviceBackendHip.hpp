#ifndef TRIBOL_EXECUTION_DEVICEBACKENDHIP_HPP_
#define TRIBOL_EXECUTION_DEVICEBACKENDHIP_HPP_

#include "tribol/core/Config.hpp"

#include "RAJA/RAJA.hpp"

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace tribol::execution::detail {

class HipDeviceBackend {
 public:
  using ForallPolicy = RAJA::hip_exec_async<128>;
  using Resource = RAJA::resources::Hip;

  static Resource resource() { return Resource::HipFromStream( nullptr ); }

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

  static void checkLaunch( const char* operation ) { require( hipGetLastError(), operation ); }

  [[nodiscard]] static bool available()
  {
    int count{};
    return hipGetDeviceCount( &count ) == hipSuccess && count > 0;
  }

  template <typename Input, typename Output, typename Reduction, typename Value>
  static std::size_t reduceTemporaryBytes( Input input, Output output, Index count, Reduction reduction, Value initial )
  {
    std::size_t bytes{};
    require( hipcub::DeviceReduce::Reduce( nullptr, bytes, input, output, count, reduction, initial,
                                           resource().get_stream() ),
             "size HIP device reduction" );
    return bytes;
  }

  template <typename Input, typename Output, typename Reduction, typename Value>
  static void reduce( void* temporary, std::size_t bytes, Input input, Output output, Index count, Reduction reduction,
                      Value initial )
  {
    require( hipcub::DeviceReduce::Reduce( temporary, bytes, input, output, count, reduction, initial,
                                           resource().get_stream() ),
             "HIP device reduction" );
  }

  template <typename KeyInput, typename KeyOutput, typename ValueInput, typename ValueOutput>
  static std::size_t sortPairsTemporaryBytes( KeyInput keys, KeyOutput sorted_keys, ValueInput values,
                                              ValueOutput sorted_values, Index count )
  {
    std::size_t bytes{};
    require( hipcub::DeviceRadixSort::SortPairs( nullptr, bytes, keys, sorted_keys, values, sorted_values, count, 0,
                                                 sizeof( *keys ) * 8, resource().get_stream() ),
             "size HIP device radix sort" );
    return bytes;
  }

  template <typename KeyInput, typename KeyOutput, typename ValueInput, typename ValueOutput>
  static void sortPairs( void* temporary, std::size_t bytes, KeyInput keys, KeyOutput sorted_keys, ValueInput values,
                         ValueOutput sorted_values, Index count )
  {
    require( hipcub::DeviceRadixSort::SortPairs( temporary, bytes, keys, sorted_keys, values, sorted_values, count, 0,
                                                 sizeof( *keys ) * 8, resource().get_stream() ),
             "HIP device radix sort" );
  }

  template <typename Input, typename Unique, typename Counts, typename NumberRuns>
  static std::size_t runLengthEncodeTemporaryBytes( Input input, Unique unique, Counts counts, NumberRuns number_runs,
                                                    Index count )
  {
    std::size_t bytes{};
    require( hipcub::DeviceRunLengthEncode::Encode( nullptr, bytes, input, unique, counts, number_runs, count,
                                                    resource().get_stream() ),
             "size HIP run-length encoding" );
    return bytes;
  }

  template <typename Input, typename Unique, typename Counts, typename NumberRuns>
  static void runLengthEncode( void* temporary, std::size_t bytes, Input input, Unique unique, Counts counts,
                               NumberRuns number_runs, Index count )
  {
    require( hipcub::DeviceRunLengthEncode::Encode( temporary, bytes, input, unique, counts, number_runs, count,
                                                    resource().get_stream() ),
             "HIP run-length encoding" );
  }

  template <typename Input, typename Output>
  static std::size_t exclusiveSumTemporaryBytes( Input input, Output output, Index count )
  {
    std::size_t bytes{};
    require( hipcub::DeviceScan::ExclusiveSum( nullptr, bytes, input, output, count, resource().get_stream() ),
             "size HIP exclusive scan" );
    return bytes;
  }

  template <typename Input, typename Output>
  static void exclusiveSum( void* temporary, std::size_t bytes, Input input, Output output, Index count )
  {
    require( hipcub::DeviceScan::ExclusiveSum( temporary, bytes, input, output, count, resource().get_stream() ),
             "HIP exclusive scan" );
  }

 private:
  static void require( hipError_t error, const char* operation )
  {
    if ( error != hipSuccess ) {
      throw std::runtime_error( std::string( operation ) + ": " + hipGetErrorString( error ) );
    }
  }
};

}  // namespace tribol::execution::detail

#endif
