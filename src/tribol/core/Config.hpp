#ifndef TRIBOL_CORE_CONFIG_HPP_
#define TRIBOL_CORE_CONFIG_HPP_

#include <cstdint>

#if !defined( TRIBOL_HOST_DEVICE )
#if defined( __CUDACC__ ) || defined( __HIPCC__ )
#define TRIBOL_HOST_DEVICE __host__ __device__
#else
#define TRIBOL_HOST_DEVICE
#endif
#endif

namespace tribol {

#if defined( TRIBOL_USE_SINGLE_PRECISION )
using Real = float;
#else
using Real = double;
#endif

#if defined( TRIBOL_USE_64BIT_INDEXTYPE )
using Index = std::int64_t;
#else
using Index = std::int32_t;
#endif

}  // namespace tribol

#endif
