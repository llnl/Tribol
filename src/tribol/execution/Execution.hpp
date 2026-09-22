#ifndef TRIBOL_EXECUTION_EXECUTION_HPP_
#define TRIBOL_EXECUTION_EXECUTION_HPP_

#include <concepts>
#include <type_traits>

namespace tribol::execution {

namespace detail {

struct ExecutionCategory {};

}  // namespace detail

struct Sequential {
  using execution_category = detail::ExecutionCategory;
};

struct Deterministic {
  using execution_category = detail::ExecutionCategory;
};

struct OpenMP {
  using execution_category = detail::ExecutionCategory;
};

struct Cuda {
  using execution_category = detail::ExecutionCategory;
};

struct Hip {
  using execution_category = detail::ExecutionCategory;
};

template <typename T>
concept Policy = requires { typename T::execution_category; } &&
                 std::same_as<typename T::execution_category, detail::ExecutionCategory>;

static_assert( Policy<Sequential> );
static_assert( Policy<Deterministic> );
static_assert( Policy<OpenMP> );
static_assert( Policy<Cuda> );
static_assert( Policy<Hip> );

template <typename T>
concept DevicePolicy = std::same_as<T, Cuda> || std::same_as<T, Hip>;

}  // namespace tribol::execution

#endif
