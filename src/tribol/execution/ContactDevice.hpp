#ifndef TRIBOL_EXECUTION_CONTACTDEVICE_HPP_
#define TRIBOL_EXECUTION_CONTACTDEVICE_HPP_

#include "tribol/execution/DeviceContact.hpp"
#include "tribol/execution/DevicePenalty.hpp"
#include "tribol/execution/Execution.hpp"
#include "tribol/method/Method.hpp"
#include "tribol/search/Search.hpp"

#include <concepts>

namespace tribol::execution {

template <typename MethodType, typename Execution>
concept SupportedContactExecution =
    SupportedMethod<MethodType> && std::same_as<typename MethodType::linearization_policy, linearization::Exact> &&
    Policy<Execution> && ( !DevicePolicy<Execution> || std::same_as<MethodType, DefaultMethod> );

template <typename Search, typename Execution>
concept SupportedContactSearch =
    SearchPolicy<Search> && Policy<Execution> && ( !DevicePolicy<Execution> || std::same_as<Search, search::Bvh> );

}  // namespace tribol::execution

#endif
