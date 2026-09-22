#include "tribol/execution/DevicePenalty.hpp"
#include "tribol/execution/DeviceBackendHip.hpp"

namespace tribol::execution {

using ActiveExecution = Hip;
using ActiveDeviceBackend = detail::HipDeviceBackend;

}  // namespace tribol::execution

#include "tribol/execution/DevicePenaltyImplementation.inl"
