#include "tribol/execution/DeviceContact.hpp"

#include "tribol/execution/DeviceBackendHip.hpp"

namespace tribol::execution {

using ActiveExecution = Hip;
using ActiveDeviceBackend = detail::HipDeviceBackend;

}  // namespace tribol::execution

#include "tribol/execution/DeviceContactImplementation.inl"

namespace tribol::execution {

bool hipDeviceAvailable() { return ActiveDeviceBackend::available(); }

}  // namespace tribol::execution
