#include "tribol/execution/DeviceContact.hpp"

#include "tribol/execution/DeviceBackendCuda.hpp"

namespace tribol::execution {

using ActiveExecution = Cuda;
using ActiveDeviceBackend = detail::CudaDeviceBackend;

}  // namespace tribol::execution

#include "tribol/execution/DeviceContactImplementation.inl"
#include "tribol/execution/DeviceRowImplementation.inl"

namespace tribol::execution {

bool cudaDeviceAvailable() { return ActiveDeviceBackend::available(); }

}  // namespace tribol::execution
