#include "tribol/constraint/NormalConstraint.hpp"
#include "tribol/core/ExactTangent.hpp"
#include "tribol/evaluation/PointwisePenalty.hpp"
#include "tribol/execution/PenaltyKernel.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace tribol::execution {

namespace {

#include "tribol/execution/CudaContactMemory.inl"
#include "tribol/execution/CudaContactSearch.inl"
#include "tribol/execution/CudaContactKernels.inl"

}  // namespace

template <>
class DeviceContactWorkspace<ActiveExecution>::Impl {
 public:
#include "tribol/execution/CudaContactWorkspacePublic.inl"

#include "tribol/execution/CudaContactWorkspacePrivate.inl"
};

template <>
DeviceContactWorkspace<ActiveExecution>::DeviceContactWorkspace() : impl_( new Impl )
{
}

template <>
DeviceContactWorkspace<ActiveExecution>::~DeviceContactWorkspace()
{
  delete impl_;
}

template <>
void DeviceContactWorkspace<ActiveExecution>::uploadSurfaces( SurfacePairView surfaces )
{
  impl_->uploadSurfaces( surfaces );
}

template <>
void DeviceContactWorkspace<ActiveExecution>::updateGeometry( SurfacePairView surfaces )
{
  impl_->updateGeometry( surfaces );
}

template <>
void DeviceContactWorkspace<ActiveExecution>::invalidate()
{
  impl_->invalidate();
}

template <>
Index DeviceContactWorkspace<ActiveExecution>::findCandidates( const search::Bvh::Parameters& parameters,
                                                               bool self_contact, bool exclude_adjacent )
{
  return impl_->findCandidates( parameters, self_contact, exclude_adjacent );
}

template <>
void DeviceContactWorkspace<ActiveExecution>::setCandidates( ArrayView<const ElementPair> candidates )
{
  impl_->setCandidates( candidates );
}

template <>
void DeviceContactWorkspace<ActiveExecution>::downloadCandidates( std::vector<ElementPair>& candidates ) const
{
  impl_->downloadCandidates( candidates );
}

template <>
Index DeviceContactWorkspace<ActiveExecution>::interactionCount() const
{
  return impl_->interactionCount();
}

template <>
ArrayView<const ElementPair> DeviceContactWorkspace<ActiveExecution>::deviceCandidates() const
{
  return impl_->deviceCandidates();
}

template <>
SurfacePairView DeviceContactWorkspace<ActiveExecution>::deviceSurfaces() const
{
  return impl_->deviceSurfaces();
}

template <>
EvaluationSummary DeviceContactWorkspace<ActiveExecution>::evaluateDefault(
    const DefaultMethod::Parameters& parameters, const ContactStateView& state,
    const timestep::Kinematic::Parameters& timestep_parameters )
{
  return impl_->evaluate( parameters, state, timestep_parameters );
}

template <>
void DeviceContactWorkspace<ActiveExecution>::downloadResult( ContactOutputView output ) const
{
  impl_->downloadResult( output );
}

template <>
ContactResultView DeviceContactWorkspace<ActiveExecution>::deviceResult( EvaluationSummary summary,
                                                                         GeometryVersion geometry_version,
                                                                         InteractionVersion interaction_version ) const
{
  return impl_->deviceResult( summary, geometry_version, interaction_version );
}

template <>
DevicePipelineView DeviceContactWorkspace<ActiveExecution>::pipelineView( EvaluationSummary summary,
                                                                          GeometryVersion geometry_version,
                                                                          InteractionVersion interaction_version ) const
{
  return { .surfaces = impl_->deviceSurfaces(),
           .candidates = impl_->deviceCandidates(),
           .patches = impl_->devicePatches(),
           .result = impl_->deviceResult( summary, geometry_version, interaction_version ) };
}

template <>
void DeviceContactWorkspace<ActiveExecution>::applyDefaultCoordinateDerivative(
    const DefaultMethod::Parameters& parameters, const ContactStateView& state, ContactDirectionView direction,
    ContactResidualView derivative )
{
  impl_->applyDerivative( parameters, state, direction, derivative );
}

}  // namespace tribol::execution
