#include "tribol/execution/CudaContact.hpp"

#include "tribol/constraint/NormalConstraint.hpp"
#include "tribol/core/ExactTangent.hpp"
#include "tribol/evaluation/PointwisePenalty.hpp"
#include "tribol/execution/PenaltyKernel.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

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

void requireCuda( cudaError_t error, const char* operation )
{
  if ( error != cudaSuccess ) {
    throw std::runtime_error( std::string( operation ) + ": " + cudaGetErrorString( error ) );
  }
}

void requireKernel( const char* operation ) { requireCuda( cudaGetLastError(), operation ); }

#include "tribol/execution/CudaContactMemory.inl"
#include "tribol/execution/CudaContactSearch.inl"
#include "tribol/execution/CudaContactKernels.inl"

}  // namespace

class CudaContactWorkspace::Impl {
 public:
#include "tribol/execution/CudaContactWorkspacePublic.inl"

 private:
#include "tribol/execution/CudaContactWorkspacePrivate.inl"
};

CudaContactWorkspace::CudaContactWorkspace() : impl_( new Impl ) {}

CudaContactWorkspace::~CudaContactWorkspace() { delete impl_; }

void CudaContactWorkspace::uploadSurfaces( SurfacePairView surfaces ) { impl_->uploadSurfaces( surfaces ); }

void CudaContactWorkspace::updateGeometry( SurfacePairView surfaces ) { impl_->updateGeometry( surfaces ); }

void CudaContactWorkspace::invalidate() { impl_->invalidate(); }

Index CudaContactWorkspace::findCandidates( const search::Bvh::Parameters& parameters, bool self_contact,
                                            bool exclude_adjacent )
{
  return impl_->findCandidates( parameters, self_contact, exclude_adjacent );
}

void CudaContactWorkspace::setCandidates( ArrayView<const ElementPair> candidates )
{
  impl_->setCandidates( candidates );
}

void CudaContactWorkspace::downloadCandidates( std::vector<ElementPair>& candidates ) const
{
  impl_->downloadCandidates( candidates );
}

Index CudaContactWorkspace::interactionCount() const { return impl_->interactionCount(); }

ArrayView<const ElementPair> CudaContactWorkspace::deviceCandidates() const { return impl_->deviceCandidates(); }

SurfacePairView CudaContactWorkspace::deviceSurfaces() const { return impl_->deviceSurfaces(); }

EvaluationSummary CudaContactWorkspace::evaluateDefault( const DefaultMethod::Parameters& parameters,
                                                         const ContactStateView& state,
                                                         const timestep::Kinematic::Parameters& timestep_parameters )
{
  return impl_->evaluate( parameters, state, timestep_parameters );
}

void CudaContactWorkspace::downloadResult( ContactOutputView output ) const { impl_->downloadResult( output ); }

ContactResultView CudaContactWorkspace::deviceResult( EvaluationSummary summary, GeometryVersion geometry_version,
                                                      InteractionVersion interaction_version ) const
{
  return impl_->deviceResult( summary, geometry_version, interaction_version );
}

CudaPipelineView CudaContactWorkspace::pipelineView( EvaluationSummary summary, GeometryVersion geometry_version,
                                                     InteractionVersion interaction_version ) const
{
  return { .surfaces = impl_->deviceSurfaces(),
           .candidates = impl_->deviceCandidates(),
           .patches = impl_->devicePatches(),
           .result = impl_->deviceResult( summary, geometry_version, interaction_version ) };
}

void CudaContactWorkspace::applyDefaultCoordinateDerivative( const DefaultMethod::Parameters& parameters,
                                                             const ContactStateView& state,
                                                             ContactDirectionView direction,
                                                             ContactResidualView derivative )
{
  impl_->applyDerivative( parameters, state, direction, derivative );
}

}  // namespace tribol::execution
