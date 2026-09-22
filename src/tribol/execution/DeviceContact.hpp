#ifndef TRIBOL_EXECUTION_DEVICECONTACT_HPP_
#define TRIBOL_EXECUTION_DEVICECONTACT_HPP_

#include "tribol/evaluation/State.hpp"
#include "tribol/execution/Execution.hpp"
#include "tribol/geom/ProjectedOverlap.hpp"
#include "tribol/method/Method.hpp"
#include "tribol/search/Search.hpp"
#include "tribol/timestep/KinematicVote.hpp"

#include <vector>

namespace tribol::execution {

struct DevicePipelineView {
  SurfacePairView surfaces{};
  ArrayView<const ElementPair> candidates{};
  ArrayView<const InteractionPatch> patches{};
  ContactResultView result{};
};

using CudaPipelineView = DevicePipelineView;
using HipPipelineView = DevicePipelineView;

template <typename Execution>
class DeviceContactWorkspace {
 public:
  class Impl;

  DeviceContactWorkspace();
  ~DeviceContactWorkspace();

  DeviceContactWorkspace( const DeviceContactWorkspace& ) = delete;
  DeviceContactWorkspace& operator=( const DeviceContactWorkspace& ) = delete;
  DeviceContactWorkspace( DeviceContactWorkspace&& ) = delete;
  DeviceContactWorkspace& operator=( DeviceContactWorkspace&& ) = delete;

  void uploadSurfaces( SurfacePairView surfaces );
  void updateGeometry( SurfacePairView surfaces );
  void invalidate();

  Index findCandidates( const search::Bvh::Parameters& parameters, bool self_contact, bool exclude_adjacent );
  void setCandidates( ArrayView<const ElementPair> candidates );
  void downloadCandidates( std::vector<ElementPair>& candidates ) const;

  [[nodiscard]] Index interactionCount() const;
  [[nodiscard]] ArrayView<const ElementPair> deviceCandidates() const;
  [[nodiscard]] SurfacePairView deviceSurfaces() const;

  EvaluationSummary evaluateDefault( const DefaultMethod::Parameters& parameters, const ContactStateView& state,
                                     const timestep::Kinematic::Parameters& timestep );
  void downloadResult( ContactOutputView output ) const;
  [[nodiscard]] ContactResultView deviceResult( EvaluationSummary summary, GeometryVersion geometry_version,
                                                InteractionVersion interaction_version ) const;
  [[nodiscard]] DevicePipelineView pipelineView( EvaluationSummary summary, GeometryVersion geometry_version,
                                                 InteractionVersion interaction_version ) const;

  void applyDefaultCoordinateDerivative( const DefaultMethod::Parameters& parameters, const ContactStateView& state,
                                         ContactDirectionView direction, ContactResidualView derivative );

 private:
  Impl* impl_{};
};

using CudaContactWorkspace = DeviceContactWorkspace<Cuda>;
using HipContactWorkspace = DeviceContactWorkspace<Hip>;

[[nodiscard]] bool cudaDeviceAvailable();
[[nodiscard]] bool hipDeviceAvailable();

static_assert( std::is_trivially_copyable_v<DevicePipelineView> );

}  // namespace tribol::execution

#endif
