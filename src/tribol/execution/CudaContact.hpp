#ifndef TRIBOL_EXECUTION_CUDACONTACT_HPP_
#define TRIBOL_EXECUTION_CUDACONTACT_HPP_

#include "tribol/evaluation/State.hpp"
#include "tribol/geom/ProjectedOverlap.hpp"
#include "tribol/method/Method.hpp"
#include "tribol/search/Search.hpp"
#include "tribol/timestep/KinematicVote.hpp"

#include <vector>

namespace tribol::execution {

struct CudaPipelineView {
  SurfacePairView surfaces{};
  ArrayView<const ElementPair> candidates{};
  ArrayView<const InteractionPatch> patches{};
  ContactResultView result{};
};

class CudaContactWorkspace {
 public:
  CudaContactWorkspace();
  ~CudaContactWorkspace();

  CudaContactWorkspace( const CudaContactWorkspace& ) = delete;
  CudaContactWorkspace& operator=( const CudaContactWorkspace& ) = delete;
  CudaContactWorkspace( CudaContactWorkspace&& ) = delete;
  CudaContactWorkspace& operator=( CudaContactWorkspace&& ) = delete;

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
  [[nodiscard]] CudaPipelineView pipelineView( EvaluationSummary summary, GeometryVersion geometry_version,
                                               InteractionVersion interaction_version ) const;

  void applyDefaultCoordinateDerivative( const DefaultMethod::Parameters& parameters, const ContactStateView& state,
                                         ContactDirectionView direction, ContactResidualView derivative );

 private:
  class Impl;
  Impl* impl_{};
};

[[nodiscard]] bool cudaDeviceAvailable();

}  // namespace tribol::execution

#endif
