#ifndef TRIBOL_EXECUTION_CUDAPENALTY_HPP_
#define TRIBOL_EXECUTION_CUDAPENALTY_HPP_

#include "tribol/execution/PenaltyKernel.hpp"

namespace tribol::execution {

[[nodiscard]] bool cudaDeviceAvailable();

struct HostPenaltyWorkspace {};

class CudaPenaltyWorkspace {
 public:
  CudaPenaltyWorkspace() = default;
  ~CudaPenaltyWorkspace();

  CudaPenaltyWorkspace( const CudaPenaltyWorkspace& ) = delete;
  CudaPenaltyWorkspace& operator=( const CudaPenaltyWorkspace& ) = delete;
  CudaPenaltyWorkspace( CudaPenaltyWorkspace&& ) = delete;
  CudaPenaltyWorkspace& operator=( CudaPenaltyWorkspace&& ) = delete;

  void reserve( Index patch_capacity );
  void evaluatePointwise( ArrayView<const InteractionPatch> patches, Real stiffness,
                          constraint::GapActivationParameters activation,
                          ArrayView<PenaltyContribution> contributions );

 private:
  InteractionPatch* device_patches_{};
  PenaltyContribution* device_contributions_{};
  Index capacity_{};
};

void evaluatePointwisePenaltyPatches( ArrayView<const InteractionPatch> patches, Real stiffness,
                                      ArrayView<PenaltyContribution> contributions, Cuda,
                                      constraint::GapActivationParameters activation = {} );

void evaluatePenaltyPatches( ArrayView<const InteractionPatch> patches, Real stiffness,
                             ArrayView<PenaltyContribution> contributions, Cuda, integration::Polygon<2> rule = {} );

}  // namespace tribol::execution

#endif
