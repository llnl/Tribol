#ifndef TRIBOL_EXECUTION_DEVICEPENALTY_HPP_
#define TRIBOL_EXECUTION_DEVICEPENALTY_HPP_

#include "tribol/execution/Execution.hpp"
#include "tribol/execution/PenaltyKernel.hpp"

namespace tribol::execution {

struct HostPenaltyWorkspace {};

template <typename Execution>
class DevicePenaltyWorkspace {
 public:
  class Impl;

  DevicePenaltyWorkspace();
  ~DevicePenaltyWorkspace();

  DevicePenaltyWorkspace( const DevicePenaltyWorkspace& ) = delete;
  DevicePenaltyWorkspace& operator=( const DevicePenaltyWorkspace& ) = delete;
  DevicePenaltyWorkspace( DevicePenaltyWorkspace&& ) = delete;
  DevicePenaltyWorkspace& operator=( DevicePenaltyWorkspace&& ) = delete;

  void reserve( Index patch_capacity );
  void evaluatePointwise( ArrayView<const InteractionPatch> patches, Real stiffness,
                          constraint::GapActivationParameters activation,
                          ArrayView<PenaltyContribution> contributions );
  void evaluate( ArrayView<const InteractionPatch> patches, Real stiffness,
                 ArrayView<PenaltyContribution> contributions );

 private:
  Impl* impl_{};
};

using CudaPenaltyWorkspace = DevicePenaltyWorkspace<Cuda>;
using HipPenaltyWorkspace = DevicePenaltyWorkspace<Hip>;

[[nodiscard]] bool cudaDeviceAvailable();
[[nodiscard]] bool hipDeviceAvailable();

void evaluatePointwisePenaltyPatches( ArrayView<const InteractionPatch> patches, Real stiffness,
                                      ArrayView<PenaltyContribution> contributions, Cuda,
                                      constraint::GapActivationParameters activation = {} );
void evaluatePointwisePenaltyPatches( ArrayView<const InteractionPatch> patches, Real stiffness,
                                      ArrayView<PenaltyContribution> contributions, Hip,
                                      constraint::GapActivationParameters activation = {} );

void evaluatePenaltyPatches( ArrayView<const InteractionPatch> patches, Real stiffness,
                             ArrayView<PenaltyContribution> contributions, Cuda, integration::Polygon<2> rule = {} );
void evaluatePenaltyPatches( ArrayView<const InteractionPatch> patches, Real stiffness,
                             ArrayView<PenaltyContribution> contributions, Hip, integration::Polygon<2> rule = {} );

}  // namespace tribol::execution

#endif
