#include "tribol/execution/DevicePenalty.hpp"

#include <cstddef>
#include <stdexcept>

namespace tribol::execution {

template <>
class DevicePenaltyWorkspace<ActiveExecution>::Impl {
 public:
  ~Impl()
  {
    ActiveDeviceBackend::deallocate( device_contributions_ );
    ActiveDeviceBackend::deallocate( device_patches_ );
  }

  void reserve( Index patch_capacity )
  {
    if ( patch_capacity < 0 ) {
      throw std::invalid_argument( "Device penalty capacity cannot be negative." );
    }
    if ( patch_capacity <= capacity_ ) {
      return;
    }
    InteractionPatch* new_patches = ActiveDeviceBackend::template allocate<InteractionPatch>( patch_capacity );
    PenaltyContribution* new_contributions{};
    try {
      new_contributions = ActiveDeviceBackend::template allocate<PenaltyContribution>( patch_capacity );
    } catch ( ... ) {
      ActiveDeviceBackend::deallocate( new_patches );
      throw;
    }
    ActiveDeviceBackend::deallocate( device_contributions_ );
    ActiveDeviceBackend::deallocate( device_patches_ );
    device_patches_ = new_patches;
    device_contributions_ = new_contributions;
    capacity_ = patch_capacity;
  }

  void evaluatePointwise( ArrayView<const InteractionPatch> patches, Real stiffness,
                          constraint::GapActivationParameters activation, ArrayView<PenaltyContribution> contributions )
  {
    requireValidArguments( patches, stiffness, contributions );
    requireDevice();
    reserve( patches.size() );
    copyInput( patches );
    const InteractionPatch* device_patches = device_patches_;
    PenaltyContribution* device_contributions = device_contributions_;
    const Index count = patches.size();
    RAJA::forall<typename ActiveDeviceBackend::ForallPolicy>(
        ActiveDeviceBackend::resource(), RAJA::TypedRangeSegment<Index>( 0, count ), [=] RAJA_DEVICE( Index patch ) {
          device_contributions[patch] = evaluatePointwisePenaltyPatch( device_patches[patch], stiffness, activation );
        } );
    ActiveDeviceBackend::checkLaunch( "launch pointwise device penalty kernel" );
    copyOutput( contributions );
  }

  void evaluate( ArrayView<const InteractionPatch> patches, Real stiffness,
                 ArrayView<PenaltyContribution> contributions )
  {
    requireValidArguments( patches, stiffness, contributions );
    requireDevice();
    reserve( patches.size() );
    copyInput( patches );
    const InteractionPatch* device_patches = device_patches_;
    PenaltyContribution* device_contributions = device_contributions_;
    const Index count = patches.size();
    RAJA::forall<typename ActiveDeviceBackend::ForallPolicy>(
        ActiveDeviceBackend::resource(), RAJA::TypedRangeSegment<Index>( 0, count ), [=] RAJA_DEVICE( Index patch ) {
          device_contributions[patch] =
              evaluatePenaltyPatch( device_patches[patch], stiffness, integration::Polygon<2>{} );
        } );
    ActiveDeviceBackend::checkLaunch( "launch device penalty kernel" );
    copyOutput( contributions );
  }

  static void requireValidArguments( ArrayView<const InteractionPatch> patches, Real stiffness,
                                     ArrayView<PenaltyContribution> contributions )
  {
    if ( stiffness < 0.0 || contributions.size() != patches.size() || !patches.isStructurallyValid() ||
         !contributions.isStructurallyValid() ) {
      throw std::invalid_argument(
          "Device penalty execution requires nonnegative stiffness and matching valid views." );
    }
  }

  static void requireDevice()
  {
    if ( !ActiveDeviceBackend::available() ) {
      throw std::runtime_error( "Device execution was requested, but no compatible device is available." );
    }
  }

  void copyInput( ArrayView<const InteractionPatch> patches )
  {
    if ( !patches.empty() ) {
      ActiveDeviceBackend::copy( device_patches_, patches.data(),
                                 static_cast<std::size_t>( patches.size() ) * sizeof( InteractionPatch ) );
    }
  }

  void copyOutput( ArrayView<PenaltyContribution> contributions )
  {
    if ( !contributions.empty() ) {
      ActiveDeviceBackend::copy( contributions.data(), device_contributions_,
                                 static_cast<std::size_t>( contributions.size() ) * sizeof( PenaltyContribution ) );
    }
  }

  InteractionPatch* device_patches_{};
  PenaltyContribution* device_contributions_{};
  Index capacity_{};
};

template <>
DevicePenaltyWorkspace<ActiveExecution>::DevicePenaltyWorkspace() : impl_( new Impl )
{
}

template <>
DevicePenaltyWorkspace<ActiveExecution>::~DevicePenaltyWorkspace()
{
  delete impl_;
}

template <>
void DevicePenaltyWorkspace<ActiveExecution>::reserve( Index patch_capacity )
{
  impl_->reserve( patch_capacity );
}

template <>
void DevicePenaltyWorkspace<ActiveExecution>::evaluatePointwise( ArrayView<const InteractionPatch> patches,
                                                                 Real stiffness,
                                                                 constraint::GapActivationParameters activation,
                                                                 ArrayView<PenaltyContribution> contributions )
{
  impl_->evaluatePointwise( patches, stiffness, activation, contributions );
}

template <>
void DevicePenaltyWorkspace<ActiveExecution>::evaluate( ArrayView<const InteractionPatch> patches, Real stiffness,
                                                        ArrayView<PenaltyContribution> contributions )
{
  impl_->evaluate( patches, stiffness, contributions );
}

void evaluatePointwisePenaltyPatches( ArrayView<const InteractionPatch> patches, Real stiffness,
                                      ArrayView<PenaltyContribution> contributions, ActiveExecution,
                                      constraint::GapActivationParameters activation )
{
  DevicePenaltyWorkspace<ActiveExecution> workspace;
  workspace.evaluatePointwise( patches, stiffness, activation, contributions );
}

void evaluatePenaltyPatches( ArrayView<const InteractionPatch> patches, Real stiffness,
                             ArrayView<PenaltyContribution> contributions, ActiveExecution, integration::Polygon<2> )
{
  DevicePenaltyWorkspace<ActiveExecution> workspace;
  workspace.evaluate( patches, stiffness, contributions );
}

}  // namespace tribol::execution
