#include "tribol/Tribol.hpp"

#include "ParentTraceFixture.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>

// Requirements: OUTPUT-003

namespace {

using namespace tribol;
using tribol::test::segmentTrace;

struct SamplingFixture {
  test::SegmentTraceStorage mortar = segmentTrace( 0.0, 1.0, 0.0, false, 101, 3 );
  test::SegmentTraceStorage nonmortar = segmentTrace( 0.0, 1.0, -0.1, true, 202, 7 );
  std::array<ElementPair, 1> candidates{ ElementPair{ 0, 0 } };
  OverlapCellBatch cells = buildOverlapCellBatch<normal::MeanPlane>(
      { mortar.view(), nonmortar.view() }, { candidates.data(), static_cast<Index>( candidates.size() ) } );
  std::array<Real, 4> mortar_coordinates{ 0.0, -0.3, 1.0, -0.3 };
  std::array<Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  std::array<Real, 4> mortar_velocity{ 0.0, -0.6, 0.0, -0.6 };
  std::array<Real, 4> nonmortar_velocity{ 0.0, -0.1, 0.0, -0.1 };

  [[nodiscard]] ParentTracePairView traces() const { return { mortar.view(), nonmortar.view() }; }

  [[nodiscard]] ParentTraceFieldPairView fields( bool velocity ) const
  {
    return {
        .mortar_coordinates = { { mortar_coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar_coordinates = { { nonmortar_coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .mortar_velocity = velocity
                               ? FieldView<const Real>{ { mortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved }
                               : FieldView<const Real>{},
        .nonmortar_velocity =
            velocity ? FieldView<const Real>{ { nonmortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved }
                     : FieldView<const Real>{},
    };
  }
};

bool automaticPhysicalSamplingContract()
{
  const SamplingFixture fixture;
  const auto storage = diagnostics::sampleQuadrature( fixture.traces(), fixture.cells.view(), fixture.fields( false ) );
  const auto samples = storage.view();
  if ( !samples.isStructurallyValid() || samples.numberOfCells() != 1 || samples.numberOfSamples() != 2 ||
       samples.hasNormalRates() || samples.owner_ranks[0] != 3 ) {
    return false;
  }
  Real weight{};
  for ( Index sample = 0; sample < samples.numberOfSamples(); ++sample ) {
    weight += samples.integration_weights[sample];
    if ( std::abs( samples.physical_gaps[sample] + 0.2 ) > 1.0e-13 ) {
      return false;
    }
  }
  return std::abs( weight - 1.0 ) < 1.0e-13;
}

bool independentExactnessContract()
{
  const SamplingFixture fixture;
  const auto enforcement = buildConstraintRowBatch( fixture.traces(), fixture.cells.view(), fixture.fields( false ) );
  const auto diagnostic_storage = diagnostics::sampleQuadrature( fixture.traces(), fixture.cells.view(),
                                                                 fixture.fields( false ), { .exactness = 8 } );
  const auto samples = diagnostic_storage.view();
  Real eighth_moment{};
  for ( Index sample = 0; sample < samples.numberOfSamples(); ++sample ) {
    eighth_moment +=
        samples.integration_weights[sample] * std::pow( samples.mortar_parent_reference_coordinates( sample, 0 ), 8 );
  }
  return enforcement.view().numberOfRows() == 2 && samples.numberOfSamples() == 5 &&
         std::abs( eighth_moment - 1.0 / 9.0 ) < 1.0e-13;
}

bool rateAndOwnershipContract()
{
  const SamplingFixture fixture;
  const auto storage = diagnostics::sampleQuadrature( fixture.traces(), fixture.cells.view(), fixture.fields( true ),
                                                      { .include_normal_rates = true } );
  const auto samples = storage.view();
  if ( !samples.hasNormalRates() ) {
    return false;
  }
  for ( Index sample = 0; sample < samples.numberOfSamples(); ++sample ) {
    if ( std::abs( samples.normal_rates[sample] + 0.5 ) > 1.0e-13 ) {
      return false;
    }
  }
  const auto owned = diagnostics::summarizeQuadrature( samples, 3 );
  const auto ghost = diagnostics::summarizeQuadrature( samples, 7 );
  return owned.samples == 2 && owned.owned_samples == 2 && std::abs( owned.maximum_penetration - 0.2 ) < 1.0e-13 &&
         std::abs( owned.maximum_closing_rate - 0.5 ) < 1.0e-13 && ghost.samples == 2 && ghost.owned_samples == 0 &&
         ghost.maximum_penetration == 0.0 && ghost.maximum_closing_rate == 0.0;
}

static_assert( std::is_trivially_copyable_v<diagnostics::QuadratureSamplingOptions> );
static_assert( std::is_trivially_copyable_v<diagnostics::QuadratureSampleView> );
static_assert( std::is_trivially_copyable_v<diagnostics::QuadratureSampleSummary> );

}  // namespace

int main()
{
  const bool automatic = automaticPhysicalSamplingContract();
  const bool exactness = independentExactnessContract();
  const bool ownership = rateAndOwnershipContract();
  if ( !automatic || !exactness || !ownership ) {
    std::cerr << "automatic=" << automatic << " exactness=" << exactness << " ownership=" << ownership << '\n';
    return 1;
  }
  return 0;
}
