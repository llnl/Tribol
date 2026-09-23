#include "tribol/Tribol.hpp"

#include "ConstraintRowFixture.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

// Requirements: TIME-002

namespace {

using namespace tribol;
using test::TwoRowFixture;

struct StabilityFixture {
  TwoRowFixture rows;
  std::array<Real, 4> mortar_inverse_mass{ 0.0, 1.0, 0.0, 1.0 };
  std::array<Real, 4> nonmortar_inverse_mass{ 0.0, 1.0, 0.0, 1.0 };
  std::array<Real, 4> mortar_workspace{};
  std::array<Real, 4> nonmortar_workspace{};

  [[nodiscard]] timestep::DiagonalInverseMassView inverseMass() const
  {
    return {
        .mortar = { { mortar_inverse_mass.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar = { { nonmortar_inverse_mass.data(), 4 }, 2, 2, FieldLayout::Interleaved },
    };
  }

  [[nodiscard]] timestep::ExplicitStabilityWorkspaceView workspace()
  {
    return {
        .mortar_row_sums = { { mortar_workspace.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar_row_sums = { { nonmortar_workspace.data(), 4 }, 2, 2, FieldLayout::Interleaved },
    };
  }
};

bool physicalMassAndIntegratorContract()
{
  StabilityFixture fixture;
  constexpr std::array<Real, 2> stiffness{ 5.0, 5.0 };
  constexpr std::array<Real, 2> rate{ 1.0, 1.0 };
  auto integrator = timestep::IntegratorStabilityLimits::centralDifference();
  integrator.safety_factor = 0.8;
  const auto result = timestep::explicitPenaltyStabilityVote( fixture.rows.view(),
                                                              { .row_stiffness = { stiffness.data(), 2 },
                                                                .row_rate_coefficient = { rate.data(), 2 },
                                                                .inverse_mass = fixture.inverseMass(),
                                                                .integrator = integrator },
                                                              fixture.workspace() );
  const Real expected_unscaled = ( std::sqrt( 44.0 ) - 2.0 ) / 10.0;
  return result.active_stiffness_rows == 2 && result.active_rate_rows == 2 &&
         std::abs( result.stiffness_frequency_squared_bound - 10.0 ) < 1.0e-13 &&
         std::abs( result.rate_frequency_bound - 2.0 ) < 1.0e-13 &&
         std::abs( result.timestep_vote - 0.8 * expected_unscaled ) < 1.0e-13;
}

bool massScalingAndInactiveContract()
{
  StabilityFixture fixture;
  constexpr std::array<Real, 2> stiffness{ 5.0, 5.0 };
  constexpr std::array<Real, 2> inactive{};
  const auto baseline = timestep::explicitPenaltyStabilityVote(
      fixture.rows.view(),
      { .row_stiffness = { stiffness.data(), 2 },
        .inverse_mass = fixture.inverseMass(),
        .integrator = timestep::IntegratorStabilityLimits::centralDifference() },
      fixture.workspace() );
  for ( Real& value : fixture.mortar_inverse_mass ) {
    value *= 0.25;
  }
  for ( Real& value : fixture.nonmortar_inverse_mass ) {
    value *= 0.25;
  }
  const auto heavier = timestep::explicitPenaltyStabilityVote(
      fixture.rows.view(),
      { .row_stiffness = { stiffness.data(), 2 },
        .inverse_mass = fixture.inverseMass(),
        .integrator = timestep::IntegratorStabilityLimits::centralDifference() },
      fixture.workspace() );
  const auto no_contact = timestep::explicitPenaltyStabilityVote(
      fixture.rows.view(),
      { .row_stiffness = { inactive.data(), 2 },
        .inverse_mass = fixture.inverseMass(),
        .integrator = timestep::IntegratorStabilityLimits::centralDifference() },
      fixture.workspace() );
  return std::abs( heavier.stiffness_frequency_squared_bound - 0.25 * baseline.stiffness_frequency_squared_bound ) <
             1.0e-13 &&
         std::abs( heavier.timestep_vote - 2.0 * baseline.timestep_vote ) < 1.0e-13 &&
         std::isinf( no_contact.timestep_vote );
}

bool spectralValidationContract()
{
  StabilityFixture fixture;
  constexpr std::array<Real, 2> stiffness{ 5.0, 5.0 };
  std::array<Real, 4> iterate_mortar{};
  std::array<Real, 4> iterate_nonmortar{};
  std::array<Real, 4> direction_mortar{};
  std::array<Real, 4> direction_nonmortar{};
  std::array<Real, 4> image_mortar{};
  std::array<Real, 4> image_nonmortar{};
  std::array<Real, 2> row_values{};
  const auto field = []( std::array<Real, 4>& values ) {
    return FieldView<Real>{ { values.data(), 4 }, 2, 2, FieldLayout::Interleaved };
  };
  const Real estimate = timestep::estimateMaximumMassScaledEigenvalue(
      fixture.rows.view(), { stiffness.data(), 2 }, fixture.inverseMass(),
      { .iterate_mortar = field( iterate_mortar ),
        .iterate_nonmortar = field( iterate_nonmortar ),
        .direction_mortar = field( direction_mortar ),
        .direction_nonmortar = field( direction_nonmortar ),
        .image_mortar = field( image_mortar ),
        .image_nonmortar = field( image_nonmortar ),
        .row_values = { row_values.data(), 2 } } );
  return std::abs( estimate - 10.0 ) < 1.0e-10;
}

bool centralDifferenceBoundaryContract()
{
  constexpr Real stiffness_frequency_squared = 10.0;
  constexpr Real rate_frequency = 2.0;
  const Real critical_step =
      ( std::sqrt( rate_frequency * rate_frequency + 4.0 * stiffness_frequency_squared ) - rate_frequency ) /
      stiffness_frequency_squared;
  const auto maximum_displacement = []( Real timestep ) {
    Real previous = 1.0;
    Real current = 1.0;
    Real maximum = 1.0;
    for ( int step = 0; step < 500; ++step ) {
      const Real next =
          ( 2.0 - rate_frequency * timestep - stiffness_frequency_squared * timestep * timestep ) * current -
          ( 1.0 - rate_frequency * timestep ) * previous;
      previous = current;
      current = next;
      maximum = std::max( maximum, std::abs( current ) );
    }
    return maximum;
  };
  return maximum_displacement( 0.99 * critical_step ) < 2.0 && maximum_displacement( 1.01 * critical_step ) > 10.0;
}

bool missingContractsAreRejected()
{
  StabilityFixture fixture;
  constexpr std::array<Real, 2> stiffness{ 5.0, 5.0 };
  try {
    (void)timestep::explicitPenaltyStabilityVote(
        fixture.rows.view(), { .row_stiffness = { stiffness.data(), 2 }, .inverse_mass = fixture.inverseMass() },
        fixture.workspace() );
  } catch ( const std::invalid_argument& ) {
    return true;
  }
  return false;
}

}  // namespace

int main()
{
  return physicalMassAndIntegratorContract() && massScalingAndInactiveContract() && spectralValidationContract() &&
                 centralDifferenceBoundaryContract() && missingContractsAreRejected()
             ? 0
             : 1;
}
