#include "tribol/Tribol.hpp"

#include "ConstraintRowFixture.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

// Requirements: DIFF-008, PHYS-004

namespace {

using namespace tribol;

using test::TwoRowFixture;

bool forceEnergyAndWorkContract()
{
  const TwoRowFixture fixture;
  constexpr std::array<Real, 2> stiffness{ 10.0, 10.0 };
  constexpr std::array<Real, 2> rate{ 2.0, 2.0 };
  std::array<Real, 4> mortar_force{};
  std::array<Real, 4> nonmortar_force{};
  std::array<Real, 2> effective_gap{};
  std::array<Real, 2> pressure{};
  std::array<Real, 2> row_force{};
  std::array<Real, 2> row_stiffness{};
  std::array<Real, 2> row_rate{};
  const auto summary =
      evaluateExplicitPenaltyRows( fixture.view(), { { stiffness.data(), 2 }, { rate.data(), 2 } },
                                   { .mortar = { { mortar_force.data(), 4 }, 2, 2, FieldLayout::Interleaved },
                                     .nonmortar = { { nonmortar_force.data(), 4 }, 2, 2, FieldLayout::Interleaved } },
                                   { .effective_gap = { effective_gap.data(), 2 },
                                     .pressure = { pressure.data(), 2 },
                                     .row_force = { row_force.data(), 2 },
                                     .row_stiffness = { row_stiffness.data(), 2 },
                                     .row_rate_coefficient = { row_rate.data(), 2 } } );
  return summary.rows == 2 && summary.active_rows == 2 && std::abs( summary.energy - 0.05 ) < 1.0e-13 &&
         std::abs( summary.contact_power - 0.09 ) < 1.0e-13 && std::abs( pressure[0] + 1.4 ) < 1.0e-13 &&
         std::abs( pressure[1] + 1.0 ) < 1.0e-13 && std::abs( row_force[0] + 0.7 ) < 1.0e-13 &&
         std::abs( row_force[1] + 0.5 ) < 1.0e-13 && std::abs( row_stiffness[0] - 5.0 ) < 1.0e-13 &&
         std::abs( row_rate[0] - 1.0 ) < 1.0e-13 && std::abs( row_rate[1] ) < 1.0e-13 &&
         std::abs( mortar_force[1] + 0.65 ) < 1.0e-13 && std::abs( mortar_force[3] + 0.55 ) < 1.0e-13 &&
         std::abs( nonmortar_force[1] - 0.65 ) < 1.0e-13 && std::abs( nonmortar_force[3] - 0.55 ) < 1.0e-13;
}

bool missingRateIsRejected()
{
  const TwoRowFixture fixture;
  constexpr std::array<Real, 2> stiffness{ 10.0, 10.0 };
  constexpr std::array<Real, 2> rate{ 2.0, 2.0 };
  std::array<Real, 4> mortar_force{};
  std::array<Real, 4> nonmortar_force{};
  try {
    (void)evaluateExplicitPenaltyRows(
        fixture.view( false ), { { stiffness.data(), 2 }, { rate.data(), 2 } },
        { .mortar = { { mortar_force.data(), 4 }, 2, 2, FieldLayout::Interleaved },
          .nonmortar = { { nonmortar_force.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );
  } catch ( const std::invalid_argument& ) {
    return true;
  }
  return false;
}

bool operatorContract()
{
  const TwoRowFixture fixture;
  constexpr std::array<Real, 4> mortar_input{ 0.0, 1.0, 0.0, 2.0 };
  constexpr std::array<Real, 4> nonmortar_input{ 0.0, -0.5, 0.0, 0.5 };
  const ContactDirectionView direction{
      .mortar = { { mortar_input.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_input.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  std::array<Real, 2> row_values{};
  operators::applyB( fixture.view(), direction, { row_values.data(), 2 } );
  constexpr std::array<Real, 2> dual{ 2.0, -3.0 };
  std::array<Real, 4> mortar_transpose{};
  std::array<Real, 4> nonmortar_transpose{};
  const ContactResidualView transpose{
      .mortar = { { mortar_transpose.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_transpose.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  operators::applyBT( fixture.view(), { dual.data(), 2 }, transpose );
  const Real row_dot = row_values[0] * dual[0] + row_values[1] * dual[1];
  Real parent_dot{};
  for ( int value = 0; value < 4; ++value ) {
    parent_dot += mortar_input[value] * mortar_transpose[value] + nonmortar_input[value] * nonmortar_transpose[value];
  }

  constexpr std::array<Real, 2> row_stiffness{ 5.0, 5.0 };
  std::array<Real, 2> workspace{};
  std::array<Real, 4> mortar_tangent{};
  std::array<Real, 4> nonmortar_tangent{};
  const ContactResidualView tangent{
      .mortar = { { mortar_tangent.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_tangent.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  operators::applyContactTangent( fixture.view(), { row_stiffness.data(), 2 }, direction, { workspace.data(), 2 },
                                  tangent );
  Real quadratic_form{};
  for ( int value = 0; value < 4; ++value ) {
    quadratic_form += mortar_input[value] * mortar_tangent[value] + nonmortar_input[value] * nonmortar_tangent[value];
  }
  return std::abs( row_values[0] - 1.5 ) < 1.0e-13 && std::abs( row_values[1] - 1.5 ) < 1.0e-13 &&
         std::abs( row_dot - parent_dot ) < 1.0e-13 && std::abs( quadratic_form - 22.5 ) < 1.0e-13;
}

bool tangentFiniteDifferenceContract()
{
  const TwoRowFixture fixture;
  constexpr std::array<Real, 4> mortar_direction_values{ 0.0, 1.0, 0.0, 2.0 };
  constexpr std::array<Real, 4> nonmortar_direction_values{ 0.0, -0.5, 0.0, 0.5 };
  const ContactDirectionView direction{
      .mortar = { { mortar_direction_values.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_direction_values.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  constexpr std::array<Real, 2> row_stiffness{ 5.0, 5.0 };
  std::array<Real, 2> workspace{};
  std::array<Real, 4> mortar_tangent{};
  std::array<Real, 4> nonmortar_tangent{};
  operators::applyContactTangent(
      fixture.view(), { row_stiffness.data(), 2 }, direction, { workspace.data(), 2 },
      { .mortar = { { mortar_tangent.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar = { { nonmortar_tangent.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );

  std::array<Real, 2> row_direction{};
  operators::applyB( fixture.view(), direction, { row_direction.data(), 2 } );
  constexpr Real step = 1.0e-7;
  TwoRowFixture positive = fixture;
  TwoRowFixture negative = fixture;
  for ( Index row = 0; row < 2; ++row ) {
    positive.gaps[row] += step * row_direction[row];
    negative.gaps[row] -= step * row_direction[row];
  }
  constexpr std::array<Real, 2> pressure_stiffness{ 10.0, 10.0 };
  std::array<Real, 4> positive_mortar{};
  std::array<Real, 4> positive_nonmortar{};
  std::array<Real, 4> negative_mortar{};
  std::array<Real, 4> negative_nonmortar{};
  (void)evaluateExplicitPenaltyRows(
      positive.view(), { .pressure_stiffness = { pressure_stiffness.data(), 2 } },
      { .mortar = { { positive_mortar.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar = { { positive_nonmortar.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );
  (void)evaluateExplicitPenaltyRows(
      negative.view(), { .pressure_stiffness = { pressure_stiffness.data(), 2 } },
      { .mortar = { { negative_mortar.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar = { { negative_nonmortar.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );
  for ( int value = 0; value < 4; ++value ) {
    if ( std::abs( ( positive_mortar[value] - negative_mortar[value] ) / ( 2.0 * step ) - mortar_tangent[value] ) >
             1.0e-8 ||
         std::abs( ( positive_nonmortar[value] - negative_nonmortar[value] ) / ( 2.0 * step ) -
                   nonmortar_tangent[value] ) > 1.0e-8 ) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main()
{
  return forceEnergyAndWorkContract() && missingRateIsRejected() && operatorContract() &&
                 tangentFiniteDifferenceContract()
             ? 0
             : 1;
}
