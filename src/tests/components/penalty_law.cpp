#include "tribol/Tribol.hpp"

#include <cmath>
#include <type_traits>

// Requirements: ENF-005

namespace {

using namespace tribol;

bool scalarContract()
{
  constraint::GapActivationParameters activation;
  activation.residual_gap = 0.05;
  activation.gap_tolerance = 0.01;
  const auto active = enforcement::evaluateNormalPenalty( enforcement::NormalPenaltyInput{
      .gap = -0.2,
      .normal_rate = -0.3,
      .pressure_stiffness = 10.0,
      .rate_coefficient = 2.0,
      .activation = activation,
  } );
  const auto opening = enforcement::evaluateNormalPenalty( enforcement::NormalPenaltyInput{
      .gap = 0.2,
      .normal_rate = 0.1,
      .pressure_stiffness = 10.0,
      .rate_coefficient = 2.0,
      .activation = activation,
  } );
  const auto closing = enforcement::evaluateNormalPenalty( enforcement::NormalPenaltyInput{
      .gap = 0.2,
      .normal_rate = -0.3,
      .pressure_stiffness = 10.0,
      .rate_coefficient = 2.0,
      .activation = activation,
  } );
  return std::abs( enforcement::seriesPenaltyStiffness( 12.0, 18.0 ) - 7.2 ) < 1.0e-14 && active.active() &&
         active.spring_active && active.rate_active && std::abs( active.effective_gap + 0.25 ) < 1.0e-14 &&
         std::abs( active.pressure + 3.1 ) < 1.0e-14 && std::abs( active.pressure_gap_derivative - 10.0 ) < 1.0e-14 &&
         std::abs( active.pressure_rate_derivative - 2.0 ) < 1.0e-14 &&
         std::abs( active.potential_density - 0.3125 ) < 1.0e-14 && !opening.active() && closing.active() &&
         !closing.spring_active && closing.rate_active && std::abs( closing.pressure + 0.6 ) < 1.0e-14;
}

bool exactDerivativeContract()
{
  using Tangent = linearization_detail::ExactTangent;
  const auto exact = enforcement::evaluateNormalPenalty( enforcement::NormalPenaltyInputT<Tangent>{
      .gap = Tangent{ -0.2, 0.7 },
      .normal_rate = Tangent{ -0.3, -0.4 },
      .pressure_stiffness = Tangent{ 10.0 },
      .rate_coefficient = Tangent{ 2.0 },
  } );
  constexpr Real step = 1.0e-7;
  const auto evaluate = []( Real offset ) {
    return enforcement::evaluateNormalPenalty( enforcement::NormalPenaltyInput{
        .gap = -0.2 + 0.7 * offset,
        .normal_rate = -0.3 - 0.4 * offset,
        .pressure_stiffness = 10.0,
        .rate_coefficient = 2.0,
    } );
  };
  const Real pressure_difference = ( evaluate( step ).pressure - evaluate( -step ).pressure ) / ( 2.0 * step );
  const Real energy_difference =
      ( evaluate( step ).potential_density - evaluate( -step ).potential_density ) / ( 2.0 * step );
  return std::abs( linearization_detail::tangent( exact.pressure ) - pressure_difference ) < 1.0e-8 &&
         std::abs( linearization_detail::tangent( exact.potential_density ) - energy_difference ) < 1.0e-8;
}

bool ratePolicyContract()
{
  using None = enforcement::Penalty<stiffness::Constant, rate::None>;
  using Constant = enforcement::Penalty<stiffness::Constant, rate::Constant>;
  using Percentage = enforcement::Penalty<stiffness::Constant, rate::Percentage>;
  None::Parameters none;
  Constant::Parameters constant;
  Percentage::Parameters percentage;
  constant.rate.value = 4.0;
  constant.rate.mortar_scale = 0.5;
  constant.rate.nonmortar_scale = 1.5;
  percentage.rate.ratio = 0.25;
  percentage.rate.mortar_scale = 0.5;
  percentage.rate.nonmortar_scale = 1.5;
  return enforcement::penaltyRateCoefficient<None>( none, 8.0 ) == 0.0 &&
         enforcement::penaltyRateCoefficient<Constant>( constant, 8.0 ) == 4.0 &&
         enforcement::penaltyRateCoefficient<Percentage>( percentage, 8.0 ) == 2.0;
}

static_assert( std::is_trivially_copyable_v<enforcement::NormalPenaltyInput> );
static_assert( std::is_trivially_copyable_v<enforcement::NormalPenaltyResponse> );

}  // namespace

int main() { return scalarContract() && exactDerivativeContract() && ratePolicyContract() ? 0 : 1; }
