#ifndef TRIBOL_ENFORCEMENT_PENALTYLAW_HPP_
#define TRIBOL_ENFORCEMENT_PENALTYLAW_HPP_

#include "tribol/core/ExactTangent.hpp"
#include "tribol/method/Method.hpp"

#include <type_traits>

namespace tribol::enforcement {

template <typename Scalar>
struct NormalPenaltyInputT {
  Scalar gap{};
  Scalar normal_rate{};
  Scalar pressure_stiffness{};
  Scalar rate_coefficient{};
  constraint::GapActivationParameters activation{};
};

using NormalPenaltyInput = NormalPenaltyInputT<Real>;

template <typename Scalar>
struct NormalPenaltyResponseT {
  Scalar effective_gap{};
  Scalar active_gap{};
  Scalar active_rate{};
  Scalar pressure_stiffness{};
  Scalar rate_coefficient{};
  Scalar pressure{};
  Scalar pressure_gap_derivative{};
  Scalar pressure_rate_derivative{};
  Scalar potential_density{};
  bool spring_active{};
  bool rate_active{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool active() const { return spring_active || rate_active; }
};

using NormalPenaltyResponse = NormalPenaltyResponseT<Real>;

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline Scalar seriesPenaltyStiffness( Scalar mortar, Scalar nonmortar )
{
  const Scalar sum = mortar + nonmortar;
  return linearization_detail::absolute( linearization_detail::primal( sum ) ) > 1.0e-28 ? mortar * nonmortar / sum
                                                                                         : Scalar{};
}

template <typename Enforcement, typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline Scalar penaltyRateCoefficient(
    const typename Enforcement::Parameters& parameters, Scalar pressure_stiffness )
{
  using Rate = typename Enforcement::rate_policy;
  if constexpr ( std::same_as<Rate, rate::None> ) {
    return Scalar{};
  } else if constexpr ( std::same_as<Rate, rate::Constant> ) {
    return 0.5 * parameters.rate.value * ( parameters.rate.mortar_scale + parameters.rate.nonmortar_scale );
  } else {
    return 0.5 * parameters.rate.ratio * ( parameters.rate.mortar_scale + parameters.rate.nonmortar_scale ) *
           pressure_stiffness;
  }
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline NormalPenaltyResponseT<Scalar> evaluateEffectiveNormalPenalty(
    Scalar effective_gap, Scalar normal_rate, Scalar pressure_stiffness, Scalar rate_coefficient, Real gap_tolerance )
{
  NormalPenaltyResponseT<Scalar> result;
  result.effective_gap = effective_gap;
  result.spring_active = linearization_detail::primal( effective_gap ) <= gap_tolerance;
  result.rate_active = linearization_detail::primal( normal_rate ) < 0.0;
  result.active_gap = result.spring_active ? result.effective_gap : Scalar{};
  result.active_rate = result.rate_active ? normal_rate : Scalar{};
  result.pressure_stiffness = pressure_stiffness;
  result.rate_coefficient = rate_coefficient;
  result.pressure = pressure_stiffness * result.active_gap + rate_coefficient * result.active_rate;
  result.pressure_gap_derivative = result.spring_active ? pressure_stiffness : Scalar{};
  result.pressure_rate_derivative = result.rate_active ? rate_coefficient : Scalar{};
  result.potential_density = 0.5 * pressure_stiffness * result.active_gap * result.active_gap;
  return result;
}

template <typename Scalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline NormalPenaltyResponseT<Scalar> evaluateNormalPenalty(
    const NormalPenaltyInputT<Scalar>& input )
{
  return evaluateEffectiveNormalPenalty( input.gap - Scalar{ input.activation.residual_gap }, input.normal_rate,
                                         input.pressure_stiffness, input.rate_coefficient,
                                         input.activation.gap_tolerance );
}

static_assert( std::is_trivially_copyable_v<NormalPenaltyInput> );
static_assert( std::is_trivially_copyable_v<NormalPenaltyResponse> );

}  // namespace tribol::enforcement

#endif
