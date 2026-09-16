#ifndef TRIBOL_METHOD_TRAITS_HPP_
#define TRIBOL_METHOD_TRAITS_HPP_

#include "tribol/method/Method.hpp"

#include <type_traits>

namespace tribol {

struct MethodCapabilities {
  bool needs_velocity{};
  bool needs_reference_coordinates{};
  bool needs_material_fields{};
  bool needs_multiplier{};
  bool accepts_external_pressure{};
  bool produces_energy{};
  bool produces_constraint_residual{};
  bool produces_diagnostic_weights{};
  bool supports_self_contact{};
  bool supports_native_high_order{};
};

namespace detail {

template <typename Enforcement>
struct EnforcementTraits {
  static constexpr bool needs_velocity = false;
  static constexpr bool needs_material_fields = false;
  static constexpr bool needs_multiplier = false;
  static constexpr bool accepts_external_pressure = false;
};

template <StiffnessPolicy Stiffness, RatePolicy Rate>
struct EnforcementTraits<enforcement::Penalty<Stiffness, Rate>> {
  static constexpr bool needs_velocity = !std::same_as<Rate, rate::None>;
  static constexpr bool needs_material_fields = std::same_as<Stiffness, stiffness::Material>;
  static constexpr bool needs_multiplier = false;
  static constexpr bool accepts_external_pressure = false;
};

template <>
struct EnforcementTraits<enforcement::LagrangeMultiplier> {
  static constexpr bool needs_velocity = false;
  static constexpr bool needs_material_fields = false;
  static constexpr bool needs_multiplier = true;
  static constexpr bool accepts_external_pressure = false;
};

template <>
struct EnforcementTraits<enforcement::ExternalPressure> {
  static constexpr bool needs_velocity = false;
  static constexpr bool needs_material_fields = false;
  static constexpr bool needs_multiplier = false;
  static constexpr bool accepts_external_pressure = true;
};

template <>
struct EnforcementTraits<enforcement::None> {
  static constexpr bool needs_velocity = false;
  static constexpr bool needs_material_fields = false;
  static constexpr bool needs_multiplier = false;
  static constexpr bool accepts_external_pressure = false;
};

}  // namespace detail

template <SupportedMethod MethodType>
struct MethodTraits {
 private:
  using Enforcement = typename MethodType::enforcement_policy;
  using Response = typename MethodType::response_policy;
  using Formulation = typename MethodType::formulation_policy;
  using EnforcementInfo = detail::EnforcementTraits<Enforcement>;

 public:
  static constexpr MethodCapabilities capabilities{
      .needs_velocity = EnforcementInfo::needs_velocity || std::same_as<Response, response::ViscousTangential>,
      .needs_reference_coordinates =
          std::same_as<Response, response::TiedNormal> || std::same_as<Response, response::TiedFull>,
      .needs_material_fields = EnforcementInfo::needs_material_fields,
      .needs_multiplier = EnforcementInfo::needs_multiplier,
      .accepts_external_pressure = EnforcementInfo::accepts_external_pressure,
      .produces_energy = std::same_as<Formulation, formulation::Variational> &&
                         ( detail::is_penalty_v<Enforcement> || EnforcementInfo::accepts_external_pressure ),
      .produces_constraint_residual = EnforcementInfo::needs_multiplier,
      .produces_diagnostic_weights = std::same_as<Formulation, formulation::DiagnosticWeights>,
      .supports_self_contact = std::same_as<Formulation, formulation::PointwiseTraction>,
      .supports_native_high_order = false,
  };
};

}  // namespace tribol

#endif
