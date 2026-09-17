#ifndef TRIBOL_METHOD_METHOD_HPP_
#define TRIBOL_METHOD_METHOD_HPP_

#include "tribol/core/Config.hpp"

#include <concepts>
#include <type_traits>

namespace tribol {
namespace detail {
struct NormalCategory {};
struct GeometryCategory {};
struct IntegrationCategory {};
struct BasisCategory {};
struct ConstraintCategory {};
struct StiffnessCategory {};
struct RateCategory {};
struct EnforcementCategory {};
struct ResponseCategory {};
struct FormulationCategory {};
struct LinearizationCategory {};

template <typename Policy, typename Category>
concept PolicyOf = requires {
  typename Policy::policy_category;
} && std::same_as<typename Policy::policy_category, Category> && requires { typename Policy::Parameters; };

}  // namespace detail
template <typename T>
concept NormalPolicy = detail::PolicyOf<T, detail::NormalCategory>;

template <typename T>
concept GeometryPolicy = detail::PolicyOf<T, detail::GeometryCategory>;

template <typename T>
concept IntegrationPolicy = detail::PolicyOf<T, detail::IntegrationCategory>;

template <typename T>
concept BasisPolicy = detail::PolicyOf<T, detail::BasisCategory>;

template <typename T>
concept ConstraintPolicy = detail::PolicyOf<T, detail::ConstraintCategory>;

template <typename T>
concept StiffnessPolicy = detail::PolicyOf<T, detail::StiffnessCategory>;

template <typename T>
concept RatePolicy = detail::PolicyOf<T, detail::RateCategory>;

template <typename T>
concept EnforcementPolicy = detail::PolicyOf<T, detail::EnforcementCategory>;

template <typename T>
concept ResponsePolicy = detail::PolicyOf<T, detail::ResponseCategory>;

template <typename T>
concept FormulationPolicy = detail::PolicyOf<T, detail::FormulationCategory>;

template <typename T>
concept LinearizationPolicy = detail::PolicyOf<T, detail::LinearizationCategory>;

namespace normal {

struct MeanPlane {
  using policy_category = detail::NormalCategory;
  struct Parameters {};
};

struct MortarSurface {
  using policy_category = detail::NormalCategory;
  struct Parameters {};
};

}  // namespace normal

namespace geometry {

template <NormalPolicy Normal = normal::MeanPlane>
struct ProjectedOverlap {
  using policy_category = detail::GeometryCategory;
  using normal_policy = Normal;

  struct Parameters {
    typename Normal::Parameters normal{};
    Real minimum_measure{ 1.0e-14 };
    Real minimum_overlap_fraction{};
  };
};

struct ConformingOverlap {
  using policy_category = detail::GeometryCategory;

  struct Parameters {
    Real alignment_tolerance{ 1.0e-12 };
  };
};

}  // namespace geometry

namespace integration {

struct Centroid {
  using policy_category = detail::IntegrationCategory;
  struct Parameters {};
};

template <int Order = 2>
struct Polygon {
  static_assert( Order > 0, "Polygon integration order must be positive." );
  using policy_category = detail::IntegrationCategory;
  static constexpr int order = Order;
  struct Parameters {};
};

template <int Order = 2>
struct Face {
  static_assert( Order > 0, "Face integration order must be positive." );
  using policy_category = detail::IntegrationCategory;
  static constexpr int order = Order;
  struct Parameters {};
};

template <int Points = 3>
struct SmoothedSegment {
  static_assert( Points >= 1 && Points <= 3, "Smoothed-segment integration supports one to three points." );
  using policy_category = detail::IntegrationCategory;
  static constexpr int points = Points;

  struct Parameters {
    Real endpoint_width{ 0.1 };
  };
};

}  // namespace integration

namespace basis {

struct Primal {
  using policy_category = detail::BasisCategory;
  struct Parameters {};
};

struct Dual {
  using policy_category = detail::BasisCategory;
  struct Parameters {};
};

}  // namespace basis

namespace constraint {

struct GapActivationParameters {
  Real residual_gap{};
  Real gap_tolerance{};
  bool reject_excessive_penetration{};
  Real maximum_penetration_fraction{ 0.95 };
};

struct Pointwise {
  using policy_category = detail::ConstraintCategory;
  struct Parameters {
    GapActivationParameters activation{};
  };
};

template <BasisPolicy Basis = basis::Primal>
struct Nodal {
  using policy_category = detail::ConstraintCategory;
  using basis_policy = Basis;

  struct Parameters {
    typename Basis::Parameters basis{};
    GapActivationParameters activation{};
  };
};

struct QuadraturePoint {
  using policy_category = detail::ConstraintCategory;
  struct Parameters {
    GapActivationParameters activation{};
  };
};

}  // namespace constraint

namespace stiffness {

struct Constant {
  using policy_category = detail::StiffnessCategory;

  struct Parameters {
    Real value{ 1.0 };
    Real mortar_scale{ 1.0 };
    Real nonmortar_scale{ 1.0 };
  };
};

struct Material {
  using policy_category = detail::StiffnessCategory;

  struct Parameters {
    Real scale{ 1.0 };
    Real mortar_scale{ 1.0 };
    Real nonmortar_scale{ 1.0 };
  };
};

}  // namespace stiffness

namespace rate {

struct None {
  using policy_category = detail::RateCategory;
  struct Parameters {};
};

struct Constant {
  using policy_category = detail::RateCategory;

  struct Parameters {
    Real value{};
    Real mortar_scale{ 1.0 };
    Real nonmortar_scale{ 1.0 };
  };
};

struct Percentage {
  using policy_category = detail::RateCategory;

  struct Parameters {
    Real ratio{};
    Real mortar_scale{ 1.0 };
    Real nonmortar_scale{ 1.0 };
  };
};

}  // namespace rate

namespace enforcement {

template <StiffnessPolicy Stiffness = stiffness::Constant, RatePolicy Rate = rate::None>
struct Penalty {
  using policy_category = detail::EnforcementCategory;
  using stiffness_policy = Stiffness;
  using rate_policy = Rate;

  struct Parameters {
    typename Stiffness::Parameters stiffness{};
    typename Rate::Parameters rate{};
  };
};

struct LagrangeMultiplier {
  using policy_category = detail::EnforcementCategory;
  struct Parameters {};
};

struct ExternalPressure {
  using policy_category = detail::EnforcementCategory;
  struct Parameters {};
};

struct None {
  using policy_category = detail::EnforcementCategory;
  struct Parameters {};
};

}  // namespace enforcement

namespace response {

struct Frictionless {
  using policy_category = detail::ResponseCategory;
  struct Parameters {};
};

struct ViscousTangential {
  using policy_category = detail::ResponseCategory;

  struct Parameters {
    Real damping{};
  };
};

struct TiedNormal {
  using policy_category = detail::ResponseCategory;
  struct Parameters {};
};

struct TiedFull {
  using policy_category = detail::ResponseCategory;
  struct Parameters {};
};

}  // namespace response

namespace formulation {

struct PointwiseTraction {
  using policy_category = detail::FormulationCategory;
  struct Parameters {};
};

struct WeightedWeakForm {
  using policy_category = detail::FormulationCategory;
  struct Parameters {};
};

struct Variational {
  using policy_category = detail::FormulationCategory;
  struct Parameters {};
};

struct DiagnosticWeights {
  using policy_category = detail::FormulationCategory;
  struct Parameters {};
};

}  // namespace formulation

namespace linearization {

struct Exact {
  using policy_category = detail::LinearizationCategory;
  struct Parameters {};
};

struct Analytic {
  using policy_category = detail::LinearizationCategory;
  struct Parameters {};
};

struct Enzyme {
  using policy_category = detail::LinearizationCategory;
  struct Parameters {};
};

}  // namespace linearization

template <GeometryPolicy Geometry = geometry::ProjectedOverlap<>, IntegrationPolicy Integration = integration::Centroid,
          ConstraintPolicy Constraint = constraint::Pointwise, EnforcementPolicy Enforcement = enforcement::Penalty<>,
          ResponsePolicy Response = response::Frictionless,
          FormulationPolicy Formulation = formulation::PointwiseTraction,
          LinearizationPolicy Linearization = linearization::Exact>
struct Method {
  using geometry_policy = Geometry;
  using integration_policy = Integration;
  using constraint_policy = Constraint;
  using enforcement_policy = Enforcement;
  using response_policy = Response;
  using formulation_policy = Formulation;
  using linearization_policy = Linearization;

  struct Parameters {
    typename Geometry::Parameters geometry{};
    typename Integration::Parameters integration{};
    typename Constraint::Parameters constraint{};
    typename Enforcement::Parameters enforcement{};
    typename Response::Parameters response{};
    typename Formulation::Parameters formulation{};
    typename Linearization::Parameters linearization{};
  };
};

#include "tribol/method/MethodCompatibility.inl"

using DefaultMethod = Method<>;

static_assert( SupportedMethod<DefaultMethod> );
}  // namespace tribol

#endif
