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

struct Pointwise {
  using policy_category = detail::ConstraintCategory;
  struct Parameters {};
};

template <BasisPolicy Basis = basis::Primal>
struct Nodal {
  using policy_category = detail::ConstraintCategory;
  using basis_policy = Basis;

  struct Parameters {
    typename Basis::Parameters basis{};
  };
};

struct QuadraturePoint {
  using policy_category = detail::ConstraintCategory;
  struct Parameters {};
};

}  // namespace constraint

namespace stiffness {

struct Constant {
  using policy_category = detail::StiffnessCategory;

  struct Parameters {
    Real value{ 1.0 };
  };
};

struct Material {
  using policy_category = detail::StiffnessCategory;

  struct Parameters {
    Real scale{ 1.0 };
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
  };
};

struct Percentage {
  using policy_category = detail::RateCategory;

  struct Parameters {
    Real ratio{};
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

namespace detail {

template <typename T>
inline constexpr bool is_projected_overlap_v = false;

template <NormalPolicy Normal>
inline constexpr bool is_projected_overlap_v<geometry::ProjectedOverlap<Normal>> = true;

template <typename T>
inline constexpr bool is_mean_plane_overlap_v = false;

template <>
inline constexpr bool is_mean_plane_overlap_v<geometry::ProjectedOverlap<normal::MeanPlane>> = true;

template <typename T>
inline constexpr bool is_polygon_integration_v = false;

template <int Order>
inline constexpr bool is_polygon_integration_v<integration::Polygon<Order>> = Order == 1 || Order == 2;

template <typename T>
inline constexpr bool is_face_integration_v = false;

template <int Order>
inline constexpr bool is_face_integration_v<integration::Face<Order>> = Order == 1 || Order == 2;

template <typename T>
inline constexpr bool is_nodal_constraint_v = false;

template <BasisPolicy Basis>
inline constexpr bool is_nodal_constraint_v<constraint::Nodal<Basis>> = true;

template <typename T>
inline constexpr bool is_primal_nodal_constraint_v = false;

template <>
inline constexpr bool is_primal_nodal_constraint_v<constraint::Nodal<basis::Primal>> = true;

template <typename T>
inline constexpr bool is_dual_nodal_constraint_v = false;

template <>
inline constexpr bool is_dual_nodal_constraint_v<constraint::Nodal<basis::Dual>> = true;

template <typename T>
inline constexpr bool is_penalty_v = false;

template <StiffnessPolicy Stiffness, RatePolicy Rate>
inline constexpr bool is_penalty_v<enforcement::Penalty<Stiffness, Rate>> = true;

template <typename T>
inline constexpr bool is_no_rate_penalty_v = false;

template <StiffnessPolicy Stiffness>
inline constexpr bool is_no_rate_penalty_v<enforcement::Penalty<Stiffness, rate::None>> = true;

template <typename T>
inline constexpr bool is_supported_response_v =
    std::same_as<T, response::Frictionless> || std::same_as<T, response::ViscousTangential> ||
    std::same_as<T, response::TiedNormal> || std::same_as<T, response::TiedFull>;

template <typename T>
struct BuiltInMethodCompatibility : std::false_type {};

template <GeometryPolicy Geometry, IntegrationPolicy Integration, ConstraintPolicy Constraint,
          EnforcementPolicy Enforcement, ResponsePolicy Response, FormulationPolicy Formulation,
          LinearizationPolicy Linearization>
struct BuiltInMethodCompatibility<
    Method<Geometry, Integration, Constraint, Enforcement, Response, Formulation, Linearization>> {
 private:
  static constexpr bool pointwise =
      is_mean_plane_overlap_v<Geometry> && std::same_as<Integration, integration::Centroid> &&
      std::same_as<Constraint, constraint::Pointwise> && is_penalty_v<Enforcement> &&
      is_supported_response_v<Response> &&
      ( std::same_as<Response, response::Frictionless> || std::same_as<Response, response::ViscousTangential> ||
        is_no_rate_penalty_v<Enforcement> ) &&
      std::same_as<Formulation, formulation::PointwiseTraction>;

  static constexpr bool projected_weak_form =
      is_projected_overlap_v<Geometry> && is_polygon_integration_v<Integration> &&
      is_dual_nodal_constraint_v<Constraint> && std::same_as<Enforcement, enforcement::LagrangeMultiplier> &&
      std::same_as<Response, response::Frictionless> && std::same_as<Formulation, formulation::WeightedWeakForm>;

  static constexpr bool conforming_weak_form =
      std::same_as<Geometry, geometry::ConformingOverlap> && is_face_integration_v<Integration> &&
      is_dual_nodal_constraint_v<Constraint> && std::same_as<Enforcement, enforcement::LagrangeMultiplier> &&
      std::same_as<Response, response::Frictionless> && std::same_as<Formulation, formulation::WeightedWeakForm>;

  static constexpr bool variational_penalty =
      is_projected_overlap_v<Geometry> && is_polygon_integration_v<Integration> &&
      ( is_primal_nodal_constraint_v<Constraint> || std::same_as<Constraint, constraint::QuadraturePoint> ) &&
      is_no_rate_penalty_v<Enforcement> && std::same_as<Response, response::Frictionless> &&
      std::same_as<Formulation, formulation::Variational>;

  static constexpr bool variational_multiplier =
      is_projected_overlap_v<Geometry> && is_polygon_integration_v<Integration> &&
      is_primal_nodal_constraint_v<Constraint> && std::same_as<Enforcement, enforcement::LagrangeMultiplier> &&
      std::same_as<Response, response::Frictionless> && std::same_as<Formulation, formulation::Variational>;

  static constexpr bool variational_external_pressure =
      is_projected_overlap_v<Geometry> && is_polygon_integration_v<Integration> &&
      is_primal_nodal_constraint_v<Constraint> && std::same_as<Enforcement, enforcement::ExternalPressure> &&
      std::same_as<Response, response::Frictionless> && std::same_as<Formulation, formulation::Variational>;

  static constexpr bool diagnostic =
      is_projected_overlap_v<Geometry> && is_polygon_integration_v<Integration> &&
      is_dual_nodal_constraint_v<Constraint> && std::same_as<Enforcement, enforcement::None> &&
      std::same_as<Response, response::Frictionless> && std::same_as<Formulation, formulation::DiagnosticWeights>;

  static constexpr bool implemented_linearization = std::same_as<Linearization, linearization::Exact>;

 public:
  static constexpr bool value =
      implemented_linearization && ( pointwise || projected_weak_form || conforming_weak_form || variational_penalty ||
                                     variational_multiplier || variational_external_pressure || diagnostic );
};

}  // namespace detail

template <typename T>
struct MethodCompatibility : std::bool_constant<detail::BuiltInMethodCompatibility<T>::value> {};

template <typename T>
concept MethodDefinition =
    requires {
      typename T::geometry_policy;
      typename T::integration_policy;
      typename T::constraint_policy;
      typename T::enforcement_policy;
      typename T::response_policy;
      typename T::formulation_policy;
      typename T::linearization_policy;
      typename T::Parameters;
    } && GeometryPolicy<typename T::geometry_policy> && IntegrationPolicy<typename T::integration_policy> &&
    ConstraintPolicy<typename T::constraint_policy> && EnforcementPolicy<typename T::enforcement_policy> &&
    ResponsePolicy<typename T::response_policy> && FormulationPolicy<typename T::formulation_policy> &&
    LinearizationPolicy<typename T::linearization_policy>;

template <typename T>
concept SupportedMethod = MethodDefinition<T> && MethodCompatibility<T>::value;

using DefaultMethod = Method<>;

static_assert( SupportedMethod<DefaultMethod> );

}  // namespace tribol

#endif
