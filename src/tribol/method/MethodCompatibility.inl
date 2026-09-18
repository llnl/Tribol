#ifndef TRIBOL_METHOD_METHODCOMPATIBILITY_INL_
#define TRIBOL_METHOD_METHODCOMPATIBILITY_INL_

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
inline constexpr bool is_polygon_integration_v<integration::Polygon<Order>> = Order == 1 || Order == 2 || Order == 4;

template <typename T>
inline constexpr bool is_smoothed_segment_integration_v = false;

template <int Points>
inline constexpr bool is_smoothed_segment_integration_v<integration::SmoothedSegment<Points>> =
    Points >= 1 && Points <= 3;

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
inline constexpr bool is_constant_no_rate_penalty_v = false;

template <>
inline constexpr bool is_constant_no_rate_penalty_v<enforcement::Penalty<stiffness::Constant, rate::None>> = true;

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
      is_primal_nodal_constraint_v<Constraint> && std::same_as<Enforcement, enforcement::LagrangeMultiplier> &&
      std::same_as<Response, response::Frictionless> && std::same_as<Formulation, formulation::WeightedWeakForm>;

  static constexpr bool conforming_weak_form =
      std::same_as<Geometry, geometry::ConformingOverlap> && is_face_integration_v<Integration> &&
      is_primal_nodal_constraint_v<Constraint> && std::same_as<Enforcement, enforcement::LagrangeMultiplier> &&
      std::same_as<Response, response::Frictionless> && std::same_as<Formulation, formulation::WeightedWeakForm>;

  static constexpr bool variational_integration =
      is_polygon_integration_v<Integration> || is_smoothed_segment_integration_v<Integration>;

  static constexpr bool variational_penalty =
      is_projected_overlap_v<Geometry> && variational_integration &&
      ( is_primal_nodal_constraint_v<Constraint> || std::same_as<Constraint, constraint::QuadraturePoint> ) &&
      is_constant_no_rate_penalty_v<Enforcement> && std::same_as<Response, response::Frictionless> &&
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
      is_primal_nodal_constraint_v<Constraint> && std::same_as<Enforcement, enforcement::None> &&
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

#endif
