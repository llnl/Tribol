#include "tests/spec/SupportedMethods.hpp"

#include "tribol/Tribol.hpp"

#include <array>
#include <stdexcept>
#include <type_traits>

// Requirements: API-001, API-002, API-003, API-004, CORE-001, DIFF-001, SEARCH-001

namespace {

using namespace tribol;
using namespace tribol::test::spec;

static_assert( SupportedMethod<DefaultMethod> );
static_assert( std::same_as<typename DefaultMethod::linearization_policy, linearization::Exact> );
static_assert( SupportedMethod<PointwisePenalty> );
static_assert( SupportedMethod<PointwiseMaterialRateViscous> );
static_assert( SupportedMethod<PointwiseTiedNormal> );
static_assert( SupportedMethod<PointwiseTiedFull> );
static_assert( SupportedMethod<ProjectedMultiplier> );
static_assert( SupportedMethod<ConformingMultiplier> );
static_assert( SupportedMethod<DiagnosticWeights> );
static_assert( SupportedMethod<VariationalNodalPenalty> );
static_assert( SupportedMethod<VariationalQuadraturePenalty> );
static_assert( SupportedMethod<VariationalMultiplier> );
static_assert( SupportedMethod<VariationalExternalPressure> );

using UnsupportedPointwiseMultiplier =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::LagrangeMultiplier, response::Frictionless, formulation::PointwiseTraction>;
using UnsupportedConformingPenalty =
    Method<geometry::ConformingOverlap, integration::Face<2>, constraint::Nodal<basis::Dual>, enforcement::Penalty<>,
           response::Frictionless, formulation::WeightedWeakForm>;
using UnsupportedViscousVariational =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::QuadraturePoint,
           enforcement::Penalty<>, response::ViscousTangential, formulation::Variational>;
using UnsupportedAnalytic =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::Penalty<>, response::Frictionless, formulation::PointwiseTraction, linearization::Analytic>;

static_assert( !SupportedMethod<UnsupportedPointwiseMultiplier> );
static_assert( !SupportedMethod<UnsupportedConformingPenalty> );
static_assert( !SupportedMethod<UnsupportedViscousVariational> );
static_assert( !SupportedMethod<UnsupportedAnalytic> );
static_assert( execution::SupportedContactExecution<DefaultMethod, execution::Cuda> );
static_assert( !execution::SupportedContactExecution<ProjectedMultiplier, execution::Cuda> );

static_assert( std::is_trivially_copyable_v<ArrayView<const Real>> );
static_assert( std::is_trivially_copyable_v<SurfaceMeshView> );

static_assert( MethodTraits<PointwiseMaterialRateViscous>::capabilities.needs_velocity );
static_assert( MethodTraits<PointwiseMaterialRateViscous>::capabilities.needs_material_fields );
static_assert( MethodTraits<PointwiseTiedNormal>::capabilities.needs_reference_coordinates );
static_assert( MethodTraits<ProjectedMultiplier>::capabilities.needs_multiplier );
static_assert( MethodTraits<VariationalExternalPressure>::capabilities.accepts_external_pressure );
static_assert( MethodTraits<VariationalNodalPenalty>::capabilities.produces_energy );
static_assert( MethodTraits<DiagnosticWeights>::capabilities.produces_diagnostic_weights );

bool meshViewContract()
{
  constexpr std::array<Real, 8> coordinates{ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
  constexpr std::array<Index, 3> offsets{ 0, 2, 4 };
  constexpr std::array<Index, 4> connectivity{ 0, 1, 2, 3 };
  constexpr std::array<ElementTopology, 2> topologies{ ElementTopology::Segment, ElementTopology::Segment };
  constexpr std::array<int, 2> attributes{ 1, 2 };

  const SurfaceMeshView mesh{
      .dimension = 2,
      .coordinates =
          FieldView<const Real>{ ArrayView<const Real>{ coordinates.data(), static_cast<Index>( coordinates.size() ) },
                                 4, 2, FieldLayout::Interleaved },
      .element_offsets = ArrayView<const Index>{ offsets.data(), static_cast<Index>( offsets.size() ) },
      .connectivity = ArrayView<const Index>{ connectivity.data(), static_cast<Index>( connectivity.size() ) },
      .topologies = ArrayView<const ElementTopology>{ topologies.data(), static_cast<Index>( topologies.size() ) },
      .attributes = ArrayView<const int>{ attributes.data(), static_cast<Index>( attributes.size() ) },
  };
  auto invalid_mesh = mesh;
  constexpr std::array<Index, 4> invalid_connectivity{ 0, 1, 2, 4 };
  invalid_mesh.connectivity = { invalid_connectivity.data(), static_cast<Index>( invalid_connectivity.size() ) };
  return mesh.isStructurallyValid() && !invalid_mesh.isStructurallyValid() && mesh.numberOfNodes() == 4 &&
         mesh.numberOfElements() == 2 && mesh.coordinates( 3, 1 ) == 1.0;
}

bool contactLifecycleContract()
{
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> moved_nonmortar_coordinates{ 0.0, 0.05, 1.0, 0.05 };
  constexpr std::array<Index, 2> offsets{ 0, 2 };
  constexpr std::array<Index, 2> connectivity{ 0, 1 };
  constexpr std::array<ElementTopology, 1> topologies{ ElementTopology::Segment };
  constexpr std::array<int, 1> attributes{ 1 };

  const auto make_surface = [&]( const std::array<Real, 4>& coordinates ) {
    return SurfaceMeshView{
        .dimension = 2,
        .coordinates = FieldView<const Real>{ ArrayView<const Real>{ coordinates.data(),
                                                                     static_cast<Index>( coordinates.size() ) },
                                              2, 2, FieldLayout::Interleaved },
        .element_offsets = ArrayView<const Index>{ offsets.data(), static_cast<Index>( offsets.size() ) },
        .connectivity = ArrayView<const Index>{ connectivity.data(), static_cast<Index>( connectivity.size() ) },
        .topologies = ArrayView<const ElementTopology>{ topologies.data(), static_cast<Index>( topologies.size() ) },
        .attributes = ArrayView<const int>{ attributes.data(), static_cast<Index>( attributes.size() ) },
    };
  };

  const SurfacePairView initial{ make_surface( mortar_coordinates ), make_surface( nonmortar_coordinates ) };
  Contact<> contact( initial );
  if ( contact.hasInteractions() ) {
    return false;
  }

  contact.updateInteractions();
  if ( !contact.hasInteractions() || contact.interactions().size() != 1 ||
       contact.interactions()[0] != ElementPair{ 0, 0 } ) {
    return false;
  }

  const auto interaction_version = contact.interactionVersion();
  contact.updateGeometry( { make_surface( mortar_coordinates ), make_surface( moved_nonmortar_coordinates ) } );
  if ( contact.geometryVersion().value() != 1 || contact.interactionVersion() != interaction_version ||
       contact.interactions().size() != 1 ) {
    return false;
  }

  contact.rebuildGeometry( initial );
  return !contact.hasInteractions() && contact.interactions().empty() && contact.geometryVersion().value() == 2;
}

bool pointwisePenaltyContract()
{
  constexpr std::array<Real, 4> mortar_coordinates{ 1.0, 0.0, 0.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  constexpr std::array<Index, 2> offsets{ 0, 2 };
  constexpr std::array<Index, 2> connectivity{ 0, 1 };
  constexpr std::array<ElementTopology, 1> topologies{ ElementTopology::Segment };
  constexpr std::array<int, 1> attributes{ 1 };
  const auto make_surface = [&]( const std::array<Real, 4>& coordinates ) {
    return SurfaceMeshView{
        .dimension = 2,
        .coordinates = FieldView<const Real>{ ArrayView<const Real>{ coordinates.data(),
                                                                     static_cast<Index>( coordinates.size() ) },
                                              2, 2, FieldLayout::Interleaved },
        .element_offsets = ArrayView<const Index>{ offsets.data(), static_cast<Index>( offsets.size() ) },
        .connectivity = ArrayView<const Index>{ connectivity.data(), static_cast<Index>( connectivity.size() ) },
        .topologies = ArrayView<const ElementTopology>{ topologies.data(), static_cast<Index>( topologies.size() ) },
        .attributes = ArrayView<const int>{ attributes.data(), static_cast<Index>( attributes.size() ) },
    };
  };

  using ContactType = Contact<PointwisePenalty>;
  ContactType::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 10.0;
  ContactType contact( { make_surface( mortar_coordinates ), make_surface( nonmortar_coordinates ) }, options );
  contact.updateInteractions();

  std::array<Real, 4> mortar_residual{};
  std::array<Real, 4> nonmortar_residual{};
  ContactResidualView residual{
      .mortar = FieldView<Real>{ ArrayView<Real>{ mortar_residual.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = FieldView<Real>{ ArrayView<Real>{ nonmortar_residual.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  const auto summary = contact.addResidual( {}, residual );

  std::array<Real, 4> mortar_direction{};
  std::array<Real, 4> nonmortar_direction{ 0.0, 1.0, 0.0, 1.0 };
  const ContactDirectionView direction{
      .mortar =
          FieldView<const Real>{ ArrayView<const Real>{ mortar_direction.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = FieldView<const Real>{ ArrayView<const Real>{ nonmortar_direction.data(), 4 }, 2, 2,
                                          FieldLayout::Interleaved },
  };
  std::array<Real, 4> mortar_derivative{};
  std::array<Real, 4> nonmortar_derivative{};
  ContactResidualView derivative{
      .mortar = FieldView<Real>{ ArrayView<Real>{ mortar_derivative.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = FieldView<Real>{ ArrayView<Real>{ nonmortar_derivative.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  Contact<> exact_contact( { make_surface( mortar_coordinates ), make_surface( nonmortar_coordinates ) },
                           Contact<>::Options{ .search = { .expansion = 0.2 } } );
  exact_contact.updateInteractions();
  exact_contact.applyCoordinateDerivative( {}, direction, derivative );

  const Real tolerance = 1.0e-12;
  const bool energy_matches = std::abs( summary.energy - 0.05 ) < tolerance;
  const bool active_pair = summary.active_interactions == 1;
  const bool expected_force =
      std::abs( mortar_residual[1] - 0.5 ) < tolerance && std::abs( mortar_residual[3] - 0.5 ) < tolerance &&
      std::abs( nonmortar_residual[1] + 0.5 ) < tolerance && std::abs( nonmortar_residual[3] + 0.5 ) < tolerance;
  const bool balanced =
      std::abs( mortar_residual[0] + mortar_residual[2] + nonmortar_residual[0] + nonmortar_residual[2] ) < tolerance &&
      std::abs( mortar_residual[1] + mortar_residual[3] + nonmortar_residual[1] + nonmortar_residual[3] ) < tolerance;
  const bool derivative_matches =
      std::abs( mortar_derivative[1] + 0.5 ) < 1.0e-8 && std::abs( mortar_derivative[3] + 0.5 ) < 1.0e-8 &&
      std::abs( nonmortar_derivative[1] - 0.5 ) < 1.0e-8 && std::abs( nonmortar_derivative[3] - 0.5 ) < 1.0e-8;
  return energy_matches && active_pair && expected_force && balanced && derivative_matches;
}

}  // namespace

int main() { return meshViewContract() && contactLifecycleContract() && pointwisePenaltyContract() ? 0 : 1; }
