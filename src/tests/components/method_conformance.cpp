#include "tests/spec/SupportedMethods.hpp"

#include "tribol/Tribol.hpp"

#include <array>
#include <cmath>

// Requirements: CONS-001, ENF-001, ENF-003, ENF-004, FORM-001, GEOM-002, OUTPUT-001, PHYS-001, PHYS-002, PHYS-003

namespace {

using namespace tribol;
using namespace tribol::test::spec;

SurfaceMeshView makeSurface( const std::array<Real, 12>& coordinates )
{
  static constexpr std::array<Index, 2> offsets{ 0, 4 };
  static constexpr std::array<Index, 4> connectivity{ 0, 1, 2, 3 };
  static constexpr std::array<ElementTopology, 1> topologies{ ElementTopology::Quadrilateral };
  static constexpr std::array<int, 1> attributes{ 1 };
  return {
      .dimension = 3,
      .coordinates =
          FieldView<const Real>{ ArrayView<const Real>{ coordinates.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .element_offsets = ArrayView<const Index>{ offsets.data(), 2 },
      .connectivity = ArrayView<const Index>{ connectivity.data(), 4 },
      .topologies = ArrayView<const ElementTopology>{ topologies.data(), 1 },
      .attributes = ArrayView<const int>{ attributes.data(), 1 },
  };
}

SurfacePairView parallelFaces()
{
  static constexpr std::array<Real, 12> mortar_coordinates{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                            1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  static constexpr std::array<Real, 12> nonmortar_coordinates{ 0.0, 0.0, -0.1, 0.0, 1.0, -0.1,
                                                               1.0, 1.0, -0.1, 1.0, 0.0, -0.1 };
  return { makeSurface( mortar_coordinates ), makeSurface( nonmortar_coordinates ) };
}

Real totalComponent( FieldView<const Real> field, int component )
{
  Real result{};
  for ( Index node = 0; node < field.entities; ++node ) {
    result += field( node, component );
  }
  return result;
}

bool projectedMultiplierContract()
{
  Contact<ProjectedMultiplier>::Options options;
  options.search.expansion = 0.2;
  Contact<ProjectedMultiplier> contact( parallelFaces(), options );
  contact.updateInteractions();
  constexpr std::array<Real, 4> multiplier{ -2.0, -2.0, -2.0, -2.0 };
  const auto result = contact.evaluate( { .multiplier = { multiplier.data(), 4 } } );
  const Real tolerance = 1.0e-12;
  Real constraint_sum{};
  for ( Real value : result.constraint_residual ) {
    constraint_sum += value;
  }
  return result.summary.active_interactions == 1 && result.summary.quadrature_points == 6 &&
         std::abs( totalComponent( result.mortar_force, 2 ) - 2.0 ) < tolerance &&
         std::abs( totalComponent( result.nonmortar_force, 2 ) + 2.0 ) < tolerance &&
         std::abs( constraint_sum + 0.1 ) < tolerance;
}

bool conformingMultiplierContract()
{
  Contact<ConformingMultiplier>::Options options;
  options.search.expansion = 0.2;
  Contact<ConformingMultiplier> contact( parallelFaces(), options );
  contact.updateInteractions();
  constexpr std::array<Real, 4> multiplier{ -1.0, -1.0, -1.0, -1.0 };
  const auto result = contact.evaluate( { .multiplier = { multiplier.data(), 4 } } );
  return result.summary.active_interactions == 1 &&
         std::abs( totalComponent( result.mortar_force, 2 ) - 1.0 ) < 1.0e-12;
}

bool variationalMultiplierContract()
{
  Contact<VariationalMultiplier>::Options options;
  options.search.expansion = 0.2;
  Contact<VariationalMultiplier> contact( parallelFaces(), options );
  contact.updateInteractions();
  constexpr std::array<Real, 4> multiplier{ -2.0, -2.0, -2.0, -2.0 };
  const auto result = contact.evaluate( { .multiplier = { multiplier.data(), 4 } } );
  Real constraint_sum{};
  for ( Real value : result.constraint_residual ) {
    constraint_sum += value;
  }
  return result.summary.active_interactions == 1 &&
         std::abs( totalComponent( result.mortar_force, 2 ) - 2.0 ) < 1.0e-12 &&
         std::abs( totalComponent( result.nonmortar_force, 2 ) + 2.0 ) < 1.0e-12 &&
         std::abs( constraint_sum + 0.1 ) < 1.0e-12;
}

bool diagnosticWeightsContract()
{
  Contact<DiagnosticWeights>::Options options;
  options.search.expansion = 0.2;
  Contact<DiagnosticWeights> contact( parallelFaces(), options );
  contact.updateInteractions();
  const auto result = contact.evaluate();
  Real total_weight{};
  for ( Real value : result.mortar_weights ) {
    total_weight += value;
  }
  return result.summary.active_interactions == 1 && std::abs( total_weight - 1.0 ) < 1.0e-12;
}

bool variationalPenaltyContract()
{
  Contact<VariationalNodalPenalty>::Options options;
  options.method.enforcement.stiffness.value = 10.0;
  options.search.expansion = 0.2;
  Contact<VariationalNodalPenalty> contact( parallelFaces(), options );
  contact.updateInteractions();
  const auto result = contact.evaluate();
  Real weighted_gap{};
  Real tributary_area{};
  for ( Index node = 0; node < result.weighted_gap.size(); ++node ) {
    weighted_gap += result.weighted_gap[node];
    tributary_area += result.tributary_area[node];
  }
  return std::abs( result.summary.energy - 0.05 ) < 1.0e-12 &&
         std::abs( totalComponent( result.mortar_force, 2 ) - 1.0 ) < 1.0e-12 &&
         std::abs( weighted_gap + 0.1 ) < 1.0e-12 && std::abs( tributary_area - 1.0 ) < 1.0e-12;
}

bool quadratureAndExternalPressureContract()
{
  Contact<VariationalQuadraturePenalty>::Options penalty_options;
  penalty_options.method.enforcement.stiffness.value = 10.0;
  penalty_options.search.expansion = 0.2;
  Contact<VariationalQuadraturePenalty> penalty( parallelFaces(), penalty_options );
  penalty.updateInteractions();
  const auto penalty_result = penalty.evaluate();

  Contact<VariationalExternalPressure>::Options external_options;
  external_options.search.expansion = 0.2;
  Contact<VariationalExternalPressure> external( parallelFaces(), external_options );
  external.updateInteractions();
  constexpr std::array<Real, 4> pressure{ -3.0, -3.0, -3.0, -3.0 };
  const auto external_result = external.evaluate( { .external_pressure = { pressure.data(), 4 } } );
  return penalty_result.quadrature_gap.size() == 6 && penalty_result.quadrature_pressure.size() == 6 &&
         std::abs( penalty_result.summary.energy - 0.05 ) < 1.0e-12 &&
         std::abs( totalComponent( external_result.mortar_force, 2 ) - 3.0 ) < 1.0e-12 &&
         std::abs( external_result.summary.energy - 0.3 ) < 1.0e-12;
}

}  // namespace

int main()
{
  return projectedMultiplierContract() && conformingMultiplierContract() && variationalMultiplierContract() &&
                 diagnosticWeightsContract() && variationalPenaltyContract() && quadratureAndExternalPressureContract()
             ? 0
             : 1;
}
