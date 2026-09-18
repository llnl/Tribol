#include "tests/spec/SupportedMethods.hpp"

#include "tribol/Tribol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

// Requirements: LEGACY-002, DIFF-001, GEOM-003, PHYS-002, PHYS-003

namespace {

using namespace tribol;
using namespace tribol::test::spec;

SurfaceMeshView makeQuad( const std::array<Real, 12>& coordinates )
{
  static constexpr std::array<Index, 2> offsets{ 0, 4 };
  static constexpr std::array<Index, 4> connectivity{ 0, 1, 2, 3 };
  static constexpr std::array<ElementTopology, 1> topologies{ ElementTopology::Quadrilateral };
  static constexpr std::array<int, 1> attributes{ 1 };
  return {
      .dimension = 3,
      .coordinates = { { coordinates.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .element_offsets = { offsets.data(), 2 },
      .connectivity = { connectivity.data(), 4 },
      .topologies = { topologies.data(), 1 },
      .attributes = { attributes.data(), 1 },
  };
}

SurfacePairView legacyMisalignedQuads()
{
  static constexpr std::array<Real, 12> mortar{ -1.0, 1.0, 0.1, -1.0, -1.0, 0.1, 1.0, -1.0, 0.1, 1.0, 1.0, 0.1 };
  static constexpr std::array<Real, 12> nonmortar{ 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0, -2.0, 0.0 };
  return { makeQuad( nonmortar ), makeQuad( mortar ) };
}

SurfacePairView legacyAlignedQuads()
{
  static constexpr std::array<Real, 12> mortar{ -1.0, 1.0, 0.1, -1.0, -1.0, 0.1, 1.0, -1.0, 0.1, 1.0, 1.0, 0.1 };
  static constexpr std::array<Real, 12> nonmortar{ -1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0 };
  return { makeQuad( nonmortar ), makeQuad( mortar ) };
}

Real totalComponent( FieldView<const Real> field, int component )
{
  Real total{};
  for ( Index node = 0; node < field.entities; ++node ) {
    total += field( node, component );
  }
  return total;
}

bool legacyMortarWeightProjection()
{
  Contact<DiagnosticWeights>::Options options;
  options.search.expansion = 0.2;
  Contact<DiagnosticWeights> contact( legacyMisalignedQuads(), options );
  contact.updateInteractions();
  const auto result = contact.evaluate();
  constexpr std::array<Real, 16> expected_coupling{
      25.0 / 576.0, 65.0 / 576.0, 169.0 / 576.0, 65.0 / 576.0, 5.0 / 576.0, 13.0 / 576.0, 65.0 / 576.0, 25.0 / 576.0,
      1.0 / 576.0,  5.0 / 576.0,  25.0 / 576.0,  5.0 / 576.0,  5.0 / 576.0, 25.0 / 576.0, 65.0 / 576.0, 13.0 / 576.0 };
  constexpr std::array<Real, 16> expected_mass{
      49.0 / 144.0, 7.0 / 72.0, 1.0 / 36.0,  7.0 / 72.0, 7.0 / 72.0, 7.0 / 144.0, 1.0 / 72.0, 1.0 / 36.0,
      1.0 / 36.0,   1.0 / 72.0, 1.0 / 144.0, 1.0 / 72.0, 7.0 / 72.0, 1.0 / 36.0,  1.0 / 72.0, 7.0 / 144.0 };
  constexpr std::array<Real, 4> expected_mortar{ 0.5625, 0.1875, 0.0625, 0.1875 };
  constexpr std::array<Real, 4> expected_nonmortar{ 0.0625, 0.1875, 0.5625, 0.1875 };
  bool matches = true;
  for ( Index mortar_node = 0; mortar_node < 4; ++mortar_node ) {
    Real row_sum{};
    Real column_sum{};
    Real mass_row_sum{};
    for ( Index nonmortar_node = 0; nonmortar_node < 4; ++nonmortar_node ) {
      const Index entry = mortar_node * 4 + nonmortar_node;
      if ( std::abs( result.mortar_weights[entry] - expected_coupling[entry] ) > 1.0e-12 ||
           std::abs( result.mortar_mass_weights[entry] - expected_mass[entry] ) > 1.0e-12 ) {
        std::cerr << "weight entry=" << entry << " coupling=" << result.mortar_weights[entry]
                  << " expected-coupling=" << expected_coupling[entry] << " mass=" << result.mortar_mass_weights[entry]
                  << " expected-mass=" << expected_mass[entry] << '\n';
        matches = false;
      }
      row_sum += result.mortar_weights[entry];
      column_sum += result.mortar_weights[nonmortar_node * 4 + mortar_node];
      mass_row_sum += result.mortar_mass_weights[entry];
    }
    if ( std::abs( row_sum - expected_mortar[mortar_node] ) > 1.0e-12 ||
         std::abs( column_sum - expected_nonmortar[mortar_node] ) > 1.0e-12 ||
         std::abs( mass_row_sum - expected_mortar[mortar_node] ) > 1.0e-12 ) {
      std::cerr << "weight node=" << mortar_node << " row=" << row_sum
                << " expected-row=" << expected_mortar[mortar_node] << " column=" << column_sum
                << " expected-column=" << expected_nonmortar[mortar_node] << " mass-row=" << mass_row_sum << '\n';
      matches = false;
    }
  }
  return matches;
}

bool legacyMortarGapAndForce()
{
  Contact<ProjectedMultiplier>::Options options;
  options.search.expansion = 0.2;
  Contact<ProjectedMultiplier> contact( legacyMisalignedQuads(), options );
  contact.updateInteractions();
  constexpr std::array<Real, 4> multiplier{ 1.0, 1.0, 1.0, 1.0 };
  const auto result = contact.evaluate( { .multiplier = { multiplier.data(), 4 } } );
  Real weighted_gap{};
  for ( Real value : result.constraint_residual ) {
    weighted_gap += value;
  }
  return std::abs( weighted_gap + 0.1 ) < 1.0e-12 &&
         std::abs( std::abs( totalComponent( result.mortar_force, 2 ) ) - 1.0 ) < 1.0e-12 &&
         std::abs( totalComponent( result.mortar_force, 2 ) + totalComponent( result.nonmortar_force, 2 ) ) < 1.0e-12;
}

bool legacyAlignedGap()
{
  Contact<ConformingMultiplier>::Options options;
  options.search.expansion = 0.2;
  options.method.geometry.alignment_tolerance = 1.0e-12;
  Contact<ConformingMultiplier> contact( legacyAlignedQuads(), options );
  contact.updateInteractions();
  constexpr std::array<Real, 4> multiplier{};
  const auto result = contact.evaluate( { .multiplier = { multiplier.data(), 4 } } );
  Real gap_sum{};
  for ( Real value : result.gap ) {
    gap_sum += value;
  }
  return std::abs( gap_sum + 0.4 ) < 1.0e-12;
}

SurfaceMeshView makeSegment( const std::array<Real, 4>& coordinates )
{
  static constexpr std::array<Index, 2> offsets{ 0, 2 };
  static constexpr std::array<Index, 2> connectivity{ 0, 1 };
  static constexpr std::array<ElementTopology, 1> topologies{ ElementTopology::Segment };
  static constexpr std::array<int, 1> attributes{ 1 };
  return {
      .dimension = 2,
      .coordinates = { { coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .element_offsets = { offsets.data(), 2 },
      .connectivity = { connectivity.data(), 2 },
      .topologies = { topologies.data(), 1 },
      .attributes = { attributes.data(), 1 },
  };
}

SurfacePairView energySegments( const std::array<Real, 4>& mortar, const std::array<Real, 4>& nonmortar )
{
  return { makeSegment( mortar ), makeSegment( nonmortar ) };
}

bool legacyEnergyMortarActivation()
{
  constexpr std::array<Real, 4> mortar{ 1.0, 0.0, 0.0, 0.0 };
  constexpr std::array<Real, 4> open_nonmortar{ 0.2, 0.1, 0.8, 0.1 };
  constexpr std::array<Real, 4> zero_nonmortar{ 0.2, 0.0, 0.8, 0.0 };
  Contact<VariationalQuadraturePenalty>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 3.0;

  Contact<VariationalQuadraturePenalty> open_contact( energySegments( mortar, open_nonmortar ), options );
  open_contact.updateInteractions();
  const auto open = open_contact.evaluate();
  const auto open_jacobian = open_contact.assembleCoordinateJacobian( {} );
  if ( open.summary.energy != 0.0 || std::any_of( open_jacobian.values.begin(), open_jacobian.values.end(),
                                                  []( Real value ) { return value != 0.0; } ) ) {
    return false;
  }

  Contact<VariationalQuadraturePenalty> zero_contact( energySegments( mortar, zero_nonmortar ), options );
  zero_contact.updateInteractions();
  const auto zero = zero_contact.evaluate();
  const auto zero_jacobian = zero_contact.assembleCoordinateJacobian( {} );
  return zero.summary.energy == 0.0 && std::any_of( zero_jacobian.values.begin(), zero_jacobian.values.end(),
                                                    []( Real value ) { return value != 0.0; } );
}

bool legacyEnergyMortarResidualGap()
{
  constexpr std::array<Real, 4> mortar{ 1.0, 0.0, 0.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar{ 0.2, 0.1, 0.8, 0.1 };
  Contact<VariationalNodalPenalty>::Options base_options;
  base_options.search.expansion = 0.2;
  Contact<VariationalNodalPenalty> base( energySegments( mortar, nonmortar ), base_options );
  base.updateInteractions();
  const auto base_result = base.evaluate();
  std::array<Real, 2> base_gap{ base_result.weighted_gap[0], base_result.weighted_gap[1] };
  std::array<Real, 2> base_area{ base_result.tributary_area[0], base_result.tributary_area[1] };

  auto shifted_options = base_options;
  shifted_options.method.constraint.activation.residual_gap = 0.15;
  Contact<VariationalNodalPenalty> shifted( energySegments( mortar, nonmortar ), shifted_options );
  shifted.updateInteractions();
  const auto shifted_result = shifted.evaluate();
  for ( Index node = 0; node < 2; ++node ) {
    if ( std::abs( shifted_result.tributary_area[node] - base_area[node] ) > 1.0e-14 ||
         std::abs( shifted_result.weighted_gap[node] - ( base_gap[node] - 0.15 * base_area[node] ) ) > 1.0e-14 ) {
      return false;
    }
  }
  return shifted_result.summary.energy > 0.0;
}

bool legacyEnergyMortarDerivatives()
{
  constexpr std::array<Real, 4> mortar{ 1.0, 0.0, 0.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar{ 0.2, -0.1, 0.8, -0.1 };
  Contact<VariationalQuadraturePenalty>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 3.0;
  options.method.constraint.activation.residual_gap = 0.15;
  Contact<VariationalQuadraturePenalty> contact( energySegments( mortar, nonmortar ), options );
  contact.updateInteractions();
  const auto base = contact.evaluate();
  const Real exact_force = base.mortar_force( 0, 1 );
  const auto jacobian = contact.assembleCoordinateJacobian( {} );
  constexpr Index mortar_y0 = 1;
  constexpr Real step = 1.0e-6;
  auto plus_mortar = mortar;
  auto minus_mortar = mortar;
  plus_mortar[mortar_y0] += step;
  minus_mortar[mortar_y0] -= step;
  contact.updateGeometry( energySegments( plus_mortar, nonmortar ) );
  const Real plus_energy = contact.evaluate().summary.energy;
  const Real plus_force = contact.evaluate().mortar_force( 0, 1 );
  contact.updateGeometry( energySegments( minus_mortar, nonmortar ) );
  const Real minus_energy = contact.evaluate().summary.energy;
  const Real minus_force = contact.evaluate().mortar_force( 0, 1 );
  const Real force_fd = ( plus_energy - minus_energy ) / ( 2.0 * step );
  const Real stiffness_fd = ( plus_force - minus_force ) / ( 2.0 * step );
  return std::abs( exact_force - force_fd ) < 1.0e-7 &&
         std::abs( jacobian( mortar_y0, mortar_y0 ) - stiffness_fd ) < 1.0e-5;
}

bool legacySmoothedIntegration()
{
  constexpr std::array<Real, 4> mortar{ 1.0, 0.0, 0.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar{ -0.05, -0.1, 0.6, -0.1 };
  Contact<SmoothedVariationalNodalPenalty>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 3.0;
  options.method.integration.endpoint_width = 0.1;
  Contact<SmoothedVariationalNodalPenalty> contact( energySegments( mortar, nonmortar ), options );
  contact.setInteractions( { std::array<ElementPair, 1>{ ElementPair{ 0, 0 } }.data(), 1 } );
  const auto result = contact.evaluate();

  constexpr Real lower = 0.4;
  constexpr Real upper = 0.99375;
  constexpr Real first_area = ( upper - lower ) - 0.5 * ( upper * upper - lower * lower );
  constexpr Real second_area = 0.5 * ( upper * upper - lower * lower );
  constexpr Real expected_energy = 0.5 * 3.0 * 0.1 * 0.1 * ( upper - lower );
  return std::abs( result.tributary_area[0] - first_area ) < 1.0e-13 &&
         std::abs( result.tributary_area[1] - second_area ) < 1.0e-13 &&
         std::abs( result.weighted_gap[0] + 0.1 * first_area ) < 1.0e-13 &&
         std::abs( result.weighted_gap[1] + 0.1 * second_area ) < 1.0e-13 &&
         std::abs( result.summary.energy - expected_energy ) < 1.0e-13;
}

template <SupportedMethod MethodType>
bool smoothedEnergyDerivativesMatchFiniteDifference()
{
  const std::array<Real, 4> mortar{ 1.05, 0.03, -0.1, -0.02 };
  const std::array<Real, 4> nonmortar{ -0.04, -0.14, 0.76, -0.08 };
  typename Contact<MethodType>::Options options;
  options.search.expansion = 0.4;
  options.method.enforcement.stiffness.value = 3.0;
  options.method.integration.endpoint_width = 0.1;
  options.method.constraint.activation.residual_gap = 0.03;
  Contact<MethodType> contact( energySegments( mortar, nonmortar ), options );
  const std::array<ElementPair, 1> pairs{ ElementPair{ 0, 0 } };
  contact.setInteractions( { pairs.data(), 1 } );
  const auto base = contact.evaluate();
  std::array<Real, 8> base_force{};
  for ( Index node = 0; node < 2; ++node ) {
    for ( int component = 0; component < 2; ++component ) {
      base_force[2 * node + component] = base.mortar_force( node, component );
      base_force[4 + 2 * node + component] = base.nonmortar_force( node, component );
    }
  }
  const auto jacobian = contact.assembleCoordinateJacobian( {} );
  std::array<Real, 64> exact_jacobian{};
  std::copy( jacobian.values.begin(), jacobian.values.end(), exact_jacobian.begin() );

  constexpr Real step = 1.0e-6;
  for ( Index column = 0; column < 8; ++column ) {
    auto plus_mortar = mortar;
    auto minus_mortar = mortar;
    auto plus_nonmortar = nonmortar;
    auto minus_nonmortar = nonmortar;
    if ( column < 4 ) {
      plus_mortar[column] += step;
      minus_mortar[column] -= step;
    } else {
      plus_nonmortar[column - 4] += step;
      minus_nonmortar[column - 4] -= step;
    }
    contact.updateGeometry( energySegments( plus_mortar, plus_nonmortar ) );
    const auto plus = contact.evaluate();
    const Real plus_energy = plus.summary.energy;
    std::array<Real, 8> plus_force{};
    for ( Index node = 0; node < 2; ++node ) {
      for ( int component = 0; component < 2; ++component ) {
        plus_force[2 * node + component] = plus.mortar_force( node, component );
        plus_force[4 + 2 * node + component] = plus.nonmortar_force( node, component );
      }
    }
    contact.updateGeometry( energySegments( minus_mortar, minus_nonmortar ) );
    const auto minus = contact.evaluate();
    const Real energy_gradient = ( plus_energy - minus.summary.energy ) / ( 2.0 * step );
    if ( std::abs( base_force[column] - energy_gradient ) > 2.0e-7 ) {
      std::cerr << "smoothed energy gradient mismatch at " << column << ": " << base_force[column] << " versus "
                << energy_gradient << '\n';
      return false;
    }
    for ( Index row = 0; row < 8; ++row ) {
      const Index node = row % 4 / 2;
      const int component = row % 2;
      const Real minus_force =
          row < 4 ? minus.mortar_force( node, component ) : minus.nonmortar_force( node, component );
      const Real finite_difference = ( plus_force[row] - minus_force ) / ( 2.0 * step );
      if ( std::abs( exact_jacobian[row * 8 + column] - finite_difference ) > 2.0e-5 ) {
        std::cerr << "smoothed Hessian mismatch at " << row << ',' << column << ": " << exact_jacobian[row * 8 + column]
                  << " versus " << finite_difference << '\n';
        return false;
      }
    }
  }
  return true;
}

bool assembledNodalGapDefinesEnergy()
{
  constexpr std::array<Real, 6> mortar_coordinates{ 2.0, 0.0, 1.0, 0.0, 0.0, 0.0 };
  constexpr std::array<Real, 8> nonmortar_coordinates{ 1.0, -0.1, 2.0, -0.1, 0.0, -0.2, 1.0, -0.2 };
  constexpr std::array<Index, 3> mortar_offsets{ 0, 2, 4 };
  constexpr std::array<Index, 4> mortar_connectivity{ 0, 1, 1, 2 };
  constexpr std::array<Index, 3> nonmortar_offsets{ 0, 2, 4 };
  constexpr std::array<Index, 4> nonmortar_connectivity{ 0, 1, 2, 3 };
  constexpr std::array<ElementTopology, 2> topologies{ ElementTopology::Segment, ElementTopology::Segment };
  constexpr std::array<int, 2> attributes{ 1, 1 };
  const SurfaceMeshView mortar{
      .dimension = 2,
      .coordinates = { { mortar_coordinates.data(), 6 }, 3, 2, FieldLayout::Interleaved },
      .element_offsets = { mortar_offsets.data(), 3 },
      .connectivity = { mortar_connectivity.data(), 4 },
      .topologies = { topologies.data(), 2 },
      .attributes = { attributes.data(), 2 },
  };
  const SurfaceMeshView nonmortar{
      .dimension = 2,
      .coordinates = { { nonmortar_coordinates.data(), 8 }, 4, 2, FieldLayout::Interleaved },
      .element_offsets = { nonmortar_offsets.data(), 3 },
      .connectivity = { nonmortar_connectivity.data(), 4 },
      .topologies = { topologies.data(), 2 },
      .attributes = { attributes.data(), 2 },
  };
  Contact<VariationalNodalPenalty>::Options options;
  options.method.enforcement.stiffness.value = 2.0;
  Contact<VariationalNodalPenalty> contact( { mortar, nonmortar }, options );
  constexpr std::array<ElementPair, 2> pairs{ ElementPair{ 0, 0 }, ElementPair{ 1, 1 } };
  contact.setInteractions( { pairs.data(), 2 } );
  const auto result = contact.evaluate();
  return std::abs( result.gap[0] + 0.1 ) < 1.0e-13 && std::abs( result.gap[1] + 0.15 ) < 1.0e-13 &&
         std::abs( result.gap[2] + 0.2 ) < 1.0e-13 && std::abs( result.summary.energy - 0.0475 ) < 1.0e-13;
}

}  // namespace

int main()
{
  if ( !legacyMortarWeightProjection() ) {
    return 1;
  }
  if ( !legacyMortarGapAndForce() ) {
    return 2;
  }
  if ( !legacyAlignedGap() ) {
    return 3;
  }
  if ( !legacyEnergyMortarActivation() ) {
    return 4;
  }
  if ( !legacyEnergyMortarResidualGap() ) {
    return 5;
  }
  if ( !legacyEnergyMortarDerivatives() ) {
    return 6;
  }
  if ( !legacySmoothedIntegration() ) {
    return 7;
  }
  if ( !smoothedEnergyDerivativesMatchFiniteDifference<SmoothedVariationalQuadraturePenalty>() ||
       !smoothedEnergyDerivativesMatchFiniteDifference<SmoothedVariationalNodalPenalty>() ) {
    return 8;
  }
  return assembledNodalGapDefinesEnergy() ? 0 : 9;
}
