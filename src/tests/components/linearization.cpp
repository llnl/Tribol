#include "tests/spec/SupportedMethods.hpp"

#include "tribol/Tribol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

// Requirements: DIFF-001, DIFF-002, DIFF-003

namespace {

using namespace tribol;
using namespace tribol::test::spec;

using UnsupportedAnalytic =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::QuadraturePoint,
           enforcement::Penalty<>, response::Frictionless, formulation::Variational, linearization::Analytic>;
using UnsupportedEnzyme =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<2>, constraint::QuadraturePoint,
           enforcement::Penalty<>, response::Frictionless, formulation::Variational, linearization::Enzyme>;

static_assert( !SupportedMethod<UnsupportedAnalytic> );
static_assert( !SupportedMethod<UnsupportedEnzyme> );

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
      .element_offsets = { offsets.data(), 2 },
      .connectivity = { connectivity.data(), 4 },
      .topologies = { topologies.data(), 1 },
      .attributes = { attributes.data(), 1 },
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

SurfacePairView projectedFaces()
{
  static constexpr std::array<Real, 12> mortar_coordinates{ 0.0, 0.0, 0.0,  1.0, 0.0, 0.02,
                                                            1.0, 1.0, 0.01, 0.0, 1.0, -0.01 };
  static constexpr std::array<Real, 12> nonmortar_coordinates{ 0.15, 0.10, -0.12, 0.12, 0.88, -0.10,
                                                               0.84, 0.91, -0.08, 0.87, 0.09, -0.11 };
  return { makeSurface( mortar_coordinates ), makeSurface( nonmortar_coordinates ) };
}

template <SupportedMethod MethodType>
SurfacePairView derivativeSurfaces()
{
  if constexpr ( std::same_as<typename MethodType::geometry_policy, geometry::ConformingOverlap> ) {
    return parallelFaces();
  } else {
    return projectedFaces();
  }
}

struct ResidualSnapshot {
  std::array<Real, 12> mortar{};
  std::array<Real, 12> nonmortar{};
  std::array<Real, 4> constraint{};
};

template <SupportedMethod MethodType>
ResidualSnapshot evaluateResidual( Contact<MethodType>& contact, const ContactStateView& state,
                                   const SurfacePairView& surfaces )
{
  contact.updateGeometry( surfaces );
  ResidualSnapshot result;
  ContactResidualView residual{
      .mortar = { { result.mortar.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .nonmortar = { { result.nonmortar.data(), 12 }, 4, 3, FieldLayout::Interleaved },
  };
  if constexpr ( MethodTraits<MethodType>::capabilities.produces_constraint_residual ) {
    residual.constraint = { result.constraint.data(), 4 };
  }
  contact.addResidual( state, residual );
  return result;
}

void perturbCoordinates( const SurfaceMeshView& surface, FieldView<const Real> direction, Real scale,
                         std::array<Real, 12>& coordinates )
{
  for ( Index node = 0; node < surface.numberOfNodes(); ++node ) {
    for ( int component = 0; component < surface.dimension; ++component ) {
      const Index entry = node * surface.dimension + component;
      coordinates[entry] = surface.coordinates( node, component ) + scale * direction( node, component );
    }
  }
}

bool closeDerivative( Real exact, Real finite_difference, const char* field, Index entry )
{
  constexpr Real absolute_tolerance = 2.0e-7;
  constexpr Real relative_tolerance = 2.0e-5;
  const Real scale = std::max( std::abs( exact ), std::abs( finite_difference ) );
  if ( !std::isfinite( exact ) || !std::isfinite( finite_difference ) ||
       std::abs( exact - finite_difference ) > absolute_tolerance + relative_tolerance * scale ) {
    std::cerr << "finite-difference mismatch field=" << field << " entry=" << entry << " exact=" << exact
              << " finite_difference=" << finite_difference << '\n';
    return false;
  }
  return true;
}

template <SupportedMethod MethodType>
bool assembledMatchesAction( typename Contact<MethodType>::Options options, const ContactStateView& state )
{
  options.search.expansion = 0.2;
  if constexpr ( std::same_as<typename MethodType::geometry_policy, geometry::ConformingOverlap> ) {
    options.method.geometry.alignment_tolerance = 1.0e-3;
  }
  const SurfacePairView surfaces = derivativeSurfaces<MethodType>();
  Contact<MethodType> contact( surfaces, options );
  contact.updateInteractions();
  std::array<Real, 12> mortar_direction{ 0.02, -0.01, 0.03, -0.01, 0.02, -0.02, 0.01, 0.03, 0.04, -0.02, -0.01, 0.01 };
  std::array<Real, 12> nonmortar_direction{ -0.01, 0.02, 0.05, 0.03, -0.01, 0.02,
                                            -0.02, 0.01, 0.04, 0.01, -0.03, 0.03 };
  const ContactDirectionView direction{
      .mortar = { { mortar_direction.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_direction.data(), 12 }, 4, 3, FieldLayout::Interleaved },
  };
  std::array<Real, 12> mortar_derivative{};
  std::array<Real, 12> nonmortar_derivative{};
  std::array<Real, 4> constraint_derivative{};
  ContactResidualView derivative{
      .mortar = { { mortar_derivative.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_derivative.data(), 12 }, 4, 3, FieldLayout::Interleaved },
  };
  if constexpr ( MethodTraits<MethodType>::capabilities.produces_constraint_residual ) {
    derivative.constraint = { constraint_derivative.data(), 4 };
  }
  contact.applyCoordinateDerivative( state, direction, derivative );

  constexpr Real finite_difference_step = 1.0e-6;
  std::array<Real, 12> plus_mortar_coordinates{};
  std::array<Real, 12> plus_nonmortar_coordinates{};
  std::array<Real, 12> minus_mortar_coordinates{};
  std::array<Real, 12> minus_nonmortar_coordinates{};
  perturbCoordinates( surfaces.mortar, direction.mortar, finite_difference_step, plus_mortar_coordinates );
  perturbCoordinates( surfaces.nonmortar, direction.nonmortar, finite_difference_step, plus_nonmortar_coordinates );
  perturbCoordinates( surfaces.mortar, direction.mortar, -finite_difference_step, minus_mortar_coordinates );
  perturbCoordinates( surfaces.nonmortar, direction.nonmortar, -finite_difference_step, minus_nonmortar_coordinates );
  const auto plus = evaluateResidual(
      contact, state, { makeSurface( plus_mortar_coordinates ), makeSurface( plus_nonmortar_coordinates ) } );
  const auto minus = evaluateResidual(
      contact, state, { makeSurface( minus_mortar_coordinates ), makeSurface( minus_nonmortar_coordinates ) } );
  for ( Index entry = 0; entry < 12; ++entry ) {
    const Real mortar_finite_difference =
        ( plus.mortar[entry] - minus.mortar[entry] ) / ( 2.0 * finite_difference_step );
    const Real nonmortar_finite_difference =
        ( plus.nonmortar[entry] - minus.nonmortar[entry] ) / ( 2.0 * finite_difference_step );
    if ( !closeDerivative( mortar_derivative[entry], mortar_finite_difference, "mortar", entry ) ||
         !closeDerivative( nonmortar_derivative[entry], nonmortar_finite_difference, "nonmortar", entry ) ) {
      return false;
    }
  }
  if constexpr ( MethodTraits<MethodType>::capabilities.produces_constraint_residual ) {
    for ( Index entry = 0; entry < 4; ++entry ) {
      const Real finite_difference =
          ( plus.constraint[entry] - minus.constraint[entry] ) / ( 2.0 * finite_difference_step );
      if ( !closeDerivative( constraint_derivative[entry], finite_difference, "constraint", entry ) ) {
        return false;
      }
    }
  }

  contact.updateGeometry( surfaces );

  const auto matrix = contact.assembleCoordinateJacobian( state );
  const Index expected_rows = 24 + ( MethodTraits<MethodType>::capabilities.produces_constraint_residual ? 4 : 0 );
  if ( matrix.rows != expected_rows || matrix.columns != 24 ) {
    return false;
  }
  for ( Index row = 0; row < matrix.rows; ++row ) {
    Real assembled_action{};
    for ( Index column = 0; column < matrix.columns; ++column ) {
      const Real direction_value = column < 12 ? mortar_direction[column] : nonmortar_direction[column - 12];
      assembled_action += matrix( row, column ) * direction_value;
    }
    const Real matrix_free = row < 12   ? mortar_derivative[row]
                             : row < 24 ? nonmortar_derivative[row - 12]
                                        : constraint_derivative[row - 24];
    if ( !std::isfinite( assembled_action ) || std::abs( assembled_action - matrix_free ) > 1.0e-11 ) {
      std::cerr << "assembled mismatch row=" << row << " action=" << assembled_action << " matrix_free=" << matrix_free
                << '\n';
      return false;
    }
  }
  return true;
}

bool allSupportedFamiliesLinearize()
{
  constexpr std::array<Real, 1> material{ 2.0 };
  constexpr std::array<Real, 1> thickness{ 0.5 };
  constexpr std::array<Real, 4> multiplier{ -2.0, -1.5, -1.0, -0.5 };
  constexpr std::array<Real, 4> pressure{ -3.0, -2.0, -1.0, -0.5 };
  constexpr std::array<Real, 12> velocity{ 0.2, 0.0, -0.1, 0.1, 0.0, -0.1, 0.2, 0.0, -0.1, 0.1, 0.0, -0.1 };
  constexpr std::array<Real, 12> zero_velocity{};
  constexpr std::array<Real, 12> mortar_reference{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 12> nonmortar_reference{ 0.0, 0.0, -0.1, 0.0, 1.0, -0.1, 1.0, 1.0, -0.1, 1.0, 0.0, -0.1 };

  Contact<PointwisePenalty>::Options pointwise;
  Contact<PointwiseMaterialRateViscous>::Options material_rate;
  material_rate.method.enforcement.stiffness.scale = 2.0;
  material_rate.method.enforcement.rate.ratio = 0.1;
  material_rate.method.response.damping = 0.25;
  const ContactStateView material_rate_state{
      .mortar_velocity = { { velocity.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .nonmortar_velocity = { { zero_velocity.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .mortar_element_thickness = { thickness.data(), 1 },
      .nonmortar_element_thickness = { thickness.data(), 1 },
      .mortar_material_modulus = { material.data(), 1 },
      .nonmortar_material_modulus = { material.data(), 1 },
  };
  Contact<PointwiseTiedNormal>::Options tied;
  Contact<PointwiseTiedFull>::Options tied_full;
  const ContactStateView tied_state{
      .mortar_reference_coordinates = { { mortar_reference.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .nonmortar_reference_coordinates = { { nonmortar_reference.data(), 12 }, 4, 3, FieldLayout::Interleaved },
  };
  Contact<ProjectedMultiplier>::Options projected_multiplier;
  Contact<ConformingMultiplier>::Options conforming_multiplier;
  Contact<VariationalMultiplier>::Options variational_multiplier;
  const ContactStateView multiplier_state{ .multiplier = { multiplier.data(), 4 } };
  Contact<DiagnosticWeights>::Options diagnostic;
  Contact<VariationalNodalPenalty>::Options nodal_penalty;
  Contact<VariationalQuadraturePenalty>::Options quadrature_penalty;
  Contact<VariationalExternalPressure>::Options external_pressure;
  const ContactStateView pressure_state{ .external_pressure = { pressure.data(), 4 } };

  return assembledMatchesAction<PointwisePenalty>( pointwise, {} ) &&
         assembledMatchesAction<PointwiseMaterialRateViscous>( material_rate, material_rate_state ) &&
         assembledMatchesAction<PointwiseTiedNormal>( tied, tied_state ) &&
         assembledMatchesAction<PointwiseTiedFull>( tied_full, tied_state ) &&
         assembledMatchesAction<ProjectedMultiplier>( projected_multiplier, multiplier_state ) &&
         assembledMatchesAction<ConformingMultiplier>( conforming_multiplier, multiplier_state ) &&
         assembledMatchesAction<DiagnosticWeights>( diagnostic, {} ) &&
         assembledMatchesAction<VariationalNodalPenalty>( nodal_penalty, {} ) &&
         assembledMatchesAction<VariationalQuadraturePenalty>( quadrature_penalty, {} ) &&
         assembledMatchesAction<VariationalMultiplier>( variational_multiplier, multiplier_state ) &&
         assembledMatchesAction<VariationalExternalPressure>( external_pressure, pressure_state );
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

bool segmentDerivativeMatchesFiniteDifference()
{
  constexpr std::array<Real, 4> mortar_coordinates{ 1.1, 0.03, -0.1, -0.02 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.15, -0.15, 0.85, -0.11 };
  constexpr std::array<Real, 4> mortar_direction{ 0.03, -0.02, -0.01, 0.04 };
  constexpr std::array<Real, 4> nonmortar_direction{ -0.02, 0.05, 0.04, 0.01 };
  const SurfacePairView surfaces{ makeSegment( mortar_coordinates ), makeSegment( nonmortar_coordinates ) };
  Contact<PointwisePenalty>::Options options;
  options.search.expansion = 0.3;
  Contact<PointwisePenalty> contact( surfaces, options );
  contact.updateInteractions();
  const ContactDirectionView direction{
      .mortar = { { mortar_direction.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_direction.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  std::array<Real, 4> exact_mortar{};
  std::array<Real, 4> exact_nonmortar{};
  contact.applyCoordinateDerivative(
      {}, direction,
      { .mortar = { { exact_mortar.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar = { { exact_nonmortar.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );

  constexpr Real step = 1.0e-6;
  std::array<Real, 4> plus_mortar{};
  std::array<Real, 4> plus_nonmortar{};
  std::array<Real, 4> minus_mortar{};
  std::array<Real, 4> minus_nonmortar{};
  for ( std::size_t entry = 0; entry < mortar_coordinates.size(); ++entry ) {
    plus_mortar[entry] = mortar_coordinates[entry] + step * mortar_direction[entry];
    plus_nonmortar[entry] = nonmortar_coordinates[entry] + step * nonmortar_direction[entry];
    minus_mortar[entry] = mortar_coordinates[entry] - step * mortar_direction[entry];
    minus_nonmortar[entry] = nonmortar_coordinates[entry] - step * nonmortar_direction[entry];
  }
  auto evaluate = [&]( const std::array<Real, 4>& mortar, const std::array<Real, 4>& nonmortar ) {
    contact.updateGeometry( { makeSegment( mortar ), makeSegment( nonmortar ) } );
    std::array<std::array<Real, 4>, 2> residual{};
    contact.addResidual( {}, { .mortar = { { residual[0].data(), 4 }, 2, 2, FieldLayout::Interleaved },
                               .nonmortar = { { residual[1].data(), 4 }, 2, 2, FieldLayout::Interleaved } } );
    return residual;
  };
  const auto plus = evaluate( plus_mortar, plus_nonmortar );
  const auto minus = evaluate( minus_mortar, minus_nonmortar );
  for ( Index entry = 0; entry < 4; ++entry ) {
    if ( !closeDerivative( exact_mortar[entry], ( plus[0][entry] - minus[0][entry] ) / ( 2.0 * step ), "segment-mortar",
                           entry ) ||
         !closeDerivative( exact_nonmortar[entry], ( plus[1][entry] - minus[1][entry] ) / ( 2.0 * step ),
                           "segment-nonmortar", entry ) ) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() { return allSupportedFamiliesLinearize() && segmentDerivativeMatchesFiniteDifference() ? 0 : 1; }
