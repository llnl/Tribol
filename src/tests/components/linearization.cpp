#include "tests/spec/SupportedMethods.hpp"

#include "tribol/Tribol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

// Requirements: DIFF-001, DIFF-002, DIFF-003, DIFF-005

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
  Real energy{};
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
  result.energy = contact.addResidual( state, residual ).energy;
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

bool csrMatchesDense( CsrMatrixView sparse, DenseMatrixView dense )
{
  if ( sparse.rows != dense.rows || sparse.columns != dense.columns || sparse.row_offsets.size() != sparse.rows + 1 ||
       sparse.column_indices.size() != sparse.values.size() ||
       sparse.row_offsets[sparse.rows] != sparse.values.size() ) {
    return false;
  }
  for ( Index row = 0; row < dense.rows; ++row ) {
    Index sparse_entry = sparse.row_offsets[row];
    for ( Index column = 0; column < dense.columns; ++column ) {
      Real sparse_value{};
      if ( sparse_entry < sparse.row_offsets[row + 1] && sparse.column_indices[sparse_entry] == column ) {
        sparse_value = sparse.values[sparse_entry++];
      }
      if ( std::abs( sparse_value - dense( row, column ) ) > 1.0e-14 ) {
        return false;
      }
    }
    if ( sparse_entry != sparse.row_offsets[row + 1] ) {
      return false;
    }
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
  quadrature_penalty.method.constraint.activation.residual_gap = 0.15;
  Contact<VariationalExternalPressure>::Options external_pressure;
  external_pressure.method.constraint.activation.residual_gap = 0.15;
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

template <SupportedMethod MethodType>
bool multiplierDerivativeMatchesFiniteDifference( typename Contact<MethodType>::Options options )
{
  static_assert( MethodTraits<MethodType>::capabilities.needs_multiplier );
  options.search.expansion = 0.2;
  if constexpr ( std::same_as<typename MethodType::geometry_policy, geometry::ConformingOverlap> ) {
    options.method.geometry.alignment_tolerance = 1.0e-3;
  }
  const auto surfaces = derivativeSurfaces<MethodType>();
  Contact<MethodType> contact( surfaces, options );
  contact.updateInteractions();
  constexpr std::array<Real, 4> multiplier{ -2.0, -1.5, -1.0, -0.5 };
  constexpr std::array<Real, 4> direction{ 0.3, -0.2, 0.1, 0.4 };
  const ContactStateView state{ .multiplier = { multiplier.data(), 4 } };
  std::array<Real, 12> mortar_derivative{};
  std::array<Real, 12> nonmortar_derivative{};
  std::array<Real, 4> constraint_derivative{};
  const ContactResidualView derivative{
      .mortar = { { mortar_derivative.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_derivative.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .constraint = { constraint_derivative.data(), 4 },
  };
  contact.applyMultiplierDerivative( state, { direction.data(), 4 }, derivative );

  constexpr Real step = 1.0e-6;
  std::array<Real, 4> plus_multiplier{};
  std::array<Real, 4> minus_multiplier{};
  for ( Index entry = 0; entry < 4; ++entry ) {
    plus_multiplier[entry] = multiplier[entry] + step * direction[entry];
    minus_multiplier[entry] = multiplier[entry] - step * direction[entry];
  }
  const auto plus =
      evaluateResidual( contact, ContactStateView{ .multiplier = { plus_multiplier.data(), 4 } }, surfaces );
  const auto minus =
      evaluateResidual( contact, ContactStateView{ .multiplier = { minus_multiplier.data(), 4 } }, surfaces );
  for ( Index entry = 0; entry < 12; ++entry ) {
    if ( !closeDerivative( mortar_derivative[entry], ( plus.mortar[entry] - minus.mortar[entry] ) / ( 2.0 * step ),
                           "multiplier-mortar", entry ) ||
         !closeDerivative( nonmortar_derivative[entry],
                           ( plus.nonmortar[entry] - minus.nonmortar[entry] ) / ( 2.0 * step ), "multiplier-nonmortar",
                           entry ) ) {
      return false;
    }
  }
  for ( Index entry = 0; entry < 4; ++entry ) {
    if ( !closeDerivative( constraint_derivative[entry],
                           ( plus.constraint[entry] - minus.constraint[entry] ) / ( 2.0 * step ),
                           "multiplier-constraint", entry ) ) {
      return false;
    }
  }

  contact.updateGeometry( surfaces );
  const auto matrix = contact.assembleSystemJacobian( state );
  if ( matrix.rows != 28 || matrix.columns != 28 ) {
    return false;
  }
  for ( Index row = 0; row < matrix.rows; ++row ) {
    Real action{};
    for ( Index column = 0; column < 4; ++column ) {
      action += matrix( row, 24 + column ) * direction[column];
    }
    const Real expected = row < 12   ? mortar_derivative[row]
                          : row < 24 ? nonmortar_derivative[row - 12]
                                     : constraint_derivative[row - 24];
    if ( std::abs( action - expected ) > 1.0e-11 ) {
      return false;
    }
  }
  const auto sparse = contact.assembleSystemJacobianCsr( state );
  const auto refreshed_matrix = contact.assembleSystemJacobian( state );
  if ( !csrMatchesDense( sparse, refreshed_matrix ) ) {
    return false;
  }
  return true;
}

bool externalPressureDerivativeMatchesFiniteDifference()
{
  const auto surfaces = derivativeSurfaces<VariationalExternalPressure>();
  Contact<VariationalExternalPressure>::Options options;
  options.search.expansion = 0.2;
  Contact<VariationalExternalPressure> contact( surfaces, options );
  contact.updateInteractions();
  constexpr std::array<Real, 4> pressure{ -2.0, -1.5, -1.0, -0.5 };
  constexpr std::array<Real, 4> direction{ 0.3, -0.2, 0.1, 0.4 };
  const ContactStateView state{ .external_pressure = { pressure.data(), 4 } };
  std::array<Real, 12> mortar_derivative{};
  std::array<Real, 12> nonmortar_derivative{};
  const ContactResidualView derivative{
      .mortar = { { mortar_derivative.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_derivative.data(), 12 }, 4, 3, FieldLayout::Interleaved },
  };
  contact.applyExternalPressureDerivative( state, { direction.data(), 4 }, derivative );

  constexpr Real step = 1.0e-6;
  std::array<Real, 4> plus_pressure{};
  std::array<Real, 4> minus_pressure{};
  for ( Index entry = 0; entry < 4; ++entry ) {
    plus_pressure[entry] = pressure[entry] + step * direction[entry];
    minus_pressure[entry] = pressure[entry] - step * direction[entry];
  }
  const auto plus =
      evaluateResidual( contact, ContactStateView{ .external_pressure = { plus_pressure.data(), 4 } }, surfaces );
  const auto minus =
      evaluateResidual( contact, ContactStateView{ .external_pressure = { minus_pressure.data(), 4 } }, surfaces );
  for ( Index entry = 0; entry < 12; ++entry ) {
    if ( !closeDerivative( mortar_derivative[entry], ( plus.mortar[entry] - minus.mortar[entry] ) / ( 2.0 * step ),
                           "pressure-mortar", entry ) ||
         !closeDerivative( nonmortar_derivative[entry],
                           ( plus.nonmortar[entry] - minus.nonmortar[entry] ) / ( 2.0 * step ), "pressure-nonmortar",
                           entry ) ) {
      return false;
    }
  }
  return true;
}

struct ExternalLawData {
  std::array<Real, 4> potential{};
  std::array<Real, 4> pressure{};
  std::array<Real, 4> tangent{};

  [[nodiscard]] ContactStateView view() const
  {
    return {
        .external_potential_density = { potential.data(), 4 },
        .external_pressure = { pressure.data(), 4 },
        .external_pressure_tangent = { tangent.data(), 4 },
    };
  }
};

ExternalLawData quadraticExternalLaw( Contact<VariationalExternalPressure>& contact )
{
  constexpr Real stiffness = 7.0;
  const auto nodal = contact.evaluateNodalKinematics();
  ExternalLawData data;
  for ( Index node = 0; node < nodal.gap.size(); ++node ) {
    const Real active_gap = std::min( nodal.gap[node], 0.0 );
    data.potential[node] = 0.5 * stiffness * active_gap * active_gap;
    data.pressure[node] = stiffness * active_gap;
    data.tangent[node] = nodal.gap[node] < 0.0 ? stiffness : 0.0;
  }
  return data;
}

bool externalPressureCoordinateDerivativeMatchesFiniteDifference()
{
  const auto surfaces = derivativeSurfaces<VariationalExternalPressure>();
  Contact<VariationalExternalPressure>::Options options;
  options.search.expansion = 0.2;
  Contact<VariationalExternalPressure> contact( surfaces, options );
  contact.updateInteractions();
  const auto state = quadraticExternalLaw( contact );
  std::array<Real, 12> mortar_direction{ 0.02, -0.01, 0.03, -0.01, 0.02, -0.02, 0.01, 0.03, 0.04, -0.02, -0.01, 0.01 };
  std::array<Real, 12> nonmortar_direction{ -0.01, 0.02, 0.05, 0.03, -0.01, 0.02,
                                            -0.02, 0.01, 0.04, 0.01, -0.03, 0.03 };
  const ContactDirectionView direction{
      .mortar = { { mortar_direction.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_direction.data(), 12 }, 4, 3, FieldLayout::Interleaved },
  };
  std::array<Real, 12> mortar_derivative{};
  std::array<Real, 12> nonmortar_derivative{};
  const auto base = evaluateResidual( contact, state.view(), surfaces );
  contact.applyCoordinateDerivative(
      state.view(), direction,
      { .mortar = { { mortar_derivative.data(), 12 }, 4, 3, FieldLayout::Interleaved },
        .nonmortar = { { nonmortar_derivative.data(), 12 }, 4, 3, FieldLayout::Interleaved } } );

  constexpr Real step = 1.0e-6;
  std::array<Real, 12> plus_mortar{};
  std::array<Real, 12> plus_nonmortar{};
  std::array<Real, 12> minus_mortar{};
  std::array<Real, 12> minus_nonmortar{};
  perturbCoordinates( surfaces.mortar, direction.mortar, step, plus_mortar );
  perturbCoordinates( surfaces.nonmortar, direction.nonmortar, step, plus_nonmortar );
  perturbCoordinates( surfaces.mortar, direction.mortar, -step, minus_mortar );
  perturbCoordinates( surfaces.nonmortar, direction.nonmortar, -step, minus_nonmortar );
  const SurfacePairView plus_surfaces{ makeSurface( plus_mortar ), makeSurface( plus_nonmortar ) };
  const SurfacePairView minus_surfaces{ makeSurface( minus_mortar ), makeSurface( minus_nonmortar ) };
  contact.updateGeometry( plus_surfaces );
  const auto plus_state = quadraticExternalLaw( contact );
  const auto plus = evaluateResidual( contact, plus_state.view(), plus_surfaces );
  contact.updateGeometry( minus_surfaces );
  const auto minus_state = quadraticExternalLaw( contact );
  const auto minus = evaluateResidual( contact, minus_state.view(), minus_surfaces );
  Real force_action{};
  for ( Index entry = 0; entry < 12; ++entry ) {
    force_action += base.mortar[entry] * mortar_direction[entry] + base.nonmortar[entry] * nonmortar_direction[entry];
  }
  if ( !closeDerivative( force_action, ( plus.energy - minus.energy ) / ( 2.0 * step ), "external-law-energy", 0 ) ) {
    return false;
  }
  for ( Index entry = 0; entry < 12; ++entry ) {
    if ( !closeDerivative( mortar_derivative[entry], ( plus.mortar[entry] - minus.mortar[entry] ) / ( 2.0 * step ),
                           "external-law-mortar", entry ) ||
         !closeDerivative( nonmortar_derivative[entry],
                           ( plus.nonmortar[entry] - minus.nonmortar[entry] ) / ( 2.0 * step ),
                           "external-law-nonmortar", entry ) ) {
      return false;
    }
  }
  return true;
}

bool stateDerivativesLinearize()
{
  return multiplierDerivativeMatchesFiniteDifference<ProjectedMultiplier>( {} ) &&
         multiplierDerivativeMatchesFiniteDifference<ConformingMultiplier>( {} ) &&
         multiplierDerivativeMatchesFiniteDifference<VariationalMultiplier>( {} ) &&
         externalPressureDerivativeMatchesFiniteDifference();
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

bool inactiveMultiplierRowsAreIdentities()
{
  constexpr std::array<Real, 8> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0 };
  constexpr std::array<Real, 8> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1, 2.0, -0.1, 3.0, -0.1 };
  constexpr std::array<Index, 3> offsets{ 0, 2, 4 };
  constexpr std::array<Index, 4> connectivity{ 0, 1, 2, 3 };
  constexpr std::array<ElementTopology, 2> topologies{ ElementTopology::Segment, ElementTopology::Segment };
  constexpr std::array<int, 2> attributes{ 1, 1 };
  const auto surface = [&]( const std::array<Real, 8>& coordinates ) {
    return SurfaceMeshView{
        .dimension = 2,
        .coordinates = { { coordinates.data(), 8 }, 4, 2, FieldLayout::Interleaved },
        .element_offsets = { offsets.data(), 3 },
        .connectivity = { connectivity.data(), 4 },
        .topologies = { topologies.data(), 2 },
        .attributes = { attributes.data(), 2 },
    };
  };
  Contact<ProjectedMultiplier> contact( { surface( mortar_coordinates ), surface( nonmortar_coordinates ) } );
  constexpr std::array<ElementPair, 1> interactions{ ElementPair{ 0, 0 } };
  contact.setInteractions( { interactions.data(), 1 } );
  constexpr std::array<Real, 4> multiplier{ -1.0, -1.0, 0.0, 0.0 };
  const auto matrix = contact.assembleSystemJacobian( { .multiplier = { multiplier.data(), 4 } } );
  constexpr Index coordinate_values = 16;
  return std::abs( matrix( coordinate_values + 0, coordinate_values + 0 ) ) < 1.0e-14 &&
         std::abs( matrix( coordinate_values + 1, coordinate_values + 1 ) ) < 1.0e-14 &&
         std::abs( matrix( coordinate_values + 2, coordinate_values + 2 ) - 1.0 ) < 1.0e-14 &&
         std::abs( matrix( coordinate_values + 3, coordinate_values + 3 ) - 1.0 ) < 1.0e-14;
}

}  // namespace

int main()
{
  return allSupportedFamiliesLinearize() && stateDerivativesLinearize() && segmentDerivativeMatchesFiniteDifference() &&
                 externalPressureCoordinateDerivativeMatchesFiniteDifference() && inactiveMultiplierRowsAreIdentities()
             ? 0
             : 1;
}
