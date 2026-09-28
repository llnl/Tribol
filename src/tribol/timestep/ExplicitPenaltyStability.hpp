#ifndef TRIBOL_TIMESTEP_EXPLICITPENALTYSTABILITY_HPP_
#define TRIBOL_TIMESTEP_EXPLICITPENALTYSTABILITY_HPP_

#include "tribol/core/ConstraintRowView.hpp"
#include "tribol/core/Version.hpp"
#include "tribol/operators/ContactOperators.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tribol::timestep {

struct DiagonalInverseMassView {
  FieldView<const Real> mortar{};
  FieldView<const Real> nonmortar{};
  MassVersion version{};
};

struct IntegratorStabilityLimits {
  Real maximum_undamped_frequency_step{};
  Real maximum_decay_step{};
  Real safety_factor{ 1.0 };
  IntegratorVersion version{};

  [[nodiscard]] static constexpr IntegratorStabilityLimits centralDifference() { return { 2.0, 2.0, 1.0, {} }; }
};

struct ExplicitStabilityInput {
  ArrayView<const Real> row_stiffness{};
  ArrayView<const Real> row_rate_coefficient{};
  DiagonalInverseMassView inverse_mass{};
  IntegratorStabilityLimits integrator{};
  GeometryVersion geometry_version{};
  InteractionVersion interaction_version{};
  PenaltyVersion penalty_version{};
};

struct ExplicitStabilityWorkspaceView {
  FieldView<Real> mortar_row_sums{};
  FieldView<Real> nonmortar_row_sums{};
};

struct ExplicitStabilityContributionsView {
  FieldView<const Real> mortar_stiffness_row_sums{};
  FieldView<const Real> nonmortar_stiffness_row_sums{};
  FieldView<const Real> mortar_rate_row_sums{};
  FieldView<const Real> nonmortar_rate_row_sums{};
};

struct ExplicitStabilityResult {
  Real stiffness_frequency_squared_bound{};
  Real rate_frequency_bound{};
  Real timestep_vote{ std::numeric_limits<Real>::infinity() };
  Index active_stiffness_rows{};
  Index active_rate_rows{};
  GeometryVersion geometry_version{};
  InteractionVersion interaction_version{};
  MassVersion mass_version{};
  PenaltyVersion penalty_version{};
  IntegratorVersion integrator_version{};
};

struct SpectralWorkspaceView {
  FieldView<Real> iterate_mortar{};
  FieldView<Real> iterate_nonmortar{};
  FieldView<Real> direction_mortar{};
  FieldView<Real> direction_nonmortar{};
  FieldView<Real> image_mortar{};
  FieldView<Real> image_nonmortar{};
  ArrayView<Real> row_values{};
};

namespace explicit_stability_detail {

[[nodiscard]] inline FieldView<const Real> asConst( FieldView<Real> field )
{
  return { { field.values.data(), field.values.size() }, field.entities, field.components, field.layout };
}

template <typename Scalar>
inline void validateField( FieldView<Scalar> field, int dimension, const char* message )
{
  if ( !field.isStructurallyValid() || field.components != dimension ) {
    throw std::invalid_argument( message );
  }
}

inline void validateMassValues( FieldView<const Real> inverse_mass )
{
  for ( Real value : inverse_mass.values ) {
    if ( !std::isfinite( value ) || value < 0.0 ) {
      throw std::invalid_argument( "Inverse mass entries must be finite and nonnegative." );
    }
  }
}

inline void validateDofs( ArrayView<const Index> dofs, Index entities )
{
  for ( Index dof : dofs ) {
    if ( dof < 0 || dof >= entities ) {
      throw std::out_of_range( "A stability row references a DOF outside the inverse-mass field." );
    }
  }
}

inline void validate( ConstraintRowBatchView rows, const ExplicitStabilityInput& input,
                      ExplicitStabilityWorkspaceView workspace )
{
  if ( !rows.isStructurallyValid() ) {
    throw std::invalid_argument( "Explicit stability requires a structurally valid row batch." );
  }
  if ( !input.row_stiffness.isStructurallyValid() || input.row_stiffness.size() != rows.numberOfRows() ||
       !input.row_rate_coefficient.isStructurallyValid() ||
       ( !input.row_rate_coefficient.empty() && input.row_rate_coefficient.size() != rows.numberOfRows() ) ) {
    throw std::invalid_argument( "Explicit stability requires row-sized derivative data." );
  }
  validateField( input.inverse_mass.mortar, rows.dimension,
                 "Explicit stability requires a valid mortar inverse-mass field." );
  validateField( input.inverse_mass.nonmortar, rows.dimension,
                 "Explicit stability requires a valid nonmortar inverse-mass field." );
  validateField( workspace.mortar_row_sums, rows.dimension, "Explicit stability requires a valid mortar workspace." );
  validateField( workspace.nonmortar_row_sums, rows.dimension,
                 "Explicit stability requires a valid nonmortar workspace." );
  if ( workspace.mortar_row_sums.entities != input.inverse_mass.mortar.entities ||
       workspace.nonmortar_row_sums.entities != input.inverse_mass.nonmortar.entities ) {
    throw std::invalid_argument( "Explicit stability workspace and inverse-mass fields must have matching shapes." );
  }
  validateDofs( rows.mortar_dofs, input.inverse_mass.mortar.entities );
  validateDofs( rows.nonmortar_dofs, input.inverse_mass.nonmortar.entities );
  validateMassValues( input.inverse_mass.mortar );
  validateMassValues( input.inverse_mass.nonmortar );
  if ( !std::isfinite( input.integrator.maximum_undamped_frequency_step ) ||
       input.integrator.maximum_undamped_frequency_step <= 0.0 ||
       !std::isfinite( input.integrator.maximum_decay_step ) || input.integrator.maximum_decay_step <= 0.0 ||
       !std::isfinite( input.integrator.safety_factor ) || input.integrator.safety_factor <= 0.0 ||
       input.integrator.safety_factor > 1.0 ) {
    throw std::invalid_argument(
        "Explicit stability requires positive finite integrator limits and a safety factor "
        "in (0, 1]." );
  }
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    if ( !std::isfinite( input.row_stiffness[row] ) || input.row_stiffness[row] < 0.0 ||
         ( !input.row_rate_coefficient.empty() &&
           ( !std::isfinite( input.row_rate_coefficient[row] ) || input.row_rate_coefficient[row] < 0.0 ) ) ) {
      throw std::invalid_argument( "Explicit stability row derivatives must be finite and nonnegative." );
    }
  }
}

inline void clear( FieldView<Real> field )
{
  for ( Real& value : field.values ) {
    value = 0.0;
  }
}

[[nodiscard]] inline Real sideMassScaledOneNorm( ConstraintRowBatchView rows, Index row, ArrayView<const Index> offsets,
                                                 ArrayView<const Index> dofs, ArrayView<const Real> basis_values,
                                                 FieldView<const Real> inverse_mass )
{
  Real norm{};
  for ( Index entry = offsets[row]; entry < offsets[row + 1]; ++entry ) {
    for ( int component = 0; component < rows.dimension; ++component ) {
      norm += std::abs( basis_values[entry] * rows.normals( row, component ) ) *
              std::sqrt( inverse_mass( dofs[entry], component ) );
    }
  }
  return norm;
}

inline void accumulateSideRowSum( ConstraintRowBatchView rows, Index row, ArrayView<const Index> offsets,
                                  ArrayView<const Index> dofs, ArrayView<const Real> basis_values,
                                  FieldView<const Real> inverse_mass, Real row_scale, Real row_one_norm,
                                  FieldView<Real> row_sums )
{
  for ( Index entry = offsets[row]; entry < offsets[row + 1]; ++entry ) {
    for ( int component = 0; component < rows.dimension; ++component ) {
      const Real coefficient = std::abs( basis_values[entry] * rows.normals( row, component ) ) *
                               std::sqrt( inverse_mass( dofs[entry], component ) );
      row_sums( dofs[entry], component ) += row_scale * coefficient * row_one_norm;
    }
  }
}

[[nodiscard]] inline Real rowSumBound( ConstraintRowBatchView rows, ArrayView<const Real> row_scale,
                                       DiagonalInverseMassView inverse_mass, ExplicitStabilityWorkspaceView workspace )
{
  clear( workspace.mortar_row_sums );
  clear( workspace.nonmortar_row_sums );
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    if ( row_scale[row] == 0.0 ) {
      continue;
    }
    const Real row_one_norm = sideMassScaledOneNorm( rows, row, rows.mortar_dof_offsets, rows.mortar_dofs,
                                                     rows.mortar_basis_values, inverse_mass.mortar ) +
                              sideMassScaledOneNorm( rows, row, rows.nonmortar_dof_offsets, rows.nonmortar_dofs,
                                                     rows.nonmortar_basis_values, inverse_mass.nonmortar );
    accumulateSideRowSum( rows, row, rows.mortar_dof_offsets, rows.mortar_dofs, rows.mortar_basis_values,
                          inverse_mass.mortar, row_scale[row], row_one_norm, workspace.mortar_row_sums );
    accumulateSideRowSum( rows, row, rows.nonmortar_dof_offsets, rows.nonmortar_dofs, rows.nonmortar_basis_values,
                          inverse_mass.nonmortar, row_scale[row], row_one_norm, workspace.nonmortar_row_sums );
  }
  Real bound{};
  for ( Real row_sum : workspace.mortar_row_sums.values ) {
    bound = std::max( bound, row_sum );
  }
  for ( Real row_sum : workspace.nonmortar_row_sums.values ) {
    bound = std::max( bound, row_sum );
  }
  return bound;
}

[[nodiscard]] inline Real stabilityVote( Real stiffness_bound, Real rate_bound,
                                         const IntegratorStabilityLimits& integrator )
{
  if ( stiffness_bound == 0.0 && rate_bound == 0.0 ) {
    return std::numeric_limits<Real>::infinity();
  }
  const Real quadratic =
      stiffness_bound / ( integrator.maximum_undamped_frequency_step * integrator.maximum_undamped_frequency_step );
  const Real linear = rate_bound / integrator.maximum_decay_step;
  const Real unscaled_vote =
      quadratic == 0.0 ? 1.0 / linear : 2.0 / ( linear + std::sqrt( linear * linear + 4.0 * quadratic ) );
  return integrator.safety_factor * unscaled_vote;
}

inline void requireSameShape( FieldView<Real> field, FieldView<const Real> reference, const char* message )
{
  validateField( field, reference.components, message );
  if ( field.entities != reference.entities ) {
    throw std::invalid_argument( message );
  }
}

inline void scaleBySqrtInverseMass( FieldView<const Real> inverse_mass, FieldView<const Real> input,
                                    FieldView<Real> output )
{
  for ( Index entity = 0; entity < input.entities; ++entity ) {
    for ( int component = 0; component < input.components; ++component ) {
      output( entity, component ) = std::sqrt( inverse_mass( entity, component ) ) * input( entity, component );
    }
  }
}

[[nodiscard]] inline Real dot( FieldView<const Real> left_mortar, FieldView<const Real> left_nonmortar,
                               FieldView<const Real> right_mortar, FieldView<const Real> right_nonmortar )
{
  Real result{};
  for ( Index value = 0; value < left_mortar.values.size(); ++value ) {
    result += left_mortar.values[value] * right_mortar.values[value];
  }
  for ( Index value = 0; value < left_nonmortar.values.size(); ++value ) {
    result += left_nonmortar.values[value] * right_nonmortar.values[value];
  }
  return result;
}

inline void initializeIterate( DiagonalInverseMassView inverse_mass, SpectralWorkspaceView workspace )
{
  for ( Index value = 0; value < workspace.iterate_mortar.values.size(); ++value ) {
    workspace.iterate_mortar.values[value] =
        inverse_mass.mortar.values[value] > 0.0 ? 1.0 + static_cast<Real>( ( value * 17 + 3 ) % 23 ) / 23.0 : 0.0;
  }
  for ( Index value = 0; value < workspace.iterate_nonmortar.values.size(); ++value ) {
    workspace.iterate_nonmortar.values[value] = inverse_mass.nonmortar.values[value] > 0.0
                                                    ? -( 1.0 + static_cast<Real>( ( value * 19 + 5 ) % 29 ) / 29.0 )
                                                    : 0.0;
  }
}

inline void normalize( FieldView<Real> mortar, FieldView<Real> nonmortar )
{
  const Real norm =
      std::sqrt( dot( asConst( mortar ), asConst( nonmortar ), asConst( mortar ), asConst( nonmortar ) ) );
  if ( norm == 0.0 ) {
    return;
  }
  for ( Real& value : mortar.values ) {
    value /= norm;
  }
  for ( Real& value : nonmortar.values ) {
    value /= norm;
  }
}

}  // namespace explicit_stability_detail

[[nodiscard]] inline ExplicitStabilityResult explicitPenaltyStabilityVote( ConstraintRowBatchView rows,
                                                                           const ExplicitStabilityInput& input,
                                                                           ExplicitStabilityWorkspaceView workspace )
{
  explicit_stability_detail::validate( rows, input, workspace );
  ExplicitStabilityResult result{
      .geometry_version = input.geometry_version,
      .interaction_version = input.interaction_version,
      .mass_version = input.inverse_mass.version,
      .penalty_version = input.penalty_version,
      .integrator_version = input.integrator.version,
  };
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    result.active_stiffness_rows += input.row_stiffness[row] > 0.0 ? 1 : 0;
    result.active_rate_rows += !input.row_rate_coefficient.empty() && input.row_rate_coefficient[row] > 0.0 ? 1 : 0;
  }
  result.stiffness_frequency_squared_bound =
      explicit_stability_detail::rowSumBound( rows, input.row_stiffness, input.inverse_mass, workspace );
  if ( !input.row_rate_coefficient.empty() ) {
    result.rate_frequency_bound =
        explicit_stability_detail::rowSumBound( rows, input.row_rate_coefficient, input.inverse_mass, workspace );
  }
  result.timestep_vote = explicit_stability_detail::stabilityVote( result.stiffness_frequency_squared_bound,
                                                                   result.rate_frequency_bound, input.integrator );
  return result;
}

[[nodiscard]] inline Real estimateMaximumMassScaledEigenvalue( ConstraintRowBatchView rows,
                                                               ArrayView<const Real> row_scale,
                                                               DiagonalInverseMassView inverse_mass,
                                                               SpectralWorkspaceView workspace, int iterations = 64 )
{
  if ( iterations <= 0 ) {
    throw std::invalid_argument( "The spectral estimator requires a positive iteration count." );
  }
  const ExplicitStabilityInput input{ .row_stiffness = row_scale,
                                      .inverse_mass = inverse_mass,
                                      .integrator = IntegratorStabilityLimits::centralDifference() };
  explicit_stability_detail::validate(
      rows, input, { .mortar_row_sums = workspace.image_mortar, .nonmortar_row_sums = workspace.image_nonmortar } );
  explicit_stability_detail::requireSameShape( workspace.iterate_mortar, inverse_mass.mortar,
                                               "Invalid mortar spectral iterate workspace." );
  explicit_stability_detail::requireSameShape( workspace.iterate_nonmortar, inverse_mass.nonmortar,
                                               "Invalid nonmortar spectral iterate workspace." );
  explicit_stability_detail::requireSameShape( workspace.direction_mortar, inverse_mass.mortar,
                                               "Invalid mortar spectral direction workspace." );
  explicit_stability_detail::requireSameShape( workspace.direction_nonmortar, inverse_mass.nonmortar,
                                               "Invalid nonmortar spectral direction workspace." );
  if ( !workspace.row_values.isStructurallyValid() || workspace.row_values.size() != rows.numberOfRows() ) {
    throw std::invalid_argument( "The spectral estimator requires one row workspace value per constraint row." );
  }

  explicit_stability_detail::initializeIterate( inverse_mass, workspace );
  explicit_stability_detail::normalize( workspace.iterate_mortar, workspace.iterate_nonmortar );
  Real eigenvalue{};
  for ( int iteration = 0; iteration < iterations; ++iteration ) {
    explicit_stability_detail::scaleBySqrtInverseMass( inverse_mass.mortar,
                                                       explicit_stability_detail::asConst( workspace.iterate_mortar ),
                                                       workspace.direction_mortar );
    explicit_stability_detail::scaleBySqrtInverseMass(
        inverse_mass.nonmortar, explicit_stability_detail::asConst( workspace.iterate_nonmortar ),
        workspace.direction_nonmortar );
    operators::applyContactTangent(
        rows, row_scale,
        { .mortar = explicit_stability_detail::asConst( workspace.direction_mortar ),
          .nonmortar = explicit_stability_detail::asConst( workspace.direction_nonmortar ) },
        workspace.row_values, { .mortar = workspace.image_mortar, .nonmortar = workspace.image_nonmortar } );
    explicit_stability_detail::scaleBySqrtInverseMass(
        inverse_mass.mortar, explicit_stability_detail::asConst( workspace.image_mortar ), workspace.direction_mortar );
    explicit_stability_detail::scaleBySqrtInverseMass( inverse_mass.nonmortar,
                                                       explicit_stability_detail::asConst( workspace.image_nonmortar ),
                                                       workspace.direction_nonmortar );
    eigenvalue = explicit_stability_detail::dot( explicit_stability_detail::asConst( workspace.iterate_mortar ),
                                                 explicit_stability_detail::asConst( workspace.iterate_nonmortar ),
                                                 explicit_stability_detail::asConst( workspace.direction_mortar ),
                                                 explicit_stability_detail::asConst( workspace.direction_nonmortar ) );
    for ( Index value = 0; value < workspace.iterate_mortar.values.size(); ++value ) {
      workspace.iterate_mortar.values[value] = workspace.direction_mortar.values[value];
    }
    for ( Index value = 0; value < workspace.iterate_nonmortar.values.size(); ++value ) {
      workspace.iterate_nonmortar.values[value] = workspace.direction_nonmortar.values[value];
    }
    explicit_stability_detail::normalize( workspace.iterate_mortar, workspace.iterate_nonmortar );
  }
  return eigenvalue;
}

static_assert( std::is_trivially_copyable_v<DiagonalInverseMassView> );
static_assert( std::is_trivially_copyable_v<IntegratorStabilityLimits> );
static_assert( std::is_trivially_copyable_v<ExplicitStabilityInput> );
static_assert( std::is_trivially_copyable_v<ExplicitStabilityContributionsView> );
static_assert( std::is_trivially_copyable_v<ExplicitStabilityWorkspaceView> );
static_assert( std::is_trivially_copyable_v<ExplicitStabilityResult> );
static_assert( std::is_trivially_copyable_v<SpectralWorkspaceView> );

}  // namespace tribol::timestep

#endif
