#ifndef TRIBOL_EVALUATION_EXPLICITROWEVALUATOR_HPP_
#define TRIBOL_EVALUATION_EXPLICITROWEVALUATOR_HPP_

#include "tribol/core/ConstraintRowView.hpp"
#include "tribol/enforcement/PenaltyLaw.hpp"
#include "tribol/evaluation/State.hpp"

#include <stdexcept>
#include <type_traits>

namespace tribol {

struct ExplicitRowSummary {
  Real energy{};
  Real contact_power{};
  Index active_rows{};
  Index active_stiffness_rows{};
  Index active_rate_rows{};
  Index rows{};
};

struct ExplicitRowEvaluation {
  enforcement::NormalPenaltyResponse response{};
  Real integrated_force{};
  Real integrated_stiffness{};
  Real integrated_rate_coefficient{};
  Real energy{};
  Real contact_power{};
};

namespace explicit_row_detail {

[[nodiscard]] TRIBOL_HOST_DEVICE inline ExplicitRowEvaluation evaluateRow( ConstraintRowBatchView rows,
                                                                           PenaltyRowInputView penalty, Index row )
{
  const Real rate_coefficient = penalty.rate_coefficients.empty() ? 0.0 : penalty.rate_coefficients[row];
  const auto response = enforcement::evaluateNormalPenalty( enforcement::NormalPenaltyInput{
      .gap = rows.gaps[row],
      .normal_rate = rows.hasNormalRates() ? rows.normal_rates[row] : 0.0,
      .pressure_stiffness = penalty.pressure_stiffness[row],
      .rate_coefficient = rate_coefficient,
      .activation = penalty.activation,
  } );
  return {
      .response = response,
      .integrated_force = rows.integration_weights[row] * response.pressure,
      .integrated_stiffness = rows.integration_weights[row] * response.pressure_gap_derivative,
      .integrated_rate_coefficient = rows.integration_weights[row] * response.pressure_rate_derivative,
      .energy = rows.integration_weights[row] * response.potential_density,
      .contact_power =
          rows.integration_weights[row] * response.pressure * ( rows.hasNormalRates() ? rows.normal_rates[row] : 0.0 ),
  };
}

template <typename T>
inline void requireOptionalRowArray( ArrayView<T> values, Index rows, const char* message )
{
  if ( !values.isStructurallyValid() || ( !values.empty() && values.size() != rows ) ) {
    throw std::invalid_argument( message );
  }
}

inline void requireResidual( FieldView<Real> residual, int dimension, const char* message )
{
  if ( !residual.isStructurallyValid() || residual.components != dimension ) {
    throw std::invalid_argument( message );
  }
}

inline void validateExplicitRows( ConstraintRowBatchView rows, PenaltyRowInputView penalty,
                                  ContactResidualView residual, PenaltyRowOutputView output )
{
  if ( !rows.isStructurallyValid() || !penalty.pressure_stiffness.isStructurallyValid() ||
       penalty.pressure_stiffness.size() != rows.numberOfRows() || !penalty.rate_coefficients.isStructurallyValid() ||
       ( !penalty.rate_coefficients.empty() && penalty.rate_coefficients.size() != rows.numberOfRows() ) ) {
    throw std::invalid_argument( "Explicit row evaluation requires valid row-sized penalty data." );
  }
  if ( !penalty.rate_coefficients.empty() && !rows.hasNormalRates() ) {
    throw std::invalid_argument( "Penalty rate coefficients require row normal rates." );
  }
  requireResidual( residual.mortar, rows.dimension, "Explicit rows require a valid mortar residual." );
  requireResidual( residual.nonmortar, rows.dimension, "Explicit rows require a valid nonmortar residual." );
  requireOptionalRowArray( output.effective_gap, rows.numberOfRows(), "Invalid effective-gap output." );
  requireOptionalRowArray( output.pressure, rows.numberOfRows(), "Invalid pressure output." );
  requireOptionalRowArray( output.row_force, rows.numberOfRows(), "Invalid row-force output." );
  requireOptionalRowArray( output.row_stiffness, rows.numberOfRows(), "Invalid row-stiffness output." );
  requireOptionalRowArray( output.row_rate_coefficient, rows.numberOfRows(), "Invalid row-rate output." );
  for ( Index entry = 0; entry < rows.mortar_dofs.size(); ++entry ) {
    if ( rows.mortar_dofs[entry] < 0 || rows.mortar_dofs[entry] >= residual.mortar.entities ) {
      throw std::out_of_range( "A mortar row references a residual DOF outside the field." );
    }
  }
  for ( Index entry = 0; entry < rows.nonmortar_dofs.size(); ++entry ) {
    if ( rows.nonmortar_dofs[entry] < 0 || rows.nonmortar_dofs[entry] >= residual.nonmortar.entities ) {
      throw std::out_of_range( "A nonmortar row references a residual DOF outside the field." );
    }
  }
}

template <typename T>
inline void assignIfPresent( ArrayView<T> values, Index row, T value )
{
  if ( !values.empty() ) {
    values[row] = value;
  }
}

inline void scatterSide( ConstraintRowBatchView rows, Index row, ArrayView<const Index> offsets,
                         ArrayView<const Index> dofs, ArrayView<const Real> basis_values, Real sign, Real row_force,
                         FieldView<Real> residual )
{
  for ( Index entry = offsets[row]; entry < offsets[row + 1]; ++entry ) {
    for ( int component = 0; component < rows.dimension; ++component ) {
      residual( dofs[entry], component ) += sign * basis_values[entry] * row_force * rows.normals( row, component );
    }
  }
}

}  // namespace explicit_row_detail

[[nodiscard]] inline ExplicitRowSummary evaluateExplicitPenaltyRows( ConstraintRowBatchView rows,
                                                                     PenaltyRowInputView penalty,
                                                                     ContactResidualView residual,
                                                                     PenaltyRowOutputView output = {} )
{
  explicit_row_detail::validateExplicitRows( rows, penalty, residual, output );
  ExplicitRowSummary summary;
  summary.rows = rows.numberOfRows();
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    const auto evaluation = explicit_row_detail::evaluateRow( rows, penalty, row );
    explicit_row_detail::scatterSide( rows, row, rows.mortar_dof_offsets, rows.mortar_dofs, rows.mortar_basis_values,
                                      1.0, evaluation.integrated_force, residual.mortar );
    explicit_row_detail::scatterSide( rows, row, rows.nonmortar_dof_offsets, rows.nonmortar_dofs,
                                      rows.nonmortar_basis_values, -1.0, evaluation.integrated_force,
                                      residual.nonmortar );
    explicit_row_detail::assignIfPresent( output.effective_gap, row, evaluation.response.effective_gap );
    explicit_row_detail::assignIfPresent( output.pressure, row, evaluation.response.pressure );
    explicit_row_detail::assignIfPresent( output.row_force, row, evaluation.integrated_force );
    explicit_row_detail::assignIfPresent( output.row_stiffness, row, evaluation.integrated_stiffness );
    explicit_row_detail::assignIfPresent( output.row_rate_coefficient, row, evaluation.integrated_rate_coefficient );
    summary.energy += evaluation.energy;
    summary.contact_power += evaluation.contact_power;
    summary.active_rows += evaluation.response.active() ? 1 : 0;
    summary.active_stiffness_rows += evaluation.integrated_stiffness > 0.0 ? 1 : 0;
    summary.active_rate_rows += evaluation.integrated_rate_coefficient > 0.0 ? 1 : 0;
  }
  return summary;
}

static_assert( std::is_trivially_copyable_v<ExplicitRowSummary> );
static_assert( std::is_trivially_copyable_v<ExplicitRowEvaluation> );

}  // namespace tribol

#endif
