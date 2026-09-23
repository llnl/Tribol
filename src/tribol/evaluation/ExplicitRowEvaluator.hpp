#ifndef TRIBOL_EVALUATION_EXPLICITROWEVALUATOR_HPP_
#define TRIBOL_EVALUATION_EXPLICITROWEVALUATOR_HPP_

#include "tribol/core/ConstraintRowView.hpp"
#include "tribol/enforcement/PenaltyLaw.hpp"
#include "tribol/evaluation/State.hpp"

#include <stdexcept>

namespace tribol {

struct ExplicitRowSummary {
  Real energy{};
  Real contact_power{};
  Index active_rows{};
  Index rows{};
};

namespace explicit_row_detail {

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
    const Real rate_coefficient = penalty.rate_coefficients.empty() ? 0.0 : penalty.rate_coefficients[row];
    const auto response = enforcement::evaluateNormalPenalty( enforcement::NormalPenaltyInput{
        .gap = rows.gaps[row],
        .normal_rate = rows.hasNormalRates() ? rows.normal_rates[row] : 0.0,
        .pressure_stiffness = penalty.pressure_stiffness[row],
        .rate_coefficient = rate_coefficient,
        .activation = penalty.activation,
    } );
    const Real integrated_force = rows.integration_weights[row] * response.pressure;
    explicit_row_detail::scatterSide( rows, row, rows.mortar_dof_offsets, rows.mortar_dofs, rows.mortar_basis_values,
                                      1.0, integrated_force, residual.mortar );
    explicit_row_detail::scatterSide( rows, row, rows.nonmortar_dof_offsets, rows.nonmortar_dofs,
                                      rows.nonmortar_basis_values, -1.0, integrated_force, residual.nonmortar );
    explicit_row_detail::assignIfPresent( output.effective_gap, row, response.effective_gap );
    explicit_row_detail::assignIfPresent( output.pressure, row, response.pressure );
    explicit_row_detail::assignIfPresent( output.row_force, row, integrated_force );
    explicit_row_detail::assignIfPresent( output.row_stiffness, row,
                                          rows.integration_weights[row] * response.pressure_gap_derivative );
    explicit_row_detail::assignIfPresent( output.row_rate_coefficient, row,
                                          rows.integration_weights[row] * response.pressure_rate_derivative );
    summary.energy += rows.integration_weights[row] * response.potential_density;
    summary.contact_power += integrated_force * ( rows.hasNormalRates() ? rows.normal_rates[row] : 0.0 );
    summary.active_rows += response.active() ? 1 : 0;
  }
  return summary;
}

}  // namespace tribol

#endif
