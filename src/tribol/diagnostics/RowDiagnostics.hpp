#ifndef TRIBOL_DIAGNOSTICS_ROWDIAGNOSTICS_HPP_
#define TRIBOL_DIAGNOSTICS_ROWDIAGNOSTICS_HPP_

#include "tribol/core/ConstraintRowView.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tribol::diagnostics {

struct ExplicitRowDiagnostics {
  Index rows{};
  Index owned_rows{};
  Index active_rows{};
  Index nonfinite_rows{};
  Real minimum_gap{ std::numeric_limits<Real>::infinity() };
  Real maximum_penetration{};
  Real maximum_closing_rate{};
  std::array<Real, 3> mortar_force{};
  std::array<Real, 3> mortar_moment{};
  std::array<Real, 3> impulse{};
  Real maximum_traction{};
  Real spring_energy{};
  Real dashpot_dissipation{};
  Real contact_work{};
};

namespace row_diagnostics_detail {

[[nodiscard]] TRIBOL_HOST_DEVICE inline Real absolute( Real value ) { return value < 0.0 ? -value : value; }

[[nodiscard]] TRIBOL_HOST_DEVICE inline bool finite( Real value )
{
  return value == value && value != std::numeric_limits<Real>::infinity() &&
         value != -std::numeric_limits<Real>::infinity();
}

inline void requireRowArray( ArrayView<const Real> values, Index rows, const char* message )
{
  if ( !values.isStructurallyValid() || values.size() != rows ) {
    throw std::invalid_argument( message );
  }
}

}  // namespace row_diagnostics_detail

[[nodiscard]] TRIBOL_HOST_DEVICE inline ExplicitRowDiagnostics explicitRowDiagnostic(
    ConstraintRowBatchView rows, PenaltyRowOutputViewT<const Real> output, Index row, Real time_step,
    int local_rank = -1 )
{
  ExplicitRowDiagnostics result;
  result.rows = 1;
  result.owned_rows = local_rank < 0 || rows.owner_ranks[row] == local_rank ? 1 : 0;
  const Real gap = rows.gaps[row];
  const Real rate = rows.hasNormalRates() ? rows.normal_rates[row] : 0.0;
  const Real pressure = output.pressure[row];
  const Real force = output.row_force[row];
  const Real stiffness = output.row_stiffness[row];
  const Real damping = output.row_rate_coefficient[row];
  result.nonfinite_rows =
      row_diagnostics_detail::finite( gap ) && row_diagnostics_detail::finite( rate ) &&
              row_diagnostics_detail::finite( pressure ) && row_diagnostics_detail::finite( force ) &&
              row_diagnostics_detail::finite( stiffness ) && row_diagnostics_detail::finite( damping )
          ? 0
          : 1;
  result.minimum_gap = gap;
  result.maximum_penetration = gap < 0.0 ? -gap : 0.0;
  result.maximum_closing_rate = rate < 0.0 ? -rate : 0.0;
  result.active_rows = pressure != 0.0 ? 1 : 0;
  result.maximum_traction = row_diagnostics_detail::absolute( pressure );
  const Real active_gap = output.effective_gap[row] < 0.0 ? output.effective_gap[row] : 0.0;
  result.spring_energy = 0.5 * stiffness * active_gap * active_gap;
  result.dashpot_dissipation = time_step * damping * result.maximum_closing_rate * result.maximum_closing_rate;
  result.contact_work = time_step * force * rate;
  for ( int component = 0; component < rows.dimension; ++component ) {
    result.mortar_force[component] = force * rows.normals( row, component );
    result.impulse[component] = time_step * result.mortar_force[component];
  }
  const std::array<Real, 3> position{ rows.mortar_positions( row, 0 ), rows.mortar_positions( row, 1 ),
                                      rows.mortar_positions( row, 2 ) };
  result.mortar_moment = {
      position[1] * result.mortar_force[2] - position[2] * result.mortar_force[1],
      position[2] * result.mortar_force[0] - position[0] * result.mortar_force[2],
      position[0] * result.mortar_force[1] - position[1] * result.mortar_force[0],
  };
  return result;
}

TRIBOL_HOST_DEVICE inline void accumulate( ExplicitRowDiagnostics& total, const ExplicitRowDiagnostics& row )
{
  total.rows += row.rows;
  total.owned_rows += row.owned_rows;
  total.active_rows += row.active_rows;
  total.nonfinite_rows += row.nonfinite_rows;
  total.minimum_gap = row.minimum_gap < total.minimum_gap ? row.minimum_gap : total.minimum_gap;
  total.maximum_penetration =
      row.maximum_penetration > total.maximum_penetration ? row.maximum_penetration : total.maximum_penetration;
  total.maximum_closing_rate =
      row.maximum_closing_rate > total.maximum_closing_rate ? row.maximum_closing_rate : total.maximum_closing_rate;
  total.maximum_traction =
      row.maximum_traction > total.maximum_traction ? row.maximum_traction : total.maximum_traction;
  total.spring_energy += row.spring_energy;
  total.dashpot_dissipation += row.dashpot_dissipation;
  total.contact_work += row.contact_work;
  for ( int component = 0; component < 3; ++component ) {
    total.mortar_force[component] += row.mortar_force[component];
    total.mortar_moment[component] += row.mortar_moment[component];
    total.impulse[component] += row.impulse[component];
  }
}

[[nodiscard]] inline ExplicitRowDiagnostics summarizeExplicitRows( ConstraintRowBatchView rows,
                                                                   PenaltyRowOutputViewT<const Real> output,
                                                                   Real time_step = 0.0, int local_rank = -1 )
{
  if ( !rows.isStructurallyValid() || !std::isfinite( time_step ) || time_step < 0.0 ) {
    throw std::invalid_argument( "Explicit row diagnostics require valid rows and a nonnegative timestep." );
  }
  row_diagnostics_detail::requireRowArray( output.effective_gap, rows.numberOfRows(),
                                           "Explicit row diagnostics require effective gaps." );
  row_diagnostics_detail::requireRowArray( output.pressure, rows.numberOfRows(),
                                           "Explicit row diagnostics require pressures." );
  row_diagnostics_detail::requireRowArray( output.row_force, rows.numberOfRows(),
                                           "Explicit row diagnostics require integrated forces." );
  row_diagnostics_detail::requireRowArray( output.row_stiffness, rows.numberOfRows(),
                                           "Explicit row diagnostics require integrated stiffness." );
  row_diagnostics_detail::requireRowArray( output.row_rate_coefficient, rows.numberOfRows(),
                                           "Explicit row diagnostics require integrated rate coefficients." );
  ExplicitRowDiagnostics result;
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    accumulate( result, explicitRowDiagnostic( rows, output, row, time_step, local_rank ) );
  }
  if ( rows.numberOfRows() == 0 ) {
    result.minimum_gap = 0.0;
  }
  return result;
}

static_assert( std::is_trivially_copyable_v<ExplicitRowDiagnostics> );

}  // namespace tribol::diagnostics

#endif
