#include "tribol/diagnostics/ContactOutput.hpp"

#include "ConstraintRowFixture.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

// Requirements: API-006, OUTPUT-001, OUTPUT-002, OUTPUT-003

namespace {

using namespace tribol;

diagnostics::RankProductionDiagnostics rankZero()
{
  const Real infinity = std::numeric_limits<Real>::infinity();
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  return {
      .rank = 0,
      .ranks = 2,
      .local = { .rows = { .rows = 3,
                           .owned_rows = 2,
                           .active_rows = 2,
                           .nonfinite_rows = 1,
                           .minimum_gap = -0.5,
                           .maximum_penetration = 0.5,
                           .maximum_closing_rate = 0.75,
                           .mortar_force = { 1.0, infinity, -0.0 },
                           .mortar_moment = { 2.0, 3.0, 4.0 },
                           .impulse = { 0.1, 0.2, 0.3 },
                           .maximum_traction = infinity,
                           .spring_energy = nan,
                           .dashpot_dissipation = 0.25,
                           .contact_work = -0.5 },
                 .quadrature = { .samples = 4,
                                 .owned_samples = 2,
                                 .maximum_penetration = infinity,
                                 .maximum_closing_rate = 0.75 } },
      .owned = { .rows = { .rows = 2,
                           .owned_rows = 2,
                           .active_rows = 1,
                           .nonfinite_rows = 0,
                           .minimum_gap = -0.25,
                           .maximum_penetration = 0.25,
                           .maximum_closing_rate = 0.5,
                           .mortar_force = { 1.0, 2.0, 3.0 },
                           .mortar_moment = { 2.0, 3.0, 4.0 },
                           .impulse = { 0.1, 0.2, 0.3 },
                           .maximum_traction = 4.0,
                           .spring_energy = 2.0,
                           .dashpot_dissipation = 0.25,
                           .contact_work = -0.5 },
                 .quadrature = {
                     .samples = 2, .owned_samples = 2, .maximum_penetration = 0.25, .maximum_closing_rate = 0.5 } } };
}

diagnostics::RankProductionDiagnostics rankOne()
{
  return {
      .rank = 1,
      .ranks = 2,
      .local = { .rows = { .rows = 2,
                           .owned_rows = 1,
                           .active_rows = 2,
                           .nonfinite_rows = 0,
                           .minimum_gap = -0.4,
                           .maximum_penetration = 0.4,
                           .maximum_closing_rate = 0.3,
                           .mortar_force = { 5.0, 7.0, 9.0 },
                           .mortar_moment = { 3.0, 4.0, 5.0 },
                           .impulse = { 0.5, 0.7, 0.9 },
                           .maximum_traction = 7.0,
                           .spring_energy = 4.0,
                           .dashpot_dissipation = 0.75,
                           .contact_work = 1.5 },
                 .quadrature =
                     { .samples = 3, .owned_samples = 1, .maximum_penetration = 0.4, .maximum_closing_rate = 0.3 } },
      .owned = { .rows = { .rows = 1,
                           .owned_rows = 1,
                           .active_rows = 1,
                           .nonfinite_rows = 0,
                           .minimum_gap = -0.4,
                           .maximum_penetration = 0.4,
                           .maximum_closing_rate = 0.3,
                           .mortar_force = { 4.0, 5.0, 6.0 },
                           .mortar_moment = { 1.0, 2.0, 3.0 },
                           .impulse = { 0.4, 0.5, 0.6 },
                           .maximum_traction = 7.0,
                           .spring_energy = 3.0,
                           .dashpot_dissipation = 0.75,
                           .contact_work = 1.5 },
                 .quadrature = {
                     .samples = 1, .owned_samples = 1, .maximum_penetration = 0.4, .maximum_closing_rate = 0.3 } } };
}

bool localAndOwnedSummaryContract()
{
  test::TwoRowFixture fixture;
  fixture.owner_ranks[1] = 1;
  fixture.gaps[1] = -0.3;
  fixture.normal_rates[1] = -0.8;
  constexpr std::array<Real, 2> effective_gap{ -0.1, -0.3 };
  constexpr std::array<Real, 2> pressure{ 1.0, 2.0 };
  constexpr std::array<Real, 2> row_force{ 1.0, 2.0 };
  constexpr std::array<Real, 2> row_stiffness{ 10.0, 10.0 };
  constexpr std::array<Real, 2> row_rate{ 0.25, 0.25 };
  const auto rows = fixture.view();
  const diagnostics::QuadratureSampleView quadrature{
      .dimension = rows.dimension,
      .cell_sample_offsets = rows.cell_row_offsets,
      .contact_pair_ids = rows.contact_pair_ids,
      .cell_ids = rows.cell_ids,
      .owner_ranks = rows.owner_ranks,
      .overlap_cells = rows.overlap_cells,
      .normals = rows.normals,
      .integration_weights = rows.integration_weights,
      .mortar_parent_reference_coordinates = rows.mortar_parent_reference_coordinates,
      .nonmortar_parent_reference_coordinates = rows.nonmortar_parent_reference_coordinates,
      .physical_gaps = rows.gaps,
      .normal_rates = rows.normal_rates,
  };
  const auto summary = diagnostics::summarizeProductionDiagnostics( rows,
                                                                    { { effective_gap.data(), 2 },
                                                                      { pressure.data(), 2 },
                                                                      { row_force.data(), 2 },
                                                                      { row_stiffness.data(), 2 },
                                                                      { row_rate.data(), 2 } },
                                                                    quadrature, 0.2, 0, 2 );
  return diagnostics::isStructurallyValid( summary ) && summary.local.rows.rows == 2 &&
         summary.local.rows.owned_rows == 1 && summary.owned.rows.rows == 1 &&
         summary.local.rows.mortar_force[1] == 3.0 && summary.owned.rows.mortar_force[1] == 1.0 &&
         summary.local.quadrature.samples == 2 && summary.local.quadrature.owned_samples == 1 &&
         summary.owned.quadrature.samples == 1 && summary.local.quadrature.maximum_penetration == 0.3 &&
         summary.owned.quadrature.maximum_penetration == 0.1 && summary.local.quadrature.maximum_closing_rate == 0.8 &&
         summary.owned.quadrature.maximum_closing_rate == 0.2;
}

bool reductionContract()
{
  const std::array ranks{ rankZero(), rankOne() };
  const auto global =
      diagnostics::reduceOwnedDiagnostics( ArrayView<const diagnostics::RankProductionDiagnostics>{ ranks.data(), 2 } );
  if ( global.rows.rows != 3 || global.rows.owned_rows != 3 || global.rows.active_rows != 2 ||
       global.quadrature.samples != 3 || global.quadrature.owned_samples != 3 ) {
    return false;
  }
  return global.rows.mortar_force == std::array<Real, 3>{ 5.0, 7.0, 9.0 } &&
         global.rows.mortar_moment == std::array<Real, 3>{ 3.0, 5.0, 7.0 } &&
         std::abs( global.rows.impulse[0] - 0.5 ) < 1.0e-12 && std::abs( global.rows.impulse[1] - 0.7 ) < 1.0e-12 &&
         std::abs( global.rows.impulse[2] - 0.9 ) < 1.0e-12 && global.rows.maximum_traction == 7.0 &&
         global.rows.spring_energy == 5.0 && global.rows.dashpot_dissipation == 1.0 &&
         global.rows.contact_work == 1.0 && global.quadrature.maximum_penetration == 0.4 &&
         global.quadrature.maximum_closing_rate == 0.5;
}

bool payloadAndJsonContract()
{
  ContactAttemptLifecycle lifecycle;
  ContactGenerations generations;
  static_cast<void>( lifecycle.begin( generations ) );
  lifecycle.reject( FailureCategory::Geometry );
  generations.coordinates.advance();
  static_cast<void>( lifecycle.begin( generations ) );
  lifecycle.accept();

  const std::array ranks{ rankZero(), rankOne() };
  const auto global =
      diagnostics::reduceOwnedDiagnostics( ArrayView<const diagnostics::RankProductionDiagnostics>{ ranks.data(), 2 } );
  const auto payload =
      diagnostics::makeProductionDiagnosticsPayload( ranks[0], global, diagnostics::summarizeAttempts( lifecycle ) );
  std::ostringstream first;
  first << std::fixed << std::setprecision( 1 );
  diagnostics::writeJson( first, payload );
  std::ostringstream second;
  second << std::scientific << std::setprecision( 3 );
  diagnostics::writeJson( second, payload );
  const std::string text = first.str();
  const auto local = text.find( "\"local\"" );
  const auto owned = text.find( "\"owned\"", local );
  const auto global_scope = text.find( "\"global\"", owned );
  return text == second.str() &&
         text.starts_with(
             "{\"schema\":\"tribol.milestone-1.production-diagnostics\",\"version\":1,\"rank\":0,"
             "\"ranks\":2" ) &&
         text.find( "\"accepted\":1,\"rejected\":1" ) != std::string::npos &&
         text.find( "\"status\":\"accepted\",\"failure\":\"none\"" ) != std::string::npos &&
         text.find( "\"status\":\"rejected\",\"failure\":\"geometry\"" ) != std::string::npos &&
         text.find( "\"mortar_force\":[1,null,0]" ) != std::string::npos &&
         text.find( "\"spring_energy\":null" ) != std::string::npos &&
         text.find( "\"maximum\":{\"traction\":null,\"penetration\":null,\"closing_rate\":0.75}" ) !=
             std::string::npos &&
         text.find( "\"mortar_force\":[5,7,9]" ) != std::string::npos &&
         text.find( "\"maximum\":{\"traction\":7,\"penetration\":0.40000000000000002,\"closing_rate\":0.5}" ) !=
             std::string::npos &&
         local < owned && owned < global_scope && text.ends_with( "}\n" );
}

bool rankOrderingContract()
{
  const std::array ranks{ rankOne(), rankZero() };
  try {
    static_cast<void>( diagnostics::reduceOwnedDiagnostics(
        ArrayView<const diagnostics::RankProductionDiagnostics>{ ranks.data(), 2 } ) );
  } catch ( const std::invalid_argument& ) {
    return true;
  }
  return false;
}

bool noContactContract()
{
  const auto payload = diagnostics::makeProductionDiagnosticsPayload( diagnostics::RankProductionDiagnostics{} );
  std::ostringstream stream;
  diagnostics::writeJson( stream, payload );
  const std::string text = stream.str();
  return text.find(
             "\"local\":{\"semantics\":\"all_rows_and_diagnostic_samples_resident_on_rank\","
             "\"counts\":{\"rows\":0,\"owned_rows\":0,\"active_rows\":0,\"nonfinite_rows\":0,"
             "\"diagnostic_samples\":0,\"owned_diagnostic_samples\":0}" ) != std::string::npos &&
         text.find(
             "\"global\":{\"semantics\":\"rank_ordered_sum_and_maximum_of_owned_values\","
             "\"counts\":{\"rows\":0,\"owned_rows\":0,\"active_rows\":0,\"nonfinite_rows\":0,"
             "\"diagnostic_samples\":0,\"owned_diagnostic_samples\":0}" ) != std::string::npos &&
         text.find( "\"maximum\":{\"traction\":0,\"penetration\":0,\"closing_rate\":0}" ) != std::string::npos;
}

static_assert( std::is_trivially_copyable_v<diagnostics::ProductionDiagnosticSummary> );
static_assert( std::is_trivially_copyable_v<diagnostics::RankProductionDiagnostics> );
static_assert( std::is_trivially_copyable_v<diagnostics::AttemptDiagnostics> );
static_assert( std::is_trivially_copyable_v<diagnostics::ProductionDiagnosticsPayload> );

}  // namespace

int main()
{
  return localAndOwnedSummaryContract() && reductionContract() && payloadAndJsonContract() && rankOrderingContract() &&
                 noContactContract()
             ? 0
             : 1;
}
