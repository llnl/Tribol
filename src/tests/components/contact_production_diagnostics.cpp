#include "tribol/Tribol.hpp"

#include "ParentTraceFixture.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Requirements: API-006, PHYS-004, OUTPUT-001, OUTPUT-003

namespace {

using namespace tribol;

using NativeDampedPenalty =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Automatic, constraint::QuadraturePoint,
           enforcement::Penalty<stiffness::Constant, rate::Constant>, response::Frictionless,
           formulation::PointwiseTraction, linearization::Exact>;

struct NativeFixture {
  test::SegmentTraceStorage mortar = test::segmentTrace( 0.0, 1.0, 0.0, false, 101, 0 );
  test::SegmentTraceStorage nonmortar = test::segmentTrace( 0.0, 1.0, -0.1, true, 202, 1 );
  std::array<Real, 4> mortar_coordinates{ 0.0, -0.3, 1.0, -0.3 };
  std::array<Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  std::array<Real, 4> mortar_velocity{ 0.0, -0.6, 0.0, -0.6 };
  std::array<Real, 4> nonmortar_velocity{ 0.0, -0.1, 0.0, -0.1 };
  std::array<Real, 4> inverse_mass{ 1.0, 1.0, 1.0, 1.0 };

  [[nodiscard]] ParentTracePairView traces() const { return { mortar.view(), nonmortar.view() }; }

  [[nodiscard]] ParentTraceFieldPairView fields() const
  {
    return {
        .mortar_coordinates = { { mortar_coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar_coordinates = { { nonmortar_coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
    };
  }

  [[nodiscard]] ContactStateView state() const
  {
    return {
        .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
    };
  }

  [[nodiscard]] timestep::DiagonalInverseMassView inverseMass() const
  {
    return {
        .mortar = { { inverse_mass.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar = { { inverse_mass.data(), 4 }, 2, 2, FieldLayout::Interleaved },
    };
  }
};

[[nodiscard]] NativeDampedPenalty::Parameters parameters()
{
  NativeDampedPenalty::Parameters result;
  result.enforcement.stiffness.value = 10.0;
  result.enforcement.rate.value = 2.0;
  return result;
}

[[nodiscard]] Contact<NativeDampedPenalty>::Options options()
{
  Contact<NativeDampedPenalty>::Options result{ .method = parameters() };
  result.search.expansion = 0.25;
  return result;
}

[[nodiscard]] bool close( Real left, Real right ) { return std::abs( left - right ) < 1.0e-12; }

template <typename Function>
[[nodiscard]] bool throwsLogicError( Function&& function )
{
  try {
    function();
  } catch ( const std::logic_error& ) {
    return true;
  }
  return false;
}

bool analyticTotalsContract()
{
  NativeFixture fixture;
  Contact<NativeDampedPenalty> contact( fixture.traces(), fixture.fields(), options() );
  contact.updateInteractions();
  const auto result = contact.evaluate( fixture.state(), fixture.inverseMass(),
                                        timestep::IntegratorStabilityLimits::centralDifference() );
  const auto summary = contact.productionDiagnostics( fixture.state(), 0.25 );
  const auto samples = contact.diagnosticQuadrature();
  const auto& rows = summary.local.rows;
  const bool valid = result.summary.quadrature_points == 2 && rows.rows == 2 && rows.owned_rows == 2 &&
                     rows.active_rows == 2 && rows.nonfinite_rows == 0 && close( rows.minimum_gap, -0.2 ) &&
                     close( rows.maximum_penetration, 0.2 ) && close( rows.maximum_closing_rate, 0.5 ) &&
                     close( rows.mortar_force[0], 0.0 ) && close( rows.mortar_force[1], -2.0 ) &&
                     close( rows.mortar_moment[2], -1.0 ) && close( rows.impulse[1], -0.5 ) &&
                     close( rows.maximum_traction, 2.0 ) && close( rows.spring_energy, 0.1 ) &&
                     close( rows.dashpot_dissipation, 0.125 ) && close( rows.contact_work, 0.25 ) &&
                     close( result.summary.energy, rows.spring_energy ) && samples.isStructurallyValid();
  if ( !valid ) {
    std::cerr << "analytic rows=" << rows.rows << " owned=" << rows.owned_rows << " active=" << rows.active_rows
              << " gap=" << rows.minimum_gap << " penetration=" << rows.maximum_penetration
              << " rate=" << rows.maximum_closing_rate << " force=" << rows.mortar_force[1]
              << " moment=" << rows.mortar_moment[2] << " impulse=" << rows.impulse[1]
              << " traction=" << rows.maximum_traction << " energy=" << rows.spring_energy
              << " dissipation=" << rows.dashpot_dissipation << " work=" << rows.contact_work
              << " result_energy=" << result.summary.energy << " q=" << result.summary.quadrature_points << '\n';
  }
  return valid;
}

bool independentSamplingContract()
{
  NativeFixture fixture;
  Contact<NativeDampedPenalty> contact( fixture.traces(), fixture.fields(), options() );
  contact.updateInteractions();
  const auto result = contact.evaluate( fixture.state(), fixture.inverseMass(),
                                        timestep::IntegratorStabilityLimits::centralDifference() );
  const Real force = result.mortar_force( 0, 1 ) + result.mortar_force( 1, 1 );
  const auto low = contact.productionDiagnostics( fixture.state(), 0.25, 0, 1, { .exactness = 0 } );
  const auto high = contact.productionDiagnostics( fixture.state(), 0.25, 0, 1, { .exactness = 8 } );
  const auto samples = contact.diagnosticQuadrature();
  const bool valid = low.local.rows.rows == high.local.rows.rows && low.local.quadrature.samples == 1 &&
                     high.local.quadrature.samples == 5 && samples.numberOfSamples() == 5 &&
                     close( result.mortar_force( 0, 1 ) + result.mortar_force( 1, 1 ), force ) &&
                     close( high.local.rows.spring_energy, result.summary.energy );
  if ( !valid ) {
    std::cerr << "independent rows=" << low.local.rows.rows << ',' << high.local.rows.rows
              << " samples=" << low.local.quadrature.samples << ',' << high.local.quadrature.samples
              << " latest=" << samples.numberOfSamples() << " force=" << force << "->"
              << result.mortar_force( 0, 1 ) + result.mortar_force( 1, 1 ) << '\n';
  }
  return valid;
}

bool optionalRateAndNoContactContract()
{
  NativeFixture fixture;
  const auto contact_options = options();
  Contact<NativeDampedPenalty> contact( fixture.traces(), fixture.fields(), contact_options );
  contact.updateInteractions();
  static_cast<void>( contact.evaluate( fixture.state(), fixture.inverseMass(),
                                       timestep::IntegratorStabilityLimits::centralDifference() ) );
  const auto without_rate = contact.productionDiagnostics( fixture.state(), 0.25 );
  if ( contact.diagnosticQuadrature().hasNormalRates() || without_rate.local.quadrature.maximum_closing_rate != 0.0 ) {
    std::cerr << "unexpected default diagnostic rates\n";
    return false;
  }
  const auto with_rate = contact.productionDiagnostics( fixture.state(), 0.25, 0, 1, { .include_normal_rates = true } );
  if ( !contact.diagnosticQuadrature().hasNormalRates() ||
       !close( with_rate.local.quadrature.maximum_closing_rate, 0.5 ) ) {
    std::cerr << "rate=" << with_rate.local.quadrature.maximum_closing_rate << '\n';
    return false;
  }

  Contact<NativeDampedPenalty> empty( fixture.traces(), fixture.fields(), contact_options );
  empty.setInteractions( {} );
  const auto result = empty.evaluate( fixture.state(), fixture.inverseMass(),
                                      timestep::IntegratorStabilityLimits::centralDifference() );
  const auto diagnostics = empty.productionDiagnostics( fixture.state(), 0.25 );
  const bool valid = result.summary.active_interactions == 0 && diagnostics.local.rows.rows == 0 &&
                     diagnostics.local.rows.minimum_gap == 0.0 && diagnostics.local.quadrature.samples == 0 &&
                     empty.diagnosticQuadrature().numberOfSamples() == 0;
  if ( !valid ) {
    std::cerr << "empty active=" << result.summary.active_interactions << " rows=" << diagnostics.local.rows.rows
              << " min=" << diagnostics.local.rows.minimum_gap << " samples=" << diagnostics.local.quadrature.samples
              << '\n';
  }
  return valid;
}

bool emptyRankReductionContract()
{
  std::array<diagnostics::RankProductionDiagnostics, 2> ranks{};
  ranks[0] = { .rank = 0, .ranks = 2 };
  ranks[0].local.rows.minimum_gap = 0.25;
  ranks[0].owned.rows = { .rows = 1, .owned_rows = 1, .minimum_gap = 0.25 };
  ranks[0].local.rows = ranks[0].owned.rows;
  ranks[1] = { .rank = 1, .ranks = 2 };
  ranks[1].local.rows.minimum_gap = 0.0;
  ranks[1].owned.rows.minimum_gap = 0.0;
  const auto global = diagnostics::reduceOwnedDiagnostics( { ranks.data(), 2 } );
  return global.rows.rows == 1 && global.rows.owned_rows == 1 && close( global.rows.minimum_gap, 0.25 );
}

bool staleEvaluationContract()
{
  NativeFixture fixture;
  Contact<NativeDampedPenalty> contact( fixture.traces(), fixture.fields(), options() );
  contact.updateInteractions();
  if ( !throwsLogicError( [&] { static_cast<void>( contact.productionDiagnostics( fixture.state(), 0.25 ) ); } ) ) {
    return false;
  }
  static_cast<void>( contact.evaluate( fixture.state(), fixture.inverseMass(),
                                       timestep::IntegratorStabilityLimits::centralDifference() ) );
  static_cast<void>( contact.productionDiagnostics( fixture.state(), 0.25 ) );
  static_cast<void>( contact.evaluate( fixture.state(), fixture.inverseMass(),
                                       timestep::IntegratorStabilityLimits::centralDifference() ) );
  if ( !throwsLogicError( [&] { static_cast<void>( contact.diagnosticQuadrature() ); } ) ) {
    return false;
  }
  static_cast<void>( contact.productionDiagnostics( fixture.state(), 0.25 ) );

  MassVersion mass;
  mass.advance();
  contact.setMassGeneration( mass );
  if ( !throwsLogicError( [&] { static_cast<void>( contact.productionDiagnostics( fixture.state(), 0.25 ) ); } ) ||
       !throwsLogicError( [&] { static_cast<void>( contact.diagnosticQuadrature() ); } ) ) {
    return false;
  }

  static_cast<void>( contact.evaluate( fixture.state(), fixture.inverseMass(),
                                       timestep::IntegratorStabilityLimits::centralDifference() ) );
  static_cast<void>( contact.productionDiagnostics( fixture.state(), 0.25 ) );
  fixture.mortar_coordinates[1] -= 0.01;
  fixture.mortar_coordinates[3] -= 0.01;
  contact.updateGeometry( fixture.traces(), fixture.fields() );
  if ( !throwsLogicError( [&] { static_cast<void>( contact.productionDiagnostics( fixture.state(), 0.25 ) ); } ) ) {
    return false;
  }

  static_cast<void>( contact.evaluate( fixture.state(), fixture.inverseMass(),
                                       timestep::IntegratorStabilityLimits::centralDifference() ) );
  static_cast<void>( contact.productionDiagnostics( fixture.state(), 0.25 ) );
  contact.updateInteractions();
  return throwsLogicError( [&] { static_cast<void>( contact.productionDiagnostics( fixture.state(), 0.25 ) ); } );
}

}  // namespace

int main()
{
  const bool analytic = analyticTotalsContract();
  const bool independent = independentSamplingContract();
  const bool optional = optionalRateAndNoContactContract();
  const bool empty_rank = emptyRankReductionContract();
  const bool stale = staleEvaluationContract();
  if ( !analytic || !independent || !optional || !empty_rank || !stale ) {
    std::cerr << "analytic=" << analytic << " independent=" << independent << " optional=" << optional
              << " empty_rank=" << empty_rank << " stale=" << stale << '\n';
  }
  return analytic && independent && optional && empty_rank && stale ? 0 : 1;
}
