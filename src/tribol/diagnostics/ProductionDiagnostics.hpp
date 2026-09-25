#ifndef TRIBOL_DIAGNOSTICS_PRODUCTIONDIAGNOSTICS_HPP_
#define TRIBOL_DIAGNOSTICS_PRODUCTIONDIAGNOSTICS_HPP_

#include "tribol/core/AttemptLifecycle.hpp"
#include "tribol/diagnostics/QuadratureSampling.hpp"
#include "tribol/diagnostics/RowDiagnostics.hpp"

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace tribol::diagnostics {

struct ProductionDiagnosticSummary {
  ExplicitRowDiagnostics rows{};
  QuadratureSampleSummary quadrature{};
};

struct RankProductionDiagnostics {
  int rank{};
  int ranks{ 1 };
  ProductionDiagnosticSummary local{};
  ProductionDiagnosticSummary owned{};
};

struct AttemptDiagnostics {
  ContactAttemptRecord current{};
  ContactAttemptRecord last_accepted{};
  ContactAttemptRecord last_rejected{};
  std::uint64_t accepted_attempts{};
  std::uint64_t rejected_attempts{};
};

struct ProductionDiagnosticsPayload {
  std::uint32_t schema_version{ 1 };
  RankProductionDiagnostics rank{};
  ProductionDiagnosticSummary global{};
  AttemptDiagnostics attempts{};
};

namespace production_diagnostics_detail {

[[nodiscard]] inline bool validCounts( const ExplicitRowDiagnostics& rows )
{
  return rows.rows >= 0 && rows.owned_rows >= 0 && rows.owned_rows <= rows.rows && rows.active_rows >= 0 &&
         rows.active_rows <= rows.rows && rows.nonfinite_rows >= 0 && rows.nonfinite_rows <= rows.rows;
}

[[nodiscard]] inline bool validCounts( const QuadratureSampleSummary& quadrature )
{
  return quadrature.samples >= 0 && quadrature.owned_samples >= 0 && quadrature.owned_samples <= quadrature.samples;
}

[[nodiscard]] inline ExplicitRowDiagnostics summarizeOwnedRows( ConstraintRowBatchView rows,
                                                                PenaltyRowOutputViewT<const Real> output,
                                                                Real time_step, int local_rank )
{
  ExplicitRowDiagnostics result;
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    if ( rows.owner_ranks[row] == local_rank ) {
      accumulate( result, explicitRowDiagnostic( rows, output, row, time_step, local_rank ) );
    }
  }
  if ( result.rows == 0 ) {
    result.minimum_gap = 0.0;
  }
  return result;
}

}  // namespace production_diagnostics_detail

[[nodiscard]] inline bool isStructurallyValid( const RankProductionDiagnostics& diagnostics )
{
  return diagnostics.rank >= 0 && diagnostics.ranks > 0 && diagnostics.rank < diagnostics.ranks &&
         production_diagnostics_detail::validCounts( diagnostics.local.rows ) &&
         production_diagnostics_detail::validCounts( diagnostics.owned.rows ) &&
         production_diagnostics_detail::validCounts( diagnostics.local.quadrature ) &&
         production_diagnostics_detail::validCounts( diagnostics.owned.quadrature ) &&
         diagnostics.local.rows.owned_rows == diagnostics.owned.rows.rows &&
         diagnostics.owned.rows.owned_rows == diagnostics.owned.rows.rows &&
         diagnostics.local.quadrature.owned_samples == diagnostics.owned.quadrature.samples &&
         diagnostics.owned.quadrature.owned_samples == diagnostics.owned.quadrature.samples;
}

[[nodiscard]] inline bool isStructurallyValid( const ProductionDiagnosticsPayload& payload )
{
  return payload.schema_version == 1 && isStructurallyValid( payload.rank ) &&
         production_diagnostics_detail::validCounts( payload.global.rows ) &&
         production_diagnostics_detail::validCounts( payload.global.quadrature ) &&
         payload.global.rows.owned_rows == payload.global.rows.rows &&
         payload.global.quadrature.owned_samples == payload.global.quadrature.samples &&
         payload.global.rows.rows >= payload.rank.owned.rows.rows &&
         payload.global.quadrature.samples >= payload.rank.owned.quadrature.samples;
}

[[nodiscard]] inline RankProductionDiagnostics summarizeProductionDiagnostics( ConstraintRowBatchView rows,
                                                                               PenaltyRowOutputViewT<const Real> output,
                                                                               QuadratureSampleView quadrature,
                                                                               Real time_step, int local_rank,
                                                                               int ranks = 1 )
{
  if ( local_rank < 0 || ranks <= 0 || local_rank >= ranks ) {
    throw std::invalid_argument( "Production diagnostics require a valid local rank and rank count." );
  }
  RankProductionDiagnostics result{ .rank = local_rank, .ranks = ranks };
  result.local.rows = summarizeExplicitRows( rows, output, time_step, local_rank );
  result.owned.rows = production_diagnostics_detail::summarizeOwnedRows( rows, output, time_step, local_rank );
  result.local.quadrature = summarizeQuadrature( quadrature );
  result.owned.quadrature = summarizeQuadrature( quadrature, local_rank );
  result.local.quadrature.owned_samples = result.owned.quadrature.owned_samples;
  result.owned.quadrature.samples = result.owned.quadrature.owned_samples;
  return result;
}

[[nodiscard]] inline ProductionDiagnosticSummary reduceOwnedDiagnostics(
    ArrayView<const RankProductionDiagnostics> rank_diagnostics )
{
  if ( !rank_diagnostics.isStructurallyValid() || rank_diagnostics.empty() ) {
    throw std::invalid_argument( "Global production diagnostics require one contribution per rank." );
  }
  ProductionDiagnosticSummary result;
  for ( Index rank = 0; rank < rank_diagnostics.size(); ++rank ) {
    const auto& contribution = rank_diagnostics[rank];
    if ( contribution.rank != rank || contribution.ranks != rank_diagnostics.size() ||
         !isStructurallyValid( contribution ) ) {
      throw std::invalid_argument( "Production diagnostic contributions must be complete and ordered by rank." );
    }
    accumulate( result.rows, contribution.owned.rows );
    result.quadrature.samples += contribution.owned.quadrature.samples;
    result.quadrature.owned_samples += contribution.owned.quadrature.owned_samples;
    result.quadrature.maximum_penetration =
        contribution.owned.quadrature.maximum_penetration > result.quadrature.maximum_penetration
            ? contribution.owned.quadrature.maximum_penetration
            : result.quadrature.maximum_penetration;
    result.quadrature.maximum_closing_rate =
        contribution.owned.quadrature.maximum_closing_rate > result.quadrature.maximum_closing_rate
            ? contribution.owned.quadrature.maximum_closing_rate
            : result.quadrature.maximum_closing_rate;
  }
  if ( result.rows.rows == 0 ) {
    result.rows.minimum_gap = 0.0;
  }
  return result;
}

[[nodiscard]] inline AttemptDiagnostics summarizeAttempts( const ContactAttemptLifecycle& lifecycle )
{
  return { .current = lifecycle.current(),
           .last_accepted = lifecycle.lastAccepted(),
           .last_rejected = lifecycle.lastRejected(),
           .accepted_attempts = lifecycle.acceptedAttempts(),
           .rejected_attempts = lifecycle.rejectedAttempts() };
}

[[nodiscard]] inline ProductionDiagnosticsPayload makeProductionDiagnosticsPayload( RankProductionDiagnostics rank,
                                                                                    ProductionDiagnosticSummary global,
                                                                                    AttemptDiagnostics attempts = {} )
{
  ProductionDiagnosticsPayload result{ .rank = rank, .global = global, .attempts = attempts };
  if ( !isStructurallyValid( result ) ) {
    throw std::invalid_argument( "Production diagnostics require consistent local, owned, and global counts." );
  }
  return result;
}

[[nodiscard]] inline ProductionDiagnosticsPayload makeProductionDiagnosticsPayload( RankProductionDiagnostics rank,
                                                                                    AttemptDiagnostics attempts = {} )
{
  if ( rank.ranks != 1 || rank.rank != 0 ) {
    throw std::invalid_argument( "Global diagnostics must be supplied for a multi-rank payload." );
  }
  return makeProductionDiagnosticsPayload( rank, rank.owned, attempts );
}

static_assert( std::is_trivially_copyable_v<ProductionDiagnosticSummary> );
static_assert( std::is_trivially_copyable_v<RankProductionDiagnostics> );
static_assert( std::is_trivially_copyable_v<AttemptDiagnostics> );
static_assert( std::is_trivially_copyable_v<ProductionDiagnosticsPayload> );

}  // namespace tribol::diagnostics

#endif
