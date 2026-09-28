#ifndef TRIBOL_ADAPTERS_MFEM_MFEMPRODUCTIONDIAGNOSTICS_HPP_
#define TRIBOL_ADAPTERS_MFEM_MFEMPRODUCTIONDIAGNOSTICS_HPP_

#include "tribol/adapters/mfem/MfemSurface.hpp"
#include "tribol/diagnostics/ProductionDiagnostics.hpp"

#include <mpi.h>

#include <cstdint>
#include <limits>

namespace tribol::mfem::detail {

struct CommunicatorInfo {
  int rank{};
  int size{ 1 };
};

[[nodiscard]] inline CommunicatorInfo communicatorInfo( MPI_Comm communicator )
{
  CommunicatorInfo result;
  MPI_Comm_rank( communicator, &result.rank );
  MPI_Comm_size( communicator, &result.size );
  return result;
}

[[nodiscard]] inline diagnostics::ProductionDiagnosticSummary reduceOwnedProductionDiagnostics(
    MPI_Comm communicator, const diagnostics::ProductionDiagnosticSummary& owned )
{
  std::int64_t local_counts[]{ static_cast<std::int64_t>( owned.rows.rows ),
                               static_cast<std::int64_t>( owned.rows.owned_rows ),
                               static_cast<std::int64_t>( owned.rows.active_rows ),
                               static_cast<std::int64_t>( owned.rows.nonfinite_rows ),
                               static_cast<std::int64_t>( owned.quadrature.samples ),
                               static_cast<std::int64_t>( owned.quadrature.owned_samples ) };
  std::int64_t global_counts[6]{};
  MPI_Allreduce( local_counts, global_counts, 6, MPI_INT64_T, MPI_SUM, communicator );

  Real local_sums[]{ owned.rows.mortar_force[0],  owned.rows.mortar_force[1],     owned.rows.mortar_force[2],
                     owned.rows.mortar_moment[0], owned.rows.mortar_moment[1],    owned.rows.mortar_moment[2],
                     owned.rows.impulse[0],       owned.rows.impulse[1],          owned.rows.impulse[2],
                     owned.rows.spring_energy,    owned.rows.dashpot_dissipation, owned.rows.contact_work };
  Real global_sums[12]{};
  MPI_Allreduce( local_sums, global_sums, 12, realMpiType(), MPI_SUM, communicator );

  const Real negative_infinity = -std::numeric_limits<Real>::infinity();
  Real local_maxima[]{ owned.rows.rows > 0 ? -owned.rows.minimum_gap : negative_infinity,
                       owned.rows.maximum_penetration,
                       owned.rows.maximum_closing_rate,
                       owned.rows.maximum_traction,
                       owned.quadrature.maximum_penetration,
                       owned.quadrature.maximum_closing_rate };
  Real global_maxima[6]{};
  MPI_Allreduce( local_maxima, global_maxima, 6, realMpiType(), MPI_MAX, communicator );

  diagnostics::ProductionDiagnosticSummary global;
  global.rows.rows = static_cast<Index>( global_counts[0] );
  global.rows.owned_rows = global.rows.rows;
  global.rows.active_rows = static_cast<Index>( global_counts[2] );
  global.rows.nonfinite_rows = static_cast<Index>( global_counts[3] );
  global.rows.minimum_gap = global.rows.rows > 0 ? -global_maxima[0] : 0.0;
  global.rows.maximum_penetration = global_maxima[1];
  global.rows.maximum_closing_rate = global_maxima[2];
  global.rows.maximum_traction = global_maxima[3];
  global.rows.mortar_force = { global_sums[0], global_sums[1], global_sums[2] };
  global.rows.mortar_moment = { global_sums[3], global_sums[4], global_sums[5] };
  global.rows.impulse = { global_sums[6], global_sums[7], global_sums[8] };
  global.rows.spring_energy = global_sums[9];
  global.rows.dashpot_dissipation = global_sums[10];
  global.rows.contact_work = global_sums[11];
  global.quadrature.samples = static_cast<Index>( global_counts[4] );
  global.quadrature.owned_samples = global.quadrature.samples;
  global.quadrature.maximum_penetration = global_maxima[4];
  global.quadrature.maximum_closing_rate = global_maxima[5];
  return global;
}

}  // namespace tribol::mfem::detail

#endif
