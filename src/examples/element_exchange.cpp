// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "mfem.hpp"

#include "redecomp/redecomp.hpp"

int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );

  int rank = 0;
  int rank_count = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &rank_count );

  // Each rank owns two variable-size element blocks. Sending one block in
  // each direction demonstrates that the schedule depends on provenance and
  // offsets rather than a materialized destination mesh.
  mfem::Array<int> source_offsets{ 0, rank + 1, rank + 3 };
  std::vector<mfem::Array<int>> requests( static_cast<size_t>( rank_count ) );
  requests[static_cast<size_t>( ( rank + 1 ) % rank_count )].Append( 0 );
  requests[static_cast<size_t>( ( rank + rank_count - 1 ) % rank_count )].Append( 1 );
  auto plan = std::make_shared<redecomp::ElementExchangePlan>( MPI_COMM_WORLD, std::move( source_offsets ),
                                                               std::move( requests ) );
  redecomp::ElementExchangeOperator exchange( plan );

  mfem::Vector source( exchange.Width() );
  mfem::Vector patch_dual( exchange.Height() );
  for ( int value = 0; value < source.Size(); ++value ) source[value] = 100.0 * rank + value;
  patch_dual = 1.0;

  mfem::Vector patch_values( exchange.Height() );
  mfem::Vector source_dual( exchange.Width() );
  exchange.Mult( source, patch_values );
  exchange.MultTranspose( patch_dual, source_dual );

  // The transpose returns independent destination duals additively. Equality
  // of these two global work values verifies that no requested copy is lost or
  // counted twice.
  const double local_patch_work = patch_values * patch_dual;
  const double local_source_work = source * source_dual;
  double global_patch_work = 0.0;
  double global_source_work = 0.0;
  MPI_Allreduce( &local_patch_work, &global_patch_work, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_source_work, &global_source_work, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  const double adjoint_error = std::abs( global_patch_work - global_source_work );

  auto assembled = exchange.Assemble();
  mfem::Vector assembled_values( exchange.Height() );
  assembled.get().Mult( source, assembled_values );
  assembled_values -= patch_values;

  const double local_error = assembled_values.Norml2();
  double global_error = 0.0;
  MPI_Allreduce( &local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  const int local_sent_elements = plan->NumSentElements();
  int global_sent_elements = 0;
  MPI_Allreduce( &local_sent_elements, &global_sent_elements, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  if ( rank == 0 ) {
    std::cout << "Exchanged " << global_sent_elements << " element blocks; adjoint error = " << adjoint_error
              << "; matrix-free/sparse error = " << global_error << '\n';
  }

  MPI_Finalize();
  return adjoint_error <= 1.0e-12 && global_error <= 1.0e-12 ? 0 : 1;
}
