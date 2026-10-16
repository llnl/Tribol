// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/OperatorAssembly.hpp"

#include <utility>

namespace redecomp {

mfem::SparseMatrix AssembleLocalOperator( const mfem::Operator& operator_to_assemble )
{
  const int local_rows = operator_to_assemble.Height();
  const int local_columns = operator_to_assemble.Width();
  mfem::SparseMatrix matrix( local_rows, local_columns );
  mfem::Vector source_basis( local_columns );
  mfem::Vector destination_column( local_rows );
  for ( int column = 0; column < local_columns; ++column ) {
    source_basis = 0.0;
    source_basis[column] = 1.0;
    operator_to_assemble.Mult( source_basis, destination_column );
    for ( int row = 0; row < local_rows; ++row ) {
      if ( destination_column[row] != 0.0 ) {
        matrix.Add( row, column, destination_column[row] );
      }
    }
  }
  matrix.Finalize();
  return matrix;
}

namespace {

mfem::Array<HYPRE_BigInt> BuildPartition( int local_size, MPI_Comm communicator )
{
  int rank = 0;
  int num_ranks = 1;
  MPI_Comm_rank( communicator, &rank );
  MPI_Comm_size( communicator, &num_ranks );
  mfem::Array<int> local_sizes( num_ranks );
  MPI_Allgather( &local_size, 1, MPI_INT, local_sizes.GetData(), 1, MPI_INT, communicator );
  mfem::Array<HYPRE_BigInt> starts( num_ranks + 1 );
  starts[0] = 0;
  for ( int owner = 0; owner < num_ranks; ++owner ) {
    starts[owner + 1] = starts[owner] + local_sizes[owner];
  }
  if ( HYPRE_AssumedPartitionCheck() ) {
    const HYPRE_BigInt global_size = starts.Last();
    const HYPRE_BigInt local_begin = starts[rank];
    const HYPRE_BigInt local_end = starts[rank + 1];
    starts.SetSize( 3 );
    starts[0] = local_begin;
    starts[1] = local_end;
    starts[2] = global_size;
  }
  return starts;
}

}  // namespace

shared::ParSparseMat AssembleBlockDiagonalOperator( mfem::SparseMatrix local_matrix, MPI_Comm communicator )
{
  auto row_starts = BuildPartition( local_matrix.Height(), communicator );
  auto column_starts = BuildPartition( local_matrix.Width(), communicator );
  const HYPRE_BigInt global_rows = row_starts.Last();
  const HYPRE_BigInt global_columns = column_starts.Last();
  shared::ParSparseMat matrix( communicator, global_rows, global_columns, row_starts.GetData(),
                               column_starts.GetData(), std::move( local_matrix ) );
  // MFEM's constructor aliases the supplied partitions. The sparse factor must
  // retain them after this routine returns and releases its temporary arrays.
  matrix->CopyRowStarts();
  matrix->CopyColStarts();
  return matrix;
}

}  // namespace redecomp
