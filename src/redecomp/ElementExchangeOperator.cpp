// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/ElementExchangeOperator.hpp"

#include <algorithm>

#include "axom/slic.hpp"

namespace redecomp {

ElementExchangeOperator::ElementExchangeOperator( std::shared_ptr<const ElementExchangePlan> plan )
    : mfem::Operator( plan->DestinationDataSize(), plan->SourceDataSize() ),
      plan_{ std::move( plan ) },
      send_buffer_( plan_->SendScalarCounts().Sum() ),
      receive_buffer_( plan_->SendScalarCounts().Sum() )
{
}

void ElementExchangeOperator::PackSourceBlocks( const mfem::Vector& source, mfem::Vector& send_buffer ) const
{
  const auto& offsets = plan_->SourceElementOffsets();
  const auto& elements_by_destination = plan_->SourceElementsByDestination();
  const auto& scalar_displacements = plan_->SendScalarDisplacements();
  const mfem::real_t* source_values = source.HostRead();
  mfem::real_t* packed_values = send_buffer.HostWrite();

  for ( int destination = 0; destination < plan_->NumRanks(); ++destination ) {
    int packed_offset = scalar_displacements[destination];
    for ( int source_element : elements_by_destination[destination] ) {
      const int source_begin = offsets[source_element];
      const int block_size = offsets[source_element + 1] - source_begin;
      std::copy_n( source_values + source_begin, block_size, packed_values + packed_offset );
      packed_offset += block_size;
    }
  }
}

void ElementExchangeOperator::Mult( const mfem::Vector& source, mfem::Vector& destination ) const
{
  SLIC_ERROR_ROOT_IF( source.Size() != Width(), "ElementExchangeOperator source size does not match its width." );
  SLIC_ERROR_ROOT_IF( destination.Size() != Height(),
                      "ElementExchangeOperator destination size does not match its height." );

  PackSourceBlocks( source, send_buffer_ );
  MPI_Alltoallv( send_buffer_.HostRead(), plan_->SendScalarCounts().GetData(),
                 plan_->SendScalarDisplacements().GetData(), MFEM_MPI_REAL_T, destination.HostWrite(),
                 plan_->ReceiveScalarCounts().GetData(), plan_->ReceiveScalarDisplacements().GetData(), MFEM_MPI_REAL_T,
                 plan_->GetComm() );
}

void ElementExchangeOperator::AddReturnedBlocks( const mfem::Vector& receive_buffer, mfem::Vector& source_dual,
                                                 mfem::real_t scale ) const
{
  const auto& offsets = plan_->SourceElementOffsets();
  const auto& elements_by_destination = plan_->SourceElementsByDestination();
  const auto& scalar_displacements = plan_->SendScalarDisplacements();
  const mfem::real_t* returned_values = receive_buffer.HostRead();
  mfem::real_t* source_values = source_dual.HostReadWrite();

  for ( int destination = 0; destination < plan_->NumRanks(); ++destination ) {
    int returned_offset = scalar_displacements[destination];
    for ( int source_element : elements_by_destination[destination] ) {
      const int source_begin = offsets[source_element];
      const int block_size = offsets[source_element + 1] - source_begin;
      for ( int i = 0; i < block_size; ++i ) {
        source_values[source_begin + i] += scale * returned_values[returned_offset + i];
      }
      returned_offset += block_size;
    }
  }
}

void ElementExchangeOperator::MultTranspose( const mfem::Vector& destination_dual, mfem::Vector& source_dual ) const
{
  source_dual = 0.0;
  AddMultTranspose( destination_dual, source_dual );
}

void ElementExchangeOperator::AddMultTranspose( const mfem::Vector& destination_dual, mfem::Vector& source_dual,
                                                const mfem::real_t scale ) const
{
  SLIC_ERROR_ROOT_IF( destination_dual.Size() != Height(),
                      "ElementExchangeOperator destination dual size does not match its height." );
  SLIC_ERROR_ROOT_IF( source_dual.Size() != Width(),
                      "ElementExchangeOperator source dual size does not match its width." );

  MPI_Alltoallv( destination_dual.HostRead(), plan_->ReceiveScalarCounts().GetData(),
                 plan_->ReceiveScalarDisplacements().GetData(), MFEM_MPI_REAL_T, receive_buffer_.HostWrite(),
                 plan_->SendScalarCounts().GetData(), plan_->SendScalarDisplacements().GetData(), MFEM_MPI_REAL_T,
                 plan_->GetComm() );
  AddReturnedBlocks( receive_buffer_, source_dual, scale );
}

shared::ParSparseMat ElementExchangeOperator::Assemble() const
{
  const int local_rows = Height();
  const auto& row_starts = plan_->GlobalDestinationOffsets();
  const auto& column_starts = plan_->GlobalSourceOffsets();

  auto* row_offsets = new HYPRE_Int[local_rows + 1];
  auto* global_columns = local_rows > 0 ? new HYPRE_BigInt[local_rows] : nullptr;
  auto* values = local_rows > 0 ? new mfem::real_t[local_rows] : nullptr;
  row_offsets[0] = 0;

  const auto& destination_offsets = plan_->DestinationElementOffsets();
  const auto& source_global_offsets = plan_->DestinationGlobalSourceOffsets();
  for ( int destination_element = 0; destination_element < plan_->NumDestinationElements(); ++destination_element ) {
    const int block_size = destination_offsets[destination_element + 1] - destination_offsets[destination_element];
    const HYPRE_BigInt source_global_begin = source_global_offsets[destination_element];
    for ( int i = 0; i < block_size; ++i ) {
      const int row = destination_offsets[destination_element] + i;
      row_offsets[row + 1] = row + 1;
      global_columns[row] = source_global_begin + i;
      values[row] = 1.0;
    }
  }

  HYPRE_BigInt local_row_partition[2]{ row_starts[plan_->Rank()], row_starts[plan_->Rank() + 1] };
  HYPRE_BigInt local_column_partition[2]{ column_starts[plan_->Rank()], column_starts[plan_->Rank() + 1] };
  return shared::ParSparseMat( plan_->GetComm(), local_rows, row_starts.Last(), column_starts.Last(), row_offsets,
                               global_columns, values, local_row_partition, local_column_partition );
}

}  // namespace redecomp
