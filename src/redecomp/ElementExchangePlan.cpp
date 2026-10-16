// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/ElementExchangePlan.hpp"

#include <vector>

#include "axom/slic.hpp"

namespace redecomp {

namespace {

mfem::Array<int> BuildDisplacements( const mfem::Array<int>& counts )
{
  mfem::Array<int> displacements( counts.Size() );
  int offset = 0;
  for ( int rank = 0; rank < counts.Size(); ++rank ) {
    displacements[rank] = offset;
    offset += counts[rank];
  }
  return displacements;
}

}  // namespace

ElementExchangePlan::ElementExchangePlan( MPI_Comm comm, mfem::Array<int> source_element_offsets,
                                          std::vector<mfem::Array<int>> source_elements_by_destination )
    : comm_{ comm },
      source_element_offsets_{ std::move( source_element_offsets ) },
      source_elements_by_destination_{ std::move( source_elements_by_destination ) }
{
  MPI_Comm_rank( comm_, &rank_ );
  MPI_Comm_size( comm_, &num_ranks_ );
  ValidateInputs();
  BuildDestinationLayout();
  BuildScalarSchedule();
  BuildGlobalOffsets();
}

void ElementExchangePlan::ValidateInputs() const
{
  SLIC_ERROR_ROOT_IF( source_element_offsets_.Size() == 0,
                      "ElementExchangePlan requires at least the initial source offset." );
  SLIC_ERROR_ROOT_IF( source_element_offsets_[0] != 0, "ElementExchangePlan source offsets must start at zero." );
  for ( int i = 0; i + 1 < source_element_offsets_.Size(); ++i ) {
    SLIC_ERROR_ROOT_IF( source_element_offsets_[i + 1] < source_element_offsets_[i],
                        "ElementExchangePlan source offsets must be monotone." );
  }
  SLIC_ERROR_ROOT_IF( static_cast<int>( source_elements_by_destination_.size() ) != num_ranks_,
                      "ElementExchangePlan requires one source-element list per MPI rank." );

  const int num_source_elements = source_element_offsets_.Size() - 1;
  for ( int destination = 0; destination < num_ranks_; ++destination ) {
    for ( int source_element : source_elements_by_destination_[destination] ) {
      SLIC_ERROR_ROOT_IF( source_element < 0 || source_element >= num_source_elements,
                          "ElementExchangePlan contains an out-of-range source element." );
    }
  }
}

void ElementExchangePlan::BuildDestinationLayout()
{
  mfem::Array<int> send_element_counts( num_ranks_ );
  mfem::Array<int> receive_element_counts( num_ranks_ );
  for ( int destination = 0; destination < num_ranks_; ++destination ) {
    send_element_counts[destination] = source_elements_by_destination_[destination].Size();
  }
  MPI_Alltoall( send_element_counts.GetData(), 1, MPI_INT, receive_element_counts.GetData(), 1, MPI_INT, comm_ );

  const auto send_element_displacements = BuildDisplacements( send_element_counts );
  const auto receive_element_displacements = BuildDisplacements( receive_element_counts );
  const int num_sent_elements = send_element_counts.Sum();
  const int num_received_elements = receive_element_counts.Sum();

  mfem::Array<int> packed_source_elements( num_sent_elements );
  for ( int destination = 0; destination < num_ranks_; ++destination ) {
    const int destination_offset = send_element_displacements[destination];
    for ( int i = 0; i < source_elements_by_destination_[destination].Size(); ++i ) {
      packed_source_elements[destination_offset + i] = source_elements_by_destination_[destination][i];
    }
  }

  destination_source_elements_.SetSize( num_received_elements );
  MPI_Alltoallv( packed_source_elements.GetData(), send_element_counts.GetData(), send_element_displacements.GetData(),
                 MPI_INT, destination_source_elements_.GetData(), receive_element_counts.GetData(),
                 receive_element_displacements.GetData(), MPI_INT, comm_ );

  destination_source_ranks_.SetSize( num_received_elements );
  for ( int source_rank = 0; source_rank < num_ranks_; ++source_rank ) {
    const int begin = receive_element_displacements[source_rank];
    const int end = begin + receive_element_counts[source_rank];
    for ( int i = begin; i < end; ++i ) {
      destination_source_ranks_[i] = source_rank;
    }
  }

  mfem::Array<int> packed_block_sizes( num_sent_elements );
  mfem::Array<int> packed_source_scalar_offsets( num_sent_elements );
  for ( int i = 0; i < num_sent_elements; ++i ) {
    const int source_element = packed_source_elements[i];
    packed_block_sizes[i] = source_element_offsets_[source_element + 1] - source_element_offsets_[source_element];
    packed_source_scalar_offsets[i] = source_element_offsets_[source_element];
  }
  mfem::Array<int> destination_block_sizes( num_received_elements );
  MPI_Alltoallv( packed_block_sizes.GetData(), send_element_counts.GetData(), send_element_displacements.GetData(),
                 MPI_INT, destination_block_sizes.GetData(), receive_element_counts.GetData(),
                 receive_element_displacements.GetData(), MPI_INT, comm_ );

  mfem::Array<int> destination_source_scalar_offsets( num_received_elements );
  MPI_Alltoallv( packed_source_scalar_offsets.GetData(), send_element_counts.GetData(),
                 send_element_displacements.GetData(), MPI_INT, destination_source_scalar_offsets.GetData(),
                 receive_element_counts.GetData(), receive_element_displacements.GetData(), MPI_INT, comm_ );
  destination_global_source_offsets_.SetSize( num_received_elements );
  for ( int i = 0; i < num_received_elements; ++i ) {
    destination_global_source_offsets_[i] = destination_source_scalar_offsets[i];
  }

  destination_element_offsets_.SetSize( num_received_elements + 1 );
  destination_element_offsets_[0] = 0;
  for ( int i = 0; i < num_received_elements; ++i ) {
    destination_element_offsets_[i + 1] = destination_element_offsets_[i] + destination_block_sizes[i];
  }
}

void ElementExchangePlan::BuildScalarSchedule()
{
  send_scalar_counts_.SetSize( num_ranks_ );
  receive_scalar_counts_.SetSize( num_ranks_ );
  for ( int destination = 0; destination < num_ranks_; ++destination ) {
    int scalar_count = 0;
    for ( int source_element : source_elements_by_destination_[destination] ) {
      scalar_count += source_element_offsets_[source_element + 1] - source_element_offsets_[source_element];
    }
    send_scalar_counts_[destination] = scalar_count;
  }
  MPI_Alltoall( send_scalar_counts_.GetData(), 1, MPI_INT, receive_scalar_counts_.GetData(), 1, MPI_INT, comm_ );
  send_scalar_displacements_ = BuildDisplacements( send_scalar_counts_ );
  receive_scalar_displacements_ = BuildDisplacements( receive_scalar_counts_ );
}

void ElementExchangePlan::BuildGlobalOffsets()
{
  mfem::Array<int> source_sizes( num_ranks_ );
  mfem::Array<int> destination_sizes( num_ranks_ );
  const int source_size = SourceDataSize();
  const int destination_size = DestinationDataSize();
  MPI_Allgather( &source_size, 1, MPI_INT, source_sizes.GetData(), 1, MPI_INT, comm_ );
  MPI_Allgather( &destination_size, 1, MPI_INT, destination_sizes.GetData(), 1, MPI_INT, comm_ );

  global_source_offsets_.SetSize( num_ranks_ + 1 );
  global_destination_offsets_.SetSize( num_ranks_ + 1 );
  global_source_offsets_[0] = 0;
  global_destination_offsets_[0] = 0;
  for ( int rank = 0; rank < num_ranks_; ++rank ) {
    global_source_offsets_[rank + 1] = global_source_offsets_[rank] + source_sizes[rank];
    global_destination_offsets_[rank + 1] = global_destination_offsets_[rank] + destination_sizes[rank];
  }
  for ( int destination_element = 0; destination_element < destination_global_source_offsets_.Size();
        ++destination_element ) {
    destination_global_source_offsets_[destination_element] +=
        global_source_offsets_[destination_source_ranks_[destination_element]];
  }
}

int ElementExchangePlan::NumSentElements() const
{
  int count = 0;
  for ( const auto& destination_elements : source_elements_by_destination_ ) {
    count += destination_elements.Size();
  }
  return count;
}

}  // namespace redecomp
