// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_ELEMENTEXCHANGEPLAN_HPP_
#define SRC_REDECOMP_ELEMENTEXCHANGEPLAN_HPP_

#include <mpi.h>

#include <vector>

#include "mfem.hpp"

namespace redecomp {

/**
 * @brief Immutable communication schedule for moving variable-size element blocks
 *
 * Each rank describes which of its local source elements are required by every
 * destination rank. The plan exchanges element provenance and block sizes once,
 * then reuses the resulting scalar communication schedule for every field update.
 */
class ElementExchangePlan {
 public:
  /**
   * @brief Build an element exchange schedule
   *
   * @param comm Communicator shared by source and destination distributions
   * @param source_element_offsets Offsets into the source element-block vector
   * @param source_elements_by_destination Local source element IDs requested by each destination rank
   */
  ElementExchangePlan( MPI_Comm comm, mfem::Array<int> source_element_offsets,
                       std::vector<mfem::Array<int>> source_elements_by_destination );

  MPI_Comm GetComm() const { return comm_; }
  int Rank() const { return rank_; }
  int NumRanks() const { return num_ranks_; }

  const mfem::Array<int>& SourceElementOffsets() const { return source_element_offsets_; }
  const std::vector<mfem::Array<int>>& SourceElementsByDestination() const { return source_elements_by_destination_; }
  const mfem::Array<int>& DestinationSourceRanks() const { return destination_source_ranks_; }
  const mfem::Array<int>& DestinationSourceElements() const { return destination_source_elements_; }
  const mfem::Array<int>& DestinationElementOffsets() const { return destination_element_offsets_; }
  const mfem::Array<HYPRE_BigInt>& DestinationGlobalSourceOffsets() const { return destination_global_source_offsets_; }

  int SourceDataSize() const { return source_element_offsets_.Last(); }
  int DestinationDataSize() const { return destination_element_offsets_.Last(); }
  int NumDestinationElements() const { return destination_source_elements_.Size(); }
  int NumSentElements() const;

  const mfem::Array<int>& SendScalarCounts() const { return send_scalar_counts_; }
  const mfem::Array<int>& SendScalarDisplacements() const { return send_scalar_displacements_; }
  const mfem::Array<int>& ReceiveScalarCounts() const { return receive_scalar_counts_; }
  const mfem::Array<int>& ReceiveScalarDisplacements() const { return receive_scalar_displacements_; }
  const mfem::Array<HYPRE_BigInt>& GlobalSourceOffsets() const { return global_source_offsets_; }
  const mfem::Array<HYPRE_BigInt>& GlobalDestinationOffsets() const { return global_destination_offsets_; }

 private:
  void ValidateInputs() const;
  void BuildDestinationLayout();
  void BuildScalarSchedule();
  void BuildGlobalOffsets();

  MPI_Comm comm_;
  int rank_{ 0 };
  int num_ranks_{ 1 };
  mfem::Array<int> source_element_offsets_;
  std::vector<mfem::Array<int>> source_elements_by_destination_;
  mfem::Array<int> destination_source_ranks_;
  mfem::Array<int> destination_source_elements_;
  mfem::Array<int> destination_element_offsets_;
  mfem::Array<HYPRE_BigInt> destination_global_source_offsets_;
  mfem::Array<int> send_scalar_counts_;
  mfem::Array<int> send_scalar_displacements_;
  mfem::Array<int> receive_scalar_counts_;
  mfem::Array<int> receive_scalar_displacements_;
  mfem::Array<HYPRE_BigInt> global_source_offsets_;
  mfem::Array<HYPRE_BigInt> global_destination_offsets_;
};

}  // namespace redecomp

#endif  // SRC_REDECOMP_ELEMENTEXCHANGEPLAN_HPP_
