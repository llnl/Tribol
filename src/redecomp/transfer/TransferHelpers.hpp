// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_TRANSFER_TRANSFERHELPERS_HPP_
#define SRC_REDECOMP_TRANSFER_TRANSFERHELPERS_HPP_

#include "mfem.hpp"

#include "redecomp/utils/MPIArray.hpp"

namespace redecomp {
namespace transfer_helpers {

/**
 * @brief Field values received from each MPI rank
 */
using FieldValuesByRank = MPIArray<double, mfem::Vector>;

/**
 * @brief Expand scalar finite-element DOFs into vector DOFs in MFEM Ordering::byNODES traversal order
 *
 * @param finite_element_space Space that defines vector dimension and ordering
 * @param scalar_dofs Scalar DOFs to expand
 * @return Vector DOFs in scalar-DOF then component order, equivalent to MFEM Ordering::byNODES
 */
mfem::Array<int> ExpandScalarDofsToVDofs( const mfem::FiniteElementSpace& finite_element_space,
                                          const mfem::Array<int>& scalar_dofs );

/**
 * @brief Gather field values without changing the source memory policy
 *
 * @param source Field containing the values
 * @param source_vdofs Vector DOFs to gather
 * @return Gathered values in source_vdofs order
 */
mfem::Vector GatherValuesFromVDofs( const mfem::Vector& source, const mfem::Array<int>& source_vdofs );

/**
 * @brief Assign received values to destination vector DOFs
 *
 * @param received_values Values received through MPI
 * @param destination_vdofs Vector DOFs receiving the values
 * @param destination Destination field
 */
void AssignValuesToVDofs( const mfem::Vector& received_values, const mfem::Array<int>& destination_vdofs,
                          mfem::Vector& destination );

/**
 * @brief Keep only parent VDofs owned by the local rank
 *
 * Replicated dual values are already assembled before transfer. Zeroing non-owned parent copies ensures a later P^T
 * assembly counts each shared response once.
 *
 * @param parent_values Parent local-DOF values to filter
 * @param parent_fes Parent parallel finite-element space
 */
void KeepOnlyOwnedParentVDofs( mfem::Vector& parent_values, const mfem::ParFiniteElementSpace& parent_fes );

/**
 * @brief Fail when a replicated candidate differs from the selected value
 *
 * @param field_name Field name included in the diagnostic
 * @param parent_vdof Parent local vector DOF
 * @param selected_rank Redecomp rank that supplied the selected value
 * @param selected_value Value selected for the parent field
 * @param candidate_rank Redecomp rank that supplied the candidate value
 * @param candidate_value Replicated value being checked
 */
void VerifyReplicatedValueMatchesSelected( const char* field_name, int parent_vdof, int selected_rank,
                                           double selected_value, int candidate_rank, double candidate_value );

}  // namespace transfer_helpers
}  // namespace redecomp

#endif /* SRC_REDECOMP_TRANSFER_TRANSFERHELPERS_HPP_ */
