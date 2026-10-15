// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "TransferByElements.hpp"

#include <cstring>
#include <vector>

#include "axom/slic.hpp"

#include "shared/infrastructure/Profiling.hpp"

#include "redecomp/RedecompMesh.hpp"
#include "redecomp/transfer/TransferHelpers.hpp"

namespace redecomp {
namespace {

using transfer_helpers::FieldValuesByRank;

// Element-buffer transfers rely on matching finite-element layouts. Reject unrelated spaces before using element
// positions from RedecompMesh as a communication contract.
const RedecompMesh& ValidateRelatedElementSpaces( const mfem::ParFiniteElementSpace& parent_fes,
                                                  const mfem::FiniteElementSpace& redecomp_fes )
{
  const auto* redecomp = dynamic_cast<const RedecompMesh*>( redecomp_fes.GetMesh() );
  SLIC_ERROR_ROOT_IF( redecomp == nullptr, "The redecomp finite element space must use a RedecompMesh." );
  SLIC_ERROR_ROOT_IF( parent_fes.GetParMesh() != &redecomp->getParent(),
                      "The parent and redecomp finite element spaces must use related meshes." );
  SLIC_ERROR_ROOT_IF( strcmp( parent_fes.FEColl()->Name(), redecomp_fes.FEColl()->Name() ) != 0,
                      "The parent and redecomp finite element collections must match." );
  SLIC_ERROR_ROOT_IF( parent_fes.GetVDim() != redecomp_fes.GetVDim(),
                      "The parent and redecomp finite element spaces must have the same vector dimension." );
  return *redecomp;
}

// Parent element lists contain every copy needed on a redecomp rank, including ghost-layer copies. Preserve list order
// so the receiving redecomp element range can reconstruct the same vector-DOF layout.
mfem::Array<int> CollectParentVDofsToSendToRedecompRank( const mfem::ParFiniteElementSpace& parent_fes,
                                                        const RedecompMesh& redecomp, int redecomp_rank )
{
  const auto& parent_elements = redecomp.getParentToRedecompElems().first[redecomp_rank];
  mfem::Array<int> parent_vdofs;
  if ( parent_elements.Size() == 0 ) {
    return parent_vdofs;
  }

  mfem::Array<int> element_vdofs;
  parent_fes.GetElementVDofs( parent_elements[0], element_vdofs );
  parent_vdofs.Reserve( element_vdofs.Size() * parent_elements.Size() );
  for ( auto parent_element : parent_elements ) {
    parent_fes.GetElementVDofs( parent_element, element_vdofs );
    parent_vdofs.Append( element_vdofs );
  }
  return parent_vdofs;
}

// RedecompMesh stores elements received from each parent rank in one contiguous range. Traversing that range in mesh
// order recreates the sender's parent-element list order.
mfem::Array<int> CollectRedecompVDofsReceivedFromParentRank( const mfem::FiniteElementSpace& redecomp_fes,
                                                            const RedecompMesh& redecomp, int parent_rank )
{
  const int first_element = redecomp.getRedecompToParentElemOffsets()[parent_rank];
  const int last_element = redecomp.getRedecompToParentElemOffsets()[parent_rank + 1];
  mfem::Array<int> redecomp_vdofs;
  if ( first_element == last_element ) {
    return redecomp_vdofs;
  }

  mfem::Array<int> element_vdofs;
  redecomp_fes.GetElementVDofs( first_element, element_vdofs );
  redecomp_vdofs.Reserve( element_vdofs.Size() * ( last_element - first_element ) );
  for ( int element{ first_element }; element < last_element; ++element ) {
    redecomp_fes.GetElementVDofs( element, element_vdofs );
    redecomp_vdofs.Append( element_vdofs );
  }
  return redecomp_vdofs;
}

// Only non-ghost elements can return assigned values to their parent rank. Preserve rank-grouped element order so each
// sent vector-DOF entry matches the corresponding parent-element entry on the receiving rank.
mfem::Array<int> CollectReturnEligibleRedecompVDofsForParentRank( const mfem::FiniteElementSpace& redecomp_fes,
                                                                 const RedecompMesh& redecomp, int parent_rank )
{
  const int first_element = redecomp.getRedecompToParentElemOffsets()[parent_rank];
  const int last_element = redecomp.getRedecompToParentElemOffsets()[parent_rank + 1];
  mfem::Array<int> redecomp_vdofs;
  if ( first_element == last_element ) {
    return redecomp_vdofs;
  }

  mfem::Array<int> element_vdofs;
  redecomp_fes.GetElementVDofs( first_element, element_vdofs );
  redecomp_vdofs.Reserve( element_vdofs.Size() * ( last_element - first_element ) );
  const auto& ghost_elements = redecomp.getRedecompToParentGhostElems()[parent_rank];
  int ghost_position = 0;
  for ( int element{ first_element }; element < last_element; ++element ) {
    const bool element_is_ghost = ghost_position < ghost_elements.Size() && ghost_elements[ghost_position] == element;
    if ( element_is_ghost ) {
      ++ghost_position;
    } else {
      redecomp_fes.GetElementVDofs( element, element_vdofs );
      redecomp_vdofs.Append( element_vdofs );
    }
  }
  return redecomp_vdofs;
}

// The parent-to-redecomp element lists mirror the sender's rank-grouped element order. Skipping entries marked as
// ghosts keeps received return values aligned with these parent vector DOFs.
mfem::Array<int> CollectParentVDofsReceivingReturnedValues( const mfem::ParFiniteElementSpace& parent_fes,
                                                            const RedecompMesh& redecomp, int redecomp_rank )
{
  const auto& parent_elements = redecomp.getParentToRedecompElems().first[redecomp_rank];
  const auto& element_is_ghost = redecomp.getParentToRedecompElems().second[redecomp_rank];
  mfem::Array<int> parent_vdofs;
  if ( parent_elements.Size() == 0 ) {
    return parent_vdofs;
  }

  mfem::Array<int> element_vdofs;
  parent_fes.GetElementVDofs( parent_elements[0], element_vdofs );
  parent_vdofs.Reserve( element_vdofs.Size() * parent_elements.Size() );
  for ( int element_position{ 0 }; element_position < parent_elements.Size(); ++element_position ) {
    if ( !element_is_ghost[element_position] ) {
      parent_fes.GetElementVDofs( parent_elements[element_position], element_vdofs );
      parent_vdofs.Append( element_vdofs );
    }
  }
  return parent_vdofs;
}

// Parent-to-redecomp replication sends every element copy requested by each redecomp rank. Element-list order is the
// communication contract, so no persistent node map is required.
FieldValuesByRank ExchangeParentElementValuesToRedecomp( const mfem::ParGridFunction& source,
                                                         const RedecompMesh& redecomp )
{
  FieldValuesByRank received_values( &redecomp.getMPIUtility() );
  received_values.SendRecvEach(
      [&source, &redecomp]( int redecomp_rank ) {
        auto parent_vdofs =
            CollectParentVDofsToSendToRedecompRank( *source.ParFESpace(), redecomp, redecomp_rank );
        return transfer_helpers::GatherValuesFromVDofs( source, parent_vdofs );
      },
      source.UseDevice() );
  return received_values;
}

// Values received from a parent rank cover that rank's contiguous redecomp element range in the same order used by the
// sender. Assigning the full range also fills ghost-layer copies required by contact search.
void AssignParentElementValuesOnRedecomp( FieldValuesByRank& received_values, mfem::GridFunction& destination,
                                          const RedecompMesh& redecomp )
{
  for ( int parent_rank{ 0 }; parent_rank < redecomp.getMPIUtility().NRanks(); ++parent_rank ) {
    auto redecomp_vdofs =
        CollectRedecompVDofsReceivedFromParentRank( *destination.FESpace(), redecomp, parent_rank );
    transfer_helpers::AssignValuesToVDofs( received_values[parent_rank], redecomp_vdofs, destination );
  }
}

// Return transfers send only complete non-ghost element copies. The paired element lists give sender and receiver the
// same vector-DOF order without constructing a cached node map.
FieldValuesByRank ExchangeReturnEligibleElementValues( const mfem::Vector& source,
                                                        const mfem::FiniteElementSpace& redecomp_fes,
                                                        const RedecompMesh& redecomp )
{
  FieldValuesByRank received_values( &redecomp.getMPIUtility() );
  received_values.SendRecvEach(
      [&source, &redecomp_fes, &redecomp]( int parent_rank ) {
        auto redecomp_vdofs =
            CollectReturnEligibleRedecompVDofsForParentRank( redecomp_fes, redecomp, parent_rank );
        return transfer_helpers::GatherValuesFromVDofs( source, redecomp_vdofs );
      },
      source.UseDevice() );
  return received_values;
}

#if defined( TRIBOL_DEBUG )
// Assigning one redecomp copy is valid only when all return-eligible copies agree. Track the first value and rank seen
// for each parent vector DOF so a conflicting copy identifies both sources.
void VerifyReturnedElementValuesAgree( const char* field_name, int redecomp_rank,
                                       const mfem::Array<int>& parent_vdofs, const mfem::Vector& candidate_values,
                                       std::vector<double>& selected_values, std::vector<int>& selected_ranks )
{
  const auto* candidate_data = candidate_values.HostRead();
  for ( int entry{ 0 }; entry < parent_vdofs.Size(); ++entry ) {
    mfem::real_t sign = 1.0;
    const int parent_vdof = mfem::FiniteElementSpace::DecodeDof( parent_vdofs[entry], sign );
    const double candidate = sign * candidate_data[entry];
    if ( selected_ranks[parent_vdof] >= 0 ) {
      transfer_helpers::VerifyReplicatedValueMatchesSelected( field_name, parent_vdof,
                                                              selected_ranks[parent_vdof],
                                                              selected_values[parent_vdof], redecomp_rank, candidate );
    } else {
      selected_values[parent_vdof] = candidate;
      selected_ranks[parent_vdof] = redecomp_rank;
    }
  }
}
#endif

// Parent vector DOFs may occur in several returned elements. Assign them directly because replicated values must agree;
// debug builds verify that requirement before any later rank overwrites an earlier identical value.
void AssignReturnedElementValuesToParent( FieldValuesByRank& received_values, mfem::Vector& destination,
                                          const mfem::ParFiniteElementSpace& parent_fes,
                                          const RedecompMesh& redecomp,
                                          const char* field_name [[maybe_unused]] )
{
#if defined( TRIBOL_DEBUG )
  std::vector<double> selected_values( destination.Size(), 0.0 );
  std::vector<int> selected_ranks( destination.Size(), -1 );
#endif
  for ( int redecomp_rank{ 0 }; redecomp_rank < redecomp.getMPIUtility().NRanks(); ++redecomp_rank ) {
    auto parent_vdofs = CollectParentVDofsReceivingReturnedValues( parent_fes, redecomp, redecomp_rank );
#if defined( TRIBOL_DEBUG )
    VerifyReturnedElementValuesAgree( field_name, redecomp_rank, parent_vdofs, received_values[redecomp_rank],
                                      selected_values, selected_ranks );
#endif
    transfer_helpers::AssignValuesToVDofs( received_values[redecomp_rank], parent_vdofs, destination );
  }
}

}  // namespace

void TransferByElements::TransferToSerial( const mfem::ParGridFunction& src, mfem::GridFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  const auto& redecomp = ValidateRelatedElementSpaces( *src.ParFESpace(), *dst.FESpace() );
  auto received_values = ExchangeParentElementValuesToRedecomp( src, redecomp );
  AssignParentElementValuesOnRedecomp( received_values, dst, redecomp );
}

void TransferByElements::TransferToParallel( const mfem::GridFunction& src, mfem::ParGridFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  const auto& redecomp = ValidateRelatedElementSpaces( *dst.ParFESpace(), *src.FESpace() );
  auto received_values = ExchangeReturnEligibleElementValues( src, *src.FESpace(), redecomp );
  AssignReturnedElementValuesToParent( received_values, dst, *dst.ParFESpace(), redecomp, "primal field" );
}

void TransferByElements::CollapseReplicatedDualToParallel( const mfem::LinearForm& src, mfem::ParLinearForm& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  const auto& redecomp = ValidateRelatedElementSpaces( *dst.ParFESpace(), *src.FESpace() );
  dst = 0.0;
  auto received_values = ExchangeReturnEligibleElementValues( src, *src.FESpace(), redecomp );
  AssignReturnedElementValuesToParent( received_values, dst, *dst.ParFESpace(), redecomp, "replicated dual" );
  transfer_helpers::KeepOnlyOwnedParentVDofs( dst, *dst.ParFESpace() );
}

}  // end namespace redecomp
