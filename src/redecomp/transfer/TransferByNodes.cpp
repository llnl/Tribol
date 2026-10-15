// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "TransferByNodes.hpp"

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "axom/slic.hpp"

#include "shared/infrastructure/Profiling.hpp"

#include "redecomp/RedecompMesh.hpp"
#include "redecomp/common/TypeDefs.hpp"
#include "redecomp/transfer/TransferHelpers.hpp"

namespace redecomp {
namespace {

using transfer_helpers::FieldValuesByRank;

// Both constructors use parent DOF numbering to build redecomp data. Confirm that the supplied parent space belongs to
// the parent mesh before either path uses those DOFs.
void ValidateParentSpaceUsesRedecompParent( const mfem::ParFiniteElementSpace& parent_fes,
                                            const RedecompMesh& redecomp )
{
  SLIC_ERROR_ROOT_IF( parent_fes.GetParMesh() != &redecomp.getParent(),
                      "The parent finite element space must use the parent of the redecomp mesh." );
}

// Both redecomp-to-parent paths use the same cached return map. Require each field to use the spaces that produced that
// map before selecting a redecomp copy.
void ValidateRedecompToParentFieldSpaces( const mfem::FiniteElementSpace* source_fes,
                                          const mfem::ParFiniteElementSpace* destination_fes,
                                          const mfem::FiniteElementSpace& redecomp_fes,
                                          const mfem::ParFiniteElementSpace& parent_fes )
{
  SLIC_ERROR_ROOT_IF( source_fes != &redecomp_fes,
                      "The source field must use the redecomp finite element space in TransferByNodes." );
  SLIC_ERROR_ROOT_IF( destination_fes != &parent_fes,
                      "The destination field must use the parent finite element space in TransferByNodes." );
}

// Debug validation needs every complete redecomp copy, including copies not chosen by the designated transfer. Skip
// ghost-only entries because they may not contain the complete assembled nodal value.
[[maybe_unused]] mfem::Array<int> CollectReturnEligibleVDofs( const mfem::FiniteElementSpace& finite_element_space,
                                                              const EntityIndexByRank& nodes_by_rank, int rank )
{
  const auto& scalar_dofs = nodes_by_rank.first[rank];
  const auto& copy_is_ghost = nodes_by_rank.second[rank];
  mfem::Array<int> return_eligible_dofs;
  return_eligible_dofs.Reserve( scalar_dofs.Size() );
  for ( int position{ 0 }; position < scalar_dofs.Size(); ++position ) {
    if ( !copy_is_ghost[position] ) {
      return_eligible_dofs.Append( scalar_dofs[position] );
    }
  }
  return transfer_helpers::ExpandScalarDofsToVDofs( finite_element_space, return_eligible_dofs );
}

// Cached node maps give each sender and receiver the same scalar-DOF order for every rank. Expand that order only while
// packing values so repeated transfers avoid duplicate element-boundary entries without storing vector-DOF maps.
FieldValuesByRank ExchangeNodeValuesUsingCachedMap( const mfem::Vector& source,
                                                     const mfem::FiniteElementSpace& source_fes,
                                                     const MPIArray<int>& source_dofs_by_rank,
                                                     const RedecompMesh& redecomp )
{
  FieldValuesByRank received_values( &redecomp.getMPIUtility() );
  received_values.SendRecvEach(
      [&source, &source_fes, &source_dofs_by_rank]( int destination_rank ) {
        auto source_vdofs =
            transfer_helpers::ExpandScalarDofsToVDofs( source_fes, source_dofs_by_rank[destination_rank] );
        return transfer_helpers::GatherValuesFromVDofs( source, source_vdofs );
      },
      source.UseDevice() );
  return received_values;
}

// The receiver map mirrors the sender's cached scalar-DOF order. Assign each received rank buffer through the shared
// device-aware vector-DOF helper.
void AssignNodeValuesUsingCachedMap( FieldValuesByRank& received_values, mfem::Vector& destination,
                                     const mfem::FiniteElementSpace& destination_fes,
                                     const MPIArray<int>& destination_dofs_by_rank,
                                     const RedecompMesh& redecomp )
{
  for ( int source_rank{ 0 }; source_rank < redecomp.getMPIUtility().NRanks(); ++source_rank ) {
    auto destination_vdofs =
        transfer_helpers::ExpandScalarDofsToVDofs( destination_fes, destination_dofs_by_rank[source_rank] );
    transfer_helpers::AssignValuesToVDofs( received_values[source_rank], destination_vdofs, destination );
  }
}

// A scalar DOF may occur in several elements. Record it once, and clear its ghost flag when any incident element is a
// complete non-ghost copy.
void AddElementDofsToUniqueNodeList( const mfem::Array<int>& element_dofs, bool element_is_ghost,
                                     mfem::Array<int>& unique_dofs, mfem::Array<bool>& dof_is_ghost,
                                     std::unordered_map<int, int>& dof_positions )
{
  for ( auto element_dof : element_dofs ) {
    const auto [position, inserted] = dof_positions.emplace( element_dof, unique_dofs.Size() );
    if ( inserted ) {
      unique_dofs.Append( element_dof );
      dof_is_ghost.Append( element_is_ghost );
    } else if ( !element_is_ghost ) {
      dof_is_ghost[position->second] = false;
    }
  }
}

// Node lists persist for every field transfer. Release reserve capacity left from element-level estimates once a rank's
// unique list is complete.
void ReleaseUnusedNodeListCapacity( mfem::Array<int>& unique_dofs, mfem::Array<bool>& dof_is_ghost )
{
  unique_dofs.ShrinkToFit();
  dof_is_ghost.ShrinkToFit();
}

}  // namespace

TransferByNodes::TransferByNodes( const mfem::ParFiniteElementSpace& parent_fes,
                                  const mfem::FiniteElementSpace& redecomp_fes )
    : parent_fes_{ &parent_fes },
      redecomp_fes_{ &redecomp_fes },
      redecomp_{ dynamic_cast<const RedecompMesh*>( redecomp_fes.GetMesh() ) }
{
  TRIBOL_MARK_FUNCTION;
  SLIC_ERROR_ROOT_IF( redecomp_ == nullptr,
                      "The Redecomp mesh pointer is null.  Does the redecomp_fes contain an "
                      "underlying Redecomp mesh?" );
  ValidateParentSpaceUsesRedecompParent( parent_fes, *redecomp_ );

  parent_to_redecomp_nodes_ = BuildParentToRedecompNodeList( false );
  redecomp_to_parent_nodes_ = BuildRedecompToParentNodeList();
  BuildDesignatedReturnNodeLists();
}

TransferByNodes::TransferByNodes( const mfem::ParFiniteElementSpace& parent_fes, const RedecompMesh& redecomp )
    : parent_fes_{ &parent_fes }, redecomp_{ &redecomp }
{
  ValidateParentSpaceUsesRedecompParent( parent_fes, redecomp );
}

void TransferByNodes::TransferToSerial( const mfem::ParGridFunction& src, mfem::GridFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  SLIC_ERROR_ROOT_IF( dst.FESpace() != redecomp_fes_,
                      "The FiniteElementSpace of GridFunction dst must match the FiniteElementSpace "
                      "in TransferByNodes." );
  SLIC_ERROR_ROOT_IF( src.ParFESpace() != parent_fes_,
                      "The ParFiniteElementSpace of GridFunction src must match the ParFiniteElementSpace "
                      "in TransferByNodes." );

  auto received_values =
      ExchangeNodeValuesUsingCachedMap( src, *parent_fes_, parent_to_redecomp_nodes_.first, *redecomp_ );
  AssignNodeValuesUsingCachedMap( received_values, dst, *redecomp_fes_, redecomp_to_parent_nodes_.first,
                                  *redecomp_ );
}

void TransferByNodes::TransferToParallel( const mfem::GridFunction& src, mfem::ParGridFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  ValidateRedecompToParentFieldSpaces( src.FESpace(), dst.ParFESpace(), *redecomp_fes_, *parent_fes_ );

  TransferDesignatedValuesToParent( src, dst, false, "primal field" );
}

void TransferByNodes::CollapseReplicatedDualToParallel( const mfem::LinearForm& src, mfem::ParLinearForm& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  ValidateRedecompToParentFieldSpaces( src.FESpace(), dst.ParFESpace(), *redecomp_fes_, *parent_fes_ );

  TransferDesignatedValuesToParent( src, dst, true, "replicated dual" );
}

void TransferByNodes::TransferDesignatedValuesToParent( const mfem::Vector& src, mfem::Vector& dst,
                                                        bool keep_only_parent_owned_dofs,
                                                        const char* field_name [[maybe_unused]] ) const
{
  dst = 0.0;

  auto received_values =
      ExchangeNodeValuesUsingCachedMap( src, *redecomp_fes_, redecomp_dofs_supplying_designated_values_,
                                        *redecomp_ );
  AssignNodeValuesUsingCachedMap( received_values, dst, *parent_fes_,
                                  parent_dofs_receiving_designated_values_, *redecomp_ );

#if defined( TRIBOL_DEBUG )
  CheckReturnEligibleCopies( src, dst, field_name );
#endif

  if ( keep_only_parent_owned_dofs ) {
    transfer_helpers::KeepOnlyOwnedParentVDofs( dst, *parent_fes_ );
  }
}

#if defined( TRIBOL_DEBUG )
void TransferByNodes::CheckReturnEligibleCopies( const mfem::Vector& src,
                                                 const mfem::Vector& selected_parent_values,
                                                 const char* field_name ) const
{
  // The designated-copy transfer above is the production algorithm in every build. Debug builds exchange the
  // additional eligible copies only to verify that selecting one of them is valid.
  auto received_copies = MPIArray<double, mfem::Vector>( &redecomp_->getMPIUtility() );
  received_copies.SendRecvEach(
      [this, &src]( int parent_rank ) {
        auto redecomp_vdofs =
            CollectReturnEligibleVDofs( *redecomp_fes_, redecomp_to_parent_nodes_, parent_rank );
        return transfer_helpers::GatherValuesFromVDofs( src, redecomp_vdofs );
      },
      src.UseDevice() );

  std::vector<int> designated_ranks( parent_fes_->GetVSize(), -1 );
  for ( int redecomp_rank{ 0 }; redecomp_rank < redecomp_->getMPIUtility().NRanks(); ++redecomp_rank ) {
    auto parent_vdofs =
        transfer_helpers::ExpandScalarDofsToVDofs( *parent_fes_,
                                                   parent_dofs_receiving_designated_values_[redecomp_rank] );
    for ( auto encoded_parent_vdof : parent_vdofs ) {
      mfem::real_t sign = 1.0;
      const int parent_vdof = mfem::FiniteElementSpace::DecodeDof( encoded_parent_vdof, sign );
      designated_ranks[parent_vdof] = redecomp_rank;
    }
  }

  const auto* selected_values = selected_parent_values.HostRead();
  for ( int redecomp_rank{ 0 }; redecomp_rank < redecomp_->getMPIUtility().NRanks(); ++redecomp_rank ) {
    auto parent_vdofs =
        CollectReturnEligibleVDofs( *parent_fes_, parent_to_redecomp_nodes_, redecomp_rank );
    const auto* candidate_values = received_copies[redecomp_rank].HostRead();
    for ( int candidate_index{ 0 }; candidate_index < parent_vdofs.Size(); ++candidate_index ) {
      mfem::real_t sign = 1.0;
      const int parent_vdof = mfem::FiniteElementSpace::DecodeDof( parent_vdofs[candidate_index], sign );
      const double selected = selected_values[parent_vdof];
      const double candidate = sign * candidate_values[candidate_index];
      transfer_helpers::VerifyReplicatedValueMatchesSelected( field_name, parent_vdof,
                                                              designated_ranks[parent_vdof], selected,
                                                              redecomp_rank, candidate );
    }
  }
}
#endif

void TransferByNodes::BuildDesignatedReturnNodeLists()
{
  TRIBOL_MARK_FUNCTION;
  const auto& mpi = redecomp_->getMPIUtility();
  parent_dofs_receiving_designated_values_ = MPIArray<int>( &mpi );
  auto designated_positions_by_redecomp_rank = MPIArray<int>( &mpi );

  // A parent DOF can have complete copies on several redecomp ranks. The lowest eligible rank is the stable owner for
  // reverse primal transfer and replicated-dual collapse.
  std::vector<int> designated_redecomp_rank( parent_fes_->GetNDofs(), mpi.NRanks() );
  for ( int redecomp_rank{ 0 }; redecomp_rank < mpi.NRanks(); ++redecomp_rank ) {
    const auto& parent_dofs = parent_to_redecomp_nodes_.first[redecomp_rank];
    const auto& copy_is_ghost = parent_to_redecomp_nodes_.second[redecomp_rank];
    for ( int position{ 0 }; position < parent_dofs.Size(); ++position ) {
      if ( !copy_is_ghost[position] ) {
        designated_redecomp_rank[parent_dofs[position]] =
            std::min( designated_redecomp_rank[parent_dofs[position]], redecomp_rank );
      }
    }
  }

  for ( int redecomp_rank{ 0 }; redecomp_rank < mpi.NRanks(); ++redecomp_rank ) {
    const auto& parent_dofs = parent_to_redecomp_nodes_.first[redecomp_rank];
    for ( int position{ 0 }; position < parent_dofs.Size(); ++position ) {
      const int parent_dof = parent_dofs[position];
      if ( designated_redecomp_rank[parent_dof] == redecomp_rank ) {
        parent_dofs_receiving_designated_values_[redecomp_rank].Append( parent_dof );
        designated_positions_by_redecomp_rank[redecomp_rank].Append( position );
      }
    }
  }

  auto designated_positions_by_parent_rank = MPIArray<int>( &mpi );
  designated_positions_by_parent_rank.SendRecvArrayEach( designated_positions_by_redecomp_rank );
  redecomp_dofs_supplying_designated_values_ = MPIArray<int>( &mpi );
  for ( int parent_rank{ 0 }; parent_rank < mpi.NRanks(); ++parent_rank ) {
    for ( auto position : designated_positions_by_parent_rank[parent_rank] ) {
      SLIC_ERROR_ROOT_IF( position < 0 || position >= redecomp_to_parent_nodes_.first[parent_rank].Size(),
                          "Designated transfer received an invalid redecomp DOF position." );
      redecomp_dofs_supplying_designated_values_[parent_rank].Append(
          redecomp_to_parent_nodes_.first[parent_rank][position] );
    }
  }
}

EntityIndexByRank TransferByNodes::BuildParentToRedecompNodeList( bool use_global_ids )
{
  TRIBOL_MARK_FUNCTION;
  auto parent_dofs_by_redecomp_rank = MPIArray<int>( &redecomp_->getMPIUtility() );
  auto parent_dof_is_ghost = MPIArray<bool>( &redecomp_->getMPIUtility() );
  const auto& parent_elements_by_redecomp_rank = redecomp_->getParentToRedecompElems().first;
  const auto& parent_element_is_ghost = redecomp_->getParentToRedecompElems().second;
  for ( int redecomp_rank{ 0 }; redecomp_rank < redecomp_->getMPIUtility().NRanks(); ++redecomp_rank ) {
    const auto& parent_elements = parent_elements_by_redecomp_rank[redecomp_rank];
    const int element_count = parent_elements.Size();
    if ( element_count > 0 ) {
      const int dofs_per_element = parent_fes_->GetFE( parent_elements[0] )->GetDof();
      parent_dofs_by_redecomp_rank[redecomp_rank].Reserve( element_count * dofs_per_element );
      parent_dof_is_ghost[redecomp_rank].Reserve( element_count * dofs_per_element );
      std::unordered_map<int, int> dof_positions;
      for ( int element_position{ 0 }; element_position < element_count; ++element_position ) {
        const bool element_is_ghost = parent_element_is_ghost[redecomp_rank][element_position];
        mfem::Array<int> element_dofs;
        parent_fes_->GetElementDofs( parent_elements[element_position], element_dofs );
        if ( use_global_ids ) {
          for ( int dof_position{ 0 }; dof_position < element_dofs.Size(); ++dof_position ) {
            element_dofs[dof_position] = parent_fes_->GetGlobalTDofNumber( element_dofs[dof_position] );
          }
        }
        AddElementDofsToUniqueNodeList( element_dofs, element_is_ghost,
                                        parent_dofs_by_redecomp_rank[redecomp_rank],
                                        parent_dof_is_ghost[redecomp_rank], dof_positions );
      }
      ReleaseUnusedNodeListCapacity( parent_dofs_by_redecomp_rank[redecomp_rank],
                                     parent_dof_is_ghost[redecomp_rank] );
    }
  }
  return { std::move( parent_dofs_by_redecomp_rank ), std::move( parent_dof_is_ghost ) };
}

EntityIndexByRank TransferByNodes::BuildRedecompToParentNodeList()
{
  TRIBOL_MARK_FUNCTION;
  auto redecomp_dofs_by_parent_rank = MPIArray<int>( &redecomp_->getMPIUtility() );
  auto redecomp_dof_is_ghost = MPIArray<bool>( &redecomp_->getMPIUtility() );
  for ( int parent_rank{ 0 }; parent_rank < redecomp_->getMPIUtility().NRanks(); ++parent_rank ) {
    const int first_element = redecomp_->getRedecompToParentElemOffsets()[parent_rank];
    const int last_element = redecomp_->getRedecompToParentElemOffsets()[parent_rank + 1];
    const int element_count = last_element - first_element;
    if ( element_count > 0 ) {
      const int dofs_per_element = redecomp_fes_->GetFE( first_element )->GetDof();
      redecomp_dofs_by_parent_rank[parent_rank].Reserve( element_count * dofs_per_element );
      redecomp_dof_is_ghost[parent_rank].Reserve( element_count * dofs_per_element );
      std::unordered_map<int, int> dof_positions;
      int ghost_position = 0;
      const auto& ghost_elements = redecomp_->getRedecompToParentGhostElems()[parent_rank];
      for ( int element{ first_element }; element < last_element; ++element ) {
        bool element_is_ghost = false;
        if ( ghost_position < ghost_elements.Size() && ghost_elements[ghost_position] == element ) {
          ++ghost_position;
          element_is_ghost = true;
        }
        mfem::Array<int> element_dofs;
        redecomp_fes_->GetElementDofs( element, element_dofs );
        AddElementDofsToUniqueNodeList( element_dofs, element_is_ghost,
                                        redecomp_dofs_by_parent_rank[parent_rank],
                                        redecomp_dof_is_ghost[parent_rank], dof_positions );
      }
      ReleaseUnusedNodeListCapacity( redecomp_dofs_by_parent_rank[parent_rank],
                                     redecomp_dof_is_ghost[parent_rank] );
    }
  }
  return { std::move( redecomp_dofs_by_parent_rank ), std::move( redecomp_dof_is_ghost ) };
}

}  // end namespace redecomp
