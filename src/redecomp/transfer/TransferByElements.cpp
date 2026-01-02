// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "TransferByElements.hpp"

#include "axom/slic.hpp"

#include "shared/infrastructure/Profiling.hpp"

#include "redecomp/RedecompMesh.hpp"

namespace redecomp {

void TransferByElements::TransferToSerial( const mfem::ParGridFunction& src, mfem::GridFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  // checks to make sure src and dst are valid
  auto redecomp = dynamic_cast<RedecompMesh*>( dst.FESpace()->GetMesh() );
  SLIC_ERROR_ROOT_IF( redecomp == nullptr, "The Mesh of GridFunction dst must be a Redecomp mesh." );
  SLIC_ERROR_ROOT_IF( src.ParFESpace()->GetParMesh() != &redecomp->getParent(),
                      "The Meshes of the specified GridFunctions are not related in a "
                      "Redecomp -> ParMesh relationship." );
  SLIC_ERROR_ROOT_IF( strcmp( src.FESpace()->FEColl()->Name(), dst.FESpace()->FEColl()->Name() ) != 0,
                      "The FiniteElementCollections of the specified GridFunctions are not "
                      "the same." );
  SLIC_ERROR_ROOT_IF( src.FESpace()->GetVDim() != dst.FESpace()->GetVDim(),
                      "The vdim of the FiniteElementSpaces of the specified GridFunctions are "
                      "not the same." );

  // send and receive DOF values from other ranks
  auto dst_dofs = MPIArray<double>( &redecomp->getMPIUtility() );
  dst_dofs.SendRecvEach( [redecomp, &src]( int dest ) {
    auto src_dofs = axom::Array<double>();
    const auto& src_elem_idx = redecomp->getParentToRedecompElems().first[dest];
    auto n_els = src_elem_idx.size();
    if ( n_els > 0 ) {
      auto elem_vdofs = mfem::Array<int>();
      auto dof_vals = mfem::Vector();
      // guess the size of src_dofs based on the size of the first element
      src.FESpace()->GetElementVDofs( src_elem_idx[0], elem_vdofs );
      src_dofs.reserve( elem_vdofs.Size() * n_els );
      auto vdof_ct = 0;
      for ( int e{ 0 }; e < n_els; ++e ) {
        src.FESpace()->GetElementVDofs( src_elem_idx[e], elem_vdofs );
        src.GetSubVector( elem_vdofs, dof_vals );
        src_dofs.insert( vdof_ct, dof_vals.Size(), dof_vals.GetData() );
        vdof_ct += dof_vals.Size();
      }
    }
    return src_dofs;
  } );

  // map received DOF values to local DOFs
  auto elem_vdofs = mfem::Array<int>();
  auto n_ranks = redecomp->getMPIUtility().NRanks();
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    auto vdof_ct = 0;
    auto first_el = redecomp->getRedecompToParentElemOffsets()[r];
    auto last_el = redecomp->getRedecompToParentElemOffsets()[r + 1];
    for ( int e{ first_el }; e < last_el; ++e ) {
      dst.FESpace()->GetElementVDofs( e, elem_vdofs );
      auto dof_vals = mfem::Vector( &dst_dofs[r][vdof_ct], elem_vdofs.Size() );
      dst.SetSubVector( elem_vdofs, dof_vals );
      vdof_ct += elem_vdofs.Size();
    }
  }
}

void TransferByElements::TransferToParallel( const mfem::GridFunction& src, mfem::ParGridFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  // checks to make sure src and dst are valid
  auto redecomp = dynamic_cast<RedecompMesh*>( src.FESpace()->GetMesh() );
  SLIC_ERROR_ROOT_IF( redecomp == nullptr, "The Mesh of GridFunction dst must be a Redecomp mesh." );
  SLIC_ERROR_ROOT_IF( dst.ParFESpace()->GetParMesh() != &redecomp->getParent(),
                      "The Meshes of the specified GridFunctions are not related in a"
                      "Redecomp -> ParMesh relationship." );
  SLIC_ERROR_ROOT_IF( strcmp( dst.FESpace()->FEColl()->Name(), src.FESpace()->FEColl()->Name() ) != 0,
                      "The FiniteElementCollections of the specified GridFunctions are not"
                      "the same." );
  SLIC_ERROR_ROOT_IF( dst.FESpace()->GetVDim() != src.FESpace()->GetVDim(),
                      "The vdim of the FiniteElementSpaces of the specified GridFunctions are"
                      "not the same." );

  // send and receive non-ghost DOF values from other ranks
  auto dst_dofs = MPIArray<double>( &redecomp->getMPIUtility() );
  dst_dofs.SendRecvEach( [redecomp, &src]( int dest ) {
    auto src_dofs = axom::Array<double>();
    auto first_el = redecomp->getRedecompToParentElemOffsets()[dest];
    auto last_el = redecomp->getRedecompToParentElemOffsets()[dest + 1];
    auto n_els = last_el - first_el;
    if ( n_els > 0 ) {
      auto elem_vdofs = mfem::Array<int>();
      auto dof_vals = mfem::Vector();
      // guess the size of src_dofs based on the size of the first element
      src.FESpace()->GetElementVDofs( first_el, elem_vdofs );
      src_dofs.reserve( elem_vdofs.Size() * n_els );
      auto vdof_ct = 0;
      auto ghost_ct = 0;
      for ( int e{ first_el }; e < last_el; ++e ) {
        // skip ghost elements
        if ( ghost_ct < redecomp->getRedecompToParentGhostElems()[dest].size() &&
             redecomp->getRedecompToParentGhostElems()[dest][ghost_ct] == e ) {
          ++ghost_ct;
        } else {
          src.FESpace()->GetElementVDofs( e, elem_vdofs );
          src.GetSubVector( elem_vdofs, dof_vals );
          src_dofs.insert( vdof_ct, dof_vals.Size(), dof_vals.GetData() );
          vdof_ct += dof_vals.Size();
        }
      }
    }
    return src_dofs;
  } );

  // map received non-ghost DOF values to local DOFs
  auto elem_vdofs = mfem::Array<int>();
  auto n_ranks = redecomp->getMPIUtility().NRanks();
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    auto vdof_ct = 0;
    for ( int e{ 0 }; e < redecomp->getParentToRedecompElems().first[r].size(); ++e ) {
      // skip ghost elements
      if ( !redecomp->getParentToRedecompElems().second[r][e] ) {
        dst.FESpace()->GetElementVDofs( redecomp->getParentToRedecompElems().first[r][e], elem_vdofs );
        auto dof_vals = mfem::Vector( &dst_dofs[r][vdof_ct], elem_vdofs.Size() );
        dst.SetSubVector( elem_vdofs, dof_vals );
        vdof_ct += elem_vdofs.Size();
      }
    }
  }
}

}  // end namespace redecomp
