// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "TransferByElements.hpp"

#include <cstring>

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
  MPIArray<double, mfem::Vector> dst_dofs( &redecomp->getMPIUtility() );
  dst_dofs.SendRecvEach(
      [redecomp, &src]( int dest ) {
        mfem::Vector src_dofs;
        src_dofs.UseDevice( src.UseDevice() );
        const auto& src_elem_idx = redecomp->getParentToRedecompElems().first[dest];
        auto n_els = src_elem_idx.Size();
        if ( n_els > 0 ) {
          mfem::Array<int> elem_vdofs;
          mfem::Array<int> all_vdofs;
          // guess the size of src_dofs based on the size of the first element
          src.FESpace()->GetElementVDofs( src_elem_idx[0], elem_vdofs );
          all_vdofs.Reserve( elem_vdofs.Size() * n_els );
          for ( int e{ 0 }; e < n_els; ++e ) {
            src.FESpace()->GetElementVDofs( src_elem_idx[e], elem_vdofs );
            all_vdofs.Append( elem_vdofs );
          }

          src_dofs.SetSize( all_vdofs.Size() );
          all_vdofs.GetMemory().UseDevice( src.UseDevice() );
          src_dofs.UseDevice( src.UseDevice() );
          src.GetSubVector( all_vdofs, src_dofs );
        }
        return src_dofs;
      },
      src.UseDevice() );

  // map received DOF values to local DOFs
  mfem::Array<int> elem_vdofs;
  auto n_ranks = redecomp->getMPIUtility().NRanks();
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    auto first_el = redecomp->getRedecompToParentElemOffsets()[r];
    auto last_el = redecomp->getRedecompToParentElemOffsets()[r + 1];
    auto n_els = last_el - first_el;
    if ( n_els > 0 ) {
      mfem::Array<int> all_vdofs;
      // guess the size of all_vdofs based on the size of the first element
      dst.FESpace()->GetElementVDofs( first_el, elem_vdofs );
      all_vdofs.Reserve( elem_vdofs.Size() * n_els );
      for ( int e{ first_el }; e < last_el; ++e ) {
        dst.FESpace()->GetElementVDofs( e, elem_vdofs );
        all_vdofs.Append( elem_vdofs );
      }
      all_vdofs.GetMemory().UseDevice( dst.UseDevice() );
      // set explicitly in case e.g. src is on device and dst is on host or vice versa
      dst_dofs[r].Read( dst.UseDevice() );
      dst_dofs[r].UseDevice( dst.UseDevice() );
      dst.SetSubVector( all_vdofs, dst_dofs[r] );
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
  MPIArray<double, mfem::Vector> dst_dofs( &redecomp->getMPIUtility() );
  dst_dofs.SendRecvEach(
      [redecomp, &src]( int dest ) {
        mfem::Vector src_dofs;
        src_dofs.UseDevice( src.UseDevice() );
        auto first_el = redecomp->getRedecompToParentElemOffsets()[dest];
        auto last_el = redecomp->getRedecompToParentElemOffsets()[dest + 1];
        auto n_els = last_el - first_el;
        if ( n_els > 0 ) {
          mfem::Array<int> elem_vdofs;
          mfem::Array<int> all_vdofs;
          // guess the size of src_dofs based on the size of the first element
          src.FESpace()->GetElementVDofs( first_el, elem_vdofs );
          all_vdofs.Reserve( elem_vdofs.Size() * n_els );
          auto ghost_ct = 0;
          for ( int e{ first_el }; e < last_el; ++e ) {
            // skip ghost elements
            if ( ghost_ct < redecomp->getRedecompToParentGhostElems()[dest].Size() &&
                 redecomp->getRedecompToParentGhostElems()[dest][ghost_ct] == e ) {
              ++ghost_ct;
            } else {
              src.FESpace()->GetElementVDofs( e, elem_vdofs );
              all_vdofs.Append( elem_vdofs );
            }
          }
          src_dofs.SetSize( all_vdofs.Size() );
          all_vdofs.GetMemory().UseDevice( src.UseDevice() );
          src_dofs.UseDevice( src.UseDevice() );
          src.GetSubVector( all_vdofs, src_dofs );
        }
        return src_dofs;
      },
      src.UseDevice() );

  // map received non-ghost DOF values to local DOFs
  mfem::Array<int> elem_vdofs;
  auto n_ranks = redecomp->getMPIUtility().NRanks();
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    auto n_els = redecomp->getParentToRedecompElems().first[r].Size();
    if ( n_els > 0 ) {
      mfem::Array<int> all_vdofs;
      // guess the size of all_vdofs based on the size of the first element
      dst.FESpace()->GetElementVDofs( redecomp->getParentToRedecompElems().first[r][0], elem_vdofs );
      all_vdofs.Reserve( n_els * elem_vdofs.Size() );
      for ( int e{ 0 }; e < n_els; ++e ) {
        // skip ghost elements
        if ( !redecomp->getParentToRedecompElems().second[r][e] ) {
          dst.FESpace()->GetElementVDofs( redecomp->getParentToRedecompElems().first[r][e], elem_vdofs );
          all_vdofs.Append( elem_vdofs );
        }
      }
      if ( all_vdofs.Size() > 0 ) {
        all_vdofs.GetMemory().UseDevice( dst.UseDevice() );
        // set explicitly in case e.g. src is on device and dst is on host or vice versa
        dst_dofs[r].Read( dst.UseDevice() );
        dst_dofs[r].UseDevice( dst.UseDevice() );
      }
      dst.SetSubVector( all_vdofs, dst_dofs[r] );
    }
  }
}

}  // end namespace redecomp
