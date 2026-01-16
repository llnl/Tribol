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
  auto dst_dofs = MPIArray<double>( &redecomp->getMPIUtility() );
  dst_dofs.SendRecvEach(
      [redecomp, &src]( int dest ) {
        auto src_dofs = MPIArray<double>::ArrayT();
        if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
          src_dofs.GetMemory().UseDevice( src.UseDevice() );
        }
        const auto& src_elem_idx = redecomp->getParentToRedecompElems().first[dest];
        auto n_els = src_elem_idx.Size();
        if ( n_els > 0 ) {
          auto elem_vdofs = mfem::Array<int>();
          auto all_vdofs = mfem::Array<int>();
          // guess the size of src_dofs based on the size of the first element
          src.FESpace()->GetElementVDofs( src_elem_idx[0], elem_vdofs );
          all_vdofs.Reserve( elem_vdofs.Size() * n_els );
          for ( int e{ 0 }; e < n_els; ++e ) {
            src.FESpace()->GetElementVDofs( src_elem_idx[e], elem_vdofs );
            all_vdofs.Append( elem_vdofs );
          }
          if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
            all_vdofs.GetMemory().UseDevice( src.UseDevice() );
            all_vdofs.Read( src.UseDevice() );
          }

          mfem::Vector src_dofs_vec( all_vdofs.Size() );
          if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
            src_dofs_vec.UseDevice( src.UseDevice() );
          }
          src_dofs_vec = 0.0;
          src.GetSubVector( all_vdofs, src_dofs_vec );

          src_dofs.SetSize( src_dofs_vec.Size() );
          if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
            auto d_src = src_dofs.Write( src.UseDevice() );
            auto d_vec = src_dofs_vec.Read( src.UseDevice() );
            mfem::forall_switch( src.UseDevice(), src_dofs.Size(),
                                 [=] MFEM_HOST_DEVICE( int i ) { d_src[i] = d_vec[i]; } );
          } else {
            auto h_src = src_dofs.HostWrite();
            auto h_vec = src_dofs_vec.HostRead();
            std::memcpy( h_src, h_vec, src_dofs.Size() * sizeof( double ) );
          }
        }
        return src_dofs;
      },
      src.UseDevice() );

  // map received DOF values to local DOFs
  auto elem_vdofs = mfem::Array<int>();
  auto n_ranks = redecomp->getMPIUtility().NRanks();
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    auto first_el = redecomp->getRedecompToParentElemOffsets()[r];
    auto last_el = redecomp->getRedecompToParentElemOffsets()[r + 1];
    auto n_els = last_el - first_el;
    if ( n_els > 0 ) {
      auto all_vdofs = mfem::Array<int>();
      // guess the size of all_vdofs based on the size of the first element
      dst.FESpace()->GetElementVDofs( first_el, elem_vdofs );
      all_vdofs.Reserve( elem_vdofs.Size() * n_els );
      for ( int e{ first_el }; e < last_el; ++e ) {
        dst.FESpace()->GetElementVDofs( e, elem_vdofs );
        all_vdofs.Append( elem_vdofs );
      }
      if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
        // in case src is not using device and dst is (or vice versa)
        dst_dofs[r].GetMemory().UseDevice( dst.UseDevice() );
        dst_dofs[r].Read( dst.UseDevice() );
        all_vdofs.GetMemory().UseDevice( dst.UseDevice() );
        all_vdofs.Read( dst.UseDevice() );
      }
      mfem::Vector dof_vals( dst_dofs[r].Size() );
      dof_vals = 0.0;
      if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
        dof_vals.UseDevice( dst.UseDevice() );
        auto d_dest = dof_vals.Write( dst.UseDevice() );
        auto d_src = dst_dofs[r].Read( dst.UseDevice() );
        mfem::forall_switch( dst.UseDevice(), dof_vals.Size(),
                             [=] MFEM_HOST_DEVICE( int i ) { d_dest[i] = d_src[i]; } );
      } else {
        auto h_dest = dof_vals.HostWrite();
        auto h_src = dst_dofs[r].HostRead();
        std::memcpy( h_dest, h_src, dof_vals.Size() * sizeof( double ) );
      }
      dst.SetSubVector( all_vdofs, dof_vals );
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
  dst_dofs.SendRecvEach(
      [redecomp, &src]( int dest ) {
        auto src_dofs = MPIArray<double>::ArrayT();
        if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
          src_dofs.GetMemory().UseDevice( src.UseDevice() );
        }
        auto first_el = redecomp->getRedecompToParentElemOffsets()[dest];
        auto last_el = redecomp->getRedecompToParentElemOffsets()[dest + 1];
        auto n_els = last_el - first_el;
        if ( n_els > 0 ) {
          auto elem_vdofs = mfem::Array<int>();
          auto all_vdofs = mfem::Array<int>();
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
          if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
            all_vdofs.GetMemory().UseDevice( src.UseDevice() );
            all_vdofs.Read( src.UseDevice() );
          }

          mfem::Vector src_dofs_vec( all_vdofs.Size() );
          if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
            src_dofs_vec.UseDevice( src.UseDevice() );
          }
          src_dofs_vec = 0.0;
          src.GetSubVector( all_vdofs, src_dofs_vec );

          src_dofs.SetSize( src_dofs_vec.Size() );
          if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
            src_dofs.GetMemory().UseDevice( src.UseDevice() );
            auto d_src = src_dofs.Write( src.UseDevice() );
            auto d_vec = src_dofs_vec.Read( src.UseDevice() );
            mfem::forall_switch( src.UseDevice(), src_dofs.Size(),
                                 [=] MFEM_HOST_DEVICE( int i ) { d_src[i] = d_vec[i]; } );
          } else {
            auto h_src = src_dofs.HostWrite();
            auto h_vec = src_dofs_vec.HostRead();
            std::memcpy( h_src, h_vec, src_dofs.Size() * sizeof( double ) );
          }
        }
        return src_dofs;
      },
      src.UseDevice() );

  // map received non-ghost DOF values to local DOFs
  auto elem_vdofs = mfem::Array<int>();
  auto n_ranks = redecomp->getMPIUtility().NRanks();
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    auto n_els = redecomp->getParentToRedecompElems().first[r].Size();
    if ( n_els > 0 ) {
      auto all_vdofs = mfem::Array<int>();
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
        if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
          // in case src is not using device and dst is (or vice versa)
          dst_dofs[r].GetMemory().UseDevice( dst.UseDevice() );
          dst_dofs[r].Read( dst.UseDevice() );
          all_vdofs.GetMemory().UseDevice( dst.UseDevice() );
          all_vdofs.Read( dst.UseDevice() );
        }
        mfem::Vector dof_vals( dst_dofs[r].Size() );
        dof_vals = 0.0;
        if constexpr ( std::is_same_v<MPIArray<double>::ArrayT, mfem::Array<double>> ) {
          dof_vals.UseDevice( dst.UseDevice() );
          auto d_dest = dof_vals.Write( dst.UseDevice() );
          auto d_src = dst_dofs[r].Read( dst.UseDevice() );
          mfem::forall_switch( dst.UseDevice(), dof_vals.Size(),
                               [=] MFEM_HOST_DEVICE( int i ) { d_dest[i] = d_src[i]; } );
        } else {
          auto h_dest = dof_vals.HostWrite();
          auto h_src = dst_dofs[r].HostRead();
          std::memcpy( h_dest, h_src, dof_vals.Size() * sizeof( double ) );
        }
        dst.SetSubVector( all_vdofs, dof_vals );
      }
    }
  }
}

}  // end namespace redecomp
