// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "TransferByNodes.hpp"

#include <unordered_set>

#include "axom/slic.hpp"

#include "shared/infrastructure/Profiling.hpp"

#include "redecomp/RedecompMesh.hpp"
#include "redecomp/common/TypeDefs.hpp"

namespace redecomp {

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
  SLIC_ERROR_ROOT_IF( parent_fes_->GetParMesh() != &redecomp_->getParent(),
                      "The ParMesh associated with both parent_fes and the redecomp mesh must match." );

  // p2r = parent to redecomp
  p2r_nodes_ = P2RNodeList( false );
  // r2p = redecomp to parent
  r2p_nodes_ = R2PNodeList();
}

TransferByNodes::TransferByNodes( const mfem::ParFiniteElementSpace& parent_fes, const RedecompMesh& redecomp )
    : parent_fes_{ &parent_fes }, redecomp_{ &redecomp }
{
  SLIC_ERROR_ROOT_IF( parent_fes_->GetParMesh() != &redecomp_->getParent(),
                      "The ParMesh associated with both parent_fes and redecomp mesh must match." );
}

void TransferByNodes::TransferToSerial( const mfem::ParGridFunction& src, mfem::GridFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  // define transfer-specific data
  auto src_fes = src.ParFESpace();
  auto dst_fes = dst.FESpace();
  // p2r = parent to redecomp
  const auto& src_nodes = p2r_nodes_;
  // r2p = redecomp to parent
  const auto& dst_nodes = r2p_nodes_;

  // checks to make sure src and dst are valid
  SLIC_ERROR_ROOT_IF( dst_fes != redecomp_fes_,
                      "The FiniteElementSpace of GridFunction dst must match the FiniteElementSpace "
                      "in TransferByNodes." );
  SLIC_ERROR_ROOT_IF( src_fes != parent_fes_,
                      "The ParFiniteElementSpace of GridFunction src must match the ParFiniteElementSpace "
                      "in TransferByNodes." );

  // send and receive DOF values from other ranks
  auto dst_dofs = MPIArray<double, mfem::Vector>( &redecomp_->getMPIUtility() );
  dst_dofs.SendRecvEach(
      [src_fes, &src_nodes, &src]( int dst_rank ) {
        mfem::Vector src_dofs;
        src_dofs.UseDevice( src.UseDevice() );
        auto n_vdofs = src_fes->GetVDim();
        auto n_src_dofs = src_nodes.first[dst_rank].Size();
        if ( n_src_dofs > 0 ) {
          mfem::Array<int> all_vdofs;
          all_vdofs.Reserve( n_vdofs * n_src_dofs );
          for ( int d{ 0 }; d < n_vdofs; ++d ) {
            for ( int j{ 0 }; j < n_src_dofs; ++j ) {
              all_vdofs.Append( src_fes->DofToVDof( src_nodes.first[dst_rank][j], d ) );
            }
          }

          all_vdofs.GetMemory().UseDevice( src.UseDevice() );
          src_dofs.SetSize( all_vdofs.Size() );
          src.GetSubVector( all_vdofs, src_dofs );
        }
        return src_dofs;
      },
      src.UseDevice() );

  // map received DOF values to local DOFs
  auto n_vdofs = src_fes->GetVDim();
  auto n_ranks = redecomp_->getMPIUtility().NRanks();
  for ( int i{ 0 }; i < n_ranks; ++i ) {
    if ( dst_nodes.first[i].Size() > 0 ) {
      mfem::Array<int> all_vdofs;
      all_vdofs.Reserve( n_vdofs * dst_nodes.first[i].Size() );
      for ( int d{ 0 }; d < n_vdofs; ++d ) {
        for ( int j{ 0 }; j < dst_nodes.first[i].Size(); ++j ) {
          all_vdofs.Append( dst_fes->DofToVDof( dst_nodes.first[i][j], d ) );
        }
      }
      all_vdofs.GetMemory().UseDevice( dst.UseDevice() );
      // set explicitly in case e.g. src is on device and dst is on host or vice versa
      dst_dofs[i].Read( dst.UseDevice() );
      dst_dofs[i].UseDevice( dst.UseDevice() );
      dst.SetSubVector( all_vdofs, dst_dofs[i] );
    }
  }
}

void TransferByNodes::TransferToParallel( const mfem::GridFunction& src, mfem::ParGridFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  // define transfer specific data
  auto src_fes = src.FESpace();
  auto dst_fes = dst.ParFESpace();
  // r2p = redecomp to parent
  const auto& src_nodes = r2p_nodes_;
  // p2r = parent to redecomp
  const auto& dst_nodes = p2r_nodes_;

  // checks to make sure src and dst are valid
  SLIC_ERROR_ROOT_IF( src.FESpace() != redecomp_fes_,
                      "The FiniteElementSpace of GridFunction src must match the FiniteElementSpace "
                      "in TransferByNodes." );
  SLIC_ERROR_ROOT_IF( dst.ParFESpace() != parent_fes_,
                      "The ParFiniteElementSpace of GridFunction dst must match the ParFiniteElementSpace "
                      "in TransferByNodes." );

  // send and receive non-ghost DOF values from other ranks
  auto dst_dofs = MPIArray<double, mfem::Vector>( &redecomp_->getMPIUtility() );
  dst_dofs.SendRecvEach(
      [src_fes, &src_nodes, &src]( int dst_rank ) {
        mfem::Vector src_dofs;
        src_dofs.UseDevice( src.UseDevice() );
        auto n_vdofs = src_fes->GetVDim();
        auto n_src_dofs = src_nodes.first[dst_rank].Size();
        auto count = 0;
        for ( int j{ 0 }; j < n_src_dofs; ++j ) {
          if ( !src_nodes.second[dst_rank][j] ) {
            ++count;
          }
        }
        if ( count > 0 ) {
          mfem::Array<int> all_vdofs;
          all_vdofs.Reserve( n_vdofs * count );
          for ( int j{ 0 }; j < n_src_dofs; ++j ) {
            if ( !src_nodes.second[dst_rank][j] ) {
              for ( int d{ 0 }; d < n_vdofs; ++d ) {
                all_vdofs.Append( src_fes->DofToVDof( src_nodes.first[dst_rank][j], d ) );
              }
            }
          }

          all_vdofs.GetMemory().UseDevice( src.UseDevice() );
          src_dofs.SetSize( all_vdofs.Size() );
          src.GetSubVector( all_vdofs, src_dofs );
        }
        return src_dofs;
      },
      src.UseDevice() );

  // map received non-ghost DOF values to dst
  auto n_vdofs = src_fes->GetVDim();
  auto n_ranks = redecomp_->getMPIUtility().NRanks();
  for ( int i{ 0 }; i < n_ranks; ++i ) {
    auto count = 0;
    for ( int j{ 0 }; j < dst_nodes.first[i].Size(); ++j ) {
      if ( !dst_nodes.second[i][j] ) {
        ++count;
      }
    }
    if ( count > 0 ) {
      mfem::Array<int> all_vdofs;
      all_vdofs.Reserve( n_vdofs * count );
      for ( int j{ 0 }; j < dst_nodes.first[i].Size(); ++j ) {
        if ( !dst_nodes.second[i][j] ) {
          for ( int d{ 0 }; d < n_vdofs; ++d ) {
            all_vdofs.Append( dst_fes->DofToVDof( dst_nodes.first[i][j], d ) );
          }
        }
      }

      all_vdofs.GetMemory().UseDevice( dst.UseDevice() );
      // set explicitly in case e.g. src is on device and dst is on host or vice versa
      dst_dofs[i].Read( dst.UseDevice() );
      dst_dofs[i].UseDevice( dst.UseDevice() );
      dst.SetSubVector( all_vdofs, dst_dofs[i] );
    }
  }
}

EntityIndexByRank TransferByNodes::P2RNodeList( bool use_global_ids )
{
  TRIBOL_MARK_FUNCTION;
  // p2r = parent to redecomp
  auto p2r_node_idx = MPIArray<int>( &redecomp_->getMPIUtility() );
  auto p2r_node_ghost = MPIArray<bool>( &redecomp_->getMPIUtility() );
  const auto& p2r_elem_idx = redecomp_->getParentToRedecompElems().first;
  const auto& p2r_elem_ghost = redecomp_->getParentToRedecompElems().second;
  auto n_ranks = redecomp_->getMPIUtility().NRanks();
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    auto n_els = p2r_elem_idx[r].Size();
    if ( n_els > 0 ) {
      auto n_dofs = parent_fes_->GetFE( p2r_elem_idx[r][0] )->GetDof();
      p2r_node_idx[r].Reserve( n_els * n_dofs );
      p2r_node_ghost[r].Reserve( n_els * n_dofs );
      auto node_idx_map = std::unordered_map<int, int>();
      auto dof_ct = 0;
      for ( int e{ 0 }; e < p2r_elem_idx[r].Size(); ++e ) {
        auto is_elem_ghost = p2r_elem_ghost[r][e];
        auto elem_dofs = mfem::Array<int>();
        parent_fes_->GetElementDofs( p2r_elem_idx[r][e], elem_dofs );
        for ( auto elem_dof : elem_dofs ) {
          if ( use_global_ids ) {
            elem_dof = parent_fes_->GetGlobalTDofNumber( elem_dof );
          }
          auto node_idx_it = node_idx_map.emplace( elem_dof, dof_ct );
          if ( node_idx_it.second ) {
            ++dof_ct;
            p2r_node_idx[r].push_back( elem_dof );
            p2r_node_ghost[r].push_back( is_elem_ghost );
          } else if ( !is_elem_ghost ) {
            p2r_node_ghost[r][node_idx_it.first->second] = false;
          }
        }
      }
      auto tmp_p2r_node_idx = p2r_node_idx[r];
      auto tmp_p2r_node_ghost = p2r_node_ghost[r];
      std::swap( p2r_node_idx[r], tmp_p2r_node_idx );
      std::swap( p2r_node_ghost[r], tmp_p2r_node_ghost );
    }
  }
  return { std::move( p2r_node_idx ), std::move( p2r_node_ghost ) };
}

EntityIndexByRank TransferByNodes::R2PNodeList()
{
  TRIBOL_MARK_FUNCTION;
  // r2p = redecomp to parent
  auto r2p_node_idx = MPIArray<int>( &redecomp_->getMPIUtility() );
  auto r2p_node_ghost = MPIArray<bool>( &redecomp_->getMPIUtility() );
  auto n_ranks = redecomp_->getMPIUtility().NRanks();
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    auto first_el = redecomp_->getRedecompToParentElemOffsets()[r];
    auto last_el = redecomp_->getRedecompToParentElemOffsets()[r + 1];
    auto n_els = last_el - first_el;
    if ( n_els > 0 ) {
      auto n_dofs = redecomp_fes_->GetFE( first_el )->GetDof();
      r2p_node_idx[r].Reserve( n_els * n_dofs );
      r2p_node_ghost[r].Reserve( n_els * n_dofs );
      auto node_idx_map = std::unordered_map<int, int>();
      auto dof_ct = 0;
      auto ghost_ct = 0;
      for ( int e{ first_el }; e < last_el; ++e ) {
        auto is_elem_ghost = false;
        if ( ghost_ct < redecomp_->getRedecompToParentGhostElems()[r].Size() &&
             redecomp_->getRedecompToParentGhostElems()[r][ghost_ct] == e ) {
          ++ghost_ct;
          is_elem_ghost = true;
        }
        auto elem_dofs = mfem::Array<int>();
        redecomp_fes_->GetElementDofs( e, elem_dofs );
        for ( auto elem_dof : elem_dofs ) {
          auto node_idx_it = node_idx_map.emplace( elem_dof, dof_ct );
          if ( node_idx_it.second ) {
            ++dof_ct;
            r2p_node_idx[r].push_back( elem_dof );
            r2p_node_ghost[r].push_back( is_elem_ghost );
          } else if ( !is_elem_ghost ) {
            r2p_node_ghost[r][node_idx_it.first->second] = false;
          }
        }
      }
      auto tmp_r2p_node_idx = r2p_node_idx[r];
      auto tmp_r2p_node_ghost = r2p_node_ghost[r];
      std::swap( r2p_node_idx[r], tmp_r2p_node_idx );
      std::swap( r2p_node_ghost[r], tmp_r2p_node_ghost );
    }
  }
  return { std::move( r2p_node_idx ), std::move( r2p_node_ghost ) };
}

}  // end namespace redecomp
