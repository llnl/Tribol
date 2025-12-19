// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "RedecompTransfer.hpp"

#include "axom/slic.hpp"

#include "shared/infrastructure/Profiling.hpp"

#include "redecomp/RedecompMesh.hpp"
#include "redecomp/transfer/TransferByNodes.hpp"
#include "redecomp/transfer/TransferByElements.hpp"

namespace redecomp {

RedecompTransfer::RedecompTransfer( std::unique_ptr<const GridFnTransfer> gf_transfer )
    : gf_transfer_{ std::move( gf_transfer ) }
{
}

RedecompTransfer::RedecompTransfer( const mfem::ParFiniteElementSpace& parent_fes,
                                    const mfem::FiniteElementSpace& redecomp_fes )
    : RedecompTransfer( std::make_unique<const TransferByNodes>( parent_fes, redecomp_fes ) )
{
}

RedecompTransfer::RedecompTransfer() : RedecompTransfer( std::make_unique<const TransferByElements>() ) {}

void RedecompTransfer::TransferToSerial( const mfem::ParGridFunction& src, mfem::GridFunction& dst ) const
{
  gf_transfer_->TransferToSerial( src, dst );
}

void RedecompTransfer::TransferToParallel( const mfem::GridFunction& src, mfem::ParGridFunction& dst ) const
{
  gf_transfer_->TransferToParallel( src, dst );
}

void RedecompTransfer::TransferToSerial( const mfem::QuadratureFunction& src, mfem::QuadratureFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  // checks to make sure src and dst are valid
  auto redecomp = dynamic_cast<RedecompMesh*>( dst.GetSpace()->GetMesh() );
  SLIC_ERROR_ROOT_IF( redecomp == nullptr, "The Mesh of QuadratureFunction dst must be a Redecomp mesh." );
  SLIC_ERROR_ROOT_IF( src.GetSpace()->GetMesh() != &redecomp->getParent(),
                      "The Meshes of the specified QuadratureFunctions are not related in a "
                      "Redecomp -> ParMesh relationship." );

  // send and receive quadrature point values from other ranks
  TRIBOL_MARK_BEGIN( "Send and receive values over MPI" );
  auto dst_vals = MPIArray<double>( &redecomp->getMPIUtility() );
  dst_vals.SendRecvEach( [redecomp, &src]( int dest ) {
    auto src_vals = axom::Array<double>();
    const auto& src_elem_idx = redecomp->getParentToRedecompElems().first[dest];
    auto n_els = src_elem_idx.size();
    if ( n_els > 0 ) {
      auto vals = mfem::Vector();
      // guess the size of send_vals based on the size of the first element
      TRIBOL_MARK_BEGIN( "Getting element values" );
      src.GetValues( src_elem_idx[0], vals );
      TRIBOL_MARK_END( "Getting element values" );
      src_vals.reserve( vals.Size() * n_els );
      auto quadpt_ct = 0;
      for ( auto src_elem_id : src_elem_idx ) {
        TRIBOL_MARK_BEGIN( "Getting element values" );
        src.GetValues( src_elem_id, vals );
        TRIBOL_MARK_END( "Getting element values" );
        src_vals.insert( quadpt_ct, vals.Size(), vals.GetData() );
        quadpt_ct += vals.Size();
      }
    }
    return src_vals;
  } );
  TRIBOL_MARK_END( "Send and receive values over MPI" );

  // map received quadrature point values to local quadrature points
  TRIBOL_MARK_BEGIN( "Map received values to local values" );
  auto vals = mfem::Vector();
  vals.UseDevice( true );
  auto n_ranks = redecomp->getMPIUtility().NRanks();
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    auto first_el = redecomp->getRedecompToParentElemOffsets()[r];
    auto last_el = redecomp->getRedecompToParentElemOffsets()[r + 1];
    auto quadpt_ct = 0;
    for ( int e{ first_el }; e < last_el; ++e ) {
      TRIBOL_MARK_BEGIN( "Read element values" );
      dst.GetValues( e, vals );
      const mfem::Vector dof_vals( &dst_vals[r][quadpt_ct], vals.Size() );
      TRIBOL_MARK_END( "Read element values" );
      TRIBOL_MARK_BEGIN( "Copy element values" );
      vals = dof_vals;
      TRIBOL_MARK_END( "Copy element values" );
      quadpt_ct += vals.Size();
      dst.SyncMemory( vals );
    }
  }
  TRIBOL_MARK_END( "Map received values to local values" );
}

void RedecompTransfer::TransferToParallel( const mfem::QuadratureFunction& src, mfem::QuadratureFunction& dst ) const
{
  TRIBOL_MARK_FUNCTION;
  // checks to make sure src and dst are valid
  auto redecomp = dynamic_cast<RedecompMesh*>( src.GetSpace()->GetMesh() );
  SLIC_ERROR_ROOT_IF( redecomp == nullptr, "The Mesh of QuadratureFunction src must be a Redecomp mesh." );
  SLIC_ERROR_ROOT_IF( dst.GetSpace()->GetMesh() != &redecomp->getParent(),
                      "The Meshes of the specified QuadratureFunctions are not related in a "
                      "Redecomp -> ParMesh relationship." );

  // send and receive quadrature point values from other ranks
  auto dst_vals = MPIArray<double>( &redecomp->getMPIUtility() );
  dst_vals.SendRecvEach( [redecomp, &src]( int dest ) {
    auto src_vals = axom::Array<double>();
    auto first_el = redecomp->getRedecompToParentElemOffsets()[dest];
    auto last_el = redecomp->getRedecompToParentElemOffsets()[dest + 1];
    auto n_els = last_el - first_el;
    if ( n_els > 0 ) {
      auto vals = mfem::Vector();
      // guess the size of send_vals based on the size of the first element
      src.GetValues( first_el, vals );
      src_vals.reserve( vals.Size() * n_els );
      auto quadpt_ct = 0;
      auto ghost_ct = 0;
      for ( int e{ first_el }; e < last_el; ++e ) {
        // skip ghost elements
        if ( ghost_ct < redecomp->getRedecompToParentGhostElems()[dest].size() &&
             redecomp->getRedecompToParentGhostElems()[dest][ghost_ct] == e ) {
          ++ghost_ct;
        } else {
          src.GetValues( e, vals );
          src_vals.insert( quadpt_ct, vals.Size(), vals.GetData() );
          quadpt_ct += vals.Size();
        }
      }
    }
    return src_vals;
  } );

  // map received quadrature point values to local quadrature points
  auto vals = mfem::Vector();
  vals.UseDevice( true );
  auto n_ranks = redecomp->getMPIUtility().NRanks();
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    auto quadpt_ct = 0;
    for ( int e{ 0 }; e < redecomp->getParentToRedecompElems().first[r].size(); ++e ) {
      // skip ghost elements
      if ( !redecomp->getParentToRedecompElems().second[r][e] ) {
        dst.GetValues( redecomp->getParentToRedecompElems().first[r][e], vals );
        vals = &dst_vals[r][quadpt_ct];
        quadpt_ct += vals.Size();
        dst.SyncMemory( vals );
      }
    }
  }
}

}  // end namespace redecomp
