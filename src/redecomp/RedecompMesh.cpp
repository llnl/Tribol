// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "RedecompMesh.hpp"

#include "axom/slic.hpp"

#include "shared/infrastructure/Profiling.hpp"

#include "redecomp/RedecompTransfer.hpp"
#include "redecomp/transfer/TransferByNodes.hpp"
#include "redecomp/partition/Partitioner.hpp"
#include "redecomp/partition/PartitionElements.hpp"
#include "redecomp/partition/RCB.hpp"

namespace redecomp {

RedecompMesh::RedecompMesh( const mfem::ParMesh& parent, PartitionType method, int n_ranks )
    : RedecompMesh( parent, DefaultGhostLength( parent ), method, n_ranks )
{
}

RedecompMesh::RedecompMesh( const mfem::ParMesh& parent, double ghost_length, PartitionType method, int n_ranks )
    : parent_{ parent }, mpi_{ parent.GetComm() }
{
  TRIBOL_MARK_FUNCTION;
  // build partitioner
  std::unique_ptr<Partitioner> partitioner = nullptr;
  switch ( parent_.SpaceDimension() ) {
    case 2:
      switch ( method ) {
        case RCB:
          partitioner = std::make_unique<Partitioner2D>( std::make_unique<PartitionElements2D>(),
                                                         std::make_unique<RCB2D>( parent.GetComm() ) );
          break;
        default:
          SLIC_ERROR_ROOT(
              "Only recursive coordinate bisection (RCB) decompositions "
              "are currently supported." );
      }
      break;
    case 3:
      switch ( method ) {
        case RCB:
          partitioner = std::make_unique<Partitioner3D>( std::make_unique<PartitionElements3D>(),
                                                         std::make_unique<RCB3D>( parent.GetComm() ) );
          break;
        default:
          SLIC_ERROR_ROOT(
              "Only recursive coordinate bisection (RCB) decompositions "
              "are currently supported." );
      }
      break;
    default:
      SLIC_ERROR_ROOT( "Only 2D and 3D meshes are supported." );
  }

  SLIC_ERROR_ROOT_IF( n_ranks < 0, "Number of ranks must be non-negative." );
  SLIC_ERROR_ROOT_IF( n_ranks > parent.GetNRanks(),
                      "Number of ranks must be less than or equal to the number of MPI ranks." );
  auto parent_n_els = static_cast<int>( parent.GetGlobalNE() );
  if ( n_ranks == 0 ) {
    // preclude degenerate case where num elements/2 < num ranks
    // factor of 2 on num elements are due to elements being paired for contact
    // additional factor of 10 targets about 10 pairs/rank on a conforming mesh (and more than 10 on a non-conforming
    // mesh)
    n_ranks = std::max( std::min( parent.GetNRanks(), ( parent_n_els + 1 ) / 2 / 10 ), 1 );
  }
  SLIC_DEBUG_ROOT( axom::fmt::format( "Repartitioning {} elements onto {} ranks...", parent_n_els, n_ranks ) );
  // p2r = parent to redecomp
  p2r_elems_ = BuildP2RElementList( *partitioner, n_ranks, ghost_length );
  BuildRedecomp();
}

RedecompMesh::RedecompMesh( const mfem::ParMesh& parent, std::unique_ptr<const Partitioner> partitioner, int n_ranks )
    : RedecompMesh( parent, DefaultGhostLength( parent ), std::move( partitioner ), n_ranks )
{
}

RedecompMesh::RedecompMesh( const mfem::ParMesh& parent, double ghost_length,
                            std::unique_ptr<const Partitioner> partitioner, int n_ranks )
    : parent_{ parent }, mpi_{ parent.GetComm() }
{
  TRIBOL_MARK_FUNCTION;
  // check partitioner
  switch ( parent_.SpaceDimension() ) {
    case 2: {
      auto partitioner2d = dynamic_cast<const Partitioner2D*>( partitioner.get() );
      SLIC_ERROR_ROOT_IF( partitioner2d == nullptr, "Partitioner must be Partitioner2D." );
      auto partition_elems2d = dynamic_cast<const PartitionElements2D*>( partitioner2d->getPartitionEntity() );
      SLIC_ERROR_ROOT_IF( partition_elems2d == nullptr,
                          "Redecomp requires the PartitionEntity to be PartitionElements." );
      break;
    }
    case 3: {
      auto partitioner3d = dynamic_cast<const Partitioner3D*>( partitioner.get() );
      SLIC_ERROR_ROOT_IF( partitioner3d == nullptr, "Partitioner must be Partitioner3D." );
      auto partition_elems3d = dynamic_cast<const PartitionElements3D*>( partitioner3d->getPartitionEntity() );
      SLIC_ERROR_ROOT_IF( partition_elems3d == nullptr,
                          "Redecomp requires the PartitionEntity to be PartitionElements." );
      break;
    }
    default:
      SLIC_ERROR_ROOT( "Only 2D and 3D meshes are supported." );
  }

  SLIC_ERROR_ROOT_IF( n_ranks < 0, "Number of ranks must be non-negative." );
  SLIC_ERROR_ROOT_IF( n_ranks > parent.GetNRanks(),
                      "Number of ranks must be less than or equal to the number of MPI ranks." );
  auto parent_n_els = static_cast<int>( parent.GetGlobalNE() );
  if ( n_ranks == 0 ) {
    // preclude degenerate case where num elements/2 < num ranks
    // factor of 2 on num elements are due to elements being paired for contact
    // additional factor of 10 targets about 10 pairs/rank on a conforming mesh (and more than 10 on a non-conforming
    // mesh)
    n_ranks = std::max( std::min( parent.GetNRanks(), ( parent_n_els + 1 ) / 2 / 10 ), 1 );
  }
  SLIC_DEBUG_ROOT( axom::fmt::format( "Repartitioning {} elements onto {} ranks...", parent_n_els, n_ranks ) );
  // p2r = parent to redecomp
  p2r_elems_ = BuildP2RElementList( *partitioner, n_ranks, ghost_length );
  BuildRedecomp();
}

RedecompMesh::RedecompMesh( const mfem::ParMesh& parent, EntityIndexByRank&& p2r_elems )
    : parent_{ parent }, mpi_{ parent.GetComm() }, p2r_elems_{ std::move( p2r_elems ) }
{
  BuildRedecomp();
}

double RedecompMesh::DefaultGhostLength( const mfem::ParMesh& parent ) const
{
  return 1.25 * MaxElementSize( parent, MPIUtility( parent.GetComm() ) );
}

EntityIndexByRank RedecompMesh::BuildP2RElementList( const Partitioner& partitioner, int n_parts,
                                                     double ghost_length ) const
{
  SLIC_ERROR_ROOT_IF( ghost_length < 0.0, "Ghost element layer length should be 0 or larger." );
  return partitioner.generatePartitioning( n_parts, { &parent_ }, ghost_length )[0];
}

void RedecompMesh::BuildRedecomp()
{
  TRIBOL_MARK_FUNCTION;
  // dimension information
  Dim = parent_.Dimension();
  spaceDim = parent_.SpaceDimension();

  // construct list of global vertices to move from parent to redecomp domain
  auto parent_vertex_fec = std::make_unique<const mfem::H1_FECollection>( 1, parent_.Dimension() );
  auto parent_vertex_fes =
      mfem::ParFiniteElementSpace( const_cast<mfem::ParMesh*>( &parent_ ), parent_vertex_fec.get() );
  auto vertex_transfer = TransferByNodes( parent_vertex_fes, *this );
  // p2r = parent to redecomp
  auto p2r_verts = vertex_transfer.P2RNodeList( true );
  // r2p = redecomp to parent
  auto r2p_vert_idx = MPIArray<int>( &mpi_ );

  // map global parent vertex IDs to redecomp vertex IDs
  r2p_vert_idx.SendRecvArrayEach( p2r_verts.first );
  auto n_ranks = mpi_.NRanks();
  auto vert_idx_map = std::unordered_map<int, int>();
  auto vert_ct = 0;
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    for ( int v{ 0 }; v < r2p_vert_idx[r].size(); ++v ) {
      auto vert_idx_it = vert_idx_map.emplace( r2p_vert_idx[r][v], vert_ct );
      r2p_vert_idx[r][v] = vert_idx_it.first->second;
      if ( vert_idx_it.second ) {
        ++vert_ct;
      }
    }
  }

  // send vertex coords from parent to redecomp
  p2r_verts = vertex_transfer.P2RNodeList( false );
  auto redecomp_coords = MPIArray<double, 2>( &mpi_ );
  redecomp_coords.SendRecvEach( [this, &p2r_verts]( int dest ) {
    auto parent_coords = axom::Array<double, 2>();
    const auto& vert_idx = p2r_verts.first[dest];
    auto n_coords = vert_idx.size();
    parent_coords.reserve( 3 * n_coords );
    parent_coords.resize( n_coords, 3 );
    for ( int v{ 0 }; v < n_coords; ++v ) {
      for ( int d{ 0 }; d < parent_.SpaceDimension(); ++d ) {
        parent_coords( v, d ) = parent_.GetVertex( vert_idx[v] )[d];
      }
    }
    return parent_coords;
  } );

  // create vertices on redecomp
  NumOfVertices = vert_idx_map.size();
  vertices.SetSize( NumOfVertices );
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    for ( int v{ 0 }; v < r2p_vert_idx[r].size(); ++v ) {
      for ( int d{ 0 }; d < parent_.SpaceDimension(); ++d ) {
        vertices[r2p_vert_idx[r][v]]( d ) = redecomp_coords[r]( v, d );
      }
    }
  }

  // Create elements on this
  // Send and receive element types
  auto redecomp_etypes = MPIArray<int>( &mpi_ );
  redecomp_etypes.SendRecvEach( [this]( int dest ) {
    auto parent_etypes = axom::Array<int>( p2r_elems_.first[dest].size(), p2r_elems_.first[dest].size() );
    for ( int e{ 0 }; e < p2r_elems_.first[dest].size(); ++e ) {
      parent_etypes[e] = parent_.GetElementType( p2r_elems_.first[dest][e] );
    }
    return parent_etypes;
  } );
  // Find length of element connectivities
  auto parent_tot_conns = axom::Array<int>( n_ranks, n_ranks );
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    const auto& elem_idx = p2r_elems_.first[r];
    auto n_elems = elem_idx.size();
    for ( int e{ 0 }; e < n_elems; ++e ) {
      parent_tot_conns[r] += parent_.GetElement( elem_idx[e] )->GetNVertices();
    }
  }
  // Send and receive element connectivities
  auto redecomp_conns = MPIArray<int>( &mpi_ );
  redecomp_conns.SendRecvEach( [this, &parent_tot_conns, &parent_vertex_fes]( int dest ) {
    auto parent_conns = axom::Array<int>( parent_tot_conns[dest], parent_tot_conns[dest] );
    parent_tot_conns[dest] = 0;
    const auto& elem_idx = p2r_elems_.first[dest];
    for ( int e{ 0 }; e < elem_idx.size(); ++e ) {
      mfem::Array<int> elem_conn;
      parent_.GetElement( elem_idx[e] )->GetVertices( elem_conn );
      for ( int v{ 0 }; v < elem_conn.Size(); ++v ) {
        parent_conns[parent_tot_conns[dest] + v] = parent_vertex_fes.GetGlobalTDofNumber( elem_conn[v] );
      }
      parent_tot_conns[dest] += elem_conn.Size();
    }
    return parent_conns;
  } );
  // Send and receive element attributes
  auto redecomp_attribs = MPIArray<int>( &mpi_ );
  redecomp_attribs.SendRecvEach( [this]( int dest ) {
    const auto& elem_idx = p2r_elems_.first[dest];
    auto n_elems = elem_idx.size();
    auto parent_attribs = axom::Array<int>( n_elems, n_elems );
    for ( int e{ 0 }; e < n_elems; ++e ) {
      parent_attribs[e] = parent_.GetElement( elem_idx[e] )->GetAttribute();
    }
    return parent_attribs;
  } );
  // Count number of elements
  // NOTE: Don't use NumOfElements.  AddElement() will increment it.
  auto n_els = 0;
  for ( const auto& redecomp_etype : redecomp_etypes ) {
    n_els += redecomp_etype.size();
  }
  elements.SetSize( n_els );
  // Create elements
  for ( int r{ 0 }; r < n_ranks; ++r ) {
    auto conn_ct = 0;
    for ( int e{ 0 }; e < redecomp_etypes[r].size(); ++e ) {
      auto el = NewElement( redecomp_etypes[r][e] );
      el->SetVertices( &redecomp_conns[r][conn_ct] );
      for ( int k{ 0 }; k < el->GetNVertices(); ++k ) {
        el->GetVertices()[k] = vert_idx_map[el->GetVertices()[k]];
      }
      conn_ct += el->GetNVertices();
      el->SetAttribute( redecomp_attribs[r][e] );
      AddElement( el );
    }
  }

  // Finalize mesh topology
  auto generate_boundary = false;
  FinalizeTopology( generate_boundary );

  // Fill r2p_elem_offsets_ with element rank offsets
  // r2p = redecomp to parent
  r2p_elem_offsets_.reserve( n_ranks + 1 );
  r2p_elem_offsets_.resize( n_ranks + 1 );
  mpi_.SendRecvEach(
      type<axom::Array<int>>(),
      [this]( int dest ) { return axom::Array<int>( { static_cast<int>( p2r_elems_.first[dest].size() ) } ); },
      [this]( axom::Array<int>&& recv_data, int source ) { r2p_elem_offsets_[source + 1] = recv_data[0]; } );
  for ( int i{ 2 }; i < n_ranks + 1; ++i ) {
    r2p_elem_offsets_[i] += r2p_elem_offsets_[i - 1];
  }

  // Fill r2p_ghost_elems_ with local indices of ghost elements
  // r2p = redecomp to parent
  r2p_ghost_elems_ = MPIArray<int>( &mpi_ );
  r2p_ghost_elems_.SendRecvEach( [this]( int dest ) {
    auto n_elems = p2r_elems_.second[dest].size();
    auto parent_gelems = axom::Array<int>( 0, n_elems );
    for ( int i{ 0 }; i < n_elems; ++i ) {
      if ( p2r_elems_.second[dest][i] ) {
        parent_gelems.push_back( i );
      }
    }
    return parent_gelems;
  } );
  for ( int i{ 1 }; i < r2p_ghost_elems_.size(); ++i ) {
    for ( auto& recv_gelem : r2p_ghost_elems_[i] ) {
      recv_gelem += r2p_elem_offsets_[i];
    }
  }

  // p > 1 case: set mesh as curved Mesh (create Nodes) and transfer Nodes from
  // parent
  auto parent_node_fes = dynamic_cast<const mfem::ParFiniteElementSpace*>( parent_.GetNodalFESpace() );
  if ( parent_node_fes ) {
    SetCurvature( parent_node_fes->FEColl()->GetOrder(), parent_node_fes->IsDGSpace(), spaceDim,
                  parent_node_fes->GetOrdering() );

    auto parent_node_pargf = dynamic_cast<const mfem::ParGridFunction*>( parent_.GetNodes() );
    SLIC_ERROR_ROOT_IF( parent_node_pargf == nullptr, "Nodes in ParMesh parent_ must be a ParGridFunction." );
    auto node_transfer = RedecompTransfer();
    node_transfer.TransferToSerial( *parent_node_pargf, *Nodes );
  }

  SetAttributes();
  Finalize();
}

// TODO: potentially improve the way this is calculated?
double RedecompMesh::MaxElementSize( const mfem::ParMesh& parent, const MPIUtility& mpi )
{
  auto max_elem_size = 0.0;
  for ( auto i = 0; i < parent.GetNE(); ++i ) {
    // TODO: const version of GetElementSize() NOTE: h_max (type = 2) not
    // implemented for surface meshes in GetElementSize(). The version called
    // (type = 0) returns average stretch at the element center
    max_elem_size = std::max( const_cast<mfem::ParMesh&>( parent ).GetElementSize( i, 0 ), max_elem_size );
  }

  // Accounts for element diagonal lengths greater than aligned lengths
  max_elem_size = max_elem_size * std::sqrt( static_cast<double>( parent.Dimension() ) );

  return mpi.AllreduceValue( max_elem_size, MPI_MAX );
}

}  // end namespace redecomp
