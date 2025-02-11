// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "MultiRedecomp.hpp"

#include "axom/slic.hpp"

#include "redecomp/partition/Partitioner.hpp"
#include "redecomp/partition/PartitionElements.hpp"
#include "redecomp/partition/RCB.hpp"

namespace redecomp {

MultiRedecomp::MultiRedecomp( int dim, MPI_Comm comm, PartitionType method, double ghost_len_multiplier )
    : ghost_len_multiplier_{ ghost_len_multiplier }
{
  // create partitioner
  switch ( dim ) {
    case 2:
      switch ( method ) {
        case RCB:
          partitioner_ = std::make_unique<Partitioner2D>( std::make_unique<PartitionElements2D>(),
                                                          std::make_unique<RCB2D>( comm ) );
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
          partitioner_ = std::make_unique<Partitioner3D>( std::make_unique<PartitionElements3D>(),
                                                          std::make_unique<RCB3D>( comm ) );
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
}

MultiRedecomp::MultiRedecomp( std::unique_ptr<const Partitioner> partitioner, double ghost_len_multiplier )
    : partitioner_{ std::move( partitioner ) }, ghost_len_multiplier_{ ghost_len_multiplier }
{
}

std::vector<std::unique_ptr<RedecompMesh>> MultiRedecomp::createRedecompMeshes(
    const std::vector<const mfem::ParMesh*>& parents )
{
  std::vector<std::unique_ptr<RedecompMesh>> redecomps;

  // check the validity of parents
  SLIC_ERROR_ROOT_IF( parents.empty(), "At least one mesh in parents required." );
  for ( size_t m{ 1 }; m < parents.size(); ++m ) {
    SLIC_ERROR_ROOT_IF( parents[m - 1]->SpaceDimension() != parents[m]->SpaceDimension(),
                        "SpaceDimension must match on all parent meshes." );
    SLIC_ERROR_ROOT_IF( parents[m - 1]->GetComm() != parents[m]->GetComm(),
                        "MPI_Comm must match on all parent meshes." );
  }

  // make sure the partitioner works with parents
  auto dim = parents[0]->SpaceDimension();

  switch ( dim ) {
    case 2: {
      auto partitioner2d = dynamic_cast<const Partitioner2D*>( partitioner_.get() );
      SLIC_ERROR_ROOT_IF( partitioner2d == nullptr, "Partitioner must be Partitioner2D." );
      auto partition_elems2d = dynamic_cast<const PartitionElements2D*>( partitioner2d->getPartitionEntity() );
      SLIC_ERROR_ROOT_IF( partition_elems2d == nullptr,
                          "Redecomp requires the PartitionEntity "
                          "to be PartitionElements." );
      break;
    }
    case 3: {
      auto partitioner3d = dynamic_cast<const Partitioner3D*>( partitioner_.get() );
      SLIC_ERROR_ROOT_IF( partitioner3d == nullptr, "Partitioner must be Partitioner3D." );
      auto partition_elems3d = dynamic_cast<const PartitionElements3D*>( partitioner3d->getPartitionEntity() );
      SLIC_ERROR_ROOT_IF( partition_elems3d == nullptr,
                          "Redecomp requires the PartitionEntity "
                          "to be PartitionElements." );
      break;
    }
    default:
      SLIC_ERROR_ROOT( "Only 2D and 3D meshes are supported." );
  }

  // Compute the number of parts the redecomp mesh should have and the size of
  // the ghost layer
  auto n_parts = 0;
  auto ghost_size = 0.0;
  for ( auto parent : parents ) {
    n_parts = std::max( n_parts, static_cast<int>( parent->GetGlobalNE() ) );
    ghost_size = std::max( ghost_size, RedecompMesh::MaxElementSize( *parent, partitioner_->getMPIUtility() ) );
  }
  n_parts = std::min( partitioner_->getMPIUtility().NRanks(), n_parts );
  ghost_size *= ghost_len_multiplier_;

  // Generate the new parent to redecomp (p2r) element partitioning for all
  // parent meshes
  auto p2r_elems_by_mesh = partitioner_->generatePartitioning( n_parts, parents, ghost_size );

  // Build redecomp meshes using the generated parent-to-redecomp partitioning
  // of all parent meshes
  redecomps.reserve( parents.size() );
  for ( size_t m{ 0 }; m < parents.size(); ++m ) {
    redecomps.emplace_back( new RedecompMesh( *parents[m], std::move( p2r_elems_by_mesh[m] ) ) );
  }

  return redecomps;
}

}  // end namespace redecomp
