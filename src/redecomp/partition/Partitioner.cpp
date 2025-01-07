// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "Partitioner.hpp"

#include "redecomp/partition/PartitionMethod.hpp"
#include "redecomp/partition/PartitionEntity.hpp"

namespace redecomp {

template <int NDIMS>
PartitionerByDim<NDIMS>::PartitionerByDim( std::unique_ptr<const PartitionEntity<NDIMS>> partition_entity,
                                           std::unique_ptr<const PartitionMethod<NDIMS>> partition_method )
    : partition_entity_{ std::move( partition_entity ) }, partition_method_{ std::move( partition_method ) }
{
}

template <int NDIMS>
const MPIUtility& PartitionerByDim<NDIMS>::getMPIUtility() const
{
  return partition_method_->getMPIUtility();
}

template <int NDIMS>
std::vector<EntityIndexByRank> PartitionerByDim<NDIMS>::generatePartitioning(
    int n_parts, const std::vector<const mfem::ParMesh*>& par_meshes, double ghost_size ) const
{
  return partition_method_->generatePartitioning( n_parts, partition_entity_->EntityCoordinates( par_meshes ),
                                                  ghost_size );
}

template <int NDIMS>
const PartitionEntity<NDIMS>* PartitionerByDim<NDIMS>::getPartitionEntity() const
{
  return partition_entity_.get();
}

template <int NDIMS>
const PartitionMethod<NDIMS>* PartitionerByDim<NDIMS>::getPartitionMethod() const
{
  return partition_method_.get();
}

template class PartitionerByDim<2>;
template class PartitionerByDim<3>;

}  // end namespace redecomp
