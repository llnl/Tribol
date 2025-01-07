// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "PartitionMethod.hpp"

namespace redecomp {

template <int NDIMS>
PartitionMethod<NDIMS>::PartitionMethod( const MPI_Comm& comm ) : mpi_{ comm }
{
}

template <int NDIMS>
const MPIUtility& PartitionMethod<NDIMS>::getMPIUtility() const
{
  return mpi_;
}

template class PartitionMethod<2>;
template class PartitionMethod<3>;

}  // end namespace redecomp
