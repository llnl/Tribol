// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "PartitionElements.hpp"

namespace redecomp {

template <int NDIMS>
std::vector<axom::Array<Point<NDIMS>>> PartitionElements<NDIMS>::EntityCoordinates(
    const std::vector<const mfem::ParMesh*>& par_meshes ) const
{
  auto elem_centroids = std::vector<axom::Array<Point<NDIMS>>>();
  elem_centroids.reserve( par_meshes.size() );
  for ( auto par_mesh : par_meshes ) {
    auto n_elems = par_mesh->GetNE();
    elem_centroids.emplace_back( n_elems, n_elems );
    for ( int i{ 0 }; i < n_elems; ++i ) {
      auto vec_centroid = mfem::Vector( elem_centroids.back()[i].data(), NDIMS );
      // TODO: const version of GetElementCenter()
      const_cast<mfem::ParMesh*>( par_mesh )->GetElementCenter( i, vec_centroid );
    }
  }
  return elem_centroids;
}

template class PartitionElements<2>;
template class PartitionElements<3>;

}  // end namespace redecomp
