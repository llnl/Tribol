// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "PartitionElements.hpp"

#include "mfem.hpp"

namespace redecomp {

template <int NDIMS>
std::vector<CoordList<NDIMS>> PartitionElements<NDIMS>::EntityCoordinates(
    const std::vector<const mfem::ParMesh*>& par_meshes ) const
{
  auto elem_centroids = std::vector<CoordList<NDIMS>>();
  elem_centroids.reserve( par_meshes.size() );

  for ( auto par_mesh : par_meshes ) {
    auto n_elems = par_mesh->GetNE();
    if ( n_elems == 0 ) {
      elem_centroids.emplace_back( mfem::Vector() );
      continue;
    }

    auto compute_centroids = [&]( const mfem::ParGridFunction& coords ) {
      mfem::Vector mesh_centroids( n_elems * NDIMS );
      mesh_centroids.UseDevice( coords.UseDevice() );
      // Create an IntegrationRule with a single point: the reference center
      // Assuming a single element type in the mesh
      mfem::Geometry::Type geom_type = par_mesh->GetElementBaseGeometry( 0 );
      mfem::IntegrationRule ir;
      ir.Append( mfem::Geometries.GetCenter( geom_type ) );

      // Setup QuadratureInterpolator. We want to evaluate the 'nodes' GridFunction at the integration points (centers).
      auto fes = coords.FESpace();
      const mfem::QuadratureInterpolator* qi = fes->GetQuadratureInterpolator( ir );

      //    We need the ElementRestriction to convert the global 'nodes' vector to E-vector.
      //    We use LEXICOGRAPHIC ordering which is required for Tensor product evaluation
      //    often used on device for quads/hexes.
      const mfem::Operator* er = fes->GetElementRestriction( mfem::ElementDofOrdering::LEXICOGRAPHIC );

      mfem::Vector e_vec( er->Height() );

      // Perform the calculation on device.
      //    a) Global to Element (E-vector)
      er->Mult( coords, e_vec );
      //    b) Interpolate to Quadrature points (Q-vector)
      qi->Values( e_vec, mesh_centroids );

      return mesh_centroids;
    };

    auto coords = dynamic_cast<const mfem::ParGridFunction*>( par_mesh->GetNodes() );
    if ( coords ) {
      elem_centroids.emplace_back( compute_centroids( *coords ) );
    } else {
      // Create temporary nodes
      auto* non_const_mesh = const_cast<mfem::ParMesh*>( par_mesh );
      mfem::H1_FECollection fec( 1, NDIMS );
      mfem::ParFiniteElementSpace fes( non_const_mesh, &fec, NDIMS );
      mfem::ParGridFunction nodes( &fes );
      non_const_mesh->GetNodes( nodes );
      elem_centroids.emplace_back( compute_centroids( nodes ) );
    }
  }
  return elem_centroids;
}

template class PartitionElements<2>;
template class PartitionElements<3>;

}  // end namespace redecomp
