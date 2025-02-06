#include "MeshBuilder.hpp"

#include "axom/slic.hpp"

namespace shared {

MeshBuilder MeshBuilder::Merged( std::initializer_list<MeshBuilder> meshes )
{
  std::vector<mfem::Mesh*> mesh_list;
  mesh_list.reserve( meshes.size() );
  for ( auto& mesh : meshes ) {
    // NOTE: the const cast is because the constructor requires a non-const mesh, even though the data is copied and not
    // altered
    mesh_list.push_back( &const_cast<mfem::Mesh&>( *mesh ) );
  }
  return mfem::Mesh( mesh_list.data(), mesh_list.size() );
}

MeshBuilder MeshBuilder::SquareMesh( int n_x_els, int n_y_els )
{
  return mfem::Mesh::MakeCartesian2D( n_x_els, n_y_els, mfem::Element::QUADRILATERAL );
}

MeshBuilder MeshBuilder::CubeMesh( int n_x_els, int n_y_els, int n_z_els )
{
  return mfem::Mesh::MakeCartesian3D( n_x_els, n_y_els, n_z_els, mfem::Element::HEXAHEDRON );
}

MeshBuilder::MeshBuilder( mfem::Mesh&& mesh ) : mesh_{ std::move( mesh ) } { mesh_.EnsureNodes(); }

MeshBuilder&& MeshBuilder::translate( std::initializer_list<double> dx )
{
  SLIC_ERROR_ROOT_IF( static_cast<int>( dx.size() ) != mesh_.SpaceDimension(), "Invalid size for dx" );
  auto& coords = *mesh_.GetNodes();
  for ( int d = 0; d < mesh_.SpaceDimension(); ++d ) {
    for ( int i = 0; i < mesh_.GetNV(); ++i ) {
      auto vdof = coords.FESpace()->DofToVDof( i, d );
      coords[vdof] += *( dx.begin() + d );
    }
  }
  return std::move( *this );
}

MeshBuilder&& MeshBuilder::updateAttrib( int old_attrib, int new_attrib )
{
  for ( int i = 0; i < mesh_.GetNE(); ++i ) {
    if ( mesh_.GetAttribute( i ) == old_attrib ) {
      mesh_.SetAttribute( i, new_attrib );
    }
  }
  return std::move( *this );
}

MeshBuilder&& MeshBuilder::updateBdrAttrib( int old_attrib, int new_attrib )
{
  for ( int i = 0; i < mesh_.GetNBE(); ++i ) {
    if ( mesh_.GetBdrAttribute( i ) == old_attrib ) {
      mesh_.SetBdrAttribute( i, new_attrib );
    }
  }
  return std::move( *this );
}

MeshBuilder::operator mfem::Mesh*() { return &mesh_; }

MeshBuilder::operator const mfem::Mesh*() const { return &mesh_; }

MeshBuilder::operator mfem::Mesh&() { return mesh_; }

MeshBuilder::operator const mfem::Mesh&() const { return mesh_; }

ParMeshBuilder::ParMeshBuilder( MPI_Comm comm, MeshBuilder&& mesh ) : pmesh_{ comm, mesh } {}

ParMeshBuilder&& ParMeshBuilder::setNodesFEColl( mfem::H1_FECollection fe_coll )
{
  mfem::FiniteElementCollection* fe_coll_ptr = fe_coll.Clone( fe_coll.GetOrder() );
  mfem::ParFiniteElementSpace* fe_space =
      new mfem::ParFiniteElementSpace( &pmesh_, fe_coll_ptr, pmesh_.SpaceDimension() );
  pmesh_.SetNodalFESpace( fe_space );
  pmesh_.GetNodes()->MakeOwner( fe_coll_ptr );
  return std::move( *this );
}

mfem::ParGridFunction& ParMeshBuilder::getNodes()
{
  // static_cast should be OK; MeshBuilder meshes always have Nodes so this won't be null and this should never be a
  // GridFunction
  return *static_cast<mfem::ParGridFunction*>( pmesh_.GetNodes() );
}

mfem::ParFiniteElementSpace& ParMeshBuilder::getNodesFESpace() { return *getNodes().ParFESpace(); }

ParMeshBuilder::operator const mfem::ParMesh&() const { return pmesh_; }

}  // namespace shared