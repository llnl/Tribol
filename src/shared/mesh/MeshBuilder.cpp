#include "MeshBuilder.hpp"

#include "axom/slic.hpp"
#include "axom/primal.hpp"

namespace shared {

MeshBuilder MeshBuilder::Unify( std::initializer_list<MeshBuilder> meshes )
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

MeshBuilder MeshBuilder::CubeMesh( int n_x_els, int n_y_els, int n_z_els, mfem::Element::Type elem_type )
{
  return mfem::Mesh::MakeCartesian3D( n_x_els, n_y_els, n_z_els, elem_type );
}

MeshBuilder MeshBuilder::HypercubeMesh( int dim, int n_els )
{
  switch ( dim ) {
    case 1:
      return mfem::Mesh::MakeCartesian1D( n_els );
    case 2:
      return mfem::Mesh::MakeCartesian2D( n_els, n_els, mfem::Element::QUADRILATERAL );
    case 3:
      return mfem::Mesh::MakeCartesian3D( n_els, n_els, n_els, mfem::Element::HEXAHEDRON );
    default:
      SLIC_ERROR_ROOT( "Mesh dimension must be between 1 and 3." );
      return mfem::Mesh();
  }
}

MeshBuilder::MeshBuilder( mfem::Mesh&& mesh ) : mesh_{ std::move( mesh ) } { mesh_.EnsureNodes(); }

MeshBuilder&& MeshBuilder::scale( std::initializer_list<double> scale_factors )
{
  SLIC_ERROR_ROOT_IF( static_cast<int>( scale_factors.size() ) != mesh_.SpaceDimension(),
                      "scale_factors size does not match mesh dimension." );
  auto& coords = *mesh_.GetNodes();
  for ( int d = 0; d < mesh_.SpaceDimension(); ++d ) {
    for ( int i = 0; i < mesh_.GetNV(); ++i ) {
      auto vdof = coords.FESpace()->DofToVDof( i, d );
      coords[vdof] *= *( scale_factors.begin() + d );
    }
  }
  return std::move( *this );
}

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

MeshBuilder&& MeshBuilder::translateNode( int node_id, std::initializer_list<double> dx )
{
  SLIC_ERROR_ROOT_IF( static_cast<int>( dx.size() ) != mesh_.SpaceDimension(), "Invalid size for dx" );
  auto& coords = *mesh_.GetNodes();
  for ( int d = 0; d < mesh_.SpaceDimension(); ++d ) {
    auto vdof = coords.FESpace()->DofToVDof( node_id, d );
    coords[vdof] += *( dx.begin() + d );
  }
  return std::move( *this );
}

MeshBuilder&& MeshBuilder::refine( int n_times )
{
  for ( int i = 0; i < n_times; ++i ) {
    mesh_.UniformRefinement();
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
  // add the new attribute to the mesh's list of attributes
  auto old_attrib_idx = mesh_.attributes.Find( old_attrib );
  if ( old_attrib_idx >= 0 ) {
    mesh_.attributes[old_attrib_idx] = new_attrib;
  }
  mesh_.attributes.Sort();
  mesh_.attributes.Unique();
  return std::move( *this );
}

MeshBuilder&& MeshBuilder::bdrAttribInfo()
{
  for ( int i{ 0 }; i < mesh_.bdr_attributes.Size(); ++i ) {
    auto current_attrib = mesh_.bdr_attributes[i];
    auto num_elems = 0;
    auto attrib_bbox = axom::primal::BoundingBox<double, 3>();
    for ( int j = 0; j < mesh_.GetNBE(); ++j ) {
      if ( mesh_.GetBdrAttribute( j ) == current_attrib ) {
        num_elems++;
        mfem::Array<int> bdr_dofs;
        mesh_.GetNodes()->FESpace()->GetBdrElementDofs( j, bdr_dofs );
        for ( int k{ 0 }; k < bdr_dofs.Size(); ++k ) {
          axom::primal::Point<double, 3> vert;
          for ( int d{ 0 }; d < mesh_.SpaceDimension(); ++d ) {
            auto vdof = mesh_.GetNodes()->FESpace()->DofToVDof( bdr_dofs[k], d );
            vert[d] = ( *mesh_.GetNodes() )[vdof];
          }
          attrib_bbox.addPoint( vert );
        }
      }
    }
    if ( num_elems > 0 ) {
      std::cout << "Boundary attribute " << current_attrib << std::endl;
      std::cout << "  Number of elements: " << num_elems << std::endl;
      std::cout << "  Min coordinate:     " << attrib_bbox.getMin() << std::endl;
      std::cout << "  Max coordinate:     " << attrib_bbox.getMax() << std::endl;
      std::cout << "  Coordinate range:   " << attrib_bbox.range() << std::endl;
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
  // add the new boundary attribute to the mesh's list of boundary attributes
  auto old_attrib_idx = mesh_.bdr_attributes.Find( old_attrib );
  if ( old_attrib_idx >= 0 ) {
    mesh_.bdr_attributes[old_attrib_idx] = new_attrib;
  }
  mesh_.bdr_attributes.Sort();
  mesh_.bdr_attributes.Unique();
  return std::move( *this );
}

MeshBuilder::operator mfem::Mesh*() { return &mesh_; }

MeshBuilder::operator const mfem::Mesh*() const { return &mesh_; }

MeshBuilder::operator mfem::Mesh&() { return mesh_; }

MeshBuilder::operator const mfem::Mesh&() const { return mesh_; }

MeshBuilder::operator mfem::Mesh&&() { return std::move( mesh_ ); }

#ifdef TRIBOL_USE_MPI

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

const mfem::ParGridFunction& ParMeshBuilder::getNodes() const
{
  // static_cast should be OK; MeshBuilder meshes always have Nodes so this won't be null and this should never be a
  // GridFunction
  return *static_cast<const mfem::ParGridFunction*>( pmesh_.GetNodes() );
}

mfem::ParFiniteElementSpace& ParMeshBuilder::getNodesFESpace() { return *getNodes().ParFESpace(); }

const mfem::ParFiniteElementSpace& ParMeshBuilder::getNodesFESpace() const { return *getNodes().ParFESpace(); }

ParMeshBuilder::operator const mfem::ParMesh&() const { return pmesh_; }

#endif

}  // namespace shared
