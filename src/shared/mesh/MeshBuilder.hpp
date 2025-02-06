#ifndef SRC_SHARED_MESHBUILDER_HPP_
#define SRC_SHARED_MESHBUILDER_HPP_

#include <initializer_list>

#include "mfem.hpp"

#include "tribol/config.hpp"

namespace shared {

class MeshBuilder {
 public:
  static MeshBuilder Merged( std::initializer_list<MeshBuilder> meshes );
  template <typename... Args>
  static MeshBuilder Merged( Args&&... meshes )
  {
    return Merged( { std::forward<Args>( meshes )... } );
  }
  static MeshBuilder SquareMesh( int n_x_els, int n_y_els );
  static MeshBuilder CubeMesh( int n_x_els, int n_y_els, int n_z_els );
  static MeshBuilder HypercubeMesh( int dim, int n_els );

  MeshBuilder( mfem::Mesh&& mesh );

  MeshBuilder&& translate( std::initializer_list<double> dx );

  MeshBuilder&& updateAttrib( int old_attrib, int new_attrib );

  MeshBuilder&& bdrAttribInfo();
  MeshBuilder&& updateBdrAttrib( int old_attrib, int new_attrib );

  operator mfem::Mesh*();
  operator const mfem::Mesh*() const;

  operator mfem::Mesh&();
  operator const mfem::Mesh&() const;

 private:
  mfem::Mesh mesh_;
};

#ifdef TRIBOL_USE_MPI

#include <mpi.h>

class ParMeshBuilder {
 public:
  ParMeshBuilder( MPI_Comm comm, MeshBuilder&& mesh );

  ParMeshBuilder&& setNodesFEColl( mfem::H1_FECollection fe_coll );
  mfem::ParGridFunction& getNodes();
  const mfem::ParGridFunction& getNodes() const;
  mfem::ParFiniteElementSpace& getNodesFESpace();
  const mfem::ParFiniteElementSpace& getNodesFESpace() const;

  operator const mfem::ParMesh&() const;

 private:
  mfem::ParMesh pmesh_;
};

#endif

}  // namespace shared

#endif  // SRC_SHARED_MESHBUILDER_HPP_
