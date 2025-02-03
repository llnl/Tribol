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

  MeshBuilder( mfem::Mesh&& mesh );

  MeshBuilder&& translate( std::initializer_list<double> dx );
  MeshBuilder&& updateAttrib( int old_attrib, int new_attrib );
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

  void setNodesFEColl( mfem::H1_FECollection fe_coll );
  mfem::ParGridFunction& getNodes();
  mfem::ParFiniteElementSpace& getNodesFESpace();

  operator const mfem::ParMesh&() const;

 private:
  mfem::ParMesh pmesh_;
};

#endif

}  // namespace shared

#endif  // SRC_SHARED_MESHBUILDER_HPP_