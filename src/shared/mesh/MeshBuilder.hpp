#ifndef SRC_SHARED_MESH_MESHBUILDER_HPP_
#define SRC_SHARED_MESH_MESHBUILDER_HPP_

#include <initializer_list>

#include "mfem.hpp"

#include "shared/config.hpp"

/**
 * @file MeshBuilder.hpp
 * @brief Contains the MeshBuilder and ParMeshBuilder classes to simplify constructing and manipulating MFEM meshes.
 */

namespace shared {

/**
 * @class MeshBuilder
 * @brief A class for building and manipulating MFEM meshes.
 */
class MeshBuilder {
 public:
  /**
   * @brief Creates a single mesh from a list of MeshBuilder objects.
   * @note Each mesh is added sequentially to the new mesh, with node numbering of the next mesh following the last node
   * of the previous mesh. Element attributes and boundary element attributes are not changed. Nodes are not merged by
   * this function. If two nodes have the same coordinates, they will remain separate entities in the unified mesh.
   * @param meshes A list of MeshBuilder objects from which the mesh will be created.
   * @return A new MeshBuilder object representing the unified mesh.
   */
  static MeshBuilder Unify( std::initializer_list<MeshBuilder> meshes );

  /**
   * @brief Creates a single mesh from multiple MeshBuilder objects.
   * @note Each mesh is added sequentially to the new mesh, with node numbering of the next mesh following the last node
   * of the previous mesh. Element attributes and boundary element attributes are not changed. Nodes are not merged by
   * this function. If two nodes have the same coordinates, they will remain separate entities in the unified mesh.
   * @tparam Args Variadic template parameter for MeshBuilder objects.
   * @param meshes MeshBuilder objects from which the mesh will be created.
   * @return A new MeshBuilder object representing the unified mesh.
   */
  template <typename... Args>
  static MeshBuilder Unify( Args&&... meshes );

  /**
   * @brief Creates a square mesh occupying the unit square, [0, 1]^2.
   * @param n_x_els Number of elements in the x direction (>0).
   * @param n_y_els Number of elements in the y direction (>0).
   * @return A new MeshBuilder object representing the square mesh.
   */
  static MeshBuilder SquareMesh( int n_x_els, int n_y_els );

  /**
   * @brief Creates a cube mesh occupying the unit cube, [0, 1]^3.
   * @param n_x_els Number of elements in the x direction (>0).
   * @param n_y_els Number of elements in the y direction (>0).
   * @param n_z_els Number of elements in the z direction (>0).
   * @param elem_type The type of elements to use in the mesh (default is HEXAHEDRON).
   * @return A new MeshBuilder object representing the cube mesh.
   */
  static MeshBuilder CubeMesh( int n_x_els, int n_y_els, int n_z_els,
                               mfem::Element::Type elem_type = mfem::Element::HEXAHEDRON );

  /**
   * @brief Creates a hypercube mesh occupying the unit hypercube, [0, 1]^dim.
   * @param dim Dimension of the hypercube (between 1 and 3).
   * @param n_els Number of elements in each direction of each dimension (>0).
   * @return A new MeshBuilder object representing the hypercube mesh.
   */
  static MeshBuilder HypercubeMesh( int dim, int n_els );

  /**
   * @brief Constructs a MeshBuilder object from an mfem::Mesh.
   * @param mesh An rvalue reference to an mfem::Mesh object.
   */
  MeshBuilder( mfem::Mesh&& mesh );

  /**
   * @brief Grows or shrinks the mesh relative to the origin by the scale factors in the given vector.
   * @param scale_factors A list of scale factors to apply to each dimension.
   * @return An rvalue reference to the updated MeshBuilder object.
   */
  MeshBuilder&& scale( std::initializer_list<double> scale_factors );

  /**
   * @brief Translates the mesh by a given vector.
   * @param dx A list of translation distances for each dimension.
   * @return An rvalue reference to the updated MeshBuilder object.
   */
  MeshBuilder&& translate( std::initializer_list<double> dx );

  /**
   * @brief Translates a specific node in the mesh by a given vector.
   * @param node_id The ID of the node to be translated.
   * @param dx A list of translation distances for each dimension.
   * @return An rvalue reference to the updated MeshBuilder object.
   */
  MeshBuilder&& translateNode( int node_id, std::initializer_list<double> dx );

  /**
   * @brief Refines the mesh uniformly a specified number of times.
   * @param n_times The number of times to refine the mesh.
   * @return An rvalue reference to the updated MeshBuilder object.
   */
  MeshBuilder&& refine( int n_times );

  /**
   * @brief Updates an attribute in the mesh.
   * @param old_attrib The old attribute value.
   * @param new_attrib The new attribute value.
   * @return An rvalue reference to the updated MeshBuilder object.
   */
  MeshBuilder&& updateAttrib( int old_attrib, int new_attrib );

  /**
   * @brief Prints boundary attribute information.
   * @return An rvalue reference to the updated MeshBuilder object.
   */
  MeshBuilder&& bdrAttribInfo();

  /**
   * @brief Updates a boundary attribute in the mesh.
   * @param old_attrib The old boundary attribute value.
   * @param new_attrib The new boundary attribute value.
   * @return An rvalue reference to the updated MeshBuilder object.
   */
  MeshBuilder&& updateBdrAttrib( int old_attrib, int new_attrib );

  /**
   * @brief Implicit conversion to a pointer to mfem::Mesh.
   * @return A pointer to the underlying mfem::Mesh object.
   */
  operator mfem::Mesh*();

  /**
   * @brief Implicit conversion to a const pointer to mfem::Mesh.
   * @return A const pointer to the underlying mfem::Mesh object.
   */
  operator const mfem::Mesh*() const;

  /**
   * @brief Implicit conversion to a reference to mfem::Mesh.
   * @return A reference to the underlying mfem::Mesh object.
   */
  operator mfem::Mesh&();

  /**
   * @brief Implicit conversion to a const reference to mfem::Mesh.
   * @return A const reference to the underlying mfem::Mesh object.
   */
  operator const mfem::Mesh&() const;

  /**
   * @brief Implicit conversion to a rvalue reference to mfem::Mesh.
   * @return An rvalue reference to the underlying mfem::Mesh object
   */
  operator mfem::Mesh&&();

 private:
  mfem::Mesh mesh_;  ///< The underlying mesh object.
};

#ifdef TRIBOL_USE_MPI

/**
 * @class ParMeshBuilder
 * @brief A class for building and manipulating parallel MFEM meshes.
 */
class ParMeshBuilder {
 public:
  /**
   * @brief Constructs a ParMeshBuilder object.
   * @param comm The MPI communicator.
   * @param mesh An rvalue reference to a MeshBuilder object.
   */
  ParMeshBuilder( MPI_Comm comm, MeshBuilder&& mesh );

  /**
   * @brief Sets the finite element collection for the Nodes grid function (holding nodal coordinates of the mesh).
   * @param fe_coll The finite element collection.
   * @return An rvalue reference to the updated ParMeshBuilder object.
   */
  ParMeshBuilder&& setNodesFEColl( mfem::H1_FECollection fe_coll );

  /**
   * @brief Gets the grid function for the nodal coordinates.
   * @return A reference to the mfem::ParGridFunction object representing the nodal coordinates.
   */
  mfem::ParGridFunction& getNodes();

  /**
   * @brief Gets the grid function for the nodal coordinates.
   * @return A const reference to the mfem::ParGridFunction object representing the nodal coordinates.
   */
  const mfem::ParGridFunction& getNodes() const;

  /**
   * @brief Gets the finite element space for the nodal coordinates.
   * @return A reference to the mfem::ParFiniteElementSpace object representing the nodal coordinates.
   */
  mfem::ParFiniteElementSpace& getNodesFESpace();

  /**
   * @brief Gets the finite element space for the nodal coordinates.
   * @return A const reference to the mfem::ParFiniteElementSpace object representing the nodal coordinates.
   */
  const mfem::ParFiniteElementSpace& getNodesFESpace() const;

  /**
   * @brief Implicit conversion to a const reference to mfem::ParMesh.
   * @return A const reference to the underlying mfem::ParMesh object.
   */
  operator const mfem::ParMesh&() const;

 private:
  mfem::ParMesh pmesh_;  ///< The underlying parallel mesh object.
};

#endif

}  // namespace shared

#endif  // SRC_SHARED_MESH_MESHBUILDER_HPP_
