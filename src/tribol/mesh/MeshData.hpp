// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_MESH_MESHDATA_HPP_
#define SRC_MESH_MESHDATA_HPP_

#include <ostream>

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/utils/DataManager.hpp"

namespace tribol
{

/*!
 * \brief Struct to hold method specific nodal fields
 */
struct MeshNodalData 
{
  int m_num_nodes;

  /////////////////////////
  // MORTAR NODAL FIELDS //
  /////////////////////////
  ArrayViewT<RealT>       m_node_gap;      ///< scalar nodal gap (used on nonmortar mesh) 
  ArrayViewT<const RealT> m_node_pressure; ///< scalar nodal pressure (used on nonmortar mesh) 

  bool m_is_node_gap_set      {false}; ///< true if nodal gap field is set
  bool m_is_node_pressure_set {false}; ///< true if nodal pressure field is set
  /////////////////////////

  bool m_is_velocity_set           {false}; ///< true if nodal velocities have been registered
  bool m_is_nodal_displacement_set {false}; ///< true if nodal displacements have been registered
  bool m_is_nodal_response_set     {false}; ///< true if the nodal responses have been registered

}; // end of struct MeshNodalData

/*!
 * \brief Struct to hold method specific element data
 */
struct MeshElemData 
{
  int m_num_cells;

  //////////////////////////////////////
  // PENALTY ENFORCEMENT ELEMENT DATA //
  //////////////////////////////////////
  ArrayViewT<const RealT> m_mat_mod;   ///< Bulk/Young's modulus for contact faces
  ArrayViewT<const RealT> m_thickness; ///< Volume element thickness associated with each contact face

  RealT m_penalty_stiffness {0.};      ///< single scalar kinematic penalty stiffness for each mesh
  RealT m_penalty_scale {1.};          ///< scale factor applied to kinematic penalty only
  RealT m_rate_penalty_stiffness {0.}; ///< single scalar rate penalty stiffness for each mesh
  RealT m_rate_percent_stiffness {0.}; ///< rate penalty is percentage of gap penalty

  bool m_is_kinematic_constant_penalty_set {false}; ///< True if single kinematic constant penalty is set
  bool m_is_kinematic_element_penalty_set  {false}; ///< True if the element-wise kinematic penalty is set
  bool m_is_rate_constant_penalty_set      {false}; ///< True if the constant rate penalty is set
  bool m_is_rate_percent_penalty_set       {false}; ///< True if the rate percent penalty is set

  bool m_is_element_thickness_set          {false}; ///< True if element thickness is set

  /*!
  * \brief Checks if the kinematic penalty data is valid
  *
  * \param [in] pen_enfrc penalty enforcement option struct
  *
  * \return true if the kinematic penalty option has valid data
  */
  bool isValidKinematicPenalty( PenaltyEnforcementOptions& pen_options );

  /*!
  * \brief Checks if the rate penalty data is valid
  *
  * \param [in] pen_enfrc to penalty enforcement option struct
  *
  * \return true if the rate penalty option has valid data
  */
  bool isValidRatePenalty( PenaltyEnforcementOptions& pen_options ); ///< True if rate penalty option is valid

};

class MeshData
{
public:
  /**
   * @brief Nested class for holding views (non-owned, shallow copies) of mesh data
   */
  class Viewer
  {
  public:
    /**
     * @brief Construct a new MeshData::Viewer object
     * 
     * @param mesh MeshData to create a view of
     */
    Viewer(MeshData& mesh);

    /**
     * @brief Obtain the mesh ID for the current mesh view
     * 
     * @return mesh ID
     */
    TRIBOL_HOST_DEVICE IndexT meshId() const { return m_mesh_id; }

    /**
     * @brief Get the element type for the current mesh view
     *
     * @note Tribol supports a single element type for each mesh
     * 
     * @return element type
     */
    TRIBOL_HOST_DEVICE InterfaceElementType getElementType() const { return m_element_type; }

    /**
     * @brief Get the memory space of the data for the current mesh view
     * 
     * @return memory space
     */
    TRIBOL_HOST_DEVICE MemorySpace getMemorySpace() const { return m_mem_space; }

    /**
     * @brief Get the allocator ID of the data for the current mesh view
     *
     * @note Corresponds to an umpire allocator ID if Tribol is built with
     * Umpire; zero otherwise
     * 
     * @return allocator ID
     */
    TRIBOL_HOST_DEVICE int getAllocatorId() const { return m_allocator_id; }

    /**
     * @brief Get the mesh nodal field data
     * 
     * @return nodal data for the mesh
     */
    TRIBOL_HOST_DEVICE MeshNodalData& getNodalFields() { return m_nodal_fields; }

    /// @overload
    TRIBOL_HOST_DEVICE const MeshNodalData& getNodalFields() const { return m_nodal_fields; }

    /**
     * @brief Get the mesh element data
     * 
     * @return element data for the mesh
     */
    TRIBOL_HOST_DEVICE MeshElemData& getElementData() { return m_element_data; }

    /// @overload
    TRIBOL_HOST_DEVICE const MeshElemData& getElementData() const { return m_element_data; }

    /**
     * @brief Spatial dimension of the mesh
     * 
     * @return spatial dimension
     */
    TRIBOL_HOST_DEVICE int spatialDimension() const { return m_position.size(); }

    /**
     * @brief Number of nodes in the mesh
     * 
     * @return node count
     */
    TRIBOL_HOST_DEVICE IndexT numberOfNodes() const { return m_num_nodes; }

    /**
     * @brief Number of elements in the mesh
     * 
     * @return element count
     */
    TRIBOL_HOST_DEVICE IndexT numberOfElements() const { return m_connectivity.shape()[0]; }

    /**
     * @brief Number of nodes in each element of the mesh
     * 
     * @return nodes per element
     */
    TRIBOL_HOST_DEVICE IndexT numberOfNodesPerElement() const { return m_connectivity.shape()[1]; }

    /**
     * @brief Get the global node ID
     * 
     * @param element_id which element the node belongs to
     * @param local_node_id node ID for the local element
     * @return global node ID
     */
    TRIBOL_HOST_DEVICE IndexT getGlobalNodeId(IndexT element_id, IndexT local_node_id) const
    {
      return m_connectivity(element_id, local_node_id);
    }

    /**
     * @brief Get the nodal position array views
     * 
     * @return array view of the nodal position arrays
     */
    TRIBOL_HOST_DEVICE const MultiViewArrayView<const RealT>& getPosition() const
    {
      return m_position;
    }
    
    /**
     * @brief Is the displacement vector populated?
     * 
     * @return true if non-empty; false otherwise
     */
    TRIBOL_HOST_DEVICE bool hasDisplacement() const { return !m_disp.empty(); }

    /**
     * @brief Get the nodal displacement array views
     * 
     * @return array view of the nodal displacement arrays
     */
    TRIBOL_HOST_DEVICE const MultiViewArrayView<const RealT>& getDisplacement() const
    {
      return m_disp;
    }

    /**
     * @brief Is the velocity vector populated?
     * 
     * @return true if non-empty; false otherwise
     */
    TRIBOL_HOST_DEVICE bool hasVelocity() const { return !m_vel.empty(); }

    /**
     * @brief Get the nodal velocity array views
     * 
     * @return array view of the nodal velocity arrays
     */
    TRIBOL_HOST_DEVICE const MultiViewArrayView<const RealT>& getVelocity() const
    {
      return m_vel;
    }
    
    /**
     * @brief Is the nodal response vector populated?
     * 
     * @return true if non-empty; false otherwise
     */
    TRIBOL_HOST_DEVICE bool hasResponse() const { return !m_response.empty(); }

    /**
     * @brief Get the nodal response array views
     * 
     * @return array view of the nodal response arrays
     */
    TRIBOL_HOST_DEVICE const MultiViewArrayView<RealT>& getResponse() const
    {
      return m_response;
    }
    
    /**
     * @brief Is the nodal normal vector populated?
     * 
     * @return true if non-empty; false otherwise
     */
    TRIBOL_HOST_DEVICE bool hasNodalNormals() const { return !m_node_n.empty(); }

    /**
     * @brief Get an array view of the nodal normals
     * 
     * @return array view of the nodal normals
     */
    TRIBOL_HOST_DEVICE const Array2DView<RealT>& getNodalNormals() const
    {
      return m_node_n;
    }

    /**
     * @brief Is the element centroids vector populated?
     * 
     * @return true if non-empty; false otherwise
     */
    TRIBOL_HOST_DEVICE bool hasElementCentroids() const { return !m_c.empty(); }

    /**
     * @brief Get an array view of the element centroids
     * 
     * @return array view of element centroids
     */
    TRIBOL_HOST_DEVICE const Array2DView<RealT>& getElementCentroids() const
    {
      return m_c;
    }

    /**
     * @brief Is the element normals vector populated?
     * 
     * @return true if non-empty; false otherwise
     */
    TRIBOL_HOST_DEVICE bool hasElementNormals() const { return !m_n.empty(); }

    /**
     * @brief Get an array view of the element normals
     * 
     * @return array view of the element normals
     */
    TRIBOL_HOST_DEVICE const Array2DView<RealT>& getElementNormals() const
    {
      return m_n;
    }

    /**
     * @brief Is the element face radii vector populated?
     * 
     * @return true if non-empty; false otherwise
     */
    TRIBOL_HOST_DEVICE bool hasFaceRadius() const { return !m_face_radius.empty(); }

    /**
     * @brief Get an array view of the element face radii
     * 
     * @return array view of the element face radii
     */
    TRIBOL_HOST_DEVICE const Array1DView<RealT>& getFaceRadius() const
    {
      return m_face_radius;
    }

    /**
     * @brief Is the element area vector populated?
     * 
     * @return true if non-empty; false otherwise
     */
    TRIBOL_HOST_DEVICE bool hasElementAreas() const { return !m_area.empty(); }

    /**
     * @brief Get an array view of the element areas
     * 
     * @return array view of the element areas
     */
    TRIBOL_HOST_DEVICE const Array1DView<RealT>& getElementAreas() const
    {
      return m_area;
    }

    /**
     * @brief Get an array view of the element connectivity
     * 
     * @return array view of element connectivity
     */
    TRIBOL_HOST_DEVICE const Array2DView<const IndexT>& getConnectivity() const 
    {
      return m_connectivity;
    }
  
    /*!
    *
    * \brief returns pointer to array of stacked nodal coordinates for given face
    *
    * \param [in] face_id integer id of face
    * \param [in/out] coords pointer to an array of stacked (x,y,z) nodal coordinates
    *
    */
    TRIBOL_HOST_DEVICE void getFaceCoords( IndexT face_id, RealT* coords ) const;

    /*!
    *
    * \brief returns pointer to array of stacked nodal velocities for given face
    *
    * \param [in] face_id integer id of face
    * \param [in/out] nodalVel pointer to an array of stacked (x,y,z) nodal velocities
    *
    */
    TRIBOL_HOST_DEVICE void getFaceVelocities( IndexT face_id, RealT* vels ) const;

    /*!
    *
    * \brief returns pointer to array of stacked normal components 
    *
    * \param [in] face_id integer id of face
    * \param [in/out] nrml pointer to array of stacked components of the face normal vector
    *
    */
    TRIBOL_HOST_DEVICE void getFaceNormal( IndexT face_id, RealT* nrml ) const;
    
  private:
    /// Unique mesh ID
    const IndexT m_mesh_id;

    /// Type of elements in the mesh
    const InterfaceElementType m_element_type;

    /// Number of nodes in the mesh
    const IndexT m_num_nodes;

    /// Memory space of the mesh data
    const MemorySpace m_mem_space;

    /// Umpire allocator ID of the memory space (0 if no Umpire)
    const int m_allocator_id;

    /// Array of views of nodal position data
    const MultiViewArrayView<const RealT> m_position;
    
    /// Array of views of nodal displacement data
    const MultiViewArrayView<const RealT> m_disp;

    /// Array of views of nodal velocity data
    const MultiViewArrayView<const RealT> m_vel;

    /// Array of views of nodal response data
    const MultiViewArrayView<RealT> m_response;

    /// Array view of 2D nodal normal data
    const Array2DView<RealT> m_node_n;

    /// Array view of 2D element connectivity data
    const Array2DView<const IndexT> m_connectivity;

    /// Array view of element centroid data
    const Array2DView<RealT> m_c;

    /// Array view of 2D element normal data
    const Array2DView<RealT> m_n;

    /// Array view of element face radius data
    const ArrayViewT<RealT> m_face_radius;

    /// Array view of element area data
    const ArrayViewT<RealT> m_area;
    
    MeshNodalData m_nodal_fields; ///< method specific nodal fields
    MeshElemData  m_element_data; ///< method/enforcement specific element data

  }; // end class MeshData::Viewer

  /**
   * @brief Construct a new MeshData object
   * 
   * \param [in] mesh_id the ID of the contact surface
   * \param [in] num_elements the number of elements on the contact surface
   * \param [in] num_nodes length of the data arrays being registered
   * \param [in] connectivity mesh connectivity array for the contact surface
   * \param [in] element_type the cell type of the contact surface elements
   * \param [in] x array of x-components of the mesh coordinates
   * \param [in] y array of y-components of the mesh coordinates
   * \param [in] z array of z-components of the mesh coordinates (3D only)
   * \param [in] m_space Memory space of the connectivity and coordinate arrays
   *
   * \pre connectivity != nullptr
   * \pre x != nullptr
   * \pre y != nullptr
   * \pre z != nullptr (3D only)
   *
   * \note connectivity is a 2D array with num_elements rows and num_nodes
   * columns with row-major ordering
   */
  MeshData( IndexT mesh_id,
            IndexT num_elements,
            IndexT num_nodes,
            const IndexT* connectivity,
            InterfaceElementType element_type,
            const RealT* x,
            const RealT* y,
            const RealT* z,
            MemorySpace mem_space );

  /**
   * @brief Get the element type
   *
   * @note Tribol supports a single element type for each mesh
   * 
   * @return element type
   */
  InterfaceElementType getElementType() const { return m_element_type; }
  
  /**
   * @brief Spatial dimension of the mesh
   * 
   * @return spatial dimension
   */
  int spatialDimension() const { return m_dim; }

  /**
   * @brief Get the memory space of nodal/element data stored in the mesh
   * 
   * @return memory space
   */
  MemorySpace getMemorySpace() const { return m_mem_space; }

  /**
   * @brief Get the allocator ID of the nodal/element data stored in the mesh
   *
   * @note Corresponds to an umpire allocator ID if Tribol is built with
   * Umpire; zero otherwise
   * 
   * @return allocator ID
   */
  int getAllocatorId() const { return m_allocator_id; }

  /**
   * @brief Set the allocator ID of the nodal/element data stored in the mesh
   * 
   * @param allocator_id Umpire allocator ID (if built with Umpire; zero otherwise)
   */
  void updateAllocatorId(int allocator_id ) { m_allocator_id = allocator_id; }

  /**
   * @brief Marker which can indicate mesh validity
   *
   * @note The MeshData constructor verifies coordinate data is not null for
   * non-empty meshes. If null, the mesh is marked as not valid.
   *
   * @warning The marker can be updated outside MeshData so the definition of a
   * valid mesh may not be consistent.
   * 
   * @return true the marker is set to true
   * @return false the marker is set to false
   */
  bool& isMeshValid() { return m_is_valid; }

  /**
   * @brief Get the mesh nodal field data
   * 
   * @return nodal data for the mesh
   */
  MeshNodalData& getNodalFields() { return m_nodal_fields; }
  
  /**
   * @brief Get the mesh element data
   * 
   * @return element data for the mesh
   */
  MeshElemData& getElementData() { return m_element_data; }

  /// @overload
  const MeshElemData& getElementData() const { return m_element_data; }
  
  /**
   * @brief Number of nodes in the mesh
   * 
   * @return node count
   */
  IndexT numberOfNodes() const { return m_num_nodes; }

  /**
   * @brief Number of elements in the mesh
   * 
   * @return element count
   */
  IndexT numberOfElements() const { return m_connectivity.shape()[0]; }

  /**
   * @brief Number of nodes in each element of the mesh
   * 
   * @return nodes per element
   */
  IndexT numberOfNodesPerElement() const { return m_connectivity.shape()[1]; }

  /**
   * @brief Get the global node ID
   * 
   * @param element_id which element the node belongs to
   * @param local_node_id node ID for the local element
   * @return global node ID
   */
  IndexT getGlobalNodeId(IndexT element_id, IndexT local_node_id) const
  {
    return m_connectivity(element_id, local_node_id);
  }

  /**
   * @brief Set the pointers to the nodal position data
   * 
   * @param x array of x-components of the nodal position
   * @param y array of y-components of the nodal position
   * @param z array of z-components of the nodal position
   */
  void setPosition( const RealT* x,
                    const RealT* y,
                    const RealT* z );

  /**
   * @brief Set the pointers to the nodal displacement data
   * 
   * @param ux array of x-components of the nodal displacement
   * @param uy array of y-components of the nodal displacement
   * @param uz array of z-components of the nodal displacement
   */
  void setDisplacement( const RealT* ux,
                        const RealT* uy,
                        const RealT* uz );

  /**
   * @brief Set the pointers to the nodal velocity data
   * 
   * @param vx array of x-components of the nodal velocity
   * @param vy array of y-components of the nodal velocity
   * @param vz array of z-components of the nodal velocity
   */
  void setVelocity( const RealT* vx,
                    const RealT* vy,
                    const RealT* vz );
  
  /**
   * @brief Is the velocity vector populated?
   * 
   * @return true vector is non-empty
   * @return false vector is empty
   */
  bool hasVelocity() const { return !m_vel.empty(); }

  /**
   * @brief Set the pointers to the nodal response data
   * 
   * @param rx array of x-components of the nodal response
   * @param ry array of y-components of the nodal response
   * @param rz array of z-components of the nodal response
   */
  void setResponse( RealT* rx, RealT* ry, RealT* rz );

  /**
   * @brief Construct a non-owned, shallow copy of the MeshData
   * 
   * @return MeshData::Viewer type
   */
  Viewer getView() { return *this; }

  /// sorts unique surface node ids from connectivity and stores them on the mesh object in ascending order
  Array1D<IndexT> sortSurfaceNodeIds();

private:
  /**
   * @brief Converts a Tribol element type to a mesh spatial dimension
   * 
   * @return spatial dimension
   */
  int getDimFromElementType() const;

  /**
   * @brief Converts pointers to components of a vector to views of the vector
   * components
   * 
   * @tparam T underlying type of the components
   * @param x pointer to array of x-components
   * @param y pointer to array of y-components
   * @param z pointer to array of z-components
   * @return Array of array views of vector components
   */
  template <typename T>
  MultiArrayView<T> createNodalVector( T* x, 
                                       T* y,
                                       T* z ) const;

  /**
   * @brief Converts pointer to element connectivity to an array view
   * 
   * @param num_elements element count; number of rows in connectivity array
   * @param connectivity pointer to array of connectivity data
   * @return Array view of element connectivity
   */
  Array2DView<const IndexT> createConnectivity( IndexT num_elements, 
                                                  const IndexT* connectivity );

  IndexT m_mesh_id;                    ///< Mesh Id associated with this data
  InterfaceElementType m_element_type; ///< Type of interface element in mesh
  int m_dim;                           ///< Spatial dimension of the mesh coordinates
  IndexT m_num_nodes;

  MemorySpace m_mem_space;             ///< Memory space for mesh data
  int m_allocator_id;                  ///< Allocator for mesh data memory
  
  bool m_is_valid;                     ///< True if the mesh is valid

  MeshNodalData m_nodal_fields;        ///< method specific nodal fields
  MeshElemData  m_element_data;        ///< method/enforcement specific element data

  // Nodal field data
  MultiArrayView<const RealT> m_position; ///< Coordinates of nodes in mesh
  MultiArrayView<const RealT> m_disp;     ///< Nodal displacements
  MultiArrayView<const RealT> m_vel;      ///< Nodal velocity
  MultiArrayView<RealT> m_response;       ///< Nodal responses (forces)

  Array2D<RealT> m_node_n;             ///< Outward unit node normals

  // Element field data
  Array2DView<const IndexT> m_connectivity;  ///< Element connectivity arrays

  Array2D<RealT> m_c;           ///< Vertex averaged element centroids
  Array2D<RealT> m_n;           ///< Outward unit element normals
  Array1D<RealT> m_face_radius; ///< Face radius used in low level proximity check
  Array1D<RealT> m_area;        ///< Element areas

public:

  /*!
  * \brief Checks for valid Lagrange multiplier enforcement data
  *
  */
  int checkLagrangeMultiplierData();

  /*!
  * \brief Checks for valid penalty enforcement data 
  *
  * \param [in] p_enfrc_options penalty enforcement options guiding check
  */
  int checkPenaltyData( PenaltyEnforcementOptions& p_enfrc_options );

  /*!
  * \brief Computes the face normals and centroids for all faces in the mesh
  *
  * \param [in] exec_mode defines where loops should be executed
  * \return true if face calculations do not encounter errors or warnings
  * 
  * This routine accounts for warped faces by computing an average normal.
  */
  bool computeFaceData(ExecutionMode exec_mode);
  
  /*!
  * \brief Computes average nodal normals for use with mortar methods
  *
  * \note this routine computes average nodal normals for all nodes in the mesh.
  *
  * \param [in] dim Dimension of the problem
  */
  void computeNodalNormals( int const dim );
  
  /*!
  *
  * \brief compute the surface edge/segment length
  *
  * \param [in] edgeId edge id
  *
  * \return edge length
  *
  */
  RealT computeEdgeLength( int edgeId );

  /// Prints information associated with this mesh to \a os
  void print(std::ostream& os) const;

}; // end class MeshData

//------------------------------------------------------------------------------
template <typename T>
MultiArrayView<T> MeshData::createNodalVector( T* x, T* y, T* z ) const
{
  MultiArrayView<T> host_nodal_vector(m_dim, m_dim);
  host_nodal_vector[0] = ArrayViewT<T>(x, m_num_nodes);
  host_nodal_vector[1] = ArrayViewT<T>(y, m_num_nodes);
  if (m_dim == 3)
  {
    host_nodal_vector[2] = ArrayViewT<T>(z, m_num_nodes);
  }
  return MultiArrayView<T>(host_nodal_vector, m_allocator_id);
}

using MeshManager = DataManager<MeshData>;

} // end namespace tribol

/// \a ostream operator to print a \a MeshData instance to \a os
std::ostream& operator<<(std::ostream& os, const tribol::MeshData& md);

#endif /* SRC_MESH_MESHDATA_HPP_ */
