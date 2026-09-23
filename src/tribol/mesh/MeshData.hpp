// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_MESH_MESHDATA_HPP_
#define SRC_TRIBOL_MESH_MESHDATA_HPP_

// C++ includes
#include <ostream>

// Shared includes
#include "tribol/common/ExecModel.hpp"

// Tribol includes
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/utils/DataManager.hpp"

namespace tribol {

/*!
 * \brief Struct to hold method specific nodal fields
 */
struct MeshNodalData {
  int m_num_nodes;

  /////////////////////////
  // MORTAR NODAL FIELDS //
  /////////////////////////
  ArrayViewT<RealT> m_node_gap;             ///< scalar nodal gap (used on nonmortar mesh)
  ArrayViewT<const RealT> m_node_pressure;  ///< scalar nodal pressure (used on nonmortar mesh)

  bool m_is_node_gap_set{ false };       ///< true if nodal gap field is set
  bool m_is_node_pressure_set{ false };  ///< true if nodal pressure field is set
  /////////////////////////

  bool m_is_velocity_set{ false };            ///< true if nodal velocities have been registered
  bool m_is_nodal_displacement_set{ false };  ///< true if nodal displacements have been registered
  bool m_is_nodal_response_set{ false };      ///< true if the nodal responses have been registered

};  // end of struct MeshNodalData

/*!
 * \brief Struct to hold method specific element data
 */
struct MeshElemData {
  int m_num_cells;

  //////////////////////////////////////
  // PENALTY ENFORCEMENT ELEMENT DATA //
  //////////////////////////////////////
  ArrayViewT<const RealT> m_mat_mod;    ///< Bulk/Young's modulus for contact faces
  ArrayViewT<const RealT> m_thickness;  ///< Volume element thickness associated with each contact face

  RealT m_penalty_stiffness{ 0. };       ///< single scalar kinematic penalty stiffness for each mesh
  RealT m_penalty_scale{ 1. };           ///< scale factor applied to kinematic penalty only
  RealT m_rate_penalty_stiffness{ 0. };  ///< single scalar rate penalty stiffness for each mesh
  RealT m_rate_percent_stiffness{ 0. };  ///< rate penalty is percentage of gap penalty

  RealT m_viscous_damping_coeff{ 0. };

  bool m_is_kinematic_constant_penalty_set{ false };  ///< True if single kinematic constant penalty is set
  bool m_is_kinematic_element_penalty_set{ false };   ///< True if the element-wise kinematic penalty is set
  bool m_is_rate_constant_penalty_set{ false };       ///< True if the constant rate penalty is set
  bool m_is_rate_percent_penalty_set{ false };        ///< True if the rate percent penalty is set
  bool m_is_viscous_damping_coeff_set{ false };

  bool m_is_element_thickness_set{ false };  ///< True if element thickness is set

  /*!
   * \brief Checks if the kinematic penalty data is valid
   *
   * \param [in] pen_enfrc penalty enforcement option struct
   *
   * \return true if the kinematic penalty option has valid data
   */
  bool isValidKinematicPenalty( PenaltyEnforcementOptions& pen_options, ExecutionMode exec_mode, int alloc_id );

  /*!
   * \brief Checks if the rate penalty data is valid
   *
   * \param [in] pen_enfrc to penalty enforcement option struct
   *
   * \return true if the rate penalty option has valid data
   */
  bool isValidRatePenalty( PenaltyEnforcementOptions& pen_options );  ///< True if rate penalty option is valid
};

/**
 * @brief Non-owning views of native parent-face mapping data for contact surface elements.
 *
 * MFEM contact may use a low-order-refined (LOR) surface for search and overlap
 * geometry.  Each Tribol surface element then needs a documented path back to
 * the native parent boundary face.  The arrays in this structure are indexed by
 * the Tribol surface element identifier and remain valid until the next MFEM
 * parallel-decomposition update.
 */
struct ParentFaceData {
  /** Tolerance for validating mapped native parent reference coordinates. */
  static constexpr RealT reference_coordinate_tolerance{ 1.e-12 };

  /** Maximum supported dimension of a parent-face reference coordinate. */
  static constexpr int max_reference_dimension{ 2 };

  /** Maximum number of vertices on a supported LOR surface element. */
  static constexpr int max_lor_face_vertices{ 4 };

  /** MFEM geometry identifier for each LOR face. */
  Array1DView<const int> m_lor_face_geometries;

  /** Polynomial order of the native parent coordinate finite element. */
  Array1DView<const int> m_parent_face_orders;

  /** Number of vertices used by each child-to-parent reference map. */
  Array1DView<const int> m_reference_vertex_counts;

  /**
   * Parent reference coordinates at the LOR face vertices.
   *
   * The second index uses vertex-major ordering with
   * `max_reference_dimension` entries per vertex.
   */
  Array2DView<const RealT> m_parent_reference_vertex_coordinates;

  /**
   * @brief Return whether native parent-face mapping data are available.
   *
   * @return true when all required mapping arrays are populated
   */
  TRIBOL_HOST_DEVICE bool isValid() const
  {
    return !m_lor_face_geometries.empty() && !m_parent_face_orders.empty() && !m_reference_vertex_counts.empty() &&
           !m_parent_reference_vertex_coordinates.empty();
  }
};

class MeshData {
 public:
  /**
   * @brief Nested class for holding views (non-owned, shallow copies) of mesh data
   */
  class Viewer {
   public:
    /**
     * @brief Construct a new MeshData::Viewer object
     *
     * @param mesh MeshData to create a view of
     */
    Viewer( MeshData& mesh );

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
     * @brief Return whether native parent-face mapping data are registered.
     *
     * @return true when this mesh has valid parent-face mapping data
     */
    TRIBOL_HOST_DEVICE bool hasParentFaceData() const { return m_parent_face_data.isValid(); }

    /**
     * @brief Get native parent-face mapping data for this mesh.
     *
     * @return non-owning views of parent-face mapping arrays
     */
    TRIBOL_HOST_DEVICE const ParentFaceData& getParentFaceData() const { return m_parent_face_data; }

    /**
     * @brief Map a LOR face reference point to its native parent-face reference point.
     *
     * The mapping interpolates the stored parent reference coordinates at the
     * LOR face vertices. Segment, triangle, and quadrilateral LOR faces are
     * supported.
     *
     * @param face_id Tribol surface element identifier
     * @param lor_reference_coordinates LOR face reference coordinates
     * @param parent_reference_coordinates Mapped native parent-face reference coordinates
     * @return true when the face mapping and geometry are valid
     */
    TRIBOL_HOST_DEVICE bool mapToParentReference( IndexT face_id, const RealT* lor_reference_coordinates,
                                                  RealT* parent_reference_coordinates ) const;

    /**
     * @brief Spatial dimension of the mesh
     *
     * @return spatial dimension
     */
    TRIBOL_HOST_DEVICE int spatialDimension() const { return static_cast<int>( m_position.size() ); }

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
    TRIBOL_HOST_DEVICE IndexT getGlobalNodeId( IndexT element_id, IndexT local_node_id ) const
    {
      return m_connectivity( element_id, local_node_id );
    }

    /**
     * @brief Get the nodal position array views
     *
     * @return array view of the nodal position arrays
     */
    TRIBOL_HOST_DEVICE const MultiViewArrayView<const RealT>& getPosition() const { return m_position; }

    /**
     * @brief Is the reference position vector populated?
     *
     * @return true if non-empty; false otherwise
     */
    TRIBOL_HOST_DEVICE bool hasReferencePosition() const { return !m_ref_position.empty(); }

    /**
     * @brief Get the nodal reference position array views
     *
     * @return array view of the nodal reference position arrays
     */
    TRIBOL_HOST_DEVICE const MultiViewArrayView<const RealT>& getReferencePosition() const { return m_ref_position; }

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
    TRIBOL_HOST_DEVICE const MultiViewArrayView<const RealT>& getDisplacement() const { return m_disp; }

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
    TRIBOL_HOST_DEVICE const MultiViewArrayView<const RealT>& getVelocity() const { return m_vel; }

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
    TRIBOL_HOST_DEVICE const MultiViewArrayView<RealT>& getResponse() const { return m_response; }

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
    TRIBOL_HOST_DEVICE const Array2DView<RealT>& getNodalNormals() const { return m_node_n; }

    /// @overload
    TRIBOL_HOST_DEVICE Array2DView<RealT>& getNodalNormals() { return m_node_n; }

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
    TRIBOL_HOST_DEVICE const Array2DView<RealT>& getElementCentroids() const { return m_c; }

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
    TRIBOL_HOST_DEVICE const Array2DView<RealT>& getElementNormals() const { return m_n; }

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
    TRIBOL_HOST_DEVICE const Array1DView<RealT>& getFaceRadius() const { return m_face_radius; }

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
    TRIBOL_HOST_DEVICE const Array1DView<RealT>& getElementAreas() const { return m_area; }

    /**
     * @brief Get an array view of the element connectivity
     *
     * @return array view of element connectivity
     */
    TRIBOL_HOST_DEVICE const Array2DView<const IndexT>& getConnectivity() const { return m_connectivity; }

    /*!
     *
     * \brief returns pointer to array of stacked nodal coordinates for given face
     *
     * \param [in] face_id integer id of face
     * \param [in/out] coords pointer to an array of stacked (x,y,z) nodal coordinates
     *
     */
    TRIBOL_HOST_DEVICE inline void getFaceCoords( IndexT face_id, RealT* coords ) const;

    /*!
     *
     * \brief returns pointer to array of stacked nodal velocities for given face
     *
     * \param [in] face_id integer id of face
     * \param [in/out] nodalVel pointer to an array of stacked (x,y,z) nodal velocities
     *
     */
    TRIBOL_HOST_DEVICE inline void getFaceVelocities( IndexT face_id, RealT* vels ) const;

    /*!
     *
     * \brief returns pointer to array of stacked normal components
     *
     * \param [in] face_id integer id of face
     * \param [in/out] nrml pointer to array of stacked components of the face normal vector
     *
     */
    TRIBOL_HOST_DEVICE inline void getFaceNormal( IndexT face_id, RealT* nrml ) const;

    /*!
     *
     * \brief returns pointer to array of stacked face centroid components
     *
     * \param [in] face_id integer id of face
     * \param [in/out] cx pointer to array of stacked components of the face centroid components
     *
     */
    TRIBOL_HOST_DEVICE inline void getFaceCentroid( IndexT face_id, RealT* cx ) const;

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

    /// Array of views of nodal reference position data
    const MultiViewArrayView<const RealT> m_ref_position;

    /// Array of views of nodal displacement data
    const MultiViewArrayView<const RealT> m_disp;

    /// Array of views of nodal velocity data
    const MultiViewArrayView<const RealT> m_vel;

    /// Array of views of nodal response data
    const MultiViewArrayView<RealT> m_response;

    /// Array view of 2D nodal normal data
    Array2DView<RealT> m_node_n;

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

    MeshNodalData m_nodal_fields;  ///< method specific nodal fields
    MeshElemData m_element_data;   ///< method/enforcement specific element data

    /** Native parent-face mapping data for the contact surface elements. */
    const ParentFaceData m_parent_face_data;

  };  // end class MeshData::Viewer

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
  MeshData( IndexT mesh_id, IndexT num_elements, IndexT num_nodes, const IndexT* connectivity,
            InterfaceElementType element_type, const RealT* x, const RealT* y, const RealT* z, MemorySpace mem_space );

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
  void updateAllocatorId( int allocator_id ) { m_allocator_id = allocator_id; }

  /**
   * @brief Register native parent-face mapping data using an existing collection of views.
   *
   * @param parent_face_data Non-owning parent-face mapping views
   */
  void setParentFaceData( const ParentFaceData& parent_face_data ) { m_parent_face_data = parent_face_data; }

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
  IndexT getGlobalNodeId( IndexT element_id, IndexT local_node_id ) const
  {
    return m_connectivity( element_id, local_node_id );
  }

  /**
   * @brief Set the pointers to the nodal position data
   *
   * @param x array of x-components of the nodal position
   * @param y array of y-components of the nodal position
   * @param z array of z-components of the nodal position
   */
  void setPosition( const RealT* x, const RealT* y, const RealT* z );

  /**
   * @brief Set the pointers to the nodal reference position data
   *
   * @param xref array of x-components of the nodal reference position
   * @param yref array of y-components of the nodal reference position
   * @param zref array of z-components of the nodal reference position
   */
  void setReferencePosition( const RealT* xref, const RealT* yref, const RealT* zref );

  /**
   * @brief Is the reference position vector populated?
   *
   * @return true vector is non-empty
   * @return false vector is empty
   */
  bool hasReferencePosition() const { return !m_ref_position.empty(); }

  /**
   * @brief Set the pointers to the nodal displacement data
   *
   * @param ux array of x-components of the nodal displacement
   * @param uy array of y-components of the nodal displacement
   * @param uz array of z-components of the nodal displacement
   */
  void setDisplacement( const RealT* ux, const RealT* uy, const RealT* uz );

  /**
   * @brief Set the pointers to the nodal velocity data
   *
   * @param vx array of x-components of the nodal velocity
   * @param vy array of y-components of the nodal velocity
   * @param vz array of z-components of the nodal velocity
   */
  void setVelocity( const RealT* vx, const RealT* vy, const RealT* vz );

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
  MultiArrayView<T> createNodalVector( T* x, T* y, T* z ) const;

  /**
   * @brief Converts pointer to element connectivity to an array view
   *
   * @param num_elements element count; number of rows in connectivity array
   * @param connectivity pointer to array of connectivity data
   * @return Array view of element connectivity
   */
  Array2DView<const IndexT> createConnectivity( IndexT num_elements, const IndexT* connectivity );

  IndexT m_mesh_id;                     ///< Mesh Id associated with this data
  InterfaceElementType m_element_type;  ///< Type of interface element in mesh
  int m_dim;                            ///< Spatial dimension of the mesh coordinates
  IndexT m_num_nodes;

  MemorySpace m_mem_space;  ///< Memory space for mesh data
  int m_allocator_id;       ///< Allocator for mesh data memory

  bool m_is_valid;  ///< True if the mesh is valid

  MeshNodalData m_nodal_fields;  ///< method specific nodal fields
  MeshElemData m_element_data;   ///< method/enforcement specific element data

  /** Non-owning native parent-face mapping data registered by the MFEM interface. */
  ParentFaceData m_parent_face_data;

  // Nodal field data
  MultiArrayView<const RealT> m_position;      ///< Coordinates of nodes in mesh
  MultiArrayView<const RealT> m_ref_position;  ///< Reference coordinates of nodes in mesh
  MultiArrayView<const RealT> m_disp;          ///< Nodal displacements
  MultiArrayView<const RealT> m_vel;           ///< Nodal velocity
  MultiArrayView<RealT> m_response;            ///< Nodal responses (forces)

  ArrayT<RealT, 2> m_node_n;  ///< Outward unit node normals

  // Element field data
  Array2DView<const IndexT> m_connectivity;  ///< Element connectivity arrays

  ArrayT<RealT, 2> m_c;          ///< Vertex averaged element centroids
  ArrayT<RealT, 2> m_n;          ///< Outward unit element normals
  Array1D<RealT> m_face_radius;  ///< Face radius used in low level proximity check
  Array1D<RealT> m_area;         ///< Element areas

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
  int checkPenaltyData( PenaltyEnforcementOptions& p_enfrc_options, ExecutionMode exec_mode );

  /*!
   * \brief Computes the face normals and centroids for all faces in the mesh
   *
   * \param [in] exec_mode defines where loops should be executed
   * \return true if face calculations do not encounter errors or warnings
   *
   * This routine accounts for warped faces by computing an average normal.
   */
  template <typename ElemNormalMethod>
  bool computeFaceData( ExecutionMode exec_mode, ElemNormalMethod elem_normal );

  /**
   * @brief Allocates and initializes memory to hold nodal normals
   */
  void allocateNodalNormals();

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
  void print( std::ostream& os ) const;

};  // end class MeshData

//------------------------------------------------------------------------------
template <typename T>
MultiArrayView<T> MeshData::createNodalVector( T* x, T* y, T* z ) const
{
  MultiArrayView<T> host_nodal_vector( m_dim, m_dim );
  host_nodal_vector[0] = ArrayViewT<T>( x, m_num_nodes );
  host_nodal_vector[1] = ArrayViewT<T>( y, m_num_nodes );
  if ( m_dim == 3 ) {
    host_nodal_vector[2] = ArrayViewT<T>( z, m_num_nodes );
  }
  return MultiArrayView<T>( host_nodal_vector, m_allocator_id );
}

using MeshManager = DataManager<MeshData>;

//-----------------------------------------------------------------------------
// Implementations
//-----------------------------------------------------------------------------

TRIBOL_HOST_DEVICE inline void MeshData::Viewer::getFaceCoords( IndexT face_id, RealT* coords ) const
{
  auto dim = spatialDimension();

  for ( IndexT a{ 0 }; a < numberOfNodesPerElement(); ++a ) {
    IndexT node_id = getGlobalNodeId( face_id, a );
    for ( int d{ 0 }; d < dim; ++d ) {
      coords[dim * a + d] = m_position[d][node_id];
    }
  }

  return;

}  // end MeshData::Viewer::getFaceCoords()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE inline void MeshData::Viewer::getFaceVelocities( IndexT face_id, RealT* vels ) const
{
  auto dim = spatialDimension();

  for ( IndexT a{ 0 }; a < numberOfNodesPerElement(); ++a ) {
    IndexT node_id = getGlobalNodeId( face_id, a );
    for ( int d{ 0 }; d < dim; ++d ) {
      vels[dim * a + d] = m_vel[d][node_id];
    }
  }

  return;

}  // end MeshData::Viewer::getFaceVelocities()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE inline void MeshData::Viewer::getFaceNormal( IndexT face_id, RealT* nrml ) const
{
  for ( int d{ 0 }; d < spatialDimension(); ++d ) {
    nrml[d] = m_n[d][face_id];
  }
  return;

}  // end MeshData::getFaceNormal()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE inline void MeshData::Viewer::getFaceCentroid( IndexT face_id, RealT* cx ) const
{
  for ( int d{ 0 }; d < spatialDimension(); ++d ) {
    cx[d] = m_c[d][face_id];
  }
  return;

}  // end MeshData::Viewer::getFaceCentroid()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE inline bool MeshData::Viewer::mapToParentReference( IndexT face_id,
                                                                       const RealT* lor_reference_coordinates,
                                                                       RealT* parent_reference_coordinates ) const
{
  if ( !hasParentFaceData() || face_id < 0 || face_id >= numberOfElements() || lor_reference_coordinates == nullptr ||
       parent_reference_coordinates == nullptr || face_id >= m_parent_face_data.m_lor_face_geometries.size() ||
       face_id >= m_parent_face_data.m_parent_face_orders.size() ||
       face_id >= m_parent_face_data.m_reference_vertex_counts.size() ||
       face_id >= m_parent_face_data.m_parent_reference_vertex_coordinates.shape()[0] ||
       m_parent_face_data.m_parent_reference_vertex_coordinates.shape()[1] <
           ParentFaceData::max_lor_face_vertices * ParentFaceData::max_reference_dimension ) {
    return false;
  }

  const int reference_dimension = spatialDimension() - 1;
  const int number_of_vertices = m_parent_face_data.m_reference_vertex_counts[face_id];
  const auto lor_face_geometry = static_cast<InterfaceElementType>( m_parent_face_data.m_lor_face_geometries[face_id] );
  const RealT reference_coordinate_tolerance = ParentFaceData::reference_coordinate_tolerance;
  const RealT first_coordinate = lor_reference_coordinates[0];

  if ( first_coordinate < -reference_coordinate_tolerance || first_coordinate > 1.0 + reference_coordinate_tolerance ) {
    return false;
  }

  RealT shape_values[ParentFaceData::max_lor_face_vertices] = { 0.0, 0.0, 0.0, 0.0 };
  if ( lor_face_geometry == LINEAR_EDGE && number_of_vertices == 2 && reference_dimension == 1 ) {
    shape_values[0] = 1.0 - first_coordinate;
    shape_values[1] = first_coordinate;
  } else if ( lor_face_geometry == LINEAR_TRIANGLE && number_of_vertices == 3 && reference_dimension == 2 ) {
    const RealT second_coordinate = lor_reference_coordinates[1];
    if ( second_coordinate < -reference_coordinate_tolerance ||
         first_coordinate + second_coordinate > 1.0 + reference_coordinate_tolerance ) {
      return false;
    }
    shape_values[0] = 1.0 - first_coordinate - second_coordinate;
    shape_values[1] = first_coordinate;
    shape_values[2] = second_coordinate;
  } else if ( lor_face_geometry == LINEAR_QUAD && number_of_vertices == 4 && reference_dimension == 2 ) {
    const RealT second_coordinate = lor_reference_coordinates[1];
    if ( second_coordinate < -reference_coordinate_tolerance ||
         second_coordinate > 1.0 + reference_coordinate_tolerance ) {
      return false;
    }
    shape_values[0] = ( 1.0 - first_coordinate ) * ( 1.0 - second_coordinate );
    shape_values[1] = first_coordinate * ( 1.0 - second_coordinate );
    shape_values[2] = first_coordinate * second_coordinate;
    shape_values[3] = ( 1.0 - first_coordinate ) * second_coordinate;
  } else {
    return false;
  }

  for ( int coordinate_component = 0; coordinate_component < reference_dimension; ++coordinate_component ) {
    parent_reference_coordinates[coordinate_component] = 0.0;
    for ( int vertex_index = 0; vertex_index < number_of_vertices; ++vertex_index ) {
      const int coordinate_index = vertex_index * ParentFaceData::max_reference_dimension + coordinate_component;
      parent_reference_coordinates[coordinate_component] +=
          shape_values[vertex_index] *
          m_parent_face_data.m_parent_reference_vertex_coordinates( face_id, coordinate_index );
    }
    if ( parent_reference_coordinates[coordinate_component] < -reference_coordinate_tolerance ||
         parent_reference_coordinates[coordinate_component] > 1.0 + reference_coordinate_tolerance ) {
      return false;
    }
  }
  if ( lor_face_geometry == LINEAR_TRIANGLE &&
       parent_reference_coordinates[0] + parent_reference_coordinates[1] > 1.0 + reference_coordinate_tolerance ) {
    return false;
  }
  return true;
}

}  // end namespace tribol

/// \a ostream operator to print a \a MeshData instance to \a os
std::ostream& operator<<( std::ostream& os, const tribol::MeshData& md );

#endif /* SRC_TRIBOL_MESH_MESHDATA_HPP_ */
