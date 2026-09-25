// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_MESH_METHODCOUPLINGDATA_HPP_
#define SRC_TRIBOL_MESH_METHODCOUPLINGDATA_HPP_

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Containers.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"

// MFEM includes
#include "mfem.hpp"

namespace tribol {

// Forward Declarations
class InterfacePairs;

//------------------------------------------------------------------------------
/*!
 * \brief Struct to hold data associated with a surface
 *        contact element
 */
struct SurfaceContactElem {
  enum JacBlock
  {
    JguBlock,
    JrpBlock
  };

  /// Default constructor
  SurfaceContactElem();

  /// Overloaded Constructor
  TRIBOL_HOST_DEVICE SurfaceContactElem( int dimension,                  ///< [in] Dimension of the problem
                                         RealT* x1,                      ///< [in] Vertex coordinates of first face
                                         RealT* x2,                      ///< [in] Vertex coordinates of second face
                                         RealT* xOverlap,                ///< [in] Vertex coordinates of overlap
                                         int nFV,                        ///< [in] Number of face vertices
                                         int nPV,                        ///< [in] Number of overlap vertices
                                         const MeshData::Viewer* mesh1,  ///< [in] View of mesh 1
                                         const MeshData::Viewer* mesh2,  ///< [in] View of mesh 2
                                         int fId1,                       ///< [in] Id for face 1
                                         int fId2                        ///< [in] Id for face 2
                                         )
      : dim( dimension ),
        m_mesh1( mesh1 ),
        m_mesh2( mesh2 ),
        faceId1( fId1 ),
        faceId2( fId2 ),
        faceCoords1( x1 ),
        faceCoords2( x2 ),
        faceNormal1( nullptr ),
        faceNormal2( nullptr ),
        overlapCoords( xOverlap ),
        overlapNormal( nullptr ),
        numFaceVert( nFV ),
        numPolyVert( nPV ),
        overlapArea( 0. ),
        mortarWts( nullptr ),
        numWts( 0 ),
        numActiveGaps( 0 ),
        blockJ( 3 )

  {
  }

  /// Destructor
  TRIBOL_HOST_DEVICE ~SurfaceContactElem() { this->deallocateElem(); }

  int dim;                          ///< Problem dimension
  const MeshData::Viewer* m_mesh1;  ///< Mesh view for face 1 (mortar)
  const MeshData::Viewer* m_mesh2;  ///< Mesh view for face 2 (nonmortar)
  int faceId1;                      ///< Face Id for face 1 (mortar)
  int faceId2;                      ///< Face Id for face 2 (nonmortar)
  RealT* faceCoords1;               ///< Coordinates of face 1 in 3D
  RealT* faceCoords2;               ///< Coordinates of face 2 in 3D
  RealT* faceNormal1;               ///< Components of face 1 normal
  RealT* faceNormal2;               ///< Components of face 2 normal
  RealT* overlapCoords;             ///< Coordinates of overlap vertices in 3D
  RealT* overlapNormal;             ///< Components of overlap normal
  int numFaceVert;                  ///< Number of face vertices/nodes
  int numPolyVert;                  ///< Number of overlap vertices
  RealT overlapArea;                ///< Area of polygonal overlap

  RealT* mortarWts;  ///< Stacked array of mortar wts for mortar methods
  int numWts;        ///< Number of mortar weights

  int numActiveGaps;  ///< Number of local face-pair active gaps

  StackArray<DeviceArray2D<RealT>, 9> blockJ;  ///< Block element Jacobian contributions

  /// routine to allocate space to store mortar weights
  void allocateMortarWts();

  /// routine to initialize mortar weights
  void initializeMortarWts();

  /*!
   * \brief routine to return mortar-nonmortar mortar weight
   *
   * \param [in] a MORTAR node id
   * \param [in] b NONMORTAR node id
   *
   */
  RealT getMortarNonmortarWt( const int a, const int b );

  /*!
   * \brief routine to return nonmortar-mortar mortar weights
   *
   * \param [in] a NONMORTAR node id
   * \param [in] b MORTAR node id
   *
   */
  RealT getNonmortarMortarWt( const int a, const int b );

  /*!
   * \brief routine to return nonmortar-nonmortar mortar weight
   *
   * \param [in] a NONMORTAR node id
   * \param [in] b NONMORTAR node id
   *
   */
  RealT getNonmortarNonmortarWt( const int a, const int b );

  /*!
   * \brief get array index for x-dimension face-pair Jacobian contribution
   *
   * Given a jacobian block and local node indices a (in row space) and b (in
   * col space), this method returns an integer index of the first dimension
   * (usually x-dimension) of the Jacobian array.  The index corresponds to
   * column major ordering and the inner loop is over the nodes, which matches
   * element Jacobian outputs in MFEM.
   *
   * \param [in] block JacBlock for the jacobian contribution whose index is
   * sought
   * \param [in] a row index of node in block
   * \param [in] b column index of node in block
   *
   * \return integer index for indexing into Jacobian storage array
   *
   * \note ordering is column major (matches mfem::DenseMatrix)
   *
   * \note inner loop is over nodes (mfem::Ordering::byNODES) (matches
   * mfem::Ordering)
   *
   * \note array indices for the (a, b) pair in the y-dimension and z-dimension
   *       can be obtained by adding the offset returned by
   *       getJacobianDimOffset()
   *
   */
  int getJacobianIndex( JacBlock block, const int a, const int b ) const;

  /*!
   * \brief get element-pair Jacobian array offset due to incrementing the
   * spatial dimension
   *
   * \param [in] block JacBlock for the jacobian contribution whose offset is
   * sought
   *
   * \return integer offset in Jacobian index from incrementing the spatial
   * dimension
   *
   * \note ordering is column major (matches mfem::DenseMatrix)
   *
   * \note inner loop is over nodes (mfem::Ordering::byNODES) (matches
   * mfem::Ordering)
   *
   * \note dimension offset returned here should be paired with the index of the
   * x-dimension given by getJacobianIndex()
   *
   */
  int getJacobianDimOffset( JacBlock block ) const;

  /*!
   * \brief routine to allocate space to store contact element Jacobians
   *
   * \param [in] method contact method
   *
   */
  void allocateBlockJ( EnforcementMethod enf );

  /// delete routine
  TRIBOL_HOST_DEVICE void deallocateElem()
  {
    if ( this->mortarWts != nullptr ) {
      delete[] this->mortarWts;
      this->mortarWts = nullptr;
    }
  }

};  // end of SurfaceContactElem definition

//------------------------------------------------------------------------------
class MethodData {
 public:
  /*!
   * \brief Constructor
   */
  MethodData();

  /*!
   * \brief Destructor
   */
  virtual ~MethodData() {};

  /*!
   * \brief allocate element Jacobian matrix storage
   *
   * \param [in] blockJSpaces list of block spaces used in the Jacobian matrix
   * \param [in] nPairs approximate number of contacting face-pairs (used to
   * allocate memory in the ArrayT)
   */
  void reserveBlockJ( ArrayT<BlockSpace>&& blockJSpaces, int nPairs );

  /*!
   * \brief store an element contribution to all blocks of the Jacobian matrix
   *
   * \param [in] blockJElemIds list of element ids on each block space
   ^
   * \param [in] blockJ 2D array of element Jacobian contributions (each array
   * entry corresponds to a block of the Jacobian matrix)
   */
  void storeElemBlockJ( ArrayT<int>&& blockJElemIds, const StackArray<DeviceArray2D<RealT>, 9>& blockJ );

  /*!
   * \brief Returns the number of blocks in the Jacobian matrix
   *
   * See @ref getElementBlockJacobians for a definition of the blocks.
   */
  IndexT getNSpaces() const { return m_blockJSpaces.size(); }

  /*!
   * \brief Get the element ids for each entry of the getBlockJ 2D ArrayT
   * sorted by block space
   *
   * \note Method returns a nested array. With getBlockJElementIds()[i][j], the
   * index i identifies the block space index (mapped using getBlockJSpaces())
   * and j gives the j-th element id for the block space identified by i.  For
   * example, to find the element id of the 3rd element matrix contribution on
   * the 0th block space, use getBlockJElementIds()[0][3].
   *
   * \return nested array identifying element ids for a given block space
   */
  const ArrayT<ArrayT<int>>& getBlockJElementIds() const { return m_blockJElemIds; }

  /*!
   * \brief Get element Jacobian contributions sorted by block space and element
   *
   * \note Method returns a nested array. With getBlockJ()(i,j)[k], the index i
   * identifies the block space index of the test (row) space (mapped using
   * getBlockJSpaces()), the index j identifies the block space index of the trial
   * (column) space (also mapped using getBlockJSpaces()), and k gives the k-th
   * element id for the block space identified by i and j.
   *
   * \return nested array identifying element Jacobian contributions for given
   * test and trial block spaces
   */
  const ArrayT<ArrayT<mfem::DenseMatrix>, 2>& getBlockJ() const { return m_blockJ; }

  /// @overload
  ArrayT<ArrayT<mfem::DenseMatrix>, 2>& getBlockJ() { return m_blockJ; }

 private:
  ArrayT<BlockSpace> m_blockJSpaces;              ///< list of Jacobian blocks in use
  ArrayT<ArrayT<int>> m_blockJElemIds;            ///< element ids for element Jacobian contributions
  ArrayT<ArrayT<mfem::DenseMatrix>, 2> m_blockJ;  ///< element Jacobian contributions by block
};

//------------------------------------------------------------------------------
/**
 * @brief Evaluation status for one CommonPlane overlap-cell row batch.
 */
enum class CommonPlanePairEvaluationStatus : int
{
  UNINITIALIZED,           ///< The overlap cell has not been evaluated.
  VALID,                   ///< Every requested row was generated successfully.
  INVALID_PARENT_DATA,     ///< Native parent data are missing or inconsistent between the two faces.
  INVALID_PARENT_MAPPING,  ///< At least one quadrature point could not be mapped to a parent face.
  DEGENERATE_OVERLAP,      ///< The accepted overlap cell has no positive integration measure.
  INCONSISTENT_NORMAL      ///< The CommonPlane normal is invalid or inconsistently oriented.
};

/**
 * @brief Device-resident CommonPlane quadrature rows shared by explicit operators.
 *
 * Each active CommonPlane face pair owns a fixed-capacity range of rows. The
 * pair-local row index gives a deterministic attempt-local identity without a
 * host prefix sum. Only the first entry in each range is used for a one-point
 * rule. Multipoint rules use one range entry per segment point or per
 * triangle-fan point in a three-dimensional overlap polygon.
 */
class CommonPlaneContactData : public MethodData {
 public:
  /** Maximum number of overlap polygon vertices supported by CommonPlane geometry. */
  static constexpr int maximum_overlap_vertices{ 10 };

  /** Maximum number of quadrature points in a supported triangle rule. */
  static constexpr int maximum_triangle_quadrature_points{ 25 };

  /** Maximum number of rows reserved for one accepted overlap cell. */
  static constexpr int maximum_rows_per_pair{ maximum_overlap_vertices * maximum_triangle_quadrature_points };

  /**
   * @brief Non-owning device views of a CommonPlane quadrature row batch.
   */
  struct Viewer {
    /** Number of active CommonPlane face pairs represented by this batch. */
    IndexT number_of_pairs{ 0 };

    /** Number of allocated row slots in this batch. */
    IndexT row_capacity{ 0 };

    /** Spatial dimension of the coupling scheme. */
    int spatial_dimension{ 0 };

    /** Number of generated rows for each face pair. */
    Array1DView<int> pair_row_counts;

    /** CommonPlanePairEvaluationStatus value for each face pair. */
    Array1DView<int> pair_evaluation_statuses;

    /** One for a generated row and zero for an unused row slot. */
    Array1DView<int> row_is_valid;

    /** One when the row satisfies the normal contact activation criterion. */
    Array1DView<int> row_is_active;

    /** Stable active-pair identifier for each generated row. */
    Array1DView<IndexT> contact_pair_ids;

    /** First Tribol LOR face identifier for each generated row. */
    Array1DView<IndexT> first_face_ids;

    /** Second Tribol LOR face identifier for each generated row. */
    Array1DView<IndexT> second_face_ids;

    /** Number of basis values on the first field face for each generated row. */
    Array1DView<int> first_basis_counts;

    /** Number of basis values on the second field face for each generated row. */
    Array1DView<int> second_basis_counts;

    /** One when rows scatter directly to native parent-face response storage. */
    Array1DView<int> row_uses_parent_fields;

    /** Physical CommonPlane integration-point coordinates. */
    Array2DView<RealT> integration_points;

    /** First native parent-face reference coordinates. */
    Array2DView<RealT> first_parent_reference_coordinates;

    /** Second native parent-face reference coordinates. */
    Array2DView<RealT> second_parent_reference_coordinates;

    /** First face position evaluated at each integration point. */
    Array2DView<RealT> first_positions;

    /** Second face position evaluated at each integration point. */
    Array2DView<RealT> second_positions;

    /** First face velocity evaluated at each integration point. */
    Array2DView<RealT> first_velocities;

    /** Second face velocity evaluated at each integration point. */
    Array2DView<RealT> second_velocities;

    /** Consistently oriented CommonPlane unit normal for each row. */
    Array2DView<RealT> normals;

    /** First face basis values evaluated at each integration point. */
    Array2DView<RealT> first_basis_values;

    /** Second face basis values evaluated at each integration point. */
    Array2DView<RealT> second_basis_values;

    /** Physical overlap measure multiplied by the reference quadrature weight. */
    Array1DView<RealT> integration_weights;

    /** Signed normal gap evaluated from native face positions. */
    Array1DView<RealT> gaps;

    /** Signed normal relative velocity evaluated from native face velocities. */
    Array1DView<RealT> normal_velocity_gaps;

    /** Kinematic penalty stiffness per unit overlap measure. */
    Array1DView<RealT> penalty_stiffnesses;

    /** Normal rate-penalty coefficient per unit overlap measure. */
    Array1DView<RealT> rate_penalty_coefficients;

    /** Tangential viscous coefficient per unit overlap measure. */
    Array1DView<RealT> tangential_viscous_coefficients;

    /**
     * @brief Return the first row slot assigned to an active face pair.
     *
     * @param pair_id Active face-pair identifier
     * @return First row slot reserved for the face pair
     */
    TRIBOL_HOST_DEVICE IndexT pairRowOffset( IndexT pair_id ) const { return pair_id * maximum_rows_per_pair; }
  };

  /**
   * @brief Allocate and clear storage for one CommonPlane update.
   *
   * @param number_of_pairs Number of active CommonPlane face pairs
   * @param spatial_dimension Coupling-scheme spatial dimension
   * @param allocator_id Umpire allocator identifier used by execution kernels
   */
  void resize( IndexT number_of_pairs, int spatial_dimension, int allocator_id );

  /**
   * @brief Return writable non-owning views of every row field.
   *
   * @return Device-copyable row-batch view
   */
  Viewer getView();

  /**
   * @brief Return read-only access to per-pair evaluation statuses.
   *
   * @return Evaluation-status array
   */
  const Array1D<int>& getPairEvaluationStatuses() const { return pair_evaluation_statuses_; }

  /**
   * @brief Return read-only access to generated row counts.
   *
   * @return Per-pair generated-row counts
   */
  const Array1D<int>& getPairRowCounts() const { return pair_row_counts_; }

  /**
   * @brief Return the allocated number of row slots.
   *
   * @return Number of allocated row slots
   */
  IndexT getRowCapacity() const { return row_capacity_; }

 private:
  /** Number of active CommonPlane face pairs represented by this batch. */
  IndexT number_of_pairs_{ 0 };

  /** Number of allocated row slots in this batch. */
  IndexT row_capacity_{ 0 };

  /** Spatial dimension of the coupling scheme. */
  int spatial_dimension_{ 0 };

  /** Number of generated rows for each face pair. */
  Array1D<int> pair_row_counts_;

  /** CommonPlanePairEvaluationStatus value for each face pair. */
  Array1D<int> pair_evaluation_statuses_;

  /** One for generated rows and zero for unused row slots. */
  Array1D<int> row_is_valid_;

  /** One for rows that satisfy the normal contact activation criterion. */
  Array1D<int> row_is_active_;

  /** Stable active-pair identifier for each generated row. */
  Array1D<IndexT> contact_pair_ids_;

  /** First Tribol LOR face identifier for each generated row. */
  Array1D<IndexT> first_face_ids_;

  /** Second Tribol LOR face identifier for each generated row. */
  Array1D<IndexT> second_face_ids_;

  /** Number of basis values on the first field face for each generated row. */
  Array1D<int> first_basis_counts_;

  /** Number of basis values on the second field face for each generated row. */
  Array1D<int> second_basis_counts_;

  /** One when rows scatter directly to native parent-face response storage. */
  Array1D<int> row_uses_parent_fields_;

  /** Physical CommonPlane integration-point coordinates. */
  Array2D<RealT> integration_points_;

  /** First native parent-face reference coordinates. */
  Array2D<RealT> first_parent_reference_coordinates_;

  /** Second native parent-face reference coordinates. */
  Array2D<RealT> second_parent_reference_coordinates_;

  /** First face position evaluated at each integration point. */
  Array2D<RealT> first_positions_;

  /** Second face position evaluated at each integration point. */
  Array2D<RealT> second_positions_;

  /** First face velocity evaluated at each integration point. */
  Array2D<RealT> first_velocities_;

  /** Second face velocity evaluated at each integration point. */
  Array2D<RealT> second_velocities_;

  /** Consistently oriented CommonPlane unit normal for each row. */
  Array2D<RealT> normals_;

  /** First face basis values evaluated at each integration point. */
  Array2D<RealT> first_basis_values_;

  /** Second face basis values evaluated at each integration point. */
  Array2D<RealT> second_basis_values_;

  /** Physical overlap measure multiplied by the reference quadrature weight. */
  Array1D<RealT> integration_weights_;

  /** Signed normal gap evaluated from native face positions. */
  Array1D<RealT> gaps_;

  /** Signed normal relative velocity evaluated from native face velocities. */
  Array1D<RealT> normal_velocity_gaps_;

  /** Kinematic penalty stiffness per unit overlap measure. */
  Array1D<RealT> penalty_stiffnesses_;

  /** Normal rate-penalty coefficient per unit overlap measure. */
  Array1D<RealT> rate_penalty_coefficients_;

  /** Tangential viscous coefficient per unit overlap measure. */
  Array1D<RealT> tangential_viscous_coefficients_;
};

//------------------------------------------------------------------------------
class MortarData : public MethodData {
 public:
  /*!
   * \brief Constructor
   */
  MortarData();

  /*!
   * \brief Destructor
   */
  ~MortarData();

  int m_numTotalNodes;

  /*!
   * \brief allocate object's mfem sparse matrix
   *
   * \param [in] numRows number of rows in matrix
   *
   * \note number of columns is same as number of rows
   *
   */
  void allocateMfemSparseMatrix( const int numRows )
  {
    if ( this->m_smat != nullptr ) {
      delete this->m_smat;
      this->m_smat = nullptr;
      this->m_smat = new mfem::SparseMatrix( numRows, numRows );
    }

    this->m_smat = new mfem::SparseMatrix( numRows, numRows );
  }

  /// get mfem sparse matrix object
  mfem::SparseMatrix* getMfemSparseMatrix() const { return m_smat; }

  /*!
   * \brief get the underlying CSR arrays of mfem sparse matrix object
   *
   * \param [out] I offsets array
   * \param [out] J column index array for each nonzero value
   * \param [out] vals nonzero values array
   * \param [out] n_offsets pointer to the number of offsets (size of I array)
   * \param [out] n_nonzero pointer to the number of non zeros
   *                        (size of J and vals arrays)
   *
   * \post n_offsets will store the number of offsets, if a non-nullptr was passed in
   * \post n_nonzero will store the number of non-zeros, if a non-nullptr was passed in
   */
  void getCSRArrays( int** I, int** J, RealT** vals, int* n_offsets = nullptr, int* n_nonzero = nullptr );

  /*!
   * \brief Assembles local contact element Jacobian contributions into
   *        MFEM sparse matrix on the coupling scheme object
   *
   * \param [in] elem surface contact element struct with Jacobian data
   * \param [in] sparse mode for assembly
   *
   */
  void assembleJacobian( SurfaceContactElem& elem, SparseMode s_mode ) const;

  /*!
   * \brief Assembles local contact element mortar weights into
   *        MFEM sparse matrix on the coupling scheme object
   *
   * \param [in] elem surface contact element struct with Jacobian data
   * \param [in] s_mode sparse mode option
   *
   */
  void assembleMortarWts( SurfaceContactElem& elem, SparseMode s_mode ) const;

 private:
  // mfem sparse matrix for Jacobian contributions for both meshes
  // involved in a coupling scheme or for mortar weights
  mutable mfem::SparseMatrix* m_smat;  ///< mfem sparse matrix for Jacobian or weights storage

  DISABLE_COPY_AND_ASSIGNMENT( MortarData );
  DISABLE_MOVE_AND_ASSIGNMENT( MortarData );
};

}  // end namespace tribol
#endif /* SRC_TRIBOL_MESH_METHODCOUPLINGDATA_HPP_ */
