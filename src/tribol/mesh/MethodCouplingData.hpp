// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_MESH_METHODCOUPLINGDATA_HPP_
#define SRC_MESH_METHODCOUPLINGDATA_HPP_

#include "tribol/types.hpp"
#include "tribol/common/Parameters.hpp"

// Axom includes
#include "axom/core.hpp"
#include "axom/slic.hpp"

// MFEM includes
#include "mfem.hpp"
#include <axom/core/Types.hpp>

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
  SurfaceContactElem( integer dimension,  ///< [in] Dimension of the problem
                      real* x1,           ///< [in] Vertex coordinates of first face
                      real* x2,           ///< [in] Vertex coordinates of second face
                      real* xOverlap,     ///< [in] Vertex coordinates of overlap
                      integer nFV,        ///< [in] Number of face vertices
                      integer nPV,        ///< [in] Number of overlap vertices
                      integer mId1,       ///< [in] Id for mesh 1
                      integer mId2,       ///< [in] Id for mesh 2
                      integer fId1,       ///< [in] Id for face 1
                      integer fId2        ///< [in] Id for face 2
                      )
      : dim( dimension ),
        meshId1( mId1 ),
        meshId2( mId2 ),
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
        numActiveGaps( 0 )

  {
  }

  /// Destructor
  ~SurfaceContactElem() { this->deallocateElem(); }

  integer dim;          ///< Problem dimension
  integer meshId1;      ///< Mesh Id for face 1 (mortar)
  integer meshId2;      ///< Mesh Id for face 2 (nonmortar)
  integer faceId1;      ///< Face Id for face 1 (mortar)
  integer faceId2;      ///< Face Id for face 2 (nonmortar)
  real* faceCoords1;    ///< Coordinates of face 1 in 3D
  real* faceCoords2;    ///< Coordinates of face 2 in 3D
  real* faceNormal1;    ///< Components of face 1 normal
  real* faceNormal2;    ///< Components of face 2 normal
  real* overlapCoords;  ///< Coordinates of overlap vertices in 3D
  real* overlapNormal;  ///< Components of overlap normal
  integer numFaceVert;  ///< Number of face vertices/nodes
  integer numPolyVert;  ///< Number of overlap vertices
  real overlapArea;     ///< Area of polygonal overlap

  real* mortarWts;  ///< Stacked array of mortar wts for mortar methods
  int numWts;       ///< Number of mortar weights

  int numActiveGaps;  ///< Number of local face-pair active gaps

  axom::Array<mfem::DenseMatrix, 2> blockJ;

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
  real getMortarNonmortarWt( const int a, const int b );

  /*!
   * \brief routine to return nonmortar-mortar mortar weights
   *
   * \param [in] a NONMORTAR node id
   * \param [in] b MORTAR node id
   *
   */
  real getNonmortarMortarWt( const int a, const int b );

  /*!
   * \brief routine to return nonmortar-nonmortar mortar weight
   *
   * \param [in] a NONMORTAR node id
   * \param [in] b NONMORTAR node id
   *
   */
  real getNonmortarNonmortarWt( const int a, const int b );

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
  void deallocateElem();

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
  ~MethodData(){};

  /*!
   * \brief allocate element Jacobian matrix storage
   *
   * \param [in] blockJSpaces list of block spaces used in the Jacobian matrix
   * \param [in] nPairs approximate number of contacting face-pairs (used to
   * allocate memory in the axom::Array)
   */
  void reserveBlockJ( axom::Array<BlockSpace>&& blockJSpaces, int nPairs );

  /*!
   * \brief store an element contribution to all blocks of the Jacobian matrix
   *
   * \param [in] blockJElemIds list of element ids on each block space
   ^
   * \param [in] blockJ 2D array of element Jacobian contributions (each array
   * entry corresponds to a block of the Jacobian matrix)
   */
  void storeElemBlockJ( axom::Array<integer>&& blockJElemIds, axom::Array<mfem::DenseMatrix, 2>& blockJ );

  /*!
   * \brief Returns the number of blocks in the Jacobian matrix
   *
   * See @ref getElementBlockJacobians for a definition of the blocks.
   */
  integer getNSpaces() const { return m_blockJSpaces.size(); }

  /*!
   * \brief Get the element ids for each entry of the getBlockJ 2D axom::Array
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
  const axom::Array<axom::Array<integer>>& getBlockJElementIds() const { return m_blockJElemIds; }

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
  const axom::Array<axom::Array<mfem::DenseMatrix>, 2>& getBlockJ() const { return m_blockJ; }

 private:
  axom::Array<BlockSpace> m_blockJSpaces;                   ///< list of Jacobian blocks in use
  axom::Array<axom::Array<integer>> m_blockJElemIds;        ///< element ids for element Jacobian contributions
  axom::Array<axom::Array<mfem::DenseMatrix>, 2> m_blockJ;  ///< element Jacobian contributions by block
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

  integer m_numTotalNodes;

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
  void getCSRArrays( int** I, int** J, real** vals, int* n_offsets = nullptr, int* n_nonzero = nullptr );

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
#endif /* SRC_MESH_METHODCOUPLINGDATA_HPP_ */
