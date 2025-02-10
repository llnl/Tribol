// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_HPP_
#define TRIBOL_HPP_

#include "tribol/types.hpp"
#include "tribol/geom/ContactPlaneManager.hpp"

#include <string>

// MFEM includes
#include "mfem.hpp"

namespace tribol {

/// \name Contact Library Initialization methods
/// @{

/*!
 * \brief Initializes the contact library.
 *
 * \param [in] dimension the problem dimension.
 * \param [in] comm the MPI communicator
 *
 * \pre dimension==2 || dimensions==3
 * \pre comm != MPI_COMM_NULL
 */
void initialize( integer dimension, CommType comm );

/// @}

/// \name Set Parameter methods
/// @{

/*!
 * \brief Sets the penalty enforcement option
 *
 * \param [in] couplingSchemeIndex coupling scheme index
 * \param [in] pen_enfrc_option enum with the type of penalty enforcement to be used
 * \param [in] kinematic_calc kinematic penalty stiffness calculation option
 * \param [in] rate_calc rate penalty stiffness calculation option
 * \pre user must register coupling scheme prior to setting penalty enforcement options for that scheme
 */
void setPenaltyOptions( int couplingSchemeIndex, PenaltyConstraintType pen_enfrc_option,
                        KinematicPenaltyCalculation kinematic_calc,
                        RatePenaltyCalculation rate_calc = NO_RATE_PENALTY );

/*!
 * \brief Sets the constant kinematic penalty stiffness
 * \param [in] meshId mesh id for penalty stiffness
 * \param [in] k constant kinematic penalty stiffness
 */
void setKinematicConstantPenalty( int meshId, double k );

/*!
 * \brief Sets the kinematic element penalty stiffness data
 * \param [in] meshId mesh id for penalty stiffness
 * \param [in] material_modulus pointer to element array of bulk or Young's moduli
 * \param [in] element_thickness pointer to element array of through element thicknesses
 *
 * \note the length of the arrays that material_modulus and element_thickness point to
 *       is the number of contact faces registered for mesh with id, \p meshId.
 */
void setKinematicElementPenalty( int meshId, const double* material_modulus, const double* element_thickness );

/*!
 * \brief Sets the constant rate penalty stiffness
 * \param [in] meshId mesh id for penalty stiffness
 * \param [in] r_k constant rate penalty stiffness
 */
void setRateConstantPenalty( int meshId, double r_k );

/*!
 * \brief Sets the percent rate penalty stiffness
 * \param [in] meshId mesh id for penalty stiffness
 * \param [in] r_p rate penalty as percent of kinematic penalty
 */
void setRatePercentPenalty( int meshId, double r_p );

/*!
 *
 * \brief sets the contact interpen fraction on the parameters struct
 *
 * \param [in] frac the maximum allowable interpenetration factor
 *
 * \note this is only used for common-plane with penalty enforcement
 *
 */
void setContactPenFrac( double frac );

/*!
 * \brief Sets the area fraction for inclusion of a contact overlap
 * \param [in] frac area fraction tolerance
 *
 * \note the area fraction under consideration is the ratio of the
 *       contact overlap with the largest of the two consituent
 *       faces. A default ratio is provided by Tribol.
 */
void setContactAreaFrac( double frac );

/*!
 * \brief Sets the penalty scale
 *
 * \param [in] meshId ID for the mesh the penalty scale will be applied to
 * \param [in] scale the penalty scale
 */
void setPenaltyScale( int meshId, double scale );

/*!
 * \brief Sets the Lagrange multiplier enforcement options
 *
 * \param [in] couplingSchemeIndex the index for the coupling scheme
 * \param [in] evalMode evaluation mode (see enum definition)
 * \param [in] sparseMode options for how the sparse matrix is initialized
 *
 */
void setLagrangeMultiplierOptions( int couplingSchemeIndex, ImplicitEvalMode evalMode,
                                   SparseMode sparseMode = SparseMode::MFEM_LINKED_LIST );

/*!
 * \brief Sets the plot cycle increment for visualization
 * \param [in] incr cycle increment between writing plot data
 */
void setPlotCycleIncrement( double incr );

/*!
 * \brief Sets the plot options for interface visualization
 *
 * \param [in] v_type visualization option
 */
void setPlotOptions( enum VisType v_type );

/*!
 * \brief Sets the directory for dumping files.
 * \param [in] dir the path of the output directory
 */
void setOutputDirectory( const std::string& dir );

/*!
 * \brief Optionally sets the logging level per coupling scheme
 * \param [in] csId coupling scheme id
 * \param [in] log_level the desired logging level
 *
 * \note this overrides the logging level set in initialize().
 */
void setLoggingLevel( int csId, LoggingLevel log_level );

/// @}

/// \name Contact Surface Registration Methods
/// @{

/*!
 * \brief Registers the mesh description for a contact surface.
 *
 * \param [in] meshId the ID of the contact surface.
 * \param [in] numCells the number of cells on the contact surface.
 * \param [in] lengthNodalData length of the data arrays being registered.
 * \param [in] connectivity mesh connectivity array for the contact surface.
 * \param [in] cellType the cell type of the contact surface elements.
 * \param [in] x array of x-components of the mesh coordinates
 * \param [in] y array of y-components of the mesh coordinates
 * \param [in] z array of z-components of the mesh coordinates (3D only)
 *
 * \note numMeshNodes may be the number of nodes on the surface
 *       (i.e. surface mesh only), or they may include nodes in the volume,
 *       but the number is specific to the contact body to which the surface
 *       defined by the surface element connectivity array belongs.
 *
 * \pre connectivity != nullptr
 * \pre x != nullptr
 * \pre y != nullptr
 * \pre z != nullptr (3D only)
 */
void registerMesh( integer meshId, integer numCells, integer lengthNodalData, const IndexType* connectivity,
                   integer cellType, const real* x, const real* y, const real* z = nullptr );

/*!
 * \brief Registers nodal displacements on the contact surface.
 *
 * \param [in] meshId the ID of the contact surface.
 * \param [in] dx array consisting of the x-component displacements
 * \param [in] dy array consisting of the y-component displacements
 * \param [in] dz array consisting of the z-component displacements
 *
 * \pre dx != nullptr
 * \pre dy != nullptr
 * \pre dz != nullptr (3D only)
 *
 * \note A mesh for the given contact surface must have already been registered
 *  prior to calling this method via registerMesh()
 */
void registerNodalDisplacements( integer meshId, const real* dx, const real* dy, const real* dz = nullptr );

/*!
 * \brief Registers nodal velocities on the contact surface.
 *
 * \param [in] meshId the ID of the contact surface.
 * \param [in] vx array consisting of the velocity x-components
 * \param [in] vy array consisting of the velocity y-components
 * \param [in] vz array consisting of the velocity z-components
 *
 * \pre vx != nullptr
 * \pre vy != nullptr
 * \pre vz != nullptr (3D only)
 *
 *  \note A mesh for the given contact surface must have already been registered
 *   prior to calling this method] via registerMesh()
 */
void registerNodalVelocities( integer meshId, const real* vx, const real* vy, const real* vz = nullptr );

/*!
 * \brief Registers nodal response buffers.
 *
 * \param [in] meshId the ID of the contact surface.
 * \param [in,out] rx buffer of the x-component of the contact response
 * \param [in,out] ry buffer of the y-component of the contact response
 * \param [in,out] rz buffer of the z-component of the contact response
 *
 * \pre rx != nullptr
 * \pre ry != nullptr
 * \pre rz != nullptr (3D only)
 *
 * \note A mesh for the given contact surface must have already been registered
 *  prior to calling this method.
 */
void registerNodalResponse( integer meshId, real* rx, real* ry, real* rz = nullptr );

/*!
 * \brief Get mfem sparse matrix for method specific Jacobian matrix output
 *
 * \param [in,out] sMat double pointer to mfem sparse matrix object
 * \param [in] csId Coupling scheme id
 *
 * \return 0 for success, nonzero for failure
 *
 * \pre *sMat = nullptr
 *
 * \note Mortar Method: The sizing of the sparse matrix assumes that all
 *       nonmortar and mortar nodes may have a Lagrange multiplier associated
 *       with them. This allows us to use the global connectivity array,
 *       which assumes contiguous and unique node ids between mortar and
 *       nonmortar meshes registered in a given coupling scheme.
 */
int getJacobianSparseMatrix( mfem::SparseMatrix** sMat, int csId );

/*!
 * \brief Gets CSR storage arrays for method specific Jacobian matrix output
 *
 * \param [out] I pointer to row offset integer array
 * \param [out] J pointer to column index array
 * \param [out] vals pointer to nonzero value array
 * \param [in]  csId coupling scheme id
 * \param [out] n_offsets optional pointer to the number of offsets (size of I array)
 * \param [out] n_nonzero optional pointer to the number of non zeros
 *                        (size of J and vals arrays)
 *
 * \pre I == nullptr
 * \pre J == nullptr
 * \pre vals == nullptr
 *
 * \post n_offsets will store the number of offsets, if a non-nullptr was passed in
 * \post n_nonzero will store the number of non-zeros, if a non-nullptr was passed in
 *
 * \return 0 success (if CSR data exists and pointed to), nonzero for failure
 *
 */
int getJacobianCSRMatrix( int** I, int** J, real** vals, int csId, int* n_offsets = nullptr, int* n_nonzero = nullptr );

/*!
 * \brief Get element Jacobian matrix contributions for a given block
 *
 * The element Jacobians are stored in blocks associated with the mortar
 * surface, nonmortar surface, and Lagrange multipliers (if applicable).  The
 * mortar-mortar, nonmortar-nonmortar, mortar-nonmortar, and nonmortar-mortar
 * blocks are associated with the equilibrium contributions (derivative of the
 * weak form contact integral with respect to displacement degrees of freedom)
 * and the blocks involving the Lagrange multiplier field are associated with
 * the constraint block (with the exception of the Lagrange multiplier-Lagrange
 * multiplier block, which is zero).
 *
 * The structure of the blocks is:
 *       M    NM   LM
 *     ----------------  M  = BlockSpace::MORTAR
 *   M | 00 | 01 | 02 |
 *     |----|----|----|  NM = BlockSpace::NONMORTAR
 *  NM | 10 | 11 | 12 |
 *     |----|----|----|  LM = BlockSpace::LAGRANGE_MULTIPLIER
 *  LM | 20 | 21 | 22 |
 *     ----------------
 * For example, requesting row_block = BlockSpace::MORTAR and col_block =
 * BlockSpace::LAGRANGE_MULTIPLIER will return the block Jacobians in position
 * 02. row_elem_idx will be an array of mortar element indices and col_elem_idx
 * will be an array of Lagrange multiplier element indices. The length of
 * row_elem_idx, col_elem_idx, and jacobians will match.  Each entry in the
 * array corresponds to a single coupled element pair, so element indices will
 * not be unique, in general.  For instance, if a nonmortar face interacts with
 * multiple mortar faces and vice-versa.
 *
 * \param [in]  csId Coupling scheme id
 * \param [in]  row_block Row Jacobian block (MORTAR, NONMORTAR, or
 * LAGRANGE_MULTIPLIER)
 * \param [in]  col_block Column Jacobian block (MORTAR, NONMORTAR, or
 * LAGRANGE_MULTIPLIER)
 * \param [out] row_elem_idx Pointer to pointer to array of element indices for
 * the row block
 * \param [out] col_elem_idx Pointer to pointer to array of element indices for
 * the column block
 * \param [out] jacobians Pointer to pointer to array of Jacobian dense matrices
 *
 * @note The second pointer of the double pointer is updated by this function to
 * point to internally stored arrays of indices and Jacobian values.
 *
 * \return 0 success (if Jacobians exist), nonzero for failure
 */
int getElementBlockJacobians( integer csId, BlockSpace row_block, BlockSpace col_block,
                              const axom::Array<integer>** row_elem_idx, const axom::Array<integer>** col_elem_idx,
                              const axom::Array<mfem::DenseMatrix>** jacobians );

/*!
 * \brief Register gap field on a nonmortar surface mesh associated with the
 * mortar method
 *
 * \param meshId Mesh id
 * \param gaps Array of degree-of-freedom values on the nodes of the mesh
 * representing the scalar gap field
 */
void registerMortarGaps( integer meshId, real* gaps );

/*!
 * \brief Register pressure field on a nonmortar surface mesh associated with
 * the mortar method
 *
 * \param meshId Mesh id
 * \param gaps Array of degree-of-freedom values on the nodes of the mesh
 * representing the scalar pressure field
 */
void registerMortarPressures( integer meshId, const real* pressures );

/// register an integer nodal field
void registerIntNodalField( integer meshId, const IntNodalFields field, integer* fieldVariable );

/// register a real element field or parameter
void registerRealElementField( int meshId, const RealElementFields field, const double* fieldVariable );

/// register an integer element field
void registerIntElementField( int meshId, const IntElementFields field, integer* fieldVariable );

/// @}

///  \name Contact Surface Coupling Scheme Registration Methods
/// @{

/*!
 * \brief Registers a contact coupling scheme between two contact surfaces.
 *
 * \param [in] couplingSchemeIndex Index to use for this coupling scheme.
 * \param [in] meshId1 Id of the first contact surface.
 * \param [in] mehsId2 Id of the second contact surface.
 * \param [in] contact_mode
 * \param [in] contact_case
 * \param [in] contact_method
 * \param [in] contact_model
 * \param [in] enforcement_method
 * \param [in] binning_method
 *
 * \note A mesh for the given contact surface must have already been registered
 *  prior to calling this method.
 */
void registerCouplingScheme( integer couplingSchemeIndex, integer meshId1, integer meshId2, integer contact_mode,
                             integer contact_case, integer contact_method, integer contact_model,
                             integer enforcement_method, integer binning_method = DEFAULT_BINNING_METHOD );
/// @}

/*!
 * \brief Sets the interacting cell-pairs manually.
 *
 * \param [in] couplingSchemeIndex The index of the coupling scheme to which
 * we are associating these interface pairs
 * \param [in] numPairs   number of cell-pairs to be registered
 * \param [in] meshId1    meshId of the first cell in the pair list
 * \param [in] pairType1  cell type of the first cell in the pair list
 * \param [in] pairIndex1 index of the first cell in the pair list
 * \param [in] meshId2    meshId of the second cell in the pair list
 * \param [in] pairType2  cell type of the second cell in the pair list
 * \param [in] pairIndex2 index of the second cell in the pair list
 *
 */
void setInterfacePairs( integer couplingSchemeIndex, IndexType numPairs, IndexType const* meshId1,
                        IndexType const* pairType1, IndexType const* pairIndex1, IndexType const* meshId2,
                        IndexType const* pairType2, IndexType const* pairIndex2 );

/*!
 * \brief Computes the contact response at the given cycle.
 *
 * \param [in] cycle the current cycle.
 * \param [in] t the corresponding simulation time at the given cycle.
 * \param [in/out] dt the simulation timestep input with Tribol timestep vote output
 *
 * \return rc return code, a non-zero return code indicates an error.
 */
integer update( integer cycle, real t, real& dt );

/// \name Contact Library finalization methods
/// @{

/*!
 * \brief Finalizes
 */
void finalize();

/// @}

} /* namespace tribol */

#endif /* TRIBOL_HPP_ */
