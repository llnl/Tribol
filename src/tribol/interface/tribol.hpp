// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTERFACE_TRIBOL_HPP_
#define SRC_TRIBOL_INTERFACE_TRIBOL_HPP_

#include "tribol/common/ExecModel.hpp"
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Parameters.hpp"

#include <string>

// MFEM includes
#include "mfem.hpp"

namespace tribol {

/// \name Contact Library Initialization methods
/// @{

/*!
 * \brief Deprecated (previously initialized the contact library)
 *
 * \param [in] dimension the problem dimension
 * \param [in] comm the MPI communicator
 *
 * Problem dimension is now set by the registered meshes and MPI communicator is
 * stored at the coupling scheme level (see setMPIComm()).
 */
void initialize( int dimension, CommT comm );

/// @}

/// \name Set Parameter methods
/// @{

/**
 * @brief Sets the MPI communicator for a coupling scheme
 *
 * @param cs_id coupling scheme id
 * @param comm MPI communicator
 * \pre MPI-enabled Tribol
 *
 * If the MPI communicator is not set using this function, MPI_COMM_WORLD is
 * assumed.
 */
void setMPIComm( IndexT cs_id, CommT comm );

/*!
 * \brief Sets the penalty enforcement option
 *
 * \param [in] cs_id coupling scheme id
 * \param [in] pen_enfrc_option enum with the type of penalty enforcement to be used
 * \param [in] kinematic_calc kinematic penalty stiffness calculation option
 * \param [in] rate_calc rate penalty stiffness calculation option
 * \pre user must register coupling scheme prior to setting penalty enforcement options for that scheme
 */
void setPenaltyOptions( IndexT cs_id, PenaltyConstraintType pen_enfrc_option,
                        KinematicPenaltyCalculation kinematic_calc,
                        RatePenaltyCalculation rate_calc = NO_RATE_PENALTY );

/*!
 * \brief Sets the constant kinematic penalty stiffness
 * \param [in] mesh_id mesh id for penalty stiffness
 * \param [in] k constant kinematic penalty stiffness
 */
void setKinematicConstantPenalty( IndexT mesh_id, RealT k );

/*!
 * \brief Sets the kinematic element penalty stiffness data
 * \param [in] mesh_id mesh id for penalty stiffness
 * \param [in] material_modulus pointer to element array of bulk or Young's moduli
 * \param [in] element_thickness pointer to element array of through element thicknesses
 *
 * \note the length of the arrays that material_modulus and element_thickness point to
 *       is the number of contact faces registered for mesh with id, \p mesh_id.
 */
void setKinematicElementPenalty( IndexT mesh_id, const RealT* material_modulus, const RealT* element_thickness );

/*!
 * \brief Sets the constant rate penalty stiffness
 * \param [in] mesh_id mesh id for penalty stiffness
 * \param [in] r_k constant rate penalty stiffness
 */
void setRateConstantPenalty( IndexT mesh_id, RealT r_k );

/*!
 * \brief Sets the tangential viscous damping coefficient
 * \param [in] mesh_id mesh id for penalty stiffness
 * \param [in] coeff viscous damping coefficient
 */
void setViscousDampingCoeff( IndexT mesh_id, RealT coeff );

/*!
 * \brief Sets the percent rate penalty stiffness
 * \param [in] mesh_id mesh id for penalty stiffness
 * \param [in] r_p rate penalty as percent of kinematic penalty
 */
void setRatePercentPenalty( IndexT mesh_id, RealT r_p );

/*!
 *
 * \brief sets the auto-contact interpen fraction on the parameters struct
 *
 * \param [in] cs_id coupling scheme id
 * \param [in] scale the scale applied to the element thickness to determine the auto-contact length scale
 *
 * \note this is only used for common-plane with penalty enforcement. A sacle < 1.0 may
 * result in missed contact face-pairs in softer contact responses
 *
 *
 */
void setAutoContactPenScale( IndexT cs_id, RealT scale );

/*!
 *
 * \brief sets the timestep interpen fraction on the parameters struct
 *
 * \param [in] cs_id coupling scheme id
 * \param [in] frac the maximum allowable interpenetration factor triggering a timestep vote
 *
 * \note this is only used for common-plane with penalty enforcement. This is the
 * fraction of the element thickness that is allowed prior to triggering a timestep vote.
 *
 */
void setTimestepPenFrac( IndexT cs_id, RealT frac );

/*!
 *
 * \brief sets the timestep scale factor applied to the timestep vote
 *
 * \param [in] cs_id coupling scheme id
 * \param [in] scale the scale factor applied to the timestep vote
 *
 * \pre scale > 0
 *
 */
void setTimestepScale( IndexT cs_id, RealT scale );

/*!
 * \brief Sets the area fraction for inclusion of a contact overlap
 *
 * \param [in] cs_id coupling scheme id
 * \param [in] frac area fraction tolerance
 *
 * \note the area fraction under consideration is the ratio of the
 *       contact overlap with the largest of the two consituent
 *       faces. A default ratio is provided by Tribol.
 */
void setContactAreaFrac( IndexT cs_id, RealT frac );

/*!
 * \brief Sets the penalty scale
 *
 * \param [in] mesh_id ID for the mesh the penalty scale will be applied to
 * \param [in] scale the penalty scale
 */
void setPenaltyScale( IndexT mesh_id, RealT scale );

/*!
 * \brief Sets the Lagrange multiplier enforcement options
 *
 * \param [in] cs_id coupling scheme id
 * \param [in] evalMode evaluation mode (see enum definition)
 * \param [in] sparseMode options for how the sparse matrix is initialized
 *
 */
void setLagrangeMultiplierOptions( IndexT cs_id, ImplicitEvalMode evalMode,
                                   SparseMode sparseMode = SparseMode::MFEM_LINKED_LIST );

/*!
 * \brief Sets the plot cycle increment for visualization
 *
 * \param [in] cs_id coupling scheme id
 * \param [in] incr cycle increment between writing plot data
 */
void setPlotCycleIncrement( IndexT cs_id, int incr );

/*!
 * \brief Sets the plot options for interface visualization
 *
 * \param [in] cs_id coupling scheme id
 * \param [in] v_type visualization option
 */
void setPlotOptions( IndexT cs_id, enum VisType v_type );

/*!
 * \brief Sets the directory for dumping files.
 *
 * \param [in] cs_id coupling scheme id
 * \param [in] dir the path of the output directory
 */
void setOutputDirectory( IndexT cs_id, const std::string& dir );

/*!
 * \brief Optionally sets the logging level per coupling scheme
 * \param [in] cs_id coupling scheme id
 * \param [in] log_level the desired logging level
 *
 * \note this overrides the logging level set in initialize().
 */
void setLoggingLevel( IndexT cs_id, LoggingLevel log_level );

/*!
 * @brief Optionally set the binning proximity scale; an element length multiplier for including pairs in coarse binning
 *
 * @param cs_id coupling scheme id
 * @param binning_proximity_scale proximity scale
 */
void setBinningProximityScale( IndexT cs_id, RealT binning_proximity_scale );

/*!
 * \brief Enable the contact timestep vote
 *
 * \param [in] cs_id coupling scheme id
 * \param [in] enable the timestep vote will be calculated and returned if true
 *
 * \note default behavior is to not enable timestep calculation
 *
 */
void enableTimestepVote( IndexT cs_id, const bool enable );

/**
 * @brief Enable Enzyme AD for Jacobian calculations
 *
 * @note Requires Tribol built with Enzyme support
 *
 * @param cs_id coupling scheme id
 * @param use_enzyme Enzyme will be used if true
 */
void enableEnzyme( IndexT cs_id, bool use_enzyme );

/// @}

/// \name Contact Surface Registration Methods
/// @{

/*!
 * \brief Registers the mesh description for a contact surface
 *
 * \param [in] mesh_id the ID of the contact mesh
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
 * \note connectivity is a 2D array with num_elements rows and num_nodes columns
 * with row-major ordering
 */
void registerMesh( IndexT mesh_id, IndexT num_elements, IndexT num_nodes, const IndexT* connectivity, int element_type,
                   const RealT* x, const RealT* y, const RealT* z = nullptr, MemorySpace m_space = MemorySpace::Host );

/*!
 * \brief Registers nodal displacements on the contact surface.
 *
 * \param [in] mesh_id the ID of the contact mesh
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
void registerNodalDisplacements( IndexT mesh_id, const RealT* dx, const RealT* dy, const RealT* dz = nullptr );

/*!
 * \brief Registers nodal velocities on the contact surface.
 *
 * \param [in] mesh_id the ID of the contact mesh
 * \param [in] vx array consisting of the velocity x-components
 * \param [in] vy array consisting of the velocity y-components
 * \param [in] vz array consisting of the velocity z-components
 *
 * \pre vx != nullptr
 * \pre vy != nullptr
 * \pre vz != nullptr (3D only)
 *
 *  \note A mesh for the given contact surface must have already been registered prior to calling this method via
 * registerMesh()
 */
void registerNodalVelocities( IndexT mesh_id, const RealT* vx, const RealT* vy, const RealT* vz = nullptr );

/*!
 * \brief Registers nodal reference coords on the contact surface. Reference coordinates refer to the original mesh
 * coordinates, i.e. at time = 0.
 *
 * \param [in] mesh_id the ID of the contact mesh
 * \param [in] xref array consisting of the reference coords x-components
 * \param [in] yref array consisting of the reference coords y-components
 * \param [in] zref array consisting of the reference coords z-components
 *
 * \pre xref != nullptr
 * \pre yref != nullptr
 * \pre zref != nullptr (3D only)
 *
 *  \note A mesh for the given contact surface must have already been registered prior to calling this method via
 * registerMesh()
 */
void registerNodalReferenceCoords( IndexT mesh_id, const RealT* xref, const RealT* yref, const RealT* zref = nullptr );

/*!
 * \brief Registers nodal response buffers.
 *
 * \param [in] mesh_id the ID of the contact mesh
 * \param [in,out] rx buffer of the x-component of the contact response
 * \param [in,out] ry buffer of the y-component of the contact response
 * \param [in,out] rz buffer of the z-component of the contact response
 *
 * \pre rx != nullptr
 * \pre ry != nullptr
 * \pre rz != nullptr (3D only)
 *
 * \note A mesh for the given contact surface must have already been registered prior to calling this method.
 */
void registerNodalResponse( IndexT mesh_id, RealT* rx, RealT* ry, RealT* rz = nullptr );

/*!
 * \brief Get mfem sparse matrix for method specific Jacobian matrix output
 *
 * \param [in,out] sMat double pointer to mfem sparse matrix object
 * \param [in] cs_id coupling scheme id
 *
 * \return 0 for success, nonzero for failure
 *
 * \pre *sMat = nullptr
 *
 * \note Mortar Method: The sizing of the sparse matrix assumes that all nonmortar and mortar nodes may have a Lagrange
 *       multiplier associated with them. This allows us to use the global connectivity array, which assumes contiguous
 *       and unique node ids between mortar and nonmortar meshes registered in a given coupling scheme.
 */
int getJacobianSparseMatrix( mfem::SparseMatrix** sMat, IndexT cs_id );

/*!
 * \brief Gets CSR storage arrays for method specific Jacobian matrix output
 *
 * \param [out] I pointer to row offset integer array
 * \param [out] J pointer to column index array
 * \param [out] vals pointer to nonzero value array
 * \param [in]  cs_id coupling scheme id
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
int getJacobianCSRMatrix( int** I, int** J, RealT** vals, IndexT cs_id, int* n_offsets = nullptr,
                          int* n_nonzero = nullptr );

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
 * \param [in]  cs_id coupling scheme id
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
int getElementBlockJacobians( IndexT cs_id, BlockSpace row_block, BlockSpace col_block,
                              const ArrayT<int>** row_elem_idx, const ArrayT<int>** col_elem_idx,
                              const ArrayT<mfem::DenseMatrix>** jacobians );

/*!
 * \brief Register gap field on a nonmortar surface mesh associated with the
 * mortar method
 *
 * \param mesh_id Mesh id
 * \param gaps Array of degree-of-freedom values on the nodes of the mesh
 * representing the scalar gap field
 */
void registerMortarGaps( IndexT mesh_id, RealT* gaps );

/*!
 * \brief Register pressure field on a nonmortar surface mesh associated with
 * the mortar method
 *
 * \param mesh_id Mesh id
 * \param gaps Array of degree-of-freedom values on the nodes of the mesh
 * representing the scalar pressure field
 */
void registerMortarPressures( IndexT mesh_id, const RealT* pressures );

/// register an integer nodal field
void registerIntNodalField( IndexT mesh_id, const IntNodalFields field, int* fieldVariable );

/// register a real element field or parameter
void registerRealElementField( IndexT mesh_id, const RealElementFields field, const RealT* fieldVariable );

/// register an integer element field
void registerIntElementField( IndexT mesh_id, const IntElementFields field, int* fieldVariable );

/// @}

///  \name Contact Surface Coupling Scheme Registration Methods
/// @{

/*!
 * \brief Registers a contact coupling scheme between two contact surfaces.
 *
 * \param [in] cs_id coupling scheme id
 * \param [in] mesh_id1 id of the first contact surface
 * \param [in] mesh_id2 id of the second contact surface
 * \param [in] contact_mode the type of contact, e.g. SURFACE_TO_SURFACE
 * \param [in] contact_case the specific case of contact application, e.g. auto
 * \param [in] contact_method the contact method, e.g. SINGLE_MORTAR
 * \param [in] contact_model the contact model, e.g. FRICTION_COULOMB
 * \param [in] enforcement_method the enforcement method, e.g. PENALTY
 * \param [in] binning_method the binning method, e.g. BINNING_GRID
 * \param [in] given_exec_mode preferred execution mode for RAJA kernels
 *
 * \note A mesh for the given contact surface must have already been registered
 *  prior to calling this method.
 */
void registerCouplingScheme( IndexT cs_id, IndexT mesh_id1, IndexT mesh_id2, int contact_mode, int contact_case,
                             int contact_method, int contact_model, int enforcement_method,
                             int binning_method = DEFAULT_BINNING_METHOD,
                             ExecutionMode exec_mode = ExecutionMode::Dynamic );
/// @}

/*!
 * \brief Sets the interacting cell-pairs manually.
 *
 * \param [in] cs_id      coupling scheme id
 * \param [in] numPairs   number of cell-pairs to be registered
 * \param [in] mesh_id1   mesh id of the first cell in the pair list
 * \param [in] pairType1  cell type of the first cell in the pair list
 * \param [in] pairIndex1 index of the first cell in the pair list
 * \param [in] mesh_id2    mesh id of the second cell in the pair list
 * \param [in] pairType2  cell type of the second cell in the pair list
 * \param [in] pairIndex2 index of the second cell in the pair list
 *
 */
void setInterfacePairs( IndexT cs_id, IndexT numPairs, IndexT const* mesh_id1, IndexT const* pairType1,
                        IndexT const* pairIndex1, IndexT const* mesh_id2, IndexT const* pairType2,
                        IndexT const* pairIndex2 );

/*!
 * \brief Computes the contact response at the given cycle.
 *
 * \param [in] cycle the current cycle.
 * \param [in] t the corresponding simulation time at the given cycle.
 * \param [in/out] dt the simulation timestep input with Tribol timestep vote output
 *
 * \return rc return code, a non-zero return code indicates an error.
 */
int update( int cycle, RealT t, RealT& dt );

/// \name Contact Library finalization methods
/// @{

/*!
 * \brief Finalizes
 */
void finalize();

/// @}

} /* namespace tribol */

#endif /* SRC_TRIBOL_INTERFACE_TRIBOL_HPP_ */
