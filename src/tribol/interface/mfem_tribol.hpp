// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTERFACE_MFEM_TRIBOL_HPP_
#define SRC_TRIBOL_INTERFACE_MFEM_TRIBOL_HPP_

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP

#include "mfem.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/common/ExecModel.hpp"
#include "tribol/physics/CoulombFriction.hpp"
#include "tribol/physics/ContactResidual.hpp"

namespace tribol {

/**
 * @brief Define and register a coupling scheme over an MFEM mesh
 *
 * This function is designed to enable simple registration of a contact coupling scheme over a parallel MFEM mesh.
 * Contact surfaces are defined via a list of one or more boundary attributes, which is used to construct a single
 * contact surface mfem::ParSubMesh (a boundary submesh extracted from the volume mesh). Calling this function stores
 * the data needed to create a coupling scheme; however, the coupling scheme is not created until
 * updateMfemParallelDecomposition() is called. The meshes used in the Tribol coupling scheme are created after the
 * parallel domains are rebalanced (using the redecomp library) through the call to updateMfemParallelDecomposition().
 *
 * If the registered mesh contains a higher-order Nodes grid function, then a low-order refined (LOR) mesh is created
 * (if needed) to use Tribol's linear contact methodologies. LOR support is currently limited to methods that do not
 * require Jacobian calculations, and is still experimental. The low-order refinement factor equals the order of the
 * Nodes grid function, but can be overridden using setMfemLowOrderRefinedFactor().
 *
 * Mortar methods implemented through ContactFormulation use the first mesh as the non-mortar integration surface
 * and the second as the mortar projection surface. Legacy SINGLE_MORTAR uses the opposite surface order.
 *
 * @param [in] cs_id Index to use for the coupling scheme
 * @param [in] mesh_id_1 The first ID of the contact surface mesh
 * @param [in] mesh_id_2 The second ID of the contact surface mesh
 * @param [in] mesh MFEM volume mesh
 * @param [in] current_coords Coordinates associated with the MFEM volume mesh
 * @param [in] b_attributes_1 Boundary attributes defining the first mesh
 * @param [in] b_attributes_2 Boundary attributes defining the second mesh
 * @param [in] contact_mode
 * @param [in] contact_case
 * @param [in] contact_method
 * @param [in] contact_model
 * @param [in] enforcement_method
 * @param [in] binning_method
 * @param [in] exec_mode Execution mode for the coupling scheme (Sequential by default)
 *
 * Quadrature-point ENERGY_MORTAR uses parent-only Jacobian transfer and creates no pressure/gap fields. Nodal
 * enforcement creates those fields when selected; switching locations rebuilds the formulation and transfer data.
 */
void registerMfemCouplingScheme( IndexT cs_id, int mesh_id_1, int mesh_id_2, const mfem::ParMesh& mesh,
                                 const mfem::ParGridFunction& current_coords, std::set<int> b_attributes_1,
                                 std::set<int> b_attributes_2, ContactMode contact_mode, ContactCase contact_case,
                                 ContactMethod contact_method, ContactModel contact_model,
                                 EnforcementMethod enforcement_method,
                                 BinningMethod binning_method = DEFAULT_BINNING_METHOD,
                                 ExecutionMode exec_mode = ExecutionMode::Sequential );

/**
 * @brief Configure Puso-Laursen mortar with direct Coulomb multipliers.
 * @param cs_id Registered MFEM SINGLE_MORTAR/FRICTION_COULOMB/LAGRANGE_MULTIPLIER scheme.
 * @param options Finite coefficient, positive equation scales, and positive overlap tolerance.
 * @throws std::invalid_argument Options or configuration are unsupported.
 * @throws std::logic_error A load step is active.
 * This function is collective on the parent communicator. Supports linear 3D CPU TRI3/TRI3 and
 * QUAD4/QUAD4 surfaces, with fixed conforming parent topology and partition. Redecomp may change during a step.
 */
void setCoulombFrictionOptions( IndexT cs_id, const CoulombFrictionOptions& options );
/**
 * @brief Access Cartesian traction multiplier input; call update() after changing it.
 * @param cs_id Registered MFEM scheme with Cartesian multiplier enforcement, currently Coulomb mortar.
 * @return Mutable vector grid function on the parent-linked boundary submesh.
 * @throws std::invalid_argument The scheme does not provide a Cartesian multiplier field.
 * This function is collective. For Coulomb mortar, lambda dot outward unit normal is compression-positive.
 * Scalar normal multipliers remain available through getMfemPressure() or getMfemContactPressure().
 */
mfem::ParGridFunction& getMfemContactTraction( IndexT cs_id );
/**
 * @brief Return owning copies of the latest contact residuals and available diagnostics.
 * @param cs_id Registered MFEM Coulomb mortar or ContactFormulation scheme.
 * @return Owned true-DOF residuals; unavailable diagnostics are empty and penalty constraints are null.
 * @throws std::invalid_argument The scheme does not provide assembled residuals, or Coulomb inputs are stale.
 * @throws std::logic_error No successful calculation is available; call update() first.
 * This function is collective. A nonzero residual means unconverged equations, not a failed calculation.
 * Frictionless nodal mortar returns its weighted gap as the multiplier equation; Coulomb mortar returns its
 * normal/tangential contact-law equations. See ContactResidual for ordering and method-dependent diagnostics.
 */
ContactResidual getMfemContactResidual( IndexT cs_id );
/**
 * @brief Begin a load step by snapshotting the accepted configuration for methods with history.
 * @param cs_id Registered MFEM scheme; first call updateMfemParallelDecomposition().
 * @throws std::invalid_argument The scheme is absent, unsupported, or has invalid inputs.
 * @throws std::logic_error A history-dependent load step is already active or not configured.
 * This collective call does nothing for stateless methods. Coulomb mortar saves coordinates, traction, and gap vectors.
 */
void beginContactStep( IndexT cs_id );
/**
 * @brief Accept the last evaluation; the application is responsible for convergence acceptance.
 * @param cs_id Registered MFEM scheme; methods with history require a successful update() at unchanged inputs.
 * @throws std::logic_error No history-dependent step is active.
 * @throws std::invalid_argument The scheme is absent or the latest history-dependent result is absent or stale.
 * This collective call does nothing for stateless methods. Calculation success is separate from convergence acceptance.
 */
void commitContactStep( IndexT cs_id );
/**
 * @brief Discard trial results and restore accepted multipliers; application restores its coordinate field.
 * @param cs_id Registered MFEM scheme; methods with history require an active step.
 * @throws std::invalid_argument The scheme is absent or not registered through MFEM.
 * @throws std::logic_error No history-dependent step is active.
 * This collective call does nothing for stateless methods. Caller-owned coordinates must be restored by the
 * application.
 */
void rollbackContactStep( IndexT cs_id );

/**
 * @brief Opt in to a direct common-plane residual and Enzyme Jacobian, or restore legacy Common Plane.
 * @param cs_id MFEM COMMON_PLANE coupling scheme with positive constant penalties already configured.
 * @param enabled True selects the formulation; false restores the legacy response path.
 * @throws std::invalid_argument Unsupported method, geometry, penalty, or execution settings; selection is unchanged.
 * @throws std::runtime_error Enzyme support is unavailable.
 *
 * Supports linear 2D MFEM frictionless penalty contact without LOR subdivision on host. Use getMfemContactForce()
 * and getMfemDfDx() after update(), rather than getMfemResponse(). The returned vector is a contact residual; its
 * negative is the physical force returned by legacy Common Plane. Nodal gaps, energy, and timestep output are
 * unsupported.
 */
void setMfemUseCommonPlaneFormulation( IndexT cs_id, bool enabled );

/**
 * @brief Sets factor of refinement in low-order refined (LOR) representation of
 * the contact mesh
 *
 * This method sets the low-order mesh refinement factor for coupling schemes
 * registered with MFEM meshes.  Low-order refinement enables representation of
 * higher-order contact surfaces and field quantities on a refined, low-order
 * mesh.  A low-order refinement factor of e.g. 2 results in twice the number of
 * elements on the LOR mesh in each dimension.  By default, the low-order
 * refinement factor equals the order of the mesh, such that the number of
 * degrees of freedom remain constant in the higher-order mesh and the LOR mesh.
 * This method allows the LOR mesh to be further refined, if desired.
 *
 * Transfer of field quantities between higher-order and LOR representations is
 * mass preserving and is accomplished through L2 minimization.  Given a
 * higher-order MFEM mesh, low-order refinement enables linear mesh contact
 * methodologies to be applied on higher-order meshes.
 *
 * @pre Coupling scheme cs_id must be registered using
 * registerMfemCouplingScheme()
 *
 * @param [in] cs_id The ID of the coupling scheme
 * @param [in] lor_factor The refinement factor of the LOR mesh
 */
void setMfemLORFactor( IndexT cs_id, int lor_factor );

/**
 * @brief Sets the displacement threshold for triggering a new parallel decomposition
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 *
 * @param [in] cs_id The ID of the coupling scheme
 * @param [in] val Threshold value
 */
void setMfemRedecompTriggerDisplacement( IndexT cs_id, RealT val );

/**
 * @brief Clears existing penalty data and sets kinematic constant penalty
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 *
 * @param cs_id The ID of the coupling scheme with the MFEM mesh
 * @param mesh1_penalty Penalty parameter for the first contact surface mesh
 * @param mesh2_penalty Penalty parameter for the second contact surface mesh
 */
void setMfemKinematicConstantPenalty( IndexT cs_id, RealT mesh1_penalty, RealT mesh2_penalty );

/**
 * @brief Clears existing penalty data and sets kinematic element penalty
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 *
 * @note modulus_coefficient attributes correspond to the boundary attributes of the parent mesh
 *
 * @note modulus_coefficient is only evaluated on the parent-linked boundary submesh
 *
 * @param cs_id The ID of the coupling scheme with the MFEM mesh
 * @param modulus_coefficient MFEM coefficient defining bulk modulus over the parent-linked boundary submesh
 */
void setMfemKinematicElementPenalty( IndexT cs_id, mfem::Coefficient& modulus_coefficient );

/**
 * @brief Adds constant gap rate penalty to the existing kinematic penalty
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 * @pre Kinematic penalty must be initialized through setMfemKinematicConstantPenalty() or
 * setMfemKinematicElementPenalty()
 *
 * @param cs_id The ID of the coupling scheme with the MFEM mesh
 * @param mesh1_penalty Penalty parameter for the first contact surface mesh
 * @param mesh2_penalty Penalty parameter for the second contact surface mesh
 */
void setMfemRateConstantPenalty( IndexT cs_id, RealT mesh1_penalty, RealT mesh2_penalty );

/**
 * @brief Adds gap rate penalty as a scaling of the existing kinematic penalty
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 * @pre Kinematic penalty must be initialized through setMfemKinematicConstantPenalty() or
 * setMfemKinematicElementPenalty()
 *
 * @param cs_id The ID of the coupling scheme with the MFEM mesh
 * @param mesh1_ratio Scaling coefficient of the kinematic penalty for the first contact surface mesh
 * @param mesh2_ratio Scaling coefficient of the kinematic penalty for the second contact surface mesh
 */
void setMfemRatePercentPenalty( IndexT cs_id, RealT mesh1_ratio, RealT mesh2_ratio );

/**
 * @brief Adds a scale to the computed kinematic penalty
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 * @pre Kinematic penalty must be initialized through setMfemKinematicConstantPenalty() or
 * setMfemKinematicElementPenalty()
 *
 * @note The scaling is applied to the kinematic penalty used in rate percent penalty enforcement.
 *
 * @param cs_id The ID of the coupling scheme with the MFEM mesh
 * @param mesh1_scale Scaling coefficient of the kinematic penalty for the first contact surface mesh
 * @param mesh2_scale Scaling coefficient of the kinematic penalty for the second contact surface mesh
 */
void setMfemKinematicPenaltyScale( IndexT cs_id, RealT mesh1_scale, RealT mesh2_scale );

/**
 * @brief Adds a tangential viscous damping coefficient to each mesh
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 *
 * @param cs_id The ID of the coupling scheme with the MFEM mesh
 * @param mesh1_coeff Tangential viscous damping coefficient for the first contact surface mesh
 * @param mesh2_coeff Tangential viscous damping coefficient for the second contact surface mesh
 */
void setMfemViscousDampingCoeff( IndexT cs_id, RealT mesh1_coeff, RealT mesh2_coeff );

/**
 * @brief Computes element thickness for the volume elements associated with the contact surface mesh.
 *
 * Element thickness is calculated at the origin of the isoparametric volume element in the direction given by the
 * normal at the center of the associated isoparametric surface element.
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 * @pre Kinematic element penalty must be initialized through setMfemKinematicElementPenalty()
 *
 * @note Element thickness is automatically set when setMfemKinematicElementPenalty() is called. Call this method to
 * update element thicknesses.
 *
 * @param cs_id The ID of the coupling scheme with the MFEM mesh
 */
void updateMfemElemThickness( IndexT cs_id );

/**
 * @brief Sets the bulk modulus for the volume elements associated with the contact surface mesh.
 *
 * @note For attribute-based coefficients, the attributes are based on the parent element's boundary attributes.
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 * @pre Kinematic element penalty must be initialized through setMfemKinematicElementPenalty()
 *
 * @param cs_id The ID of the coupling scheme with the MFEM mesh
 * @param modulus_coefficient Coefficient field projected onto the domain
 *
 * @note Material modulus is autoamtically set when setMfemKinematicElementPenalty() is called. Call this method to
 * update material bulk moduli.
 */
void updateMfemMaterialModulus( IndexT cs_id, mfem::Coefficient& modulus_coefficient );

/**
 * @brief Registers a velocity field on a MFEM mesh-defined coupling scheme
 *
 * @pre Coupling scheme cs_id must be registered using
 * registerMfemCouplingScheme()
 *
 * @param [in] cs_id The ID of the coupling scheme with the MFEM mesh
 * @param [in] v MFEM velocity ParGridFunction defined over the parent mesh
 */
void registerMfemVelocity( IndexT cs_id, const mfem::ParGridFunction& v );

/**
 * @brief Registers a reference coord field on a MFEM mesh-defined coupling scheme
 *
 * @pre Coupling scheme cs_id must be registered using
 * registerMfemCouplingScheme()
 *
 * @param [in] cs_id The ID of the coupling scheme with the MFEM mesh
 * @param [in] reference_coords MFEM reference coord ParGridFunction defined over the parent mesh
 */
void registerMfemReferenceCoords( IndexT cs_id, const mfem::ParGridFunction& reference_coords );

/**
 * @brief Returns the response (RHS) vector to a given mfem::Vector
 *
 * @note This is stored as a dual vector, meaning the shared DOFs must be summed over all ranks to obtain their value.
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 * @pre Redecomp mesh must be created and up to date by calling updateMfemParallelDecomposition()
 * @pre Tribol data must be up to date for current geometry by calling update()
 *
 * @param [in] cs_id The ID of the coupling scheme with the MFEM mesh
 * @param [out] r mfem::Vector of the response (RHS) vector (properly sized, pre-allocated, and initialized)
 */
void getMfemResponse( IndexT cs_id, mfem::Vector& r );

/**
 * @brief Return the assembled contact residual on parent true DOFs
 *
 * @note Values on shared DOFs are already consistent across ranks. For the Common Plane formulation, the negative of
 * this residual is the physical force returned by the legacy getMfemResponse() path.
 * Coulomb mortar returns the same displacement residual contribution as getMfemContactResidual().force.
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 * @pre Redecomp mesh must be created and up to date by calling updateMfemParallelDecomposition()
 * @pre Tribol data must be up to date for current geometry by calling update()
 *
 * @param [in] cs_id The ID of the coupling scheme with the MFEM mesh
 * @return mfem::HypreParVector True-dof contact force vector (returned by value) on the parent mesh
 */
mfem::HypreParVector getMfemContactForce( IndexT cs_id );

/**
 * @brief Get assembled contact contributions for the Jacobian matrix
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 * @pre Redecomp mesh must be created and up to date by calling updateMfemParallelDecomposition()
 * @pre Tribol data must be up to date for current geometry by calling update()
 *
 * This method requires registration of an mfem::ParMesh with the coupling scheme. The Jacobian contributions are split
 * into a 2x2 block structure. The first row (or column) block is associated with the coordinate grid function degrees
 * of freedom and the second row (or column) block is associated with scalar pressure or Cartesian traction multiplier
 * degrees of freedom. With f = displacement residual, u = displacement, r = multiplier residual, and lambda =
 * multiplier:
 *
 *   df/du | df/dlambda
 *  ---------------
 *   dr/du | dr/dlambda
 *
 * Note, for contact methods with Lagrange multiplier constraint enforcement, Tribol will have contact contribution
 * placeholders for nodes on both contact surfaces. For instance, in a mortar method, both mortar and nonmortar nodes
 * will have placeholders in the contact contribution data structures. If, however, SINGLE_MORTAR is used as the
 * ContactMethod, the multiplier degrees of freedom are ONLY on the nonmortar surface. Tribol returns empty rows of
 * dr/du, empty columns of df/dlambda, and identity rows in dr/dlambda for unused mortar multiplier placeholders.
 * Coulomb mortar retains physical dr/dlambda derivatives on supported multiplier nodes and returns all four
 * nonsymmetric blocks as owning copies. Its force blocks are differentiated independently of the constraint blocks.
 *
 * @param cs_id Coupling scheme id with a registered MFEM mesh
 * @return Jacobian contributions as an mfem::BlockOperator
 */
std::unique_ptr<mfem::BlockOperator> getMfemBlockJacobian( IndexT cs_id );

/**
 * @brief Get the derivative of the force with respect to displacement
 *
 * Transfers the latest result out of the coupling scheme. Returns null if it was not calculated or already taken.
 *
 * @param cs_id Coupling scheme id with a registered MFEM mesh
 * @return Unique pointer to MFEM HypreParMatrix
 */
std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx( IndexT cs_id );

/**
 * @brief Get the derivative of the force with respect to pressure
 *
 * Transfers the latest result out of the coupling scheme. Returns null if it was not calculated or already taken.
 *
 * @param cs_id Coupling scheme id with a registered MFEM mesh
 * @return Unique pointer to MFEM HypreParMatrix
 */
std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp( IndexT cs_id );

/**
 * @brief Get the derivative of the gap with respect to displacement
 *
 * This is the derivative of weighted nodal gaps. Transfers the latest result out of the coupling scheme and returns
 * null if it was not calculated or already taken.
 *
 * @param cs_id Coupling scheme id with a registered MFEM mesh
 * @return Unique pointer to MFEM HypreParMatrix
 */
std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx( IndexT cs_id );

/**
 * @brief Returns a legacy-method gap vector to a given mfem::Vector
 *
 * @throws std::logic_error The scheme uses ContactFormulation; use getMfemContactGap() instead.
 *
 * @note This is stored as an MFEM dual vector, meaning the shared DOFs expect to be summed over all ranks to obtain
 * their value.
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 * @pre Redecomp mesh must be created and up to date by calling updateMfemParallelDecomposition()
 * @pre Response vector must be up to date for current geometry by calling update()
 *
 * @param [in] cs_id Coupling scheme id with a registered MFEM mesh
 * @param [out] g Nodal gap values (values do not have to be pre-allocated) on the parent-linked boundary submesh
 */
void getMfemGap( IndexT cs_id, mfem::Vector& g );

/**
 * @brief Returns the true-dof contact gap vector
 *
 * @note Values on shared DOFs are already consistent across ranks. Nodal Energy Mortar returns normalized gaps for
 * penalty enforcement and weighted gaps for LM enforcement. Quadrature-point methods have no nodal gap output.
 * @throws std::logic_error The latest calculation did not provide nodal gaps.
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 * @pre Redecomp mesh must be created and up to date by calling updateMfemParallelDecomposition()
 * @pre Tribol data must be up to date for current geometry by calling update()
 *
 * @param [in] cs_id The ID of the coupling scheme with the MFEM mesh
 * @return mfem::HypreParVector True-dof contact gap vector (returned by value) on the parent-linked boundary submesh
 */
mfem::HypreParVector getMfemContactGap( IndexT cs_id );

/**
 * @brief Returns reference to nodal pressure vector on the submesh surface
 *
 * @pre Coupling scheme cs_id must be registered using
 * registerMfemCouplingScheme()
 *
 * @param cs_id Coupling scheme id with a registered MFEM mesh
 * @return mfem::ParGridFunction& Nodal pressure vector defined on the
 * parent-linked boundary submesh
 */
mfem::ParGridFunction& getMfemPressure( IndexT cs_id );

/**
 * @brief Returns a reference to the true-dof contact pressure vector
 *
 * For nodal penalty enforcement, returns the pressure in the latest coupling-scheme result. For LM enforcement,
 * returns the caller-supplied multiplier input for the next update(), available immediately after registration.
 * Formulations retain neither this input nor computed pressure. Quadrature-point methods have no nodal pressure.
 *
 * @throws std::logic_error No nodal pressure output or multiplier input is available.
 *
 * @pre Coupling scheme cs_id must be registered using registerMfemCouplingScheme()
 *
 * @param [in] cs_id The ID of the coupling scheme with the MFEM mesh
 * @return mfem::HypreParVector& Reference to the true-dof pressure vector defined on the parent-linked boundary submesh
 */
mfem::HypreParVector& getMfemContactPressure( IndexT cs_id );

/**
 * @brief Updates mesh parallel decomposition and related grid functions/Jacobian when coordinates are updated
 *
 * @param n_ranks Number of ranks in the parallel decomposition; automatically determine when set to 0 (default)
 * @param force_new_redecomp If true, construct a new RedecompMesh even if displacement threshold is not met (default =
 * false)
 *
 * @pre Coupling schemes must be registered using registerMfemCouplingScheme()
 */
void updateMfemParallelDecomposition( int n_ranks = 0, bool force_new_redecomp = false );

/**
 * @brief Create VisIt output of the parallel repartitioned RedecompMesh
 *
 * @pre Coupling schemes must be registered using registerMfemCouplingScheme()
 * @pre Redecomp mesh must be created and up to date by calling updateMfemParallelDecomposition()
 *
 * @param output_id Unique identifier in the saved file name (usually cycle number)
 */
void saveRedecompMesh( int output_id );

} /* namespace tribol */

#endif /* BUILD_REDECOMP */

#endif /* SRC_TRIBOL_INTERFACE_MFEM_TRIBOL_HPP_ */
