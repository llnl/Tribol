// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)
 
#ifndef TRIBOL_PARAMETERS_HPP_
#define TRIBOL_PARAMETERS_HPP_

// Tribol includes
#include "tribol/common/BasicTypes.hpp"

#include <string>

namespace tribol
{

// Internal Helper Method for Enums
namespace
{

//------------------------------------------------------------------------------
inline bool in_range( int target, int N )
{
  // NOTE: assumes indexing starts from 0
  return( (target >= 0) && ( target < N ) );
}

} // end anonymous namespace

constexpr int ANY_MESH = -1;

/*!
 * \brief Enumerates the logging level options
 */
enum LoggingLevel
{
   TRIBOL_UNDEFINED, ///! Undefined 
   TRIBOL_DEBUG,     ///! Debug and higher
   TRIBOL_INFO,      ///! Info and higher
   TRIBOL_WARNING,   ///! Warning and higher
   TRIBOL_ERROR,     ///! Errors only
   NUM_LOGGING_LEVELS = TRIBOL_ERROR
};

/*!
 * \brief Enumerates the interface element types  
 */
enum InterfaceElementType
{
   UNDEFINED_ELEMENT, ///! Undefined
   LINEAR_EDGE,       ///! 1D linear edge 
   LINEAR_TRIANGLE,   ///! 2D linear triangle
   LINEAR_QUAD,       ///! 2D linear quadrilateral
   LINEAR_TET,        ///! 3D linear tetrahedron (volume methods and test mesh class)
   LINEAR_HEX,        ///! 3D linear hexahedron (volume methods and test mesh class)
   NUM_CONTACT_ELEMENTS = LINEAR_HEX
};

/*!
 * \brief Enumerates the visualization options
 */
enum VisType
{
   UNDEFINED_VIS,               ///! Undefined
   VIS_MESH,                    ///! Print registered mesh(es)
   VIS_FACES,                   ///! Print active interface-faces (method specific)
   VIS_OVERLAPS,                ///! Print interface face-face overlaps
   VIS_MESH_AND_OVERLAPS,       ///! Print registered mesh(es) and interface overlaps
   VIS_FACES_AND_OVERLAPS,      ///! Print active interface-faces and face-face overlaps
   VIS_MESH_FACES_AND_OVERLAPS, ///! Print registered mesh(es), active faces, and overlaps
   NUM_VIS_TYPES = VIS_MESH_FACES_AND_OVERLAPS
};

/*!
 * \brief Enumerates the contact modes that specify paired topologies in an interaction
 *
 *
 * The contact mode enumerates the two-sided pairing of mesh entities 
 * (element topologies) in an interaction. These may be combinations 
 * of surface and volume interactions.
 */
enum ContactMode
{
  SURFACE_TO_SURFACE,            ///! surface-to-surface interaction
  SURFACE_TO_SURFACE_CONFORMING, ///! conforming surface-to-surface interaction
  SURFACE_TO_VOLUME,             ///! surface-to-volume interaction
  VOLUME_TO_VOLUME,              ///! volume-to-volume interaction
  NUM_CONTACT_MODES
};

/*!
 * \brief Enumerates the available contact cases  
 *
 * The contact case enumerates specializations, or Tribol use-cases that require
 * special algorithmic considerations beyond standard Lagrangian contact. Note,
 * the use of auto-contact cannot be used with the tied contact variants. This 
 * may be a limitation down the road, but for now, the very use case of TIED_*
 * implies that a host-code knows what two surfaces are tied in a given interaction,
 * and is able to explicitly specify these when registering the meshes, coupling schemes,
 * and/or boundary attributes.
 */
enum ContactCase
{
  NO_CASE,       ///! No case specified for chosen mode and/or method
  AUTO,          ///! Auto contact
  TIED_NORMAL,   ///! Tied in the surface normal direction
  TIED_FULL,     ///! Tied in the surface normal and tangential directions
  NO_SLIDING,    ///! User may specify no sliding, simplifying search update
  NUM_CONTACT_CASES
};

/*!
 * \brief Enumerates the available contact method options.
 *
 * The contact method is the numerical method used to discretize the
 * contact surface in order to integrate the weak form contact integrals.
 */
enum ContactMethod // all mortar methods go first
{
  SINGLE_MORTAR,      ///! Single mortar per Puso 2003
  ALIGNED_MORTAR,     ///! Aligned mortar to be used with ContactCase = NO_SLIDING
  MORTAR_WEIGHTS,     ///! Method that only returns mortar weights per single mortar method
  COMMON_PLANE,       ///! Common plane method, currently with single integration point
  NUM_CONTACT_METHODS
};

/*!
 * \brief Enumerates the available contact model options.
 *
 * The contact model enumerates interface constitutive modeling options.
 * These may be paired exclusively with certain contact cases and methods
 * depending on appropriate physics and/or available implementations.
 */
enum ContactModel
{
  NO_CONTACT,                     ///! No contact
  FRICTIONLESS,                   ///! Frictionless, normal contact only
  COULOMB,                        ///! Coulomb friction model, not supported
  ADHESION_SEPARATION_SCALAR_LAW, ///! Scalar pressure law for the separation of adhered surfaces (Used with tied contact)
  NULL_MODEL,                     ///! Null model, for use with ContactMethod = MORTAR_WEIGHTS
  NUM_CONTACT_MODELS
};

/*!
 * \brief Enumerates the available enforcement method options.
 *
 * The enforcement method is the method used to enforce the contact
 * constraints paired with a given contact numerical method.
 */
enum EnforcementMethod
{
  PENALTY,               ///! Penalty enforcement method for gap only
  LAGRANGE_MULTIPLIER,   ///! Lagrange multiplier with system output
  NULL_ENFORCEMENT,      ///! Null enforcement, for use with ContactMethod = MORTAR_WEIGHTS
  NUM_ENFORCEMENT_METHODS
};

/*!
 * \brief Enumerates the available spatial binning methods
 */
enum BinningMethod
{
  BINNING_GRID,               ///! Uses a spatial index to compute the pairs
  BINNING_CARTESIAN_PRODUCT,  ///! Generates all element pairs between the meshes
  BINNING_BVH,                ///! Uses a bounding volume hierarchy tree to compute the pairs
  NUM_BINNING_METHODS,
  DEFAULT_BINNING_METHOD = BINNING_GRID
};

/*!
 * \brief Enumerates the available penalty enforcement options 
 */
enum PenaltyConstraintType
{
   KINEMATIC,
   KINEMATIC_AND_RATE,
   NUM_PENALTY_OPTIONS
};

/*!
 * \brief Enumerates the available kinematic penalty stiffness calculation options
 */
enum KinematicPenaltyCalculation
{
   KINEMATIC_CONSTANT, ///! Constant penalty stiffness applied to all contacting face-pairs
   KINEMATIC_ELEMENT,  ///! Element-wise penalty stiffness calculation
   NUM_KINEMATIC_PENALTY_CALCULATION
};

/*!
 * \brief Enumerates the available rate penalty stiffness calculation options
 */
enum RatePenaltyCalculation
{
   NO_RATE_PENALTY,
   RATE_CONSTANT,  ///! Constant rate penalty stiffness
   RATE_PERCENT,   ///! Rate penalty stiffness as a percentage of the kinematic penalty stiffness
   NUM_RATE_PENALTY_CALCULATION
};

/*!
 * \brief Enumerates supported real element fields
 */
enum RealElementFields
{
   UNDEFINED_REAL_ELEMENT_FIELD,
   KINEMATIC_CONSTANT_STIFFNESS, ///! Constant kinematic penalty stiffness
   RATE_CONSTANT_STIFFNESS,      ///! Constant rate penalty stiffness
   RATE_PERCENT_STIFFNESS,       ///! Percent rate penalty stiffness
   BULK_MODULUS,                 ///! Element bulk modulus
   YOUNGS_MODULUS,               ///! Element Young's modulus
   ELEMENT_THICKNESS,            ///! Element thickness in contact normal direction
   NUM_REAL_ELEMENT_FIELDS = ELEMENT_THICKNESS 
};

/*!
 * \brief Enumerates supported integer element fields
 */
enum IntElementFields
{
   UNDEFINED_INT_ELEMENT_FIELD
};

/*!
 * \brief Enumerates supported integer nodal fields
 */
enum IntNodalFields
{
   UNDEFINED_INT_NODAL_FIELD
};

/*!
 * \brief Enumerates the available integration rules over convex polygons
 */
enum PolyInteg
{
  SINGLE_POINT,    ///! Single point integration at centroid of polygon
  FULL_TRI_DECOMP, ///! Full integration using triangular decomposition
  NUM_INTEG_RULES
};

/*!
 * \brief Enumerates the available integration methods
 */
enum IntegMethod
{
   TWB,            ///! Taylor-Wingate-Bos integration on low order triangles
   TRIBOL_INV_ISO, ///! Inverse isoparametric mapping of integration points to parent space
   MFEM,           ///! MFEM integration rule and integration methods
   NUM_INTEG_METHODS
};

/*!
 * \brief Enumerates the finite element spaces of the Jacobian contributions
 */
enum class BlockSpace
{
   MORTAR,              ///! The coordinate space on the mortar contact surface
   NONMORTAR,           ///! The coordinate space on the nonmortar contact surface
   LAGRANGE_MULTIPLIER, ///! The Lagrange multiplier space
   NUM_BLOCK_SPACES
};

/*!
 * \brief Enumerates the available implicit evaluations modes
 */
enum class ImplicitEvalMode
{
   MORTAR_JACOBIAN,          ///! Contact Jacobian evaluation only
   MORTAR_RESIDUAL,          ///! Contact residual evaluation only
   MORTAR_RESIDUAL_JACOBIAN, ///! Contact residual AND Jacobian evaluation
   MORTAR_GAP,               ///! Contact gap evaluation only
   MORTAR_WEIGHTS_EVAL       ///! Mortar weight evaluation only
};

/*!
 * \brief Enumerates the available sparse output modes
 */
enum class SparseMode
{
   MFEM_INDEX_SET,     ///! initialize mfem sparse matrix with I, J, and data
   MFEM_LINKED_LIST,   ///! initialize mfem sparse matrix with flexible, linked list option
   MFEM_ELEMENT_DENSE  ///! Stores element Jacobian contributions in an ArrayT of mfem::DenseMatrixs
};

/*!
 * \brief Enumerates the order and type of the finite element face
 */
enum FaceOrderType
{
   LINEAR,                ///! Linear Lagrange (default)
   QUADRATIC_LAGRANGE,    ///! Quadratic 9-node Lagrange face
   QUADRATIC_SERENDIPITY, ///! Quadratic 8-node serendipity face
   CUBIC_LAGRANGE,        ///! Cubic Lagrange face
   NUM_ORDERS_TYPES,
   UNDEFINED_ORDER_TYPE = NUM_ORDERS_TYPES
};

/*!
 * \brief Enumerates physical or parent basis evaluation
 */
enum BasisEvalType
{
   UNDEFINED_BASIS_EVAL_TYPE,
   PARENT,   ///! Evaluate basis in parent space
   PHYSICAL, ///! Evaluate basis in physical space
   NUM_BASIS_EVAL_TYPES = PHYSICAL
};

/*!
 * \brief Enumerates face-pair computational geometry errors
 */
enum FaceGeomError
{
   NO_FACE_GEOM_ERROR,                         ///! No face geometry error
   FACE_ORIENTATION,                           ///! Face vertices not ordered consistent with outward unit normal
   INVALID_FACE_INPUT,                         ///! Invalid input
   DEGENERATE_OVERLAP,                         ///! Issues with overlap calculation resulting in degenerate overlap
   FACE_VERTEX_INDEX_EXCEEDS_OVERLAP_VERTICES, ///! Very specific debug indexing error where face vertex count exceeds overlap vertex count in cg routine
   NUM_FACE_GEOM_ERRORS
};

/*!
 * \brief Enumerates ContactMode errors
 */
enum ModeError
{
   INVALID_MODE,
   NO_MODE_IMPLEMENTATION,
   NO_MODE_ERROR,
   NUM_MODE_ERRORS
};

/*!
 * \brief Enumerates ContactCase errors
 */
enum CaseError
{
   INVALID_CASE,
   NO_CASE_IMPLEMENTATION,
   INVALID_CASE_DATA,
   NO_CASE_ERROR,
   NUM_CASE_ERRORS
};

/*!
 * \brief Enumerates ContactMethod errors
 */
enum MethodError
{
   INVALID_ELEMENT_TYPE,
   INVALID_METHOD,
   NO_METHOD_IMPLEMENTATION,
   DIFFERENT_FACE_TYPES,
   SAME_MESH_IDS,
   SAME_MESH_IDS_INVALID_DIM,
   INVALID_DIM,
   NULL_NODAL_RESPONSE,
   NO_METHOD_ERROR,
   NUM_METHOD_ERRORS
};

/*!
 * \brief Enumerates ContactModel Errors
 */
enum ModelError
{
   INVALID_MODEL,
   NO_MODEL_IMPLEMENTATION,
   NO_MODEL_IMPLEMENTATION_FOR_REGISTERED_METHOD,
   NO_MODEL_ERROR,
   NUM_MODEL_ERRORS
};

/*!
 * \brief Enumerates ContactEnforcement Errors
 */
enum EnforcementError
{
   INVALID_ENFORCEMENT,
   INVALID_ENFORCEMENT_FOR_REGISTERED_METHOD,
   INVALID_ENFORCEMENT_OPTION,
   OPTIONS_NOT_SET,
   NO_ENFORCEMENT_IMPLEMENTATION,
   NO_ENFORCEMENT_IMPLEMENTATION_FOR_REGISTERED_METHOD,
   NO_ENFORCEMENT_IMPLEMENTATION_FOR_REGISTERED_OPTION,
   NO_ENFORCEMENT_ERROR,
   NUM_ENFORCEMENT_ERRORS
};

/*!
 * \brief Enumerates enforcement data errors
 */
enum EnforcementDataErrors
{
   ERROR_IN_REGISTERED_ENFORCEMENT_DATA,
   NO_ENFORCEMENT_DATA_ERROR,
   NUM_ENFORCEMENT_DATA_ERRORS
};

/*!
 * \brief Enumerates ContactCase info
 */
enum CaseInfo
{
   SPECIFYING_NO_SLIDING_WITH_REGISTERED_MODE,
   SPECIFYING_NO_SLIDING_WITH_REGISTERED_METHOD,
   SPECIFYING_NONE_WITH_REGISTERED_METHOD,
   NO_CASE_INFO,
   NUM_CASE_INFO
};

/*!
 * \brief Enumerates EnforcementMethod info
 */
enum EnforcementInfo
{
   SPECIFYING_NULL_ENFORCEMENT_WITH_REGISTERED_METHOD,
   NO_ENFORCEMENT_INFO,
   NUM_ENFORCEMENT_INFO
};

/*!
 * \brief Struct to hold Lagrange multiplier enforcement and implicit evaluation options
 */
struct LagrangeMultiplierImplicitOptions
{
public:
   bool enforcement_option_set {false};
   
   ImplicitEvalMode eval_mode;    ///! Implicit evaluation mode for residual, jacobian and gaps
   SparseMode sparse_mode;        ///! Mode for assembling sparse matrix contributions
};

/*!
 * \brief Struct to hold penalty enforcement options
 */
struct PenaltyEnforcementOptions
{
public:
   PenaltyConstraintType constraint_type;
   KinematicPenaltyCalculation kinematic_calculation;
   RatePenaltyCalculation rate_calculation;
 
   bool constraint_type_set {false};
   bool kinematic_calc_set  {false};
   bool rate_calc_set       {false};

   RealT tiny_length   {1.e-12}; ///! Small length to avoid division by zero
   RealT tiny_penalty  {1.e-12}; ///! Small penalty to avoid division by zero
};

/*!
 * \brief Struct wrapping constraint enforcement options
 */
struct EnforcementOptions
{
public:
   PenaltyEnforcementOptions         penalty_options;
   LagrangeMultiplierImplicitOptions lm_implicit_options;
};

  /*!
  * \brief Coupling scheme parameters struct
  */
  struct Parameters
  {
    CommT problem_comm = TRIBOL_COMM_WORLD;  ///! MPI communicator for the problem

    RealT overlap_area_frac     = 1.0e-8;  ///! Ratio of overlap area to largest face area for contact inclusion
    RealT gap_tol_ratio         = 1.0e-12; ///! Ratio for determining tolerance for active contact gaps 
    RealT gap_separation_ratio  = 0.75;    ///! Ratio for determining allowable separation in geometric filtering
    RealT gap_tied_tol          = 0.1;     ///! Ratio for determining max separation tied contact can support
    RealT len_collapse_ratio    = 1.0e-8;  ///! Ratio of face length providing topology collapse length tolerance
    RealT projection_ratio      = 1.0e-10; ///! Ratio for defining nonzero projections
    RealT auto_contact_pen_frac = 0.95;    ///! Max allowable interpenetration as percent of element thickness for contact candidacy
    RealT timestep_pen_frac     = 3.0e-1;  ///! Max allowable interpenetration as percent of element thickness prior to triggering timestep vote
    RealT timestep_scale        = 1.0;     ///! Scale factor (>0) applied to the timestep vote giving users some control over the vote

    int vis_cycle_incr          = 100;     ///! Frequency for visualizations dumps
    VisType vis_type            = VIS_OVERLAPS; ///! Type of interface physics visualization output
    bool enable_timestep_vote   = false;   ///! True if host-code desires the timestep vote to be calculated and returned

    bool auto_interpen_check    = false;   ///! True if the auto-contact interpenetration check is used for full-overlap pairs

    double auto_contact_len_scale_factor;  ///! Scale factor applied to element thickness for auto contact length scale
    
   // Interpenetration check for auto-contact. If true, this will check a full-overlap 
   // face-pair configuration in the computational geoemtry routines to preclude 
   // auto-contact of opposite sides of thin structures/plates. If the full-overlap 
   // interpenetration kinematic gap is more than the smallest thickness of the 
   // constituent face elements, then we don't consider the face-pair a contact candidate.
   // Note, auto-contact will require registration of element thicknesses.
    bool auto_contact_check     = false;   ///! True if auto-contact checks should be enabled
  };

} // namespace tribol


#endif /* TRIBOL_PARAMETERS_HPP_ */


