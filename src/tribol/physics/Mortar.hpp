// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_MORTAR_HPP_
#define SRC_TRIBOL_PHYSICS_MORTAR_HPP_

#include "tribol/common/Parameters.hpp"
#include "Physics.hpp"

namespace tribol {

// forward declarations
struct SurfaceContactElem;

enum VariableType
{
  PRIMAL,
  DUAL,

  NUM_VARIABLES
};

/*!
 *
 * \brief computes the integral of (phi_a * phi_b) over a contact
 *        overlap for all (a,b) combinations.
 *
 * \note the mortar weights are stored on the SurfaceContactElem object
 *
 * \param [in] elem surface contact element object for contact face-pair
 *
 *
 */
void ComputeMortarWeights( SurfaceContactElem& elem );

/*!
 *
 * \brief computes all of the nonmortar gaps to determine active set of contact constraints
 *
 * \param [in] cs pointer to coupling scheme
 *
 */
void ComputeSingleMortarGaps( CouplingScheme* cs );

/*!
 *
 * \brief compute a contact element's contribution to nodal gaps
 *
 * \param [in] elem surface contact element object for contact face-pair
 *
 */
template <ContactMethod M>
void ComputeNodalGap( SurfaceContactElem& elem );

/*!
 *
 * \brief compute a contact element's contribution to nodal gaps
 *
 * \note explicit specialization for single mortar method
 *
 * \param [in] elem surface contact element object for contact face-pair
 *
 */
template <>
void ComputeNodalGap<SINGLE_MORTAR>( SurfaceContactElem& elem );

/*!
 *
 * \brief method to compute the Jacobian contributions of the contact residual
 *        term with respect to either the primal or dual variable for a single
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template <ContactMethod M, VariableType V>
void ComputeResidualJacobian( SurfaceContactElem& elem );

/*!
 *
 * \brief method to compute the Jacobian contributions of the contact gap
 *        constraint with respect to either the primal or dual variable for a single
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template <ContactMethod M, VariableType V>
void ComputeConstraintJacobian( SurfaceContactElem& elem );

/*!
 *
 * \brief routine to apply interface physics in the direction normal to the interface
 *
 * \param [in] cs pointer to the coupling scheme
 *
 * \return 0 if no error
 *
 */
template <>
int ApplyNormal<SINGLE_MORTAR, LAGRANGE_MULTIPLIER>( CouplingScheme* cs );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of
 *        the contact residual term with respect to the primal variable for a single
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template <>
void ComputeResidualJacobian<SINGLE_MORTAR, PRIMAL>( SurfaceContactElem& elem );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of
 *        the contact residual term with respect to the dual variable for a single
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template <>
void ComputeResidualJacobian<SINGLE_MORTAR, DUAL>( SurfaceContactElem& elem );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of
 *        the contact gap  constraint with respect to the primal variable for a single
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template <>
void ComputeConstraintJacobian<SINGLE_MORTAR, PRIMAL>( SurfaceContactElem& elem );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of
 *        the contact gap  constraint with respect to the dual variable for a single
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template <>
void ComputeConstraintJacobian<SINGLE_MORTAR, DUAL>( SurfaceContactElem& elem );

/*!
 *
 * \brief wrapper to call specific routines to compute block Jacobian contributions
 *
 * \param [in] elem surface contact element struct
 *
 */
void ComputeSingleMortarJacobian( SurfaceContactElem& elem );

#ifdef TRIBOL_USE_ENZYME
/*!
 * \brief Computes frictionless mortar forces and Jacobians (using Enzyme) following Puso and Laursen (2004)
 *
 * \param [in] cs pointer to the coupling scheme
 *
 * \return 0 if no error
 *
 */
int ApplyNormalEnzyme( CouplingScheme* cs );

/**
 * @brief Computes the frictionless mortar forces for a 3D quad element (following Puso and Laursen (2004))
 *
 * @param [in] x1 Nodal coordinates for element 1 (stored by node, e.g. [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2,
 *                z3])
 * @param [in] n1 Nodal unit normal vectors for element 1 (stored by node, e.g. [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1,
 *                z2, z3])
 * @param [in] p1 Nodal pressures for element 1
 * @param [out] f1 Nodal forces for element 1 (stored by node, e.g. [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3])
 * @param [out] g1 Nodal gaps for element 1
 * @param [in] size1 Number of nodes on element 1
 * @param [in] x2 Nodal coordinates for element 2 (stored by node, e.g. [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2,
 *                z3])
 * @param [out] f2 Nodal forces for element 2 (stored by node, e.g. [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3])
 * @param [in] size2 Number of nodes on element 2
 */
void ComputeMortarForceEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* g1, int size1,
                               const RealT* x2, RealT* f2, int size2 );

/**
 * @brief Computes the frictionless mortar forces and Jacobian for a 3D quad element (following Puso and Laursen (2004))
 *
 * @param [in] x1 Nodal coordinates for element 1 (stored by node, e.g. [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2,
 *                z3])
 * @param [in] n1 Nodal unit normal vectors for element 1 (stored by node, e.g. [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1,
 *                z2, z3])
 * @param [in] p1 Nodal pressures for element 1
 * @param [out] f1 Nodal forces for element 1 (stored by node, e.g. [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3])
 * @param [out] df1dx1 Derivative of element 1 nodal forces with respect to element 1 nodal coordinates (size =
 *                     num_nodes_per_elem^2 x spatial dim^2)
 * @param [out] df1dx2 Derivative of element 1 nodal forces with respect to element 2 nodal coordinates (size =
 *                     num_nodes_per_elem^2 x spatial dim^2)
 * @param [out] df1dn1 Derivative of element 1 nodal forces with respect to element 1 nodal normals (size =
 *                     num_nodes_per_elem^2 x spatial dim^2)
 * @param [out] df1dp1 Derivative of element 1 nodal forces with respect to element 1 nodal pressures (size =
 *                     num_nodes_per_elem^2 x spatial dim)
 * @param [out] g1 Nodal gaps for element 1
 * @param [out] dg1dx1 Derivative of element 1 nodal gaps with respect to element 1 nodal coordinates (size =
 *                     num_nodes_per_elem^2 x spatial dim^2)
 * @param [out] dg1dx2 Derivative of element 1 nodal gaps with respect to element 2 nodal coordinates (size =
 *                     num_nodes_per_elem^2 x spatial dim^2)
 * @param [out] dg1dn1 Derivative of element 1 nodal gaps with respect to element 1 nodal normals (size =
 *                     num_nodes_per_elem^2 x spatial dim^2)
 * @param [in] size1 Number of nodes on element 1
 * @param [in] x2 Nodal coordinates for element 2 (stored by node, e.g. [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2,
 *                z3])
 * @param [out] f2 Nodal forces for element 2 (stored by node, e.g. [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3])
 * @param [out] df2dx1 Derivative of element 2 nodal forces with respect to element 1 nodal coordinates (size =
 *                     num_nodes_per_elem^2 x spatial dim^2)
 * @param [out] df2dx2 Derivative of element 2 nodal forces with respect to element 2 nodal coordinates (size =
 *                     num_nodes_per_elem^2 x spatial dim^2)
 * @param [out] df2dn1 Derivative of element 2 nodal forces with respect to element 1 nodal normals (size =
 *                     num_nodes_per_elem^2 x spatial dim^2)
 * @param [out] df2dp1 Derivative of element 2 nodal forces with respect to element 1 nodal pressures (size =
 *                     num_nodes_per_elem^2 x spatial dim)
 * @param [in] size2 Number of nodes on element 2
 */
void ComputeMortarJacobianEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* df1dx1,
                                  RealT* df1dx2, RealT* df1dn1, RealT* df1dp1, RealT* g1, RealT* dg1dx1, RealT* dg1dx2,
                                  RealT* dg1dn1, int size1, const RealT* x2, RealT* f2, RealT* df2dx1, RealT* df2dx2,
                                  RealT* df2dn1, RealT* df2dp1, int size2 );
#endif

/*!
 *
 * \brief method to compute mortar weights for MORTAR_WEIGHTS method
 *
 * \param [in] cs pointer to coupling scheme
 *
 * \return 0 if no error
 *
 */
template <>
int GetMethodData<MORTAR_WEIGHTS>( CouplingScheme* cs );

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_MORTAR_HPP_ */
