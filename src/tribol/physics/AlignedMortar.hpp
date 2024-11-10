// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_PHYSICS_ALIGNEDMORTAR_HPP_
#define SRC_PHYSICS_ALIGNEDMORTAR_HPP_

#include "Mortar.hpp"
#include "Physics.hpp"

namespace tribol
{

// forward declarations
struct SurfaceContactElem;

/*!
 *
 * \brief compute all aligned mortar nonmortar gaps
 *
 * \param [in] cs pointer to coupling scheme
 *
 *
 */
void ComputeAlignedMortarGaps( CouplingScheme* cs );

/*!
 *
 * \brief computes the integral of (phi_a * phi_b) over a contact 
 *        overlap for all (a,b) combinations. 
 *
 * \note the mortar weights are stored on the SurfaceContactElem object
 *
 * \param [in] elem surface contact element object for contact face-pair
 *
 * \note this is a specialized case for aligned faces using Gauss 
 *       quadrature on a four node quad.
 *
 */
void ComputeAlignedMortarWeights( SurfaceContactElem & elem );

/*!
 *
 * \brief compute a contact element's contribution to nodal gaps
 *
 * \param [in] cs pointer to the coupling scheme
 *
 */
template< ContactMethod M >
void ComputeNodalGap( SurfaceContactElem & elem );

/*!
 *
 * \brief compute a contact element's contribution to nodal gaps
 *
 * \note explicit specialization for single mortar method
 *
 * \param [in] cs pointer to the coupling scheme
 *
 */
template< >
void ComputeNodalGap< ALIGNED_MORTAR >( SurfaceContactElem & elem );

/*!
 *
 * \brief routine to apply interface physics in the direction normal to the interface 
 *
 * \param [in] cs pointer to the coupling scheme
 *
 * \return 0 if no error
 *
 */
template< >
int ApplyNormal< ALIGNED_MORTAR, LAGRANGE_MULTIPLIER >( CouplingScheme* cs );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of 
 *        the contact residual term with respect to the primal variable for a single 
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template< >
void ComputeResidualJacobian< ALIGNED_MORTAR, PRIMAL >( SurfaceContactElem & elem );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of 
 *        the contact residual term with respect to the dual variable for a single 
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template< >
void ComputeResidualJacobian< ALIGNED_MORTAR, DUAL >( SurfaceContactElem & elem );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of 
 *        the contact gap  constraint with respect to the primal variable for a single 
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template< >
void ComputeConstraintJacobian< ALIGNED_MORTAR, PRIMAL >( SurfaceContactElem & elem );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of 
 *        the contact gap  constraint with respect to the dual variable for a single 
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template< >
void ComputeConstraintJacobian< ALIGNED_MORTAR, DUAL >( SurfaceContactElem & elem );

/*!
 *
 * \brief wrapper to call specific routines to compute block Jacobian contributions
 *
 * \param [in] elem surface contact element struct
 *
 */
void ComputeAlignedMortarJacobian( SurfaceContactElem & elem );

} // end of namespace Tribol

#endif /* SRC_PHYSICS_ALIGNEDMORTAR_HPP_ */
