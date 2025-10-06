// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_PHYSICS_MORTAR_HPP_
#define SRC_PHYSICS_MORTAR_HPP_

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
 * \brief computes all of the nonmortar gaps to determine active set of contact constraints
 *
 * \param [in] cs pointer to coupling scheme
 *
 */
void ComputeSmoothMortarGaps( CouplingScheme* cs );

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
 * \note explicit specialization for Smooth mortar method
 *
 * \param [in] elem surface contact element object for contact face-pair
 *
 */
template <>
void ComputeNodalGap<SMOOTH_MORTAR>( SurfaceContactElem& elem );

/*!
 *
 * \brief method to compute the Jacobian contributions of the contact residual
 *        term with respect to either the primal or dual variable for a Smooth
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
 *        constraint with respect to either the primal or dual variable for a Smooth
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
int ApplyNormal<SMOOTH_MORTAR, LAGRANGE_MULTIPLIER>( CouplingScheme* cs );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of
 *        the contact residual term with respect to the primal variable for a Smooth
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template <>
void ComputeResidualJacobian<SMOOTH_MORTAR, PRIMAL>( SurfaceContactElem& elem );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of
 *        the contact residual term with respect to the dual variable for a Smooth
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template <>
void ComputeResidualJacobian<SMOOTH_MORTAR, DUAL>( SurfaceContactElem& elem );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of
 *        the contact gap  constraint with respect to the primal variable for a Smooth
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template <>
void ComputeConstraintJacobian<SMOOTH_MORTAR, PRIMAL>( SurfaceContactElem& elem );

/*!
 *
 * \brief explicit specialization of method to compute the Jacobian contributions of
 *        the contact gap  constraint with respect to the dual variable for a Smooth
 *        contact face-pair.
 *
 * \param [in] elem surface contact element struct
 *
 */
template <>
void ComputeConstraintJacobian<SMOOTH_MORTAR, DUAL>( SurfaceContactElem& elem );

/*!
 *
 * \brief wrapper to call specific routines to compute block Jacobian contributions
 *
 * \param [in] elem surface contact element struct
 *
 */
void ComputeSmoothMortarJacobian( SurfaceContactElem& elem );

#ifdef TRIBOL_USE_ENZYME
int ApplySmoothNormalEnzyme( CouplingScheme* cs );

void find_normal(const RealT* coord1, const RealT* coord2, RealT* normal);

void determine_legendre_nodes(int N, RealT* N_i);

void determine_legendre_weights(int N, RealT* W);

void iso_map(const RealT* coord1, const RealT* coord2, RealT xi, RealT* mapped_coord);

bool segmentsIntersect(const RealT A0[2], const RealT A1[2],
                       const RealT B0[2], const RealT B1[2],
                       RealT intersection[2]);

void get_projections(const RealT* A0, const RealT* A1, const RealT* B0, const RealT* B1, RealT* projections, RealT del);

void compute_integration_bounds(const RealT* projections, RealT* integration_bounds, RealT del);

void modify_bounds(const RealT* integration_bounds, RealT del, RealT* modified_bounds);

void modify_bounds_for_weight(const RealT* integration_bounds, RealT del, RealT* modified_bounds);

void compute_quadrature_point(const RealT* integration_bounds, const RealT* A0, const RealT* A1, int N, RealT* quad_points);

void assign_weights(const RealT* integration_bounds, int N, RealT* weights);

RealT compute_gap(const RealT* p, const RealT* B0, const RealT* B1, const RealT* nB, const RealT* A0, const RealT* A1);

RealT compute_modified_gap(RealT gap, RealT* nA, RealT* nB); 

RealT compute_contact_potential(RealT gap, RealT k1, RealT k2);

void ComputeSmoothMortarEnergyEnzyme(const RealT* coords, RealT del, RealT k1, RealT k2, int N, RealT lenA, RealT* projections, RealT* energy);

void ComputeSmoothMortarForceEnzyme(RealT* coords, RealT del, RealT k1, RealT k2, int N, RealT lenA, RealT* projections, RealT* force);

void ComputeSmoothMortarJacobianEnzyme(RealT* coords, RealT del, RealT k1, RealT k2, int N, RealT lenA, RealT* projections, RealT* force, RealT* jacobian);




// void ComputeSmoothMortarEnergyEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* g1, int size1,
//                                      const RealT* x2, RealT* f2, int size2 );


// void ComputeSmoothMortarForceEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* g1, int size1,
//                                      const RealT* x2, RealT* f2, int size2 );

// void ComputeSmoothMortarJacobianEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* df1dx1,
//                                         RealT* df1dx2, RealT* df1dn1, RealT* df1dp1, RealT* g1, RealT* dg1dx1, RealT* dg1dx2,
//                                         RealT* dg1dn1, int size1, const RealT* x2, RealT* f2, RealT* df2dx1, RealT* df2dx2,
//                                         RealT* df2dn1, RealT* df2dp1, int size2 );
#endif

}  // namespace tribol

#endif /* SRC_PHYSICS_MORTAR_HPP_ */
