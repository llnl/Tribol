// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_

#include "Physics.hpp"

namespace tribol {
/*!
 *
 * \brief computes penalty stiffness for Common Plane + Penalty
 *
 * \param [in] K1/t1 contact spring stiffness for face 1 (bulk_modulus/element_thickness for face 1)
 * \param [in] K2/t2 contact spring stiffness for face 2 (bulk_modulus/element_thickness for face 2)
 *
 * \return face-pair based, element-wise penalty stiffness per area
 *
 *
 * \pre Bulk modulus and element thickness arrays are registered by host code
 *
 */
TRIBOL_HOST_DEVICE inline RealT ComputePenaltyStiffnessPerArea( const RealT K1_over_t1, const RealT K2_over_t2 )
{
  // compute face-pair specific penalty stiffness per unit area.
  // Note: This assumes that each face has a spring stiffness
  // equal to that side's material Bulk modulus, K, over the
  // thickness of the volume element to which that face belongs,
  // times the overlap area. That is, K1_over_t1 * A and K2_over_t2 * A. We
  // then assume the two springs are in series and compute an
  // equivalent spring stiffness as,
  // k_eq = A*(K1_over_t1)*(K2_over_t2) / ((K1_over_t1)+(K2_over_t2).
  // Note, the host code registers each face's (K/t) as a penalty scale.
  //
  // UNITS: we multiply k_eq above by the overlap area A, to get a
  // stiffness per unit area. This will make the force calculations
  // commensurate with the previous calculations using only the
  // constant registered penalty scale.

  return K1_over_t1 * K2_over_t2 / ( K1_over_t1 + K2_over_t2 );

}  // end ComputePenaltyStiffnessPerArea

/**
 * @brief Build the shared CommonPlane quadrature rows for the current overlap cells.
 *
 * The generated rows contain mapped parent-face coordinates, field values,
 * basis values, integration weights, normals, gaps, and penalty coefficients.
 * Explicit force, damping, diagnostics, and stability calculations consume
 * this same batch.
 *
 * @param [in,out] cs CommonPlane coupling scheme that owns the row batch
 * @return zero on success and nonzero if row generation fails
 */
int BuildCommonPlaneContactRows( CouplingScheme* cs );

/**
 * @brief Compute the explicit penalty stability vote from the shared CommonPlane rows.
 *
 * @param [in,out] cs CommonPlane coupling scheme that owns the row batch and diagnostics
 * @param [in,out] timestep Application timestep reduced by the contact stability vote
 * @return zero on success and nonzero when required stability data are unavailable or invalid
 */
int ComputeCommonPlanePenaltyStabilityTimeStep( CouplingScheme* cs, RealT& timestep );

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
int ApplyNormal<COMMON_PLANE, PENALTY>( CouplingScheme* cs );

/*!
 *
 * \brief routine to apply interface physics in the direction tangential to the interface
 *
 * \param [in] cs pointer to the coupling scheme
 *
 * \return 0 if no error
 *
 */
template <>
int ApplyTangential<COMMON_PLANE, PENALTY, VISCOUS_TANGENTIAL>( CouplingScheme* cs );

}  // end namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_ */
