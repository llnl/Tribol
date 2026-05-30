// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_

#include "Physics.hpp"
#include "tribol/common/Parameters.hpp"

namespace tribol {
/*!
 *
 * \brief computes the individual (uncoupled) kinematic penalty spring stiffness for a single face
 *
 * \details This routine calculates the stiffness of a single contact spring on one side of
 *          the contact pair (either mortar or non-mortar), completely independent of (uncoupled from)
 *          the other side. The resulting uncoupled stiffnesses from both sides are later combined
 *          in series (harmonic mean) via ComputePenaltyStiffnessPerArea() to obtain the coupled
 *          equivalent interface stiffness per unit area.
 *
 * \param [in] kinematic_calc calculation option (constant or element-based)
 * \param [in] thickness element thickness
 * \param [in] tiny_length numeric safeguard offset to avoid division by zero
 * \param [in] pen_scale mesh penalty scale
 * \param [in] mat_mod material modulus (bulk modulus)
 * \param [in] const_penalty constant penalty value
 * \param [out] stiffness computed uncoupled spring stiffness for this face/side
 *
 */
TRIBOL_HOST_DEVICE inline void ComputeUncoupledStiffness( const KinematicPenaltyCalculation kinematic_calc,
                                                          const RealT thickness, const RealT tiny_length,
                                                          const RealT pen_scale, const RealT mat_mod,
                                                          const RealT const_penalty, RealT& stiffness )
{
  if ( kinematic_calc == KINEMATIC_CONSTANT ) {
    stiffness = pen_scale * const_penalty;
  } else if ( kinematic_calc == KINEMATIC_ELEMENT ) {
    RealT denom = thickness;
    if ( denom < tiny_length ) {
      denom = tiny_length;
    }
    stiffness = pen_scale * mat_mod / denom;
  }
}

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
