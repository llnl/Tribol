// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANE_HPP_

#include "Physics.hpp"
#include "tribol/mesh/MeshData.hpp"

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

namespace detail {

/**
 * @brief Compute the effective series kinematic penalty stiffness per unit area for one Common Plane face pair.
 * @param mesh1 First surface mesh view.
 * @param mesh2 Second surface mesh view.
 * @param pen_options Penalty enforcement options selecting KINEMATIC_CONSTANT or KINEMATIC_ELEMENT.
 * @param index1 Element index on mesh1.
 * @param index2 Element index on mesh2.
 * @param[out] penalty_stiff_per_area Computed series penalty stiffness per unit area.
 * @return True if thicknesses are valid; false if negative element thickness was encountered.
 */
TRIBOL_HOST_DEVICE inline bool computeCommonPlanePenaltyStiffnessPerArea( const MeshData::Viewer& mesh1,
                                                                          const MeshData::Viewer& mesh2,
                                                                          const PenaltyEnforcementOptions& pen_options,
                                                                          IndexT index1, IndexT index2,
                                                                          RealT& penalty_stiff_per_area )
{
  penalty_stiff_per_area = 0.0;
  const RealT pen_scale1 = mesh1.getElementData().m_penalty_scale;
  const RealT pen_scale2 = mesh2.getElementData().m_penalty_scale;
  switch ( pen_options.kinematic_calculation ) {
    case KINEMATIC_CONSTANT: {
      const RealT stiffness1 = pen_scale1 * mesh1.getElementData().m_penalty_stiffness;
      const RealT stiffness2 = pen_scale2 * mesh2.getElementData().m_penalty_stiffness;
      penalty_stiff_per_area = ComputePenaltyStiffnessPerArea( stiffness1, stiffness2 );
      return true;
    }
    case KINEMATIC_ELEMENT: {
      const RealT t1 = mesh1.getElementData().m_thickness[index1] + pen_options.tiny_length;
      const RealT t2 = mesh2.getElementData().m_thickness[index2] + pen_options.tiny_length;
      const RealT stiffness1 = pen_scale1 * mesh1.getElementData().m_mat_mod[index1] / t1;
      const RealT stiffness2 = pen_scale2 * mesh2.getElementData().m_mat_mod[index2] / t2;
      penalty_stiff_per_area = ComputePenaltyStiffnessPerArea( stiffness1, stiffness2 );
      return !( t1 < 0.0 || t2 < 0.0 );
    }
    default:
      return true;
  }
}

/**
 * @brief Compute the effective normal gap-rate penalty coefficient per unit area for one Common Plane face pair.
 * @param mesh1 First surface mesh view.
 * @param mesh2 Second surface mesh view.
 * @param element_penalty Effective kinematic penalty stiffness per unit area.
 * @param rate_calc Rate penalty calculation option.
 * @return Rate penalty coefficient per unit area, or 0.0 when no rate penalty is configured.
 */
TRIBOL_HOST_DEVICE inline RealT computeCommonPlaneRatePenalty( const MeshData::Viewer& mesh1,
                                                               const MeshData::Viewer& mesh2, RealT element_penalty,
                                                               RatePenaltyCalculation rate_calc )
{
  switch ( rate_calc ) {
    case RATE_CONSTANT:
      return 0.5 *
             ( mesh1.getElementData().m_rate_penalty_stiffness + mesh2.getElementData().m_rate_penalty_stiffness );
    case RATE_PERCENT:
      return element_penalty * 0.5 *
             ( mesh1.getElementData().m_rate_percent_stiffness + mesh2.getElementData().m_rate_percent_stiffness );
    case NO_RATE_PENALTY:
    default:
      return 0.0;
  }
}

}  // namespace detail

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
