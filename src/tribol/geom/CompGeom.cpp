// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "CompGeom.hpp"

#include "tribol/common/BasicTypes.hpp"

namespace tribol {

void ContactPlanePair::globalTo2DLocalCoords( RealT pX, RealT pY, RealT pZ, RealT& pLX, RealT& pLY,
                                              int TRIBOL_UNUSED_PARAM( size ) )
{
  // compute the vector between the point on the plane and the contact plane point
  RealT vX = pX - m_cX;
  RealT vY = pY - m_cY;
  RealT vZ = pZ - m_cZ;

  // project this vector onto the {e1,e2} local basis. This vector is
  // in the plane so the out-of-plane component should be zero.
  pLX = vX * m_e1X + vY * m_e1Y + vZ * m_e1Z;  // projection onto e1
  pLY = vX * m_e2X + vY * m_e2Y + vZ * m_e2Z;  // projection onto e2

  return;

}  // end ContactPlanePair::globalTo2DLocalCoords()

}  // namespace tribol
