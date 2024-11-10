// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_PHYSICS_PHYSICS_HPP_
#define SRC_PHYSICS_PHYSICS_HPP_

#include "tribol/common/Parameters.hpp"

namespace tribol
{

// forward declarations
struct InterfacePair;
struct SurfaceContactElem;
class CouplingScheme;

/*!
 *
 * \brief applies interface physics method to obtain residual and/or jacobian evaluations
 *
 * \param [in] cs pointer to the coupling scheme
 * \param [in] cycle simulation cycle
 * \param [in] t simulation time step
 *
 * \return 0 if no errors
 *
 */
int ApplyInterfacePhysics( CouplingScheme* cs,
                           int cycle, RealT t );
/*!
 *
 * \brief applies interface method in the normal direction
 *
 * \param [in] cs pointer to the coupling scheme
 *
 * \return 0 if no error
 *
 */
template< ContactMethod M, EnforcementMethod E >
int ApplyNormal( CouplingScheme* cs );

/*!
 *
 * \brief applies interface method in the tangential direction
 *
 * \param [in] cs pointer to the coupling scheme
 *
 * \return 0 if no error
 *
 */
template< ContactMethod M, 
          EnforcementMethod E,
          ContactModel Model >
int ApplyTangential( CouplingScheme* cs );

/*!
 *
 * \brief computes and stores interface physics method specific data
 *
 * \param [in] cs pointer to the coupling scheme
 *
 * \return 0 if no error
 *
 */
template< ContactMethod M >
int GetMethodData( CouplingScheme* cs );

} // end namespace TRIBOL

#endif /* SRC_PHYSICS_PHYSICS_HPP_ */
