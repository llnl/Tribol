// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// tribol includes
#include "Physics.hpp"

#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/utils/ContactPlaneOutput.hpp"

// Axom includes
#include "axom/slic.hpp"

namespace tribol {

int ApplyInterfacePhysics( CouplingScheme* cs, int TRIBOL_UNUSED_PARAM( cycle ), RealT TRIBOL_UNUSED_PARAM( t ) )
{
  // call the appropriate normal and tangential interface physics
  // routines based on method, enforcement strategy, and interface
  // model combinations. Note that combinations that are not yet
  // implemented or are not possible are checked in a call to the
  // API to register a coupling scheme; such checks are not required
  // here.

  int err_nrml = false;
  int err_tang = false;
  int err_data = false;

  // switch over numerical method
  switch ( cs->getContactMethod() ) {
    case COMMON_PLANE: {

      // switch over enforcement method for normal (i.e. normal direction) enforcement
      switch ( cs->getEnforcementMethod() ) {
        case PENALTY: {

          switch ( cs->getContactCase() ) {
            case TIED_FULL: {

              err_nrml = ApplyNormal<COMMON_PLANE, PENALTY>( cs );

              switch ( cs->getContactModel() ) {
                case FRICTIONLESS: {
                  err_tang = ApplyTangential<COMMON_PLANE, PENALTY, TIED_FULL, FRICTIONLESS>( cs );
                  break;
                }

                case ADHESION_SEPARATION_SCALAR_LAW: {
                  // no-op
                  // TODO implement this
                  break;
                }

                default: {
                  // no-op; no other tangential components
                }

              }  // end switch on contact model
              break;
            } // end case tied full

            default: { // apply normal for all other contact cases
              err_nrml = ApplyNormal<COMMON_PLANE, PENALTY>( cs );
              break;
            }

          } // end switch on contactCase
          break;

        } // end case penalty

        default: {
          // no-op; no other enforcement methods for common plane
          break;
        }

      } // end switch over enforcement method

      break;
    } // end case COMMON_PLANE

    case SINGLE_MORTAR: {

      switch ( cs->getEnforcementMethod() ) {
        case LAGRANGE_MULTIPLIER: {

          switch ( cs->getContactModel() ) {
            case FRICTIONLESS: {

              err_nrml = ApplyNormal<SINGLE_MORTAR, LAGRANGE_MULTIPLIER>( cs );
              break;
            }
 
            default: {
              break;
            }

          }  // end switch on contact model
          break;
        } // end case Lagrange multiplier

        default: {
          // no-op for all other enforcement methods
          break;
        }
      } // end switch on enforcement method
    }// end case SINGLE_MORTAR

    case ALIGNED_MORTAR: {

      switch ( cs->getEnforcementMethod() ) {
        case LAGRANGE_MULTIPLIER: {

          switch ( cs->getContactModel() ) {
            case FRICTIONLESS: {

              err_nrml = ApplyNormal<ALIGNED_MORTAR, LAGRANGE_MULTIPLIER>( cs );
              break;
            }

            default: {
              // no-op; no other models for aligned mortar
              break;
            }
          }  // end switch on contact model
          break;
        } // end case Lagrange multiplier

        default: {
          // no-op; no other enforcement methods
          break;
        }

      } // end switch on enforcement method
      break;
    } // end case ALIGNED_MORTAR

    case MORTAR_WEIGHTS: {
      // no enforcement for this method and no need to call visualization.
      err_data = GetMethodData<MORTAR_WEIGHTS>( cs );
      break;

    default:
      // don't do anything. Note, no need to throw an error here as unimplemented
      // interface methods will already have been caught
      break;

    } // end case mortar weights
  }  // end switch (method)

  // error checking
  if ( err_nrml != 0 ) {
    // note, not all ranks will get here if a rank has null-meshes
    SLIC_WARNING( "ApplyInterfacePhysics: error in application of "
                  << "'normal' physics method for "
                  << "coupling scheme, " << cs->getId() << "." );

    return err_nrml;
  } else if ( err_tang != 0 ) {
    // note, not all ranks will get here if a rank has null-meshes
    SLIC_WARNING( "ApplyInterfacePhysics: error in application of "
                  << "'tangential' physics method for "
                  << "coupling scheme, " << cs->getId() << "." );

    return err_tang;
  } else if ( err_data != 0 ) {
    // note, not all ranks will get here if a rank has null-meshes
    SLIC_WARNING( "ApplyInterfacePhysics: error in call to  "
                  << "GetMethodData for coupling scheme, " << cs->getId() << "." );
    return err_data;
  } else {
    // no error
    return 0;
  }

}  // end ApplyInterfacePhysics

}  // end namespace tribol
