// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/CouplingScheme.hpp"

// Tribol includes
#include "tribol/types.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/MeshManager.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/utils/ContactPlaneOutput.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/search/InterfacePairFinder.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/geom/ContactPlane.hpp"
#include "tribol/geom/ContactPlaneManager.hpp"
#include "tribol/physics/Physics.hpp"
#include "tribol/integ/FE.hpp"

// Axom includes
#include "axom/slic.hpp"

// MFEM includes
#include "mfem.hpp"

// C++ includes
#include <cmath>

namespace tribol {

//------------------------------------------------------------------------------
// INTERNAL HELPER METHODS
//------------------------------------------------------------------------------
namespace {

//------------------------------------------------------------------------------
inline bool validMeshID( integer meshID )
{
  MeshManager& meshManager = MeshManager::getInstance();
  return ( meshID == ANY_MESH ) || meshManager.hasMesh( meshID );
}

} /* end anonymous namespace */

//------------------------------------------------------------------------------
// Struct implementation for CouplingSchemeErrors
//------------------------------------------------------------------------------
void CouplingSchemeErrors::printModeErrors()
{
  switch ( this->cs_mode_error ) {
    case INVALID_MODE: {
      SLIC_WARNING_ROOT( "The specified ContactMode is invalid." );
      break;
    }
    case NO_MODE_IMPLEMENTATION: {
      SLIC_WARNING_ROOT( "The specified ContactMode has no implementation." );
      break;
    }
    case NO_MODE_ERROR: {
      break;
    }
    default:
      break;
  }  // end switch over mode errors
}  // end CouplingSchemeErrors::printModeErrors()

//------------------------------------------------------------------------------
void CouplingSchemeErrors::printCaseErrors()
{
  switch ( this->cs_case_error ) {
    case INVALID_CASE: {
      SLIC_WARNING_ROOT( "The specified ContactCase is invalid." );
      break;
    }
    case NO_CASE_IMPLEMENTATION: {
      SLIC_WARNING_ROOT( "The specified ContactCase has no implementation." );
      break;
    }
    case NO_CASE_ERROR: {
      break;
    }
    default:
      break;
  }  // end switch over case errors
}  // end CouplingSchemeErrors::printCaseErrors()

//------------------------------------------------------------------------------
void CouplingSchemeErrors::printMethodErrors()
{
  switch ( this->cs_method_error ) {
    case INVALID_METHOD: {
      SLIC_WARNING_ROOT( "The specified ContactMethod is invalid." );
      break;
    }
    case NO_METHOD_IMPLEMENTATION: {
      SLIC_WARNING_ROOT( "The specified ContactMethod has no implementation." );
      break;
    }
    case DIFFERENT_FACE_TYPES: {
      SLIC_WARNING_ROOT( "The specified ContactMethod does not support different face types." );
      break;
    }
    case SAME_MESH_IDS: {
      SLIC_WARNING_ROOT( "The specified ContactMethod cannot be used in coupling schemes with identical mesh IDs." );
      break;
    }
    case SAME_MESH_IDS_INVALID_DIM: {
      SLIC_WARNING_ROOT( "The specified ContactMethod is not implemented for the problem dimension and "
                         << "cannot be used in coupling schemes with identical mesh IDs." );
      break;
    }
    case INVALID_DIM: {
      SLIC_WARNING_ROOT( "The specified ContactMethod is not implemented for the problem dimension." );
      break;
    }
    case NULL_NODAL_RESPONSE: {
      SLIC_WARNING_ROOT( "User must call tribol::registerNodalResponse() for each mesh to use this ContactMethod." );
      break;
    }
    case NO_METHOD_ERROR: {
      break;
    }
    default:
      break;
  }  // end switch over method errors
}  // end CouplingSchemeErrors::printMethodErrors()

//------------------------------------------------------------------------------
void CouplingSchemeErrors::printModelErrors()
{
  switch ( this->cs_model_error ) {
    case INVALID_MODEL: {
      SLIC_WARNING_ROOT( "The specified ContactModel is invalid." );
      break;
    }
    case NO_MODEL_IMPLEMENTATION: {
      SLIC_WARNING_ROOT( "The specified ContactModel has no implementation." );
      break;
    }
    case NO_MODEL_IMPLEMENTATION_FOR_REGISTERED_METHOD: {
      SLIC_WARNING_ROOT( "The specified ContactModel has no implementation for the registered ContactMethod." );
      break;
    }
    case NO_MODEL_ERROR: {
      break;
    }
    default:
      break;
  }  // end switch over model errors
}  // end CouplingSchemeErrors::printModelErrors()

//------------------------------------------------------------------------------
void CouplingSchemeErrors::printEnforcementErrors()
{
  switch ( this->cs_enforcement_error ) {
    case INVALID_ENFORCEMENT: {
      SLIC_WARNING_ROOT( "The specified EnforcementMethod is invalid." );
      break;
    }
    case INVALID_ENFORCEMENT_FOR_REGISTERED_METHOD: {
      SLIC_WARNING_ROOT( "The specified EnforcementMethod is invalid for the registered METHOD." );
      break;
    }
    case INVALID_ENFORCEMENT_OPTION: {
      SLIC_WARNING_ROOT( "The specified enforcement option is invalid." );
      break;
    }
    case OPTIONS_NOT_SET: {
      SLIC_WARNING_ROOT( "User must call 'tribol::set<EnforcementMethod>Options(..)' to set options for "
                         << "registered EnforcementMethod." );
      break;
    }
    case NO_ENFORCEMENT_IMPLEMENTATION: {
      SLIC_WARNING_ROOT( "The specified enforcement option has no implementation." );
      break;
    }
    case NO_ENFORCEMENT_IMPLEMENTATION_FOR_REGISTERED_METHOD: {
      SLIC_WARNING_ROOT( "The specified enforcement option has no implementation for the registered ContactMethod." );
      break;
    }
    case NO_ENFORCEMENT_IMPLEMENTATION_FOR_REGISTERED_OPTION: {
      SLIC_WARNING_ROOT(
          "The specified enforcement option has no implementation for the specified EnforcementMethod." );
      break;
    }
    case NO_ENFORCEMENT_ERROR: {
      break;
    }
    default:
      break;
  }  // end switch over enforcement errors
}  // end CouplingSchemeErrors::printEnforcementErrors()

//------------------------------------------------------------------------------
void CouplingSchemeErrors::printEnforcementDataErrors()
{
  switch ( this->cs_enforcement_data_error ) {
    case ERROR_IN_REGISTERED_ENFORCEMENT_DATA: {
      SLIC_WARNING_ROOT( "Error in registered enforcement data; see warnings." );
      break;
    }
    case NO_ENFORCEMENT_DATA_ERROR: {
      break;
    }
    default:
      break;
  }  // end switch over enforcement data errors
}  // end CouplingSchemeErrors::printEnforcementDataErrors()

//------------------------------------------------------------------------------
// Struct implementation for CouplingSchemeInfo
//------------------------------------------------------------------------------
void CouplingSchemeInfo::printCaseInfo()
{
  switch ( this->cs_case_info ) {
    case SPECIFYING_NO_SLIDING_WITH_REGISTERED_MODE: {
      SLIC_DEBUG_ROOT( "Overriding with ContactCase=NO_SLIDING with registered ContactMode." );
      break;
    }
    case SPECIFYING_NO_SLIDING_WITH_REGISTERED_METHOD: {
      SLIC_DEBUG_ROOT( "Overriding with ContactCase=NO_SLIDING with registered ContactMethod." );
      break;
    }
    case SPECIFYING_NONE_WITH_REGISTERED_METHOD: {
      SLIC_DEBUG_ROOT( "Overriding with ContactCase=NO_CASE with registered ContactMethod." );
      break;
    }
    case SPECIFYING_NONE_WITH_TWO_REGISTERED_MESHES: {
      SLIC_DEBUG_ROOT(
          "ContactCase=AUTO not supported with two different meshes; overriding with ContactCase=NO_CASE." );
      break;
    }
    case NO_CASE_INFO: {
      break;
    }
    default:
      break;
  }  // end switch over case info
}  // end CouplingSchemeInfo::printCaseInfo()

//------------------------------------------------------------------------------
void CouplingSchemeInfo::printEnforcementInfo()
{
  switch ( this->cs_enforcement_info ) {
    case SPECIFYING_NULL_ENFORCEMENT_WITH_REGISTERED_METHOD: {
      SLIC_DEBUG_ROOT( "Overriding with EnforcementMethod=NULL_ENFORCEMENT with registered ContactMethod." );
      break;
    }
    case NO_ENFORCEMENT_INFO: {
      break;
    }
    default:
      break;
  }  // end switch over enforcement info
}  // end CouplingSchemeInfo::printEnforcementInfo()

//------------------------------------------------------------------------------
// CouplingScheme class implementation
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
CouplingScheme::CouplingScheme( integer couplingSchemeId, integer meshId1, integer meshId2, integer contact_mode,
                                integer contact_case, integer contact_method, integer contact_model,
                                integer enforcement_method, integer binning_method )
    : m_id( couplingSchemeId ),
      m_meshId1( meshId1 ),
      m_meshId2( meshId2 ),
      m_numTotalNodes( 0 ),
      m_fixedBinning( false ),
      m_isBinned( false ),
      m_isTied( false ),
      m_numActivePairs( 0 ),
      m_methodData( nullptr )
{
  // error sanity checks
  SLIC_ERROR_ROOT_IF( meshId1 == ANY_MESH, "meshId1 cannot be set to ANY_MESH" );
  SLIC_ERROR_ROOT_IF( !validMeshID( m_meshId1 ), "invalid meshId1=" << meshId1 );
  SLIC_ERROR_ROOT_IF( !validMeshID( m_meshId2 ), "invalid meshId2=" << meshId2 );

  SLIC_ERROR_ROOT_IF( !in_range( contact_mode, NUM_CONTACT_MODES ), "invalid contact_mode=" << contact_mode );
  SLIC_ERROR_ROOT_IF( !in_range( contact_method, NUM_CONTACT_METHODS ), "invalid contact_method=" << contact_method );
  SLIC_ERROR_ROOT_IF( !in_range( contact_model, NUM_CONTACT_MODELS ), "invalid contact_model=" << contact_model );
  SLIC_ERROR_ROOT_IF( !in_range( enforcement_method, NUM_ENFORCEMENT_METHODS ),
                      "invalid enforcement_method=" << enforcement_method );
  SLIC_ERROR_ROOT_IF( !in_range( binning_method, NUM_BINNING_METHODS ), "invalid binning_method=" << binning_method );

  m_contactMode = static_cast<ContactMode>( contact_mode );
  m_contactCase = static_cast<ContactCase>( contact_case );
  m_contactMethod = static_cast<ContactMethod>( contact_method );
  m_contactModel = static_cast<ContactModel>( contact_model ),
  m_enforcementMethod = static_cast<EnforcementMethod>( enforcement_method );
  m_binningMethod = static_cast<BinningMethod>( binning_method );

  m_couplingSchemeErrors.cs_mode_error = NO_MODE_ERROR;
  m_couplingSchemeErrors.cs_case_error = NO_CASE_ERROR;
  m_couplingSchemeErrors.cs_method_error = NO_METHOD_ERROR;
  m_couplingSchemeErrors.cs_model_error = NO_MODEL_ERROR;
  m_couplingSchemeErrors.cs_enforcement_error = NO_ENFORCEMENT_ERROR;

  m_couplingSchemeInfo.cs_case_info = NO_CASE_INFO;
  m_couplingSchemeInfo.cs_enforcement_info = NO_ENFORCEMENT_INFO;

  m_loggingLevel = TRIBOL_UNDEFINED;

  // STEP 0: create contact-pairs object associated with this coupling scheme
  m_interfacePairs = new InterfacePairs();

}  // end CouplingScheme::CouplingScheme()

//------------------------------------------------------------------------------
CouplingScheme::~CouplingScheme()
{
  delete m_interfacePairs;
  delete m_methodData;
}

//------------------------------------------------------------------------------
bool CouplingScheme::isValidCouplingScheme()
{
  bool valid{ true };
  MeshManager& meshManager = MeshManager::getInstance();
  if ( !meshManager.hasMesh( this->m_meshId1 ) || !meshManager.hasMesh( this->m_meshId2 ) ) {
    SLIC_WARNING_ROOT( "Please register meshes for coupling scheme, " << this->m_id << "." );
    return false;
  }

  MeshData& mesh1 = meshManager.GetMeshInstance( this->m_meshId1 );
  MeshData& mesh2 = meshManager.GetMeshInstance( this->m_meshId2 );

  // check for invalid mesh topology matches in a coupling scheme
  if ( mesh1.m_elementType != mesh2.m_elementType ) {
    SLIC_WARNING_ROOT( "Coupling scheme, " << this->m_id << ", does not support meshes with "
                                           << "different surface element types." );
    mesh1.m_isValid = false;
    mesh2.m_isValid = false;
  }

  // check for invalid meshes. A mesh could be deemed invalid when registered.
  if ( !mesh1.m_isValid || !mesh2.m_isValid ) {
    return false;
  }

  // set boolean for null meshes
  if ( mesh1.m_numCells <= 0 || mesh2.m_numCells <= 0 ) {
    this->m_nullMeshes = true;
    valid = true;  // a null-mesh coupling scheme should still be valid
  }

  // check valid contact mode. Not all modes have an implementation
  if ( !this->isValidMode() ) {
    this->m_couplingSchemeErrors.printModeErrors();
    valid = false;
  }

  if ( !this->isValidCase() ) {
    this->m_couplingSchemeErrors.printCaseErrors();
    valid = false;
  } else {
    // print reasons why case may have been modified
    this->m_couplingSchemeInfo.printCaseInfo();
  }

  if ( !this->isValidMethod() ) {
    this->m_couplingSchemeErrors.printMethodErrors();
    valid = false;
  }

  if ( !this->isValidModel() ) {
    this->m_couplingSchemeErrors.printModelErrors();
    valid = false;
  }

  if ( !this->isValidEnforcement() ) {
    this->m_couplingSchemeErrors.printEnforcementErrors();
    valid = false;
  } else if ( this->checkEnforcementData() != 0 ) {
    this->m_couplingSchemeErrors.printEnforcementDataErrors();
    valid = false;
  }

  return valid;
}  // end CouplingScheme::isValidCouplingScheme()

//------------------------------------------------------------------------------
bool CouplingScheme::isValidMode()
{
  // check if contactMode is not an existing option
  if ( !in_range( this->m_contactMode, NUM_CONTACT_MODES ) ) {
    this->m_couplingSchemeErrors.cs_mode_error = INVALID_MODE;
    return false;
  } else if ( this->m_contactMode != SURFACE_TO_SURFACE && this->m_contactMode != SURFACE_TO_SURFACE_CONFORMING ) {
    this->m_couplingSchemeErrors.cs_mode_error = NO_MODE_IMPLEMENTATION;
    return false;
  } else {
    this->m_couplingSchemeErrors.cs_mode_error = NO_MODE_ERROR;
  }
  return true;
}  // end CouplingScheme::isValidMode()

//------------------------------------------------------------------------------
bool CouplingScheme::isValidCase()
{
  // check if contactCase is not an existing option
  if ( !in_range( this->m_contactCase, NUM_CONTACT_CASES ) ) {
    this->m_couplingSchemeErrors.cs_case_error = INVALID_CASE;
    return false;
  }

  // modify incompatible case with SURFACE_TO_SURFACE_CONFORMING to
  // NO_SLIDING
  if ( this->m_contactMode == SURFACE_TO_SURFACE_CONFORMING && this->m_contactCase != NO_SLIDING ) {
    this->m_couplingSchemeInfo.cs_case_info = SPECIFYING_NO_SLIDING_WITH_REGISTERED_MODE;
    this->m_contactCase = NO_SLIDING;
  }

  // make sure NO_SLIDING case is specified with ALIGNED_MORTAR
  if ( this->m_contactMethod == ALIGNED_MORTAR && this->m_contactCase != NO_SLIDING ) {
    this->m_couplingSchemeInfo.cs_case_info = SPECIFYING_NO_SLIDING_WITH_REGISTERED_METHOD;
    this->m_contactCase = NO_SLIDING;
  }

  // catch invalid case with SINGLE_MORTAR and MORTAR_WEIGHTS and switch
  // case to NONE (no case required).
  if ( ( this->m_contactMethod == SINGLE_MORTAR || this->m_contactMethod == MORTAR_WEIGHTS ) &&
       ( this->m_contactCase != NO_CASE && this->m_contactCase != NO_SLIDING ) ) {
    this->m_couplingSchemeInfo.cs_case_info = SPECIFYING_NONE_WITH_REGISTERED_METHOD;
    this->m_contactCase = NO_CASE;
  }

  // catch incorrectly specified AUTO contact case
  if ( this->m_contactCase == AUTO && ( this->m_meshId1 != this->m_meshId2 ) ) {
    this->m_couplingSchemeInfo.cs_case_info = SPECIFYING_NONE_WITH_TWO_REGISTERED_MESHES;
    this->m_contactCase = NO_CASE;
  }

  // if we are here we have modified the case with no error.
  this->m_couplingSchemeErrors.cs_case_error = NO_CASE_ERROR;

  return true;
}  // end CouplingScheme::isValidCase()

//------------------------------------------------------------------------------
bool CouplingScheme::isValidMethod()
{
  ////////////////////////
  //        NOTE        //
  ////////////////////////
  // Any new method has to be added as a case in the switch statement, even
  // if there are no specific checks, otherwise Tribol will error out assuming
  // that there is no implementation for a method in the ContactMethod enum list

  // check if contactMethod is not an existing option
  if ( !in_range( this->m_contactMethod, NUM_CONTACT_METHODS ) ) {
    this->m_couplingSchemeErrors.cs_method_error = INVALID_METHOD;
    return false;
  }

  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& mesh1 = meshManager.GetMeshInstance( this->m_meshId1 );
  MeshData& mesh2 = meshManager.GetMeshInstance( this->m_meshId2 );
  integer dim = this->spatialDimension();

  // check all methods for basic validity issues for non-null meshes
  if ( !this->m_nullMeshes ) {
    if ( this->m_contactMethod == ALIGNED_MORTAR || this->m_contactMethod == MORTAR_WEIGHTS ||
         this->m_contactMethod == SINGLE_MORTAR ) {
      if ( mesh1.m_numNodesPerCell != mesh2.m_numNodesPerCell ) {
        this->m_couplingSchemeErrors.cs_method_error = DIFFERENT_FACE_TYPES;
        return false;
      }
      if ( this->m_meshId1 == this->m_meshId2 ) {
        this->m_couplingSchemeErrors.cs_method_error = SAME_MESH_IDS;
        if ( dim != 3 ) {
          this->m_couplingSchemeErrors.cs_method_error = SAME_MESH_IDS_INVALID_DIM;
        }
        return false;
      }

      if ( dim != 3 ) {
        this->m_couplingSchemeErrors.cs_method_error = INVALID_DIM;
        return false;
      }
    } else if ( this->m_contactMethod == COMMON_PLANE ) {
      // check for different face types. This is not yet supported
      if ( mesh1.m_numNodesPerCell != mesh2.m_numNodesPerCell ) {
        this->m_couplingSchemeErrors.cs_method_error = DIFFERENT_FACE_TYPES;
        return false;
      }
    }  // end switch on contact method
    else {
      // if we are here there may be a method with no implementation.
      // See note at top of routine.
      this->m_couplingSchemeErrors.cs_method_error = NO_METHOD_IMPLEMENTATION;
      return false;
    }

    if ( this->m_contactMethod == ALIGNED_MORTAR || this->m_contactMethod == SINGLE_MORTAR ||
         this->m_contactMethod == COMMON_PLANE ) {
      if ( mesh1.m_numCells > 0 && !mesh1.m_nodalFields.m_is_nodal_response_set ) {
        this->m_couplingSchemeErrors.cs_method_error = NULL_NODAL_RESPONSE;
        return false;
      }

      if ( mesh2.m_numCells > 0 && !mesh2.m_nodalFields.m_is_nodal_response_set ) {
        this->m_couplingSchemeErrors.cs_method_error = NULL_NODAL_RESPONSE;
        return false;
      }
    }
  }  // end if-check on non-null meshes

  // TODO check for nodal displacements for methods that require this data

  // no method error if here
  this->m_couplingSchemeErrors.cs_method_error = NO_METHOD_ERROR;
  return true;

}  // end CouplingScheme::isValidMethod()

//------------------------------------------------------------------------------
bool CouplingScheme::isValidModel()
{
  // Note: add a method check for compatible models when implementing a new
  // method in Tribol

  // check if the m_contactModel is not an existing option
  if ( !in_range( this->m_contactModel, NUM_CONTACT_MODELS ) ) {
    this->m_couplingSchemeErrors.cs_model_error = INVALID_MODEL;
    return false;
  }

  // check for model and method compatibility issues
  switch ( this->m_contactMethod ) {
    case SINGLE_MORTAR:
    case ALIGNED_MORTAR:
    case MORTAR_WEIGHTS: {
      if ( this->m_contactModel != FRICTIONLESS && this->m_contactModel != NULL_MODEL ) {
        this->m_couplingSchemeErrors.cs_model_error = NO_MODEL_IMPLEMENTATION_FOR_REGISTERED_METHOD;
        return false;
      }
      break;
    }

    case COMMON_PLANE: {
      if ( this->m_contactModel != FRICTIONLESS && this->m_contactModel != NULL_MODEL &&
           this->m_contactModel != TIED ) {
        this->m_couplingSchemeErrors.cs_model_error = NO_MODEL_IMPLEMENTATION_FOR_REGISTERED_METHOD;
        return false;
      }
      break;
    }

    default: {
      // Don't need to add default error/info. Compatibility is driven by existing
      // method implementations, which are checked in isValidMethod()
      break;
    }
  }  // end switch

  this->m_couplingSchemeErrors.cs_model_error = NO_MODEL_ERROR;
  return true;
}  // end isValidModel()

//------------------------------------------------------------------------------
bool CouplingScheme::isValidEnforcement()
{
  // NOTE: Add a method check here for compatible enforcement when adding a
  // new method to Tribol

  // check if the enforcementMethod is not an existing option
  if ( !in_range( this->m_enforcementMethod, NUM_ENFORCEMENT_METHODS ) ) {
    this->m_couplingSchemeErrors.cs_enforcement_error = INVALID_ENFORCEMENT;
    return false;
  }

  // check for invalid method/enforcement compatibility
  switch ( this->m_contactMethod ) {
    case MORTAR_WEIGHTS: {
      // force NULL_ENFORCEMENT for MORTAR_WEIGHTS. Only possible choice
      if ( this->m_enforcementMethod != NULL_ENFORCEMENT ) {
        this->m_couplingSchemeInfo.cs_enforcement_info = SPECIFYING_NULL_ENFORCEMENT_WITH_REGISTERED_METHOD;
        this->m_enforcementMethod = NULL_ENFORCEMENT;
        // don't return
      }
      if ( this->m_enforcementOptions.lm_implicit_options.eval_mode != ImplicitEvalMode::MORTAR_WEIGHTS_EVAL ) {
        // Note, not adding a cs_enforcement_info note here since MORTAR_WEIGHTS only
        // works with this eval mode. This is simply protecting a user from specifying
        // something that doesn't make sense for this specialized 'method'. This does
        // not affect requirements on registered data or output for the user.
        this->m_enforcementOptions.lm_implicit_options.eval_mode = ImplicitEvalMode::MORTAR_WEIGHTS_EVAL;
        // don't return
      }
      if ( this->m_enforcementOptions.lm_implicit_options.sparse_mode != SparseMode::MFEM_LINKED_LIST ) {
        this->m_couplingSchemeErrors.cs_enforcement_error = NO_ENFORCEMENT_IMPLEMENTATION_FOR_REGISTERED_OPTION;
        return false;
      }
      break;
    }  // end case MORTAR_WEIGHTS

    case ALIGNED_MORTAR:
    case SINGLE_MORTAR: {
      if ( this->m_enforcementMethod == PENALTY ) {
        this->m_couplingSchemeErrors.cs_enforcement_error = NO_ENFORCEMENT_IMPLEMENTATION_FOR_REGISTERED_METHOD;
        return false;
      } else if ( this->m_enforcementMethod != LAGRANGE_MULTIPLIER ) {
        // Don't change to valid enforcement method. Data required
        // for valid method likely not registered
        this->m_couplingSchemeErrors.cs_enforcement_error = INVALID_ENFORCEMENT_FOR_REGISTERED_METHOD;
        return false;
      } else if ( this->m_enforcementMethod == LAGRANGE_MULTIPLIER ) {
        if ( !this->m_enforcementOptions.lm_implicit_options.is_enforcement_option_set() ) {
          this->m_couplingSchemeErrors.cs_enforcement_error = OPTIONS_NOT_SET;
          return false;
        } else if ( this->m_enforcementOptions.lm_implicit_options.sparse_mode != SparseMode::MFEM_LINKED_LIST &&
                    this->m_enforcementOptions.lm_implicit_options.sparse_mode != SparseMode::MFEM_ELEMENT_DENSE ) {
          this->m_couplingSchemeErrors.cs_enforcement_error = NO_ENFORCEMENT_IMPLEMENTATION_FOR_REGISTERED_OPTION;
          return false;
        } else if ( this->m_enforcementOptions.lm_implicit_options.eval_mode ==
                    ImplicitEvalMode::MORTAR_WEIGHTS_EVAL ) {
          this->m_couplingSchemeErrors.cs_enforcement_error = NO_ENFORCEMENT_IMPLEMENTATION_FOR_REGISTERED_OPTION;
          return false;
        }
      }
      break;
    }  // end case SINGLE_MORTAR

    case COMMON_PLANE: {
      // check if PENALTY is not chosen. This is the only possible (and foreseeable)
      // choice for COMMON_PLANE
      if ( this->m_enforcementMethod != PENALTY ) {
        this->m_couplingSchemeErrors.cs_enforcement_error = INVALID_ENFORCEMENT_FOR_REGISTERED_METHOD;
        return false;
      } else if ( !this->m_enforcementOptions.penalty_options.is_constraint_type_set() ) {
        this->m_couplingSchemeErrors.cs_enforcement_error = OPTIONS_NOT_SET;
        return false;
      }
      break;
    }

    default: {
      // no default check. These are method driven and method checks are performed
      // in isValidMethod().
      break;
    }
  }  // end switch

  this->m_couplingSchemeErrors.cs_enforcement_error = NO_ENFORCEMENT_ERROR;
  return true;
}  // end CouplingScheme::isValidEnforcement()

//------------------------------------------------------------------------------
int CouplingScheme::checkEnforcementData()
{
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& mesh1 = meshManager.GetMeshInstance( this->m_meshId1 );
  MeshData& mesh2 = meshManager.GetMeshInstance( this->m_meshId2 );
  this->m_couplingSchemeErrors.cs_enforcement_data_error = NO_ENFORCEMENT_DATA_ERROR;

  int err = 0;
  switch ( this->m_contactMethod ) {
    case MORTAR_WEIGHTS:
      // no-op for now
      break;
    case ALIGNED_MORTAR:
      // don't break
    case SINGLE_MORTAR: {
      switch ( this->m_enforcementMethod ) {
        case LAGRANGE_MULTIPLIER: {
          // check LM data. Note, this routine is guarded against null-meshes
          if ( mesh2.checkLagrangeMultiplierData() != 0 )  // nonmortar side only
          {
            this->m_couplingSchemeErrors.cs_enforcement_data_error = ERROR_IN_REGISTERED_ENFORCEMENT_DATA;
            err = 1;
          }
          break;
        }  // end case LAGRANGE_MULTIPLIER
        default:
          // no-op
          break;
      }  // end switch over enforcement method
      break;
    }  // end case SINGLE_MORTAR
    case COMMON_PLANE: {
      switch ( this->m_enforcementMethod ) {
        case PENALTY: {
          // check penalty data. Note, this routine is guarded against null-meshes
          PenaltyEnforcementOptions& pen_enfrc_options = this->m_enforcementOptions.penalty_options;
          if ( mesh1.checkPenaltyData( pen_enfrc_options ) != 0 || mesh2.checkPenaltyData( pen_enfrc_options ) != 0 ) {
            this->m_couplingSchemeErrors.cs_enforcement_data_error = ERROR_IN_REGISTERED_ENFORCEMENT_DATA;
            err = 1;
          }
          break;
        }  // end case PENALTY
        default:
          // no-op
          break;
      }  // end switch over enforcement method
    }    // end case COMMON_PLANE
    default:
      // no-op
      break;
  }  // end switch on method

  return err;

}  // end CouplingScheme::checkEnforcementData()
//------------------------------------------------------------------------------
void CouplingScheme::performBinning()
{
  // Find the interacting pairs for this coupling scheme. Will not use
  // binning if setInterfacePairs has been called.
  if ( !this->m_nullMeshes ) {
    if ( !this->hasFixedBinning() ) {
      m_interfacePairs->clear();

      InterfacePairFinder finder( this );
      finder.initialize();
      finder.findInterfacePairs();

      // For Cartesian binning, we only need to compute the binning once
      if ( this->getBinningMethod() == BINNING_CARTESIAN_PRODUCT ) {
        this->setFixedBinning( true );
      }

      // set fixed binning depending on contact case,
      // e.g. NO_SLIDING
      this->setFixedBinningPerCase();
    }
  }  // end if-non-null meshes
  return;
}
//------------------------------------------------------------------------------
int CouplingScheme::apply( integer cycle, real t, real& dt )
{
  // set dimension on the contact plane manager
  parameters_t& params = parameters_t::getInstance();
  ContactPlaneManager& cpMgr = ContactPlaneManager::getInstance();

  // clear contact plane manager to be populated/allocated anew for this
  // coupling-scheme/cycle.
  cpMgr.clearCPManager();
  cpMgr.setSpaceDim( params.dimension );

  // loop over number of interface pairs
  IndexType numPairs = m_interfacePairs->getNumPairs();

  SLIC_DEBUG( "Coupling scheme " << m_id << " has " << numPairs << " pairs." );

  // loop over all pairs and perform geometry checks to see if they
  // are interacting
  int numActivePairs = 0;
  int pair_err = 0;
  for ( IndexType kp = 0; kp < numPairs; ++kp ) {
    InterfacePair pair = m_interfacePairs->getInterfacePair( kp );

    // call wrapper around the contact method/case specific
    // geometry checks to determine whether to include a pair
    // in the active set
    bool interact = false;
    FaceGeomError interact_err = CheckInterfacePair( pair, m_contactMethod, m_contactCase, interact );

    // TODO refine how these errors are handled. Here we skip over face-pairs with errors. That is,
    // they are not registered for contact, but we don't error out.
    if ( interact_err != NO_FACE_GEOM_ERROR ) {
      pair_err = 1;
      pair.isContactCandidate = false;
      // TODO consider printing offending face(s) coordinates for debugging
      SLIC_DEBUG( "Face geometry error, " << static_cast<int>( interact_err ) << "for pair, " << kp << "." );
      continue;
    } else if ( !interact ) {
      pair.isContactCandidate = false;
    } else {
      pair.isContactCandidate = true;
      ++numActivePairs;
    }

    // update the InterfacePairs container on the coupling scheme
    // to reflect any change to contact candidacy
    m_interfacePairs->updateInterfacePair( pair, kp );

  }  // end loop over pairs

  // TODO refine how this logging is handled. This just detects an issue with a face-pair geometry
  // (which has been skipped over for contact eligibility) and reports this warning. Do we want to
  // error out, or let a user detect bad contact behavior, but with a contact interaction that still
  // runs?
  SLIC_WARNING_IF( pair_err != 0, "CouplingScheme::apply(): error with orientation, input, "
                                      << "or invalid overlaps in CheckInterfacePair()." );

  this->m_numActivePairs = numActivePairs;

  SLIC_ERROR_IF( numActivePairs != cpMgr.size(),
                 "CouplingScheme::apply(): "
                     << "number of active pairs does not match number of contact planes." );

  // aggregate across ranks for this coupling scheme? SRW
  SLIC_DEBUG( "Number of active interface pairs: " << numActivePairs );

  // wrapper around contact method, case, and
  // enforcement to apply the interface physics in both
  // normal and tangential directions. This function loops
  // over the pairs on the coupling scheme and applies the
  // appropriate physics in the normal and tangential directions.
  int err = ApplyInterfacePhysics( this, cycle, t );

  SLIC_WARNING_IF( err != 0, "CouplingScheme::apply(): error in ApplyInterfacePhysics for "
                                 << "coupling scheme, " << this->m_id << "." );

  // compute Tribol timestep vote on the coupling scheme
  if ( err == 0 && numActivePairs > 0 ) {
    computeTimeStep( dt );
  }

  // write output
  writeInterfaceOutput( params.output_directory, params.vis_type, cycle, t );

  if ( err != 0 || pair_err != 0 ) {
    return 1;
  } else {
    return 0;
  }

}  // end CouplingScheme::apply()

//------------------------------------------------------------------------------
bool CouplingScheme::init()
{
  // check for valid coupling scheme only for non-null-meshes
  bool valid = false;
  valid = this->isValidCouplingScheme();
  this->m_isValid = valid;
  if ( this->m_isValid ) {
    // set individual coupling scheme logging level
    this->setSlicLoggingLevel();
    this->allocateMethodData();

    // compute the face data
    MeshManager& meshManager = MeshManager::getInstance();
    MeshData& mesh1 = meshManager.GetMeshInstance( this->m_meshId1 );
    mesh1.computeFaceData( mesh1.m_dim );
    if ( this->m_meshId2 != this->m_meshId1 ) {
      MeshData& mesh2 = meshManager.GetMeshInstance( this->m_meshId2 );
      mesh2.computeFaceData( mesh2.m_dim );
    }

    return true;
  } else {
    return false;
  }
}
//------------------------------------------------------------------------------
void CouplingScheme::setSlicLoggingLevel()
{
  // set slic logging level for coupling schemes that have API modified logging levels
  if ( this->m_loggingLevel != TRIBOL_UNDEFINED ) {
    switch ( this->m_loggingLevel ) {
      case TRIBOL_DEBUG: {
        axom::slic::setLoggingMsgLevel( axom::slic::message::Debug );
        break;
      }
      case TRIBOL_INFO: {
        axom::slic::setLoggingMsgLevel( axom::slic::message::Info );
        break;
      }
      case TRIBOL_WARNING: {
        axom::slic::setLoggingMsgLevel( axom::slic::message::Warning );
        break;
      }
      case TRIBOL_ERROR: {
        axom::slic::setLoggingMsgLevel( axom::slic::message::Error );
        break;
      }
      default: {
        axom::slic::setLoggingMsgLevel( axom::slic::message::Warning );
        break;
      }
    }  // end switch
  }    // end if
}
//------------------------------------------------------------------------------
void CouplingScheme::allocateMethodData()
{
  // check for valid coupling schemes for those with non-null meshes.
  // Note: keep if-block for non-null meshes here. A valid coupling scheme
  // may have null meshes, but we don't want to allocate unnecessary memory here.
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& mesh1 = meshManager.GetMeshInstance( this->m_meshId1 );
  MeshData& mesh2 = meshManager.GetMeshInstance( this->m_meshId2 );
  if ( mesh1.m_numCells > 0 && mesh2.m_numCells > 0 ) {
    this->m_numTotalNodes = mesh1.m_lengthNodalData;

    // dynamically allocate method data object for mortar method
    switch ( this->m_contactMethod ) {
      case ALIGNED_MORTAR:
      case MORTAR_WEIGHTS:
      case SINGLE_MORTAR: {
        // dynamically allocate method data object
        this->m_methodData = new MortarData;
        static_cast<MortarData*>( m_methodData )->m_numTotalNodes = this->m_numTotalNodes;
        break;
      }  // end case SINGLE_MORTAR
      default: {
        this->m_methodData = nullptr;
        break;
      }
    }  // end if on non-null meshes

  }  // end if on non-null-meshes
}  // end CouplingScheme::allocateMethodData()

//------------------------------------------------------------------------------
real CouplingScheme::getGapTol( int fid1, int fid2 ) const
{
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& mesh1 = meshManager.GetMeshInstance( m_meshId1 );
  MeshData& mesh2 = meshManager.GetMeshInstance( m_meshId2 );
  parameters_t& params = parameters_t::getInstance();
  real gap_tol = 0.;

  // add debug warning if this routine is called for interface methods
  // that do not require gap tolerances
  switch ( m_contactMethod ) {
    case SINGLE_MORTAR:
      SLIC_WARNING( "CouplingScheme::getGapTol(): 'SINGLE_MORTAR' "
                    << "method does not require use of a gap tolerance." );
      break;

    case ALIGNED_MORTAR:
      SLIC_WARNING( "CouplingScheme::getGapTol(): 'ALIGNED_MORTAR' "
                    << "method does not require use of a gap tolerance." );
      break;

    case MORTAR_WEIGHTS:
      SLIC_WARNING( "CouplingScheme::getGapTol(): 'MORTAR_WEIGHTS' "
                    << "method does not require use of a gap tolerance." );
      break;

    case COMMON_PLANE:

      switch ( m_contactModel ) {
        case TIED:
          gap_tol = params.gap_tied_tol * axom::utilities::max( mesh1.m_faceRadius[fid1], mesh2.m_faceRadius[fid2] );
          break;

        default:
          gap_tol =
              -1. * params.gap_tol_ratio * axom::utilities::max( mesh1.m_faceRadius[fid1], mesh2.m_faceRadius[fid2] );
          break;

      }  // end switch over m_contactModel
      break;

    default:
      break;
  }  // end switch over m_contactMethod

  return gap_tol;
}

//------------------------------------------------------------------------------
void CouplingScheme::computeTimeStep( real& dt )
{
  // make sure velocities are registered
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& mesh1 = meshManager.GetMeshInstance( m_meshId1 );
  MeshData& mesh2 = meshManager.GetMeshInstance( m_meshId2 );

  if ( dt < 1.e-8 ) {
    // current timestep too small for Tribol vote. Leave unchanged and return
    return;
  }

  // check for null velocities needed to compute timestep. Allow for null-meshes
  // (i.e. zero-element on rank meshes)
  bool meshVel1 = true;
  bool meshVel2 = true;
  if ( this->spatialDimension() == 2 ) {
    if ( mesh1.m_velX == nullptr || mesh1.m_velY == nullptr ) {
      meshVel1 = false;
    }

    if ( mesh2.m_velX == nullptr || mesh2.m_velY == nullptr ) {
      meshVel2 = false;
    }
  } else {
    if ( mesh1.m_velX == nullptr || mesh1.m_velY == nullptr || mesh1.m_velZ == nullptr ) {
      meshVel1 = false;
    }

    if ( mesh2.m_velX == nullptr || mesh2.m_velY == nullptr || mesh2.m_velZ == nullptr ) {
      meshVel2 = false;
    }
  }  // end if-check on dim for velocity registration

  if ( !meshVel1 || !meshVel2 ) {
    if ( mesh1.m_numCells > 0 && mesh2.m_numCells > 0 ) {
      // invalid registration of nodal velocities for non-null meshes
      dt = -1.0;
      return;
    } else {
      // at least one null mesh with allowable null velocities; don't modify dt
      return;
    }
  }

  // if we are here we have registered velocities for non-null meshes
  // and can compute the timestep vote
  switch ( m_contactMethod ) {
    case SINGLE_MORTAR:
      // no-op
      break;
    case ALIGNED_MORTAR:
      // no-op
      break;
    case MORTAR_WEIGHTS:
      // no-op
      break;
    case COMMON_PLANE:
      if ( m_enforcementMethod == PENALTY ) {
        this->computeCommonPlaneTimeStep( dt );
      }
      break;
    default:
      break;
  }  // end-switch
}
//------------------------------------------------------------------------------
void CouplingScheme::computeCommonPlaneTimeStep( real& dt )
{
  // note: the timestep vote is based on a velocity projection
  // and does not account for the spring stiffness in a CFL-like
  // timestep constraint. A constant penalty everywhere is not necessarily
  // tuned to the underlying material that occurs with 'element_wise'
  // and may result in contact instabilities that this timestep vote
  // does not yet address. Tuning the penalty to the underlying material
  // stiffness implicitly scales the penalty stiffness to approximately
  // correspond to a host-code timestep governed by an underlying
  // element-wise CFL constraint. The timestep vote in this routine
  // catches the case where too large of a timestep results in too
  // much face-pair interpenetration, which may also lead to contact
  // instabilities.

  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& mesh1 = meshManager.GetMeshInstance( m_meshId1 );
  MeshData& mesh2 = meshManager.GetMeshInstance( m_meshId2 );

  // issue warning that this timestep vote does not address
  // contact instabilities that may present themselves with the use
  // of a constant penalty everywhere; then, return. If constant penalty
  // is used then likely element thicknesses have not been registered.
  PenaltyEnforcementOptions& pen_enfrc_options = this->m_enforcementOptions.penalty_options;
  KinematicPenaltyCalculation kin_calc = pen_enfrc_options.kinematic_calculation;
  if ( kin_calc == KINEMATIC_CONSTANT ) {
    // Tribol timestep vote only used with KINEMATIC_ELEMENT penalty
    // because element thicknesses are supplied
    return;
  }

  parameters_t& parameters = parameters_t::getInstance();
  real proj_ratio = parameters.contact_pen_frac;
  ContactPlaneManager& cpMgr = ContactPlaneManager::getInstance();
  // int num_sides = 2; // always 2 sides in a single coupling scheme
  int dim = this->spatialDimension();
  int numNodesPerCell1 = mesh1.m_numNodesPerCell;
  int numNodesPerCell2 = mesh2.m_numNodesPerCell;

  // loop over each interface pair. Even if pair is not in contact,
  // we still do a velocity projection for that proximate face-pair
  // to see if interpenetration next cycle 'may' be too much
  IndexType numPairs = m_interfacePairs->getNumPairs();
  real dt_temp1 = dt;
  real dt_temp2 = dt;
  int cpID = 0;
  bool tiny_vel_msg = false;
  for ( IndexType kp = 0; kp < numPairs; ++kp ) {
    InterfacePair pair = m_interfacePairs->getInterfacePair( kp );

    // guard against the case where a pair does not have a contact plane; that is,
    // the pair is not a contact candidate
    if ( !pair.isContactCandidate ) {
      continue;
    }

    real x1[dim * numNodesPerCell1];
    real v1[dim * numNodesPerCell1];
    mesh1.getFaceCoords( pair.pairIndex1, &x1[0] );
    mesh1.getFaceNodalVelocities( pair.pairIndex1, &v1[0] );

    real x2[dim * numNodesPerCell2];
    real v2[dim * numNodesPerCell2];
    mesh2.getFaceCoords( pair.pairIndex2, &x2[0] );
    mesh2.getFaceNodalVelocities( pair.pairIndex2, &v2[0] );

    /////////////////////////////////////////////////////////////
    // calculate face velocities at projected overlap centroid //
    /////////////////////////////////////////////////////////////
    real vel_f1[dim];
    real vel_f2[dim];
    initRealArray( &vel_f1[0], dim, 0. );
    initRealArray( &vel_f2[0], dim, 0. );

    // interpolate nodal velocity at overlap centroid as projected
    // onto face 1
    double cXf1 = cpMgr.m_cXf1[cpID];
    double cYf1 = cpMgr.m_cYf1[cpID];
    double cZf1 = ( dim == 3 ) ? cpMgr.m_cZf1[cpID] : 0.;
    GalerkinEval( &x1[0], cXf1, cYf1, cZf1, LINEAR, PHYSICAL, dim, dim, &v1[0], &vel_f1[0] );

    // interpolate nodal velocity at overlap centroid as projected
    // onto face 2
    double cXf2 = cpMgr.m_cXf2[cpID];
    double cYf2 = cpMgr.m_cYf2[cpID];
    double cZf2 = ( dim == 3 ) ? cpMgr.m_cZf2[cpID] : 0.;
    GalerkinEval( &x2[0], cXf2, cYf2, cZf2, LINEAR, PHYSICAL, dim, dim, &v2[0], &vel_f2[0] );

    ////////////////////////////////////////////////
    //                                            //
    // Compute Timestep Vote Based on a Few Cases //
    //                                            //
    ////////////////////////////////////////////////

    ///////////////////////////////////////////////
    // compute data common to all timestep votes //
    ///////////////////////////////////////////////

    // compute velocity projections:
    // compute the dot product between the face velocities
    // at the overlap-centroid-to-face projected centroid and each
    // face's outward unit normal AND the overlap normal. The
    // former is used to compute projections and the latter is
    // used to indicate further contact using a velocity projection
    real v1_dot_n, v2_dot_n, v1_dot_n1, v2_dot_n2;
    real overlapNormal[dim];
    cpMgr.getContactPlaneNormal( cpID, dim, &overlapNormal[0] );

    // get face normals
    real fn1[dim], fn2[dim];
    mesh1.getFaceNormal( pair.pairIndex1, dim, &fn1[0] );
    mesh2.getFaceNormal( pair.pairIndex2, dim, &fn2[0] );

    // compute projections
    v1_dot_n = dotProd( &vel_f1[0], &overlapNormal[0], dim );
    v2_dot_n = dotProd( &vel_f2[0], &overlapNormal[0], dim );
    v1_dot_n1 = dotProd( &vel_f1[0], &fn1[0], dim );
    v2_dot_n2 = dotProd( &vel_f2[0], &fn2[0], dim );

    // add tiny amount to velocity-normal projections to avoid
    // division by zero. Note that if these projections are close to
    // zero, there may stationary interactions or tangential motion.
    // In this case, any timestep estimate will be very large, and
    // not control the simulation
    real tiny = 1.e-10;
    real tiny1 = ( v1_dot_n >= 0. ) ? tiny : -1. * tiny;
    real tiny2 = ( v2_dot_n >= 0. ) ? tiny : -1. * tiny;
    v1_dot_n += tiny1;
    v2_dot_n += tiny2;
    v1_dot_n1 += tiny1;
    v2_dot_n2 += tiny2;

    // get volume element thicknesses associated with each
    // face in this pair and find minimum
    real t1 = mesh1.m_elemData.m_thickness[pair.pairIndex1];
    real t2 = mesh2.m_elemData.m_thickness[pair.pairIndex2];

    // compute the gap vector (recall gap is x1-x2 by convention)
    real gapVec[dim];
    gapVec[0] = cpMgr.m_cXf1[cpID] - cpMgr.m_cXf2[cpID];
    gapVec[1] = cpMgr.m_cYf1[cpID] - cpMgr.m_cYf2[cpID];
    if ( dim == 3 ) {
      gapVec[2] = cpMgr.m_cZf1[cpID] - cpMgr.m_cZf2[cpID];
    }

    // compute the dot product between gap vector and the outward
    // unit face normals. Note: the amount of interpenetration is
    // going to be compared to a length/thickness parameter that
    // is computed in the direction of the outward unit normal,
    // NOT the normal of the contact plane. This is despite the
    // fact that the contact nodal forces are resisting contact
    // in the direction of the overlap normal.
    real gap_f1_n1 = dotProd( &gapVec[0], &fn1[0], dim );
    real gap_f2_n2 = dotProd( &gapVec[0], &fn2[0], dim );

    real dt1 = 1.e6;    // initialize as large number
    real dt2 = 1.e6;    // initialize as large number
    real alpha = 0.75;  // multiplier on timestep estimate
    bool dt1_check1 = false;
    bool dt2_check1 = false;
    bool dt1_vel_check = false;
    bool dt2_vel_check = false;

    real max_delta1 = proj_ratio * t1;
    real max_delta2 = proj_ratio * t2;

    // Trigger for check 1 and 2:
    // check if there is further interpen or separation based on the
    // velocity projection in the direction of the common-plane normal.
    // The two cases are:
    // if v1*n < 0 there is interpen
    // if v2*n > 0 there is interpen
    dt1_vel_check = ( v1_dot_n < 0. ) ? true : false;
    dt2_vel_check = ( v2_dot_n > 0. ) ? true : false;

    ////////////////////////////////////////////////////////////////////
    // 1. Current interpenetration gap exceeds max allowable interpen //
    ////////////////////////////////////////////////////////////////////

    // check if pair is in contact per Common Plane method. Note: this check
    // to see if the face-pair is in contact uses the gap computed on the
    // contact plane, which is in the direction of the overlap normal
    if ( cpMgr.m_inContact[cpID] )  // gap < gap_tol
    {
      // check for nearly zero velocity and a gap that's too large.
      real tiny_vel_proj = 1.e-8;
      real tiny_vel_tol = 1.e-6;  // make larger than tiny_vel_proj
      real tiny_vel_diff1 = std::abs( v1_dot_n - tiny_vel_proj );
      real tiny_vel_diff2 = std::abs( v2_dot_n - tiny_vel_proj );
      if ( tiny_vel_diff1 < tiny_vel_tol || tiny_vel_diff2 < tiny_vel_tol ) {
        tiny_vel_msg = true;
      }

      // compute the difference between the 'face-gaps' and the max allowable
      // interpen as a function of element thickness.
      real delta1 = max_delta1 - gap_f1_n1;  // >0 not exceeding max allowable
      real delta2 = max_delta2 + gap_f2_n2;  // >0 not exceeding max allowable

      dt1_check1 = ( dt1_vel_check ) ? ( delta1 < 0. ) : false;
      dt2_check1 = ( dt2_vel_check ) ? ( delta2 < 0. ) : false;

      // compute dt for face 1 and 2 based on the velocity projection in the
      // direction of that face's outward unit normal
      // Note, this calculation takes a fraction
      // of the computed dt to reduce the amount of face-displacement in a given
      // cycle.
      dt1 = ( dt1_check1 ) ? -alpha * delta1 / v1_dot_n1 : dt1;
      dt2 = ( dt2_check1 ) ? -alpha * delta2 / v2_dot_n2 : dt2;

      SLIC_ERROR_IF( dt1 < 0., "Common plane timestep vote for gap-check of face 1 is negative." );
      SLIC_ERROR_IF( dt2 < 0., "Common plane timestep vote for gap-check of face 2 is negative." );

      dt_temp1 = axom::utilities::min( dt_temp1, axom::utilities::min( dt1, dt2 ) );

    }  // end case 1

    ///////////////////////////////////////////////////////////
    // 2. Velocity projection exceeds interpen tolerance     //
    //    Note: This is performed for all contact candidates //
    //          even if they are not 'in contact' per the    //
    //          common-plane method                          //
    ///////////////////////////////////////////////////////////

    // compute delta between velocity projection of face-projected
    // overlap centroid and the OTHER face's face-projected overlap
    // centroid
    real proj_delta_x1 = cpMgr.m_cXf1[cpID] + dt * vel_f1[0] - cpMgr.m_cXf2[cpID];
    real proj_delta_y1 = cpMgr.m_cYf1[cpID] + dt * vel_f1[1] - cpMgr.m_cYf2[cpID];
    real proj_delta_z1 = 0.;

    real proj_delta_x2 = cpMgr.m_cXf2[cpID] + dt * vel_f2[0] - cpMgr.m_cXf1[cpID];
    real proj_delta_y2 = cpMgr.m_cYf2[cpID] + dt * vel_f2[1] - cpMgr.m_cYf1[cpID];
    real proj_delta_z2 = 0.;

    // compute the dot product between each face's delta and the OTHER
    // face's outward unit normal. This is the magnitude of interpenetration
    // of one face's projected overlap-centroid in the 'thickness-direction'
    // of the other face (with whom in may be in contact currently, or in
    // a velocity projected sense).
    real proj_delta_n_1 = proj_delta_x1 * fn2[0] + proj_delta_y1 * fn2[1];
    real proj_delta_n_2 = proj_delta_x2 * fn1[0] + proj_delta_y2 * fn1[1];

    if ( dim == 3 ) {
      proj_delta_z1 = cpMgr.m_cZf1[cpID] + dt * vel_f1[2] - cpMgr.m_cZf2[cpID];
      proj_delta_z2 = cpMgr.m_cZf2[cpID] + dt * vel_f2[2] - cpMgr.m_cZf1[cpID];

      proj_delta_n_1 += proj_delta_z1 * fn2[2];
      proj_delta_n_2 += proj_delta_z2 * fn1[2];
    }

    // If delta_n_i < 0, (i=1,2) there is interpen. Check this interpen
    // against the maximum allowable to determine if a velocity projection
    // timestep estimate is still required.
    if ( dt1_vel_check ) {
      dt1_vel_check = ( proj_delta_n_1 < 0. ) ? ( ( std::abs( proj_delta_n_1 ) > max_delta1 ) ? true : false ) : false;
    }

    if ( dt2_vel_check ) {
      dt2_vel_check = ( proj_delta_n_2 < 0. ) ? ( ( std::abs( proj_delta_n_2 ) > max_delta2 ) ? true : false ) : false;
    }

    // if the 'case 1' check was not triggered for face 1 or 2, then
    // check the sign of the delta-projections to determine if interpen
    // is occuring. If so, check against maximum allowable interpen.
    // In both cases if delta_n_i (i=1,2) < 0 there is interpen
    dt1 = ( dt1_vel_check ) ? -alpha * ( proj_delta_n_1 + max_delta1 ) / v1_dot_n1 : dt1;
    dt2 = ( dt2_vel_check ) ? -alpha * ( proj_delta_n_2 + max_delta2 ) / v2_dot_n2 : dt2;

    SLIC_ERROR_IF( dt1 < 0, "Common plane timestep vote for velocity projection of face 1 is negative." );
    SLIC_ERROR_IF( dt2 < 0, "Common plane timestep vote for velocity projection of face 2 is negative." );

    // update dt_temp2
    dt_temp2 = axom::utilities::min( dt_temp2, axom::utilities::min( dt1, dt2 ) );

    // end check 2

    ++cpID;
  }  // end loop over interface pairs

  // Can we output this message on root? SRW
  if ( tiny_vel_msg ) {
    SLIC_INFO( "tribol::computeCommonPlaneTimeStep(): initial mesh overlap is too large "
               << "with very small velocity. Cannot provide timestep vote. "
               << "Reduce overlap in initial configuration, otherwise penalty "
               << "instability may result." );
  }

  dt = axom::utilities::min( dt_temp1, dt_temp2 );
}

//------------------------------------------------------------------------------
void CouplingScheme::writeInterfaceOutput( const std::string& dir, const VisType v_type, const integer cycle,
                                           const real t )
{
  parameters_t& parameters = parameters_t::getInstance();
  int dim = this->spatialDimension();
  if ( parameters.vis_cycle_incr > 0 && !( cycle % parameters.vis_cycle_incr ) ) {
    switch ( m_contactMethod ) {
      case SINGLE_MORTAR:
      case ALIGNED_MORTAR:
      case MORTAR_WEIGHTS:
      case COMMON_PLANE:
        WriteContactPlaneMeshToVtk( dir, v_type, m_id, m_meshId1, m_meshId2, dim, cycle, t );
        break;
      default:
        // Can this be called on root? SRW
        SLIC_INFO( "CouplingScheme::writeInterfaceOutput(): "
                   << "output routine not yet written for interface method. " );
        break;
    }  // end-switch
  }    // end-if
  return;
}

//------------------------------------------------------------------------------

} /* namespace tribol */
