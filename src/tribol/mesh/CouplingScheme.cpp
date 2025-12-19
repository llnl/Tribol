// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/CouplingScheme.hpp"

// Tribol includes
#include "tribol/common/ExecModel.hpp"
#include "tribol/geom/ElementNormal.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/utils/ContactPlaneOutput.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/search/InterfacePairFinder.hpp"
#include "tribol/common/Parameters.hpp"
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
inline bool validMeshID( IndexT mesh_id )
{
  MeshManager& meshManager = MeshManager::getInstance();
  return ( mesh_id == ANY_MESH ) || meshManager.findData( mesh_id );
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
    case INVALID_CASE_DATA: {
      SLIC_WARNING_ROOT( "The specified ContactCase has invalid data. "
                         << "AUTO contact requires element thickness registration." );
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
    case INVALID_ELEMENT_TYPE: {
      SLIC_WARNING_ROOT( "The specified element type is invalid." );
      break;
    }
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
    case NULL_REFERENCE_COORDS: {
      SLIC_WARNING_ROOT(
          "User must call tribol::registerNodalReferenceCoords() for one or both meshes per registered "
          "ContactMethod." );
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
      SLIC_WARNING_ROOT( "The specified EnforcementMethod is invalid for the registered ContactMethod." );
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
void CouplingSchemeErrors::printExecutionModeErrors()
{
  switch ( this->cs_execution_mode_error ) {
    case ExecutionModeError::NON_MATCHING_MEMORY_SPACE: {
      SLIC_WARNING_ROOT( "Memory spaces for meshes do not match; see warnings." );
      break;
    }
    case ExecutionModeError::BAD_MODE_FOR_MEMORY_SPACE: {
      SLIC_WARNING_ROOT( "Memory space is not compatible with execution mode; see warnings." );
      break;
    }
    case ExecutionModeError::INCOMPATIBLE_METHOD: {
      SLIC_WARNING_ROOT( "Contact method not compatible with execution mode; see warnings." );
      break;
    }
    default:
      break;
  }  // end switch over execution mode errors
}  // end CouplingSchemeErrors::printExecutionModeErrors()

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
    case NO_CASE_INFO: {
      break;
    }
    default:
      break;
  }  // end switch over case info
}  // end CouplingSchemeInfo::printCaseInfo()

//------------------------------------------------------------------------------
void CouplingSchemeInfo::printExecutionModeInfo()
{
  switch ( this->cs_execution_mode_info ) {
    case ExecutionModeInfo::NONOPTIMAL_MODE_FOR_MEMORY_SPACE: {
      SLIC_WARNING_ROOT( "Execution mode is not optimal for memory space; see warnings." );
      break;
    }
    default:
      break;
  }  // end switch over execution mode info
}  // end CouplingSchemeInfo::printExecutionModeInfo()

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
CouplingScheme::CouplingScheme( IndexT cs_id, IndexT mesh_id1, IndexT mesh_id2, int contact_mode, int contact_case,
                                int contact_method, int contact_model, int enforcement_method, int binning_method,
                                ExecutionMode given_exec_mode )
    : m_id( cs_id ),
      m_mesh_id1( mesh_id1 ),
      m_mesh_id2( mesh_id2 ),
      m_given_exec_mode( given_exec_mode ),
      m_numTotalNodes( 0 ),
      m_fixedBinning( false ),
      m_isBinned( false ),
      m_isTied( false ),
      m_methodData( nullptr )
{
  // error sanity checks
  SLIC_ERROR_ROOT_IF( mesh_id1 == ANY_MESH, "mesh_id1 cannot be set to ANY_MESH" );
  SLIC_ERROR_ROOT_IF( !validMeshID( m_mesh_id1 ), "invalid mesh_id1=" << mesh_id1 );
  SLIC_ERROR_ROOT_IF( !validMeshID( m_mesh_id2 ), "invalid mesh_id2=" << mesh_id2 );

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

  m_loggingLevel = LoggingLevel::UNDEFINED;

}  // end CouplingScheme::CouplingScheme()

//------------------------------------------------------------------------------
const ContactPlanePair& CouplingScheme::getContactPlanePair( IndexT id ) const
{
  switch ( this->m_contactMethod ) {
    case COMMON_PLANE: {
      return m_cg_pairs.getCommonPlane( id );
      break;
    }
    case MORTAR_WEIGHTS:
    case SINGLE_MORTAR: {
      return m_cg_pairs.getMortarPlane( id );
      break;
    }
    case ALIGNED_MORTAR: {
      return m_cg_pairs.getAlignedMortarPlane( id );
      break;
    }
    default: {
      // won't get here, but just return something to suppress warnings
      return m_cg_pairs.getMortarPlane( id );
      break;
    }
  }  // end switch
}

//------------------------------------------------------------------------------
bool CouplingScheme::setMeshPointers()
{
  // verify meshes exist and set pointers to meshes
  MeshManager& meshManager = MeshManager::getInstance();
  if ( !meshManager.findData( this->m_mesh_id1 ) || !meshManager.findData( this->m_mesh_id2 ) ) {
    SLIC_WARNING_ROOT( "Please register meshes for coupling scheme, " << this->m_id << "." );
    return false;
  }

  this->m_mesh1 = &MeshManager::getInstance().at( this->m_mesh_id1 );
  this->m_mesh2 = &MeshManager::getInstance().at( this->m_mesh_id2 );

  return true;
}

//------------------------------------------------------------------------------
bool CouplingScheme::isValidCouplingScheme()
{
  bool valid{ true };

  if ( !setMeshPointers() ) {
    return false;
  }

  // set boolean for null meshes
  this->m_nullMeshes = this->m_mesh1->numberOfElements() <= 0 || this->m_mesh2->numberOfElements() <= 0;

  // check for invalid mesh topology matches in a coupling scheme
  if ( this->m_mesh1->getElementType() != this->m_mesh2->getElementType() ) {
    SLIC_WARNING_ROOT( "Coupling scheme " << this->m_id << " does not support meshes with "
                                          << "different surface element types." );
    this->m_mesh1->isMeshValid() = false;
    this->m_mesh2->isMeshValid() = false;
  }

  // check for invalid meshes. A mesh could be deemed invalid when registered.
  if ( !this->m_mesh1->isMeshValid() || !this->m_mesh2->isMeshValid() ) {
    return false;
  }

  // check valid contact mode. Not all modes have an implementation
  if ( !this->isValidMode() ) {
    this->m_couplingSchemeErrors.printModeErrors();
    valid = false;
  }

  // TODO check whether info should be printed before
  // errors in case AUTO needs to be change to NO_CASE
  // and the check on element thickness needs to be modified
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

  switch ( this->checkExecutionModeData() ) {
    case 1:
      this->m_couplingSchemeErrors.printExecutionModeErrors();
      valid = false;
      break;
    case 2:
      this->m_couplingSchemeInfo.printExecutionModeInfo();
      break;
    default:
      // no info or error messages
      break;
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
  // set to no error until otherwise noted
  this->m_couplingSchemeErrors.cs_case_error = NO_CASE_ERROR;
  bool isValid = true;  // pre-set to a valid case

  // check if contactCase is not an existing option
  if ( !in_range( this->m_contactCase, NUM_CONTACT_CASES ) ) {
    this->m_couplingSchemeErrors.cs_case_error = INVALID_CASE;
    isValid = false;
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

  if ( this->m_contactMethod == COMMON_PLANE ) {
    switch ( this->m_contactCase ) {
      case AUTO: {
        // enable auto-contact specific checks through boolean, and check to
        // make sure element thicknesses have been registered for auto-contact
        // length scale calculations
        this->m_parameters.auto_contact_check = true;

        if ( !this->m_mesh1->getElementData().m_is_element_thickness_set ||
             !this->m_mesh2->getElementData().m_is_element_thickness_set ) {
          this->m_couplingSchemeErrors.cs_case_error = INVALID_CASE_DATA;
          isValid = false;
        }
        break;
      }
      case TIED_FULL: {
        // uncomment when there is an implementation
        // this->m_parameters.auto_contact_check = false;
        this->m_couplingSchemeErrors.cs_case_error = NO_CASE_IMPLEMENTATION;
        isValid = false;
        break;
      }
      default:
        this->m_parameters.auto_contact_check = false;
    }  // end switch on case
  }  // end if check on common-plane

  return isValid;
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

  int dim = this->spatialDimension();

  // check all methods for basic validity issues for non-null meshes
  if ( !this->m_nullMeshes ) {
    if ( ( this->m_mesh1->getElementType() != LINEAR_EDGE && this->m_mesh1->getElementType() != LINEAR_QUAD &&
           this->m_mesh1->getElementType() != LINEAR_TRIANGLE ) ||
         ( this->m_mesh2->getElementType() != LINEAR_EDGE && this->m_mesh2->getElementType() != LINEAR_QUAD &&
           this->m_mesh2->getElementType() != LINEAR_TRIANGLE ) ) {
      this->m_couplingSchemeErrors.cs_method_error = INVALID_ELEMENT_TYPE;
      return false;
    }

    if ( this->m_contactMethod == ALIGNED_MORTAR || this->m_contactMethod == MORTAR_WEIGHTS ||
         this->m_contactMethod == SINGLE_MORTAR ) {
      if ( this->m_mesh1->numberOfNodesPerElement() != this->m_mesh2->numberOfNodesPerElement() ) {
        this->m_couplingSchemeErrors.cs_method_error = DIFFERENT_FACE_TYPES;
        return false;
      }
      if ( this->m_mesh_id1 == this->m_mesh_id2 ) {
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
      if ( this->m_mesh1->numberOfNodesPerElement() != this->m_mesh2->numberOfNodesPerElement() ) {
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
      if ( this->m_mesh1->numberOfElements() > 0 && !this->m_mesh1->getNodalFields().m_is_nodal_response_set ) {
        this->m_couplingSchemeErrors.cs_method_error = NULL_NODAL_RESPONSE;
        return false;
      }

      if ( this->m_mesh2->numberOfElements() > 0 && !this->m_mesh2->getNodalFields().m_is_nodal_response_set ) {
        this->m_couplingSchemeErrors.cs_method_error = NULL_NODAL_RESPONSE;
        return false;
      }
    }

    if ( this->m_contactMethod == SINGLE_MORTAR && this->m_useEnzyme ) {
      // this is only needed on the nonmortar mesh (def'd as mesh2 for Enzyme mortar method)
      if ( this->m_mesh2->numberOfElements() > 0 && !this->m_mesh2->hasReferencePosition() ) {
        this->m_couplingSchemeErrors.cs_method_error = NULL_REFERENCE_COORDS;
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
           this->m_contactModel != VISCOUS_TANGENTIAL ) {
        this->m_couplingSchemeErrors.cs_model_error = NO_MODEL_IMPLEMENTATION_FOR_REGISTERED_METHOD;
        return false;
      }
      if ( this->m_contactCase == TIED_NORMAL && this->m_contactModel == ADHESION_SEPARATION_SCALAR_LAW ) {
        this->m_couplingSchemeErrors.cs_model_error = NO_MODEL_IMPLEMENTATION;
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
        if ( !this->m_enforcementOptions.lm_implicit_options.enforcement_option_set ) {
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
      } else if ( !this->m_enforcementOptions.penalty_options.constraint_type_set ) {
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
          if ( this->m_mesh2->checkLagrangeMultiplierData() != 0 )  // nonmortar side only
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
          if ( this->m_mesh1->checkPenaltyData( pen_enfrc_options, this->m_exec_mode ) != 0 ||
               this->m_mesh2->checkPenaltyData( pen_enfrc_options, this->m_exec_mode ) != 0 ) {
            this->m_couplingSchemeErrors.cs_enforcement_data_error = ERROR_IN_REGISTERED_ENFORCEMENT_DATA;
            err = 1;
          }
          break;
        }  // end case PENALTY
        default:
          // no-op
          break;
      }  // end switch over enforcement method
    }  // end case COMMON_PLANE
    default:
      // no-op
      break;
  }  // end switch on method

  return err;

}  // end CouplingScheme::checkEnforcementData()

//------------------------------------------------------------------------------
int CouplingScheme::checkExecutionModeData()
{
  int err = 0;
  this->m_exec_mode = ExecutionMode::Sequential;
  this->m_couplingSchemeErrors.cs_execution_mode_error = ExecutionModeError::NO_ERROR;
  this->m_couplingSchemeInfo.cs_execution_mode_info = ExecutionModeInfo::NO_INFO;

  if ( this->m_mesh1->getMemorySpace() != this->m_mesh2->getMemorySpace() ) {
    SLIC_WARNING_ROOT( "Coupling scheme " << this->m_id << ": Paired meshes reside in "
                                          << "different memory spaces." );
    this->m_mesh1->isMeshValid() = false;
    this->m_mesh2->isMeshValid() = false;
    this->m_couplingSchemeErrors.cs_execution_mode_error = ExecutionModeError::NON_MATCHING_MEMORY_SPACE;
    err = 1;
    return err;
  }

  switch ( this->m_mesh1->getMemorySpace() ) {
    case MemorySpace::Dynamic:
#ifdef TRIBOL_USE_UMPIRE
      // trust the user here...
      this->m_exec_mode = this->m_given_exec_mode;
      if ( this->m_exec_mode == ExecutionMode::Dynamic ) {
        SLIC_WARNING_ROOT(
            "Dynamic execution with dynamic memory space. "
            "Unable to determine execution mode." );
        this->m_couplingSchemeErrors.cs_execution_mode_error = ExecutionModeError::BAD_MODE_FOR_MEMORY_SPACE;
        err = 1;
      }
#else
      // if we have RAJA but no Umpire, execute serially on host
      this->m_exec_mode = ExecutionMode::Sequential;
#endif
      break;
#ifdef TRIBOL_USE_UMPIRE
    case MemorySpace::Unified:
      // this should be able to run anywhere. let the user decide.
      this->m_exec_mode = this->m_given_exec_mode;
      if ( this->m_exec_mode == ExecutionMode::Dynamic ) {
#if defined( TRIBOL_USE_CUDA )
        SLIC_INFO_ROOT(
            "Dynamic execution with unified memory space. "
            "Assuming CUDA parallel execution." );
        this->m_exec_mode = ExecutionMode::Cuda;
#elif defined( TRIBOL_USE_HIP )
        SLIC_INFO_ROOT(
            "Dynamic execution with unified memory space. "
            "Assuming HIP parallel execution." );
        this->m_exec_mode = ExecutionMode::Hip;
#elif defined( TRIBOL_USE_OPENMP )
        SLIC_INFO_ROOT(
            "Dynamic execution with unified memory space. "
            "Tribol not built with CUDA/HIP support. "
            "Assuming OpenMP parallel execution." );
        this->m_exec_mode = ExecutionMode::OpenMP;
        this->m_couplingSchemeInfo.cs_execution_mode_info = ExecutionModeInfo::NONOPTIMAL_MODE_FOR_MEMORY_SPACE;
        err = 2;
#else
        SLIC_INFO_ROOT(
            "Dynamic execution with unified memory space. "
            "Tribol not built with CUDA/HIP/OpenMP support. "
            "Assuming sequential execution." );
        this->m_exec_mode = ExecutionMode::Sequential;
        this->m_couplingSchemeInfo.cs_execution_mode_info = ExecutionModeInfo::NONOPTIMAL_MODE_FOR_MEMORY_SPACE;
        err = 2;
#endif
      }
      break;
#endif
    case MemorySpace::Host:
      switch ( this->m_given_exec_mode ) {
        case ExecutionMode::Sequential:
#ifdef TRIBOL_USE_OPENMP
        case ExecutionMode::OpenMP:
#endif
          this->m_exec_mode = this->m_given_exec_mode;
          break;
        case ExecutionMode::Dynamic:
#ifdef TRIBOL_USE_OPENMP
          SLIC_INFO_ROOT(
              "Dynamic execution with a host memory space. "
              "Assuming OpenMP parallel execution." );
          this->m_exec_mode = ExecutionMode::OpenMP;
#else
          SLIC_INFO_ROOT(
              "Dynamic execution with a host memory space. "
              "Assuming sequential execution." );
          this->m_exec_mode = ExecutionMode::Sequential;
#endif
          break;
        default:
          SLIC_WARNING_ROOT(
              "Unsupported execution mode for host memory. "
              "Unable to determine execution mode." );
          this->m_couplingSchemeErrors.cs_execution_mode_error = ExecutionModeError::BAD_MODE_FOR_MEMORY_SPACE;
          err = 1;
          break;
      }
      break;
#ifdef TRIBOL_USE_UMPIRE
    case MemorySpace::Device:
      switch ( this->m_given_exec_mode ) {
#if defined( TRIBOL_USE_CUDA ) || defined( TRIBOL_USE_HIP )
#ifdef TRIBOL_USE_CUDA
        case ExecutionMode::Cuda:
#endif
#ifdef TRIBOL_USE_HIP
        case ExecutionMode::Hip:
#endif
          this->m_exec_mode = this->m_given_exec_mode;
          break;
#endif
        case ExecutionMode::Dynamic:
#if defined( TRIBOL_USE_CUDA )
          this->m_exec_mode = ExecutionMode::Cuda;
          break;
#elif defined( TRIBOL_USE_HIP )
          this->m_exec_mode = ExecutionMode::Hip;
          break;
#endif
        default:
          SLIC_ERROR_ROOT(
              "Unknown execution mode for device memory. "
              "Unable to determine execution mode." );
          this->m_couplingSchemeErrors.cs_execution_mode_error = ExecutionModeError::BAD_MODE_FOR_MEMORY_SPACE;
          err = 1;
          break;
      }
#endif
  }

  // Update memory spaces of mesh data which are originally set as dynamic
  // (ensures data owned by MeshData is in the right memory space)
  if ( this->m_mesh1->getMemorySpace() == MemorySpace::Dynamic ) {
    switch ( this->m_exec_mode ) {
      case ExecutionMode::Sequential:
#ifdef TRIBOL_USE_OPENMP
      case ExecutionMode::OpenMP:
#endif
        this->m_mesh1->updateAllocatorId( getResourceAllocatorID( MemorySpace::Host ) );
        this->m_mesh2->updateAllocatorId( getResourceAllocatorID( MemorySpace::Host ) );
        break;
#ifdef TRIBOL_USE_CUDA
      case ExecutionMode::Cuda:
        this->m_mesh1->updateAllocatorId( getResourceAllocatorID( MemorySpace::Device ) );
        this->m_mesh2->updateAllocatorId( getResourceAllocatorID( MemorySpace::Device ) );
        break;
#endif
#ifdef TRIBOL_USE_HIP
      case ExecutionMode::Hip:
        this->m_mesh1->updateAllocatorId( getResourceAllocatorID( MemorySpace::Device ) );
        this->m_mesh2->updateAllocatorId( getResourceAllocatorID( MemorySpace::Device ) );
        break;
#endif
      default:
        // no-op
        break;
    }
  }
  this->m_allocator_id = this->m_mesh1->getAllocatorId();

  if ( m_contactMethod != COMMON_PLANE ) {
    if ( m_exec_mode != ExecutionMode::Sequential ) {
      SLIC_WARNING_ROOT( "Only sequential execution on host supported for contact methods other than COMMON_PLANE." );
      this->m_couplingSchemeErrors.cs_execution_mode_error = ExecutionModeError::INCOMPATIBLE_METHOD;
      err = 1;
    }
  }

  return err;
}

//------------------------------------------------------------------------------
void CouplingScheme::performBinning()
{
  // Find the interacting pairs for this coupling scheme. Will not use
  // binning if setInterfacePairs has been called.
  if ( !this->hasFixedBinning() ) {
    // create interface pairs based on allocator id
    m_interface_pairs = ArrayT<InterfacePair>( 0, 0, m_allocator_id );

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
  return;
}

//------------------------------------------------------------------------------
int CouplingScheme::apply( int cycle, RealT t, RealT& dt )
{
  auto& params = m_parameters;

  // loop over number of interface pairs
  IndexT numPairs = m_interface_pairs.size();

  SLIC_DEBUG( "Coupling scheme " << m_id << " has " << numPairs << " pairs." );

  // loop over all pairs and perform geometry checks to see if they are
  // interacting
  auto pairs = getInterfacePairs().view();
  auto contact_method = m_contactMethod;
  auto contact_case = m_contactCase;
  ArrayT<int> pair_err_data( 1, 1, getAllocatorId() );
  auto pair_err = pair_err_data.view();

  // allocate planes based on contact method
  m_cg_pairs.allocatePlanePairs( contact_method, numPairs, getAllocatorId() );

  // get comp geom container view after allocating pair arrays
  auto cg_view = m_cg_pairs.getView();

  // get mesh views
  auto mesh1 = getMesh1().getView();
  auto mesh2 = getMesh2().getView();
  // array of size one for counting number of planes on device
  ArrayT<IndexT> planes_ct_data( 1, 1, getAllocatorId() );
  auto planes_ct = planes_ct_data.view();
  forAllExec( getExecutionMode(), numPairs,
              [pairs, mesh1, mesh2, params, contact_method, contact_case, cg_view, planes_ct,
               pair_err] TRIBOL_HOST_DEVICE( IndexT i ) mutable {
                auto& pair = pairs[i];

                // call wrapper around the contact method/case specific
                // geometry checks to determine whether to include a pair
                // in the active set
                bool interact = false;

                FaceGeomException interact_err = CheckInterfacePair(
                    pair, mesh1, mesh2, params, contact_method, contact_case, interact, cg_view, planes_ct.data() );

                // // Update pair reporting data for this coupling scheme
                // this->updatePairReportingData( interact_err );

                // TODO refine how these errors are handled. Here we skip over face-pairs with errors. That is,
                // they are not registered for contact, but we don't error out.
                if ( interact_err != NO_FACE_GEOM_EXCEPTION ) {
                  pair_err[0] = 1;
                  pair.m_is_contact_candidate = false;
                  // TODO consider printing offending face(s) coordinates for debugging
                  // SLIC_DEBUG("Face geometry error, " << static_cast<int>(interact_err) << "for pair, " << kp << ".");
                  // continue; // TODO SRW why do we need this? Seems like we want to update interface pair below
                  // if-statements
                } else if ( !interact ) {
                  pair.m_is_contact_candidate = false;
                } else {
                  pair.m_is_contact_candidate = true;
                }
              } );

  ArrayT<IndexT, 1, MemorySpace::Host> planes_ct_host( planes_ct_data );
  // shrink array to actual number of contact planes
  m_cg_pairs.resizeActivePairs( contact_method, planes_ct_host[0] );

  // Here, the pair_err is checked, which detects an issue with a face-pair geometry
  // (which has been skipped over for contact eligibility) and reports this warning.
  // This is intended to indicate to a user that there may be bad geometry, or issues with
  // complex cg calculations that need debugging.
  //
  // This is complex because a host-code may have unavoidable 'bad' geometry and wish
  // to continue the simulation. In this case, we may 'punt' on those face-pairs, which
  // may be reasonable and not an error. Alternatively, this warning may indicate a bug
  // or issue in the cg that a host-code does desire to have resolved. For this reason, this
  // message is kept at the warning level.
  ArrayT<int, 1, MemorySpace::Host> pair_err_host( pair_err_data );
  SLIC_DEBUG_IF( pair_err_host[0] != 0, "CouplingScheme::apply(): possible issues with orientation, "
                                            << "input, or invalid overlaps in CheckInterfacePair()." );

  // aggregate across ranks for this coupling scheme? SRW
  SLIC_DEBUG( "Number of active interface pairs: " << getNumActivePairs() );

  // wrapper around contact method, case, and
  // enforcement to apply the interface physics in both
  // normal and tangential directions. This function loops
  // over the pairs on the coupling scheme and applies the
  // appropriate physics in the normal and tangential directions.
  int err = ApplyInterfacePhysics( this, cycle, t );

  SLIC_WARNING_IF( err != 0, "CouplingScheme::apply(): error in ApplyInterfacePhysics for " << "coupling scheme, "
                                                                                            << this->m_id << "." );

  // compute Tribol timestep vote on the coupling scheme
  if ( err == 0 && getNumActivePairs() > 0 ) {
    computeTimeStep( dt );
  }

  // write contact plane output if it is on host
  if ( !isOnDevice( this->m_exec_mode ) ) {
    writeInterfaceOutput( m_output_directory, params.vis_type, cycle, t );
  }

  if ( err != 0 ) {
    return 1;
  } else {
    // here we don't have any error in the application of interface physics,
    // but may have face-pair data reporting skipped pair statistics for debug print
    this->printPairReportingData();
    return 0;
  }

}  // end CouplingScheme::apply()

//------------------------------------------------------------------------------
bool CouplingScheme::init()
{
  // check for valid coupling scheme only for non-null-meshes
  this->m_isValid = this->isValidCouplingScheme();

  if ( this->m_isValid ) {
    // set individual coupling scheme logging level
    this->setSlicLoggingLevel();

    // set effective binning proximity, based on binning proximity scale and LOR factor
    this->m_effective_binning_proximity_scale = this->getParameters().binning_proximity_scale;
#ifdef BUILD_REDECOMP
    if ( this->hasMfemData() && this->getMfemMeshData()->GetLORFactor() > 1 ) {
      this->m_effective_binning_proximity_scale *= static_cast<RealT>( this->getMfemMeshData()->GetLORFactor() );
    }
#endif

    // compute the face data
    // different element normals for enzyme + mortar (matching Puso and Laursen)
    if ( this->isEnzymeEnabled() && this->m_contactMethod == SINGLE_MORTAR ) {
      this->m_mesh1->computeFaceData( this->m_exec_mode, ElementCentroidNormal() );
      if ( this->m_mesh_id2 != this->m_mesh_id1 ) {
        this->m_mesh2->computeFaceData( this->m_exec_mode, ElementCentroidNormal() );
      }
    } else {
      this->m_mesh1->computeFaceData( this->m_exec_mode, PalletAvgNormal() );
      if ( this->m_mesh_id2 != this->m_mesh_id1 ) {
        this->m_mesh2->computeFaceData( this->m_exec_mode, PalletAvgNormal() );
      }
    }

    this->allocateMethodData();

    return true;
  } else {
    return false;
  }
}

//------------------------------------------------------------------------------
void CouplingScheme::setSlicLoggingLevel()
{
  // set slic logging level for coupling schemes that have API modified logging levels
  if ( this->m_loggingLevel != LoggingLevel::UNDEFINED ) {
    switch ( this->m_loggingLevel ) {
      case LoggingLevel::DEBUG: {
        axom::slic::setLoggingMsgLevel( axom::slic::message::Debug );
        break;
      }
      case LoggingLevel::INFO: {
        axom::slic::setLoggingMsgLevel( axom::slic::message::Info );
        break;
      }
      case LoggingLevel::WARNING: {
        axom::slic::setLoggingMsgLevel( axom::slic::message::Warning );
        break;
      }
      case LoggingLevel::ERROR: {
        axom::slic::setLoggingMsgLevel( axom::slic::message::Error );
        break;
      }
      default: {
        axom::slic::setLoggingMsgLevel( axom::slic::message::Warning );
        break;
      }
    }  // end switch
  }  // end if
}

//------------------------------------------------------------------------------
void CouplingScheme::allocateMethodData()
{
  auto& mesh1 = MeshManager::getInstance().getData( m_mesh_id1 );
  this->m_numTotalNodes = mesh1.numberOfNodes();

  // dynamically allocate method data object for mortar method
  switch ( this->m_contactMethod ) {
    case ALIGNED_MORTAR:
    case MORTAR_WEIGHTS:
    case SINGLE_MORTAR: {
      // dynamically allocate method data object
      this->m_methodData = std::make_unique<MortarData>();
      static_cast<MortarData*>( m_methodData.get() )->m_numTotalNodes = this->m_numTotalNodes;
      break;
    }  // end case SINGLE_MORTAR
    default: {
      this->m_methodData = nullptr;
      break;
    }
  }
}  // end CouplingScheme::allocateMethodData()

//------------------------------------------------------------------------------
void CouplingScheme::computeTimeStep( RealT& dt )
{
  if ( dt < 1.e-8 ) {
    // current timestep too small for Tribol vote. Leave unchanged and return
    return;
  }

  // make sure velocities are registered
  if ( !m_mesh1->hasVelocity() || !m_mesh2->hasVelocity() ) {
    if ( m_mesh1->numberOfElements() > 0 && m_mesh2->numberOfElements() > 0 ) {
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
        if ( m_parameters.enable_timestep_vote ) {
          this->computeCommonPlaneTimeStep( dt );
        }
      }
      break;
    default:
      break;
  }  // end-switch
}

//------------------------------------------------------------------------------
void CouplingScheme::computeCommonPlaneTimeStep( RealT& dt )
{
  // note: the timestep vote is based on a maximum allowable interpenetration
  // approach checking current gaps and then performing a velocity projection.
  // This timestep vote is not derived from a stability analysis and does not
  // account for the spring stiffness in a CFL-like timestep constraint.
  // The timestep vote in this routine is intended to avoid missing contact or
  // inadequately resolving contact by detecting 'too much' interpenetration.
  // To do this, the first check is a 'gap check' where the current gap is
  // checked against a fraction of the element-thicknesses. The second check
  // assumes zero gap, and then does a velocity projection followed by the
  // 'gap check' using this new, projected gap.

  auto& mesh1 = getMesh1();
  auto& mesh2 = getMesh2();

  // check if the element thicknesses have been set. This is now
  // regardless of exact penalty calculation type, since element
  // thicknesses are required for auto contact even if a constant
  // penalty is used
  if ( !mesh1.getElementData().m_is_element_thickness_set || !mesh2.getElementData().m_is_element_thickness_set ) {
    return;
  }

  RealT proj_ratio = m_parameters.timestep_pen_frac;
  // int num_sides = 2; // always 2 sides in a single coupling scheme
  int dim = spatialDimension();

  // Loop over each contact plane. Even if pair is not in contact, we still do a
  // velocity projection for that proximate face-pair to see if interpenetration
  // next cycle 'may' be too much. Note, if a contact plane exists, then it is a
  // contact candidate. For this calculation, we want to compute a timestep for
  // all candidates, not necessarily ones that are deemed to be in contact per
  // the gap constraint.
  auto cs_view = getView();
  ArrayT<RealT> dt_temp_data( { dt, dt }, getAllocatorId() );
  ArrayViewT<RealT> dt_temp = dt_temp_data;
  // [0]: exceed_max_gap1, [1]: exceed_max_gap2, [2]: neg_dt_gap_msg, [3]: neg_dt_vel_proj_msg
  ArrayT<IndexT> msg_data( { static_cast<IndexT>( false ), static_cast<IndexT>( false ), static_cast<IndexT>( false ),
                             static_cast<IndexT>( false ) },
                           getAllocatorId() );
  ArrayViewT<IndexT> msg = msg_data;
  forAllExec( getExecutionMode(), getNumActivePairs(),
              [cs_view, dim, proj_ratio, msg, dt_temp, dt] TRIBOL_HOST_DEVICE( IndexT i ) {
                auto& cg_view = cs_view.getCompGeomView();
                auto& plane = cg_view.getCommonPlane( i );

                auto& mesh1 = cs_view.getMesh1View();
                auto& mesh2 = cs_view.getMesh2View();

                // get pair indices
                IndexT index1 = plane.getCpElementId1();
                IndexT index2 = plane.getCpElementId2();

                constexpr int max_dim = 3;
                constexpr int max_nodes_per_elem = 4;
                StackArrayT<RealT, max_dim * max_nodes_per_elem> x1;
                StackArrayT<RealT, max_dim * max_nodes_per_elem> v1;
                mesh1.getFaceCoords( index1, x1 );
                mesh1.getFaceVelocities( index1, v1 );

                StackArrayT<RealT, max_dim * max_nodes_per_elem> x2;
                StackArrayT<RealT, max_dim * max_nodes_per_elem> v2;
                mesh2.getFaceCoords( index2, x2 );
                mesh2.getFaceVelocities( index2, v2 );

                /////////////////////////////////////////////////////////////
                // calculate face velocities at projected overlap centroid //
                /////////////////////////////////////////////////////////////
                StackArrayT<RealT, max_dim> vel_f1;
                StackArrayT<RealT, max_dim> vel_f2;
                initRealArray( vel_f1, dim, 0.0 );
                initRealArray( vel_f2, dim, 0.0 );

                // interpolate nodal velocity at overlap centroid as projected
                // onto face 1
                RealT cXf1 = plane.m_cXf1;
                RealT cYf1 = plane.m_cYf1;
                RealT cZf1 = ( dim == 3 ) ? plane.m_cZf1 : 0.;
                GalerkinEval( x1, cXf1, cYf1, cZf1, LINEAR, PHYSICAL, dim, dim, v1, vel_f1 );
                // interpolate nodal velocity at overlap centroid as projected
                // onto face 2
                RealT cXf2 = plane.m_cXf2;
                RealT cYf2 = plane.m_cYf2;
                RealT cZf2 = ( dim == 3 ) ? plane.m_cZf2 : 0.;
                GalerkinEval( x2, cXf2, cYf2, cZf2, LINEAR, PHYSICAL, dim, dim, v2, vel_f2 );

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
                // face's outward unit normal AND the overlap normal.
                RealT v1_dot_n, v2_dot_n, v1_dot_n1, v2_dot_n2;
                RealT overlapNormal[max_dim];
                overlapNormal[0] = plane.m_nX;
                overlapNormal[1] = plane.m_nY;
                if ( dim == 3 ) {
                  overlapNormal[2] = plane.m_nZ;
                }

                // get face normals
                RealT fn1[max_dim], fn2[max_dim];
                mesh1.getFaceNormal( index1, fn1 );
                mesh2.getFaceNormal( index2, fn2 );

                // compute projections
                v1_dot_n = dotProd( vel_f1, overlapNormal, dim );
                v2_dot_n = dotProd( vel_f2, overlapNormal, dim );
                v1_dot_n1 = dotProd( vel_f1, fn1, dim );
                v2_dot_n2 = dotProd( vel_f2, fn2, dim );

                // Keep debug print statements. This routine is still in the testing phase
                // std::cout << "face 1 normal: " << fn1[0] << ", " << fn1[1] << ", " << fn1[2] << std::endl;
                // std::cout << "face 2 normal: " << fn2[0] << ", " << fn2[1] << ", " << fn2[2] << std::endl;
                // std::cout << " " << std::endl;
                // std::cout << "face 1 vel: " << vel_f1[0] << ", " << vel_f1[1] << ", " << vel_f1[2] << std::endl;
                // std::cout << "face 2 vel: " << vel_f2[0] << ", " << vel_f2[1] << ", " << vel_f2[2] << std::endl;
                // std::cout << " " << std::endl;
                // std::cout << "First v1_dot_n1 calc: " << v1_dot_n1 << std::endl;
                // std::cout << "First v2_dot_n2 calc: " << v2_dot_n2 << std::endl;
                // std::cout << "First v1_dot_n: " << v1_dot_n << std::endl;
                // std::cout << "First v2_dot_n: " << v2_dot_n << std::endl;

                // add tiny amount to velocity projections to avoid division by zero.
                // Note that if these projections are close to zero, there may be
                // stationary interactions or tangential motion. In this case, any
                // timestep estimate will be very large, and not control the simulation
                RealT tiny = 1.e-12;
                RealT tiny1 = ( v1_dot_n >= 0. ) ? tiny : -1. * tiny;
                RealT tiny2 = ( v2_dot_n >= 0. ) ? tiny : -1. * tiny;
                v1_dot_n += tiny1;
                v2_dot_n += tiny2;
                // reset tiny velocity based on face normal projections.
                tiny1 = ( v1_dot_n1 >= 0. ) ? tiny : -1. * tiny;
                tiny2 = ( v2_dot_n2 >= 0. ) ? tiny : -1. * tiny;
                v1_dot_n1 += tiny1;
                v2_dot_n2 += tiny2;

                // Keep debug print statements. This routine is still in the testing phase
                // std::cout << "Second v1_dot_n1 calc: " << v1_dot_n1 << std::endl;
                // std::cout << "Second v2_dot_n2 calc: " << v2_dot_n2 << std::endl;
                // std::cout << "Second v1_dot_n: " << v1_dot_n << std::endl;
                // std::cout << "Second v2_dot_n: " << v2_dot_n << std::endl;

                // get volume element thicknesses associated with each face in this pair
                RealT t1 = mesh1.getElementData().m_thickness[index1];
                RealT t2 = mesh2.getElementData().m_thickness[index2];

                // compute the existing gap vector (recall gap is x1-x2 by convention)
                RealT gapVec[max_dim];
                gapVec[0] = plane.m_cXf1 - plane.m_cXf2;
                gapVec[1] = plane.m_cYf1 - plane.m_cYf2;
                if ( dim == 3 ) {
                  gapVec[2] = plane.m_cZf1 - plane.m_cZf2;
                }

                // compute the dot product between gap vector and the outward unit face normals.
                RealT gap_f1_n1 = dotProd( gapVec, fn1, dim );
                RealT gap_f2_n2 = dotProd( gapVec, fn2, dim );

                RealT dt1 = 1.e6;                          // initialize as large number
                RealT dt2 = 1.e6;                          // initialize as large number
                RealT alpha = cs_view.getTimestepScale();  // multiplier on timestep estimate
                bool dt1_check1 = false;
                bool dt2_check1 = false;
                bool dt1_vel_check = false;
                bool dt2_vel_check = false;

                // maximum allowable interpenetration in the normal direction of each element
                RealT max_delta1 = proj_ratio * t1;
                RealT max_delta2 = proj_ratio * t2;

                // Separation or interpenetration trigger for check 1 and 2:
                // check if there is further interpen or separation based on the
                // velocity projection in the direction of the common-plane normal,
                // which is in the direction of face-2 normal.
                // The two cases are:
                // if v1*n < 0 there is interpen
                // if v2*n > 0 there is interpen
                //
                // Note: we compare strictly to 0. here since a 'tiny' value was
                // appropriately added to the velocity projections, which is akin
                // to some tolerancing effect
                dt1_vel_check = ( v1_dot_n < 0. ) ? true : false;
                dt2_vel_check = ( v2_dot_n > 0. ) ? true : false;

                //////////////////////////////////////////////////////////////////////////
                // Check 1. Current interpenetration gap exceeds max allowable interpen //
                //////////////////////////////////////////////////////////////////////////

                // check if face-pair is in contact (i.e. gap < gap_tol), which is determined
                // in Common Plane ApplyNormal<>() routine
                if ( plane.m_inContact ) {
                  // compute the difference between the 'face-gaps' and the max allowable
                  // interpen as a function of element thickness. Note, we have to use the
                  // gap projected onto the outward unit face-normal to check against the
                  // max allowable gap as a factor of the thickness in the element normal
                  // direction
                  RealT delta1 = max_delta1 - gap_f1_n1;  // >0 not exceeding max allowable
                  RealT delta2 = max_delta2 + gap_f2_n2;  // >0 not exceeding max allowable

                  auto exceed_max_gap1 = ( delta1 < 0. ) ? true : false;
                  auto exceed_max_gap2 = ( delta2 < 0. ) ? true : false;

                  // if velocity projection indicates further interpenetration, and the gaps
                  // EXCEED max allowable, then compute time step estimates to reduce overlap
                  dt1_check1 = ( dt1_vel_check ) ? exceed_max_gap1 : false;
                  dt2_check1 = ( dt2_vel_check ) ? exceed_max_gap2 : false;

#ifdef TRIBOL_USE_RAJA
                  RAJA::atomicMax<RAJA::auto_atomic>( &msg[0], static_cast<IndexT>( exceed_max_gap1 ) );
                  RAJA::atomicMax<RAJA::auto_atomic>( &msg[1], static_cast<IndexT>( exceed_max_gap2 ) );
#else
                  msg[0] = exceed_max_gap1;
                  msg[1] = exceed_max_gap2;
#endif

                  // compute dt for face 1 and 2 based on the velocity and gap projections onto
                  // the face-normals for faces where currect gap exceeds max allowable gap.
                  //
                  // NOTE:
                  //
                  // This calculation RESETS the current gap to be g = 0, and computes a timestep
                  // such that the velocity projection of the overlap-to-face projected overlap
                  // centroid does not exceed the max allowable gap.
                  //
                  // This avoid a timestep crash in the case that the current gap barely exceeds
                  // the max allowable and also allows a soft contact response with interpen
                  // in excess of the max allowable gap without causing timestep crashes.
                  //
                  // v1_dot_n1 > 0 and v2_dot_n2 > 0 for further interpen
                  dt1 = ( dt1_check1 ) ? alpha * max_delta1 / v1_dot_n1 : dt1;
                  dt2 = ( dt2_check1 ) ? alpha * max_delta2 / v2_dot_n2 : dt2;

                  // Keep debug print statements. This routine is still in the testing phase
                  // std::cout << "dt1_check1, delta1 and v1_dot_n1: " << dt1_check1 << ", " << max_delta1 << ", " <<
                  // v1_dot_n1 << std::endl; std::cout << "dt2_check1, delta2 and v2_dot_n2: " << dt2_check1 << ", " <<
                  // max_delta2 << ", " << v2_dot_n2 << std::endl; std::cout << "dt1 and dt2: " << dt1 << ", " << dt2 <<
                  // std::endl;

                  // update dt_temp1 only for positive dt1 and/or dt2
                  if ( dt1 > 0. ) {
#ifdef TRIBOL_USE_RAJA
                    RAJA::atomicMin<RAJA::auto_atomic>( &dt_temp[0], axom::utilities::min( dt1, 1.e6 ) );
#else
                    dt_temp[0] = axom::utilities::min(dt_temp[0], axom::utilities::min(dt1, 1.e6));
#endif
                  }
                  if ( dt2 > 0. ) {
#ifdef TRIBOL_USE_RAJA
                    RAJA::atomicMin<RAJA::auto_atomic>( &dt_temp[0], axom::utilities::min( 1.e6, dt2 ) );
#else
                    dt_temp[0] = axom::utilities::min(dt_temp[0], axom::utilities::min(1.e6, dt2));
#endif
                  }

                  if ( dt1 < 0. || dt2 < 0. ) {
#ifdef TRIBOL_USE_RAJA
                    RAJA::atomicMax<RAJA::auto_atomic>( &msg[2], static_cast<IndexT>( true ) );
#else
                    msg[2] = true;
#endif
                  }
                }  // end case 1

                ////////////////////////////////////////////////////////////////////////
                // 2. Velocity projection exceeds max interpenetration                //
                //                                                                    //
                //    Note: This is performed for all contact candidates even if they //
                //          are not 'in contact' per the common-plane method. Every   //
                //          contact candidate has a contact plane                     //
                ////////////////////////////////////////////////////////////////////////

                {
                  // compute the delta between the velocity projection of each face-projected
                  // common plane centroid location
                  //
                  // First project each face-projected common plane centroid using linear velocity
                  // projection as approximation of configuration next cycle
                  RealT proj_delta_x1 = plane.m_cXf1 + dt * vel_f1[0];
                  RealT proj_delta_y1 = plane.m_cYf1 + dt * vel_f1[1];
                  RealT proj_delta_z1 = 0.;

                  RealT proj_delta_x2 = plane.m_cXf2 + dt * vel_f2[0];
                  RealT proj_delta_y2 = plane.m_cYf2 + dt * vel_f2[1];
                  RealT proj_delta_z2 = 0.;

                  // Second compute the amount of interpenetration as the difference between the two
                  // velocity projected points
                  RealT proj_delta_x1_fixed = proj_delta_x1;
                  RealT proj_delta_y1_fixed = proj_delta_y1;

                  proj_delta_x1 -= proj_delta_x2;
                  proj_delta_y1 -= proj_delta_y2;

                  proj_delta_x2 -= proj_delta_x1_fixed;
                  proj_delta_y2 -= proj_delta_y1_fixed;

                  // compute the dot product between each face's delta and the OTHER
                  // face's outward unit normal. This is the magnitude of interpenetration
                  // of one face's projected overlap-centroid in the 'thickness-direction'
                  // of the other face (with whom in may be in contact currently, or in
                  // a velocity projected sense).
                  RealT proj_delta_n_1 = proj_delta_x1 * fn2[0] + proj_delta_y1 * fn2[1];
                  RealT proj_delta_n_2 = proj_delta_x2 * fn1[0] + proj_delta_y2 * fn1[1];

                  if ( dim == 3 ) {
                    // project the z-component
                    proj_delta_z1 = plane.m_cZf1 + dt * vel_f1[2];
                    proj_delta_z2 = plane.m_cZf2 + dt * vel_f2[2];

                    RealT proj_delta_z1_fixed = proj_delta_z1;

                    // compute difference between each projected point
                    proj_delta_z1 -= proj_delta_z2;
                    proj_delta_z2 -= proj_delta_z1_fixed;

                    // add the z-component of the projection onto the face normal
                    proj_delta_n_1 += proj_delta_z1 * fn2[2];
                    proj_delta_n_2 += proj_delta_z2 * fn1[2];
                  }

                  // Reset the dt velocity check only for faces with continued interpen that exceeds the
                  // max allowable gap AND where the current gap did NOT exceed that face's max allowable
                  // gap per check 1 (would result in same dt calc).
                  //
                  // Note:
                  // If proj_delta_n_i < 0, (i=1,2) there is interpen from the velocity projection.
                  // Check this interpen against the maximum allowable to determine if a velocity projection
                  // timestep estimate is still required.
                  if ( dt1_vel_check && !dt1_check1 )  // continued interpen
                  {
                    dt1_vel_check = ( proj_delta_n_1 < 0. )
                                        ? ( ( std::abs( proj_delta_n_1 ) > max_delta1 ) ? true : false )
                                        : false;
                  }

                  if ( dt2_vel_check && !dt2_check1 )  // continued interpen
                  {
                    dt2_vel_check = ( proj_delta_n_2 < 0. )
                                        ? ( ( std::abs( proj_delta_n_2 ) > max_delta2 ) ? true : false )
                                        : false;
                  }

                  // compute velocity projection based dt (check 2) using a RESET gap (g=0) such that
                  // the velocity projected gap does not exceed the max allowable gap. This avoid timestep
                  // crashes for velocity projected gaps slightly in excess of the max allowable and still
                  // allows for a soft contact response without a timestep crash.
                  //
                  // v1_dot_n1 > 0 and v2_dot_n2 > 0 for further interpen
                  dt1 = ( dt1_vel_check ) ? alpha * max_delta1 / v1_dot_n1 : dt1;
                  dt2 = ( dt2_vel_check ) ? alpha * max_delta2 / v2_dot_n2 : dt2;

                  // Keep debug print statements. This routine is still in the testing phase
                  // std::cout << "dt1_vel_check, (proj_delta_n_1+max_delta1), v1_dot_n1: " << dt1_vel_check << ", "
                  //          << proj_delta_n_1+max_delta1 << ", " << v1_dot_n1 << std::endl;
                  // std::cout << "dt2_vel_check, (proj_delta_n_2+max_delta2), v2_dot_n2: " << dt2_vel_check << ", "
                  //          << proj_delta_n_2+max_delta2 << ", " << v2_dot_n2 << std::endl;
                  // std::cout << "dt1 and dt2: " << dt1 << ", " << dt2 << std::endl;

                  // update dt_temp2 only for positive dt1 and/or dt2
                  if ( dt1 > 0. ) {
#ifdef TRIBOL_USE_RAJA
                    RAJA::atomicMin<RAJA::auto_atomic>( &dt_temp[1], axom::utilities::min( dt1, 1.e6 ) );
#else
                    dt_temp[1] = axom::utilities::min(dt_temp[1], axom::utilities::min(dt1, 1.e6));
#endif
                  }
                  if ( dt2 > 0. ) {
#ifdef TRIBOL_USE_RAJA
                    RAJA::atomicMin<RAJA::auto_atomic>( &dt_temp[1], axom::utilities::min( 1.e6, dt2 ) );
#else
                    dt_temp[1] = axom::utilities::min(dt_temp[1], axom::utilities::min(1.e6, dt2));
#endif
                  }
                  if ( dt1 < 0. || dt2 < 0. ) {
#ifdef TRIBOL_USE_RAJA
                    RAJA::atomicMax<RAJA::auto_atomic>( &msg[3], static_cast<IndexT>( true ) );
#else
                    msg[3] = true;
#endif
                  }

                }  // end check 2
              } );

  // print general messages once
  // Can we output this message on root? SRW
  ArrayT<IndexT, 1, MemorySpace::Host> msg_host( msg_data );
  SLIC_DEBUG_IF( msg_host[0] || msg_host[1], "tribol::computeCommonPlaneTimeStep(): "
                                                 << "there are locations where mesh overlap may be too large. "
                                                 << "Cannot provide timestep vote. Reduce timestep and/or increase "
                                                 << "penalty." );

  SLIC_DEBUG_IF( msg_host[2], "tribol::computeCommonPlaneTimeStep():  "
                                  << "one or more face-pairs have a negative timestep vote based on "
                                  << "maximum gap check." );

  SLIC_DEBUG_IF( msg_host[3], "tribol::computeCommonPlaneTimeStep(): "
                                  << "one or more face-pairs have a negative timestep vote based on "
                                  << "velocity projection calculation." );

  ArrayT<RealT, 1, MemorySpace::Host> dt_temp_host( dt_temp_data );
  dt = axom::utilities::min( dt_temp_host[0], dt_temp_host[1] );
}

//------------------------------------------------------------------------------
void CouplingScheme::writeInterfaceOutput( const std::string& dir, const VisType v_type, const int cycle,
                                           const RealT t )
{
  int dim = this->spatialDimension();
  if ( m_parameters.vis_cycle_incr > 0 && !( cycle % m_parameters.vis_cycle_incr ) ) {
    switch ( m_contactMethod ) {
      case SINGLE_MORTAR:
      case ALIGNED_MORTAR:
      case MORTAR_WEIGHTS:
      case COMMON_PLANE:
        WriteContactPlaneMeshToVtk( dir, v_type, m_id, m_mesh_id1, m_mesh_id2, dim, cycle, t );
        break;
      default:
        // Can this be called on root? SRW
        SLIC_INFO(
            "CouplingScheme::writeInterfaceOutput(): " << "output routine not yet written for interface method. " );
        break;
    }  // end-switch
  }  // end-if
  return;
}

//------------------------------------------------------------------------------
void CouplingScheme::updatePairReportingData( const FaceGeomException face_exception )
{
  switch ( face_exception ) {
    case NO_FACE_GEOM_EXCEPTION: {
      // no-op
      break;
    }
    case FACE_ORIENTATION: {
      ++this->m_pairReportingData.numBadOrientation;
      break;
    }
    case INVALID_FACE_INPUT: {
      ++this->m_pairReportingData.numBadFaceGeometry;
      break;
    }
    case DEGENERATE_OVERLAP: {
      ++this->m_pairReportingData.numBadOverlaps;
      break;
    }
    case FACE_VERTEX_INDEX_EXCEEDS_OVERLAP_VERTICES: {
      // no-op; this is a very specific, in-the-weeds computational geometry
      // debug print and does not indicate an issue with the host-code mesh
      break;
    }
    default:
      break;
  }  // end switch
}

//------------------------------------------------------------------------------
void CouplingScheme::printPairReportingData()
{
  SLIC_DEBUG_IF( getInterfacePairs().size() > 0, this->getNumActivePairs() * 100. / getInterfacePairs().size()
                                                     << "% of binned interface pairs are active contact candidates." );

  SLIC_DEBUG_IF( this->m_pairReportingData.numBadOrientation > 0,
                 "Number of bad orientations is "
                     << this->m_pairReportingData.numBadOrientation << " equaling "
                     << this->m_pairReportingData.numBadOrientation * 100. / getInterfacePairs().size()
                     << "% of total number of binned interface pairs." );

  SLIC_DEBUG_IF( this->m_pairReportingData.numBadFaceGeometry > 0,
                 "Number of bad face geometries is "
                     << this->m_pairReportingData.numBadFaceGeometry << " equaling "
                     << this->m_pairReportingData.numBadFaceGeometry * 100. / getInterfacePairs().size()
                     << "% of total number of binned interface pairs." );

  SLIC_DEBUG_IF( this->m_pairReportingData.numBadOverlaps > 0,
                 "Number of bad contact overlaps is "
                     << this->m_pairReportingData.numBadOverlaps << " equaling "
                     << this->m_pairReportingData.numBadOverlaps * 100. / getInterfacePairs().size()
                     << "% of total number of binned interface pairs." );
}

//------------------------------------------------------------------------------
void CouplingScheme::enableEnzyme( [[maybe_unused]] bool useEnzyme )
{
#ifdef TRIBOL_USE_ENZYME
  this->m_useEnzyme = useEnzyme;
#else
  SLIC_WARNING( "CouplingScheme::enableEnzyme(): Tribol is not built with Enzyme. Continuing without Enzyme support." );
#endif
}

//------------------------------------------------------------------------------
void CouplingScheme::createNodalNormalJacobianData()
{
  m_dfdnJacobian = std::make_unique<MethodData>();
  m_dndxJacobian = std::make_unique<MethodData>();
}

//------------------------------------------------------------------------------
CouplingScheme::Viewer::Viewer( CouplingScheme& cs )
    : m_parameters( cs.m_parameters ),
      m_effective_binning_proximity_scale( cs.m_effective_binning_proximity_scale ),
      m_contact_mode( cs.m_contactMode ),
      m_contact_case( cs.m_contactCase ),
      m_contact_method( cs.m_contactMethod ),
      m_enforcement_options( cs.m_enforcementOptions ),
      m_mesh1( cs.getMesh1().getView() ),
      m_mesh2( cs.getMesh2().getView() ),
      m_cg_pairs( cs.m_cg_pairs )
{
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE ContactPlanePair& CouplingScheme::Viewer::getContactPlanePair( IndexT id ) const
{
  switch ( m_contact_method ) {
    case COMMON_PLANE: {
      return m_cg_pairs.getCommonPlane( id );
      break;
    }
    case MORTAR_WEIGHTS:
    case SINGLE_MORTAR: {
      return m_cg_pairs.getMortarPlane( id );
      break;
    }
    case ALIGNED_MORTAR: {
      return m_cg_pairs.getAlignedMortarPlane( id );
      break;
    }
    default: {
      // won't get here, but just return something to suppress warnings
      return m_cg_pairs.getMortarPlane( id );
      break;
    }
  }  // end switch
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE RealT CouplingScheme::Viewer::getGapTol( int fid1, int fid2 ) const
{
  RealT gap_tol = 0.;
  // add debug warning if this routine is called for interface methods
  // that do not require gap tolerances
  switch ( m_contact_method ) {
    case SINGLE_MORTAR:
#ifdef TRIBOL_USE_HOST
      SLIC_WARNING( "CouplingScheme::getGapTol(): 'SINGLE_MORTAR' "
                    << "method does not require use of a gap tolerance." );
#endif
      break;

    case ALIGNED_MORTAR:
#ifdef TRIBOL_USE_HOST
      SLIC_WARNING( "CouplingScheme::getGapTol(): 'ALIGNED_MORTAR' "
                    << "method does not require use of a gap tolerance." );
#endif
      break;

    case MORTAR_WEIGHTS:
#ifdef TRIBOL_USE_HOST
      SLIC_WARNING( "CouplingScheme::getGapTol(): 'MORTAR_WEIGHTS' "
                    << "method does not require use of a gap tolerance." );
#endif
      break;

    case COMMON_PLANE:

      switch ( m_contact_case ) {
        case TIED_NORMAL:
          gap_tol = m_parameters.binning_proximity_scale *
                    axom::utilities::max( m_mesh1.getFaceRadius()[fid1], m_mesh2.getFaceRadius()[fid2] );
          break;

        default:
          gap_tol = -1. * m_parameters.gap_tol_ratio *
                    axom::utilities::max( m_mesh1.getFaceRadius()[fid1], m_mesh2.getFaceRadius()[fid2] );
          break;

      }  // end switch over m_contactModel
      break;

    default:
      break;
  }  // end switch over m_contactMethod

  return gap_tol;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE bool CouplingScheme::Viewer::pruneMethodFacePair( const IndexT fid1, const IndexT fid2 ) const
{
  constexpr int max_dim = 3;
  constexpr int max_nodes_per_face = 4;

  auto& mesh1 = this->getMesh1View();
  auto& mesh2 = this->getMesh2View();
  int dim = mesh1.spatialDimension();
  int num_nodes_face_1 = mesh1.numberOfNodesPerElement();
  int num_nodes_face_2 = mesh2.numberOfNodesPerElement();

  RealT fn1[max_dim], cx1[max_dim];
  mesh1.getFaceNormal( fid1, fn1 );
  mesh1.getFaceCentroid( fid1, cx1 );

  RealT fn2[max_dim], cx2[max_dim];
  mesh2.getFaceNormal( fid2, fn2 );
  mesh2.getFaceCentroid( fid2, cx2 );

  // get each face's nodal coordinates
  RealT x1[max_nodes_per_face];
  RealT y1[max_nodes_per_face];
  RealT z1[max_nodes_per_face];

  RealT x2[max_nodes_per_face];
  RealT y2[max_nodes_per_face];
  RealT z2[max_nodes_per_face];

  for ( int i = 0; i < mesh1.numberOfNodesPerElement(); ++i ) {
    const int nodeId_1 = mesh1.getGlobalNodeId( fid1, i );
    x1[i] = mesh1.getPosition()[0][nodeId_1];
    y1[i] = mesh1.getPosition()[1][nodeId_1];
    if ( dim == 3 ) {
      z1[i] = mesh1.getPosition()[2][nodeId_1];
    }
  }

  for ( int i = 0; i < mesh2.numberOfNodesPerElement(); ++i ) {
    const int nodeId_2 = mesh2.getGlobalNodeId( fid2, i );
    x2[i] = mesh2.getPosition()[0][nodeId_2];
    y2[i] = mesh2.getPosition()[1][nodeId_2];
    if ( dim == 3 ) {
      z2[i] = mesh2.getPosition()[2][nodeId_2];
    }
  }

  RealT nrml[max_dim], cx[max_dim];

  switch ( m_contact_method ) {
    case ALIGNED_MORTAR:
    case MORTAR_WEIGHTS:
    case SINGLE_MORTAR: {
      // specify the point-normal data used for mortar contact plane methods
      // This is taken as the face 2 (nonmortar) normal and centroid
      for ( int i = 0; i < dim; ++i ) {
        nrml[i] = fn2[i];
        cx[i] = cx2[i];
      }

      if ( !IsOverlappingOnPlane( &x1[0], &y1[0], &z1[0], &x2[0], &y2[0], &z2[0], &nrml[0], &cx[0], num_nodes_face_1,
                                  num_nodes_face_2, dim ) ) {
        return true;
      }

      break;
    }
    case COMMON_PLANE: {
      // define the common plane
      for ( int i = 0; i < dim; ++i ) {
        nrml[i] = 0.5 * ( fn2[i] - fn1[i] );
        cx[i] = 0.5 * ( cx1[i] + cx2[i] );
      }

      // normalize the intermediate plane normal
      RealT mag;
      if ( dim == 3 ) {
        mag = magnitude( nrml[0], nrml[1], nrml[2] );
      } else {
        mag = magnitude( nrml[0], nrml[1], 0. );
      }
      RealT invMag = 1.0 / mag;

      for ( int i = 0; i < dim; ++i ) {
        nrml[i] *= invMag;
      }

      RealT x1_prime[max_nodes_per_face];
      RealT y1_prime[max_nodes_per_face];
      RealT z1_prime[max_nodes_per_face];
      RealT x2_prime[max_nodes_per_face];
      RealT y2_prime[max_nodes_per_face];
      RealT z2_prime[max_nodes_per_face];

      // project faces to average face planes
      if ( dim == 3 ) {
        ProjectFaceNodesToPlane( mesh1, fid1, fn1[0], fn1[1], fn1[2], cx1[0], cx1[1], cx1[2], &x1_prime[0],
                                 &y1_prime[0], &z1_prime[0] );
        ProjectFaceNodesToPlane( mesh2, fid2, fn2[0], fn2[1], fn2[2], cx2[0], cx2[1], cx2[2], &x2_prime[0],
                                 &y2_prime[0], &z2_prime[0] );
      } else {
        for ( int i = 0; i < num_nodes_face_1; ++i ) {
          x1_prime[i] = x1[i];
          y1_prime[i] = y1[i];
        }
        for ( int i = 0; i < num_nodes_face_2; ++i ) {
          x2_prime[i] = x2[i];
          y2_prime[i] = y2[i];
        }
      }

      if ( !IsOverlappingOnPlane( &x1_prime[0], &y1_prime[0], &z1_prime[0], &x2_prime[0], &y2_prime[0], &z2_prime[0],
                                  &nrml[0], &cx[0], num_nodes_face_1, num_nodes_face_2, dim ) ) {
        return true;
      }
      break;
    }
    default: {
#ifdef TRIBOL_USE_HOST
      SLIC_ERROR( "CouplingScheme::performMethodPruning(): your contact method does not have a pruning routine." );
#endif
      break;
    }
  }  // end switch

  return false;
}

} /* namespace tribol */
