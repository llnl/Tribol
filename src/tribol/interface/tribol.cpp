// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol.hpp"

// Tribol includes
#include "tribol/common/Parameters.hpp"
#include "tribol/types.hpp"

#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/CouplingSchemeManager.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/MeshManager.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

#include "tribol/geom/ContactPlane.hpp"
#include "tribol/geom/ContactPlaneManager.hpp"
#include "tribol/geom/GeomUtilities.hpp"

#include "tribol/physics/Physics.hpp"

#include "tribol/search/InterfacePairFinder.hpp"

#include "tribol/utils/Math.hpp"

// Axom includes
#include "axom/core.hpp"
#include "axom/slic.hpp"

// C/C++ includes
#include <string>
#include <unordered_map>
#include <fstream>

//------------------------------------------------------------------------------
// Interface Implementation
//------------------------------------------------------------------------------
namespace tribol {

namespace internal {

/*!
 * \brief Sets various default parameters
 */
void set_defaults()
{
  parameters_t& parameters = parameters_t::getInstance();

  //////////////////////////////
  // visualization parameters //
  //////////////////////////////
  parameters.vis_cycle_incr = 100;  // default contact output every 100 cycles
  parameters.vis_type = VIS_OVERLAPS;
  parameters.output_directory = "";

  ///////////////////////////////////////
  // computational geometry parameters //
  ///////////////////////////////////////
  parameters.overlap_area_frac = 1.E-8;  // note: API supports setting this
  parameters.gap_tol_ratio = 1.e-12;     // numerically "zero". Note: API doesn't support setting this
  parameters.gap_separation_ratio =
      0.75;                       // applied to the face-radius for geometry filtering. API doesn't support setting this
  parameters.gap_tied_tol = 0.1;  // tolerance for how much separation can occur before opposing faces are let go
  parameters.len_collapse_ratio = 1.E-8;
  parameters.projection_ratio = 1.E-10;
  parameters.contact_pen_frac = 3.e-1;  // allows for up to 30% penetration used in timestep vote calculation
}

}  // namespace internal

//------------------------------------------------------------------------------
void initialize( integer dimension, CommType comm )
{
  // sanity checks
  SLIC_ASSERT( ( dimension == 2 ) || ( dimension == 3 ) );
  SLIC_ASSERT( comm != TRIBOL_COMM_NULL );

  parameters_t& parameters = parameters_t::getInstance();
  parameters.dimension = dimension;
  parameters.problem_comm = comm;

  internal::set_defaults();
}

//------------------------------------------------------------------------------
void setPenaltyOptions( int couplingSchemeIndex, PenaltyConstraintType pen_enfrc_option,
                        KinematicPenaltyCalculation kinematic_calc, RatePenaltyCalculation rate_calc )
{
  CouplingSchemeManager& csManager = CouplingSchemeManager::getInstance();

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF( !csManager.hasCoupling( couplingSchemeIndex ),
                      "tribol::setPenaltyOptions(): call tribol::registerCouplingScheme() "
                          << "prior to calling this routine." );

  // get access to coupling scheme
  CouplingScheme* couplingScheme = csManager.getCoupling( couplingSchemeIndex );

  // get access to struct on coupling scheme holding penalty options
  EnforcementOptions& enforcement_options = couplingScheme->getEnforcementOptions();
  PenaltyEnforcementOptions& penalty_options = enforcement_options.penalty_options;

  // check that penalty enforcement option is valid
  if ( !in_range( pen_enfrc_option, NUM_PENALTY_OPTIONS ) ) {
    SLIC_WARNING_ROOT( "tribol::setPenaltyOptions(): penalty enforcement option not available." );
  } else {
    penalty_options.constraint_type = pen_enfrc_option;
    penalty_options.constraint_type_set = true;
  }

  // check that kinematic penalty calculation is valid
  if ( !in_range( kinematic_calc, NUM_KINEMATIC_PENALTY_CALCULATION ) ) {
    SLIC_WARNING_ROOT( "tribol::setPenaltyOptions(): kinematic penalty calculation not available." );
  } else {
    penalty_options.kinematic_calculation = kinematic_calc;
    penalty_options.kinematic_calc_set = true;
  }

  // check that the rate penalty calculation is valid
  if ( !in_range( rate_calc, NUM_RATE_PENALTY_CALCULATION ) ) {
    SLIC_WARNING_ROOT( "tribol::setPenaltyOptions(): rate penalty calculation not available." );
  } else {
    penalty_options.rate_calculation = rate_calc;
    penalty_options.rate_calc_set = true;
  }

}  // end setPenaltyOptions()

//------------------------------------------------------------------------------
void setKinematicConstantPenalty( int meshId, double k )
{
  // note, error checking done in the following registration routine
  registerRealElementField( meshId, KINEMATIC_CONSTANT_STIFFNESS, &k );

}  // end setKinematicConstantPenalty()

//------------------------------------------------------------------------------
void setKinematicElementPenalty( int meshId, const double* material_modulus, const double* element_thickness )
{
  // note, error checking done in the following registration routine
  registerRealElementField( meshId, BULK_MODULUS, material_modulus );
  registerRealElementField( meshId, ELEMENT_THICKNESS, element_thickness );

}  // end setKinematicElementPenalty()

//------------------------------------------------------------------------------
void setRateConstantPenalty( int meshId, double r_k )
{
  // note, error checking done in the following registration routine
  registerRealElementField( meshId, RATE_CONSTANT_STIFFNESS, &r_k );

}  // end setRateConstantPenalty()

//------------------------------------------------------------------------------
void setRatePercentPenalty( int meshId, double r_p )
{
  // note, error checking done in the following registration routine
  registerRealElementField( meshId, RATE_PERCENT_STIFFNESS, &r_p );

}  // end setRatePercentPenalty()

//------------------------------------------------------------------------------
void setContactPenFrac( double frac )
{
  parameters_t& parameters = parameters_t::getInstance();
  if ( frac <= 0. ) {
    // Don't set the contact_pen_frac. This will use default of 30%
    return;
  }

  parameters.contact_pen_frac = frac;

}  // end setContactPenFrac()

//------------------------------------------------------------------------------
void setContactAreaFrac( double frac )
{
  parameters_t& parameters = parameters_t::getInstance();
  if ( frac < 1.e-12 ) {
    SLIC_DEBUG_ROOT( "tribol::setContactAreaFrac(): area fraction too small or negative; "
                     << "setting to default 1.e-8." );
    frac = 1.e-8;
  }
  parameters.overlap_area_frac = frac;
}

//------------------------------------------------------------------------------
void setPenaltyScale( int meshId, double scale )
{
  MeshManager& meshManager = MeshManager::getInstance();

  SLIC_ERROR_ROOT_IF( !meshManager.hasMesh( meshId ), "tribol::setPenaltyScale(): "
                                                          << "no mesh with id, " << meshId << "exists." );

  MeshData& mesh = meshManager.GetMeshInstance( meshId );
  if ( scale > 1.e-6 ) {
    mesh.m_elemData.m_penalty_scale = scale;
  } else {
    // still set small penalty to allow for zeroing out kinematic penalty
    // enforcement allowing for rate only enforcement
    mesh.m_elemData.m_penalty_scale = scale;
    SLIC_WARNING_ROOT( "tribol::setPenaltyScale(): input scale factor is "
                       << "close to zero or negative; kinematic contact may "
                       << "not be properly enforced." );
  }

}  // end setPenaltyScale()

//------------------------------------------------------------------------------
void setLagrangeMultiplierOptions( int couplingSchemeIndex, ImplicitEvalMode evalMode, SparseMode sparseMode )
{
  // get access to coupling scheme
  CouplingSchemeManager& csManager = CouplingSchemeManager::getInstance();

  SLIC_ERROR_ROOT_IF( !csManager.hasCoupling( couplingSchemeIndex ),
                      "tribol::setLagrangeMultiplierOptions(): call tribol::registerCouplingScheme() "
                          << "prior to calling this routine." );

  CouplingScheme* couplingScheme = csManager.getCoupling( couplingSchemeIndex );

  // get access to struct on coupling scheme holding penalty options
  EnforcementOptions& enforcement_options = couplingScheme->getEnforcementOptions();
  LagrangeMultiplierImplicitOptions& lm_options = enforcement_options.lm_implicit_options;

  lm_options.eval_mode = evalMode;
  lm_options.sparse_mode = sparseMode;

#ifdef BUILD_REDECOMP

  if ( couplingScheme->hasMfemData() ) {
    // MFEM_ELEMENT_DENSE is required to use the MFEM interface
    lm_options.sparse_mode = SparseMode::MFEM_ELEMENT_DENSE;
    if ( !couplingScheme->hasMfemJacobianData() &&
         ( lm_options.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN ||
           lm_options.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ) ) {
      couplingScheme->setMfemJacobianData( std::make_unique<MfemJacobianData>( *couplingScheme->getMfemMeshData(),
                                                                               *couplingScheme->getMfemSubmeshData(),
                                                                               couplingScheme->getContactMethod() ) );
    }
  }

#endif /* BUILD_REDECOMP */

  lm_options.enforcement_option_set = true;

}  // end setLagrangeMultiplierOptions()

//------------------------------------------------------------------------------
void setPlotCycleIncrement( double incr )
{
  parameters_t& parameters = parameters_t::getInstance();
  parameters.vis_cycle_incr = incr;
}

//------------------------------------------------------------------------------
void setPlotOptions( enum VisType v_type )
{
  parameters_t& parameters = parameters_t::getInstance();
  parameters.vis_type = v_type;
}

//------------------------------------------------------------------------------
void setOutputDirectory( const std::string& dir )
{
  // Create path if it doesn't already exist
  if ( !axom::utilities::filesystem::pathExists( dir ) ) {
    SLIC_INFO_ROOT( "Creating output path '" << dir << "'" );
    axom::utilities::filesystem::makeDirsForPath( dir );
  }

  parameters_t& parameters = parameters_t::getInstance();
  parameters.output_directory = dir;
}

//------------------------------------------------------------------------------
void setLoggingLevel( int csId, LoggingLevel log_level )
{
  CouplingSchemeManager& csManager = CouplingSchemeManager::getInstance();
  SLIC_ERROR_IF( !csManager.hasCoupling( csId ), "tribol::setLoggingLevel(): "
                                                     << "invalid CouplingScheme id." );
  CouplingScheme* couplingScheme = csManager.getCoupling( csId );

  if ( !in_range( static_cast<int>( log_level ), static_cast<int>( tribol::NUM_LOGGING_LEVELS ) ) ) {
    SLIC_INFO_ROOT( "tribol::setLoggingLevel(): Logging level not an option; "
                    << "using 'warning' level." );
    couplingScheme->setLoggingLevel( tribol::TRIBOL_WARNING );
  } else {
    couplingScheme->setLoggingLevel( log_level );
  }
}
//------------------------------------------------------------------------------
void registerMesh( integer meshId, integer numCells, integer lengthNodalData, const IndexType* connectivity,
                   integer elementType, const real* x, const real* y, const real* z )
{
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& mesh = meshManager.CreateMesh( meshId );

  // check supported element types
  if ( static_cast<InterfaceElementType>( elementType ) != LINEAR_EDGE &&
       static_cast<InterfaceElementType>( elementType ) != LINEAR_TRIANGLE &&
       static_cast<InterfaceElementType>( elementType ) != LINEAR_QUAD ) {
    SLIC_WARNING_ROOT( "tribol::registerMesh(): mesh topology not supported "
                       << "for mesh id, " << meshId << "." );
    mesh.m_isValid = false;
  }

  const int dim = ( z == nullptr ) ? 2 : 3;

  // check for null pointers on ranks with non-null meshes
  if ( numCells > 0 ) {
    if ( x == nullptr || y == nullptr ) {
      SLIC_WARNING_ROOT( "tribol::registerMesh(): pointer to x or y-component "
                         << "mesh coordinate arrays are null pointers "
                         << " for mesh id, " << meshId << "." );
      mesh.m_isValid = false;
    }

    if ( dim == 3 ) {
      if ( z == nullptr ) {
        SLIC_WARNING_ROOT( "tribol::registerMesh(): pointer to z-component "
                           << "mesh coordinates is null for mesh id, " << meshId << "." );
        mesh.m_isValid = false;
      }
    }
  }

  // Setup mesh data; input argument pointers are allowed to be null
  // since Tribol supports null meshes. This is not uncommon in parallel
  // contact simulations
  mesh.m_meshId = meshId;
  mesh.m_dim = dim;
  mesh.m_positionX = x;
  mesh.m_positionY = y;
  mesh.m_positionZ = z;
  mesh.m_connectivity = connectivity;
  mesh.m_elementType = static_cast<InterfaceElementType>( elementType );
  mesh.m_lengthNodalData = lengthNodalData;
  mesh.m_numCells = numCells;

  // set the number of cells on the mesh element data struct
  mesh.m_elemData.m_numCells = numCells;

  // set the number of nodes on the mesh nodal data struct
  mesh.m_nodalFields.m_numNodes = lengthNodalData;

  // set the number of nodes per cell on the mesh.
  switch ( mesh.m_elementType ) {
    case tribol::LINEAR_EDGE: {
      mesh.m_numNodesPerCell = 2;
      break;
    }
    case tribol::LINEAR_TRIANGLE: {
      mesh.m_numNodesPerCell = 3;
      break;
    }
    case tribol::LINEAR_QUAD: {
      mesh.m_numNodesPerCell = 4;
      break;
    }
    default:
      SLIC_ERROR_ROOT( "Element type not supported." );
      break;
  }  // end switch over element type

  // compute the number of unique surface nodes from the connectivity
  // Note: this routine assigns mesh.m_numSurfaceNodes and allocates
  // space for m_sortedSurfaceNodeIds containing list of unique sorted
  // connectivity node ids in ascending order
  if ( mesh.m_numCells > 0 ) {
    mesh.sortSurfaceNodeIds();
  }

  if ( mesh.m_numCells > 0 ) {
    mesh.allocateArrays( dim );
    initRealArray( mesh.m_nX, mesh.m_numCells, 0. );
    initRealArray( mesh.m_nY, mesh.m_numCells, 0. );
    initRealArray( mesh.m_cX, mesh.m_numCells, 0. );
    initRealArray( mesh.m_cY, mesh.m_numCells, 0. );
    initRealArray( mesh.m_area, mesh.m_numCells, 0. );
  }

  if ( mesh.m_dim == 3 && mesh.m_numCells > 0 ) {
    initRealArray( mesh.m_nZ, mesh.m_numCells, 0. );
    initRealArray( mesh.m_cZ, mesh.m_numCells, 0. );
  }

}  // end of registerMesh()

//------------------------------------------------------------------------------
void registerNodalDisplacements( integer meshId, const real* dx, const real* dy, const real* dz )
{
  MeshManager& meshManager = MeshManager::getInstance();

  SLIC_ERROR_ROOT_IF( !meshManager.hasMesh( meshId ), "tribol::registerNodalDisplacements(): "
                                                          << "no mesh with id, " << meshId << "exists." );

  MeshData& mesh = meshManager.GetMeshInstance( meshId );
  mesh.m_nodalFields.m_is_nodal_displacement_set = true;

  if ( dx == nullptr || dy == nullptr ) {
    mesh.m_nodalFields.m_is_nodal_displacement_set = false;
  }

  if ( mesh.m_dim == 3 ) {
    if ( dz == nullptr ) {
      mesh.m_nodalFields.m_is_nodal_displacement_set = false;
    }
  }

  mesh.m_dispX = dx;
  mesh.m_dispY = dy;
  mesh.m_dispZ = dz;

}  // end registerNodalDisplacements()

//------------------------------------------------------------------------------
void registerNodalVelocities( integer meshId, const real* vx, const real* vy, const real* vz )
{
  MeshManager& meshManager = MeshManager::getInstance();

  SLIC_ERROR_ROOT_IF( !meshManager.hasMesh( meshId ), "tribol::registerNodalVelocities(): "
                                                          << "no mesh with id, " << meshId << "exists." );

  MeshData& mesh = meshManager.GetMeshInstance( meshId );
  mesh.m_nodalFields.m_is_velocity_set = true;

  if ( vx == nullptr || vy == nullptr ) {
    mesh.m_nodalFields.m_is_velocity_set = false;
  }

  if ( mesh.m_dim == 3 ) {
    if ( vz == nullptr ) {
      mesh.m_nodalFields.m_is_velocity_set = false;
    }
  }

  mesh.m_velX = vx;
  mesh.m_velY = vy;
  mesh.m_velZ = vz;

}  // end registerNodalVelocities()

//------------------------------------------------------------------------------
void registerNodalResponse( integer meshId, real* rx, real* ry, real* rz )
{
  MeshManager& meshManager = MeshManager::getInstance();

  SLIC_ERROR_ROOT_IF( !meshManager.hasMesh( meshId ), "tribol::registerNodalResponse(): "
                                                          << "no mesh with id, " << meshId << "exists." );

  MeshData& mesh = meshManager.GetMeshInstance( meshId );
  mesh.m_nodalFields.m_is_nodal_response_set = true;
  if ( rx == nullptr || ry == nullptr ) {
    mesh.m_nodalFields.m_is_nodal_response_set = false;
  }

  if ( mesh.m_dim == 3 ) {
    if ( rz == nullptr ) {
      mesh.m_nodalFields.m_is_nodal_response_set = false;
    }
  }

  mesh.m_forceX = rx;
  mesh.m_forceY = ry;
  mesh.m_forceZ = rz;

}  // end registerNodalResponse()

//------------------------------------------------------------------------------
int getJacobianSparseMatrix( mfem::SparseMatrix** sMat, int csId )
{
  // note, SLIC_ERROR_ROOT_IF is not used here because it's possible not all ranks
  // will have method (i.e. mortar) data.
  SLIC_ERROR_IF( *sMat != nullptr, "tribol::getMfemSparseMatrix(): "
                                       << "sparse matrix pointer not null." );

  CouplingSchemeManager& csManager = CouplingSchemeManager::getInstance();

  SLIC_ERROR_IF( !csManager.hasCoupling( csId ), "tribol::getMfemSparseMatrix(): "
                                                     << "invalid CouplingScheme id." );

  CouplingScheme* couplingScheme = csManager.getCoupling( csId );

  switch ( couplingScheme->getContactMethod() ) {
    case MORTAR_WEIGHTS:
    case ALIGNED_MORTAR:
    case SINGLE_MORTAR: {
      *sMat = static_cast<MortarData*>( couplingScheme->getMethodData() )->getMfemSparseMatrix();
      return 0;
    }
    default: {
      SLIC_WARNING( "tribol::getMfemSparseMatrix(): interface method does not return matrix data." );
      return 1;
    }
  }
}  // end getMfemSparseMatrix()

//------------------------------------------------------------------------------
int getJacobianCSRMatrix( int** I, int** J, real** vals, int csId, int* n_offsets, int* n_nonzero )
{
  // check to make sure input pointers are null
  if ( *I != nullptr || *J != nullptr || *vals != nullptr ) {
    SLIC_WARNING( "tribol::getJacobianCSRMatrix(): input pointers must be null." );
    return 1;
  }

  CouplingSchemeManager& csManager = CouplingSchemeManager::getInstance();

  // Note, SLIC_<>_ROOT macros are not here because it's possible not all ranks will have
  // method data.
  SLIC_ERROR_IF( !csManager.hasCoupling( csId ), "tribol::getJacobianCSRMatrix(): invalid "
                                                     << "CouplingScheme id." );

  CouplingScheme* couplingScheme = csManager.getCoupling( csId );

  switch ( couplingScheme->getContactMethod() ) {
    case ALIGNED_MORTAR: {
      SLIC_WARNING( "tribol::getJacobianCSRMatrix(): CSR format not currently implemented with "
                    << "ALIGNED_MORTAR. Use MFEM sparse matrix registration." );
      return 1;
    }
    case MORTAR_WEIGHTS: {
      static_cast<MortarData*>( couplingScheme->getMethodData() )->getCSRArrays( I, J, vals, n_offsets, n_nonzero );
      return 0;
    }
    case SINGLE_MORTAR: {
      SLIC_WARNING(
          "tribol::getJacobianCSRMatrix(): CSR format not currently implemented with "
          "SINGLE_MORTAR. Use MFEM sparse matrix registration." );
      return 1;
    }
    default: {
      SLIC_WARNING( "tribol::getJacobianCSRMatrix(): method does not return matrix data; "
                    << "invalid call." );
      return 1;
    }
  }
}  // end getCSRMatrix()

//------------------------------------------------------------------------------
int getElementBlockJacobians( integer csId, BlockSpace row_block, BlockSpace col_block,
                              const axom::Array<integer>** row_elem_idx, const axom::Array<integer>** col_elem_idx,
                              const axom::Array<mfem::DenseMatrix>** jacobians )
{
  SparseMode sparse_mode =
      CouplingSchemeManager::getInstance().getCoupling( csId )->getEnforcementOptions().lm_implicit_options.sparse_mode;
  if ( sparse_mode != SparseMode::MFEM_ELEMENT_DENSE ) {
    SLIC_WARNING(
        "Jacobian is assembled and can be accessed by "
        "getMfemSparseMatrix() or getCSRMatrix(). For (unassembled) element "
        "Jacobian contributions, call setLagrangeMultiplierOptions() with "
        "SparseMode::MFEM_ELEMENT_DENSE before calling update()." );
    return 1;
  }
  MethodData* method_data = CouplingSchemeManager::getInstance().getCoupling( csId )->getMethodData();
  *row_elem_idx = &method_data->getBlockJElementIds()[static_cast<int>( row_block )];
  *col_elem_idx = &method_data->getBlockJElementIds()[static_cast<int>( col_block )];
  *jacobians = &method_data->getBlockJ()( static_cast<int>( row_block ), static_cast<int>( col_block ) );
  return 0;
}

//------------------------------------------------------------------------------
void registerMortarGaps( integer meshId, real* gaps )
{
  MeshManager& meshManager = MeshManager::getInstance();

  SLIC_ERROR_ROOT_IF( !meshManager.hasMesh( meshId ), "tribol::registerMortarGaps(): "
                                                          << "no mesh with id " << meshId << " exists." );

  MeshData& mesh = meshManager.GetMeshInstance( meshId );

  if ( gaps == nullptr && mesh.m_numCells > 0 ) {
    SLIC_WARNING( "tribol::registerMortarGaps(): null pointer to gap data "
                  << "on non-null mesh " << meshId << "." );
    mesh.m_isValid = false;
  } else {
    mesh.m_nodalFields.m_node_gap = gaps;
    mesh.m_nodalFields.m_is_node_gap_set = true;
  }
}

//------------------------------------------------------------------------------
void registerMortarPressures( integer meshId, const real* pressures )
{
  MeshManager& meshManager = MeshManager::getInstance();

  SLIC_ERROR_ROOT_IF( !meshManager.hasMesh( meshId ), "tribol::registerMortarPressures(): "
                                                          << "no mesh with id " << meshId << " exists." );

  MeshData& mesh = meshManager.GetMeshInstance( meshId );

  if ( pressures == nullptr && mesh.m_numCells > 0 ) {
    SLIC_WARNING( "tribol::registerMortarPressures(): null pointer to pressure data "
                  << "on non-null mesh " << meshId << "." );
    mesh.m_isValid = false;
  } else {
    mesh.m_nodalFields.m_node_pressure = pressures;
    mesh.m_nodalFields.m_is_node_pressure_set = true;
  }
}

//------------------------------------------------------------------------------
void registerIntNodalField( integer meshId, const IntNodalFields field, integer* TRIBOL_UNUSED_PARAM( fieldVariable ) )
{
  MeshManager& meshManager = MeshManager::getInstance();

  SLIC_ERROR_ROOT_IF( !meshManager.hasMesh( meshId ), "tribol::registerIntNodalField(): "
                                                          << "no mesh with id " << meshId << " exists." );

  switch ( field ) {
    case UNDEFINED_INT_NODAL_FIELD:
    default:
      SLIC_ERROR_ROOT( "tribol::registerIntNodalField() not yet implemented." );
  }  // end switch over field

}  // end registerIntNodalField()

//------------------------------------------------------------------------------
void registerRealElementField( integer meshId, const RealElementFields field, const real* fieldVariable )
{
  MeshManager& meshManager = MeshManager::getInstance();

  SLIC_ERROR_IF( !meshManager.hasMesh( meshId ), "tribol::registerRealElementField(): "
                                                     << "no mesh with id " << meshId << " exists." );

  MeshData& mesh = meshManager.GetMeshInstance( meshId );

  switch ( field ) {
    case KINEMATIC_CONSTANT_STIFFNESS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh.m_numCells > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for "
                      << "'KINEMATIC_CONSTANT_STIFFNESS' on mesh " << meshId << "." );
          mesh.m_elemData.m_is_kinematic_constant_penalty_set = false;
        } else {
          mesh.m_elemData.m_is_kinematic_constant_penalty_set = true;
        }
      } else {
        mesh.m_elemData.m_penalty_stiffness = *fieldVariable;
        mesh.m_elemData.m_is_kinematic_constant_penalty_set = true;
      }
      break;
    }
    case RATE_CONSTANT_STIFFNESS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh.m_numCells > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for "
                      << "'RATE_CONSTANT_STIFFNESS' on mesh " << meshId << "." );
          mesh.m_elemData.m_is_rate_constant_penalty_set = false;
        } else {
          mesh.m_elemData.m_is_rate_constant_penalty_set = true;
        }
      } else {
        mesh.m_elemData.m_rate_penalty_stiffness = *fieldVariable;
        mesh.m_elemData.m_is_rate_constant_penalty_set = true;
      }
      break;
    }
    case RATE_PERCENT_STIFFNESS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh.m_numCells > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for "
                      << "'RATE_PERCENT_STIFFNESS' on mesh " << meshId << "." );
          mesh.m_elemData.m_is_rate_percent_penalty_set = false;
        } else {
          mesh.m_elemData.m_is_rate_percent_penalty_set = true;
        }
      } else {
        mesh.m_elemData.m_rate_percent_stiffness = *fieldVariable;
        mesh.m_elemData.m_is_rate_percent_penalty_set = true;
      }
      break;
    }
    case BULK_MODULUS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh.m_numCells > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for "
                      << "'BULK_MODULUS' on mesh " << meshId << "." );
          mesh.m_elemData.m_is_kinematic_element_penalty_set = false;
        } else {
          // set boolean to true for zero element meshes (acceptable registration)
          mesh.m_elemData.m_is_kinematic_element_penalty_set = true;
        }
      } else {
        mesh.m_elemData.m_mat_mod = fieldVariable;

        // Only set boolean to true if the element thickness has been registered
        // for nonzero element meshes. This will be true if the element thickness
        // was registered first (need both).
        if ( mesh.m_elemData.m_thickness != nullptr ) {
          mesh.m_elemData.m_is_kinematic_element_penalty_set = true;
        }
      }

      break;
    }
    case YOUNGS_MODULUS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh.m_numCells > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for "
                      << "'YOUNGS_MODULUS' on mesh " << meshId << "." );
          mesh.m_elemData.m_is_kinematic_element_penalty_set = false;
        } else {
          // set boolean to true for zero element meshes (acceptable registration)
          mesh.m_elemData.m_is_kinematic_element_penalty_set = true;
        }
      } else {
        mesh.m_elemData.m_mat_mod = fieldVariable;

        // Only set boolean to true if the element thickness has been registered
        // for nonzero element meshes. This will be true if the element thickness
        // was registered first (need both).
        if ( mesh.m_elemData.m_thickness != nullptr ) {
          mesh.m_elemData.m_is_kinematic_element_penalty_set = true;
        }
      }

      break;
    }
    case ELEMENT_THICKNESS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh.m_numCells > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for "
                      << "'ELEMENT_THICKNESS' on mesh " << meshId << "." );
          mesh.m_elemData.m_is_kinematic_element_penalty_set = false;
        } else {
          // set boolean to true for zero element meshes (acceptable registration)
          mesh.m_elemData.m_is_kinematic_element_penalty_set = true;
        }
      } else {
        mesh.m_elemData.m_thickness = fieldVariable;

        // Only set boolean to true if the material modulus has been registered for
        // nonzero element meshes. This will set to true if the material modulus was
        // registered first (need both).
        if ( mesh.m_elemData.m_mat_mod != nullptr ) {
          mesh.m_elemData.m_is_kinematic_element_penalty_set = true;
        }
      }

      break;
    }
    default: {
      SLIC_ERROR( "tribol::registerRealElementField(): the field argument "
                  << "on mesh " << meshId << " is not an accepted tribol real element field." );
    }
  }  // end switch over field

}  // end registerRealElementField()

//------------------------------------------------------------------------------
void registerIntElementField( integer meshId, const IntElementFields field,
                              integer* TRIBOL_UNUSED_PARAM( fieldVariable ) )
{
  MeshManager& meshManager = MeshManager::getInstance();

  SLIC_ERROR_ROOT_IF( !meshManager.hasMesh( meshId ), "tribol::registerIntElementField(): "
                                                          << "no mesh with id " << meshId << " exists." );

  switch ( field ) {
    case UNDEFINED_INT_ELEMENT_FIELD:
    default:
      SLIC_ERROR_ROOT( "tribol::registerIntElementField() not yet implemented." );
  }  // end switch over field

}  // end registerIntElementField()

//------------------------------------------------------------------------------
void registerCouplingScheme( integer couplingSchemeIndex, integer meshId1, integer meshId2, integer contact_mode,
                             integer contact_case, integer contact_method, integer contact_model,
                             integer enforcement_method, integer binning_method )
{
  // add coupling scheme. Checks for valid schemes will be performed later
  CouplingSchemeManager& couplingSchemeManager = CouplingSchemeManager::getInstance();

  CouplingScheme* scheme = new CouplingScheme( couplingSchemeIndex, meshId1, meshId2, contact_mode, contact_case,
                                               contact_method, contact_model, enforcement_method, binning_method );

  // add coupling scheme to manager. Validity checks are performed in
  // tribol::update() when each coupling scheme is initialized.
  couplingSchemeManager.addCoupling( couplingSchemeIndex, scheme );

}  // end registerCouplingScheme()

//------------------------------------------------------------------------------
void setInterfacePairs( integer couplingSchemeIndex, IndexType numPairs, IndexType const* const meshId1,
                        IndexType const* const pairType1, IndexType const* const pairIndex1,
                        IndexType const* const meshId2, IndexType const* const pairType2,
                        IndexType const* const pairIndex2 )
{
  CouplingSchemeManager& csManager = CouplingSchemeManager::getInstance();

  SLIC_ERROR_ROOT_IF( !csManager.hasCoupling( couplingSchemeIndex ),
                      "tribol::setInterfacePairs(): invalid coupling scheme index." );

  auto* couplingScheme = csManager.getCoupling( couplingSchemeIndex );
  auto* pairs = couplingScheme->getInterfacePairs();

  pairs->clear();

  // copy the interaction pairs
  for ( int i = 0; i < numPairs; ++i ) {
    InterfacePair pair{ meshId1[i], pairType1[i], pairIndex1[i], meshId2[i], pairType2[i], pairIndex2[i], i };
    ContactMode mode = couplingScheme->getContactMode();

    // perform initial face-pair validity checks to add valid face-pairs
    // to interface pair manager. Note, further computational geometry
    // filtering will be performed on each face-pair indendifying
    // contact candidates.
    bool check = geomFilter( pair, mode );

    if ( check ) {
      pair.isContactCandidate = true;
      pairs->addInterfacePair( pair );
    } else {
      pair.isContactCandidate = false;
    }
  }

  // Disable per-cycle rebinning
  couplingScheme->setFixedBinning( true );
}

//------------------------------------------------------------------------------
integer update( integer cycle, real t, real& dt )
{
  CouplingSchemeManager& csManager = CouplingSchemeManager::getInstance();
  int numCouplings = csManager.getNumberOfCouplings();
  bool err_cs = false;

  /////////////////////////////////////////////////////////////////////////
  //                                                                     //
  // Loop over coupling schemes.                                         //
  //                                                                     //
  // Note, numCouplings is always 1 greater than the highest registered  //
  // coupling index. This allows for non-contiguous coupling scheme ids, //
  // which may arise from host-code registration or from skipped schemes //
  //                                                                     //
  /////////////////////////////////////////////////////////////////////////
  for ( int csIndex = 0; csIndex < numCouplings; ++csIndex ) {
    if ( !csManager.hasCoupling( csIndex ) ) {
      continue;
    }

    CouplingScheme* couplingScheme = csManager.getCoupling( csIndex );

    // initialize and check for valid coupling scheme. If not valid, the coupling
    // scheme will not be valid across all ranks and we will skip this coupling scheme
    if ( !couplingScheme->init() ) {
      SLIC_WARNING_ROOT( "tribol::update(): skipping invalid CouplingScheme " << couplingScheme->getId()
                                                                              << "Please see warnings." );
      continue;
    }

    // perform binning between meshes on the coupling scheme
    // Note, this routine is guarded against null meshes
    couplingScheme->performBinning();

    // apply the coupling scheme. Note, there are appropriate guards against zero
    // element meshes, or null-mesh coupling schemes
    err_cs = couplingScheme->apply( cycle, t, dt );

    if ( err_cs != 0 ) {
      SLIC_WARNING( "tribol::update(): coupling scheme " << csIndex << " returned with an error." );
    }

  }  // end of coupling scheme loop

  return err_cs;

}  // end update()

//------------------------------------------------------------------------------
void finalize()
{
  CouplingSchemeManager& csManager = CouplingSchemeManager::getInstance();
  int numCouplings = csManager.getNumberOfCouplings();
  if ( numCouplings > 0 ) {
    csManager.clearAllCouplings();
  }
}

//------------------------------------------------------------------------------
}  // namespace tribol
