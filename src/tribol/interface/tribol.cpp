// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol.hpp"

// Tribol includes
#include "tribol/common/Parameters.hpp"

#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

// #include "tribol/geom/CompGeom.hpp"
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

//------------------------------------------------------------------------------
void initialize( int, CommT )
{
  SLIC_WARNING_ROOT(
      "Initialization of Tribol is no longer needed. Dimension\n"
      "is set by registered meshes and MPI communicator is stored on the\n"
      "coupling scheme (see setMPIComm())." );
}

//------------------------------------------------------------------------------
void setMPIComm( IndexT cs_id, CommT comm )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF(
      !cs, "tribol::setMPIComm(): call tribol::registerCouplingScheme() " << "prior to calling this routine." );

  cs->setMPIComm( comm );
}

//------------------------------------------------------------------------------
void setPenaltyOptions( IndexT cs_id, PenaltyConstraintType pen_enfrc_option,
                        KinematicPenaltyCalculation kinematic_calc, RatePenaltyCalculation rate_calc )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF(
      !cs, "tribol::setPenaltyOptions(): call tribol::registerCouplingScheme() " << "prior to calling this routine." );

  // get access to struct on coupling scheme holding penalty options
  EnforcementOptions& enforcement_options = cs->getEnforcementOptions();
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
void setKinematicConstantPenalty( IndexT mesh_id, RealT k )
{
  // note, error checking done in the following registration routine
  registerRealElementField( mesh_id, KINEMATIC_CONSTANT_STIFFNESS, &k );

}  // end setKinematicConstantPenalty()

//------------------------------------------------------------------------------
void setKinematicElementPenalty( IndexT mesh_id, const RealT* material_modulus, const RealT* element_thickness )
{
  // note, error checking done in the following registration routine
  registerRealElementField( mesh_id, BULK_MODULUS, material_modulus );
  registerRealElementField( mesh_id, ELEMENT_THICKNESS, element_thickness );

}  // end setKinematicElementPenalty()

//------------------------------------------------------------------------------
void setRateConstantPenalty( IndexT mesh_id, RealT r_k )
{
  // note, error checking done in the following registration routine
  registerRealElementField( mesh_id, RATE_CONSTANT_STIFFNESS, &r_k );

}  // end setRateConstantPenalty()

//------------------------------------------------------------------------------
void setRatePercentPenalty( IndexT mesh_id, RealT r_p )
{
  // note, error checking done in the following registration routine
  registerRealElementField( mesh_id, RATE_PERCENT_STIFFNESS, &r_p );

}  // end setRatePercentPenalty()

//------------------------------------------------------------------------------
void setViscousDampingCoeff( IndexT mesh_id, RealT coeff )
{
  registerRealElementField( mesh_id, VISCOUS_DAMPING_COEFF, &coeff );
}

//------------------------------------------------------------------------------
void setAutoContactPenScale( IndexT cs_id, RealT scale )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF( !cs, "tribol::setAutoContactPenScale(): call tribol::registerCouplingScheme() "
                               << "prior to calling this routine." );

  // check for strict positivity of the input parameter
  SLIC_WARNING_ROOT_IF( scale < 0., "tribol::setAutoContactPenScale(): "
                                        << "input for the auto-contact length scale factor must be positive." );

  cs->getParameters().auto_contact_pen_frac = scale;

}  // end setAutoContactPenScale()

//------------------------------------------------------------------------------
void setTimestepPenFrac( IndexT cs_id, RealT frac )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF(
      !cs, "tribol::setTimestepPenFrac(): call tribol::registerCouplingScheme() " << "prior to calling this routine." );

  if ( frac <= 0. ) {
    // Don't set the timestep_pen_frac. This will use default
    return;
  }

  cs->getParameters().timestep_pen_frac = frac;

}  // end setTimestepPenFrac()

//------------------------------------------------------------------------------
void setTimestepScale( IndexT cs_id, RealT scale )
{
  if ( scale <= 0. ) {
    // Don't set the timestep_scale.  This will use default
    return;
  }

  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF(
      !cs, "tribol::setTimestepScale(): call tribol::registerCouplingScheme() " << "prior to calling this routine." );

  cs->getParameters().timestep_scale = scale;
}
//------------------------------------------------------------------------------
void setContactAreaFrac( IndexT cs_id, RealT frac )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF(
      !cs, "tribol::setContactAreaFrac(): call tribol::registerCouplingScheme() " << "prior to calling this routine." );

  if ( frac <= 0.0 ) {
    SLIC_DEBUG_ROOT( "tribol::setContactAreaFrac(): area fraction <= 0.0; " << "setting to default 1.e-8." );
    frac = 1.e-8;
  }
  cs->getParameters().overlap_area_frac = frac;

}  // end setPenaltyScale()

//------------------------------------------------------------------------------
void setPenaltyScale( IndexT mesh_id, RealT scale )
{
  auto mesh = MeshManager::getInstance().findData( mesh_id );

  SLIC_ERROR_ROOT_IF( !mesh, "tribol::setPenaltyScale(): " << "no mesh with id, " << mesh_id << "exists." );

  if ( scale > 1.e-6 ) {
    mesh->getElementData().m_penalty_scale = scale;
  } else {
    // still set small penalty to allow for zeroing out kinematic penalty
    // enforcement allowing for rate only enforcement
    mesh->getElementData().m_penalty_scale = scale;
    SLIC_WARNING_ROOT( "tribol::setPenaltyScale(): input scale factor is "
                       << "close to zero or negative; kinematic contact may "
                       << "not be properly enforced." );
  }

}  // end setPenaltyScale()

//------------------------------------------------------------------------------
void setLagrangeMultiplierOptions( IndexT cs_id, ImplicitEvalMode evalMode, SparseMode sparseMode )
{
  // get access to coupling scheme
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  SLIC_ERROR_ROOT_IF( !cs, "tribol::setLagrangeMultiplierOptions(): call tribol::registerCouplingScheme() "
                               << "prior to calling this routine." );

  // get access to struct on coupling scheme holding penalty options
  EnforcementOptions& enforcement_options = cs->getEnforcementOptions();
  LagrangeMultiplierImplicitOptions& lm_options = enforcement_options.lm_implicit_options;

  lm_options.eval_mode = evalMode;
  lm_options.sparse_mode = sparseMode;

#ifdef BUILD_REDECOMP

  if ( cs->hasMfemData() ) {
    // MFEM_ELEMENT_DENSE is required to use the MFEM interface
    lm_options.sparse_mode = SparseMode::MFEM_ELEMENT_DENSE;
    if ( !cs->hasMfemJacobianData() && ( lm_options.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN ||
                                         lm_options.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ) ) {
      cs->setMfemJacobianData( std::make_unique<MfemJacobianData>( *cs->getMfemMeshData(), *cs->getMfemSubmeshData(),
                                                                   cs->getContactMethod() ) );
    }
  }

#endif /* BUILD_REDECOMP */

  lm_options.enforcement_option_set = true;

}  // end setLagrangeMultiplierOptions()

//------------------------------------------------------------------------------
void setPlotCycleIncrement( IndexT cs_id, int incr )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF( !cs, "tribol::setPlotCycleIncrement(): call tribol::registerCouplingScheme() "
                               << "prior to calling this routine." );

  cs->getParameters().vis_cycle_incr = incr;

}  // end setPlotCycleIncrement()

//------------------------------------------------------------------------------
void setPlotOptions( IndexT cs_id, enum VisType v_type )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF(
      !cs, "tribol::setPlotOptions(): call tribol::registerCouplingScheme() " << "prior to calling this routine." );

  cs->getParameters().vis_type = v_type;

}  // end setPlotOptions()

//------------------------------------------------------------------------------
void setOutputDirectory( IndexT cs_id, const std::string& dir )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF(
      !cs, "tribol::setOutputDirectory(): call tribol::registerCouplingScheme() " << "prior to calling this routine." );

  // Create path if it doesn't already exist
  if ( !axom::utilities::filesystem::pathExists( dir ) ) {
    SLIC_INFO_ROOT( "Creating output path '" << dir << "'" );
    axom::utilities::filesystem::makeDirsForPath( dir );
  }

  cs->setOutputDirectory( dir );

}  // end setOutputDirectory()

//------------------------------------------------------------------------------
void setLoggingLevel( IndexT cs_id, LoggingLevel log_level )
{
  // get access to coupling scheme
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  SLIC_ERROR_IF( !cs, "tribol::setLoggingLevel(): " << "invalid CouplingScheme id." );

  if ( !in_range( static_cast<int>( log_level ), static_cast<int>( LoggingLevel::NUM_LOGGING_LEVELS ) ) ) {
    SLIC_INFO_ROOT( "tribol::setLoggingLevel(): Logging level not an option; " << "using 'warning' level." );
    cs->setLoggingLevel( LoggingLevel::WARNING );
  } else {
    cs->setLoggingLevel( log_level );
  }

}  // end setLoggingLevel()

//------------------------------------------------------------------------------
void setBinningProximityScale( IndexT cs_id, RealT binning_proximity_scale )
{
  // get access to coupling scheme
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  SLIC_ERROR_ROOT_IF( !cs, "tribol::setBinningProximityScale(): call tribol::registerCouplingScheme() "
                               << "prior to calling this routine." );
  if ( binning_proximity_scale < 2.0 ) {
    binning_proximity_scale = 2.0;
    SLIC_WARNING_ROOT(
        "Setting binning proximity to less than 2.0 can lead to missed contact pairs.  Resetting to 2.0." );
  }

  cs->getParameters().binning_proximity_scale = binning_proximity_scale;

}  // end setBinningProximityScale()

//------------------------------------------------------------------------------
void enableTimestepVote( IndexT cs_id, const bool enable )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF(
      !cs, "tribol::enableTimestepVote(): call tribol::registerCouplingScheme() " << "prior to calling this routine." );

  cs->getParameters().enable_timestep_vote = enable;

}  // end enableTimestepVote()

//------------------------------------------------------------------------------
void enableEnzyme( IndexT cs_id, [[maybe_unused]] bool use_enzyme )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // check to see if coupling scheme exists
  SLIC_ERROR_ROOT_IF(
      !cs, "tribol::enableEnzyme(): call tribol::registerCouplingScheme() " << "prior to calling this routine." );

#ifdef TRIBOL_USE_ENZYME
  cs->enableEnzyme( use_enzyme );
#else
  SLIC_WARNING_ROOT( "tribol::enableEnzyme(): Tribol not built with Enzyme support." );
#endif

}  // end enableEnzyme()

//------------------------------------------------------------------------------
void registerMesh( IndexT mesh_id, IndexT num_elements, IndexT num_nodes, const IndexT* connectivity, int element_type,
                   const RealT* x, const RealT* y, const RealT* z, MemorySpace mem_space )
{
  MeshManager::getInstance().addData(
      mesh_id, MeshData( mesh_id, num_elements, num_nodes, connectivity,
                         static_cast<InterfaceElementType>( element_type ), x, y, z, mem_space ) );
}  // end registerMesh()

//------------------------------------------------------------------------------
void registerNodalDisplacements( IndexT mesh_id, const RealT* dx, const RealT* dy, const RealT* dz )
{
  auto mesh = MeshManager::getInstance().findData( mesh_id );

  SLIC_ERROR_ROOT_IF( !mesh, "tribol::registerNodalDisplacements(): " << "no mesh with id, " << mesh_id << "exists." );

  mesh->getNodalFields().m_is_nodal_displacement_set = true;

  if ( dx == nullptr || dy == nullptr ) {
    mesh->getNodalFields().m_is_nodal_displacement_set = false;
  }

  if ( mesh->spatialDimension() == 3 ) {
    if ( dz == nullptr ) {
      mesh->getNodalFields().m_is_nodal_displacement_set = false;
    }
  }

  mesh->setDisplacement( dx, dy, dz );

}  // end registerNodalDisplacements()

//------------------------------------------------------------------------------
void registerNodalVelocities( IndexT mesh_id, const RealT* vx, const RealT* vy, const RealT* vz )
{
  auto mesh = MeshManager::getInstance().findData( mesh_id );

  SLIC_ERROR_ROOT_IF( !mesh, "tribol::registerNodalVelocities(): " << "no mesh with id, " << mesh_id << "exists." );

  mesh->getNodalFields().m_is_velocity_set = true;

  if ( vx == nullptr || vy == nullptr ) {
    mesh->getNodalFields().m_is_velocity_set = false;
  }

  if ( mesh->spatialDimension() == 3 ) {
    if ( vz == nullptr ) {
      mesh->getNodalFields().m_is_velocity_set = false;
    }
  }

  mesh->setVelocity( vx, vy, vz );

}  // end registerNodalVelocities()

//------------------------------------------------------------------------------
void registerNodalReferenceCoords( IndexT mesh_id, const RealT* xref, const RealT* yref, const RealT* zref )
{
  auto mesh = MeshManager::getInstance().findData( mesh_id );

  SLIC_ERROR_ROOT_IF( !mesh, "tribol::registerNodalVelocities(): " << "no mesh with id, " << mesh_id << "exists." );

  mesh->setReferencePosition( xref, yref, zref );

}  // end registerNodalReferenceCoords()

//------------------------------------------------------------------------------
void registerNodalResponse( IndexT mesh_id, RealT* rx, RealT* ry, RealT* rz )
{
  auto mesh = MeshManager::getInstance().findData( mesh_id );

  SLIC_ERROR_ROOT_IF( !mesh, "tribol::registerNodalResponse(): " << "no mesh with id, " << mesh_id << "exists." );

  mesh->getNodalFields().m_is_nodal_response_set = true;
  if ( rx == nullptr || ry == nullptr ) {
    mesh->getNodalFields().m_is_nodal_response_set = false;
  }

  if ( mesh->spatialDimension() == 3 ) {
    if ( rz == nullptr ) {
      mesh->getNodalFields().m_is_nodal_response_set = false;
    }
  }

  mesh->setResponse( rx, ry, rz );

}  // end registerNodalResponse()

//------------------------------------------------------------------------------
int getJacobianSparseMatrix( mfem::SparseMatrix** sMat, IndexT cs_id )
{
  // note, SLIC_ERROR_ROOT_IF is not used here because it's possible not all ranks
  // will have method (i.e. mortar) data.
  SLIC_ERROR_IF( *sMat != nullptr, "tribol::getJacobianSparseMatrix(): " << "sparse matrix pointer not null." );

  // get access to coupling scheme
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  SLIC_ERROR_IF( !cs, "tribol::getJacobianSparseMatrix(): " << "invalid CouplingScheme id." );

  switch ( cs->getContactMethod() ) {
    case MORTAR_WEIGHTS:
    case ALIGNED_MORTAR:
    case SINGLE_MORTAR: {
      *sMat = static_cast<MortarData*>( cs->getMethodData() )->getMfemSparseMatrix();
      return 0;
    }
    default: {
      SLIC_WARNING( "tribol::getJacobianSparseMatrix(): interface method does not return matrix data." );
      return 1;
    }
  }

}  // end getJacobianSparseMatrix()

//------------------------------------------------------------------------------
int getJacobianCSRMatrix( int** I, int** J, RealT** vals, IndexT cs_id, int* n_offsets, int* n_nonzero )
{
  // check to make sure input pointers are null
  if ( *I != nullptr || *J != nullptr || *vals != nullptr ) {
    SLIC_WARNING( "tribol::getJacobianCSRMatrix(): input pointers must be null." );
    return 1;
  }

  // get access to coupling scheme
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // Note, SLIC_<>_ROOT macros are not here because it's possible not all ranks will have
  // method data.
  SLIC_ERROR_IF( !cs, "tribol::getJacobianCSRMatrix(): invalid " << "CouplingScheme id." );

  switch ( cs->getContactMethod() ) {
    case ALIGNED_MORTAR: {
      SLIC_WARNING( "tribol::getJacobianCSRMatrix(): CSR format not currently implemented with "
                    << "ALIGNED_MORTAR. Use MFEM sparse matrix registration." );
      return 1;
    }
    case MORTAR_WEIGHTS: {
      static_cast<MortarData*>( cs->getMethodData() )->getCSRArrays( I, J, vals, n_offsets, n_nonzero );
      return 0;
    }
    case SINGLE_MORTAR: {
      SLIC_WARNING(
          "tribol::getJacobianCSRMatrix(): CSR format not currently implemented with "
          "SINGLE_MORTAR. Use MFEM sparse matrix registration." );
      return 1;
    }
    default: {
      SLIC_WARNING( "tribol::getJacobianCSRMatrix(): method does not return matrix data; " << "invalid call." );
      return 1;
    }
  }

}  // end getJacobianCSRMatrix()

//------------------------------------------------------------------------------
int getElementBlockJacobians( IndexT cs_id, BlockSpace row_block, BlockSpace col_block,
                              const ArrayT<int>** row_elem_idx, const ArrayT<int>** col_elem_idx,
                              const ArrayT<mfem::DenseMatrix>** jacobians )
{
  // get access to coupling scheme
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  // Note, SLIC_<>_ROOT macros are not here because it's possible not all ranks will have
  // method data.
  SLIC_ERROR_IF( !cs, "tribol::getElementBlockJacobians(): invalid " << "CouplingScheme id." );

  SparseMode sparse_mode = cs->getEnforcementOptions().lm_implicit_options.sparse_mode;
  if ( sparse_mode != SparseMode::MFEM_ELEMENT_DENSE ) {
    SLIC_WARNING(
        "Jacobian is assembled and can be accessed by "
        "getMfemSparseMatrix() or getCSRMatrix(). For (unassembled) element "
        "Jacobian contributions, call setLagrangeMultiplierOptions() with "
        "SparseMode::MFEM_ELEMENT_DENSE before calling update()." );
    return 1;
  }
  MethodData* method_data = cs->getMethodData();
  if ( method_data != nullptr ) {
    *row_elem_idx = &method_data->getBlockJElementIds()[static_cast<int>( row_block )];
    *col_elem_idx = &method_data->getBlockJElementIds()[static_cast<int>( col_block )];
    *jacobians = &method_data->getBlockJ()( static_cast<int>( row_block ), static_cast<int>( col_block ) );
  }
  return 0;

}  // end getElementBlockJacobians()

//------------------------------------------------------------------------------
void registerMortarGaps( IndexT mesh_id, RealT* gaps )
{
  auto mesh = MeshManager::getInstance().findData( mesh_id );

  SLIC_ERROR_ROOT_IF( !mesh, "tribol::registerMortarGaps(): " << "no mesh with id " << mesh_id << " exists." );

  if ( gaps == nullptr && mesh->numberOfElements() > 0 ) {
    SLIC_WARNING( "tribol::registerMortarGaps(): null pointer to gap data " << "on non-null mesh " << mesh_id << "." );
    mesh->isMeshValid() = false;
  } else {
    mesh->getNodalFields().m_node_gap = ArrayViewT<RealT>( gaps, mesh->numberOfNodes() );
    mesh->getNodalFields().m_is_node_gap_set = true;
  }

}  // end registerMortarGaps()

//------------------------------------------------------------------------------
void registerMortarPressures( IndexT mesh_id, const RealT* pressures )
{
  auto mesh = MeshManager::getInstance().findData( mesh_id );

  SLIC_ERROR_ROOT_IF( !mesh, "tribol::registerMortarPressures(): " << "no mesh with id " << mesh_id << " exists." );

  if ( pressures == nullptr && mesh->numberOfElements() > 0 ) {
    SLIC_WARNING( "tribol::registerMortarPressures(): null pointer to pressure data " << "on non-null mesh " << mesh_id
                                                                                      << "." );
    mesh->isMeshValid() = false;
  } else {
    mesh->getNodalFields().m_node_pressure = ArrayViewT<const RealT>( pressures, mesh->numberOfNodes() );
    mesh->getNodalFields().m_is_node_pressure_set = true;
  }

}  // end registerMortarPressures()

//------------------------------------------------------------------------------
void registerIntNodalField( IndexT mesh_id, const IntNodalFields field, int* TRIBOL_UNUSED_PARAM( fieldVariable ) )
{
  auto mesh = MeshManager::getInstance().findData( mesh_id );

  SLIC_ERROR_ROOT_IF( !mesh, "tribol::registerIntNodalField(): " << "no mesh with id " << mesh_id << " exists." );

  switch ( field ) {
    case UNDEFINED_INT_NODAL_FIELD:
    default:
      SLIC_ERROR_ROOT( "tribol::registerIntNodalField() not yet implemented." );
  }  // end switch over field

}  // end registerIntNodalField()

//------------------------------------------------------------------------------
void registerRealElementField( IndexT mesh_id, const RealElementFields field, const RealT* fieldVariable )
{
  auto mesh = MeshManager::getInstance().findData( mesh_id );

  SLIC_ERROR_IF( !mesh, "tribol::registerRealElementField(): " << "no mesh with id " << mesh_id << " exists." );

  switch ( field ) {
    case KINEMATIC_CONSTANT_STIFFNESS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh->numberOfElements() > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for "
                      << "'KINEMATIC_CONSTANT_STIFFNESS' on mesh " << mesh_id << "." );
          mesh->getElementData().m_is_kinematic_constant_penalty_set = false;
        } else {
          mesh->getElementData().m_is_kinematic_constant_penalty_set = true;
        }
      } else {
        mesh->getElementData().m_penalty_stiffness = *fieldVariable;
        mesh->getElementData().m_is_kinematic_constant_penalty_set = true;
      }
      break;
    }
    case RATE_CONSTANT_STIFFNESS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh->numberOfElements() > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for "
                      << "'RATE_CONSTANT_STIFFNESS' on mesh " << mesh_id << "." );
          mesh->getElementData().m_is_rate_constant_penalty_set = false;
        } else {
          mesh->getElementData().m_is_rate_constant_penalty_set = true;
        }
      } else {
        mesh->getElementData().m_rate_penalty_stiffness = *fieldVariable;
        mesh->getElementData().m_is_rate_constant_penalty_set = true;
      }
      break;
    }
    case RATE_PERCENT_STIFFNESS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh->numberOfElements() > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for "
                      << "'RATE_PERCENT_STIFFNESS' on mesh " << mesh_id << "." );
          mesh->getElementData().m_is_rate_percent_penalty_set = false;
        } else {
          mesh->getElementData().m_is_rate_percent_penalty_set = true;
        }
      } else {
        mesh->getElementData().m_rate_percent_stiffness = *fieldVariable;
        mesh->getElementData().m_is_rate_percent_penalty_set = true;
      }
      break;
    }
    case BULK_MODULUS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh->numberOfElements() > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for " << "'BULK_MODULUS' on mesh "
                                                                                      << mesh_id << "." );
          mesh->getElementData().m_is_kinematic_element_penalty_set = false;
        } else {
          // set boolean to true for zero element meshes (acceptable registration)
          mesh->getElementData().m_is_kinematic_element_penalty_set = true;
        }
      } else {
        mesh->getElementData().m_mat_mod = ArrayViewT<const RealT>( fieldVariable, mesh->numberOfElements() );

        // Only set boolean to true if the element thickness has been registered
        // for nonzero element meshes. This will be true if the element thickness
        // was registered first (need both).
        if ( !mesh->getElementData().m_thickness.empty() ) {
          mesh->getElementData().m_is_kinematic_element_penalty_set = true;
        }
      }

      break;
    }
    case YOUNGS_MODULUS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh->numberOfElements() > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for " << "'YOUNGS_MODULUS' on mesh "
                                                                                      << mesh_id << "." );
          mesh->getElementData().m_is_kinematic_element_penalty_set = false;
        } else {
          // set boolean to true for zero element meshes (acceptable registration)
          mesh->getElementData().m_is_kinematic_element_penalty_set = true;
        }
      } else {
        mesh->getElementData().m_mat_mod = ArrayViewT<const RealT>( fieldVariable, mesh->numberOfElements() );

        // Only set boolean to true if the element thickness has been registered
        // for nonzero element meshes. This will be true if the element thickness
        // was registered first (need both).
        if ( !mesh->getElementData().m_thickness.empty() ) {
          mesh->getElementData().m_is_kinematic_element_penalty_set = true;
        }
      }

      break;
    }
    case ELEMENT_THICKNESS: {
      if ( fieldVariable == nullptr ) {
        if ( mesh->numberOfElements() > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for " << "'ELEMENT_THICKNESS' on mesh "
                                                                                      << mesh_id << "." );
          mesh->getElementData().m_is_kinematic_element_penalty_set = false;
        } else {
          // set booleans to true for zero element meshes (acceptable registration)
          mesh->getElementData().m_is_kinematic_element_penalty_set = true;
          mesh->getElementData().m_is_element_thickness_set = true;
        }
      } else {
        mesh->getElementData().m_thickness = ArrayViewT<const RealT>( fieldVariable, mesh->numberOfElements() );
        mesh->getElementData().m_is_element_thickness_set = true;

        // Only set boolean to true if the material modulus has been registered for
        // nonzero element meshes. This will set to true if the material modulus was
        // registered first (need both).
        if ( !mesh->getElementData().m_mat_mod.empty() ) {
          mesh->getElementData().m_is_kinematic_element_penalty_set = true;
        }
      }

      break;
    }
    case VISCOUS_DAMPING_COEFF: {
      if ( fieldVariable == nullptr ) {
        if ( mesh->numberOfElements() > 0 ) {
          SLIC_ERROR( "tribol::registerRealElementField(): null pointer to data for "
                      << "'VISCOUS_DAMPING_COEFF' on mesh " << mesh_id << "." );
          mesh->getElementData().m_is_viscous_damping_coeff_set = false;
        } else {
          mesh->getElementData().m_is_viscous_damping_coeff_set = true;
        }
      } else {
        mesh->getElementData().m_viscous_damping_coeff = *fieldVariable;
        mesh->getElementData().m_is_viscous_damping_coeff_set = true;
      }
      break;
    }
    default: {
      SLIC_ERROR( "tribol::registerRealElementField(): the field argument "
                  << "on mesh " << mesh_id << " is not an accepted tribol real element field." );
    }
  }  // end switch over field

}  // end registerRealElementField()

//------------------------------------------------------------------------------
void registerIntElementField( IndexT mesh_id, const IntElementFields field, int* TRIBOL_UNUSED_PARAM( fieldVariable ) )
{
  auto mesh = MeshManager::getInstance().findData( mesh_id );

  SLIC_ERROR_ROOT_IF( !mesh, "tribol::registerIntElementField(): " << "no mesh with id " << mesh_id << " exists." );

  switch ( field ) {
    case UNDEFINED_INT_ELEMENT_FIELD:
    default:
      SLIC_ERROR_ROOT( "tribol::registerIntElementField() not yet implemented." );
  }  // end switch over field

}  // end registerIntElementField()

//------------------------------------------------------------------------------
void registerCouplingScheme( IndexT cs_id, IndexT mesh_id1, IndexT mesh_id2, int contact_mode, int contact_case,
                             int contact_method, int contact_model, int enforcement_method, int binning_method,
                             ExecutionMode given_exec_mode )
{
  CouplingScheme scheme( cs_id, mesh_id1, mesh_id2, contact_mode, contact_case, contact_method, contact_model,
                         enforcement_method, binning_method, given_exec_mode );

  // add coupling scheme to manager. Validity checks are performed in
  // tribol::update() when each coupling scheme is initialized.
  CouplingSchemeManager::getInstance().addData( cs_id, std::move( scheme ) );

}  // end registerCouplingScheme()

//------------------------------------------------------------------------------
void setInterfacePairs( IndexT cs_id, IndexT numPairs, IndexT const* const pairIndex1, IndexT const* const pairIndex2 )
{
  // get access to coupling scheme
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );

  SLIC_ERROR_ROOT_IF( !cs, "tribol::setInterfacePairs(): invalid coupling scheme index." );

  auto& pairs = cs->getInterfacePairs();
  pairs.clear();
  pairs.reserve( numPairs );

  // copy the interaction pairs
  for ( int i = 0; i < numPairs; ++i ) {
    // perform initial face-pair validity checks to add valid face-pairs
    // to interface pair manager. Note, further computational geometry
    // filtering will be performed on each face-pair indendifying
    // contact candidates.
    if ( geomFilter( cs->getView(), pairIndex1[i], pairIndex2[i] ) ) {
      pairs.emplace_back( pairIndex1[i], pairIndex2[i], true );
    }
  }

  // Disable per-cycle rebinning
  cs->setFixedBinning( true );

}  // end setInterfacePairs()

//------------------------------------------------------------------------------
int update( int cycle, RealT t, RealT& dt )
{
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
  for ( auto& cs_pair : CouplingSchemeManager::getInstance() ) {
    auto& cs = cs_pair.second;

    // initialize and check for valid coupling scheme. If not valid, the coupling
    // scheme will not be valid across all ranks and we will skip this coupling scheme
    if ( !cs.init() ) {
      SLIC_WARNING_ROOT( "tribol::update(): skipping invalid CouplingScheme " << cs_pair.first
                                                                              << ". Please see warnings." );
      continue;
    }

    // perform binning between meshes on the coupling scheme
    // Note, this routine is guarded against null meshes
    cs.performBinning();

    // apply the coupling scheme. Note, there are appropriate guards against zero
    // element meshes, or null-mesh coupling schemes
    err_cs = cs.apply( cycle, t, dt );

    if ( err_cs != 0 ) {
      SLIC_WARNING( "tribol::update(): coupling scheme " << cs_pair.first << " returned with an error." );
    }

  }  // end of coupling scheme loop

  return err_cs;

}  // end update()

//------------------------------------------------------------------------------
void finalize()
{
  CouplingSchemeManager::getInstance().clear();
  MeshManager::getInstance().clear();
}

//------------------------------------------------------------------------------
}  // namespace tribol
