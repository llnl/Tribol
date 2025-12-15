// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol.hpp"

// Tribol includes
#include "tribol/common/ExecModel.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/simple_tribol.hpp"

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
//
//------------------------------------------------------------------------------
// free functions for simple API usage
//------------------------------------------------------------------------------

int Initialize( bool init_slic )
{
  // initialize slic
  if ( init_slic ) {
    axom::slic::finalize();
    axom::slic::initialize();
    std::string format = "[<LEVEL>]: <MESSAGE> \n";
    axom::slic::setLoggingMsgLevel( axom::slic::message::Info );

    axom::slic::addStreamToAllMsgLevels( new axom::slic::GenericOutputStream( &std::cout, format ) );
  }

  return 0;
}

int Finalize( bool finalize_slic )
{
  // finalize tribol
  tribol::finalize();

  // finalize slic
  if ( finalize_slic ) {
    axom::slic::finalize();
  }

  return 0;
}

void SimpleCouplingSetup( const int dim, int cell_type, int contact_method, int mortar_numCells,
                          int mortar_lengthNodalData, const tribol::IndexT* mortar_connectivity, const double* mortar_x,
                          const double* mortar_y, const double* mortar_z, int nonmortar_numCells,
                          int nonmortar_lengthNodalData, const tribol::IndexT* nonmortar_connectivity,
                          const double* nonmortar_x, const double* nonmortar_y, const double* nonmortar_z,
                          const double area_frac, double* mortar_gaps, double* mortar_pressures )
{
  (void)dim;  // quiet compiler

  if ( contact_method != tribol::MORTAR_WEIGHTS ) {
    SLIC_ERROR( "SimpleCouplingSetup: simple API only works " << "for MORTAR_WEIGHTS method." );
  }

  // register mortar mesh
  int mortarMeshId = 0;
  tribol::registerMesh( mortarMeshId, mortar_numCells, mortar_lengthNodalData, mortar_connectivity, cell_type, mortar_x,
                        mortar_y, mortar_z, tribol::MemorySpace::Host );

  // register nonmortar mesh
  int nonmortarMeshId = 1;
  tribol::registerMesh( nonmortarMeshId, nonmortar_numCells, nonmortar_lengthNodalData, nonmortar_connectivity,
                        cell_type, nonmortar_x, nonmortar_y, nonmortar_z, tribol::MemorySpace::Host );

  // Register mortar gaps and pressures, if provided
  if ( mortar_gaps != nullptr ) {
    tribol::registerMortarGaps( nonmortarMeshId, mortar_gaps );
  }
  if ( mortar_pressures != nullptr ) {
    tribol::registerMortarPressures( nonmortarMeshId, mortar_pressures );
  }

  // note the use of NULL_ENFORCEMENT reflects that this routine is used
  // to initially setup tests for MORTAR_WEIGHTS only!
  tribol::registerCouplingScheme( 0, mortarMeshId, nonmortarMeshId, tribol::SURFACE_TO_SURFACE, tribol::AUTO,
                                  contact_method, tribol::NULL_MODEL, tribol::NULL_ENFORCEMENT,
                                  tribol::DEFAULT_BINNING_METHOD, tribol::ExecutionMode::Sequential );

  // set contact area fraction
  tribol::setContactAreaFrac( 0, area_frac );
  tribol::setPlotCycleIncrement( 0, 1 );

  // set enforcement options for MORTAR_WEIGHTS
  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_WEIGHTS_EVAL,
                                        tribol::SparseMode::MFEM_LINKED_LIST );

  axom::slic::flushStreams();

  return;
}

//------------------------------------------------------------------------------
int Update( double& dt )
{
  int err = tribol::update( 1, 1., dt );

  axom::slic::flushStreams();

  return err;
}

//------------------------------------------------------------------------------
int GetSimpleCouplingCSR( int** I, int** J, double** vals, int* n_offsets, int* n_nonzeros )
{
  int err = tribol::getJacobianCSRMatrix( I, J, vals, 0, n_offsets, n_nonzeros );
  return err;
}
