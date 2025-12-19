// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_EXAMPLES_EXAMPLES_COMMON_HPP_
#define SRC_EXAMPLES_EXAMPLES_COMMON_HPP_

// axom includes
#include "axom/core.hpp"
#include "axom/primal.hpp"
#include "axom/slic.hpp"
#include "axom/CLI11.hpp"
#include "axom/fmt.hpp"

// tribol includes
#include "tribol/interface/tribol.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/utils/TestUtils.hpp"

// C/C++ includes
#include <algorithm>  // for std::fill_n()
#include <iostream>

// namespace aliases
namespace primal = axom::primal;
namespace slic = axom::slic;
namespace utilities = axom::utilities;
namespace CLI = axom::CLI;

using RealT = tribol::RealT;

//------------------------------------------------------------------------------
// COMMON DATA STRUCTURE DEFINITIONS
//------------------------------------------------------------------------------

/*!
 * \brief Enumerates the boundary condition types for two-block examples
 */
enum BLOCK_EX_BCS
{
  NO_BCS,     ///! no boundary conditions
  PATCH_BCS,  ///! patch test boundary conditions
  NUM_BLOCK_EX_BCS
};

/**
 * Custom CLI11 Validator to check that a command line argument
 * is a number greater than or equal to \a n
 */
class NumberAtLeast : public CLI::Validator {
 public:
  NumberAtLeast( int n ) : Validator( axom::fmt::format( ">={} ", n ) )
  {
    func_ = [=]( std::string& number_str ) {
      int number;
      if ( !CLI::detail::lexical_cast( number_str, number ) ) {
        return "Failed parsing as a number " + number_str;
      }
      if ( number < n ) {
        return axom::fmt::format( "Number ({}) must be at least {} ", number, n );
      }
      return std::string();
    };
  }
};

/// Utility function to return an axom::Point from a vector<T>
template <typename T, int D = 3>
primal::Point<T, D> getPoint( const std::vector<T>& vec )
{
  return primal::Point<T, D>( vec.data(), D );
}

/// Utility function to return an axom BoundingBox from a pair of vector<T>.
/// The expectation is that the vectors represent the min and max coords of the box
template <typename T, int D = 3>
primal::BoundingBox<T, D> getBoundingBox( const std::vector<T>& mins, const std::vector<T>& maxs )
{
  return primal::BoundingBox<T, D>( getPoint( mins ), getPoint( maxs ) );
}

/*!
 * \brief Holds command line arguments (with defaults) for the examples.
 */
struct Arguments {
  int dimension{ 3 };              // problem dimension, i.e., 2 or 3.
  int refinement_factor{ 0 };      // factor to control number of contact cells
  int numcycles{ 10 };             // total number of cycles
  int m;                           // number of vertices in polygon A (intersection ex.)
  int n;                           // number of vertices in polygon B (intersection ex.)
  std::string output_dir{};        // directory for file output
  double penalty_stiffness{ 1. };  // penalty stiffness
  bool dump_vis{ false };          // should the example dump visualization files?

  /// input arguments for examples that use the TestMesh class
  std::vector<int> block1_res{ 4, 4, 4 };                 // block1 -- number of elements in each direction
  std::vector<tribol::RealT> block1_min{ 0., 0., 0. };    // block1 -- bounding box min
  std::vector<tribol::RealT> block1_max{ 1., 1., 1.05 };  // block1 -- bounding box max
  std::vector<int> block2_res{ 4, 4, 4 };                 // block2 -- number of elements in each direction
  std::vector<tribol::RealT> block2_min{ 0., 0., 0.95 };  // block2 -- bounding box min
  std::vector<tribol::RealT> block2_max{ 1., 1., 2. };    // block2 -- bounding box max
};

//------------------------------------------------------------------------------
// FUNCTION PROTOTYPES
//------------------------------------------------------------------------------

/*!
 * \brief Parse command line arguments
 * \param [in]
 */
int parse_command_line_args( std::string ex_name, Arguments& args, int argc, char** argv )
{
  CLI::App app{ ex_name };
  app.add_option( "--block1_res", args.block1_res, "Mesh 1 x,y,z discretization" )->expected( 3 );

  app.add_option( "--block1_min", args.block1_min, "Bounding box min for Mesh 1" )->expected( 3 );

  app.add_option( "--block1_max", args.block1_max, "Bounding box max for Mesh 1" )->expected( 3 );

  app.add_option( "--block2_res", args.block2_res, "Mesh 2 x,y,z discretization" )->expected( 3 );

  app.add_option( "--block2_min", args.block2_min, "Bounding box min for Mesh 2" )->expected( 3 );
  app.add_option( "--block2_max", args.block2_max, "Bounding box max for Mesh 2" )->expected( 3 );

  app.add_flag( "--dump-vis,!--no-dump-vis", args.dump_vis, "Dump vis files?" )->capture_default_str();

  app.get_formatter()->column_width( 35 );

  CLI11_PARSE( app, argc, argv );

  // print parsed args to screen
  SLIC_INFO( "Mesh 1" << "\n\t x,y,z discretization: " << getPoint( args.block1_res )
                      << "\n\t Bounding box: " << getBoundingBox( args.block1_min, args.block1_max ) );
  SLIC_INFO( "Mesh 2" << "\n\t x,y,z discretization: " << getPoint( args.block2_res )
                      << "\n\t Bounding box: " << getBoundingBox( args.block2_min, args.block2_max ) );

  return 0;
}

void build_mesh_3D( tribol::TestMesh& mesh, Arguments& args, BLOCK_EX_BCS bc_type )
{
  auto& res1 = args.block1_res;
  auto& min1 = args.block1_min;
  auto& max1 = args.block1_max;

  auto& res2 = args.block2_res;
  auto& min2 = args.block2_min;
  auto& max2 = args.block2_max;

  mesh.setupContactMeshHex( res1[0], res1[1], res1[2], min1[0], min1[1], min1[2], max1[0], max1[1], max1[2], res2[0],
                            res2[1], res2[2], min2[0], min2[1], min2[2], max2[0], max2[1], max2[2], 0., 0. );

  switch ( bc_type ) {
    case NO_BCS: {
      // no boundary conditions required on mesh object
      break;
    }
    case PATCH_BCS: {
      // setup block 1 homogeneous Dirichlet BCs
      mesh.setupPatchTestDirichletBCs( mesh.mortarMeshId, res1[0], res1[1], res1[2], 0, false, 0. );

      // setup block 2 homogeneous Dirichlet BCs
      mesh.setupPatchTestDirichletBCs( mesh.nonmortarMeshId, res2[0], res2[1], res2[2], mesh.numMortarNodes, false,
                                       0. );

      // setup DUMMY block 1 pressure dof array
      mesh.setupPatchTestPressureDofs( mesh.mortarMeshId, res1[0], res1[1], res1[2], 0, false );

      // setup block 2 pressure dofs
      mesh.setupPatchTestPressureDofs( mesh.nonmortarMeshId, res2[0], res2[1], res2[2], mesh.numMortarNodes, true );
      break;
    }
    default: {
      SLIC_ERROR( "build_mesh_3D(): boundary condition type on 3D test mesh not supported." );
    }
  }

  return;
}

int tribol_register_and_update( tribol::TestMesh& mesh, tribol::ContactMethod method,
                                tribol::EnforcementMethod enforcement, tribol::ContactModel model, bool visualization,
                                tribol::TestControlParameters* params )
{
  /////////////////////////////////////////////
  //                                         //
  // STEP 1: register the interacting meshes //
  //                                         //
  // note: the physics application           //
  // (host code) calling Tribol will have    //
  // their own mesh data. For this example,  //
  // Tribol makes use of an internal         //
  // mesh class typically used for testing.  //
  //                                         //
  /////////////////////////////////////////////
  const int cellType = mesh.cellType;

  // set the mesh ids for block 1 and block 2
  const int block1_id = 0;
  const int block2_id = 1;

  // register the two meshes with Tribol
  tribol::registerMesh( block1_id, mesh.numMortarFaces, mesh.numTotalNodes, mesh.faceConn1, cellType, mesh.x, mesh.y,
                        mesh.z, tribol::MemorySpace::Host );
  tribol::registerMesh( block2_id, mesh.numNonmortarFaces, mesh.numTotalNodes, mesh.faceConn2, cellType, mesh.x, mesh.y,
                        mesh.z, tribol::MemorySpace::Host );
  ///////////////////////////////////
  //                               //
  // STEP 2: register field arrays //
  //                               //
  ///////////////////////////////////
  {
    ///////////////////////////////////////////////
    //                                           //
    // STEP 2.0: register the nodal force arrays //
    //                                           //
    // note: these are the residuals for each    //
    // mesh sized such that the connectivity     //
    // array ids can properly index into each    //
    // array. For clarity, separate x,y, and z   //
    // arrays are created for each block, but    //
    // in practice these can be the same         //
    // arrays for both blocks.                   //
    //                                           //
    ///////////////////////////////////////////////
    tribol::allocRealArray( &mesh.fx1, mesh.numTotalNodes, 0. );
    tribol::allocRealArray( &mesh.fy1, mesh.numTotalNodes, 0. );
    tribol::allocRealArray( &mesh.fz1, mesh.numTotalNodes, 0. );
    tribol::allocRealArray( &mesh.fx2, mesh.numTotalNodes, 0. );
    tribol::allocRealArray( &mesh.fy2, mesh.numTotalNodes, 0. );
    tribol::allocRealArray( &mesh.fz2, mesh.numTotalNodes, 0. );

    tribol::registerNodalResponse( block1_id, mesh.fx1, mesh.fy1, mesh.fz1 );

    tribol::registerNodalResponse( block2_id, mesh.fx2, mesh.fy2, mesh.fz2 );

    //////////////////////////////////////////////////////
    //                                                  //
    // STEP 2.1: register nodal pressure and gap arrays //
    //                                                  //
    // note: this is for mortar-based methods ONLY.     //
    //                                                  //
    //////////////////////////////////////////////////////
    if ( method == tribol::SINGLE_MORTAR || method == tribol::ALIGNED_MORTAR ) {
      tribol::allocRealArray( &mesh.gaps, mesh.numTotalNodes, 0. );
      tribol::allocRealArray( &mesh.pressures, mesh.numTotalNodes, 1. );

      // register nodal gaps and pressures. Note: for single sided mortar
      // methods these fields are only registered for one mesh. By
      // convention, this is the second mesh/block.
      tribol::registerMortarGaps( block2_id, mesh.gaps );
      tribol::registerMortarPressures( block2_id, mesh.pressures );
    } else if ( method == tribol::MORTAR_WEIGHTS ) {
      tribol::allocRealArray( &mesh.gaps, mesh.numTotalNodes, 0. );
      mesh.pressures = nullptr;  // not needed

      tribol::registerMortarGaps( block2_id, mesh.gaps );
    }

  }  // end STEP 2 scope

  /////////////////////////////////////////////
  //                                         //
  // STEP 3: register enforcement parameters //
  //                                         //
  // note: these are method specific         //
  // and may not be applicable for a given   //
  // method.                                 //
  //                                         //
  /////////////////////////////////////////////
  {
    ///////////////////////////////////////////////////////
    //                                                   //
    // STEP 3.0: register penalty enforcement parameters //
    //                                                   //
    ///////////////////////////////////////////////////////
    if ( enforcement == tribol::PENALTY ) {
      if ( params == nullptr ) {
        SLIC_ERROR( "tribol_register_and_update(): " << "tribol::TestControlParameters pointer is null. "
                                                     << "Parameters are required for penalty enforcement." );
      }

      if ( !params->penalty_ratio ) {
        tribol::setKinematicConstantPenalty( block1_id, params->const_penalty );
        tribol::setKinematicConstantPenalty( block2_id, params->const_penalty );
      }

      else {
        // mortar penalty data
        tribol::allocRealArray( &mesh.mortar_bulk_mod, mesh.numMortarFaces, params->const_penalty );
        tribol::allocRealArray( &mesh.mortar_element_thickness, mesh.numMortarFaces, 1.0 );
        tribol::registerRealElementField( block1_id, tribol::BULK_MODULUS, mesh.mortar_bulk_mod );
        tribol::registerRealElementField( block1_id, tribol::ELEMENT_THICKNESS, mesh.mortar_element_thickness );

        // nonmortar penalty data
        tribol::allocRealArray( &mesh.nonmortar_bulk_mod, mesh.numNonmortarFaces, params->const_penalty );
        tribol::allocRealArray( &mesh.nonmortar_element_thickness, mesh.numNonmortarFaces, 1.0 );
        tribol::registerRealElementField( block2_id, tribol::BULK_MODULUS, mesh.nonmortar_bulk_mod );
        tribol::registerRealElementField( block2_id, tribol::ELEMENT_THICKNESS, mesh.nonmortar_element_thickness );
      }
    }  // end if-penalty

  }  // end STEP 3 scope

  ///////////////////////////////////////////////
  //                                           //
  // STEP 4: register Tribol "coupling scheme" //
  //                                           //
  ///////////////////////////////////////////////
  const int csIndex = 0;
  registerCouplingScheme( csIndex, block1_id, block2_id, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, method, model,
                          enforcement, tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );

  //////////////////////////////////////////////////////////
  //                                                      //
  // STEP 4.5: set enforcement options on coupling scheme //
  //                                                      //
  //////////////////////////////////////////////////////////
  if ( method == tribol::COMMON_PLANE && enforcement == tribol::PENALTY ) {
    tribol::PenaltyConstraintType constraint_type = ( params->constant_rate_penalty || params->percent_rate_penalty )
                                                        ? tribol::KINEMATIC_AND_RATE
                                                        : tribol::KINEMATIC;
    tribol::KinematicPenaltyCalculation pen_calc =
        ( params->penalty_ratio ) ? tribol::KINEMATIC_ELEMENT : tribol::KINEMATIC_CONSTANT;

    tribol::RatePenaltyCalculation rate_calc;
    rate_calc = tribol::NO_RATE_PENALTY;
    if ( params->constant_rate_penalty ) {
      rate_calc = tribol::RATE_CONSTANT;
    } else if ( params->percent_rate_penalty ) {
      rate_calc = tribol::RATE_PERCENT;
    }

    // set penalty options after registering coupling scheme
    tribol::setPenaltyOptions( csIndex, constraint_type, pen_calc, rate_calc );
  } else if ( ( method == tribol::SINGLE_MORTAR || method == tribol::ALIGNED_MORTAR ) &&
              enforcement == tribol::LAGRANGE_MULTIPLIER ) {
    tribol::setLagrangeMultiplierOptions( csIndex, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN,
                                          tribol::SparseMode::MFEM_LINKED_LIST );
  } else if ( method == tribol::MORTAR_WEIGHTS ) {
    tribol::setLagrangeMultiplierOptions( csIndex, tribol::ImplicitEvalMode::MORTAR_WEIGHTS_EVAL,
                                          tribol::SparseMode::MFEM_LINKED_LIST );
  } else {
    SLIC_ERROR( "Method and enforcement are not supported in the examples." );
  }

  ///////////////////////////////////
  //                               //
  // STEP 5: optional registration //
  //                               //
  ///////////////////////////////////
  {
    ////////////////////////////////////////////////////////////////////
    //                                                                //
    // STEP 5.0 (optional): register plot cycle increment             //
    //                      for visualizing registered meshes,        //
    //                      active surface faces, and patch overlaps. //
    //                      This is hardcoded as default behavior     //
    //                      for examples.                             //
    //                                                                //
    ////////////////////////////////////////////////////////////////////
    if ( visualization ) {
      tribol::setPlotCycleIncrement( csIndex, 1 );
      tribol::setPlotOptions( csIndex, tribol::VIS_MESH_FACES_AND_OVERLAPS );
    }

    //////////////////////////////////////////////////////////////
    //                                                          //
    // STEP 5.1 (optional): register the area fraction for the  //
    //                      smallest considered contact overlap //
    //                      patch                               //
    //                                                          //
    //  note: this is not required and a default value is       //
    //  used by Tribol that should be sufficient for all        //
    //  methods.                                                //
    //                                                          //
    //////////////////////////////////////////////////////////////
    tribol::setContactAreaFrac( csIndex, 1.e-6 );

  }  // end STEP 5 scope

  //////////////////////////////////////
  //                                  //
  // STEP 6: call Tribol update       //
  //                                  //
  // note: this updates all pertinent //
  // fields registered with Tribol    //
  // per a given interface numerical  //
  // method.                          //
  //                                  //
  //////////////////////////////////////
  double dt = 1.0;
  int err = tribol::update( 1, 1., dt );

  // specific to TestMesh class; dump mesh to vtk for visualization
  if ( visualization ) {
    mesh.testMeshToVtk( "", 1, 1 );
  }

  return err;
}

/*!
 * \brief Initialize logger
 * \param [in] problem_comm MPI communicator
 */
void initialize_logger( tribol::CommT problem_comm )
{
  slic::initialize();
  slic::setLoggingMsgLevel( slic::message::Debug );
  std::string msg_format = "[<LEVEL>] <MESSAGE>\n";

#ifdef TRIBOL_USE_MPI
  slic::addStreamToAllMsgLevels( new slic::SynchronizedStream( &std::cout, problem_comm, msg_format ) );
#else
  slic::addStreamToAllMsgLevels( new slic::GenericOutputStream( &std::cout, msg_format ) );
  static_cast<void>( problem_comm );  // elide warning about unused vars
#endif
}

/*!
 * \brief Finalize logger
 */
void finalize_logger() { slic::finalize(); }

#endif /* SRC_EXAMPLES_EXAMPLES_COMMON_HPP_ */
