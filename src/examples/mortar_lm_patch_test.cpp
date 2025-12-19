// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "examples_common.hpp"  // for common functionality used in examples

// Tribol includes
#include "tribol/interface/tribol.hpp"
#include "tribol/utils/TestUtils.hpp"
#include "tribol/common/Parameters.hpp"

// Axom includes
#include "axom/slic.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// C/C++ includes
#include <string>  // for std::string and operators

//----------------------------------------------------------------------------------
// Example command line arguments for running this example. This test creates two
// rectangular blocks of dimensions (l x w x h). The dimensions are set such that
// an initial block-intersection exists, triggering the contact interaction. The
// blocks are discretized per the "block#_res xx yy zz" input arguments, where
// "xx", "yy", and "zz" are the number of elements in the x, y and z directions.
// Varying "xx" and "yy#" will vary the number of contact overlaps that exist
// between the two surfaces, and therefore, the amount of contact work.
//
// This example also sets homogeneous Dirichlet boundary conditions on the TestMesh
// object suitable for a contact patch test. This example calls Tribol to obtain
// the contact contributions to the residual and Jacobian, and calls MFEM to obtain
// these contributions from the stress divergence residual term
// (i.e. equilibrium contributions). Then, the Tribol and equilibrium contributions
// are combined into one monolithic system of equations and the pressure Lagrange
// multipliers and displacement degrees of freedom are solved for. This example
// mimics how a physics application would interact with Tribol, pre and post
// tribol::update() call. More specifically, this example demonstrates how to use
// the Tribol API to obtain Jacobian (i.e. sparse matrix) contributions.
//
// Note: This example uses MFEM linear algebra features; namely, the LU solver.
// There may be issues if the mesh blocks are too refined. Additionally, the
// runtime will be very slow.
//
// srun -n1 ./mortar_lm_patch_test_ex --block1_res 5 5 1  --block1_min 0 0 0. --block1_max 1 1 1.05 --block2_res 4 4 1
// --block2_min 0 0 0.95 --block2_max 1 1 2
//
//----------------------------------------------------------------------------------

/*!
 * \brief Program main.
 *
 * This example runs the mortar + Lagrange multiplier algorithm for two opposing blocks,
 * and shows an example of how to post-process Tribol data to solve for the
 * Lagrange multiplier field
 *
 * \param [in] argc argument counter
 * \param [in] argv argument vector
 *
 * \return rc return code, a non-zero return code indicates an error.
 */
int main( int argc, char** argv )
{
  ////////////////////////////////
  //                            //
  // SETUP THE EXAMPLE AND MESH //
  //                            //
  ////////////////////////////////

  // initialize
#ifdef TRIBOL_USE_MPI
  MPI_Init( &argc, &argv );
#endif
  tribol::CommT problem_comm = TRIBOL_COMM_WORLD;

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  initialize_logger( problem_comm );

  // parse command line arguments. This is
  Arguments args;
  {
    // set common plane + penalty specific options
    args.dimension = 3;  // problems currently only setup for 3D
    args.penalty_stiffness = 1.0;
    args.dump_vis = true;

    // parse the command line arguments
    parse_command_line_args( "Mortar Lm patch test example", args, argc, argv );
  }

  // instantiate test mesh object. Note, this mesh object is a Tribol
  // utility for testing. In general, a physics application will have
  // their own mesh data.
  tribol::TestMesh mesh;
  build_mesh_3D( mesh, args, PATCH_BCS );

  ////////////////////////////////////////////////////
  //                                                //
  // CALL TRIBOL API FUNCTIONS FOR INTERFACE UPDATE //
  //                                                //
  // note: look at this routine in                  //
  // example_common.hpp for example of all          //
  // API function calls                             //
  //                                                //
  ////////////////////////////////////////////////////
  int err = tribol_register_and_update( mesh, tribol::SINGLE_MORTAR, tribol::LAGRANGE_MULTIPLIER, tribol::FRICTIONLESS,
                                        args.dump_vis, nullptr );

  if ( err == 1 ) {
    SLIC_WARNING( "Returned from tribol_register_and_update with error." );
  } else {
    SLIC_INFO( "Tribol update executed successfully. " << "Preparing mfem mesh and linear algebra for "
                                                       << "computation of pressure Lagrange multipliers." );
  }

  //////////////////////////////////////////////////////////
  //                                                      //
  // DEMONSTRATE POST TRIBOL UPDATE ACCESS TO SPARSE DATA //
  // FOR DISPLACEMENT AND LAGRANGE MULTIPLIER SOLUTION    //
  //                                                      //
  // note: for the SINGLE_MORTAR method the sparse data   //
  // is the Jacobian contributions to the global system   //
  // of equations. Here we will access this data making   //
  // use of MFEM data objects and linear algebra features //
  // for convenience. We will form the global system of   //
  // equations and solve it. In the Lagrange multiplier   //
  // enforcement of this method, the system of equations  //
  // is modified to include gap constraint equations      //
  // and a stacked solution vector of nodal displacements //
  // AND pressure Lagrange multipliers, which are the     //
  // auxiliary variables enforcing the gap constraints.   //
  //                                                      //
  //////////////////////////////////////////////////////////
  {
    // setup MFEM mesh representation of the TestMesh
    // in order to facilitate sparse data structure and linear
    // algebra features
    mesh.setupMfemMesh();

    // get Jacobian sparse matrix from Tribol. Note: the Tribol MFEM
    // matrix option does not return CSR data. Rather, the matrix is in
    // a row-wise linked list format. This will make it convenient (i.e. easy)
    // to sum in MFEM equilibrium contributions after which the matrix
    // will be finalized.
    mfem::SparseMatrix* tribolJac{ nullptr };
    int sparseMatErr = tribol::getJacobianSparseMatrix( &tribolJac, 0 );

    SLIC_ERROR_IF( sparseMatErr != 0, "mortar_lm_patch_test.cpp: error " << "gettting Tribol mfem sparse matrix." );

    SLIC_INFO( "Setup mfem mesh and grabbed Tribol sparse data." );

    ////////////////////////////////////////////////////
    //                                                //
    // Compute Equilibrium contributions on MFEM mesh //
    //                                                //
    // note: Tribol only provides the contact         //
    // contributions. Here we compute the equilibrium //
    // physics contributions making use of the MFEM   //
    // linear elasticity integrator.                  //
    //                                                //
    ////////////////////////////////////////////////////

    // setup material properties
    RealT nu_val{ 0.33 };      // Poisson's ratio
    RealT youngs_val{ 3.E7 };  // Young's modulus (approximate, mild steel (psi))

    /////////////////////////////////////////////////////////////////////////
    // compute equilibrium contributions and sum into Tribol sparse matrix //
    /////////////////////////////////////////////////////////////////////////
    mesh.computeEquilibriumJacobian( tribolJac, nu_val, youngs_val );

    SLIC_INFO( "Assembled equilibrium contributions from mfem." );

    // finalize the sparse matrix. This converts the row-wise linked list
    // format into CSR format. We could use CSR data directly, but for ease
    // we will then convert the sparse matrix to a dense matrix hopefully
    // to ease clarity
    tribolJac->Finalize();
    mfem::DenseMatrix tribolA;  // full global system in dense matrix format
    tribolJac->ToDenseMatrix( tribolA );

    // instantiate mfem vector for RHS contributions
    int rhs_size = mesh.dim * mesh.numTotalNodes +  // equilibrium equations
                   mesh.numNonmortarSurfaceNodes;   // gap equations
    mfem::Vector rhs( rhs_size );
    rhs = 0.0;  // initialize

    SLIC_INFO( "Finalized initial oversized sparse matrix and created rhs vector." );

    //////////////////////////////////////////////////////////
    //                                                      //
    // Condense the matrix tribolA into a system that is    //
    // (dim*numTotalNodes + numPressureDof). For clarity,   //
    // sum contributions into another sparse matrix of this //
    // proper size.                                         //
    //                                                      //
    // note: The Tribol Jacobian was oversized in order to  //
    // use the global connectivity provided by the physics  //
    // application. Specifically, we had to account for     //
    // block 1 pressure dofs when there are none in a       //
    // single mortar method. Now, we have to condense the   //
    // system of equations into the proper size and then    //
    // solve.                                               //
    //                                                      //
    //////////////////////////////////////////////////////////
    int solveSize = rhs_size;
    mfem::SparseMatrix A_s( solveSize );  // A_s for "s"parse matrix "A"
    // int numRows = A_s.NumRows();

    mesh.tribolMatrixToSystemMatrix( &tribolA, &A_s );

    SLIC_INFO( "Condensed oversized matrix into properly sized matrix." );

    /////////////////////////////////////////
    // populate RHS with gap contributions //
    /////////////////////////////////////////
    mesh.getGapEvals( rhs.GetData() );

    SLIC_INFO( "Populated RHS gap contributions." );

    //////////////////////////////////////////////////////////////////////////
    // zero out all homogeneous Dirichlet BC components for each mesh block //
    //////////////////////////////////////////////////////////////////////////
    mesh.enforceDirichletBCs( &A_s, &rhs );

    SLIC_INFO( "Applied homogeneous Dirichlet BCs to global system." );

    // Finalize A_s and convert to dense matrix to solve using
    // MFEM LU solver
    A_s.Finalize();
    mfem::DenseMatrix A;
    A_s.ToDenseMatrix( A );
    int rank = A.Rank( 1.e-15 );
    SLIC_INFO( "Matrix rank: " << rank );

    // instantiate MFEM dense matrix inverse object
    // and solution vector
    mfem::DenseMatrixInverse invA( A );
    mfem::Vector sol;
    invA.Mult( rhs, sol );            // solve the system
    RealT* sol_data = sol.GetData();  // get solution data

    SLIC_INFO( "Solved global system of equations." );

    RealT pressureSum = 0.;
    for ( int i = 0; i < mesh.numNonmortarSurfaceNodes; ++i ) {
      int offset = mesh.dim * mesh.numTotalNodes;
      // exploit offset and contiguous node numbering in the indexing here.
      pressureSum += sol_data[offset + i];
    }

    SLIC_INFO( "Pressure Sum: " << pressureSum << "." );

  }  // end of post Tribol scope

  tribol::finalize();

  SLIC_INFO( "Example has run successfully." );

  axom::slic::flushStreams();
  finalize_logger();

#ifdef TRIBOL_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}
