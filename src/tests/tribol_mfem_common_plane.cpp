// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <set>

#include <gtest/gtest.h>

// Tribol includes
#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/utils/TestUtils.hpp"

// Shared includes
#include "shared/mesh/MeshBuilder.hpp"

// Redecomp includes
#include "redecomp/redecomp.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// MFEM includes
#include "mfem.hpp"

// Axom includes
#include "axom/CLI11.hpp"
#include "axom/slic.hpp"

/**
 * @brief This tests the Tribol MFEM interface running a small common plane explicit contact example using a central
 * difference explicit time integration scheme.
 *
 * Both the element penalty and a constant penalty are tested, with the constant penalty tuned to match the element
 * penalty for this case.  As a result, the test comparisons are the same for both penalty types.
 *
 */
class MfemCommonPlaneTest : public testing::TestWithParam<std::tuple<int, tribol::KinematicPenaltyCalculation>> {
 protected:
  tribol::RealT max_disp_;
  void SetUp() override
  {
    // number of times to uniformly refine the serial mesh before constructing the
    // parallel mesh
    int ref_levels = 2;
    // polynomial order of the finite element discretization
    int order = std::get<0>( GetParam() );
    // initial separation between the two blocks
    double initial_sep = -0.001;
    // initial velocity
    tribol::RealT initial_v = 0.0;
    // timestep size
    tribol::RealT dt = 0.001;
    // end time
    tribol::RealT t_end = 0.01;
    // material density
    tribol::RealT rho = 1000.0;
    // lame parameter
    tribol::RealT lambda = 100000.0;
    // lame parameter (shear modulus)
    tribol::RealT mu = 100000.0;
    // kinematic constant penalty stiffness equivalent to the element-wise calculation,
    // which is bulk-modulus over element thickness.
    tribol::RealT p_kine = ( lambda + 2.0 / 3.0 * mu ) / ( 1.0 / std::pow( 2.0, ref_levels ) );

    // fixed options
    // boundary element attributes of contact surface 1
    auto contact_surf_1 = std::set<int>( { 6 } );
    // boundary element attributes of contact surface 2
    auto contact_surf_2 = std::set<int>( { 7 } );
    // boundary element attributes of fixed surface (points on z = 0, all t)
    auto fixed_attrs = std::set<int>( { 1 } );
    // element attribute corresponding to volume elements where an initial
    // velocity will be applied
    auto moving_attrs = std::set<int>( { 2 } );

#if defined( TRIBOL_USE_CUDA )
    tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Cuda;
#elif defined( TRIBOL_USE_HIP )
    tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Hip;
#else
    tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Sequential;
#endif

    // read mesh
    // clang-format off
    mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::Unify( {
      shared::MeshBuilder::CubeMesh( 1, 1, 1 ),
      shared::MeshBuilder::CubeMesh( 1, 1, 1 )
        .translate( { 0.0, 0.0, 1.0 + initial_sep } )
        .updateAttrib( 1, 2 )    // changes attribute in all volume elements from 1 to 2 so it doesn't clash with the 
                                 // first (bottom) mesh
        .updateBdrAttrib( 1, 7 ) // set the bottom surface to boundary attribute 7
        .updateBdrAttrib( 6, 8 ) // set the top surface to boundary attribute 8
    } ).refine( ref_levels ) );
    // clang-format on

    // grid function for higher-order nodes
    auto fe_coll = mfem::H1_FECollection( order, mesh.SpaceDimension() );
    auto par_fe_space = mfem::ParFiniteElementSpace( &mesh, &fe_coll, mesh.SpaceDimension() );
    auto coords = mfem::ParGridFunction( &par_fe_space );
    if ( order > 1 ) {
      mesh.SetNodalGridFunction( &coords, false );
    } else {
      mesh.GetNodes( coords );
    }

    mfem::ParGridFunction ref_coords{ coords };

    // grid function for displacement
    mfem::ParGridFunction displacement{ &par_fe_space };
    displacement = 0.0;

    // grid function for velocity
    mfem::ParGridFunction velocity{ &par_fe_space };
    velocity = 0.0;

    // set initial velocity
    mfem::Vector init_velocity_vector( { 0.0, 0.0, -std::abs( initial_v ) } );
    mfem::VectorConstantCoefficient init_velocity_coeff( init_velocity_vector );
    mfem::Array<int> moving_attrs_array;
    mfem::Array<mfem::VectorCoefficient*> init_velocity_coeff_array;
    moving_attrs_array.Reserve( moving_attrs.size() );
    init_velocity_coeff_array.Reserve( moving_attrs.size() );
    for ( auto moving_attr : moving_attrs ) {
      moving_attrs_array.Append( moving_attr );
      init_velocity_coeff_array.Append( &init_velocity_coeff );
    }
    mfem::PWVectorCoefficient initial_v_coeff( mesh.SpaceDimension(), moving_attrs_array, init_velocity_coeff_array );
    velocity.ProjectCoefficient( initial_v_coeff );

    // recover dirichlet bc tdof list
    mfem::Array<int> ess_vdof_list;
    {
      mfem::Array<int> ess_vdof_marker;
      mfem::Array<int> ess_bdr( mesh.bdr_attributes.Max() );
      ess_bdr = 0;
      for ( auto fixed_attr : fixed_attrs ) {
        ess_bdr[fixed_attr - 1] = 1;
      }
      par_fe_space.GetEssentialVDofs( ess_bdr, ess_vdof_marker );
      mfem::FiniteElementSpace::MarkerToList( ess_vdof_marker, ess_vdof_list );
    }

    // set up mfem elasticity bilinear form
    mfem::ConstantCoefficient rho_coeff{ rho };
    mfem::ConstantCoefficient lambda_coeff{ lambda };
    mfem::ConstantCoefficient mu_coeff{ mu };
    mfem_ext::ExplicitMechanics op{ par_fe_space, rho_coeff, lambda_coeff, mu_coeff };

    // set up time integrator
    mfem_ext::CentralDiffSolver solver{ ess_vdof_list };
    solver.Init( op );

    // set up tribol
    int coupling_scheme_id = 0;
    int mesh1_id = 0;
    int mesh2_id = 1;
    tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, contact_surf_1,
                                        contact_surf_2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                                        tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                        tribol::BINNING_BVH, exec_mode );
    tribol::registerMfemVelocity( 0, velocity );
    if ( std::get<1>( GetParam() ) == tribol::KINEMATIC_CONSTANT ) {
      tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, p_kine, p_kine );
    } else {
      mfem::Vector bulk_moduli_by_bdry_attrib( mesh.bdr_attributes.Max() );
      bulk_moduli_by_bdry_attrib = lambda + 2.0 / 3.0 * mu;
      mfem::PWConstCoefficient mat_coeff( bulk_moduli_by_bdry_attrib );
      tribol::setMfemKinematicElementPenalty( coupling_scheme_id, mat_coeff );
    }

    int cycle{ 0 };
    for ( tribol::RealT t{ 0.0 }; t < t_end; t += dt ) {
      // build new parallel decomposed redecomp mesh and update grid functions
      // on each mesh
      tribol::updateMfemParallelDecomposition();
      tribol::update( cycle, t, dt );
      op.f_ext = 0.0;
      tribol::getMfemResponse( 0, op.f_ext );

      op.SetTime( t );
      solver.Step( displacement, velocity, t, dt );

      coords.Set( 1.0, ref_coords );
      coords += displacement;
      if ( order == 1 ) {
        coords.HostRead();
        mesh.SetVertices( coords );
      }

      ++cycle;
    }

    max_disp_ = displacement.Max();
    MPI_Allreduce( MPI_IN_PLACE, &max_disp_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

    tribol::finalize();
  }
};

TEST_P( MfemCommonPlaneTest, common_plane )
{
  // make sure there is some contact response after 10 cycles
  EXPECT_GT( max_disp_, 1.0e-4 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( tribol, MfemCommonPlaneTest,
                          testing::Values( std::make_tuple( 1, tribol::KINEMATIC_CONSTANT ),
                                           std::make_tuple( 1, tribol::KINEMATIC_ELEMENT ),
                                           std::make_tuple( 2, tribol::KINEMATIC_CONSTANT ),
                                           std::make_tuple( 2, tribol::KINEMATIC_ELEMENT ) ) );

//------------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

#if defined( TRIBOL_USE_CUDA )
  std::string device_str( "cuda" );
#elif defined( TRIBOL_USE_HIP )
  std::string device_str( "hip" );
#elif defined( TRIBOL_USE_OPENMP )
  std::string device_str( "omp" );
#else
  std::string device_str( "cpu" );
#endif

  mfem::Device device( device_str );
  device.Print();

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
