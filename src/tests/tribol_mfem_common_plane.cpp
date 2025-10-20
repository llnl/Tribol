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
class MfemCommonPlaneTest : public testing::TestWithParam<std::pair<int, tribol::KinematicPenaltyCalculation>> {
 protected:
  tribol::RealT max_disp_;
  void SetUp() override
  {
    // number of times to uniformly refine the serial mesh before constructing the
    // parallel mesh
    int ref_levels = 2;
    // polynomial order of the finite element discretization
    int order = GetParam().first;
    // initial velocity
    tribol::RealT initial_v = 0.02;
    // timestep size
    tribol::RealT dt = 0.001;
    // end time
    tribol::RealT t_end = 2.0;
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
    // location of mesh file. TRIBOL_REPO_DIR is defined in tribol/config.hpp
    std::string mesh_file = TRIBOL_REPO_DIR "/data/two_hex_apart.mesh";
    // boundary element attributes of contact surface 1
    auto contact_surf_1 = std::set<int>( { 4 } );
    // boundary element attributes of contact surface 2
    auto contact_surf_2 = std::set<int>( { 5 } );
    // boundary element attributes of fixed surface (points on z = 0, all t)
    auto fixed_attrs = std::set<int>( { 3 } );
    // element attribute corresponding to volume elements where an initial
    // velocity will be applied
    auto moving_attrs = std::set<int>( { 2 } );

    // read mesh
    std::unique_ptr<mfem::ParMesh> pmesh{ nullptr };
    {
      // read serial mesh
      auto mesh = std::make_unique<mfem::Mesh>( mesh_file.c_str(), 1, 1 );

      // refine serial mesh
      if ( ref_levels > 0 ) {
        for ( int i{ 0 }; i < ref_levels; ++i ) {
          mesh->UniformRefinement();
        }
      }

      // create parallel mesh from serial
      pmesh = std::make_unique<mfem::ParMesh>( MPI_COMM_WORLD, *mesh );
      mesh.reset( nullptr );

      // further refinement of parallel mesh
      {
        int par_ref_levels = 0;
        for ( int i{ 0 }; i < par_ref_levels; ++i ) {
          pmesh->UniformRefinement();
        }
      }
    }

    // grid function for higher-order nodes
    auto fe_coll = mfem::H1_FECollection( order, pmesh->SpaceDimension() );
    auto par_fe_space = mfem::ParFiniteElementSpace( pmesh.get(), &fe_coll, pmesh->SpaceDimension() );
    auto coords = mfem::ParGridFunction( &par_fe_space );
    if ( order > 1 ) {
      pmesh->SetNodalGridFunction( &coords, false );
    } else {
      pmesh->GetNodes( coords );
    }

    mfem::ParGridFunction ref_coords{ coords };

    // grid function for displacement
    mfem::ParGridFunction displacement{ &par_fe_space };
    displacement = 0.0;

    // grid function for velocity
    mfem::ParGridFunction v{ &par_fe_space };
    v = 0.0;

    // set initial velocity
    {
      mfem::Array<int> attrib_marker( pmesh->attributes.Max() );
      attrib_marker = 0;
      for ( auto moving_attr : moving_attrs ) {
        attrib_marker[moving_attr - 1] = 1;
      }
      mfem::Array<int> vdof_marker( par_fe_space.GetVSize() );
      vdof_marker = 0;
      for ( int e{ 0 }; e < pmesh->GetNE(); ++e ) {
        if ( attrib_marker[pmesh->GetElement( e )->GetAttribute() - 1] ) {
          mfem::Array<int> vdofs;
          par_fe_space.GetElementDofs( e, vdofs );
          par_fe_space.DofsToVDofs( 2, vdofs );
          for ( auto vdof : vdofs ) {
            vdof_marker[vdof] = 1;
          }
        }
      }
      for ( int i{ 0 }; i < vdof_marker.Size(); ++i ) {
        if ( vdof_marker[i] ) {
          v[i] = -std::abs( initial_v );
        }
      }
    }

    // recover dirichlet bc tdof list
    mfem::Array<int> ess_vdof_list;
    {
      mfem::Array<int> ess_vdof_marker;
      mfem::Array<int> ess_bdr( pmesh->bdr_attributes.Max() );
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
    tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, *pmesh, coords, contact_surf_1,
                                        contact_surf_2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                                        tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                        tribol::BINNING_GRID );
    tribol::registerMfemVelocity( 0, v );
    if ( GetParam().second == tribol::KINEMATIC_CONSTANT ) {
      tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, p_kine, p_kine );
    } else {
      mfem::Vector bulk_moduli_by_bdry_attrib( pmesh->bdr_attributes.Max() );
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
      solver.Step( displacement, v, t, dt );

      coords.Set( 1.0, ref_coords );
      coords += displacement;
      if ( order == 1 ) {
        pmesh->SetVertices( coords );
      }

      ++cycle;
    }

    max_disp_ = displacement.Max();
    MPI_Allreduce( MPI_IN_PLACE, &max_disp_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  }
};

TEST_P( MfemCommonPlaneTest, common_plane )
{
  EXPECT_LT( std::abs( max_disp_ - 0.0136390367035 ), 1.0e-5 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( tribol, MfemCommonPlaneTest,
                          testing::Values( std::make_pair( 1, tribol::KINEMATIC_CONSTANT ),
                                           std::make_pair( 1, tribol::KINEMATIC_ELEMENT ),
                                           std::make_pair( 2, tribol::KINEMATIC_CONSTANT ),
                                           std::make_pair( 2, tribol::KINEMATIC_ELEMENT ) ) );

//------------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  tribol::finalize();
  MPI_Finalize();

  return result;
}
