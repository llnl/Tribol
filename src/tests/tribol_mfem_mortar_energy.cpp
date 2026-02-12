// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <set>

#include <gtest/gtest.h>

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// MFEM includes
#include "mfem.hpp"

// Axom includes
#include "axom/CLI11.hpp"
#include "axom/slic.hpp"

// Shared includes
#include "shared/mesh/MeshBuilder.hpp"

// Redecomp includes
#include "redecomp/redecomp.hpp"

// Tribol includes
#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

/**
 * @brief This tests the Tribol MFEM interface running a contact patch test using ENERGY_MORTAR.
 *
 */
class MfemMortarEnergyTest : public testing::TestWithParam<std::tuple<int>> {
 protected:
  tribol::RealT max_disp_;
  void SetUp() override
  {
    // number of times to uniformly refine the serial mesh before constructing the
    // parallel mesh
    int ref_levels = std::get<0>( GetParam() );
    // polynomial order of the finite element discretization
    int order = 1;

    // fixed options
    // boundary element attributes of mortar surface (bottom of top square)
    auto mortar_attrs = std::set<int>( { 5 } );
    // boundary element attributes of nonmortar surface (top of bottom square)
    auto nonmortar_attrs = std::set<int>( { 3 } );
    // boundary element attributes of x-fixed surfaces (left side)
    auto xfixed_attrs = std::set<int>( { 4 } );
    // boundary element attributes of y-fixed surfaces (bottom of bottom square, top of top square)
    auto yfixed_attrs = std::set<int>( { 1, 6 } );

    // build mesh of 2 squares
    int nel_per_dir = std::pow( 2, ref_levels );

    // clang-format off
    mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
      shared::MeshBuilder::SquareMesh(nel_per_dir, nel_per_dir) // Bottom mesh [0,1]x[0,1]
        .updateBdrAttrib(1, 1) // Bottom (Fixed Y)
        .updateBdrAttrib(2, 2) // Right
        .updateBdrAttrib(3, 3) // Top (NonMortar)
        .updateBdrAttrib(4, 4), // Left (Fixed X)
      shared::MeshBuilder::SquareMesh(nel_per_dir, nel_per_dir) // Top mesh [0,1]x[0,1]
        .translate({0.0, 0.99}) // Shift up to [0,1]x[0.99, 1.99]. Overlap 0.01.
        .updateBdrAttrib(1, 5) // Bottom (Mortar)
        .updateBdrAttrib(2, 2) // Right
        .updateBdrAttrib(3, 6) // Top (Fixed Y)
        .updateBdrAttrib(4, 4) // Left (Fixed X)
    }));
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

    // grid function for displacement
    mfem::ParGridFunction displacement{ &par_fe_space };
    displacement = 0.0;

    // recover dirichlet bc tdof list
    mfem::Array<int> ess_tdof_list;
    {
      mfem::Array<int> ess_vdof_marker;
      mfem::Array<int> ess_bdr( mesh.bdr_attributes.Max() );
      ess_bdr = 0;
      for ( auto xfixed_attr : xfixed_attrs ) {
        if ( xfixed_attr <= ess_bdr.Size() ) ess_bdr[xfixed_attr - 1] = 1;
      }
      par_fe_space.GetEssentialVDofs( ess_bdr, ess_vdof_marker, 0 );
      mfem::Array<int> new_ess_vdof_marker;
      ess_bdr = 0;
      for ( auto yfixed_attr : yfixed_attrs ) {
        if ( yfixed_attr <= ess_bdr.Size() ) ess_bdr[yfixed_attr - 1] = 1;
      }
      par_fe_space.GetEssentialVDofs( ess_bdr, new_ess_vdof_marker, 1 );
      for ( int i{ 0 }; i < ess_vdof_marker.Size(); ++i ) {
        ess_vdof_marker[i] = ess_vdof_marker[i] || new_ess_vdof_marker[i];
      }
      mfem::Array<int> ess_tdof_marker;
      par_fe_space.GetRestrictionMatrix()->BooleanMult( ess_vdof_marker, ess_tdof_marker );
      mfem::FiniteElementSpace::MarkerToList( ess_tdof_marker, ess_tdof_list );
    }

    // set up mfem elasticity bilinear form
    mfem::ParBilinearForm a( &par_fe_space );
    mfem::ConstantCoefficient lambda( 50.0 );
    mfem::ConstantCoefficient mu( 50.0 );
    a.AddDomainIntegrator( new mfem::ElasticityIntegrator( lambda, mu ) );
    a.Assemble();

    // compute elasticity contribution to stiffness
    auto A = std::make_unique<mfem::HypreParMatrix>();
    a.FormSystemMatrix( ess_tdof_list, *A );

    // set up tribol
    coords.ReadWrite();
    int coupling_scheme_id = 0;
    int mesh1_id = 0;
    int mesh2_id = 1;
    tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs,
                                        nonmortar_attrs, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING,
                                        tribol::ENERGY_MORTAR, tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER,
                                        tribol::BINNING_GRID );
    tribol::setLagrangeMultiplierOptions( coupling_scheme_id, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN );

    // Set Penalty options
    tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, 1000.0, 1000.0 );

    coords.ReadWrite();
    // update tribol (compute contact contribution to force and stiffness)
    tribol::updateMfemParallelDecomposition();
    tribol::RealT dt{ 1.0 };  // time is arbitrary here (no timesteps)
    tribol::update( 1, 1.0, dt );

    // retrieve contact stiffness matrix
    auto A_cont = tribol::getMfemDfDx( coupling_scheme_id );

    // retrieve contact force (response)
    auto f_contact = tribol::getMfemTDofForce( coupling_scheme_id );
    f_contact.Neg();
    for ( int i{ 0 }; i < ess_tdof_list.Size(); ++i ) {
      f_contact( ess_tdof_list[i] ) = 0.0;
    }

    // Add contact stiffness to elasticity stiffness
    auto A_total = std::unique_ptr<mfem::HypreParMatrix>( mfem::Add( 1.0, *A, 1.0, *A_cont ) );
    A_total->EliminateRowsCols( ess_tdof_list );

    // Solve for X (displacement)
    mfem::Vector X( par_fe_space.GetTrueVSize() );
    X = 0.0;

    mfem::MINRESSolver solver( MPI_COMM_WORLD );
    solver.SetRelTol( 1.0e-8 );
    solver.SetAbsTol( 1.0e-12 );
    solver.SetMaxIter( 5000 );
    solver.SetPrintLevel( 3 );
    solver.SetOperator( *A_total );
    solver.Mult( f_contact, X );

    // move displacements to grid function
    {
      auto& P = *par_fe_space.GetProlongationMatrix();
      P.Mult( X, displacement );
    }

    // We can check max displacement magnitude.
    auto local_max = displacement.Max();
    max_disp_ = 0.0;
    MPI_Allreduce( &local_max, &max_disp_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  }
};

TEST_P( MfemMortarEnergyTest, check_mortar_displacement )
{
  // Penalty enforcement with nonlinear contact enforcement. Let's just check that max_disp_ is positive and roughly
  // correct order of magnitude.
  EXPECT_GT( max_disp_, 0.0 );
  EXPECT_LT( max_disp_, 0.01 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( tribol, MfemMortarEnergyTest, testing::Values( std::make_tuple( 2 ) ) );

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

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
