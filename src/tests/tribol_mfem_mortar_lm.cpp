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
 * @brief This tests the Tribol MFEM interface running a contact patch test.
 *
 */
class MfemMortarTest : public testing::TestWithParam<std::tuple<int, mfem::Element::Type>> {
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
    // boundary element attributes of mortar surface
    auto mortar_attrs = std::set<int>( { 4 } );
    // boundary element attributes of nonmortar surface
    auto nonmortar_attrs = std::set<int>( { 5 } );
    // boundary element attributes of x-fixed surfaces (at x = 0)
    auto xfixed_attrs = std::set<int>( { 1 } );
    // boundary element attributes of y-fixed surfaces (at y = 0)
    auto yfixed_attrs = std::set<int>( { 2 } );
    // boundary element attributes of z-fixed surfaces (3: surface at z = 0, 6: surface at z = 1.99)
    auto zfixed_attrs = std::set<int>( { 3, 6 } );

    // build mesh of 2 cubes
    int nel_per_dir = std::pow( 2, ref_levels );
    // clang-format off
    mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
      shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir, std::get<1>(GetParam()))
        .updateBdrAttrib(3, 7)
        .updateBdrAttrib(1, 3)
        .updateBdrAttrib(4, 7)
        .updateBdrAttrib(5, 1)
        .updateBdrAttrib(6, 4),
      shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir, std::get<1>(GetParam()))
        .translate({0.0, 0.0, 0.99})
        .updateBdrAttrib(1, 8)
        .updateBdrAttrib(3, 7)
        .updateBdrAttrib(4, 7)
        .updateBdrAttrib(5, 1)
        .updateBdrAttrib(8, 5)
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
        ess_bdr[xfixed_attr - 1] = 1;
      }
      par_fe_space.GetEssentialVDofs( ess_bdr, ess_vdof_marker, 0 );
      mfem::Array<int> new_ess_vdof_marker;
      ess_bdr = 0;
      for ( auto yfixed_attr : yfixed_attrs ) {
        ess_bdr[yfixed_attr - 1] = 1;
      }
      par_fe_space.GetEssentialVDofs( ess_bdr, new_ess_vdof_marker, 1 );
      for ( int i{ 0 }; i < ess_vdof_marker.Size(); ++i ) {
        ess_vdof_marker[i] = ess_vdof_marker[i] || new_ess_vdof_marker[i];
      }
      ess_bdr = 0;
      for ( auto zfixed_attr : zfixed_attrs ) {
        ess_bdr[zfixed_attr - 1] = 1;
      }
      par_fe_space.GetEssentialVDofs( ess_bdr, new_ess_vdof_marker, 2 );
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
    int coupling_scheme_id = 0;
    int mesh1_id = 0;
    int mesh2_id = 1;
    tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs,
                                        nonmortar_attrs, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING,
                                        tribol::SINGLE_MORTAR, tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER,
                                        tribol::BINNING_GRID );
    tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN );

    // update tribol (compute contact contribution to force and stiffness)
    tribol::updateMfemParallelDecomposition();
    tribol::RealT dt{ 1.0 };  // time is arbitrary here (no timesteps)
    tribol::update( 1, 1.0, dt );

    // retrieve block stiffness matrix
    auto A_blk = tribol::getMfemBlockJacobian( 0 );
    A_blk->SetBlock( 0, 0, A.release() );

    // create block solution and RHS vectors
    mfem::BlockVector B_blk{ A_blk->ColOffsets() };
    B_blk = 0.0;
    mfem::BlockVector X_blk{ A_blk->RowOffsets() };
    X_blk = 0.0;

    // retrieve gap vector (RHS) from contact
    mfem::ParGridFunction g;
    tribol::getMfemGap( 0, g );

    // prolongation transpose operator on submesh: maps dofs stored in g to tdofs stored in G
    {
      auto& G = B_blk.GetBlock( 1 );
      auto& P_submesh = *tribol::getMfemPressure( 0 ).ParFESpace()->GetProlongationMatrix();
      P_submesh.MultTranspose( g, G );
    }

    // solve for X_blk
    mfem::MINRESSolver solver( MPI_COMM_WORLD );
    solver.SetRelTol( 1.0e-8 );
    solver.SetAbsTol( 1.0e-12 );
    solver.SetMaxIter( 5000 );
    solver.SetPrintLevel( 3 );
    solver.SetOperator( *A_blk );
    solver.Mult( B_blk, X_blk );

    // move block displacements to grid function
    {
      auto& U = X_blk.GetBlock( 0 );
      auto& P = *par_fe_space.GetProlongationMatrix();
      P.Mult( U, displacement );
    }
    displacement.Neg();

    auto local_max = displacement.Max();
    max_disp_ = 0.0;
    MPI_Allreduce( &local_max, &max_disp_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  }
};

TEST_P( MfemMortarTest, mass_matrix_transfer )
{
  EXPECT_LT( std::abs( max_disp_ - 0.005 ), 1.0e-6 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( tribol, MfemMortarTest,
                          testing::Values( std::make_tuple( 2, mfem::Element::Type::HEXAHEDRON ),
                                           std::make_tuple( 2, mfem::Element::Type::TETRAHEDRON ) ) );

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
