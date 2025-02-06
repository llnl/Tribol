// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <iostream>
#include <set>

#include <gtest/gtest.h>

// Tribol includes
#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

#include "shared/mesh/MeshBuilder.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// MFEM includes
#include "mfem.hpp"

/**
 * @brief This tests the Tribol MFEM interface running a small common plane explicit contact example using a central
 * difference explicit time integration scheme.
 *
 * Both the element penalty and a constant penalty are tested, with the constant penalty tuned to match the element
 * penalty for this case.  As a result, the test comparisons are the same for both penalty types.
 *
 */
class ProximityTest : public testing::TestWithParam<std::tuple<int, tribol::RealT, tribol::RealT, bool>> {
 protected:
  double max_force_;

  std::array<tribol::BinningMethod, 3> binning_methods_{ tribol::BINNING_CARTESIAN_PRODUCT, tribol::BINNING_GRID,
                                                         tribol::BINNING_BVH };

  void UpdateTribol( shared::ParMeshBuilder& mesh, int coupling_scheme_id )
  {
    tribol::updateMfemParallelDecomposition();
    constexpr int cycle = 0;
    constexpr tribol::RealT t = 0.0;
    tribol::RealT dt = 1.0;
    tribol::update( cycle, t, dt );

    mfem::LinearForm r( &mesh.getNodesFESpace() );
    r = 0.0;
    tribol::getMfemResponse( coupling_scheme_id, r );
    max_force_ = r.Max();

    r.Print();

    MPI_Allreduce( MPI_IN_PLACE, &max_force_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  }

  void SetUpCommonPlaneProblem( shared::ParMeshBuilder& mesh, const std::set<int>& contact_surf_1,
                                const std::set<int>& contact_surf_2, tribol::RealT penalty,
                                tribol::RealT binning_proximity, tribol::BinningMethod binning_method )
  {
    // set up tribol
    constexpr int coupling_scheme_id = 0;
    constexpr int mesh1_id = 0;
    constexpr int mesh2_id = 1;
    tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, mesh.getNodes(), contact_surf_1,
                                        contact_surf_2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                                        tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY, binning_method );
    tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, penalty, penalty );
    tribol::setBinningProximityScale( coupling_scheme_id, binning_proximity );

    UpdateTribol( mesh, coupling_scheme_id );
  }

  void SetUp2DCommonPlaneProblem( tribol::BinningMethod binning_method )
  {
    constexpr int dim = 2;
    // polynomial order of the finite element discretization
    auto order = std::get<0>( GetParam() );
    // binning proximity value
    auto binning_proximity = std::get<1>( GetParam() );
    // mesh interpenetration
    auto mesh_interpenetration = std::get<2>( GetParam() );

    // fixed options

    // kinematic constant penalty stiffness
    constexpr tribol::RealT penalty = 1.0;
    // boundary element attributes of contact surface 1, the top surface of the bottom block
    auto contact_surf_1 = std::set<int>( { 3 } );
    // boundary element attributes of contact surface 2, the bottom surface of the top block
    auto contact_surf_2 = std::set<int>( { 5 } );

    // create a mesh with two blocks
    // clang-format off
    shared::ParMeshBuilder mesh( MPI_COMM_WORLD, shared::MeshBuilder::Merged( {
      shared::MeshBuilder::SquareMesh( 1, 1 ),
      shared::MeshBuilder::SquareMesh( 1, 1 )
        .translate( { 0.0, 1.0 - mesh_interpenetration } )
        .updateAttrib( 1, 2 )
        .updateBdrAttrib( 1, 5 )
        .updateBdrAttrib( 3, 6 )
      } ) );
    // clang-format on

    // grid function for higher-order nodes
    mesh.setNodesFEColl( mfem::H1_FECollection( order, dim ) );

    SetUpCommonPlaneProblem( mesh, contact_surf_1, contact_surf_2, penalty, binning_proximity, binning_method );
  }

  std::tuple<shared::ParMeshBuilder, std::set<int>, std::set<int>> SetUp3DMesh()
  {
    constexpr int dim = 3;
    // polynomial order of the finite element discretization
    auto order = std::get<0>( GetParam() );
    // mesh interpenetration (scaled by element radius)
    auto mesh_interpenetration = std::get<2>( GetParam() ) * std::sqrt( 2 );

    // fixed options

    // boundary element attributes of contact surface 1, the top surface of the bottom block
    auto contact_surf_1 = std::set<int>( { 6 } );
    // boundary element attributes of contact surface 2, the bottom surface of the top block
    auto contact_surf_2 = std::set<int>( { 7 } );

    // create a mesh with two cubes
    // clang-format off
    shared::ParMeshBuilder mesh( MPI_COMM_WORLD, shared::MeshBuilder::Merged( {
      shared::MeshBuilder::HypercubeMesh( dim, 1 ),
      shared::MeshBuilder::HypercubeMesh( dim, 1 )
        .translate( { 0.0, 0.0, 1.0 - mesh_interpenetration } )
        .updateAttrib( 1, 2 )
        .updateBdrAttrib( 1, 7 )
        .updateBdrAttrib( 6, 8 )
      } ) );
    // clang-format on

    // grid function for higher-order nodes
    mesh.setNodesFEColl( mfem::H1_FECollection( order, dim ) );

    return { std::move( mesh ), std::move( contact_surf_1 ), std::move( contact_surf_2 ) };
  }

  void SetUp3DCommonPlaneProblem( tribol::BinningMethod binning_method )
  {
    // binning proximity value
    auto binning_proximity = std::get<1>( GetParam() );
    // kinematic constant penalty stiffness
    constexpr tribol::RealT penalty = 1.0;

    auto mesh = SetUp3DMesh();

    SetUpCommonPlaneProblem( std::get<0>( mesh ), std::get<1>( mesh ), std::get<2>( mesh ), penalty, binning_proximity,
                             binning_method );
  }

  void SetUpMortarProblem( tribol::BinningMethod binning_method )
  {
    // binning proximity value
    auto binning_proximity = std::get<1>( GetParam() );

    auto mesh = SetUp3DMesh();

    // set up tribol
    constexpr int coupling_scheme_id = 0;
    constexpr int mesh1_id = 0;
    constexpr int mesh2_id = 1;
    tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, std::get<0>( mesh ),
                                        std::get<0>( mesh ).getNodes(), std::get<1>( mesh ), std::get<2>( mesh ),
                                        tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::SINGLE_MORTAR,
                                        tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, binning_method );
    tribol::setLagrangeMultiplierOptions( coupling_scheme_id, tribol::ImplicitEvalMode::MORTAR_RESIDUAL );
    tribol::getMfemPressure( coupling_scheme_id ) = 1.0;
    tribol::setBinningProximityScale( coupling_scheme_id, binning_proximity );

    UpdateTribol( std::get<0>( mesh ), coupling_scheme_id );
  }
};

TEST_P( ProximityTest, CheckForceValues2DCommonPlane )
{
  auto should_have_force = std::get<3>( GetParam() );

  for ( auto binning_method : binning_methods_ ) {
    std::cout << "Binning method = " << binning_method << std::endl;
    SetUp2DCommonPlaneProblem( binning_method );
    std::cout << "  max_force_ = " << max_force_ << std::endl;
    if ( should_have_force ) {
      EXPECT_GT( max_force_, 1.0e-6 );
    } else {
      EXPECT_LT( max_force_, 1.0e-15 );
    }
  }

  MPI_Barrier( MPI_COMM_WORLD );
}

TEST_P( ProximityTest, CheckForceValues3DCommonPlane )
{
  auto should_have_force = std::get<3>( GetParam() );

  for ( auto binning_method : binning_methods_ ) {
    std::cout << "Binning method = " << binning_method << std::endl;
    SetUp3DCommonPlaneProblem( binning_method );
    std::cout << "  max_force_ = " << max_force_ << std::endl;
    if ( should_have_force ) {
      EXPECT_GT( max_force_, 1.0e-6 );
    } else {
      EXPECT_LT( max_force_, 1.0e-15 );
    }
  }

  MPI_Barrier( MPI_COMM_WORLD );
}

TEST_P( ProximityTest, CheckForceValues3DMortar )
{
  auto should_have_force = std::get<3>( GetParam() );

  for ( auto binning_method : binning_methods_ ) {
    std::cout << "Binning method = " << binning_method << std::endl;
    SetUpMortarProblem( binning_method );
    std::cout << "  max_force_ = " << max_force_ << std::endl;
    if ( should_have_force ) {
      EXPECT_GT( max_force_, 1.0e-6 );
    } else {
      EXPECT_LT( max_force_, 1.0e-15 );
    }
  }

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P(
    tribol, ProximityTest,
    testing::Values( std::make_tuple( 1, 0.0, 2.0, true ), std::make_tuple( 1, 0.0, 2.01, false ),
                     std::make_tuple( 2, 0.0, 1.3333,
                                      true ),  // this should be 2.0, but lumped mass is affecting LOR accuracy
                     std::make_tuple( 2, 0.0, 1.3433,
                                      false ),  // this should be 2.01, but lumped mass is affecting LOR accuracy
                     std::make_tuple( 1, 4.0, 4.0, true ), std::make_tuple( 1, 4.0, 4.01, false ),
                     std::make_tuple( 2, 4.0, 2.6666,
                                      true ),  // this should be 4.0, but lumped mass is affecting LOR accuracy
                     std::make_tuple( 2, 4.0, 2.6766,
                                      false ) ) );  // this should be 4.01, but lumped mass is affecting LOR accuracy

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
