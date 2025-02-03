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
class ProximityTest : public testing::TestWithParam<std::tuple<int, tribol::RealT, tribol::RealT>> {
 protected:
  double max_force_;
  void SetUp2DProblem()
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

    // set up tribol
    constexpr int coupling_scheme_id = 0;
    constexpr int mesh1_id = 0;
    constexpr int mesh2_id = 1;
    tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, mesh.getNodes(), contact_surf_1,
                                        contact_surf_2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                                        tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                        tribol::BINNING_GRID );
    tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, penalty, penalty );
    tribol::setBinningProximityScale( coupling_scheme_id, binning_proximity );

    tribol::updateMfemParallelDecomposition();
    constexpr int cycle = 0;
    constexpr tribol::RealT t = 0.0;
    tribol::RealT dt = 1.0;
    tribol::update( cycle, t, dt );

    mfem::LinearForm r( &mesh.getNodesFESpace() );
    tribol::getMfemResponse( coupling_scheme_id, r );
    max_force_ = r.Max();

    r.Print();

    MPI_Allreduce( MPI_IN_PLACE, &max_force_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  }
};

TEST_P( ProximityTest, CheckForceValues )
{
  SetUp2DProblem();
  std::cout << "max_force_: " << max_force_ << std::endl;
  // EXPECT_LT( std::abs( max_disp_ - 0.013637427890739103 ), 1.5e-6 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( tribol, ProximityTest,
                          testing::Values( std::make_tuple( 1, 0.0, 1.0 ), std::make_tuple( 2, 0.0, 1.0 ) ) );

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
