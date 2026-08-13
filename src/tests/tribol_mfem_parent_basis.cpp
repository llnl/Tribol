#include <cmath>
#include <set>

#include <gtest/gtest.h>

#include "axom/slic.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

namespace {

struct ParentBasisResult {
  double response_norm;
  double stability_dt;
};

ParentBasisResult RunParentBasisCase( int lor_factor )
{
  auto surface1 = std::set<int>( { 3 } );
  auto surface2 = std::set<int>( { 5 } );

  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
    shared::MeshBuilder::SquareMesh(1, 1)
      .updateBdrAttrib(1, 1)
      .updateBdrAttrib(2, 2)
      .updateBdrAttrib(3, 3)
      .updateBdrAttrib(4, 4),
    shared::MeshBuilder::SquareMesh(1, 1)
      .translate({0.0, 0.99})
      .updateAttrib(1, 2)
      .updateBdrAttrib(1, 5)
      .updateBdrAttrib(2, 6)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(4, 8)
  }));
  // clang-format on

  mesh.SetCurvature( 2 );
  auto* nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
  EXPECT_NE( nodes, nullptr );
  if ( nodes == nullptr ) {
    return { 0., 0. };
  }
  mfem::ParGridFunction coords( nodes->ParFESpace() );
  coords = *nodes;
  mfem::ParGridFunction velocity( nodes->ParFESpace() );
  velocity = 0.;
  mfem::ParGridFunction inverse_mass( nodes->ParFESpace() );
  inverse_mass = 1.;

  constexpr int coupling_scheme_id = 710;
  constexpr int mesh1_id = 1420;
  constexpr int mesh2_id = 1421;
  tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, surface1, surface2,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                      tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                      tribol::ExecutionMode::Sequential );
  tribol::setMfemLORFactor( coupling_scheme_id, lor_factor );
  tribol::setMfemSurfaceBasis( coupling_scheme_id, tribol::MfemSurfaceBasis::PARENT );
  tribol::registerMfemVelocity( coupling_scheme_id, velocity );
  tribol::registerMfemInverseMass( coupling_scheme_id, inverse_mass );
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, 100., 100. );
  tribol::setCommonPlaneIntegrationOptions( coupling_scheme_id, tribol::MULTI_POINT, 3 );

  int num_ranks = 1;
  MPI_Comm_size( MPI_COMM_WORLD, &num_ranks );
  tribol::updateMfemParallelDecomposition( num_ranks, true );
  tribol::update( 0, 0., 1.e-3 );

  mfem::Vector response( coords.Size() );
  response = 0.;
  tribol::getMfemResponse( coupling_scheme_id, response );
  double response_norm_squared = response * response;
  MPI_Allreduce( MPI_IN_PLACE, &response_norm_squared, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  const double stability_dt = tribol::getPenaltyStabilityTimestep( coupling_scheme_id );
  tribol::finalize();
  return { std::sqrt( response_norm_squared ), stability_dt };
}

TEST( MfemParentBasis, StraightQ2EdgeIndependentOfSegmentation )
{
  const ParentBasisResult result_lor2 = RunParentBasisCase( 2 );
  const ParentBasisResult result_lor4 = RunParentBasisCase( 4 );
  EXPECT_GT( result_lor2.response_norm, 0. );
  EXPECT_TRUE( std::isfinite( result_lor2.stability_dt ) );
  EXPECT_NEAR( result_lor2.response_norm, result_lor4.response_norm, 1.e-9 * result_lor2.response_norm );
  EXPECT_NEAR( result_lor2.stability_dt, result_lor4.stability_dt, 1.e-9 * result_lor2.stability_dt );
}

}  // namespace

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();
#endif

  mfem::Device device( "cpu" );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
