#include <mpi.h>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/interface/tribol.hpp"

TEST( TimestepVoteMPI, ReturnsCommunicatorMinimum )
{
  constexpr tribol::IndexT cs_id = 0;
  constexpr tribol::IndexT mesh_id_1 = 0;
  constexpr tribol::IndexT mesh_id_2 = 1;

  tribol::registerMesh( mesh_id_1, 0, 0, nullptr, tribol::LINEAR_EDGE, nullptr, nullptr, nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( mesh_id_2, 0, 0, nullptr, tribol::LINEAR_EDGE, nullptr, nullptr, nullptr,
                        tribol::MemorySpace::Host );
  tribol::setKinematicConstantPenalty( mesh_id_1, 1. );
  tribol::setKinematicConstantPenalty( mesh_id_2, 1. );
  tribol::registerCouplingScheme( cs_id, mesh_id_1, mesh_id_2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                                  tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                  tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
  tribol::setPenaltyOptions( cs_id, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::enableTimestepVote( cs_id, true );
  tribol::setMPIComm( cs_id, MPI_COMM_WORLD );

  int rank = 0;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  tribol::RealT dt = rank == 0 ? 0.25 : 1.;

  EXPECT_EQ( tribol::update( 1, 0., dt ), 0 );
  EXPECT_DOUBLE_EQ( dt, 0.25 );

  tribol::finalize();
}

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  ::testing::InitGoogleTest( &argc, argv );

  int result = 0;
  {
    axom::slic::SimpleLogger logger;
    result = RUN_ALL_TESTS();
  }

  MPI_Finalize();
  return result;
}
