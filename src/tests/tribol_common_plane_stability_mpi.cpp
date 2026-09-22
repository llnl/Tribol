// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <mpi.h>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/physics/CommonPlane.hpp"

/** Verify communicator-wide stability voting when one rank has no local contact rows. */
TEST( CommonPlanePenaltyStabilityMPI, IncludesRanksWithoutLocalRows )
{
  // Rank zero owns one active two-body row and rank one owns no local contact
  // mesh. Both ranks must enter the stability reductions and receive the same
  // finite vote; this is the historical empty-rank collective regression.
  int rank = 0;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  constexpr tribol::IndexT coupling_scheme_id = 0;
  constexpr tribol::IndexT first_mesh_id = 0;
  constexpr tribol::IndexT second_mesh_id = 1;
  tribol::IndexT connectivity[2] = { 0, 1 };
  tribol::RealT first_x[2] = { 0.0, 1.0 };
  tribol::RealT first_y[2] = { 0.0, 0.0 };
  tribol::RealT second_x[2] = { 0.0, 1.0 };
  tribol::RealT second_y[2] = { -0.1, -0.1 };
  tribol::RealT zero_values[2] = { 0.0, 0.0 };
  tribol::RealT unit_inverse_mass[2] = { 1.0, 1.0 };
  const tribol::IndexT local_face_count = rank == 0 ? 1 : 0;
  const tribol::IndexT local_node_count = rank == 0 ? 2 : 0;

  tribol::registerMesh( first_mesh_id, local_face_count, local_node_count, rank == 0 ? connectivity : nullptr,
                        tribol::LINEAR_EDGE, rank == 0 ? first_x : nullptr, rank == 0 ? first_y : nullptr, nullptr,
                        tribol::MemorySpace::Host );
  tribol::registerMesh( second_mesh_id, local_face_count, local_node_count, rank == 0 ? connectivity : nullptr,
                        tribol::LINEAR_EDGE, rank == 0 ? second_x : nullptr, rank == 0 ? second_y : nullptr, nullptr,
                        tribol::MemorySpace::Host );
  if ( rank == 0 ) {
    tribol::registerNodalResponse( first_mesh_id, zero_values, zero_values );
    tribol::registerNodalResponse( second_mesh_id, zero_values, zero_values );
    tribol::registerNodalVelocities( first_mesh_id, zero_values, zero_values );
    tribol::registerNodalVelocities( second_mesh_id, zero_values, zero_values );
    tribol::registerNodalInverseMass( first_mesh_id, zero_values, unit_inverse_mass );
    tribol::registerNodalInverseMass( second_mesh_id, zero_values, unit_inverse_mass );
  }
  tribol::setKinematicConstantPenalty( first_mesh_id, 1.0 );
  tribol::setKinematicConstantPenalty( second_mesh_id, 1.0 );
  tribol::registerCouplingScheme( coupling_scheme_id, first_mesh_id, second_mesh_id, tribol::SURFACE_TO_SURFACE,
                                  tribol::NO_CASE, tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                  tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
  tribol::setPenaltyOptions( coupling_scheme_id, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setExplicitIntegratorStabilityFactor( coupling_scheme_id, 2.0 );
  tribol::setMPIComm( coupling_scheme_id, MPI_COMM_WORLD );

  tribol::CouplingScheme& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  ASSERT_TRUE( coupling_scheme.init() );
  auto* common_plane_data = static_cast<tribol::CommonPlaneContactData*>( coupling_scheme.getMethodData() );
  common_plane_data->resize( local_face_count, 2, coupling_scheme.getAllocatorId() );

  if ( rank == 0 ) {
    tribol::CommonPlaneContactData::Viewer rows = common_plane_data->getView();
    constexpr tribol::IndexT row_id = 0;
    rows.pair_row_counts[0] = 1;
    rows.row_is_valid[row_id] = 1;
    rows.row_is_active[row_id] = 1;
    rows.contact_pair_ids[row_id] = 0;
    rows.first_face_ids[row_id] = 0;
    rows.second_face_ids[row_id] = 0;
    rows.first_basis_counts[row_id] = 2;
    rows.second_basis_counts[row_id] = 2;
    rows.row_uses_parent_fields[row_id] = 0;
    rows.first_basis_values( row_id, 0 ) = 1.0;
    rows.second_basis_values( row_id, 0 ) = 1.0;
    rows.normals( row_id, 1 ) = 1.0;
    rows.integration_weights[row_id] = 1.0;
    rows.penalty_stiffnesses[row_id] = 8.0;
  }

  tribol::RealT timestep = 10.0;
  EXPECT_EQ( tribol::ComputeCommonPlanePenaltyStabilityTimeStep( &coupling_scheme, timestep ), 0 );
  EXPECT_DOUBLE_EQ( timestep, 0.5 );
  EXPECT_DOUBLE_EQ( coupling_scheme.getExplicitPenaltyStiffnessBound(), 16.0 );

  tribol::finalize();
  MPI_Barrier( MPI_COMM_WORLD );
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
