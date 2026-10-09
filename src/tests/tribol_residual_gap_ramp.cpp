// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <cmath>
#include <memory>
#include <numbers>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/mesh/MfemData.hpp"

namespace {

double residualGapAtVertex( const tribol::MfemSubmeshData& data, int vertex )
{
  mfem::Array<int> dofs;
  data.GetSubmeshFESpace().GetVertexDofs( vertex, dofs );
  EXPECT_EQ( dofs.Size(), 1 );
  return data.GetSubmeshResidualGap()[dofs[0]];
}

double ballWeightAtVertex( const tribol::MfemSubmeshData& data, int vertex )
{
  mfem::Array<int> dofs;
  data.GetSubmeshFESpace().GetVertexDofs( vertex, dofs );
  EXPECT_EQ( dofs.Size(), 1 );
  return data.GetSubmeshBallWeight()[dofs[0]];
}

TEST( ResidualGapRamp, ConcaveCornerRampsOverMultipleElements )
{
  auto parent = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::CShapeMesh( 8, 8, 2 ) );
  mfem::Array<int> attributes{ 2, 3 };
  auto submesh = mfem::ParSubMesh::CreateFromBoundary( parent, attributes );
  auto fec = std::make_unique<mfem::H1_FECollection>( 1, 2 );
  tribol::MfemSubmeshData data( submesh, nullptr, std::move( fec ), 1, false );

  constexpr double residual_gap = 0.05;
  const tribol::Parameters parameters{};
  EXPECT_DOUBLE_EQ( parameters.energy_mortar_residual_gap_ramp_angle,
                    tribol::energy_mortar::default_residual_gap_ramp_angle );
  data.UpdateResidualGapField( residual_gap, parameters.energy_mortar_residual_gap_ramp_angle );

  bool found_tip = false;
  bool found_ramp = false;
  bool found_full = false;
  for ( int vertex = 0; vertex < submesh.GetNV(); ++vertex ) {
    double x[2];
    submesh.GetNode( vertex, x );
    const double value = residualGapAtVertex( data, vertex );
    if ( std::abs( x[0] - 0.25 ) < 1.0e-12 && std::abs( x[1] - 0.75 ) < 1.0e-12 ) {
      EXPECT_NEAR( value, 0.0, 1.0e-12 );
      found_tip = true;
    } else if ( value > 0.0 && value < residual_gap ) {
      found_ramp = true;
    } else if ( std::abs( value - residual_gap ) < 1.0e-12 ) {
      found_full = true;
    }
  }
  EXPECT_TRUE( found_tip );
  EXPECT_TRUE( found_ramp );
  EXPECT_TRUE( found_full );
}

TEST( ResidualGapRamp, ConvexCornerAndZeroAngleRemainUniform )
{
  auto parent = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::CShapeMesh( 4, 4, 1 ) );
  mfem::Array<int> attributes{ 1, 6 };
  auto submesh = mfem::ParSubMesh::CreateFromBoundary( parent, attributes );
  auto fec = std::make_unique<mfem::H1_FECollection>( 1, 2 );
  tribol::MfemSubmeshData data( submesh, nullptr, std::move( fec ), 1, false );

  constexpr double residual_gap = 0.05;
  constexpr double pi = std::numbers::pi_v<double>;
  data.UpdateResidualGapField( residual_gap, 10.0 * pi / 180.0 );
  for ( int vertex = 0; vertex < submesh.GetNV(); ++vertex ) {
    EXPECT_NEAR( residualGapAtVertex( data, vertex ), residual_gap, 1.0e-12 );
  }

  data.UpdateResidualGapField( residual_gap, 0.0 );
  for ( int vertex = 0; vertex < submesh.GetNV(); ++vertex ) {
    EXPECT_NEAR( residualGapAtVertex( data, vertex ), residual_gap, 1.0e-12 );
  }
}

TEST( ResidualGapRamp, BallWeightsSelectConvexCorners )
{
  auto parent = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::CShapeMesh( 8, 8, 2 ) );
  mfem::Array<int> attributes{ 1, 2, 3, 4, 5, 6, 7 };
  auto submesh = mfem::ParSubMesh::CreateFromBoundary( parent, attributes );
  auto fec = std::make_unique<mfem::H1_FECollection>( 1, 2 );
  tribol::MfemSubmeshData data( submesh, nullptr, std::move( fec ), 1, false );

  constexpr double residual_gap = 0.05;
  data.UpdateResidualGapField( residual_gap, 0.0 );

  bool found_concave = false;
  bool found_convex = false;
  bool found_straight = false;
  for ( int vertex = 0; vertex < submesh.GetNV(); ++vertex ) {
    double x[2];
    submesh.GetNode( vertex, x );
    if ( std::abs( x[0] - 0.25 ) < 1.0e-12 && std::abs( x[1] - 0.75 ) < 1.0e-12 ) {
      EXPECT_DOUBLE_EQ( ballWeightAtVertex( data, vertex ), 0.0 );
      found_concave = true;
    } else if ( std::abs( x[0] ) < 1.0e-12 && std::abs( x[1] ) < 1.0e-12 ) {
      EXPECT_DOUBLE_EQ( ballWeightAtVertex( data, vertex ), 1.0 );
      found_convex = true;
    } else if ( std::abs( x[0] ) < 1.0e-12 && std::abs( x[1] - 0.5 ) < 1.0e-12 ) {
      EXPECT_DOUBLE_EQ( ballWeightAtVertex( data, vertex ), 0.0 );
      found_straight = true;
    }
    EXPECT_NEAR( residualGapAtVertex( data, vertex ), residual_gap, 1.0e-12 );
  }
  EXPECT_TRUE( found_concave );
  EXPECT_TRUE( found_convex );
  EXPECT_TRUE( found_straight );
}

TEST( ResidualGapRamp, OpenEndpointsRemainEligibleAfterDeformation )
{
  auto parent = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( 2, 1 ) );
  mfem::Array<int> attributes{ 1 };
  auto submesh = mfem::ParSubMesh::CreateFromBoundary( parent, attributes );
  auto fec = std::make_unique<mfem::H1_FECollection>( 1, 2 );
  tribol::MfemSubmeshData data( submesh, nullptr, std::move( fec ), 1, false );

  constexpr double residual_gap = 0.05;
  data.UpdateResidualGapField( residual_gap, 0.0 );

  int endpoint = -1;
  for ( int vertex = 0; vertex < submesh.GetNV(); ++vertex ) {
    if ( ballWeightAtVertex( data, vertex ) > 0.5 ) {
      endpoint = vertex;
      EXPECT_DOUBLE_EQ( ballWeightAtVertex( data, vertex ), 1.0 );
      break;
    }
  }
  ASSERT_GE( endpoint, 0 );

  double moved_endpoint[2];
  submesh.GetNode( endpoint, moved_endpoint );
  moved_endpoint[1] += 0.25;
  submesh.SetNode( endpoint, moved_endpoint );
  data.UpdateResidualGapField( residual_gap, 0.0 );
  EXPECT_DOUBLE_EQ( ballWeightAtVertex( data, endpoint ), 1.0 );
}

TEST( ResidualGapRamp, DetectsObtuseCornerCreatedByDeformation )
{
  auto parent = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::CShapeMesh( 8, 8, 2 ) );
  mfem::Array<int> attributes{ 2, 3 };
  auto submesh = mfem::ParSubMesh::CreateFromBoundary( parent, attributes );

  int tip = -1;
  for ( int vertex = 0; vertex < submesh.GetNV(); ++vertex ) {
    double x[2];
    submesh.GetNode( vertex, x );
    if ( std::abs( x[0] - 0.25 ) < 1.0e-12 && std::abs( x[1] - 0.75 ) < 1.0e-12 ) {
      tip = vertex;
      break;
    }
  }
  ASSERT_GE( tip, 0 );

  mfem::Array<int> neighbors;
  for ( int element = 0; element < submesh.GetNE(); ++element ) {
    mfem::Array<int> vertices;
    submesh.GetElementVertices( element, vertices );
    if ( vertices[0] == tip ) {
      neighbors.Append( vertices[1] );
    } else if ( vertices[1] == tip ) {
      neighbors.Append( vertices[0] );
    }
  }
  ASSERT_EQ( neighbors.Size(), 2 );

  double neighbor0[2];
  double neighbor1[2];
  submesh.GetNode( neighbors[0], neighbor0 );
  submesh.GetNode( neighbors[1], neighbor1 );
  const double straight_tip[2] = { 0.5 * ( neighbor0[0] + neighbor1[0] ), 0.5 * ( neighbor0[1] + neighbor1[1] ) };
  submesh.SetNode( tip, straight_tip );

  auto fec = std::make_unique<mfem::H1_FECollection>( 1, 2 );
  tribol::MfemSubmeshData data( submesh, nullptr, std::move( fec ), 1, false );
  // Keep the propagation length shorter than an adjacent edge so this test isolates corner eligibility.
  constexpr double residual_gap = 0.005;
  constexpr double pi = std::numbers::pi_v<double>;
  data.UpdateResidualGapField( residual_gap, 10.0 * pi / 180.0 );
  EXPECT_NEAR( residualGapAtVertex( data, tip ), residual_gap, 1.0e-12 );

  const double sharp_tip[2] = { 0.25, 0.75 };
  const double obtuse_tip[2] = { 0.5 * ( straight_tip[0] + sharp_tip[0] ), 0.5 * ( straight_tip[1] + sharp_tip[1] ) };
  submesh.SetNode( tip, obtuse_tip );
  data.UpdateResidualGapField( residual_gap, 10.0 * pi / 180.0 );
  EXPECT_NEAR( residualGapAtVertex( data, tip ), 0.0, 1.0e-12 );

  submesh.SetNode( tip, sharp_tip );
  data.UpdateResidualGapField( residual_gap, 10.0 * pi / 180.0 );
  EXPECT_NEAR( residualGapAtVertex( data, tip ), 0.0, 1.0e-12 );
}

TEST( ResidualGapRamp, ThreeDimensionalSurfaceRetainsUniformGap )
{
  auto parent = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::CubeMesh( 1, 1, 1 ) );
  mfem::Array<int> attributes{ 1, 2 };
  auto submesh = mfem::ParSubMesh::CreateFromBoundary( parent, attributes );
  auto fec = std::make_unique<mfem::H1_FECollection>( 1, 3 );
  tribol::MfemSubmeshData data( submesh, nullptr, std::move( fec ), 1, false );

  constexpr double residual_gap = 0.05;
  constexpr double pi = std::numbers::pi_v<double>;
  data.UpdateResidualGapField( residual_gap, 10.0 * pi / 180.0 );
  for ( int vertex = 0; vertex < submesh.GetNV(); ++vertex ) {
    EXPECT_NEAR( residualGapAtVertex( data, vertex ), residual_gap, 1.0e-12 );
    EXPECT_DOUBLE_EQ( ballWeightAtVertex( data, vertex ), 0.0 );
  }
}

}  // namespace

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
