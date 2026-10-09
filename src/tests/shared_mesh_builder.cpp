// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "gtest/gtest.h"

#include <array>

#include "axom/slic/core/SimpleLogger.hpp"
#include "shared/mesh/MeshBuilder.hpp"

TEST( MeshBuilder, Stitch )
{
  auto builder = shared::MeshBuilder::Stitch( {
      shared::MeshBuilder::SquareMesh( 1, 1 ),
      shared::MeshBuilder::SquareMesh( 1, 1 ).translate( { 1.0, 0.0 } ),
  } );
  const auto& mesh = static_cast<const mfem::Mesh&>( builder );

  EXPECT_EQ( mesh.GetNE(), 2 );
  EXPECT_EQ( mesh.GetNV(), 6 );
  EXPECT_EQ( mesh.GetNBE(), 6 );
}

TEST( MeshBuilder, StitchRejectsMismatchedDimensionsBeforeUnifying )
{
  axom::slic::SimpleLogger logger;
  axom::slic::ScopedAbortToThrow abort_to_throw;
  EXPECT_THROW( shared::MeshBuilder::Stitch(
                    { shared::MeshBuilder::SquareMesh( 1, 1 ), shared::MeshBuilder::CubeMesh( 1, 1, 1 ) } ),
                axom::slic::SlicAbortException );
}

TEST( MeshBuilder, StitchRejectsHigherOrderMeshBeforeUnifying )
{
  axom::slic::SimpleLogger logger;
  axom::slic::ScopedAbortToThrow abort_to_throw;
  auto quadratic = shared::MeshBuilder::SquareMesh( 1, 1 );
  static_cast<mfem::Mesh&>( quadratic ).SetCurvature( 2 );
  EXPECT_THROW( shared::MeshBuilder::Stitch( { shared::MeshBuilder::SquareMesh( 1, 1 ), std::move( quadratic ) } ),
                axom::slic::SlicAbortException );
}

TEST( MeshBuilder, CShapeMesh )
{
  auto builder = shared::MeshBuilder::CShapeMesh( 4, 4, 1 );
  const auto& mesh = static_cast<const mfem::Mesh&>( builder );

  EXPECT_EQ( mesh.GetNE(), 10 );
  EXPECT_EQ( mesh.GetNV(), 22 );
  EXPECT_EQ( mesh.GetNBE(), 22 );

  const std::array<int, 7> expected_bdr_element_counts = { 4, 3, 2, 3, 2, 4, 4 };
  std::array<int, 7> bdr_element_counts{};
  for ( int i = 0; i < mesh.GetNBE(); ++i ) {
    ++bdr_element_counts[mesh.GetBdrAttribute( i ) - 1];

    mfem::Array<int> vertices;
    mesh.GetBdrElementVertices( i, vertices );
    for ( int j = 0; j < vertices.Size(); ++j ) {
      std::array<double, 2> coordinate;
      mesh.GetNode( vertices[j], coordinate.data() );
      switch ( mesh.GetBdrAttribute( i ) ) {
        case 1:
          EXPECT_DOUBLE_EQ( coordinate[0], 0.0 );
          break;
        case 2:
          EXPECT_DOUBLE_EQ( coordinate[1], 0.75 );
          break;
        case 3:
          EXPECT_DOUBLE_EQ( coordinate[0], 0.25 );
          break;
        case 4:
          EXPECT_DOUBLE_EQ( coordinate[1], 0.25 );
          break;
        case 5:
          EXPECT_DOUBLE_EQ( coordinate[0], 1.0 );
          break;
        case 6:
          EXPECT_DOUBLE_EQ( coordinate[1], 1.0 );
          break;
        case 7:
          EXPECT_DOUBLE_EQ( coordinate[1], 0.0 );
          break;
      }
    }
  }
  EXPECT_EQ( bdr_element_counts, expected_bdr_element_counts );

  const std::array<int, 3> expected_element_counts = { 4, 3, 3 };
  std::array<int, 3> element_counts{};
  for ( int i = 0; i < mesh.GetNE(); ++i ) {
    ++element_counts[mesh.GetAttribute( i ) - 1];
  }
  EXPECT_EQ( element_counts, expected_element_counts );
}
