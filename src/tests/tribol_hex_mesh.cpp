// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/utils/TestUtils.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/common/Parameters.hpp"

// Axom includes
#include "axom/slic.hpp"

// gtest includes
#include "gtest/gtest.h"

// c++ includes
#include <cmath>  // std::abs, std::cos, std::sin
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

using RealT = tribol::RealT;

/*!
 * Test fixture class with some setup necessary to test
 * the TestMesh hex mesh feature
 */
class HexMeshTest : public ::testing::Test {
 public:
  tribol::TestMesh m_mesh;

 protected:
  void SetUp() override {}

  void TearDown() override { this->m_mesh.clear(); }

 protected:
};

TEST_F( HexMeshTest, build_and_check_hex_mesh )
{
  this->m_mesh.mortarMeshId = 0;
  this->m_mesh.nonmortarMeshId = 1;

  int nMortarElems = 4;
  int nElemsXM = nMortarElems;
  int nElemsYM = nMortarElems;
  int nElemsZM = nMortarElems;

  int nNonmortarElems = 3;
  int nElemsXS = nNonmortarElems;
  int nElemsYS = nNonmortarElems;
  int nElemsZS = nNonmortarElems;

  // mesh bounding box with 0.1 interpenetration gap
  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.05;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 0.95;
  RealT x_max2 = 1.;
  RealT y_max2 = 1.;
  RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 5.,
                                    -5. );

  // sanity checks for this specific mesh
  EXPECT_EQ( this->m_mesh.mesh_constructed, true );
  EXPECT_EQ( this->m_mesh.cellType, (int)( tribol::LINEAR_QUAD ) );
  EXPECT_EQ( this->m_mesh.numNodesPerFace, 4 );
  EXPECT_EQ( this->m_mesh.numNodesPerElement, 8 );
  EXPECT_EQ( this->m_mesh.numMortarElements, nMortarElems * nMortarElems * nMortarElems );
  EXPECT_EQ( this->m_mesh.numNonmortarElements, nNonmortarElems * nNonmortarElems * nNonmortarElems );
  EXPECT_EQ( this->m_mesh.numTotalElements, this->m_mesh.numMortarElements + this->m_mesh.numNonmortarElements );
  EXPECT_EQ( this->m_mesh.numMortarFaces, nMortarElems * nMortarElems );
  EXPECT_EQ( this->m_mesh.numNonmortarFaces, nNonmortarElems * nNonmortarElems );
  EXPECT_EQ( this->m_mesh.numMortarNodes, ( nMortarElems + 1 ) * ( nMortarElems + 1 ) * ( nMortarElems + 1 ) );
  EXPECT_EQ( this->m_mesh.numNonmortarNodes,
             ( nNonmortarElems + 1 ) * ( nNonmortarElems + 1 ) * ( nNonmortarElems + 1 ) );

  // this->m_mesh.testMeshToVtk( "", 1, 1 );
}

TEST_F( HexMeshTest, build_and_check_mfem_hex_mesh )
{
  this->m_mesh.mortarMeshId = 0;
  this->m_mesh.nonmortarMeshId = 1;

  int nMortarElems = 5;
  int nElemsXM = nMortarElems;
  int nElemsYM = nMortarElems;
  int nElemsZM = nMortarElems;

  int nNonmortarElems = 4;
  int nElemsXS = nNonmortarElems;
  int nElemsYS = nNonmortarElems;
  int nElemsZS = nNonmortarElems;

  // mesh bounding box with 0.1 interpenetration gap
  RealT x_min1 = 0.;
  RealT y_min1 = 0.;
  RealT z_min1 = 0.;
  RealT x_max1 = 1.;
  RealT y_max1 = 1.;
  RealT z_max1 = 1.05;

  RealT x_min2 = 0.;
  RealT y_min2 = 0.;
  RealT z_min2 = 0.95;
  RealT x_max2 = 1.;
  RealT y_max2 = 1.;
  RealT z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 5.,
                                    -5. );

  // setup mfem mesh, but don't fix orientations in order to
  // check if underlying TestMesh hex mesh has orientation issues
  this->m_mesh.setupMfemMesh( false );

  // perform mfem mesh based sanity checks
  EXPECT_EQ( this->m_mesh.mfem_mesh->SpaceDimension(), 3 );
  SLIC_INFO( "number of elements: " << this->m_mesh.mfem_mesh->GetNE() << "." );
  EXPECT_EQ( this->m_mesh.mfem_mesh->GetNE(), this->m_mesh.numMortarElements + this->m_mesh.numNonmortarElements );
  EXPECT_EQ( this->m_mesh.mfem_mesh->GetNV(), this->m_mesh.numMortarNodes + this->m_mesh.numNonmortarNodes );
  EXPECT_EQ( this->m_mesh.mfem_mesh->GetNFbyType( mfem::FaceType::Boundary ),
             6 * ( nMortarElems * nMortarElems + nNonmortarElems * nNonmortarElems ) );

  // check for inverted elements
  int wrong_elem_orientation = -1;
  int wrong_bdr_elem_orientation = -1;
  bool fix_it = false;
  wrong_elem_orientation = this->m_mesh.mfem_mesh->CheckElementOrientation( fix_it );
  wrong_bdr_elem_orientation = this->m_mesh.mfem_mesh->CheckBdrElementOrientation( fix_it );
  EXPECT_EQ( wrong_elem_orientation, 0 );
  EXPECT_EQ( wrong_bdr_elem_orientation, 0 );

  // std::ofstream mfem_file("mfem_test_mesh.vtk");
  // this->m_mesh.mfem_mesh->PrintVTK(mfem_file);
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;
  result = RUN_ALL_TESTS();

  return result;
}
