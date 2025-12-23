// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/physics/Mortar.hpp"

// Axom includes
#include "axom/slic.hpp"

// gtest includes
#include "gtest/gtest.h"

// c++ includes
#include <cmath>  // std::abs

using RealT = tribol::RealT;

/*!
 * Test fixture class with some setup necessary to compute
 * the nonmortar/nonmortar and mortar/nonmortar mortar projection weights
 * in order to test a simple projection between two
 * mis-aligned faces.
 */
class MortarWeightTest : public ::testing::Test {
 public:
  int numNodes;
  int numNodesPerFace;
  int numOverlapNodes;
  int dim;

  RealT* getXCoords( int id )
  {
    if ( id == 0 ) {
      return x1;
    } else {
      return x2;
    }
  }

  RealT* getYCoords( int id )
  {
    if ( id == 0 ) {
      return y1;
    } else {
      return y2;
    }
  }

  RealT* getZCoords( int id )
  {
    if ( id == 0 ) {
      return z1;
    } else {
      return z2;
    }
  }

  RealT* getXOverlapCoords() { return xOverlap; }

  RealT* getYOverlapCoords() { return yOverlap; }

  RealT* getZOverlapCoords() { return zOverlap; }

  void checkMortarWts( RealT pTestNonmortar[4], RealT pTestMortar[4] )
  {
    if ( this->numNodesPerFace != 4 ) {
      SLIC_ERROR( "checkMortarWts: number of nodes per face not equal to 4." );
    }

    tribol::Array2D<RealT> xyz1( this->numNodesPerFace, this->dim );
    tribol::Array2D<RealT> xyz2( this->numNodesPerFace, this->dim );
    tribol::Array2D<RealT> xyzOverlap( this->numOverlapNodes, this->dim );

    // generate stacked coordinate array
    for ( int j = 0; j < this->numNodesPerFace; ++j ) {
      xyz1( j, 0 ) = x1[j];
      xyz1( j, 1 ) = y1[j];
      xyz1( j, 2 ) = z1[j];

      xyz2( j, 0 ) = x2[j];
      xyz2( j, 1 ) = y2[j];
      xyz2( j, 2 ) = z2[j];
    }

    for ( int j = 0; j < this->numOverlapNodes; ++j ) {
      xyzOverlap( j, 0 ) = xOverlap[j];
      xyzOverlap( j, 1 ) = yOverlap[j];
      xyzOverlap( j, 2 ) = zOverlap[j];
    }  // end loop over nodes

    // instantiate SurfaceContactElem struct. Note, this object is instantiated
    // using face 1, face 2, and the set overlap polygon. Note, the mesh ids are set
    // equal to 0, and the face ids are 0 and 1, respectively.
    tribol::SurfaceContactElem elem( this->dim, xyz1.data(), xyz2.data(), xyzOverlap.data(), this->numNodesPerFace,
                                     this->numOverlapNodes, 0, 0, 0, 1 );

    // compute the mortar weights to be stored on
    // the surface contact element struct.
    tribol::ComputeMortarWeights( elem );

    // test the projection on an arbitrary vector
    RealT p[4] = { 1., 1., 1., 1. };
    for ( int i = 0; i < this->numNodesPerFace; ++i ) {
      for ( int j = 0; j < this->numNodesPerFace; ++j ) {
        int nonmortarId = this->numNodesPerFace * i + j;
        int mortarId = this->numNodesPerFace * this->numNodesPerFace + this->numNodesPerFace * i + j;
        pTestNonmortar[i] += elem.mortarWts[nonmortarId] * p[j];
        pTestMortar[i] += elem.mortarWts[mortarId] * p[j];
      }
    }
  }

 protected:
  void SetUp() override
  {
    this->numNodes = 8;
    this->numNodesPerFace = 4;
    this->numOverlapNodes = 4;
    this->dim = 3;

    if ( this->x1 == nullptr ) {
      this->x1 = new RealT[this->numNodes];
    } else {
      delete[] this->x1;
      this->x1 = new RealT[this->numNodes];
    }

    if ( this->x2 == nullptr ) {
      this->x2 = new RealT[this->numNodes];
    } else {
      delete[] this->x2;
      this->x2 = new RealT[this->numNodes];
    }

    if ( this->y1 == nullptr ) {
      this->y1 = new RealT[this->numNodes];
    } else {
      delete[] this->y1;
      this->y1 = new RealT[this->numNodes];
    }

    if ( this->y2 == nullptr ) {
      this->y2 = new RealT[this->numNodes];
    } else {
      delete[] this->y2;
      this->y2 = new RealT[this->numNodes];
    }

    if ( this->z1 == nullptr ) {
      this->z1 = new RealT[this->numNodes];
    } else {
      delete[] this->z1;
      this->z1 = new RealT[this->numNodes];
    }

    if ( this->z2 == nullptr ) {
      this->z2 = new RealT[this->numNodes];
    } else {
      delete[] this->z2;
      this->z2 = new RealT[this->numNodes];
    }

    if ( this->xOverlap == nullptr ) {
      this->xOverlap = new RealT[this->numOverlapNodes];
    } else {
      delete[] this->xOverlap;
      this->xOverlap = new RealT[this->numOverlapNodes];
    }

    if ( this->yOverlap == nullptr ) {
      this->yOverlap = new RealT[this->numOverlapNodes];
    } else {
      delete[] this->yOverlap;
      this->yOverlap = new RealT[this->numOverlapNodes];
    }
    if ( this->zOverlap == nullptr ) {
      this->zOverlap = new RealT[this->numOverlapNodes];
    } else {
      delete[] this->zOverlap;
      this->zOverlap = new RealT[this->numOverlapNodes];
    }
  }

  void TearDown() override
  {
    if ( this->x1 != nullptr ) {
      delete[] this->x1;
      this->x1 = nullptr;
    }
    if ( this->x2 != nullptr ) {
      delete[] this->x2;
      this->x2 = nullptr;
    }
    if ( this->y1 != nullptr ) {
      delete[] this->y1;
      this->y1 = nullptr;
    }
    if ( this->y2 != nullptr ) {
      delete[] this->y2;
      this->y2 = nullptr;
    }
    if ( this->z1 != nullptr ) {
      delete[] this->z1;
      this->z1 = nullptr;
    }
    if ( this->z2 != nullptr ) {
      delete[] this->z2;
      this->z2 = nullptr;
    }
    if ( this->xOverlap != nullptr ) {
      delete[] this->xOverlap;
      this->xOverlap = nullptr;
    }
    if ( this->yOverlap != nullptr ) {
      delete[] this->yOverlap;
      this->yOverlap = nullptr;
    }
    if ( this->zOverlap != nullptr ) {
      delete[] this->zOverlap;
      this->zOverlap = nullptr;
    }
  }

 protected:
  RealT* x1{ nullptr };
  RealT* y1{ nullptr };
  RealT* z1{ nullptr };

  RealT* x2{ nullptr };
  RealT* y2{ nullptr };
  RealT* z2{ nullptr };

  RealT* xOverlap{ nullptr };
  RealT* yOverlap{ nullptr };
  RealT* zOverlap{ nullptr };
};

TEST_F( MortarWeightTest, simple_projection )
{
  RealT* x1 = this->getXCoords( 0 );
  RealT* y1 = this->getYCoords( 0 );
  RealT* z1 = this->getZCoords( 0 );

  RealT* x2 = this->getXCoords( 1 );
  RealT* y2 = this->getYCoords( 1 );
  RealT* z2 = this->getZCoords( 1 );

  RealT* xOvrlp = this->getXOverlapCoords();
  RealT* yOvrlp = this->getYOverlapCoords();
  RealT* zOvrlp = this->getZOverlapCoords();

  x1[0] = -1.;
  x1[1] = -1.;
  x1[2] = 1.;
  x1[3] = 1.;

  y1[0] = 1.;
  y1[1] = -1.;
  y1[2] = -1.;
  y1[3] = 1.;

  z1[0] = 0.1;
  z1[1] = 0.1;
  z1[2] = 0.1;
  z1[3] = 0.1;

  x2[0] = 0.;
  x2[1] = 2.;
  x2[2] = 2.;
  x2[3] = 0.;

  y2[0] = 0.;
  y2[1] = 0.;
  y2[2] = -2.;
  y2[3] = -2.;

  z2[0] = 0.1;
  z2[1] = 0.1;
  z2[2] = 0.1;
  z2[3] = 0.1;

  xOvrlp[0] = 0.;
  xOvrlp[1] = 0.;
  xOvrlp[2] = 1.;
  xOvrlp[3] = 1.;

  yOvrlp[0] = 0.;
  yOvrlp[1] = -1.;
  yOvrlp[2] = -1.;
  yOvrlp[3] = 0.;

  zOvrlp[0] = 0.1;
  zOvrlp[1] = 0.1;
  zOvrlp[2] = 0.1;
  zOvrlp[3] = 0.1;

  RealT pNonmortar[4] = { 0., 0., 0., 0. };
  RealT pMortar[4] = { 0., 0., 0., 0 };
  this->checkMortarWts( pNonmortar, pMortar );

  // hard-code diffs for each element in pNonmortar and pMortar.
  // Note, these hard coded values DEPEND on the ordering of
  // the nodes in this function (above).
  RealT diffNonmortar1 = std::abs( 0.5625 - pNonmortar[0] );
  RealT diffNonmortar2 = std::abs( 0.1875 - pNonmortar[1] );
  RealT diffNonmortar3 = std::abs( 0.0625 - pNonmortar[2] );
  RealT diffNonmortar4 = std::abs( 0.1875 - pNonmortar[3] );

  RealT diffMortar1 = std::abs( 0.0625 - pMortar[0] );
  RealT diffMortar2 = std::abs( 0.1875 - pMortar[1] );
  RealT diffMortar3 = std::abs( 0.5625 - pMortar[2] );
  RealT diffMortar4 = std::abs( 0.1875 - pMortar[3] );

  RealT tol = 1.e-8;
  EXPECT_LE( diffNonmortar1, tol );
  EXPECT_LE( diffNonmortar2, tol );
  EXPECT_LE( diffNonmortar3, tol );
  EXPECT_LE( diffNonmortar4, tol );

  EXPECT_LE( diffMortar1, tol );
  EXPECT_LE( diffMortar2, tol );
  EXPECT_LE( diffMortar3, tol );
  EXPECT_LE( diffMortar4, tol );
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
