// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/integ/Integration.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/Math.hpp"

// Axom includes
#include "axom/slic.hpp"

// gtest includes
#include "gtest/gtest.h"

using RealT = tribol::RealT;

/*!
 * Test fixture class with some setup necessary to use the
 * inverse isoparametric routine
 */
class InvIsoTest : public ::testing::Test {
 public:
  int numNodes;

  RealT* getXCoords() { return this->x; }

  RealT* getYCoords() { return this->y; }

  RealT* getZCoords() { return this->z; }

  bool InvMap( RealT point[3], RealT const tol )
  {
    RealT x_sol[2];
    tribol::InvIso( point, this->x, this->y, this->z, 4, x_sol );

    // test (xi,eta) obtained from inverse isoparametric
    // mapping by performing forward map of that point and
    // compare to original point.
    RealT map_point[3] = { 0., 0, 0 };
    tribol::FwdMapLinQuad( x_sol, this->x, this->y, this->z, map_point );

    bool convrg = false;
    RealT res =
        tribol::magnitude( ( point[0] - map_point[0] ), ( point[1] - map_point[1] ), ( point[2] - map_point[2] ) );

    if ( res < tol ) {
      convrg = true;
    }

    return convrg;
  }

 protected:
  void SetUp() override
  {
    this->numNodes = 4;
    if ( this->x == nullptr ) {
      this->x = new RealT[this->numNodes];
    } else {
      delete[] this->x;
      this->x = new RealT[this->numNodes];
    }

    if ( this->y == nullptr ) {
      this->y = new RealT[this->numNodes];
    } else {
      delete[] this->y;
      this->y = new RealT[this->numNodes];
    }

    if ( this->z == nullptr ) {
      this->z = new RealT[this->numNodes];
    } else {
      delete[] this->z;
      this->z = new RealT[this->numNodes];
    }
  }

  void TearDown() override
  {
    if ( this->x != nullptr ) {
      delete[] this->x;
      this->x = nullptr;
    }
    if ( this->y != nullptr ) {
      delete[] this->y;
      this->y = nullptr;
    }
    if ( this->z != nullptr ) {
      delete[] this->z;
      this->z = nullptr;
    }
  }

  /**
   * @brief Given a coordinate point xi in the reference triangle, compute the corresponding physical coordinate. Then
   * verify InvIso maps the coordinate back to xi.
   *
   * @param x0 The zero-th coordinate of the triangle (3D, CCW ordering).
   * @param x1 The first coordinate of the triangle (3D, CCW ordering).
   * @param x2 The second coordinate of the triangle (3D, CCW ordering).
   * @param xi The point to test (xi \in [0, 1], eta \in [0, 1 - xi]).
   */
  void RunTriTest( const tribol::RealT* x0, const tribol::RealT* x1, const tribol::RealT* x2, const tribol::RealT* xi )
  {
    tribol::RealT x[3] = { 0.0, 0.0, 0.0 };
    tribol::RealT phi[3] = { 0.0, 0.0, 0.0 };
    tribol::LinIsoTriShapeFunc( xi, phi );
    for ( int i{ 0 }; i < 3; ++i ) {
      x[i] += phi[0] * x0[i] + phi[1] * x1[i] + phi[2] * x2[i];
    }
    tribol::RealT xi_inviso[2] = { 0.0, 0.0 };
    tribol::RealT xA[3] = { x0[0], x1[0], x2[0] };
    tribol::RealT yA[3] = { x0[1], x1[1], x2[1] };
    tribol::RealT zA[3] = { x0[2], x1[2], x2[2] };
    tribol::InvIso( x, xA, yA, zA, 3, xi_inviso );
    EXPECT_NEAR( xi_inviso[0], xi[0], 1.e-12 );
    EXPECT_NEAR( xi_inviso[1], xi[1], 1.e-12 );
  }

 protected:
  RealT* x{ nullptr };
  RealT* y{ nullptr };
  RealT* z{ nullptr };
};

TEST_F( InvIsoTest, nonaffine_centroid )
{
  RealT* x = this->getXCoords();
  RealT* y = this->getYCoords();
  RealT* z = this->getZCoords();

  x[0] = -0.5;
  x[1] = 0.5;
  x[2] = 0.235;
  x[3] = -0.35;

  y[0] = -0.25;
  y[1] = -0.15;
  y[2] = 0.25;
  y[3] = 0.235;

  z[0] = 0.1;
  z[1] = 0.1;
  z[2] = 0.1;
  z[3] = 0.1;

  RealT point[3];

  // initialize physical point array
  for ( int i = 0; i < 3; ++i ) {
    point[i] = 0.;
  }

  // generate physical space point to be mapped as
  // vertex averaged centroid of quad
  for ( int i = 0; i < 4; ++i ) {
    point[0] += x[i];
    point[1] += y[i];
    point[2] += z[i];
  }

  // divide by number of nodes
  point[0] /= 4;
  point[1] /= 4;
  point[2] /= 4;

  bool convrg = this->InvMap( point, 1.e-6 );

  EXPECT_EQ( convrg, true );
}

TEST_F( InvIsoTest, nonaffine_test_point )
{
  RealT* x = this->getXCoords();
  RealT* y = this->getYCoords();
  RealT* z = this->getZCoords();

  x[0] = -0.5;
  x[1] = 0.5;
  x[2] = 0.235;
  x[3] = -0.35;

  y[0] = -0.25;
  y[1] = -0.15;
  y[2] = 0.25;
  y[3] = 0.235;

  z[0] = 0.1;
  z[1] = 0.1;
  z[2] = 0.1;
  z[3] = 0.1;

  // hard code point
  RealT point[3] = { 0.215, 0.116, 0.1 };

  bool convrg = this->InvMap( point, 1.e-6 );

  EXPECT_EQ( convrg, true );
}

TEST_F( InvIsoTest, affine_test_point )
{
  RealT* x = this->getXCoords();
  RealT* y = this->getYCoords();
  RealT* z = this->getZCoords();

  x[0] = -0.5;
  x[1] = 0.5;
  x[2] = 0.5;
  x[3] = -0.5;

  y[0] = -0.5;
  y[1] = -0.5;
  y[2] = 0.5;
  y[3] = 0.5;

  z[0] = 0.1;
  z[1] = 0.1;
  z[2] = 0.1;
  z[3] = 0.1;

  RealT point[3];

  // hard-code point
  point[0] = 0.25;
  point[1] = 0.25;
  point[2] = 0.1;

  bool convrg = this->InvMap( point, 1.e-6 );

  EXPECT_EQ( convrg, true );
}

/*
       x2
       /\
      /  \
     /    \
    /      \
   /        \
  x0---------x1

  x0 = (0.0, 0.0, 0.0)
  x1 = (1.0, 0.0, 0.0)
  x2 = (0.5, 1.0, 0.0)
*/
TEST_F( InvIsoTest, basic_tri_test )
{
  RealT x0[3] = { 0.0, 0.0, 0.0 };
  RealT x1[3] = { 1.0, 0.0, 0.0 };
  RealT x2[3] = { 0.5, 1.0, 0.0 };
  // clang-format off
  RealT xi[14] = { 0.25, 0.25,
                  0.0, 0.0,
                  1.0, 0.0,
                  0.0, 1.0,
                  0.5, 0.5,
                  0.33333, 0.33333,
                  0.12352634, 0.2345422 };
  // clang-format on

  for ( int i{ 0 }; i < 7; ++i ) {
    this->RunTriTest( x0, x1, x2, xi + 2 * i );
  }
}

// tests a narrower triangle that is out of the xy plane
TEST_F( InvIsoTest, narrow_offaxis_tri_test )
{
  RealT x0[3] = { 0.2, -0.2, 0.5 };
  RealT x1[3] = { 1.0, 0.0, 0.5 };
  RealT x2[3] = { 0.5, 0.1, 0.0 };
  // clang-format off
  RealT xi[14] = { 0.25, 0.25,
                  0.0, 0.0,
                  1.0, 0.0,
                  0.0, 1.0,
                  0.5, 0.5,
                  0.33333, 0.33333,
                  0.12352634, 0.2345422 };
  // clang-format on

  for ( int i{ 0 }; i < 7; ++i ) {
    this->RunTriTest( x0, x1, x2, xi + 2 * i );
  }
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
