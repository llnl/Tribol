// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/types.hpp"
#include "tribol/utils/Math.hpp"

// Axom includes
#include "axom/slic.hpp"

// gtest includes
#include "gtest/gtest.h"

// c++ includes
#include <cmath>

using real = tribol::real;

/*!
 *  Test fixture class to test math functions
 */
class MathTest : public ::testing::Test {
 public:
  void get3DVectorComps( real& x, real& y, real& z )
  {
    x = m_vx;
    y = m_vy;
    z = m_vz;
  }

  void set3DVectorComps( real vx, real vy, real vz )
  {
    m_vx = vx;
    m_vy = vy;
    m_vz = vz;
  }

  real get3DMag() { return m_3VecMag; }

  void setSecond3DVectorComps( real vx, real vy, real vz )
  {
    m_ux = vx;
    m_uy = vy;
    m_uz = vz;
  }

  void getSecond3DVectorComps( real& x, real& y, real& z )
  {
    x = m_ux;
    y = m_uy;
    z = m_uz;
  }

  real getSecond3DMag() { return m_Second3VecMag; }

  void get2DVectorComps( real& x, real& y )
  {
    x = m_wx;
    y = m_wy;
  }

  void set2DVectorComps( real vx, real vy )
  {
    m_wx = vx;
    m_wy = vy;
  }

  real get2DMag() { return m_2VecMag; }

  void setTol( real tol ) { m_tol = tol; }

  real getTol() { return m_tol; }

 protected:
  real m_tol{ 1.E-8 };

  // components of a three dimensional vector
  real m_vx{ 1.5 };
  real m_vy{ 2.32 };
  real m_vz{ 3.1 };
  real m_3VecMag{ 4.152396898 };

  // components of another three dimensional vector
  real m_ux{ 2.1 };
  real m_uy{ 4.23 };
  real m_uz{ 5.35 };
  real m_Second3VecMag{ 7.136203472 };

  // components of a two dimensional vector
  real m_wx{ 1.5 };
  real m_wy{ 2.32 };
  real m_2VecMag{ 2.762679858 };
};

TEST_F( MathTest, magnitude )
{
  real vx;
  real vy;
  real vz;
  get3DVectorComps( vx, vy, vz );

  real mag1 = tribol::magnitude( vx, vy, vz );

  real diff1 = std::abs( mag1 - get3DMag() );

  EXPECT_LE( diff1, getTol() );

  real wx, wy;
  get2DVectorComps( wx, wy );

  real mag2 = tribol::magnitude( wx, wy );

  real diff2 = std::abs( mag2 - get2DMag() );

  EXPECT_LE( diff2, getTol() );
}

TEST_F( MathTest, dotProd )
{
  real vx;
  real vy;
  real vz;
  get3DVectorComps( vx, vy, vz );

  real prod = tribol::dotProd( vx, vy, vz, vx, vy, vz );

  real diff = std::abs( prod - ( vx * vx + vy * vy + vz * vz ) );

  EXPECT_LE( diff, getTol() );

  // test pointer input to dot product routine
  real vec[3]{
      0.,
      0.,
      0.,
  };
  real* w;
  w = vec;
  w[0] = vx;
  w[1] = vy;
  w[2] = vz;

  real prod2 = tribol::dotProd( w, w, 3 );

  real diff2 = std::abs( prod2 - ( vx * vx + vy * vy + vz * vz ) );

  EXPECT_LE( diff2, getTol() );
}

TEST_F( MathTest, crossProd )
{
  real vx, vy, vz;
  real ux, uy, uz;
  get3DVectorComps( vx, vy, vz );
  getSecond3DVectorComps( ux, uy, uz );

  real prod_x, prod_y, prod_z;
  tribol::crossProd( vx, vy, vz, ux, uy, uz, prod_x, prod_y, prod_z );

  real vec_mag = tribol::magnitude( prod_x, prod_y, prod_z );

  real prod_test_i = vy * uz - vz * uy;
  real prod_test_j = vz * ux - vx * uz;
  real prod_test_k = vx * uy - vy * ux;
  real mag_test = tribol::magnitude( prod_test_i, prod_test_j, prod_test_k );
  real diff = std::abs( vec_mag - mag_test );

  EXPECT_LE( diff, getTol() );

  // test pointer input to dot product routine
  real a[3]{ 0., 0., 0. };
  real b[3]{ 0., 0., 0. };
  a[0] = vx;
  a[1] = vy;
  a[2] = vz;
  b[0] = ux;
  b[1] = uy;
  b[2] = uz;

  vec_mag = tribol::magCrossProd( a, b );

  diff = std::abs( vec_mag - mag_test );

  EXPECT_LE( diff, getTol() );
}

TEST_F( MathTest, sort_and_search )
{
  // define integer array. Note, introduce a duplicate entry
  int a[6]{ 1, 12, 8, 2, 7, 2 };

  // sort array in increasing order
  tribol::bubble_sort( &a[0], 6 );

  EXPECT_EQ( a[0], 1 );
  EXPECT_EQ( a[1], 2 );
  EXPECT_EQ( a[2], 2 );
  EXPECT_EQ( a[3], 7 );
  EXPECT_EQ( a[4], 8 );
  EXPECT_EQ( a[5], 12 );

  // search array for integer value = 7
  real index = tribol::binary_search( &a[0], 6, 7 );
  EXPECT_EQ( index, 3 );

  // search array for integer value = 2
  index = tribol::binary_search( &a[0], 6, 2 );

  // the search will find only one of the duplicate entries
  if ( index == 1 ) {
    EXPECT_EQ( index, 1 );
  } else if ( index == 2 ) {
    EXPECT_EQ( index, 2 );
  } else {
    EXPECT_EQ( index, 1 );  // will fail if here
  }

  // search for faulty value
  index = tribol::binary_search( &a[0], 6, 21 );
  EXPECT_EQ( index, -1 );
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
