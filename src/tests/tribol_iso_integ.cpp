// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// c++ includes
#include <cmath>  // std::abs

// gtest includes
#include "gtest/gtest.h"

// Tribol includes
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/integ/Integration.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/integ/FE.hpp"

using RealT = tribol::RealT;

namespace {

RealT factorial( int n )
{
  RealT result = 1.;
  for ( int i = 2; i <= n; ++i ) {
    result *= i;
  }
  return result;
}

RealT referenceTriangleMoment( int px, int py )
{
  return factorial( px ) * factorial( py ) / factorial( px + py + 2 );
}

RealT normalizedReferenceTriangleMoment( int px, int py )
{
  return 2. * referenceTriangleMoment( px, py );
}

RealT evalTriangleRuleMoment( bool use_legacy, int order, int px, int py )
{
  RealT wts[tribol::max_symmetric_triangle_qpts] = { 0. };
  RealT coords[2 * tribol::max_symmetric_triangle_qpts] = { 0. };
  const int num_qpts =
      use_legacy ? tribol::GetLegacyTriangleRule( order, wts, coords ) : tribol::GetCommonPlaneTriangleRule( order, wts, coords );

  RealT value = 0.;
  for ( int qp = 0; qp < num_qpts; ++qp ) {
    value += wts[qp] * std::pow( coords[2 * qp], px ) * std::pow( coords[2 * qp + 1], py );
  }
  return value;
}

}  // namespace

/*!
 * Test fixture class with some setup necessary to use the
 * triangular decomposition of a quadrilateral with integration
 * points specified on each triangle's parent space, forward mapped
 * to the physical triangle, and then mapped using the inverse
 * isoparametric mapping in order to obtain (xi,eta) coordinates
 * on the parent four node quad. These tests compute the area
 * as calculated by summing integrals of shape functions defined
 * on the four node quad.
 */
class IsoIntegTest : public ::testing::Test {
 public:
  int numNodes;
  static constexpr int dim = 3;

  RealT* getXCoords() { return x; }

  RealT* getYCoords() { return y; }

  RealT* getZCoords() { return z; }

  bool integrate( RealT const tol )
  {
    tribol::Array2D<RealT> xyz( this->numNodes, dim );

    // generate stacked coordinate array
    for ( int j = 0; j < this->numNodes; ++j ) {
      xyz( j, 0 ) = x[j];
      xyz( j, 1 ) = y[j];
      xyz( j, 2 ) = z[j];
    }  // end loop over nodes

    // instantiate SurfaceContactElem struct. Note, this object is instantiated
    // using face 1 as face 2, but these faces are not used in this test so this
    // is ok.
    tribol::SurfaceContactElem elem( this->dim, xyz.data(), xyz.data(), xyz.data(), this->numNodes, this->numNodes,
                                     nullptr, nullptr, 0, 0 );

    // instantiate integration object
    tribol::IntegPts integ;

    // generate all current configuration integration point coordinates and weights
    tribol::GaussPolyIntTri( elem, integ, 2 );

    // evaluate sum_a (integral_face (phi_a) da) with outer loop over nodes, a, and
    // inner loop over number of integration points
    RealT areaTest = 0.;
    RealT phi = 0.;

    for ( int a = 0; a < this->numNodes; ++a ) {
      for ( int ip = 0; ip < integ.numIPs; ++ip ) {
        // perform inverse isoparametric mapping of current configuration
        // integration point to four node quad parent space
        RealT xp[3] = { integ.xy[dim * ip], integ.xy[dim * ip + 1], integ.xy[dim * ip + 2] };
        RealT xi[2] = { 0., 0. };
        tribol::InvIso( xp, x, y, z, this->numNodes, xi );
        tribol::LinIsoQuadShapeFunc( xi[0], xi[1], a, phi );

        areaTest += integ.wts[ip] * phi;
      }
    }

    RealT area = tribol::Area2DPolygon( x, y, this->numNodes );

    bool convrg = ( std::abs( areaTest - area ) <= tol ) ? true : false;

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

 protected:
  RealT* x{ nullptr };
  RealT* y{ nullptr };
  RealT* z{ nullptr };
};

TEST_F( IsoIntegTest, square )
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

  bool convrg = this->integrate( 1.e-8 );

  EXPECT_EQ( convrg, true );
}

TEST_F( IsoIntegTest, rect )
{
  RealT* x = this->getXCoords();
  RealT* y = this->getYCoords();
  RealT* z = this->getZCoords();

  x[0] = -0.5;
  x[1] = 0.5;
  x[2] = 0.5;
  x[3] = -0.5;

  y[0] = -0.25;
  y[1] = -0.25;
  y[2] = 0.25;
  y[3] = 0.25;

  z[0] = 0.1;
  z[1] = 0.1;
  z[2] = 0.1;
  z[3] = 0.1;

  bool convrg = this->integrate( 1.e-8 );

  EXPECT_EQ( convrg, true );
}

TEST_F( IsoIntegTest, affine )
{
  RealT* x = this->getXCoords();
  RealT* y = this->getYCoords();
  RealT* z = this->getZCoords();

  x[0] = -0.5;
  x[1] = 0.5;
  x[2] = 0.8;
  x[3] = -0.2;

  y[0] = -0.415;
  y[1] = -0.415;
  y[2] = 0.5;
  y[3] = 0.5;

  z[0] = 0.1;
  z[1] = 0.1;
  z[2] = 0.1;
  z[3] = 0.1;

  bool convrg = integrate( 1.e-8 );

  EXPECT_EQ( convrg, true );
}

TEST_F( IsoIntegTest, nonaffine )
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

  // note slightly lower convergence tol for nonaffinely
  // mapped quad
  bool convrg = integrate( 1.e-5 );

  EXPECT_EQ( convrg, true );
}

TEST( TriangleRuleTest, legacy_and_symmetric_match_on_shared_orders )
{
  for ( int order : { 2, 4 } ) {
    EXPECT_NEAR( evalTriangleRuleMoment( true, order, 0, 0 ), evalTriangleRuleMoment( false, order, 0, 0 ), 2.e-10 );
    EXPECT_NEAR( evalTriangleRuleMoment( true, order, 2, 0 ), evalTriangleRuleMoment( false, order, 2, 0 ), 2.e-10 );
    EXPECT_NEAR( evalTriangleRuleMoment( true, order, 1, 1 ), evalTriangleRuleMoment( false, order, 1, 1 ), 2.e-10 );
  }
}

TEST( TriangleRuleTest, gauss_poly_int_tri_supports_order_10 )
{
  constexpr int dim = 3;
  constexpr int num_nodes = 3;
  RealT xyz[dim * num_nodes] = { 0., 0., 0., 1., 0., 0., 0., 1., 0. };

  tribol::SurfaceContactElem elem( dim, xyz, xyz, xyz, num_nodes, num_nodes, nullptr, nullptr, 0, 0 );
  tribol::IntegPts integ;
  tribol::GaussPolyIntTri( elem, integ, 10 );

  RealT area = 0.;
  RealT moment73 = 0.;
  for ( int ip = 0; ip < integ.numIPs; ++ip ) {
    const RealT x = integ.xy[dim * ip];
    const RealT y = integ.xy[dim * ip + 1];
    area += integ.wts[ip];
    moment73 += integ.wts[ip] * std::pow( x, 7 ) * std::pow( y, 3 );
  }

  EXPECT_EQ( integ.numIPs, 75 );
  EXPECT_NEAR( area, 0.5, 1.e-14 );
  EXPECT_NEAR( moment73, referenceTriangleMoment( 7, 3 ), 1.e-14 );
  EXPECT_NEAR( evalTriangleRuleMoment( false, 10, 7, 3 ), normalizedReferenceTriangleMoment( 7, 3 ), 1.e-14 );
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
