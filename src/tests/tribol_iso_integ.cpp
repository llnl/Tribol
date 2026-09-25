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

/** Return the factorial of a nonnegative integer as a Tribol scalar. */
RealT Factorial( int value )
{
  RealT result = 1.;
  for ( int factor = 2; factor <= value; ++factor ) {
    result *= factor;
  }
  return result;
}

/** Return the exact x^p y^q moment on the area-one-half unit right triangle. */
RealT ReferenceTriangleMoment( int first_exponent, int second_exponent )
{
  return Factorial( first_exponent ) * Factorial( second_exponent ) / Factorial( first_exponent + second_exponent + 2 );
}

/** Return the exact moment normalized to the unit-sum triangle-weight convention. */
RealT NormalizedReferenceTriangleMoment( int first_exponent, int second_exponent )
{
  return 2. * ReferenceTriangleMoment( first_exponent, second_exponent );
}

/**
 * @brief Evaluate one polynomial moment with either CommonPlane triangle-rule family.
 *
 * @param use_legacy_rule Whether to use the historical rule instead of the symmetric rule
 * @param order Requested quadrature order
 * @param first_exponent Exponent of the first reference coordinate
 * @param second_exponent Exponent of the second reference coordinate
 * @return Numerically integrated, unit-sum-normalized moment
 */
RealT EvaluateTriangleRuleMoment( bool use_legacy_rule, int order, int first_exponent, int second_exponent )
{
  RealT quadrature_weights[tribol::max_symmetric_triangle_qpts] = { 0. };
  RealT reference_coordinates[2 * tribol::max_symmetric_triangle_qpts] = { 0. };
  const int number_of_quadrature_points =
      use_legacy_rule ? tribol::GetLegacyTriangleRule( order, quadrature_weights, reference_coordinates )
                      : tribol::GetCommonPlaneTriangleRule( order, quadrature_weights, reference_coordinates );

  RealT value = 0.;
  for ( int quadrature_point = 0; quadrature_point < number_of_quadrature_points; ++quadrature_point ) {
    value += quadrature_weights[quadrature_point] *
             std::pow( reference_coordinates[2 * quadrature_point], first_exponent ) *
             std::pow( reference_coordinates[2 * quadrature_point + 1], second_exponent );
  }
  return value;
}

/** Evaluate one monomial moment with a CommonPlane segment quadrature rule. */
RealT EvaluateSegmentRuleMoment( int order, int exponent )
{
  RealT quadrature_weights[tribol::max_segment_gauss_legendre_qpts] = { 0. };
  RealT reference_coordinates[tribol::max_segment_gauss_legendre_qpts] = { 0. };
  const int number_of_quadrature_points =
      tribol::GetCommonPlaneSegmentRule( order, quadrature_weights, reference_coordinates );

  RealT value = 0.;
  for ( int quadrature_point = 0; quadrature_point < number_of_quadrature_points; ++quadrature_point ) {
    value += quadrature_weights[quadrature_point] * std::pow( reference_coordinates[quadrature_point], exponent );
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

/** Verify low-order compatibility between the legacy and symmetric triangle rules. */
TEST( TriangleRuleTest, legacy_and_symmetric_match_on_shared_orders )
{
  // Orders 2 and 4 are supported by both rule implementations and should integrate the
  // same low-order reference-triangle moments.
  for ( int order : { 2, 4 } ) {
    EXPECT_NEAR( EvaluateTriangleRuleMoment( true, order, 0, 0 ), EvaluateTriangleRuleMoment( false, order, 0, 0 ),
                 2.e-10 );
    EXPECT_NEAR( EvaluateTriangleRuleMoment( true, order, 2, 0 ), EvaluateTriangleRuleMoment( false, order, 2, 0 ),
                 2.e-10 );
    EXPECT_NEAR( EvaluateTriangleRuleMoment( true, order, 1, 1 ), EvaluateTriangleRuleMoment( false, order, 1, 1 ),
                 2.e-10 );
  }
}

/** Verify polynomial exactness for every supported CommonPlane segment rule. */
TEST( SegmentRuleTest, integrates_polynomials_through_each_supported_order )
{
  // An n-point Gauss-Legendre rule integrates every polynomial through degree
  // 2n-1 exactly. Exercising each exposed order also validates its point count.
  constexpr RealT integration_tolerance = 2.e-14;
  for ( int order = 2; order <= 10; ++order ) {
    for ( int exponent = 0; exponent <= 2 * order - 1; ++exponent ) {
      const RealT exact_moment = 1. / static_cast<RealT>( exponent + 1 );
      EXPECT_NEAR( EvaluateSegmentRuleMoment( order, exponent ), exact_moment, integration_tolerance )
          << "quadrature order " << order << ", polynomial exponent " << exponent;
    }
  }
}

/** Verify polynomial exactness for every supported symmetric triangle rule. */
TEST( TriangleRuleTest, integrates_polynomials_through_each_supported_order )
{
  // The symmetric order-p rule must reproduce every reference-triangle monomial
  // whose total degree does not exceed p. This validates every table from 2–10.
  constexpr RealT integration_tolerance = 5.e-14;
  for ( int order = 2; order <= 10; ++order ) {
    for ( int first_exponent = 0; first_exponent <= order; ++first_exponent ) {
      for ( int second_exponent = 0; second_exponent <= order - first_exponent; ++second_exponent ) {
        const RealT exact_moment = NormalizedReferenceTriangleMoment( first_exponent, second_exponent );
        EXPECT_NEAR( EvaluateTriangleRuleMoment( false, order, first_exponent, second_exponent ), exact_moment,
                     integration_tolerance )
            << "quadrature order " << order << ", polynomial exponents " << first_exponent << " and "
            << second_exponent;
      }
    }
  }
}

/** Verify polygon-fan integration with the highest supported triangle rule. */
TEST( TriangleRuleTest, gauss_poly_int_tri_supports_order_10 )
{
  // CommonPlane triangle-decomposition integration supports the order-10 symmetric rule
  // on triangular overlap facets in 3D.
  constexpr int spatial_dimension = 3;
  constexpr int number_of_nodes = 3;
  RealT coordinates[spatial_dimension * number_of_nodes] = { 0., 0., 0., 1., 0., 0., 0., 1., 0. };

  tribol::SurfaceContactElem contact_element( spatial_dimension, coordinates, coordinates, coordinates, number_of_nodes,
                                              number_of_nodes, nullptr, nullptr, 0, 0 );
  tribol::IntegPts integration_points;
  tribol::GaussPolyIntTri( contact_element, integration_points, 10 );

  RealT area = 0.;
  RealT seventh_third_moment = 0.;
  for ( int integration_point = 0; integration_point < integration_points.numIPs; ++integration_point ) {
    const RealT x_coordinate = integration_points.xy[spatial_dimension * integration_point];
    const RealT y_coordinate = integration_points.xy[spatial_dimension * integration_point + 1];
    area += integration_points.wts[integration_point];
    seventh_third_moment +=
        integration_points.wts[integration_point] * std::pow( x_coordinate, 7 ) * std::pow( y_coordinate, 3 );
  }

  EXPECT_EQ( integration_points.numIPs, 75 );
  EXPECT_NEAR( area, 0.5, 1.e-14 );
  // Integral of x^7 y^3 over the physical unit right triangle.
  EXPECT_NEAR( seventh_third_moment, ReferenceTriangleMoment( 7, 3 ), 1.e-14 );
  EXPECT_NEAR( EvaluateTriangleRuleMoment( false, 10, 7, 3 ), NormalizedReferenceTriangleMoment( 7, 3 ), 1.e-14 );
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
