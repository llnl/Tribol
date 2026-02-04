// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/integ/FE.hpp"

#include "tribol/utils/Math.hpp"

namespace tribol {

void FwdMapLinTri( const RealT xi[2], RealT xa[3], RealT ya[3], RealT za[3], RealT x[3] )
{
  // initialize output array
  initRealArray( &x[0], 3, 0. );

  // obtain shape function evaluations at (xi,eta)
  RealT phi[3] = { 0., 0., 0. };
  LinIsoTriShapeFunc( xi[0], xi[1], 0, phi[0] );
  LinIsoTriShapeFunc( xi[0], xi[1], 1, phi[1] );
  LinIsoTriShapeFunc( xi[0], xi[1], 2, phi[2] );

  for ( int j = 0; j < 3; ++j ) {
    x[0] += xa[j] * phi[j];
    x[1] += ya[j] * phi[j];
    x[2] += za[j] * phi[j];
  }
  return;
}

//------------------------------------------------------------------------------
void FwdMapLinQuad( const RealT xi[2], RealT xa[4], RealT ya[4], RealT za[4], RealT x[3] )
{
  // initialize output array
  initRealArray( &x[0], 3, 0. );

  // obtain shape function evaluations at (xi,eta)
  RealT phi[4] = { 0., 0., 0., 0. };
  LinIsoQuadShapeFunc( xi[0], xi[1], 0, phi[0] );
  LinIsoQuadShapeFunc( xi[0], xi[1], 1, phi[1] );
  LinIsoQuadShapeFunc( xi[0], xi[1], 2, phi[2] );
  LinIsoQuadShapeFunc( xi[0], xi[1], 3, phi[3] );

  for ( int j = 0; j < 4; ++j ) {
    x[0] += xa[j] * phi[j];
    x[1] += ya[j] * phi[j];
    x[2] += za[j] * phi[j];
  }
  return;
}

//------------------------------------------------------------------------------
void DetJQuad( const RealT xi, const RealT eta, const RealT* x, const int dim, RealT& detJ )
{
  RealT J[4] = { 0., 0., 0., 0. };  // column major ordering

  // loop over nodes
  for ( int a = 0; a < 4; ++a ) {
    // determine (xi,eta) coord of node a
    RealT xi_node, eta_node;
    switch ( a ) {
      case 0:
        xi_node = 1.;
        eta_node = 1.;
        break;
      case 1:
        xi_node = -1.;
        eta_node = 1.;
        break;
      case 2:
        xi_node = -1.;
        eta_node = -1.;
        break;
      case 3:
        xi_node = 1.;
        eta_node = -1.;
        break;
    }

    // loop over 2D coords
    for ( int j = 0; j < 2; ++j ) {
      J[0 + j] += 0.25 * x[dim * a + j] * xi_node * ( 1. + eta_node * eta );
      J[2 + j] += 0.25 * x[dim * a + j] * eta_node * ( 1. + xi_node * xi );
    }
  }

  detJ = J[0] * J[3] - J[2] * J[1];

  // this is a hack, but I can't guarantee the correct orientation of the
  // overlap vertices with respect to some global notion of up. This
  // calculation will calculate the correct absolute value.
  detJ = ( detJ <= 0 ) ? -detJ : detJ;

  return;
}

}  // namespace tribol
