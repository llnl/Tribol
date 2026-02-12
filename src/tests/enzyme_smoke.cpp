// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

//-----------------------------------------------------------------------------
//
// file: enzyme_smoke.cpp
//
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

template <typename return_type, typename... Args>
return_type __enzyme_fwddiff( Args... );

void LinearQuadBasis( const double* xi, double* phi )
{
  phi[0] = 0.25 * ( 1 - xi[0] ) * ( 1 - xi[1] );
  phi[1] = 0.25 * ( 1 + xi[0] ) * ( 1 - xi[1] );
  phi[2] = 0.25 * ( 1 + xi[0] ) * ( 1 + xi[1] );
  phi[3] = 0.25 * ( 1 - xi[0] ) * ( 1 + xi[1] );
}

void LinearQuadBasisDeriv( const double* xi, double* phi, double* dphi_dxi, double* dphi_deta )
{
  double xi_dot[2] = { 1.0, 0.0 };
  __enzyme_fwddiff<void>( (void*)LinearQuadBasis, xi, xi_dot, phi, dphi_dxi );
  xi_dot[0] = 0.0;
  xi_dot[1] = 1.0;
  __enzyme_fwddiff<void>( (void*)LinearQuadBasis, xi, xi_dot, phi, dphi_deta );
}

void LinearQuadBasisDeriv_FD(const double* xi, double* phi, double* dphi_dxi, double* dphi_deta, double h = 1e-4) {
 // compute drivatives wrt to xi[0]
  double phi_p[4];
  double phi_m[4];
  double xi_plush[2] = {xi[0] + h, xi[1]};
  double xi_minush[2] = {xi[0] - h, xi[1]}; 
  LinearQuadBasis(xi_plush, phi_p);
  LinearQuadBasis(xi_minush, phi_m);
  for (int i = 0; i < 4; ++i) {
    dphi_dxi[i] = (phi_p[i] - phi_m[i]) / (2 * h);
  }

  //compute derivatives wrt xi[1]
  xi_plush[1] = xi[1] + h; 
  xi_plush[0] = xi[0];
  xi_minush[1] = xi[1] - h; 
  xi_minush[0] = xi[0];
  LinearQuadBasis(xi_plush, phi_p);
  LinearQuadBasis(xi_minush, phi_m);
  for (int i = 0; i < 4; ++i) {
    dphi_deta[i] = (phi_p[i] - phi_m[i]) / (2 * h);
  }
}

TEST( enzyme_smoke, basic_use )
{
  double xi[2] = { 0.2, -0.4 };
  double phi[4] = { 0.0, 0.0, 0.0, 0.0 };
  double dphi_dxi[4] = { 0.0, 0.0, 0.0, 0.0 };
  double dphi_deta[4] = { 0.0, 0.0, 0.0, 0.0 };

  double xi_fw[2] = { 0.2, -0.4 };
  double phi_fw[4] = { 0.0, 0.0, 0.0, 0.0 };
  double dphi_dxi_fw[4] = { 0.0, 0.0, 0.0, 0.0 };
  double dphi_deta_fw[4] = { 0.0, 0.0, 0.0, 0.0 };


  LinearQuadBasisDeriv_FD( xi, phi, dphi_dxi, dphi_deta);
  LinearQuadBasisDeriv(xi_fw, phi_fw, dphi_dxi_fw, dphi_deta_fw);

  EXPECT_NEAR( dphi_dxi[0], dphi_dxi_fw[0], 1e-6 );
  EXPECT_NEAR( dphi_deta[0], dphi_deta_fw[0], 1e-6 );
  EXPECT_NEAR( dphi_dxi[1], dphi_dxi_fw[1], 1e-6 );
  EXPECT_NEAR( dphi_deta[1], dphi_deta_fw[1], 1e-6 );
  EXPECT_NEAR( dphi_dxi[2], dphi_dxi_fw[2], 1e-6 );
  EXPECT_NEAR( dphi_deta[2],  dphi_deta_fw[2], 1e-6);
  EXPECT_NEAR( dphi_dxi[3], dphi_dxi_fw[3], 1e-6 );
  EXPECT_NEAR( dphi_deta[3], dphi_deta_fw[3], 1e-6 );

  //EXPECT_EQ( dphi_dxi[0], -0.25 * ( 1.0 - xi[1] ) );
  //EXPECT_EQ( dphi_deta[0], -0.25 * ( 1.0 - xi[0] ) );
  //EXPECT_EQ( dphi_dxi[1], 0.25 * ( 1.0 - xi[1] ) );
  //EXPECT_EQ( dphi_deta[1], -0.25 * ( 1.0 + xi[0] ) );
  //EXPECT_EQ( dphi_dxi[2], 0.25 * ( 1.0 + xi[1] ) );
  //EXPECT_EQ( dphi_deta[2], 0.25 * ( 1.0 + xi[0] ) );
  //EXPECT_EQ( dphi_dxi[3], -0.25 * ( 1.0 + xi[1] ) );
  //EXPECT_EQ( dphi_deta[3], 0.25 * ( 1.0 - xi[0] ) );


}
