// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

//-----------------------------------------------------------------------------
//
// file: enzyme_smoke.cpp
//
//-----------------------------------------------------------------------------

#include <iostream>

#include "tribol/config.hpp"
#include "tribol/physics/Mortar.hpp"

#include "gtest/gtest.h"

TEST(enzyme_smoke, basic_use)
{
  double xi[2] = {0.2, -0.4};
  double phi[4] = {0.0, 0.0, 0.0, 0.0};
  double dphi_dxi[4] = {0.0, 0.0, 0.0, 0.0};
  double dphi_deta[4] = {0.0, 0.0, 0.0, 0.0};
  
  tribol::LinearQuadBasisDeriv(xi, phi, dphi_dxi, dphi_deta);

  std::cout << "xi: [" << xi[0] << ", " << xi[1] << "]" << std::endl;
  std::cout << "phi: [" << phi[0] << ", " << phi[1] << ", " << phi[2] << ", " << phi[3] << "]" << std::endl;
  std::cout << "dphi_dxi: [" << dphi_dxi[0] << ", " << dphi_dxi[1] << ", " << dphi_dxi[2] << ", " << dphi_dxi[3] << "]" << std::endl;
  std::cout << "dphi_deta: [" << dphi_deta[0] << ", " << dphi_deta[1] << ", " << dphi_deta[2] << ", " << dphi_deta[3] << "]" << std::endl;

  double delta = 1.0e-7;

  // exact overlap
  // double x1[12] = { 0.0, 0.0, 0.01,
  //                   1.0, 0.0, 0.01,
  //                   1.0, 1.0, 0.01,
  //                   0.0, 1.0, 0.01 };
  // outside to inside edge
  // double x1[12] = { -0.5*delta, 0.0, 0.01,
  //                   1.0, 0.0, 0.01,
  //                   1.0, 1.0, 0.01,
  //                   0.0, 1.0, 0.01 };
  // always outside edge
  // double x1[12] = { -10.0*delta, 0.0, 0.01,
  //                   1.0, 0.0, 0.01,
  //                   1.0, 1.0, 0.01,
  //                   0.0, 1.0, 0.01 };
  // always outside edge
  double x1[12] = { -1.5*delta, -0.1, 0.01,
                    1.0, -0.1, 0.01,
                    1.0, 1.0, 0.01,
                    0.0, 1.0, 0.01 };
  double x2[12] = { 0.0, 0.0, -0.01,
                    0.0, 1.0, -0.01,
                    1.0, 1.0, -0.01,
                    1.0, 0.0, -0.01 };
  double n1[12] = { 0.0, 0.0, 1.0,
                    0.0, 0.0, 1.0,
                    0.0, 0.0, 1.0,
                    0.0, 0.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  double f1[12];
  double f2[12];
  double g1[4];

  //tribol::ComputeMortarForceEnzyme(x1, n1, p1, f1, g1, 4, x2, f2, 4);


  double df1dx1[12*12];
  double df1dx2[12*12];
  double df1dp1[12*12];
  for (int i{0}; i < 12*12; ++i)
  {
    df1dx1[i] = 0.0;
    df1dx2[i] = 0.0;
    df1dp1[i] = 0.0;
  }
  double dg1dx1[4*12];
  double dg1dx2[4*12];
  for (int i{0}; i < 4*12; ++i)
  {
    dg1dx1[i] = 0.0;
    dg1dx2[i] = 0.0;
  }
  double df2dx1[12*12];
  double df2dx2[12*12];
  double df2dp1[12*12];
  for (int i{0}; i < 12*12; ++i)
  {
    df2dx1[i] = 0.0;
    df2dx2[i] = 0.0;
    df2dp1[i] = 0.0;
  }
  tribol::ComputeMortarJacobianEnzyme(x1, n1, p1, f1, df1dx1, df1dx2, df1dp1, g1, dg1dx1, dg1dx2, 4, x2, f2, df2dx1, df2dx2, df2dp1, 4);

  double df1dx1_fd[12*12];
  //tribol::ComputeMortarForceEnzyme(x1, n1, p1, f1, g1, 4, x2, f2, 4);
  for (int i{0}; i < 12; ++i)
  {
    df1dx1_fd[i] = -f1[i];
  }
  x1[0] += delta;
  tribol::ComputeMortarForceEnzyme(x1, n1, p1, f1, g1, 4, x2, f2, 4);
  for (int i{0}; i < 12; ++i)
  {
    df1dx1_fd[i] += f1[i];
    df1dx1_fd[i] /= delta;
  }

  for (int i{0}; i < 12; ++i)
  {
    std::cout << "[" << i << "] Enzyme: " << df1dx1[i] << "  FD: " << df1dx1_fd[i] << std::endl;
  }
}
