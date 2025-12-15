// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

//-----------------------------------------------------------------------------
//
// file: tribol_enzyme_element_mortar.cpp
//
//-----------------------------------------------------------------------------

#include <iostream>

#include "tribol/common/Parameters.hpp"
#include "tribol/config.hpp"

#include "gtest/gtest.h"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

#include "redecomp/common/TypeDefs.hpp"

#include "tribol/physics/Mortar.hpp"
#include "tribol/interface/tribol.hpp"

namespace tribol {

/**
 * @brief Test fixture for the Enzyme-computed Jacobian terms of the mortar method, not including the nodal normal
 * contribution.
 */
class EnzymeElementMortarTest : public testing::Test {
 protected:
  double delta_{ 1.0e-7 };
  double approx_j_err_{ 0.01 };
  double tribol_vs_enzyme_err_{ 1.0e-13 };
  void SetUp() override {}

  /**
   * @brief Check the Jacobian terms computed by Enzyme against finite differences.
   *
   * @param x1 Coordinates of the first face
   * @param x2 Coordinates of the second face
   * @param n1 Normal of the first face
   * @param p1 Pressure of the first face
   * @param x1_stencil Stencil coordinate directions of the first face
   * @param x2_stencil Stencil coordinate directions of the second face
   */
  void FDCheck( double* x1, double* x2, double* n1, double* p1, const double* x1_stencil = nullptr,
                const double* x2_stencil = nullptr, int num_nodes = 4, double check_scale = 1.0 )
  {
    constexpr int max_num_disp_dofs = 12;
    constexpr int max_num_pres_dofs = 4;

    // Array to store forces for the first face
    double f1[max_num_disp_dofs] = { 0.0 };
    // Array to store forces for the second face
    double f2[max_num_disp_dofs] = { 0.0 };
    // Array to store gap for the first face
    double g1[max_num_pres_dofs] = { 0.0 };

    // Derivative of the force on the first face w.r.t. the coordinates of the first face
    double df1dx1[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the first face w.r.t. the coordinates of the second face
    double df1dx2[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the first face w.r.t. the normal of the first face
    double df1dn1[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the first face w.r.t. the pressure of the first face
    double df1dp1[max_num_disp_dofs * max_num_pres_dofs] = { 0.0 };
    // Derivative of the gap on the first face w.r.t. the coordinates of the first face
    double dg1dx1[max_num_pres_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the gap on the first face w.r.t. the coordinates of the second face
    double dg1dx2[max_num_pres_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the gap on the first face w.r.t. the normal of the first face
    double dg1dn1[max_num_pres_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the second face w.r.t. the coordinates of the first face
    double df2dx1[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the second face w.r.t. the coordinates of the second face
    double df2dx2[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the second face w.r.t. the normal of the first face
    double df2dn1[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the second face w.r.t. the pressure of the first face
    double df2dp1[max_num_disp_dofs * max_num_pres_dofs] = { 0.0 };

    tribol::ComputeMortarJacobianEnzyme( x1, n1, p1, f1, df1dx1, df1dx2, df1dn1, df1dp1, g1, dg1dx1, dg1dx2, dg1dn1,
                                         num_nodes, x2, f2, df2dx1, df2dx2, df2dn1, df2dp1, num_nodes );

    // Finite difference derivative of the force on the first face w.r.t. the coordinates of the first face
    double df1dx1_fd[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Finite difference derivative of the force on the second face w.r.t. the coordinates of the first face
    double df2dx1_fd[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Finite difference derivative of the force on the first face w.r.t. the coordinates of the second face
    double df1dx2_fd[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Finite difference derivative of the force on the second face w.r.t. the coordinates of the second face
    double df2dx2_fd[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Finite difference derivative of the force on the first face w.r.t. the normal of the first face
    double df1dn1_fd[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Finite difference derivative of the force on the second face w.r.t. the normal of the first face
    double df2dn1_fd[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Finite difference derivative of the gap on the first face w.r.t. the coordinates of the first face
    double dg1dx1_fd[max_num_disp_dofs * max_num_pres_dofs] = { 0.0 };
    // Finite difference derivative of the gap on the first face w.r.t. the coordinates of the second face
    double dg1dx2_fd[max_num_disp_dofs * max_num_pres_dofs] = { 0.0 };
    // Finite difference derivative of the gap on the first face w.r.t. the normal of the first face
    double dg1dn1_fd[max_num_disp_dofs * max_num_pres_dofs] = { 0.0 };

    int num_disp_dofs = num_nodes * 3;  // Number of displacement degrees of freedom
    int num_pres_dofs = num_nodes;      // Number of pressure degrees of freedom

    for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
      for ( int j{ 0 }; j < num_disp_dofs; ++j ) {
        df1dx1_fd[i * num_disp_dofs + j] = -f1[j];
        df2dx1_fd[i * num_disp_dofs + j] = -f2[j];
        df1dx2_fd[i * num_disp_dofs + j] = -f1[j];
        df2dx2_fd[i * num_disp_dofs + j] = -f2[j];
        df1dn1_fd[i * num_disp_dofs + j] = -f1[j];
        df2dn1_fd[i * num_disp_dofs + j] = -f2[j];
      }
      for ( int j{ 0 }; j < num_pres_dofs; ++j ) {
        dg1dx1_fd[i * num_pres_dofs + j] = -g1[j];
        dg1dx2_fd[i * num_pres_dofs + j] = -g1[j];
        dg1dn1_fd[i * num_pres_dofs + j] = -g1[j];
      }
    }

    // Finite difference derivative of the force on the first face w.r.t. the pressure of the first face
    double df1dp1_fd[max_num_pres_dofs * max_num_disp_dofs] = { 0.0 };
    // Finite difference derivative of the force on the second face w.r.t. the pressure of the first face
    double df2dp1_fd[max_num_pres_dofs * max_num_disp_dofs] = { 0.0 };

    for ( int i{ 0 }; i < num_pres_dofs; ++i ) {
      for ( int j{ 0 }; j < num_disp_dofs; ++j ) {
        df1dp1_fd[i * num_disp_dofs + j] = -f1[j];
        df2dp1_fd[i * num_disp_dofs + j] = -f2[j];
      }
    }
    // wiggle x1
    for ( int j{ 0 }; j < num_disp_dofs; ++j ) {
      auto shift1 = delta_;
      if ( x1_stencil ) {
        shift1 *= x1_stencil[j];
      }
      x1[j] += shift1;
      for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
        f1[i] = 0.0;
        f2[i] = 0.0;
      }
      for ( int i{ 0 }; i < num_pres_dofs; ++i ) {
        g1[i] = 0.0;
      }
      tribol::ComputeMortarForceEnzyme( x1, n1, p1, f1, g1, num_nodes, x2, f2, num_nodes );
      for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
        df1dx1_fd[j * num_disp_dofs + i] += f1[i];
        df1dx1_fd[j * num_disp_dofs + i] /= shift1;
        df2dx1_fd[j * num_disp_dofs + i] += f2[i];
        df2dx1_fd[j * num_disp_dofs + i] /= shift1;
      }
      for ( int i{ 0 }; i < num_pres_dofs; ++i ) {
        dg1dx1_fd[j * num_pres_dofs + i] += g1[i];
        dg1dx1_fd[j * num_pres_dofs + i] /= shift1;
      }
      x1[j] -= shift1;
    }
    // wiggle x2
    for ( int j{ 0 }; j < num_disp_dofs; ++j ) {
      auto shift2 = delta_;
      if ( x2_stencil ) {
        shift2 *= x2_stencil[j];
      }
      x2[j] += shift2;
      for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
        f1[i] = 0.0;
        f2[i] = 0.0;
      }
      for ( int i{ 0 }; i < num_pres_dofs; ++i ) {
        g1[i] = 0.0;
      }
      tribol::ComputeMortarForceEnzyme( x1, n1, p1, f1, g1, num_nodes, x2, f2, num_nodes );
      for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
        df1dx2_fd[j * num_disp_dofs + i] += f1[i];
        df1dx2_fd[j * num_disp_dofs + i] /= shift2;
        df2dx2_fd[j * num_disp_dofs + i] += f2[i];
        df2dx2_fd[j * num_disp_dofs + i] /= shift2;
      }
      for ( int i{ 0 }; i < num_pres_dofs; ++i ) {
        dg1dx2_fd[j * num_pres_dofs + i] += g1[i];
        dg1dx2_fd[j * num_pres_dofs + i] /= shift2;
      }
      x2[j] -= shift2;
    }
    // wiggle n1
    for ( int j{ 0 }; j < num_disp_dofs; ++j ) {
      auto shift = delta_;
      n1[j] += shift;
      for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
        f1[i] = 0.0;
        f2[i] = 0.0;
      }
      for ( int i{ 0 }; i < num_pres_dofs; ++i ) {
        g1[i] = 0.0;
      }
      tribol::ComputeMortarForceEnzyme( x1, n1, p1, f1, g1, num_nodes, x2, f2, num_nodes );
      for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
        df1dn1_fd[j * num_disp_dofs + i] += f1[i];
        df1dn1_fd[j * num_disp_dofs + i] /= shift;
        df2dn1_fd[j * num_disp_dofs + i] += f2[i];
        df2dn1_fd[j * num_disp_dofs + i] /= shift;
      }
      for ( int i{ 0 }; i < num_pres_dofs; ++i ) {
        dg1dn1_fd[j * num_pres_dofs + i] += g1[i];
        dg1dn1_fd[j * num_pres_dofs + i] /= shift;
      }
      n1[j] -= shift;
    }
    // wiggle p1
    for ( int j{ 0 }; j < num_pres_dofs; ++j ) {
      auto shift = delta_;
      p1[j] += shift;
      for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
        f1[i] = 0.0;
        f2[i] = 0.0;
      }
      for ( int i{ 0 }; i < num_pres_dofs; ++i ) {
        g1[i] = 0.0;
      }
      tribol::ComputeMortarForceEnzyme( x1, n1, p1, f1, g1, num_nodes, x2, f2, num_nodes );
      for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
        df1dp1_fd[j * num_disp_dofs + i] += f1[i];
        df1dp1_fd[j * num_disp_dofs + i] /= shift;
        df2dp1_fd[j * num_disp_dofs + i] += f2[i];
        df2dp1_fd[j * num_disp_dofs + i] /= shift;
      }
      p1[j] -= shift;
    }

    double max_diff{ 0.0 };
    double max_error{ check_scale * delta_ };

    std::cout << " df1/dx1 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs * num_disp_dofs; ++i ) {
      auto diff = std::abs( df1dx1[i] - df1dx1_fd[i] );
      max_diff = std::max( max_diff, diff );
      if ( diff > max_error ) {
        auto row = i % num_disp_dofs;
        auto col = i / num_disp_dofs;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << df1dx1[i]
                  << "   FD: " << df1dx1_fd[i] << std::endl;
      }
      EXPECT_NEAR( df1dx1[i], df1dx1_fd[i], max_error );
    }

    std::cout << " df2/dx1 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs * num_disp_dofs; ++i ) {
      auto diff = std::abs( df2dx1[i] - df2dx1_fd[i] );
      max_diff = std::max( max_diff, diff );
      if ( diff > max_error ) {
        auto row = i % num_disp_dofs;
        auto col = i / num_disp_dofs;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << df2dx1[i]
                  << "   FD: " << df2dx1_fd[i] << std::endl;
      }
      EXPECT_NEAR( df2dx1[i], df2dx1_fd[i], max_error );
    }

    std::cout << " dg1/dx1 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_pres_dofs * num_disp_dofs; ++i ) {
      auto diff = std::abs( dg1dx1[i] - dg1dx1_fd[i] );
      max_diff = std::max( max_diff, diff );
      if ( diff > max_error ) {
        auto row = i % num_pres_dofs;
        auto col = i / num_pres_dofs;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << dg1dx1[i]
                  << "   FD: " << dg1dx1_fd[i] << std::endl;
      }
      EXPECT_NEAR( dg1dx1[i], dg1dx1_fd[i], max_error );
    }

    std::cout << " df1/dx2 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs * num_disp_dofs; ++i ) {
      auto diff = std::abs( df1dx2[i] - df1dx2_fd[i] );
      max_diff = std::max( max_diff, diff );
      if ( diff > max_error ) {
        auto row = i % num_disp_dofs;
        auto col = i / num_disp_dofs;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << df1dx2[i]
                  << "   FD: " << df1dx2_fd[i] << std::endl;
      }
      EXPECT_NEAR( df1dx2[i], df1dx2_fd[i], max_error );
    }

    std::cout << " df2/dx2 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs * num_disp_dofs; ++i ) {
      auto diff = std::abs( df2dx2[i] - df2dx2_fd[i] );
      max_diff = std::max( max_diff, diff );
      if ( diff > max_error ) {
        auto row = i % num_disp_dofs;
        auto col = i / num_disp_dofs;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << df2dx2[i]
                  << "   FD: " << df2dx2_fd[i] << std::endl;
      }
      EXPECT_NEAR( df2dx2[i], df2dx2_fd[i], max_error );
    }

    std::cout << " dg1/dx2 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_pres_dofs * num_disp_dofs; ++i ) {
      auto diff = std::abs( dg1dx2[i] - dg1dx2_fd[i] );
      max_diff = std::max( max_diff, diff );
      if ( diff > max_error ) {
        auto row = i % num_pres_dofs;
        auto col = i / num_pres_dofs;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << dg1dx2[i]
                  << "   FD: " << dg1dx2_fd[i] << std::endl;
      }
      EXPECT_NEAR( dg1dx2[i], dg1dx2_fd[i], max_error );
    }

    std::cout << " df1/dn1 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs * num_disp_dofs; ++i ) {
      auto diff = std::abs( df1dn1[i] - df1dn1_fd[i] );
      max_diff = std::max( max_diff, diff );
      if ( diff > max_error ) {
        auto row = i % num_disp_dofs;
        auto col = i / num_disp_dofs;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << df1dn1[i]
                  << "   FD: " << df1dn1_fd[i] << std::endl;
      }
      EXPECT_NEAR( df1dn1[i], df1dn1_fd[i], max_error );
    }

    std::cout << " df2/dn1 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs * num_disp_dofs; ++i ) {
      auto diff = std::abs( df2dn1[i] - df2dn1_fd[i] );
      max_diff = std::max( max_diff, diff );
      if ( diff > max_error ) {
        auto row = i % num_disp_dofs;
        auto col = i / num_disp_dofs;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << df2dn1[i]
                  << "   FD: " << df2dn1_fd[i] << std::endl;
      }
      EXPECT_NEAR( df2dn1[i], df2dn1_fd[i], max_error );
    }

    std::cout << " dg1/dn1 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_pres_dofs * num_disp_dofs; ++i ) {
      auto diff = std::abs( dg1dn1[i] - dg1dn1_fd[i] );
      max_diff = std::max( max_diff, diff );
      if ( diff > max_error ) {
        auto row = i % num_pres_dofs;
        auto col = i / num_pres_dofs;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << dg1dn1[i]
                  << "   FD: " << dg1dn1_fd[i] << std::endl;
      }
      EXPECT_NEAR( dg1dn1[i], dg1dn1_fd[i], max_error );
    }

    std::cout << " df1/dp1 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_pres_dofs * num_disp_dofs; ++i ) {
      auto diff = std::abs( df1dp1[i] - df1dp1_fd[i] );
      max_diff = std::max( max_diff, diff );
      if ( diff > max_error ) {
        auto row = i % num_disp_dofs;
        auto col = i / num_disp_dofs;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << df1dp1[i]
                  << "   FD: " << df1dp1_fd[i] << std::endl;
      }
      EXPECT_NEAR( df1dp1[i], df1dp1_fd[i], max_error );
    }

    std::cout << " df2/dp1 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_pres_dofs * num_disp_dofs; ++i ) {
      auto diff = std::abs( df2dp1[i] - df2dp1_fd[i] );
      max_diff = std::max( max_diff, diff );
      if ( diff > max_error ) {
        auto row = i % num_disp_dofs;
        auto col = i / num_disp_dofs;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << df2dp1[i]
                  << "   FD: " << df2dp1_fd[i] << std::endl;
      }
      EXPECT_NEAR( df2dp1[i], df2dp1_fd[i], max_error );
    }

    std::cout << "max_diff for finite difference test: " << max_diff << std::endl;
  }

  /**
   * @brief Check the Jacobian terms computed by Enzyme against the non-Enzyme Tribol simplified Jacobian.
   *
   * @param x1 Coordinates of the first face
   * @param x2 Coordinates of the second face
   * @param n1 Normal of the first face
   * @param p1 Pressure of the first face
   */
  void SimplifiedJacobianCheck( double* x1, double* x2, double* n1, double* p1, int num_nodes = 4 )
  {
    constexpr int max_num_disp_dofs = 12;
    constexpr int max_num_pres_dofs = 4;

    // Array to store forces for the first face
    double f1[max_num_disp_dofs] = { 0.0 };
    // Array to store forces for the second face
    double f2[max_num_disp_dofs] = { 0.0 };
    // Array to store gap for the first face
    double g1[max_num_pres_dofs] = { 0.0 };
    // Derivative of the force on the first face w.r.t. the coordinates of the first face
    double df1dx1[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the first face w.r.t. the coordinates of the second face
    double df1dx2[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the first face w.r.t. the normal of the first face
    double df1dn1[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the first face w.r.t. the pressure of the first face
    double df1dp1[max_num_disp_dofs * max_num_pres_dofs] = { 0.0 };
    // Derivative of the gap on the first face w.r.t. the coordinates of the first face
    double dg1dx1[max_num_pres_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the gap on the first face w.r.t. the coordinates of the second face
    double dg1dx2[max_num_pres_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the gap on the first face w.r.t. the normal of the first face
    double dg1dn1[max_num_pres_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the second face w.r.t. the coordinates of the first face
    double df2dx1[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the second face w.r.t. the coordinates of the second face
    double df2dx2[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the second face w.r.t. the normal of the first face
    double df2dn1[max_num_disp_dofs * max_num_disp_dofs] = { 0.0 };
    // Derivative of the force on the second face w.r.t. the pressure of the first face
    double df2dp1[max_num_disp_dofs * max_num_pres_dofs] = { 0.0 };

    ComputeMortarJacobianEnzyme( x1, n1, p1, f1, df1dx1, df1dx2, df1dn1, df1dp1, g1, dg1dx1, dg1dx2, dg1dn1, num_nodes,
                                 x2, f2, df2dx1, df2dx2, df2dn1, df2dp1, num_nodes );

    // NOTE: This will still work for triangles since the unused nodes are ignored.
    IndexT conn[4] = { 0, 1, 2, 3 };
    tribol::InterfaceElementType interface_element_type = tribol::InterfaceElementType::LINEAR_QUAD;
    if ( num_nodes == 3 ) {
      interface_element_type = tribol::InterfaceElementType::LINEAR_TRIANGLE;
    }
    constexpr int num_elems = 1;

    constexpr int mesh_id1 = 0;
    registerMesh( mesh_id1, num_elems, num_nodes, conn, interface_element_type, x1, x1 + num_nodes, x1 + 2 * num_nodes,
                  MemorySpace::Host );
    constexpr int mesh_id2 = 1;
    registerMesh( mesh_id2, num_elems, num_nodes, conn, interface_element_type, x2, x2 + num_nodes, x2 + 2 * num_nodes,
                  MemorySpace::Host );
    constexpr int cs_id = 0;
    // mortar then nonmortar surfaces
    registerCouplingScheme( cs_id, mesh_id2, mesh_id1, ContactMode::SURFACE_TO_SURFACE, ContactCase::NO_CASE,
                            ContactMethod::SINGLE_MORTAR, ContactModel::FRICTIONLESS,
                            EnforcementMethod::LAGRANGE_MULTIPLIER, BinningMethod::BINNING_GRID,
                            ExecutionMode::Sequential );
    // Array to store non-Enzyme forces for the first face
    double f1t[max_num_disp_dofs] = { 0.0 };
    registerNodalResponse( mesh_id1, f1t, f1t + num_nodes, f1t + 2 * num_nodes );
    // Array to store non-Enzyme forces for the second face
    double f2t[max_num_disp_dofs] = { 0.0 };
    registerNodalResponse( mesh_id2, f2t, f2t + num_nodes, f2t + 2 * num_nodes );
    // Array to store non-Enzyme gap for the first face
    double g1t[max_num_pres_dofs] = { 0.0 };
    registerMortarGaps( mesh_id1, g1t );
    registerMortarPressures( mesh_id1, p1 );
    setLagrangeMultiplierOptions( cs_id, ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN, SparseMode::MFEM_ELEMENT_DENSE );

    int cycle = 1;
    double t = 0.0;
    double dt = 1.0;
    update( cycle, t, dt );

    const ArrayT<int>* row_elem_idx = nullptr;
    const ArrayT<int>* col_elem_idx = nullptr;
    const ArrayT<mfem::DenseMatrix>* jacobians = nullptr;
    getElementBlockJacobians( cs_id, BlockSpace::NONMORTAR, BlockSpace::LAGRANGE_MULTIPLIER, &row_elem_idx,
                              &col_elem_idx, &jacobians );

    double max_diff{ 0.0 };
    constexpr int dim = 3;
    int num_disp_dofs = num_nodes * dim;
    int num_pres_dofs = num_nodes;

    std::cout << "df1/dp -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
      for ( int j{ 0 }; j < num_pres_dofs; ++j ) {
        int idx_e = num_pres_dofs * i + j;
        auto diff = std::abs( df1dp1[idx_e] - ( *jacobians )[0].Data()[idx_e] );
        max_diff = std::max( max_diff, diff );
        if ( diff > approx_j_err_ ) {
          std::cout << "[" << idx_e << "] Enzyme: " << df1dp1[idx_e] << "  Tribol: " << ( *jacobians )[0].Data()[idx_e]
                    << std::endl;
        }
        EXPECT_NEAR( df1dp1[idx_e], ( *jacobians )[0].Data()[idx_e], approx_j_err_ );
      }
    }

    getElementBlockJacobians( cs_id, BlockSpace::MORTAR, BlockSpace::LAGRANGE_MULTIPLIER, &row_elem_idx, &col_elem_idx,
                              &jacobians );

    std::cout << "df2/dp -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
      for ( int j{ 0 }; j < num_pres_dofs; ++j ) {
        int idx_e = num_pres_dofs * i + j;
        auto diff = std::abs( df2dp1[idx_e] - ( *jacobians )[0].Data()[idx_e] );
        max_diff = std::max( max_diff, diff );
        if ( diff > approx_j_err_ ) {
          std::cout << "[" << idx_e << "] Enzyme: " << df2dp1[idx_e] << "  Tribol: " << ( *jacobians )[0].Data()[idx_e]
                    << std::endl;
        }
        EXPECT_NEAR( df2dp1[idx_e], ( *jacobians )[0].Data()[idx_e], approx_j_err_ );
      }
    }

    getElementBlockJacobians( cs_id, BlockSpace::LAGRANGE_MULTIPLIER, BlockSpace::NONMORTAR, &row_elem_idx,
                              &col_elem_idx, &jacobians );

    std::cout << "dg/dx1 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
      for ( int j{ 0 }; j < num_pres_dofs; ++j ) {
        int idx_e = num_pres_dofs * i + j;
        auto diff = std::abs( dg1dx1[idx_e] - ( *jacobians )[0].Data()[idx_e] );
        max_diff = std::max( max_diff, diff );
        if ( diff > approx_j_err_ ) {
          std::cout << "[" << idx_e << "] Enzyme: " << dg1dx1[idx_e] << "  Tribol: " << ( *jacobians )[0].Data()[idx_e]
                    << std::endl;
        }
        EXPECT_NEAR( dg1dx1[idx_e], ( *jacobians )[0].Data()[idx_e], approx_j_err_ );
      }
    }

    getElementBlockJacobians( cs_id, BlockSpace::LAGRANGE_MULTIPLIER, BlockSpace::MORTAR, &row_elem_idx, &col_elem_idx,
                              &jacobians );

    std::cout << "dg/dx2 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
      for ( int j{ 0 }; j < num_pres_dofs; ++j ) {
        int idx_e = num_pres_dofs * i + j;
        auto diff = std::abs( dg1dx2[idx_e] - ( *jacobians )[0].Data()[idx_e] );
        max_diff = std::max( max_diff, diff );
        if ( diff > approx_j_err_ ) {
          std::cout << "[" << idx_e << "] Enzyme: " << dg1dx2[idx_e] << "  Tribol: " << ( *jacobians )[0].Data()[idx_e]
                    << std::endl;
        }
        EXPECT_NEAR( dg1dx2[idx_e], ( *jacobians )[0].Data()[idx_e], approx_j_err_ );
      }
    }

    std::cout << "max_diff for approximate Jacobian comparison test: " << max_diff << std::endl;

    std::cout << "g -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_pres_dofs; ++i ) {
      auto diff = std::abs( g1[i] - g1t[i] );
      if ( diff > tribol_vs_enzyme_err_ ) {
        std::cout << "[" << i << "] Enzyme: " << g1[i] << "  Tribol: " << g1t[i] << std::endl;
      }
      EXPECT_NEAR( g1[i], g1t[i], tribol_vs_enzyme_err_ );
    }

    std::cout << "f1 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
      auto diff = std::abs( f1[i] - f1t[i] );
      if ( diff > tribol_vs_enzyme_err_ ) {
        std::cout << "[" << i << "] Enzyme: " << f1[i] << "  Tribol: " << f1t[i] << std::endl;
      }
      EXPECT_NEAR( f1[i], f1t[i], tribol_vs_enzyme_err_ );
    }

    std::cout << "f2 -------------------------------------- " << std::endl;
    for ( int i{ 0 }; i < num_disp_dofs; ++i ) {
      auto diff = std::abs( f2[i] - f2t[i] );
      if ( diff > tribol_vs_enzyme_err_ ) {
        std::cout << "[" << i << "] Enzyme: " << f2[i] << "  Tribol: " << f2t[i] << std::endl;
      }
      EXPECT_NEAR( f2[i], f2t[i], tribol_vs_enzyme_err_ );
    }
  }
};

TEST_F( EnzymeElementMortarTest, ExactOverlapZeroGap )
{
  // clang-format off
  // {x0, x1, x2, x3,
  //  y0, y1, y2, y3,
  //  z0, z1, z2, z3}
  double x1[12] = { 0.0, 1.0, 1.0, 0.0,
                    0.0, 0.0, 1.0, 1.0,
                    0.0, 0.0, 0.0, 0.0 };
  double x2[12] = { 0.0, 0.0, 1.0, 1.0,
                    0.0, 1.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0 };
  double n1[12] = { 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  double x1_stencil[12] = {  1.0, -1.0, -1.0,  1.0,
                             1.0,  1.0, -1.0, -1.0,
                            -1.0, -1.0, -1.0, -1.0 };
  double x2_stencil[12] = { -1.0, -1.0,  1.0,  1.0,
                            -1.0,  1.0,  1.0, -1.0,
                            -1.0, -1.0, -1.0, -1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1, x1_stencil, x2_stencil );
  // the simplified Tribol jacobian should match here.  verify that it does
  SimplifiedJacobianCheck( x1, x2, n1, p1 );
}

TEST_F( EnzymeElementMortarTest, ShiftedXYNonmortarElementZeroGapTri )
{
  // clang-format off
  // {x0, x1, x2,
  //  y0, y1, y2,
  //  z0, z1, z2}
  constexpr double offset = 0.01;
  double x1[9] = { 0.0 + 2.0 * offset, 1.0 + 2.0 * offset, 1.0 + 2.0 * offset,
                   0.0 + offset,       0.0 + offset,       1.0 + offset,
                   0.0,                0.0,                0.0 };
  double x2[9] = { 0.0, 1.0, 1.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 0.0 };
  double n1[9] = { 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0,
                   1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0 };
  double x1_stencil[12] = {  1.0, -1.0, -1.0,
                             1.0,  1.0, -1.0,
                            -1.0, -1.0, -1.0 };
  double x2_stencil[12] = { -1.0,  1.0,  1.0,
                            -1.0,  1.0, -1.0,
                            -1.0, -1.0, -1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1, x1_stencil, x2_stencil, 3, 2.0 );
  // the simplified Tribol jacobian should match here.  verify that it does
  SimplifiedJacobianCheck( x1, x2, n1, p1, 3 );
}

TEST_F( EnzymeElementMortarTest, SlightlySmallerNonmortarElementMinorInterpenetration )
{
  // slightly smaller
  double dx = 4.0 * delta_;
  // clang-format off
  // {x0, x1, x2, x3,
  //  y0, y1, y2, y3,
  //  z0, z1, z2, z3}
  double x1[12] = { 0.0+dx, 1.0-dx, 1.0-dx, 0.0+dx,
                    0.0+dx, 0.0+dx, 1.0-dx, 1.0-dx,
                    0.01,   0.01,   0.01,   0.01 };
  double x2[12] = { 0.0,   0.0,   1.0,   1.0,
                    0.0,   1.0,   1.0,   0.0,
                    -0.01, -0.01, -0.01, -0.01 };
  double n1[12] = { 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1 );
  // the simplified Tribol jacobian should be close here.  verify that it is
  SimplifiedJacobianCheck( x1, x2, n1, p1 );
}

TEST_F( EnzymeElementMortarTest, ShiftedXNonmortarElementMinorInterpenetration )
{
  // slightly smaller and offset
  double offset = 0.3;
  double dx = 4.0 * delta_;
  // clang-format off
  // {x0, x1, x2, x3,
  //  y0, y1, y2, y3,
  //  z0, z1, z2, z3}
  double x1[12] = { 0.0+dx+offset, 1.0-dx+offset, 1.0-dx+offset, 0.0+dx+offset,
                    0.0+dx,        0.0+dx,        1.0-dx,        1.0-dx,
                    0.01,          0.01,          0.01,          0.01 };
  double x2[12] = { 0.0,   0.0,   1.0,   1.0,
                    0.0,   1.0,   1.0,   0.0,
                    -0.01, -0.01, -0.01, -0.01 };
  double n1[12] = { 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1 );
}

TEST_F( EnzymeElementMortarTest, ShiftedXYNonmortarElementMinorInterpenetration )
{
  // slightly smaller and offset
  double offset = 0.3;
  double dx = 4.0 * delta_;
  // clang-format off
  // {x0, x1, x2, x3,
  //  y0, y1, y2, y3,
  //  z0, z1, z2, z3}
  double x1[12] = { 0.0+dx+offset, 1.0-dx+offset, 1.0-dx+offset, 0.0+dx+offset,
                    0.0+dx+offset, 0.0+dx+offset, 1.0-dx+offset, 1.0-dx+offset,
                    0.01,          0.01,          0.01,          0.01 };
  double x2[12] = { 0.0,   0.0,   1.0,   1.0,
                    0.0,   1.0,   1.0,   0.0,
                    -0.01, -0.01, -0.01, -0.01 };
  double n1[12] = { 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1 );
}

// NOTE: this configuration is designed to match a test in Smith
TEST_F( EnzymeElementMortarTest, ShiftedXYNonmortarElementMinorInterpenetrationV2 )
{
  // slightly offset
  double dx = 10.0 * delta_;
  // clang-format off
  // {x0, x1, x2, x3,
  //  y0, y1, y2, y3,
  //  z0, z1, z2, z3}
  double x1[12] = { 0.0+dx, 0.0+dx, 1.0+dx, 1.0+dx,
                    0.0+dx, 1.0+dx, 1.0+dx, 0.0+dx,
                    0.999,  0.999,  0.999,  0.999 };
  double x2[12] = { 0.0, 1.0, 1.0, 0.0,
                    0.0, 0.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0 };
  double n1[12] = { 0.0,  0.0,  0.0,  0.0,
                    0.0,  0.0,  0.0,  0.0,
                    -1.0, -1.0, -1.0, -1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1 );
}

TEST_F( EnzymeElementMortarTest, Rotated30DegNonmortarElementMinorInterpenetration )
{
  // clang-format off
  // rotate 30 degrees
  double x1[12] = { 0.0,  1.0,  1.0,  0.0,
                    0.0,  0.0,  1.0,  1.0,
                    0.01, 0.01, 0.01, 0.01 };
  // clang-format on
  double cos30 = std::cos( redecomp::pi / 6.0 );
  double sin30 = std::sin( redecomp::pi / 6.0 );
  for ( int i{ 0 }; i < 4; ++i ) {
    double x_new = x1[i] * cos30 - x1[i + 4] * sin30;
    double y_new = x1[i] * sin30 + x1[i + 4] * cos30;
    x1[i] = x_new;
    x1[i + 4] = y_new;
  }
  // shift to center the element at (0.5, 0.5)
  double x_shift = 0.25;
  double y_shift = -0.5 * ( x1[6] - 1.0 );
  for ( int i{ 0 }; i < 4; ++i ) {
    x1[i] += x_shift;
    x1[i + 4] += y_shift;
  }
  // clang-format off
  double x2[12] = { 0.0,   0.0,   1.0,   1.0,
                    0.0,   1.0,   1.0,   0.0,
                    -0.01, -0.01, -0.01, -0.01 };
  double n1[12] = { 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1 );
}

TEST_F( EnzymeElementMortarTest, NonaffineRotated45DegMortarElementMinorInterpenetration )
{
  // clang-format off
  // rotate 45 degrees
  double x1[12] = { 0.0,  1.1,  1.0,  0.0,
                    0.0,  0.0,  1.1,  1.0,
                    0.01, 0.01, 0.01, 0.01 };
  // clang-format on
  double cos45 = std::cos( redecomp::pi / 4.0 );
  double sin45 = std::sin( redecomp::pi / 4.0 );
  for ( int i{ 0 }; i < 4; ++i ) {
    double x_new = x1[i] * cos45 - x1[i + 4] * sin45;
    double y_new = x1[i] * sin45 + x1[i + 4] * cos45;
    x1[i] = x_new;
    x1[i + 4] = y_new;
  }
  // shift to center the element near (0.5, 0.5)
  double x_shift = 0.5 / std::sqrt( 2.0 ) + 0.1;
  double y_shift = -0.5 / std::sqrt( 2.0 ) + 0.1;
  for ( int i{ 0 }; i < 4; ++i ) {
    x1[i] += x_shift;
    x1[i + 4] += y_shift;
  }
  // clang-format off
  double x2[12] = { 0.0,   0.0,   1.0,   1.0,
                    0.0,   1.0,   1.0,   0.0,
                    -0.01, -0.01, -0.01, -0.01 };
  double n1[12] = { 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1 );
}

TEST_F( EnzymeElementMortarTest, NonaffineOutOfPlaneNonmortarElementMinorInterpenetration )
{
  // clang-format off
  // rotate 45 degrees
  double x1[12] = { 0.0,  1.1,  1.0,  0.0,
                    0.0,  0.0,  1.1,  1.0,
                    0.0,  0.0,  0.01, 0.01 };
  // clang-format on
  double cos45 = std::cos( redecomp::pi / 4.0 );
  double sin45 = std::sin( redecomp::pi / 4.0 );
  for ( int i{ 0 }; i < 4; ++i ) {
    double x_new = x1[i] * cos45 - x1[i + 4] * sin45;
    double y_new = x1[i] * sin45 + x1[i + 4] * cos45;
    x1[i] = x_new;
    x1[i + 4] = y_new;
  }
  // shift to center the element near (0.5, 0.5)
  double x_shift = 0.5 / std::sqrt( 2.0 ) + 0.1;
  double y_shift = -0.5 / std::sqrt( 2.0 ) + 0.1;
  for ( int i{ 0 }; i < 4; ++i ) {
    x1[i] += x_shift;
    x1[i + 4] += y_shift;
  }
  // clang-format off
  double x2[12] = { 0.0,   0.0,   1.0,   1.0,
                    0.0,   1.0,   1.0,   0.0,
                    -0.01, -0.01, -0.01, -0.01 };
  double n1[12] = { 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1 );
}

TEST_F( EnzymeElementMortarTest, NonaffineWarpedNonmortarElementMinorInterpenetration )
{
  // clang-format off
  // rotate 45 degrees
  double x1[12] = { 0.0,  1.1,   1.0,  0.0,
                    0.0,  0.0,   1.1,  1.0,
                    0.0,  -0.01, 0.01, 0.02 };
  // clang-format on
  double cos45 = std::cos( redecomp::pi / 4.0 );
  double sin45 = std::sin( redecomp::pi / 4.0 );
  for ( int i{ 0 }; i < 4; ++i ) {
    double x_new = x1[i] * cos45 - x1[i + 4] * sin45;
    double y_new = x1[i] * sin45 + x1[i + 4] * cos45;
    x1[i] = x_new;
    x1[i + 4] = y_new;
  }
  // shift to center the element near (0.5, 0.5)
  double x_shift = 0.5 / std::sqrt( 2.0 ) + 0.1;
  double y_shift = -0.5 / std::sqrt( 2.0 ) + 0.1;
  for ( int i{ 0 }; i < 4; ++i ) {
    x1[i] += x_shift;
    x1[i + 4] += y_shift;
  }
  // clang-format off
  double x2[12] = { 0.0,   0.0,   1.0,   1.0,
                    0.0,   1.0,   1.0,   0.0,
                    -0.01, -0.01, -0.01, -0.01 };
  double n1[12] = { 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1 );
}

TEST_F( EnzymeElementMortarTest, NonaffineOutOfPlaneMortarElementMinorInterpenetration )
{
  // clang-format off
  // rotate 45 degrees
  double x1[12] = { 0.0,  1.1,  1.0,  0.0,
                    0.0,  0.0,  1.1,  1.0,
                    0.01, 0.01, 0.01, 0.01 };
  // clang-format on
  double cos45 = std::cos( redecomp::pi / 4.0 );
  double sin45 = std::sin( redecomp::pi / 4.0 );
  for ( int i{ 0 }; i < 4; ++i ) {
    double x_new = x1[i] * cos45 - x1[i + 4] * sin45;
    double y_new = x1[i] * sin45 + x1[i + 4] * cos45;
    x1[i] = x_new;
    x1[i + 4] = y_new;
  }
  // shift to center the element near (0.5, 0.5)
  double x_shift = 0.5 / std::sqrt( 2.0 ) + 0.1;
  double y_shift = -0.5 / std::sqrt( 2.0 ) + 0.1;
  for ( int i{ 0 }; i < 4; ++i ) {
    x1[i] += x_shift;
    x1[i + 4] += y_shift;
  }
  // clang-format off
  double x2[12] = { 0.0,   0.0,   1.0, 1.0,
                    0.0,   1.0,   1.0, 0.0,
                    -0.01, -0.01, 0.0, 0.0 };
  double n1[12] = { 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1 );
}

TEST_F( EnzymeElementMortarTest, NonaffineWarpedMortarElementMinorInterpenetration )
{
  // clang-format off
  // rotate 45 degrees
  double x1[12] = { 0.0,  1.1,  1.0,  0.0,
                    0.0,  0.0,  1.1,  1.0,
                    0.01, 0.01, 0.01, 0.01 };
  // clang-format on
  double cos45 = std::cos( redecomp::pi / 4.0 );
  double sin45 = std::sin( redecomp::pi / 4.0 );
  for ( int i{ 0 }; i < 4; ++i ) {
    double x_new = x1[i] * cos45 - x1[i + 4] * sin45;
    double y_new = x1[i] * sin45 + x1[i + 4] * cos45;
    x1[i] = x_new;
    x1[i + 4] = y_new;
  }
  // shift to center the element near (0.5, 0.5)
  double x_shift = 0.5 / std::sqrt( 2.0 ) + 0.1;
  double y_shift = -0.5 / std::sqrt( 2.0 ) + 0.1;
  for ( int i{ 0 }; i < 4; ++i ) {
    x1[i] += x_shift;
    x1[i + 4] += y_shift;
  }
  // clang-format off
  double x2[12] = { 0.0,   0.0,   1.0, 1.0,
                    0.0,   1.0,   1.0, 0.0,
                    0.01,  -0.01, 0.0, -0.01 };
  double n1[12] = { 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1 );
}

TEST_F( EnzymeElementMortarTest, NoOverlap )
{
  // clang-format off
  double x1[12] = { 0,                  0.25061248332819264, 0.25061248347850068, 0,
                    1.0024499307581727, 1.0024499310658752,  0.75183744879681569, 0.75183744859830548,
                    0.9950243367728403, 0.99502433719083682, 0.99502433812781421, 0.99502433784796417 };
  double x2[12] = { 0,                   0.2506124842888437,  0.25061248413259829, 0,
                    0.50122496909581893, 0.50122496895699598, 0.75183745346191466, 0.75183745367891308,
                    0.99497565718874636, 0.99497565744420857, 0.99497565800865728, 0.99497565778429586 };
  double n1[12] = { 1.6760823570596968E-9,  2.8328822061264079E-9,  1.7167363802005653E-9,  1.1221425796502179E-9,
                    -4.3110314424768161E-9, -3.7570916641889438E-9, 4.1753970779955578E-10, -2.8983706722398794E-11,
                    -1.0049058641140272,    -1.0049058643326783,    -1.0049058657698271,    -1.0049058656548657 };
  double p1[4] = { -0.0039961035429747216, -0.0039669165550449692, -0.0035314820072299361, -0.0035524348165424662 };
  double x1_stencil[12] = {  1.0,  1.0,  1.0,  1.0,
                             1.0,  1.0,  1.0,  1.0,
                             1.0,  1.0,  1.0,  1.0 };
  double x2_stencil[12] = {  1.0,  1.0,  1.0,  1.0,
                             1.0,  1.0, -1.0, -1.0,
                             1.0,  1.0,  1.0,  1.0 };
  // clang-format on

  FDCheck( x1, x2, n1, p1, x1_stencil, x2_stencil );
}

}  // namespace tribol

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
