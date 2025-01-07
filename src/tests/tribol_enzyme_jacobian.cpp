// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

//-----------------------------------------------------------------------------
//
// file: tribol_enzyme_jacobian.cpp
//
//-----------------------------------------------------------------------------

#include <iostream>

#include "redecomp/common/TypeDefs.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/config.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/Mortar.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/utils/Algorithm.hpp"

#include "gtest/gtest.h"

namespace tribol {

void FDCheck(double* x1, double* x2, double* n1, double* p1);

// NOTE: Make sure no vertices on either element pass through an edge. The
// finite differencing will fail in this case.
class EnzymeJacobianTest : public testing::Test {
 protected:
  double delta_{1.0e-7};
  double approx_j_err_{0.01};
  double tribol_vs_enzyme_err_{1.0e-13};
  void SetUp() override {}

  void FDCheck(double* x1, double* x2, double* n1, double* p1, const double* x1_stencil = nullptr,
               const double* x2_stencil = nullptr)
  {
    double f1[12];
    double f2[12];
    for (int i{0}; i < 12; ++i) {
      f1[i] = 0.0;
      f2[i] = 0.0;
    }
    double g1[4];
    for (int i{0}; i < 4; ++i) {
      g1[i] = 0.0;
    }

    double df1dx1[12 * 12];
    double df1dx2[12 * 12];
    double df1dn1[12 * 12];
    for (int i{0}; i < 12 * 12; ++i) {
      df1dx1[i] = 0.0;
      df1dx2[i] = 0.0;
      df1dn1[i] = 0.0;
    }
    double df1dp1[12 * 4];
    for (int i{0}; i < 12 * 4; ++i) {
      df1dp1[i] = 0.0;
    }
    double dg1dx1[4 * 12];
    double dg1dx2[4 * 12];
    double dg1dn1[4 * 12];
    for (int i{0}; i < 4 * 12; ++i) {
      dg1dx1[i] = 0.0;
      dg1dx2[i] = 0.0;
      dg1dn1[i] = 0.0;
    }
    double df2dx1[12 * 12];
    double df2dx2[12 * 12];
    double df2dn1[12 * 12];
    for (int i{0}; i < 12 * 12; ++i) {
      df2dx1[i] = 0.0;
      df2dx2[i] = 0.0;
      df2dn1[i] = 0.0;
    }
    double df2dp1[12 * 4];
    for (int i{0}; i < 12 * 4; ++i) {
      df2dp1[i] = 0.0;
    }

    tribol::ComputeMortarJacobianEnzyme(x1, n1, p1, f1, df1dx1, df1dx2, df1dn1, df1dp1, g1, dg1dx1, dg1dx2, dg1dn1, 4,
                                        x2, f2, df2dx1, df2dx2, df2dn1, df2dp1, 4);

    double df1dx1_fd[12 * 12];
    double df2dx1_fd[12 * 12];
    for (int i{0}; i < 12; ++i) {
      for (int j{0}; j < 12; ++j) {
        df1dx1_fd[i * 12 + j] = -f1[j];
        df2dx1_fd[i * 12 + j] = -f2[j];
      }
    }
    for (int j{0}; j < 12; ++j) {
      auto shift1 = delta_;
      if (x1_stencil) {
        shift1 *= x1_stencil[j];
      }
      x1[j] += shift1;
      for (int i{0}; i < 12; ++i) {
        f1[i] = 0.0;
        f2[i] = 0.0;
      }
      double g1[4];
      for (int i{0}; i < 4; ++i) {
        g1[i] = 0.0;
      }
      tribol::ComputeMortarForceEnzyme(x1, n1, p1, f1, g1, 4, x2, f2, 4);
      for (int i{0}; i < 12; ++i) {
        df1dx1_fd[j * 12 + i] += f1[i];
        df1dx1_fd[j * 12 + i] /= shift1;
        df2dx1_fd[j * 12 + i] += f2[i];
        df2dx1_fd[j * 12 + i] /= shift1;
      }
      x1[j] -= shift1;
    }

    double max_diff{0.0};
    std::cout << " df1/dx1 -------------------------------------- " << std::endl;
    for (int i{0}; i < 144; ++i) {
      auto diff = std::abs(df1dx1[i] - df1dx1_fd[i]);
      max_diff = std::max(max_diff, diff);
      if (diff > delta_) {
        // if (std::abs(df1dx1[i]) > 1e-14 || std::abs(df1dx1_fd[i]) > delta_) {
        auto row = i % 12;
        auto col = i / 12;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << df1dx1[i]
                  << "   FD: " << df1dx1_fd[i] << std::endl;
      }
      EXPECT_NEAR(df1dx1[i], df1dx1_fd[i], delta_);
    }

    std::cout << " df2/dx1 -------------------------------------- " << std::endl;
    for (int i{0}; i < 144; ++i) {
      auto diff = std::abs(df2dx1[i] - df2dx1_fd[i]);
      max_diff = std::max(max_diff, diff);
      if (diff > delta_) {
        // if (std::abs(df2dx1[i]) > 1e-14 || std::abs(df2dx1_fd[i]) > delta_) {
        auto row = i % 12;
        auto col = i / 12;
        std::cout << "  (" << row << ", " << col << ") : Diff: " << diff << "   Enzyme: " << df2dx1[i]
                  << "   FD: " << df2dx1_fd[i] << std::endl;
      }
      EXPECT_NEAR(df2dx1[i], df2dx1_fd[i], delta_);
    }

    std::cout << "max_diff for test: " << max_diff << std::endl;
  }

  void ApproxJacobianCheck(double* x1, double* x2, double* n1, double* p1)
  {
    double f1[12];
    double f2[12];
    double g1[4];

    double df1dx1[12 * 12];
    double df1dx2[12 * 12];
    double df1dn1[12 * 12];
    for (int i{0}; i < 12 * 12; ++i) {
      df1dx1[i] = 0.0;
      df1dx2[i] = 0.0;
      df1dn1[i] = 0.0;
    }
    double df1dp1[12 * 4];
    for (int i{0}; i < 12 * 4; ++i) {
      df1dp1[i] = 0.0;
    }
    double dg1dx1[4 * 12];
    double dg1dx2[4 * 12];
    double dg1dn1[4 * 12];
    for (int i{0}; i < 4 * 12; ++i) {
      dg1dx1[i] = 0.0;
      dg1dx2[i] = 0.0;
      dg1dn1[i] = 0.0;
    }
    double df2dx1[12 * 12];
    double df2dx2[12 * 12];
    double df2dn1[12 * 12];
    for (int i{0}; i < 12 * 12; ++i) {
      df2dx1[i] = 0.0;
      df2dx2[i] = 0.0;
      df2dn1[i] = 0.0;
    }
    double df2dp1[12 * 4];
    for (int i{0}; i < 12 * 4; ++i) {
      df2dp1[i] = 0.0;
    }

    ComputeMortarJacobianEnzyme(x1, n1, p1, f1, df1dx1, df1dx2, df1dn1, df1dp1, g1, dg1dx1, dg1dx2, dg1dn1, 4, x2, f2,
                                df2dx1, df2dx2, df2dn1, df2dp1, 4);

    int conn[4] = {0, 1, 2, 3};
    constexpr int num_elems = 1;
    constexpr int num_nodes = 4;

    constexpr int mesh_id1 = 0;
    registerMesh(mesh_id1, num_elems, num_nodes, conn, InterfaceElementType::LINEAR_QUAD, x1, x1 + 4, x1 + 8);
    constexpr int mesh_id2 = 1;
    registerMesh(mesh_id2, num_elems, num_nodes, conn, InterfaceElementType::LINEAR_QUAD, x2, x2 + 4, x2 + 8);
    constexpr int cs_id = 0;
    // mortar then nonmortar surfaces
    registerCouplingScheme(cs_id, mesh_id2, mesh_id1, ContactMode::SURFACE_TO_SURFACE, ContactCase::NO_CASE,
                           ContactMethod::SINGLE_MORTAR, ContactModel::FRICTIONLESS,
                           EnforcementMethod::LAGRANGE_MULTIPLIER, BinningMethod::BINNING_GRID);
    double f1t[12];
    for (int i{0}; i < 12; ++i) {
      f1t[i] = 0.0;
    }
    registerNodalResponse(mesh_id1, f1t, f1t + 4, f1t + 8);
    double f2t[12];
    for (int i{0}; i < 12; ++i) {
      f2t[i] = 0.0;
    }
    registerNodalResponse(mesh_id2, f2t, f2t + 4, f2t + 8);
    double g1t[4];
    for (int i{0}; i < 4; ++i) {
      g1t[i] = 0.0;
    }
    registerMortarGaps(mesh_id1, g1t);
    registerMortarPressures(mesh_id1, p1);
    setLagrangeMultiplierOptions(cs_id, ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN, SparseMode::MFEM_ELEMENT_DENSE);

    int cycle = 1;
    double t = 0.0;
    double dt = 1.0;
    update(cycle, t, dt);

    const ArrayT<int>* row_elem_idx = nullptr;
    const ArrayT<int>* col_elem_idx = nullptr;
    const ArrayT<mfem::DenseMatrix>* jacobians = nullptr;
    getElementBlockJacobians(cs_id, BlockSpace::NONMORTAR, BlockSpace::LAGRANGE_MULTIPLIER, &row_elem_idx,
                             &col_elem_idx, &jacobians);

    std::cout << "df1/dp = " << std::endl;
    for (int i{0}; i < 12; ++i) {
      for (int j{0}; j < 4; ++j) {
        int idx_e = 4 * i + j;
        auto diff = std::abs(df1dp1[idx_e] - (*jacobians)[0].Data()[idx_e]);
        if (diff > approx_j_err_) {
          std::cout << "[" << idx_e << "] Enzyme: " << df1dp1[idx_e] << "  Tribol: " << (*jacobians)[0].Data()[idx_e]
                    << std::endl;
        }
        EXPECT_NEAR(df1dp1[idx_e], (*jacobians)[0].Data()[idx_e], approx_j_err_);
      }
    }

    getElementBlockJacobians(cs_id, BlockSpace::MORTAR, BlockSpace::LAGRANGE_MULTIPLIER, &row_elem_idx, &col_elem_idx,
                             &jacobians);

    std::cout << "df2/dp = " << std::endl;
    for (int i{0}; i < 12; ++i) {
      for (int j{0}; j < 4; ++j) {
        int idx_e = 4 * i + j;
        auto diff = std::abs(df2dp1[idx_e] - (*jacobians)[0].Data()[idx_e]);
        if (diff > approx_j_err_) {
          std::cout << "[" << idx_e << "] Enzyme: " << df2dp1[idx_e] << "  Tribol: " << (*jacobians)[0].Data()[idx_e]
                    << std::endl;
        }
        EXPECT_NEAR(df2dp1[idx_e], (*jacobians)[0].Data()[idx_e], approx_j_err_);
      }
    }

    getElementBlockJacobians(cs_id, BlockSpace::LAGRANGE_MULTIPLIER, BlockSpace::NONMORTAR, &row_elem_idx,
                             &col_elem_idx, &jacobians);

    std::cout << "dg/dx1 = " << std::endl;
    for (int i{0}; i < 12; ++i) {
      for (int j{0}; j < 4; ++j) {
        int idx_e = 4 * i + j;
        auto diff = std::abs(dg1dx1[idx_e] - (*jacobians)[0].Data()[idx_e]);
        if (diff > approx_j_err_) {
          std::cout << "[" << idx_e << "] Enzyme: " << dg1dx1[idx_e] << "  Tribol: " << (*jacobians)[0].Data()[idx_e]
                    << std::endl;
        }
        EXPECT_NEAR(dg1dx1[idx_e], (*jacobians)[0].Data()[idx_e], approx_j_err_);
      }
    }

    getElementBlockJacobians(cs_id, BlockSpace::LAGRANGE_MULTIPLIER, BlockSpace::MORTAR, &row_elem_idx, &col_elem_idx,
                             &jacobians);

    std::cout << "dg/dx2 = " << std::endl;
    for (int i{0}; i < 12; ++i) {
      for (int j{0}; j < 4; ++j) {
        int idx_e = 4 * i + j;
        auto diff = std::abs(dg1dx2[idx_e] - (*jacobians)[0].Data()[idx_e]);
        if (diff > approx_j_err_) {
          std::cout << "[" << idx_e << "] Enzyme: " << dg1dx2[idx_e] << "  Tribol: " << (*jacobians)[0].Data()[idx_e]
                    << std::endl;
        }
        EXPECT_NEAR(dg1dx2[idx_e], (*jacobians)[0].Data()[idx_e], approx_j_err_);
      }
    }

    std::cout << "g = " << std::endl;
    for (int i{0}; i < 4; ++i) {
      auto diff = std::abs(g1[i] - g1t[i]);
      if (diff > tribol_vs_enzyme_err_) {
        std::cout << "[" << i << "] Enzyme: " << g1[i] << "  Tribol: " << g1t[i] << std::endl;
      }
      EXPECT_NEAR(g1[i], g1t[i], tribol_vs_enzyme_err_);
    }

    std::cout << "f1 = " << std::endl;
    for (int i{0}; i < 12; ++i) {
      auto diff = std::abs(f1[i] - f1t[i]);
      if (diff > tribol_vs_enzyme_err_) {
        std::cout << "[" << i << "] Enzyme: " << f1[i] << "  Tribol: " << f1t[i] << std::endl;
      }
      EXPECT_NEAR(f1[i], f1t[i], tribol_vs_enzyme_err_);
    }

    std::cout << "f2 = " << std::endl;
    for (int i{0}; i < 12; ++i) {
      auto diff = std::abs(f2[i] - f2t[i]);
      if (diff > tribol_vs_enzyme_err_) {
        std::cout << "[" << i << "] Enzyme: " << f2[i] << "  Tribol: " << f2t[i] << std::endl;
      }
      EXPECT_NEAR(f2[i], f2t[i], tribol_vs_enzyme_err_);
    }
  }
};

TEST_F(EnzymeJacobianTest, ExactOverlap)
{
  // clang-format off
  // {x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3}
  double x1[12] = { 0.0,   1.0,   1.0,   0.0,
                    0.0,   0.0,   1.0,   1.0,
                    0.0,   0.0,   0.0,   0.0 };
  double x2[12] = { 0.0,   0.0,   1.0,   1.0,
                    0.0,   1.0,   1.0,   0.0,
                    0.0,   0.0,   0.0,   0.0 };
  double n1[12] = { 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 1.0 };
  double p1[4] = { 1.0, 1.0, 1.0, 1.0 };
  double x1_stencil[12] = {  1.0, -1.0, -1.0,  1.0,
                             1.0,  1.0, -1.0, -1.0,
                             -1.0,  -1.0,  -1.0,  -1.0 };
  double x2_stencil[12] = {  1.0,  1.0, -1.0, -1.0,
                             1.0, -1.0, -1.0,  1.0,
                             1.0,  1.0,  1.0,  1.0 };
  // clang-format on

  FDCheck(x1, x2, n1, p1, x1_stencil, x2_stencil);
  ApproxJacobianCheck(x1, x2, n1, p1);
}

TEST_F(EnzymeJacobianTest, SlightlySmallerNonmortarElement)
{
  // slightly smaller
  double dx = 4.0 * delta_;
  // clang-format off
  // {x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3}
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

  FDCheck(x1, x2, n1, p1);
  ApproxJacobianCheck(x1, x2, n1, p1);
}

TEST_F(EnzymeJacobianTest, ShiftedXNonmortarElement)
{
  // slightly smaller and offset
  double offset = 0.3;
  double dx = 4.0 * delta_;
  // clang-format off
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

  FDCheck(x1, x2, n1, p1);
}

TEST_F(EnzymeJacobianTest, ShiftedXYNonmortarElement)
{
  // slightly smaller and offset
  double offset = 0.3;
  double dx = 4.0 * delta_;
  // clang-format off
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

  FDCheck(x1, x2, n1, p1);
}

TEST_F(EnzymeJacobianTest, Rotated30DegNonmortarElement)
{
  // clang-format off
  // rotate 30 degrees
  double x1[12] = { 0.0,  1.0,  1.0,  0.0,
                    0.0,  0.0,  1.0,  1.0,
                    0.01, 0.01, 0.01, 0.01 };
  // clang-format on
  double cos30 = std::cos(redecomp::pi / 6.0);
  double sin30 = std::sin(redecomp::pi / 6.0);
  for (int i{0}; i < 4; ++i) {
    double x_new = x1[i] * cos30 - x1[i + 4] * sin30;
    double y_new = x1[i] * sin30 + x1[i + 4] * cos30;
    x1[i] = x_new;
    x1[i + 4] = y_new;
  }
  // shift to center the element at (0.5, 0.5)
  double x_shift = 0.25;
  double y_shift = -0.5 * (x1[6] - 1.0);
  for (int i{0}; i < 4; ++i) {
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

  FDCheck(x1, x2, n1, p1);
}

TEST_F(EnzymeJacobianTest, NonaffineRotated45DegMortarElement)
{
  // clang-format off
  // rotate 45 degrees
  double x1[12] = { 0.0,  1.1,  1.0,  0.0,
                    0.0,  0.0,  1.1,  1.0,
                    0.01, 0.01, 0.01, 0.01 };
  // clang-format on
  double cos45 = std::cos(redecomp::pi / 4.0);
  double sin45 = std::sin(redecomp::pi / 4.0);
  for (int i{0}; i < 4; ++i) {
    double x_new = x1[i] * cos45 - x1[i + 4] * sin45;
    double y_new = x1[i] * sin45 + x1[i + 4] * cos45;
    x1[i] = x_new;
    x1[i + 4] = y_new;
  }
  // shift to center the element near (0.5, 0.5)
  double x_shift = 0.5 / std::sqrt(2.0) + 0.1;
  double y_shift = -0.5 / std::sqrt(2.0) + 0.1;
  for (int i{0}; i < 4; ++i) {
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

  FDCheck(x1, x2, n1, p1);
}

TEST_F(EnzymeJacobianTest, NoOverlap)
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
  // clang-format on

  FDCheck(x1, x2, n1, p1);
}

}  // namespace tribol

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  int result = 0;

  MPI_Init(&argc, &argv);

  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
