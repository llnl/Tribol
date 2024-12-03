// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

//-----------------------------------------------------------------------------
//
// file: tribol_enzyme_poly_intersect.cpp
//
//-----------------------------------------------------------------------------

#include <iostream>

#include "tribol/config.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/geom/GeomUtilities.hpp"

#include "mfem.hpp"

#include "gtest/gtest.h"

namespace tribol {

class EnzymePolyIntersectTest : public testing::Test {
protected:
  static constexpr double delta_ {1.0e-8};
  void SetUp() override
  {

  }

  void CheckIntersection(RealT* x1, RealT* x2, int* stencil_dir, RealT pos_tol = 1.0e-8, RealT len_tol = 1.0e-8, std::string name = "")
  {
    RealT xi[16];
    OverlapVertexType type[8];
    IndexT edge1[8];
    IndexT edge2[8];
    for (int i{0}; i < 8; ++i)
    {
      xi[i] = 0.0;
      xi[i+8] = 0.0;
      type[i] = OverlapVertexType::A;
      edge1[i] = -1;
      edge2[i] = -1;
    }
    auto num_poly_verts = 0;
    RealT area = 0.0;
    Intersection2DPolygon(x1, x1 + 4, 4, x2, x2 + 4, 4, pos_tol, len_tol,
      xi, xi + 8, type, edge1, edge2, num_poly_verts, area, true);
    std::cout << std::setprecision(15) << "Element 1 coords" << std::endl;
    for (int i{0}; i < 4; ++i)
    {
      std::cout << "(" << x1[i] << ", " << x1[i+4] << ")\n";
    }
    std::cout << std::setprecision(15) << "Element 2 coords" << std::endl;
    for (int i{0}; i < 4; ++i)
    {
      std::cout << "(" << x2[i] << ", " << x2[i+4] << ")\n";
    }
    std::cout << std::setprecision(15) << "Number of vertices: " << num_poly_verts << "   Polygon area: "
      << area << std::endl;

    for (int i{0}; i < num_poly_verts; ++i)
    {
      std::cout << "  Coord: (" << xi[i] << ", " << xi[i+8] << ")   Type: ";
      switch (type[i])
      {
        case OverlapVertexType::A:
          std::cout << "Vertex A" << std::endl;
          break;
        case OverlapVertexType::B:
          std::cout << "Vertex B" << std::endl;
          break;
        case OverlapVertexType::EdgeEdge:
          std::cout << "Edge/Edge" << std::endl;
          break;
      }
    }

    RealT x_dot[4] = {0.0, 0.0, 0.0, 0.0};
    RealT zeros[4] = {0.0, 0.0, 0.0, 0.0};
    RealT dxidx1[16*8];
    RealT dxidx2[16*8];
    for (int i{0}; i < 16*8; ++i)
    {
      dxidx1[i] = 0.0;
      dxidx2[i] = 0.0;
    }

    RealT xi_base[16];
    for (int i{0}; i < 16; ++i)
    {
      xi_base[i] = xi[i];
    }

    for (int i{0}; i < 4; ++i)
    {
      x_dot[i] = 1.0;
      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_dup, x1, x_dot,
        enzyme_dup, x1 + 4, zeros,
        enzyme_const, 4,
        enzyme_dup, x2, zeros,
        enzyme_dup, x2 + 4, zeros,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx1 + 16*i,
        enzyme_dup, xi + 8, dxidx1 + 16*i + 8,
        enzyme_const, type,
        enzyme_const, edge1,
        enzyme_const, edge2,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true
      );
    
      std::cout << std::setprecision(15) << "Element 1 coords" << std::endl;
      for (int i{0}; i < 4; ++i)
      {
        std::cout << "(" << x1[i] << ", " << x1[i+4] << ")\n";
      }
      std::cout << std::setprecision(15) << "Element 2 coords" << std::endl;
      for (int i{0}; i < 4; ++i)
      {
        std::cout << "(" << x2[i] << ", " << x2[i+4] << ")\n";
      }
      std::cout << std::setprecision(15) << "Number of vertices: " << num_poly_verts << "   Polygon area: "
        << area << std::endl;
      for (int i{0}; i < num_poly_verts; ++i)
      {
        std::cout << "  Coord: (" << xi[i] << ", " << xi[i+8] << ")   Type: ";
        switch (type[i])
        {
          case OverlapVertexType::A:
            std::cout << "Vertex A" << std::endl;
            break;
          case OverlapVertexType::B:
            std::cout << "Vertex B" << std::endl;
            break;
          case OverlapVertexType::EdgeEdge:
            std::cout << "Edge/Edge" << std::endl;
            break;
        }
      }

      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_dup, x1, zeros,
        enzyme_dup, x1 + 4, x_dot,
        enzyme_const, 4,
        enzyme_dup, x2, zeros,
        enzyme_dup, x2 + 4, zeros,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx1 + 16*(4 + i),
        enzyme_dup, xi + 8, dxidx1 + 16*(4 + i) + 8,
        enzyme_const, type,
        enzyme_const, edge1,
        enzyme_const, edge2,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true
      );
    
      std::cout << std::setprecision(15) << "Element 1 coords" << std::endl;
      for (int i{0}; i < 4; ++i)
      {
        std::cout << "(" << x1[i] << ", " << x1[i+4] << ")\n";
      }
      std::cout << std::setprecision(15) << "Element 2 coords" << std::endl;
      for (int i{0}; i < 4; ++i)
      {
        std::cout << "(" << x2[i] << ", " << x2[i+4] << ")\n";
      }
      std::cout << std::setprecision(15) << "Number of vertices: " << num_poly_verts << "   Polygon area: "
        << area << std::endl;
      for (int i{0}; i < num_poly_verts; ++i)
      {
        std::cout << "  Coord: (" << xi[i] << ", " << xi[i+8] << ")   Type: ";
        switch (type[i])
        {
          case OverlapVertexType::A:
            std::cout << "Vertex A" << std::endl;
            break;
          case OverlapVertexType::B:
            std::cout << "Vertex B" << std::endl;
            break;
          case OverlapVertexType::EdgeEdge:
            std::cout << "Edge/Edge" << std::endl;
            break;
        }
      }
      
      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_dup, x1, zeros,
        enzyme_dup, x1 + 4, zeros,
        enzyme_const, 4,
        enzyme_dup, x2, x_dot,
        enzyme_dup, x2 + 4, zeros,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx2 + 16*i,
        enzyme_dup, xi + 8, dxidx2 + 16*i + 8,
        enzyme_const, type,
        enzyme_const, edge1,
        enzyme_const, edge2,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true
      );
    
      std::cout << std::setprecision(15) << "Element 1 coords" << std::endl;
      for (int i{0}; i < 4; ++i)
      {
        std::cout << "(" << x1[i] << ", " << x1[i+4] << ")\n";
      }
      std::cout << std::setprecision(15) << "Element 2 coords" << std::endl;
      for (int i{0}; i < 4; ++i)
      {
        std::cout << "(" << x2[i] << ", " << x2[i+4] << ")\n";
      }
      std::cout << std::setprecision(15) << "Number of vertices: " << num_poly_verts << "   Polygon area: "
        << area << std::endl;
      for (int i{0}; i < num_poly_verts; ++i)
      {
        std::cout << "  Coord: (" << xi[i] << ", " << xi[i+8] << ")   Type: ";
        switch (type[i])
        {
          case OverlapVertexType::A:
            std::cout << "Vertex A" << std::endl;
            break;
          case OverlapVertexType::B:
            std::cout << "Vertex B" << std::endl;
            break;
          case OverlapVertexType::EdgeEdge:
            std::cout << "Edge/Edge" << std::endl;
            break;
        }
      }
      
      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_dup, x1, zeros,
        enzyme_dup, x1 + 4, zeros,
        enzyme_const, 4,
        enzyme_dup, x2, zeros,
        enzyme_dup, x2 + 4, x_dot,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx2 + 16*(4 + i),
        enzyme_dup, xi + 8, dxidx2 + 16*(4 + i) + 8,
        enzyme_const, type,
        enzyme_const, edge1,
        enzyme_const, edge2,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true
      );
    
      std::cout << std::setprecision(15) << "Element 1 coords" << std::endl;
      for (int i{0}; i < 4; ++i)
      {
        std::cout << "(" << x1[i] << ", " << x1[i+4] << ")\n";
      }
      std::cout << std::setprecision(15) << "Element 2 coords" << std::endl;
      for (int i{0}; i < 4; ++i)
      {
        std::cout << "(" << x2[i] << ", " << x2[i+4] << ")\n";
      }
      std::cout << std::setprecision(15) << "Number of vertices: " << num_poly_verts << "   Polygon area: "
        << area << std::endl;
      for (int i{0}; i < num_poly_verts; ++i)
      {
        std::cout << "  Coord: (" << xi[i] << ", " << xi[i+8] << ")   Type: ";
        switch (type[i])
        {
          case OverlapVertexType::A:
            std::cout << "Vertex A" << std::endl;
            break;
          case OverlapVertexType::B:
            std::cout << "Vertex B" << std::endl;
            break;
          case OverlapVertexType::EdgeEdge:
            std::cout << "Edge/Edge" << std::endl;
            break;
        }
      }
      
      x_dot[i] = 0.0;
    }

    std::cout << "dxi/dx1 nonzero values:" << std::endl;
    for (int j{0}; j < 8; ++j)
    {
        for (int i{0}; i < 16; ++i)
        {
            auto idx = j*16 + i;
            if (std::abs(dxidx1[idx]) > 1.0e-15)
            {
                std::cout << "  (" << i << ", " << j << ") = " << dxidx1[idx] << std::endl;
            }
        }
    }

    std::cout << "dxi/dx2 nonzero values:" << std::endl;
    for (int j{0}; j < 8; ++j)
    {
        for (int i{0}; i < 16; ++i)
        {
            auto idx = j*16 + i;
            if (std::abs(dxidx2[idx]) > 1.0e-15)
            {
                std::cout << "  (" << i << ", " << j << ") = " << dxidx2[idx] << std::endl;
            }
        }
    }

    mfem::DenseMatrix dxidx1_dense(dxidx1, 16, 8);
    std::ofstream dxidx1_file(name + "_dxidx1.mat");
    dxidx1_dense.PrintMatlab(dxidx1_file);
    dxidx1_file.close();
    mfem::DenseMatrix dxidx2_dense(dxidx2, 16, 8);
    std::ofstream dxidx2_file(name + "_dxidx2.mat");
    dxidx2_dense.PrintMatlab(dxidx2_file);
    dxidx2_file.close();

    RealT dxidx1_fd[16*8];
    RealT dxidx2_fd[16*8];
    // row 1 assumes x2 is inside x1; row 2 assumes x1 is inside x2
    RealT x_sgn1[8] = {1.0, -1.0, -1.0, 1.0,
                       -1.0, 1.0, 1.0, -1.0};
    RealT y_sgn1[8] = {1.0, 1.0, -1.0, -1.0,
                       -1.0, -1.0, 1.0, 1.0};
    RealT x_sgn2[8] = {-1.0, 1.0, 1.0, -1.0,
                       1.0, -1.0, -1.0, 1.0};
    RealT y_sgn2[8] = {-1.0, -1.0, 1.0, 1.0,
                       1.0, 1.0, -1.0, -1.0};
    for (int j{0}; j < 4; ++j)
    {
      x1[j] += x_sgn1[4*stencil_dir[j] + j]*delta_;
      for (int i{0}; i < 16; ++i)
      {
        xi[i] = 0.0;
      }
      Intersection2DPolygon(x1, x1 + 4, 4, x2, x2 + 4, 4, pos_tol, len_tol,
        xi, xi + 8, type, edge1, edge2, num_poly_verts, area, true);
      for (int i{0}; i < 16; ++i)
      {
        if (j == 0 && i == 13)
        {
          std::cout << "  Entry (13, 0): " << x_sgn1[4*stencil_dir[j] + j] << " "
            << xi[i] << " " << xi_base[i] << std::endl;
        }
        dxidx1_fd[16*j + i] = x_sgn1[4*stencil_dir[j] + j]*(xi[i] - xi_base[i])/delta_;
      }
      x1[j] -= x_sgn1[4*stencil_dir[j] + j]*delta_;

      x1[j + 4] += y_sgn1[4*stencil_dir[j] + j]*delta_;
      for (int i{0}; i < 16; ++i)
      {
        xi[i] = 0.0;
      }
      Intersection2DPolygon(x1, x1 + 4, 4, x2, x2 + 4, 4, pos_tol, len_tol,
        xi, xi + 8, type, edge1, edge2, num_poly_verts, area, true);
      for (int i{0}; i < 16; ++i)
      {
        dxidx1_fd[16*(4 + j) + i] = y_sgn1[4*stencil_dir[j] + j]*(xi[i] - xi_base[i])/delta_;
      }
      x1[j + 4] -= y_sgn1[4*stencil_dir[j] + j]*delta_;

      x2[j] += x_sgn2[4*stencil_dir[j] + j]*delta_;
      for (int i{0}; i < 16; ++i)
      {
        xi[i] = 0.0;
      }
      Intersection2DPolygon(x1, x1 + 4, 4, x2, x2 + 4, 4, pos_tol, len_tol,
        xi, xi + 8, type, edge1, edge2, num_poly_verts, area, true);
      for (int i{0}; i < 16; ++i)
      {
        dxidx2_fd[16*j + i] = x_sgn2[4*stencil_dir[j] + j]*(xi[i] - xi_base[i])/delta_;
      }
      x2[j] -= x_sgn2[4*stencil_dir[j] + j]*delta_;

      x2[j + 4] += y_sgn2[4*stencil_dir[j] + j]*delta_;
      for (int i{0}; i < 16; ++i)
      {
        xi[i] = 0.0;
      }
      Intersection2DPolygon(x1, x1 + 4, 4, x2, x2 + 4, 4, pos_tol, len_tol,
        xi, xi + 8, type, edge1, edge2, num_poly_verts, area, true);
      for (int i{0}; i < 16; ++i)
      {
        dxidx2_fd[16*(4 + j) + i] = y_sgn2[4*stencil_dir[j] + j]*(xi[i] - xi_base[i])/delta_;
      }
      x2[j + 4] -= y_sgn2[4*stencil_dir[j] + j]*delta_;
    }

    // write deltas to screen
    std::cout << "dxi/dx1 ------------------------------" << std::endl;
    for (int j{0}; j < 8; ++j)
    {
      for (int i{0}; i < 16; ++i)
      {
        auto idx = j*16 + i;
        auto diff = std::abs(dxidx1[idx] - dxidx1_fd[idx]);
        if (diff > 10.0 * delta_)
        {
          std::cout << "  (" << i << ", " << j << ") : Diff: " << 
            diff << "   Ratio: " << dxidx1[idx] / dxidx1_fd[idx] << "   Enzyme: " <<
            dxidx1[idx] << "   FD: " << dxidx1_fd[idx] << std::endl;
        }
        // EXPECT_NEAR(dxidx1[idx], dxidx1_fd[idx], delta_);
      }
    }
    std::cout << "dxi/dx2 ------------------------------" << std::endl;
    for (int j{0}; j < 8; ++j)
    {
      for (int i{0}; i < 16; ++i)
      {
        auto idx = j*16 + i;
        auto diff = std::abs(dxidx2[idx] - dxidx2_fd[idx]);
        if (diff > 10.0 * delta_)
        {
          std::cout << "  (" << i << ", " << j << ") : Diff: " << 
            diff << "   Ratio: " << dxidx2[idx] / dxidx2_fd[idx] << "   Enzyme: " <<
            dxidx2[idx] << "   FD: " << dxidx2_fd[idx] << std::endl;
        }
        // EXPECT_NEAR(dxidx2[idx], dxidx2_fd[idx], delta_);
      }
    }
    
  }
};

// TEST_F(EnzymePolyIntersectTest, PerfectOverlap)
// {
//   constexpr auto pos_tol = 10.0 * delta_;
//   constexpr auto len_tol = 10.0 * delta_; 
//   RealT x1[8] = { 0.0, 1.0, 1.0, 0.0,
//                   0.0, 0.0, 1.0, 1.0 };
//   RealT x2[8] = { 0.0, 1.0, 1.0, 0.0,
//                   0.0, 0.0, 1.0, 1.0 };
//   int stencil_dir[4] = {0, 0, 0, 0};
//   CheckIntersection(x1, x2, stencil_dir, pos_tol, len_tol, "perfect_overlap");
// }

// TEST_F(EnzymePolyIntersectTest, Mesh2VertexMovedInByPosTol)
// {
//   constexpr auto pos_tol = 10.0 * delta_;
//   constexpr auto len_tol = 10.0 * delta_; 
//   RealT x1[8] = { 0.0, 1.0, 1.0, 0.0,
//                   0.0, 0.0, 1.0, 1.0 };
//   RealT x2[8] = { 0.0+pos_tol, 1.0, 1.0, 0.0,
//                   0.0, 0.0, 1.0, 1.0 };
//   int stencil_dir[4] = {1, 1, 1, 1};
//   CheckIntersection(x1, x2, stencil_dir, pos_tol, len_tol, "onevertpostol");
// }

// TEST_F(EnzymePolyIntersectTest, NearlyParallelEdges)
// {
//   constexpr auto pos_tol = 10.0 * delta_;
//   constexpr auto len_tol = 10.0 * delta_; 
//   RealT x1[8] = { 0.0, 1.0, 1.0, 0.0,
//                   0.0, 0.0, 1.0, 1.0 };
//   RealT x2[8] = { 0.0, 1.0, 1.0, 0.0,
//                   0.0-2*pos_tol, 0.0+2*pos_tol, 1.0, 1.0 };
//   int stencil_dir[4] = {0, 1, 0, 0};
//   CheckIntersection(x1, x2, stencil_dir, pos_tol, len_tol, "nearlyparallel");
// }

TEST_F(EnzymePolyIntersectTest, LessNearlyParallelEdges)
{
  constexpr auto pos_tol = 10.0 * delta_;
  constexpr auto len_tol = 10.0 * delta_; 
  constexpr auto offset = 0.2;
  // shift node 3 in a little to prevent edge class change with FD
  RealT x1[8] = { 0.0, 1.0, 1.0 - pos_tol, 0.0,
                  0.0, 0.0, 1.0 - pos_tol, 1.0 };
  RealT x2[8] = { 0.0, 1.0, 1.0, 0.0,
                  0.0-offset, 0.0+offset, 1.0, 1.0 };
  int stencil_dir[4] = {0, 1, 1, 0};
  CheckIntersection(x1, x2, stencil_dir, pos_tol, len_tol, "kindaparallel");
}

// TEST_F(EnzymePolyIntersectTest, OffsetElements)
// {
//   constexpr auto pos_tol = 10.0 * delta_;
//   constexpr auto len_tol = 10.0 * delta_;
//   constexpr auto offset = 0.2;
//   RealT x1[8] = { 0.0, 1.0, 1.0, 0.0,
//                   0.0, 0.0, 1.0, 1.0 };
//   RealT x2[8] = { 0.0+offset, 1.0+offset, 1.0+offset, 0.0+offset,
//                   0.0+offset, 0.0+offset, 1.0+offset, 1.0+offset };
//   int stencil_dir[4] = {1, 0, 0, 0};
//   CheckIntersection(x1, x2, stencil_dir, pos_tol, len_tol, "offset");
// }

// TEST_F(EnzymePolyIntersectTest, EightOverlapVertices)
// {
//   constexpr auto pos_tol = 10.0 * delta_;
//   constexpr auto len_tol = 10.0 * delta_;
//   auto xmin = -1.0 / std::sqrt(2.0) + 0.5;
//   auto xmax = 1.0 / std::sqrt(2.0) + 0.5;
//   RealT x1[8] = { 0.0, 1.0, 1.0, 0.0,
//                   0.0, 0.0, 1.0, 1.0 };
//   RealT x2[8] = { 0.5, xmax, 0.5, xmin,
//                   xmin, 0.5, xmax, 0.5 };
//   int stencil_dir[4] = {0, 0, 0, 0};
//   CheckIntersection(x1, x2, stencil_dir, pos_tol, len_tol, "twist");
// }

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

