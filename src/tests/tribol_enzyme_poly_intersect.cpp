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
  double delta_ {1.0e-7};
  void SetUp() override
  {

  }

  void CheckIntersection(RealT* x1, RealT* x2, RealT pos_tol = 1.0e-8, RealT len_tol = 1.0e-8)
  {
    RealT xi[16];
    OverlapVertexType type[8];
    auto num_poly_verts = 0;
    RealT area = 0.0;
    Intersection2DPolygon(x1, x1 + 4, 4, x2, x2 + 4, 4, pos_tol, len_tol,
      xi, xi + 8, type, num_poly_verts, area, true);
    std::cout << "Number of vertices: " << num_poly_verts << "   Polygon area: "
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
    RealT dxidx1[16*8];
    RealT dxidx2[16*8];
    for (int i{0}; i < 16*8; ++i)
    {
      dxidx1[i] = 0.0;
      dxidx2[i] = 0.0;
    }

    for (int i{0}; i < 4; ++i)
    {
      x_dot[i] = 1.0;
      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_dup, x1, x_dot,
        enzyme_const, x1 + 4,
        enzyme_const, 4,
        enzyme_const, x2,
        enzyme_const, x2 + 4,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx1 + 16*i,
        enzyme_dup, xi + 8, dxidx1 + 16*i + 8,
        enzyme_const, type,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true,
        enzyme_runtime_activity
      );
      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_const, x1,
        enzyme_dup, x1 + 4, x_dot,
        enzyme_const, 4,
        enzyme_const, x2,
        enzyme_const, x2 + 4,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx1 + 16*(4 + i),
        enzyme_dup, xi + 8, dxidx1 + 16*(4 + i) + 8,
        enzyme_const, type,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true,
        enzyme_runtime_activity
      );
      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_const, x1,
        enzyme_const, x1 + 4,
        enzyme_const, 4,
        enzyme_dup, x2, x_dot,
        enzyme_const, x2 + 4,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx2 + 16*i,
        enzyme_dup, xi + 8, dxidx2 + 16*i + 8,
        enzyme_const, type,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true,
        enzyme_runtime_activity
      );
      __enzyme_fwddiff<void>((void*)Intersection2DPolygon,
        enzyme_const, x1,
        enzyme_const, x1 + 4,
        enzyme_const, 4,
        enzyme_const, x2,
        enzyme_dup, x2 + 4, x_dot,
        enzyme_const, 4,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx2 + 16*(4 + i),
        enzyme_dup, xi + 8, dxidx2 + 16*(4 + i) + 8,
        enzyme_const, type,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true,
        enzyme_runtime_activity
      );
      x_dot[i] = 0.0;
    }

    mfem::DenseMatrix dxidx1_dense(dxidx1, 16, 8);
    std::ofstream dxidx1_file("dxidx1.mat");
    dxidx1_dense.PrintMatlab(dxidx1_file);
    dxidx1_file.close();
    mfem::DenseMatrix dxidx2_dense(dxidx2, 16, 8);
    std::ofstream dxidx2_file("dxidx2.mat");
    dxidx2_dense.PrintMatlab(dxidx2_file);
    dxidx2_file.close();

    // for (int j{0}; j < 4; ++j)
    // {
    //   x1[j] += delta_;
    //   tribol::ComputeMortarForceEnzyme(x1, n1, p1, f1, g1, 4, x2, f2, 4);
    //   for (int i{0}; i < 12; ++i)
    //   {
    //     df1dx1_fd[j*12+i] += f1[i];
    //     df1dx1_fd[j*12+i] /= delta_;
    //     df2dx1_fd[j*12+i] += f2[i];
    //     df2dx1_fd[j*12+i] /= delta_;
    //   }
    //   x1[j] -= delta_;
    // }
    
  }
};

TEST_F(EnzymePolyIntersectTest, PerfectOverlap)
{
  constexpr auto pos_tol = 1.0e-8;
  constexpr auto len_tol = 1.0e-8; 
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0,
                  0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.0, 1.0, 1.0, 0.0,
                  0.0, 0.0, 1.0, 1.0 };
  CheckIntersection(x1, x2, pos_tol, len_tol);
}

TEST_F(EnzymePolyIntersectTest, Mesh2VertexMovedInByPosTol)
{
  constexpr auto pos_tol = 1.0e-8;
  constexpr auto len_tol = 1.0e-8; 
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0,
                  0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.0+pos_tol, 1.0, 1.0, 0.0,
                  0.0, 0.0, 1.0, 1.0 };
  CheckIntersection(x1, x2, pos_tol, len_tol);
}

TEST_F(EnzymePolyIntersectTest, NearlyParallelEdges)
{
  constexpr auto pos_tol = 1.0e-8;
  constexpr auto len_tol = 1.0e-8; 
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0,
                  0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.0, 1.0, 1.0, 0.0,
                  0.0-2*pos_tol, 0.0+2*pos_tol, 1.0, 1.0 };
  CheckIntersection(x1, x2, pos_tol, len_tol);
}

TEST_F(EnzymePolyIntersectTest, LessNearlyParallelEdges)
{
  constexpr auto pos_tol = 1.0e-8;
  constexpr auto len_tol = 1.0e-8; 
  constexpr auto offset = 0.2;
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0,
                  0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.0, 1.0, 1.0, 0.0,
                  0.0-offset, 0.0+offset, 1.0, 1.0 };
  CheckIntersection(x1, x2, pos_tol, len_tol);
}

TEST_F(EnzymePolyIntersectTest, OffsetElements)
{
  constexpr auto pos_tol = 1.0e-8;
  constexpr auto len_tol = 1.0e-8;
  constexpr auto offset = 0.2;
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0,
                  0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.0+offset, 1.0+offset, 1.0+offset, 0.0+offset,
                  0.0+offset, 0.0+offset, 1.0+offset, 1.0+offset };
  CheckIntersection(x1, x2, pos_tol, len_tol);
}

TEST_F(EnzymePolyIntersectTest, EightOverlapVertices)
{
  constexpr auto pos_tol = 1.0e-8;
  constexpr auto len_tol = 1.0e-8;
  auto xmin = -1.0 / std::sqrt(2.0) + 0.5;
  auto xmax = 1.0 / std::sqrt(2.0) + 0.5;
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0,
                  0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.5, xmax, 0.5, xmin,
                  xmin, 0.5, xmax, 0.5 };
  CheckIntersection(x1, x2, pos_tol, len_tol);
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

