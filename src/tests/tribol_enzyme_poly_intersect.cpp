// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
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

#include "gtest/gtest.h"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

#include "mfem.hpp"

#include "tribol/common/Enzyme.hpp"
#include "tribol/geom/GeomUtilities.hpp"

namespace tribol {

/**
 * @brief Test fixture for the Enzyme-based derivatives of intersection polynomial calculations.
 */
class EnzymePolyIntersectTest : public testing::Test {
 protected:
  static constexpr double delta_{ 1.0e-8 };
  void SetUp() override {}

  void CheckIntersection( RealT* x1, RealT* x2, int* stencil_dir, RealT fd_tol = delta_, RealT pos_tol = 1.0e-8,
                          RealT len_tol = 1.0e-8, [[maybe_unused]] std::string name = "" )
  {
    constexpr int max_overlap_vert = 8;
    constexpr int dim = 2;
    RealT xi[max_overlap_vert * dim];
    for ( int i{ 0 }; i < max_overlap_vert; ++i ) {
      xi[i] = 0.0;
      xi[i + max_overlap_vert] = 0.0;
    }
    auto num_poly_verts = 0;
    RealT area = 0.0;
    constexpr int num_elem_coords = 4;
    Intersection2DPolygon( x1, x1 + num_elem_coords, num_elem_coords, x2, x2 + num_elem_coords, num_elem_coords,
                           pos_tol, len_tol, xi, xi + max_overlap_vert, num_poly_verts, area, true );
    std::cout << std::setprecision( 15 ) << "Element 1 coords" << std::endl;
    for ( int i{ 0 }; i < num_elem_coords; ++i ) {
      std::cout << "(" << x1[i] << ", " << x1[i + 4] << ")\n";
    }
    std::cout << std::setprecision( 15 ) << "Element 2 coords" << std::endl;
    for ( int i{ 0 }; i < num_elem_coords; ++i ) {
      std::cout << "(" << x2[i] << ", " << x2[i + 4] << ")\n";
    }
    std::cout << std::setprecision( 15 ) << "Number of vertices: " << num_poly_verts << "   Polygon area: " << area
              << std::endl;

    RealT x_dot[4] = { 0.0, 0.0, 0.0, 0.0 };
    RealT dxidx1[( max_overlap_vert * num_elem_coords ) * dim];
    RealT dxidx2[( max_overlap_vert * num_elem_coords ) * dim];
    for ( int i{ 0 }; i < ( max_overlap_vert * num_elem_coords ) * dim; ++i ) {
      dxidx1[i] = 0.0;
      dxidx2[i] = 0.0;
    }

    RealT xi_base[max_overlap_vert * dim];
    for ( int i{ 0 }; i < max_overlap_vert * dim; ++i ) {
      xi_base[i] = xi[i];
    }

    for ( int i{ 0 }; i < num_elem_coords; ++i ) {
      x_dot[i] = 1.0;
      // clang-format off
      // wiggle the xi coordinate of node i in element 1
      __enzyme_fwddiff<void>( (void*)Intersection2DPolygon,
        enzyme_dup, x1, x_dot, 
        enzyme_const, x1 + num_elem_coords, 
        enzyme_const, num_elem_coords,
        enzyme_const, x2,
        enzyme_const, x2 + num_elem_coords,
        enzyme_const, num_elem_coords,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx1 + (max_overlap_vert * dim) * i,
        enzyme_dup, xi + max_overlap_vert, dxidx1 + (max_overlap_vert * dim) * i + max_overlap_vert,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true );
      // wiggle the eta coordinate of node i in element 1
      __enzyme_fwddiff<void>( (void*)Intersection2DPolygon,
        enzyme_const, x1, 
        enzyme_dup, x1 + num_elem_coords, x_dot,
        enzyme_const, num_elem_coords,
        enzyme_const, x2,
        enzyme_const, x2 + num_elem_coords,
        enzyme_const, num_elem_coords,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx1 + (max_overlap_vert * dim) * (num_elem_coords + i),
        enzyme_dup, xi + max_overlap_vert, dxidx1 + (max_overlap_vert * dim) * (num_elem_coords + i) + max_overlap_vert,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true );
      // wiggle the xi coordinate of node i in element 2
      __enzyme_fwddiff<void>( (void*)Intersection2DPolygon,
        enzyme_const, x1,
        enzyme_const, x1 + num_elem_coords,
        enzyme_const, num_elem_coords,
        enzyme_dup, x2, x_dot, 
        enzyme_const, x2 + num_elem_coords, 
        enzyme_const, num_elem_coords,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx2 + (max_overlap_vert * dim) * i,
        enzyme_dup, xi + max_overlap_vert, dxidx2 + (max_overlap_vert * dim) * i + max_overlap_vert,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true );
      // wiggle the eta coordinate of node i in element 2
      __enzyme_fwddiff<void>( (void*)Intersection2DPolygon,
        enzyme_const, x1,
        enzyme_const, x1 + num_elem_coords,
        enzyme_const, num_elem_coords,
        enzyme_const, x2, 
        enzyme_dup, x2 + num_elem_coords, x_dot,
        enzyme_const, num_elem_coords,
        enzyme_const, pos_tol,
        enzyme_const, len_tol,
        enzyme_dup, xi, dxidx2 + (max_overlap_vert * dim) * (num_elem_coords + i),
        enzyme_dup, xi + max_overlap_vert, dxidx2 + (max_overlap_vert * dim) * (num_elem_coords + i) + max_overlap_vert,
        enzyme_const, &num_poly_verts,
        enzyme_const, &area,
        enzyme_const, true );
      // clang-format on
      x_dot[i] = 0.0;
    }

    std::cout << "dxi/dx1 nonzero values:" << std::endl;
    for ( int j{ 0 }; j < 8; ++j ) {
      for ( int i{ 0 }; i < 16; ++i ) {
        auto idx = j * 16 + i;
        if ( std::abs( dxidx1[idx] ) > 1.0e-15 ) {
          std::cout << "  (" << i << ", " << j << ") = " << dxidx1[idx] << std::endl;
        }
      }
    }

    std::cout << "dxi/dx2 nonzero values:" << std::endl;
    for ( int j{ 0 }; j < 8; ++j ) {
      for ( int i{ 0 }; i < 16; ++i ) {
        auto idx = j * 16 + i;
        if ( std::abs( dxidx2[idx] ) > 1.0e-15 ) {
          std::cout << "  (" << i << ", " << j << ") = " << dxidx2[idx] << std::endl;
        }
      }
    }

    RealT dxidx1_fd[16 * 8];
    RealT dxidx2_fd[16 * 8];
    // row 1 assumes x2 is inside x1; row 2 assumes x1 is inside x2
    RealT x_sgn1[8] = { 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0 };
    RealT y_sgn1[8] = { 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0 };
    RealT x_sgn2[8] = { -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0 };
    RealT y_sgn2[8] = { -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0 };
    for ( int j{ 0 }; j < 4; ++j ) {
      x1[j] += x_sgn1[4 * stencil_dir[j] + j] * delta_;
      for ( int i{ 0 }; i < 16; ++i ) {
        xi[i] = 0.0;
      }
      Intersection2DPolygon( x1, x1 + 4, 4, x2, x2 + 4, 4, pos_tol, len_tol, xi, xi + 8, num_poly_verts, area, true );
      for ( int i{ 0 }; i < 16; ++i ) {
        dxidx1_fd[16 * j + i] = x_sgn1[4 * stencil_dir[j] + j] * ( xi[i] - xi_base[i] ) / delta_;
      }
      x1[j] -= x_sgn1[4 * stencil_dir[j] + j] * delta_;

      x1[j + 4] += y_sgn1[4 * stencil_dir[j] + j] * delta_;
      for ( int i{ 0 }; i < 16; ++i ) {
        xi[i] = 0.0;
      }
      Intersection2DPolygon( x1, x1 + 4, 4, x2, x2 + 4, 4, pos_tol, len_tol, xi, xi + 8, num_poly_verts, area, true );
      for ( int i{ 0 }; i < 16; ++i ) {
        dxidx1_fd[16 * ( 4 + j ) + i] = y_sgn1[4 * stencil_dir[j] + j] * ( xi[i] - xi_base[i] ) / delta_;
      }
      x1[j + 4] -= y_sgn1[4 * stencil_dir[j] + j] * delta_;

      x2[j] += x_sgn2[4 * stencil_dir[j] + j] * delta_;
      for ( int i{ 0 }; i < 16; ++i ) {
        xi[i] = 0.0;
      }
      Intersection2DPolygon( x1, x1 + 4, 4, x2, x2 + 4, 4, pos_tol, len_tol, xi, xi + 8, num_poly_verts, area, true );
      for ( int i{ 0 }; i < 16; ++i ) {
        dxidx2_fd[16 * j + i] = x_sgn2[4 * stencil_dir[j] + j] * ( xi[i] - xi_base[i] ) / delta_;
      }
      x2[j] -= x_sgn2[4 * stencil_dir[j] + j] * delta_;

      x2[j + 4] += y_sgn2[4 * stencil_dir[j] + j] * delta_;
      for ( int i{ 0 }; i < 16; ++i ) {
        xi[i] = 0.0;
      }
      Intersection2DPolygon( x1, x1 + 4, 4, x2, x2 + 4, 4, pos_tol, len_tol, xi, xi + 8, num_poly_verts, area, true );
      for ( int i{ 0 }; i < 16; ++i ) {
        dxidx2_fd[16 * ( 4 + j ) + i] = y_sgn2[4 * stencil_dir[j] + j] * ( xi[i] - xi_base[i] ) / delta_;
      }
      x2[j + 4] -= y_sgn2[4 * stencil_dir[j] + j] * delta_;
    }

    // write deltas to screen
    std::cout << "dxi/dx1 ------------------------------" << std::endl;
    for ( int j{ 0 }; j < 8; ++j ) {
      for ( int i{ 0 }; i < 16; ++i ) {
        auto idx = j * 16 + i;
        auto diff = std::abs( dxidx1[idx] - dxidx1_fd[idx] );
        if ( diff > 10.0 * delta_ ) {
          std::cout << "  (" << i << ", " << j << ") : Diff: " << diff << "   Ratio: " << dxidx1[idx] / dxidx1_fd[idx]
                    << "   Enzyme: " << dxidx1[idx] << "   FD: " << dxidx1_fd[idx] << std::endl;
        }
        EXPECT_NEAR( dxidx1[idx], dxidx1_fd[idx], fd_tol );
      }
    }
    std::cout << "dxi/dx2 ------------------------------" << std::endl;
    for ( int j{ 0 }; j < 8; ++j ) {
      for ( int i{ 0 }; i < 16; ++i ) {
        auto idx = j * 16 + i;
        auto diff = std::abs( dxidx2[idx] - dxidx2_fd[idx] );
        if ( diff > 10.0 * delta_ ) {
          std::cout << "  (" << i << ", " << j << ") : Diff: " << diff << "   Ratio: " << dxidx2[idx] / dxidx2_fd[idx]
                    << "   Enzyme: " << dxidx2[idx] << "   FD: " << dxidx2_fd[idx] << std::endl;
        }
        EXPECT_NEAR( dxidx2[idx], dxidx2_fd[idx], fd_tol );
      }
    }
  }
};

TEST_F( EnzymePolyIntersectTest, PerfectOverlap )
{
  constexpr auto pos_tol = 10.0 * delta_;
  constexpr auto len_tol = 10.0 * delta_;
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
  int stencil_dir[4] = { 0, 0, 0, 0 };
  CheckIntersection( x1, x2, stencil_dir, delta_, pos_tol, len_tol, "perfect_overlap" );
}

TEST_F( EnzymePolyIntersectTest, Mesh2VertexMovedInByPosTol )
{
  constexpr auto pos_tol = 10.0 * delta_;
  constexpr auto len_tol = 10.0 * delta_;
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.0 + pos_tol, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
  int stencil_dir[4] = { 1, 1, 1, 1 };
  CheckIntersection( x1, x2, stencil_dir, delta_, pos_tol, len_tol, "onevertpostol" );
}

// NOTE: edges are too nearly parallel which makes the derivative explode for some terms.  this makes the FD inaccurate.
TEST_F( EnzymePolyIntersectTest, NearlyParallelEdges )
{
  constexpr auto pos_tol = 10.0 * delta_;
  constexpr auto len_tol = 10.0 * delta_;
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.0, 1.0, 1.0, 0.0, 0.0 - 2 * pos_tol, 0.0 + 2 * pos_tol, 1.0, 1.0 };
  int stencil_dir[4] = { 0, 1, 0, 0 };
  CheckIntersection( x1, x2, stencil_dir, 31000.0, pos_tol, len_tol, "nearlyparallel" );
}

TEST_F( EnzymePolyIntersectTest, LessNearlyParallelEdges )
{
  constexpr auto pos_tol = 10.0 * delta_;
  constexpr auto len_tol = 10.0 * delta_;
  constexpr auto offset = 0.2;
  // shift node 3 in a little to prevent edge class change with FD
  RealT x1[8] = { 0.0, 1.0, 1.0 - pos_tol, 0.0, 0.0, 0.0, 1.0 - pos_tol, 1.0 };
  RealT x2[8] = { 0.0, 1.0, 1.0, 0.0, 0.0 - offset, 0.0 + offset, 1.0, 1.0 };
  int stencil_dir[4] = { 0, 1, 1, 0 };
  CheckIntersection( x1, x2, stencil_dir, 4.0 * delta_, pos_tol, len_tol, "kindaparallel" );
}

TEST_F( EnzymePolyIntersectTest, OffsetElements )
{
  constexpr auto pos_tol = 10.0 * delta_;
  constexpr auto len_tol = 10.0 * delta_;
  constexpr auto offset = 0.2;
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.0 + offset, 1.0 + offset, 1.0 + offset, 0.0 + offset,
                  0.0 + offset, 0.0 + offset, 1.0 + offset, 1.0 + offset };
  int stencil_dir[4] = { 1, 0, 0, 0 };
  CheckIntersection( x1, x2, stencil_dir, 2.0 * delta_, pos_tol, len_tol, "offset" );
}

TEST_F( EnzymePolyIntersectTest, EightOverlapVertices )
{
  constexpr auto pos_tol = 10.0 * delta_;
  constexpr auto len_tol = 10.0 * delta_;
  auto xmin = -1.0 / std::sqrt( 2.0 ) + 0.5;
  auto xmax = 1.0 / std::sqrt( 2.0 ) + 0.5;
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.5, xmax, 0.5, xmin, xmin, 0.5, xmax, 0.5 };
  int stencil_dir[4] = { 0, 0, 0, 0 };
  CheckIntersection( x1, x2, stencil_dir, 2.0 * delta_, pos_tol, len_tol, "twist" );
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
