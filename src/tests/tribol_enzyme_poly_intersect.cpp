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

#include "tribol/geom/GeomUtilities.hpp"

namespace tribol {

/**
 * @brief Test fixture for the Enzyme-based derivatives of intersection polygon calculations.
 *
 * Specifically, this tests the derivative of the overlap polygon w.r.t. the nodal coordinates by comparing the
 * Enzyme-based derivative to the derivative computed using finite differences.
 */
class EnzymePolyIntersectTest : public testing::Test {
 protected:
  static constexpr double delta_{ 1.0e-8 };
  void SetUp() override {}

  void CheckIntersectionJacobian( RealT* x1, RealT* x2, int* stencil_dir, RealT fd_tol = delta_, RealT pos_tol = 1.0e-8,
                                  RealT len_tol = 1.0e-8 )
  {
    constexpr int max_overlap_vert = 8;
    constexpr int dim = 2;
    RealT xi[max_overlap_vert * dim];
    // track the origin of the overlap vertices
    // A = vertex in element 1
    // B = vertex in element 2
    // EdgeEdge = intersection of edges
    OverlapVertexType type[max_overlap_vert];
    // this stores which edge or vertex on element 1 the overlap vertex is associated with
    int edge1[max_overlap_vert];
    // this stores which edge or vertex on element 2 the overlap vertex is associated with
    int edge2[max_overlap_vert];
    for ( int i{ 0 }; i < max_overlap_vert; ++i ) {
      xi[i] = 0.0;
      xi[i + max_overlap_vert] = 0.0;
      // initialize these to something so they are always initialized
      type[i] = OverlapVertexType::A;
      edge1[i] = -1;
      edge2[i] = -1;
    }
    auto num_poly_verts = 0;
    auto d_num_poly_verts = 0;
    auto d_pos_tol = 0.0;
    auto d_len_tol = 0.0;
    RealT area = 0.0;
    constexpr int num_elem_coords = 4;
    constexpr bool check_orientation = true;

    // compute the overlap polygon
    Intersection2DPolygon( x1, x1 + num_elem_coords, num_elem_coords, x2, x2 + num_elem_coords, num_elem_coords,
                           pos_tol, len_tol, xi, xi + max_overlap_vert, num_poly_verts, area, check_orientation, type,
                           edge1, edge2 );
    // print some info about the overlap
    std::cout << std::setprecision( 15 ) << "Element 1 coords" << std::endl;
    for ( int i{ 0 }; i < num_elem_coords; ++i ) {
      std::cout << "(" << x1[i] << ", " << x1[i + 4] << ")\n";
    }
    std::cout << std::setprecision( 15 ) << "Element 2 coords" << std::endl;
    for ( int i{ 0 }; i < num_elem_coords; ++i ) {
      std::cout << "(" << x2[i] << ", " << x2[i + 4] << ")\n";
    }
    std::cout << std::setprecision( 15 ) << "Number of overlap vertices: " << num_poly_verts
              << "   Polygon area: " << area << std::endl;

    for ( int i{ 0 }; i < num_poly_verts; ++i ) {
      std::cout << "  Coord: (" << xi[i] << ", " << xi[i + 8] << ")   Type: ";
      switch ( type[i] ) {
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

    // compute the derivative of the overlap coords w.r.t. the nodal coordinates using Enzyme (exact)
    RealT x_dot[num_elem_coords] = { 0.0, 0.0, 0.0, 0.0 };
    RealT zeros[num_elem_coords] = { 0.0, 0.0, 0.0, 0.0 };
    constexpr auto rows = max_overlap_vert * dim;
    constexpr auto cols = num_elem_coords * dim;
    RealT dxidx1[rows * cols];
    RealT dxidx2[rows * cols];
    for ( int i{ 0 }; i < rows * cols; ++i ) {
      dxidx1[i] = 0.0;
      dxidx2[i] = 0.0;
    }
    // save the original overlap coordinates (needed for finite differencing below)
    RealT xi_base[max_overlap_vert * dim];
    for ( int i{ 0 }; i < 16; ++i ) {
      xi_base[i] = xi[i];
    }
    // loop over each nodal coordinate to compute each column of the Jacobian matrix
    for ( int i{ 0 }; i < num_elem_coords; ++i ) {
      x_dot[i] = 1.0;
      // clang-format off
      // wiggle the xi coordinate of node i in element 1
      __enzyme_fwddiff<void>( (void*)Intersection2DPolygonEnzyme,
        x1, x_dot,
        x1 + num_elem_coords, zeros,
        num_elem_coords,
        x2, zeros,
        x2 + num_elem_coords, zeros,
        num_elem_coords,
        pos_tol, d_pos_tol,
        len_tol, d_len_tol,
        xi, dxidx1 + rows * i,
        xi + max_overlap_vert, dxidx1 + rows * i + max_overlap_vert,
        &num_poly_verts, &d_num_poly_verts );
      // wiggle the eta coordinate of node i in element 1
      __enzyme_fwddiff<void>( (void*)Intersection2DPolygonEnzyme,
        x1, zeros,
        x1 + num_elem_coords, x_dot,
        num_elem_coords,
        x2, zeros,
        x2 + num_elem_coords, zeros,
        num_elem_coords,
        pos_tol, d_pos_tol,
        len_tol, d_len_tol,
        xi, dxidx1 + rows * ( num_elem_coords + i ),
        xi + max_overlap_vert, dxidx1 + rows * ( num_elem_coords + i ) + max_overlap_vert,
        &num_poly_verts, &d_num_poly_verts );
      // wiggle the xi coordinate of node i in element 2
      __enzyme_fwddiff<void>( (void*)Intersection2DPolygonEnzyme,
        x1, zeros,
        x1 + num_elem_coords, zeros,
        num_elem_coords,
        x2, x_dot,
        x2 + num_elem_coords, zeros,
        num_elem_coords,
        pos_tol, d_pos_tol,
        len_tol, d_len_tol,
        xi, dxidx2 + rows * i,
        xi + max_overlap_vert, dxidx2 + rows * i + max_overlap_vert,
        &num_poly_verts, &d_num_poly_verts );
      // wiggle the eta coordinate of node i in element 2
      __enzyme_fwddiff<void>( (void*)Intersection2DPolygonEnzyme,
        x1, zeros,
        x1 + num_elem_coords, zeros,
        num_elem_coords,
        x2, zeros,
        x2 + num_elem_coords, x_dot,
        num_elem_coords,
        pos_tol, d_pos_tol,
        len_tol, d_len_tol,
        xi, dxidx2 + rows * ( num_elem_coords + i ),
        xi + max_overlap_vert, dxidx2 + rows * ( num_elem_coords + i ) + max_overlap_vert,
        &num_poly_verts, &d_num_poly_verts );
      // clang-format on
      x_dot[i] = 0.0;
    }
    // print out non-zero derivatives for debugging
    std::cout << "dxi/dx1 nonzero values:" << std::endl;
    for ( int j{ 0 }; j < cols; ++j ) {
      for ( int i{ 0 }; i < rows; ++i ) {
        auto idx = j * rows + i;
        if ( std::abs( dxidx1[idx] ) > 1.0e-15 ) {
          std::cout << "  (" << i << ", " << j << ") = " << dxidx1[idx] << std::endl;
        }
      }
    }
    std::cout << "dxi/dx2 nonzero values:" << std::endl;
    for ( int j{ 0 }; j < cols; ++j ) {
      for ( int i{ 0 }; i < rows; ++i ) {
        auto idx = j * rows + i;
        if ( std::abs( dxidx2[idx] ) > 1.0e-15 ) {
          std::cout << "  (" << i << ", " << j << ") = " << dxidx2[idx] << std::endl;
        }
      }
    }

    // compute the derivative of the overlap coords w.r.t. the nodal coordinates using finite differencing (approx)
    RealT dxidx1_fd[rows * cols];
    RealT dxidx2_fd[rows * cols];
    // define a stencil direction so the overlap doesn't change when we wiggle the nodes
    // NOTE: row 1 assumes x2 is inside x1; row 2 assumes x1 is inside x2
    // clang-format off
    RealT x_sgn1[num_elem_coords * 2] = { 1.0, -1.0, -1.0, 1.0,
                                          -1.0, 1.0, 1.0, -1.0 };
    RealT y_sgn1[num_elem_coords * 2] = { 1.0, 1.0, -1.0, -1.0,
                                          -1.0, -1.0, 1.0, 1.0 };
    RealT x_sgn2[num_elem_coords * 2] = { -1.0, 1.0, 1.0, -1.0,
                                          1.0, -1.0, -1.0, 1.0 };
    RealT y_sgn2[num_elem_coords * 2] = { -1.0, -1.0, 1.0, 1.0,
                                          1.0, 1.0, -1.0, -1.0 };
    // clang-format on
    // loop over each nodal coordinate to compute each column of the Jacobian matrix
    for ( int j{ 0 }; j < num_elem_coords; ++j ) {
      // wiggle the xi coordinate of node i in element 1
      x1[j] += x_sgn1[num_elem_coords * stencil_dir[j] + j] * delta_;
      for ( int i{ 0 }; i < max_overlap_vert * dim; ++i ) {
        xi[i] = 0.0;
      }
      Intersection2DPolygon( x1, x1 + num_elem_coords, num_elem_coords, x2, x2 + num_elem_coords, num_elem_coords,
                             pos_tol, len_tol, xi, xi + max_overlap_vert, num_poly_verts, area, check_orientation, type,
                             edge1, edge2 );
      for ( int i{ 0 }; i < rows; ++i ) {
        dxidx1_fd[rows * j + i] = x_sgn1[num_elem_coords * stencil_dir[j] + j] * ( xi[i] - xi_base[i] ) / delta_;
      }
      x1[j] -= x_sgn1[num_elem_coords * stencil_dir[j] + j] * delta_;
      // wiggle the eta coordinate of node i in element 1
      x1[j + num_elem_coords] += y_sgn1[num_elem_coords * stencil_dir[j] + j] * delta_;
      for ( int i{ 0 }; i < max_overlap_vert * dim; ++i ) {
        xi[i] = 0.0;
      }
      Intersection2DPolygon( x1, x1 + num_elem_coords, num_elem_coords, x2, x2 + num_elem_coords, num_elem_coords,
                             pos_tol, len_tol, xi, xi + max_overlap_vert, num_poly_verts, area, check_orientation, type,
                             edge1, edge2 );
      for ( int i{ 0 }; i < rows; ++i ) {
        dxidx1_fd[rows * ( num_elem_coords + j ) + i] =
            y_sgn1[num_elem_coords * stencil_dir[j] + j] * ( xi[i] - xi_base[i] ) / delta_;
      }
      x1[j + num_elem_coords] -= y_sgn1[num_elem_coords * stencil_dir[j] + j] * delta_;
      // wiggle the xi coordinate of node i in element 2
      x2[j] += x_sgn2[num_elem_coords * stencil_dir[j] + j] * delta_;
      for ( int i{ 0 }; i < max_overlap_vert * dim; ++i ) {
        xi[i] = 0.0;
      }
      Intersection2DPolygon( x1, x1 + num_elem_coords, num_elem_coords, x2, x2 + num_elem_coords, num_elem_coords,
                             pos_tol, len_tol, xi, xi + max_overlap_vert, num_poly_verts, area, check_orientation, type,
                             edge1, edge2 );
      for ( int i{ 0 }; i < rows; ++i ) {
        dxidx2_fd[rows * j + i] = x_sgn2[num_elem_coords * stencil_dir[j] + j] * ( xi[i] - xi_base[i] ) / delta_;
      }
      x2[j] -= x_sgn2[num_elem_coords * stencil_dir[j] + j] * delta_;
      // wiggle the eta coordinate of node i in element 2
      x2[j + num_elem_coords] += y_sgn2[num_elem_coords * stencil_dir[j] + j] * delta_;
      for ( int i{ 0 }; i < max_overlap_vert * dim; ++i ) {
        xi[i] = 0.0;
      }
      Intersection2DPolygon( x1, x1 + num_elem_coords, num_elem_coords, x2, x2 + num_elem_coords, num_elem_coords,
                             pos_tol, len_tol, xi, xi + max_overlap_vert, num_poly_verts, area, check_orientation, type,
                             edge1, edge2 );
      for ( int i{ 0 }; i < rows; ++i ) {
        dxidx2_fd[rows * ( num_elem_coords + j ) + i] =
            y_sgn2[num_elem_coords * stencil_dir[j] + j] * ( xi[i] - xi_base[i] ) / delta_;
      }
      x2[j + num_elem_coords] -= y_sgn2[num_elem_coords * stencil_dir[j] + j] * delta_;
    }

    // print out derivatives where Enzyme and FD differ for debugging
    std::cout << "dxi/dx1 ------------------------------" << std::endl;
    for ( int j{ 0 }; j < cols; ++j ) {
      for ( int i{ 0 }; i < rows; ++i ) {
        auto idx = j * rows + i;
        auto diff = std::abs( dxidx1[idx] - dxidx1_fd[idx] );
        if ( diff > fd_tol ) {
          std::cout << "  (" << i << ", " << j << ") : Diff: " << diff << "   Ratio: " << dxidx1[idx] / dxidx1_fd[idx]
                    << "   Enzyme: " << dxidx1[idx] << "   FD: " << dxidx1_fd[idx] << std::endl;
        }
        EXPECT_NEAR( dxidx1[idx], dxidx1_fd[idx], fd_tol );
      }
    }
    std::cout << "dxi/dx2 ------------------------------" << std::endl;
    for ( int j{ 0 }; j < cols; ++j ) {
      for ( int i{ 0 }; i < rows; ++i ) {
        auto idx = j * rows + i;
        auto diff = std::abs( dxidx2[idx] - dxidx2_fd[idx] );
        if ( diff > fd_tol ) {
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
  CheckIntersectionJacobian( x1, x2, stencil_dir, delta_, pos_tol, len_tol );
}

TEST_F( EnzymePolyIntersectTest, Mesh2VertexMovedInByPosTol )
{
  constexpr auto pos_tol = 10.0 * delta_;
  constexpr auto len_tol = 10.0 * delta_;
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.0 + pos_tol, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
  int stencil_dir[4] = { 1, 1, 1, 1 };
  CheckIntersectionJacobian( x1, x2, stencil_dir, delta_, pos_tol, len_tol );
}

// NOTE: edges are too nearly parallel which makes the derivative explode for some terms.  this makes the FD inaccurate.
// specifically, a small change in the perpendicular component of a coordinate on the edge leads to a large (linearized)
// change in the overlap coordinate.
TEST_F( EnzymePolyIntersectTest, NearlyParallelEdges )
{
  constexpr auto pos_tol = 10.0 * delta_;
  constexpr auto len_tol = 10.0 * delta_;
  RealT x1[8] = { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 };
  RealT x2[8] = { 0.0, 1.0, 1.0, 0.0, 0.0 - 2 * pos_tol, 0.0 + 2 * pos_tol, 1.0, 1.0 };
  int stencil_dir[4] = { 0, 1, 0, 0 };
  CheckIntersectionJacobian( x1, x2, stencil_dir, 31000.0, pos_tol, len_tol );
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
  CheckIntersectionJacobian( x1, x2, stencil_dir, 4.0 * delta_, pos_tol, len_tol );
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
  CheckIntersectionJacobian( x1, x2, stencil_dir, 2.0 * delta_, pos_tol, len_tol );
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
  CheckIntersectionJacobian( x1, x2, stencil_dir, 2.0 * delta_, pos_tol, len_tol );
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
