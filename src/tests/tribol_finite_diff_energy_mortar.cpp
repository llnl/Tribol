// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <cmath>
#include <set>
#include "/usr/workspace/lutz23/Smith/repo/tribol/src/tribol/physics/new_method.hpp"
// #include <tribol/src/tribol/physics/new_method.hpp>
// #include "tribol/physics/new_method.hpp"
#include <gtest/gtest.h>

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

#include "mfem.hpp"

#include "axom/CLI11.hpp"
#include "axom/slic.hpp"

#include "shared/mesh/MeshBuilder.hpp"
#include "redecomp/redecomp.hpp"

#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"


namespace tribol{

TEST(GradientCheck, GtildeFDvsAD)
{
    // ── Geometry: two facing LINEAR_EDGE segments ────────────────────────────
    // Segment A: (0,0) -> (1,0)
    // Segment B: (0.2, 0.5) -> (0.8, 0.5)

RealT x1[2] = { 0.0, 1.0 };
RealT y1[2] = { 0.0, 0.0 };
IndexT conn1[2] = { 1, 0 };   // reversed so nA points toward B
MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );

RealT x2[2] = { 0.2, 0.8 };
RealT y2[2] = { 0.5, 0.5 };
IndexT conn2[2] = { 0, 1 };
MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

    InterfacePair pair( 0, 0 );

    // ── Evaluator setup ──────────────────────────────────────────────────────
    ContactParams params;
    params.del              = 0.1;   // smoothing m
    params.k                = 1.0;   // penalty stiffness
    params.N                = 3;     // quadrature points
    params.enzyme_quadrature = true; // use the non-Enzyme quadrature path

    ContactEvaluator evaluator( params );

    // ── Run validation ───────────────────────────────────────────────────────
    const double epsilon = 1e-7;

    // In the test, before calling validate_g_tilde:
RealT y1_plus[2] = { epsilon, 0.0 };
RealT y1_minus[2] = { -epsilon, 0.0 };
mesh1.setPosition( x1, y1_plus, nullptr );
auto [gp1, gp2] = evaluator.eval_gtilde( pair, mesh1.getView(), mesh2.getView() );
mesh1.setPosition( x1, y1_minus, nullptr );
auto [gm1, gm2] = evaluator.eval_gtilde( pair, mesh1.getView(), mesh2.getView() );
mesh1.setPosition( x1, y1, nullptr );
std::cout << "Manual FD DOF1: " << (gp1-gm1)/(2*epsilon) << ", " << (gp2-gm2)/(2*epsilon) << "\n";
    auto result = evaluator.validate_g_tilde( pair, mesh1, mesh2, epsilon );

    // ── Compare ──────────────────────────────────────────────────────────────
    const double tol = 1e-6;
    // const int num_dofs = static_cast<int>( result.node_ids.size() ) * 2;
    const std::size_t num_dofs = result.node_ids.size() * 2;

    ASSERT_EQ( result.fd_gradient_g1.size(), result.analytical_gradient_g1.size() );
    ASSERT_EQ( result.fd_gradient_g2.size(), result.analytical_gradient_g2.size() );

    for ( size_t i = 0; i < num_dofs; ++i )
    {
        EXPECT_NEAR( result.fd_gradient_g1[i], result.analytical_gradient_g1[i], tol )
            << "gtilde1 mismatch at DOF [" << i << "]"
            << "  node=" << result.node_ids[i / 2]
            << "  dir=" << ( i % 2 == 0 ? "x" : "y" )
            << "  FD="  << result.fd_gradient_g1[i]
            << "  AD="  << result.analytical_gradient_g1[i];

        EXPECT_NEAR( result.fd_gradient_g2[i], result.analytical_gradient_g2[i], tol )
            << "gtilde2 mismatch at DOF [" << i << "]"
            << "  node=" << result.node_ids[i / 2]
            << "  dir=" << ( i % 2 == 0 ? "x" : "y" )
            << "  FD="  << result.fd_gradient_g2[i]
            << "  AD="  << result.analytical_gradient_g2[i];
    }
}




TEST(HessianCheck, GtildeFDvsAD)
{
    // ── Geometry: two facing LINEAR_EDGE segments ────────────────────────────
    // Segment A: (0,0) -> (1,0)
    // Segment B: (0.2, 0.5) -> (0.8, 0.5)

    RealT x1[2] = { 0.0, 1.0 };
    RealT y1[2] = { 0.0, 0.0 };

    RealT x2[2] = { 0.2, 0.8 };
    RealT y2[2] = { 0.5, 0.5 };

    IndexT conn1[2] = { 0, 1 };
    IndexT conn2[2] = { 0, 1 };

    MeshData mesh1( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );
    MeshData mesh2( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );

    InterfacePair pair( 0, 0 );

    // ── Evaluator setup ──────────────────────────────────────────────────────
    ContactParams params;
    params.del               = 0.1;
    params.k                 = 1.0;
    params.N                 = 4;
    params.enzyme_quadrature = true;

    ContactEvaluator evaluator( params );

    // ── Run validation ───────────────────────────────────────────────────────
    const double epsilon = 1e-5;  // slightly larger epsilon works better for second derivatives
    auto result = evaluator.validate_hessian( pair, mesh1, mesh2, epsilon );

    // ── Compare ──────────────────────────────────────────────────────────────
    const double tol = 1e-4;  // looser tolerance for second derivatives
    const int ndof = 8;

    ASSERT_EQ( result.fd_gradient_g1.size(), 64u );
    ASSERT_EQ( result.fd_gradient_g2.size(), 64u );
    ASSERT_EQ( result.analytical_gradient_g1.size(), 64u );
    ASSERT_EQ( result.analytical_gradient_g2.size(), 64u );

    for ( size_t row = 0; row < ndof; ++row )
    {
        for ( size_t col = 0; col < ndof; ++col )
        {
            size_t idx = row * ndof + col;

            EXPECT_NEAR( result.fd_gradient_g1[idx], result.analytical_gradient_g1[idx], tol )
                << "Hessian g1 mismatch at [row=" << row << ", col=" << col << "]"
                << "  FD=" << result.fd_gradient_g1[idx]
                << "  AD=" << result.analytical_gradient_g1[idx];

            EXPECT_NEAR( result.fd_gradient_g2[idx], result.analytical_gradient_g2[idx], tol )
                << "Hessian g2 mismatch at [row=" << row << ", col=" << col << "]"
                << "  FD=" << result.fd_gradient_g2[idx]
                << "  AD=" << result.analytical_gradient_g2[idx];
        }
    }
}

}