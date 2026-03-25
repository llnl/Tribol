// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <cmath>
#include <set>

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
#include "tribol/interface/

FiniteDiffResult ContactEvaluator::validate_g_tilde( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
                                                     double epsilon ) const
{
  FiniteDiffResult result;

  auto viewer1 = mesh1.getView();
  auto viewer2 = mesh2.getView();

  auto projs0 = projections( pair, viewer1, viewer2 );
  auto bounds0 = smoother_.bounds_from_projections( projs0 );
  auto smooth_bounds0 = smoother_.smooth_bounds( bounds0 );
  QuadPoints qp0 = compute_quadrature( smooth_bounds0 );

  // auto [g1_base, g2_base] = eval_gtilde_fixed_qp(mesh, A, B, qp0);

  auto [g1_base, g2_base] = eval_gtilde( pair, viewer1, viewer2 );
  result.g_tilde1_baseline = g1_base;
  result.g_tilde2_baseline = g2_base;

  // Collect nodes in sorted order
  std::set<int> node_set;
  auto A_conn = viewer1.getConnectivity()( pair.m_element_id1 );
  node_set.insert( A_conn[0] );
  node_set.insert( A_conn[1] );
  auto B_conn = viewer2.getConnectivity()( pair.m_element_id2 );
  node_set.insert( B_conn[0] );
  node_set.insert( B_conn[1] );

  result.node_ids = std::vector<int>( node_set.begin(), node_set.end() );
  // std::sort(result.node_ids.begin(), result.node_ids.end()); //Redundant??

  int num_dofs = result.node_ids.size() * 2;
  result.fd_gradient_g1.resize( num_dofs );
  result.fd_gradient_g2.resize( num_dofs );

  // ===== GET AND REORDER ENZYME GRADIENTS =====
  double dgt1_dx[8] = { 0.0 };
  double dgt2_dx[8] = { 0.0 };
  grad_trib_area( pair, viewer1, viewer2, dgt1_dx, dgt2_dx );

  // Map from node_id to position in x[8]
  std::map<int, int> node_to_x_idx;
  node_to_x_idx[A_conn[0]] = 0;  // A0 → x[0,1]
  node_to_x_idx[A_conn[1]] = 1;  // A1 → x[2,3]
  node_to_x_idx[B_conn[0]] = 2;  // B0 → x[4,5]
  node_to_x_idx[B_conn[1]] = 3;  // B1 → x[6,7]

  // Reorder Enzyme gradients to match sorted node order
  result.analytical_gradient_g1.resize( num_dofs );
  result.analytical_gradient_g2.resize( num_dofs );

  for ( size_t i = 0; i < result.node_ids.size(); ++i ) {
    int node_id = result.node_ids[i];
    int x_idx = node_to_x_idx[node_id];

    result.analytical_gradient_g1[2 * i + 0] = dgt1_dx[2 * x_idx + 0];  // x component
    result.analytical_gradient_g1[2 * i + 1] = dgt1_dx[2 * x_idx + 1];  // y component
    result.analytical_gradient_g2[2 * i + 0] = dgt2_dx[2 * x_idx + 0];
    result.analytical_gradient_g2[2 * i + 1] = dgt2_dx[2 * x_idx + 1];
  }
  // =

  int dof_idx = 0;
  // X-direction

  std::set<IndexT> mesh1_nodes = { A_conn[0], A_conn[1] };
  std::set<IndexT> mesh2_nodes = { B_conn[0], B_conn[1] };

  for ( int node_id : result.node_ids ) {
    {
      bool is_in_mesh1 = ( mesh1_nodes.count( node_id ) > 0 );
      MeshData& mesh_to_perturb = is_in_mesh1 ? mesh1 : mesh2;

      // Store Original Mesh coords:
      auto pos = mesh_to_perturb.getView().getPosition();
      int num_nodes = mesh_to_perturb.numberOfNodes();
      int dim = mesh_to_perturb.spatialDimension();

      std::vector<RealT> x_original( num_nodes );
      std::vector<RealT> y_original( num_nodes );
      std::vector<RealT> z_original( num_nodes );

      for ( int i = 0; i < num_nodes; ++i ) {
        x_original[i] = pos[0][i];
        y_original[i] = pos[1][i];
        if ( dim == 3 ) z_original[i] = pos[2][i];
      }

      std::vector<RealT> x_pert = x_original;
      x_pert[node_id] += epsilon;
      mesh_to_perturb.setPosition( x_pert.data(), y_original.data(), dim == 3 ? z_original.data() : nullptr );

      // Evalaute with x_plus
      auto viewer1_plus = mesh1.getView();
      auto viewer2_plus = mesh2.getView();

      auto [g1_plus, g2_plus] = eval_gtilde_fixed_qp( pair, viewer1_plus, viewer2_plus, qp0 );

      x_pert[node_id] = x_original[node_id] - epsilon;

      mesh_to_perturb.setPosition( x_pert.data(), y_original.data(), dim == 3 ? z_original.data() : nullptr );

      auto viewer1_minus = mesh1.getView();
      auto viewer2_minus = mesh2.getView();

      auto [g1_minus, g2_minus] = eval_gtilde_fixed_qp( pair, viewer1_minus, viewer2_minus, qp0 );

      // Restore orginal
      mesh_to_perturb.setPosition( x_original.data(), y_original.data(), dim == 3 ? z_original.data() : nullptr );

      // Compute gradient
      result.fd_gradient_g1[dof_idx] = ( g1_plus - g1_minus ) / ( 2.0 * epsilon );
      result.fd_gradient_g2[dof_idx] = ( g2_plus - g2_minus ) / ( 2.0 * epsilon );

      dof_idx++;
    }
    {
      bool is_in_mesh1 = ( mesh1_nodes.count( node_id ) > 0 );
      MeshData& mesh_to_perturb = is_in_mesh1 ? mesh1 : mesh2;

      // Store Original Mesh coords:
      auto pos = mesh_to_perturb.getView().getPosition();
      int num_nodes = mesh_to_perturb.numberOfNodes();
      int dim = mesh_to_perturb.spatialDimension();

      std::vector<RealT> x_original( num_nodes );
      std::vector<RealT> y_original( num_nodes );
      std::vector<RealT> z_original( num_nodes );

      for ( int i = 0; i < num_nodes; ++i ) {
        x_original[i] = pos[0][i];
        y_original[i] = pos[1][i];
        if ( dim == 3 ) z_original[i] = pos[2][i];
      }
      std::vector<RealT> y_pert = y_original;

      y_pert[node_id] += epsilon;

      mesh_to_perturb.setPosition( x_original.data(), y_pert.data(), dim == 3 ? z_original.data() : nullptr );

      auto viewer1_plus2 = mesh1.getView();
      auto viewer2_plus2 = mesh2.getView();

      auto [g1_plus, g2_plus] = eval_gtilde_fixed_qp( pair, viewer1_plus2, viewer2_plus2, qp0 );

      y_pert[node_id] = y_original[node_id] - epsilon;

      mesh_to_perturb.setPosition( x_original.data(), y_pert.data(), dim == 3 ? z_original.data() : nullptr );

      auto viewer1_minus2 = mesh1.getView();
      auto viewer2_minus2 = mesh2.getView();
      auto [g1_minus, g2_minus] = eval_gtilde_fixed_qp( pair, viewer1_minus2, viewer2_minus2, qp0 );

      mesh_to_perturb.setPosition( x_original.data(), y_original.data(), dim == 3 ? z_original.data() : nullptr );

      result.fd_gradient_g1[dof_idx] = ( g1_plus - g1_minus ) / ( 2.0 * epsilon );
      result.fd_gradient_g2[dof_idx] = ( g2_plus - g2_minus ) / ( 2.0 * epsilon );

      dof_idx++;
    }

    //         double original = mesh.node(node_id).x;

    //         double x_plus =

    //         mesh.node(node_id).x = original + epsilon;
    //         auto [g1_plus, g2_plus] = eval_gtilde_fixed_qp(mesh, A, B, qp0);

    //         mesh.node(node_id).x = original - epsilon;
    //         auto [g1_minus, g2_minus] = eval_gtilde_fixed_qp(mesh, A, B, qp0);

    //         //Restorre orginal
    //         mesh.node(node_id).x = original;

    //         result.fd_gradient_g1[dof_idx] = (g1_plus - g1_minus) / (2.0 * epsilon);
    //         result.fd_gradient_g2[dof_idx] = (g2_plus - g2_minus) / (2.0 * epsilon);

    //         dof_idx++;
    //     }

    // //y - direction
    //     {
    //         double original = mesh.node(node_id).y;

    //         // +epsilon
    //         mesh.node(node_id).y = original + epsilon;
    //         auto [g1_plus, g2_plus] = eval_gtilde_fixed_qp(mesh, A, B, qp0);

    //         // -epsilon
    //         mesh.node(node_id).y = original - epsilon;
    //         auto [g1_minus, g2_minus] = eval_gtilde_fixed_qp(mesh, A, B, qp0);

    //         // Restore
    //         mesh.node(node_id).y = original;

    //         // Central difference
    //         result.fd_gradient_g1[dof_idx] = (g1_plus - g1_minus) / (2.0 * epsilon);
    //         result.fd_gradient_g2[dof_idx] = (g2_plus - g2_minus) / (2.0 * epsilon);

    //         dof_idx++;
    //     }
  }
  return result;
}


// void ContactEvaluator::grad_gtilde_with_qp( const InterfacePair& pair, const MeshData::Viewer& mesh1,
//                                             const MeshData::Viewer& mesh2, const QuadPoints& qp_fixed,
//                                             double dgt1_dx[8], double dgt2_dx[8] ) const
// {
//   double A0[2], A1[2], B0[2], B1[2];
//   endpoints( mesh1, pair.m_element_id1, A0, A1 );
//   endpoints( mesh2, pair.m_element_id2, B0, B1 );

//   double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

//   const int N = static_cast<int>( qp_fixed.qp.size() );

//   Gparams gp;
//   gp.N = N;
//   gp.qp = qp_fixed.qp.data();  // Use FIXED quadrature
//   gp.w = qp_fixed.w.data();

//   grad_A1( x, &gp, dgt1_dx );
//   grad_A2( x, &gp, dgt2_dx );
// }



std::pair<double, double> ContactEvaluator::eval_gtilde_fixed_qp( const InterfacePair& pair,
                                                                  const MeshData::Viewer& mesh1,
                                                                  const MeshData::Viewer& /*mesh2*/,
                                                                  const QuadPoints& qp_fixed ) const
{
  double A0[2], A1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) + ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  double gt1 = 0.0, gt2 = 0.0;

  for ( size_t i = 0; i < qp_fixed.qp.size(); ++i ) {
    const double xiA = qp_fixed.qp[i];
    const double w = qp_fixed.w[i];

    const double N1 = 0.5 - xiA;
    const double N2 = 0.5 + xiA;

    // const double gn = gap(pair, mesh1, mesh2, xiA);   // still depends on geometry
    // const double gn_active = gn;              // or your (gn<0?gn:0) logic

    gt1 += w * N1 * J;
    gt2 += w * N2 * J;
  }

  return { gt1, gt2 };
}




// FiniteDiffResult ContactEvaluator::validate_hessian(Mesh& mesh, const Element& A, const Element& B, double epsilon)
// const {
//     FiniteDiffResult result;

//     auto projs0 = projections(mesh, A, B);
//     auto bounds0 = smoother_.bounds_from_projections(projs0);
//     auto smooth_bounds0 = smoother_.smooth_bounds(bounds0);
//     QuadPoints qp0 = compute_quadrature(smooth_bounds0);
//     double hess1[64] = {0.0};
//     double hess2[64] = {0.0};
//     compute_d2A_d2u(mesh, A, B, hess1, hess2);

//     const int ndof = 8;
//     result.fd_gradient_g1.assign(ndof*ndof, 0.0);
//     result.fd_gradient_g2.assign(ndof*ndof, 0.0);
//     result.analytical_gradient_g1.resize(ndof * ndof);
//     result.analytical_gradient_g2.resize(ndof * ndof);

//     result.analytical_gradient_g1.assign(hess1, hess1 + 64);
//     result.analytical_gradient_g2.assign(hess2, hess2 + 64);

// int nodes[4] = { A.node_ids[0], A.node_ids[1], B.node_ids[0], B.node_ids[1] };

// int col = 0;
// for (int k = 0; k < 4; ++k) {
//   for (int comp = 0; comp < 2; ++comp) { // 0=x, 1=y
//     Node& n = mesh.node(nodes[k]);
//     double& coord = (comp == 0) ? n.x : n.y;
//     double orig = coord;

//     double g1p[8]={0}, g1m[8]={0}, g2p[8]={0}, g2m[8]={0};

//     coord = orig + epsilon; grad_gtilde_with_qp(mesh, A, B, qp0, g1p, g2p);
//     coord = orig - epsilon; grad_gtilde_with_qp(mesh, A, B, qp0, g1m, g2m);
//     coord = orig;

//     for (int i = 0; i < 8; ++i) {
//       result.fd_gradient_g1[i*8 + col] = (g1p[i] - g1m[i]) / (2*epsilon);
//       result.fd_gradient_g2[i*8 + col] = (g2p[i] - g2m[i]) / (2*epsilon);
//     }
//     ++col;
//   }
// }
// return result;
// }