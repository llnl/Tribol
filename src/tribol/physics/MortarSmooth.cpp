// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "MortarSmooth.hpp"

#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/geom/ContactPlane.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/integ/Integration.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/ContactPlaneOutput.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/utils/Algorithm.hpp"

// Axom includes
#include "axom/slic.hpp"

#include <iostream>
#include <iomanip>
#include <ostream>
#include <src/serac/physics/contact/contact_config.hpp>

#ifdef TRIBOL_USE_ENZYME

#include "tribol/common/Enzyme.hpp"
#endif

namespace tribol {

//------------------------------------------------------------------------------
template <>
int ApplyNormal<SMOOTH_MORTAR, LAGRANGE_MULTIPLIER>( CouplingScheme* cs )
{
#ifdef TRIBOL_USE_ENZYME
  printf("enzyme enabled\n");
  //if ( cs->isEnzymeEnabled() ) {
  return ApplySmoothNormalEnzyme( cs );
  //}
#endif

  return 0;

}  // end ApplyNormal<>()

#ifdef TRIBOL_USE_ENZYME

//------------------------------------------------------------------------------
int ApplySmoothNormalEnzyme( CouplingScheme* cs )
{
  printf("smoothed enzyme\n");
//   exit(1);
  auto planes_view = cs->get2DContactPlanes().view();
  auto& lm_opts = cs->getEnforcementOptions().lm_implicit_options;
  bool compute_jacobian = false;
  if ( lm_opts.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ||
       lm_opts.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN ) {
    if ( lm_opts.sparse_mode == SparseMode::MFEM_ELEMENT_DENSE ) {
      cs->getMethodData()->reserveBlockJ(
          { BlockSpace::NONMORTAR, BlockSpace::MORTAR}, planes_view.size() );
      compute_jacobian = true;
    } else {
      SLIC_WARNING( "Unsupported Jacobian storage method." );
      return 1;
    }
  }
  // convention: 1 = nonmortar
  //             2 = mortar
  auto mesh1 = cs->getMesh2().getView();  // switched from tribol convention
  auto mesh2 = cs->getMesh1().getView();  // switched from tribol convention
  int size1 = mesh1.numberOfNodesPerElement();
  int size2 = mesh2.numberOfNodesPerElement();
  constexpr int dim = 2;

  RealT k1 = cs->getMesh1().getElementData().m_penalty_stiffness;
  RealT k2 = cs->getMesh2().getElementData().m_penalty_stiffness;
//   std::cout << "k1: " << k1 << " k2: " << k2 << std::endl;
  constexpr RealT del = 0.01;



  

  for ( auto& plane : planes_view ) {
    int elem1 = plane.getCpElementId2();  // switched from tribol convention
    // NOTE: mfem::DenseMatrix data is stored by nodes instead of by vdim
    RealT coords[8];
    RealT force[8];
    for ( int i{ 0 }; i < size1; ++i ) {
      int node_id = mesh1.getGlobalNodeId( elem1, i );
      for ( int d{ 0 }; d < dim; ++d ) {
        coords[i * dim + d] = mesh1.getPosition()[d][node_id];
        force[i * dim + d] = 0.0;
      }
    }
    int elem2 = plane.getCpElementId1();  // switched from tribol convention
    for ( int i{ 0 }; i < size2; ++i ) {
      int node_id = mesh2.getGlobalNodeId( elem2, i );
      for ( int d{ 0 }; d < dim; ++d ) {
        coords[(i + size1) * dim + d] = mesh2.getPosition()[d][node_id];
        force[(i + size1) * dim + d] = 0.0;
      }
    }
    if ( lm_opts.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ||
         lm_opts.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN ) {

      constexpr int n_disp = 4;
  
      StackArray<DeviceArray2D<RealT>, 9> blockJ( 2 );
      for ( int i{}; i < 2; ++i ) {
        for ( int j{}; j < 2; ++j ) {
          blockJ( i, j ) = DeviceArray2D<RealT>( n_disp, n_disp );
          blockJ( i, j ).fill( 0.0 );
        }
      }

      constexpr int num_quad_points = 3;

      RealT jacobian[64] = {0.0};

    RealT A0[2] = {coords[0], coords[1]};
    RealT A1[2] = {coords[2], coords[3]};
    RealT B0[2] = {coords[4], coords[5]};
    RealT B1[2] = {coords[6], coords[7]};

        RealT projections[2];
        get_projections(A0, A1, B0, B1, projections, del);



      RealT lenA = sqrt((coords[2] - coords[0]) * (coords[2] - coords[0]) + (coords[3] - coords[1]) * (coords[3] - coords[1]));
      RealT lenB = sqrt((coords[6] - coords[4]) * (coords[6] - coords[4]) + (coords[7] - coords[5]) * (coords[7] - coords[5]));
    
      ComputeSmoothMortarJacobianEnzyme(coords, del, k1, k2, num_quad_points, lenA, projections, force, jacobian );

      
      int vdim_to_nodes[4] = { 0, 2, 1, 3 };
      for ( int br = 0; br < 2; ++br ) {
        for ( int bc = 0; bc < 2; ++bc ) {
          for ( int lr = 0; lr < 4; ++lr ) {
            for ( int lc = 0; lc < 4; ++lc ) {
              int jacobian_row = ( br * 4 + vdim_to_nodes[lr] );
              int jacobian_col = ( bc * 4 + vdim_to_nodes[lc] );
              blockJ( br, bc )( lr, lc ) = jacobian[jacobian_row * 8 + jacobian_col];
            }
          }
        }
      }

//       for (int br = 0; br < 2; ++br) {
//     for (int bc = 0; bc < 2; ++bc) {
//         std::cout << "blockJ(" << br << ", " << bc << "):" << std::endl;
//         for (int lr = 0; lr < 4; ++lr) {
//             for (int lc = 0; lc < 4; ++lc) {
//                 std::cout << blockJ(br, bc)(lr, lc) << " ";
//             }
//             std::cout << std::endl;
//         }
//         std::cout << std::endl;
//     }
// }

    

      if ( lm_opts.sparse_mode == SparseMode::MFEM_ELEMENT_DENSE ) {
        cs->getMethodData()->storeElemBlockJ( { elem1, elem2 }, blockJ );
      } else {
        SLIC_WARNING( "Unsupported Jacobian storage method." );
        return 1;
      }
    } else if ( lm_opts.eval_mode == ImplicitEvalMode::MORTAR_GAP ||
                lm_opts.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL ) {
                RealT A0[2] = {coords[0], coords[1]};
                RealT A1[2] = {coords[2], coords[3]};
                RealT B0[2] = {coords[4], coords[5]};
                RealT B1[2] = {coords[6], coords[7]};

        RealT projections[2];
        get_projections(A0, A1, B0, B1, projections, del);

      constexpr int num_quad_points = 3;
      RealT lenA = sqrt((coords[2] - coords[0]) * (coords[2] - coords[0]) + (coords[3] - coords[1]) * (coords[3] - coords[1]));
      RealT lenB = sqrt((coords[6] - coords[4]) * (coords[6] - coords[4]) + (coords[7] - coords[5]) * (coords[7] - coords[5]));
      ComputeSmoothMortarForceEnzyme( coords, del, k1, k2, num_quad_points, lenA, projections, force);
    }
    for ( int i = 0; i < size1; ++i ) {
      int node_id = mesh1.getGlobalNodeId( elem1, i );
      for ( int d = 0; d < dim; ++d ) {
        mesh1.getResponse()[d][node_id] += force[i * dim + d];     //f1[d * size1 + i];
      }
    }
    for ( int i{ 0 }; i < size2; ++i ) {
      int node_id = mesh2.getGlobalNodeId( elem2, i );
      for ( int d = 0; d < dim; ++d ) {
        mesh2.getResponse()[d][node_id] += force[dim * (size1 + i) + d];  //f2[d * size2 + i];
      }
    }
  }
  
return 0;
}

//------------------------------------------------------------------------------
void find_normal(const RealT* coord1, const RealT* coord2, RealT* normal) {
    RealT dx = coord2[0] - coord1[0];
    RealT dy = coord2[1] - coord1[1];
    RealT len = std::sqrt(dy * dy + dx * dx);
    dx /= len;
    dy /= len;
    normal[0] = dy;
    normal[1] = -dx;
}

 void determine_legendre_nodes(int N, double* N_i) {
    if (N==1) {
       N_i[0] = 0.0; 
    }
    else if(N==2) {
        N_i[0] = -1 / std::sqrt(3);
        N_i[1] = 1 / std::sqrt(3);
    }
    else if(N==3) {
        N_i[0] = -std::sqrt(3.0/5.0);
        N_i[1] = 0.0;
        N_i[2] = std::sqrt(3.0/5.0);
    }
    else {
        N_i[0] = -1.0 * std::sqrt((15 + 2 * std::sqrt(30)) / 35);
        N_i[1] = -1.0 * std::sqrt((15 - 2 * std::sqrt(30)) / 35);
        N_i[2] = -std::sqrt((15 - 2 * std::sqrt(30)) / 35);
        N_i[3] = -std::sqrt((15 + 2 * std::sqrt(30)) / 35);
    }
 }

 void determine_legendre_weights(int N, double* W) {
    if (N == 1) {
        W[0] = 2.0;
    }
    else if(N == 2) {
        W[0] = 1.0;
        W[1] = 1.0;
    }
    else if (N == 3) {
        W[0] = 5.0 / 9.0;
        W[1] = 8.0 / 9.0;
        W[2] = 5.0 / 9.0;
    }
    else {
        W[0] = (18 - std::sqrt(30)) / 36.0;
        W[1] = (18 + std::sqrt(30)) / 36.0;
        W[2] = (18 + std::sqrt(30)) / 36.0;
        W[3] = (18 - std::sqrt(30)) / 36.0;
    }
 }


void iso_map(const RealT* coord1, const RealT* coord2, RealT xi, RealT* mapped_coord){
    double N1 = 0.5 - xi;
    double N2 = 0.5 + xi;
    mapped_coord[0] = N1 * coord1[0] + N2 * coord2[0];
    mapped_coord[1] =  N1 * coord1[1] + N2 * coord2[1];
}

bool segmentsIntersect(const RealT A0[2], const RealT A1[2],
                       const RealT B0[2], const RealT B1[2],
                       RealT intersection[2]) {
    auto cross = [](RealT x0, RealT y0, RealT x1, RealT y1) {
        return x0 * y1 - y0 * x1;
    };

    RealT dxA = A1[0] - A0[0], dyA = A1[1] - A0[1];
    RealT dxB = B1[0] - B0[0], dyB = B1[1] - B0[1];
    RealT dxAB = B0[0] - A0[0], dyAB = B0[1] - A0[1];

    RealT denom = cross(dxA, dyA, dxB, dyB);
    RealT numeA = cross(dxAB, dyAB, dxB, dyB);
    RealT numeB = cross(dxAB, dyAB, dxA, dyA);

    // Collinear or parallel
    if (std::abs(denom) < 1e-12) {
        if (std::abs(numeA) > 1e-12 || std::abs(numeB) > 1e-12)
            return false; // Parallel, not collinear

        // Collinear: check for overlap
        auto between = [](RealT a, RealT b, RealT c) {
            return std::min(a, b) <= c && c <= std::max(a, b);
        };

        // Check if endpoints overlap
            if (between(A0[0], A1[0], B0[0]) && between(A0[1], A1[1], B0[1])) {
                intersection[0] = B0[0];
                intersection[1] = B0[1];
                return true;
            }
            if (between(A0[0], A1[0], B1[0]) && between(A0[1], A1[1], B1[1])) {
                intersection[0] = B1[0];
                intersection[1] = B1[1];
                return true;
            }
            if (between(B0[0], B1[0], A0[0]) && between(B0[1], B1[1], A0[1])) {
                intersection[0] = A0[0];
                intersection[1] = A0[1];
                return true;
            }
            if (between(B0[0], B1[0], A1[0]) && between(B0[1], B1[1], A1[1])) {
                intersection[0] = A1[0];
                intersection[1] = A1[1];
                return true;
            }
        // Overlap but not at a single point
        return false;
    }


    RealT ua = numeA / denom;
    RealT ub = numeB / denom;

    if (ua >= 0.0 && ua <= 1.0 && ub >= 0.0 && ub <= 1.0) {
        intersection[0] = A0[0] + ua * dxA;
        intersection[1] = A0[1] + ua * dyA;
        return true;
    }
    return false;
}


void find_intersection(const RealT* A0, const RealT* A1, const RealT* p, const RealT* nB, RealT* intersection) {
    RealT tA[2] = {A1[0] - A0[0], A1[1] - A0[1] };
    RealT d[2] = {p[0] - A0[0], p[1] - A0[1]};

    RealT det = tA[0] * nB[1] - tA[1] * nB[0];

    if(std::abs(det) < 1e-12) {
        intersection[0] = p[0];
        intersection[1] = p[1];
        return;
    }

    RealT inv_det = 1.0 / det;

    RealT alpha = (d[0] * nB[1] - d[1] * nB[0]) * inv_det;

    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;

    intersection[0] = (A0[0] + alpha * tA[0]);
    intersection[1] = A0[1]  + alpha * tA[1];

}




void get_projections(const RealT* A0, const RealT* A1, const RealT* B0, const RealT* B1, RealT* projections, RealT del) {
    RealT nB[2] = {0.0}; 
    find_normal(B0, B1, nB);
    
    RealT end_points[2] = {-0.5, 0.5}; 
    for (int i = 0; i < 2; ++i) {
        RealT p[2] = {0.0};
        iso_map(B0, B1, end_points[i], p);
        
        RealT intersection[2] = {0.0};
        find_intersection(A0, A1, p, nB, intersection);

        // Convert intersection to parametric coordinate on A
        RealT dx = A1[0] - A0[0];
        RealT dy = A1[1] - A0[1];
        RealT len2 = dx*dx + dy*dy;
        RealT xiA = ((intersection[0] - A0[0]) * dx + (intersection[1] - A0[1]) * dy) / len2;
        
        // Apply constraints and convert to reference interval
        xiA = std::max(del, std::min(1.0 - del, xiA)) - 0.5;
        projections[i] = xiA;
    }
}


// void get_projections(const RealT* A0, const RealT* A1, const RealT* B0, const RealT* B1, RealT* projections, RealT del) {
//     RealT nA[2] = {0.0};
//     RealT nB[2] = {0.0}; 
//     find_normal(A0, A1, nA);
//     find_normal(B0, B1, nB);
//     RealT end_points[2] = {-0.5, 0.5}; 
//     for (int i = 0; i < 2; ++i) {
//         RealT p[2] = {0.0};

//         RealT intersection[2] = {0.0};
//         RealT seg_intersection[2] = {0.0};
//         iso_map(B0, B1, end_points[i], p);
//         find_intersection(A0, A1, p, nB, intersection);

//         RealT dx = A1[0] - A0[0];
//         RealT dy = A1[1] - A0[1];
//         RealT len2 = dx*dx + dy*dy;
//         RealT xiA = ((intersection[0] - A0[0]) * dx + (intersection[1] - A0[1]) * dy) / len2;
//         RealT nB_unit[2] = { nB[0], nB[1] };
//         RealT norm = std::sqrt(nB_unit[0]*nB_unit[0] + nB_unit[1]*nB_unit[1]);
//         nB_unit[0] /= norm;
//         nB_unit[1] /= norm;

//         RealT dx_gap = intersection[0] - p[0];
//         RealT dy_gap = intersection[1] - p[1];
//         RealT gap = dx_gap * nB_unit[0] + dy_gap * nB_unit[1];

//         if(segmentsIntersect(A0, A1, B0, B1, seg_intersection) &&  gap > 0.0) {

//                 xiA = ((seg_intersection[0] - A0[0]) * dx + (seg_intersection[1] - A0[1]) * dy) / len2;
//                 if (xiA < del) { 
//                   xiA = del;
//                 }

//         }
//         xiA = xiA - 0.5;
//         projections[i] = xiA;
//     }
// }


void compute_integration_bounds(const RealT* projections, RealT* integration_bounds, RealT del) {
    RealT xi_min = projections[0];
    RealT xi_max = projections[0];
    for (int i = 0; i < 2; ++i) {
        if (xi_min > projections[i]) {
            xi_min = projections[i];
        }
        if(xi_max < projections[i]) {
            xi_max = projections[i]; 
        }

    }

    if (xi_max < -0.5 - del) {
        xi_max = -0.5 - del;
    }
    if(xi_min > 0.5 + del) {
        xi_min  = 0.5 + del;
    }
    if (xi_min < -0.5 - del) { 
        xi_min = -0.5 -del;
    }
    if (xi_max > 0.5 + del) {
        xi_max = 0.5 + del;
    }

    integration_bounds[0] = xi_min;
    integration_bounds[1] = xi_max;
    // std::cout << "xi min: " << xi_min << " xi_max: " << xi_max << std::endl;
}

void modify_bounds(const RealT* integration_bounds, RealT del, RealT* modified_bounds) {
    RealT xi = 0.0;

    RealT int_bound[2] = {0.0};
    for(int i = 0; i < 2; ++i) {
        int_bound[i] = integration_bounds[i];
    }
    for (int i = 0; i < 2; ++i) {
        RealT xi_hat = 0.0;
        xi = int_bound[i] + 0.5;
        if (0.0 - del <= xi && xi <= del) {
            xi_hat = (1.0/(4*del)) * (xi*xi) + 0.5 * xi + del/4.0;
        }
        else if((1.0 - del) <= xi && xi <= 1.0 + del) {
        RealT b = -1.0/(4.0*del);
        RealT c = 0.5 + 1.0/(2.0*del);
        RealT d = 1.0 - del + (1.0/(4.0*del)) * pow(1.0-del, 2) - 0.5*(1.0-del) - (1.0-del)/(2.0*del);

        xi_hat = b*xi*xi + c*xi + d;
        }
        else if(del <= xi && xi <= (1.0 - del)) { 
            xi_hat = xi;
        }
        else{ 
            std::cerr << "Xi did not fall in an expected range for modifying bounds" << std::endl;
        }
        modified_bounds[i] = xi_hat - 0.5;
    }
}

void modify_bounds_for_weight(const RealT* integration_bounds, RealT del, RealT* modified_bounds) {
    RealT xi = 0.0;
    for (int i = 0; i < 2; ++i) {
        RealT xi_hat = 0.0;
        xi = integration_bounds[i] + 0.5;

        if (xi < std::abs(1e-10)) {
            xi = 0.0;
        }
        if (0.0 <= xi && xi <= del) {
            xi_hat = (xi * xi) / (2.0 * del * (1.0 - del));
        }
        else if((1.0 - del) <= xi && xi <= 1.0) {
            xi_hat =  1.0 -(((1.0- xi) * (1.0 - xi)) / (2 * del * (1.0 - del)));
        }
        else if(del <= xi && xi <= (1.0 - del)) { 
            xi_hat = ((2.0 * xi) - del) / (2.0 * (1.0 - del));
        }
        else{ 
            std::cerr << "Xi did not fall in an expected range for modifying bounds" << std::endl;
        }
        modified_bounds[i] = xi_hat - 0.5;
    }
}


void compute_quadrature_point(const RealT* integration_bounds, const RealT* A0, const RealT* A1, int N, RealT* quad_points) {
    RealT eta_values[N];
    determine_legendre_nodes(N, eta_values);

    // for (int i = 0; i < N; ++i) {
    //     eta_values[i] *= 0.5;
    // }

    RealT xi_min = integration_bounds[0];
    RealT xi_max = integration_bounds[1];

    for ( int i = 0; i < N; ++i) {
        RealT xi_i = 0.5 * (xi_max + xi_min) + eta_values[i] * (xi_max - xi_min); 
        RealT mapped_coords[2] = {0.0, 0.0};
        iso_map(A0, A1, xi_i, mapped_coords);
        quad_points[2 * i] = mapped_coords[0];
        quad_points[2 * i + 1] = mapped_coords[1];   
    }     
}

void assign_weights(const RealT* integration_bounds, int N, RealT* weights) {
    RealT ref_weights[N];
    determine_legendre_weights(N, ref_weights);
    RealT J = 0.0;
    RealT xi_min = integration_bounds[0];
    RealT xi_max = integration_bounds[1];
    
    J = 0.5 * (xi_max - xi_min);

    for( int i = 0; i < N; ++i) {
        weights[i] = ref_weights[i] * J;
    }
}

RealT compute_gap(const RealT* p, const RealT* B0, const RealT* B1, const RealT* nA, const RealT* A0, const RealT* A1) {
    RealT nA_orig[2] = {nA[0], nA[1]};

    RealT len = std::sqrt(nA[0] * nA[0] + nA[1] * nA[1]);
    nA_orig[0] /= len;
    nA_orig[1] /= len;
    RealT intersection[2] = {0.0};
    find_intersection(A0, A1, p, nA_orig, intersection);


    RealT dx = intersection[0] - p[0];
    RealT dy = intersection[1] - p[1];

    RealT gap = dx * nA_orig[0] + dy * nA_orig[1];
    gap *= -1;
    // std::cout << "GAP: " << gap << std::endl;
    return gap;
}


// RealT compute_gap(const RealT* p, const RealT* B0, const RealT* B1, const RealT* nB, const RealT* A0, const RealT* A1) {
//     RealT nB_orig[2] = {nB[0], nB[1]};

//     RealT len = std::sqrt(nB[0] * nB[0] + nB[1] * nB[1]);
//     nB_orig[0] /= len;
//     nB_orig[1] /= len;
//     RealT intersection[2] = {0.0};
//     find_intersection(A0, A1, p, nB_orig, intersection);


//     RealT dx = intersection[0] - p[0];
//     RealT dy = intersection[1] - p[1];

//     RealT gap = dx * nB_orig[0] + dy * nB_orig[1];
//     return gap;
// }

RealT compute_modified_gap(RealT gap, RealT* nA, RealT* nB) {
    RealT dot = nA[0] * nB[0] + nA[1] * nB[1];
    RealT eta = (dot < 0) ? -dot:0.0;

    // std::cout << "GAP: " << gap * eta << std::endl;
    return gap * eta;
}

RealT compute_contact_potential(RealT gap, RealT k1, RealT k2) {
    if (gap > -1e-10) {
        return 0;
    }
    RealT gap1 = -gap;
    RealT pot = k1 * (gap1 * gap1) - k2 * (gap1 * gap1 * gap1);
    return pot;
}




void ComputeSmoothMortarEnergyEnzyme(const RealT* coords, RealT del, RealT k1, RealT k2, int N, RealT lenA, RealT* projections, RealT* energy) {
    RealT A0[2] = {coords[0], coords[1]};
    RealT A1[2] = {coords[2], coords[3]};
    RealT B0[2] = {coords[4], coords[5]};
    RealT B1[2] = {coords[6], coords[7]};

    // for (int i{0}; i < 4; ++i) {
    //         std::cout << " Coord " << i << ": ( " << coords[i*2] << ", " << coords[i*2+1] << ")" << std::endl;
    // }


    // std::cout << "len A: " << lenA << " len B: " << std::endl;

    // std::cout << "Node A0: " << A0[0] << ", " << A0[1] << std::endl;
    // std::cout << "Node A1: " << A1[0] << ", " << A1[1] << std::endl;    
    // std::cout << "Node B0: " << B0[0] << ", " << B0[1] << std::endl;
    // std::cout << "Node B1: " << B1[0] << ", " << B1[1] << std::endl;


// std::cout << "TRICK TURNED OFF" << std::endl;

//   RealT AC[2] = { 0.5 * ( A0[0] + A1[0] ), 0.5 * ( A0[1] + A1[1] ) };
//   RealT AR[2] = { 0.5 * ( A0[0] - A1[0] ), 0.5 * ( A0[1] - A1[1] ) };
//   RealT normAR = 0.5 / std::sqrt( AR[0] * AR[0] + AR[1] * AR[1] );

//   RealT BC[2] = { 0.5 * ( B0[0] + B1[0] ), 0.5 * ( B0[1] + B1[1] ) };
//   RealT BR[2] = { 0.5 * ( B1[0] - B0[0] ), 0.5 * ( B1[1] - B0[1] ) };
//   RealT normBR = 0.5 / std::sqrt( BR[0] * BR[0] + BR[1] * BR[1] );

//   A0[0] = AC[0] + AR[0] * lenA * normAR;
//   A0[1] = AC[1] + AR[1] * lenA * normAR;

//   A1[0] = AC[0] - AR[0] * lenA * normAR;
//   A1[1] = AC[1] - AR[1] * lenA * normAR;
//   B0[0] = BC[0] - BR[0] * lenB * normBR;
//   B0[1] = BC[1] - BR[1] * lenB * normBR;

//   B1[0] = BC[0] + BR[0] * lenB * normBR;
//   B1[1] = BC[1] + BR[1] * lenB * normBR;




    // RealT AC[2] = {0.5 * (A0[0]+A1[0]), 0.5*(A0[1]+A1[1])};
    // RealT AR[2] = {0.5 * (A0[0]-A1[0]), 0.5*(A0[1]-A1[1])};

    // RealT BC[2] = {0.5 * (B0[0]+B1[0]), 0.5*(B0[1]+B1[1])};
    // RealT BR[2] = {0.5 * (B0[0]-B1[0]), 0.5*(B0[1]-B1[1])};

    // RealT normAR = 0.5 / std::sqrt( AR[0] * AR[0] + AR[1] * AR[1] );
    // RealT normBR = 0.5 / std::sqrt( BR[0] * BR[0] + BR[1] * BR[1] );

    // A0[0] = AC[0] + ((AR[0] * lenA * 0.5) / normAR);
    // A0[1] = AC[1] + ((AR[1] * lenA * 0.5) / normAR);

    // A1[0] = AC[0] - ((AR[0] * lenA * 0.5) / normAR);
    // A1[1] = AC[1] - ((AR[1] * lenA * 0.5) / normAR);

    // B0[0] = BC[0] + ((BR[0] * lenB * 0.5) / normBR);
    // B0[1] = BC[1] + ((BR[1] * lenB * 0.5) / normBR);

    // B1[0] = BC[0] - ((BR[0] * lenB * 0.5) / normBR);
    // B1[1] = BC[1] - ((BR[1] * lenB * 0.5) / normBR);
    

    // double x0 = ((BR[0] * lenB * 0.5) / normBR);
    // B0[0] = BC[0] + x0;
    // B0[0] = new_B;



    // B0[0] = BC[0] + x0;



    // std::cout << "executed line 607" << std::endl;
    RealT nA[2] = {0.0};
    RealT nB[2] = {0.0};
    find_normal(A0, A1, nA);
    find_normal(B0, B1, nB);
    // std::cout << "executed line 612" << std::endl;

    RealT dot_product = nA[0] * nB[0] + nA[1] * nB[1];

    if (std::abs(dot_product) < 1e-10) {
        *energy = 0;
    }

    else{

    // RealT projections[2];
    // get_projections(A0, A1, B0, B1, projections, del);

    RealT integration_bounds[2];
    compute_integration_bounds(projections, integration_bounds, del);

    RealT modified_bounds[2];
    modify_bounds(integration_bounds, del, modified_bounds);

    RealT modified_bounds_w[2];
    modify_bounds_for_weight(integration_bounds, del, modified_bounds_w); 

    RealT quad_points[2 * N];
    compute_quadrature_point(modified_bounds, B0, B1, N, quad_points);

    RealT weights[N];
    assign_weights(modified_bounds_w, N, weights);
    // std::cout << "Integration Bounds: " << modified_bounds[0] << ", " << modified_bounds[1] << std::endl;

    *energy = 0.0;
    for(int i = 0; i < N; ++i) {
        RealT p[2] = {quad_points[2 * i], quad_points[2 * i + 1]};
        RealT gap = compute_gap(p, B0, B1, nA, A0, A1);
        RealT smooth_gap = compute_modified_gap(gap, nA, nB);
        // std::cout << "GAP: " << gap << " at point: " << B0[0] << ", " << B0[1] << " and " << B1[0] << " , " << B1[1] << " and qp is: " << p[0] << " , " << p[1] << std::endl;
        RealT potential = compute_contact_potential(smooth_gap, k1, k2);
        // std::cout << "ENERGY: " << potential * weights[i] << std::endl;
        *energy +=  weights[i] * potential;
    }
    // lenA = sqrt((coords[2] - coords[0]) * (coords[2] - coords[0]) + (coords[3] - coords[1]) * (coords[3] - coords[1]));
    *energy *= lenA * 0.5;
    // std::cout << "ENERGY: " << *energy << std::endl;

}}


// void ComputeSmoothMortarEnergyEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* g1, int size1,
//                                      const RealT* x2, RealT* f2, int size2 )

// {

// }


//--------------------------------------------------------------------------------









//--------------------------------------------------------------------------------





void ComputeSmoothMortarForceEnzyme(RealT* coords, RealT del, RealT k1, RealT k2, int N, RealT lenA, RealT* projections, RealT* dE_dX) 
{
    double dcoords[8] = {0.0};
    double E = 0.0;
    double dE = 1.0;
    __enzyme_autodiff<void>( ComputeSmoothMortarEnergyEnzyme, enzyme_dup, coords, dcoords, enzyme_const, del, enzyme_const, k1, enzyme_const, k2, enzyme_const, N, enzyme_const, lenA, enzyme_const, projections, enzyme_dup, &E, &dE);
    // std::cout << "Computed forces" << std::endl;

    for(int i = 0; i < 8; ++i) {
        dE_dX[i] = dcoords[i];
    }

    for(int i = 0; i < 8; ++i) {
        // std::cout << "Force[" << i << "]: " << dE_dX[i] << std::endl;
    }
}


// void ComputeSmoothMortarForceEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* g1, int size1,
//                                      const RealT* x2, RealT* f2, int size2 )
// {






//------------------------------------------------------------------------------

void ComputeSmoothMortarJacobianEnzyme(RealT* coords, RealT del, RealT k1, RealT k2, int N, RealT lenA, RealT* projections, RealT* force, RealT* d2E_d2X) {
    RealT dE[8] = {0.0};
    RealT d2E[8] = {0.0};
    // RealT dEf[8] = {0.0};
    // ComputeSmoothMortarForceEnzyme(coords, del, k1, k2, N, lenA, lenB, dEf);
    // for(int i = 0; i < 8; ++i) {
    //   force[i] = dEf[i];
    // }
    for(int i = 0; i < 8; ++i) {
        RealT d2coords[8] = {0.0};
        d2coords[i] = 1.0;
        RealT d2k1 = 0.0;
        RealT d2del = 0.0;
        RealT d2k2 = 0.0;
        RealT d2lenA = 0.0;
        RealT dprojections[2] = {0.0, 0.0};
        __enzyme_fwddiff<void>( ComputeSmoothMortarForceEnzyme, coords, d2coords, del, d2del, k1, d2k1, k2, d2k2, N, lenA, d2lenA, projections, dprojections, &dE, &d2E);
        for(int j = 0; j < 8; ++j) {
            d2E_d2X[8 * i + j] = d2E[j];
            // std::cout << "position: [" << i << ',' << j << "]: " << d2E_d2X[8 * i + j] << std::endl;
        }
        // std::cout << "Computed row " << i << " of the Jacobian" << std::endl;

    }
    for(int i = 0; i < 8; ++i) {
      force[i] = dE[i];
    }
//     const int NN = 8;
// int k = 0; // The DOF (column) you want

// double result[NN] = {0.0};
// for (int j = 0; j < NN; ++j) {
//     result[j] = d2E_d2X[NN * j + k];
    // This grabs the k-th column (since your matrix is row-major)
    // If you want the k-th row, swap indices
// }

// Print result to compare with J_exact
// for (int j = 0; j < NN; ++j) {
//     printf("J exact: %.17g\n", result[j]);
// }


}

// void ComputeSmoothMortarJacobianEnzyme( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* df1dx1,
//                                         RealT* df1dx2, RealT* df1dn1, RealT* df1dp1, RealT* g1, RealT* dg1dx1, RealT* dg1dx2,
//                                         RealT* dg1dn1, int size1, const RealT* x2, RealT* f2, RealT* df2dx1, RealT* df2dx2,
//                                         RealT* df2dn1, RealT* df2dp1, int size2 )
// {
// }
#endif

//------------------------------------------------------------------------------

}  // end namespace tribol
