#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include "tribol/common/Parameters.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/common/Enzyme.hpp"


template <typename return_type, typename... Args>
return_type __enzyme_autodiff( Args... );


void find_normal(const double* coord1, const double* coord2, double* normal) {
    double dx = coord2[0] - coord1[0];
    double dy = coord2[1] - coord1[1];
    double len = std::sqrt(dy * dy + dx * dx);
    dx /= len;
    dy /= len;
    normal[0] = dy;
    normal[1] = -dx;
}

void determine_lobatto_nodes(int N, double* N_i) {
    if (N == 1) {
        N_i[0] = 0.0;
    }
    else if (N == 2) {
        N_i[0] = -1.0;
        N_i[1] = 1.0;
    }
    else if (N == 3) {
        N_i[0] = -1.0;
        N_i[1] = 0.0;
        N_i[2] = 1.0;
    }
    else if(N == 4) {
        N_i[0] = -1.0;
        N_i[1] = -1.0 / std::sqrt(5.0);
        N_i[2] = 1.0 / std::sqrt(5.0);
        N_i[3] = 1.0;
    }
    else {
        N_i[0] = -1.0;
        N_i[1] = -1.0 * std::sqrt(3.0 / 7.0);
        N_i[2] = 0.0;
        N_i[3] =  std::sqrt(3.0 / 7.0);
        N_i[4] = 1.0;
    }
}

void determine_lobatto_weights(int N, double* weights) {
    if (N == 1) {
        weights[0] = 2.0;
    }
    else if (N == 2) {
        weights[0] = 1.0;
        weights[1] = 1.0;
    } else if (N == 3) {
        weights[0] = 1.0 / 3.0;
        weights[1] = 4.0 / 3.0;
        weights[2] = 1.0 / 3.0;
    } else if (N == 4) {
        weights[0] = 1.0 / 6.0;
        weights[1] = 5.0 / 6.0;
        weights[2] = 5.0 / 6.0;
        weights[3] = 1.0 / 6.0;
    } else {
        weights[0] = 1.0 / 10.0;
        weights[1] = 49.0 / 90.0;
        weights[2] = 32.0 / 45.0;
        weights[3] = 49.0 / 90.0;
        weights[4] = 1.0 / 10.0;
    }
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
        N_i[4] = -std::sqrt((15 + 2 * std::sqrt(30)) / 35);
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

void iso_map(const double* coord1, const double* coord2, double xi,  double* mapped_coord) {
    double N1 = 1.0 - xi;
    double N2 = xi;
    // double N1 = 0.5 - xi;
    // double N2 = 0.5 + xi;
    mapped_coord[0] = N1 * coord1[0] + N2 * coord2[0];
    mapped_coord[1] =  N1 * coord1[1] + N2 * coord2[1];
}

void iso_map2(const double* coord1, const double* coord2, double xi, double* mapped_coord){
    double N1 = 0.5 - xi;
    double N2 = 0.5 + xi;
    mapped_coord[0] = N1 * coord1[0] + N2 * coord2[0];
    mapped_coord[1] =  N1 * coord1[1] + N2 * coord2[1];
}



void iso_map_deriv(const double* coord1, const double* coord2, double* deriv) {
    deriv[0] = 0.5 * (coord2[0] - coord1[0]);
    deriv[1] = 0.5 * (coord2[1] - coord1[1]);
}

bool segmentsIntersect(const double A0[2], const double A1[2],
                       const double B0[2], const double B1[2],
                       double intersection[2]) {
    auto cross = [](double x0, double y0, double x1, double y1) {
        return x0 * y1 - y0 * x1;
    };

    double dxA = A1[0] - A0[0], dyA = A1[1] - A0[1];
    double dxB = B1[0] - B0[0], dyB = B1[1] - B0[1];
    double dxAB = B0[0] - A0[0], dyAB = B0[1] - A0[1];

    double denom = cross(dxA, dyA, dxB, dyB);
    double numeA = cross(dxAB, dyAB, dxB, dyB);
    double numeB = cross(dxAB, dyAB, dxA, dyA);

    // Collinear or parallel
    if (std::abs(denom) < 1e-12) {
        if (std::abs(numeA) > 1e-12 || std::abs(numeB) > 1e-12)
            return false; // Parallel, not collinear

        // Collinear: check for overlap
        auto between = [](double a, double b, double c) {
            return std::min(a, b) <= c && c <= std::max(a, b);
        };

        // Check if endpoints overlap
        for (int i = 0; i < 2; ++i) {
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
        }
        // Overlap but not at a single point
        return false;
    }

    double ua = numeA / denom;
    double ub = numeB / denom;

    if (ua >= 0.0 && ua <= 1.0 && ub >= 0.0 && ub <= 1.0) {
        intersection[0] = A0[0] + ua * dxA;
        intersection[1] = A0[1] + ua * dyA;
        return true;
    }
    return false;
}





// void lagrange_shape_functions(int N, double xi, const double* nodes, double* N_i) {
//     for(int i = 0; i < N; ++i) {
//         N_i[i] = 1.0;
//         for(int j = 0; j < N; j++){
//             if(i != j) {
//                 N_i[i] *= (xi - nodes[j]) / (nodes[i] - nodes[j]);
//             }
//         }
//     }
// }

// void iso_map(const double* coords, int N, double* mapped_coords, double xi) {
//     double nodes[N];
//     double shape_functions[N];
//     determine_lobatto_nodes(N, nodes);
//     lagrange_shape_functions(N, xi, nodes, shape_functions);
//     mapped_coords[0] = 0.0;
//     mapped_coords[1] = 0.0;
//     for(int i = 0; i < N; ++i) {
//         mapped_coords[0] += shape_functions[i] * coords[2 * i];
//         mapped_coords[1] += shape_functions[i] * coords[2 * i + 1]; 
//     }
// }


// void iso_map_deriv(double xi, const double* coords, int N, double* dxi_dx) {
//     double mapped_coords[2] = {0.0, 0.0};
//     double d_mapped_coords[2] = {0.0, 0.0};
//     double dxi = 1.0;
//     __enzyme_autodiff<void>( iso_map, enzyme_const, coords, enzyme_const, N, enzyme_dup, mapped_coords, d_mapped_coords, enzyme_dup, xi, dxi);

//     dxi_dx[0] = d_mapped_coords[0];
//     dxi_dx[1] = d_mapped_coords[1];
// }

// double compute_jacobian(const double* coords, const double* derivs, int N) {
//     double dx_dxi = 0.0;
//     double dy_dxi = 0.0;

//     for (int i = 0; i < N; ++i) {
//         dx_dxi += derivs[i] * coords[2 * i];
//         dy_dxi += derivs[i] * coords[2 * i + 1];
//     }

//     double J = 0.5 * std::sqrt(dx_dxi * dx_dxi + dy_dxi * dy_dxi);
//     return J;
// }


// double newtons_method(const double* p, const double* coord1, const double* coord2, double tol = 1e-20, int iter = 20) {
//     double xi = 0.0; //initial guess

//     for(int i = 0; i < iter; ++i) {
//         double mapped_coords[2] = {0.0, 0.0};
//         iso_map(coord1, coord2, xi, mapped_coords);

//         //compute residuals
//         double rx = mapped_coords[0] - p[0];
//         double ry = mapped_coords[1] - p[1];

//         double dx_dxi[2] = {0.0, 0.0};
//         iso_map_deriv(coord1, coord2, dx_dxi);

//         double grad = 2.0 * (rx * dx_dxi[0] + ry * dx_dxi[1]);
//         double hess = 2.0 * (dx_dxi[0] * dx_dxi[0] + dx_dxi[1] * dx_dxi[1]);
//         //newton step
//         double step = grad / hess;
//         xi -= step;

//         //clamp xi to [-1, 1] for segment
//         xi = std::max(-1.0, std::min(1.0, xi));

//         if (std::abs(step) < tol) {
//             break;
//         }

        
//     }
//     return xi;
// }




void find_intersection(const double* A0, const double* A1, const double* p, const double* nB, double* intersection) {
    double tA[2] = {A1[0] - A0[0], A1[1] - A0[1] };
    double d[2] = {p[0] - A0[0], p[1] - A0[1]};

    double det = tA[0] * nB[1] - tA[1] * nB[0];

    if(std::abs(det) < 1e-12) {
        intersection[0] = p[0];
        intersection[1] = p[1];
        return;
    }

    double inv_det = 1.0 / det;

    double alpha = (d[0] * nB[1] - d[1] * nB[0]) * inv_det;

    // if (alpha < 0.0) alpha = 0.0;
    // if (alpha > 1.0) alpha = 1.0;

    intersection[0] = (A0[0] + alpha * tA[0]);
    intersection[1] = A0[1]  + alpha * tA[1];

}


// void get_projections(const double* A0, const double* A1, const double* B0, const double* B1, double* projections) {
//     double nA[2] = {0.0};
//     double nB[2] = {0.0}; 
//     find_normal(A0, A1, nA);
//     find_normal(B0, B1, nB);
//     // double eta_values[N];
//     // determine_lobatto_nodes(N, eta_values)
//     double end_points[2] = {-0.5, 0.5}; // change for [-0.5, 0.5] mapping
//     for (int i = 0; i < 2; ++i) {
//         double p[2] = {0.0};

//         double intersection[2] = {0.0};
//         double seg_intersection[2] = {0.0};
//         iso_map2(B0, B1, end_points[i], p);

//         // std::cout << "gx: " << p[0] << "gy: " << p[1] << std::endl;
//         // // double xiA = newtons_method(p, A0, A1);
//         // // tribol::ProjectPointToSegment(p[0], p[1],  nB[0], nB[1], A0[0], A0[1], px, py); 
//         // std::cout << "px: " << p[0] << ", " << "py: " << p[1] <<std::endl;

//         find_intersection(A0, A1, p, nB, intersection);



//         // std::cout << "intersection: " << intersection[0] << ',' << intersection[1] << std::endl;


//         double dx = A1[0] - A0[0];
//         double dy = A1[1] - A0[1];
//         double len2 = dx*dx + dy*dy;
//         double xiA = ((intersection[0] - A0[0]) * dx + (intersection[1] - A0[1]) * dy) / len2;

//         // bool current_inside = (xiA >= 0.0 && xiA <= 1.0);

//         double nB_unit[2] = { nB[0], nB[1] };
//         double norm = std::sqrt(nB_unit[0]*nB_unit[0] + nB_unit[1]*nB_unit[1]);
//         nB_unit[0] /= norm;
//         nB_unit[1] /= norm;

//         double dx_gap = intersection[0] - p[0];
//         double dy_gap = intersection[1] - p[1];
//         double gap = dx_gap * nB_unit[0] + dy_gap * nB_unit[1];



//         // if (gap > 0) {
//         //     xiA_was_inside[i] = true;  // mark this slot as valid
//         // }

//         double del = 0.1;

//         if(segmentsIntersect(A0, A1, B0, B1, seg_intersection) &&  gap > 0.0) {
//             // std::cout << "Segments intersect" << std::endl;
//             // if(xiA < 0.0 || xiA > 1.0) {
//                 // std::cout << "entered loop" << std::endl;
//                 // std::cout << "Seg intersection: " << seg_intersection[0] << ", " << seg_intersection[1] << std::endl;
//                 // std::cout << "xia before: " << xiA << std::endl;
//                 xiA = ((seg_intersection[0] - A0[0]) * dx + (seg_intersection[1] - A0[1]) * dy) / len2;
//                 // std::cout << "xia after: " << xiA << std::endl;
//                 if (xiA < del) {
//                     xiA = del;
//                 }
//                 // std::cout << "xia after: " << xiA << std::endl;
//             // }
//         }
//         xiA = xiA - 0.5;
//         // xiA = (xiA + 1) / 2;
//         // std::cout << "Xia: " << xiA << std::endl;  //PICK UP HERE******
//         projections[i] = xiA;
//     }
// }

// void get_endpoint_projections(const double* A0, const double* A1, const double* B0, const double* B1, double* proj0, double* proj1) {
//     double nA[2];
//     find_normal(A0, A1, nA);
//     find_intersection(B0, B1, A0, nA, proj0);
//     find_intersection(B0, B1, A1, nA, proj1);

// }


// void get_projections(const double* A0, const double* A1, const double* B0, const double* B1, double* projections, double del) {
//     double nA[2] = {0.0}; 
//     find_normal(A0, A1, nA);
    
//     double end_points[2] = {-0.5, 0.5}; 
//     for (int i = 0; i < 2; ++i) {
//         double p[2] = {0.0};
//         iso_map2(B0, B1, end_points[i], p);
//         std::cout << "EndPoints: " << end_points[0] << ", " << end_points[1] << std::endl;
        
//         double intersection[2] = {0.0};
//         find_intersection(B0, B1, p, nA, intersection);
//         std::cout << "intersection: " << intersection[0] << ", " << intersection[1] << std::endl;

//         // Convert intersection to parametric coordinate on A
//         // double dx = A1[0] - A0[0];
//         // double dy = A1[1] - A0[1];
//         // double len2 = dx*dx + dy*dy;
//         // std::cout << "len2: " << len2 << std::endl;
//         // double xiA = ((intersection[0] - A0[0]) * dx + (intersection[1] - A0[1]) * dy) / len2;
//         // std::cout << "Xia: " << xiA << std::endl;
        
//         // Apply constraints and convert to reference interval
//         // xiA = std::max(del, std::min(1.0 - del, xiA)) - 0.5;
  
//         // xiA = 0.5 - xiA;
//         projections[i] = intersection[i];
//     }
// }
void get_projections(const double* A0, const double* A1, const double* B0, const double* B1, double* projections, double del) {
    double nB[2] = {0.0};
    find_normal(B0, B1, nB);
    double B_endpoints[2][2];
    B_endpoints[0][0] = B0[0]; B_endpoints[0][1] = B0[1];
    B_endpoints[1][0] = B1[0]; B_endpoints[1][1] = B1[1];
    
    for (int i =0; i < 2; ++i) {
        //prohect A endpoints onto B
        double intersection[2] = {0.0};
        find_intersection(A0,A1, B_endpoints[i], nB, intersection);

        // std::cout << "Intersection: " << intersection[0] << ", " << intersection[1] << std::endl;
    

        //convert to parametric coords
        double dx = A1[0] - A0[0];
        // std::cout << "dx: " << dx << std::endl;
        double dy = A1[1] - A0[1];
        // std::cout << "dy: " << dy << std::endl;
        double len2 = dx*dx + dy*dy;
        double alpha = ((intersection[0] - A0[0]) * dx + (intersection[1] - A0[1]) * dy) / len2;
        //map to xiB
        // std::cout << "alpha: " << alpha << std::endl;
        // double xiB = 0.5 - alpha;
        double xiB = alpha - 0.5;
        // xiB = std::max(-0.5, std::min(0.5, xiB));
        
        // std::cout << "xi on B: " << xiB << std::endl;
        
        projections[i] = xiB;
    }

}





// void compute_integration_bounds(const double* projections, double* integration_bounds, int N) {
//     double xi_min = projections[0];
//     double xi_max = projections[0];
//     for (int i = 0; i < 2; ++i) {
//         if (xi_min > projections[i]) {
//             xi_min = projections[i];
//         }
//         if(xi_max < projections[i]) {
//             xi_max = projections[i]; 
//         }

//     }

//     if (xi_max < -0.5) {
//         xi_max = -0.5;
//     }
//     if(xi_min > 0.5) {
//         xi_min  = 0.5;
//     }
//     if (xi_min < -0.5) { 
//         xi_min = -0.5;
//     }
//     if (xi_max > 0.5) {
//         xi_max = 0.5;
//     }

//     double del = 0.1;

//     integration_bounds[0] = xi_min;
//     integration_bounds[1] = xi_max;
//     // std::cout << "x_min: " << xi_min << "  xi_max: " << xi_max << std::endl;

// }

void compute_integration_bounds(const double* projections, double* integration_bounds, double del) {
    // std::cout << "Projections in Compute bounds: " << projections[0] << ", " <<  projections[1] << std::endl;
    double xi_min = projections[0];
    double xi_max = projections[0];
    for (int i = 0; i < 2; ++i) {
        if (xi_min > projections[i]) {
            xi_min = projections[i];
        }
        if(xi_max < projections[i]) {
            xi_max = projections[i]; 
        }

    }

    // std::cout << "BEFORE xi min: " << xi_min << " xi_max: " << xi_max << std::endl;


    if (xi_max < -0.5 -del ) {
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

    // if (xi_max < -0.5) {
    //     xi_max = -0.5;
    // }
    // if(xi_min > 0.5) {
    //     xi_min  = 0.5;
    // }
    // if (xi_min < -0.5) { 
    //     xi_min = -0.5;
    // }
    // if (xi_max > 0.5) {
    //     xi_max = 0.5;
    // }

    integration_bounds[0] = xi_min;
    integration_bounds[1] = xi_max;
    // std::cout << "xi min: " << xi_min << " xi_max: " << xi_max << std::endl;
}


void modify_bounds(double* integration_bounds, double del, double* modified_bounds) {
    double xi = 0.0;
    double int_bound[2] = {0.0};
    for(int i = 0; i < 2; ++i) {
        int_bound[i] = integration_bounds[i];
    }
    // int_bound[0] -= del;
    // int_bound[1] += del;


    for (int i = 0; i < 2; ++i) {
        double xi_hat = 0.0;
        // xi = 0.5 * (integration_bounds[i] + 1.0);
        xi = int_bound[i] + 0.5;
        // std::cout << "xi in smoothoing: " << xi << std::endl;
        if (0.0 - del <= xi && xi <= del) {
            xi_hat = (1.0/(4*del)) * (xi*xi) + 0.5 * xi + del/4.0;
            // std::cout << "zone1" << std::endl;
        }
        else if((1.0 - del) <= xi && xi <= 1.0 + del) {
        double b = -1.0/(4.0*del);
        double c = 0.5 + 1.0/(2.0*del);
        double d = 1.0 - del + (1.0/(4.0*del)) * pow(1.0-del, 2) - 0.5*(1.0-del) - (1.0-del)/(2.0*del);

        xi_hat = b*xi*xi + c*xi + d;

            // xi_hat = (1.0/del) * xi*xi - (2.0*(1.0-del)/del) * xi + (-1.0 + 1.0/del);
            // xi_hat = -1.0/del * xi*xi + 2.0/del * xi + (1.0 - 1.0/del);


            // xi_hat= (-1.0/(del*del))*pow(xi,3) + ((3.0/(del*del)) - (2.0/del))*pow(xi,2) + ((-3.0/(del*del)) + (4.0/del))*xi + (1.0 + (1.0/(del*del)) - (2.0/del));


            // xi_hat = -1.0/(del*del)*pow(xi,3) + (3.0+del)/(del*del)*pow(xi,2) + (1.0 + (-3.0-2.0*del)/(del*del))*xi + (1.0+del)/(del*del);

        //     double d = 1 - del
        //  + (1.0 / (4.0 * del * del)) * (1 - 3 * del + 3 * del * del - del * del * del)
        //  - ((-1.0 / (4.0 * del) + 3.0 / (4.0 * del * del)) * (1 - 2 * del + del * del))
        //  - ((5.0 / 4.0 + 1.0 / (2.0 * del) - 3.0 / (4.0 * del * del)) * (1 - del));
        //         xi_hat = 
        // -1.0*(xi*xi*xi) / (4.0 * del * del)
        // + (-1.0/(4.0*del) + 3.0/(4.0*del*del)) * (xi*xi)
        // + (1.25 + 1.0/(2.0*del) - 3.0/(4.0*del*del)) * xi
        // + d;
            // std::cout << "d: " << d << std::endl;
   
        //  std::cout << "zone2" << std::endl;
        }
        else if(del <= xi && xi <= (1.0 - del)) { 
            xi_hat = xi;
            // std::cout << "zone3" << std::endl;
        }
        else{ 
            // std::cerr << "Xi did not fall in an expected range for modifying bounds for 1" << std::endl;
        }
        // modified_bounds[i] = 2.0 * xi_hat - 1;
        modified_bounds[i] = xi_hat - 0.5;
    }
    // std::cout << modified_bounds[0] << ',' << modified_bounds[1] << std::endl;
}

// void modify_bounds(double* integration_bounds, double del, double* modified_bounds) {
//     double xi = 0.0;
//     // integration_bounds[0] -= del;
//     // integration_bounds[1] += del;


//     for (int i = 0; i < 2; ++i) {
//         double xi_hat = 0.0;
//         // xi = 0.5 * (integration_bounds[i] + 1.0);
//         xi = integration_bounds[i] + 0.5;
//         // std::cout << "xi: " << xi << std::endl;
//         if (0.0 <= xi && xi <= del) {
//             xi_hat = (1.0/del) * (xi*xi) + 0.5 * xi + del/4.0;
//             // std::cout << "zone1" << std::endl;
//         }
//         else if((1.0 - del) <= xi && xi <= 1.0) {
//             xi_hat =  1.0 -(((1.0 - xi) * (1.0 - xi)) / (2 * del * (1.0 - del)));
//             // std::cout << "zone2" << std::endl;
//         }
//         else if(del <= xi && xi <= (1.0)) { 
//             xi_hat = xi;
//             // std::cout << "zone3" << std::endl;
//         }
//         else{ 
//             std::cerr << "Xi did not fall in an expected range for modifying bounds for 1" << std::endl;
//         }
//         // modified_bounds[i] = 2.0 * xi_hat - 1;
//         modified_bounds[i] = xi_hat - 0.5;
//     }
//     // std::cout << "modified bounds: " << modified_bounds[0] << ',' << modified_bounds[1] << std::endl;
// }

void modify_bounds_for_weight(double* integration_bounds, double del, double* modified_bounds) {
    double xi = 0.0;
    integration_bounds[0];
    integration_bounds[1];
    for (int i = 0; i < 2; ++i) {
        double xi_hat = 0.0;
        // xi = 0.5 * (integration_bounds[i] + 1.0);
        xi = integration_bounds[i] + 0.5;
        if (xi < std::abs(1e-10)) {
            xi = 0.0;
        
        }
        // std::cout << "xi: " << xi << std::endl;
        if (0 <= xi && xi <= del) {
            xi_hat = ((xi)*(xi)) / (2.0 * del * (1.0 - del));
            // std::cout << "zone1" << std::endl;
        }
        else if((1.0 - del) <= xi && xi <= 1.0) {
            xi_hat =  1.0 -(((1.0 - xi) * (1.0 - xi)) / (2 * del * (1.0 - del)));
            // std::cout << "zone2" << std::endl;
        }
        else if(del <= xi && xi <= (1.0 - del)) { 
            xi_hat = ((2.0 * xi) - del) / (2.0 * (1.0 - del));
            // std::cout << "zone3" << std::endl;
        }
        else{ 
            std::cerr << "Xi did not fall in an expected range for modifying bounds for weight fpr 2" << std::endl;
        }
        // modified_bounds[i] = 2.0 * xi_hat - 1;
        modified_bounds[i] = xi_hat - 0.5;
    }
    // std::cout << "modified bounds: " << modified_bounds[0] << ',' << modified_bounds[1] << std::endl;
}


void compute_quadrature_point(double* integration_bounds, const double* A0, const double* A1, int N, double* quad_points) {
    // std::cout << "=== ENTERING compute_quadrature_point ===" << std::endl;
    double eta_values[N];
    determine_legendre_nodes(N, eta_values);
    // for(int i = 0; i < N; ++i) {
    //     eta_values[i] = (eta_values[i] + 1) / 2;
    // }



    // for (int i = 0; i < N; ++i) {
    //     eta_values[i] = eta_values[i] - 0.5;  // scale to [-0.5, 0.5] per suggestion of mike.
    // }

    // for (int i = 0; i < N; ++i) {
    //     eta_values[i] *= 0.5;
    // }

    double xi_min = integration_bounds[0];
    double xi_max = integration_bounds[1];
    // std::cout << "xi values: " << xi_min << ", " << xi_max << std::endl;

    for ( int i = 0; i < N; ++i) {
        double xi_i = 0.5 * (xi_max - xi_min) * eta_values[i] + 0.5 * (xi_max + xi_min); //this was th original implementation
        // double xi_i = 0.5 * (xi_max + xi_min) + eta_values[i] + 0.5 *(xi_max - xi_min); //mikes suggestions
        // double xi_i = xi_min + (xi_max - xi_min) * eta_values[i];
        // xi_i *= 0.5;
        double mapped_coords[2] = {0.0, 0.0};


        iso_map2(A0, A1, xi_i, mapped_coords);
        quad_points[2 * i] = mapped_coords[0];
        quad_points[2 * i + 1] = mapped_coords[1];
        // std::cout << "x: " << quad_points[2 * i] << " y: " << quad_points[2 * i + 1] << std::endl;
        
    }
    
     
}

void assign_weights(const double* integration_bounds, int N, double* weights) {
    double ref_weights[N];
    determine_legendre_weights(N, ref_weights);
    // std::cout << integration_bounds[0] << ' ' << integration_bounds[1] << std::endl;
    double J = 0.0;
  


    double xi_min = integration_bounds[0];
    double xi_max = integration_bounds[1];
    
    J = 0.5 * (xi_max - xi_min);

    for( int i = 0; i < N; ++i) {
        weights[i] = ref_weights[i] * J;
    }
}


// double compute_gap(const double* p, const double* B0, const double* B1, double* A0, double* A1, double* nB) {
//     double nB_orig[2] = {nB[0], nB[1]};
//     double len = std::sqrt(nB[0] * nB[0] + nB[1] * nB[1]);
//     // std::cout << len << std:: endl;
//     nB_orig[0] /= len;
//     nB_orig[1] /= len;
//     // std::cout << "nbx: " << nB_orig[0] << " nby: " << nB_orig[1] << std::endl;

//     double intersection[2] = {0.0};
//     find_intersection(B0, B1, p, nB_orig, intersection);

//     // std::cout << "intersection at B: " << intersection[0] << ", " << intersection[1] << std::endl;
  
//     //  std::cout << "intersection for gap: " << intersection[0] << ',' << intersection[1] << std::endl;

//     // double eta = newtons_method(p, B0, B1); //closest projection of p onto elem B
//     // double px, py;
//     // tribol::ProjectPointToSegment(p[0], p[1],nB_orig[0], nB_orig[1], B0[0], B0[1], px, py);

//     // double q[2] = {0.0, 0.0};
//     // iso_map(B0, B1, eta, q); //map eta back to physical space to get closest point q on A

//     double dx = intersection[0] - p[0];
//     double dy = intersection[1] - p[1];
//     // std::cout << "px: " << p[0] << "py: " << p[1] << std::endl;
//     // std::cout << "dx: " << dx << "dy: " << dy << std::endl;

//     double gap = dx * nB_orig[0] + dy * nB_orig[1];

//     // if(dx == 0 && dy == 0){
//     //     gap = (A0[1] - p[1]) * nB_orig[1];
//     //     // std::cout << "gap in loop: " << gap << std::endl;
//     //     return gap;
        

//     // }
//     // std::cout << "gap in compute_gap: " << gap << std::endl;
//     return gap;
// }

double compute_gap(const double* p, const double* B0, const double* B1, const double* nA, const double* A0, const double* A1) {
    double nA_orig[2] = {nA[0], nA[1]};
    // std::cout << "nA: " << nA_orig[0] << ", " << nA_orig[1] << std::endl;

    double len = std::sqrt(nA[0] * nA[0] + nA[1] * nA[1]);
    // std::cout << "LEN: " << len << std::endl;
    nA_orig[0] /= len;
    nA_orig[1] /= len;
    double intersection[2] = {0.0};
    find_intersection(B0, B1, p, nA_orig, intersection);
    // std::cout << "INTERSECTION: " << intersection[0] << ", " << intersection[1] << std::endl;


    double dx = intersection[0] - p[0];
    double dy = intersection[1] - p[1];

    double gap = dx * nA_orig[0] + dy * nA_orig[1];
    gap *= -1;
    // std::cout << "GAP: " << gap << std::endl;
    return gap;
}


double compute_modified_gap(double gap, double* nA, double* nB) {
    double dot = nA[0] * nB[0] + nA[1] * nB[1];
    double eta = (dot < 0) ? -dot:0.0;

//    if(nu >= 0) {
//         nu = 0;
//     } 

//     gap *= nu;
    // std::cout << "gap in modify gap: " << gap << std::endl;
    // std::cout << "eta: " << eta << std::endl;
    return gap * eta;
}


double compute_contact_potential(double gap, double k1, double k2) {
    if (gap < 1e-12) {
        return 0;
    }
    double gap1 = gap;
    double pot = k1 * (gap1 * gap1) - k2 * (gap1 * gap1 * gap1);
    // std::cout << "potential: " << pot << std::endl;
    return pot;
}


void compute_contact_energy(const double* coords, double del, double k1, double k2, int N, double lenA, double* projections, double* energy) {
    double A0[2] = {coords[0], coords[1]};
    double A1[2] = {coords[2], coords[3]};
    double B0[2] = {coords[4], coords[5]};
    double B1[2] = {coords[6], coords[7]};

    // double lenA = sqrt((A1[0] - A0[0]) * (A1[0] - A0[0]) + (A1[1] - A0[1]) * (A1[1] - A0[1]));
    double lenB = sqrt((B1[0] - B0[0]) * (B1[0] - B0[0]) + (B1[1] - B0[1]) * (B1[1] - B0[1]));
    
    double AC[2] = {0.5 * (A0[0]+A1[0]), 0.5*(A0[1]+A1[1])};
    double AR[2] = {0.5 * (A0[0]-A1[0]), 0.5*(A0[1]-A1[1])};
    double normAR = std::sqrt(AR[0]*AR[0] + AR[1]*AR[1]);

    double BC[2] = {0.5 * (B0[0]+B1[0]), 0.5*(B0[1]+B1[1])};
    double BR[2] = {0.5 * (B0[0]-B1[0]), 0.5*(B0[1]-B1[1])};
    double normBR = std::sqrt(BR[0]*BR[0] + BR[1]*BR[1]);

    A0[0] = AC[0] + AR[0] * lenA * 0.5 / normAR;
    A0[1] = AC[1] + AR[1] * lenA * 0.5 / normAR;

    A1[0] = AC[0] - AR[0] * lenA * 0.5 / normAR;
    A1[1] = AC[1] - AR[1] * lenA * 0.5 / normAR;

    B0[0] = BC[0] + BR[0] * lenB * 0.5 / normBR;
    B0[1] = BC[1] + BR[1] * lenB * 0.5 / normBR;;

    B1[0] = BC[0] - BR[0] * lenB * 0.5 / normBR;;
    B1[1] = BC[1] - BR[1] * lenB * 0.5 / normBR;;

    double nA[2] = {0.0};
    double nB[2] = {0.0};
    find_normal(A0, A1, nA);
    find_normal(B0, B1, nB);

    double dot_product = nA[0] * nB[0] + nA[1] * nB[1];

    if (std::abs(dot_product) < 1e-10) {
        *energy = 0;
    }

    else{
 
    // std::cout << "length: " << lenA << std::endl;



    // double projections[2];
    // get_projections(A0, A1, B0, B1, projections);

    double integration_bounds[2];
    compute_integration_bounds(projections, integration_bounds, del);

    // double len = sqrt((A1[0] - A0[0]) * (A1[0] - A0[0]) + (A1[1] - A0[1]) * (A1[1] - A0[1]));
    // std::cout << "length: " << len << std::endl;



    double modified_bounds[2];
    modify_bounds(integration_bounds, del, modified_bounds);
    // std::cout << "Integration Bounds Original" << integration_bounds[0] << ", " << integration_bounds[1] << std::endl;
    // std::cout << "Modifed Bounds" << modified_bounds[0] << ',' << modified_bounds[1] << std::endl;

    // double modified_bounds_w[2];
    // modify_bounds_for_weight(integration_bounds, del, modified_bounds_w);

//     std::cout << "A: x from " << A0[0] << " to " << A1[0] << std::endl;
// std::cout << "B: x from " << B0[0] << " to " << B1[0] << std::endl;
// std::cout << "Raw projections from get_projections: [" << projections[0] << ", " << projections[1] << "]" << std::endl;
// std::cout << "Integration bounds: [" << integration_bounds[0] << ", " << integration_bounds[1] << "]" << std::endl;
// std::cout << "Modified bounds for quadrature: [" << modified_bounds[0] << ", " << modified_bounds[1] << "]" << std::endl;
    

    double quad_points[2 * N];
    compute_quadrature_point(modified_bounds, A0, A1, N, quad_points);

    
    // std::cout << "integration Bounds" << integration_bounds[0] << ", " << integration_bounds[1] << std::endl;
    // double modified_bounds_w[2];
    // modify_bounds_for_weight(integration_bounds, del, modified_bounds_w);

    double weights[N];
    assign_weights(modified_bounds, N, weights); //was for weigh orginalally 

    *energy = 0.0;
    for(int i = 0; i < N; ++i) {
        // double p[2] = {quad_points[2 * i], quad_points[2 * i + 1]};
        double mapped_coords[2] = {quad_points[2 * i], quad_points[2 * i + 1]};
        // iso_map2(A0, A1, quad_points[i], mapped_coords); 
        // std::cout << "quad point: " << quad_points[2*i] << std::endl;

        // std::cout << "Mapped coords: " << mapped_coords[0] << ", " << mapped_coords[1] << std::endl;

        double gap = compute_gap(mapped_coords, B0, B1, nA, A0, A1);
        // if (gap < 0.0) {
        //     continue;
        // }
        double smooth_gap = compute_modified_gap(gap, nA, nB);
        // std::cout << "gap: " << smooth_gap << std::endl;

        double potential = compute_contact_potential(smooth_gap, k1, k2);

        *energy +=  weights[i] * potential;
        // std::cout << "energy: " << *energy << std::endl;

    }
    *energy *= lenA * 0.5;
    // std::cout << "energy: " << *energy << std::endl;
    }
}

// void compute_sym_energy(const double* coords, double del, double k1, double k2, int N, double len, double* energy) {
//     double energy1 = 0.0; 
//     compute_contact_energy(coords, del, k1, k2, N, len, &energy1); 

//     double A0[2] = {coords[0], coords[1]};
//     double A1[2] = {coords[2], coords[3]};
//     double B0[2] = {coords[4], coords[5]};
//     double B1[2] = {coords[6], coords[7]};

//     double nA[2] = {0.0};
//     double nB[2] = {0.0};
 
//     // std::cout << "length: " << len << std::endl;
//     double energy2 = 0.0;

//     find_normal(A0, A1, nA);
//     find_normal(B0, B1, nB);

//     double projections[2];
//     get_projections(A0, A1, B0, B1, projections, N);

//     double integration_bounds[2];
//     compute_integration_bounds(projections, integration_bounds, N);

//     // double switch_bounds[2] = {integration_bounds[1], integration_bounds[0]};



//     double modified_bounds[2];
//     modify_bounds(integration_bounds, del, modified_bounds);
//     // std::cout << modified_bounds[0] << ',' << modified_bounds[1] << std::endl;
    
//      double switch_bounds[2] = {modified_bounds[1], modified_bounds[0]};

//     double quad_points[2 * N];
//     compute_quadrature_point(switch_bounds, A0, A1, N, quad_points);
    
    

//     // double modified_bounds[2];
//     // modify_bounds(switch_bounds, del, modified_bounds);

//     double weights[N];
//     assign_weights(switch_bounds, N, weights);

//     *energy = 0.0;
//     for(int i = 0; i < N; ++i) {
//         double p[2] = {quad_points[2 * i], quad_points[2 * i + 1]};

//         double gap = compute_gap(p, B0, B1, A0, A1, nB);
//         double smooth_gap = compute_modified_gap(gap, nA, nB);
//         // std::cout << smooth_gap << std::endl;

//         double potential = compute_contact_potential(smooth_gap, k1, k2);

//         energy2 +=  weights[i] * potential;

//     }
//     energy2 *= len * 0.5;

//     *energy = 0.5 * (energy1 - energy2);
    

// }







void read_element_coords(int N, std::vector<double>& coords) {
    for(int i = 0; i < 2; ++i) {
        double x;
        double y;
        std::cout << "Enter x" << i+1 << ": ";
        std::cin >> x;
        
        std::cout << "Enter y" << i+1 << ": ";
        std::cin >> y;
        
        coords.push_back(x);
        coords.push_back(y);
    }
}

void populate_C_arrays(double* C, const std::vector<double>& elem) {
    for (size_t i = 0; i < elem.size(); ++i){
        C[i] = elem[i];
    }
}


// void calc_force(double* coords, double del, double k1, double k2, int N, double len, double* dE_dX) {
// double E = 0.0;
// for (int i = 0; i < 8; ++i) {
//     double dcoords[8] = {0.0};
//     dcoords[i] = 1.0;
//     double dk1 = 0.0;
//     double dk2 = 0.0;
//     double ddel = 0.0;
//     double dE = 1.0;
//     double dlen = 0.0;
//     __enzyme_fwddiff<void>( compute_contact_energy, coords, dcoords, del, ddel, k1, dk1, k2, dk2, enzyme_const, N, dlen, len, &E, &dE);
//     dE_dX[i] = -dE;

// }
// }

void calc_force_reverse(const double* coords, double del, double k1, double k2, int N, double len, double* projections, double* dE_dX) {
    double dcoords[8] = {0.0};
    double E = 0.0;
    double dE = 1.0;
    __enzyme_autodiff<void>( compute_contact_energy, enzyme_dup, coords, dcoords, enzyme_const, del, enzyme_const, k1, enzyme_const, k2, enzyme_const, N, enzyme_const, len, enzyme_const, projections, enzyme_dup, &E, &dE);

    for(int i = 0; i < 8; ++i) {
        dE_dX[i] = dcoords[i];
    }
}

// void calc_force_FD(double* coords, double del, double k1, double k2, int N, double* dE_dX, double h = 1e-10) {
//     double X_plus[8] = {0.0};
//     double X_minus[8] = {0.0};
//     double  E_plus = 0.0;
//     double E_minus;
//     for(int i = 0; i < 8; ++i) {
//         for (int j = 0; j < 8; ++j) {
//             X_plus[j] = coords[j];
//             X_minus[j] = coords[j];
//         }
//         X_plus[i] = coords[i] + h;
//         X_minus[i] = coords[i] - h;
//         compute_contact_energy(X_plus, del, k1, k2, N, len, &E_plus);
//         compute_contact_energy(X_minus, del, k1, k2, N, len, &E_minus);
//         dE_dX[i] = (E_plus - E_minus) / (2 * h);
//      }

// }


void calc_stiffness_rev_fwd(double* coords, double del, double k1, double k2, int N, double lenA, double* projections, double* force, double* d2E_d2X) {
    double dE[8] = {0.0};
    double d2E[8] = {0.0};
    double dEF[8] = {0.0};
    calc_force_reverse(coords, del, k1, k2, N, lenA, projections, dEF);
    for (int i = 0; i < 8; ++i) {
        force[i] = dEF[i];
    }
    for(int i = 0; i < 8; ++i) {
        double d2coords[8] = {0.0};
        d2coords[i] = 1.0;
        double d2k1 = 0.0;
        double d2del = 0.0;
        double d2k2 = 0.0;
        double d2lenA = 0.0;
        double d2projections[] = {0.0};
        __enzyme_fwddiff<void>( (void*) calc_force_reverse, coords, d2coords, del, d2del, k1, d2k1, k2, d2k2, N, lenA, d2lenA, projections, d2projections, dE, d2E);
        for(int j = 0; j < 8; ++j) {
            d2E_d2X[8 * i + j] = d2E[j];
        }

    }
}

// void calc_stiffness_rev_rev(double* coords, double del, double k1, double k2, int N, double lenA, double lenB, double* d2E_d2X) {
//     for (int i = 0; i < 8; ++i) {
//         double d2X[8] = {0.0};
//         double dE[8] = {0.0};
//         double d2E[8] = {0.0};
//         d2E[i] = 1.0;
//         __enzyme_autodiff<void>( (void*)calc_force_reverse, enzyme_dup, coords, d2X, enzyme_const, del, enzyme_const, k1, enzyme_const, k2, enzyme_const, N, enzyme_const, lenA, enzyme_const, lenB, enzyme_dup, dE, d2E);
//         for(int j = 0; j < 8; ++j) {
//             d2E_d2X[8 * i + j] = d2X[j];
//         }
//     }
// }

// void calc_stiffness_FD(double* coords, double del, double k1, double k2, double lenA , double lenB, int N, double *d2E_d2X, double h = 1e-7) {
//     double dX_plus[8] = {0.0};
//     double dX_minus[8] = {0.0};
//     double dW_plus[8] = {0.0};
//     double dW_minus[8] = {0.0};
//     for (int i = 0; i < 8; ++i) {
//         for (int j = 0; j < 8; ++j) {
//             dX_plus[j] = coords[j];
//             dX_minus[j] = coords[j];
//         }
//         dX_plus[i] = coords[i] + h;
//         dX_minus[i] = coords[i] - h;
        
//         calc_force_reverse(dX_plus, del, k1, k2, N, lenA, lenB, dW_plus);
//         calc_force_reverse(dX_minus, del, k1, k2, N, lenA, lenB, dW_minus);
//         for(int j = 0; j < 8; ++j){
//         d2E_d2X[8 * i + j] = (dW_plus[j] - dW_minus[j]) / (2  * h);
        
//     }

// }
// }

// void calc_ab(const double* coord1, const double* coord2, const double* normal, double* a){
//     double y_diff = coord1[1] - coord2[1];
//     double x_diff = coord1[0] - coord2[0];
//     *a = (x_diff * normal[0]) + (y_diff * normal[1]);

// }

// void analytical_integral(const double* coords, double del, double k1, double k2, int N, double len, double* energy) {
//     double A0[2] = {coords[0], coords[1]};
//     double A1[2] = {coords[2], coords[3]};
//     double B0[2] = {coords[4], coords[5]};
//     double B1[2] = {coords[6], coords[7]};

//     double nB[2] = {0.0};
//     find_normal(B0, B1, nB);
//     double a = 0.0;
//     calc_ab(A1, A0, nB, &a);
//     double b = 0.0;
//     calc_ab(A0, B0, nB, &b);

//     double projections[2] = {0.0};
//     get_projections(A0, A1, B0, B1, projections);

//     double integration_bounds[2] = {0.0};
//     compute_integration_bounds(projections, integration_bounds, N);
//     double xi[2] = {0.0};
//     modify_bounds(integration_bounds,del, xi);

//     double term_one = (k1 * ((a * a) * (xi[1] * xi[1] * xi[1] / 3) + a * b * xi[1] + (b * b * xi[1])) + k2 * ((a * a * a) * (xi[1] * xi[1] * xi[1] * xi[1]) / 4) + (a * a) * (xi[1] * xi[1]) * b + ((3 * a * (xi[1] * xi[1] * xi[1]) * b) / 2) + (b * b * b) * (xi[1]));
//     double term_two = (k1 * ((a * a) * (xi[0] * xi[0] * xi[0] / 3) + a * b * xi[0] + (b * b * xi[0])) + k2 * ((a * a * a) * (xi[0] * xi[0] * xi[0] * xi[0]) / 4) + (a * a) * (xi[0] * xi[0]) * b + ((3 * a * (xi[0] * xi[0] * xi[0]) * b) / 2) + (b * b * b) * (xi[0]));

//     *energy = term_one - term_two;
//     *energy *= len;
// }

// void calc_force_reverse_exact(double* coords, double del, double k1, double k2, int N, double len, double* dE_dX) {
//     double dcoords[8] = {0.0};
//     double E = 0.0;
//     double dE = 1.0;
//     __enzyme_autodiff<void>( analytical_integral, enzyme_dup, coords, dcoords, enzyme_const, del, enzyme_const, k1, enzyme_const, k2, enzyme_const, N, enzyme_const, len, enzyme_dup, &E, &dE);

//     for(int i = 0; i < 8; ++i) {
//         dE_dX[i] = -dcoords[i];
//     }
// }

// void calc_force_reverse_sym(double* coords, double del, double k1, double k2, int N, double len, double* dE_dX) {
//     double dcoords[8] = {0.0};
//     double E = 0.0;
//     double dE = 1.0;
//     __enzyme_autodiff<void>( compute_sym_energy, enzyme_dup, coords, dcoords, enzyme_const, del, enzyme_const, k1, enzyme_const, k2, enzyme_const, N, enzyme_const, len, enzyme_dup, &E, &dE);

//     for(int i = 0; i < 8; ++i) {
//         dE_dX[i] = -dcoords[i];
//     }
// }






int main() {
    // int N;
    // std::cout << "Enter N quadrature points: ";
    // std::cin >> N;
    
    // if(N !=3 && N != 4 && N != 5) {
    //     std::cerr << "Error: not a valid number qaud pts" << std::endl;
    // }
     
    // std::vector<double> elem_A;
    // std::vector<double> elem_B;

    // std::cout << "Enter coordinates for element A: 
    // read_element_coords(N, elem_A);

    // std::cout << "Eneter coordinates for element B: ";
    // read_element_coords(N, elem_B);

    // double A[4] = {0.0};
    // double B[4] = {0.0};

    // populate_C_arrays(A, elem_A);
    // populate_C_arrays(B, elem_B);

    int N = 3;

    // double A0[2] = {A[0], A[1]};
    // double A1[2] = {A[2], A[3]};
    // double B0[2] = {B[0], B[1]};
    // double B1[2] = {B[2], B[3]};

    double A0_i[2] = {-0.3, -0.05};
    double A1_i[2] = {0.0, -0.05};
    double B0[2] = {1.0, 0.0};
    double B1[2] = {0.1, 0.0};
    double del = 0.05;
    double k1 = 100;
    double k2 = 0.0;
    for(int i = 0; i < 140; ++i) {
        // std::cout << i << std::endl;
        double energy = 0.0;
        double energy2;
        double shift = 0.01 * i;
        
        // std::cout << i << std::endl;
        // std::cout << "location: " << shift << std::endl;
        double A0[2] = {A0_i[0] + shift, A0_i[1]};
        double A1[2] = {A1_i[0] + shift, A1_i[1]};
        
        
        // std::cout << "A0x: " << A0[0] << " A0y: " << A0[1] << std::endl; 
        // std::cout << "A1x: " << A1[0] << " A1y: " << A1[1] << std::endl; 

        double coords[8] = {A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1]};
        double lenA = sqrt((A1[0] - A0[0]) * (A1[0] - A0[0]) + (A1[1] - A0[1]) * (A1[1] - A0[1]));
        
        // double lenB = sqrt((B1[0] - B0[0]) * (B1[0] - B0[0]) + (B1[1] - B0[1]) * (B1[1] - B0[1]));
        // analytical_integral(coords, del, k1, k2, N, len, &energy2); 
        // if (i == 410) {
            // std::cout << "B0x: " << B0[0] << ' ' << "B1x: " << B1[0] << std::endl;
        // }
        // // compute_sym_energy(coords, k1, k2, del, N, len, &energy);
        // compute_contact_energy(coords, del, k1, k2, N, lenA, lenB, &energy);
 
        double dE_dX[8] = {0.0};
        double projections[2] = {0.0};
        double proj1[2];
        double proj0[2];
        // get_endpoint_projections(A0, A1, B0, B1, proj0, proj1);
        // std::cout << "Proj 0: " << proj0[0] << ", " << proj0[1] << std::endl;
        // std::cout << "Proj 1: " << proj1[0] << ", " << proj1[1] << std::endl;
        get_projections(A0, A1, B0, B1, projections, del);
        // std::cout << "Projections in Main: " << projections[0] << ", " << projections[1] << std::endl;
        compute_contact_energy(coords, del, k1, k2, N, lenA, projections, &energy);
        // calc_force_reverse_sym(coords, del, k1, k2, N, len, dE_dX);
        calc_force_reverse(coords, del, k1, k2, N, lenA, projections, dE_dX);
        //  calc_force_reverse_exact(coords, del, k1, k2, N, len, dE_dX);
        // std::cout << '[';
        // for(int j = 0; j < 8; ++j) {
       
                    for(int j = 0; j < 8; ++j) {
                      
                        if (j == 0) {
                            
                            std::cout << dE_dX[j];
        
                        }
                        else{

            std::cout << "," << dE_dX[j];
            
                        }
        }
        std::cout << std::endl;
        // }
        
// // //         std::cout << ']' << std::endl;
    //    std::cout << i * 0.01 << ',' << energy << std::endl;
       double dE_dXrev[8] = {0.0};
    //    calc_force_reverse(coords, del, k1, k2, N, len,dE_dXrev);
    //    std::cout << "[";
    //    for (int j = 0; j < 8; ++j) {
    //     std::cout << ", " << dE_dXrev[j];
    //    }
    //    std::cout << "]" << std::endl;
    //    double dE_dXFD[8] = {0.0};
    //    calc_force_FD(coords, del, k1, k2, N, dE_dXFD);
    //    std::cout << "[";
    //     for (int j = 0; j < 8; ++j) {
    //     std::cout << ", " << dE_dXFD[j];
    //    }
    //    std::cout << "]" << std::endl;
    // double d2E_d2XFD[64] = {0.0};
    // calc_stiffness_rev_fwd(coords, del, k1, k2, N, lenA, lenB, dE_dX, d2E_d2X);
    // std::cout << " rev fwd: [";
    // for (int j = 0; j < 64; ++j) {
    //     std::cout << ", " << d2E_d2X[j];
    // }
    // std::cout << "]" << std::endl;
//      double d2E_d2XFD[64] = {0.0};
    // calc_stiffness_FD(coords, del, k1, k2, lenA, lenB, N, d2E_d2XFD);
//     for (int i = 0; i < 16; ++i) {
//     // Create unit vector e_i
//     double v[16] = {0.0};
//     v[i] = 1.0;

//     // Multiply: result = K * v
//     double result[16] = {0.0};
//     for (int row = 0; row < 16; ++row) {
//         for (int col = 0; col < 16; ++col){
//             result[row] += d2E_d2XFD[16 * row + col] * v[col];
//             if (std::abs(result[row]) < 1e-10) {
//                 result[row] = 0.0;
//             }
//         }
//     }

//     std::cout << "Column " << i << ": [";
//     for (int j = 0; j < 16; ++j) {
//         std::cout << result[j];
//         if (j < 15) std::cout << ", ";
//     }
//     std::cout << "]" << std::endl;
// }

// const int N = 8;
// int k = 5; // The DOF (column) you want

// double result[N] = {0.0};
// for (int j = 0; j < N; ++j) {
//     result[j] = d2E_d2XFD[N * j + k];
//     // This grabs the k-th column (since your matrix is row-major)
//     // If you want the k-th row, swap indices
// }

// // Print result to compare with J_exact
// for (int j = 0; j < N; ++j) {
//     printf("J exact: %.17g\n", result[j]);
// }


   
    //     double d2E_d2XFD[64] = {0.0};
    // calc_stiffness_FD(coords, del, k1, k2, lenA, lenB, N, d2E_d2XFD);
    // std::cout << "FD: [";
    // for (int j = 0; j < 64; ++j) {
    //     std::cout << ", " << d2E_d2XFD[j];
    // }
    // std::cout << "]" << std::endl;

//             double d2E_d2Xrevrev[64] = {0.0};
//     calc_stiffness_rev_rev(coords, del, k1, k2, lenA, lenB, N, d2E_d2Xrevrev);
//     std::cout << "Rev rev: [";
//     for (int j = 0; j < 64; ++j) {
//         std::cout << ", " << d2E_d2Xrevrev[j];
//     }
//     std::cout << "]" << std::endl;

//     std::cout << "Difference rev fwd - FD: [";
// for (int j = 0; j < 64; ++j) {
//     std::cout << ", " << (d2E_d2X[j] - d2E_d2XFD[j]);
// }
// std::cout << "]" << std::endl;

// std::cout << "Difference rev rev - FD: [";
// for (int j = 0; j < 64; ++j) {
//     std::cout << ", " << (d2E_d2Xrevrev[j] - d2E_d2XFD[j]);
// }
// std::cout << "]" << std::endl;

    // double energy = compute_contact_energy(A0, A1, B0, B1, del, k1, k2, N);
    // std::cout << "Energy: " << energy << std::endl;
}
}