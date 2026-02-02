// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "MortarElementForce.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/Math.hpp"
#include <cmath>

namespace tribol {

void MortarElementForce::compute( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* g1, int size1,
                               const RealT* x2, RealT* f2, int size2 )
{
  // convention: elem1 = nonmortar element
  //             elem2 = mortar element
  constexpr int max_mortar_mat_size = 4 * 4;
  RealT mortar_mat1[max_mortar_mat_size];
  int mortar_mat1_size = size1 * size1;
  for ( int i{ 0 }; i < mortar_mat1_size; ++i ) {
    mortar_mat1[i] = 0.0;
  }
  RealT mortar_mat2[max_mortar_mat_size];
  int mortar_mat2_size = size1 * size2;
  for ( int i{ 0 }; i < mortar_mat2_size; ++i ) {
    mortar_mat2[i] = 0.0;
  }
  // get point x0 (geometric center of elem1)
  RealT x0[3] = { 0.0, 0.0, 0.0 };
  for ( int i{ 0 }; i < size1; ++i ) {
    for ( int d{ 0 }; d < 3; ++d ) {
      x0[d] += x1[d * size1 + i] / static_cast<RealT>( size1 );
    }
  }

  // get vector n (normal of elem1) = de1 x de2
  // clang-format off
  RealT de1[3] = { 0.0, 0.0, 0.0 };
  RealT de2[3] = { 0.0, 0.0, 0.0 };
  if ( size1 == 4 ) {
    de1[0] = -0.25*x1[0] + 0.25*x1[1] + 0.25*x1[2] - 0.25*x1[3];
    de1[1] = -0.25*x1[4] + 0.25*x1[5] + 0.25*x1[6] - 0.25*x1[7];
    de1[2] = -0.25*x1[8] + 0.25*x1[9] + 0.25*x1[10] - 0.25*x1[11];
    de2[0] = -0.25*x1[0] - 0.25*x1[1] + 0.25*x1[2] + 0.25*x1[3];
    de2[1] = -0.25*x1[4] - 0.25*x1[5] + 0.25*x1[6] + 0.25*x1[7];
    de2[2] = -0.25*x1[8] - 0.25*x1[9] + 0.25*x1[10] + 0.25*x1[11];
  } else if ( size1 == 3 ) {
    de1[0] = x1[1] - x1[0];
    de1[1] = x1[4] - x1[3];
    de1[2] = x1[7] - x1[6];
    de2[0] = x1[2] - x1[0];
    de2[1] = x1[5] - x1[3];
    de2[2] = x1[8] - x1[6];
  }
  RealT n[3] = {
    de1[1]*de2[2] - de1[2]*de2[1],
    de1[2]*de2[0] - de1[0]*de2[2],
    de1[0]*de2[1] - de1[1]*de2[0]
  };
  // clang-format on
  RealT n_mag = std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
  for ( int d{ 0 }; d < 3; ++d ) {
    n[d] /= n_mag;
  }

  // x1t = x1 coordinates projected to plane p (def'd by x0 and n) but in 3d
  constexpr int max_coord_size = 4 * 3;
  RealT x1t[max_coord_size];
  for ( int i{ 0 }; i < size1; ++i ) {
    RealT x1diff_mag = 0.0;
    for ( int d{ 0 }; d < 3; ++d ) {
      x1diff_mag += n[d] * ( x1[size1 * d + i] - x0[d] );
    }
    for ( int d{ 0 }; d < 3; ++d ) {
      x1t[size1 * d + i] = x1[size1 * d + i] - n[d] * x1diff_mag;
    }
  }
  // x2t = x2 coordinates projected to plane p but in 3d
  RealT x2t[max_coord_size];
  for ( int i{ 0 }; i < size2; ++i ) {
    RealT x2diff_mag = 0.0;
    for ( int d{ 0 }; d < 3; ++d ) {
      x2diff_mag += n[d] * ( x2[size2 * d + i] - x0[d] );
    }
    for ( int d{ 0 }; d < 3; ++d ) {
      x2t[size2 * d + i] = x2[size2 * d + i] - n[d] * x2diff_mag;
    }
  }
  // Tribol's clipping algorithm
  // create a local basis; e1 is a unit vector aligned with the first edge in element 1
  // clang-format off
   RealT e1[3] = {
      x1t[0*size1 + 1] - x1t[0*size1 + 0],
      x1t[1*size1 + 1] - x1t[1*size1 + 0],
      x1t[2*size1 + 1] - x1t[2*size1 + 0]
   };
  // clang-format on
  RealT e1_mag = std::sqrt( e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2] );
  for ( int d{ 0 }; d < 3; ++d ) {
    e1[d] /= e1_mag;
  }
  // e2 is a unit vector = n x e1
  // clang-format off
   RealT e2[3] = {
      n[1]*e1[2] - n[2]*e1[1],
      n[2]*e1[0] - n[0]*e1[2],
      n[0]*e1[1] - n[1]*e1[0]
   };
  // clang-format on
  RealT x1t_2d[4];
  RealT y1t_2d[4];
  PlaneTo2DCoords( x1t, x0, e1, e2, x1t_2d, y1t_2d, size1 );
  RealT x2t_2d[4];
  RealT y2t_2d[4];
  PlaneTo2DCoords( x2t, x0, e1, e2, x2t_2d, y2t_2d, size2 );
  // coordinates need to be CCW for both faces. the call to ElemReverse() will reverse the projected 2d coordinates of
  // element 2, which are in clockwise direction
  RealT x2t_2d_rev[4];
  RealT y2t_2d_rev[4];
  for ( int i{ 0 }; i < size2; ++i ) {
    x2t_2d_rev[i] = x2t_2d[i];
    y2t_2d_rev[i] = y2t_2d[i];
  }
  ElemReverse( x2t_2d_rev, y2t_2d_rev, size2 );
  RealT xti_2d[8];
  RealT yti_2d[8];
  int overlap_poly_size = 0;
  Intersection2DPolygonEnzyme( x1t_2d, y1t_2d, size1, x2t_2d_rev, y2t_2d_rev, size2, 1.0e-8, 1.0e-8, xti_2d, yti_2d,
                               &overlap_poly_size );
  RealT overlap_poly_area = Area2DPolygon( xti_2d, yti_2d, overlap_poly_size );
  if ( overlap_poly_area <= 0.0 ) {
    return;
  }

  // Integrate mortar matrix over the polygon
  // 1. get base triangle integration rule
  RealT base_rule_2d[12];
  RealT base_weights[6];
  {
    RealT wt1 = 0.109951743655322;
    RealT wt2 = 0.223381589678011;
    base_weights[0] = wt1;
    base_weights[1] = wt1;
    base_weights[2] = wt1;
    base_weights[3] = wt2;
    base_weights[4] = wt2;
    base_weights[5] = wt2;
    RealT base_x1 = 0.091576213509771;
    RealT base_x2 = 0.816847572980459;
    RealT base_x3 = 0.108103018168070;
    RealT base_x4 = 0.445948490915965;
    base_rule_2d[0] = base_x1;
    base_rule_2d[1] = base_x1;
    base_rule_2d[2] = base_x2;
    base_rule_2d[3] = base_x1;
    base_rule_2d[4] = base_x1;
    base_rule_2d[5] = base_x2;
    base_rule_2d[6] = base_x3;
    base_rule_2d[7] = base_x4;
    base_rule_2d[8] = base_x4;
    base_rule_2d[9] = base_x3;
    base_rule_2d[10] = base_x4;
    base_rule_2d[11] = base_x4;
  }

  // 2. build the sub-triangles
  // vert0 = centroid of overlap polygon; this will be used as the first vertex of the sub-triangles
  RealT tri_0[2];
  PolyCentroid( xti_2d, yti_2d, overlap_poly_size, tri_0[0], tri_0[1] );
  for ( int i{ 0 }; i < overlap_poly_size; ++i ) {
    int idx1 = i;
    int idx2 = ( i + 1 ) % overlap_poly_size;
    RealT tri_1[2] = { xti_2d[idx1], yti_2d[idx1] };
    RealT tri_2[2] = { xti_2d[idx2], yti_2d[idx2] };
    RealT side1[2] = { tri_2[0] - tri_1[0], tri_2[1] - tri_1[1] };
    RealT side2[2] = { tri_0[0] - tri_1[0], tri_0[1] - tri_1[1] };
    RealT area = 0.5 * ( side1[0] * side2[1] - side1[1] * side2[0] );

    // the sub-triangle is inverted.  likely something went wrong with CG.  don't try to integrate over it.
    if ( area <= 0.0 ) {
      continue;
    }

    for ( int j{ 0 }; j < 6; ++j ) {
      RealT tri_xi[2] = { base_rule_2d[j * 2 + 0], base_rule_2d[j * 2 + 1] };
      RealT tri_phi[3] = { 0.0, 0.0, 0.0 };
      LinIsoTriShapeFunc( tri_xi, tri_phi );
      RealT tri_quad_pt[2] = { tri_phi[0] * tri_0[0] + tri_phi[1] * tri_1[0] + tri_phi[2] * tri_2[0],
                               tri_phi[0] * tri_0[1] + tri_phi[1] * tri_1[1] + tri_phi[2] * tri_2[1] };

      // 3. map sub-triangle coordinate to nonmortar and mortar coordinates
      // NOTE: we ideally want to do this in 2d, but there are finite differencing errors when we do
      RealT tri_quad_pt_3d[3] = { 0.0, 0.0, 0.0 };
      Coords2DToPlane( tri_quad_pt, tri_quad_pt + 1, x0, e1, e2, tri_quad_pt_3d, 1 );
      RealT xi1[2] = { 0.0, 0.0 };
      InvIso( tri_quad_pt_3d, x1t, x1t + size1, x1t + 2 * size1, size1, xi1 );
      RealT xi2[2] = { 0.0, 0.0 };
      InvIso( tri_quad_pt_3d, x2t, x2t + size2, x2t + 2 * size2, size2, xi2 );

      RealT quad_wt = base_weights[j] * area;

      // 4. Evaluate mortar matrix (nonmortar/nonmortar contribs)
      // NOTE: Nonstandard node numbering with InvIso and LinIsoQuadShapeFunc
      for ( int k{ 0 }; k < size1; ++k ) {
        RealT phiA = 0.0;
        if ( size1 == 4 ) {
          LinIsoQuadShapeFunc( xi1[0], xi1[1], k, phiA );
        } else if ( size1 == 3 ) {
          LinIsoTriShapeFunc( xi1[0], xi1[1], k, phiA );
        }
        for ( int l{ 0 }; l < size1; ++l ) {
          RealT phiB = 0.0;
          if ( size1 == 4 ) {
            LinIsoQuadShapeFunc( xi1[0], xi1[1], l, phiB );
          } else if ( size1 == 3 ) {
            LinIsoTriShapeFunc( xi1[0], xi1[1], l, phiB );
          }
          mortar_mat1[k * size1 + l] += phiA * phiB * quad_wt;
        }
      }

      // 5. Evaluate mortar matrix (nonmortar/mortar contribs)
      for ( int k{ 0 }; k < size1; ++k ) {
        RealT phiA = 0.0;
        if ( size1 == 4 ) {
          LinIsoQuadShapeFunc( xi1[0], xi1[1], k, phiA );
        } else if ( size1 == 3 ) {
          LinIsoTriShapeFunc( xi1[0], xi1[1], k, phiA );
        }
        for ( int l{ 0 }; l < size2; ++l ) {
          RealT phiB = 0.0;
          if ( size2 == 4 ) {
            LinIsoQuadShapeFunc( xi2[0], xi2[1], l, phiB );
          } else if ( size2 == 3 ) {
            LinIsoTriShapeFunc( xi2[0], xi2[1], l, phiB );
          }
          mortar_mat2[k * size2 + l] += phiA * phiB * quad_wt;
        }
      }
    }
  }

  // compute gaps
  for ( int i{ 0 }; i < size1; ++i ) {
    g1[i] = 0.0;
    RealT gap_v[3] = { 0.0, 0.0, 0.0 };
    for ( int j{ 0 }; j < size1; ++j ) {
      for ( int d{ 0 }; d < 3; ++d ) {
        gap_v[d] -= mortar_mat1[i * size1 + j] * x1[d * size1 + j];
      }
    }
    for ( int j{ 0 }; j < size2; ++j ) {
      for ( int d{ 0 }; d < 3; ++d ) {
        gap_v[d] += mortar_mat2[i * size2 + j] * x2[d * size2 + j];
      }
    }
    for ( int d{ 0 }; d < 3; ++d ) {
      g1[i] += n1[d * size1 + i] * gap_v[d];
    }
  }

  // compute nonmortar force contributions
  for ( int i{ 0 }; i < size1; ++i ) {
    for ( int d{ 0 }; d < 3; ++d ) {
      f1[d * size1 + i] = 0.0;
    }
    for ( int j{ 0 }; j < size1; ++j ) {
      for ( int d{ 0 }; d < 3; ++d ) {
        f1[d * size1 + i] -= p1[j] * n1[d * size1 + i] * mortar_mat1[j * size1 + i];
      }
    }
  }

  // compute mortar force contributions
  for ( int i{ 0 }; i < size2; ++i ) {
    for ( int d{ 0 }; d < 3; ++d ) {
      f2[d * size2 + i] = 0.0;
    }
    for ( int j{ 0 }; j < size1; ++j ) {
      for ( int d{ 0 }; d < 3; ++d ) {
        f2[d * size2 + i] += p1[j] * n1[d * size1 + i] * mortar_mat2[j * size2 + i];
      }
    }
  }
}

}  // namespace tribol
