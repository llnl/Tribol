// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "ElementNormal.hpp"

#include <cmath>

#include "axom/slic.hpp"

#include "tribol/utils/Math.hpp"

namespace tribol {

TRIBOL_HOST_DEVICE bool PalletAvgNormal::Compute( const RealT* x, const RealT* c, RealT* n, int num_nodes,
                                                  RealT& area ) const
{
  bool face_ok = true;
  area = 0.0;
  for ( int i{ 0 }; i < num_nodes; ++i ) {
    auto i0 = i;
    auto i1 = ( i + 1 ) % num_nodes;
    // first triangle edge vector between the face's two edge nodes
    auto vX1 = x[0 * num_nodes + i1] - x[0 * num_nodes + i0];
    auto vY1 = x[1 * num_nodes + i1] - x[1 * num_nodes + i0];
    auto vZ1 = x[2 * num_nodes + i1] - x[2 * num_nodes + i0];
    // second triangle edge vector between the face centroid and the face edge's first node
    auto vX2 = c[0] - x[0 * num_nodes + i0];
    auto vY2 = c[1] - x[1 * num_nodes + i0];
    auto vZ2 = c[2] - x[2 * num_nodes + i0];
    // compute the contribution to the pallet normal as v1 x v2. Sum these into the face normal component variables
    // stored on the mesh data object
    auto nX = ( vY1 * vZ2 ) - ( vZ1 * vY2 );
    auto nY = ( vZ1 * vX2 ) - ( vX1 * vZ2 );
    auto nZ = ( vX1 * vY2 ) - ( vY1 * vX2 );
    // sum the normal component contributions into the component variables
    n[0] += nX;
    n[1] += nY;
    n[2] += nZ;
    // half the magnitude of the computed normal is the pallet area. Note: this is exact for planar faces and
    // approximate for warped faces. Face areas are used in a general sense to create a face-overlap tolerance
    area += 0.5 * magnitude( nX, nY, nZ );
  }
  // multiply the pallet normal components by fac to obtain avg.
  RealT fac = 1.0 / num_nodes;
  n[0] = fac * n[0];
  n[1] = fac * n[1];
  n[2] = fac * n[2];
  // compute the magnitude of the average pallet normal
  auto mag = magnitude( n[0], n[1], n[2] );
  constexpr RealT nrml_mag_tol = 1.0e-15;
  auto inv_mag = nrml_mag_tol;
  if ( mag >= nrml_mag_tol ) {
    inv_mag = 1.0 / mag;
  } else {
    // TODO (EBC): can we get rid of this check?
    face_ok = false;
  }
  // normalize the average normal
  n[0] *= inv_mag;
  n[1] *= inv_mag;
  n[2] *= inv_mag;
  return face_ok;
}

TRIBOL_HOST_DEVICE bool ElementCentroidNormal::Compute( const RealT* x, const RealT* c, RealT* n, int num_nodes,
                                                        RealT& area ) const
{
  area = 0.0;
  // get vector n (normal of elem1) = de1 x de2, where de1 and de2 are tangent vectors evaluated at the
  // element centroid
  RealT de1[3] = { 0.0, 0.0, 0.0 };
  RealT de2[3] = { 0.0, 0.0, 0.0 };
  if ( num_nodes == 4 ) {
    de1[0] = -0.25 * x[0] + 0.25 * x[1] + 0.25 * x[2] - 0.25 * x[3];
    de1[1] = -0.25 * x[4] + 0.25 * x[5] + 0.25 * x[6] - 0.25 * x[7];
    de1[2] = -0.25 * x[8] + 0.25 * x[9] + 0.25 * x[10] - 0.25 * x[11];
    de2[0] = -0.25 * x[0] - 0.25 * x[1] + 0.25 * x[2] + 0.25 * x[3];
    de2[1] = -0.25 * x[4] - 0.25 * x[5] + 0.25 * x[6] + 0.25 * x[7];
    de2[2] = -0.25 * x[8] - 0.25 * x[9] + 0.25 * x[10] + 0.25 * x[11];
  } else if ( num_nodes == 3 ) {
    de1[0] = x[1] - x[0];
    de1[1] = x[4] - x[3];
    de1[2] = x[7] - x[6];
    de2[0] = x[2] - x[0];
    de2[1] = x[5] - x[3];
    de2[2] = x[8] - x[6];
  } else {
// TODO: switch to TRIBOL_DEVICE_CODE when PR 147 merges
#ifdef TRIBOL_USE_HOST
    SLIC_ERROR( "ElementCentroidNormal::Compute() only 3- and 4-node elements are supported." );
#endif
  }
  n[0] = de1[1] * de2[2] - de1[2] * de2[1];
  n[1] = de1[2] * de2[0] - de1[0] * de2[2];
  n[2] = de1[0] * de2[1] - de1[1] * de2[0];
  RealT n_mag = std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
  for ( int d{ 0 }; d < 3; ++d ) {
    n[d] /= n_mag;
  }
  for ( int i{ 0 }; i < num_nodes; ++i ) {
    auto i0 = i;
    auto i1 = ( i + 1 ) % num_nodes;
    // first triangle edge vector between the face's two edge nodes
    auto vX1 = x[0 * num_nodes + i1] - x[0 * num_nodes + i0];
    auto vY1 = x[1 * num_nodes + i1] - x[1 * num_nodes + i0];
    auto vZ1 = x[2 * num_nodes + i1] - x[2 * num_nodes + i0];
    // second triangle edge vector between the face centroid and the face edge's first node
    auto vX2 = c[0] - x[0 * num_nodes + i0];
    auto vY2 = c[1] - x[1 * num_nodes + i0];
    auto vZ2 = c[2] - x[2 * num_nodes + i0];
    // compute the contribution to the pallet normal as v1 x v2. Sum these into the face normal component variables
    // stored on the mesh data object
    auto nX = ( vY1 * vZ2 ) - ( vZ1 * vY2 );
    auto nY = ( vZ1 * vX2 ) - ( vX1 * vZ2 );
    auto nZ = ( vX1 * vY2 ) - ( vY1 * vX2 );
    // half the magnitude of the computed normal is the pallet area. Note: this is exact for planar faces and
    // approximate for warped faces. Face areas are used in a general sense to create a face-overlap tolerance
    area += 0.5 * magnitude( nX, nY, nZ );
  }
  return true;
}

}  // namespace tribol