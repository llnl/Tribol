// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/NodalEnergyMortar.hpp"

#include <cmath>

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
namespace tribol {
namespace detail {

ContactFormulationResult makeNodalPenaltyGapResult( EnergyMortarNodalQuantities& quantities )
{
  // Penalty gaps are zero where contact is open and, matching the penalty energy, where the tributary area is too small
  // to support contact. ParVector::divide() would leave the weighted gap unchanged at those nodes.
  // The vectors may be valid only in device memory. Synchronize before the host loop, and mark the host copy of the
  // normalized gap as the valid one so later device reads see the result.
  shared::ParVector normalized_gap( quantities.weighted_gap.values.get() );
  const int num_nodes = quantities.weighted_gap.values.size();
  const RealT* weighted_gap = quantities.weighted_gap.values.get().HostRead();
  const RealT* area = quantities.tributary_area.values.get().HostRead();
  RealT* normalized_gap_data = normalized_gap.get().HostWrite();
  mfem::Array<int> open_nodes;
  for ( int node = 0; node < num_nodes; ++node ) {
    if ( weighted_gap[node] > 0.0 ) {
      normalized_gap_data[node] = 0.0;
      open_nodes.Append( node );
    } else if ( std::abs( area[node] ) <= nodal_area_tolerance ) {
      normalized_gap_data[node] = 0.0;
    } else {
      normalized_gap_data[node] = weighted_gap[node] / area[node];
    }
  }
  quantities.weighted_gap.jacobian.eliminateRows( open_nodes );
  ContactFormulationResult result;
  result.gap.reset( normalized_gap.release() );
  result.weighted_gap_jacobian.reset( quantities.weighted_gap.jacobian.release() );
  return result;
}

ContactFormulationResult makeNodalLagrangeMultiplierGapResult( EnergyMortarNodalQuantities& quantities )
{
  ContactFormulationResult result;
  result.gap.reset( quantities.weighted_gap.values.release() );
  result.weighted_gap_jacobian.reset( quantities.weighted_gap.jacobian.release() );
  return result;
}

}  // namespace detail
}  // namespace tribol
#endif
