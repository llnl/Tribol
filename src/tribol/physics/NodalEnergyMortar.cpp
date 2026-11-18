// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/NodalEnergyMortar.hpp"

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
namespace tribol {
namespace detail {

ContactFormulationResult makeNodalPenaltyGapResult( EnergyMortarNodalQuantities& quantities )
{
  auto normalized_gap = quantities.weighted_gap.values.divide( quantities.tributary_area.values, nodal_area_tolerance );
  // The vectors may be valid only in device memory. Synchronize before the host loop, and mark the host copy of the
  // normalized gap as the valid one so later device reads see the zeroed opening nodes.
  const int num_nodes = quantities.weighted_gap.values.size();
  const RealT* weighted_gap = quantities.weighted_gap.values.get().HostRead();
  RealT* normalized_gap_data = normalized_gap.get().HostReadWrite();
  mfem::Array<int> open_nodes;
  for ( int node = 0; node < num_nodes; ++node ) {
    if ( weighted_gap[node] > 0.0 ) {
      normalized_gap_data[node] = 0.0;
      open_nodes.Append( node );
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
