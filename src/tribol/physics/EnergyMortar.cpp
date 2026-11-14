// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortar.hpp"

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
namespace tribol {
namespace detail {

ContactFormulationResult makeNodalPenaltyGapResult( EnergyMortarNodalQuantities& quantities )
{
  auto normalized_gap = quantities.weighted_gap.values.divide( quantities.tributary_area.values, nodal_area_tolerance );
  mfem::Array<int> open_nodes;
  for ( int node = 0; node < quantities.weighted_gap.values.size(); ++node ) {
    if ( quantities.weighted_gap.values[node] > 0.0 ) {
      normalized_gap[node] = 0.0;
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
