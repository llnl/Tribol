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
  auto normalized_gap = quantities.weighted_gap.values.divide( quantities.area.values, nodal_area_tolerance );
  mfem::Array<int> open_nodes;
  for ( int node = 0; node < quantities.weighted_gap.values.size(); ++node ) {
    if ( quantities.weighted_gap.values[node] > 0.0 ) {
      normalized_gap[node] = 0.0;
      open_nodes.Append( node );
    }
  }
  quantities.weighted_gap.dvalues_dx.eliminateRows( open_nodes );
  ContactFormulationResult result;
  result.gap.reset( normalized_gap.release() );
  result.dg_dx.reset( quantities.weighted_gap.dvalues_dx.release() );
  return result;
}

ContactFormulationResult makeNodalLagrangeMultiplierGapResult( EnergyMortarNodalQuantities& quantities )
{
  ContactFormulationResult result;
  result.gap.reset( quantities.weighted_gap.values.release() );
  result.dg_dx.reset( quantities.weighted_gap.dvalues_dx.release() );
  return result;
}

}  // namespace detail
}  // namespace tribol
#endif
