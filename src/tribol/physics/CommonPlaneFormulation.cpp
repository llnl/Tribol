// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/CommonPlaneFormulation.hpp"

#ifdef BUILD_REDECOMP
#include <stdexcept>
#include <utility>

namespace tribol {

CommonPlaneFormulation::CommonPlaneFormulation( std::unique_ptr<ContactEvaluation> evaluation,
                                                PairListBuilder pair_builder )
    : evaluation_( std::move( evaluation ) ), pair_builder_( std::move( pair_builder ) )
{
  if ( !evaluation_ ) {
    throw std::invalid_argument( "Common Plane requires a residual evaluation." );
  }
}

ContactFormulationResult CommonPlaneFormulation::computeNodalForces( MeshData::Viewer mesh1,
                                                                     MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  return evaluation_->evaluateAndAssemble( pairs, mesh1, mesh2 );
}

}  // namespace tribol
#endif
