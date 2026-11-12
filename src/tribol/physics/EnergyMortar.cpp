// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortar.hpp"

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
#include <stdexcept>
#include <utility>

namespace tribol {

QuadraturePointEnergyMortar::QuadraturePointEnergyMortar( std::unique_ptr<ContactEvaluation> evaluation,
                                                          PairListBuilder pair_builder )
    : evaluation_( std::move( evaluation ) ), pair_builder_( std::move( pair_builder ) )
{
  if ( !evaluation_ ) {
    throw std::invalid_argument( "Quadrature-point Energy Mortar requires a contact evaluation." );
  }
}

ContactFormulationResult QuadraturePointEnergyMortar::computeNodalForces( MeshData::Viewer mesh1,
                                                                          MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  return evaluation_->evaluateAndAssemble( pairs, mesh1, mesh2 );
}

NodalEnergyMortar::NodalEnergyMortar( std::unique_ptr<ContactEvaluation> evaluation, PairListBuilder pair_builder )
    : evaluation_( std::move( evaluation ) ), pair_builder_( std::move( pair_builder ) )
{
  if ( !evaluation_ ) {
    throw std::invalid_argument( "Nodal Energy Mortar requires a contact evaluation." );
  }
}

ContactFormulationResult NodalEnergyMortar::computeNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  return evaluation_->evaluateAndAssembleGaps( pairs, mesh1, mesh2 );
}

ContactFormulationResult NodalEnergyMortar::computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  return evaluation_->evaluateAndAssemble( pairs, mesh1, mesh2 );
}

}  // namespace tribol
#endif
