// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/ContactFormulation.hpp"
#ifdef BUILD_REDECOMP
#include <utility>

namespace tribol {

ADScalar ContactFormulationResult::makeADEnergy( const mfem::ParFiniteElementSpace& coordinates ) const
{
  if ( !energy || !force || !df_dx )
    throw std::logic_error( "Differentiable contact energy requires energy, force, and df_dx outputs." );
  return ADScalar::fromDerivatives( coordinates, *energy,
                                    shared::ParVector( std::make_unique<mfem::HypreParVector>( *force ) ),
                                    shared::ParSparseMat( std::make_unique<mfem::HypreParMatrix>( *df_dx ) ) );
}

ADVector ContactFormulationResult::makeADResidual( const mfem::ParFiniteElementSpace& coordinates,
                                                   ADVector::WeightedHessian curvature ) const
{
  if ( !force || !df_dx ) throw std::logic_error( "Differentiable contact residual requires force and df_dx outputs." );
  return ADVector::fromDerivatives(
      coordinates, coordinates, shared::ParVector( std::make_unique<mfem::HypreParVector>( *force ) ),
      shared::ParSparseMat( std::make_unique<mfem::HypreParMatrix>( *df_dx ) ), std::move( curvature ) );
}

}  // namespace tribol
#endif
