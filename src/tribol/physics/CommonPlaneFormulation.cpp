// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/CommonPlaneFormulation.hpp"

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
#include "tribol/assembly/PairEvaluationAssembly.hpp"

namespace tribol {

template <typename BaseRule>
ContactFormulationResult CommonPlaneFormulation<BaseRule>::computeNodalForces( MeshData::Viewer mesh1,
                                                                               MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  return detail::evaluateAndAssembleResidualPairs( execution_mode_, pairs, mesh1, mesh2, function_, mesh_data_,
                                                   jacobian_data_ );
}

template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<1>>;
template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<2>>;
template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<3>>;

}  // namespace tribol
#endif
