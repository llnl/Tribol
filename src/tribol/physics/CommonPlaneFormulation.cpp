// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/CommonPlaneFormulation.hpp"

#ifdef BUILD_REDECOMP
#include "tribol/assembly/PairEvaluationAssembly.hpp"

namespace tribol {

template <typename BaseRule, ExecutionMode Exec>
ContactFormulationResult CommonPlaneFormulation<BaseRule, Exec>::computeNodalForces( MeshData::Viewer mesh1,
                                                                                     MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  if ( !jacobian_data_ ) {
    return detail::evaluateAndAssembleResidualValues<Exec>( pairs, mesh1, mesh2, function_, mesh_data_ );
  }
#ifdef TRIBOL_USE_ENZYME
  return detail::evaluateAndAssembleResidualPairs<Exec>( pairs, mesh1, mesh2, function_, mesh_data_, *jacobian_data_ );
#else
  throw std::logic_error( "Common Plane Jacobian calculation requires Enzyme." );
#endif
}

#define TRIBOL_INSTANTIATE_COMMON_PLANE( Exec )                                \
  template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<1>, Exec>; \
  template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<2>, Exec>; \
  template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<3>, Exec>

TRIBOL_INSTANTIATE_COMMON_PLANE( ExecutionMode::Sequential );
#ifdef TRIBOL_USE_OPENMP
TRIBOL_INSTANTIATE_COMMON_PLANE( ExecutionMode::OpenMP );
#endif
#ifdef TRIBOL_USE_CUDA
TRIBOL_INSTANTIATE_COMMON_PLANE( ExecutionMode::Cuda );
#endif
#ifdef TRIBOL_USE_HIP
TRIBOL_INSTANTIATE_COMMON_PLANE( ExecutionMode::Hip );
#endif
#undef TRIBOL_INSTANTIATE_COMMON_PLANE

}  // namespace tribol
#endif
