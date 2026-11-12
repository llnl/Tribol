// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/CommonPlaneFormulation.hpp"

#ifdef BUILD_REDECOMP
#include "tribol/assembly/PairEvaluationAssembly.hpp"

namespace tribol {

template <typename BaseRule, ExecutionMode Exec, typename Evaluation>
ContactFormulationResult CommonPlaneFormulation<BaseRule, Exec, Evaluation>::computeNodalForces(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  return detail::evaluateAndAssembleResidualPairs<Exec>( pairs, mesh1, mesh2, function_, evaluation_ );
}

#define TRIBOL_INSTANTIATE_COMMON_PLANE_EVALUATION( Exec, Evaluation )                     \
  template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<1>, Exec, Evaluation>; \
  template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<2>, Exec, Evaluation>; \
  template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<3>, Exec, Evaluation>
#ifdef TRIBOL_USE_ENZYME
#define TRIBOL_INSTANTIATE_COMMON_PLANE( Exec )                                \
  TRIBOL_INSTANTIATE_COMMON_PLANE_EVALUATION( Exec, ResidualForceEvaluation ); \
  TRIBOL_INSTANTIATE_COMMON_PLANE_EVALUATION( Exec, ResidualForceJacobianEvaluation )
#else
#define TRIBOL_INSTANTIATE_COMMON_PLANE( Exec ) \
  TRIBOL_INSTANTIATE_COMMON_PLANE_EVALUATION( Exec, ResidualForceEvaluation )
#endif

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
#undef TRIBOL_INSTANTIATE_COMMON_PLANE_EVALUATION

}  // namespace tribol
#endif
