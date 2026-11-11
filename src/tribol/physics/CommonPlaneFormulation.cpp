// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/CommonPlaneFormulation.hpp"

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
#include "tribol/mesh/MfemPairAssembler.hpp"
#include "tribol/physics/PairContribution.hpp"
#include "tribol/physics/PairKernel.hpp"

namespace tribol {

template <typename BaseRule, typename PenaltyStiffness>
ContactFormulationResult CommonPlaneFormulation<BaseRule, PenaltyStiffness>::computeNodalForces(
    MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const
{
  const RealT penalty = penalty_inputs_.prepare( mesh_data_, mesh1, mesh2 );
  const PairFunction function{ quadrature_, penalty, CommonPlaneResidual{ gap_ } };
  const auto pairs = pair_builder_.build( mesh1, mesh2 );
  PairValueJacobianBuffer<PairFunction::value_count, PairFunction::dof_count> contributions( pairs.size(),
                                                                                             pairs.getAllocatorID() );
  evaluatePairValuesAndJacobians( execution_mode_, pairs.view(), mesh1, mesh2, function, contributions.view() );

  checkPairQuadratureResults( execution_mode_, contributions.quadratureResults() );
  constexpr IndexT dof_count = PairFunction::dof_count;
  const IndexT pair_count = pairs.size();
  MfemPairAssembler assembler( mesh_data_, jacobian_data_, mesh1, mesh2, pair_count );
  const auto contribution_view = contributions.view();
  assembler.addPairs<dof_count>(
      execution_mode_, pairs,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row ) {
        return contribution_view.has_active_points[pair] ? contribution_view.value( pair, row ) : 0.0;
      },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return contribution_view.has_active_points[pair] ? contribution_view.jacobian( pair, row, column ) : 0.0;
      } );
  return assembler.assemble();
}

#define TRIBOL_INSTANTIATE_COMMON_PLANE( Rule, Penalty ) template class CommonPlaneFormulation<Rule, Penalty>

TRIBOL_INSTANTIATE_COMMON_PLANE( EdgeQuadratureRuleGenerator<1>, AveragePenaltyStiffness );
TRIBOL_INSTANTIATE_COMMON_PLANE( EdgeQuadratureRuleGenerator<1>, SeriesPenaltyStiffness );
TRIBOL_INSTANTIATE_COMMON_PLANE( EdgeQuadratureRuleGenerator<2>, AveragePenaltyStiffness );
TRIBOL_INSTANTIATE_COMMON_PLANE( EdgeQuadratureRuleGenerator<2>, SeriesPenaltyStiffness );
TRIBOL_INSTANTIATE_COMMON_PLANE( EdgeQuadratureRuleGenerator<3>, AveragePenaltyStiffness );
TRIBOL_INSTANTIATE_COMMON_PLANE( EdgeQuadratureRuleGenerator<3>, SeriesPenaltyStiffness );

#undef TRIBOL_INSTANTIATE_COMMON_PLANE

}  // namespace tribol
#endif
