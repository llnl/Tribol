// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/CommonPlaneFormulation.hpp"
#include "tribol/mesh/MfemPairAssembler.hpp"
#include <cmath>
#include <stdexcept>

namespace tribol {
#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
template <typename BaseQuadratureRule>
CommonPlaneFormulation<BaseQuadratureRule>::CommonPlaneFormulation( MfemMeshData& mesh_data,
                                                                    MfemJacobianData& jacobian_data,
                                                                    std::unique_ptr<CoarseSearch> coarse_search,
                                                                    std::unique_ptr<RefinedSearch> refined_search,
                                                                    RealT residual_gap )
    : mesh_data_( mesh_data ),
      jacobian_data_( jacobian_data ),
      coarse_search_( std::move( coarse_search ) ),
      refined_search_( std::move( refined_search ) ),
      residual_gap_( residual_gap )
{
  if ( !coarse_search_ || !refined_search_ )
    throw std::invalid_argument( "Common Plane requires both search objects." );
}

template <typename BaseQuadratureRule>
ContactFormulationResult CommonPlaneFormulation<BaseQuadratureRule>::computeNodalForces( MeshData::Viewer mesh1,
                                                                                         MeshData::Viewer mesh2 ) const
{
  for ( const auto& mesh : { mesh1, mesh2 } ) {
    if ( mesh.spatialDimension() != CommonPlaneResidualData::dimension || mesh.getElementType() != EDGE ||
         mesh.getFaceOrder() != LINEAR_LAGRANGE ||
         mesh.numberOfNodesPerElement() != CommonPlaneResidualData::nodes_per_element ) {
      throw std::invalid_argument( "CommonPlaneFormulation requires linear two-dimensional edges." );
    }
  }
  const RealT penalty1 = mesh1.getElementData().m_penalty_stiffness * mesh1.getElementData().m_penalty_scale;
  const RealT penalty2 = mesh2.getElementData().m_penalty_stiffness * mesh2.getElementData().m_penalty_scale;
  if ( !std::isfinite( penalty1 ) || !std::isfinite( penalty2 ) || penalty1 <= 0.0 || penalty2 <= 0.0 ) {
    throw std::invalid_argument( "Common Plane requires positive finite constant penalties." );
  }
  const RealT penalty = ComputePenaltyStiffnessPerArea( penalty1, penalty2 );
  const auto candidates = coarse_search_->findCandidatePairs( mesh1, mesh2 );
  const auto pairs = refined_search_->filterCandidatePairs( candidates, mesh1, mesh2 );
  MfemPairAssembler assembler( mesh_data_, jacobian_data_, mesh1, mesh2, pairs.size() );
  constexpr IndexT pair_dofs = CommonPlaneResidualData::coordinate_count;

  for ( const auto& pair : pairs ) {
    MdArray<RealT, LinearEdgeGeometry::vertex_count, LinearEdgeGeometry::spatial_dimension> coordinates1{},
        coordinates2{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates1.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates2.data() );
    const auto qp_data = computeCommonPlanePair<BaseQuadratureRule>( coordinates1.view(), coordinates2.view(),
                                                                     quadrature_rule_, penalty, residual_gap_ );

    if ( !qp_data.has_active_points ) {
      continue;
    }

    assembler.addPair( pair, { qp_data.residual.data(), pair_dofs },
                       { qp_data.jacobian.data(), { pair_dofs, pair_dofs } } );
  }
  return assembler.assemble();
}
template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<1>>;
template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<2>>;
template class CommonPlaneFormulation<EdgeQuadratureRuleGenerator<3>>;
#endif
}  // namespace tribol
