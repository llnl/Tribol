// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/CommonPlaneFormulation.hpp"
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
void CommonPlaneFormulation<BaseQuadratureRule>::updateNodalGaps( MeshData::Viewer, MeshData::Viewer )
{
  throw std::logic_error( "CommonPlaneFormulation has no nodal gap field." );
}

template <typename BaseQuadratureRule>
RealT CommonPlaneFormulation<BaseQuadratureRule>::getEnergy() const
{
  throw std::logic_error( "CommonPlaneFormulation defines a residual, not an energy." );
}

template <typename BaseQuadratureRule>
void CommonPlaneFormulation<BaseQuadratureRule>::updateNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 )
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
  constexpr int dimension = CommonPlaneResidualData::dimension;
  constexpr int nodes = CommonPlaneResidualData::nodes_per_element;
  constexpr int pair_size = CommonPlaneResidualData::pair_size;
  constexpr int element_dofs = dimension * nodes;
  constexpr int block_entries = element_dofs * element_dofs;
  const auto& displacement_surface_fes = mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *mesh_data_.GetRedecompResponse().FESpace();
  const auto& element_map1 = mesh_data_.GetElemMap1();
  const auto& element_map2 = mesh_data_.GetElemMap2();

  PackedPairJacobianContribs jacobian11( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                         displacement_redecomp_fes, element_map1, element_map1 );
  PackedPairJacobianContribs jacobian12( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                         displacement_redecomp_fes, element_map1, element_map2 );
  PackedPairJacobianContribs jacobian21( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                         displacement_redecomp_fes, element_map2, element_map1 );
  PackedPairJacobianContribs jacobian22( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                         displacement_redecomp_fes, element_map2, element_map2 );

  jacobian11.reserve( pairs.size(), block_entries );
  jacobian12.reserve( pairs.size(), block_entries );
  jacobian21.reserve( pairs.size(), block_entries );
  jacobian22.reserve( pairs.size(), block_entries );

  mfem::GridFunction redecomp_residual( const_cast<mfem::FiniteElementSpace*>( &displacement_redecomp_fes ) );
  redecomp_residual = 0.0;
  const int scalar_size = redecomp_residual.FESpace()->GetVSize() / redecomp_residual.FESpace()->GetVDim();

  for ( const auto& pair : pairs ) {
    const auto elem1 = static_cast<int>( pair.element_id1_ );
    const auto elem2 = static_cast<int>( pair.element_id2_ );
    MdArray<RealT, LinearEdgeGeometry::endpoint_count, LinearEdgeGeometry::spatial_dimension> coordinates1{},
        coordinates2{};
    mesh1.getFaceCoords( pair.element_id1_, coordinates1.data() );
    mesh2.getFaceCoords( pair.element_id2_, coordinates2.data() );
    const auto qp_data = computeCommonPlanePair<BaseQuadratureRule>( coordinates1.view(), coordinates2.view(),
                                                                     quadrature_rule_, penalty, residual_gap_ );

    if ( !qp_data.has_active_points ) {
      continue;
    }

    const auto connectivity1 = mesh1.getConnectivity()( elem1 );
    const auto connectivity2 = mesh2.getConnectivity()( elem2 );
    for ( int node = 0; node < nodes; ++node ) {
      for ( int component = 0; component < dimension; ++component ) {
        redecomp_residual( connectivity1[node] + component * scalar_size ) +=
            qp_data.residual[node * dimension + component];
        redecomp_residual( connectivity2[node] + component * scalar_size ) +=
            qp_data.residual[( node + nodes ) * dimension + component];
      }
    }

    RealT df_dx_blocks[pair_size][pair_size][block_entries];
    for ( int i = 0; i < pair_size; ++i ) {
      for ( int j = 0; j < pair_size; ++j ) {
        for ( int row = 0; row < element_dofs; ++row ) {
          for ( int column = 0; column < element_dofs; ++column ) {
            const int pair_row = i * element_dofs + ( row % nodes ) * dimension + row / nodes;
            const int pair_column = j * element_dofs + ( column % nodes ) * dimension + column / nodes;
            df_dx_blocks[i][j][row + column * element_dofs] = qp_data.jacobian( pair_row, pair_column );
          }
        }
      }
    }

    jacobian11.append( elem1, elem1, df_dx_blocks[0][0], block_entries );
    jacobian12.append( elem1, elem2, df_dx_blocks[0][1], block_entries );
    jacobian21.append( elem2, elem1, df_dx_blocks[1][0], block_entries );
    jacobian22.append( elem2, elem2, df_dx_blocks[1][1], block_entries );
  }

  auto* parent_fes = mesh_data_.GetParentCoords().ParFESpace();
  residual_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( parent_fes ) );
  residual_.fill( 0.0 );
  mfem::Vector parent_residual( parent_fes->GetVSize() );
  parent_residual = 0.0;
  mesh_data_.GetParentRedecompTransfer().RedecompToParent( redecomp_residual, parent_residual );
  parent_fes->GetProlongationMatrix()->MultTranspose( parent_residual, residual_.get() );

  std::vector<PackedPairJacobianContribs> df_contribs;
  df_contribs.reserve( pair_size * pair_size );
  df_contribs.push_back( std::move( jacobian11 ) );
  df_contribs.push_back( std::move( jacobian12 ) );
  df_contribs.push_back( std::move( jacobian21 ) );
  df_contribs.push_back( std::move( jacobian22 ) );
  jacobian_ = jacobian_data_.GetMfemJacobian( parent_fes, parent_fes, df_contribs );
}
template class CommonPlaneFormulation<EdgeQuadratureRule<1>>;
template class CommonPlaneFormulation<EdgeQuadratureRule<2>>;
template class CommonPlaneFormulation<EdgeQuadratureRule<3>>;
#endif
}  // namespace tribol
