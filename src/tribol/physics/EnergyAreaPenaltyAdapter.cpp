#include "tribol/physics/EnergyAreaPenaltyAdapter.hpp"
#include <utility>
#include <axom/slic/interface/slic_macros.hpp>
#include "tribol/mesh/MfemData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

EnergyAreaPenaltyAdapter::EnergyAreaPenaltyAdapter( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data,
                                                    MfemJacobianData& jac_data, MeshData& mesh1, MeshData& mesh2,
                                                    const EnergyMortarOptions& opts )
    : mesh_data_( mesh_data ),
      submesh_data_( submesh_data ),
      jac_data_( jac_data ),
      mesh1_( &mesh2 ),
      mesh2_( &mesh1 )
{
  if ( mesh1.numberOfNodes() > 0 && mesh2.numberOfNodes() > 0 ) {
    SLIC_ERROR_ROOT_IF( mesh1.spatialDimension() != 2 || mesh2.spatialDimension() != 2,
                        "ENERGY_AREA_PENALTY requires 2D meshes." );
  }

  params_ = opts;
  evaluator_ = std::make_unique<EnergyAreaPenaltyCalculator>( params_ );

  pressure_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  pressure_vec_.fill( 0.0 );

  gap_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  gap_vec_.fill( 0.0 );
}

void EnergyAreaPenaltyAdapter::updateMeshes( MeshData& mesh1, MeshData& mesh2 )
{
  mesh1_ = &mesh2;
  mesh2_ = &mesh1;
}

void EnergyAreaPenaltyAdapter::setInterfacePairs( ArrayT<InterfacePair>&& pairs, int /*check_level*/ )
{
  pairs_ = std::move( pairs );
}

void EnergyAreaPenaltyAdapter::updateIntegrationRule()
{
  // No-op: integration is handled directly in updateNodalGaps/updateNodalForces
}

void EnergyAreaPenaltyAdapter::updateNodalGaps()
{
  SLIC_ERROR_ROOT_IF( mesh1_ == nullptr || mesh2_ == nullptr, "ENERGY_AREA_PENALTY meshes not set." );
  auto mesh1_view = mesh1_->getView();
  auto mesh2_view = mesh2_->getView();

  energy_ = 0.0;

  // Use a temporary grid function on the redecomp displacement space for force scatter
  auto* fes = const_cast<mfem::FiniteElementSpace*>( mesh_data_.GetRedecompResponse().FESpace() );
  mfem::GridFunction redecomp_force( fes );
  redecomp_force = 0.0;
  double* data = redecomp_force.HostWrite();

  for ( const auto& pair : pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    double E = evaluator_->compute_energy( flipped_pair, mesh1_view, mesh2_view );
    energy_ += params_.k * E;

    double dE_dx[8];
    evaluator_->grad_energy( flipped_pair, mesh1_view, mesh2_view, dE_dx );

    auto nm_conn = mesh1_view.getConnectivity()( elem1 );
    auto m_conn = mesh2_view.getConnectivity()( elem2 );

    // Scatter gradient to redecomp grid function (multiply by k)
    // dE_dx layout: [A0x, A0y, A1x, A1y, B0x, B0y, B1x, B1y]
    data[fes->DofToVDof( nm_conn[0], 0 )] += params_.k * dE_dx[0];
    data[fes->DofToVDof( nm_conn[0], 1 )] += params_.k * dE_dx[1];
    data[fes->DofToVDof( nm_conn[1], 0 )] += params_.k * dE_dx[2];
    data[fes->DofToVDof( nm_conn[1], 1 )] += params_.k * dE_dx[3];
    data[fes->DofToVDof( m_conn[0], 0 )]  += params_.k * dE_dx[4];
    data[fes->DofToVDof( m_conn[0], 1 )]  += params_.k * dE_dx[5];
    data[fes->DofToVDof( m_conn[1], 0 )]  += params_.k * dE_dx[6];
    data[fes->DofToVDof( m_conn[1], 1 )]  += params_.k * dE_dx[7];
  }

  // Transfer redecomp force → parent mesh L-vector → true-dof vector
  auto* parent_fes = const_cast<mfem::ParFiniteElementSpace*>( mesh_data_.GetParentCoords().ParFESpace() );
  mfem::Vector parent_response( parent_fes->GetVSize() );
  parent_response = 0.0;
  mesh_data_.GetParentRedecompTransfer().RedecompToParent( redecomp_force, parent_response );

  force_vec_ = shared::ParVector( parent_fes );
  force_vec_.fill( 0.0 );
  parent_fes->GetProlongationMatrix()->MultTranspose( parent_response, force_vec_.get() );
}

void EnergyAreaPenaltyAdapter::updateNodalForces()
{
  SLIC_ERROR_ROOT_IF( mesh1_ == nullptr || mesh2_ == nullptr, "ENERGY_AREA_PENALTY meshes not set." );
  auto mesh1_view = mesh1_->getView();
  auto mesh2_view = mesh2_->getView();

  const bool use_lor = ( mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes = use_lor ? *mesh_data_.GetLORMeshFESpace() : mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *mesh_data_.GetRedecompResponse().FESpace();
  const auto& mortar_elem_map = mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = mesh_data_.GetElemMap2();

  PackedPairJacobianContribs df_nm_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_nm_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs df_m_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, mortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_m_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                     displacement_redecomp_fes, mortar_elem_map, mortar_elem_map );

  df_nm_nm.reserve( pairs_.size(), 16 );
  df_nm_m.reserve( pairs_.size(), 16 );
  df_m_nm.reserve( pairs_.size(), 16 );
  df_m_m.reserve( pairs_.size(), 16 );

  // node_idx remaps from kernel DOF order to element DOF order used by PackedPairJacobianContribs
  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  for ( const auto& pair : pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    double d2E_dx2[64];
    evaluator_->hessian_energy( flipped_pair, mesh1_view, mesh2_view, d2E_dx2 );

    // Remap and scale by k
    double df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            const auto idx = node_idx[l + i * 4] * 8 + node_idx[k + j * 4];
            df_dx_blocks[i][j][l + k * 4] = params_.k * d2E_dx2[idx];
          }
        }
      }
    }

    df_nm_nm.append( elem1, elem1, df_dx_blocks[0][0], 16 );
    df_nm_m.append( elem1, elem2, df_dx_blocks[0][1], 16 );
    df_m_nm.append( elem2, elem1, df_dx_blocks[1][0], 16 );
    df_m_m.append( elem2, elem2, df_dx_blocks[1][1], 16 );
  }

  std::vector<PackedPairJacobianContribs> df_contribs;
  df_contribs.reserve( 4 );
  df_contribs.push_back( std::move( df_nm_nm ) );
  df_contribs.push_back( std::move( df_nm_m ) );
  df_contribs.push_back( std::move( df_m_nm ) );
  df_contribs.push_back( std::move( df_m_m ) );
  df_dx_ = jac_data_.GetMfemJacobian( mesh_data_.GetParentCoords().ParFESpace(),
                                      mesh_data_.GetParentCoords().ParFESpace(), df_contribs );
}

RealT EnergyAreaPenaltyAdapter::computeTimeStep()
{
  return 1.0;
}

std::unique_ptr<mfem::HypreParMatrix> EnergyAreaPenaltyAdapter::getMfemDfDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( df_dx_.release() );
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
