// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortarFieldData.hpp"

#include "axom/slic.hpp"

#include <functional>

namespace tribol {

namespace {

void hashCombine( std::size_t& seed, std::size_t value )
{
  seed ^= value + 0x9e3779b9 + ( seed << 6 ) + ( seed >> 2 );
}

}  // namespace

TribolFieldData::TribolFieldData( IndexT mortar_mesh_id, IndexT nonmortar_mesh_id )
    : FieldDataBase( FieldDataBackend::Tribol ),
      mortar_mesh_id_( mortar_mesh_id ),
      nonmortar_mesh_id_( nonmortar_mesh_id )
{
  update();
}

TribolFieldData::TribolFieldData( MeshData& mortar_mesh, MeshData& nonmortar_mesh )
    : FieldDataBase( FieldDataBackend::Tribol ),
      mortar_mesh_id_( mortar_mesh.getView().meshId() ),
      nonmortar_mesh_id_( nonmortar_mesh.getView().meshId() )
{
  update( mortar_mesh, nonmortar_mesh );
}

void TribolFieldData::update()
{
  auto* mortar_mesh = MeshManager::getInstance().findData( mortar_mesh_id_ );
  auto* nonmortar_mesh = MeshManager::getInstance().findData( nonmortar_mesh_id_ );
  SLIC_ERROR_ROOT_IF( mortar_mesh == nullptr || nonmortar_mesh == nullptr,
                      "ENERGY_MORTAR FieldData references an unregistered mesh." );
  update( *mortar_mesh, *nonmortar_mesh );
}

void TribolFieldData::update( MeshData& mortar_mesh, MeshData& nonmortar_mesh )
{
  const bool topology_changed = topologyChanged( mortar_mesh, nonmortar_mesh );
  mortar_mesh_ = &mortar_mesh;
  nonmortar_mesh_ = &nonmortar_mesh;
  topology_hash_ = topologyHash( mortar_mesh, nonmortar_mesh );
  if ( topology_changed ) {
    dual_field_.assign( static_cast<std::size_t>( nonmortar_mesh.numberOfNodes() ), 0.0 );
    markTopologyUpdated();
  } else {
    markFieldsUpdated();
  }
}

MeshData& TribolFieldData::mortarMesh() const
{
  SLIC_ERROR_ROOT_IF( mortar_mesh_ == nullptr, "ENERGY_MORTAR mortar mesh is not available." );
  return *mortar_mesh_;
}

MeshData& TribolFieldData::nonmortarMesh() const
{
  SLIC_ERROR_ROOT_IF( nonmortar_mesh_ == nullptr, "ENERGY_MORTAR nonmortar mesh is not available." );
  return *nonmortar_mesh_;
}

void TribolFieldData::setContactPressure( const std::vector<RealT>& pressure )
{
  SLIC_ERROR_ROOT_IF( pressure.size() != static_cast<std::size_t>( nonmortarNodeCount() ),
                      "Contact pressure size does not match the nonmortar node count." );
  dual_field_ = pressure;
}

bool TribolFieldData::topologyChanged( const MeshData& mortar_mesh, const MeshData& nonmortar_mesh ) const
{
  return mortar_mesh_ == nullptr || nonmortar_mesh_ == nullptr ||
      topology_hash_ != topologyHash( mortar_mesh, nonmortar_mesh );
}

std::size_t TribolFieldData::topologyHash( const MeshData& mortar_mesh, const MeshData& nonmortar_mesh ) const
{
  std::size_t hash = 0;
  const auto add_mesh = [&hash]( const MeshData& mesh ) {
    hashCombine( hash, static_cast<std::size_t>( mesh.numberOfNodes() ) );
    hashCombine( hash, static_cast<std::size_t>( mesh.numberOfElements() ) );
    hashCombine( hash, static_cast<std::size_t>( mesh.numberOfNodesPerElement() ) );
    auto connectivity = const_cast<MeshData&>( mesh ).getView().getConnectivity();
    for ( IndexT element = 0; element < mesh.numberOfElements(); ++element ) {
      for ( IndexT node = 0; node < mesh.numberOfNodesPerElement(); ++node ) {
        hashCombine( hash, static_cast<std::size_t>( connectivity( element, node ) ) );
      }
    }
  };
  add_mesh( mortar_mesh );
  add_mesh( nonmortar_mesh );
  return hash;
}

#ifdef BUILD_REDECOMP

MfemFieldData::MfemFieldData( std::unique_ptr<MfemMeshData> mesh_data,
                              std::unique_ptr<MfemSubmeshData> submesh_data,
                              std::unique_ptr<MfemJacobianData> jacobian_data )
    : FieldDataBase( FieldDataBackend::Mfem ),
      mesh_data_( std::move( mesh_data ) ),
      submesh_data_( std::move( submesh_data ) ),
      jacobian_data_( std::move( jacobian_data ) )
{
  SLIC_ERROR_ROOT_IF( !mesh_data_ || !submesh_data_ || !jacobian_data_,
                      "MFEM ENERGY_MORTAR FieldData requires mesh, submesh, and Jacobian data." );
  dual_field_ = std::make_unique<mfem::HypreParVector>(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_->GetSubmeshFESpace() ) );
  *dual_field_ = 0.0;
}

bool MfemFieldData::update( RealT effective_binning_proximity, int n_ranks, bool force_new_redecomp )
{
  const bool new_redecomp =
      mesh_data_->UpdateMfemMeshData( effective_binning_proximity, n_ranks, force_new_redecomp );
  submesh_data_->UpdateMfemSubmeshData( mesh_data_->GetRedecompMesh(), new_redecomp );
  if ( new_redecomp ) {
    jacobian_data_->UpdateJacobianXfer();
    markTopologyUpdated();
  } else {
    markFieldsUpdated();
  }
  return new_redecomp;
}

MeshData& MfemFieldData::mortarMesh() const
{
  auto* mesh = MeshManager::getInstance().findData( mesh_data_->GetMesh1ID() );
  SLIC_ERROR_ROOT_IF( mesh == nullptr, "MFEM ENERGY_MORTAR mortar mesh has not been synchronized with Tribol." );
  return *mesh;
}

MeshData& MfemFieldData::nonmortarMesh() const
{
  auto* mesh = MeshManager::getInstance().findData( mesh_data_->GetMesh2ID() );
  SLIC_ERROR_ROOT_IF( mesh == nullptr, "MFEM ENERGY_MORTAR nonmortar mesh has not been synchronized with Tribol." );
  return *mesh;
}

void MfemFieldData::rebuildJacobianData()
{
  jacobian_data_ = std::make_unique<MfemJacobianData>( *mesh_data_, *submesh_data_ );
  markTopologyUpdated();
}

void MfemFieldData::setContactPressure( const mfem::HypreParVector& pressure )
{
  SLIC_ERROR_ROOT_IF( pressure.Size() != dual_field_->Size(),
                      "MFEM contact pressure size does not match the dual true-dof space." );
  *dual_field_ = pressure;
}

#endif

}  // namespace tribol
