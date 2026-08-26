// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTARFIELDDATA_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTARFIELDDATA_HPP_

#include "tribol/config.hpp"

#include "tribol/mesh/MeshData.hpp"
#include "tribol/physics/EnergyMortarData.hpp"
#include "tribol/physics/FieldData.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#ifdef BUILD_REDECOMP
#include "tribol/mesh/MfemData.hpp"
#endif

namespace tribol {

class TribolFieldData : public FieldDataBase {
 public:
  using GapData = TribolGapData;
  using ForceData = TribolForceData;
  using ContactData = TribolContactData;

  TribolFieldData( IndexT mortar_mesh_id, IndexT nonmortar_mesh_id );
  TribolFieldData( MeshData& mortar_mesh, MeshData& nonmortar_mesh );

  void update();
  void update( MeshData& mortar_mesh, MeshData& nonmortar_mesh );

  MeshData& mortarMesh() const;
  MeshData& nonmortarMesh() const;

  void setContactPressure( const std::vector<RealT>& pressure );
  const std::vector<RealT>& contactPressure() const { return dual_field_; }

  IndexT mortarNodeCount() const { return mortarMesh().numberOfNodes(); }
  IndexT nonmortarNodeCount() const { return nonmortarMesh().numberOfNodes(); }

 private:
  bool topologyChanged( const MeshData& mortar_mesh, const MeshData& nonmortar_mesh ) const;
  std::size_t topologyHash( const MeshData& mortar_mesh, const MeshData& nonmortar_mesh ) const;

  IndexT mortar_mesh_id_;
  IndexT nonmortar_mesh_id_;
  MeshData* mortar_mesh_{ nullptr };
  MeshData* nonmortar_mesh_{ nullptr };
  std::size_t topology_hash_{ 0 };
  std::vector<RealT> dual_field_;
};

#ifdef BUILD_REDECOMP

class MfemFieldData : public FieldDataBase {
 public:
  using GapData = MfemGapData;
  using ForceData = MfemForceData;
  using ContactData = MfemContactData;

  MfemFieldData( std::unique_ptr<MfemMeshData> mesh_data, std::unique_ptr<MfemSubmeshData> submesh_data,
                 std::unique_ptr<MfemJacobianData> jacobian_data );

  bool update( RealT effective_binning_proximity, int n_ranks = 0, bool force_new_redecomp = false );

  MfemMeshData& meshData() { return *mesh_data_; }
  const MfemMeshData& meshData() const { return *mesh_data_; }
  MfemSubmeshData& submeshData() { return *submesh_data_; }
  const MfemSubmeshData& submeshData() const { return *submesh_data_; }
  MfemJacobianData& jacobianData() { return *jacobian_data_; }
  const MfemJacobianData& jacobianData() const { return *jacobian_data_; }

  MeshData& mortarMesh() const;
  MeshData& nonmortarMesh() const;
  IndexT mortarNodeCount() const { return mesh_data_->GetNV(); }
  IndexT nonmortarNodeCount() const { return mesh_data_->GetNV(); }

  void rebuildJacobianData();
  void setContactPressure( const mfem::HypreParVector& pressure );
  const mfem::HypreParVector& contactPressure() const { return *dual_field_; }

 private:
  std::unique_ptr<MfemMeshData> mesh_data_;
  std::unique_ptr<MfemSubmeshData> submesh_data_;
  std::unique_ptr<MfemJacobianData> jacobian_data_;
  std::unique_ptr<mfem::HypreParVector> dual_field_;
};

#endif

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_ENERGYMORTARFIELDDATA_HPP_ */
