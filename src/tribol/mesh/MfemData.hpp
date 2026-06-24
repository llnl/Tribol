// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_MESH_MFEMDATA_HPP_
#define SRC_TRIBOL_MESH_MFEMDATA_HPP_

// Tribol config include
#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP

// C++ includes
#include <set>
#include <utility>
#include <vector>

// MFEM includes
#include "mfem.hpp"

// Axom includes
#include "axom/core.hpp"

// Shared includes
#include "shared/math/ParSparseMat.hpp"

// Redecomp includes
#include "redecomp/redecomp.hpp"

// Tribol includes
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"

namespace tribol {

/**
 * @brief Facilitates transfer of fields to/from parent-linked boundary submesh
 * (a higher-order mesh) to LOR mesh
 *
 * This class simplifies transferring 1) a primal field such as displacement and
 * velocity from a higher-order grid function on a parent-linked boundary
 * submesh to a low-order grid function on a LOR boundary mesh and 2) the
 * energetic conjugate to a primal field (such as nodal force, which is
 * conjugate to nodal displacement) from a low-order grid function on a LOR mesh
 * to a higher-order grid function on a parent-linked boundary submesh.
 *
 * Field data on the LOR mesh are stored internally in this class, and accessed
 * through the GetLORGridFn() method.
 */
class SubmeshLORTransfer {
 public:
  /**
   * @brief Construct a new SubmeshLORTransfer object
   *
   * @param submesh_fes Higher order finite element space on the parent-linked
   * boundary submesh
   * @param lor_mesh LOR mesh
   * @param use_ea Whether to use device-friendly element assembly for the transfer (default: false)
   */
  SubmeshLORTransfer( mfem::ParFiniteElementSpace& submesh_fes, mfem::ParMesh& lor_mesh, bool use_ea = false );

  /**
   * @brief Transfers data from a higher-order grid function on a parent-linked
   * submesh
   *
   * Data is transferred to the low-order grid function on the LOR mesh that can
   * be accessed using GetLORGridFn().
   *
   * @param submesh_src Source higher-order grid function on the parent-linked
   * boundary submesh
   */
  void TransferToLORGridFn( const mfem::ParGridFunction& submesh_src );

  /**
   * @brief Transfers data to a higher-order vector on a parent-linked boundary submesh
   *
   * Data must be stored in the low-order vector on the LOR mesh accessed using GetLORVector().
   *
   * @param submesh_dst Destination higher-order vector on the parent-linked boundary submesh
   */
  void TransferFromLORVector( mfem::Vector& submesh_dst ) const;

  /**
   * @brief Transfer grid function on parent-linked boundary submesh to grid
   * function on LOR mesh
   *
   * @param [in] submesh_src Grid function on parent-linked boundary submesh
   * @param [out] lor_dst Zero-valued grid function on LOR mesh
   */
  void SubmeshToLOR( const mfem::ParGridFunction& submesh_src, mfem::ParGridFunction& lor_dst );

  /**
   * @brief Access the local low-order grid function on the LOR mesh
   *
   * @return mfem::ParGridFunction&
   */
  mfem::ParGridFunction& GetLORGridFn() { return *lor_gridfn_; }

  /**
   * @brief Access the local low-order grid function on the LOR mesh
   *
   * @return mfem::ParGridFunction&
   */
  const mfem::ParGridFunction& GetLORGridFn() const { return *lor_gridfn_; }

  /**
   * @brief Access the local low-order vector on the LOR mesh
   *
   * @return mfem::Vector&
   */
  mfem::Vector& GetLORVector() { return *lor_gridfn_; }

  /**
   * @brief Access the local low-order vector on the LOR mesh
   *
   * @return const mfem::Vector&
   */
  const mfem::Vector& GetLORVector() const { return *lor_gridfn_; }

 private:
  /**
   * @brief Create low-order grid function on the LOR mesh
   *
   * @param lor_mesh LOR mesh
   * @param lor_fec Finite element collection to apply to grid function
   * @param vdim Vector dimension of the grid function
   * @return mfem::ParGridFunction on lor_mesh, with lor_fec and vdim specified
   */
  static std::unique_ptr<mfem::ParGridFunction> CreateLORGridFunction(
      mfem::ParMesh& lor_mesh, std::unique_ptr<mfem::FiniteElementCollection> lor_fec, int vdim );

  /**
   * @brief Local low-order grid function on the LOR mesh
   */
  std::unique_ptr<mfem::ParGridFunction> lor_gridfn_;

  /**
   * @brief Low-order refined <-> higher-order coarse transfer object
   */
  mutable mfem::L2ProjectionGridTransfer lor_xfer_;
};

/**
 * @brief Facilitates transferring variables from the submesh to redecomp mesh
 * levels
 *
 * This class simplifies transferring field variables from/to a grid function on
 * an mfem::ParSubMesh to/from a grid function on a redecomp::RedecompMesh.  If
 * transferring to a low-order grid function on a LOR mesh is also required,
 * this class will perform that transfer as well.
 *
 * The hierarchy of transfer is: submesh <--> LOR mesh (optional) <--> redecomp
 * mesh
 *
 * @note This is used to transfer variables defined at the mfem::ParSubMesh
 * level (e.g. pressure and gap).
 */
class SubmeshRedecompTransfer {
 public:
  /**
   * @brief Construct a new SubmeshRedecompTransfer object
   *
   * @param submesh_fes Finite element space on the parent-linked boundary
   * submesh
   * @param submesh_lor_xfer Submesh to LOR grid function transfer object (if
   * using LOR; nullptr otherwise)
   * @param redecomp_mesh RedecompMesh of the redecomposed contact surface mesh
   */
  SubmeshRedecompTransfer( mfem::ParFiniteElementSpace& submesh_fes, SubmeshLORTransfer* submesh_lor_xfer,
                           redecomp::RedecompMesh& redecomp_mesh );

  /**
   * @brief Transfer grid function on parent-linked boundary submesh to grid
   * function on redecomp mesh
   *
   * @param [in] submesh_src Grid function on parent-linked boundary submesh
   * @param [out] redecomp_dst Zero-valued grid function on redecomp mesh
   */
  void SubmeshToRedecomp( const mfem::ParGridFunction& submesh_src, mfem::GridFunction& redecomp_dst ) const;

  /**
   * @brief Transfer grid function on redecomp mesh to vector on parent-linked boundary submesh
   *
   * @note The redecomp_src GridFunction is expected to have values at shared DOFs equal.  The submesh_dst will need
   * parallel summation for shared DOF values to be equal.  This arrangement of DOF values is in line with dual vectors
   * in MFEM.
   *
   * @param redecomp_src Grid function on redecomp mesh
   * @param submesh_dst Zero-valued vector on parent-linked boundary submesh
   */
  void RedecompToSubmesh( const mfem::GridFunction& redecomp_src, mfem::Vector& submesh_dst ) const;

  /**
   * @brief Get the parent-linked boundary submesh associated with the
   * SubmeshRedecompTransfer object
   *
   * @return const mfem::ParSubMesh&
   */
  const mfem::ParSubMesh& GetSubmesh() const
  {
    return static_cast<const mfem::ParSubMesh&>( *submesh_fes_.GetParMesh() );
  }

  /**
   * @brief Returns finite element space on the redecomp mesh associated with
   * this transfer object
   *
   * @return mfem::FiniteElementSpace&
   */
  mfem::FiniteElementSpace& GetRedecompFESpace() { return *redecomp_fes_; }

 private:
  /**
   * @brief Create a finite element space on the redecomp mesh
   *
   * @param redecomp_mesh RedecompMesh of the redecomposed contact surface mesh
   * @param submesh_fes Finite element space on the parent-linked boundary
   * submesh
   * @return std::unique_ptr<mfem::FiniteElementSpace>
   */
  static std::unique_ptr<mfem::FiniteElementSpace> CreateRedecompFESpace( redecomp::RedecompMesh& redecomp_mesh,
                                                                          mfem::ParFiniteElementSpace& submesh_fes );

  /**
   * @brief Finite element space on the parent-linked boundary submesh
   */
  mfem::ParFiniteElementSpace& submesh_fes_;

  /**
   * @brief Finite element space on the redecomp mesh
   */
  mutable std::unique_ptr<mfem::FiniteElementSpace> redecomp_fes_;

  /**
   * @brief Transfer object between low-order and higher-order grid functions
   */
  SubmeshLORTransfer* submesh_lor_xfer_;

  /**
   * @brief Transfer object between parent-linked boundary submesh and redecomp
   * mesh
   */
  const redecomp::RedecompTransfer redecomp_xfer_;
};

/**
 * @brief Facilitates transferring variables from the parent mesh to the
 * redecomp mesh levels
 *
 * This class simplifies transferring field variables from/to a grid function on
 * a parent mfem::ParMesh to/from a grid function on a redecomp::RedecompMesh.
 * If transferring to a low-order grid function on a LOR mesh is also required,
 * this class will perform that transfer as well.
 *
 * The hierarchy of transfer is:
 * parent mesh <--> submesh <--> LOR mesh (optional) <--> redecomp mesh
 *                 \---------------------------------------------------/
 *                handled through SubmeshRedecompTransfer member variable
 */
class ParentRedecompTransfer {
 public:
  /**
   * @brief Construct a new ParentRedecompTransfer object
   *
   * @param parent_fes Finite element space on the parent mesh
   * @param submesh_gridfn Grid function on the parent-linked boundary submesh
   * used to temporarily store variables being transferred
   * @param submesh_lor_xfer Submesh to LOR grid function transfer object (if
   * using LOR; nullptr otherwise)
   * @param redecomp_mesh RedecompMesh of the redecomposed contact surface mesh
   */
  ParentRedecompTransfer( const mfem::ParFiniteElementSpace& parent_fes, mfem::ParGridFunction& submesh_gridfn,
                          SubmeshLORTransfer* submesh_lor_xfer, redecomp::RedecompMesh& redecomp_mesh );

  /**
   * @brief Transfer grid function on parent mesh to grid function on redecomp
   * mesh
   *
   * @param [in] parent_src Grid function on parent mesh
   * @param [out] redecomp_dst Zero-valued grid function on redecomp mesh
   */
  void ParentToRedecomp( const mfem::ParGridFunction& parent_src, mfem::GridFunction& redecomp_dst ) const;

  /**
   * @brief Transfer grid function on redecomp mesh to vector on parent mesh
   *
   * @note The redecomp_src GridFunction is expected to have values at shared DOFs equal.  The parallel_dst will need
   * parallel summation for shared DOF values to be equal.  This arrangement of DOF values is in line with dual vectors
   * in MFEM.
   *
   * @param [in] redecomp_src Grid function on RedecompMesh
   * @param [out] parent_dst Zero-valued vector on parent mesh
   */
  void RedecompToParent( const mfem::GridFunction& redecomp_src, mfem::Vector& parent_dst ) const;

  /**
   * @brief Get the parent-linked boundary submesh finite element space
   * associated with this transfer object
   *
   * @return const mfem::ParFiniteElementSpace&
   */
  const mfem::ParFiniteElementSpace& GetSubmeshFESpace() const { return *submesh_gridfn_.ParFESpace(); }

  /**
   * @brief Returns finite element space on the redecomp mesh associated with
   * this transfer object
   *
   * @return mfem::FiniteElementSpace&
   */
  mfem::FiniteElementSpace& GetRedecompFESpace() { return submesh_redecomp_xfer_.GetRedecompFESpace(); }

 private:
  /**
   * @brief Finite element space on the parent mesh
   */
  mutable mfem::ParFiniteElementSpace parent_fes_;

  /**
   * @brief Grid function on the parent-linked boundary submesh
   */
  mfem::ParGridFunction& submesh_gridfn_;

  /**
   * @brief Object to transfer variables to/from the parent-linked boundary
   * submesh level from/to the redecomp level
   */
  SubmeshRedecompTransfer submesh_redecomp_xfer_;
};

/**
 * @brief Vector field variable that lives on the parent mesh
 *
 * This class stores a vector field variable defined on the parent mesh and
 * handles transferring field data to/from different mesh representations used
 * by Tribol.
 *
 * @note Example parent vector fields include displacement and velocity.
 */
class ParentField {
 public:
  /**
   * @brief Construct a new ParentField object
   *
   * @param parent Grid function on the parent mesh
   */
  ParentField( const mfem::ParGridFunction& parent_gridfn );

  /**
   * @brief Set a new grid function on the parent mesh
   *
   * @param parent Grid function on the parent mesh
   */
  void SetParentGridFn( const mfem::ParGridFunction& parent_gridfn );

  /**
   * @brief Set a new transfer object when the redecomp mesh has been updated
   *
   * @param xfer Updated parent mesh to redecomp mesh transfer object
   * @param use_device If true, use device memory for the redecomp grid function
   */
  void UpdateField( ParentRedecompTransfer& parent_redecomp_xfer, bool use_device );

  /**
   * @brief Get the parent grid function
   *
   * @return const mfem::ParGridFunction&
   */
  const mfem::ParGridFunction& GetParentGridFn() const { return parent_gridfn_; }

  /**
   * @brief Get the redecomp mesh grid function
   *
   * @return mfem::GridFunction&
   */
  mfem::GridFunction& GetRedecompGridFn() { return GetUpdateData().redecomp_gridfn_; }

  /**
   * @brief Get the redecomp mesh grid function
   *
   * @return const mfem::GridFunction&
   */
  const mfem::GridFunction& GetRedecompGridFn() const { return GetUpdateData().redecomp_gridfn_; }

  /**
   * @brief Get pointers to component arrays of the redecomp mesh grid function
   *
   * @return std::vector<const RealT*> of length 3
   *
   * @note The third entry is nullptr in two dimensions
   */
  std::vector<const RealT*> GetRedecompFieldPtrs() const;

  /**
   * @brief Get pointers to component arrays of the redecomp mesh grid function
   *
   * @param redecomp_gridfn Redecomp mesh grid function
   * @return std::vector<RealT*> of length 3
   *
   * @note The third entry is nullptr in two dimensions
   */
  static std::vector<RealT*> GetRedecompFieldPtrs( mfem::GridFunction& redecomp_gridfn );

 private:
  /**
   * @brief Creates and stores data that changes when the redecomp mesh is
   * updated
   */
  struct UpdateData {
    /**
     * @brief Construct a new UpdateData object
     *
     * @param parent_redecomp_xfer Parent to redecomp field transfer object
     * @param parent_gridfn Grid function on the original, parent mesh
     * @param use_device If true, use device memory for the redecomp grid function
     */
    UpdateData( ParentRedecompTransfer& parent_redecomp_xfer, const mfem::ParGridFunction& parent_gridfn,
                bool use_device );

    /**
     * @brief Parent to redecomp field transfer object
     */
    const ParentRedecompTransfer& parent_redecomp_xfer_;

    /**
     * @brief Grid function values on the redecomp mesh
     */
    mfem::GridFunction redecomp_gridfn_;
  };

  /**
   * @brief Get the UpdateData object
   *
   * @return UpdateData&
   */
  UpdateData& GetUpdateData();

  /**
   * @brief Get the UpdateData object
   *
   * @return const UpdateData&
   */
  const UpdateData& GetUpdateData() const;

  /**
   * @brief Grid function on the parent mesh
   *
   * @note Stored as a reference wrapper so the reference can be updated
   */
  std::reference_wrapper<const mfem::ParGridFunction> parent_gridfn_;

  /**
   * @brief UpdateData object created upon call to UpdateField()
   */
  std::unique_ptr<UpdateData> update_data_;
};

/**
 * @brief Stores a pressure field variable that lives on the parent-linked
 * boundary submesh
 *
 * This class handles transferring pressure field data to/from representations
 * used by MFEM from/to representations used by Tribol.
 */
class PressureField {
 public:
  /**
   * @brief Construct a new PressureField object
   *
   * @param submesh_gridfn Grid function on the parent-linked boundary submesh
   */
  PressureField( const mfem::ParGridFunction& submesh_gridfn );

  /**
   * @brief Sets a new grid function on the parent-linked boundary submesh
   *
   * @param submesh_gridfn Grid function on the parent-linked boundary submesh
   */
  void SetSubmeshField( const mfem::ParGridFunction& submesh_gridfn );

  /**
   * @brief Sets a new transfer object when the redecomp mesh has been updated
   *
   * @param xfer Updated submesh to redecomp transfer object
   */
  void UpdateField( SubmeshRedecompTransfer& submesh_redecomp_xfer );

  /**
   * @brief Get the parent-linked boundary submesh grid function
   *
   * @return const mfem::ParGridFunction&
   */
  const mfem::ParGridFunction& GetSubmeshGridFn() const { return submesh_gridfn_; }

  /**
   * @brief Get the redecomp mesh grid function
   *
   * @return mfem::GridFunction&
   */
  mfem::GridFunction& GetRedecompGridFn() { return GetUpdateData().redecomp_gridfn_; }

  /**
   * @brief Get the redecomp mesh grid function
   *
   * @return const mfem::GridFunction&
   */
  const mfem::GridFunction& GetRedecompGridFn() const { return GetUpdateData().redecomp_gridfn_; }

  /**
   * @brief Get pointers to component arrays of the redecomp mesh grid function
   *
   * @return std::vector<const RealT*> of length 3
   *
   * @note Unused entries are nullptr.  Only the first entry is used with
   * frictionless contact.
   */
  std::vector<const RealT*> GetRedecompFieldPtrs() const;

  /**
   * @brief Get pointers to component arrays of a redecomp mesh grid function
   *
   * @param redecomp_gridfn Redecomp mesh grid function
   * @return std::vector<RealT*> of length 3
   *
   * @note Unused entries are nullptr.  Only the first entry is used with
   * frictionless contact.
   */
  static std::vector<RealT*> GetRedecompFieldPtrs( mfem::GridFunction& redecomp_gridfn );

 private:
  /**
   * @brief Creates and stores data that changes when the redecomp mesh is
   * updated
   */
  struct UpdateData {
    /**
     * @brief Construct a new UpdateData object
     *
     * @param submesh_redecomp_xfer Submesh to redecomp field transfer object
     * @param submesh_gridfn Grid function on the parent-linked boundary submesh
     */
    UpdateData( SubmeshRedecompTransfer& submesh_redecomp_xfer, const mfem::ParGridFunction& submesh_gridfn );

    /**
     * @brief Submesh to redecomp field transfer object
     */
    const SubmeshRedecompTransfer& submesh_redecomp_xfer_;

    /**
     * @brief Grid function values on the redecomp mesh
     */
    mfem::GridFunction redecomp_gridfn_;
  };

  /**
   * @brief Get the UpdateData object
   *
   * @return UpdateData&
   */
  UpdateData& GetUpdateData();

  /**
   * @brief Get the UpdateData object
   *
   * @return const UpdateData&
   */
  const UpdateData& GetUpdateData() const;

  /**
   * @brief Grid function on the parent-linked boundary submesh
   *
   * @note Stored as a reference wrapper so the reference can be updated
   */
  std::reference_wrapper<const mfem::ParGridFunction> submesh_gridfn_;

  /**
   * @brief UpdateData object created upon call to UpdateField()
   */
  std::unique_ptr<UpdateData> update_data_;
};

/**
 * @brief Stores MFEM and transfer data associated with parent vector fields
 * (displacement and velocity)
 */
class MfemMeshData {
 public:
  /**
   * @brief Construct a new MfemMeshData object
   *
   * @param mesh_id_1 Integer identifier for first Tribol registered mesh
   * @param mesh_id_2 Integer identifier for second Tribol registered mesh
   * @param parent_mesh Parent mesh, i.e. volume mesh of parent domain
   * @param current_coords Grid function on parent mesh holding current
   * coordinates
   * @param attributes_1 Mesh boundary attributes identifying surface elements
   * in the first Tribol registered mesh
   * @param attributes_2 Mesh boundary attributes identifying surface elements
   * in the second Tribol registered mesh
   */
  MfemMeshData( IndexT mesh_id_1, IndexT mesh_id_2, const mfem::ParMesh& parent_mesh,
                const mfem::ParGridFunction& current_coords, std::set<int>&& attributes_1, std::set<int>&& attributes_2,
                ExecutionMode exec_mode, MemorySpace mem_space );

  /**
   * @brief Get coordinate grid function on the parent mesh
   *
   * @return const mfem::ParGridFunction&
   */
  const mfem::ParGridFunction& GetParentCoords() const { return coords_.GetParentGridFn(); }

  /**
   * @brief Sets a new coordinate grid function on the parent mesh
   *
   * @param current_coords Coordinate grid function on the parent mesh
   */
  void SetParentCoords( const mfem::ParGridFunction& current_coords );

  /**
   * @brief Sets a new reference coordinate grid function on the parent mesh
   *
   * @param reference_coords Reference coordinate grid function on the parent mesh
   */
  void SetParentReferenceCoords( const mfem::ParGridFunction& reference_coords );

  /**
   * @brief Determine if a reference coords grid function has been set
   *
   * @return true: Reference coords grid function has been set
   * @return false: Reference coords grid function has not been set
   */
  bool HasReferenceCoords() const { return reference_coords_ != nullptr; }

  /**
   * @brief Get pointers to component arrays of the reference coords on the RedecompMesh
   *
   * @return std::vector<const RealT*> of length 3
   *
   * @note The third entry is nullptr in two dimensions
   */
  std::vector<const RealT*> GetRedecompReferenceCoordsPtrs() const { return reference_coords_->GetRedecompFieldPtrs(); }

  /**
   * @brief Build a new redecomp mesh and update grid functions on the redecomp mesh
   *
   * @param binning_proximity_scale Element length multiplier for coarse binning and proximity detection inclusion. This
   * is needed to size the ghost element layer in the redecomp mesh.
   * @param n_ranks Number of ranks in the parallel decomposition
   * @param force_new_redecomp If true, construct a new RedecompMesh even if displacement threshold is not met (default
   * = false)
   * @return True if a new RedecompMesh is created by this method
   *
   * @note This method should be called after the coordinate grid function is updated.
   */
  bool UpdateMfemMeshData( RealT binning_proximity_scale, int n_ranks, bool force_new_redecomp = false );

  /**
   * @brief Get the integer identifier for the first Tribol registered mesh
   *
   * @return IndexT
   */
  IndexT GetMesh1ID() const { return mesh_id_1_; }

  /**
   * @brief Get the integer identifier for the second Tribol registered mesh
   *
   * @return IndexT
   */
  IndexT GetMesh2ID() const { return mesh_id_2_; }

  /**
   * @brief Get the number of elements in the first Tribol registered mesh
   *
   * @return int
   */
  int GetMesh1NE() const { return GetUpdateData().conn_1_.size() / GetUpdateData().num_verts_per_elem_; }

  /**
   * @brief Get the number of elements in the second Tribol registered mesh
   *
   * @return int
   */
  int GetMesh2NE() const { return GetUpdateData().conn_2_.size() / GetUpdateData().num_verts_per_elem_; }

  /**
   * @brief Get the total number of vertices in both Tribol registered meshes
   *
   * @return int
   */
  int GetNV() const { return GetUpdateData().redecomp_mesh_.GetNV(); }

  /**
   * @brief Get the connectivity for the first Tribol registered mesh
   *
   * @return const IndexType*
   */
  const IndexT* GetMesh1Conn() const { return GetUpdateData().conn_1_.data(); }

  /**
   * @brief Get the connectivity for the second Tribol registered mesh
   *
   * @return const IndexType*
   */
  const IndexT* GetMesh2Conn() const { return GetUpdateData().conn_2_.data(); }

  /**
   * @brief Get the element type for both Tribol registered meshes
   *
   * @return InterfaceElementType
   */
  InterfaceElementType GetElemType() const { return GetUpdateData().elem_type_; }

  /**
   * @brief Get pointers to component arrays of the coordinates on the
   * redecomp mesh
   *
   * @return std::vector<const RealT*> of length 3
   *
   * @note The third entry is nullptr in two dimensions
   */
  std::vector<const RealT*> GetRedecompCoordsPtrs() const { return coords_.GetRedecompFieldPtrs(); }

  /**
   * @brief Get pointers to component arrays of the nodal response on the
   * redecomp mesh
   *
   * @return std::vector<RealT*> of length 3
   *
   * @note The third entry is nullptr in two dimensions
   */
  std::vector<RealT*> GetRedecompResponsePtrs() { return ParentField::GetRedecompFieldPtrs( *redecomp_response_ ); }

  /**
   * @brief Get the nodal response grid function on the redecomp mesh
   *
   * @return const mfem::GridFunction&
   */
  const mfem::GridFunction& GetRedecompResponse() const { return *redecomp_response_; }

  /**
   * @brief Get the nodal response vector on the parent mesh
   *
   * @note This is stored as an MFEM dual vector, meaning the shared DOFs are expected to be summed over all ranks to
   * obtain their value.
   *
   * @param [out] r Pre-allocated, initialized mfem::Vector to which response vector is added
   */
  void GetParentResponse( mfem::Vector& r ) const;

  /**
   * @brief Get the parent to redecomp grid function transfer object
   *
   * @return const ParentRedecompTransfer&
   */
  const ParentRedecompTransfer& GetParentRedecompTransfer() const { return GetUpdateData().vector_xfer_; }

  /**
   * @brief Add/replace the parent velocity grid function
   *
   * @param velocity Velocity grid function on the parent mesh
   */
  void SetParentVelocity( const mfem::ParGridFunction& velocity );

  /**
   * @brief Determine if a velocity grid function has been set
   *
   * @return true: Velocity grid function has been set
   * @return false: Velocity grid function has not been set
   */
  bool HasVelocity() const { return velocity_ != nullptr; }

  /**
   * @brief Get pointers to component arrays of the velocity on the RedecompMesh
   *
   * @return std::vector<const RealT*> of length 3
   *
   * @note The third entry is nullptr in two dimensions
   */
  std::vector<const RealT*> GetRedecompVelocityPtrs() const { return velocity_->GetRedecompFieldPtrs(); }

  /**
   * @brief Clears all kinematic and rate penalty data
   */
  void ClearAllPenaltyData();

  /**
   * @brief Clears rate penalty data
   */
  void ClearRatePenaltyData();

  /**
   * @brief Sets the kinematic constant penalty parameter for the first registered Tribol mesh
   *
   * @param penalty Penalty value for the first registered Tribol mesh
   */
  void SetMesh1KinematicConstantPenalty( RealT penalty )
  {
    kinematic_constant_penalty_1_ = std::make_unique<RealT>( penalty );
  }

  /**
   * @brief Sets the kinematic constant penalty parameter for the second registered Tribol mesh
   *
   * @param penalty Penalty value for the second registered Tribol mesh
   */
  void SetMesh2KinematicConstantPenalty( RealT penalty )
  {
    kinematic_constant_penalty_2_ = std::make_unique<RealT>( penalty );
  }

  /**
   * @brief Get the kinematic constant penalty parameter for the first registered Tribol mesh
   *
   * @return const RealT*
   */
  const RealT* GetMesh1KinematicConstantPenalty() const { return kinematic_constant_penalty_1_.get(); }

  /**
   * @brief Get the kinematic constant penalty parameter for the second registered Tribol mesh
   *
   * @return const RealT*
   */
  const RealT* GetMesh2KinematicConstantPenalty() const { return kinematic_constant_penalty_2_.get(); }

  /**
   * @brief Sets the kinematic penalty scale for the first registered Tribol mesh
   *
   * @param scale Penalty scale value for the first registered Tribol mesh
   */
  void SetMesh1KinematicPenaltyScale( RealT scale ) { kinematic_penalty_scale_1_ = std::make_unique<RealT>( scale ); }

  /**
   * @brief Sets the kinematic penalty scale for the second registered Tribol mesh
   *
   * @param scale Penalty scale value for the second registered Tribol mesh
   */
  void SetMesh2KinematicPenaltyScale( RealT scale ) { kinematic_penalty_scale_2_ = std::make_unique<RealT>( scale ); }

  /**
   * @brief Get the kinematic penalty scale for the first registered Tribol mesh
   *
   * @return const RealT*
   */
  const RealT* GetMesh1KinematicPenaltyScale() const { return kinematic_penalty_scale_1_.get(); }

  /**
   * @brief Get the kinematic penalty scale for the second registered Tribol mesh
   *
   * @return const RealT*
   */
  const RealT* GetMesh2KinematicPenaltyScale() const { return kinematic_penalty_scale_2_.get(); }

  /**
   * @brief Sets the rate constant penalty for the first registered Tribol mesh
   *
   * @param penalty Rate constant penalty value for the first registered Tribol mesh
   */
  void SetMesh1RateConstantPenalty( RealT penalty ) { rate_constant_penalty_1_ = std::make_unique<RealT>( penalty ); }

  /**
   * @brief Sets the rate constant penalty for the second registered Tribol mesh
   *
   * @param penalty Rate penalty value for the second registered Tribol mesh
   */
  void SetMesh2RateConstantPenalty( RealT penalty ) { rate_constant_penalty_2_ = std::make_unique<RealT>( penalty ); }

  /**
   * @brief Get the rate constant penalty for the first registered Tribol mesh
   *
   * @return const RealT*
   */
  const RealT* GetMesh1RateConstantPenalty() const { return rate_constant_penalty_1_.get(); }

  /**
   * @brief Get the rate constant penalty for the second registered Tribol mesh
   *
   * @return const RealT*
   */
  const RealT* GetMesh2RateConstantPenalty() const { return rate_constant_penalty_2_.get(); }

  /**
   * @brief Sets the rate penalty as a ratio of the kinematic penalty for the
   * first registered Tribol mesh
   *
   * @param ratio Rate ratio for the first registered Tribol mesh
   */
  void SetMesh1RatePercentPenalty( RealT ratio ) { rate_percent_ratio_1_ = std::make_unique<RealT>( ratio ); }

  /**
   * @brief Sets the rate penalty as a ratio of the kinematic penalty for the
   * second registered Tribol mesh
   *
   * @param ratio Rate ratio for the second registered Tribol mesh
   */
  void SetMesh2RatePercentPenalty( RealT ratio ) { rate_percent_ratio_2_ = std::make_unique<RealT>( ratio ); }

  /**
   * @brief Get the rate penalty ratio for the first registered Tribol mesh
   *
   * @return const RealT*
   */
  const RealT* GetMesh1RatePercentPenalty() const { return rate_percent_ratio_1_.get(); }

  /**
   * @brief Get the rate penalty ratio for the second registered Tribol mesh
   *
   * @return const RealT*
   */
  const RealT* GetMesh2RatePercentPenalty() const { return rate_percent_ratio_2_.get(); }

  /**
   * @brief Sets the tangential viscous damping coefficient on mesh 1
   *
   * @param coeff coefficient for viscous damping
   */
  void SetMesh1ViscousDampingCoeff( RealT coeff ) { viscous_damping_coeff_1_ = std::make_unique<RealT>( coeff ); }

  /**
   * @brief Sets the tangential viscous damping coefficient on mesh 2
   *
   * @param coeff coefficient for viscous damping
   */
  void SetMesh2ViscousDampingCoeff( RealT coeff ) { viscous_damping_coeff_2_ = std::make_unique<RealT>( coeff ); }

  /**
   * @brief Get the tangential viscous damping coefficient for the first registered Tribol mesh
   *
   * @return const RealT*
   */
  const RealT* GetMesh1ViscousDampingCoeff() const { return viscous_damping_coeff_1_.get(); }

  /**
   * @brief Get the tangential viscous damping coefficient for the second registered Tribol mesh
   *
   * @return const RealT*
   */
  const RealT* GetMesh2ViscousDampingCoeff() const { return viscous_damping_coeff_2_.get(); }

  /**
   * @brief Get a pointer to the element thickness array for the first Tribol
   * registered mesh
   *
   * @return const RealT*
   */
  const RealT* GetRedecompElemThickness1() const { return tribol_elem_thickness_1_->data(); }

  /**
   * @brief Get a pointer to the element thickness array for the second Tribol
   * registered mesh
   *
   * @return const RealT*
   */
  const RealT* GetRedecompElemThickness2() const { return tribol_elem_thickness_2_->data(); }

  /**
   * @brief Get a pointer to the material modulus array for the first Tribol
   * registered mesh
   *
   * @return const RealT*
   */
  const RealT* GetRedecompMaterialModulus1() const { return tribol_material_modulus_1_->data(); }

  /**
   * @brief Get a pointer to the material modulus array for the second Tribol
   * registered mesh
   *
   * @return const RealT*
   */
  const RealT* GetRedecompMaterialModulus2() const { return tribol_material_modulus_2_->data(); }

  /**
   * @brief Get the map from Tribol registered mesh 1 element indices to
   * redecomp mesh element indices
   *
   * @return const ArrayT<int>&
   */
  const Array1D<int>& GetElemMap1() const { return GetUpdateData().elem_map_1_; }

  /**
   * @brief Get the map from Tribol registered mesh 2 element indices to
   * redecomp mesh element indices
   *
   * @return const ArrayT<int>&
   */
  const Array1D<int>& GetElemMap2() const { return GetUpdateData().elem_map_2_; }

  /**
   * @brief Get the parent-linked boundary submesh containing both contact
   * surfaces
   *
   * @return mfem::ParSubMesh&
   */
  mfem::ParSubMesh& GetSubmesh() { return submesh_; }

  /**
   * @brief Get the parent-linked boundary submesh containing both contact
   * surfaces
   *
   * @return const mfem::ParSubMesh&
   */
  const mfem::ParSubMesh& GetSubmesh() const { return submesh_; }

  /**
   * @brief Get the LOR mesh containing both contact surfaces
   *
   * @return mfem::ParMesh*
   *
   * @note nullptr if no refined mesh exists (polynomial order of parent is 1)
   */
  mfem::ParMesh* GetLORMesh() { return lor_mesh_.get(); }

  /**
   * @brief Get the LOR mesh containing both contact surfaces
   *
   * @return const mfem::ParMesh*
   *
   * @note nullptr if no refined mesh exists (polynomial order of parent is 1)
   */
  const mfem::ParMesh* GetLORMesh() const { return lor_mesh_.get(); }

  /**
   * @brief Get the redecomp mesh containing redecomposed contact surfaces
   *
   * @return redecomp::RedecompMesh&
   */
  redecomp::RedecompMesh& GetRedecompMesh() { return GetUpdateData().redecomp_mesh_; }

  /**
   * @brief Get the set of boundary attributes on the parent mesh corresponding
   * to surface elements contained in the first Tribol registered mesh
   *
   * @return Set of boundary attributes
   */
  const std::set<int>& GetBoundaryAttribs1() const { return attributes_1_; }

  /**
   * @brief Get the set of boundary attributes on the parent mesh corresponding
   * to surface elements contained in the second Tribol registered mesh
   *
   * @return Set of boundary attributes
   */
  const std::set<int>& GetBoundaryAttribs2() const { return attributes_2_; }

  /**
   * @brief Get the finite element space on the parent-linked boundary submesh
   *
   * @return const mfem::ParFiniteElementSpace&
   */
  const mfem::ParFiniteElementSpace& GetSubmeshFESpace() const { return *submesh_xfer_gridfn_.ParFESpace(); }

  /**
   * @brief Get the finite element space on the LOR mesh
   *
   * @return const mfem::ParFiniteElementSpace*
   *
   * @note nullptr if no LOR mesh exists (polynomial order of parent is 1)
   */
  const mfem::ParFiniteElementSpace* GetLORMeshFESpace() const
  {
    return submesh_lor_xfer_ ? submesh_lor_xfer_->GetLORGridFn().ParFESpace() : nullptr;
  }

  /**
   * @brief Get the LOR factor
   *
   * @note The LOR factor corresponds to the number of LOR elements per HO element applied to each dimension on the LOR
   * mesh.
   *
   * @return int
   */
  int GetLORFactor() const { return lor_factor_; }

  /**
   * @brief Set the LOR factor
   *
   * @note The LOR factor corresponds to the number of LOR elements per HO element applied to each dimension on the LOR
   * mesh.
   *
   * @param lor_factor LOR factor
   */
  void SetLORFactor( int lor_factor );

  /**
   * @brief Computes element thicknesses for volume elements attached to the contact surface
   */
  void ComputeElementThicknesses();

  /**
   * @brief Compute material modulus field at each element
   *
   * @param modulus_field An mfem::Coefficient which spatially evaluates to the material modulus value
   */
  void SetMaterialModulus( mfem::Coefficient& modulus_field );

  /**
   * @brief Get the memory space used by the Tribol fields
   *
   * @return MemorySpace
   */
  MemorySpace GetMemorySpace() const { return mem_space_; }

  /**
   * @brief Sets the displacement threshold for triggering a new parallel decomposition
   *
   * @param val Threshold value
   */
  void SetRedecompTriggerDisplacement( RealT val ) { redecomp_trigger_displacement_ = val; }

 private:
  /**
   * @brief Creates and stores data that changes when the RedecompMesh is updated
   */
  struct UpdateData {
    /**
     * @brief Construct a new UpdateData object
     *
     * @param submesh Parent-linked boundary submesh of contact elements
     * @param lor_mesh LOR mesh of contact elements (if using LOR; nullptr otherwise)
     * @param parent_fes Vector finite element space on the original parent mesh
     * @param submesh_gridfn Grid function on the parent-linked boundary submesh used to temporarily store variables
     *        being transferred
     * @param submesh_lor_xfer Submesh to LOR grid function transfer object (if using LOR; nullptr otherwise)
     * @param attributes_1 Set of boundary attributes identifying elements in the first Tribol registered mesh
     * @param attributes_2 Set of boundary attributes identifying elements in the second Tribol registered mesh
     * @param binning_proximity_scale Element length multiplier for coarse binning and proximity detection inclusion.
     *        This is needed to size the ghost element layer in the redecomp mesh.
     * @param n_ranks Number of ranks in the parallel decomposition
     * @param allocator_id Allocation space ID for Tribol memory
     * @param redecomp_trigger_displacement Additional length to add to redecomp ghost length equal to the
     *        displacement required to trigger a new RedecompMesh to be built.
     */
    UpdateData( mfem::ParSubMesh& submesh, mfem::ParMesh* lor_mesh, const mfem::ParFiniteElementSpace& parent_fes,
                mfem::ParGridFunction& submesh_gridfn, SubmeshLORTransfer* submesh_lor_xfer,
                const std::set<int>& attributes_1, const std::set<int>& attributes_2, RealT binning_proximity_scale,
                int n_ranks, int allocator_id, RealT redecomp_trigger_displacement );

    /**
     * @brief Redecomposed boundary element mesh
     */
    redecomp::RedecompMesh redecomp_mesh_;

    /**
     * @brief Parent mesh to redecomp mesh field transfer object
     */
    ParentRedecompTransfer vector_xfer_;

    /**
     * @brief Redecomp mesh element connectivity for the first Tribol registered
     * mesh
     */
    Array2D<IndexT> conn_1_;

    /**
     * @brief Redecomp mesh element connectivity for the second Tribol
     * registered mesh
     */
    Array2D<IndexT> conn_2_;

    /**
     * @brief Map from first Tribol registered mesh element indices to redecomp
     * mesh element indices
     */
    Array1D<int> elem_map_1_;

    /**
     * @brief Map from second Tribol registered mesh element indices to redecomp
     * mesh element indices
     */
    Array1D<int> elem_map_2_;

    /**
     * @brief Type of elements on the contact meshes
     */
    InterfaceElementType elem_type_;

    /**
     * @brief Number of vertices on each element in the contact meshes
     */
    int num_verts_per_elem_;

    /**
     * @brief Umpire allocator ID for Tribol data
     */
    int allocator_id_;

   private:
    /**
     * @brief Builds connectivity arrays and redecomp mesh to Tribol registered
     * mesh element maps
     *
     * @param attributes_1 Set of boundary attributes for the first Tribol
     * registered mesh
     * @param attributes_2 Set of boundary attributes for the second Tribol
     * registered mesh
     */
    void UpdateConnectivity( const std::set<int>& attributes_1, const std::set<int>& attributes_2 );

    /**
     * @brief Sets the number of vertices per element and the element type for the redecomp mesh
     */
    void SetElementData();
  };

  /**
   * @brief Get the UpdateData object
   *
   * @return UpdateData&
   */
  UpdateData& GetUpdateData();

  /**
   * @brief Get the UpdateData object
   *
   * @return const UpdateData&
   */
  const UpdateData& GetUpdateData() const;

  /**
   * @brief Create the parent-linked boundary submesh
   *
   * @param parent_mesh Parent mesh, i.e. volume mesh of parent domain
   * @param attributes_1 Mesh boundary attributes identifying surface elements
   * in the first Tribol registered mesh
   * @param attributes_2 Mesh boundary attributes identifying surface elements
   * in the second Tribol registered mesh
   * @return mfem::ParSubMesh
   */
  static mfem::ParSubMesh CreateSubmesh( const mfem::ParMesh& parent_mesh, const std::set<int>& attributes_1,
                                         const std::set<int>& attributes_2 );

  /**
   * @brief First mesh identifier
   */
  IndexT mesh_id_1_;

  /**
   * @brief Second mesh identifier
   */
  IndexT mesh_id_2_;

  /**
   * @brief Volume mesh of parent domain
   */
  const mfem::ParMesh& parent_mesh_;

  /**
   * @brief Mesh boundary attributes identifying first mesh
   */
  const std::set<int> attributes_1_;

  /**
   * @brief Mesh boundary attributes identifying second mesh
   */
  const std::set<int> attributes_2_;

  /**
   * @brief Submesh containing boundary elements of both contact surfaces
   */
  mfem::ParSubMesh submesh_;

  /**
   * @brief Coordinates grid function and transfer operators
   */
  ParentField coords_;

  /**
   * @brief Contains reference coords grid function and transfer operators if set; nullptr otherwise
   */
  std::unique_ptr<ParentField> reference_coords_;

  /**
   * @brief Submesh grid function to temporarily hold values being transferred
   */
  mfem::ParGridFunction submesh_xfer_gridfn_;

  /**
   * @brief Refinement factor for refined mesh
   */
  int lor_factor_;

  /**
   * @brief Contains LOR mesh if low-order refinement is being used; nullptr
   * otherwise
   */
  std::unique_ptr<mfem::ParMesh> lor_mesh_;

  /**
   * @brief Contains LOR mesh to submesh transfer operators if LOR is being
   * used; nullptr otherwise
   */
  std::unique_ptr<SubmeshLORTransfer> submesh_lor_xfer_;

  /**
   * @brief Contains velocity grid function and transfer operators if set;
   * nullptr otherwise
   */
  std::unique_ptr<ParentField> velocity_;

  /**
   * @brief Kinematic constant contact penalty for the first Tribol registered mesh
   */
  std::unique_ptr<RealT> kinematic_constant_penalty_1_;

  /**
   * @brief Kinematic constant contact penalty for the second Tribol registered mesh
   */
  std::unique_ptr<RealT> kinematic_constant_penalty_2_;

  /**
   * @brief Scaling of kinematic penalty for the first Tribol registered mesh
   */
  std::unique_ptr<RealT> kinematic_penalty_scale_1_;

  /**
   * @brief Scaling of kinematic penalty for the second Tribol registered mesh
   */
  std::unique_ptr<RealT> kinematic_penalty_scale_2_;

  /**
   * @brief Rate constant contact penalty for the first Tribol registered mesh
   */
  std::unique_ptr<RealT> rate_constant_penalty_1_;

  /**
   * @brief Rate constant contact penalty for the second Tribol registered mesh
   */
  std::unique_ptr<RealT> rate_constant_penalty_2_;

  /**
   * @brief Rate percent penalty as a ratio of kinematic penalty for the first
   * Tribol registered mesh
   */
  std::unique_ptr<RealT> rate_percent_ratio_1_;

  /**
   * @brief Rate percent penalty as a ratio of kinematic penalty for the second
   * Tribol registered mesh
   */
  std::unique_ptr<RealT> rate_percent_ratio_2_;

  /**
   * @brief Tangential viscous damping coefficient for the first Tribol registered mesh
   */
  std::unique_ptr<RealT> viscous_damping_coeff_1_;

  /**
   * @brief Tangential viscous damping coefficient for the second Tribol registered mesh
   */
  std::unique_ptr<RealT> viscous_damping_coeff_2_;

  /**
   * @brief Stores element thicknesses for element-based penalty calculations on
   * the submesh or the LOR mesh (if it exists); nullptr otherwise
   */
  std::unique_ptr<mfem::QuadratureFunction> elem_thickness_;

  /**
   * @brief Element thickness stored on the redecomp mesh
   */
  std::unique_ptr<mfem::QuadratureFunction> redecomp_elem_thickness_;

  /**
   * @brief Element thicknesses for the first Tribol registered mesh
   */
  std::unique_ptr<ArrayT<RealT>> tribol_elem_thickness_1_;

  /**
   * @brief Element thicknesses for the second Tribol registered mesh
   */
  std::unique_ptr<ArrayT<RealT>> tribol_elem_thickness_2_;

  /**
   * @brief Stores material moduli for element-based penalty calculations on
   * the submesh or the LOR mesh (if it exists); nullptr otherwise
   */
  std::unique_ptr<mfem::QuadratureFunction> material_modulus_;

  /**
   * @brief Material modulus stored on the redecomp mesh
   */
  std::unique_ptr<mfem::QuadratureFunction> redecomp_material_modulus_;

  /**
   * @brief Material moduli for the first Tribol registered mesh
   */
  std::unique_ptr<ArrayT<RealT>> tribol_material_modulus_1_;

  /**
   * @brief Material moduli for the second Tribol registered mesh
   */
  std::unique_ptr<ArrayT<RealT>> tribol_material_modulus_2_;

  /**
   * @brief UpdateData object created upon call to UpdateMeshData()
   */
  std::unique_ptr<UpdateData> update_data_;

  /**
   * @brief Nodal response grid function on the redecomp mesh
   */
  std::unique_ptr<mfem::GridFunction> redecomp_response_;
  // NOTE: redecomp_response_ doesn't need to be a pointer, but SetSpace() doesn't seem to register memory correctly
  // when on device. TODO: Debug this and remove the pointer.

  /**
   * @brief Execution mode for Tribol
   */
  ExecutionMode exec_mode_;

  /**
   * @brief Memory space for Tribol
   */
  MemorySpace mem_space_;

  /**
   * @brief Whether to use device memory for MFEM data
   */
  bool use_device_;

  /**
   * @brief Umpire allocator ID for Tribol data
   */
  int allocator_id_;

  /**
   * @brief Threshold for max nodal displacement change to trigger a new parallel decomposition
   */
  RealT redecomp_trigger_displacement_;

  /**
   * @brief Parent coordinate values at the time of the last parallel decomposition
   */
  mutable mfem::Vector coords_at_last_redecomp_;

  /**
   * @brief Merges two STL containers
   *
   * @tparam T container type
   * @param container_1 First container
   * @param container_2 Second container
   * @return T merged container
   */
  template <typename T>
  static T mergeContainers( T container_1, T container_2 )
  {
    auto merged = container_1;
    merged.insert( container_2.begin(), container_2.end() );
    return merged;
  }

  /**
   * @brief Converts a std::set to an mfem::Array
   *
   * @tparam T type held in the set and array
   * @param orig original set
   * @return mfem::Array<T> output array holding entries in orig
   */
  template <typename T>
  static mfem::Array<T> arrayFromSet( std::set<T> orig )
  {
    auto array = mfem::Array<T>();
    array.Reserve( static_cast<int>( orig.size() ) );
    for ( const auto& val : orig ) {
      array.Append( val );
    }
    return array;
  }
};

/**
 * @brief Stores MFEM and transfer data associated with parent-linked boundary
 * submesh pressure and gap fields
 */
class MfemSubmeshData {
 public:
  /**
   * @brief Construct a new MfemSubmeshData object
   *
   * @param submesh Parent-linked boundary submesh
   * @param lor_mesh LOR mesh of contact surfaces (if using LOR; nullptr
   * otherwise)
   * @param pressure_fec Finite element collection of the pressure field
   * @param pressure_vdim Vector dimension of the pressure field
   * @param use_device Whether to use device memory
   */
  MfemSubmeshData( mfem::ParSubMesh& submesh, mfem::ParMesh* lor_mesh,
                   std::unique_ptr<mfem::FiniteElementCollection> pressure_fec, int pressure_vdim, bool use_device );

  /**
   * @brief Update (or clear) the LOR mesh used for pressure/gap transfers
   *
   * @note This should be called if the owning MfemMeshData rebuilds its LOR mesh
   * (e.g., via MfemMeshData::SetLORFactor()). This resets internal transfer data
   * so the next UpdateMfemSubmeshData() rebuilds consistent redecomp transfers.
   *
   * @param lor_mesh New LOR mesh pointer (nullptr disables LOR transfers)
   */
  void SetLORMesh( mfem::ParMesh* lor_mesh );

  /**
   * @brief Build a new transfer operator and update redecomp-level grid
   * functions
   *
   * @param redecomp_mesh Updated redecomp mesh
   * @param new_redecomp If true, construct an updated SubmeshRedecompTransfer object for new RedecompMesh
   */
  void UpdateMfemSubmeshData( redecomp::RedecompMesh& redecomp_mesh, bool new_redecomp = true );

  /**
   * @brief Get pointers to component arrays of the pressure on the redecomp
   * mesh
   *
   * @return std::vector<const RealT*> of length 3
   *
   * @note Unused entries are nullptr.  Only the first entry is used with
   * frictionless contact.
   */
  std::vector<const RealT*> GetRedecompPressurePtrs() const { return pressure_.GetRedecompFieldPtrs(); }

  /**
   * @brief Get the parent-linked boundary submesh pressure grid function
   *
   * @return mfem::ParGridFunction&
   */
  mfem::ParGridFunction& GetSubmeshPressure() { return submesh_pressure_; }

  /**
   * @brief Get the parent-linked boundary submesh pressure grid function
   *
   * @return const mfem::ParGridFunction&
   */
  const mfem::ParGridFunction& GetSubmeshPressure() const { return submesh_pressure_; }

  /**
   * @brief Get pointers to component arrays of the gap on the redecomp mesh
   *
   * @return std::vector<RealT*> of length 3
   *
   * @note Unused entries are nullptr.  Only the first entry is used with
   * frictionless contact.
   */
  std::vector<RealT*> GetRedecompGapPtrs() { return PressureField::GetRedecompFieldPtrs( redecomp_gap_ ); }

  /**
   * @brief Get the gap grid function on the redecomp mesh
   *
   * @return const mfem::GridFunction&
   */
  const mfem::GridFunction& GetRedecompGap() const { return redecomp_gap_; }

  /**
   * @brief Get the gap grid function on the redecomp mesh
   *
   * @return const mfem::GridFunction&
   */
  mfem::GridFunction& GetRedecompGap() { return redecomp_gap_; }

  /**
   * @brief Get the gap vector on the parent-linked boundary submesh
   *
   * @note This is stored as an MFEM dual vector, meaning the shared DOFs are expected to be summed over all ranks to
   * obtain their value.
   *
   * @param [out] g Un-initialized mfem::Vector holding the nodal gap values
   */
  void GetSubmeshGap( mfem::Vector& g ) const;

  /**
   * @brief Get the parent-linked boundary submesh to redecomp mesh pressure
   * transfer object
   *
   * @return const SubmeshRedecompTransfer&
   */
  const SubmeshRedecompTransfer& GetPressureTransfer() const { return GetUpdateData().pressure_xfer_; }

  /**
   * @brief Get the finite element space on the parent-linked boundary submesh
   *
   * @return const mfem::ParFiniteElementSpace&
   */
  const mfem::ParFiniteElementSpace& GetSubmeshFESpace() const { return *submesh_pressure_.ParFESpace(); }

  /**
   * @brief Get the finite element space on the LOR mesh
   *
   * @return const mfem::ParFiniteElementSpace* or nullptr if no LOR mesh
   * (polynomial order of parent is 1)
   */
  const mfem::ParFiniteElementSpace* GetLORMeshFESpace() const
  {
    return submesh_lor_xfer_ ? submesh_lor_xfer_->GetLORGridFn().ParFESpace() : nullptr;
  }

 private:
  /**
   * @brief Creates and stores data that changes when the redecomp mesh is
   * updated
   */
  struct UpdateData {
    /**
     * @brief Construct a new UpdateData object
     *
     * @param submesh_fes Pressure finite element space on the parent-linked
     * boundary submesh
     * @param submesh_lor_xfer Submesh to LOR grid function transfer object (if
     * using LOR; nullptr otherwise)
     * @param redecomp Redecomp mesh
     */
    UpdateData( mfem::ParFiniteElementSpace& submesh_fes, SubmeshLORTransfer* submesh_lor_xfer,
                redecomp::RedecompMesh& redecomp_mesh );

    /**
     * @brief Parent-linked boundary submesh to redecomp mesh field transfer
     * object
     */
    SubmeshRedecompTransfer pressure_xfer_;
  };

  /**
   * @brief Get the UpdateData object
   *
   * @return UpdateData&
   */
  UpdateData& GetUpdateData();

  /**
   * @brief Get the UpdateData object
   *
   * @return const UpdateData&
   */
  const UpdateData& GetUpdateData() const;

  /**
   * @brief Pressure grid function on the parent-linked boundary submesh
   */
  mfem::ParGridFunction submesh_pressure_;

  /**
   * @brief Pressure grid function and transfer operators
   */
  PressureField pressure_;

  /**
   * @brief Contains LOR mesh transfer operators if LOR is being used; nullptr
   * otherwise
   */
  std::unique_ptr<SubmeshLORTransfer> submesh_lor_xfer_;

  /**
   * @brief Gap grid function on the redecomp mesh
   */
  mfem::GridFunction redecomp_gap_;

  /**
   * @brief UpdateData object created upon call to UpdateSubmeshData()
   */
  std::unique_ptr<UpdateData> update_data_;

  /**
   * @brief Whether to use device memory for MFEM data
   */
  bool use_device_;
};

/**
 * @brief HO -> LOR transfer action (matrix-free apply) for one FE-space pair
 *
 * This is intentionally matrix-free: it only provides Mult() and does not
 * expose any assembled operator.
 */
class HoToLorTransferAction : public mfem::Operator {
 public:
  /**
   * @brief Construct an action that maps HO DOFs to LOR DOFs for one field space
   *
   * @param ho_fes Higher-order (HO) finite element space
   * @param lor_fes Low-order refined (LOR) finite element space
   * @param use_ea Whether to use element assembly in MFEM's transfer operator
   */
  HoToLorTransferAction( const mfem::ParFiniteElementSpace& ho_fes, const mfem::ParFiniteElementSpace& lor_fes,
                         bool use_ea );

  /**
   * @brief Apply MFEM's runtime HO->LOR transfer without explicitly assembling a matrix
   *
   * @param x Input vector on the HO DOF space
   * @param y Output vector on the LOR DOF space (overwritten)
   */
  void Mult( const mfem::Vector& x, mfem::Vector& y ) const override;

 private:
  /// Higher-order (HO) surface finite element space
  const mfem::ParFiniteElementSpace& ho_fes_;
  /// Low-order refined (LOR) surface finite element space
  const mfem::ParFiniteElementSpace& lor_fes_;
  /// MFEM HO->LOR transfer object (constructed lazily but reused across Mult() calls)
  mutable std::unique_ptr<mfem::L2ProjectionGridTransfer> transfer_;
};

/**
 * @brief HO -> LOR transfer matrix builder for one FE-space pair
 *
 * This type is assembled-only and returns an explicit HypreParMatrix wrapper.
 */
class HoToLorTransferMat {
 public:
  /**
   * @brief Construct a builder that maps HO DOFs to LOR DOFs for one field space
   *
   * @param ho_fes Higher-order (HO) finite element space
   * @param lor_fes Low-order refined (LOR) finite element space
   * @param ho_scalar_fes Scalar (vdim=1) HO companion space (used to assemble component operators)
   * @param lor_scalar_fes Scalar (vdim=1) LOR companion space (used to assemble component operators)
   */
  HoToLorTransferMat( const mfem::ParFiniteElementSpace& ho_fes, const mfem::ParFiniteElementSpace& lor_fes,
                      const mfem::ParFiniteElementSpace& ho_scalar_fes,
                      const mfem::ParFiniteElementSpace& lor_scalar_fes );

  /**
   * @brief Assemble a sparse matrix matching MFEM's HO->LOR transfer action
   */
  shared::ParSparseMat Assemble() const;

 private:
  /// Higher-order (HO) surface finite element space
  const mfem::ParFiniteElementSpace& ho_fes_;
  /// Low-order refined (LOR) surface finite element space
  const mfem::ParFiniteElementSpace& lor_fes_;
  /// Scalar (vdim=1) companion HO space used to build component operators
  const mfem::ParFiniteElementSpace& ho_scalar_fes_;
  /// Scalar (vdim=1) companion LOR space used to build component operators
  const mfem::ParFiniteElementSpace& lor_scalar_fes_;
};

/**
 * @brief Submesh -> parent DOF transfer matrix builder
 */
class SubmeshParentTransferMat {
 public:
  /**
   * @brief Construct a builder that injects boundary-submesh DOFs into the parent FE space
   *
   * @param submesh_fes Boundary submesh finite element space
   * @param parent_fes Parent volume finite element space
   * @param submesh2parent_vdof_list Global parent vdof for each local submesh vdof
   */
  SubmeshParentTransferMat( const mfem::ParFiniteElementSpace& submesh_fes,
                            const mfem::ParFiniteElementSpace& parent_fes,
                            const mfem::Array<HYPRE_BigInt>& submesh2parent_vdof_list );

  /**
   * @brief Assemble the submesh->parent DOF injection matrix
   */
  shared::ParSparseMat Assemble() const;

 private:
  /// Boundary submesh finite element space
  const mfem::ParFiniteElementSpace& submesh_fes_;
  /// Parent volume finite element space
  const mfem::ParFiniteElementSpace& parent_fes_;
  /// Global parent vdof for each local submesh vdof
  const mfem::Array<HYPRE_BigInt>& submesh2parent_vdof_list_;
};

/**
 * @brief Packed Jacobian contributions for one row/col FE-space + Tribol mesh pairing
 *
 * This struct stores stacked element-pair Jacobian contributions along with:
 * - the row/col surface finite element spaces the assembled matrix lives on, and
 * - the row/col redecomp finite element spaces and Tribol mesh element-id maps
 *
 * In this context, "surface" refers to the parent FE spaces of the redecomp FE spaces used during redecomp transfer.
 * In the MFEM integration those parent (surface) FE spaces live on either the LOR surface mesh (when LOR is active)
 * or the boundary submesh (otherwise).
 *
 * Each instance is tied to a single row/col surface FE space pairing and a single row/col Tribol mesh
 * (via the element maps).
 */
struct PackedPairJacobianContribs {
  /// Surface FE space for the assembled row DOF layout (LOR mesh if active; otherwise submesh)
  const mfem::ParFiniteElementSpace* row_surface_fes{ nullptr };
  /// Surface FE space for the assembled column DOF layout (LOR mesh if active; otherwise submesh)
  const mfem::ParFiniteElementSpace* col_surface_fes{ nullptr };
  /// Redecomp FE space that defines the row element-id domain and element DOF layout
  const mfem::FiniteElementSpace* row_redecomp_fes{ nullptr };
  /// Redecomp FE space that defines the column element-id domain and element DOF layout
  const mfem::FiniteElementSpace* col_redecomp_fes{ nullptr };
  /// Tribol element-id -> redecomp element-id map for row element ids
  const Array1D<int>* row_elem_map{ nullptr };
  /// Tribol element-id -> redecomp element-id map for column element ids
  const Array1D<int>* col_elem_map{ nullptr };
  Array1D<int, MemorySpace::Host> row_elem_ids;          ///< Tribol element IDs for rows
  Array1D<int, MemorySpace::Host> col_elem_ids;          ///< Tribol element IDs for columns
  Array1D<double, MemorySpace::Host> jacobian_data;      ///< Flattened Jacobian data
  Array1D<int, MemorySpace::Host> value_offsets;         ///< Offsets into jacobian_data for each element

  PackedPairJacobianContribs() = default;

  PackedPairJacobianContribs( const mfem::ParFiniteElementSpace& row_fes, const mfem::ParFiniteElementSpace& col_fes,
                              const mfem::FiniteElementSpace& row_redecomp_fes_in,
                              const mfem::FiniteElementSpace& col_redecomp_fes_in, const Array1D<int>& row_map,
                              const Array1D<int>& col_map )
      : row_surface_fes( &row_fes ),
        col_surface_fes( &col_fes ),
        row_redecomp_fes( &row_redecomp_fes_in ),
        col_redecomp_fes( &col_redecomp_fes_in ),
        row_elem_map( &row_map ),
        col_elem_map( &col_map )
  {
  }

  /**
   * @brief Reserve packed storage for a batch of element contributions
   *
   * @param n_pairs Number of element-pair contributions that will be appended.
   * @param n_jacobian_scalar_values Total number of scalar Jacobian values that will be appended across all pairs.
   * This is used to reserve capacity in @ref jacobian_data.
   */
  void reserve( int n_pairs, int n_jacobian_scalar_values )
  {
    row_elem_ids.reserve( n_pairs );
    col_elem_ids.reserve( n_pairs );
    jacobian_data.reserve( n_jacobian_scalar_values );
    value_offsets.reserve( n_pairs );
  }

  /**
   * @brief Append one flattened element Jacobian contribution
   *
   * @param row_elem_id Tribol element id for the row side of this contribution.
   * @param col_elem_id Tribol element id for the column side of this contribution.
   * @param data Pointer to a contiguous, column-major dense block of Jacobian values.
   * @param size Number of scalar entries in @p data (typically `n_row_dofs * n_col_dofs`).
   */
  void append( int row_elem_id, int col_elem_id, const double* data, int size )
  {
    row_elem_ids.push_back( row_elem_id );
    col_elem_ids.push_back( col_elem_id );
    value_offsets.push_back( jacobian_data.size() );
    if ( size > 0 ) {
      jacobian_data.append( axom::ArrayView<const double>( data, size ) );
    }
  }

  /**
   * @brief Number of element contributions stored in this packed block
   */
  int numEntries() const { return row_elem_ids.size(); }
};

/**
 * @brief Assemble a Jacobian on LOR/submesh surface FE spaces from redecomp element contributions
 *
 * Here, "surface" refers to the parent FE spaces of the redecomp FE spaces used for the transfer. In the MFEM
 * integration those parent (surface) FE spaces live on either the LOR surface mesh (when LOR is active) or the
 * boundary submesh (otherwise).
 */
class RedecompJacobianAssembler {
 public:
  /**
   * @brief Construct one redecomp->surface Jacobian transfer for a row/column pairing
   *
   * @param transfer Redecomp transfer route for the given row/col surface FE spaces
   * @param contributions Packed element Jacobian contributions for one row/col FE-space pairing
   */
  RedecompJacobianAssembler( const redecomp::MatrixTransfer& transfer,
                             std::vector<PackedPairJacobianContribs> contributions );

  /**
   * @brief Assemble the redecomp-stage matrix from flattened element contributions
   */
  shared::ParSparseMat Assemble() const;

 private:
  /// Redecomp transfer route used to map redecomp element contributions onto the surface DOF layout
  const redecomp::MatrixTransfer& transfer_;
  /// Packed element Jacobian contributions (possibly aggregated from multiple sources)
  std::vector<PackedPairJacobianContribs> contributions_;
};

/**
 * @brief Assemble a solver-visible Jacobian by composing a LOR/submesh Jacobian with explicit transfer operators
 *
 * The transfer operators must be explicitly assembled sparse matrices. Each transfer list is an ordered chain that maps
 * from solver true DOFs into the corresponding DOF space of the input Jacobian:
 *   x0 (true dofs) -> op[0] -> op[1] -> ... -> xN (lor/submesh dofs)
 *
 * The assembled solver-visible Jacobian is:
 *   J = M_row^T * A * M_col
 * where A is the lor/submesh Jacobian and M_* are the products of the supplied transfer chains.
 */
class JacobianAssembler {
 public:
  /**
   * @brief Construct a composition utility with explicit row/col transfer chains
   */
  JacobianAssembler( std::vector<shared::ParSparseMatView> row_transfer_ops,
                     std::vector<shared::ParSparseMatView> col_transfer_ops );

  /**
   * @brief Assemble the solver-visible Jacobian by composing the provided lor/submesh Jacobian
   *
   * @param lor_or_submesh_jacobian Jacobian assembled on LOR or submesh DOF spaces (moved in)
   */
  shared::ParSparseMat Assemble( shared::ParSparseMat lor_or_submesh_jacobian ) const;

 private:
  /// Transfer operators mapping solver row true DOFs into the Jacobian's row DOF space
  std::vector<shared::ParSparseMatView> row_transfer_ops_;
  /// Transfer operators mapping solver col true DOFs into the Jacobian's col DOF space
  std::vector<shared::ParSparseMatView> col_transfer_ops_;
};

/**
 * @brief A fixed transfer pathway from a solver-visible FE space (true dofs) to a surface FE space (dofs)
 *
 * This stores an ordered chain of explicit sparse matrices that map from the true dofs of @ref final_fes to the dofs
 * of @ref surface_fes. Here, "surface" refers to the parent FE space of the redecomp FE space used during transfer:
 * it is either on the LOR surface mesh (when LOR is active) or on the boundary submesh (otherwise):
 *   x_true -> ops[0] -> ops[1] -> ... -> x_surface
 *
 * The first operator is typically MFEM's prolongation matrix `final_fes->Dof_TrueDof_Matrix()`. Additional operators
 * may include submesh restriction and HO->LOR transfer matrices, depending on the chosen surface space.
 *
 * @note Performance: this struct currently stores the factorized chain. If profiling shows the repeated sparse
 * matrix multiplications in JacobianAssembler are a bottleneck, consider precomputing and caching the combined
 * pathway operator `M = ops[0] * ops[1] * ...` in UpdateJacobianXfer() and storing just `M`.
 */
struct MfemJacobianPath {
  /// Solver-visible final FE space (defines the input true dofs)
  const mfem::ParFiniteElementSpace* final_fes{ nullptr };
  /// Surface FE space that the intermediate redecomp-to-surface Jacobian is assembled on
  const mfem::ParFiniteElementSpace* surface_fes{ nullptr };

  /// Owned assembled operators that appear in @ref ops
  std::vector<std::unique_ptr<shared::ParSparseMat>> owned_ops;
  /// Operator chain mapping final true dofs into surface dofs
  std::vector<shared::ParSparseMatView> ops;
};

/**
 * @brief End-to-end Jacobian transfer: redecomp contributions -> surface Jacobian -> solver-visible Jacobian
 *
 * This composes a redecomp-to-surface Jacobian (assembled from element-pair contributions) with explicit row/col
 * transfer pathways to produce a solver-visible Jacobian on the true dof spaces of the chosen final FE spaces.
 */
class MfemJacobianTransfer {
 public:
  /**
   * @brief Construct a transfer pipeline for one solver-visible (row_final, col_final) pairing
   */
  MfemJacobianTransfer( const MfemJacobianPath& row_path, const MfemJacobianPath& col_path );

  /**
   * @brief Assemble a solver-visible Jacobian from packed redecomp element contributions
   *
   * @note The contributions must be for a single row/col surface FE-space pairing and must match the provided paths.
   */
  shared::ParSparseMat Assemble( const std::vector<PackedPairJacobianContribs>& contributions ) const;

 private:
  const MfemJacobianPath& row_path_;
  const MfemJacobianPath& col_path_;
};
/**
 * @brief Simplifies transfer of Jacobian matrix data between MFEM and Tribol
 */
class MfemJacobianData {
 public:
  /**
   * @brief Construct a new MfemJacobianData object
   *
   * @param parent_data MFEM data associated with displacement and velocity
   * @param submesh_data MFEM data associated with pressure and gap
   */
  MfemJacobianData( const MfemMeshData& parent_data, const MfemSubmeshData& submesh_data );

  /**
   * @brief Builds new transfer data after a new redecomp mesh has been built
   */
  void UpdateJacobianXfer();

  /**
   * @brief Assemble a solver-visible Jacobian on caller-provided final FE spaces
   *
   * This composes the intermediate LOR/submesh DOF-level Jacobian (assembled from redecomp element contributions)
   * with explicit transfer operators to produce a Jacobian on the true-dof spaces of @p row_final_fes and
   * @p col_final_fes.
   *
   * @note @p row_final_fes and @p col_final_fes must be one of:
   * - parent FE space (parent_data_.GetParentCoords().ParFESpace())
   * - multiplier submesh FE space (submesh_data_.GetSubmeshFESpace())
   */
  shared::ParSparseMat GetMfemJacobian( const mfem::ParFiniteElementSpace* row_final_fes,
                                        const mfem::ParFiniteElementSpace* col_final_fes,
                                        const std::vector<PackedPairJacobianContribs>& contributions ) const;

  /**
   * @brief Access the transfer pathway for parent-final Jacobians
   *
   * This maps from the parent true dofs into the selected primary surface DOF space (LOR if active, otherwise submesh).
   *
   * @note Requires UpdateJacobianXfer() to have been called.
   */
  const MfemJacobianPath& ParentPath() const;

  /**
   * @brief Access the transfer pathway for submesh-final Jacobians
   *
   * This maps from the multiplier submesh true dofs into the selected dual surface DOF space (LOR if active, otherwise
   * submesh).
   *
   * @note Requires UpdateJacobianXfer() to have been called.
   */
  const MfemJacobianPath& SubmeshPath() const;

 private:
  /**
   * @brief Creates and stores data that changes when the redecomp mesh is updated
   */
  struct UpdateData {
    /**
     * @brief Construct a new UpdateData object
     *
     * @param parent_data MFEM data associated with displacement and velocity
     * @param submesh_data MFEM data associated with pressure and gap
     * @param submesh2parent_vdof_list Global parent vdof for each local submesh vdof
     */
    UpdateData( const MfemMeshData& parent_data, const MfemSubmeshData& submesh_data,
                const mfem::Array<HYPRE_BigInt>& submesh2parent_vdof_list );

    /**
     * @brief Transfer pathway for parent-final Jacobians
     */
    MfemJacobianPath parent_path_;

    /**
     * @brief Transfer pathway for submesh-final Jacobians
     */
    MfemJacobianPath submesh_path_;
  };

  /**
   * @brief Get the UpdateData object
   *
   * @return UpdateData&
   */
  UpdateData& GetUpdateData();

  /**
   * @brief Get the UpdateData object
   *
   * @return const UpdateData&
   */
  const UpdateData& GetUpdateData() const;

  /**
   * @brief MFEM and transfer data associated with displacement and velocity
   */
  const MfemMeshData& parent_data_;

  /**
   * @brief MFEM and transfer data associated with pressure and gap
   */
  const MfemSubmeshData& submesh_data_;

  /**
   * @brief List giving global parent vdof given the submesh vdof
   */
  mfem::Array<HYPRE_BigInt> submesh2parent_vdof_list_;

  /**
   * @brief UpdateData object created upon calling UpdateMatrixXfer()
   */
  std::unique_ptr<UpdateData> update_data_;
};

}  // end namespace tribol

#endif /* BUILD_REDECOMP */

#endif /* SRC_TRIBOL_MESH_MFEMDATA_HPP_ */
