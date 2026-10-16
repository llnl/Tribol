// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_MESH_MFEMCONTACTDATA_HPP_
#define SRC_TRIBOL_MESH_MFEMCONTACTDATA_HPP_

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "mfem.hpp"

#include "redecomp/ContactPatchOperator.hpp"
#include "redecomp/ElementFieldExchange.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/MfemContact.hpp"
#include "tribol/mesh/MfemMultiplierSpace.hpp"

namespace tribol {

class MethodData;

/** @brief One caller-owned solver space and its Tribol-owned true-DOF contact residual */
struct MfemContactResidual {
  const mfem::ParFiniteElementSpace* space{ nullptr };
  mfem::Vector true_dof_values;
};

/**
 * @brief Mesh-free linear child-facet patch consumed by one Tribol surface
 *
 * Coordinates are current physical coordinates. Reference coordinates retain
 * the caller's parent/reference representation. Nodal arrays are
 * component-major; connectivity and element fields use child-element order.
 * Response and gap-like arrays store independent evaluation-rank duals until
 * MfemContactData applies the exact transpose.
 */
struct MfemContactPatchView {
  IndexT mesh_id{ -1 };
  InterfaceElementType element_type{ LINEAR_EDGE };
  int space_dimension{ 0 };
  std::vector<IndexT> connectivity;
  std::array<mfem::Vector, 3> coordinates;
  std::array<mfem::Vector, 3> response;
  std::array<mfem::Vector, 3> velocity;
  std::array<mfem::Vector, 3> reference_coordinates;
  mfem::Vector element_thickness;
  mfem::Vector material_modulus;
  bool has_velocity{ false };
  bool has_reference_coordinates{ false };

  int NumElements() const;
  int NumNodes() const;
  int VerticesPerElement() const;
};

/**
 * @brief Coordinates search, sparse face exchange, virtual refinement, and MFEM results
 *
 * This is the single owner for the operator-based MFEM contact path. Parent
 * meshes and fields remain non-owning references, while exchange schedules,
 * patch arrays, and true-DOF result buffers are cached here.
 */
class MfemContactData {
 public:
  MfemContactData( MfemContactSurface surface_1, MfemContactSurface surface_2, MfemContactOptions options,
                   ContactMethod contact_method, EnforcementMethod enforcement_method = PENALTY );
  ~MfemContactData();

  /**
   * @brief Refresh the contact patch and rebuild its topology only when required
   *
   * Rebuilds after an explicit rebuild request, parent mesh/space sequence changes,
   * contact-reach changes, or accumulated physical motion reaches the skin.
   * Otherwise this collective call reuses allocations and MPI schedules and
   * exchanges only field buffers.
   *
   * @param contact_reach Nonnegative physical search distance added to exact virtual-facet bounds
   * @param force_rebuild Rebuild regardless of cached validity
   * @return true when topology and communication schedules were rebuilt
   */
  bool Update( RealT contact_reach, bool force_rebuild = false );

  /** @brief Force the distributed search and exchange schedules to rebuild during the next update */
  void ForcePatchRebuildOnNextUpdate() { patch_rebuild_requested_ = true; }

  /**
   * @brief Overwrite contact residuals by applying each patch operator transpose
   *
   * Call after Tribol has accumulated independent child-patch forces. Shared
   * true DOFs are summed by MFEM's transpose; no rank-count averaging occurs.
   */
  void CollectResiduals();

  const MfemContactPatchView& SurfacePatch( int surface ) const { return surface_patches_[surface]; }
  MfemContactPatchView& SurfacePatch( int surface ) { return surface_patches_[surface]; }
  const std::vector<IndexT>& PairSurface1Elements() const { return pair_surface_1_elements_; }
  const std::vector<IndexT>& PairSurface2Elements() const { return pair_surface_2_elements_; }
  const std::vector<MfemContactResidual>& ContactResiduals() const { return contact_residuals_; }
  const MfemContactOptions& Options() const { return options_; }
  MfemMultiplierSpace* MultiplierSpace() { return multiplier_space_.get(); }
  const MfemMultiplierSpace* MultiplierSpace() const { return multiplier_space_.get(); }
  MPI_Comm Communicator() const { return communicator_; }

  std::array<int, 2> SentFaceCounts() const;
  std::array<int, 2> ReceivedFaceCounts() const;

  /**
   * @brief Assemble solver-visible residual blocks as @f$H_r^T A_{rc} H_c@f$
   *
   * The first call lazily assembles and caches each field operator H. The
   * returned BlockOperator owns its blocks and follows ContactResiduals() order;
   * the multiplier residual is last when present.
   */
  std::unique_ptr<mfem::BlockOperator> BuildJacobian( const MethodData& method_data );

  void SetKinematicConstantPenalty( RealT first, RealT second ) { kinematic_constant_penalty_ = { first, second }; }
  const std::optional<std::array<RealT, 2>>& KinematicConstantPenalty() const { return kinematic_constant_penalty_; }

  void SetViscousDampingCoefficient( RealT first, RealT second ) { viscous_damping_coefficient_ = { first, second }; }
  const std::optional<std::array<RealT, 2>>& ViscousDampingCoefficient() const { return viscous_damping_coefficient_; }

 private:
  struct SurfaceState;
  struct ParentFacePair {
    int first_rank;
    int first_element;
    int second_rank;
    int second_element;
  };

  void ValidateRegistration() const;
  bool ContactPatchNeedsRebuild( RealT contact_reach ) const;
  void BuildPatch( RealT contact_reach );
  void RefreshPatchFields();
  void InitializeContactResiduals();
  void BuildChildPairs();
  MfemContactPlacement ResolvePlacement() const;

  std::array<MfemContactSurface, 2> surfaces_;
  MfemContactOptions options_;
  ContactMethod contact_method_;
  EnforcementMethod enforcement_method_;
  MPI_Comm communicator_;
  int rank_{ 0 };
  int num_ranks_{ 1 };
  int evaluation_rank_count_{ 1 };
  RealT motion_skin_{ 0.0 };
  RealT rebuild_contact_reach_{ 0.0 };
  bool patch_rebuild_requested_{ true };
  std::array<std::unique_ptr<SurfaceState>, 2> surface_states_;
  std::array<MfemContactPatchView, 2> surface_patches_;
  std::vector<ParentFacePair> parent_pairs_;
  std::vector<IndexT> pair_surface_1_elements_;
  std::vector<IndexT> pair_surface_2_elements_;
  std::vector<MfemContactResidual> contact_residuals_;
  std::unique_ptr<MfemMultiplierSpace> multiplier_space_;
  std::optional<std::array<RealT, 2>> kinematic_constant_penalty_;
  std::optional<std::array<RealT, 2>> viscous_damping_coefficient_;
};

}  // namespace tribol

#endif  // BUILD_REDECOMP

#endif  // SRC_TRIBOL_MESH_MFEMCONTACTDATA_HPP_
