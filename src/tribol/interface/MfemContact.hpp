// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTERFACE_MFEMCONTACT_HPP_
#define SRC_TRIBOL_INTERFACE_MFEMCONTACT_HPP_

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP

#include <set>
#include <utility>

#include "mfem.hpp"

#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/ExecModel.hpp"

namespace tribol {

/** @brief Policy assigning each parent-face candidate pair to exactly one evaluation rank */
enum class MfemContactPlacement
{
  /// Select ANCHOR_OWNER for asymmetric mortar, BALANCED_ANCHOR for other two-surface contact, and SPATIAL_PARTITION
  /// for self-contact.
  AUTO,
  /// Keep each anchor's pairs on its owner when that owner evaluates contact; restricted rank sets fold excluded
  /// owners deterministically onto an evaluation rank.
  ANCHOR_OWNER,
  /// Hash complete anchor groups across evaluation ranks without building a global load table.
  BALANCED_ANCHOR,
  /// Evaluate each pair on the distributed spatial-search rank that found it.
  SPATIAL_PARTITION
};

/**
 * @brief Non-owning definition of one MFEM contact surface
 *
 * Registered MFEM objects must outlive the coupling scheme. Boundary
 * attributes select exterior faces directly from the parent mesh. Optional
 * nodal and quadrature fields remain on the parent mesh and use the same
 * sparse face requests; no submesh stages geometry or field values.
 */
class MfemContactSurface {
 public:
  /**
   * @brief Describe one contact boundary on a caller-owned MFEM parent mesh
   *
   * @param mesh_id Tribol mesh ID reserved for this surface's transient patch view
   * @param mesh Parent volume ParMesh
   * @param coordinates Current physical coordinates on mesh, with one component per physical dimension
   * @param boundary_attributes Parent boundary attributes included in this contact surface
   */
  MfemContactSurface( IndexT mesh_id, const mfem::ParMesh& mesh, const mfem::ParGridFunction& coordinates,
                      std::set<int> boundary_attributes )
      : mesh_id_{ mesh_id },
        mesh_{ &mesh },
        coordinates_{ &coordinates },
        boundary_attributes_{ std::move( boundary_attributes ) }
  {
  }

  /** @brief Bind current physical velocity; the field may use a different fixed nodal H1 order than coordinates */
  MfemContactSurface& SetVelocity( const mfem::ParGridFunction& velocity )
  {
    velocity_ = &velocity;
    return *this;
  }

  /** @brief Bind parent/reference coordinates; the field may use a different fixed nodal H1 order than coordinates */
  MfemContactSurface& SetReferenceCoordinates( const mfem::ParGridFunction& reference_coordinates )
  {
    reference_coordinates_ = &reference_coordinates;
    return *this;
  }

  /** @brief Bind one scalar thickness value per parent volume element */
  MfemContactSurface& SetElementThickness( const mfem::QuadratureFunction& thickness )
  {
    element_thickness_ = &thickness;
    return *this;
  }

  /** @brief Bind one scalar material-modulus value per parent volume element */
  MfemContactSurface& SetMaterialModulus( const mfem::QuadratureFunction& modulus )
  {
    material_modulus_ = &modulus;
    return *this;
  }

  IndexT MeshId() const { return mesh_id_; }
  const mfem::ParMesh& Mesh() const { return *mesh_; }
  const mfem::ParGridFunction& Coordinates() const { return *coordinates_; }
  const std::set<int>& BoundaryAttributes() const { return boundary_attributes_; }
  const mfem::ParGridFunction* Velocity() const { return velocity_; }
  const mfem::ParGridFunction* ReferenceCoordinates() const { return reference_coordinates_; }
  const mfem::QuadratureFunction* ElementThickness() const { return element_thickness_; }
  const mfem::QuadratureFunction* MaterialModulus() const { return material_modulus_; }

 private:
  IndexT mesh_id_;
  const mfem::ParMesh* mesh_;
  const mfem::ParGridFunction* coordinates_;
  std::set<int> boundary_attributes_;
  const mfem::ParGridFunction* velocity_{ nullptr };
  const mfem::ParGridFunction* reference_coordinates_{ nullptr };
  const mfem::QuadratureFunction* element_thickness_{ nullptr };
  const mfem::QuadratureFunction* material_modulus_{ nullptr };
};

/** @brief Configuration fixed when an MFEM contact scheme is registered */
struct MfemContactOptions {
  static constexpr RealT AUTO_MOTION_SKIN = -1.0;

  /// AUTO selects owner placement for asymmetric mortar, balanced placement for symmetric contact, and spatial
  /// placement for self-contact.
  MfemContactPlacement placement{ MfemContactPlacement::AUTO };
  /// Number of lowest communicator ranks available for evaluation; zero uses every rank.
  int evaluation_rank_count{ 0 };
  /// Uniform child-facet count per parent direction; zero uses the coordinate-space order.
  int refinement_factor{ 0 };
  /// Physical-distance rebuild threshold; AUTO uses one quarter of the global maximum child-facet diameter.
  RealT motion_skin{ AUTO_MOTION_SKIN };
  /// Tribol execution policy for local contact calculations; device modes are rejected until patch buffers are
  /// device-resident.
  ExecutionMode execution_mode{ ExecutionMode::Dynamic };
  /// Storage location for Tribol's non-owning patch-array views; currently Host or Dynamic only.
  MemorySpace memory_space{ MemorySpace::Dynamic };
};

}  // namespace tribol

#endif  // BUILD_REDECOMP

#endif  // SRC_TRIBOL_INTERFACE_MFEMCONTACT_HPP_
