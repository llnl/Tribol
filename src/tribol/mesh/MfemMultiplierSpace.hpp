// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_MESH_MFEMMULTIPLIERSPACE_HPP_
#define SRC_TRIBOL_MESH_MFEMMULTIPLIERSPACE_HPP_

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP

#include <memory>
#include <vector>

#include "mfem.hpp"

#include "redecomp/ElementPatchOperator.hpp"
#include "shared/math/ParSparseMat.hpp"
#include "tribol/interface/MfemContact.hpp"

namespace tribol {

/**
 * @brief Solver-visible mortar multiplier state on the designated dual surface
 *
 * The ParSubMesh exists only to define the multiplier finite element space.
 * Pressure and gap are canonical true-DOF vectors; neither uses the submesh as
 * a geometry or Tribol-field staging area. The registered parent mesh must
 * outlive this object. BuildPatchOperator() replaces the cached schedule after
 * a contact-patch rebuild and invalidates the lazily assembled sparse matrix.
 */
class MfemMultiplierSpace {
 public:
  explicit MfemMultiplierSpace( const MfemContactSurface& dual_surface );

  void BuildPatchOperator( mfem::Array<int> selected_parent_boundary_elements,
                           std::vector<mfem::Array<int>> elements_by_destination,
                           std::vector<redecomp::VirtualFaceSpecification> destination_faces,
                           mfem::Array<int> occurrence_to_node, int num_patch_nodes );
  /** @brief Gather canonical pressure true DOFs and overwrite patch pressure values */
  void GatherPressure();

  /** @brief Overwrite canonical gap true DOFs with the additive patch-gap transpose */
  void ReturnGap();

  mfem::ParFiniteElementSpace& Space() { return *space_; }
  const mfem::ParFiniteElementSpace& Space() const { return *space_; }
  mfem::Vector& Pressure() { return pressure_true_dofs_; }
  const mfem::Vector& Pressure() const { return pressure_true_dofs_; }
  const mfem::Vector& Gap() const { return gap_true_dofs_; }
  mfem::Vector& PatchPressure() { return patch_pressure_; }
  mfem::Vector& PatchGap() { return patch_gap_; }
  /** @brief Return the lazily assembled true-DOF-to-patch operator owned by this object */
  const shared::ParSparseMat& AssembledPatchOperator();

 private:
  std::unique_ptr<mfem::ParSubMesh> submesh_;
  std::unique_ptr<mfem::FiniteElementCollection> collection_;
  std::unique_ptr<mfem::ParFiniteElementSpace> space_;
  std::unique_ptr<redecomp::ElementPatchOperator> patch_operator_;
  std::unique_ptr<shared::ParSparseMat> assembled_patch_operator_;
  mfem::Vector pressure_true_dofs_;
  mfem::Vector gap_true_dofs_;
  mfem::Vector patch_pressure_;
  mfem::Vector patch_gap_;
};

}  // namespace tribol

#endif  // BUILD_REDECOMP

#endif  // SRC_TRIBOL_MESH_MFEMMULTIPLIERSPACE_HPP_
