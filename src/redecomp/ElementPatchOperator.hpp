// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_ELEMENTPATCHOPERATOR_HPP_
#define SRC_REDECOMP_ELEMENTPATCHOPERATOR_HPP_

#include <vector>

#include "mfem.hpp"

#include "redecomp/DistributedElementRestriction.hpp"
#include "redecomp/PatchNodalLayoutOperator.hpp"
#include "redecomp/VirtualRefinementCollectionOperator.hpp"
#include "shared/math/ParSparseMat.hpp"

namespace redecomp {

/**
 * @brief Map true DOFs on a caller-owned ParSubMesh space to mesh-free patch nodes
 *
 * This variant supports the solver-visible multiplier space whose elements
 * already are surface elements. MultTranspose() returns independent patch
 * duals additively through MFEM's true-DOF restriction.
 */
class ElementPatchOperator : public mfem::Operator {
 public:
  ElementPatchOperator( const mfem::ParFiniteElementSpace& source_space, mfem::Array<int> selected_elements,
                        std::vector<mfem::Array<int>> selected_elements_by_destination,
                        std::vector<VirtualFaceSpecification> destination_faces, mfem::Array<int> occurrence_to_node,
                        int num_patch_nodes );

  void Mult( const mfem::Vector& source_true_dofs, mfem::Vector& patch_values ) const override;
  void MultTranspose( const mfem::Vector& patch_dual, mfem::Vector& source_true_dof_dual ) const override;
  void AddMultTranspose( const mfem::Vector& patch_dual, mfem::Vector& source_true_dof_dual,
                         mfem::real_t scale = 1.0 ) const override;

  /** @brief Assemble the matrix-free action when solver-visible composition requires it */
  shared::ParSparseMat Assemble() const;

 private:
  const mfem::ParFiniteElementSpace& source_space_;
  DistributedElementRestriction restriction_;
  VirtualRefinementCollectionOperator refinement_;
  PatchNodalLayoutOperator nodal_layout_;
  mutable mfem::Vector exchanged_values_;
  mutable mfem::Vector repeated_values_;
  mutable mfem::Vector repeated_dual_;
  mutable mfem::Vector exchanged_dual_;
  mutable mfem::Vector returned_true_dual_;
};

}  // namespace redecomp

#endif  // SRC_REDECOMP_ELEMENTPATCHOPERATOR_HPP_
