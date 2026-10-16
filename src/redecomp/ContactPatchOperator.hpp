// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_CONTACTPATCHOPERATOR_HPP_
#define SRC_REDECOMP_CONTACTPATCHOPERATOR_HPP_

#include <vector>

#include "mfem.hpp"

#include "redecomp/DistributedBoundaryRestriction.hpp"
#include "redecomp/PatchNodalLayoutOperator.hpp"
#include "redecomp/VirtualRefinementCollectionOperator.hpp"
#include "shared/math/ParSparseMat.hpp"

namespace redecomp {

/**
 * @brief Matrix-free true-DOF to contact-patch operator @f$H = C B X R@f$
 *
 * `Mult()` gathers a primal field to Tribol's unique virtual nodes. The exact
 * transpose adds independent nodal contact duals directly to source true DOFs.
 * The source is a caller-owned parent ParFiniteElementSpace; the destination
 * is a mesh-free, component-major patch-node vector on each evaluation rank.
 * MFEM prolongation, constraints, signed boundary VDofs, and orientation
 * transforms are included before MPI exchange. The object owns its immutable
 * schedule and work buffers but does not own source_space.
 */
class ContactPatchOperator : public mfem::Operator {
 public:
  ContactPatchOperator( const mfem::ParFiniteElementSpace& source_space, mfem::Array<int> selected_boundary_elements,
                        std::vector<mfem::Array<int>> selected_elements_by_destination,
                        std::vector<VirtualFaceSpecification> destination_faces, mfem::Array<int> occurrence_to_node,
                        int num_patch_nodes );

  void Mult( const mfem::Vector& source_true_dofs, mfem::Vector& patch_values ) const override;
  void MultTranspose( const mfem::Vector& patch_dual, mfem::Vector& source_true_dof_dual ) const override;
  void AddMultTranspose( const mfem::Vector& patch_dual, mfem::Vector& source_true_dof_dual,
                         mfem::real_t scale = 1.0 ) const override;

  /** @brief Assemble the matrix-free action when solver-visible composition requires it */
  shared::ParSparseMat Assemble() const;

  const ElementExchangePlan& ExchangePlan() const { return restriction_.Plan(); }
  const VirtualRefinementCollectionOperator& Refinement() const { return refinement_; }
  const PatchNodalLayoutOperator& NodalLayout() const { return nodal_layout_; }

 private:
  const mfem::ParFiniteElementSpace& source_space_;
  DistributedBoundaryRestriction restriction_;
  VirtualRefinementCollectionOperator refinement_;
  PatchNodalLayoutOperator nodal_layout_;
  mutable mfem::Vector exchanged_values_;
  mutable mfem::Vector repeated_values_;
  mutable mfem::Vector repeated_dual_;
  mutable mfem::Vector exchanged_dual_;
  mutable mfem::Vector returned_true_dual_;
};

}  // namespace redecomp

#endif  // SRC_REDECOMP_CONTACTPATCHOPERATOR_HPP_
