// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_SELECTEDBOUNDARYRESTRICTION_HPP_
#define SRC_REDECOMP_SELECTEDBOUNDARYRESTRICTION_HPP_

#include "mfem.hpp"

namespace redecomp {

/**
 * @brief Restrict parallel true DOFs to selected boundary-element E-vectors
 *
 * MFEM supplies signed boundary VDofs and an optional orientation transform
 * for each boundary element. Applying those operations around the parallel
 * prolongation makes the transpose add contact duals to the correct source
 * true DOFs, including constrained and remotely owned DOFs.
 */
class SelectedBoundaryRestriction : public mfem::Operator {
 public:
  SelectedBoundaryRestriction( const mfem::ParFiniteElementSpace& source_space,
                               mfem::Array<int> selected_boundary_elements );

  void Mult( const mfem::Vector& true_dofs, mfem::Vector& selected_element_values ) const override;
  void MultTranspose( const mfem::Vector& selected_element_dual, mfem::Vector& true_dof_dual ) const override;
  void AddMultTranspose( const mfem::Vector& selected_element_dual, mfem::Vector& true_dof_dual,
                         mfem::real_t scale = 1.0 ) const override;

  const mfem::Array<int>& SelectedBoundaryElements() const { return selected_boundary_elements_; }
  const mfem::Array<int>& SelectedElementOffsets() const { return selected_element_offsets_; }
  /** @brief Assemble the selected boundary E-vector map from rank-local VDofs */
  mfem::SparseMatrix AssembleLocal() const;

 private:
  void BuildElementOffsets();
  void GatherBoundaryElement( int selected_element, mfem::Vector& element_values ) const;
  void AddBoundaryElementDual( int selected_element, const mfem::Vector& element_dual ) const;

  const mfem::ParFiniteElementSpace& source_space_;
  const mfem::Operator& prolongation_;
  mfem::Array<int> selected_boundary_elements_;
  mfem::Array<int> selected_element_offsets_;
  mutable mfem::Vector local_values_;
  mutable mfem::Vector local_dual_;
  mutable mfem::Vector returned_true_dual_;
  mutable mfem::Array<int> boundary_vdofs_;
  mutable mfem::Vector element_buffer_;
};

}  // namespace redecomp

#endif  // SRC_REDECOMP_SELECTEDBOUNDARYRESTRICTION_HPP_
