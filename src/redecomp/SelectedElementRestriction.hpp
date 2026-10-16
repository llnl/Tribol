// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_SELECTEDELEMENTRESTRICTION_HPP_
#define SRC_REDECOMP_SELECTEDELEMENTRESTRICTION_HPP_

#include "mfem.hpp"

namespace redecomp {

/**
 * @brief Restrict parallel true DOFs to element-local values for selected elements
 *
 * MFEM's prolongation and element restriction perform the framework-specific
 * constraint and orientation work. This class only selects complete element
 * blocks from the resulting E-vector and applies the exact transpose on return.
 */
class SelectedElementRestriction : public mfem::Operator {
 public:
  SelectedElementRestriction( const mfem::ParFiniteElementSpace& source_space, mfem::Array<int> selected_elements,
                              mfem::ElementDofOrdering element_ordering = mfem::ElementDofOrdering::NATIVE );

  void Mult( const mfem::Vector& true_dofs, mfem::Vector& selected_element_values ) const override;
  void MultTranspose( const mfem::Vector& selected_element_dual, mfem::Vector& true_dof_dual ) const override;
  void AddMultTranspose( const mfem::Vector& selected_element_dual, mfem::Vector& true_dof_dual,
                         const mfem::real_t scale = 1.0 ) const override;

  const mfem::Array<int>& SelectedElements() const { return selected_elements_; }
  const mfem::Array<int>& SelectedElementOffsets() const { return selected_element_offsets_; }
  /** @brief Assemble the selected element E-vector map from rank-local VDofs */
  mfem::SparseMatrix AssembleLocal() const;

 private:
  void ValidateUniformElementLayout();
  void GatherSelectedBlocks( const mfem::Vector& all_element_values, mfem::Vector& selected_values ) const;
  void InjectSelectedBlocks( const mfem::Vector& selected_values, mfem::Vector& all_element_values,
                             mfem::real_t scale ) const;

  const mfem::ParFiniteElementSpace& source_space_;
  const mfem::Operator& prolongation_;
  const mfem::ElementRestrictionOperator& element_restriction_;
  mfem::Array<int> selected_elements_;
  mfem::Array<int> selected_element_offsets_;
  int element_block_size_{ 0 };
  mutable mfem::Vector local_values_;
  mutable mfem::Vector all_element_values_;
  mutable mfem::Vector local_dual_;
  mutable mfem::Vector all_element_dual_;
  mutable mfem::Vector returned_true_dual_;
};

}  // namespace redecomp

#endif  // SRC_REDECOMP_SELECTEDELEMENTRESTRICTION_HPP_
