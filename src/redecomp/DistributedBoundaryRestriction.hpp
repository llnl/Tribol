// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_DISTRIBUTEDBOUNDARYRESTRICTION_HPP_
#define SRC_REDECOMP_DISTRIBUTEDBOUNDARYRESTRICTION_HPP_

#include <memory>
#include <vector>

#include "mfem.hpp"

#include "redecomp/ElementExchangeOperator.hpp"
#include "redecomp/SelectedBoundaryRestriction.hpp"

namespace redecomp {

/** @brief Gather selected MFEM boundary E-vectors and add returned duals to source true DOFs */
class DistributedBoundaryRestriction : public mfem::Operator {
 public:
  DistributedBoundaryRestriction( const mfem::ParFiniteElementSpace& source_space,
                                  mfem::Array<int> selected_boundary_elements,
                                  std::vector<mfem::Array<int>> selected_elements_by_destination );

  void Mult( const mfem::Vector& true_dofs, mfem::Vector& destination_element_values ) const override;
  void MultTranspose( const mfem::Vector& destination_element_dual, mfem::Vector& true_dof_dual ) const override;
  void AddMultTranspose( const mfem::Vector& destination_element_dual, mfem::Vector& true_dof_dual,
                         mfem::real_t scale = 1.0 ) const override;

  const ElementExchangePlan& Plan() const { return exchange_.Plan(); }
  const ElementExchangeOperator& Exchange() const { return exchange_; }
  const SelectedBoundaryRestriction& LocalRestriction() const { return local_restriction_; }

 private:
  SelectedBoundaryRestriction local_restriction_;
  std::shared_ptr<const ElementExchangePlan> plan_;
  ElementExchangeOperator exchange_;
  mutable mfem::Vector selected_values_;
  mutable mfem::Vector selected_dual_;
  mutable mfem::Vector returned_true_dual_;
};

}  // namespace redecomp

#endif  // SRC_REDECOMP_DISTRIBUTEDBOUNDARYRESTRICTION_HPP_
