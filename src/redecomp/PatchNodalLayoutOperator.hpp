// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_PATCHNODALLAYOUTOPERATOR_HPP_
#define SRC_REDECOMP_PATCHNODALLAYOUTOPERATOR_HPP_

#include "mfem.hpp"

namespace redecomp {

/**
 * @brief Collapse repeated element vertices into Tribol's unique patch nodes
 *
 * Input and output vectors use component-major storage. The forward action
 * selects one deterministic occurrence for each patch node. Its transpose
 * returns each nodal dual through that same occurrence exactly once, so a node
 * repeated by adjacent child facets does not multiply the physical force.
 * Mult() overwrites nodal_values. MultTranspose() overwrites repeated_dual;
 * AddMultTranspose() accumulates into it. Debug builds also verify that all
 * repeated primal occurrences agree within roundoff.
 */
class PatchNodalLayoutOperator : public mfem::Operator {
 public:
  PatchNodalLayoutOperator( mfem::Array<int> occurrence_to_node, int num_nodes, int vector_dimension );

  void Mult( const mfem::Vector& repeated_values, mfem::Vector& nodal_values ) const override;
  void MultTranspose( const mfem::Vector& nodal_dual, mfem::Vector& repeated_dual ) const override;
  void AddMultTranspose( const mfem::Vector& nodal_dual, mfem::Vector& repeated_dual,
                         mfem::real_t scale = 1.0 ) const override;

  int NumOccurrences() const { return occurrence_to_node_.Size(); }
  int NumNodes() const { return num_nodes_; }
  int VectorDimension() const { return vector_dimension_; }
  const mfem::Array<int>& OccurrenceToNode() const { return occurrence_to_node_; }
  const mfem::Array<int>& DesignatedOccurrences() const { return designated_occurrences_; }

 private:
  void ChooseDesignatedOccurrences();
  void VerifyRepeatedValuesAgree( const mfem::Vector& repeated_values ) const;

  mfem::Array<int> occurrence_to_node_;
  mfem::Array<int> designated_occurrences_;
  int num_nodes_;
  int vector_dimension_;
};

}  // namespace redecomp

#endif  // SRC_REDECOMP_PATCHNODALLAYOUTOPERATOR_HPP_
