// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_VIRTUALREFINEMENTOPERATOR_HPP_
#define SRC_REDECOMP_VIRTUALREFINEMENTOPERATOR_HPP_

#include "mfem.hpp"

namespace redecomp {

/**
 * @brief Evaluate one high-order nodal element on uniformly refined linear child elements
 *
 * The output stores a separate value for every child-element vertex. Shared
 * child vertices remain repeated so a later patch-layout operator can choose
 * deterministic representatives without changing element-local contact data.
 * Segment points use MFEM's [0,1] parent interval; triangle points use
 * @f$x >= 0, y >= 0, x+y <= 1@f$; quadrilateral points use [0,1]^2.
 *
 * This operator uses MFEM's high-order nodal basis directly because every
 * destination point is already known in parent-reference coordinates.
 * Tribol's InvIso/GalerkinEval path instead maps physical points to linear
 * contact elements and therefore does not preserve a parent high-order basis.
 * Inputs use the source element's MFEM E-vector ordering. Outputs use
 * component-major storage with one entry for every repeated child vertex.
 */
class VirtualRefinementOperator : public mfem::Operator {
 public:
  VirtualRefinementOperator( const mfem::FiniteElement& source_element, int vector_dimension, int refinement_factor,
                             mfem::Ordering::Type source_ordering = mfem::Ordering::byNODES );

  /** @brief Overwrite child-vertex primal values by evaluating the parent nodal basis */
  void Mult( const mfem::Vector& source_element_values, mfem::Vector& child_vertex_values ) const override;

  /** @brief Overwrite the parent E-vector dual with the exact basis-evaluation transpose */
  void MultTranspose( const mfem::Vector& child_vertex_dual, mfem::Vector& source_element_dual ) const override;
  void AddMultTranspose( const mfem::Vector& child_vertex_dual, mfem::Vector& source_element_dual,
                         const mfem::real_t scale = 1.0 ) const override;

  int NumChildren() const { return num_children_; }
  int VerticesPerChild() const { return vertices_per_child_; }
  int RefinementFactor() const { return refinement_factor_; }
  const mfem::IntegrationRule& ChildVertexReferencePoints() const { return child_vertex_reference_points_; }

 private:
  void BuildChildVertexReferencePoints();
  void BuildEvaluationMatrix();

  const mfem::FiniteElement& source_element_;
  int vector_dimension_{ 1 };
  mfem::Ordering::Type source_ordering_{ mfem::Ordering::byVDIM };
  int refinement_factor_{ 1 };
  int num_children_{ 0 };
  int vertices_per_child_{ 0 };
  mfem::IntegrationRule child_vertex_reference_points_;
  mfem::DenseMatrix evaluation_matrix_;
};

}  // namespace redecomp

#endif  // SRC_REDECOMP_VIRTUALREFINEMENTOPERATOR_HPP_
