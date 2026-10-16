// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_VIRTUALREFINEMENTCOLLECTIONOPERATOR_HPP_
#define SRC_REDECOMP_VIRTUALREFINEMENTCOLLECTIONOPERATOR_HPP_

#include <memory>
#include <vector>

#include "mfem.hpp"

#include "redecomp/VirtualRefinementOperator.hpp"

namespace redecomp {

/** @brief Basis and refinement information for one exchanged surface element */
struct VirtualFaceSpecification {
  mfem::Geometry::Type geometry;
  int order;
  int refinement_factor;
  int basis_type{ mfem::BasisType::GaussLobatto };
};

/**
 * @brief Apply virtual refinement independently to variable-size surface-element blocks
 *
 * The input follows the exchanged element-block layout. The output is one
 * component-major array of repeated child-facet vertices, which can be passed
 * directly to PatchNodalLayoutOperator.
 */
class VirtualRefinementCollectionOperator : public mfem::Operator {
 public:
  VirtualRefinementCollectionOperator( const mfem::Array<int>& source_element_offsets,
                                       std::vector<VirtualFaceSpecification> face_specifications, int vector_dimension,
                                       mfem::Ordering::Type source_ordering = mfem::Ordering::byNODES );

  void Mult( const mfem::Vector& source_element_values, mfem::Vector& repeated_child_values ) const override;
  void MultTranspose( const mfem::Vector& repeated_child_dual, mfem::Vector& source_element_dual ) const override;
  void AddMultTranspose( const mfem::Vector& repeated_child_dual, mfem::Vector& source_element_dual,
                         mfem::real_t scale = 1.0 ) const override;

  int NumFaces() const { return static_cast<int>( refinements_.size() ); }
  int NumOccurrences() const { return occurrence_offsets_.Last(); }
  const mfem::Array<int>& OccurrenceOffsets() const { return occurrence_offsets_; }
  const VirtualRefinementOperator& FaceRefinement( int face ) const { return *refinements_[face]; }

 private:
  mfem::Array<int> source_element_offsets_;
  mfem::Array<int> occurrence_offsets_;
  int vector_dimension_;
  std::vector<std::unique_ptr<mfem::FiniteElementCollection>> collections_;
  std::vector<std::unique_ptr<VirtualRefinementOperator>> refinements_;
  mutable mfem::Vector source_face_buffer_;
  mutable mfem::Vector refined_face_buffer_;
};

}  // namespace redecomp

#endif  // SRC_REDECOMP_VIRTUALREFINEMENTCOLLECTIONOPERATOR_HPP_
