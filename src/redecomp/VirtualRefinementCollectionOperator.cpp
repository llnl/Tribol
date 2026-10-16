// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/VirtualRefinementCollectionOperator.hpp"

#include <algorithm>
#include <utility>

#include "axom/slic.hpp"

namespace redecomp {

VirtualRefinementCollectionOperator::VirtualRefinementCollectionOperator(
    const mfem::Array<int>& source_element_offsets, std::vector<VirtualFaceSpecification> face_specifications,
    int vector_dimension, mfem::Ordering::Type source_ordering )
    : mfem::Operator(),
      source_element_offsets_{ source_element_offsets },
      occurrence_offsets_( static_cast<int>( face_specifications.size() ) + 1 ),
      vector_dimension_{ vector_dimension }
{
  SLIC_ERROR_ROOT_IF( source_element_offsets_.Size() != static_cast<int>( face_specifications.size() ) + 1,
                      "Virtual refinement requires one source block for each face specification." );
  SLIC_ERROR_ROOT_IF( vector_dimension_ <= 0, "Virtual refinement requires a positive vector dimension." );
  occurrence_offsets_[0] = 0;
  collections_.reserve( face_specifications.size() );
  refinements_.reserve( face_specifications.size() );
  const int face_dimension =
      face_specifications.empty() ? 1 : mfem::Geometry::Dimension[face_specifications.front().geometry];
  for ( int face = 0; face < static_cast<int>( face_specifications.size() ); ++face ) {
    const auto& specification = face_specifications[face];
    SLIC_ERROR_ROOT_IF( mfem::Geometry::Dimension[specification.geometry] != face_dimension,
                        "A virtual surface patch cannot mix topological dimensions." );
    collections_.push_back(
        std::make_unique<mfem::H1_FECollection>( specification.order, face_dimension, specification.basis_type ) );
    const auto* element = collections_.back()->FiniteElementForGeometry( specification.geometry );
    SLIC_ERROR_ROOT_IF( element == nullptr, "MFEM did not provide the requested virtual-face finite element." );
    refinements_.push_back( std::make_unique<VirtualRefinementOperator>(
        *element, vector_dimension_, specification.refinement_factor, source_ordering ) );
    const int source_block_size = source_element_offsets_[face + 1] - source_element_offsets_[face];
    SLIC_ERROR_ROOT_IF( source_block_size != refinements_.back()->Width(),
                        "Exchanged boundary block does not match its virtual-face finite element." );
    const int face_occurrences = refinements_.back()->Height() / vector_dimension_;
    occurrence_offsets_[face + 1] = occurrence_offsets_[face] + face_occurrences;
  }
  width = source_element_offsets_.Last();
  height = occurrence_offsets_.Last() * vector_dimension_;
}

void VirtualRefinementCollectionOperator::Mult( const mfem::Vector& source_element_values,
                                                mfem::Vector& repeated_child_values ) const
{
  SLIC_ERROR_ROOT_IF( source_element_values.Size() != Width(), "Virtual refinement source size mismatch." );
  SLIC_ERROR_ROOT_IF( repeated_child_values.Size() != Height(), "Virtual refinement destination size mismatch." );
  const int total_occurrences = occurrence_offsets_.Last();
  for ( int face = 0; face < NumFaces(); ++face ) {
    const int source_offset = source_element_offsets_[face];
    const int source_size = source_element_offsets_[face + 1] - source_offset;
    mfem::Vector source_face_view( const_cast<mfem::real_t*>( source_element_values.HostRead() ) + source_offset,
                                   source_size );
    refined_face_buffer_.SetSize( refinements_[face]->Height() );
    refinements_[face]->Mult( source_face_view, refined_face_buffer_ );
    const int face_occurrences = occurrence_offsets_[face + 1] - occurrence_offsets_[face];
    for ( int component = 0; component < vector_dimension_; ++component ) {
      std::copy_n( refined_face_buffer_.HostRead() + component * face_occurrences, face_occurrences,
                   repeated_child_values.HostWrite() + component * total_occurrences + occurrence_offsets_[face] );
    }
  }
}

void VirtualRefinementCollectionOperator::MultTranspose( const mfem::Vector& repeated_child_dual,
                                                         mfem::Vector& source_element_dual ) const
{
  source_element_dual = 0.0;
  AddMultTranspose( repeated_child_dual, source_element_dual );
}

void VirtualRefinementCollectionOperator::AddMultTranspose( const mfem::Vector& repeated_child_dual,
                                                            mfem::Vector& source_element_dual,
                                                            mfem::real_t scale ) const
{
  SLIC_ERROR_ROOT_IF( repeated_child_dual.Size() != Height(), "Virtual refinement dual size mismatch." );
  SLIC_ERROR_ROOT_IF( source_element_dual.Size() != Width(), "Virtual refinement source dual size mismatch." );
  const int total_occurrences = occurrence_offsets_.Last();
  for ( int face = 0; face < NumFaces(); ++face ) {
    const int face_occurrences = occurrence_offsets_[face + 1] - occurrence_offsets_[face];
    refined_face_buffer_.SetSize( face_occurrences * vector_dimension_ );
    for ( int component = 0; component < vector_dimension_; ++component ) {
      std::copy_n( repeated_child_dual.HostRead() + component * total_occurrences + occurrence_offsets_[face],
                   face_occurrences, refined_face_buffer_.HostWrite() + component * face_occurrences );
    }
    source_face_buffer_.SetSize( refinements_[face]->Width() );
    refinements_[face]->MultTranspose( refined_face_buffer_, source_face_buffer_ );
    const int source_offset = source_element_offsets_[face];
    for ( int i = 0; i < source_face_buffer_.Size(); ++i ) {
      source_element_dual[source_offset + i] += scale * source_face_buffer_[i];
    }
  }
}

}  // namespace redecomp
