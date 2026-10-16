// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/PatchNodalLayoutOperator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "axom/slic.hpp"

namespace redecomp {

PatchNodalLayoutOperator::PatchNodalLayoutOperator( mfem::Array<int> occurrence_to_node, int num_nodes,
                                                    int vector_dimension )
    : mfem::Operator( num_nodes * vector_dimension, occurrence_to_node.Size() * vector_dimension ),
      occurrence_to_node_{ std::move( occurrence_to_node ) },
      designated_occurrences_( num_nodes ),
      num_nodes_{ num_nodes },
      vector_dimension_{ vector_dimension }
{
  SLIC_ERROR_ROOT_IF( num_nodes_ < 0, "PatchNodalLayoutOperator requires a nonnegative node count." );
  SLIC_ERROR_ROOT_IF( vector_dimension_ <= 0, "PatchNodalLayoutOperator requires a positive vector dimension." );
  ChooseDesignatedOccurrences();
}

void PatchNodalLayoutOperator::ChooseDesignatedOccurrences()
{
  designated_occurrences_ = -1;
  for ( int occurrence = 0; occurrence < occurrence_to_node_.Size(); ++occurrence ) {
    const int node = occurrence_to_node_[occurrence];
    SLIC_ERROR_ROOT_IF( node < 0 || node >= num_nodes_,
                        "PatchNodalLayoutOperator contains an out-of-range patch node." );
    if ( designated_occurrences_[node] < 0 ) {
      designated_occurrences_[node] = occurrence;
    }
  }
  for ( int node = 0; node < num_nodes_; ++node ) {
    SLIC_ERROR_ROOT_IF( designated_occurrences_[node] < 0,
                        "PatchNodalLayoutOperator contains a patch node with no element occurrence." );
  }
}

void PatchNodalLayoutOperator::VerifyRepeatedValuesAgree( const mfem::Vector& repeated_values ) const
{
#if defined( TRIBOL_DEBUG )
  constexpr double tolerance_multiplier = 64.0;
  for ( int component = 0; component < vector_dimension_; ++component ) {
    for ( int occurrence = 0; occurrence < NumOccurrences(); ++occurrence ) {
      const int node = occurrence_to_node_[occurrence];
      const int designated = designated_occurrences_[node];
      const double selected = repeated_values[designated + component * NumOccurrences()];
      const double candidate = repeated_values[occurrence + component * NumOccurrences()];
      const double scale = std::max( { 1.0, std::abs( selected ), std::abs( candidate ) } );
      const double tolerance = tolerance_multiplier * std::numeric_limits<double>::epsilon() * scale;
      SLIC_ERROR_ROOT_IF( std::abs( selected - candidate ) > tolerance,
                          "Repeated virtual-node values disagree for patch node "
                              << node << ", component " << component << ": designated occurrence " << designated
                              << " has value " << selected << " while occurrence " << occurrence << " has value "
                              << candidate << "." );
    }
  }
#else
  static_cast<void>( repeated_values );
#endif
}

void PatchNodalLayoutOperator::Mult( const mfem::Vector& repeated_values, mfem::Vector& nodal_values ) const
{
  SLIC_ERROR_ROOT_IF( repeated_values.Size() != Width(), "Patch repeated-value size mismatch." );
  SLIC_ERROR_ROOT_IF( nodal_values.Size() != Height(), "Patch nodal-value size mismatch." );
  VerifyRepeatedValuesAgree( repeated_values );
  for ( int component = 0; component < vector_dimension_; ++component ) {
    for ( int node = 0; node < num_nodes_; ++node ) {
      nodal_values[node + component * num_nodes_] =
          repeated_values[designated_occurrences_[node] + component * NumOccurrences()];
    }
  }
}

void PatchNodalLayoutOperator::MultTranspose( const mfem::Vector& nodal_dual, mfem::Vector& repeated_dual ) const
{
  repeated_dual = 0.0;
  AddMultTranspose( nodal_dual, repeated_dual );
}

void PatchNodalLayoutOperator::AddMultTranspose( const mfem::Vector& nodal_dual, mfem::Vector& repeated_dual,
                                                 mfem::real_t scale ) const
{
  SLIC_ERROR_ROOT_IF( nodal_dual.Size() != Height(), "Patch nodal-dual size mismatch." );
  SLIC_ERROR_ROOT_IF( repeated_dual.Size() != Width(), "Patch repeated-dual size mismatch." );
  for ( int component = 0; component < vector_dimension_; ++component ) {
    for ( int node = 0; node < num_nodes_; ++node ) {
      repeated_dual[designated_occurrences_[node] + component * NumOccurrences()] +=
          scale * nodal_dual[node + component * num_nodes_];
    }
  }
}

}  // namespace redecomp
