// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/search/PairIterator.hpp"

#include "tribol/utils/Algorithm.hpp"

namespace tribol {

AllPairIterator::AllPairIterator( const MeshData& mesh1, const MeshData& mesh2 )
    : repeated_mesh_( &mesh1 == &mesh2 ),
      mesh1_num_elems_( mesh1.numberOfElements() ),
      mesh2_num_elems_( mesh2.numberOfElements() ),
      num_pairs_( repeated_mesh_ ? mesh1_num_elems_ * ( mesh1_num_elems_ + 1 ) / 2
                                 : mesh1_num_elems_ * mesh2_num_elems_ )
{
}

TRIBOL_HOST_DEVICE std::pair<size_t, size_t> AllPairIterator::getPair( size_t idx ) const
{
#ifndef TRIBOL_DEVICE_CODE
  if ( idx >= num_pairs_ ) {
    SLIC_ERROR( "Invalid index for pair iterator" );
  }
#endif

  if ( repeated_mesh_ ) {
    size_t row = algorithm::symmMatrixRow( idx, mesh1_num_elems_ );
    size_t offset = row * ( row + 1 ) / 2;
    return std::make_pair( row, idx - offset );
  } else {
    return std::make_pair( idx / mesh2_num_elems_, idx % mesh2_num_elems_ );
  }
}

ListIterator::ListIterator( Array<size_t>::view_type candidates, Array<size_t>::view_type offsets,
                            Array<size_t>::view_type counts )
    : candidates_( candidates ), offsets_( offsets ), counts_( counts )
{
}

TRIBOL_HOST_DEVICE std::pair<size_t, size_t> ListIterator::getPair( size_t idx ) const
{
#ifndef TRIBOL_DEVICE_CODE
  if ( idx >= numPairs() ) {
    SLIC_ERROR( "Invalid index for pair iterator" );
  }
#endif

  return std::make_pair( algorithm::binarySearch( offsets_, counts_, idx ), candidates_[idx] );
}

}  // end namespace tribol
