// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_PAIR_ITERATOR_HPP_
#define SRC_TRIBOL_SEARCH_PAIR_ITERATOR_HPP_

#include "tribol/common/Arrays.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

class AllPairIterator {
 public:
  AllPairIterator( const MeshData& mesh1, const MeshData& mesh2 );

  size_t numPairs() const { return num_pairs_; }

  TRIBOL_HOST_DEVICE std::pair<size_t, size_t> getPair( size_t idx ) const;

 private:
  bool repeated_mesh_;
  size_t mesh1_num_elems_;
  size_t mesh2_num_elems_;
  size_t num_pairs_;
};

class ListIterator {
 public:
  ListIterator( Array<size_t>::view_type candidates, Array<size_t>::view_type offsets,
                Array<size_t>::view_type counts );

  size_t numPairs() const { return candidates_.size(); }

  TRIBOL_HOST_DEVICE std::pair<size_t, size_t> getPair( size_t idx ) const;

 private:
  Array<size_t>::view_type candidates_;
  Array<size_t>::view_type offsets_;
  Array<size_t>::view_type counts_;
};

}  // end namespace tribol

#endif /* SRC_TRIBOL_SEARCH_PAIR_ITERATOR_HPP_ */
