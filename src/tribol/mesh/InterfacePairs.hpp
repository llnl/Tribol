// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_MESH_INTERFACE_PAIRS_HPP_
#define SRC_TRIBOL_MESH_INTERFACE_PAIRS_HPP_

// Shared includes
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/utils/Algorithm.hpp"

#include <type_traits>
#include <variant>

namespace tribol {

/**
 * @brief Pair of element indices produced by coarse search
 *
 * Search pairs intentionally contain no contact state. Whether a coarse pair is
 * suitable for a contact method is decided by the consumer.
 */
struct ElementPair {
  TRIBOL_HOST_DEVICE ElementPair( IndexT element_id1, IndexT element_id2 )
      : element_id1( element_id1 ), element_id2( element_id2 )
  {
  }

  TRIBOL_HOST_DEVICE ElementPair() : element_id1( -1 ), element_id2( -1 ) {}

  IndexT element_id1;
  IndexT element_id2;
};

/**
 * @brief Random-access view of explicitly stored coarse-search pairs
 */
class PairListView {
 public:
  PairListView() = default;
  explicit PairListView( ArrayViewT<const ElementPair> pairs ) : pairs_( pairs ) {}

  TRIBOL_HOST_DEVICE IndexT size() const { return pairs_.size(); }
  TRIBOL_HOST_DEVICE ElementPair operator[]( IndexT index ) const { return pairs_[index]; }

 private:
  ArrayViewT<const ElementPair> pairs_;
};

/**
 * @brief Random-access view that generates a Cartesian product lazily
 */
class CartesianPairView {
 public:
  CartesianPairView() = default;
  CartesianPairView( IndexT num_elements1, IndexT num_elements2, bool symmetric )
      : num_elements1_( num_elements1 ), num_elements2_( num_elements2 ), symmetric_( symmetric )
  {
  }

  TRIBOL_HOST_DEVICE IndexT size() const
  {
    return symmetric_ ? num_elements1_ * ( num_elements1_ - 1 ) / 2 : num_elements1_ * num_elements2_;
  }

  /** @brief Return this device-copyable view for generic pair-range consumers. */
  TRIBOL_HOST_DEVICE CartesianPairView view() const { return *this; }

  TRIBOL_HOST_DEVICE ElementPair operator[]( IndexT index ) const
  {
    if ( !symmetric_ ) {
      return { index / num_elements2_, index % num_elements2_ };
    }

    const IndexT row = algorithm::symmMatrixRow( index, num_elements1_ - 1 ) + 1;
    const IndexT offset = row * ( row - 1 ) / 2;
    return { row, index - offset };
  }

 private:
  IndexT num_elements1_{ 0 };
  IndexT num_elements2_{ 0 };
  bool symmetric_{ false };
};

/**
 * @brief Host-side owning storage for a contact-pair range
 *
 * Explicit pairs may reside in device memory. Visit this range on the host to
 * obtain a device-copyable PairListView or CartesianPairView for a kernel.
 */
using ContactPairRange = std::variant<ArrayT<ElementPair>, CartesianPairView>;

template <typename Visitor>
decltype( auto ) visitContactPairs( const ContactPairRange& pairs, Visitor&& visitor )
{
  return std::visit(
      [&visitor]( const auto& pair_storage ) -> decltype( auto ) {
        using PairStorage = std::decay_t<decltype( pair_storage )>;
        if constexpr ( std::is_same_v<PairStorage, ArrayT<ElementPair>> ) {
          return visitor( PairListView( pair_storage.view() ) );
        } else {
          return visitor( pair_storage );
        }
      },
      pairs );
}

struct InterfacePair {
  TRIBOL_HOST_DEVICE InterfacePair( IndexT element_id1, IndexT element_id2, bool is_contact_candidate = true )
      : m_element_id1( element_id1 ), m_element_id2( element_id2 ), m_is_contact_candidate( is_contact_candidate )
  {
  }

  // overload constructor to handle zero input arguments
  TRIBOL_HOST_DEVICE InterfacePair() : m_element_id1( -1 ), m_element_id2( -1 ), m_is_contact_candidate( true ) {}

  // Element id for face 1
  IndexT m_element_id1;

  // Element id for face 2
  IndexT m_element_id2;

  // boolean indicating if a binned pair is a contact candidate.
  // A contact candidate is defined as a face-pair that is deemed geometrically proximate
  // by the binning coarse search, and one that passes the finer computational geometry
  // (CG) filter/checks. These finer checks identify face-pairs that are intersecting or
  // nearly intersecting with positive areas of overlap. These checks do not indicate
  // whether a face-pair is contacting, since the definition of 'contacting' is specific
  // to a particular contact method and its enforced contstraints. Rather, the CG filter
  // identifies contact candidacy, or face-pairs likely in contact.
  bool m_is_contact_candidate;
};

} /* namespace tribol */

#endif /* SRC_TRIBOL_MESH_INTERFACE_PAIRS_HPP_ */
