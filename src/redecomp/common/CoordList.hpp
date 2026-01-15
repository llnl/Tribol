// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_COMMON_COORDLIST_HPP_
#define SRC_REDECOMP_COMMON_COORDLIST_HPP_

#include "mfem.hpp"
#include "redecomp/common/TypeDefs.hpp"

namespace redecomp {

/**
 * @brief CoordList class that holds a 2D mfem::Vector and provides access to points.
 * 
 * @tparam NDIMS Number of dimensions
 * @tparam ORDERING Element DOF ordering (LEXICOGRAPHIC or NATIVE)
 */
template <int NDIMS, mfem::ElementDofOrdering ORDERING = mfem::ElementDofOrdering::LEXICOGRAPHIC>
class CoordList {
public:
  /**
   * @brief Construct a new Coord List object
   * 
   * @param coords Vector containing coordinates. Expected layout: ByVDIM 
   * (all x coordinates, then all y coordinates, etc.)
   */
  CoordList( mfem::Vector coords ) : coords_( std::move( coords ) ) {}

  /**
   * @brief Create a Point object for the k-th node
   * 
   * @param k Index of the node
   * @return Point<NDIMS> 
   */
  Point<NDIMS> GetPoint( int k ) const
  {
    Point<NDIMS> pt;
    for ( int d = 0; d < NDIMS; ++d ) {
      pt[d] = coords_( k * NDIMS + d );
    }
    return pt;
  }

  /**
   * @brief Get the Ordering
   * 
   * @return mfem::ElementDofOrdering 
   */
  static constexpr mfem::ElementDofOrdering GetOrdering() { return ORDERING; }

  /**
   * @brief Get the underlying coordinate vector
   * 
   * @return const mfem::Vector& 
   */
  const mfem::Vector& GetCoords() const { return coords_; }

  /**
   * @brief Get the number of coordinates
   * 
   * @return int 
   */
  int GetNumCoords() const { return coords_.Size() / NDIMS; }

private:
  mfem::Vector coords_;
};

}  // namespace redecomp

#endif /* SRC_REDECOMP_COMMON_COORDLIST_HPP_ */
