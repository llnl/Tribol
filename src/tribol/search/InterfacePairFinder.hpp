// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_
#define SRC_TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_

#include <axom/core/execution/execution_space.hpp>
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/search/PairIterator.hpp"

namespace tribol {

// Forward Declarations
class CouplingScheme;
class SearchBase;

/// Free functions

/*!
 * \brief Basic geometry/proximity checks for face pairs
 *
 * \param [in] element_id1 id of 1st element in pair
 * \param [in] element_id2 id of 2nd element in pair
 * \param [in] mesh1 mesh view for 1st element in pair
 * \param [in] mesh2 mesh view for 2nd element in pair
 * \param [in] mode ContactMode
 * \param [in] auto_contact_check Is auto-contact assumed?
 * \param [in] element_radius_multiplier Scaling applied to max element radius distance check
 *
 */
TRIBOL_HOST_DEVICE bool geomFilter( IndexT element_id1, IndexT element_id2, const MeshData::Viewer& mesh1,
                                    const MeshData::Viewer& mesh2, ContactMode mode, bool auto_contact_check,
                                    RealT element_radius_multiplier );

class CartesianProduct {
 public:
  using pair_iterator = AllPairIterator;

  CartesianProduct( const MeshData& mesh1, const MeshData& mesh2, const Parameters& );
  pair_iterator getPairCandidates();

 private:
  const MeshData& mesh1_;
  const MeshData& mesh2_;
};

template <int Dim>
class BVHSearch {
 public:
  using pair_iterator = ListIterator;
  // using bvh_type = axom::spin::BVH < Dim, axom::e>

  BVHSearch( const MeshData& mesh1, const MeshData& mesh2, const Parameters& params );
  pair_iterator getPairCandidates();

 private:
  Array<size_t> m_candidate_offsets;
  Array<size_t> m_candidate_counts;
  Array<size_t> m_candidates;
};

/*!
 * \class InterfacePairFinder
 *
 * \brief This class finds pairs of interfering elements in the meshes
 * referred to by the CouplingScheme
 */
class InterfacePairFinder {
 public:
  InterfacePairFinder( CouplingScheme* cs );

  ~InterfacePairFinder();

  /*!
   * Initializes structures for the candidate search
   */
  void initialize();

  /*!
   * Computes the interacting interface pairs between the meshes
   * specified in \a m_coupling_scheme
   */
  void findInterfacePairs();

 private:
  CouplingScheme* m_coupling_scheme;
  SearchBase* m_search;  // The search strategy
};

}  // end namespace tribol

#endif /* SRC_TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_ */
