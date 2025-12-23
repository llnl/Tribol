// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_
#define SRC_TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_

#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

namespace tribol {

// Forward Declarations
class SearchBase;

/// Free functions

/*!
 * \brief Basic geometry/proximity checks for face pairs
 *
 * \param [in] cs_view View of the coupling scheme
 * \param [in] element_id1 id of 1st element in pair
 * \param [in] element_id2 id of 2nd element in pair
 *
 */
TRIBOL_HOST_DEVICE bool geomFilter( const CouplingScheme::Viewer& cs_view, IndexT element_id1, IndexT element_id2 );

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
