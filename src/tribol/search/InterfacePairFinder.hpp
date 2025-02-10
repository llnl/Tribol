// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)
#ifndef TRIBOL_INTERFACE_PAIR_FINDER_HPP_
#define TRIBOL_INTERFACE_PAIR_FINDER_HPP_

#include "tribol/common/Parameters.hpp"

namespace tribol {

// Forward Declarations
class CouplingScheme;
template <int D>
class GridSearch;
struct InterfacePair;

/// Free functions

/*!
 * \brief Basic geometry/proximity checks for face pairs
 *
 * \param [in] iPair InterfacePair struct for two faces
 * \param [in] mode ContactMode
 *
 */
bool geomFilter( InterfacePair& iPair, ContactMode const mode );

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
   * specified in \a m_couplingScheme
   */
  void findInterfacePairs();

 private:
  CouplingScheme* m_couplingScheme;

  GridSearch<2>* m_gridSearch2D;
  GridSearch<3>* m_gridSearch3D;
};

}  // end namespace tribol

#endif /* TRIBOL_INTERFACE_PAIR_FINDER_HPP_ */
