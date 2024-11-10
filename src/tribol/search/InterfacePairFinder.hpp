// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_
#define TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_

#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol
{

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
 *
 */
TRIBOL_HOST_DEVICE bool geomFilter( IndexT element_id1, IndexT element_id2,
                                    const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                    ContactMode mode, bool auto_contact_check );

/*!
 * \class InterfacePairFinder
 *
 * \brief This class finds pairs of interfering elements in the meshes
 * referred to by the CouplingScheme
 */
class InterfacePairFinder
{
public:
   InterfacePairFinder(CouplingScheme* cs);

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

} // end namespace tribol



#endif /* TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_ */
