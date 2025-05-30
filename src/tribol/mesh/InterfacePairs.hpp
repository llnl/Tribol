// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_MESH_INTERFACE_PAIRS_HPP_
#define SRC_TRIBOL_MESH_INTERFACE_PAIRS_HPP_

#include "tribol/common/BasicTypes.hpp"

namespace tribol {

struct InterfacePair {
  TRIBOL_HOST_DEVICE InterfacePair( IndexT element_id1, IndexT element_id2, bool is_contact_candidate = true );

  // overload constructor to handle zero input arguments
  TRIBOL_HOST_DEVICE InterfacePair();

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
