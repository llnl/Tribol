// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_MESH_INTERFACE_PAIRS_HPP_
#define SRC_TRIBOL_MESH_INTERFACE_PAIRS_HPP_

#include "tribol/common/BasicTypes.hpp"

namespace tribol {

struct InterfacePair {
  TRIBOL_HOST_DEVICE InterfacePair( IndexT element_id1, IndexT element_id2 )
      : m_element_id1( element_id1 ), m_element_id2( element_id2 )
  {
  }

  // overload constructor to handle zero input arguments
  TRIBOL_HOST_DEVICE InterfacePair() : InterfacePair( -1, -1 ) {}

  // Element id for face 1
  IndexT m_element_id1;

  // Element id for face 2
  IndexT m_element_id2;
};

} /* namespace tribol */

#endif /* SRC_TRIBOL_MESH_INTERFACE_PAIRS_HPP_ */
