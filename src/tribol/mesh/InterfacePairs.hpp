// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_MESH_INTERFACE_PAIRS_HPP_
#define SRC_TRIBOL_MESH_INTERFACE_PAIRS_HPP_

#include <type_traits>

// Shared includes
#include "tribol/common/BasicTypes.hpp"

namespace tribol {

/** @brief Identifies one element from each mesh without storing method-specific contact state */
struct InterfacePair {
  IndexT m_element_id1{ -1 };  ///< Element ID from the first mesh
  IndexT m_element_id2{ -1 };  ///< Element ID from the second mesh
};

static_assert( std::is_aggregate_v<InterfacePair>, "InterfacePair must support aggregate initialization." );
static_assert( std::is_trivially_copyable_v<InterfacePair>, "InterfacePair must remain inexpensive to copy." );
static_assert( sizeof( InterfacePair ) == 2 * sizeof( IndexT ), "InterfacePair must store only its two element IDs." );
static_assert( InterfacePair{}.m_element_id1 == -1 && InterfacePair{}.m_element_id2 == -1,
               "A default InterfacePair must remain invalid." );

} /* namespace tribol */

#endif /* SRC_TRIBOL_MESH_INTERFACE_PAIRS_HPP_ */
