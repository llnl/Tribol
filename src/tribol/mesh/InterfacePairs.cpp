// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/InterfacePairs.hpp"

namespace tribol {

TRIBOL_HOST_DEVICE InterfacePair::InterfacePair( IndexT element_id1, IndexT element_id2, bool is_contact_candidate )
    : m_element_id1( element_id1 ), m_element_id2( element_id2 ), m_is_contact_candidate( is_contact_candidate )
{
}

TRIBOL_HOST_DEVICE InterfacePair::InterfacePair()
    : m_element_id1( -1 ), m_element_id2( -1 ), m_is_contact_candidate( true )
{
}

}  // namespace tribol
