// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/InterfacePairs.hpp"

namespace tribol {

void InterfacePairs::reserve( integer new_size )
{
  m_pairIndex1.reserve( new_size );
  m_pairIndex2.reserve( new_size );
  m_isContactCandidate.reserve( new_size );
}

void InterfacePairs::clear()
{
  m_pairIndex1.clear();
  m_pairIndex2.clear();
  m_isContactCandidate.clear();
}

void InterfacePairs::addInterfacePair( InterfacePair const& pair )
{
  // set mesh ids and pair types. This is redundant since these
  // only need to be set once.
  m_meshId1 = pair.meshId1;
  m_meshId2 = pair.meshId2;
  m_pairType1 = pair.pairType1;
  m_pairType2 = pair.pairType2;

  // add face ids to containers
  m_pairIndex1.push_back( pair.pairIndex1 );
  m_pairIndex2.push_back( pair.pairIndex2 );

  // set contact candidate boolean container entry
  m_isContactCandidate.push_back( pair.isContactCandidate );
}

void InterfacePairs::updateInterfacePair( InterfacePair const& pair, integer const idx )
{
  m_isContactCandidate[idx] = pair.isContactCandidate;
}

InterfacePair InterfacePairs::getInterfacePair( IndexType idx ) const
{
  SLIC_ERROR_IF( idx >= getNumPairs(), "Index out of range." );

  return InterfacePair{ m_meshId1,   m_pairType1,       m_pairIndex1[idx],         m_meshId2,
                        m_pairType2, m_pairIndex2[idx], m_isContactCandidate[idx], idx };
}

}  // namespace tribol
