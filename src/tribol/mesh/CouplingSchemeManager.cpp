// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)
#include "tribol/mesh/CouplingSchemeManager.hpp"

#include "tribol/mesh/CouplingScheme.hpp"

namespace tribol {

CouplingSchemeManager::~CouplingSchemeManager() { clearAllCouplings(); }

CouplingSchemeManager& CouplingSchemeManager::getInstance()
{
  static CouplingSchemeManager instance;
  return instance;
}

void CouplingSchemeManager::clearAllCouplings()
{
  const int numCouplings = m_coupling_schemes.size();
  for ( int i = 0; i < numCouplings; ++i ) {
    delete m_coupling_schemes[i];
    m_coupling_schemes[i] = nullptr;
  }

  m_coupling_schemes.clear();
}

int CouplingSchemeManager::addCoupling( CouplingScheme* cs )
{
  SLIC_ASSERT( cs != nullptr );

  m_coupling_schemes.push_back( cs );

  return getNumberOfCouplings() - 1;
}

int CouplingSchemeManager::addCoupling( int idx, CouplingScheme* cs )
{
  SLIC_ASSERT_MSG( idx >= 0, "Index for new coupling scheme cannot be negative" );
  SLIC_ASSERT_MSG( cs != nullptr, "Attempted to add invalid coupling scheme to "
                                      << " coupling scheme manager. New entries cannot be null." );

  // Expand storage to contain index idx
  if ( idx >= getNumberOfCouplings() ) {
    m_coupling_schemes.resize( idx + 1, nullptr );
  }

  if ( hasCoupling( idx ) ) {
    removeCoupling( idx );
  }

  m_coupling_schemes[idx] = cs;

  return idx;
}

void CouplingSchemeManager::removeCoupling( int idx )
{
  if ( hasCoupling( idx ) ) {
    delete m_coupling_schemes[idx];
    m_coupling_schemes[idx] = nullptr;
  }
}

} /* namespace tribol */
