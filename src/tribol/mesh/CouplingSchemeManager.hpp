// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)
#ifndef TRIBOL_COUPLINGSCHEMEMANAGER_HPP_
#define TRIBOL_COUPLINGSCHEMEMANAGER_HPP_

#include "axom/core.hpp"
#include "axom/slic.hpp"

// C/C++ include
#include <vector>

namespace tribol {

// Forward Declarations
class CouplingScheme;

/*!
 * \brief Singleton class to manage and store CouplingScheme instances.
 *
 * \see CouplingScheme
 */
class CouplingSchemeManager {
 public:
  /*!
   * \brief Destructor.
   */
  ~CouplingSchemeManager();

  /*!
   * \brief Returns a reference to the CouplingSchemeManager instance.
   * \return C a reference to the CouplingSchemeManager instance.
   */
  static CouplingSchemeManager& getInstance();

  /*!
   * Removes all registered couplings from the manager and reclaims
   * associated memory
   */
  void clearAllCouplings();

  /*!
   * \brief Adds the CouplingScheme to the CouplingSchemeManager
   *
   * \param [in] cs pointer to a CouplingScheme instance.
   *
   * \note After a coupling scheme is added to the CouplingSchemeManager,
   *  ownership of the CouplingScheme object is assumed by the manager.
   *
   * \pre cs != nullptr
   */
  int addCoupling( CouplingScheme* cs );

  /*!
   * \brief Adds the CouplingScheme to the CouplingSchemeManager instance.
   *
   * \param [in] idx index for the new CouplingScheme
   * \param [in] cs pointer to a CouplingScheme instance.
   *
   * \note After a coupling scheme is added to the CouplingSchemeManager,
   *  ownership of the CouplingScheme object is assumed by the manager.
   *
   * \note If there is an existing CouplingScheme in the provided index \a idx,
   * the former will be replaced with the new one and deleted.
   *
   * \pre cs != nullptr
   * \pre idx >= 0
   */
  int addCoupling( int idx, CouplingScheme* cs );

  /*!
   * Removes the CouplingScheme at index \a idx and reclaims memory
   */
  void removeCoupling( int idx );

  /*!
   * \brief Returns pointer to the CouplingScheme at the given index
   *
   * \param idx the index of the CouplingScheme being queried.
   * \return cs pointer to the CouplingScheme
   *
   * \pre idx >= 0 && idx < getNumberOfCouplings()
   * \note Returned pointer might be null
   */
  inline CouplingScheme* getCoupling( int idx ) const;

  /*!
   * \brief Returns the number of CouplingSchemes registered with the manager.
   * \return N the number of coupling schemes
   * \post N >= 0
   */
  inline int getNumberOfCouplings() const { return static_cast<int>( m_coupling_schemes.size() ); }

  /*!
   * Predicate to check if a CouplingScheme with the given index
   * has been registered.
   *
   * \param idx The index to check
   * \return True if there is a registered coupling scheme with index \a idx
   */
  bool hasCoupling( int idx ) const { return isValidIndex( idx ) && m_coupling_schemes[idx] != nullptr; }

 private:
  /*!
   * Predicate to check if a given index \a idx is valid
   *
   * An index is valid when it is in the range [0, \a getNumberOfCouplings() )
   *
   * \param idx The index to check
   * \return True when the index is valid, false otherwise.
   */
  bool isValidIndex( int idx ) const { return idx >= 0 && idx < getNumberOfCouplings(); }

  /*!
   * \brief Default constructor.
   *
   * \note Made private in order to prevent users from instantiating this
   *  this object.
   */
  CouplingSchemeManager() {}

  std::vector<CouplingScheme*> m_coupling_schemes;  ///< List of coupling schemes

  DISABLE_COPY_AND_ASSIGNMENT( CouplingSchemeManager );
  DISABLE_MOVE_AND_ASSIGNMENT( CouplingSchemeManager );
};

//------------------------------------------------------------------------------
// IMPLEMENTATION OF INLINE METHODS
//------------------------------------------------------------------------------

inline CouplingScheme* CouplingSchemeManager::getCoupling( int idx ) const
{
  SLIC_ASSERT( isValidIndex( idx ) );

  return m_coupling_schemes[idx];
}

} /* namespace tribol */

#endif /* TRIBOL_COUPLINGSCHEMEMANAGER_HPP_ */
