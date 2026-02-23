// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_
#define SRC_TRIBOL_SEARCH_INTERFACE_PAIR_FINDER_HPP_

#include "tribol/common/BasicTypes.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/utils/Math.hpp"

namespace tribol {

class SearchBase;

/**
 * @brief Performs a geometric filter on two elements to determine if they are a contact candidate
 *
 * @param cs_view View of the coupling scheme
 * @param element_id1 ID of element in the first mesh
 * @param element_id2 ID of element in the second mesh
 * @return true Elements are a contact candidate
 * @return false Elements are NOT a contact candidate
 */
TRIBOL_HOST_DEVICE inline bool geomFilter( const CouplingScheme::Viewer& cs_view, IndexT element_id1,
                                           IndexT element_id2 )
{
  auto& mesh1 = cs_view.getMesh1View();
  auto& mesh2 = cs_view.getMesh2View();
  bool auto_contact_check = cs_view.getParameters().auto_contact_check;
  // we want binning proximity scaled by LOR factor on HO meshes, i.e. the effective binning proximity
  auto binning_proximity_scale = cs_view.getEffectiveBinningProximityScale();
  auto dim = cs_view.spatialDimension();
  auto mode = cs_view.getContactMode();
  auto& params = cs_view.getParameters();
  RealT residual_gap = params.residual_gap;

  /// CHECK #1: Check to make sure the two face ids are not the same
  ///           and the two mesh ids are not the same.
  if ( ( mesh1.meshId() == mesh2.meshId() ) && ( element_id1 == element_id2 ) ) {
    return false;
  }

  /// CHECK #2: Auto-contact precludes faces that share a common
  ///           node(s). We want to preclude two adjacent faces from interacting
  ///           due to problematic configurations, such as corners where the
  ///           configuration and opposing normals appear to be in contact, but
  ///           are not.
  ///
  ///           Note: non-auto-contact coupling schemes should typically be amongst
  ///                 topologically disconnected surfaces unless it is known apriori that
  ///                 face-pairs with shared nodes can in fact contact.
  if ( auto_contact_check ) {
    for ( IndexT i{ 0 }; i < mesh1.numberOfNodesPerElement(); ++i ) {
      int node1 = mesh1.getGlobalNodeId( element_id1, i );
      for ( IndexT j{ 0 }; j < mesh2.numberOfNodesPerElement(); ++j ) {
        int node2 = mesh2.getGlobalNodeId( element_id2, j );
        if ( node1 == node2 ) {
          return false;
        }
      }
    }
  }

  /// CHECK #3: Check that face normals are opposing up to some tolerance.
  ///           This uses a hard coded normal tolerance for this check.
  RealT nrmlTol = 0.0;  // taken as cos(90) between face pair

  RealT nrmlCheck = 0.0;
  for ( int d{ 0 }; d < dim; ++d ) {
    nrmlCheck += mesh1.getElementNormals()[d][element_id1] * mesh2.getElementNormals()[d][element_id2];
  }

  // check normal projection against tolerance
  if ( nrmlCheck > nrmlTol ) {
    return false;
  }

  /// CHECK #4: Perform radius check, which involves seeing if
  ///           the distance between the two face vertex averaged
  ///           centroid is less than the sum of the two face radii
  ///           premultiplied by a binning scale factor.
  ///           The face radii are taken to be the magnitude of the
  ///           longest vector from that face's vertex averaged
  ///           centroid to one its nodes.
  constexpr RealT offset_tol = 0.05;

  if ( dim == 3 ) {
    // get face radius off the mesh data
    RealT r1 = mesh1.getFaceRadius()[element_id1];
    RealT r2 = mesh2.getFaceRadius()[element_id2];

    RealT distMax = binning_proximity_scale * ( r1 + r2 ) + residual_gap;

    // check if the contact mode is conforming, in which case the
    // faces are supposed to be aligned
    if ( mode == SURFACE_TO_SURFACE_CONFORMING ) {
      // use 5% of max face radius for conforming case as
      // tolerance on face offsets
      distMax = offset_tol * binning_proximity_scale * ( r1 + r2 ) + residual_gap;
    }

    // compute the distance between the two face centroids
    RealT distX = mesh2.getElementCentroids()[0][element_id2] - mesh1.getElementCentroids()[0][element_id1];
    RealT distY = mesh2.getElementCentroids()[1][element_id2] - mesh1.getElementCentroids()[1][element_id1];
    RealT distZ = mesh2.getElementCentroids()[2][element_id2] - mesh1.getElementCentroids()[2][element_id1];

    RealT distMag = magnitude( distX, distY, distZ );

    if ( distMag > ( distMax ) ) {
      return false;
    }
  }  // end of dim == 3
  else if ( dim == 2 ) {
    // get 1/2 edge length off the mesh data
    RealT e1 = 0.5 * mesh1.getElementAreas()[element_id1];
    RealT e2 = 0.5 * mesh2.getElementAreas()[element_id2];

    RealT distMax = binning_proximity_scale * ( e1 + e2 ) + residual_gap;

    // check if the contact mode is conforming, in which case the
    // edges are supposed to be aligned
    if ( mode == SURFACE_TO_SURFACE_CONFORMING ) {
      // use 5% of max face radius for conforming case as
      // tolerance on face offsets
      distMax = offset_tol * binning_proximity_scale * ( e1 + e2 ) + residual_gap;
    }

    // compute the distance between the two edge centroids
    RealT distX = mesh2.getElementCentroids()[0][element_id2] - mesh1.getElementCentroids()[0][element_id1];
    RealT distY = mesh2.getElementCentroids()[1][element_id2] - mesh1.getElementCentroids()[1][element_id1];

    RealT distMag = magnitude( distX, distY );

    // include faces where separation equals distMax
    if ( distMag > ( distMax ) ) {
      return false;
    }
  }  // end of dim == 2

  /// Check #5: Check to see if there is a positive area of overlap when both faces/edges
  ///           are projected onto an intermediate plane
  if ( cs_view.pruneMethodFacePair( element_id1, element_id2 ) ) {
    return false;
  }

  // if we made it here we passed all checks
  return true;
}

/*!
 * \class SearchBase
 *
 * \brief This is the base class for the candidate search
 */
class SearchBase {
 public:
  SearchBase( CouplingScheme* cs ) : m_coupling_scheme( cs ) {}

  virtual ~SearchBase() {}

  /*!
   * Initializes the search strategy
   */
  virtual void initialize() = 0;

  /*!
   * Performs the candidate search
   */
  virtual void findInterfacePairs() = 0;

 protected:
  CouplingScheme* m_coupling_scheme;

  /**
   * @brief Performs a geometric filter on two elements to determine if they are a contact candidate
   *
   * @param element_id1 ID of element in the first mesh
   * @param element_id2 ID of element in the second mesh
   * @return true Elements are a contact candidate
   * @return false Elements are NOT a contact candidate
   */
  TRIBOL_HOST_DEVICE inline bool geomFilter( const IndexT element_id1, const IndexT element_id2 ) const
  {
    return tribol::geomFilter( m_coupling_scheme->getView(), element_id1, element_id2 );
  }
};

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
