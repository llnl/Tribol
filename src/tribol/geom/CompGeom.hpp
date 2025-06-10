// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_GEOM_COMPGEOM_HPP_
#define SRC_GEOM_COMPGEOM_HPP_

#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/common/Parameters.hpp"
#include "axom/slic.hpp"

#include <string>

namespace tribol {

/*!
 * \brief checks if the vertices on face2 have interpenetrated the level set
 *        defined by face 1 and vice-versa, or if the faces are entirely in separation
 *
 * \param [in] mesh1 mesh data viewer for mesh 1 to which face 1 belongs
 * \param [in] mesh2 mesh data viewer for mesh 2 to which face 2 belongs
 * \param [in] fId1 id for face 1
 * \param [in] fId2 id for face 2
 *
 * \return true if all nodes on one face are on the other side of the plane defined
 *         by the other face
 *
 * This uses face1 as a level set and checks the projection
 * of the vector defined by differencing the face1 center and a face2
 * node onto the face1 normal. If this projection is positive then
 * interpenetration has occured and face2 intersects the plane defined
 * by face1 (i.e. the zero level set).
 *
 */
TRIBOL_HOST_DEVICE bool FullFaceCheck( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, int fId1,
                                       int fId2 );

/*!
 *
 * \brief checks if edge 2 interpenetrates the level set defined by edge 1 and vice-versa
 *
 * \param [in] mesh1 mesh data viewer for mesh 1
 * \param [in] mesh2 mesh data viewer for mesh 2
 * \param [in] eId1 edge id for edge belonging to mesh 1
 * \param [in] eId2 edge id for edge belonging to mesh 2
 *
 * \return true if edge 2 interpenetrates the level set defined by edge 1 and vice-versa
 *
 */
TRIBOL_HOST_DEVICE bool FullEdgeCheck( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, int eId1,
                                       int eId2 );


//-----------------------------------------------------------------------------
// Computational Geometry base classes
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Comp geom container class
//-----------------------------------------------------------------------------
class CompGeom {

  public:

   /**
    * @brief Nested class for holding views (non-owned, shallow copies) of the CompGeom data
    */
   class Viewer {
    public:

     /**
      * @brief Construct a new CompGeom::Viewer object
      *
      * @param cg CompGeom object to create a view of
      */
     Viewer ( CompGeom& cg )
       : m_common_plane_pairs( cg.m_common_plane_pairs ),
         m_mortar_plane_pairs( cg.m_mortar_plane_pairs ),
         m_aligned_mortar_plane_pairs( cg.m_aligned_mortar_plane_pairs )
     {
     }

     /**
      * @brief Get the array view of common plane pairs
      *
      * @return ArrayViewT of common plane pairs
      */
     TRIBOL_HOST_DEVICE const ArrayViewT<const CommonPlanePair>& getCommonPlanePairs() const { return m_common_plane_pairs; }

     /**
      * @brief Get a single common plane from the array view
      *
      * @return common plane object
      */
     TRIBOL_HOST_DEVICE CommonPlanePair& getCommonPlane( int id ) const { return m_common_plane_pairs[id]; }

     /**
      * @brief Get the array view of mortar plane pairs
      *
      * @return ArrayViewT of mortar plane pairs
      */
     TRIBOL_HOST_DEVICE const ArrayViewT<const MortarPlanePair>& getMortarPlanePairs() const { return m_mortar_plane_pairs; }

     /**
      * @brief Get a single mortar plane from the array view
      *
      * @return mortar plane object
      */
     TRIBOL_HOST_DEVICE MortarPlanePair& getMortarPlane( int id ) const { return m_mortar_plane_pairs[id]; }

     /**
      * @brief Get the array view of aligned mortar plane pairs
      *
      * @return ArrayViewT of aligned mortar plane pairs
      */
     TRIBOL_HOST_DEVICE const ArrayViewT<const AlignedMortarPlanePair>& getAlignedMortarPlanePairs() const { return m_aligned_mortar_plane_pairs; }

     /**
      * @brief Get a single aligned mortar plane from the array view
      *
      * @return algined mortar plane object
      */
     TRIBOL_HOST_DEVICE AlignedMortarPlanePair& getAlignedMortarPlane( int id ) const { return m_aligned_mortar_plane_pairs[id]; }

     /**
      * @brief Add a contact plane to the appropriate array view
      *
      */
      TRIBOL_HOST_DEVICE addContactPlane( ContactPlanePair& contact_plane, const int id, const ContactMethod method ) {
        switch (method) {
          case COMMON_PLANE: {
            m_common_plane_pairs[id] = std::move( static_cast<CommonPlanePair&>(contact_plane) );
            break;
          }
          case SINGLE_MORTAR:
          case MORTAR_WEIGHTS: {
            m_mortar_plane_pairs[id] = std::move( static_cast<MortarPlanePair&>(contact_plane) );
            break;
          }
          case ALIGNED_MORTAR: {
            m_aligned_mortar_plane_pairs[id] = std::move( static_cast<AlignedMortarPlanePair&>(contact_plane) );
            break;
          }
          default: {
            // no-op
            break;
          }
        } // end switch
      }

     private:

      ArrayViewT<CommonPlanePair> m_common_plane_pairs;
      ArrayViewT<MortarPlanePair> m_mortar_plane_pairs;
      ArrayViewT<AlignedMortarPlanePair> m_aligned_mortar_plane_pairs;

   }; // end CompGeom::Viewer
  
   /*!
    * @brief Constructs a comp geom object
    *
    */
   TRIBOL_HOST_DEVICE CompGeom() {};

   /*!
    * @brief Destructor
    *
    */
   virtual ~CompGeom() = default; 

   /**
    * @brief Construct a non-owned, shallow copy of the CompGeom data
    *
    * @return CompGeom::Viewer type
    */
   CompGeom::Viewer getView() { return *this; }

   /**
    * @brief Get the list of common plane pairs
    *
    * @return ArrayT of common plane pairs
    */
   ArrayT<CommonPlanePair>& getCommonPlanePairs() { return m_common_plane_pairs; }

   /**
    * @brief Get a single common plane
    *
    * @return common plane object
    */
   CommonPlanePair& getCommonPlane( int id ) const { return m_common_plane_pairs[id]; }

   /**
    * @brief Get the list of mortar plane pairs
    *
    * @return ArrayT of mortar plane pairs
    */
   ArrayT<MortarPlanePair>& getMortarPlanePairs() { return m_mortar_plane_pairs; }

   /**
    * @brief Get a single mortar plane
    *
    * @return mortar plane object
    */
   MortarPlanePair& getMortarPlane( int id ) const { return m_mortar_plane_pairs[id]; }

   /**
    * @brief Get the list of aligned mortar plane pairs
    *
    * @return ArrayT of aligned mortar plane pairs
    */
   ArrayT<AlignedMortarPlanePair>& getAlignedMortarPlanePairs() { return m_aligned_mortar_plane_pairs; }

   /**
    * @brief Get a single aligned mortar plane
    *
    * @return aligned mortar plane object
    */
   AlignedMortarPlanePair& getAlignedMortarPlane( int id ) const { return m_aligned_mortar_plane_pairs[id]; }

   int getNumActivePairs( const ContactMethod method )
   {
    switch (method) {
      case COMMON_PLANE: {
        return m_common_plane_pairs.size();
        break;
      }
      case SINGLE_MORTAR:
      case MORTAR_WEIGHTS: {
        return m_mortar_plane_pairs.size();
        break;
      }
      case ALIGNED_MORTAR: {
        return m_aligned_mortar_plane_pairs.size();
        break;
      }
      default: {
        // no-op
        break;
      }
    } // end switch
    return 0;
   } // end getNumActivePairs()

   TRIBOL_HOST_DEVICE resizeActivePairs( ContactMethod method, int size ) {
    switch (method) {
      case COMMON_PLANE: {
        m_common_plane_pairs.resize( size );
        break;
      }
      case SINGLE_MORTAR:
      case MORTAR_WEIGHTS: {
        m_mortar_plane_pairs.resize( size );
        break;
      }
      case ALIGNED_MORTAR: {
        m_aligned_mortar_plane_pairs.resize( size );
        break;
      }
      default: {
        // no-op
        break;
      }
    } // end switch
   } // end resizeActivePairs()

  private:

   ArrayT<CommonPlanePair> m_common_plane_pairs;
   ArrayT<MortarPlanePair> m_mortar_plane_pairs;
   ArrayT<AlignedMortarPlanePair> m_aligned_mortar_plane_pairs;
};

//-----------------------------------------------------------------------------
// Computational geometry base class 
// (can be used to extend non-contact-plane classes)
//-----------------------------------------------------------------------------
class CompGeomPair {

 protected:
  InterfacePair* m_pair; ///< Face-pair struct for two constituent faces

  TRIBOL_HOST_DEVICE CompGeomPair() {};

  TRIBOL_HOST_DEVICE CompGeomPair( InterfacePair* pair, Parameters& params, const int dim )
      : m_pair( pair ),
        m_params( params ),
        m_dim( dim ) 
  {
  }

  virtual ~CompGeomPair() = default; 

 public:
  int m_dim;
  Parameters& m_params;
  TRIBOL_HOST_DEVICE virtual FaceGeomError checkInterfacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) = 0;
};

//-----------------------------------------------------------------------------
// ContactPlane abstract base class
//-----------------------------------------------------------------------------
class ContactPlanePair : public CompGeomPair {
 protected:

  /**
   * @brief Constructs a ContactPlane object
   *
   */
  TRIBOL_HOST_DEVICE ContactPlanePair() {};

  /**
   * @brief Overloaded constructor
   *
   */
  TRIBOL_HOST_DEVICE ContactPlanePair( InterfacePair* pair, Parameters& params, const int dim );

  virtual ~ContactPlanePair() = default;

  static constexpr int max_nodes_per_overlap{8};

 public:
  bool m_inContact; ///< True if face-pair is in contact
  RealT m_gap;      ///< Face-pair gap
  RealT m_gapTol;   ///< Face-pair gap tolerance

  RealT m_e1X;  ///< Global x-component of first in-plane basis vector
  RealT m_e1Y;  ///< Global y-component of first in-plane basis vector
  RealT m_e1Z;  ///< Global z-component of first in-plane basis vector

  RealT m_e2X;  ///< Global x-component of second in-plane basis vector
  RealT m_e2Y;  ///< Global y-component of second in-plane basis vector
  RealT m_e2Z;  ///< Global z-component of second in-plane basis vector

  RealT m_cX;  ///< Contact plane overlap centroid global x-coordinate
  RealT m_cY;  ///< Contact plane overlap centroid global y-coordinate
  RealT m_cZ;  ///< Contact plane overlap centroid global z-coordinate (zero out for 2D)

  RealT m_overlapCX;  ///< Local x-coordinate of overlap centroid
  RealT m_overlapCY;  ///< Local y-coordinate of overlap centroid

  RealT m_cXf1;  ///< Global x-coordinate of contact plane centroid projected to face 1
  RealT m_cYf1;  ///< Global y-coordinate of contact plane centroid projected to face 1
  RealT m_cZf1;  ///< Global z-coordinate of contact plane centroid projected to face 1

  RealT m_cXf2;  ///< global x-coordinate of contact plane centroid projected to face 2
  RealT m_cYf2;  ///< global y-coordinate of contact plane centroid projected to face 2
  RealT m_cZf2;  ///< global z-coordinate of contact plane centroid projected to face 2

  RealT m_nX;  ///< Global x-component of contact plane unit normal
  RealT m_nY;  ///< Global y-component of contact plane unit normal
  RealT m_nZ;  ///< Global z-component of contact plane unit normal (zero out for 2D)

  int m_numPolyVert;  ///< Number of vertices in overlapping polygon
  RealT m_polyX[ max_nodes_per_overlap ]; ///< Global x-components of overlap polygon's vertices
  RealT m_polyY[ max_nodes_per_overlap ]; ///< Global y-components of overlap polygon's vertices
  RealT m_polyZ[ max_nodes_per_overlap ]; ///< Global z-components of overlap polygon's vertices

  RealT m_polyLocX[ max_nodes_per_overlap ];  ///< Pointer to local x-components of overlap polygon's vertices
  RealT m_polyLocY[ max_nodes_per_overlap ];  ///< Pointer to local y-components of overlap polygon's vertices
  
  // cp area
  bool  m_fullOverlap {true}; ///< Indicates if a full overlap (true) or interpen overlap (false) is used
  RealT m_areaFrac; ///< Face area fraction used to determine overlap area cutoff
  RealT m_areaMin;  ///< Minimum overlap area for inclusion into the active set
  RealT m_area;     ///< Overlap area

  /// \name Contact plane routines
  /// @{

  /*!
   * \brief check to see if interface pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry error
   */
  TRIBOL_HOST_DEVICE virtual FaceGeomError checkInterfacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) = 0;

  /*!
   * \brief check to see if face-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry error
   */
  TRIBOL_HOST_DEVICE virtual FaceGeomError checkFacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) = 0;

  /*!
   * \brief check to see if edge-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry error
   */
  TRIBOL_HOST_DEVICE virtual FaceGeomError checkEdgePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) = 0;

  /*!
   * \brief Compute the projected overlap in 2D 
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] params Coupling scheme-dependent parameters
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   */
  TRIBOL_HOST_DEVICE FaceGeomError computeOverlap2D( const MeshData::Viewer& m1, const MeshData::Viewer& m2,
                                                     const Parameters& params ) = 0;
  /*!
   * \brief Compute the projected overlap in 3D
   *
   * \param [in] x1 x-coordinates of the first planar quadrilateral
   * \param [in] y1 y-coordinates of the first planar quadrilateral
   * \param [in] z1 z-coordinates of the first planar quadrilateral
   * \param [in] x2 x-coordinates of the second planar quadrilateral
   * \param [in] y2 y-coordinates of the second planar quadrilateral
   * \param [in] z2 z-coordinates of the second planar quadrilateral
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] params Coupling scheme-dependent parameters
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   *
   * \pre this routine assumes that the two four node quadrilaterals are planar
   */
  TRIBOL_HOST_DEVICE FaceGeomError computeOverlap3D( const RealT* x1, const RealT* y1, const RealT* z1,
                                                     const RealT* x2, const RealT* y2, const RealT* z2,
                                                     const MeshData::Viewer& m1, const MeshData::Viewer& m2,const Parameters& params ) = 0;
  /*!
   * \brief Compute a local basis on the contact plane
   *
   * \param [in] m1 mesh data viewer for mesh 1
   */
  TRIBOL_HOST_DEVICE void computeLocalBasis( const MeshData::Viewer& m1 );

  /*!
   * \brief Compute the contact plane normal
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   */
  TRIBOL_HOST_DEVICE virtual void computeNormal( const MeshData::Viewer& m1, const MeshData::Viewer& m2 ) = 0;

  /*!
   * \brief Compute the contact plane point
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   */
  TRIBOL_HOST_DEVICE virtual void computePlanePoint( const MeshData::Viewer& m1, const MeshData::Viewer& m2 ) = 0;

  /*!
   * \brief Compute the contact plane area tolerance
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] params Coupling scheme-dependent parameters
   */
  TRIBOL_HOST_DEVICE void computeAreaTol( const MeshData::Viewer& m1, const MeshData::Viewer& m2,
                                          const Parameters& params );

  /// @}

  /// \name Coordinate projection helper routines
  /// @{

  /*!
   * \brief Compute the local 2D coordinates of an array of points on the
   *  contact plane
   *
   * \param [in] pX array of global x coordinates for input points
   * \param [in] pY array of global y coordinates for input points
   * \param [in] pZ array of global z coordinates for input points
   * \param [in,out] pLX array of local x coordinates of transformed points
   * \param [in,out] pLY array of local y coordinates of transformed points
   * \param [in] size number of points in arrays
   *
   * \pre length(pX), length(pY), length(pZ) >= size
   * \pre length(pLX), length(pLY) >= size
   */
  TRIBOL_HOST_DEVICE void globalTo2DLocalCoords( RealT* pX, RealT* pY, RealT* pZ, RealT* pLX, RealT* pLY, int size );

  /*!
   * \brief Compute the local 2D coordinates of a point on the contact plane
   *
   * \param [in] pX global x coordinate of point
   * \param [in] pY global y coordinate of point
   * \param [in] pZ global z coordinate of point
   * \param [in,out] pLX local x coordinate of point on contact plane
   * \param [in,out] pLY local y coordinate of point on contact plane
   *
   * \note Overloaded member function to compute local coordinates of
   *  a single point on the contact plane
   */
  void globalTo2DLocalCoords( RealT pX, RealT pY, RealT pZ, RealT& pLX, RealT& pLY, int size );

  /*!
   * \brief Transform a local 2D point on the contact plane to global 3D
   *  coordinates
   *
   * \param [in] xloc local x coordinate of point
   * \param [in] yloc local y coordinate of point
   * \param [in,out] xg global x coordinate of point
   * \param [in,out] yg global y coordinate of point
   * \param [in,out] zg global z coordinate of point
   *
   */
  TRIBOL_HOST_DEVICE void local2DToGlobalCoords( RealT xloc, RealT yloc, RealT& xg, RealT& yg, RealT& zg );

  /// @}

  /// \name Getters and setters
  /// @{

  /*!
   * \brief Get the id of the first element that forms the contact plane
   *
   * \return Face id
   */
  TRIBOL_HOST_DEVICE int getCpElementId1() const { return m_pair->m_element_id1; }

  /*!
   * \brief Get the id of the second element that forms the contact plane
   *
   * \return Face id
   */
  TRIBOL_HOST_DEVICE int getCpElementId2() const { return m_pair->m_element_id2; }

  /*!
   * \brief Set the first contact plane element id
   *
   * \param [in] element_id element id
   */
  void setCpElementId1( IndexT element_id ) { m_pair->m_element_id1 = element_id; }

  /*!
   * \brief Set the second contact plane element id
   *
   * \param [in] element_id element id
   */
  void setCpElementId2( IndexT element_id ) { m_pair->m_element_id2 = element_id; }

  /// @}
};

//-----------------------------------------------------------------------------
// Common Plane Computational Geometry Class
//-----------------------------------------------------------------------------
class CommonPlanePair : public ContactPlanePair {

 public:

  /*!
   * @brief Constructs a common plane contact plane
   *
   */
  TRIBOL_HOST_DEVICE CommonPlanePair() {};

  /*!
   * @brief Overloaded constructor
   *
   */
  TRIBOL_HOST_DEVICE CommonPlanePair( InterfacePair* pair, Parameters& params, const int dim );

  /*!
   * \brief Destructor
   *
   */
  ~CommonPlanePair() = default;

 protected:

  // Assuming a convex quadrilateral in 3D with only TWO line/edge plane intersections,
  // you can have a max of 5 vertices associated with the interpenetrating portion of the
  // four node quadrilateral face. This configuration is a 1-3 configuration with 3 nodes
  // interpenetrating and one node not.
  static constexpr max_nodes_per_intersection{ 5 };

 public:

  int m_numInterpenPoly1Vert; ///< Number of vertices on face 1 interpenetrating polygon
  RealT m_interpenG1X[ max_nodes_per_intersection ];  ///< Global x-coordinate of face 1 interpenetrating polygon as projected onto the common plane 
  RealT m_interpenG1Y[ max_nodes_per_intersection ];  ///< Global y-coordinate of face 1 interpenetrating polygon as projected onto the common plane
  RealT m_interpenG1Z[ max_nodes_per_intersection ];  ///< Global z-coordinate of face 1 interpenetrating polygon as projected onto the common plane

  int m_numInterpenPoly2Vert; ///< Number of vertices on face 2 interpenetrating polygon
  RealT m_interpenG2X[ max_nodes_per_intersection ];  ///< Global x-coordinate of face 2 interpenetrating polygon as projected onto the common plane
  RealT m_interpenG2Y[ max_nodes_per_intersection ];  ///< Global y-coordinate of face 2 interpenetrating polygon as projected onto the common plane 
  RealT m_interpenG2Z[ max_nodes_per_intersection ];  ///< Global z-coordinate of face 2 interpenetrating polygon as projected onto the common plane

  RealT m_velGap; ///< Velocity gap
  RealT m_ratePressure; ///< gap-rate pressure
  RealT m_pressure; ///< kinematic contact pressure

  /*!
   * \brief check to see if face-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   * 
   * \return face geometry error
   */
  TRIBOL_HOST_DEVICE FaceGeomError checkInterfacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) override;

  /*!
   * \brief check to see if common plane face-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   */
   TRIBOL_HOST_DEVICE FaceGeomError checkFacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) override;

  /*!
   * \brief check to see if common plane edge-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   */
   TRIBOL_HOST_DEVICE FaceGeomError checkEdgePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) override;

  /*!
   * \brief Compute the projected overlap of the interpenetrating portions of each face in 2D
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] params Coupling scheme-dependent parameters
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   */
  TRIBOL_HOST_DEVICE FaceGeomError computeOverlap2D( const MeshData::Viewer& m1, const MeshData::Viewer& m2,
                                                     const Parameters& params ) override;
  /*!
   * \brief Compute the overlap of the interpenetrating portions of each face in 3D
   *
   * \param [in] x1 x-coordinates of the first planar quadrilateral
   * \param [in] y1 y-coordinates of the first planar quadrilateral
   * \param [in] z1 z-coordinates of the first planar quadrilateral
   * \param [in] x2 x-coordinates of the second planar quadrilateral
   * \param [in] y2 y-coordinates of the second planar quadrilateral
   * \param [in] z2 z-coordinates of the second planar quadrilateral
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] params Coupling scheme-dependent parameters
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   *
   * \pre this routine assumes that the two four node quadrilaterals are planar
   */
  TRIBOL_HOST_DEVICE FaceGeomError computeOverlap3D( const RealT* x1, const RealT* y1, const RealT* z1,
                                                     const RealT* x2, const RealT* y2, const RealT* z2,
                                                     const MeshData::Viewer& m1, const MeshData::Viewer& m2,const Parameters& params ) override;

  /*!
   * \brief Project face or interpen vertices onto common plane and compute overlap
   *
   * \param [in] fx1 x-coordinates of the first planar whole or partial face 
   * \param [in] fy1 y-coordinates of the first planar whole or partial face 
   * \param [in] fz1 z-coordinates of the first planar whole or partial face 
   * \param [in] fx2 x-coordinates of the second planar whole or partial face 
   * \param [in] fy2 y-coordinates of the second planar whole or partial face 
   * \param [in] fz2 z-coordinates of the second planar whole or partial face 
   * \param [in] num_vert_1 number of vertices in first whole or partial face
   * \param [in] num_vert_2 number of vertices in second whole or partial face
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] params Coupling scheme-dependent parameters
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   *
   * \pre this routine assumes each whole or partial face is planar
   */
  TRIBOL_HOST_DEVICE FaceGeomError projectPointsAndComputeOverlap( RealT const* const fx1, RealT const* const fy1, RealT const* const fz1,
                                                                   RealT const* const fx2, RealT const* const fy2, RealT const* const fz2,
                                                                   const int num_vert_1, const int num_vert_2, MeshData::Viewer& m1, MeshData::Viewer& m2,
                                                                   Parameters& params );
  /*!
   * \brief Compute the unit normal that defines the contact plane
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   */
  TRIBOL_HOST_DEVICE void computeNormal( const MeshData::Viewer& m1, const MeshData::Viewer& m2 ) override;

  /*!
   * \brief Computes a reference point on the plane locating it in 3-space
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   *
   * \note This is taken as the average of the vertex averaged centroids of
   *  the two faces that are used to define a local contact plane
   */
  TRIBOL_HOST_DEVICE void computePlanePoint( const MeshData::Viewer& m1, const MeshData::Viewer& m2 ) override;

  /*!
   * \brief Compute a local basis on the contact plane
   *
   * \param [in] m1 mesh data viewer for mesh 1
   */
  TRIBOL_HOST_DEVICE void computeLocalBasis( const MeshData::Viewer& m1 ) override;

  /*!
   * \brief Recomputes the reference point that locates the plane in 3-space
   *        and the gap between the projected `intersection` poly centroids
   *
   * \note This projects the projected area of overlap's centroid (from the
   *  polygon intersection routine) back to each face that are used to form
   *  the contact plane and then averages these projected points.
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   */
  TRIBOL_HOST_DEVICE void resetPlanePointAndCentroidGap( const MeshData::Viewer& m1, const MeshData::Viewer& m2 );

  /*!
   * \brief Check whether two polygons (faces) have a positive area of overlap
   *
   * \note Wrapper routine that calls the polygon intersection routine. That routine
   *  does not return vertices, just overlap area. This is the FULL overlap calculation.
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] projLocX1 2D x-coordinates of projected element 1 vertices
   * \param [in] projLocY1 2D y-coordinates of projected element 1 vertices
   * \param [in] projLocX2 2D x-coordinates of projected element 2 vertices
   * \param [in] projLocY2 2D y-coordinates of projected element 2 vertices
   * \param [in] isym 0 for planar symmetry, 1 for axial symmetry
   */
  TRIBOL_HOST_DEVICE void checkPolyOverlap( const MeshData::Viewer& m1, const MeshData::Viewer& m2, RealT* projLocX1,
                                            RealT* projLocY1, RealT* projLocX2, RealT* projLocY2, const int isym );

  /*!
   * \brief Computes the discrete scalar gap between the two projections of the contact
   *        plane centroid onto each constituent face.
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] scale Scale to help find centroid-to-face projections
   * 
   * \note this routine computes and stores the gap on the CommonPlane object
   */
  void centroidGap( const MeshData::Viewer& m1, const MeshData::Viewer& m2, RealT scale );

  /*!
   *
   * \brief checks the contact plane gap against the maximum allowable interpenetration
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   * \param [in] faceId1 face id for face belonging to mesh 1
   * \param [in] faceId2 face id for face belonging to mesh 2
   * \param [in] auto_contact_pen_frac Allowable interpenetration as a fraction of element thickness for auto-contact
   * \param [in] gap the contact plane gap
   *
   * \return true if the gap exceeds the max allowable interpenetration
   *
   * \pre this function is for use with ContactCase = AUTO to preclude face-pairs on opposite
   *      sides of thin structures/plates
   *
   */
  TRIBOL_HOST_DEVICE bool ExceedsMaxAutoInterpen( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                                  const int faceId1, const int faceId2, RealT auto_contact_pen_frac,
                                                  const RealT gap );

};

//-----------------------------------------------------------------------------
// Single Mortar Computational Geometry Class
//-----------------------------------------------------------------------------
class MortarPlanePair : public ContactPlanePair {

 public:

  /*!
   * @brief Constructs a Mortar contact plane
   *
   */
  TRIBOL_HOST_DEVICE MortarPlanePair() {};

  /*!
   * @brief Overloaded constructor
   *
   */
  TRIBOL_HOST_DEVICE MortarPlanePair( InterfacePair* pair, Parameters& params, const int dim );

  /*!
   * \brief Destructor
   *
   */
  ~MortarPlanePair() = default;

  /*!
   * \brief check to see if face-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry error
   */
  TRIBOL_HOST_DEVICE FaceGeomError checkInterfacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) override;

  /*!
   * \brief check to see if mortar plane face-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   */
   TRIBOL_HOST_DEVICE FaceGeomError checkFacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 );

  /*!
   * \brief check to see if mortar plane edge-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   */
   TRIBOL_HOST_DEVICE FaceGeomError checkEdgePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) override;

  /*!
   * \brief Compute the projected overlap in 2D
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] params Coupling scheme-dependent parameters
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   */
  TRIBOL_HOST_DEVICE FaceGeomError computeOverlap2D( const MeshData::Viewer& m1, const MeshData::Viewer& m2,
                                                     const Parameters& params ) override;

  /*!
   * \brief Compute the projected overlap in 3D 
   *
   * \param [in] x1 x-coordinates of the first planar quadrilateral
   * \param [in] y1 y-coordinates of the first planar quadrilateral
   * \param [in] z1 z-coordinates of the first planar quadrilateral
   * \param [in] x2 x-coordinates of the second planar quadrilateral
   * \param [in] y2 y-coordinates of the second planar quadrilateral
   * \param [in] z2 z-coordinates of the second planar quadrilateral
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] params Coupling scheme-dependent parameters
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   *
   * \pre this routine assumes that the two four node quadrilaterals are planar
   */
  TRIBOL_HOST_DEVICE FaceGeomError computeOverlap3D( const RealT* x1, const RealT* y1, const RealT* z1,
                                                     const RealT* x2, const RealT* y2, const RealT* z2,
                                                     const MeshData::Viewer& m1, const MeshData::Viewer& m2,const Parameters& params ) override;

  /*!
   * \brief Compute the unit normal that defines the contact plane
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   */
  TRIBOL_HOST_DEVICE void computeNormal( const MeshData::Viewer& m1, const MeshData::Viewer& m2 ) override;

  /*!
   * \brief Computes a reference point on the plane locating it in 3-space
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   *
   * \note this is taken as the vertex average centroid of the nonmortar face
   */
  TRIBOL_HOST_DEVICE void computePlanePoint( const MeshData::Viewer& m1, const MeshData::Viewer& m2 ) override;

  /*!
   * \brief Compute a local basis on the contact plane
   *
   * \param [in] m1 mesh data viewer for mesh 1
   */
  TRIBOL_HOST_DEVICE void computeLocalBasis( const MeshData::Viewer& m1 ) override;

};

//-----------------------------------------------------------------------------
// Aligned Mortar Computational Geometry Class
//-----------------------------------------------------------------------------
class AlignedMortarPlanePair : public ContactPlanePair {

 public:

  /*!
   * @brief Constructs a Mortar-based contact plane
   *
   */
  TRIBOL_HOST_DEVICE AlignedMortarPlanePair() {};

  /*!
   * @brief Overloaded constructor
   *
   */
  TRIBOL_HOST_DEVICE AlignedMortarPlanePair( InterfacePair* pair, Parameters& params, const int dim );

  /*!
   * \brief Destructor
   *
   */
  ~AlignedMortarPlanePair() = default;

  /*!
   * \brief check to see if face-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry error
   */
  TRIBOL_HOST_DEVICE FaceGeomError checkInterfacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) override;

  /*!
   * \brief check to see if aligned mortar plane face-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \note Aligned mortar only works in 3D (e.g. face-pairs)
   */
   TRIBOL_HOST_DEVICE FaceGeomError checkFacePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) override;

  /*!
   * \brief check to see if aligned mortar plane edge-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   */
   TRIBOL_HOST_DEVICE FaceGeomError checkEdgePair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) override;

  /*!
   * \brief Compute the projected overlap in 2D
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] params Coupling scheme-dependent parameters
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   */
  TRIBOL_HOST_DEVICE FaceGeomError computeOverlap2D( const MeshData::Viewer& m1, const MeshData::Viewer& m2,
                                                     const Parameters& params ) override;
  /*!
   * \brief Compute the projected overlap in 3D 
   *
   * \param [in] x1 x-coordinates of the first planar quadrilateral
   * \param [in] y1 y-coordinates of the first planar quadrilateral
   * \param [in] z1 z-coordinates of the first planar quadrilateral
   * \param [in] x2 x-coordinates of the second planar quadrilateral
   * \param [in] y2 y-coordinates of the second planar quadrilateral
   * \param [in] z2 z-coordinates of the second planar quadrilateral
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] params Coupling scheme-dependent parameters
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   *
   * \pre this routine assumes that the two four node quadrilaterals are planar
   */
  TRIBOL_HOST_DEVICE FaceGeomError computeOverlap3D( const RealT* x1, const RealT* y1, const RealT* z1,
                                                     const RealT* x2, const RealT* y2, const RealT* z2,
                                                     const MeshData::Viewer& m1, const MeshData::Viewer& m2,const Parameters& params ) override;

  /*!
   * \brief Compute the unit normal that defines the contact plane
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   */
  TRIBOL_HOST_DEVICE void computeNormal( const MeshData::Viewer& m1, const MeshData::Viewer& m2 ) override;

  /*!
   * \brief Computes a reference point on the plane locating it in 3-space
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   *
   * \note this is taken as the vertex average centroid of the nonmortar face
   */
  TRIBOL_HOST_DEVICE void computePlanePoint( const MeshData::Viewer& m1, const MeshData::Viewer& m2 ) override;

  /*!
   * \brief Compute a local basis on the contact plane
   *
   * \param [in] m1 mesh data viewer for mesh 1
   */
  TRIBOL_HOST_DEVICE void computeLocalBasis( const MeshData::Viewer& m1 ) override;

};

//-----------------------------------------------------------------------------
// Free functions
//-----------------------------------------------------------------------------
/*!
 * \brief higher level routine wrapping face and edge-pair interaction checks
 *
 * \param [in] pair interface pair containing pair related indices
 * \param [in] mesh1 mesh data viewer for mesh 1
 * \param [in] mesh2 mesh data viewer for mesh 2
 * \param [in] params coupling-scheme specific parameters
 * \param [in] cMethod the Tribol contact method
 * \param [in] cCase the Tribol contact Case
 * \param [in,out] isInteracting true if pair passes all computational geometry filters
 * \param [in,out] cg viewer of the computational geometry container
 * \param [in,out] plane_ct number of contact planes in the array views
 *
 * \note isInteracting is true indicating a contact candidate for intersecting or
 *       nearly intersecting face-pairs with a positive area of overlap
 *
 * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
 *
 * \note will need the contact case for specialized geometry checks
 *
 */
TRIBOL_HOST_DEVICE FaceGeomError CheckInterfacePair( InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                     const MeshData::Viewer& mesh2, const Parameters& params,
                                                     ContactMethod const cMethod, ContactCase const cCase,
                                                     bool& isInteracting, CompGeom::Viewer& cg, IndexT* plane_ct );

}  // namespace tribol

#endif /* SRC_GEOM_COMPGEOM_HPP_ */
