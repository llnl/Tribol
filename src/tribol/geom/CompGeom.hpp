// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_COMPGEOM_HPP_
#define SRC_TRIBOL_GEOM_COMPGEOM_HPP_

#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/common/Parameters.hpp"

#include <string>

namespace tribol {

//-----------------------------------------------------------------------------
// Computational geometry base class
// (can be used to extend non-contact-plane classes)
//-----------------------------------------------------------------------------
class CompGeomPair {
 protected:
  InterfacePair* m_pair;  ///< Face-pair struct for two constituent faces

  TRIBOL_HOST_DEVICE CompGeomPair() {};

  TRIBOL_HOST_DEVICE CompGeomPair( InterfacePair* pair, const Parameters& params, const int dim )
      : m_pair( pair ), m_dim( dim ), m_params( params )
  {
  }

  virtual ~CompGeomPair() = default;

 public:
  int m_dim;
  Parameters m_params;
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
  TRIBOL_HOST_DEVICE ContactPlanePair( InterfacePair* pair, const Parameters& params, const int dim );

  /*!
   * \brief Compute a local basis on the contact plane
   *
   */
  TRIBOL_HOST_DEVICE void computeLocalBasis();

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
   */
  TRIBOL_HOST_DEVICE void computeAreaTol( const MeshData::Viewer& m1, const MeshData::Viewer& m2 );

  static constexpr int max_nodes_per_overlap{ 10 };

  static constexpr int max_face_nodes{ 4 };

 public:
  bool m_face1_convex{ true };  // assume convex faces
  bool m_face2_convex{ true };

  RealT m_x1_prime[max_face_nodes];
  RealT m_y1_prime[max_face_nodes];
  RealT m_z1_prime[max_face_nodes];

  RealT m_x2_prime[max_face_nodes];
  RealT m_y2_prime[max_face_nodes];
  RealT m_z2_prime[max_face_nodes];

  RealT m_x1_bar[max_face_nodes];
  RealT m_y1_bar[max_face_nodes];
  RealT m_z1_bar[max_face_nodes];

  RealT m_x2_bar[max_face_nodes];
  RealT m_y2_bar[max_face_nodes];
  RealT m_z2_bar[max_face_nodes];

  bool m_inContact;  ///< True if face-pair is in contact
  RealT m_gap;       ///< Face-pair gap
  RealT m_gapTol;    ///< Face-pair gap tolerance

  RealT m_e1X;  ///< Global x-component of first in-plane basis vector
  RealT m_e1Y;  ///< Global y-component of first in-plane basis vector
  RealT m_e1Z;  ///< Global z-component of first in-plane basis vector

  RealT m_e2X;  ///< Global x-component of second in-plane basis vector
  RealT m_e2Y;  ///< Global y-component of second in-plane basis vector
  RealT m_e2Z;  ///< Global z-component of second in-plane basis vector

  RealT m_cX;  ///< Contact plane overlap centroid global x-coordinate
  RealT m_cY;  ///< Contact plane overlap centroid global y-coordinate
  RealT m_cZ;  ///< Contact plane overlap centroid global z-coordinate (zero out for 2D)

  RealT m_cXf1;  ///< Global x-coordinate of contact plane centroid projected to face 1
  RealT m_cYf1;  ///< Global y-coordinate of contact plane centroid projected to face 1
  RealT m_cZf1;  ///< Global z-coordinate of contact plane centroid projected to face 1

  RealT m_cXf2;  ///< global x-coordinate of contact plane centroid projected to face 2
  RealT m_cYf2;  ///< global y-coordinate of contact plane centroid projected to face 2
  RealT m_cZf2;  ///< global z-coordinate of contact plane centroid projected to face 2

  RealT m_nX;  ///< Global x-component of contact plane unit normal
  RealT m_nY;  ///< Global y-component of contact plane unit normal
  RealT m_nZ;  ///< Global z-component of contact plane unit normal (zero out for 2D)

  int m_numPolyVert;                     ///< Number of vertices in overlapping polygon
  RealT m_polyX[max_nodes_per_overlap];  ///< Global x-components of overlap polygon's vertices
  RealT m_polyY[max_nodes_per_overlap];  ///< Global y-components of overlap polygon's vertices
  RealT m_polyZ[max_nodes_per_overlap];  ///< Global z-components of overlap polygon's vertices

  RealT m_polyLocX[max_nodes_per_overlap];  ///< Pointer to local x-components of overlap polygon's vertices
  RealT m_polyLocY[max_nodes_per_overlap];  ///< Pointer to local y-components of overlap polygon's vertices

  // cp area
  bool m_fullOverlap{ true };  ///< Indicates if a full overlap (true) or interpen overlap (false) is used
  RealT m_areaFrac;            ///< Face area fraction used to determine overlap area cutoff
  RealT m_areaMin;             ///< Minimum overlap area for inclusion into the active set
  RealT m_area;                ///< Overlap area

  /// \name Contact plane routines
  /// @{

  /*!
   * \brief check to see if face-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry exception
   */
  TRIBOL_HOST_DEVICE virtual FaceGeomException checkFacePair(
      const MeshData::Viewer& TRIBOL_UNUSED_PARAM( mesh1 ), const MeshData::Viewer& TRIBOL_UNUSED_PARAM( mesh2 ) ) = 0;

  /*!
   * \brief check to see if edge-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry exception
   */
  TRIBOL_HOST_DEVICE virtual FaceGeomException checkEdgePair(
      const MeshData::Viewer& TRIBOL_UNUSED_PARAM( mesh1 ), const MeshData::Viewer& TRIBOL_UNUSED_PARAM( mesh2 ) ) = 0;

  TRIBOL_HOST_DEVICE void computePlaneData( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
  {
    computeNormal( mesh1, mesh2 );
    computePlanePoint( mesh1, mesh2 );
    computeAreaTol( mesh1, mesh2 );
    if ( m_dim == 3 ) {
      computeLocalBasis();
    }
  }

  TRIBOL_HOST_DEVICE void getFace1Coords( RealT* x1, int num_coords ) const;

  TRIBOL_HOST_DEVICE void getFace2Coords( RealT* x2, int num_coords ) const;
  ;

  TRIBOL_HOST_DEVICE void getFace1ProjectedCoords( RealT* x1_proj, int num_coords ) const;

  TRIBOL_HOST_DEVICE void getFace2ProjectedCoords( RealT* x2_proj, int num_coords ) const;

  TRIBOL_HOST_DEVICE void getContactPlaneNormal( RealT* normal ) const
  {
    normal[0] = m_nX;
    normal[1] = m_nY;
    if ( m_dim == 3 ) {
      normal[2] = m_nZ;
    }
  }

  TRIBOL_HOST_DEVICE void getOverlapVertices( RealT* overlap_verts ) const;

  /*!
   * \brief Compute the projected overlap in 2D
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   *
   * \return 0 if no exception, non-zero (via FaceGeomException enum) otherwise
   */
  TRIBOL_HOST_DEVICE virtual FaceGeomException computeOverlap2D( const MeshData::Viewer& m1,
                                                                 const MeshData::Viewer& m2 ) = 0;
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
   *
   * \return 0 if no exception, non-zero (via FaceGeomException enum) otherwise
   *
   * \pre this routine assumes that the two four node quadrilaterals are planar
   */
  TRIBOL_HOST_DEVICE virtual FaceGeomException computeOverlap3D( const RealT* x1, const RealT* y1, const RealT* z1,
                                                                 const RealT* x2, const RealT* y2, const RealT* z2,
                                                                 const MeshData::Viewer& m1,
                                                                 const MeshData::Viewer& m2 ) = 0;
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
  TRIBOL_HOST_DEVICE void globalTo2DLocalCoords( const RealT* pX, const RealT* pY, const RealT* pZ, RealT* pLX,
                                                 RealT* pLY, int size );

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
  TRIBOL_HOST_DEVICE CommonPlanePair( InterfacePair* pair, const Parameters& params, const int dim );

  /*!
   * \brief Destructor
   *
   */
  ~CommonPlanePair() = default;

  /*!
   * \brief check to see if common plane face-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry exception (0 no exception, >0 otherwise)
   */
  TRIBOL_HOST_DEVICE FaceGeomException checkFacePair( const MeshData::Viewer& mesh1,
                                                      const MeshData::Viewer& mesh2 ) override;

  /*!
   * \brief check to see if common plane edge-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry exception (0 no exception, >0 otherwise)
   */
  TRIBOL_HOST_DEVICE FaceGeomException checkEdgePair( const MeshData::Viewer& mesh1,
                                                      const MeshData::Viewer& mesh2 ) override;

 protected:
  // Assuming a convex quadrilateral in 3D with only TWO line/edge plane intersections,
  // you can have a max of 5 vertices associated with the interpenetrating portion of the
  // four node quadrilateral face. This configuration is a 1-3 configuration with 3 nodes
  // interpenetrating and one node not.
  static constexpr int max_nodes_per_intersection{ 5 };

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

 public:
  int m_numInterpenPoly1Vert;                       ///< Number of vertices on face 1 interpenetrating polygon
  RealT m_interpenG1X[max_nodes_per_intersection];  ///< Global x-coordinate of face 1 interpenetrating polygon as
                                                    ///< projected onto the common plane
  RealT m_interpenG1Y[max_nodes_per_intersection];  ///< Global y-coordinate of face 1 interpenetrating polygon as
                                                    ///< projected onto the common plane
  RealT m_interpenG1Z[max_nodes_per_intersection];  ///< Global z-coordinate of face 1 interpenetrating polygon as
                                                    ///< projected onto the common plane

  int m_numInterpenPoly2Vert;                       ///< Number of vertices on face 2 interpenetrating polygon
  RealT m_interpenG2X[max_nodes_per_intersection];  ///< Global x-coordinate of face 2 interpenetrating polygon as
                                                    ///< projected onto the common plane
  RealT m_interpenG2Y[max_nodes_per_intersection];  ///< Global y-coordinate of face 2 interpenetrating polygon as
                                                    ///< projected onto the common plane
  RealT m_interpenG2Z[max_nodes_per_intersection];  ///< Global z-coordinate of face 2 interpenetrating polygon as
                                                    ///< projected onto the common plane

  RealT m_velGap;        ///< Velocity gap
  RealT m_ratePressure;  ///< gap-rate pressure
  RealT m_pressure;      ///< kinematic contact pressure

  /*!
   * \brief Compute the projected overlap of the interpenetrating portions of each face in 2D
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   *
   * \return 0 if no exception, non-zero (via FaceGeomException enum) otherwise
   */
  TRIBOL_HOST_DEVICE FaceGeomException computeOverlap2D( const MeshData::Viewer& m1,
                                                         const MeshData::Viewer& m2 ) override;

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
   *
   * \return 0 if no exception, non-zero (via FaceGeomException enum) otherwise
   *
   * \pre this routine assumes that the two four node quadrilaterals are planar
   */
  TRIBOL_HOST_DEVICE FaceGeomException computeOverlap3D( const RealT* x1, const RealT* y1, const RealT* z1,
                                                         const RealT* x2, const RealT* y2, const RealT* z2,
                                                         const MeshData::Viewer& m1,
                                                         const MeshData::Viewer& m2 ) override;

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
   *
   * \return 0 if no exception, non-zero (via FaceGeomException enum) otherwise
   *
   * \pre this routine assumes each whole or partial face is planar
   */
  TRIBOL_HOST_DEVICE FaceGeomException projectPointsAndComputeOverlap( RealT const* const fx1, RealT const* const fy1,
                                                                       RealT const* const fz1, RealT const* const fx2,
                                                                       RealT const* const fy2, RealT const* const fz2,
                                                                       const int num_vert_1, const int num_vert_2,
                                                                       const MeshData::Viewer& m1,
                                                                       const MeshData::Viewer& m2 );
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
   * \brief Computes the discrete scalar gap between the two projections of the contact
   *        plane centroid onto each constituent face.
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   * \param [in] scale Scale to help find centroid-to-face projections
   *
   * \note this routine computes and stores the gap on the CommonPlane object
   */
  TRIBOL_HOST_DEVICE void centroidGap( const MeshData::Viewer& m1, const MeshData::Viewer& m2, RealT scale );

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
  TRIBOL_HOST_DEVICE bool exceedsMaxAutoInterpen( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                                  const int faceId1, const int faceId2, const Parameters& params,
                                                  const RealT gap );

};  // end class CommonPlanePair

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
  TRIBOL_HOST_DEVICE MortarPlanePair( InterfacePair* pair, const Parameters& params, const int dim );

  /*!
   * \brief Destructor
   *
   */
  ~MortarPlanePair() = default;
  /*!
   * \brief check to see if mortar plane face-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry exception (0 if no exception, >0 otherwise)
   */
  TRIBOL_HOST_DEVICE FaceGeomException checkFacePair( const MeshData::Viewer& mesh1,
                                                      const MeshData::Viewer& mesh2 ) override;

  /*!
   * \brief check to see if mortar plane edge-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry exception (0 if no exception, >0 otherwise)
   */
  TRIBOL_HOST_DEVICE FaceGeomException checkEdgePair( const MeshData::Viewer& mesh1,
                                                      const MeshData::Viewer& mesh2 ) override;

 protected:
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

 public:
  /*!
   * \brief Compute the projected overlap in 2D
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   *
   * \return 0 if no exception, non-zero (via FaceGeomException enum) otherwise
   */
  TRIBOL_HOST_DEVICE FaceGeomException computeOverlap2D( const MeshData::Viewer& m1,
                                                         const MeshData::Viewer& m2 ) override;

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
   *
   * \return 0 if no exception, non-zero (via FaceGeomException enum) otherwise
   *
   * \pre this routine assumes that the two four node quadrilaterals are planar
   */
  TRIBOL_HOST_DEVICE FaceGeomException computeOverlap3D( const RealT* x1, const RealT* y1, const RealT* z1,
                                                         const RealT* x2, const RealT* y2, const RealT* z2,
                                                         const MeshData::Viewer& m1,
                                                         const MeshData::Viewer& m2 ) override;

};  // end class MortarPlanePair

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
  TRIBOL_HOST_DEVICE AlignedMortarPlanePair( InterfacePair* pair, const Parameters& params, const int dim );

  /*!
   * \brief Destructor
   *
   */
  ~AlignedMortarPlanePair() = default;

  /*!
   * \brief check to see if aligned mortar plane face-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \note Aligned mortar only works in 3D (e.g. face-pairs)
   *
   * \return face geometry exception (0 if no exception, >0 otherwise)
   */
  TRIBOL_HOST_DEVICE FaceGeomException checkFacePair( const MeshData::Viewer& mesh1,
                                                      const MeshData::Viewer& mesh2 ) override;

  /*!
   * \brief check to see if aligned mortar plane edge-pairs are interacting
   *
   * \param [in] mesh1 mesh data viewer for mesh 1
   * \param [in] mesh2 mesh data viewer for mesh 2
   *
   * \return face geometry exception (0 if no exception, >0 otherwise)
   */
  TRIBOL_HOST_DEVICE FaceGeomException checkEdgePair( const MeshData::Viewer& mesh1,
                                                      const MeshData::Viewer& mesh2 ) override;

 protected:
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

 public:
  /*!
   * \brief Compute the projected overlap in 2D
   *
   * \param [in] m1 mesh data viewer for mesh 1
   * \param [in] m2 mesh data viewer for mesh 2
   *
   * \return 0 if no exception, non-zero (via FaceGeomException enum) otherwise
   */
  TRIBOL_HOST_DEVICE FaceGeomException computeOverlap2D( const MeshData::Viewer& m1,
                                                         const MeshData::Viewer& m2 ) override;
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
   *
   * \return 0 if no exception, non-zero (via FaceGeomException enum) otherwise
   *
   * \pre this routine assumes that the two four node quadrilaterals are planar
   */
  TRIBOL_HOST_DEVICE FaceGeomException computeOverlap3D( const RealT* x1, const RealT* y1, const RealT* z1,
                                                         const RealT* x2, const RealT* y2, const RealT* z2,
                                                         const MeshData::Viewer& m1,
                                                         const MeshData::Viewer& m2 ) override;

};  // end class AlignedMortarPlanePair

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
    Viewer( CompGeom& cg )
        : m_common_plane_pairs( cg.m_common_plane_pairs.view() ),
          m_mortar_plane_pairs( cg.m_mortar_plane_pairs.view() ),
          m_aligned_mortar_plane_pairs( cg.m_aligned_mortar_plane_pairs.view() )
    {
    }

    /**
     * @brief Get a single common plane from the array view
     *
     * @return common plane object
     */
    TRIBOL_HOST_DEVICE CommonPlanePair& getCommonPlane( int id ) const { return m_common_plane_pairs[id]; }

    /**
     * @brief Get a single mortar plane from the array view
     *
     * @return mortar plane object
     */
    TRIBOL_HOST_DEVICE MortarPlanePair& getMortarPlane( int id ) const { return m_mortar_plane_pairs[id]; }

    /**
     * @brief Get a single aligned mortar plane from the array view
     *
     * @return algined mortar plane object
     */
    TRIBOL_HOST_DEVICE AlignedMortarPlanePair& getAlignedMortarPlane( int id ) const
    {
      return m_aligned_mortar_plane_pairs[id];
    }

    template <typename T>
    TRIBOL_HOST_DEVICE T& getPlane( int id );

   private:
    ArrayViewT<CommonPlanePair> m_common_plane_pairs;
    ArrayViewT<MortarPlanePair> m_mortar_plane_pairs;
    ArrayViewT<AlignedMortarPlanePair> m_aligned_mortar_plane_pairs;

  };  // end CompGeom::Viewer

  /*!
   * @brief Constructs a comp geom object
   *
   */
  CompGeom() {};

  /*!
   * @brief Destructor
   *
   */
  virtual ~CompGeom() = default;

  // Prevent copying
  CompGeom( const CompGeom& other ) = delete;
  CompGeom& operator=( const CompGeom& other ) = delete;
  // Enable moving
  CompGeom( CompGeom&& other ) = default;
  CompGeom& operator=( CompGeom&& other ) = default;

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
  const ArrayT<CommonPlanePair>& getCommonPlanePairs() const { return m_common_plane_pairs; }

  /**
   * @brief Get a single common plane
   *
   * @return common plane object
   */
  const CommonPlanePair& getCommonPlane( int id ) const { return m_common_plane_pairs[id]; }

  /**
   * @brief Get the list of mortar plane pairs
   *
   * @return ArrayT of mortar plane pairs
   */
  const ArrayT<MortarPlanePair>& getMortarPlanePairs() const { return m_mortar_plane_pairs; }

  /**
   * @brief Get a single mortar plane
   *
   * @return mortar plane object
   */
  const MortarPlanePair& getMortarPlane( int id ) const { return m_mortar_plane_pairs[id]; }

  /**
   * @brief Get the list of aligned mortar plane pairs
   *
   * @return ArrayT of aligned mortar plane pairs
   */
  const ArrayT<AlignedMortarPlanePair>& getAlignedMortarPlanePairs() const { return m_aligned_mortar_plane_pairs; }

  /**
   * @brief Get a single aligned mortar plane
   *
   * @return aligned mortar plane object
   */
  const AlignedMortarPlanePair& getAlignedMortarPlane( int id ) const { return m_aligned_mortar_plane_pairs[id]; }

  /**
   * @brief Allocate contact plane pairs arrays based on contact method
   *
   */
  void allocatePlanePairs( const ContactMethod method, const int num_pairs, const int allocator_id )
  {
    // clear and allocate the appropriate computational geometry pairs
    switch ( method ) {
      case COMMON_PLANE: {
        m_common_plane_pairs = ArrayT<CommonPlanePair>( num_pairs, num_pairs, allocator_id );
        break;
      }
      case SINGLE_MORTAR:
      case MORTAR_WEIGHTS: {
        m_mortar_plane_pairs = ArrayT<MortarPlanePair>( num_pairs, num_pairs, allocator_id );
        break;
      }
      case ALIGNED_MORTAR: {
        m_aligned_mortar_plane_pairs = ArrayT<AlignedMortarPlanePair>( num_pairs, num_pairs, allocator_id );
        break;
      }
      default: {
        // no-op
        break;
      }
    }  // end switch
  }

  int getNumActivePairs( const ContactMethod method ) const
  {
    switch ( method ) {
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
    }  // end switch
    return 0;
  }  // end getNumActivePairs()

  /**
   * @brief Resize the appropriate contact plane array view
   *
   */
  void resizeActivePairs( ContactMethod method, int size )
  {
    switch ( method ) {
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
    }  // end switch
  }  // end resizeActivePairs()

 private:
  ArrayT<CommonPlanePair> m_common_plane_pairs;
  ArrayT<MortarPlanePair> m_mortar_plane_pairs;
  ArrayT<AlignedMortarPlanePair> m_aligned_mortar_plane_pairs;
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
 * \return 0 if no exception, non-zero (via FaceGeomException enum) otherwise
 *
 * \note will need the contact case for specialized geometry checks
 *
 */
TRIBOL_HOST_DEVICE FaceGeomException CheckInterfacePair( InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                         const MeshData::Viewer& mesh2, const Parameters& params,
                                                         ContactMethod const cMethod, ContactCase const cCase,
                                                         bool& isInteracting, CompGeom::Viewer& cg, IndexT* plane_ct );

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_COMPGEOM_HPP_ */
