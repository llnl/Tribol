// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_GEOM_CONTACTPLANE_HPP_
#define SRC_GEOM_CONTACTPLANE_HPP_

#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/common/Parameters.hpp"
#include "axom/slic.hpp" 

#include <string>

namespace tribol
{

/*!
 *
 * \brief projects all the nodes (vertices) of a given FE face to a 
 *  specified plane
 *
 * \param [in] mesh mesh data viewer
 * \param [in] faceId id for given face
 * \param [in] nrmlX x component of plane's unit normal
 * \param [in] nrmlY y component of plane's unit normal
 * \param [in] nrmlZ z component of plane's unit normal
 * \param [in] cX x coordinate of reference point on the plane
 * \param [in] cY y coordinate of reference point on the plane
 * \param [in] cZ z coordinate of reference point on the plane
 * \param [in,out] pX array of x coordinates of projected nodes
 * \param [in,out] pY array of y coordinates of projected nodes
 * \param [in,out] pZ array of z coordinates of projected nodes
 *
 * \pre length(pX), length(pY), length(pZ) >= number of nodes on face
 */
TRIBOL_HOST_DEVICE void ProjectFaceNodesToPlane( const MeshData::Viewer& mesh, int faceId,
                              RealT nrmlX, RealT nrmlY, RealT nrmlZ,
                              RealT cX, RealT cY, RealT cZ,
                              RealT* pX, RealT* pY, 
                              RealT* pZ );

/*!
 *
 * \brief projects nodes belonging to a surface edge to a contact segment 
 *
 * \param [in] mesh mesh data viewer
 * \param [in] edgeId edge id
 * \param [in] nrmlX x-component of the contact segment normal
 * \param [in] nrmlY y-component of the contact segment normal
 * \param [in] cX x-coordinate of a point on the contact segment
 * \param [in] cY y-coordinate of a point on the contact segment
 * \param [in,out] pX pointer to array of projected nodal x-coordinates
 * \param [in,out] pY pointer to array of projected nodal y-coordinates
 *
 */
TRIBOL_HOST_DEVICE void ProjectEdgeNodesToSegment( const MeshData::Viewer& mesh, int edgeId, 
                                RealT nrmlX, RealT nrmlY, RealT cX, 
                                RealT cY, RealT* pX, 
                                RealT* pY );

/*!
 * \brief checks if the vertices on face2 have interpenetrated the level set 
 *        defined by face 1
 *
 * \param [in] mesh1 mesh data viewer for mesh 1 to which face 1 belongs
 * \param [in] mesh2 mesh data viewer for mesh 2 to which face 2 belongs
 * \param [in] fId1 id for face 1
 * \param [in] fId2 id for face 2
 * \param [in] tol tolerance for including vertices "close" to face1 plane
 * \param [in,out] allVerts indicates if all vertices interpenetrate face1
 *
 * \return true if face 2 intersects the level set of face 1, otherwise false
 *
 * This uses face1 as a level set and checks the projection 
 * of the vector defined by differencing the face1 center and a face2 
 * node onto the face1 normal. If this projection is positive then 
 * interpenetration has occured and face2 intersects the plane defined 
 * by face1 (i.e. the zero level set). 
 * 
 */
TRIBOL_HOST_DEVICE bool FaceInterCheck( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, 
                                        int fId1, int fId2, RealT tol, bool& allVerts );

/*!
 *
 * \brief checks if edge 2 interpenetrates the level set defined by edge 1
 *
 * \param [in] mesh1 mesh data viewer for mesh 1
 * \param [in] mesh2 mesh data viewer for mesh 2
 * \param [in] eId1 edge id for edge belonging to mesh 1
 * \param [in] eId2 edge id for edge belonging to mesh 2
 * \param [in] tol interpenetration tolerance
 * \param [in,out] allVerts true if all of edge2 has interpenetrated 
 *                 edge 1 
 *
 * \return true if edge 2 interpenetrates the level set defined by edge 1
 *
 */
TRIBOL_HOST_DEVICE bool EdgeInterCheck( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, 
                                        int eId1, int eId2, RealT tol, bool& allVerts );

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
TRIBOL_HOST_DEVICE bool ExceedsMaxAutoInterpen( const MeshData::Viewer& mesh1, 
                                                const MeshData::Viewer& mesh2,
                                                const int faceId1, 
                                                const int faceId2, 
                                                RealT auto_contact_pen_frac,
                                                const RealT gap );

//-----------------------------------------------------------------------------
// Contact Plane base class
//-----------------------------------------------------------------------------

class ContactPlane
{
protected:
   InterfacePair* m_pair; ///< Face-pair struct for two constituent faces

   /**
    * @brief Constructs a contact plane
    * 
    * @param pair Proximate candidate interface pair
    * @param areaFrac Sets the minimum allowable area for an overlap
    * @param interpenOverlap If true, overlap includes only parts of face where constraint is violated
    * @param interPlane If true, a common plane is used; if false, a mortar plane is used
    * @param dim Plane dimension
    */
   TRIBOL_HOST_DEVICE ContactPlane( InterfacePair* pair, 
                                    RealT areaFrac, 
                                    bool interpenOverlap, 
                                    bool interPlane,
                                    int dim );

   TRIBOL_HOST_DEVICE ContactPlane();

   virtual ~ContactPlane() = default;

   static constexpr int max_nodes_per_overlap {8};

public:

   int m_dim;            ///< Problem dimension
   int m_numFaces;       ///< Number of constituent faces

   bool m_intermediatePlane; ///< True if intermediate plane is used
   bool m_inContact;         ///< True if face-pair is in contact
   bool m_interpenOverlap;   ///< True if using interpenetration overlap algorithm

   RealT m_cX; ///< Contact plane point global x-coordinate 
   RealT m_cY; ///< Contact plane point global y-coordinate 
   RealT m_cZ; ///< Contact plane point global z-coordinate (zero out for 2D)

   RealT m_cXf1; ///< Global x-coordinate of contact plane centroid projected to face 1
   RealT m_cYf1; ///< Global y-coordinate of contact plane centroid projected to face 1
   RealT m_cZf1; ///< Global z-coordinate of contact plane centroid projected to face 1
  
   RealT m_cXf2; ///< global x-coordinate of contact plane centroid projected to face 2
   RealT m_cYf2; ///< global y-coordinate of contact plane centroid projected to face 2
   RealT m_cZf2; ///< global z-coordinate of contact plane centroid projected to face 2

   int m_numInterpenPoly1Vert; ///< Number of vertices on face 1 interpenetrating polygon
   RealT m_interpenG1X[max_nodes_per_overlap];   ///< Global x-coordinate of face 1 interpenetrating polygon
   RealT m_interpenG1Y[max_nodes_per_overlap];   ///< Global y-coordinate of face 1 interpenetrating polygon
   RealT m_interpenG1Z[max_nodes_per_overlap];   ///< Global z-coordinate of face 1 interpenetrating polygon
   
   int m_numInterpenPoly2Vert; ///< Number of vertices on face 2 interpenetrating polygon
   RealT m_interpenG2X[max_nodes_per_overlap];   ///< Global x-coordinate of face 2 interpenetrating polygon
   RealT m_interpenG2Y[max_nodes_per_overlap];   ///< Global y-coordinate of face 2 interpenetrating polygon
   RealT m_interpenG2Z[max_nodes_per_overlap];   ///< Global z-coordinate of face 2 interpenetrating polygon

   RealT m_nX; ///< Global x-component of contact plane unit normal 
   RealT m_nY; ///< Global y-component of contact plane unit normal
   RealT m_nZ; ///< Global z-component of contact plane unit normal (zero out for 2D)

   RealT m_gap;    ///< Face-pair gap
   RealT m_gapTol; ///< Face-pair gap tolerance

   // cp area
   RealT m_areaFrac;     ///< Face area fraction used to determine overlap area cutoff
   RealT m_areaMin;      ///< Minimum overlap area for inclusion into the active set
   RealT m_area;         ///< Overlap area
   RealT m_interpenArea; ///< Interpenetrating overlap area

   RealT m_velGap;
   RealT m_ratePressure;

   RealT m_pressure;

   /// \name Contact plane routines
   /// @{

   /*!
    * \brief Compute the contact plane normal
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    */
   TRIBOL_HOST_DEVICE virtual void computeNormal( const MeshData::Viewer& m1, 
                                                  const MeshData::Viewer& m2 ) = 0 ; 

   /*!
    * \brief Compute the contact plane point
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    */
   TRIBOL_HOST_DEVICE virtual void computePlanePoint( const MeshData::Viewer& m1,
                                                      const MeshData::Viewer& m2 ) = 0 ;


   /*!
    * \brief Recomputes the reference point that locates the plane in 3-space
    *        and the gap between the projected intersection poly centroids
    *
    * \note This projects the projected area of overlap's centroid (from the 
    *  polygon intersection routine) back to each face that are used to form 
    *  the contact plane and then averages these projected points.
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    * \param [in] scale Scale to help find the centroid-to-face projections
    */
   TRIBOL_HOST_DEVICE void planePointAndCentroidGap( const MeshData::Viewer& m1,
                                                     const MeshData::Viewer& m2,
                                                     RealT scale );

   /*!
    * \brief Compute the contact plane integral gap expression
    */
   virtual void computeIntegralGap() = 0 ;
    
   /*!
    * \brief Compute the contact plane area tolerance
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    * \param [in] params Coupling scheme-dependent parameters
    */
   TRIBOL_HOST_DEVICE virtual void computeAreaTol( const MeshData::Viewer& m1,
                                                   const MeshData::Viewer& m2,
                                                   const Parameters& params ) = 0 ;

   /*!
    * \brief Compute the contact plane centroid gap
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    * \param [in] scale Scale to help find centroid-to-face projections
    */
   virtual void centroidGap( const MeshData::Viewer& m1,
                             const MeshData::Viewer& m2,
                             RealT scale ) = 0 ;

   /*!
    * \brief Compute the contact plane integral gap expression
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    * \param [in] params Coupling scheme-dependent parameters
    * \param [in,out] interpen true if the two faces interpenetrate
    *
    * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
    */
   TRIBOL_HOST_DEVICE virtual FaceGeomError computeLocalInterpenOverlap( 
      const MeshData::Viewer& m1, const MeshData::Viewer& m2, 
      const Parameters& params, bool& interpen ) = 0;

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
    * \brief Get the number of faces used to form the contact plane 
    *
    * \return Number of faces
    *
    * \note Number of faces should typically be 2
    *
    */
   int getCpNumFaces() const
   {
     return m_numFaces;
   }
   
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
   
   /*!
    * \brief Set the number of faces involved in forming the contact plane
    *
    * \param [in] num Number of faces
    */
   void setCpNumFaces( int num )
   {
     m_numFaces = num;
   }

   /// @}
};

//-----------------------------------------------------------------------------
// Contact Plane 3D class
//-----------------------------------------------------------------------------

class ContactPlane3D : public ContactPlane
{
public:

   /*!
    * @brief Constructs a 3D contact plane
    * 
    * @param pair Proximate candidate interface pair
    * @param areaFrac Sets the minimum allowable area for an overlap
    * @param interpenOverlap If true, overlap includes only parts of face where constraint is violated
    * @param interPlane If true, a common plane is used; if false, a mortar plane is used
    */
   TRIBOL_HOST_DEVICE ContactPlane3D( InterfacePair* pair,
                                      RealT areaFrac,
                                      bool interpenOverlap,
                                      bool interPlane );

   /*!
    * Overload constructor with no argument list
    *
    */
   TRIBOL_HOST_DEVICE ContactPlane3D();

   RealT m_e1X; ///< Global x-component of first in-plane basis vector
   RealT m_e1Y; ///< Global y-component of first in-plane basis vector
   RealT m_e1Z; ///< Global z-component of first in-plane basis vector

   RealT m_e2X; ///< Global x-component of second in-plane basis vector
   RealT m_e2Y; ///< Global y-component of second in-plane basis vector
   RealT m_e2Z; ///< Global z-component of second in-plane basis vector

   RealT m_polyLocX[max_nodes_per_overlap]; ///< Pointer to local x-components of overlap polygon's vertices
   RealT m_polyLocY[max_nodes_per_overlap]; ///< Pointer to local y-components of overlap polygon's vertices 

   RealT m_polyX[max_nodes_per_overlap]; ///< Global x-components of overlap polygon's vertices
   RealT m_polyY[max_nodes_per_overlap]; ///< Global y-components of overlap polygon's vertices
   RealT m_polyZ[max_nodes_per_overlap]; ///< Global z-components of overlap polygon's vertices

   int m_numPolyVert; ///< Number of vertices in overlapping polygon

   RealT m_overlapCX; ///< Local x-coordinate of overlap centroid
   RealT m_overlapCY; ///< Local y-coordinate of overlap centroid

   RealT m_interpenPoly1X[max_nodes_per_overlap]; ///< Local x-coordinates of face 1 interpenetrating overlap
   RealT m_interpenPoly1Y[max_nodes_per_overlap]; ///< Local y-coordinates of face 1 interpenetrating overlap

   RealT m_interpenPoly2X[max_nodes_per_overlap]; ///< Local x-coordinates of face 2 interpenetrating overlap
   RealT m_interpenPoly2Y[max_nodes_per_overlap]; ///< Local y-coordinates of face 2 interpenetrating overlap

   /*!
    * \brief Compute the unit normal that defines the contact plane
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    */
   TRIBOL_HOST_DEVICE void computeNormal( const MeshData::Viewer& m1, 
                                          const MeshData::Viewer& m2 ) override;
   
   /*!
    * \brief Computes a reference point on the plane locating it in 3-space
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    *
    * \note This is taken as the average of the vertex averaged centroids of 
    *  the two faces that are used to define a local contact plane
    */
   TRIBOL_HOST_DEVICE void computePlanePoint( const MeshData::Viewer& m1,
                                              const MeshData::Viewer& m2 ) override;
   
   /*!
    * \brief Compute a local basis on the contact plane
    *
    * \param [in] m1 mesh data viewer for mesh 1
    */
   TRIBOL_HOST_DEVICE void computeLocalBasis( const MeshData::Viewer& m1 );
   
   /*!
    * \brief Compute the weak (integral form) gap between the two faces. 
    */
   void computeIntegralGap() override;
   
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
   TRIBOL_HOST_DEVICE void globalTo2DLocalCoords( RealT* pX, RealT* pY, 
                                                  RealT* pZ, RealT* pLX, 
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
   void globalTo2DLocalCoords( RealT pX, RealT pY, RealT pZ,
                               RealT& pLX, RealT& pLY, int size );
   
   /*!
    * \brief Computes the area tolerance for accepting a face pair
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    * \param [in] params Coupling scheme-dependent parameters
    */
   TRIBOL_HOST_DEVICE void computeAreaTol( const MeshData::Viewer& m1,
                                           const MeshData::Viewer& m2,
                                           const Parameters& params ) override;
   
   /*!
    * \brief Check whether two polygons (faces) have a positive area of overlap
    *
    * \note Wrapper routine that calls the polygon intersection routine. That routine
    *  does not return vertices, just overlap area.
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    * \param [in] projLocX1 2D x-coordinates of projected element 1 vertices
    * \param [in] projLocY1 2D y-coordinates of projected element 1 vertices
    * \param [in] projLocX2 2D x-coordinates of projected element 2 vertices
    * \param [in] projLocY2 2D y-coordinates of projected element 2 vertices
    * \param [in] isym 0 for planar symmetry, 1 for axial symmetry
    */
   TRIBOL_HOST_DEVICE void checkPolyOverlap( const MeshData::Viewer& m1,
                                             const MeshData::Viewer& m2,
                                             RealT* projLocX1, RealT* projLocY1, 
                                             RealT* projLocX2, RealT* projLocY2, 
                                             const int isym);

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
   
   /*!
    * \brief Computes the gap between the two projections of the contact 
    *        plane centroid onto each constituent face.
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    * \param [in] scale Scale to help find centroid-to-face projections
    */
   void centroidGap( const MeshData::Viewer& m1,
                     const MeshData::Viewer& m2,
                     RealT scale ) override;

   /*!
    * \brief Compute the contact plane integral gap expression
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    * \param [in] params Coupling scheme-dependent parameters
    * \param [in,out] interpen true if the two faces interpenetrate
    *
    * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
    */
   TRIBOL_HOST_DEVICE FaceGeomError computeLocalInterpenOverlap( 
      const MeshData::Viewer& m1, const MeshData::Viewer& m2, 
      const Parameters& params, bool& interpen ) override;

};


//-----------------------------------------------------------------------------
// Contact Plane 2D class
//-----------------------------------------------------------------------------

class ContactPlane2D : public ContactPlane
{
public:

   RealT m_segX[2]; ///< Global x-components of overlap segment vertices
   RealT m_segY[2]; ///< Global y-components of overlap segment vertices

public:

   /*!
    * \brief Constructor 
    *        
    * \param [in] pair InterfacePair struct 
    * \param [in] lenFrac Length fraction used for overlap segment cutoff
    * \param [in] interpenOverlap True if using interpenetration overlap algorithm
    * \param [in] interPlane True if intermediate (i.e. common) plane is used
    * \param [in] dimension Dimension of problem
    */
   TRIBOL_HOST_DEVICE ContactPlane2D( InterfacePair* pair,
                                      RealT lenFrac,
                                      bool interpenOverlap,
                                      bool interPlane ) ;

   /*!
    * \brief Overloaded constructor with no arguments
    *        
    */
   TRIBOL_HOST_DEVICE ContactPlane2D();

   /*!
    * \brief Destructor 
    *        
    */
   ~ContactPlane2D() = default;

   /*!
    * \brief Compute the unit normal that defines the contact plane
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    */
   TRIBOL_HOST_DEVICE void computeNormal( const MeshData::Viewer& m1, 
                                          const MeshData::Viewer& m2 ) override;

   /*!
    * \brief Computes a reference point on the plane locating it in 3-space
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    *
    * \note This is taken as the average of the vertex averaged centroids of 
    *  the two faces that are used to define a local contact plane
    */
   TRIBOL_HOST_DEVICE void computePlanePoint( const MeshData::Viewer& m1,
                                              const MeshData::Viewer& m2 ) override;

   /*!
    * \brief Compute the weak (integral form) gap between the two faces. 
    */
   void computeIntegralGap() override;
    
   /*!
    * \brief Computes the area tolerance for accepting a face pair
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    * \param [in] params Coupling scheme-dependent parameters
    */
   TRIBOL_HOST_DEVICE void computeAreaTol( const MeshData::Viewer& m1,
                                           const MeshData::Viewer& m2,
                                           const Parameters& params ) override;

   /*!
    * \brief Computes the gap between the two projections of the contact 
    *        plane centroid onto each constituent face.
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    * \param [in] scale Scale to help find centroid-to-face projections
    */
   void centroidGap( const MeshData::Viewer& m1,
                     const MeshData::Viewer& m2,
                     RealT scale  ) override;

   /*!
    * \brief Check whether two segments have a positive length of overlap 
    *
    */
   TRIBOL_HOST_DEVICE void checkSegOverlap( const RealT* const pX1, const RealT* const pY1, 
                                            const RealT* const pX2, const RealT* const pY2, 
                                            const int nV1, const int nV2 );

   /*!
    * \brief Compute the contact plane integral gap expression
    *
    * \param [in] m1 mesh data viewer for mesh 1
    * \param [in] m2 mesh data viewer for mesh 2
    * \param [in] params Coupling scheme-dependent parameters
    * \param [in,out] interpen true if the two faces interpenetrate
    *
    * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
    */
   TRIBOL_HOST_DEVICE FaceGeomError computeLocalInterpenOverlap(
      const MeshData::Viewer& m1, const MeshData::Viewer& m2, 
      const Parameters& params, bool& interpen ) override;

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
 * \param [in,out] planes_2d array view of 2D contact planes
 * \param [in,out] planes_3d array view of 3D contact planes
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
TRIBOL_HOST_DEVICE FaceGeomError CheckInterfacePair( InterfacePair& pair,
                                  const MeshData::Viewer& mesh1,
                                  const MeshData::Viewer& mesh2,
                                  const Parameters& params,
                                  ContactMethod const cMethod,
                                  ContactCase const cCase,
                                  bool& isInteracting,
                                  ArrayViewT<ContactPlane2D>& planes_2d,
                                  ArrayViewT<ContactPlane3D>& planes_3d,
                                  IndexT* plane_ct );


//-----------------------------------------------------------------------------
// Free functions returning Contact Plane objects
//-----------------------------------------------------------------------------

/*!
 * \brief Checks if face-pair (3D) candidate is actual local contact interaction.
 *
 * \param [in,out] cp contact plane object to be populated
 * \param [in] mesh1 mesh data viewer for mesh 1
 * \param [in] mesh2 mesh data viewer for mesh 2
 * \param [in] params coupling-scheme specific parameters
 * \param [in] fullOverlap True if full overlap calculation is used, false if interpenetration calculation is used
 *
 * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
 * 
 */
TRIBOL_HOST_DEVICE FaceGeomError CheckFacePair( ContactPlane3D& cp,
                                                const MeshData::Viewer& mesh1,
                                                const MeshData::Viewer& mesh2,
                                                const Parameters& params,
                                                bool fullOverlap );

/*!
 * \brief Checks if face-pair (3D) candidate is aligned and actual local contact interaction.
 *
 * \param [in] pair interface pair containing pair related indices
 * \param [in] mesh1 mesh data viewer for mesh 1
 * \param [in] mesh2 mesh data viewer for mesh 2
 * \param [in] params coupling-scheme specific parameters
 *
 * \return 3D contact plane object with boolean indicating if face-pair form a local contact interaction
 * 
 */
TRIBOL_HOST_DEVICE ContactPlane3D CheckAlignedFacePair( InterfacePair& pair,
                                                        const MeshData::Viewer& mesh1,
                                                        const MeshData::Viewer& mesh2,
                                                        const Parameters& params );

/*!
 * \brief Checks if 2D edge-pair candidate is actual local contact interaction.
 *
 * \param [in,out] cp contact plane object to be populated
 * \param [in] mesh1 mesh data viewer for mesh 1
 * \param [in] mesh2 mesh data viewer for mesh 2
 * \param [in] params coupling-scheme specific parameters
 * \param [in] fullOverlap True if full overlap calculation is used, false if interpenetration calculation is used
 *
 * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
 * 
 */
TRIBOL_HOST_DEVICE FaceGeomError CheckEdgePair( ContactPlane2D& cp,
                                                const MeshData::Viewer& mesh1,
                                                const MeshData::Viewer& mesh2,
                                                const Parameters& params,
                                                bool fullOverlap );
}

#endif /* SRC_GEOM_CONTACTPLANE_HPP_ */
