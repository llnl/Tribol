// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_GEOM_CONTACTPLANE_HPP_
#define SRC_GEOM_CONTACTPLANE_HPP_

#include "tribol/types.hpp"
#include "tribol/mesh/MeshManager.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/common/Parameters.hpp"
#include "axom/slic.hpp"

#include <string>

namespace tribol {

// forward declaration(s)
class ContactPlaneManager;

//-----------------------------------------------------------------------------
// Free functions
//-----------------------------------------------------------------------------
/*!
 * \brief higher level routine wrapping face and edge-pair interaction checks
 *
 * \param [in] pair interface pair containing pair related indices
 * \param [in] cMethod the Tribol contact method
 * \param [in] cCase the Tribol contact Case
 * \param [in] isInteracting true if pair passes all computational geometry filters
 *
 * \note isInteracting is true indicating a contact candidate for intersecting or
 *       nearly intersecting face-pairs with a positive area of overlap
 *
 * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
 *
 * \note will need the contact case for specialized geometry checks
 *
 */
FaceGeomError CheckInterfacePair( InterfacePair& pair, ContactMethod const cMethod, ContactCase const cCase,
                                  bool& isInteracting );

/*!
 *
 * \brief projects all the nodes (vertices) of a given FE face to a
 *  specified plane
 *
 * \param [in] mesh mesh data
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
void ProjectFaceNodesToPlane( const MeshData& mesh, int faceId, real nrmlX, real nrmlY, real nrmlZ, real cX, real cY,
                              real cZ, real* RESTRICT pX, real* RESTRICT pY, real* RESTRICT pZ );

/*!
 *
 * \brief projects nodes belonging to a surface edge to a contact segment
 *
 * \param [in] mesh MeshData object
 * \param [in] edgeId edge id
 * \param [in] nrmlX x-component of the contact segment normal
 * \param [in] nrmlY y-component of the contact segment normal
 * \param [in] cX x-coordinate of a point on the contact segment
 * \param [in] cY y-coordinate of a point on the contact segment
 * \param [in,out] pX pointer to array of projected nodal x-coordinates
 * \param [in,out] pY pointer to array of projected nodal y-coordinates
 *
 */
void ProjectEdgeNodesToSegment( const MeshData& mesh, int edgeId, real nrmlX, real nrmlY, real cX, real cY,
                                real* RESTRICT pX, real* RESTRICT pY );

/*!
 * \brief checks if the vertices on face2 have interpenetrated the level set
 *        defined by face 1
 *
 * \param [in] meshDat1 mesh data object for mesh 1 to which face 1 belongs
 * \param [in] meshDat2 mesh data object for mesh 2 to which face 2 belongs
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
bool FaceInterCheck( const MeshData& meshDat1, const MeshData& meshDat2, int fId1, int fId2, real tol, bool& allVerts );

/*!
 *
 * \brief checks if edge 2 interpenetrates the level set defined by edge 1
 *
 * \param [in] meshDat1 MeshData object for mesh 1
 * \param [in] meshDat2 MeshData object for mesh 2
 * \param [in] eId1 edge id for edge belonging to mesh 1
 * \param [in] eId2 edge id for edge belonging to mesh 2
 * \param [in] tol interpenetration tolerance
 * \param [in,out] allVerts true if all of edge2 has interpenetrated
 *                 edge 1
 *
 * \return true if edge 2 interpenetrates the level set defined by edge 1
 *
 */
bool EdgeInterCheck( const MeshData& meshDat1, const MeshData& meshDat2, int eId1, int eId2, real tol, bool& allVerts );

//-----------------------------------------------------------------------------
// Contact Plane base class
//-----------------------------------------------------------------------------

class ContactPlane {
 protected:
  int dim;               ///< Problem dimension
  int m_numFaces;        ///< Number of constituent faces
  InterfacePair m_pair;  ///< Face-pair struct for two constituent faces

 public:
  ContactPlane(){};
  virtual ~ContactPlane(){};

  bool m_intermediatePlane;  ///< True if intermediate plane is used
  bool m_inContact;          ///< True if face-pair is in contact
  bool m_interpenOverlap;    ///< True if using interpenetration overlap algorithm

  real m_cX;  ///< Contact plane point global x-coordinate
  real m_cY;  ///< Contact plane point global y-coordinate
  real m_cZ;  ///< Contact plane point global z-coordinate (zero out for 2D)

  real m_cXf1;  ///< Global x-coordinate of contact plane centroid projected to face 1
  real m_cYf1;  ///< Global y-coordinate of contact plane centroid projected to face 1
  real m_cZf1;  ///< Global z-coordinate of contact plane centroid projected to face 1

  real m_cXf2;  ///< global x-coordinate of contact plane centroid projected to face 2
  real m_cYf2;  ///< global y-coordinate of contact plane centroid projected to face 2
  real m_cZf2;  ///< global z-coordinate of contact plane centroid projected to face 2

  int m_numInterpenPoly1Vert;  ///< Number of vertices on face 1 interpenetrating polygon
  real* m_interpenG1X;         ///< Global x-coordinate of face 1 interpenetrating polygon
  real* m_interpenG1Y;         ///< Global y-coordinate of face 1 interpenetrating polygon
  real* m_interpenG1Z;         ///< Global z-coordinate of face 1 interpenetrating polygon

  int m_numInterpenPoly2Vert;  ///< Number of vertices on face 2 interpenetrating polygon
  real* m_interpenG2X;         ///< Global x-coordinate of face 2 interpenetrating polygon
  real* m_interpenG2Y;         ///< Global y-coordinate of face 2 interpenetrating polygon
  real* m_interpenG2Z;         ///< Global z-coordinate of face 2 interpenetrating polygon

  real m_nX;  ///< Global x-component of contact plane unit normal
  real m_nY;  ///< Global y-component of contact plane unit normal
  real m_nZ;  ///< Global z-component of contact plane unit normal (zero out for 2D)

  real m_gap;     ///< Face-pair gap
  real m_gapTol;  ///< Face-pair gap tolerance

  // cp area
  real m_areaFrac;      ///< Face area fraction used to determine overlap area cutoff
  real m_areaMin;       ///< Minimum overlap area for inclusion into the active set
  real m_area;          ///< Overlap area
  real m_interpenArea;  ///< Interpenetrating overlap area

 public:
  /// \name Contact plane routines
  /// @{

  /*!
   * \brief Compute the contact plane normal
   */
  virtual void computeNormal() = 0;

  /*!
   * \brief Compute the contact plane point
   */
  virtual void computePlanePoint() = 0;

  /*!
   * \brief Compute the contact plane integral gap expression
   */
  virtual void computeIntegralGap() = 0;

  /*!
   * \brief Compute the contact plane area tolerance
   */
  virtual void computeAreaTol() = 0;

  /*!
   * \brief Compute the contact plane centroid gap
   *
   * \param [in] scale Scale to help find centroid-to-face projections
   */
  virtual void centroidGap( real scale ) = 0;

  /*!
   * \brief Compute the contact plane centroid gap and relocate contact plane point
   *
   * \param [in] scale Scale to help find the centroid-to-face projections
   */
  virtual void planePointAndCentroidGap( real scale ) = 0;

  /*!
   * \brief Compute the contact plane integral gap expression
   *
   * \param [in,out] interpen true if the two faces interpenetrate
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   */
  virtual FaceGeomError computeLocalInterpenOverlap( bool& interpen ) = 0;

  /*!
   * \brief Copy the contact plane object
   *
   * \param [in] cPlane Pointer to existing contact plane object to be copied
   */
  virtual void copyContactPlane( ContactPlane* RESTRICT cPlane ) = 0;

  /// @}

  /// \name Getters and setters
  /// @{

  /*!
   * \brief Get mesh data associated with a particular mesh id
   *
   * \param [in] meshId Integer id for mesh
   * \return MeshData object
   */
  MeshData& getCpMeshData( int meshId )
  {
    MeshManager& meshManager = MeshManager::getInstance();
    MeshData& mesh = meshManager.GetMeshInstance( meshId );
    return mesh;
  }

  /*!
   * \brief Get the mesh id of one of the two faces used to form the contact plane
   *
   * \param [in] meshID Integer id for mesh
   * \return Mesh id
   */
  int getCpMeshId( int meshID ) const
  {
    if ( meshID != 1 && meshID != 2 ) {
      SLIC_ERROR( "getCpMeshId input mesh ID value must be 1 or 2." );
    }

    int id = -1;
    if ( meshID == 1 ) id = m_pair.meshId1;
    if ( meshID == 2 ) id = m_pair.meshId2;

    return id;
  }

  /*!
   * \brief Get the face id of one of the two faces used to form the contact plane
   *
   * \param [in] faceID Integer id for mesh
   * \return Face id
   */
  int getCpFaceId( int faceID ) const
  {
    if ( faceID != 1 && faceID != 2 ) {
      SLIC_ERROR( "getCpFaceId input face ID value must be 1 or 2." );
    }

    int id = -1;
    if ( faceID == 1 ) id = m_pair.pairIndex1;
    if ( faceID == 2 ) id = m_pair.pairIndex2;

    return id;
  }

  /*!
   * \brief Get the number of faces used to form the contact plane
   *
   * \return Number of faces
   *
   * \note Number of faces should typically be 2
   *
   */
  int getCpNumFaces() const { return m_numFaces; }

  /*!
   * \brief Get the dimension of the problem
   *
   * \return Space dimension
   *
   */
  int getDim() { return dim; };

  /*!
   * \brief Set the contact plane mesh id for a given face id
   *
   * \param [in] i Face id
   * \param [in] meshID Integer Id of the mesh
   */
  void setCpMeshId( int i, int meshID )
  {
    if ( i != 1 && i != 2 ) {
      SLIC_ERROR( "setCpMeshId input argument _i_ value must be 1 or 2." );
    }
    if ( i == 1 ) m_pair.meshId1 = meshID;
    if ( i == 2 ) m_pair.meshId2 = meshID;
  }

  /*!
   * \brief Set a contact plane face id
   *
   * \param [in] i Face number
   * \param [in] faceID Integer Id of the i^th face
   */
  void setCpFaceId( int i, int faceID )
  {
    if ( i != 1 && i != 2 ) {
      SLIC_ERROR( "setCpMeshId input argument _i_ value must be 1 or 2." );
    }
    if ( i == 1 ) m_pair.pairIndex1 = faceID;
    if ( i == 2 ) m_pair.pairIndex2 = faceID;
  }

  /*!
   * \brief Set the pair type for each face
   *
   * \param [in] i Face number
   * \param [in] pairType Face type
   */
  void setCpPairType( int id, integer pairType )
  {
    if ( id != 1 && id != 2 ) {
      SLIC_ERROR( "setCpPairType input argument _id_ value must be 1 or 2." );
    }
    if ( id == 1 ) m_pair.pairType1 = pairType;
    if ( id == 2 ) m_pair.pairType2 = pairType;
  }

  /*!
   * \brief Set the number of faces involved in forming the contact plane
   *
   * \param [in] num Number of faces
   */
  void setCpNumFaces( int num ) { m_numFaces = num; }

  /*!
   * \brief Set the dimension of the problem
   *
   * \param [in] dimension Dimension of the problem
   */
  void setDim( int dimension ) { dim = dimension; };

  /// @}
};

//-----------------------------------------------------------------------------
// Contact Plane 3D class
//-----------------------------------------------------------------------------

class ContactPlane3D : public ContactPlane {
 public:
  real m_e1X;  ///< Global x-component of first in-plane basis vector
  real m_e1Y;  ///< Global y-component of first in-plane basis vector
  real m_e1Z;  ///< Global z-component of first in-plane basis vector

  real m_e2X;  ///< Global x-component of second in-plane basis vector
  real m_e2Y;  ///< Global y-component of second in-plane basis vector
  real m_e2Z;  ///< Global z-component of second in-plane basis vector

  real* m_polyLocX;  ///< Pointer to local x-components of overlap polygon's vertices
  real* m_polyLocY;  ///< Pointer to local y-components of overlap polygon's vertices

  real* m_polyX;  ///< Pointer to global x-components of overlap polygon's vertices
  real* m_polyY;  ///< Pointer to global y-components of overlap polygon's vertices
  real* m_polyZ;  ///< Pointer to global z-components of overlap polygon's vertices

  int m_numPolyVert;  ///< Number of vertices in overlapping polygon

  real m_overlapCX;  ///< Local x-coordinate of overlap centroid
  real m_overlapCY;  ///< Local y-coordinate of overlap centroid

  real* m_interpenPoly1X;  ///< Local x-coordinates of face 1 interpenetrating overlap
  real* m_interpenPoly1Y;  ///< Local y-coordinates of face 1 interpenetrating overlap

  real* m_interpenPoly2X;  ///< Local x-coordinates of face 2 interpenetrating overlap
  real* m_interpenPoly2Y;  ///< Local y-coordinates of face 2 interpenetrating overlap

 public:
  /*!
   * Constructor
   *
   * \param [in] pair InterfacePair struct
   * \param [in] areaFrac Area fraction for inclusion of contact plane
   * \param [in] interpenOverlap True if using interpenetration overlap algorithm
   * \param [in] interPlane True if intermediate (i.e. common) plane is used
   * \param [in] dimension Dimension of the problem
   */
  ContactPlane3D( InterfacePair& pair, real areaFrac, bool interpenOverlap, bool interPlane, int dimension );

  /*!
   * Overload constructor with no argument list
   *
   */
  ContactPlane3D();

  /*!
   * Destructor
   *
   */
  ~ContactPlane3D()
  {
    if ( m_polyX != nullptr ) delete[] m_polyX;
    if ( m_polyY != nullptr ) delete[] m_polyY;
    if ( m_polyZ != nullptr ) delete[] m_polyZ;
    if ( m_polyLocX != nullptr ) delete[] m_polyLocX;
    if ( m_polyLocY != nullptr ) delete[] m_polyLocY;
    if ( m_interpenPoly1X != nullptr ) delete[] m_interpenPoly1X;
    if ( m_interpenPoly1Y != nullptr ) delete[] m_interpenPoly1Y;
    if ( m_interpenPoly2X != nullptr ) delete[] m_interpenPoly2X;
    if ( m_interpenPoly2Y != nullptr ) delete[] m_interpenPoly2Y;
    if ( m_interpenG1X != nullptr ) delete[] m_interpenG1X;
    if ( m_interpenG1Y != nullptr ) delete[] m_interpenG1Y;
    if ( m_interpenG1Z != nullptr ) delete[] m_interpenG1Z;
    if ( m_interpenG2X != nullptr ) delete[] m_interpenG2X;
    if ( m_interpenG2Y != nullptr ) delete[] m_interpenG2Y;
    if ( m_interpenG2Z != nullptr ) delete[] m_interpenG2Z;
  }

  /*!
   * \brief Compute the unit normal that defines the contact plane
   */
  virtual void computeNormal();

  /*!
   * \brief Computes a reference point on the plane locating it in 3-space
   *
   * \note This is taken as the average of the vertex averaged centroids of
   *  the two faces that are used to define a local contact plane
   */
  virtual void computePlanePoint();

  /*!
   * \brief Compute a local basis on the contact plane
   */
  void computeLocalBasis();

  /*!
   * \brief Compute the weak (integral form) gap between the two faces.
   */
  virtual void computeIntegralGap();

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
  void globalTo2DLocalCoords( real* RESTRICT pX, real* RESTRICT pY, real* RESTRICT pZ, real* RESTRICT pLX,
                              real* RESTRICT pLY, int size );

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
  void globalTo2DLocalCoords( real pX, real pY, real pZ, real& pLX, real& pLY, int size );

  /*!
   * \brief Computes the area tolerance for accepting a face pair
   */
  virtual void computeAreaTol();

  /*!
   * \brief Check whether two polygons (faces) have a positive area of overlap
   *
   * \note Wrapper routine that calls the ALE3D polygon intersection routine
   */
  void checkPolyOverlap( real* RESTRICT projLocX1, real* RESTRICT projLocY1, real* RESTRICT projLocX2,
                         real* RESTRICT projLocY2, const int isym );

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
  void local2DToGlobalCoords( real xloc, real yloc, real& xg, real& yg, real& zg );

  /*!
   * \brief Recomputes the reference point that locates the plane in 3-space
   *        and the gap between the projected intersection poly centroids
   *
   * \note This projects the projected area of overlap's centroid (from the
   *  polygon intersection routine) back to each face that are used to form
   *  the contact plane and then averages these projected points.
   */
  virtual void planePointAndCentroidGap( real scale );

  /*!
   * \brief Computes the gap between the two projections of the contact
   *        plane centroid onto each constituent face.
   *
   * \param [in] scale
   */
  virtual void centroidGap( real scale );

  /*!
   * \brief Computes the polygonal overlap between the portion of two
   *        faces lying on the contact plane that are interpenetrating the contact
   *        plane.
   *
   * \param [in,out] interpen true if the two faces interpenetrate
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   */
  virtual FaceGeomError computeLocalInterpenOverlap( bool& interpen );

  /*!
   * \brief Copies one contact plane object's data to another
   *
   * \param [in] cPlane input contact plane object to be copied
   */
  virtual void copyContactPlane( ContactPlane* RESTRICT cPlane );
};

//-----------------------------------------------------------------------------
// Contact Plane 2D class
//-----------------------------------------------------------------------------

class ContactPlane2D : public ContactPlane {
 public:
  real m_segX[2];  ///< Global x-components of overlap segment vertices
  real m_segY[2];  ///< Global y-components of overlap segment vertices

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
  ContactPlane2D( InterfacePair& pair, real lenFrac, bool interpenOverlap, bool interPlane, int dimension );

  /*!
   * \brief Overloaded constructor with no arguments
   *
   */
  ContactPlane2D();

  /*!
   * \brief Destructor
   *
   */
  ~ContactPlane2D()
  {
    if ( m_interpenG1X != nullptr ) delete[] m_interpenG1X;
    if ( m_interpenG1Y != nullptr ) delete[] m_interpenG1Y;
    if ( m_interpenG1Z != nullptr ) delete[] m_interpenG1Z;
    if ( m_interpenG2X != nullptr ) delete[] m_interpenG2X;
    if ( m_interpenG2Y != nullptr ) delete[] m_interpenG2Y;
    if ( m_interpenG2Z != nullptr ) delete[] m_interpenG2Z;
  };

  /*!
   * \brief Compute the unit normal that defines the contact plane
   */
  virtual void computeNormal();

  /*!
   * \brief Computes a reference point on the plane locating it in 3-space
   *
   * \note This is taken as the average of the vertex averaged centroids of
   *  the two faces that are used to define a local contact plane
   */
  virtual void computePlanePoint();

  /*!
   * \brief Compute the weak (integral form) gap between the two faces.
   */
  virtual void computeIntegralGap();

  /*!
   * \brief Computes the area tolerance for accepting a face pair
   */
  virtual void computeAreaTol();

  /*!
   * \brief Computes the gap between the two projections of the contact
   *        plane centroid onto each constituent face.
   *
   * \param [in] scale
   */
  virtual void centroidGap( real scale );

  /*!
   * \brief Recomputes the reference point that locates the plane in 3-space
   *        and the gap between the projected intersection poly centroids
   *
   * \note This projects the projected area of overlap's centroid (from the
   *  polygon intersection routine) back to each face that are used to form
   *  the contact plane and then averages these projected points.
   */
  virtual void planePointAndCentroidGap( real scale );

  /*!
   * \brief Computes the polygonal overlap between the portion of two
   *        faces lying on the contact plane that are interpenetrating the contact
   *        plane.
   *
   * \param [in,out] interpen true if the two faces interpenetrate
   *
   * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
   */
  virtual FaceGeomError computeLocalInterpenOverlap( bool& interpen );

  /*!
   * \brief Copies one contact plane object's data to another
   *
   * \param [in] cPlane input contact plane object to be copied
   */
  virtual void copyContactPlane( ContactPlane* RESTRICT cPlane );

  /*!
   * \brief Check whether two segments have a positive length of overlap
   *
   */
  void checkSegOverlap( const real* const RESTRICT pX1, const real* const RESTRICT pY1, const real* const RESTRICT pX2,
                        const real* const RESTRICT pY2, const int nV1, const int nV2 );
};

//-----------------------------------------------------------------------------
// Free functions returning Contact Plane objects
//-----------------------------------------------------------------------------

/*!
 * \brief Checks if face-pair (3D) candidate is actual local contact interaction.
 *
 * \param [in] pair interface pair containing pair related indices
 * \param [in] fullOverlap True if full overlap calculation is used, false if interpenetration calculation is used
 * \param [in,out] cp contact plane object to be populated
 *
 * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
 *
 */
FaceGeomError CheckFacePair( InterfacePair& pair, bool fullOverlap, ContactPlane3D& cp );

/*!
 * \brief Checks if face-pair (3D) candidate is aligned and actual local contact interaction.
 *
 * \param [in] pair interface pair containing pair related indices
 *
 * \return 3D contact plane object with boolean indicating if face-pair form a local contact interaction
 *
 */
ContactPlane3D CheckAlignedFacePair( InterfacePair& pair );

/*!
 * \brief Checks if 2D edge-pair candidate is actual local contact interaction.
 *
 * \param [in] pair interface pair containing pair related indices
 * \param [in] fullOverlap True if full overlap calculation is used, false if interpenetration calculation is used
 * \param [in,out] cp contact plane object to be populated
 *
 * \return 0 if no error, non-zero (via FaceGeomError enum) otherwise
 *
 */
FaceGeomError CheckEdgePair( InterfacePair& pair, bool fullOverlap, ContactPlane2D& cp );

}  // namespace tribol

#endif /* SRC_GEOM_CONTACTPLANE_HPP_ */
