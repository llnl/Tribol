// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/geom/ContactPlaneManager.hpp"
#include "tribol/geom/ContactPlane.hpp"
#include "tribol/mesh/MeshManager.hpp"
#include "tribol/utils/Math.hpp"
#include "axom/slic.hpp"

// C++ includes
#include <iostream>

namespace tribol {

ContactPlaneManager::ContactPlaneManager()
{
  m_spaceDim = -1;
  m_numContactPlanes = 0;
  m_allocatedNumContactPlanes = 0;
  m_resized = false;
}

//------------------------------------------------------------------------------
ContactPlaneManager& ContactPlaneManager::getInstance()
{
  static ContactPlaneManager instance;
  return instance;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::resize( IndexType const newSize )
{
  m_numContactPlanes = newSize;

  if ( m_numContactPlanes > m_allocatedNumContactPlanes ) {
    reserve( static_cast<IndexType>( m_numContactPlanes * 1.1 ) );
  }

  switch ( m_spaceDim ) {
    case 2:
      resize2D();
      break;
    case 3:
      resize3D();
      break;
    default:
      SLIC_ERROR( "ContactPlaneManager::resize(): incorrect problem dimension, " << m_spaceDim );
      break;
  }

  return;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::resize3D()
{
  m_numFaces.resize( m_numContactPlanes );

  m_meshId1.resize( m_numContactPlanes );
  m_meshId2.resize( m_numContactPlanes );

  m_fId1.resize( m_numContactPlanes );
  m_fId2.resize( m_numContactPlanes );

  m_pressure.resize( m_numContactPlanes );
  m_ratePressure.resize( m_numContactPlanes );

  m_inContact.resize( m_numContactPlanes );
  m_interpenOverlap.resize( m_numContactPlanes );

  m_cX.resize( m_numContactPlanes );
  m_cY.resize( m_numContactPlanes );
  m_cZ.resize( m_numContactPlanes );

  m_e1X.resize( m_numContactPlanes );
  m_e1Y.resize( m_numContactPlanes );
  m_e1Z.resize( m_numContactPlanes );

  m_e2X.resize( m_numContactPlanes );
  m_e2Y.resize( m_numContactPlanes );
  m_e2Z.resize( m_numContactPlanes );

  m_nX.resize( m_numContactPlanes );
  m_nY.resize( m_numContactPlanes );
  m_nZ.resize( m_numContactPlanes );

  m_polyLocX.resize( m_numContactPlanes );
  m_polyLocY.resize( m_numContactPlanes );

  m_numPolyVert.resize( m_numContactPlanes );

  m_polyX.resize( m_numContactPlanes );
  m_polyY.resize( m_numContactPlanes );
  m_polyZ.resize( m_numContactPlanes );

  m_overlapCX.resize( m_numContactPlanes );
  m_overlapCY.resize( m_numContactPlanes );

  m_gap.resize( m_numContactPlanes );
  m_velGap.resize( m_numContactPlanes );
  m_gapTol.resize( m_numContactPlanes );

  m_numInterpenPoly1Vert.resize( m_numContactPlanes );

  m_interpenPoly1X.resize( m_numContactPlanes );
  m_interpenPoly1Y.resize( m_numContactPlanes );

  m_numInterpenPoly2Vert.resize( m_numContactPlanes );

  m_interpenPoly2X.resize( m_numContactPlanes );
  m_interpenPoly2Y.resize( m_numContactPlanes );

  m_interpenG1X.resize( m_numContactPlanes );
  m_interpenG1Y.resize( m_numContactPlanes );
  m_interpenG1Z.resize( m_numContactPlanes );

  m_interpenG2X.resize( m_numContactPlanes );
  m_interpenG2Y.resize( m_numContactPlanes );
  m_interpenG2Z.resize( m_numContactPlanes );

  m_areaFrac.resize( m_numContactPlanes );
  m_areaMin.resize( m_numContactPlanes );
  m_area.resize( m_numContactPlanes );
  m_interpenArea.resize( m_numContactPlanes );

  m_cXf1.resize( m_numContactPlanes );
  m_cYf1.resize( m_numContactPlanes );
  m_cZf1.resize( m_numContactPlanes );

  m_cXf2.resize( m_numContactPlanes );
  m_cYf2.resize( m_numContactPlanes );
  m_cZf2.resize( m_numContactPlanes );

  m_resized = true;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::resize2D()
{
  m_numFaces.resize( m_numContactPlanes );

  m_meshId1.resize( m_numContactPlanes );
  m_meshId2.resize( m_numContactPlanes );

  m_fId1.resize( m_numContactPlanes );
  m_fId2.resize( m_numContactPlanes );

  m_pressure.resize( m_numContactPlanes );
  m_ratePressure.resize( m_numContactPlanes );

  m_inContact.resize( m_numContactPlanes );
  m_interpenOverlap.resize( m_numContactPlanes );

  m_cX.resize( m_numContactPlanes );
  m_cY.resize( m_numContactPlanes );

  m_nX.resize( m_numContactPlanes );
  m_nY.resize( m_numContactPlanes );

  m_gap.resize( m_numContactPlanes );
  m_velGap.resize( m_numContactPlanes );
  m_gapTol.resize( m_numContactPlanes );

  m_numPolyVert.resize( m_numContactPlanes );

  m_numInterpenPoly1Vert.resize( m_numContactPlanes );
  m_numInterpenPoly2Vert.resize( m_numContactPlanes );

  m_interpenG1X.resize( m_numContactPlanes );
  m_interpenG1Y.resize( m_numContactPlanes );

  m_interpenG2X.resize( m_numContactPlanes );
  m_interpenG2Y.resize( m_numContactPlanes );

  m_areaFrac.resize( m_numContactPlanes );
  m_areaMin.resize( m_numContactPlanes );
  m_area.resize( m_numContactPlanes );
  m_interpenArea.resize( m_numContactPlanes );

  m_cXf1.resize( m_numContactPlanes );
  m_cYf1.resize( m_numContactPlanes );

  m_cXf2.resize( m_numContactPlanes );
  m_cYf2.resize( m_numContactPlanes );

  // add resize call for overlap segment vertices
  m_segX.resize( m_numContactPlanes );
  m_segY.resize( m_numContactPlanes );

  m_resized = true;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::reserve( IndexType const newSize )
{
  if ( m_spaceDim == 3 ) {
    reserve3D( newSize );
  } else if ( m_spaceDim == 2 ) {
    reserve2D( newSize );
  } else {
    SLIC_ERROR( "ContactPlaneManager::reserve(): must set problem dimension." );
  }

  return;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::reserve3D( IndexType const newSize )
{
  m_allocatedNumContactPlanes = newSize;

  m_numFaces.reserve( m_allocatedNumContactPlanes );

  m_meshId1.reserve( m_allocatedNumContactPlanes );
  m_meshId2.reserve( m_allocatedNumContactPlanes );

  m_fId1.reserve( m_allocatedNumContactPlanes );
  m_fId2.reserve( m_allocatedNumContactPlanes );

  m_pressure.reserve( m_allocatedNumContactPlanes );
  m_ratePressure.reserve( m_allocatedNumContactPlanes );

  m_inContact.reserve( m_allocatedNumContactPlanes );
  m_interpenOverlap.reserve( m_allocatedNumContactPlanes );

  m_cX.reserve( m_allocatedNumContactPlanes );
  m_cY.reserve( m_allocatedNumContactPlanes );
  m_cZ.reserve( m_allocatedNumContactPlanes );

  m_e1X.reserve( m_allocatedNumContactPlanes );
  m_e1Y.reserve( m_allocatedNumContactPlanes );
  m_e1Z.reserve( m_allocatedNumContactPlanes );

  m_e2X.reserve( m_allocatedNumContactPlanes );
  m_e2Y.reserve( m_allocatedNumContactPlanes );
  m_e2Z.reserve( m_allocatedNumContactPlanes );

  m_nX.reserve( m_allocatedNumContactPlanes );
  m_nY.reserve( m_allocatedNumContactPlanes );
  m_nZ.reserve( m_allocatedNumContactPlanes );

  m_numPolyVert.reserve( m_allocatedNumContactPlanes );

  m_polyLocX.reserve( m_allocatedNumContactPlanes );
  m_polyLocY.reserve( m_allocatedNumContactPlanes );

  m_polyX.reserve( m_allocatedNumContactPlanes );
  m_polyY.reserve( m_allocatedNumContactPlanes );
  m_polyZ.reserve( m_allocatedNumContactPlanes );

  m_overlapCX.reserve( m_allocatedNumContactPlanes );
  m_overlapCY.reserve( m_allocatedNumContactPlanes );

  m_gap.reserve( m_allocatedNumContactPlanes );
  m_velGap.reserve( m_allocatedNumContactPlanes );
  m_gapTol.reserve( m_allocatedNumContactPlanes );

  m_numInterpenPoly1Vert.reserve( m_allocatedNumContactPlanes );

  m_interpenPoly1X.reserve( m_allocatedNumContactPlanes );
  m_interpenPoly1Y.reserve( m_allocatedNumContactPlanes );

  m_numInterpenPoly2Vert.reserve( m_allocatedNumContactPlanes );

  m_interpenPoly2X.reserve( m_allocatedNumContactPlanes );
  m_interpenPoly2Y.reserve( m_allocatedNumContactPlanes );

  m_interpenG1X.reserve( m_allocatedNumContactPlanes );
  m_interpenG1Y.reserve( m_allocatedNumContactPlanes );
  m_interpenG1Z.reserve( m_allocatedNumContactPlanes );

  m_interpenG2X.reserve( m_allocatedNumContactPlanes );
  m_interpenG2Y.reserve( m_allocatedNumContactPlanes );
  m_interpenG2Z.reserve( m_allocatedNumContactPlanes );

  m_areaFrac.reserve( m_allocatedNumContactPlanes );
  m_areaMin.reserve( m_allocatedNumContactPlanes );
  m_area.reserve( m_allocatedNumContactPlanes );
  m_interpenArea.reserve( m_allocatedNumContactPlanes );

  m_cXf1.reserve( m_allocatedNumContactPlanes );
  m_cYf1.reserve( m_allocatedNumContactPlanes );
  m_cZf1.reserve( m_allocatedNumContactPlanes );

  m_cXf2.reserve( m_allocatedNumContactPlanes );
  m_cYf2.reserve( m_allocatedNumContactPlanes );
  m_cZf2.reserve( m_allocatedNumContactPlanes );
}

//------------------------------------------------------------------------------
void ContactPlaneManager::reserve2D( IndexType const newSize )
{
  m_allocatedNumContactPlanes = newSize;

  m_numFaces.reserve( m_allocatedNumContactPlanes );

  m_meshId1.reserve( m_allocatedNumContactPlanes );
  m_meshId2.reserve( m_allocatedNumContactPlanes );

  m_fId1.reserve( m_allocatedNumContactPlanes );
  m_fId2.reserve( m_allocatedNumContactPlanes );

  m_pressure.reserve( m_allocatedNumContactPlanes );
  m_ratePressure.reserve( m_allocatedNumContactPlanes );

  m_inContact.reserve( m_allocatedNumContactPlanes );
  m_interpenOverlap.reserve( m_allocatedNumContactPlanes );

  m_cX.reserve( m_allocatedNumContactPlanes );
  m_cY.reserve( m_allocatedNumContactPlanes );

  m_nX.reserve( m_allocatedNumContactPlanes );
  m_nY.reserve( m_allocatedNumContactPlanes );

  m_gap.reserve( m_allocatedNumContactPlanes );
  m_velGap.reserve( m_allocatedNumContactPlanes );
  m_gapTol.reserve( m_allocatedNumContactPlanes );

  m_numPolyVert.reserve( m_allocatedNumContactPlanes );

  m_numInterpenPoly1Vert.reserve( m_allocatedNumContactPlanes );
  m_numInterpenPoly2Vert.reserve( m_allocatedNumContactPlanes );

  m_interpenG1X.reserve( m_allocatedNumContactPlanes );
  m_interpenG1Y.reserve( m_allocatedNumContactPlanes );

  m_interpenG2X.reserve( m_allocatedNumContactPlanes );
  m_interpenG2Y.reserve( m_allocatedNumContactPlanes );

  m_areaFrac.reserve( m_allocatedNumContactPlanes );
  m_areaMin.reserve( m_allocatedNumContactPlanes );
  m_area.reserve( m_allocatedNumContactPlanes );
  m_interpenArea.reserve( m_allocatedNumContactPlanes );

  m_cXf1.reserve( m_allocatedNumContactPlanes );
  m_cYf1.reserve( m_allocatedNumContactPlanes );

  m_cXf2.reserve( m_allocatedNumContactPlanes );
  m_cYf2.reserve( m_allocatedNumContactPlanes );

  // add reserve call for the 2D contact segment overlap vertex
  // pointers
  m_segX.reserve( m_allocatedNumContactPlanes );
  m_segY.reserve( m_allocatedNumContactPlanes );
}

//------------------------------------------------------------------------------
void ContactPlaneManager::setContactPlaneData( const ContactPlane3D& cp )
{
  if ( !m_resized ) {
    SLIC_ERROR( "Resize ContactPlaneManager prior to adding new ContactPlane" );
    return;
  }

  int id = m_numContactPlanes - 1;

  m_numFaces[id] = cp.getCpNumFaces();

  m_meshId1[id] = cp.getCpMeshId( 1 );
  m_meshId2[id] = cp.getCpMeshId( 2 );

  m_fId1[id] = cp.getCpFaceId( 1 );
  m_fId2[id] = cp.getCpFaceId( 2 );

  m_inContact[id] = cp.m_inContact;
  m_interpenOverlap[id] = cp.m_interpenOverlap;

  m_cX[id] = cp.m_cX;
  m_cY[id] = cp.m_cY;
  m_cZ[id] = cp.m_cZ;

  m_e1X[id] = cp.m_e1X;
  m_e1Y[id] = cp.m_e1Y;
  m_e1Z[id] = cp.m_e1Z;

  m_e2X[id] = cp.m_e2X;
  m_e2Y[id] = cp.m_e2Y;
  m_e2Z[id] = cp.m_e2Z;

  m_nX[id] = cp.m_nX;
  m_nY[id] = cp.m_nY;
  m_nZ[id] = cp.m_nZ;

  // allocate new memory for m_polyLocX, m_polyLocY, m_polyX, and
  // m_polyY, m_polyZ as the contact plane object memory will be deleted
  // when the object goes out of scope. Note, must resize the container
  // arrays prior to this

  m_numPolyVert[id] = cp.m_numPolyVert;

  // TODO revisit whether this needs to be stored. It is not used
  // NOR populated in ALIGNED_MORTAR
  if ( cp.m_polyLocX != nullptr ) {
    allocRealArray( &m_polyLocX[id], cp.m_numPolyVert, cp.m_polyLocX );
    allocRealArray( &m_polyLocY[id], cp.m_numPolyVert, cp.m_polyLocY );
  }

  allocRealArray( &m_polyX[id], cp.m_numPolyVert, cp.m_polyX );
  allocRealArray( &m_polyY[id], cp.m_numPolyVert, cp.m_polyY );
  allocRealArray( &m_polyZ[id], cp.m_numPolyVert, cp.m_polyZ );

  m_overlapCX[id] = cp.m_overlapCX;
  m_overlapCY[id] = cp.m_overlapCY;
  m_gap[id] = cp.m_gap;
  m_gapTol[id] = cp.m_gapTol;

  m_numInterpenPoly1Vert[id] = cp.m_numInterpenPoly1Vert;
  m_numInterpenPoly2Vert[id] = cp.m_numInterpenPoly2Vert;

  // allocate new memory for m_interpenPoly1X, m_interpenPoly1Y,
  // m_interpenPoly2X, m_interpenPoly2Y and all m_interpenG1(2)X(Y,Z).
  // Note, must resize the container arrays prior to this.

  if ( m_interpenOverlap[id] ) {
    allocRealArray( &m_interpenPoly1X[id], cp.m_numInterpenPoly1Vert, cp.m_interpenPoly1X );
    allocRealArray( &m_interpenPoly1Y[id], cp.m_numInterpenPoly1Vert, cp.m_interpenPoly1Y );
    allocRealArray( &m_interpenPoly2X[id], cp.m_numInterpenPoly2Vert, cp.m_interpenPoly2X );
    allocRealArray( &m_interpenPoly2Y[id], cp.m_numInterpenPoly2Vert, cp.m_interpenPoly2Y );

    allocRealArray( &m_interpenG1X[id], cp.m_numInterpenPoly1Vert, cp.m_interpenG1X );
    allocRealArray( &m_interpenG1Y[id], cp.m_numInterpenPoly1Vert, cp.m_interpenG1Y );
    allocRealArray( &m_interpenG1Z[id], cp.m_numInterpenPoly1Vert, cp.m_interpenG1Z );

    allocRealArray( &m_interpenG2X[id], cp.m_numInterpenPoly2Vert, cp.m_interpenG2X );
    allocRealArray( &m_interpenG2Y[id], cp.m_numInterpenPoly2Vert, cp.m_interpenG2Y );
    allocRealArray( &m_interpenG2Z[id], cp.m_numInterpenPoly2Vert, cp.m_interpenG2Z );
  } else {
    m_interpenPoly1X[id] = nullptr;
    m_interpenPoly1Y[id] = nullptr;
    m_interpenPoly2X[id] = nullptr;
    m_interpenPoly2Y[id] = nullptr;

    m_interpenG1X[id] = nullptr;
    m_interpenG1Y[id] = nullptr;
    m_interpenG1Z[id] = nullptr;

    m_interpenG2X[id] = nullptr;
    m_interpenG2Y[id] = nullptr;
    m_interpenG2Z[id] = nullptr;
  }

  m_areaFrac[id] = cp.m_areaFrac;
  m_areaMin[id] = cp.m_areaMin;
  m_area[id] = cp.m_area;
  m_interpenArea[id] = cp.m_interpenArea;

  m_cXf1[id] = cp.m_cXf1;
  m_cYf1[id] = cp.m_cYf1;
  m_cZf1[id] = cp.m_cZf1;

  m_cXf2[id] = cp.m_cXf2;
  m_cYf2[id] = cp.m_cYf2;
  m_cZf2[id] = cp.m_cZf2;

  // reset "resize" to false
  m_resized = false;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::addContactPlane( const ContactPlane3D& cp )
{
  // add dimension check
  if ( m_spaceDim == 2 ) {
    SLIC_ASSERT( "Cannot add 3D contact plane for 2D problem." );
    return;
  }

  // resize the contact plane manager container arrays
  resize( m_numContactPlanes + 1 );

  // copy over contact plane data
  setContactPlaneData( cp );

  return;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::setContactPlaneData( const ContactPlane2D& cp )
{
  if ( !m_resized ) {
    SLIC_ERROR( "Resize ContactPlaneManager prior to adding new ContactPlane" );
    return;
  }

  int id = m_numContactPlanes - 1;

  m_numFaces[id] = cp.getCpNumFaces();

  m_meshId1[id] = cp.getCpMeshId( 1 );
  m_meshId2[id] = cp.getCpMeshId( 2 );

  m_fId1[id] = cp.getCpFaceId( 1 );
  m_fId2[id] = cp.getCpFaceId( 2 );

  m_inContact[id] = cp.m_inContact;
  m_interpenOverlap[id] = cp.m_interpenOverlap;

  m_cX[id] = cp.m_cX;
  m_cY[id] = cp.m_cY;

  m_nX[id] = cp.m_nX;
  m_nY[id] = cp.m_nY;

  m_gap[id] = cp.m_gap;
  m_gapTol[id] = cp.m_gapTol;

  m_numPolyVert[id] = 2;  // always 2 for 2D

  m_numInterpenPoly1Vert[id] = cp.m_numInterpenPoly1Vert;
  m_numInterpenPoly2Vert[id] = cp.m_numInterpenPoly2Vert;

  // allocate space for the container array pointers
  if ( m_interpenOverlap[id] ) {
    allocRealArray( &m_interpenG1X[id], m_numInterpenPoly1Vert[id], cp.m_interpenG1X );
    allocRealArray( &m_interpenG1Y[id], m_numInterpenPoly1Vert[id], cp.m_interpenG1Y );
    allocRealArray( &m_interpenG2X[id], m_numInterpenPoly2Vert[id], cp.m_interpenG2X );
    allocRealArray( &m_interpenG2Y[id], m_numInterpenPoly2Vert[id], cp.m_interpenG2Y );
  } else {
    m_interpenG1X[id] = nullptr;
    m_interpenG1Y[id] = nullptr;
    m_interpenG2X[id] = nullptr;
    m_interpenG2Y[id] = nullptr;
  }

  m_areaFrac[id] = cp.m_areaFrac;
  m_areaMin[id] = cp.m_areaMin;
  m_area[id] = cp.m_area;
  m_interpenArea[id] = cp.m_interpenArea;

  m_cXf1[id] = cp.m_cXf1;
  m_cYf1[id] = cp.m_cYf1;

  m_cXf2[id] = cp.m_cXf2;
  m_cYf2[id] = cp.m_cYf2;

  // allocate and set the 2D overlap segment vertex data
  allocRealArray( &m_segX[id], 2, cp.m_segX );
  allocRealArray( &m_segY[id], 2, cp.m_segY );

  // reset "resize" to false
  m_resized = false;

  return;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::addContactPlane( const ContactPlane2D& cp )
{
  // add dimension check
  if ( m_spaceDim == 3 ) {
    SLIC_ERROR( "Cannot add 2D contact plane for 3D problem." );
    return;
  }

  // resize the contact plane manager container arrays
  resize( m_numContactPlanes + 1 );

  // copy over contact plane data
  setContactPlaneData( cp );

  return;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::clearCPManager()
{
  // delete allocated storage
  {
    const int sz = m_polyLocX.size();
    for ( int i = 0; i < sz; ++i ) {
      if ( m_polyLocX[i] != nullptr ) delete[] m_polyLocX[i];
      if ( m_polyLocY[i] != nullptr ) delete[] m_polyLocY[i];
    }
  }

  {
    const int sz = m_polyX.size();
    for ( int i = 0; i < sz; ++i ) {
      if ( m_polyX[i] != nullptr ) delete[] m_polyX[i];
      if ( m_polyY[i] != nullptr ) delete[] m_polyY[i];
      if ( m_polyZ[i] != nullptr ) delete[] m_polyZ[i];
    }
  }

  {
    const int sz = m_interpenPoly1X.size();
    for ( int i = 0; i < sz; ++i ) {
      if ( m_interpenPoly1X[i] != nullptr ) delete[] m_interpenPoly1X[i];
      if ( m_interpenPoly1Y[i] != nullptr ) delete[] m_interpenPoly1Y[i];
    }
  }

  {
    const int sz = m_interpenPoly2X.size();
    for ( int i = 0; i < sz; ++i ) {
      if ( m_interpenPoly2X[i] != nullptr ) delete[] m_interpenPoly2X[i];
      if ( m_interpenPoly2Y[i] != nullptr ) delete[] m_interpenPoly2Y[i];
    }
  }

  {
    const int sz = m_interpenG1X.size();
    for ( int i = 0; i < sz; ++i ) {
      if ( m_interpenG1X[i] != nullptr ) delete[] m_interpenG1X[i];
      if ( m_interpenG1Y[i] != nullptr ) delete[] m_interpenG1Y[i];
    }
  }

  // delete the z-component separately as it is not used in 2D
  {
    const int sz = m_interpenG1Z.size();
    for ( int i = 0; i < sz; ++i ) {
      if ( m_interpenG1Z[i] != nullptr ) delete[] m_interpenG1Z[i];
    }
  }

  {
    const int sz = m_interpenG2X.size();
    for ( int i = 0; i < sz; ++i ) {
      if ( m_interpenG2X[i] != nullptr ) delete[] m_interpenG2X[i];
      if ( m_interpenG2Y[i] != nullptr ) delete[] m_interpenG2Y[i];
    }
  }

  // delete the z-component separately as it is not used in 2D
  {
    const int sz = m_interpenG2Z.size();
    for ( int i = 0; i < sz; ++i ) {
      if ( m_interpenG2Z[i] != nullptr ) delete[] m_interpenG2Z[i];
    }
  }

  {
    const int sz = m_segX.size();
    for ( int i = 0; i < sz; ++i ) {
      if ( m_segX[i] != nullptr ) delete[] m_segX[i];
      if ( m_segY[i] != nullptr ) delete[] m_segY[i];
    }
  }

  // clear each containerArray
  m_numContactPlanes = 0;
  m_allocatedNumContactPlanes = 0;
  m_resized = false;

  m_numFaces.clear();
  m_meshId1.clear();
  m_meshId2.clear();
  m_fId1.clear();
  m_fId2.clear();
  m_pressure.clear();
  m_ratePressure.clear();
  m_inContact.clear();
  m_interpenOverlap.clear();
  m_cX.clear();
  m_cY.clear();
  m_cZ.clear();
  m_e1X.clear();
  m_e1Y.clear();
  m_e1Z.clear();
  m_e2X.clear();
  m_e2Y.clear();
  m_e2Z.clear();
  m_nX.clear();
  m_nY.clear();
  m_nZ.clear();
  m_numPolyVert.clear();
  m_polyLocX.clear();
  m_polyLocY.clear();
  m_polyX.clear();
  m_polyY.clear();
  m_polyZ.clear();
  m_overlapCX.clear();
  m_overlapCY.clear();
  m_gap.clear();
  m_velGap.clear();
  m_gapTol.clear();
  m_numInterpenPoly1Vert.clear();
  m_interpenPoly1X.clear();
  m_interpenPoly1Y.clear();
  m_numInterpenPoly2Vert.clear();
  m_interpenPoly2X.clear();
  m_interpenPoly2Y.clear();
  m_interpenG1X.clear();
  m_interpenG1Y.clear();
  m_interpenG1Z.clear();
  m_interpenG2X.clear();
  m_interpenG2Y.clear();
  m_interpenG2Z.clear();
  m_areaFrac.clear();
  m_areaMin.clear();
  m_area.clear();
  m_interpenArea.clear();
  m_cXf1.clear();
  m_cYf1.clear();
  m_cZf1.clear();
  m_cXf2.clear();
  m_cYf2.clear();
  m_cZf2.clear();
  m_segX.clear();
  m_segY.clear();

  return;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::getContactPlaneOverlapVerts( int const id, int const numVerts, real* coords )
{
  if ( numVerts != m_numPolyVert[id] ) {
    SLIC_ERROR( "ContactPlaneManager::getContactPlaneOverlapVerts(): "
                << "input arg. numVerts != m_numPolyVert[id]" );
  }

  // point to the appropriate overlap vertex coordinate
  // data structure
  real const* vertCoordsX = nullptr;
  real const* vertCoordsY = nullptr;
  real const* vertCoordsZ = nullptr;
  if ( m_spaceDim == 3 ) {
    vertCoordsX = m_polyX[id];
    vertCoordsY = m_polyY[id];
    vertCoordsZ = m_polyZ[id];
  } else {
    vertCoordsX = m_segX[id];
    vertCoordsY = m_segY[id];
  }

  for ( IndexType ivert = 0; ivert < numVerts; ++ivert ) {
    coords[m_spaceDim * ivert] = vertCoordsX[ivert];
    coords[m_spaceDim * ivert + 1] = vertCoordsY[ivert];

    if ( m_spaceDim == 3 ) {
      coords[m_spaceDim * ivert + 2] = vertCoordsZ[ivert];
    }
  }

  return;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::getContactPlaneNormal( int const id, int const dim, real* nrml )
{
  nrml[0] = this->m_nX[id];
  nrml[1] = this->m_nY[id];
  if ( dim == 3 ) {
    nrml[2] = this->m_nZ[id];
  }
  return;
}

//------------------------------------------------------------------------------
void ContactPlaneManager::getProjectedFaceCoords( int const id, int const faceId, real* coords )
{
  // Note: at the moment there is no bounds check for coords.
  // Doxygen comments in the function prototype have been added
  // to note the expected length of the coords array.
  IndexType const numNodesPerFace = ( m_spaceDim == 3 ) ? 4 : 2;
  IndexType meshId, fId;

  if ( faceId == 0 ) {
    meshId = m_meshId1[id];
  } else {
    meshId = m_meshId2[id];
  }

  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& mesh = meshManager.GetMeshInstance( meshId );

  if ( faceId == 0 ) {
    fId = m_fId1[id];
  } else {
    fId = m_fId2[id];
  }

  real projX[numNodesPerFace];
  real projY[numNodesPerFace];
  real projZ[numNodesPerFace];

  for ( int i = 0; i < numNodesPerFace; ++i ) {
    projX[i] = 0.;
    projY[i] = 0.;
    projZ[i] = 0.;
  }

  if ( m_spaceDim == 2 ) {
    ProjectEdgeNodesToSegment( mesh, fId, m_nX[id], m_nY[id], m_cX[id], m_cY[id], &projX[0], &projY[0] );
  } else {
    ProjectFaceNodesToPlane( mesh, fId, m_nX[id], m_nY[id], m_nZ[id], m_cX[id], m_cY[id], m_cZ[id], &projX[0],
                             &projY[0], &projZ[0] );
  }

  for ( int i = 0; i < numNodesPerFace; ++i ) {
    coords[m_spaceDim * i] = projX[i];
    coords[m_spaceDim * i + 1] = projY[i];
  }

  if ( m_spaceDim == 3 ) {
    for ( int i = 0; i < numNodesPerFace; ++i ) {
      coords[m_spaceDim * i + 2] = projZ[i];
    }
  }

  return;
}

//------------------------------------------------------------------------------

}  // end of namespace tribol
