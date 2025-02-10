// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MeshData.hpp"
#include "tribol/utils/Math.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

#include "axom/slic.hpp"
#include "axom/fmt.hpp"

namespace fmt = axom::fmt;

namespace tribol {

///////////////////////////////////////////
//                                       //
// Routines for the MeshNodalData struct //
//                                       //
///////////////////////////////////////////
MeshNodalData::MeshNodalData()
{
  this->m_node_gap = nullptr;
  this->m_node_pressure = nullptr;
  return;
}

///////////////////////////////////////////
//                                       //
// Routines for the MeshElemData struct //
//                                       //
///////////////////////////////////////////
MeshElemData::MeshElemData()
{
  this->m_mat_mod = nullptr;
  this->m_thickness = nullptr;
  return;
}

//------------------------------------------------------------------------------
MeshElemData::~MeshElemData()
{
  // Don't delete any pointers to host code data (e.g. force, velocity,
  // penalty data, etc.). Don't nullify pointers as the host code
  // could register these pointers at the very beginning of a simulation
  // and not every cycle.
}

//------------------------------------------------------------------------------
bool MeshElemData::isValidKinematicPenalty( PenaltyEnforcementOptions& pen_options )
{
  // Note, this routine is, and should be called only for non-null meshes
  KinematicPenaltyCalculation kin_calc = pen_options.kinematic_calculation;

  // check kinematic penalty calculation data
  if ( !in_range( kin_calc, NUM_KINEMATIC_PENALTY_CALCULATION ) ) {
    // warning already issued when penalty options were set. If penalty options
    // were not set, the following else-if will catch this
    return false;
  }
  // a kinematic penalty should always be set. Right now Tribol does not support
  // rate only enforcement
  else if ( !pen_options.is_kinematic_calculation_set() ) {
    SLIC_WARNING( "MeshElemData::isValidKinematic(): kinematic penalty calculation data not set; "
                  << "call tribol::setPenaltyOptions()." );
    return false;
  }

  switch ( kin_calc ) {
    case KINEMATIC_CONSTANT: {
      if ( !this->m_is_kinematic_constant_penalty_set ) {
        SLIC_WARNING( "MeshElemData::isValidKinematicPenalty(): "
                      << "single stiffness penalty not set." );
        return false;
      } else if ( this->m_penalty_stiffness < pen_options.tiny_penalty ) {
        SLIC_WARNING( "MeshElemData::isValidKinematicPenalty(): "
                      << "single stiffness penalty less than threshold (" << pen_options.tiny_penalty
                      << "). Consider increasing "
                      << "for your problem." );
        return false;
      }
      break;
    }  // end case KINEMATIC_CONSTANT
    case KINEMATIC_ELEMENT: {
      if ( !this->m_is_kinematic_element_penalty_set ) {
        SLIC_WARNING( "MeshElemData::isValidKinematicPenalty(): "
                      << "element-wise penalty data not set." );
        return false;
      }

      // check for positive material modulus and thickness values
      bool isValidMatMod = true;
      bool isValidElemThickness = true;
      for ( int i = 0; i < this->m_numCells; ++i ) {
        if ( this->m_mat_mod[i] <= 0. ) {
          isValidMatMod = false;
        }
        if ( this->m_thickness[i] <= 0. ) {
          isValidElemThickness = false;
        }

        SLIC_WARNING_IF( !isValidMatMod, "MeshElemData::isValidKinematicPenalty(): "
                                             << "invalid nonpositive element material modulus encountered." );

        SLIC_WARNING_IF( !isValidElemThickness, "MeshElemData::isValidKinematicPenalty(): "
                                                    << "invalid nonpositive element thickness encountered." );

        if ( !isValidMatMod || !isValidElemThickness ) {
          return false;
        }
      }  // end for loop over elements
      break;
    }  // end case KINEMATIC_ELEMENT
    default:
      // no-op, quiet compiler
      break;
  }  // end switch over kinematic calculation

  return true;

}  // end MeshElemData::isValidKinematicPenalty()

//------------------------------------------------------------------------------
bool MeshElemData::isValidRatePenalty( PenaltyEnforcementOptions& pen_options )
{
  // Note, this method is and should only be called for non-null meshes
  RatePenaltyCalculation rate_calc = pen_options.rate_calculation;

  // check rate penalty calculation data
  if ( !in_range( rate_calc, NUM_RATE_PENALTY_CALCULATION ) ) {
    // warning already issued when penalty options were set. If penalty options
    // were not set, the following else-if will catch this
    return false;
  }
  // the rate_calc could be set to NONE and this boolean will be true
  else if ( !pen_options.is_rate_calculation_set() ) {
    SLIC_WARNING( "MeshElemData::isValidRatePenalty(): rate penalty calculation data not set. "
                  << "call tribol::setPenaltyOptions()." );
    return false;
  }

  switch ( rate_calc ) {
    case NO_RATE_PENALTY: {
      // double check that mesh booleans are consistent with rate penalty
      // calculation option
      if ( this->m_is_rate_constant_penalty_set ) {
        this->m_is_rate_constant_penalty_set = false;
      } else if ( this->m_is_rate_percent_penalty_set ) {
        this->m_is_rate_percent_penalty_set = false;
      }
      break;
    }  // end case NONE
    case RATE_CONSTANT: {
      if ( !this->m_is_rate_constant_penalty_set ) {
        SLIC_WARNING( "MeshElemData::isValidRatePenalty(): "
                      << "constant rate penalty data not set." );
        return false;
      } else if ( this->m_rate_penalty_stiffness < pen_options.tiny_penalty ) {
        SLIC_WARNING( "MeshElemData::isValidRatePenalty(): "
                      << "constant rate penalty less than threshold (" << pen_options.tiny_penalty
                      << "). Consider increasing "
                      << "for your problem." );
        return false;
      }
      break;
    }  // end case RATE_CONSTANT
    case RATE_PERCENT: {
      if ( !this->m_is_rate_percent_penalty_set ) {
        SLIC_WARNING( "MeshElemData::isValidRatePenalty(): "
                      << "percent rate penalty data not set." );
        return false;
      } else if ( this->m_rate_percent_stiffness < pen_options.tiny_penalty ||
                  this->m_rate_percent_stiffness > ( 1. - pen_options.tiny_penalty ) ) {
        SLIC_WARNING( "MeshElemData::isValidRatePenalty(): "
                      << "rate percent penalty not in (0,1)." );
        return false;
      }
      break;
    }  // end case RATE_PERCENT
    default:
      // no-op, quiet compiler
      break;
  }  // end switch on rate calculation

  return true;

}  // end MeshElemData::isValidRatePenalty()

//------------------------------------------------------------------------------

/////////////////////////////////
//                             //
// Routines for MeshData class //
//                             //
/////////////////////////////////
MeshData::MeshData()
    : m_elementType( tribol::UNDEFINED_ELEMENT ),
      m_lengthNodalData( 0 ),
      m_numSurfaceNodes( 0 ),
      m_numCells( 0 ),
      m_numNodesPerCell( 0 ),
      m_isValid( true ),
      m_positionX( nullptr ),
      m_positionY( nullptr ),
      m_positionZ( nullptr ),
      m_dispX( nullptr ),
      m_dispY( nullptr ),
      m_dispZ( nullptr ),
      m_forceX( nullptr ),
      m_forceY( nullptr ),
      m_forceZ( nullptr ),
      m_velX( nullptr ),
      m_velY( nullptr ),
      m_velZ( nullptr ),
      m_nX( nullptr ),
      m_nY( nullptr ),
      m_nZ( nullptr ),
      m_cX( nullptr ),
      m_cY( nullptr ),
      m_cZ( nullptr ),
      m_faceRadius( nullptr ),
      m_area( nullptr ),
      m_connectivity( nullptr ),
      m_sortedSurfaceNodeIds( nullptr ),
      m_node_nX( nullptr ),
      m_node_nY( nullptr ),
      m_node_nZ( nullptr )
{
}

//------------------------------------------------------------------------------
void MeshData::allocateArrays( int dimension )
{
  m_nX = new real[m_numCells];
  m_nY = new real[m_numCells];
  m_cX = new real[m_numCells];
  m_cY = new real[m_numCells];
  m_area = new real[m_numCells];
  m_faceRadius = new real[m_numCells];
  m_nZ = nullptr;
  m_cZ = nullptr;

  if ( dimension == 3 ) {
    m_nZ = new real[m_numCells];
    m_cZ = new real[m_numCells];
  }
}

//------------------------------------------------------------------------------
MeshData::~MeshData() { deallocateArrays(); }

//------------------------------------------------------------------------------
void MeshData::deallocateArrays()
{
  if ( m_nX != nullptr ) {
    delete[] m_nX;
    m_nX = nullptr;
  }

  if ( m_nY != nullptr ) {
    delete[] m_nY;
    m_nY = nullptr;
  }

  if ( m_nZ != nullptr ) {
    delete[] m_nZ;
    m_nZ = nullptr;
  }

  if ( m_cX != nullptr ) {
    delete[] m_cX;
    m_cX = nullptr;
  }

  if ( m_cY != nullptr ) {
    delete[] m_cY;
    m_cY = nullptr;
  }

  if ( m_cZ != nullptr ) {
    delete[] m_cZ;
    m_cZ = nullptr;
  }

  if ( m_node_nX != nullptr ) {
    delete[] m_node_nX;
    m_node_nX = nullptr;
  }

  if ( m_node_nY != nullptr ) {
    delete[] m_node_nY;
    m_node_nY = nullptr;
  }

  if ( m_node_nZ != nullptr ) {
    delete[] m_node_nZ;
    m_node_nZ = nullptr;
  }

  if ( m_faceRadius != nullptr ) {
    delete[] m_faceRadius;
    m_faceRadius = nullptr;
  }

  if ( m_area != nullptr ) {
    delete[] m_area;
    m_area = nullptr;
  }

  if ( m_sortedSurfaceNodeIds != nullptr ) {
    delete[] m_sortedSurfaceNodeIds;
    m_sortedSurfaceNodeIds = nullptr;
  }
}

//------------------------------------------------------------------------------
bool MeshData::computeFaceData( int const dim )
{
  bool faceDataOk = true;
  real fac = 1.0 / m_numNodesPerCell;
  constexpr real nrmlMagTol = 1.0e-15;

  // loop over all cells in the mesh
  for ( int i = 0; i < m_numCells; ++i ) {
    // compute the vertex average centroid. This will lie in the
    // plane of the face for planar faces, and will be used as
    // an approximate centroid for warped faces, both in 3D.

    // loop over the nodes per cell
    for ( int j = 0; j < m_numNodesPerCell; ++j ) {
      auto nodeIndex = m_numNodesPerCell * i + j;
      auto nodeId = m_connectivity[nodeIndex];
      // always compute the x and y components for 2D and 3D
      m_cX[i] += m_positionX[nodeId];
      m_cY[i] += m_positionY[nodeId];
    }  // end loop over nodes

    m_cX[i] = fac * m_cX[i];
    m_cY[i] = fac * m_cY[i];

    if ( dim == 3 ) {
      for ( int j = 0; j < m_numNodesPerCell; ++j ) {
        auto nodeIndex = m_numNodesPerCell * i + j;
        auto nodeId = m_connectivity[nodeIndex];
        m_cZ[i] += m_positionZ[nodeId];
      }  // end loop over nodes
      m_cZ[i] = fac * m_cZ[i];
    }       // end if-dim
    else {  // dim == 2, nullify just in case
      m_cZ = nullptr;
    }

    // compute the outward facing normal
    if ( dim == 2 ) {
      // the 2D calculation over a 2-node, 1D segment assumes a
      // counter-clockwise ordering of the quad4 area element
      // to which the 1D line segment belongs. This is to properly
      // orient the normal outward
      auto nodeIndex = m_numNodesPerCell * i;
      auto nodeId = m_connectivity[nodeIndex];
      auto nextNodeId = m_connectivity[nodeIndex + 1];
      real lambdaX = m_positionX[nextNodeId] - m_positionX[nodeId];
      real lambdaY = m_positionY[nextNodeId] - m_positionY[nodeId];

      m_nX[i] = lambdaY;
      m_nY[i] = -lambdaX;

      m_nZ = nullptr;

      // compute the length of the segment
      m_area[i] = magnitude( lambdaX, lambdaY );

      // normalize normal vector
      auto mag = magnitude( m_nX[i], m_nY[i] );
      auto invMag = nrmlMagTol;
      if ( mag >= nrmlMagTol ) {
        invMag = 1.0 / mag;
      } else {
        faceDataOk = false;
      }
      m_nX[i] *= invMag;
      m_nY[i] *= invMag;
    }

    // compute face radius for both 2D and 3D. For 2D, this is a duplicate of
    // the length, but allows for some uniformity in accessing this value
    // for tolerance ratios
    m_faceRadius[i] = this->computeFaceRadius( i );

    if ( dim == 3 ) {
      // this method of computing an outward unit normal breaks the
      // face into triangular pallets by connecting two consecutive
      // nodes with the approximate centroid.
      // The average outward unit normal for the face is the average of
      // those of the pallets. This is exact for non-warped faces. To
      // compute the pallet normal, you only need edge vectors for the
      // pallet. These are constructed from the face centroid and the face
      // edge's first node and the face edge's two nodes

      // declare triangle edge vector components and normal components
      real vX1, vY1, vZ1;
      real vX2, vY2, vZ2;
      real nX, nY, nZ;

      // loop over m_numNodesPerCell-1 cell edges and compute pallet normal
      for ( int j = 0; j < ( m_numNodesPerCell - 1 ); ++j ) {
        auto nodeIndex = m_numNodesPerCell * i + j;
        auto nodeId = m_connectivity[nodeIndex];
        auto nextNodeId = m_connectivity[nodeIndex + 1];
        // first triangle edge vector between the face's two
        // edge nodes
        vX1 = m_positionX[nextNodeId] - m_positionX[nodeId];
        vY1 = m_positionY[nextNodeId] - m_positionY[nodeId];
        vZ1 = m_positionZ[nextNodeId] - m_positionZ[nodeId];

        // second triangle edge vector between the face centroid
        // and the face edge's first node
        vX2 = m_cX[i] - m_positionX[nodeId];
        vY2 = m_cY[i] - m_positionY[nodeId];
        vZ2 = m_cZ[i] - m_positionZ[nodeId];

        // compute the contribution to the pallet normal as v1 x v2. Sum
        // these into the face normal component variables stored on the mesh data
        // object
        nX = ( vY1 * vZ2 ) - ( vZ1 * vY2 );
        nY = ( vZ1 * vX2 ) - ( vX1 * vZ2 );
        nZ = ( vX1 * vY2 ) - ( vY1 * vX2 );

        // sum the normal component contributions into the component variables
        m_nX[i] += nX;
        m_nY[i] += nY;
        m_nZ[i] += nZ;

        // half the magnitude of the computed normal is the pallet area. Note: this is exact
        // for planar faces and approximate for warped faces. Face areas are used in a general
        // sense to create a face-overlap tolerance
        m_area[i] += 0.5 * magnitude( nX, nY, nZ );
      }

      // compute the pallet normal contribution for the last pallet
      auto nodeIndex = m_numNodesPerCell * i;
      auto nodeId = m_connectivity[nodeIndex];
      auto nextNodeId = m_connectivity[nodeIndex + m_numNodesPerCell - 1];
      vX1 = m_positionX[nodeId] - m_positionX[nextNodeId];
      vY1 = m_positionY[nodeId] - m_positionY[nextNodeId];
      vZ1 = m_positionZ[nodeId] - m_positionZ[nextNodeId];

      vX2 = m_cX[i] - m_positionX[nextNodeId];
      vY2 = m_cY[i] - m_positionY[nextNodeId];
      vZ2 = m_cZ[i] - m_positionZ[nextNodeId];

      nX = ( vY1 * vZ2 ) - ( vZ1 * vY2 );
      nY = ( vZ1 * vX2 ) - ( vX1 * vZ2 );
      nZ = ( vX1 * vY2 ) - ( vY1 * vX2 );

      // sum the normal component contributions into the component variables
      m_nX[i] += nX;
      m_nY[i] += nY;
      m_nZ[i] += nZ;

      // half the magnitude of the computed normal is the pallet area. Note: this is exact
      // for planar faces and approximate for warped faces. Face areas are used in a general
      // sense to create a face-overlap tolerance
      m_area[i] += 0.5 * magnitude( nX, nY, nZ );

      // multiply the pallet normal components by fac to obtain avg.
      m_nX[i] = fac * m_nX[i];
      m_nY[i] = fac * m_nY[i];
      m_nZ[i] = fac * m_nZ[i];

      // compute the magnitude of the average pallet normal
      auto mag = magnitude( m_nX[i], m_nY[i], m_nZ[i] );
      auto invMag = nrmlMagTol;
      if ( mag >= nrmlMagTol ) {
        invMag = 1.0 / mag;
      } else {
        faceDataOk = false;
      }

      // normalize the average normal
      m_nX[i] *= invMag;
      m_nY[i] *= invMag;
      m_nZ[i] *= invMag;

    }  // end if (dim == 3)

  }  // end cell loop

  SLIC_WARNING_IF(
      !faceDataOk,
      axom::fmt::format( "There are faces with a normal magnitude less than tolerance ({:e}).", nrmlMagTol ) );

  return faceDataOk;

}  // end MeshData::computeFaceData()

//------------------------------------------------------------------------------
real MeshData::computeFaceRadius( int faceId )
{
  // loop over nodes of the face and determine the maximum
  // "link" vector from the ith node to the face center
  real sqrRadius = 0.0;
  for ( int i = 0; i < m_numNodesPerCell; ++i ) {
    const int nodeId = getFaceNodeId( faceId, i );
    real lvx = m_positionX[nodeId] - m_cX[faceId];
    real lvy = m_positionY[nodeId] - m_cY[faceId];

    real lvz;
    if ( m_positionZ != nullptr ) {  // for 3D
      lvz = m_positionZ[nodeId] - m_cZ[faceId];
    } else {
      lvz = 0.0;
    }

    real sqrLinkMag = lvx * lvx + lvy * lvy + lvz * lvz;

    if ( sqrLinkMag > sqrRadius ) {
      sqrRadius = sqrLinkMag;
    }
  }

  return sqrt( sqrRadius );

}  // end MeshData::computeFaceRadius()

//------------------------------------------------------------------------------
real MeshData::computeEdgeLength( int faceId )
{
  // compute the length of the edge as the magnitude of
  // the vector defined between the two edge vertices
  const int nodeId1 = getFaceNodeId( faceId, 0 );
  const int nodeId2 = getFaceNodeId( faceId, 1 );
  real lvx = m_positionX[nodeId2] - m_positionX[nodeId1];
  real lvy = m_positionY[nodeId2] - m_positionY[nodeId1];

  real len = magnitude( lvx, lvy );

  return len;

}  // end MeshData::computeEdgeLength()

//------------------------------------------------------------------------------
void MeshData::getFaceCoords( int const faceId, real* coords )
{
  int dim = ( m_positionZ != nullptr ) ? 3 : 2;

  for ( IndexType a = 0; a < m_numNodesPerCell; ++a ) {
    IndexType nodeId = m_connectivity[faceId * m_numNodesPerCell + a];

    coords[dim * a] = m_positionX[nodeId];
    coords[dim * a + 1] = m_positionY[nodeId];

    if ( dim == 3 ) {
      coords[dim * a + 2] = m_positionZ[nodeId];
    }
  }

  return;

}  // end MeshData::getFaceCoords()

//------------------------------------------------------------------------------
void MeshData::getFaceNodalVelocities( int const faceId, real* nodalVel )
{
  if ( m_velX == nullptr || m_velY == nullptr ) {
    SLIC_ERROR( "MeshData::getFaceNodalVelocities(): velocity arrays on each "
                << "mesh must be registered with Tribol." );
  }

  int dim = ( m_velZ != nullptr ) ? 3 : 2;

  for ( IndexType a = 0; a < m_numNodesPerCell; ++a ) {
    IndexType nodeId = m_connectivity[faceId * m_numNodesPerCell + a];

    nodalVel[dim * a] = m_velX[nodeId];
    nodalVel[dim * a + 1] = m_velY[nodeId];

    if ( dim == 3 ) {
      nodalVel[dim * a + 2] = m_velZ[nodeId];
    }
  }

  return;

}  // end MeshData::getFaceNodalVelocities()

//------------------------------------------------------------------------------
void MeshData::computeNodalNormals( int const dim )
{
  int* numFaceNrmlsToNodes;
  if ( this->m_numCells > 0 ) {
    // check to make sure face normals have been computed with
    // a call to computeFaceData
    if ( this->m_nX == nullptr || this->m_nY == nullptr ) {
      SLIC_ERROR( "MeshData::computeNodalNormals: required face normals not computed." );
    }

    // allocate space for nodal normal array
    int size = this->m_lengthNodalData;
    if ( this->m_node_nX != nullptr ) {
      delete[] m_node_nX;
      m_node_nX = new real[size];
    } else {
      m_node_nX = new real[size];
    }

    if ( this->m_node_nY != nullptr ) {
      delete[] m_node_nY;
      m_node_nY = new real[size];
    } else {
      m_node_nY = new real[size];
    }

    if ( dim == 3 ) {
      if ( this->m_node_nZ != nullptr ) {
        delete[] m_node_nZ;
        m_node_nZ = new real[size];
      } else {
        m_node_nZ = new real[size];
      }

      // initialize z component
      initRealArray( m_node_nZ, size, 0. );
    }

    // initialize x and y components
    initRealArray( m_node_nX, size, 0. );
    initRealArray( m_node_nY, size, 0. );

    // allocate scratch array to hold number of faces whose
    // normals contribute to a given node. Most of the time
    // this will be four face normals contributing to an
    // averaged nodal normal for linear quad elements, but
    // we want to handle arbitrary meshes and edge cases
    allocIntArray( &numFaceNrmlsToNodes, size, 0 );
  }  // end if-check on null mesh

  // loop over cells
  for ( int i = 0; i < this->m_numCells; ++i ) {
    // loop over cell nodes
    for ( int j = 0; j < this->m_numNodesPerCell; ++j ) {
      // SRW: note the connectivity array must be local to the mesh for indexing into
      // the mesh nodal normal array. If it is not, then nodeId will access some other
      // piece of memory and there may be a memory issue when numFaceNrmlsToNodes is deleted
      // at the end of this routine.
      integer nodeId = this->m_connectivity[this->m_numNodesPerCell * i + j];
      m_node_nX[nodeId] += this->m_nX[i];  // m_nX[i] is the ith face normal x-component
      m_node_nY[nodeId] += this->m_nY[i];  // see above...

      // increment face-normal-to-node contribution counter
      ++numFaceNrmlsToNodes[nodeId];

    }  // end loop over cell nodes

    // populate z-component for 3D problems
    if ( dim == 3 ) {
      // repeat loop over cell nodes for z-component
      for ( int k = 0; k < this->m_numNodesPerCell; ++k ) {
        integer nodeId = this->m_connectivity[this->m_numNodesPerCell * i + k];
        m_node_nZ[nodeId] += this->m_nZ[i];
      }  // end loop over cell nodes

    }  // end if (dim == 3)

  }  // end loop over cells

  // average the nodal normals
  if ( this->m_numCells > 0 ) {
    for ( int i = 0; i < this->m_lengthNodalData; ++i ) {
      m_node_nX[i] /= numFaceNrmlsToNodes[i];
      m_node_nY[i] /= numFaceNrmlsToNodes[i];
      if ( dim == 3 ) {
        m_node_nZ[i] /= numFaceNrmlsToNodes[i];
      }
    }  // end loop over nodes
  }

  // normalize the nodal normals
  if ( this->m_numCells > 0 ) {
    if ( dim == 3 ) {
      for ( int i = 0; i < this->m_lengthNodalData; ++i ) {
        real mag = magnitude( m_node_nX[i], m_node_nY[i], m_node_nZ[i] );
        m_node_nX[i] /= mag;
        m_node_nY[i] /= mag;
        m_node_nZ[i] /= mag;
      }
    } else {
      for ( int i = 0; i < this->m_lengthNodalData; ++i ) {
        real mag = magnitude( m_node_nX[i], m_node_nY[i] );
        m_node_nX[i] /= mag;
        m_node_nY[i] /= mag;
      }
    }

    delete[] numFaceNrmlsToNodes;
  }  // end if-check on null mesh

  return;
}  // end MeshData::computeNodalNormals()

//------------------------------------------------------------------------------
void MeshData::getFaceNormal( int const faceId, int const dim, real* nrml )
{
  nrml[0] = this->m_nX[faceId];
  nrml[1] = this->m_nY[faceId];

  if ( dim == 3 ) {
    nrml[2] = this->m_nZ[faceId];
  }

  return;

}  // end MeshData::getFaceNormal()

//------------------------------------------------------------------------------
void MeshData::sortSurfaceNodeIds()
{
  // check that the number of cells is greater than 0
  SLIC_ERROR_IF( ( this->m_numCells == 0 ), "MeshData::sortSurfaceNodeIds(): "
                                                << "m_numCells on mesh cannot be equal to zero." );

  // check that the number of cell nodes is greater than 2 (minimum number of cell
  // nodes for 1D segment
  SLIC_ERROR_IF( ( this->m_numNodesPerCell < 2 ), "MeshData::sortSurfaceNodeIds(): "
                                                      << "m_numNodesPerCell on mesh cannot be equal to zero." );

  int conn_size = this->m_numCells * this->m_numNodesPerCell;

  int sorted_conn[conn_size];
  for ( int i = 0; i < conn_size; ++i ) {
    sorted_conn[i] = this->m_connectivity[i];
  }

  bubble_sort( &sorted_conn[0], conn_size );

  // count number of duplicate entries
  int num_dup = 0;
  for ( int i = 1; i < conn_size; ++i ) {
    if ( sorted_conn[i] == sorted_conn[i - 1] ) {
      ++num_dup;
    }
  }

  // compute number of unique integer ids
  int unique_size = conn_size - num_dup;

  SLIC_ERROR_IF( ( unique_size <= 0 ), "MeshData::sortSurfaceNodeIds(): "
                                           << "invalid connectivity array; "
                                           << "only single unique id in connectivity array." );

  // allocate array to store unique, sorted node ids on mesh object
  this->m_sortedSurfaceNodeIds = new int[unique_size];
  this->m_numSurfaceNodes = unique_size;

  // populate sorted node id list
  this->m_sortedSurfaceNodeIds[0] = sorted_conn[0];
  int k = 1;
  for ( int i = 1; i < conn_size; ++i ) {
    if ( sorted_conn[i] != sorted_conn[i - 1] ) {
      this->m_sortedSurfaceNodeIds[k] = sorted_conn[i];
      ++k;
    }
  }

  return;
}  // end MeshData::sortSurfaceNodeIds()

//------------------------------------------------------------------------------
int MeshData::checkLagrangeMultiplierData()
{
  int err = 0;
  if ( this->m_numCells > 0 ) {
    if ( !this->m_nodalFields.m_is_node_gap_set || !this->m_nodalFields.m_is_node_pressure_set ) {
      err = 1;
    }
  }  // end if-non-null mesh
  return err;
}
//------------------------------------------------------------------------------
int MeshData::checkPenaltyData( PenaltyEnforcementOptions& p_enfrc_options )
{
  int err = 0;
  if ( this->m_numCells > 0 ) {
    PenaltyConstraintType constraint_type = p_enfrc_options.constraint_type;
    // switch over penalty enforcement options and check for required data
    switch ( constraint_type ) {
      case KINEMATIC: {
        if ( !this->m_elemData.isValidKinematicPenalty( p_enfrc_options ) ) {
          err = 1;
        }
        break;
      }  // end KINEMATIC case

      case KINEMATIC_AND_RATE: {
        if ( !this->m_elemData.isValidKinematicPenalty( p_enfrc_options ) ) {
          err = 1;
        }
        if ( !this->m_elemData.isValidRatePenalty( p_enfrc_options ) ) {
          err = 1;
        }
        if ( !this->m_nodalFields.m_is_velocity_set ) {
          SLIC_WARNING( "Nodal velocities not set or null pointers; please set for "
                        << "use with gap rate penalty enforcement." );
          err = 1;
        }
        break;
      }  // end case KINEMATIC_AND_RATE
      default:
        // no-op, quiet compiler
        break;
    }  // end switch over constraint types
  }    // end if-non-null mesh

  return err;
}  // end MeshData::checkPenaltyData()

//------------------------------------------------------------------------------
int MeshData::localIdFromConn( const int connectivityId )
{
  return binary_search( this->m_sortedSurfaceNodeIds, this->m_numSurfaceNodes, connectivityId );

}  // end MeshData::localIdFromConn()

//------------------------------------------------------------------------------
void MeshData::print( std::ostream& os ) const
{
  const int num_verts = m_lengthNodalData;
  const int num_elem = m_numCells;
  const int dim = m_positionZ == nullptr ? 2 : 3;

  if ( num_verts <= 0 ) {
    os << "{}";
    return;
  }

  os << "{\n";
  os << fmt::format( "  verts ({}) {{", num_verts );
  // positions
  os << fmt::format( "\n\tx: {}", fmt::join( m_positionX, m_positionX + num_verts, ", " ) );
  os << fmt::format( "\n\ty: {}", fmt::join( m_positionY, m_positionY + num_verts, ", " ) );
  if ( dim == 3 ) {
    os << fmt::format( "\n\tz: {}", fmt::join( m_positionZ, m_positionZ + num_verts, ", " ) );
  }
  // contact force
  if ( m_forceX != nullptr ) {
    os << fmt::format( "\n\tfx: {}", fmt::join( m_forceX, m_forceX + num_verts, ", " ) );
    os << fmt::format( "\n\tfy: {}", fmt::join( m_forceY, m_forceY + num_verts, ", " ) );
    if ( dim == 3 ) {
      os << fmt::format( "\n\tfz: {}", fmt::join( m_forceZ, m_forceZ + num_verts, ", " ) );
    }
  }
  os << "\n  }";

  os << fmt::format( "\n  elems ({}) {{", num_elem );

  if ( m_connectivity != nullptr ) {
    os << fmt::format( "\n\tconnectivity: {{ {} }}",
                       fmt::join( m_connectivity, m_connectivity + ( num_elem * m_numNodesPerCell ), ", " ) );
  }

  // normals
  if ( m_nX != nullptr ) {
    os << fmt::format( "\n\tnx: {}", fmt::join( m_nX, m_nX + num_elem, ", " ) );
    os << fmt::format( "\n\tny: {}", fmt::join( m_nY, m_nY + num_elem, ", " ) );
    if ( dim == 3 ) {
      os << fmt::format( "\n\tnz: {}", fmt::join( m_nZ, m_nZ + num_elem, ", " ) );
    }
  }
  os << "\n  }";

  os << "\n}";
}

//------------------------------------------------------------------------------
}  // namespace tribol

std::ostream& operator<<( std::ostream& os, const tribol::MeshData& md )
{
  md.print( os );
  return os;
}
