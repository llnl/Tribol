// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MeshData.hpp"
#include "tribol/common/ExecModel.hpp"
#include "tribol/utils/Math.hpp"

#include <cmath> 
#include <iostream> 
#include <sstream>
#include <iomanip>
#include <fstream>

#include "axom/slic.hpp"
#include "axom/fmt.hpp"

namespace tribol
{

//------------------------------------------------------------------------------
bool MeshElemData::isValidKinematicPenalty( PenaltyEnforcementOptions& pen_options )
{
  // Note, this routine is, and should be called only for non-null meshes
  KinematicPenaltyCalculation kin_calc = pen_options.kinematic_calculation;

  // check kinematic penalty calculation data
  if ( !in_range( kin_calc, NUM_KINEMATIC_PENALTY_CALCULATION) )
  {
    // warning already issued when penalty options were set. If penalty options 
    // were not set, the following else-if will catch this
    return false;
  }
  // a kinematic penalty should always be set. Right now Tribol does not support 
  // rate only enforcement
  else if ( !pen_options.kinematic_calc_set )
  {
    SLIC_WARNING( "MeshElemData::isValidKinematic(): kinematic penalty calculation data not set; " << 
                  "call tribol::setPenaltyOptions()." );
    return false;
  }

  switch (kin_calc)
  {
    case KINEMATIC_CONSTANT:
    {
      if ( !this->m_is_kinematic_constant_penalty_set )
      {
        SLIC_WARNING("MeshElemData::isValidKinematicPenalty(): " << 
                     "single stiffness penalty not set.");
        return false;
      }
      else if ( this->m_penalty_stiffness < pen_options.tiny_penalty )
      {
        SLIC_WARNING("MeshElemData::isValidKinematicPenalty(): " << 
                     "single stiffness penalty less than threshold (" << 
                     pen_options.tiny_penalty << "). Consider increasing " << 
                     "for your problem.");
        return false;
      }
      break;
    } // end case KINEMATIC_CONSTANT
    case KINEMATIC_ELEMENT:
    {
      if ( !this->m_is_kinematic_element_penalty_set )
      {
        SLIC_WARNING("MeshElemData::isValidKinematicPenalty(): " << 
                     "element-wise penalty data not set.");
        return false;
      }

      // check for positive material modulus and thickness values
      bool isValidMatMod = true;
      bool isValidElemThickness = true;
      for (int i=0; i<this->m_num_cells; ++i)
      {
        if (this->m_mat_mod[i] <= 0.)
        {
          isValidMatMod = false;
        }
        if (this->m_thickness[i] <= 0.)
        {
          isValidElemThickness = false;
        }

        SLIC_WARNING_IF(!isValidMatMod, "MeshElemData::isValidKinematicPenalty(): " << 
                        "invalid nonpositive element material modulus encountered.");

        SLIC_WARNING_IF(!isValidElemThickness, "MeshElemData::isValidKinematicPenalty(): " << 
                        "invalid nonpositive element thickness encountered.");

        if (!isValidMatMod || !isValidElemThickness)
        {
          return false;
        }
      } // end for loop over elements
      break;
    } // end case KINEMATIC_ELEMENT
    default:
      // no-op, quiet compiler
      break;
  } // end switch over kinematic calculation

  return true;
 
} // end MeshElemData::isValidKinematicPenalty()

//------------------------------------------------------------------------------
bool MeshElemData::isValidRatePenalty( PenaltyEnforcementOptions& pen_options )
{
  // Note, this method is and should only be called for non-null meshes
  RatePenaltyCalculation rate_calc = pen_options.rate_calculation; 

  // check rate penalty calculation data
  if ( !in_range( rate_calc, NUM_RATE_PENALTY_CALCULATION) )
  {
    // warning already issued when penalty options were set. If penalty options 
    // were not set, the following else-if will catch this
    return false;
  }
  // the rate_calc could be set to NONE and this boolean will be true
  else if ( !pen_options.rate_calc_set )
  {
    SLIC_WARNING( "MeshElemData::isValidRatePenalty(): rate penalty calculation data not set. " << 
                  "call tribol::setPenaltyOptions()." );
    return false;
  }

  switch (rate_calc)
  {
    case NO_RATE_PENALTY:
    {
      // double check that mesh booleans are consistent with rate penalty 
      // calculation option
      if (this->m_is_rate_constant_penalty_set)
      {
        this->m_is_rate_constant_penalty_set = false;
      }
      else if (this->m_is_rate_percent_penalty_set)
      {
        this->m_is_rate_percent_penalty_set = false;
      }
      break;
    } // end case NONE
    case RATE_CONSTANT:
    {
      if ( !this->m_is_rate_constant_penalty_set )
      {
        SLIC_WARNING("MeshElemData::isValidRatePenalty(): " << 
                     "constant rate penalty data not set.");
        return false;
      }
      else if ( this->m_rate_penalty_stiffness < pen_options.tiny_penalty )
      {
        SLIC_WARNING("MeshElemData::isValidRatePenalty(): " << 
                     "constant rate penalty less than threshold (" << 
                     pen_options.tiny_penalty << "). Consider increasing " << 
                     "for your problem.");
        return false;
      }
      break;
    } // end case RATE_CONSTANT
    case RATE_PERCENT:
    {
      if ( !this->m_is_rate_percent_penalty_set )
      {
        SLIC_WARNING("MeshElemData::isValidRatePenalty(): " << 
                     "percent rate penalty data not set.");
        return false;
      }
      else if ( this->m_rate_percent_stiffness < pen_options.tiny_penalty ||
                this->m_rate_percent_stiffness > (1.-pen_options.tiny_penalty) )
      {
        SLIC_WARNING("MeshElemData::isValidRatePenalty(): " << 
                     "rate percent penalty not in (0,1).");
        return false;
      }
      break;
    } // end case RATE_PERCENT
    default:
      // no-op, quiet compiler
      break;
  } // end switch on rate calculation

  return true;

} // end MeshElemData::isValidRatePenalty()

//------------------------------------------------------------------------------

/////////////////////////////////
//                             //
// Routines for MeshData class //
//                             //
/////////////////////////////////
MeshData::MeshData( IndexT mesh_id, IndexT num_elements, IndexT num_nodes,
                    const IndexT* connectivity, InterfaceElementType element_type,
                    const RealT* x, const RealT* y, const RealT* z,
                    MemorySpace mem_space )
  : m_mesh_id( mesh_id )
  , m_element_type( element_type )
  , m_dim( getDimFromElementType() )
  , m_num_nodes( num_nodes )
  , m_mem_space( mem_space )
  , m_allocator_id( getResourceAllocatorID(mem_space) )
  , m_is_valid( true )
  , m_position( createNodalVector(x, y, z) )
  , m_connectivity( createConnectivity(num_elements, connectivity) )
{
  // mesh verification
  if (num_elements > 0)
  {
    if (m_dim == 2 && (x == nullptr || y == nullptr))
    {
      SLIC_WARNING_ROOT("tribol::MeshData(): pointer to x and/or y-component " << 
                        "mesh coordinate array is a null pointer " <<
                        "for mesh id " << m_mesh_id << ".");
      m_is_valid = false;
    }
    else if (m_dim == 3 && (x == nullptr || y == nullptr || z == nullptr))
    {
      SLIC_WARNING_ROOT("tribol::MeshData(): pointer to x, y, and/or z-component " << 
                        "mesh coordinate array is a null pointer " <<
                        "for mesh id " << m_mesh_id << ".");
      m_is_valid = false;
    }
    if (connectivity == nullptr)
    {
      SLIC_WARNING_ROOT("tribol::MeshData(): pointer to mesh connectivity is " <<
                        "a null pointer for mesh id " << m_mesh_id << ".");
    }
  }

  getElementData().m_num_cells = num_elements;
  getNodalFields().m_num_nodes = num_nodes;
}

//------------------------------------------------------------------------------
void MeshData::setPosition( const RealT* x,
                            const RealT* y,
                            const RealT* z )
{
  m_position = createNodalVector(x, y, z);
}

//------------------------------------------------------------------------------
void MeshData::setDisplacement( const RealT* ux,
                                const RealT* uy,
                                const RealT* uz )
{
  m_disp = createNodalVector(ux, uy, uz);
}

//------------------------------------------------------------------------------
void MeshData::setVelocity( const RealT* vx,
                            const RealT* vy,
                            const RealT* vz )
{
  m_vel = createNodalVector(vx, vy, vz);
}

//------------------------------------------------------------------------------
void MeshData::setResponse( RealT* rx,
                            RealT* ry,
                            RealT* rz )
{
  m_response = createNodalVector(rx, ry, rz);
}

//------------------------------------------------------------------------------
int MeshData::getDimFromElementType() const
{
  switch (m_element_type)
  {
    case LINEAR_EDGE:
    {
      return 2;
    }
    case LINEAR_TRIANGLE:
    case LINEAR_QUAD:
    {
      return 3;
    }
    default:
    {
      SLIC_ERROR_ROOT("Unsupported element type for a contact mesh.");
      return 0;
    }
  }
}

//------------------------------------------------------------------------------
ArrayViewT<const IndexT, 2> MeshData::createConnectivity(IndexT num_elements, const IndexT* connectivity)
{
  switch (m_element_type)
  {
    case LINEAR_EDGE:
    {
      return ArrayViewT<const IndexT, 2>(connectivity, {num_elements, 2});
    }
    case LINEAR_TRIANGLE:
    {
      return ArrayViewT<const IndexT, 2>(connectivity, {num_elements, 3});
    }
    case LINEAR_QUAD:
    {
      return ArrayViewT<const IndexT, 2>(connectivity, {num_elements, 4});
    }
    default:
    {
      SLIC_ERROR_ROOT("Unsupported element type for a contact mesh.");
      return ArrayViewT<const IndexT, 2>(connectivity, {num_elements, 0});
    }
  }
}

//------------------------------------------------------------------------------
Array1D<IndexT> MeshData::sortSurfaceNodeIds()
{
  ArrayT<IndexT> sorted_conn(0, m_connectivity.size());
  for (auto node_id : m_connectivity)
  {
    sorted_conn.push_back(node_id);
  }

  bubble_sort(sorted_conn.data(), sorted_conn.size());

  // count number of duplicate entries
  int num_dup = 0;
  for (IndexT i{1}; i < sorted_conn.size(); ++i)
  {
    if (sorted_conn[i] == sorted_conn[i-1])
    {
      ++num_dup;
    }
  }

  // compute number of unique integer ids
  int unique_size = sorted_conn.size() - num_dup;

  SLIC_ERROR_IF(unique_size <= 0, "MeshData::sortSurfaceNodeIds(): " << 
                "invalid connectivity array; " <<
                "only single unique id in connectivity array.");

  // allocate array to store unique, sorted node ids on mesh object
  auto sorted_surface_node_ids = ArrayT<IndexT>(0, unique_size);

  // populate sorted node id list
  sorted_surface_node_ids.push_back(sorted_conn[0]);
  for (IndexT i{1}; i < sorted_conn.size(); ++i)
  {
    if ( sorted_conn[i] != sorted_conn[i-1] )
    { 
      sorted_surface_node_ids.push_back(sorted_conn[i]);
    }
  }

  return sorted_surface_node_ids;
} // end MeshData::sortSurfaceNodeIds()

//------------------------------------------------------------------------------
bool MeshData::computeFaceData(ExecutionMode exec_mode)
{
  constexpr RealT nrml_mag_tol = 1.0e-15;

  // allocate m_c
  m_c = Array2D<RealT>({m_dim, numberOfElements()}, m_allocator_id);
  // initialize m_c
  m_c.fill(0.0);

  // allocate m_n
  m_n = Array2D<RealT>({m_dim, numberOfElements()}, m_allocator_id);
  // initialize m_n
  m_n.fill(0.0);

  // allocate m_area (initialized to 0.0)
  m_area = Array1D<RealT>(numberOfElements(), numberOfElements(), m_allocator_id);

  // allocate face_radius (initialized to 0.0)
  m_face_radius = Array1D<RealT>(numberOfElements(), numberOfElements(), m_allocator_id);
  
  ArrayT<IndexT> face_data_ok_data({static_cast<IndexT>(true)}, m_allocator_id);

  // loop over all elements in the mesh
  Array2DView<RealT> c = m_c;
  MultiViewArrayView<const RealT> x = m_position;
  Array2DView<RealT> n = m_n;
  Array1DView<RealT> area = m_area;
  Array1DView<RealT> radius = m_face_radius;
  auto dim = m_dim;
  auto conn = m_connectivity;
  ArrayViewT<IndexT> face_data_ok = face_data_ok_data;
  forAllExec(exec_mode, numberOfElements(), 
    [c, x, n, area, radius, dim, conn, face_data_ok] TRIBOL_HOST_DEVICE (IndexT i) {

      // compute the vertex average centroid. This will lie in the 
      // plane of the face for planar faces, and will be used as 
      // an approximate centroid for warped faces, both in 3D.

      // loop over the nodes per element
      auto num_nodes_per_elem = conn.shape()[1];
      for (int j=0; j<num_nodes_per_elem; ++j) {
        auto node_id = conn(i, j);
        for (IndexT d{0}; d < dim; ++d)
        {
          c[d][i] += x[d][ node_id ];
        }
      } // end loop over nodes
     
      RealT fac = 1.0 / num_nodes_per_elem;
      for (IndexT d{0}; d < dim; ++d)
      {
        c[d][i] = fac * c[d][i];
      }

      // compute face radius for both 2D and 3D. For 2D, this is a duplicate of 
      // the length, but allows for some uniformity in accessing this value 
      // for tolerance ratios
      // loop over nodes of the face and determine the maximum 
      // "link" vector from the ith node to the face center
      RealT sqr_radius = 0.0;
      for (int j=0; j<num_nodes_per_elem; ++j) {
        const int node_id = conn(i, j);
        RealT sqr_link_mag = 0.0;
        for (IndexT d{0}; d < dim; ++d)
        {
          RealT lv = x[d][node_id] - c[d][i];
          sqr_link_mag += lv * lv;
        }
        if (sqr_link_mag > sqr_radius) {
          sqr_radius = sqr_link_mag;
        }
      } 
      radius[i] = sqrt(sqr_radius);

      // compute the outward facing normal
      if (dim == 2) {

        // the 2D calculation over a 2-node, 1D segment assumes a 
        // counter-clockwise ordering of the quad4 area element
        // to which the 1D line segment belongs. This is to properly 
        // orient the normal outward
        auto node_id = conn(i, 0);
        auto next_node_id = conn(i, 1);
        RealT lambdaX = x[0][ next_node_id ] - x[0][ node_id ];
        RealT lambdaY = x[1][ next_node_id ] - x[1][ node_id ];
  
        n[0][i] = lambdaY;
        n[1][i] = -lambdaX;

        // compute the length of the segment
        area[i] = magnitude( lambdaX, lambdaY );

        // normalize normal vector
        auto mag = magnitude( n[0][i], n[1][i] );
        auto inv_mag = nrml_mag_tol;
        if (mag >= nrml_mag_tol) {
          inv_mag = 1.0 / mag;
        } else {
          face_data_ok[0] = static_cast<IndexT>(false);
        }
        n[0][i] *= inv_mag;
        n[1][i] *= inv_mag;

      }
      else if (dim == 3) {

        // this method of computing an outward unit normal breaks the 
        // face into triangular pallets by connecting two consecutive 
        // nodes with the approximate centroid.
        // The average outward unit normal for the face is the average of 
        // those of the pallets. This is exact for non-warped faces. To 
        // compute the pallet normal, you only need edge vectors for the 
        // pallet. These are constructed from the face centroid and the face 
        // edge's first node and the face edge's two nodes

        // loop over num_nodes_per_elem-1 element edges and compute pallet
        // normal
        for (int j=0; j<num_nodes_per_elem; ++j) 
        {
          auto node_id = conn(i, j);
          auto next_node_id = conn(i, 0);
          if (j < num_nodes_per_elem - 1)
          {
            next_node_id = conn(i, j+1);
          }
          // first triangle edge vector between the face's two edge nodes
          auto vX1 = x[0][ next_node_id ] - x[0][ node_id ];
          auto vY1 = x[1][ next_node_id ] - x[1][ node_id ];
          auto vZ1 = x[2][ next_node_id ] - x[2][ node_id ];
          
          // second triangle edge vector between the face centroid 
          // and the face edge's first node
          auto vX2 = c[0][i] - x[0][ node_id ];
          auto vY2 = c[1][i] - x[1][ node_id ];
          auto vZ2 = c[2][i] - x[2][ node_id ];

          // compute the contribution to the pallet normal as v1 x v2. Sum these
          // into the face normal component variables stored on the mesh data
          // object
          auto nX = (vY1 * vZ2) - (vZ1 * vY2);
          auto nY = (vZ1 * vX2) - (vX1 * vZ2);
          auto nZ = (vX1 * vY2) - (vY1 * vX2);

          // sum the normal component contributions into the component variables
          n[0][i] += nX;
          n[1][i] += nY;
          n[2][i] += nZ;

          // half the magnitude of the computed normal is the pallet area. Note:
          // this is exact for planar faces and approximate for warped faces.
          // Face areas are used in a general sense to create a face-overlap
          // tolerance
          area[i] += 0.5 * magnitude( nX, nY, nZ );
        }

        // multiply the pallet normal components by fac to obtain avg.
        n[0][i] = fac * n[0][i];
        n[1][i] = fac * n[1][i];
        n[2][i] = fac * n[2][i];

        // compute the magnitude of the average pallet normal
        auto mag = magnitude(n[0][i], n[1][i], n[2][i] );
        auto inv_mag = nrml_mag_tol;
        if (mag >= nrml_mag_tol) {
          inv_mag = 1.0 / mag;
        } else {
          face_data_ok[0] = static_cast<IndexT>(false);
        }

        // normalize the average normal
        n[0][i] *= inv_mag;
        n[1][i] *= inv_mag;
        n[2][i] *= inv_mag;

      } // end if (dim == 3)

  }); // end element loop

  ArrayT<IndexT, 1, MemorySpace::Host> face_data_ok_host(face_data_ok_data);
  SLIC_WARNING_IF(!face_data_ok_host[0], 
      axom::fmt::format("There are faces with a normal magnitude less than tolerance ({:e}).", nrml_mag_tol));

  return face_data_ok_host[0];

} // end MeshData::computeFaceData()

//------------------------------------------------------------------------------
RealT MeshData::computeEdgeLength( int faceId ) 
{
   // compute the length of the edge as the magnitude of 
   // the vector defined between the two edge vertices
   const int nodeId1 = getGlobalNodeId( faceId, 0 );
   const int nodeId2 = getGlobalNodeId( faceId, 1 );
   RealT lvx = m_position[0][nodeId2] - m_position[0][nodeId1];
   RealT lvy = m_position[1][nodeId2] - m_position[1][nodeId1];

   RealT len = magnitude( lvx, lvy );

   return len;

} // end MeshData::computeEdgeLength()

//------------------------------------------------------------------------------
void MeshData::computeNodalNormals( int const dim )
{
   int * numFaceNrmlsToNodes;
   if (this->numberOfElements() > 0)
   {
      // check to make sure face normals have been computed with 
      // a call to computeFaceData
      if (m_n.empty())
      {
         SLIC_ERROR("MeshData::computeNodalNormals: required face normals not computed.");
      }

      m_node_n = Array2D<RealT>({m_dim, m_num_nodes}, m_allocator_id);
      m_node_n.fill(0.0);

      // allocate space for nodal normal array
      int size = m_num_nodes; 
      // allocate scratch array to hold number of faces whose 
      // normals contribute to a given node. Most of the time 
      // this will be four face normals contributing to an 
      // averaged nodal normal for linear quad elements, but 
      // we want to handle arbitrary meshes and edge cases
      allocIntArray( &numFaceNrmlsToNodes, size, 0 );
   } // end if-check on null mesh

   // loop over elements
   for (int i=0; i<this->numberOfElements(); ++i)
   {
      // loop over element nodes
      for (int j=0; j<this->numberOfNodesPerElement(); ++j)
      {

         // SRW: note the connectivity array must be local to the mesh for indexing into 
         // the mesh nodal normal array. If it is not, then nodeId will access some other
         // piece of memory and there may be a memory issue when numFaceNrmlsToNodes is deleted
         // at the end of this routine.
         int nodeId = getGlobalNodeId(i, j);
         for (int d=0; d<this->spatialDimension(); ++d)
         {
            m_node_n(d, nodeId) += this->m_n(d, i); // m_n(d, i) is the ith face normal d-component
         }

         // increment face-normal-to-node contribution counter
         ++numFaceNrmlsToNodes[ nodeId ];

      } // end loop over element nodes

   } // end loop over elements

   // average the nodal normals
   if (this->numberOfElements() > 0)
   {
      for (int i=0; i<m_num_nodes; ++i)
      {
         m_node_n[0][i] /= numFaceNrmlsToNodes[i];
         m_node_n[1][i] /= numFaceNrmlsToNodes[i];
         if (dim == 3)
         {
            m_node_n[2][i] /= numFaceNrmlsToNodes[i];
         }
      } // end loop over nodes
   }

   // normalize the nodal normals
   if (this->numberOfElements() > 0)
   {
      if (dim == 3)
      {
         for (int i=0; i<m_num_nodes; ++i)
         {
            RealT mag = magnitude( m_node_n[0][i], m_node_n[1][i], m_node_n[2][i] );
            m_node_n[0][ i ] /= mag;
            m_node_n[1][ i ] /= mag;
            m_node_n[2][ i ] /= mag;
         }
      }
      else 
      {
         for (int i=0; i<m_num_nodes; ++i)
         {
            RealT mag = magnitude( m_node_n[0][i], m_node_n[1][i] );
            m_node_n[0][ i ] /= mag;
            m_node_n[1][ i ] /= mag;
         }
      }

      delete [] numFaceNrmlsToNodes;
   } // end if-check on null mesh

   return;
} // end MeshData::computeNodalNormals()

//------------------------------------------------------------------------------
int MeshData::checkLagrangeMultiplierData()
{
   int err = 0;
   if (this->numberOfElements()>0)
   {
      if (!m_nodal_fields.m_is_node_gap_set ||
          !m_nodal_fields.m_is_node_pressure_set)
      {
         err = 1;
      }
   } // end if-non-null mesh
   return err; 
}
//------------------------------------------------------------------------------
int MeshData::checkPenaltyData( PenaltyEnforcementOptions& p_enfrc_options )
{
   int err = 0;
   if (this->numberOfElements()>0)
   {
      PenaltyConstraintType constraint_type = p_enfrc_options.constraint_type;
      // switch over penalty enforcement options and check for required data
      switch (constraint_type)
      {
         case KINEMATIC:
         {
            if (!m_element_data.isValidKinematicPenalty( p_enfrc_options ))
            {
               err = 1;
            }
            break;
         } // end KINEMATIC case

         case KINEMATIC_AND_RATE:
         {
            if (!m_element_data.isValidKinematicPenalty( p_enfrc_options ))
            {
               err = 1;
            }
            if (!m_element_data.isValidRatePenalty( p_enfrc_options ))
            {
               err = 1;
            }
            if (!m_nodal_fields.m_is_velocity_set)
            {
               SLIC_WARNING("Nodal velocities not set or null pointers; please set for " << 
                            "use with gap rate penalty enforcement.");
               err = 1;
            }
            break;
         } // end case KINEMATIC_AND_RATE
         default:
            // no-op, quiet compiler
            break;
      } // end switch over constraint types
   } // end if-non-null mesh

   return err;
} // end MeshData::checkPenaltyData()

//------------------------------------------------------------------------------
void MeshData::print(std::ostream& os) const
{
   const int num_verts = m_num_nodes;
   const int num_elem = numberOfElements();

   if(num_verts <= 0)
   {
      os << "{}";
      return;
   }

   os << "{\n";
   os << axom::fmt::format("  verts ({}) {{",num_verts);
   // positions
   os << axom::fmt::format("\n\tx: {}", axom::fmt::join(m_position[0].data(), m_position[0].data()+num_verts, ", "));
   os << axom::fmt::format("\n\ty: {}", axom::fmt::join(m_position[1].data(), m_position[1].data()+num_verts, ", "));
   if(m_dim == 3)
   {  
      os << axom::fmt::format("\n\tz: {}", axom::fmt::join(m_position[2].data(), m_position[2].data()+num_verts, ", "));
   }
   // contact response (force)
   if( !m_response.empty() )
   {
      os << axom::fmt::format("\n\tfx: {}", axom::fmt::join(m_response[0].data(), m_response[0].data()+num_verts, ", "));
      os << axom::fmt::format("\n\tfy: {}", axom::fmt::join(m_response[1].data(), m_response[1].data()+num_verts, ", "));
      if(m_dim == 3)
      {  
         os << axom::fmt::format("\n\tfz: {}", axom::fmt::join(m_response[2].data(), m_response[2].data()+num_verts, ", "));
      }
   }
   os << "\n  }";

   os << axom::fmt::format("\n  elems ({}) {{", num_elem);  
   
   if( !m_connectivity.empty() )
   {
      os << axom::fmt::format("\n\tconnectivity: {{ {} }}", axom::fmt::join(m_connectivity.data(), m_connectivity.data()+(num_elem*numberOfNodesPerElement()), ", "));
   }

   // normals
   if( !m_n.empty() )
   {
      os << "\n\tnx: " << m_n(0, 0);
      for (int e{1}; e < num_elem; ++e)
      {
         os << ", " <<  m_n(e, 0);
      }
      os << "\n\tny: " << m_n(0, 1);
      for (int e{1}; e < num_elem; ++e)
      {
         os << ", " <<  m_n(e, 1);
      }
      if(m_dim == 3)
      {  
         os << "\n\tnz: " << m_n(0, 2);
         for (int e{1}; e < num_elem; ++e)
         {
            os << ", " <<  m_n(e, 2);
         }
      }
   }
   os << "\n  }";

   os << "\n}";
}

//------------------------------------------------------------------------------
MeshData::Viewer::Viewer(MeshData& mesh)
: m_mesh_id( mesh.m_mesh_id )
, m_element_type( mesh.m_element_type )
, m_num_nodes( mesh.m_num_nodes )
, m_mem_space( mesh.m_mem_space )
, m_allocator_id( mesh.m_allocator_id )
, m_position( mesh.m_position )
, m_disp( mesh.m_disp )
, m_vel( mesh.m_vel )
, m_response( mesh.m_response )
, m_node_n( mesh.m_node_n )
, m_connectivity( mesh.m_connectivity )
, m_c( mesh.m_c )
, m_n( mesh.m_n )
, m_face_radius( mesh.m_face_radius )
, m_area( mesh.m_area )
, m_nodal_fields( mesh.m_nodal_fields )
, m_element_data( mesh.m_element_data )
{}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void MeshData::Viewer::getFaceCoords( IndexT face_id, RealT* coords ) const
{
  auto dim = spatialDimension();

  for (IndexT a{0}; a < numberOfNodesPerElement(); ++a)
  {
    IndexT node_id = getGlobalNodeId(face_id, a);
    for (int d{0}; d < dim; ++d)
    {
      coords[dim*a + d] = m_position[d][node_id];
    }
  }

  return;

}  // end MeshData::Viewer::getFaceCoords()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void MeshData::Viewer::getFaceVelocities( IndexT face_id, RealT* vels ) const
{
  auto dim = spatialDimension();

  for (IndexT a{0}; a < numberOfNodesPerElement(); ++a)
  {
    IndexT node_id = getGlobalNodeId(face_id, a);
    for (int d{0}; d < dim; ++d)
    {
      vels[dim*a + d] = m_vel[d][node_id];
    }
  }

  return;

}  // end MeshData::Viewer::getFaceVelocities()

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void MeshData::Viewer::getFaceNormal( int const face_id, RealT * nrml ) const
{
  for (int d{0}; d < spatialDimension(); ++d)
  {
    nrml[d] = m_n[d][face_id];
  }
  return;

} // end MeshData::getFaceNormal()

} // end tribol namespace

std::ostream& operator<<(std::ostream& os, const tribol::MeshData& md)
{
   md.print(os);
   return os;
}
