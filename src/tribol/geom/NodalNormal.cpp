// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "NodalNormal.hpp"

#include "tribol/common/Enzyme.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/utils/Math.hpp"

#ifdef TRIBOL_DEBUG
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#ifdef TRIBOL_USE_MPI
#include <mpi.h>
#endif
#endif

namespace tribol {

// forward declare free functions for enzyme.  these shouldn't be used outside the class, so no need to put them in the
// header.

/**
 * @brief Computes the normal direction at all the nodal coordinates of the element.
 *
 * @note This is a free function to allow for Enzyme support
 *
 * @param [in] x Nodal coordinates for the element (stored by nodes, i.e. [x0, x1, x2, y0, y1, y2, z0, z1, z2])
 * @param [in] xref Reference nodal coordinates for the element (at t = 0) (stored by nodes)
 * @param [out] n Unit vectors giving the normal direction for each node (stored by nodes)
 * @param [in] num_nodes_per_elem Number of nodes in the element
 */
void ElementEdgeAvgNodalNormal( const RealT* x, const RealT* xref, RealT* n, int num_nodes_per_elem );

/**
 * @brief Computes the normal direction and Jacobian at all the nodal coordinates of the element.
 *
 * @note Requires Tribol built with Enzyme support
 *
 * @param [in] x Nodal coordinates for the element (stored by nodes, i.e. [x0, x1, x2, y0, y1, y2, z0, z1, z2])
 * @param [in] xref Reference nodal coordinates for the element (at t = 0) (stored by nodes)
 * @param [out] n Unit vectors giving the normal direction for each node (stored by nodes)
 * @param [out] dndx Derivative of the unit normal vectors for each node (size = num_nodes_per_elem^2 x spatial dim^2)
 * @param [in] num_nodes_per_elem Number of nodes in the element
 */
void ElementEdgeAvgNodalNormalJacobian( const RealT* x, const RealT* xref, RealT* n, RealT* dndx,
                                        int num_nodes_per_elem );

#ifdef TRIBOL_DEBUG
namespace {

bool hasNaN( const RealT* data, int n )
{
  for ( int i = 0; i < n; ++i ) {
    if ( std::isnan( static_cast<double>( data[i] ) ) ) {
      return true;
    }
  }
  return false;
}

void normalize3( RealT& x, RealT& y, RealT& z )
{
  const RealT mag = std::sqrt( x * x + y * y + z * z );
  if ( mag > RealT( 0 ) ) {
    x /= mag;
    y /= mag;
    z /= mag;
  }
}

void computeElementNormalFromCoords( const RealT* x, int nnode, RealT& nx, RealT& ny, RealT& nz )
{
  // Use (v1 - v0) x (v2 - v0), assuming at least 3 vertices.
  const RealT v0x = x[0 * nnode + 0];
  const RealT v0y = x[1 * nnode + 0];
  const RealT v0z = x[2 * nnode + 0];
  const RealT e1x = x[0 * nnode + 1] - v0x;
  const RealT e1y = x[1 * nnode + 1] - v0y;
  const RealT e1z = x[2 * nnode + 1] - v0z;
  const RealT e2x = x[0 * nnode + 2] - v0x;
  const RealT e2y = x[1 * nnode + 2] - v0y;
  const RealT e2z = x[2 * nnode + 2] - v0z;

  nx = e1y * e2z - e1z * e2y;
  ny = e1z * e2x - e1x * e2z;
  nz = e1x * e2y - e1y * e2x;
  normalize3( nx, ny, nz );
}

bool referenceEdgeMagnitudesValid( const RealT* xref, int nnode, RealT tol )
{
  for ( int i{ 0 }; i < nnode; ++i ) {
    const int node0 = ( i - 1 + nnode ) % nnode;
    const int node1 = i;
    const int node2 = ( i + 1 ) % nnode;

    RealT e1[3] = { xref[0 * nnode + node2] - xref[0 * nnode + node1],
                    xref[1 * nnode + node2] - xref[1 * nnode + node1],
                    xref[2 * nnode + node2] - xref[2 * nnode + node1] };
    RealT e2[3] = { xref[0 * nnode + node0] - xref[0 * nnode + node1],
                    xref[1 * nnode + node0] - xref[1 * nnode + node1],
                    xref[2 * nnode + node0] - xref[2 * nnode + node1] };

    const RealT ni_ref[3] = { e1[1] * e2[2] - e1[2] * e2[1],
                              e1[2] * e2[0] - e1[0] * e2[2],
                              e1[0] * e2[1] - e1[1] * e2[0] };
    const RealT ni_mag = std::sqrt( ni_ref[0] * ni_ref[0] + ni_ref[1] * ni_ref[1] + ni_ref[2] * ni_ref[2] );
    if ( !std::isfinite( static_cast<double>( ni_mag ) ) || ni_mag < tol ) {
      return false;
    }
  }
  return true;
}

void writeEdgeAvgNormalElemToVTK( int elem_id, int nnode, const RealT* x, const RealT* n, const std::string& tag )
{
#ifdef TRIBOL_USE_MPI
  int rank = 0;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
#endif

  std::ostringstream base;
  base << "tribol_edge_avg_nodal_normal_nan_elem_" << elem_id;
  if ( !tag.empty() ) {
    base << "_" << tag;
  }
#ifdef TRIBOL_USE_MPI
  base << "_rank_" << rank;
#endif
  const std::string vtk_path = base.str() + ".vtk";

  std::ofstream out( vtk_path );
  if ( !out ) {
    SLIC_WARNING_ROOT( "ElementEdgeAvgNodalNormalJacobian() produced NaNs. Failed to write VTK debug file: " << vtk_path );
    return;
  }

  out << "# vtk DataFile Version 3.0\n";
  out << "Tribol EdgeAvgNodalNormal NaN debug (elem=" << elem_id << ")\n";
  out << "ASCII\n";
  out << "DATASET POLYDATA\n";
  out << "POINTS " << nnode << " float\n";
  for ( int i = 0; i < nnode; ++i ) {
    out << x[0 * nnode + i] << " " << x[1 * nnode + i] << " " << x[2 * nnode + i] << "\n";
  }

  out << "POLYGONS 1 " << ( 1 + nnode ) << "\n";
  out << nnode;
  for ( int i = 0; i < nnode; ++i ) {
    out << " " << i;
  }
  out << "\n";

  out << "POINT_DATA " << nnode << "\n";
  out << "VECTORS normals float\n";

  RealT enx = 0.0, eny = 0.0, enz = 0.0;
  computeElementNormalFromCoords( x, nnode, enx, eny, enz );
  for ( int i = 0; i < nnode; ++i ) {
    const RealT nx = n[0 * nnode + i];
    const RealT ny = n[1 * nnode + i];
    const RealT nz = n[2 * nnode + i];
    const bool n_has_nan = std::isnan( static_cast<double>( nx ) ) || std::isnan( static_cast<double>( ny ) ) ||
                           std::isnan( static_cast<double>( nz ) );
    if ( n_has_nan ) {
      out << enx << " " << eny << " " << enz << "\n";
    } else {
      out << nx << " " << ny << " " << nz << "\n";
    }
  }

  SLIC_WARNING_ROOT( "ElementEdgeAvgNodalNormalJacobian() produced NaNs. Wrote VTK debug file: " << vtk_path
                                                                                                 << " (open in ParaView; use Glyph on `normals`)." );
}

}  // namespace
#endif  // TRIBOL_DEBUG

void ElementAvgNodalNormal::Compute( MeshData& mesh, MethodData* jacobian_data )
{
  if ( mesh.numberOfElements() == 0 ) {
    return;
  }

  SLIC_ERROR_IF( jacobian_data != nullptr, "ElementAvgNodalNormal does not support computing Jacobian data." );

  mesh.allocateNodalNormals();

  auto mesh_view = mesh.getView();
  // check to make sure face normals have been computed with
  // a call to computeFaceData
  SLIC_ERROR_IF( !mesh_view.hasElementNormals(), "MeshData::computeNodalNormals: required face normals not computed." );

  // loop over elements
  for ( int i = 0; i < mesh_view.numberOfElements(); ++i ) {
    // loop over element nodes
    for ( int j = 0; j < mesh_view.numberOfNodesPerElement(); ++j ) {
      // SRW: note the connectivity array must be local to the mesh for indexing into
      // the mesh nodal normal array. If it is not, then nodeId will access some other
      // piece of memory and there may be a memory issue when numFaceNrmlsToNodes is deleted
      // at the end of this routine.
      int nodeId = mesh_view.getGlobalNodeId( i, j );
      for ( int d = 0; d < mesh_view.spatialDimension(); ++d ) {
        mesh_view.getNodalNormals()( d, nodeId ) += mesh_view.getElementNormals()( d, i );
      }

    }  // end loop over element nodes

  }  // end loop over elements

  // normalize the nodal normals
  if ( mesh_view.spatialDimension() == 3 ) {
    for ( int i = 0; i < mesh_view.numberOfNodes(); ++i ) {
      RealT mag = magnitude( mesh_view.getNodalNormals()( 0, i ), mesh_view.getNodalNormals()( 1, i ),
                             mesh_view.getNodalNormals()( 2, i ) );
      if ( mag >= 1.0e-15 ) {
        mesh_view.getNodalNormals()( 0, i ) /= mag;
        mesh_view.getNodalNormals()( 1, i ) /= mag;
        mesh_view.getNodalNormals()( 2, i ) /= mag;
      }
    }
  } else {
    for ( int i = 0; i < mesh_view.numberOfNodes(); ++i ) {
      RealT mag = magnitude( mesh_view.getNodalNormals()( 0, i ), mesh_view.getNodalNormals()( 1, i ) );
      if ( mag >= 1.0e-15 ) {
        mesh_view.getNodalNormals()( 0, i ) /= mag;
        mesh_view.getNodalNormals()( 1, i ) /= mag;
      }
    }
  }
}

void EdgeAvgNodalNormal::Compute( MeshData& mesh, MethodData* jacobian_data )
{
  SLIC_ERROR_ROOT_IF( mesh.spatialDimension() != 3, "3D mesh required for vertex averaged normal." );

  mesh.allocateNodalNormals();

  auto n0 = ArrayT<RealT, 2>( { 3, mesh.numberOfNodes() }, mesh.getAllocatorId() );
  n0.fill( 0.0 );

  if ( jacobian_data != nullptr ) {
    jacobian_data->reserveBlockJ( { BlockSpace::NONMORTAR }, mesh.numberOfElements() );
  }

  auto mesh_view = mesh.getView();

  SLIC_ERROR_IF( !mesh_view.hasReferencePosition(),
                 "Reference coordinates must be registered for vertex averaged normal." );

  auto num_nodes_per_elem = mesh_view.numberOfNodesPerElement();
#ifdef TRIBOL_DEBUG
  auto first_elem_for_node = ArrayT<int>( mesh.numberOfNodes(), mesh.getAllocatorId() );
  first_elem_for_node.fill( -1 );
  constexpr RealT ref_tol = 1.0e-15;
  const RealT qnan = std::numeric_limits<RealT>::quiet_NaN();
#endif
  for ( int e{ 0 }; e < mesh_view.numberOfElements(); ++e ) {
    RealT x[12];
    RealT xref[12];
    RealT n[12];
    for ( int i{ 0 }; i < num_nodes_per_elem; ++i ) {
      int node_id = mesh_view.getGlobalNodeId( e, i );
#ifdef TRIBOL_DEBUG
      if ( first_elem_for_node[node_id] < 0 ) {
        first_elem_for_node[node_id] = e;
      }
#endif
      for ( int d{ 0 }; d < 3; ++d ) {
        x[d * num_nodes_per_elem + i] = mesh_view.getPosition()[d][node_id];
        xref[d * num_nodes_per_elem + i] = mesh_view.getReferencePosition()[d][node_id];
        n[d * num_nodes_per_elem + i] = 0.0;
      }
    }
#ifdef TRIBOL_DEBUG
    const bool ref_ok = referenceEdgeMagnitudesValid( xref, num_nodes_per_elem, ref_tol );
    if ( !ref_ok ) {
      RealT nnan[12];
      for ( int i = 0; i < 3 * num_nodes_per_elem; ++i ) {
        nnan[i] = qnan;
      }
      writeEdgeAvgNormalElemToVTK( e, num_nodes_per_elem, xref, nnan, "bad_xref" );
    }
#endif
    if ( jacobian_data != nullptr ) {
      StackArray<DeviceArray2D<RealT>, 9> blockJ( 3 );
      blockJ( 0, 0 ) = DeviceArray2D<RealT>( num_nodes_per_elem * 3, num_nodes_per_elem * 3 );
      blockJ( 0, 0 ).fill( 0.0 );
#ifdef TRIBOL_DEBUG
      if ( ref_ok ) {
        ElementEdgeAvgNodalNormalJacobian( x, xref, n, blockJ( 0, 0 ).data(), num_nodes_per_elem );
        if ( hasNaN( n, num_nodes_per_elem * 3 ) ||
             hasNaN( blockJ( 0, 0 ).data(), static_cast<int>( blockJ( 0, 0 ).size() ) ) ) {
          writeEdgeAvgNormalElemToVTK( e, num_nodes_per_elem, x, n, "jacobian_nan" );
        }
      } else {
        for ( int i = 0; i < 3 * num_nodes_per_elem; ++i ) {
          n[i] = 0.0;
        }
      }
#else
      ElementEdgeAvgNodalNormalJacobian( x, xref, n, blockJ( 0, 0 ).data(), num_nodes_per_elem );
#endif
      jacobian_data->storeElemBlockJ( { e }, blockJ );
    } else {
#ifdef TRIBOL_DEBUG
      if ( ref_ok ) {
        ElementEdgeAvgNodalNormal( x, xref, n, num_nodes_per_elem );
        if ( hasNaN( n, num_nodes_per_elem * 3 ) ) {
          writeEdgeAvgNormalElemToVTK( e, num_nodes_per_elem, x, n, "normal_nan" );
        }
      } else {
        for ( int i = 0; i < 3 * num_nodes_per_elem; ++i ) {
          n[i] = 0.0;
        }
      }
#else
      ElementEdgeAvgNodalNormal( x, xref, n, num_nodes_per_elem );
#endif
    }
    // assemble normal contribution
    for ( int i{ 0 }; i < num_nodes_per_elem; ++i ) {
      int node_id = mesh_view.getGlobalNodeId( e, i );
      for ( int d{ 0 }; d < 3; ++d ) {
        mesh_view.getNodalNormals()( d, node_id ) += n[d * num_nodes_per_elem + i];
      }
    }
    // compute reference normal
#ifdef TRIBOL_DEBUG
    if ( ref_ok ) {
      ElementEdgeAvgNodalNormal( xref, xref, n, num_nodes_per_elem );
      if ( hasNaN( n, num_nodes_per_elem * 3 ) ) {
        writeEdgeAvgNormalElemToVTK( e, num_nodes_per_elem, xref, n, "ref_normal_nan" );
        for ( int i = 0; i < 3 * num_nodes_per_elem; ++i ) {
          n[i] = 0.0;
        }
      }
    } else {
      for ( int i = 0; i < 3 * num_nodes_per_elem; ++i ) {
        n[i] = 0.0;
      }
    }
#else
    ElementEdgeAvgNodalNormal( xref, xref, n, num_nodes_per_elem );
#endif
    // assemble reference normal contribution
    for ( int i{ 0 }; i < num_nodes_per_elem; ++i ) {
      int node_id = mesh_view.getGlobalNodeId( e, i );
      for ( int d{ 0 }; d < 3; ++d ) {
        n0( d, node_id ) += n[d * num_nodes_per_elem + i];
      }
    }
  }
  for ( int i{ 0 }; i < mesh_view.numberOfNodes(); ++i ) {
    // compute magnitude of reference normal (and store it in the first column)
    n0( 0, i ) = std::sqrt( n0( 0, i ) * n0( 0, i ) + n0( 1, i ) * n0( 1, i ) + n0( 2, i ) * n0( 2, i ) );
    // scale normals by reference normal magnitude
#ifdef TRIBOL_DEBUG
    const bool n0_ok = std::isfinite( static_cast<double>( n0( 0, i ) ) ) && n0( 0, i ) >= ref_tol;
    if ( !n0_ok ) {
      const int e_dbg = first_elem_for_node[i];
      if ( e_dbg >= 0 ) {
        RealT x_dbg[12];
        RealT n_dbg[12];
        for ( int a = 0; a < num_nodes_per_elem; ++a ) {
          const int node_id = mesh_view.getGlobalNodeId( e_dbg, a );
          for ( int d = 0; d < 3; ++d ) {
            x_dbg[d * num_nodes_per_elem + a] = mesh_view.getPosition()[d][node_id];
            n_dbg[d * num_nodes_per_elem + a] = mesh_view.getNodalNormals()( d, node_id );
          }
        }
        std::ostringstream tag;
        tag << "bad_n0_node_" << i;
        writeEdgeAvgNormalElemToVTK( e_dbg, num_nodes_per_elem, x_dbg, n_dbg, tag.str() );
      }
      continue;
    }
    if ( n0_ok ) {
#else
    if ( n0( 0, i ) >= 1.0e-15 ) {
#endif
      for ( int d{ 0 }; d < 3; ++d ) {
        mesh_view.getNodalNormals()( d, i ) /= n0( 0, i );
      }
    }
  }
  // scale Jacobian contributions
  if ( jacobian_data != nullptr ) {
    auto& blockJ_mats = jacobian_data->getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ),
                                                    static_cast<int>( BlockSpace::NONMORTAR ) );
    int e_ct = 0;
    for ( auto& blockJ_mat : blockJ_mats ) {
      for ( int i{ 0 }; i < num_nodes_per_elem; ++i ) {
        int node_id = mesh_view.getGlobalNodeId( e_ct, i );
#ifdef TRIBOL_DEBUG
        const RealT den = n0( 0, node_id );
        const bool den_ok = std::isfinite( static_cast<double>( den ) ) && den >= ref_tol;
        if ( !den_ok ) {
          RealT x_dbg[12];
          RealT n_dbg[12];
          for ( int a = 0; a < num_nodes_per_elem; ++a ) {
            const int nid = mesh_view.getGlobalNodeId( e_ct, a );
            for ( int d = 0; d < 3; ++d ) {
              x_dbg[d * num_nodes_per_elem + a] = mesh_view.getPosition()[d][nid];
              n_dbg[d * num_nodes_per_elem + a] = mesh_view.getNodalNormals()( d, nid );
            }
          }
          std::ostringstream tag;
          tag << "bad_n0_node_" << node_id << "_jacobian";
          writeEdgeAvgNormalElemToVTK( e_ct, num_nodes_per_elem, x_dbg, n_dbg, tag.str() );
          if ( !std::isfinite( static_cast<double>( den ) ) ) {
            for ( int d{ 0 }; d < 3; ++d ) {
              for ( int j{ 0 }; j < 3 * num_nodes_per_elem; ++j ) {
                blockJ_mat( d * num_nodes_per_elem + i, j ) = 0.0;
              }
            }
          }
          continue;
        }
        if ( den_ok ) {
#else
        if ( n0( 0, node_id ) >= 1.0e-15 ) {
#endif
          for ( int d{ 0 }; d < 3; ++d ) {
            for ( int j{ 0 }; j < 3 * num_nodes_per_elem; ++j ) {
              blockJ_mat( d * num_nodes_per_elem + i, j ) /= n0( 0, node_id );
            }
          }
        }
      }
      ++e_ct;
    }
  }
}

void ElementEdgeAvgNodalNormal( const RealT* x, const RealT* xref, RealT* n, int num_nodes_per_elem )
{
#ifdef TRIBOL_DEBUG
  const RealT qnan = std::numeric_limits<RealT>::quiet_NaN();
#endif
  for ( int i{ 0 }; i < num_nodes_per_elem; ++i ) {
    int node0 = ( i - 1 + num_nodes_per_elem ) % num_nodes_per_elem;
    int node1 = i;
    int node2 = ( i + 1 ) % num_nodes_per_elem;
    RealT e1[3] = { x[0 * num_nodes_per_elem + node2] - x[0 * num_nodes_per_elem + node1],
                    x[1 * num_nodes_per_elem + node2] - x[1 * num_nodes_per_elem + node1],
                    x[2 * num_nodes_per_elem + node2] - x[2 * num_nodes_per_elem + node1] };
    RealT e2[3] = { x[0 * num_nodes_per_elem + node0] - x[0 * num_nodes_per_elem + node1],
                    x[1 * num_nodes_per_elem + node0] - x[1 * num_nodes_per_elem + node1],
                    x[2 * num_nodes_per_elem + node0] - x[2 * num_nodes_per_elem + node1] };
    // normal vector = e1 x e2
    RealT ni[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
    // get magnitude in reference config
    e1[0] = xref[0 * num_nodes_per_elem + node2] - xref[0 * num_nodes_per_elem + node1];
    e1[1] = xref[1 * num_nodes_per_elem + node2] - xref[1 * num_nodes_per_elem + node1];
    e1[2] = xref[2 * num_nodes_per_elem + node2] - xref[2 * num_nodes_per_elem + node1];
    e2[0] = xref[0 * num_nodes_per_elem + node0] - xref[0 * num_nodes_per_elem + node1];
    e2[1] = xref[1 * num_nodes_per_elem + node0] - xref[1 * num_nodes_per_elem + node1];
    e2[2] = xref[2 * num_nodes_per_elem + node0] - xref[2 * num_nodes_per_elem + node1];
    RealT ni_ref[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
    RealT ni_mag = std::sqrt( ni_ref[0] * ni_ref[0] + ni_ref[1] * ni_ref[1] + ni_ref[2] * ni_ref[2] );
    bool mag_ok = ( ni_mag >= 1.0e-15 );
#ifdef TRIBOL_DEBUG
    mag_ok = mag_ok && std::isfinite( static_cast<double>( ni_mag ) );
#endif
    if ( mag_ok ) {
      for ( int d{ 0 }; d < 3; ++d ) {
        n[d * num_nodes_per_elem + i] = ni[d] / ni_mag;
      }
    } else {
#ifdef TRIBOL_DEBUG
      for ( int d{ 0 }; d < 3; ++d ) {
        n[d * num_nodes_per_elem + i] = qnan;
      }
#else
      for ( int d{ 0 }; d < 3; ++d ) {
        n[d * num_nodes_per_elem + i] = 0.0;
      }
#endif
    }
  }
}

void ElementEdgeAvgNodalNormalJacobian( [[maybe_unused]] const RealT* x, [[maybe_unused]] const RealT* xref,
                                        [[maybe_unused]] RealT* n, [[maybe_unused]] RealT* dndx,
                                        [[maybe_unused]] int num_nodes_per_elem )
{
#ifdef TRIBOL_USE_ENZYME
  RealT x_dot[12] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  for ( int i{ 0 }; i < num_nodes_per_elem * 3; ++i ) {
    x_dot[i] = 1.0;
    __enzyme_fwddiff<void>( (void*)ElementEdgeAvgNodalNormal, TRIBOL_ENZYME_DUP, x, x_dot, TRIBOL_ENZYME_CONST, xref,
                            TRIBOL_ENZYME_DUP, n, &dndx[num_nodes_per_elem * 3 * i], TRIBOL_ENZYME_CONST,
                            num_nodes_per_elem );
    x_dot[i] = 0.0;
  }
#else
  SLIC_ERROR( "ElementEdgeAvgNodalNormalJacobian requires Tribol built with Enzyme support." );
#endif
}

}  // namespace tribol
