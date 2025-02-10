// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/types.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/geom/ContactPlaneManager.hpp"
#include "tribol/mesh/MeshManager.hpp"
#include "tribol/mesh/CouplingSchemeManager.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/utils/ContactPlaneOutput.hpp"

// AXOM includes
#include "axom/config.hpp"
#include "axom/slic.hpp"
#include "axom/slam.hpp"
#include "axom/fmt.hpp"

// C++ includes
#include <iomanip>
#include <sstream>
#include <fstream>

namespace fmt = axom::fmt;

namespace tribol {
/*!
 * \brief free function to return vtk element type
 */
int GetVtkElementId( const InterfaceElementType type )
{
  switch ( type ) {
    case LINEAR_EDGE:
      return 3;  // vtk 2-node line
      break;
    case LINEAR_TRIANGLE:
      return 5;  // vtk 3-node triangle
      break;
    case LINEAR_QUAD:
      return 9;  // vtk 4-node quad
      break;
    case LINEAR_HEX:
      return 12;  // vtk 8-node hex
      break;
    default:
      SLIC_ERROR( "Unsupported element type in Tribol's VTK output" );
      break;
  }  // end switch( type )
  return 0;
}  // end GetVtkElementId()

//------------------------------------------------------------------------------
void WriteContactPlaneMeshToVtk( const std::string& dir, const VisType v_type, const integer csId,
                                 const integer meshId1, const integer meshId2, const integer dim, const integer cycle,
                                 const real time )
{
  ContactPlaneManager& cpMgr = ContactPlaneManager::getInstance();
  MeshManager& meshManager = MeshManager::getInstance();
  MeshData& mesh1 = meshManager.GetMeshInstance( meshId1 );
  MeshData& mesh2 = meshManager.GetMeshInstance( meshId2 );
  CouplingSchemeManager& csManager = CouplingSchemeManager::getInstance();
  CouplingScheme* couplingScheme = csManager.getCoupling( csId );

  int nranks = 1;
  int rank = -1;
#ifdef TRIBOL_USE_MPI
  // TODO: use parameters.problem_comm ?
  MPI_Comm_rank( TRIBOL_COMM_WORLD, &rank );
  MPI_Comm_size( TRIBOL_COMM_WORLD, &nranks );
#endif

  /////////////////////////////////////////
  //                                     //
  // Write contact faces and/or overlaps //
  //                                     //
  /////////////////////////////////////////
  if ( !couplingScheme->nullMeshes() ) {
    int cpSize = cpMgr.size();
    bool overlaps{ false };
    bool faces{ false };

    switch ( v_type ) {
      case VIS_FACES:
        faces = true;
        break;
      case VIS_OVERLAPS:
        overlaps = true;
        break;
      case VIS_MESH_AND_OVERLAPS:
        overlaps = true;
        break;
      case VIS_FACES_AND_OVERLAPS:
        faces = true;
        overlaps = true;
        break;
      case VIS_MESH_FACES_AND_OVERLAPS:
        faces = true;
        overlaps = true;
        break;
      default:
        // Can this be output on root? SRW
        overlaps = true;  // set default for now; refactoring
        SLIC_INFO( "WriteInterfaceMeshToVtk: visualization type not supported."
                   << " Printing overlaps only." );
        break;
    }  // end switch( v_type )

    if ( faces ) {
      // Compose file name and open file
      std::string name = ( nranks > 1 ) ? fmt::format( "y_cntct_faces_r{:04}_{:07}.vtk", rank, cycle )
                                        : fmt::format( "y_cntct_faces_{:07}.vtk", cycle );
      std::string f_name = axom::utilities::filesystem::joinPath( dir, name );

      std::ofstream faces;
      faces.setf( std::ios::scientific );
      faces.open( f_name.c_str() );

      // write face .vtk first
      faces << "# vtk DataFile Version 3.0\n";
      faces << "vtk output\n";
      faces << "ASCII\n";
      faces << "DATASET UNSTRUCTURED_GRID\n";

      // Add the cycle and time to FieldData
      faces << "FIELD FieldData 3\n";
      faces << "TIME 1 1 double\n";
      faces << time << "\n";
      faces << "CYCLE 1 1 int\n";
      faces << cycle << "\n";
      faces << "COUPLING_SCHEME 1 1 int\n";
      faces << csId << "\n";

      // count the number of face points for all contact planes
      int numPoints = 0;
      for ( int i = 0; i < cpSize; ++i ) {
        numPoints += mesh1.m_numNodesPerCell + mesh2.m_numNodesPerCell;
      }  // end i-loop over contact planes

      // output the number of points
      faces << "POINTS " << numPoints << " float\n";

      // loop over all contact planes and output the face coordinates
      for ( int i = 0; i < cpSize; ++i ) {
        // if interpenOverlap, print interpenetrating portions of each face.
        if ( cpMgr.m_interpenOverlap[i] ) {
          for ( int j = 0; j < cpMgr.m_numInterpenPoly1Vert[i]; ++j ) {
            fmt::print( faces, "{} {} {}\n", cpMgr.m_interpenG1X[i][j], cpMgr.m_interpenG1Y[i][j],
                        dim == 3 ? cpMgr.m_interpenG1Z[i][j] : 0. );
          }

          for ( int j = 0; j < cpMgr.m_numInterpenPoly2Vert[i]; ++j ) {
            fmt::print( faces, "{} {} {}\n", cpMgr.m_interpenG2X[i][j], cpMgr.m_interpenG2Y[i][j],
                        dim == 3 ? cpMgr.m_interpenG2Z[i][j] : 0. );
          }
        }  // end if-cpMrg.m_interpenOverlap[i]

        else  // print the current configuration faces
        {
          for ( int j = 0; j < mesh1.m_numNodesPerCell; ++j ) {
            const int nodeId = mesh1.getFaceNodeId( cpMgr.m_fId1[i], j );
            fmt::print( faces, "{} {} {}\n", mesh1.m_positionX[nodeId], mesh1.m_positionY[nodeId],
                        dim == 3 ? mesh1.m_positionZ[nodeId] : 0. );
          }

          for ( int j = 0; j < mesh2.m_numNodesPerCell; ++j ) {
            const int nodeId = mesh2.getFaceNodeId( cpMgr.m_fId2[i], j );
            fmt::print( faces, "{} {} {}\n", mesh2.m_positionX[nodeId], mesh2.m_positionY[nodeId],
                        dim == 3 ? mesh2.m_positionZ[nodeId] : 0. );
          }
        }  // end else
      }    // end i-loop over contact planes outputting face coordinates

      // output polygon connectivity. Number of points is the number of polygon
      // vertices + the polygon vertices for each contact plane
      fmt::print( faces, "CELLS {} {}\n", 2 * cpSize, numPoints + ( 2 * cpSize ) );

      using RSet = axom::slam::RangeSet<int, int>;
      int connIter = 0;  // connectivity iterator

      // loop over contact plane instances and print current configuration
      // face polygon connectivity
      const int nNodes1 = mesh1.m_numNodesPerCell;
      const int nNodes2 = mesh2.m_numNodesPerCell;
      for ( int i = 0; i < cpSize; ++i ) {
        fmt::print( faces, "{} {}\n", nNodes1, fmt::join( RSet( connIter, connIter + nNodes1 ), " " ) );
        connIter += nNodes1;

        fmt::print( faces, "{} {}\n", nNodes2, fmt::join( RSet( connIter, connIter + nNodes2 ), " " ) );
        connIter += nNodes2;
      }
      faces << std::endl;

      // print cell types as VTK integer IDs
      {
        fmt::print( faces, "CELL_TYPES {}\n", 2 * cpSize );
        const int vtkid1 = dim == 3 ? 7 : 3;  // 7 is VTK_POLYGON; 3 is VTK_LINE
        const int vtkid2 = dim == 3 ? 7 : 3;

        for ( int i = 0; i < cpSize; ++i ) {
          fmt::print( faces, "{} {} ", vtkid1, vtkid2 );
        }
        faces << std::endl;
      }

      // print the contact face areas
      fmt::print( faces, "CELL_DATA {}\n", 2 * cpSize );
      fmt::print( faces, "SCALARS {} {}\n", "face_area", "float" );
      fmt::print( faces, "LOOKUP_TABLE default\n" );

      for ( int i = 0; i < cpSize; ++i ) {
        // print face areas
        fmt::print( faces, "{} {} ", mesh1.m_area[cpMgr.m_fId1[i]], mesh2.m_area[cpMgr.m_fId2[i]] );
      }  // end i-loop over contact planes
      faces << std::endl;
      faces.close();

    }  // end if-faces

    // open contact plane output file. For now we just output the overlaps
    if ( overlaps ) {
      // Compose file name and open file
      std::string name = ( nranks > 1 ) ? fmt::format( "z_cntct_overlap_r{:04}_{:07}.vtk", rank, cycle )
                                        : fmt::format( "z_cntct_overlap_{:07}.vtk", cycle );
      std::string f_name = axom::utilities::filesystem::joinPath( dir, name );

      std::ofstream overlap;
      overlap.setf( std::ios::scientific );
      overlap.open( f_name.c_str() );

      // write contact plane data
      overlap << "# vtk DataFile Version 3.0\n";
      overlap << "vtk output\n";
      overlap << "ASCII\n";
      overlap << "DATASET UNSTRUCTURED_GRID\n";

      // Add the cycle and time to FieldData
      overlap << "FIELD FieldData 3\n";
      overlap << "TIME 1 1 double\n";
      overlap << time << "\n";
      overlap << "CYCLE 1 1 int\n";
      overlap << cycle << "\n";
      overlap << "COUPLING_SCHEME 1 1 int\n";
      overlap << csId << "\n";

      // count the total number of vertices for all contact plane instances.
      int numPoints = 0;
      for ( int k = 0; k < cpSize; ++k ) {
        // add the number of overlap vertices
        numPoints += cpMgr.m_numPolyVert[k];
      }  // end k-loop over contact planes

      fmt::print( overlap, "POINTS {} float\n", numPoints );

      // loop over contact plane instances and output polygon vertices
      for ( int k = 0; k < cpSize; ++k ) {
        // output the overlap polygon. Whether interpenetrating overlap or full
        // overlap the vertex coordinates are stored in cp.m_polyX,Y,Z
        for ( int i = 0; i < cpMgr.m_numPolyVert[k]; ++i ) {
          if ( dim == 3 ) {
            fmt::print( overlap, "{} {} {}\n", cpMgr.m_polyX[k][i], cpMgr.m_polyY[k][i], cpMgr.m_polyZ[k][i] );
          } else {
            fmt::print( overlap, "{} {} {}\n", cpMgr.m_segX[k][i], cpMgr.m_segY[k][i], 0. );
          }
        }  // end i-loop over overlap vertices
      }    // end i-loop over contact planes for overlap output

      // define the polygons
      int numPolygons = cpSize;  // one overlap per contact plane object

      fmt::print( overlap, "CELLS {} {}\n", numPolygons, ( numPoints + numPolygons ) );

      // output the overlap connectivity
      using RSet = axom::slam::RangeSet<int, int>;
      int k = 0;
      for ( int i = 0; i < cpSize; ++i ) {
        const int nVerts = cpMgr.m_numPolyVert[i];
        fmt::print( overlap, "{} {}\n", nVerts, fmt::join( RSet( k, k + nVerts ), " " ) );
        k += nVerts;
      }

      // print cell types as VTK integer IDs
      {
        fmt::print( overlap, "CELL_TYPES {}\n", cpSize );
        const int vtkid = dim == 3 ? 7 : 3;  // 7 is VTK_POLYGON; 3 is VTK_LINE
        for ( int i = 0; i < cpSize; ++i ) {
          fmt::print( overlap, "{} ", vtkid );
        }
        overlap << std::endl;
      }

      /// Output scalar fields
      fmt::print( overlap, "CELL_DATA {}\n", numPolygons );

      // print the contact plane area
      {
        fmt::print( overlap, "SCALARS {} {}\n", "overlap_area", "float" );
        fmt::print( overlap, "LOOKUP_TABLE default\n" );
        for ( int i = 0; i < cpSize; ++i ) {
          fmt::print( overlap, "{} ", cpMgr.m_interpenOverlap[i] ? cpMgr.m_interpenArea[i] : cpMgr.m_area[i] );
        }
        overlap << std::endl;
      }

      // print the contact plane pressure scalar data
      {
        fmt::print( overlap, "SCALARS {} {}\n", "overlap_pressure", "float" );
        fmt::print( overlap, "LOOKUP_TABLE default\n" );
        fmt::print( overlap, "{}", fmt::join( cpMgr.m_pressure.data(), cpMgr.m_pressure.data() + cpSize, " " ) );
        overlap << std::endl;
      }

      // close file
      overlap.close();

    }  // end if-overlaps

  }  // end write faces and/or overlaps for non-null meshes

  //////////////////////////////////////////////////////////////
  //                                                          //
  // Write registered contact meshes for this coupling scheme //
  //                                                          //
  //////////////////////////////////////////////////////////////
  if ( !couplingScheme->nullMeshes() ) {
    switch ( v_type ) {
      case VIS_MESH:
        break;
      case VIS_MESH_AND_OVERLAPS:
        break;
      case VIS_MESH_FACES_AND_OVERLAPS:
        break;
      default:
        // no mesh output
        return;
    }  // end switch( v_type )

    std::string name = ( nranks > 1 ) ? fmt::format( "mesh_intrfc_cs{:02}_r{:04}_{:07}.vtk", csId, rank, cycle )
                                      : fmt::format( "mesh_intrfc_cs{:02}_{:07}.vtk", csId, cycle );
    std::string f_name = axom::utilities::filesystem::joinPath( dir, name );

    std::ofstream mesh;
    mesh.setf( std::ios::scientific );
    mesh.open( f_name.c_str() );

    mesh << "# vtk DataFile Version 3.0\n";
    mesh << "vtk output\n";
    mesh << "ASCII\n";
    mesh << "DATASET UNSTRUCTURED_GRID\n";

    // Add the cycle and time to FieldData
    mesh << "FIELD FieldData 3\n";
    mesh << "TIME 1 1 double\n";
    mesh << time << "\n";
    mesh << "CYCLE 1 1 int\n";
    mesh << cycle << "\n";
    mesh << "COUPLING_SCHEME 1 1 int\n";
    mesh << csId << "\n";

    int numTotalNodes = mesh1.m_lengthNodalData + mesh2.m_lengthNodalData;
    mesh << "POINTS " << numTotalNodes << " float\n";

    for ( int i = 0; i < mesh1.m_lengthNodalData; ++i ) {
      fmt::print( mesh, "{} {} {}\n", mesh1.m_positionX[i], mesh1.m_positionY[i],
                  dim == 3 ? mesh1.m_positionZ[i] : 0. );
    }

    for ( int i = 0; i < mesh2.m_lengthNodalData; ++i ) {
      fmt::print( mesh, "{} {} {}\n", mesh2.m_positionX[i], mesh2.m_positionY[i],
                  dim == 3 ? mesh2.m_positionZ[i] : 0. );
    }

    // print mesh element connectivity
    int numTotalElements = mesh1.m_numCells + mesh2.m_numCells;
    int numSurfaceNodes = mesh1.m_numCells * mesh1.m_numNodesPerCell + mesh2.m_numCells * mesh2.m_numNodesPerCell;

    fmt::print( mesh, "CELLS {} {}\n", numTotalElements, numTotalElements + numSurfaceNodes );

    for ( int i = 0; i < mesh1.m_numCells; ++i ) {
      mesh << mesh1.m_numNodesPerCell;
      for ( int a = 0; a < mesh1.m_numNodesPerCell; ++a ) {
        int id = mesh1.m_numNodesPerCell * i + a;
        mesh << " " << mesh1.m_connectivity[id];
      }  // end a-loop over nodes
      mesh << std::endl;
    }  // end i-loop over cells

    const int m2_offset = mesh1.m_lengthNodalData;
    for ( int i = 0; i < mesh2.m_numCells; ++i ) {
      mesh << mesh2.m_numNodesPerCell;
      for ( int a = 0; a < mesh2.m_numNodesPerCell; ++a ) {
        int id = mesh2.m_numNodesPerCell * i + a;
        mesh << " " << m2_offset + mesh2.m_connectivity[id];
      }  // end a-loop over nodes
      mesh << std::endl;
    }  // end i-loop over cells

    // specify integer id for each cell type.
    // For 4-node quad, id = 9.
    const int mesh1_element_id = GetVtkElementId( mesh1.m_elementType );
    const int mesh2_element_id = GetVtkElementId( mesh2.m_elementType );

    if ( mesh1_element_id <= 0 || mesh2_element_id <= 0 ) {
      SLIC_ERROR( "WriteInterfaceMeshToVtk(): "
                  << "element type not supported by vtk." );
    }

    mesh << "CELL_TYPES " << numTotalElements << std::endl;
    for ( int i = 0; i < mesh1.m_numCells; ++i ) {
      fmt::print( mesh, "{} ", mesh1_element_id );
    }
    for ( int i = 0; i < mesh2.m_numCells; ++i ) {
      fmt::print( mesh, "{} ", mesh2_element_id );
    }
    mesh << std::endl;

    // Add a field to label each face with its source mesh
    mesh << "CELL_DATA " << numTotalElements << std::endl;
    mesh << "SCALARS mesh_id int 1" << std::endl;
    mesh << "LOOKUP_TABLE default" << std::endl;
    for ( int i = 0; i < mesh1.m_numCells; ++i ) {
      fmt::print( mesh, "{} ", meshId1 );
    }

    for ( int i = 0; i < mesh2.m_numCells; ++i ) {
      fmt::print( mesh, "{} ", meshId2 );
    }
    mesh << std::endl;

    mesh.close();
  }  // end write mesh for non-null meshes

  return;

}  // end WriteInterfaceMeshToVtk()

//------------------------------------------------------------------------------

}  // namespace tribol
