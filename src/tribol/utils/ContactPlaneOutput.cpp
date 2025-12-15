// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/utils/ContactPlaneOutput.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

// AXOM includes
#include "axom/fmt.hpp"
#include "axom/slam.hpp"
#include "axom/slic.hpp"

// C++ includes
#include <fstream>

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
void WriteContactPlaneMeshToVtk( const std::string& dir, const VisType v_type, const IndexT cs_id,
                                 const IndexT mesh_id1, const IndexT mesh_id2, const int dim, const int cycle,
                                 const RealT time )
{
  CouplingScheme* couplingScheme = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF( !couplingScheme, "No coupling scheme registered with given cs_id." );
  const auto mesh1 = couplingScheme->getMesh1().getView();
  const auto mesh2 = couplingScheme->getMesh2().getView();

  int nranks = 1;
  int rank = -1;
#ifdef TRIBOL_USE_MPI
  MPI_Comm_rank( couplingScheme->getProblemComm(), &rank );
  MPI_Comm_size( couplingScheme->getProblemComm(), &nranks );
#endif

  /////////////////////////////////////////
  //                                     //
  // Write contact faces and/or overlaps //
  //                                     //
  /////////////////////////////////////////
  if ( !couplingScheme->nullMeshes() ) {
    int cpSize = couplingScheme->getNumActivePairs();
    bool overlaps{ false };
    bool faces{ false };
    bool meshes{ false };

    switch ( v_type ) {
      case VIS_NONE:
        break;
      case VIS_FACES:
        faces = true;
        break;
      case VIS_OVERLAPS:
        overlaps = true;
        break;
      case VIS_MESH_AND_OVERLAPS:
        overlaps = true;
        meshes = true;
        break;
      case VIS_FACES_AND_OVERLAPS:
        faces = true;
        overlaps = true;
        break;
      case VIS_MESH_FACES_AND_OVERLAPS:
        faces = true;
        overlaps = true;
        meshes = true;
        break;
      default:
        // Can this be output on root? SRW
        overlaps = true;  // set default for now; refactoring
        SLIC_INFO( "WriteInterfaceMeshToVtk: visualization type not supported." << " Printing overlaps only." );
        break;
    }  // end switch( v_type )

    if ( faces && cpSize > 0 ) {
      // Compose file name and open file
      std::string name = ( nranks > 1 ) ? axom::fmt::format( "y_cntct_faces_r{:04}_{:07}.vtk", rank, cycle )
                                        : axom::fmt::format( "y_cntct_faces_{:07}.vtk", cycle );
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
      faces << cs_id << "\n";

      // count the number of face points for all contact planes
      int numPoints = 0;
      for ( int i = 0; i < cpSize; ++i ) {
        auto& cp_base = couplingScheme->getContactPlanePair( i );
        if ( cp_base.m_fullOverlap ) {
          numPoints += mesh1.numberOfNodesPerElement() + mesh2.numberOfNodesPerElement();
        } else {
          auto& cp = couplingScheme->getCompGeom().getCommonPlane( i );
          numPoints += cp.m_numInterpenPoly1Vert + cp.m_numInterpenPoly2Vert;
        }
      }  // end i-loop over contact planes

      // output the number of points
      faces << "POINTS " << numPoints << " float\n";

      // loop over all contact planes and output the face coordinates
      for ( int i = 0; i < cpSize; ++i ) {
        auto& cp_base = couplingScheme->getContactPlanePair( i );
        // print interpenetrating portion of each face if not full overlap
        if ( !cp_base.m_fullOverlap ) {  // note this is just for common plane pairs, but mortar will always have full
                                         // overlap
          // get common plane derived type for interpen prints
          auto& cp = couplingScheme->getCompGeom().getCommonPlane( i );
          for ( int j = 0; j < cp.m_numInterpenPoly1Vert; ++j ) {
            axom::fmt::print( faces, "{} {} {}\n", cp.m_interpenG1X[j], cp.m_interpenG1Y[j],
                              dim == 3 ? cp.m_interpenG1Z[j] : 0. );
          }

          for ( int j = 0; j < cp.m_numInterpenPoly2Vert; ++j ) {
            axom::fmt::print( faces, "{} {} {}\n", cp.m_interpenG2X[j], cp.m_interpenG2Y[j],
                              dim == 3 ? cp.m_interpenG2Z[j] : 0. );
          }
        }  // end if-!m_fullOverlap

        else  // print the current configuration faces
        {
          for ( int j = 0; j < mesh1.numberOfNodesPerElement(); ++j ) {
            const int nodeId = mesh1.getGlobalNodeId( cp_base.getCpElementId1(), j );
            axom::fmt::print( faces, "{} {} {}\n", mesh1.getPosition()[0][nodeId], mesh1.getPosition()[1][nodeId],
                              dim == 3 ? mesh1.getPosition()[2][nodeId] : 0. );
          }

          for ( int j = 0; j < mesh2.numberOfNodesPerElement(); ++j ) {
            const int nodeId = mesh2.getGlobalNodeId( cp_base.getCpElementId2(), j );
            axom::fmt::print( faces, "{} {} {}\n", mesh2.getPosition()[0][nodeId], mesh2.getPosition()[1][nodeId],
                              dim == 3 ? mesh2.getPosition()[2][nodeId] : 0. );
          }
        }  // end else
      }  // end i-loop over contact planes outputting face coordinates

      // output face polygon connectivity. Number of points is the number of face polygon
      // vertices + an index for each face for each contact plane
      axom::fmt::print( faces, "CELLS {} {}\n", 2 * cpSize, numPoints + ( 2 * cpSize ) );

      using RSet = axom::slam::RangeSet<int, int>;
      int connIter = 0;  // connectivity iterator

      // loop over contact plane instances and print current configuration
      // face polygon connectivity
      int nNodes1, nNodes2;
      for ( int i = 0; i < cpSize; ++i ) {
        auto& cp_base = couplingScheme->getContactPlanePair( i );
        if ( cp_base.m_fullOverlap ) {
          nNodes1 = mesh1.numberOfNodesPerElement();
          nNodes2 = mesh2.numberOfNodesPerElement();
        } else {
          auto& cp = couplingScheme->getCompGeom().getCommonPlane( i );
          nNodes1 = cp.m_numInterpenPoly1Vert;
          nNodes2 = cp.m_numInterpenPoly2Vert;
        }
        axom::fmt::print( faces, "{} {}\n", nNodes1, axom::fmt::join( RSet( connIter, connIter + nNodes1 ), " " ) );
        connIter += nNodes1;

        axom::fmt::print( faces, "{} {}\n", nNodes2, axom::fmt::join( RSet( connIter, connIter + nNodes2 ), " " ) );
        connIter += nNodes2;
      }
      faces << std::endl;

      // print cell types as VTK integer IDs
      {
        axom::fmt::print( faces, "CELL_TYPES {}\n", 2 * cpSize );
        const int vtkid1 = dim == 3 ? 7 : 3;  // 7 is VTK_POLYGON; 3 is VTK_LINE
        const int vtkid2 = dim == 3 ? 7 : 3;

        for ( int i = 0; i < cpSize; ++i ) {
          axom::fmt::print( faces, "{} {} ", vtkid1, vtkid2 );
        }
        faces << std::endl;
      }
      faces << std::endl;
      faces.close();

    }  // end if-faces

    // open contact plane output file. For now we just output the overlaps
    if ( overlaps && cpSize > 0 ) {
      // Compose file name and open file
      std::string name = ( nranks > 1 ) ? axom::fmt::format( "z_cntct_overlap_r{:04}_{:07}.vtk", rank, cycle )
                                        : axom::fmt::format( "z_cntct_overlap_{:07}.vtk", cycle );
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
      overlap << cs_id << "\n";

      // count the total number of vertices for all contact plane instances.
      int numPoints = 2 * cpSize;  // 2D overlap is 2 vertices per contact plane
      if ( dim == 3 ) {
        numPoints = 0;
        for ( int k = 0; k < cpSize; ++k ) {
          auto& cp = couplingScheme->getContactPlanePair( k );
          // add the number of overlap vertices
          numPoints += cp.m_numPolyVert;
        }  // end k-loop over contact planes
      }

      axom::fmt::print( overlap, "POINTS {} float\n", numPoints );

      // loop over contact plane instances and output polygon vertices
      for ( int k = 0; k < cpSize; ++k ) {
        auto& cp = couplingScheme->getContactPlanePair( k );
        // output the overlap polygon. Whether interpenetrating overlap or full
        // overlap the vertex coordinates are stored in cp.m_polyX,Y,Z
        if ( dim == 2 ) {
          for ( int i = 0; i < 2; ++i ) {
            axom::fmt::print( overlap, "{} {} {}\n", cp.m_polyX[i], cp.m_polyY[i], 0. );
          }  // end i-loop over overlap vertices
        } else {
          for ( int i = 0; i < cp.m_numPolyVert; ++i ) {
            axom::fmt::print( overlap, "{} {} {}\n", cp.m_polyX[i], cp.m_polyY[i], cp.m_polyZ[i] );
          }  // end i-loop over overlap vertices
        }
      }  // end i-loop over contact planes for overlap output

      // define the polygons
      int numPolygons = cpSize;  // one overlap per contact plane object

      axom::fmt::print( overlap, "CELLS {} {}\n", numPolygons, ( numPoints + numPolygons ) );

      // output the overlap connectivity
      using RSet = axom::slam::RangeSet<int, int>;
      int k = 0;
      for ( int i = 0; i < cpSize; ++i ) {
        int nVerts = 2;
        if ( dim == 3 ) {
          auto& cp = couplingScheme->getContactPlanePair( i );
          nVerts = cp.m_numPolyVert;
        }
        axom::fmt::print( overlap, "{} {}\n", nVerts, axom::fmt::join( RSet( k, k + nVerts ), " " ) );
        k += nVerts;
      }

      // print cell types as VTK int IDs
      {
        axom::fmt::print( overlap, "CELL_TYPES {}\n", cpSize );
        const int vtkid = dim == 3 ? 7 : 3;  // 7 is VTK_POLYGON; 3 is VTK_LINE
        for ( int i = 0; i < cpSize; ++i ) {
          axom::fmt::print( overlap, "{} ", vtkid );
        }
        overlap << std::endl;
      }

      /// Output scalar fields
      axom::fmt::print( overlap, "CELL_DATA {}\n", numPolygons );

      // print the contact plane area
      {
        axom::fmt::print( overlap, "SCALARS {} {}\n", "overlap_area", "float" );
        axom::fmt::print( overlap, "LOOKUP_TABLE default\n" );
        for ( int i = 0; i < cpSize; ++i ) {
          auto& cp = couplingScheme->getContactPlanePair( i );
          axom::fmt::print( overlap, "{} ", cp.m_area );
        }
        overlap << std::endl;
      }

      // print the contact plane pressure scalar data for common plane overlaps
      {
        if ( couplingScheme->getContactMethod() == COMMON_PLANE ) {
          axom::fmt::print( overlap, "SCALARS {} {}\n", "overlap_pressure", "float" );
          axom::fmt::print( overlap, "LOOKUP_TABLE default\n" );
          for ( int i = 0; i < cpSize; ++i ) {
            auto& cp = couplingScheme->getCompGeom().getCommonPlane( i );
            axom::fmt::print( overlap, "{} ", cp.m_pressure );
          }
          overlap << std::endl;
        }
      }

      // close file
      overlap.close();

    }  // end if-overlaps

    //////////////////////////////////////////////////////////////
    //                                                          //
    // Write registered contact meshes for this coupling scheme //
    //                                                          //
    //////////////////////////////////////////////////////////////
    if ( meshes ) {
      std::string name = ( nranks > 1 )
                             ? axom::fmt::format( "mesh_intrfc_cs{:02}_r{:04}_{:07}.vtk", cs_id, rank, cycle )
                             : axom::fmt::format( "mesh_intrfc_cs{:02}_{:07}.vtk", cs_id, cycle );
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
      mesh << cs_id << "\n";

      int numTotalNodes = mesh1.numberOfNodes() + mesh2.numberOfNodes();
      mesh << "POINTS " << numTotalNodes << " float\n";

      for ( int i = 0; i < mesh1.numberOfNodes(); ++i ) {
        axom::fmt::print( mesh, "{} {} {}\n", mesh1.getPosition()[0][i], mesh1.getPosition()[1][i],
                          dim == 3 ? mesh1.getPosition()[2][i] : 0. );
      }

      for ( int i = 0; i < mesh2.numberOfNodes(); ++i ) {
        axom::fmt::print( mesh, "{} {} {}\n", mesh2.getPosition()[0][i], mesh2.getPosition()[1][i],
                          dim == 3 ? mesh2.getPosition()[2][i] : 0. );
      }

      // print mesh element connectivity
      int numTotalElements = mesh1.numberOfElements() + mesh2.numberOfElements();
      int numSurfaceNodes = mesh1.numberOfElements() * mesh1.numberOfNodesPerElement() +
                            mesh2.numberOfElements() * mesh2.numberOfNodesPerElement();

      axom::fmt::print( mesh, "CELLS {} {}\n", numTotalElements, numTotalElements + numSurfaceNodes );

      for ( int i = 0; i < mesh1.numberOfElements(); ++i ) {
        mesh << mesh1.numberOfNodesPerElement();
        for ( int a = 0; a < mesh1.numberOfNodesPerElement(); ++a ) {
          mesh << " " << mesh1.getConnectivity()( i, a );
        }  // end a-loop over nodes
        mesh << std::endl;
      }  // end i-loop over cells

      const int m2_offset = mesh1.numberOfNodes();
      for ( int i = 0; i < mesh2.numberOfElements(); ++i ) {
        mesh << mesh2.numberOfNodesPerElement();
        for ( int a = 0; a < mesh2.numberOfNodesPerElement(); ++a ) {
          mesh << " " << m2_offset + mesh2.getConnectivity()( i, a );
        }  // end a-loop over nodes
        mesh << std::endl;
      }  // end i-loop over cells

      // specify integer id for each cell type.
      // For 4-node quad, id = 9.
      const int mesh1_element_id = GetVtkElementId( mesh1.getElementType() );
      const int mesh2_element_id = GetVtkElementId( mesh2.getElementType() );

      if ( mesh1_element_id <= 0 || mesh2_element_id <= 0 ) {
        SLIC_ERROR( "WriteInterfaceMeshToVtk(): " << "element type not supported by vtk." );
      }

      mesh << "CELL_TYPES " << numTotalElements << std::endl;
      for ( int i = 0; i < mesh1.numberOfElements(); ++i ) {
        axom::fmt::print( mesh, "{} ", mesh1_element_id );
      }
      for ( int i = 0; i < mesh2.numberOfElements(); ++i ) {
        axom::fmt::print( mesh, "{} ", mesh2_element_id );
      }
      mesh << std::endl;

      // Add a field to label each face with its source mesh
      mesh << "CELL_DATA " << numTotalElements << std::endl;
      mesh << "SCALARS mesh_id int 1" << std::endl;
      mesh << "LOOKUP_TABLE default" << std::endl;
      for ( int i = 0; i < mesh1.numberOfElements(); ++i ) {
        axom::fmt::print( mesh, "{} ", mesh_id1 );
      }

      for ( int i = 0; i < mesh2.numberOfElements(); ++i ) {
        axom::fmt::print( mesh, "{} ", mesh_id2 );
      }
      mesh << std::endl;

      mesh.close();

    }  // end if (meshes)

  }  // end write data for non-null meshes

  return;

}  // end WriteInterfaceMeshToVtk()

//------------------------------------------------------------------------------

}  // namespace tribol
