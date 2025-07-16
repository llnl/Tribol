// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <gtest/gtest.h>

#include "mfem.hpp"

#include "redecomp/MultiRedecomp.hpp"
#include "tribol/config.hpp"
#include "redecomp/redecomp.hpp"

namespace redecomp {

enum class MeshType
{
  TwoStarRotated,
  TwoHexSurface
};

class MultiTransferTest : public testing::TestWithParam<std::pair<MeshType, int>> {
 protected:
  struct MeshData {
    mfem::ParMesh par_mesh;
    std::unique_ptr<RedecompMesh> redecomp_mesh;
    std::unique_ptr<mfem::H1_FECollection> h1_elems;
    std::unique_ptr<mfem::ParFiniteElementSpace> par_vector_space;
    std::unique_ptr<mfem::ParGridFunction> orig_gridfn;
    std::unique_ptr<mfem::ParGridFunction> final_gridfn;
    std::unique_ptr<mfem::FiniteElementSpace> redecomp_vector_space;
    std::unique_ptr<mfem::GridFunction> xfer_gridfn;
    std::unique_ptr<mfem::QuadratureSpace> par_quad_space;
    std::unique_ptr<mfem::QuadratureFunction> orig_quadfn;
    std::unique_ptr<mfem::QuadratureFunction> final_quadfn;
    std::unique_ptr<mfem::QuadratureSpace> redecomp_quad_space;
    std::unique_ptr<mfem::QuadratureFunction> xfer_quadfn;
  };
  std::vector<MeshData> mesh_data_;
  void CreateMeshData( std::vector<mfem::Mesh>&& serial_meshes )
  {
    std::vector<mfem::ParMesh> par_meshes;
    par_meshes.reserve( serial_meshes.size() );
    for ( auto& serial_mesh : serial_meshes ) {
      par_meshes.emplace_back( MPI_COMM_WORLD, serial_mesh );
    }
    CreateMeshData( std::move( par_meshes ) );
  }
  void CreateMeshData( std::vector<mfem::ParMesh>&& par_meshes )
  {
    mesh_data_.clear();
    mesh_data_.reserve( par_meshes.size() );
    auto par_mesh_ptrs = std::vector<const mfem::ParMesh*>();
    par_mesh_ptrs.reserve( par_meshes.size() );
    for ( auto& par_mesh : par_meshes ) {
      auto mesh_data = MeshData();
      mesh_data.par_mesh = std::move( par_mesh );
      mesh_data_.emplace_back( std::move( mesh_data ) );
      par_mesh_ptrs.push_back( &mesh_data_.back().par_mesh );
    }
    auto multiredecomp = MultiRedecomp( mesh_data_[0].par_mesh.SpaceDimension(), mesh_data_[0].par_mesh.GetComm() );
    auto redecomp_meshes = multiredecomp.createRedecompMeshes( par_mesh_ptrs );
    for ( size_t i{ 0 }; i < par_meshes.size(); ++i ) {
      mesh_data_[i].redecomp_mesh = std::move( redecomp_meshes[i] );
    }
  }
  void CreateRotatedStarMesh()
  {
    // open meshes
    auto serial_meshes = std::vector<mfem::Mesh>();
    serial_meshes.reserve( 2 );
    std::string mesh_filename = std::string( TRIBOL_REPO_DIR ) + "/data/star.mesh";
    serial_meshes.emplace_back( mesh_filename.c_str(), 1, 1, true );
    serial_meshes.emplace_back( mesh_filename.c_str(), 1, 1, true );
    // rotate second mesh 30 degrees
    auto theta = 30.0;
    theta = theta * redecomp::pi / 180.0;
    auto R = axom::Array<double, 2>( 3, 3 );
    R( 0, 0 ) = cos( theta );
    R( 0, 1 ) = -sin( theta );
    R( 0, 2 ) = 0.0;
    R( 1, 0 ) = sin( theta );
    R( 1, 1 ) = cos( theta );
    R( 1, 2 ) = 0.0;
    R( 2, 0 ) = 0.0;
    R( 2, 1 ) = 0.0;
    R( 2, 2 ) = 1.0;
    auto dim = serial_meshes[1].Dimension();
    for ( int v{ 0 }; v < serial_meshes[1].GetNV(); ++v ) {
      auto tmp_vert = axom::Array<double>( dim, dim );
      auto vert = serial_meshes[1].GetVertex( v );
      for ( int i{ 0 }; i < dim; ++i ) {
        for ( int j{ 0 }; j < dim; ++j ) {
          tmp_vert[i] = tmp_vert[i] + R( i, j ) * vert[j];
        }
      }
      for ( int i{ 0 }; i < dim; ++i ) {
        vert[i] = tmp_vert[i];
      }
    }
    auto node_gf = serial_meshes[1].GetNodes();
    if ( node_gf ) {
      auto node_fes = node_gf->FESpace();
      auto vdim = node_fes->GetVDim();
      for ( int n{ 0 }; n < node_fes->GetNDofs(); ++n ) {
        auto tmp_node = axom::Array<double>( vdim, vdim );
        for ( int i{ 0 }; i < vdim; ++i ) {
          for ( int j{ 0 }; j < vdim; ++j ) {
            tmp_node[i] = tmp_node[i] + R( i, j ) * ( *node_gf )( node_fes->DofToVDof( n, j ) );
          }
        }
        for ( int i{ 0 }; i < vdim; ++i ) {
          ( *node_gf )( node_fes->DofToVDof( n, i ) ) = tmp_node[i];
        }
      }
    }
    // finish mesh setup
    CreateMeshData( std::move( serial_meshes ) );
  }
  void CreateTwoHexMesh()
  {
    std::string mesh_filename = std::string( TRIBOL_REPO_DIR ) + "/data/two_hex.mesh";
    auto serial_mesh = mfem::Mesh( mesh_filename.c_str(), 1, 1, true );
    serial_mesh.UniformRefinement();
    serial_mesh.UniformRefinement();
    auto par_mesh = mfem::ParMesh( MPI_COMM_WORLD, serial_mesh );
    auto par_submeshes = std::vector<mfem::ParMesh>();
    par_submeshes.reserve( 2 );
    auto bdry_attrib = mfem::Array<int>( 1 );
    bdry_attrib[0] = 4;
    par_submeshes.push_back( mfem::ParSubMesh::CreateFromBoundary( par_mesh, bdry_attrib ) );
    bdry_attrib[0] = 5;
    par_submeshes.push_back( mfem::ParSubMesh::CreateFromBoundary( par_mesh, bdry_attrib ) );
    CreateMeshData( std::move( par_submeshes ) );
  }
  void CreateGridFnData()
  {
    for ( auto& mesh_data : mesh_data_ ) {
      auto dim = mesh_data.par_mesh.Dimension();
      mesh_data.h1_elems = std::make_unique<mfem::H1_FECollection>( GetParam().second, dim );
      mesh_data.par_vector_space =
          std::make_unique<mfem::ParFiniteElementSpace>( &mesh_data.par_mesh, mesh_data.h1_elems.get(), dim );
      mesh_data.orig_gridfn = std::make_unique<mfem::ParGridFunction>( mesh_data.par_vector_space.get() );
      mesh_data.final_gridfn = std::make_unique<mfem::ParGridFunction>( mesh_data.par_vector_space.get() );
      if ( GetParam().second > 1 ) {
        mesh_data.par_mesh.SetNodalGridFunction( mesh_data.orig_gridfn.get(), false );
      } else {
        mesh_data.par_mesh.GetNodes( *mesh_data.orig_gridfn );
      }
      mesh_data.redecomp_vector_space = std::make_unique<mfem::FiniteElementSpace>(
          mesh_data.redecomp_mesh.get(), mesh_data.h1_elems.get(), mesh_data.redecomp_mesh->Dimension() );
      mesh_data.xfer_gridfn = std::make_unique<mfem::GridFunction>( mesh_data.redecomp_vector_space.get() );
    }
  }
  void CreateQuadFnData()
  {
    for ( auto& mesh_data : mesh_data_ ) {
      mesh_data.par_quad_space = std::make_unique<mfem::QuadratureSpace>( &mesh_data.par_mesh, 0 );
      mesh_data.orig_quadfn = std::make_unique<mfem::QuadratureFunction>( mesh_data.par_quad_space.get() );
      mesh_data.final_quadfn = std::make_unique<mfem::QuadratureFunction>( mesh_data.par_quad_space.get() );
      for ( int e{ 0 }; e < mesh_data.par_mesh.GetNE(); ++e ) {
        auto quad_val = mfem::Vector();
        mesh_data.orig_quadfn->GetValues( e, quad_val );
        for ( int i{ 0 }; i < quad_val.Size(); ++i ) {
          quad_val[i] = static_cast<double>( mesh_data_[i].par_mesh.GetGlobalElementNum( e ) );
        }
      }
      mesh_data.redecomp_quad_space = std::make_unique<mfem::QuadratureSpace>( mesh_data.redecomp_mesh.get(), 0 );
      mesh_data.xfer_quadfn = std::make_unique<mfem::QuadratureFunction>( mesh_data.redecomp_quad_space.get() );
    }
  }
  double CalcGridFnl2Error()
  {
    auto total_error = 0.0;
    for ( auto& mesh_data : mesh_data_ ) {
      if ( mesh_data.final_gridfn->FESpace()->GetNDofs() > 0 ) {
        *mesh_data.final_gridfn -= *mesh_data.orig_gridfn;
        total_error += mesh_data.final_gridfn->Norml2() / mesh_data.orig_gridfn->Norml2();
      }
    }
    MPI_Allreduce( MPI_IN_PLACE, &total_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    return total_error;
  }
  double CalcQuadFnl2Error()
  {
    auto total_error = 0.0;
    for ( auto& mesh_data : mesh_data_ ) {
      if ( mesh_data.final_quadfn->GetSpace()->GetSize() > 0 ) {
        *mesh_data.final_quadfn -= *mesh_data.orig_quadfn;
        total_error += mesh_data.final_quadfn->Norml2() / mesh_data.orig_quadfn->Norml2();
      }
    }
    MPI_Allreduce( MPI_IN_PLACE, &total_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    return total_error;
  }
  void CreateMesh( MeshType mesh_type )
  {
    switch ( mesh_type ) {
      case MeshType::TwoStarRotated:
        CreateRotatedStarMesh();
        break;
      case MeshType::TwoHexSurface:
        CreateTwoHexMesh();
        break;
    }
  }
};

TEST_P( MultiTransferTest, element_gridfn )
{
  CreateMesh( GetParam().first );
  CreateGridFnData();
  auto transfer_map = RedecompTransfer();
  for ( auto& mesh_data : mesh_data_ ) {
    transfer_map.TransferToSerial( *mesh_data.orig_gridfn, *mesh_data.xfer_gridfn );
    transfer_map.TransferToParallel( *mesh_data.xfer_gridfn, *mesh_data.final_gridfn );
  }
  EXPECT_LT( CalcGridFnl2Error(), 1.0e-13 );
  MPI_Barrier( MPI_COMM_WORLD );
}

TEST_P( MultiTransferTest, element_quadfn )
{
  CreateMesh( GetParam().first );
  CreateQuadFnData();
  auto transfer_map = RedecompTransfer();
  for ( auto& mesh_data : mesh_data_ ) {
    transfer_map.TransferToSerial( *mesh_data.orig_quadfn, *mesh_data.xfer_quadfn );
    transfer_map.TransferToParallel( *mesh_data.xfer_quadfn, *mesh_data.final_quadfn );
  }
  EXPECT_LT( CalcQuadFnl2Error(), 1.0e-13 );
  MPI_Barrier( MPI_COMM_WORLD );
}

TEST_P( MultiTransferTest, node_gridfn )
{
  CreateMesh( GetParam().first );
  CreateGridFnData();
  for ( auto& mesh_data : mesh_data_ ) {
    auto transfer_map = RedecompTransfer( *mesh_data.par_vector_space, *mesh_data.redecomp_vector_space );
    transfer_map.TransferToSerial( *mesh_data.orig_gridfn, *mesh_data.xfer_gridfn );
    transfer_map.TransferToParallel( *mesh_data.xfer_gridfn, *mesh_data.final_gridfn );
  }
  EXPECT_LT( CalcGridFnl2Error(), 1.0e-13 );
  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( redecomp, MultiTransferTest,
                          testing::Values( std::make_pair( MeshType::TwoStarRotated, 1 ),
                                           std::make_pair( MeshType::TwoStarRotated, 3 ),
                                           std::make_pair( MeshType::TwoHexSurface, 1 ),
                                           std::make_pair( MeshType::TwoHexSurface, 3 ) ) );

}  // namespace redecomp

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
