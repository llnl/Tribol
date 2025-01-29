// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

//-----------------------------------------------------------------------------
//
// file: tribol_enzyme_jacobian.cpp
//
//-----------------------------------------------------------------------------

#include <iostream>

#include "redecomp/common/TypeDefs.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/config.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/Mortar.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/utils/Algorithm.hpp"

#include "gtest/gtest.h"

namespace tribol {

class EnzymeNormalTest : public testing::Test {
 protected:
  double delta_{ 1.0e-7 };
  void SetUp() override {}

  void FDCheck( RealT* x, MeshData& mesh )
  {
    VertexAvgNormal norm( true );
    norm.Compute( mesh );
    auto num_dofs = mesh.numberOfNodes() * mesh.spatialDimension();
    mfem::SparseMatrix dndx( num_dofs );
    auto& dndx_data = norm.getJacobianData();
    // get nonmortar/nonmortar contributions
    auto& elem_Js =
        dndx_data.getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ), static_cast<int>( BlockSpace::NONMORTAR ) );
    auto num_nodes_per_el = mesh.numberOfNodesPerElement();
    for ( int i{ 0 }; i < mesh.numberOfElements(); ++i ) {
      for ( int d1{ 0 }; d1 < mesh.spatialDimension(); ++d1 ) {
        for ( int j{ 0 }; j < num_nodes_per_el; ++j ) {
          auto global_idx_j = mesh.numberOfNodes() * d1 + mesh.getGlobalNodeId( i, j );
          auto local_idx_j = num_nodes_per_el * d1 + j;
          for ( int d2{ 0 }; d2 < mesh.spatialDimension(); ++d2 ) {
            for ( int k{ 0 }; k < num_nodes_per_el; ++k ) {
              auto global_idx_k = mesh.numberOfNodes() * d2 + mesh.getGlobalNodeId( i, k );
              auto local_idx_k = num_nodes_per_el * d2 + k;
              dndx.Add( global_idx_j, global_idx_k, elem_Js[i]( local_idx_j, local_idx_k ) );
            }
          }
        }
      }
    }
    dndx.Finalize();
    mfem::DenseMatrix dndx_enzyme;
    dndx.ToDenseMatrix( dndx_enzyme );

    mfem::DenseMatrix dndx_fd( num_dofs );
    auto mesh_view = mesh.getView();
    norm = VertexAvgNormal( false );
    Array2D<RealT> n_base = mesh_view.getNodalNormals();
    for ( int dx{ 0 }; dx < mesh.spatialDimension(); ++dx ) {
      for ( int nx{ 0 }; nx < mesh.numberOfNodes(); ++nx ) {
        auto x_idx = dx * mesh.numberOfNodes() + nx;
        x[dx * mesh.numberOfNodes() + nx] += delta_;
        norm.Compute( mesh );
        auto local_mesh_view = mesh.getView();
        for ( int dn{ 0 }; dn < mesh.spatialDimension(); ++dn ) {
          for ( int nn{ 0 }; nn < mesh.numberOfNodes(); ++nn ) {
            auto n_idx = dn * mesh.numberOfNodes() + nn;
            dndx_fd( n_idx, x_idx ) = ( local_mesh_view.getNodalNormals()( dn, nn ) - n_base( dn, nn ) ) / delta_;
          }
        }
        x[dx * mesh.numberOfNodes() + nx] -= delta_;
      }
    }

    // write deltas to screen
    std::cout << "dn/dx ------------------------------" << std::endl;
    for ( int i{ 0 }; i < num_dofs; ++i ) {
      for ( int j{ 0 }; j < num_dofs; ++j ) {
        auto diff = std::abs( dndx_enzyme( i, j ) - dndx_fd( i, j ) );
        if ( diff > delta_ ) {
          std::cout << "  (" << i << ", " << j << ") : Diff: " << diff
                    << "   Ratio: " << dndx_enzyme( i, j ) / dndx_fd( i, j ) << "   Enzyme: " << dndx_enzyme( i, j )
                    << "   FD: " << dndx_fd( i, j ) << std::endl;
        }
        EXPECT_NEAR( dndx_enzyme( i, j ), dndx_fd( i, j ), delta_ );
      }
    }
  }
};

TEST_F( EnzymeNormalTest, TwoElementsFlatNormalJacobian )
{
  // two elements flat
  double x[18] = { 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  double xref[18];
  for ( int i{ 0 }; i < 18; ++i ) {
    xref[i] = x[i];
  }
  int conn[8] = { 0, 1, 4, 3, 1, 2, 5, 4 };
  constexpr auto mesh_id = 0;
  registerMesh( mesh_id, 2, 6, conn, InterfaceElementType::LINEAR_QUAD, x, x + 6, x + 12 );
  registerNodalReferenceCoords( mesh_id, xref, xref + 6, xref + 12 );
  auto& mesh_data = MeshManager::getInstance().at( mesh_id );

  FDCheck( x, mesh_data );
}

}  // namespace tribol

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
