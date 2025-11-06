// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <gtest/gtest.h>

#include "mfem.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

#include "tribol/config.hpp"

#include "redecomp/redecomp.hpp"

namespace redecomp {

class TransferTest : public testing::TestWithParam<std::pair<std::string, int>> {
 protected:
  mfem::ParMesh par_mesh_;
  std::unique_ptr<mfem::H1_FECollection> h1_elems_;
  std::unique_ptr<mfem::ParFiniteElementSpace> par_vector_space_;
  std::unique_ptr<mfem::ParGridFunction> orig_;
  std::unique_ptr<mfem::ParGridFunction> final_;
  std::unique_ptr<RedecompMesh> redecomp_mesh_;
  std::unique_ptr<mfem::FiniteElementSpace> redecomp_vector_space_;
  std::unique_ptr<mfem::GridFunction> xfer_;
  std::unique_ptr<mfem::QuadratureSpace> par_quad_space_;
  std::unique_ptr<mfem::QuadratureFunction> orig_quad_fn_;
  std::unique_ptr<mfem::QuadratureSpace> redecomp_quad_space_;
  std::unique_ptr<mfem::QuadratureFunction> xfer_quad_fn_;
  std::unique_ptr<mfem::QuadratureFunction> final_quad_fn_;
  void SetUp() override
  {
    auto mesh_file = GetParam().first;
    auto fe_order = GetParam().second;
    std::string mesh_filename = std::string( TRIBOL_REPO_DIR ) + mesh_file;
    auto serial_mesh = mfem::Mesh( mesh_filename.c_str(), 1, 1, true );
    auto dim = serial_mesh.Dimension();
    serial_mesh.UniformRefinement();
    serial_mesh.UniformRefinement();
    par_mesh_ = mfem::ParMesh( MPI_COMM_WORLD, serial_mesh );

    // store nodal coordinate as a GridFunction
    h1_elems_ = std::make_unique<mfem::H1_FECollection>( fe_order, dim );
    par_vector_space_ = std::make_unique<mfem::ParFiniteElementSpace>( &par_mesh_, h1_elems_.get(), dim );
    orig_ = std::make_unique<mfem::ParGridFunction>( par_vector_space_.get() );
    final_ = std::make_unique<mfem::ParGridFunction>( par_vector_space_.get() );
    if ( fe_order > 1 ) {
      par_mesh_.SetNodalGridFunction( orig_.get(), false );
    } else {
      par_mesh_.GetNodes( *orig_ );
    }
    redecomp_mesh_ = std::make_unique<RedecompMesh>( par_mesh_ );
    redecomp_vector_space_ = std::make_unique<mfem::FiniteElementSpace>( redecomp_mesh_.get(), h1_elems_.get(), dim );
    xfer_ = std::make_unique<mfem::GridFunction>( redecomp_vector_space_.get() );

    // store global element number as a QuadratureFunction
    par_quad_space_ = std::make_unique<mfem::QuadratureSpace>( &par_mesh_, 0 );
    orig_quad_fn_ = std::make_unique<mfem::QuadratureFunction>( par_quad_space_.get() );
    for ( int e{ 0 }; e < par_mesh_.GetNE(); ++e ) {
      auto quad_val = mfem::Vector();
      orig_quad_fn_->GetValues( e, quad_val );
      for ( int i{ 0 }; i < quad_val.Size(); ++i ) {
        quad_val[i] = static_cast<double>( par_mesh_.GetGlobalElementNum( e ) );
      }
    }
    redecomp_quad_space_ = std::make_unique<mfem::QuadratureSpace>( redecomp_mesh_.get(), 0 );
    xfer_quad_fn_ = std::make_unique<mfem::QuadratureFunction>( redecomp_quad_space_.get() );
    final_quad_fn_ = std::make_unique<mfem::QuadratureFunction>( par_quad_space_.get() );
  }
  template <typename T>
  double Calcl2Error( const T& orig, T& final )
  {
    final -= orig;
    return final.Norml2() / orig.Norml2();
  }
};

TEST_P( TransferTest, element_gridfn_transfer )
{
  // test grid function transfer
  auto transfer_map = RedecompTransfer();
  transfer_map.TransferToSerial( *orig_, *xfer_ );
  transfer_map.TransferToParallel( *xfer_, *final_ );
  EXPECT_LT( Calcl2Error( *orig_, *final_ ), 1.0e-13 );

  MPI_Barrier( MPI_COMM_WORLD );
}

TEST_P( TransferTest, element_quadfn_transfer )
{
  // test quadrature function transfer
  auto transfer_map = RedecompTransfer();
  transfer_map.TransferToSerial( *orig_quad_fn_, *xfer_quad_fn_ );
  transfer_map.TransferToParallel( *xfer_quad_fn_, *final_quad_fn_ );
  EXPECT_LT( Calcl2Error( *orig_quad_fn_, *final_quad_fn_ ), 1.0e-13 );

  MPI_Barrier( MPI_COMM_WORLD );
}

TEST_P( TransferTest, node_gridfn_transfer )
{
  auto transfer_map = RedecompTransfer( *par_vector_space_, *redecomp_vector_space_ );
  transfer_map.TransferToSerial( *orig_, *xfer_ );
  transfer_map.TransferToParallel( *xfer_, *final_ );
  EXPECT_LT( Calcl2Error( *orig_, *final_ ), 1.0e-13 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( redecomp, TransferTest,
                          testing::Values( std::make_pair( "/data/star.mesh", 1 ),
                                           std::make_pair( "/data/star.mesh", 3 ),
                                           std::make_pair( "/data/two_hex.mesh", 1 ),
                                           std::make_pair( "/data/two_hex.mesh", 3 ) ) );

}  // namespace redecomp

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

#ifdef TRIBOL_ENABLE_CUDA
  mfem::Device device( "cuda" );
#endif

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
