// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/config.hpp"

#include <gtest/gtest.h>

#include "mfem.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

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

  void CheckReplicatedDualCollapse( const RedecompTransfer& transfer, const mfem::ParGridFunction& parent_primal,
                                    mfem::GridFunction& redecomp_primal ) const
  {
    // Transferring a known parent field creates matching values on every complete redecomp copy. Reusing those values
    // as a synthetic assembled response makes the expected collapsed dual unambiguous.
    transfer.TransferToSerial( parent_primal, redecomp_primal );
    mfem::LinearForm redecomp_dual( redecomp_primal.FESpace() );
    redecomp_dual = redecomp_primal;

    mfem::ParLinearForm parent_dual( parent_primal.ParFESpace() );
    transfer.CollapseReplicatedDualToParallel( redecomp_dual, parent_dual );

    mfem::Vector collapsed_true_dofs( parent_primal.ParFESpace()->GetTrueVSize() );
    mfem::Vector expected_true_dofs( parent_primal.ParFESpace()->GetTrueVSize() );

    // MFEM sums shared parent DOFs during assembly. Matching the original true values proves that the collapse retained
    // one physical response instead of one response per redecomp copy.
    parent_dual.ParallelAssemble( collapsed_true_dofs );
    parent_primal.GetTrueDofs( expected_true_dofs );
    collapsed_true_dofs -= expected_true_dofs;
    EXPECT_LT( collapsed_true_dofs.Norml2() / std::max( 1.0, expected_true_dofs.Norml2() ), 1.0e-13 );
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

TEST_P( TransferTest, replicated_dual_collapse )
{
  // Exercise the cached nodal maps used when several fields share the same parent and redecomp spaces.
  auto transfer_map = RedecompTransfer( *par_vector_space_, *redecomp_vector_space_ );
  CheckReplicatedDualCollapse( transfer_map, *orig_, *xfer_ );
}

TEST_P( TransferTest, element_replicated_dual_collapse )
{
  // Exercise direct element-list transfer independently of the cached nodal-map implementation.
  auto transfer_map = RedecompTransfer();
  CheckReplicatedDualCollapse( transfer_map, *orig_, *xfer_ );
}

TEST_P( TransferTest, scalar_replicated_dual_collapse )
{
  // Vector tests cover component-expanded DOF maps. A scalar field confirms that collapse also handles vector dimension
  // one without relying on component ordering.
  mfem::ParFiniteElementSpace parent_scalar_space( &par_mesh_, h1_elems_.get() );
  mfem::FiniteElementSpace redecomp_scalar_space( redecomp_mesh_.get(), h1_elems_.get() );
  mfem::ParGridFunction parent_scalar( &parent_scalar_space );
  mfem::GridFunction redecomp_scalar( &redecomp_scalar_space );
  parent_scalar = 2.5;

  auto transfer_map = RedecompTransfer( parent_scalar_space, redecomp_scalar_space );
  CheckReplicatedDualCollapse( transfer_map, parent_scalar, redecomp_scalar );
}

INSTANTIATE_TEST_SUITE_P( redecomp, TransferTest,
                          testing::Values( std::make_pair( "/data/star.mesh", 1 ),
                                           std::make_pair( "/data/star.mesh", 3 ),
                                           std::make_pair( "/data/two_hex.mesh", 1 ),
                                           std::make_pair( "/data/two_hex.mesh", 3 ) ) );

TEST( RedecompTransfer, empty_redecomp_ranks )
{
  int comm_size = 0;
  MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
  if ( comm_size < 4 ) {
    GTEST_SKIP() << "Empty-rank coverage runs with four MPI ranks.";
  }

  auto serial_mesh = mfem::Mesh::MakeCartesian2D( 4, 1, mfem::Element::QUADRILATERAL );
  mfem::ParMesh parent_mesh( MPI_COMM_WORLD, serial_mesh );
  mfem::H1_FECollection collection( 1, parent_mesh.Dimension() );
  mfem::ParFiniteElementSpace parent_space( &parent_mesh, &collection );
  mfem::ParGridFunction parent_primal( &parent_space );
  parent_primal = 3.0;

  // Restrict redecomp ownership to one rank so the remaining ranks exercise empty serial meshes and transfer maps.
  RedecompMesh redecomp_mesh( parent_mesh, RedecompMesh::RCB, 1 );
  mfem::FiniteElementSpace redecomp_space( &redecomp_mesh, &collection );
  mfem::GridFunction redecomp_primal( &redecomp_space );
  RedecompTransfer transfer( parent_space, redecomp_space );
  transfer.TransferToSerial( parent_primal, redecomp_primal );

  mfem::LinearForm redecomp_dual( &redecomp_space );
  redecomp_dual = redecomp_primal;
  mfem::ParLinearForm parent_dual( &parent_space );
  transfer.CollapseReplicatedDualToParallel( redecomp_dual, parent_dual );

  mfem::Vector collapsed_true_dofs( parent_space.GetTrueVSize() );
  mfem::Vector expected_true_dofs( parent_space.GetTrueVSize() );

  // Ranks with empty redecomp meshes must still participate in collapse without changing the response supplied by the
  // sole populated redecomp rank.
  parent_dual.ParallelAssemble( collapsed_true_dofs );
  parent_primal.GetTrueDofs( expected_true_dofs );
  collapsed_true_dofs -= expected_true_dofs;
  EXPECT_LT( collapsed_true_dofs.Norml2(), 1.0e-13 );
}

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

#ifdef TRIBOL_USE_CUDA
  mfem::Device device( "cuda" );
#elif defined( TRIBOL_USE_HIP )
  mfem::Device device( "hip" );
#endif

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
