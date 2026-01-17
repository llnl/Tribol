// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "shared/config.hpp"

#include <memory>

#include <gtest/gtest.h>

#include "mfem.hpp"
#include "redecomp/redecomp.hpp"

namespace redecomp {

#ifdef TRIBOL_USE_GPU_MPI

class GpuMpiTest : public testing::Test {
 protected:
  mfem::ParMesh par_mesh_;
  std::unique_ptr<mfem::H1_FECollection> h1_elems_;
  std::unique_ptr<mfem::ParFiniteElementSpace> par_vector_space_;
  std::unique_ptr<mfem::ParGridFunction> orig_;
  std::unique_ptr<mfem::ParGridFunction> final_;
  std::unique_ptr<RedecompMesh> redecomp_mesh_;
  std::unique_ptr<mfem::FiniteElementSpace> redecomp_vector_space_;
  std::unique_ptr<mfem::GridFunction> xfer_;

  void SetUp() override
  {
    // Create a simple mesh
    mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D( 4, 4, mfem::Element::QUADRILATERAL );
    par_mesh_ = mfem::ParMesh( MPI_COMM_WORLD, serial_mesh );

    // Setup fields
    h1_elems_ = std::make_unique<mfem::H1_FECollection>( 1, 2 );
    par_vector_space_ = std::make_unique<mfem::ParFiniteElementSpace>( &par_mesh_, h1_elems_.get(), 2 );
    orig_ = std::make_unique<mfem::ParGridFunction>( par_vector_space_.get() );
    final_ = std::make_unique<mfem::ParGridFunction>( par_vector_space_.get() );

    // Initialize orig with coordinates
    par_mesh_.GetNodes( *orig_ );

    // Move to device (important!)
    if ( mfem::Device::Allows( mfem::Backend::CUDA_MASK | mfem::Backend::HIP_MASK ) ) {
      orig_->UseDevice( true );
      orig_->Read( true );  // Allocate/Move to device
    }

    redecomp_mesh_ = std::make_unique<RedecompMesh>( par_mesh_ );
    redecomp_vector_space_ = std::make_unique<mfem::FiniteElementSpace>( redecomp_mesh_.get(), h1_elems_.get(), 2 );
    xfer_ = std::make_unique<mfem::GridFunction>( redecomp_vector_space_.get() );

    // xfer needs to be on device too
    if ( mfem::Device::Allows( mfem::Backend::CUDA_MASK | mfem::Backend::HIP_MASK ) ) {
      xfer_->UseDevice( true );
    }
  }
};

TEST_F( GpuMpiTest, VerifyTransfer )
{
  if ( !mfem::Device::GetGPUAwareMPI() ) {
    // If not enabled (maybe runtime doesn't support it), skip
    GTEST_SKIP() << "GPU Aware MPI not enabled or supported.";
  }

  // Double check that we are actually on device if expected
#if defined( TRIBOL_USE_CUDA ) || defined( TRIBOL_USE_HIP )
  ASSERT_TRUE( orig_->GetMemory().UseDevice() );
#endif

  auto transfer_map = RedecompTransfer();

  // Transfer to Serial (Redecomp)
  // orig_ is on device.
  transfer_map.TransferToSerial( *orig_, *xfer_ );

  // Transfer back
  transfer_map.TransferToParallel( *xfer_, *final_ );

  // Check error
  if ( mfem::Device::Allows( mfem::Backend::CUDA_MASK | mfem::Backend::HIP_MASK ) ) {
    final_->UseDevice( true );
  }

  // Verify values (L2 error)
  *final_ -= *orig_;
  double err = final_->Norml2() / orig_->Norml2();
  EXPECT_LT( err, 1e-12 );
}

TEST_F( GpuMpiTest, VerifyDeviceTransferStrict )
{
  if ( !mfem::Device::GetGPUAwareMPI() ) {
    GTEST_SKIP() << "GPU Aware MPI not enabled or supported.";
  }

  int rank;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  int size;
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  if ( size < 2 ) GTEST_SKIP();

  const auto& mpi = redecomp_mesh_->getMPIUtility();

  int N = 100;
  mfem::Vector send_vec( N );
  send_vec.UseDevice( true );
  send_vec = 1.0;  // Device = 1.0

  // Pollute Host to verify we are reading from Device
  {
    const double* h_read = send_vec.HostRead();  // Syncs to host (1.0)
    double* h_ptr = const_cast<double*>( h_read );
    for ( int i = 0; i < N; ++i ) h_ptr[i] = 2.0;  // Host = 2.0
  }
  // send_vec internal state: ValidDevice=true, ValidHost=true (but we hacked it).

  mfem::Vector recv_vec( N );

  // Transfer rank 0 -> rank 1
  if ( rank == 0 ) {
    mpi.Send( send_vec, 1, 999 );
  } else if ( rank == 1 ) {
    recv_vec = mpi.Recv( type<mfem::Vector>(), 0, 999, true );
  }

  MPI_Barrier( MPI_COMM_WORLD );

  if ( rank == 1 ) {
    // Check if we received 1.0 (from Device) or 2.0 (from Host)
    const double* res = recv_vec.HostRead();
    EXPECT_DOUBLE_EQ( res[0], 1.0 );
  }
}

#endif  // TRIBOL_USE_GPU_MPI

}  // namespace redecomp

// Main
#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

#if defined( TRIBOL_USE_CUDA )
  mfem::Device device( "cuda" );
#elif defined( TRIBOL_USE_HIP )
  mfem::Device device( "hip" );
#endif
  // Enable GPU Aware MPI
  mfem::Device::SetGPUAwareMPI( true );
  device.Print();

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
