// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <set>
#include <vector>
#include <chrono>

#include "mfem.hpp"
#include "axom/core.hpp"
#include "axom/slic.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  int rank, n_ranks;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &n_ranks );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();
#endif

  axom::slic::SimpleLogger logger;
  axom::slic::setIsRoot( rank == 0 );

  int ref_levels = 2;
  axom::CLI::App app{ "jacobian_transfer_comparison" };
  app.add_option( "-r,--refine", ref_levels, "Number of times to refine the mesh uniformly." )->capture_default_str();
  CLI11_PARSE( app, argc, argv );

  // 1. Setup mesh (2 blocks)
  int nel_per_dir = std::pow( 2, ref_levels );
  auto elem_type = mfem::Element::HEXAHEDRON;
  auto mortar_attrs = std::set<int>( { 4 } );
  auto nonmortar_attrs = std::set<int>( { 5 } );

  mfem::ParMesh mesh = shared::ParMeshBuilder(
      MPI_COMM_WORLD,
      shared::MeshBuilder::Unify( { shared::MeshBuilder::CubeMesh( nel_per_dir, nel_per_dir, nel_per_dir, elem_type )
                                        .updateBdrAttrib( 4, 7 )
                                        .updateBdrAttrib( 6, 4 ),
                                    shared::MeshBuilder::CubeMesh( nel_per_dir, nel_per_dir, nel_per_dir, elem_type )
                                        .translate( { 0.0, 0.0, 0.99 } )
                                        .updateBdrAttrib( 8, 5 ) } ) );

  int dim = mesh.SpaceDimension();
  mfem::H1_FECollection fec( 1, dim );
  mfem::ParFiniteElementSpace fespace( &mesh, &fec, dim );
  mfem::ParGridFunction coords( &fespace );
  mesh.GetNodes( coords );

  // 2. Register Tribol Coupling Scheme
  int cs_id = 0;
  tribol::registerMfemCouplingScheme( cs_id, 0, 1, mesh, coords, mortar_attrs, nonmortar_attrs,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::SINGLE_MORTAR,
                                      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );
  tribol::setMPIComm( cs_id, MPI_COMM_WORLD );
  tribol::updateMfemParallelDecomposition();

  // 3. Setup MfemJacobianData
  auto& cs_manager = tribol::CouplingSchemeManager::getInstance();
  auto* cs = cs_manager.findData( cs_id );
  if ( !cs->hasMfemJacobianData() ) {
    cs->setMfemJacobianData( std::make_unique<tribol::MfemJacobianData>(
        *cs->getMfemMeshData(), *cs->getMfemSubmeshData(), cs->getContactMethod() ) );
  }
  auto* jac_data = cs->getMfemJacobianData();
  jac_data->UpdateJacobianXfer();

  // 4. Synthesize Jacobian data for comparison
  // We'll create some dummy element matrices for elements on Mesh 1 (Mortar)
  int ne1 = cs->getMfemMeshData()->GetMesh1NE();
  int num_nodes_per_elem = 4;  // Quad faces
  int num_dofs_per_elem = num_nodes_per_elem * dim;

  // New format: ComputedElementData
  tribol::ComputedElementData new_data;
  new_data.row_space = tribol::BlockSpace::MORTAR;
  new_data.col_space = tribol::BlockSpace::MORTAR;
  new_data.jacobian_data.resize( ne1 * num_dofs_per_elem * num_dofs_per_elem );
  new_data.jacobian_offsets.resize( ne1 );
  new_data.row_elem_ids.resize( ne1 );
  new_data.col_elem_ids.resize( ne1 );

  // Prepare MethodData for OLD path comparison
  if ( cs->getMethodData() == nullptr ) {
    cs->allocateMethodData();
  }
  auto* method_data = cs->getMethodData();
  SLIC_ASSERT( method_data != nullptr );

  tribol::ArrayT<tribol::BlockSpace> spaces( { tribol::BlockSpace::MORTAR } );
  method_data->reserveBlockJ( std::move( spaces ), ne1 );

  for ( int e = 0; e < ne1; ++e ) {
    new_data.row_elem_ids[e] = e;
    new_data.col_elem_ids[e] = e;
    new_data.jacobian_offsets[e] = e * num_dofs_per_elem * num_dofs_per_elem;

    tribol::StackArray<tribol::DeviceArray2D<tribol::RealT>, 9> blockJ;
    blockJ[0] = tribol::DeviceArray2D<tribol::RealT>( num_dofs_per_elem, num_dofs_per_elem );

    for ( int i = 0; i < num_dofs_per_elem; ++i ) {
      for ( int j = 0; j < num_dofs_per_elem; ++j ) {
        double val = static_cast<double>( e + i + j );
        blockJ[0]( i, j ) = val;
        new_data.jacobian_data[new_data.jacobian_offsets[e] + i + j * num_dofs_per_elem] = val;
      }
    }
    tribol::ArrayT<int> ids( { e } );
    method_data->storeElemBlockJ( std::move( ids ), blockJ );
  }

  // 5. Time and assemble using Old Method
  auto start_old = std::chrono::high_resolution_clock::now();

  // Simulated OLD path logic
  auto xfer = cs->getMfemJacobianData()->GetMfemBlockJacobian( *method_data, { { 0, tribol::BlockSpace::MORTAR } },
                                                               { { 0, tribol::BlockSpace::MORTAR } } );

  auto end_old = std::chrono::high_resolution_clock::now();

  // 6. Time and assemble using New Method
  // We must call this on ALL ranks collectively.
  std::vector<tribol::ComputedElementData> contribs_vec;
  if ( ne1 > 0 ) {
    contribs_vec.push_back( std::move( new_data ) );
  }

  auto start_new = std::chrono::high_resolution_clock::now();
  auto par_J_new = jac_data->GetMfemJacobian( contribs_vec );
  auto end_new = std::chrono::high_resolution_clock::now();

  // 7. Verify match
  // We need to extract the block from BlockOperator if we used GetMfemBlockJacobian
  auto* old_hypre = dynamic_cast<mfem::HypreParMatrix*>( &xfer->GetBlock( 0, 0 ) );
  auto* new_hypre = &par_J_new.get();

  // Check difference: A_old - A_new
  shared::ParSparseMat diff_psm = shared::ParSparseMatView( old_hypre ) - shared::ParSparseMatView( new_hypre );

  // Verify match by checking NNZ of difference
  // Since we subtracted, exact match means NNZ should be 0 (or values very small)
  // mfem::HypreParMatrix doesn't have an easy "max norm" without converting to SparseMatrix
  // but we can check NNZ. Note that operator- might keep zero entries.
  // A better way is to check the data array of the resulting matrix.

  double max_err = 0.0;
  HYPRE_ParCSRMatrix diff_csr = diff_psm.get();
  hypre_ParCSRMatrix* diff_parcsr = (hypre_ParCSRMatrix*)diff_csr;
  hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag( diff_parcsr );
  double* data = hypre_CSRMatrixData( diag );
  int num_nonzeros = hypre_CSRMatrixNumNonzeros( diag );
  for ( int i = 0; i < num_nonzeros; ++i ) {
    max_err = std::max( max_err, std::abs( data[i] ) );
  }
  // Also check off-diagonal block
  hypre_CSRMatrix* offd = hypre_ParCSRMatrixOffd( diff_parcsr );
  data = hypre_CSRMatrixData( offd );
  num_nonzeros = hypre_CSRMatrixNumNonzeros( offd );
  for ( int i = 0; i < num_nonzeros; ++i ) {
    max_err = std::max( max_err, std::abs( data[i] ) );
  }

  double global_max_err = 0.0;
  MPI_Allreduce( &max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

  if ( rank == 0 ) {
    std::cout << "Old method time: "
              << std::chrono::duration_cast<std::chrono::microseconds>( end_old - start_old ).count() << " us"
              << std::endl;
    std::cout << "New method time: "
              << std::chrono::duration_cast<std::chrono::microseconds>( end_new - start_new ).count() << " us"
              << std::endl;
    std::cout << "Matrix difference max err: " << global_max_err << std::endl;
  }

  if ( global_max_err > 1e-12 ) {
    SLIC_ERROR_ROOT( "Matrices do not match!" );
  } else {
    SLIC_INFO_ROOT( "Verification successful: Matrices match." );
  }

  tribol::finalize();
  MPI_Finalize();
  return 0;
}
