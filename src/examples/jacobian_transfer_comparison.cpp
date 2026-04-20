// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <set>
#include <vector>
#include <chrono>

#include "tribol/config.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

#include "mfem.hpp"

#include "axom/core.hpp"
#include "axom/slic.hpp"

#include "shared/mesh/MeshBuilder.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MfemData.hpp"

namespace {

std::vector<tribol::ComputedElementData> BuildComputedElementData(
    const tribol::MethodData& method_data,
    const std::vector<std::pair<tribol::BlockSpace, tribol::BlockSpace>>& contribs )
{
  std::vector<tribol::ComputedElementData> computed;
  computed.reserve( contribs.size() );

  for ( const auto& pair : contribs ) {
    tribol::ComputedElementData data( pair.first, pair.second );

    const auto& J_block = method_data.getBlockJ()( static_cast<int>( pair.first ), static_cast<int>( pair.second ) );
    const auto& row_elem_ids = method_data.getBlockJElementIds()[static_cast<int>( pair.first )];
    const auto& col_elem_ids = method_data.getBlockJElementIds()[static_cast<int>( pair.second )];

    SLIC_ERROR_ROOT_IF( J_block.size() != row_elem_ids.size() || J_block.size() != col_elem_ids.size(),
                        "MethodData block Jacobians and element-id arrays must have matching sizes." );

    int total_values = 0;
    for ( int i = 0; i < J_block.size(); ++i ) {
      total_values += J_block[i].Height() * J_block[i].Width();
    }
    data.reserve( J_block.size(), total_values );

    for ( int i = 0; i < J_block.size(); ++i ) {
      const int size = J_block[i].Height() * J_block[i].Width();
      data.append( row_elem_ids[i], col_elem_ids[i], J_block[i].GetData(), size );
    }

    computed.push_back( std::move( data ) );
  }

  return computed;
}

}  // namespace

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

  // Setup two-block mesh
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

  // Register Tribol coupling scheme
  int cs_id = 0;
  tribol::registerMfemCouplingScheme( cs_id, 0, 1, mesh, coords, mortar_attrs, nonmortar_attrs,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::SINGLE_MORTAR,
                                      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );
  tribol::setMPIComm( cs_id, MPI_COMM_WORLD );
  tribol::updateMfemParallelDecomposition();
  tribol::setLagrangeMultiplierOptions( cs_id, tribol::ImplicitEvalMode::MORTAR_JACOBIAN,
                                        tribol::SparseMode::MFEM_ELEMENT_DENSE );

  // Setup MFEM Jacobian data
  auto& cs_manager = tribol::CouplingSchemeManager::getInstance();
  auto* cs = cs_manager.findData( cs_id );
  auto* jac_data = cs->getMfemJacobianData();
  jac_data->UpdateJacobianXfer();

  // Synthesize Jacobian data for comparison
  int ne1 = cs->getMfemMeshData()->GetMesh1NE();
  int num_nodes_per_elem = 4;  // Quad faces
  int num_dofs_per_elem = num_nodes_per_elem * dim;

  // Populate ComputedElementData
  tribol::ComputedElementData new_data( tribol::BlockSpace::MORTAR, tribol::BlockSpace::MORTAR );
  new_data.reserve( ne1, ne1 * num_dofs_per_elem * num_dofs_per_elem );

  // Setup MethodData for comparison
  if ( cs->getMethodData() == nullptr ) {
    cs->allocateMethodData();
  }
  auto* method_data = cs->getMethodData();
  SLIC_ASSERT( method_data != nullptr );

  tribol::ArrayT<tribol::BlockSpace> spaces( { tribol::BlockSpace::MORTAR } );
  method_data->reserveBlockJ( std::move( spaces ), ne1 );

  for ( int e = 0; e < ne1; ++e ) {
    tribol::StackArray<tribol::DeviceArray2D<tribol::RealT>, 9> blockJ;
    blockJ[0] = tribol::DeviceArray2D<tribol::RealT>( num_dofs_per_elem, num_dofs_per_elem );
    std::vector<double> elem_vals( num_dofs_per_elem * num_dofs_per_elem );

    for ( int i = 0; i < num_dofs_per_elem; ++i ) {
      for ( int j = 0; j < num_dofs_per_elem; ++j ) {
        double val = static_cast<double>( e + i + j );
        blockJ[0]( i, j ) = val;
        elem_vals[i + j * num_dofs_per_elem] = val;
      }
    }

    new_data.append( e, e, elem_vals.data(), elem_vals.size() );
    tribol::ArrayT<int> ids( { e } );
    method_data->storeElemBlockJ( std::move( ids ), blockJ );
  }

  // Time and assemble with the MethodData adapter path
  auto old_contribs = BuildComputedElementData( *method_data, { { tribol::BlockSpace::MORTAR, tribol::BlockSpace::MORTAR } } );
  auto start_old = std::chrono::high_resolution_clock::now();
  auto par_J_old = jac_data->GetMfemJacobian( old_contribs );
  auto end_old = std::chrono::high_resolution_clock::now();

  // Time and assemble with the direct computed-contribution path
  std::vector<tribol::ComputedElementData> contribs_vec;
  if ( ne1 > 0 ) {
    contribs_vec.push_back( std::move( new_data ) );
  }

  auto start_new = std::chrono::high_resolution_clock::now();
  auto par_J_new = jac_data->GetMfemJacobian( contribs_vec );
  auto end_new = std::chrono::high_resolution_clock::now();

  // Verify results
  auto* old_hypre = &par_J_old.get();
  auto* new_hypre = &par_J_new.get();

  // Check difference
  shared::ParSparseMat diff_psm = shared::ParSparseMatView( old_hypre ) - shared::ParSparseMatView( new_hypre );

  // Verify match by checking max error
  double max_err = 0.0;
  HYPRE_ParCSRMatrix diff_csr = diff_psm.get();
  hypre_ParCSRMatrix* diff_parcsr = (hypre_ParCSRMatrix*)diff_csr;
  hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag( diff_parcsr );
  double* data = hypre_CSRMatrixData( diag );
  int num_nonzeros = hypre_CSRMatrixNumNonzeros( diag );
  for ( int i = 0; i < num_nonzeros; ++i ) {
    max_err = std::max( max_err, std::abs( data[i] ) );
  }
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
