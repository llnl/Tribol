// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

// MFEM includes
#include "mfem.hpp"

// Axom includes
#include "axom/slic.hpp"

// Shared includes
#include "shared/mesh/MeshBuilder.hpp"

// Tribol includes
#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MfemData.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

class MfemJacobianTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override { tribol::finalize(); }
};

TEST_F( MfemJacobianTest, direct_jacobian_assembly )
{
  int n_ranks;
  MPI_Comm_size( MPI_COMM_WORLD, &n_ranks );

  int ref_levels = 0;
  int nel_per_dir = std::pow( 2, ref_levels );

  auto mortar_attrs = std::set<int>( { 4 } );
  auto nonmortar_attrs = std::set<int>( { 5 } );

  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(1, 3)
      .updateBdrAttrib(4, 7)
      .updateBdrAttrib(5, 1)
      .updateBdrAttrib(6, 4),
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir)
      .translate({0.0, 0.0, 0.99})
      .updateBdrAttrib(1, 8)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(4, 7)
      .updateBdrAttrib(5, 1)
      .updateBdrAttrib(8, 5)
  }));
  // clang-format on

  int dim = mesh.SpaceDimension();
  int order = 1;
  mfem::H1_FECollection fe_coll( order, dim );
  mfem::ParFiniteElementSpace par_fe_space( &mesh, &fe_coll, dim );
  mfem::ParGridFunction coords( &par_fe_space );
  mesh.GetNodes( coords );

  int cs_id = 0;
  int mesh1_id = 0;
  int mesh2_id = 1;
  tribol::registerMfemCouplingScheme( cs_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs, nonmortar_attrs,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::SINGLE_MORTAR,
                                      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );

  tribol::updateMfemParallelDecomposition();

  auto& cs_manager = tribol::CouplingSchemeManager::getInstance();
  auto* cs = cs_manager.findData( cs_id );
  ASSERT_NE( cs, nullptr );

  auto* mesh_data = cs->getMfemMeshData();
  auto* submesh_data = cs->getMfemSubmeshData();
  ASSERT_NE( mesh_data, nullptr );
  ASSERT_NE( submesh_data, nullptr );
  cs->setMfemJacobianData(
      std::make_unique<tribol::MfemJacobianData>( *mesh_data, *submesh_data, cs->getContactMethod() ) );
  auto* jac_data = cs->getMfemJacobianData();
  ASSERT_NE( jac_data, nullptr );

  jac_data->UpdateJacobianXfer();

  std::vector<tribol::ComputedElementData> contributions;

  int num_dofs_per_elem = 8 * dim;  // Hex element, 8 nodes, 3 dims
  int mat_size = num_dofs_per_elem * num_dofs_per_elem;

  tribol::ComputedElementData contrib;
  contrib.row_space = tribol::BlockSpace::MORTAR;
  contrib.col_space = tribol::BlockSpace::MORTAR;

  if ( mesh_data->GetMesh1NE() > 0 ) {
    contrib.row_elem_ids.push_back( 0 );
    contrib.col_elem_ids.push_back( 0 );

    contrib.jacobian_data.resize( mat_size );
    for ( int i = 0; i < mat_size; ++i ) contrib.jacobian_data[i] = 1.0;

    contrib.jacobian_offsets.push_back( 0 );

    contributions.push_back( contrib );
  }

  auto ParJ = jac_data->GetMfemJacobian( contributions );

  int local_contrib = contributions.size();
  int global_contrib = 0;
  MPI_Allreduce( &local_contrib, &global_contrib, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

  if ( global_contrib > 0 ) {
    EXPECT_GT( ParJ->NNZ(), 0 );
  } else {
    if ( n_ranks == 1 ) {
      FAIL() << "No surface elements found on mesh 1 in serial run.";
    }
  }
}

TEST_F( MfemJacobianTest, direct_jacobian_assembly_lor )
{
  int n_ranks;
  MPI_Comm_size( MPI_COMM_WORLD, &n_ranks );

  int ref_levels = 0;
  int nel_per_dir = std::pow( 2, ref_levels );

  auto mortar_attrs = std::set<int>( { 4 } );
  auto nonmortar_attrs = std::set<int>( { 5 } );

  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(1, 3)
      .updateBdrAttrib(4, 7)
      .updateBdrAttrib(5, 1)
      .updateBdrAttrib(6, 4),
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir)
      .translate({0.0, 0.0, 0.99})
      .updateBdrAttrib(1, 8)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(4, 7)
      .updateBdrAttrib(5, 1)
      .updateBdrAttrib(8, 5)
  }));
  // clang-format on

  int dim = mesh.SpaceDimension();
  int order = 2;
  mesh.SetCurvature( order );
  auto* nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
  ASSERT_NE( nodes, nullptr );
  mfem::ParGridFunction coords( nodes->ParFESpace() );
  coords = *nodes;

  int cs_id = 1;
  int mesh1_id = 0;
  int mesh2_id = 1;
  tribol::registerMfemCouplingScheme( cs_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs, nonmortar_attrs,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::SINGLE_MORTAR,
                                      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );

  tribol::updateMfemParallelDecomposition();

  auto& cs_manager = tribol::CouplingSchemeManager::getInstance();
  auto* cs = cs_manager.findData( cs_id );
  ASSERT_NE( cs, nullptr );

  auto* mesh_data = cs->getMfemMeshData();
  auto* submesh_data = cs->getMfemSubmeshData();
  ASSERT_NE( mesh_data, nullptr );
  ASSERT_NE( submesh_data, nullptr );
  cs->setMfemJacobianData(
      std::make_unique<tribol::MfemJacobianData>( *mesh_data, *submesh_data, cs->getContactMethod() ) );
  auto* jac_data = cs->getMfemJacobianData();
  ASSERT_NE( jac_data, nullptr );
  jac_data->UpdateJacobianXfer();

  std::vector<tribol::ComputedElementData> contributions;

  int mat_size = 0;
  if ( mesh_data->GetMesh1NE() > 0 ) {
    int num_dofs_per_elem = 8 * dim;
    mat_size = num_dofs_per_elem * num_dofs_per_elem;
    tribol::ComputedElementData contrib;
    contrib.row_space = tribol::BlockSpace::MORTAR;
    contrib.col_space = tribol::BlockSpace::MORTAR;
    contrib.row_elem_ids.push_back( 0 );
    contrib.col_elem_ids.push_back( 0 );
    contrib.jacobian_data.resize( mat_size );
    for ( int i = 0; i < mat_size; ++i ) contrib.jacobian_data[i] = 1.0;
    contrib.jacobian_offsets.push_back( 0 );
    contributions.push_back( contrib );
  }

  auto ParJ = jac_data->GetMfemJacobian( contributions );

  int local_contrib = static_cast<int>( contributions.size() );
  int global_contrib = 0;
  MPI_Allreduce( &local_contrib, &global_contrib, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

  if ( global_contrib > 0 ) {
    EXPECT_GT( ParJ->NNZ(), 0 );
  } else if ( n_ranks == 1 ) {
    FAIL() << "No surface elements found on mesh 1 in serial run.";
  }
}

TEST_F( MfemJacobianTest, lor_transfer_matches_mfem_l2projection_h1 )
{
  // Build a HO submesh + LOR mesh via Tribol's MfemMeshData, then compare:
  // (1) MFEM's L2ProjectionGridTransfer (ForwardOperator().Mult)
  // (2) Tribol's cached HO->LOR sparse transfer matrix (Mult)

  int ref_levels = 0;
  int nel_per_dir = std::pow( 2, ref_levels );

  auto mortar_attrs = std::set<int>( { 4 } );
  auto nonmortar_attrs = std::set<int>( { 5 } );

  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(1, 3)
      .updateBdrAttrib(4, 7)
      .updateBdrAttrib(5, 1)
      .updateBdrAttrib(6, 4),
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir)
      .translate({0.0, 0.0, 0.99})
      .updateBdrAttrib(1, 8)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(4, 7)
      .updateBdrAttrib(5, 1)
      .updateBdrAttrib(8, 5)
  }));
  // clang-format on

  const int order = 2;
  mesh.SetCurvature( order );
  auto* nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
  ASSERT_NE( nodes, nullptr );
  mfem::ParGridFunction coords( nodes->ParFESpace() );
  coords = *nodes;

  int cs_id = 2;
  int mesh1_id = 0;
  int mesh2_id = 1;
  tribol::registerMfemCouplingScheme( cs_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs, nonmortar_attrs,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::SINGLE_MORTAR,
                                      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );

  tribol::updateMfemParallelDecomposition();

  auto& cs_manager = tribol::CouplingSchemeManager::getInstance();
  auto* cs = cs_manager.findData( cs_id );
  ASSERT_NE( cs, nullptr );

  auto* mesh_data = cs->getMfemMeshData();
  auto* submesh_data = cs->getMfemSubmeshData();
  ASSERT_NE( mesh_data, nullptr );
  ASSERT_NE( submesh_data, nullptr );

  cs->setMfemJacobianData(
      std::make_unique<tribol::MfemJacobianData>( *mesh_data, *submesh_data, cs->getContactMethod() ) );
  auto* jac_data = cs->getMfemJacobianData();
  ASSERT_NE( jac_data, nullptr );
  jac_data->UpdateJacobianXfer();

  const auto* T = jac_data->GetDisplacementHoToLorTransfer();
  ASSERT_NE( T, nullptr );

  const mfem::ParFiniteElementSpace& ho_fes = mesh_data->GetSubmeshFESpace();
  const mfem::ParFiniteElementSpace* lor_fes_ptr = mesh_data->GetLORMeshFESpace();
  ASSERT_NE( lor_fes_ptr, nullptr );
  const mfem::ParFiniteElementSpace& lor_fes = *lor_fes_ptr;

  auto& ho_fes_nc = const_cast<mfem::ParFiniteElementSpace&>( ho_fes );
  auto& lor_fes_nc = const_cast<mfem::ParFiniteElementSpace&>( lor_fes );
  mfem::L2ProjectionGridTransfer mfem_xfer( ho_fes_nc, lor_fes_nc );
  mfem_xfer.UseEA( false );
  const mfem::Operator& F = mfem_xfer.ForwardOperator();

  // Build a *conforming/consistent* HO dof vector by starting from true-dofs
  // and prolongating. This avoids ambiguity in how shared dofs are reduced to
  // true dofs (MFEM's restriction operator assumes a valid GridFunction-like
  // vector with equal shared dof values).
  mfem::Vector x_ho_true( ho_fes.GetTrueVSize() );
  {
    int myid = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &myid );
    const auto& tdof_offsets = ho_fes.GetTrueDofOffsets();
    const double g0 = static_cast<double>( tdof_offsets[myid] );
    for ( int i = 0; i < x_ho_true.Size(); ++i ) {
      x_ho_true[i] = std::sin( 0.1 * ( g0 + static_cast<double>( i ) + 1.0 ) );
    }
  }

  mfem::Vector x_ho( ho_fes.GetVSize() );
  x_ho = 0.0;
  const mfem::Operator* P_ho = ho_fes.GetProlongationMatrix();
  if ( P_ho ) {
    P_ho->Mult( x_ho_true, x_ho );
  } else {
    x_ho = x_ho_true;
  }

  mfem::Vector y_mfem( lor_fes.GetVSize() );
  y_mfem = 0.0;
  F.Mult( x_ho, y_mfem );

  mfem::Vector y_tribol( lor_fes.GetVSize() );
  y_tribol = 0.0;
  T->get().Mult( x_ho, y_tribol );

  mfem::Vector diff( y_mfem.Size() );
  diff = 0.0;
  diff += y_mfem;
  diff -= y_tribol;

  double local_norm2 = diff * diff;
  double global_norm2 = 0.0;
  MPI_Allreduce( &local_norm2, &global_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

  EXPECT_LT( std::sqrt( global_norm2 ), 1e-10 );

  // Also compare on a small set of unit true-dof vectors, prolonged to a
  // conforming DOF vector before applying the operators. This helps localize
  // mismatches to specific true-dofs (e.g. shared/constrained dofs).
  int myrank = 0;
  int nranks = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
  MPI_Comm_size( MPI_COMM_WORLD, &nranks );

  // Choose a small set of *global* true-dof indices and broadcast it so all
  // ranks execute the same number of MPI collectives (some ranks may own zero
  // local true-dofs).
  std::vector<long long> sample_gtdofs;
  {
    const HYPRE_BigInt* tdof_offsets = ho_fes.GetTrueDofOffsets();
    ASSERT_NE( tdof_offsets, nullptr );
    const HYPRE_BigInt global_true = tdof_offsets[nranks];

    if ( myrank == 0 && global_true > 0 ) {
      sample_gtdofs = {0,
                       static_cast<long long>( global_true / 2 ),
                       static_cast<long long>( global_true - 1 ),
                       static_cast<long long>( global_true / 3 ),
                       static_cast<long long>( ( 2 * global_true ) / 3 )};
      std::sort( sample_gtdofs.begin(), sample_gtdofs.end() );
      sample_gtdofs.erase( std::unique( sample_gtdofs.begin(), sample_gtdofs.end() ), sample_gtdofs.end() );
      if ( static_cast<int>( sample_gtdofs.size() ) > 5 ) {
        sample_gtdofs.resize( 5 );
      }
    }

    int n = static_cast<int>( sample_gtdofs.size() );
    MPI_Bcast( &n, 1, MPI_INT, 0, MPI_COMM_WORLD );
    sample_gtdofs.resize( static_cast<size_t>( n ) );
    if ( n > 0 ) {
      MPI_Bcast( sample_gtdofs.data(), n, MPI_LONG_LONG, 0, MPI_COMM_WORLD );
    }
  }

  double local_max_err = 0.0;
  long long local_max_gtdof = -1;
  double global_max_err = 0.0;
  int global_max_rank = -1;
  long long global_max_gtdof = -1;
  double local_max_res_vs_pt = 0.0;
  double local_max_res_vs_wpt = 0.0;
  double global_max_res_vs_pt = 0.0;
  double global_max_res_vs_wpt = 0.0;

  mfem::Vector e_true( ho_fes.GetTrueVSize() );
  mfem::Vector x_unit( ho_fes.GetVSize() );
  mfem::Vector x_true_from_res( ho_fes.GetTrueVSize() );
  mfem::Vector x_true_from_Pt( ho_fes.GetTrueVSize() );
  mfem::Vector x_true_from_WPt( ho_fes.GetTrueVSize() );

  // Build W*P^T for diagnostics (this is what Tribol currently approximates as
  // a restriction matrix when it cannot materialize fes.GetRestrictionOperator()).
  std::unique_ptr<mfem::HypreParMatrix> P_ho_T;
  mfem::Vector multiplicity_true;
  mfem::Vector inv_multiplicity_true;
  {
    const auto* P_hypre = ho_fes.Dof_TrueDof_Matrix();
    ASSERT_NE( P_hypre, nullptr );
    P_ho_T.reset( P_hypre->Transpose() );
    ASSERT_NE( P_ho_T, nullptr );
    mfem::Vector ones_dof( ho_fes.GetVSize() );
    ones_dof = 1.0;
    multiplicity_true.SetSize( ho_fes.GetTrueVSize() );
    multiplicity_true = 0.0;
    P_ho_T->Mult( ones_dof, multiplicity_true );
    inv_multiplicity_true = multiplicity_true;
    inv_multiplicity_true.Reciprocal();
  }

  const mfem::Operator* ho_res = ho_fes.GetRestrictionOperator();
  mfem::Vector y_mfem_unit( lor_fes.GetVSize() );
  mfem::Vector y_tribol_unit( lor_fes.GetVSize() );
  mfem::Vector diff_unit( lor_fes.GetVSize() );

  for ( long long gtdof_ll : sample_gtdofs ) {
    const HYPRE_BigInt gtdof = static_cast<HYPRE_BigInt>( gtdof_ll );

    e_true = 0.0;
    {
      const auto& tdof_offsets = ho_fes.GetTrueDofOffsets();
      const HYPRE_BigInt lo = tdof_offsets[myrank];
      const HYPRE_BigInt hi = tdof_offsets[myrank + 1];
      if ( gtdof >= lo && gtdof < hi ) {
        const int local = static_cast<int>( gtdof - lo );
        e_true[local] = 1.0;
      }
    }

    x_unit = 0.0;
    if ( P_ho ) {
      P_ho->Mult( e_true, x_unit );
    } else {
      x_unit = e_true;
    }

    if ( ho_res ) {
      x_true_from_res = 0.0;
      ho_res->Mult( x_unit, x_true_from_res );
    } else {
      x_true_from_res = x_unit;
    }

    // Compare MFEM restriction against the identity on conforming vectors.
    // For x_unit = P_ho * e_true, a "true dof extraction" should recover e_true.
    {
      mfem::Vector rdiff( x_true_from_res.Size() );
      rdiff = 0.0;
      rdiff += x_true_from_res;
      rdiff -= e_true;
      double local_rnorm2 = rdiff * rdiff;
      double global_rnorm2 = 0.0;
      MPI_Allreduce( &local_rnorm2, &global_rnorm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
      // Keep as EXPECT (not ASSERT) so we still get the transfer mismatch report.
      EXPECT_LT( std::sqrt( global_rnorm2 ), 1e-12 );
    }

    // Diagnostics: compare against P^T and W*P^T.
    x_true_from_Pt = 0.0;
    P_ho_T->Mult( x_unit, x_true_from_Pt );
    x_true_from_WPt = x_true_from_Pt;
    x_true_from_WPt *= inv_multiplicity_true;

    {
      mfem::Vector rdiff_pt( x_true_from_res.Size() );
      rdiff_pt = 0.0;
      rdiff_pt += x_true_from_res;
      rdiff_pt -= x_true_from_Pt;
      double local_rnorm2 = rdiff_pt * rdiff_pt;
      double global_rnorm2 = 0.0;
      MPI_Allreduce( &local_rnorm2, &global_rnorm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
      local_max_res_vs_pt = std::max( local_max_res_vs_pt, std::sqrt( global_rnorm2 ) );
    }
    {
      mfem::Vector rdiff_wpt( x_true_from_res.Size() );
      rdiff_wpt = 0.0;
      rdiff_wpt += x_true_from_res;
      rdiff_wpt -= x_true_from_WPt;
      double local_rnorm2 = rdiff_wpt * rdiff_wpt;
      double global_rnorm2 = 0.0;
      MPI_Allreduce( &local_rnorm2, &global_rnorm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
      local_max_res_vs_wpt = std::max( local_max_res_vs_wpt, std::sqrt( global_rnorm2 ) );
    }

    y_mfem_unit = 0.0;
    y_tribol_unit = 0.0;
    F.Mult( x_unit, y_mfem_unit );
    T->get().Mult( x_unit, y_tribol_unit );

    diff_unit = 0.0;
    diff_unit += y_mfem_unit;
    diff_unit -= y_tribol_unit;

    local_norm2 = diff_unit * diff_unit;
    MPI_Allreduce( &local_norm2, &global_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    const double err = std::sqrt( global_norm2 );
    if ( err > local_max_err ) {
      local_max_err = err;
      local_max_gtdof = gtdof_ll;
    }
  }

  struct {
    double val;
    int rank;
  } in, out;
  in.val = local_max_err;
  in.rank = myrank;
  MPI_Allreduce( &in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD );
  global_max_err = out.val;
  global_max_rank = out.rank;

  global_max_gtdof = local_max_gtdof;
  MPI_Bcast( &global_max_gtdof, 1, MPI_LONG_LONG, global_max_rank, MPI_COMM_WORLD );

  MPI_Allreduce( &local_max_res_vs_pt, &global_max_res_vs_pt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  MPI_Allreduce( &local_max_res_vs_wpt, &global_max_res_vs_wpt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

  if ( global_max_err >= 1e-10 ) {
    if ( myrank == 0 ) {
      ADD_FAILURE() << "Max unit-tdof mismatch: err=" << global_max_err << " at rank=" << global_max_rank
                    << " global_tdof=" << global_max_gtdof
                    << " (max ||R_ho - P^T||=" << global_max_res_vs_pt
                    << ", max ||R_ho - W*P^T||=" << global_max_res_vs_wpt << ")";
    }
  }

  EXPECT_LT( global_max_err, 1e-10 );
}

TEST_F( MfemJacobianTest, lor_transfer_fallback_buildH1TrueRestriction_matches_mfem )
{
  // Force Tribol to use its explicit BuildH1TrueRestriction() path (instead of
  // reusing MFEM's internal true-dof operator) when building the HO->LOR transfer.
  setenv( "TRIBOL_MFEM_FORCE_LOR_FALLBACK", "1", 1 );

  int ref_levels = 0;
  int nel_per_dir = std::pow( 2, ref_levels );

  auto mortar_attrs = std::set<int>( { 4 } );
  auto nonmortar_attrs = std::set<int>( { 5 } );

  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(1, 3)
      .updateBdrAttrib(4, 7)
      .updateBdrAttrib(5, 1)
      .updateBdrAttrib(6, 4),
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir)
      .translate({0.0, 0.0, 0.99})
      .updateBdrAttrib(1, 8)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(4, 7)
      .updateBdrAttrib(5, 1)
      .updateBdrAttrib(8, 5)
  }));
  // clang-format on

  const int order = 2;
  mesh.SetCurvature( order );
  auto* nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
  ASSERT_NE( nodes, nullptr );
  mfem::ParGridFunction coords( nodes->ParFESpace() );
  coords = *nodes;

  int cs_id = 3;
  int mesh1_id = 0;
  int mesh2_id = 1;
  tribol::registerMfemCouplingScheme( cs_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs, nonmortar_attrs,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::SINGLE_MORTAR,
                                      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );
  tribol::updateMfemParallelDecomposition();

  auto& cs_manager = tribol::CouplingSchemeManager::getInstance();
  auto* cs = cs_manager.findData( cs_id );
  ASSERT_NE( cs, nullptr );

  auto* mesh_data = cs->getMfemMeshData();
  auto* submesh_data = cs->getMfemSubmeshData();
  ASSERT_NE( mesh_data, nullptr );
  ASSERT_NE( submesh_data, nullptr );

  cs->setMfemJacobianData(
      std::make_unique<tribol::MfemJacobianData>( *mesh_data, *submesh_data, cs->getContactMethod() ) );
  auto* jac_data = cs->getMfemJacobianData();
  ASSERT_NE( jac_data, nullptr );
  jac_data->UpdateJacobianXfer();

  const auto* T = jac_data->GetDisplacementHoToLorTransfer();
  ASSERT_NE( T, nullptr );

  const mfem::ParFiniteElementSpace& ho_fes = mesh_data->GetSubmeshFESpace();
  const mfem::ParFiniteElementSpace* lor_fes_ptr = mesh_data->GetLORMeshFESpace();
  ASSERT_NE( lor_fes_ptr, nullptr );
  const mfem::ParFiniteElementSpace& lor_fes = *lor_fes_ptr;

  auto& ho_fes_nc = const_cast<mfem::ParFiniteElementSpace&>( ho_fes );
  auto& lor_fes_nc = const_cast<mfem::ParFiniteElementSpace&>( lor_fes );
  mfem::L2ProjectionGridTransfer mfem_xfer( ho_fes_nc, lor_fes_nc );
  mfem_xfer.UseEA( false );
  const mfem::Operator& F = mfem_xfer.ForwardOperator();

  mfem::Vector x_ho_true( ho_fes.GetTrueVSize() );
  {
    int myid = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &myid );
    const HYPRE_BigInt* tdof_offsets = ho_fes.GetTrueDofOffsets();
    ASSERT_NE( tdof_offsets, nullptr );
    const double g0 = static_cast<double>( tdof_offsets[myid] );
    for ( int i = 0; i < x_ho_true.Size(); ++i ) {
      x_ho_true[i] = std::sin( 0.1 * ( g0 + static_cast<double>( i ) + 1.0 ) );
    }
  }

  mfem::Vector x_ho( ho_fes.GetVSize() );
  x_ho = 0.0;
  const mfem::Operator* P_ho = ho_fes.GetProlongationMatrix();
  if ( P_ho ) {
    P_ho->Mult( x_ho_true, x_ho );
  } else {
    x_ho = x_ho_true;
  }

  mfem::Vector y_mfem( lor_fes.GetVSize() );
  y_mfem = 0.0;
  F.Mult( x_ho, y_mfem );

  mfem::Vector y_tribol( lor_fes.GetVSize() );
  y_tribol = 0.0;
  T->get().Mult( x_ho, y_tribol );

  mfem::Vector diff( y_mfem.Size() );
  diff = 0.0;
  diff += y_mfem;
  diff -= y_tribol;

  double local_norm2 = diff * diff;
  double global_norm2 = 0.0;
  MPI_Allreduce( &local_norm2, &global_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_LT( std::sqrt( global_norm2 ), 1e-10 );

  unsetenv( "TRIBOL_MFEM_FORCE_LOR_FALLBACK" );
}

int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();
#endif

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  tribol::finalize();
  MPI_Finalize();

  return result;
}
