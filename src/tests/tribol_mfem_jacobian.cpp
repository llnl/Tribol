// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

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

namespace {

struct EnvVarGuard {
  explicit EnvVarGuard( const char* name, const char* value ) : name_( name )
  {
    const char* old = std::getenv( name );
    if ( old ) {
      had_old_ = true;
      old_value_ = old;
    }
    if ( value ) {
      setenv( name, value, 1 );
    } else {
      unsetenv( name );
    }
  }

  ~EnvVarGuard()
  {
    if ( had_old_ ) {
      setenv( name_.c_str(), old_value_.c_str(), 1 );
    } else {
      unsetenv( name_.c_str() );
    }
  }

 private:
  std::string name_;
  bool had_old_ = false;
  std::string old_value_;
};

enum class FieldKind
{
  Displacement,
  LagrangeMultiplier
};

struct LorTransferParams {
  int order = 2;
  int lor_factor = 2;
  FieldKind field = FieldKind::Displacement;
  bool force_fallback = false;
};

std::string ParamsToString( const testing::TestParamInfo<LorTransferParams>& info )
{
  const auto& p = info.param;
  const char* field = ( p.field == FieldKind::Displacement ) ? "disp" : "lm";
  const char* fb = p.force_fallback ? "fallback" : "mfemR";
  return "p" + std::to_string( p.order ) + "_lor" + std::to_string( p.lor_factor ) + "_" + field + "_" + fb;
}

bool BuildConsistentHoVector( const mfem::ParFiniteElementSpace& ho_fes, mfem::Vector& x_ho )
{
  mfem::Vector x_true( ho_fes.GetTrueVSize() );
  {
    int myid = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &myid );
    const HYPRE_BigInt* tdof_offsets = ho_fes.GetTrueDofOffsets();
    if ( tdof_offsets == nullptr ) {
      return false;
    }
    const double g0 = static_cast<double>( tdof_offsets[myid] );
    for ( int i = 0; i < x_true.Size(); ++i ) {
      x_true[i] = std::sin( 0.1 * ( g0 + static_cast<double>( i ) + 1.0 ) );
    }
  }

  x_ho.SetSize( ho_fes.GetVSize() );
  x_ho = 0.0;
  const mfem::Operator* P = ho_fes.GetProlongationMatrix();
  if ( P ) {
    P->Mult( x_true, x_ho );
  } else {
    x_ho = x_true;
  }
  return true;
}

void CompareHoToLorTransfers( const mfem::Operator& F, const mfem::HypreParMatrix& T_mat,
                              const mfem::ParFiniteElementSpace& ho_fes, const mfem::ParFiniteElementSpace& lor_fes )
{
  mfem::Vector x_ho;
  const bool ok = BuildConsistentHoVector( ho_fes, x_ho );
  int local_ok = ok ? 1 : 0;
  int global_ok = 0;
  MPI_Allreduce( &local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD );
  ASSERT_EQ( global_ok, 1 );

  mfem::Vector y_mfem( lor_fes.GetVSize() );
  y_mfem = 0.0;
  F.Mult( x_ho, y_mfem );

  mfem::Vector y_tribol( lor_fes.GetVSize() );
  y_tribol = 0.0;
  T_mat.Mult( x_ho, y_tribol );

  mfem::Vector diff( y_mfem.Size() );
  diff = 0.0;
  diff += y_mfem;
  diff -= y_tribol;

  double local_norm2 = diff * diff;
  double global_norm2 = 0.0;
  MPI_Allreduce( &local_norm2, &global_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_LT( std::sqrt( global_norm2 ), 1e-10 );
}

}  // namespace

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

  tribol::updateMfemParallelDecomposition( n_ranks, true );

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

  tribol::updateMfemParallelDecomposition( n_ranks, true );

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

class MfemLorTransferParamTest : public MfemJacobianTest, public testing::WithParamInterface<LorTransferParams> {};

TEST_P( MfemLorTransferParamTest, lor_transfer_matches_mfem )
{
  const auto p = GetParam();
  EnvVarGuard guard( "TRIBOL_MFEM_FORCE_LOR_FALLBACK", p.force_fallback ? "1" : nullptr );

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

  mesh.SetCurvature( p.order );
  auto* nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
  ASSERT_NE( nodes, nullptr );
  mfem::ParGridFunction coords( nodes->ParFESpace() );
  coords = *nodes;

  // Unique cs_id per test instance (process-local) so repeated registration
  // doesn't collide if a prior test aborted before TearDown().
  static int cs_id_next = 20;
  const int cs_id = cs_id_next++;
  const int mesh1_id = 0;
  const int mesh2_id = 1;

  tribol::registerMfemCouplingScheme( cs_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs, nonmortar_attrs,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::SINGLE_MORTAR,
                                      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );

  int n_ranks = 1;
  MPI_Comm_size( MPI_COMM_WORLD, &n_ranks );

  auto& cs_manager = tribol::CouplingSchemeManager::getInstance();
  auto* cs = cs_manager.findData( cs_id );
  ASSERT_NE( cs, nullptr );

  auto* mesh_data = cs->getMfemMeshData();
  ASSERT_NE( mesh_data, nullptr );

  // registerMfemCouplingScheme() creates submesh/jacobian data using the
  // default LOR factor (geometry order). Override it here, then rebuild
  // submesh + jacobian data so they use the updated LOR mesh.
  if ( p.lor_factor != mesh_data->GetLORFactor() ) {
    mesh_data->SetLORFactor( p.lor_factor );
    auto pressure_fec = std::make_unique<mfem::H1_FECollection>( p.order, mesh.SpaceDimension() );
    cs->setMfemSubmeshData( std::make_unique<tribol::MfemSubmeshData>( mesh_data->GetSubmesh(), mesh_data->GetLORMesh(),
                                                                       std::move( pressure_fec ), 1, false ) );
  }

  tribol::updateMfemParallelDecomposition( n_ranks, true );

  auto* submesh_data = cs->getMfemSubmeshData();
  ASSERT_NE( submesh_data, nullptr );

  cs->setMfemJacobianData(
      std::make_unique<tribol::MfemJacobianData>( *mesh_data, *submesh_data, cs->getContactMethod() ) );
  auto* jac_data = cs->getMfemJacobianData();
  ASSERT_NE( jac_data, nullptr );
  jac_data->UpdateJacobianXfer();

  const mfem::ParFiniteElementSpace* ho_fes = nullptr;
  const mfem::ParFiniteElementSpace* lor_fes = nullptr;
  const shared::ParSparseMat* T = nullptr;

  if ( p.field == FieldKind::Displacement ) {
    ho_fes = &mesh_data->GetSubmeshFESpace();
    lor_fes = mesh_data->GetLORMeshFESpace();
    T = jac_data->GetDisplacementHoToLorTransfer();
  } else {
    ho_fes = &submesh_data->GetSubmeshFESpace();
    lor_fes = submesh_data->GetLORMeshFESpace();
    T = jac_data->GetLagrangeMultiplierHoToLorTransfer();
  }

  ASSERT_NE( ho_fes, nullptr );
  ASSERT_NE( lor_fes, nullptr );
  ASSERT_NE( T, nullptr );

  auto& ho_fes_nc = const_cast<mfem::ParFiniteElementSpace&>( *ho_fes );
  auto& lor_fes_nc = const_cast<mfem::ParFiniteElementSpace&>( *lor_fes );
  mfem::L2ProjectionGridTransfer mfem_xfer( ho_fes_nc, lor_fes_nc );
  mfem_xfer.UseEA( false );
  const mfem::Operator& F = mfem_xfer.ForwardOperator();

  CompareHoToLorTransfers( F, T->get(), *ho_fes, *lor_fes );
}

INSTANTIATE_TEST_SUITE_P(
    MfemLorTransfer, MfemLorTransferParamTest,
    testing::Values( LorTransferParams{ 2, 2, FieldKind::Displacement, false },
                     LorTransferParams{ 2, 3, FieldKind::Displacement, false },  // lor_factor > order
                     LorTransferParams{ 2, 4, FieldKind::Displacement, false },  // lor_factor > order
                     LorTransferParams{ 3, 2, FieldKind::Displacement, false },  // lor_factor < order
                     LorTransferParams{ 3, 3, FieldKind::Displacement, false },
                     LorTransferParams{ 3, 4, FieldKind::Displacement, false },  // lor_factor > order
                     LorTransferParams{ 2, 2, FieldKind::LagrangeMultiplier, false },
                     LorTransferParams{ 2, 3, FieldKind::LagrangeMultiplier, false },
                     LorTransferParams{ 3, 3, FieldKind::LagrangeMultiplier, false },
                     LorTransferParams{ 3, 4, FieldKind::LagrangeMultiplier, false },
                     LorTransferParams{ 3, 4, FieldKind::Displacement, true },  // force BuildH1TrueRestriction()
                     LorTransferParams{ 3, 4, FieldKind::LagrangeMultiplier, true } ),
    ParamsToString );

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
