// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/config.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

// MFEM includes
#include "mfem.hpp"

#include "axom/slic.hpp"

#include "shared/mesh/MeshBuilder.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

class MfemJacobianTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override { tribol::finalize(); }
};

namespace {

enum class FieldKind
{
  Displacement,
  LagrangeMultiplier
};

enum class SolverBlock
{
  Primary,
  Dual
};

struct LorTransferParams {
  int order = 2;
  int lor_factor = 2;
  FieldKind field = FieldKind::Displacement;
};

struct JacobianTestContext {
  tribol::CouplingScheme* cs = nullptr;
  tribol::MfemMeshData* mesh_data = nullptr;
  tribol::MfemSubmeshData* submesh_data = nullptr;
  tribol::MfemJacobianData* jac_data = nullptr;
};

// Each test fixture finalizes Tribol state, so parameterized runs need fresh ids.
int NextCouplingSchemeId()
{
  // Generate a process-local unique coupling scheme id. Tests register global Tribol state, and TearDown() calls
  // tribol::finalize(), so we avoid accidental id reuse if a test fails early or is parameterized.
  static int cs_id_next = 100;
  return cs_id_next++;
}

std::string ParamsToString( const testing::TestParamInfo<LorTransferParams>& info )
{
  // Provide stable, readable parameterized-test names so failures clearly indicate which (order, lor_factor, field)
  // combination triggered the mismatch.
  const auto& p = info.param;
  const char* field = ( p.field == FieldKind::Displacement ) ? "disp" : "lm";
  return "p" + std::to_string( p.order ) + "_lor" + std::to_string( p.lor_factor ) + "_" + field;
}

std::vector<int> BuildInactiveDualTdofs( const tribol::MfemMeshData& mesh_data,
                                         const tribol::MfemSubmeshData& submesh_data );

shared::ParSparseMat AssembleSolverBlockJacobian( const JacobianTestContext& ctx,
                                                  const std::vector<tribol::PackedPairJacobianContribs>& contributions,
                                                  SolverBlock row_block, SolverBlock col_block )
{
  // Assemble one solver-visible true-dof Jacobian block by:
  // 1) assembling a LOR/submesh Jacobian from redecomp element contributions, then
  // 2) composing it with explicit true-dof and (optional) HO->LOR/submesh->parent transfer operators.
  //
  // This is test code only: it mirrors the public MFEM interface behavior so we can validate the new API and transfer
  // composition without depending on a particular physics routine.
  if ( row_block == SolverBlock::Dual && col_block == SolverBlock::Dual ) {
    // Compatibility: dual-dual blocks are not assembled through redecomp transfer. Preserve the historical inactive-LM
    // identity behavior directly in solver true-dof space.
    auto comm = ctx.mesh_data->GetParentCoords().ParFESpace()->GetComm();
    auto& dual_fes = ctx.submesh_data->GetSubmeshFESpace();
    auto expected_inactive = BuildInactiveDualTdofs( *ctx.mesh_data, *ctx.submesh_data );
    mfem::Array<int> inactive_tdofs( static_cast<int>( expected_inactive.size() ) );
    for ( int i = 0; i < static_cast<int>( expected_inactive.size() ); ++i ) {
      inactive_tdofs[i] = expected_inactive[static_cast<size_t>( i )];
    }
    return shared::ParSparseMat::diagonalMatrix( comm, dual_fes.GlobalTrueVSize(), dual_fes.GetTrueDofOffsets(), 1.0,
                                                 inactive_tdofs, false );
  }

  const mfem::ParFiniteElementSpace* row_final_fes =
      ( row_block == SolverBlock::Primary ) ? ctx.mesh_data->GetParentCoords().ParFESpace()
                                            : &ctx.submesh_data->GetSubmeshFESpace();
  const mfem::ParFiniteElementSpace* col_final_fes =
      ( col_block == SolverBlock::Primary ) ? ctx.mesh_data->GetParentCoords().ParFESpace()
                                            : &ctx.submesh_data->GetSubmeshFESpace();
  return ctx.jac_data->GetMfemJacobian( row_final_fes, col_final_fes, contributions );
}

bool BuildConsistentHoVector( const mfem::ParFiniteElementSpace& ho_fes, mfem::Vector& x_ho )
{
  // Populate the true vector with a deterministic global pattern so transfer
  // comparisons remain meaningful in parallel.
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
  // Compare MFEM's runtime HO->LOR transfer operator against Tribol's assembled transfer matrix on the same input.
  // This is a numeric regression test for the assembled builder (not a performance test).
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

double MaxAbsMatrixEntry( const shared::ParSparseMatView& mat )
{
  // Hypre stores diagonal and off-diagonal CSR blocks separately, so inspect both.
  double local_max = 0.0;
  HYPRE_ParCSRMatrix csr = mat.get();
  auto* parcsr = (hypre_ParCSRMatrix*)csr;

  auto* diag = hypre_ParCSRMatrixDiag( parcsr );
  if ( auto* data = hypre_CSRMatrixData( diag ) ) {
    const int nnz = hypre_CSRMatrixNumNonzeros( diag );
    for ( int i = 0; i < nnz; ++i ) {
      local_max = std::max( local_max, std::abs( data[i] ) );
    }
  }

  auto* offd = hypre_ParCSRMatrixOffd( parcsr );
  if ( auto* data = hypre_CSRMatrixData( offd ) ) {
    const int nnz = hypre_CSRMatrixNumNonzeros( offd );
    for ( int i = 0; i < nnz; ++i ) {
      local_max = std::max( local_max, std::abs( data[i] ) );
    }
  }

  double global_max = 0.0;
  MPI_Allreduce( &local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  return global_max;
}

double LocalDiagonalEntry( const mfem::HypreParMatrix& mat, int local_row )
{
  // Helper to read the local diagonal entry (i,i) out of Hypre's diagonal CSR block. Used to validate the dual-dual
  // constraint identity block without needing a full matrix compare.
  HYPRE_MemoryLocation old_loc;
  HYPRE_GetMemoryLocation( &old_loc );
  HYPRE_SetMemoryLocation( HYPRE_MEMORY_HOST );

  auto& nc_mat = const_cast<mfem::HypreParMatrix&>( mat );
  nc_mat.HostReadWrite();  // Ensure the underlying Hypre data are valid on host.

  double result = 0.0;
  if ( local_row >= 0 && local_row < nc_mat.NumRows() ) {
    HYPRE_ParCSRMatrix csr = nc_mat;
    auto* parcsr = (hypre_ParCSRMatrix*)csr;
    auto* diag = hypre_ParCSRMatrixDiag( parcsr );
    auto* I = hypre_CSRMatrixI( diag );
    auto* J = hypre_CSRMatrixJ( diag );
    auto* data = hypre_CSRMatrixData( diag );

    if ( I && J && data ) {
      for ( int jj = I[local_row]; jj < I[local_row + 1]; ++jj ) {
        if ( J[jj] == local_row ) {
          result = data[jj];
          break;
        }
      }
    }
  }

  HYPRE_SetMemoryLocation( old_loc );
  return result;
}

mfem::DenseMatrix ConstantDenseMatrix( int rows, int cols, double value )
{
  // Create a dense element matrix filled with a constant. Used to build synthetic per-element contributions with
  // predictable assembled results.
  mfem::DenseMatrix mat( rows, cols );
  for ( int i = 0; i < rows; ++i ) {
    for ( int j = 0; j < cols; ++j ) {
      mat( i, j ) = value;
    }
  }
  return mat;
}

JacobianTestContext CreateJacobianTestContext( mfem::ParMesh& mesh, mfem::ParGridFunction& coords, int cs_id,
                                               int lor_factor = -1 )
{
  // Register a minimal MFEM coupling scheme and initialize redecomp + Jacobian transfer data.
  // This centralizes the boilerplate so each test can focus on the specific assembly/transfer behavior under test.
  int n_ranks = 1;
  MPI_Comm_size( MPI_COMM_WORLD, &n_ranks );

  const auto mortar_attrs = std::set<int>( { 4 } );
  const auto nonmortar_attrs = std::set<int>( { 5 } );

  tribol::registerMfemCouplingScheme( cs_id, 0, 1, mesh, coords, mortar_attrs, nonmortar_attrs,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::SINGLE_MORTAR,
                                      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );

  auto& cs_manager = tribol::CouplingSchemeManager::getInstance();
  auto* cs = cs_manager.findData( cs_id );
  EXPECT_NE( cs, nullptr );
  if ( cs == nullptr ) {
    return {};
  }

  auto* mesh_data = cs->getMfemMeshData();
  auto* submesh_data = cs->getMfemSubmeshData();
  EXPECT_NE( mesh_data, nullptr );
  EXPECT_NE( submesh_data, nullptr );
  if ( mesh_data == nullptr || submesh_data == nullptr ) {
    return {};
  }

  if ( lor_factor > 0 && lor_factor != mesh_data->GetLORFactor() ) {
    tribol::setMfemLORFactor( cs_id, lor_factor );
  }

  tribol::updateMfemParallelDecomposition( n_ranks, true );

  cs->setMfemJacobianData( std::make_unique<tribol::MfemJacobianData>( *mesh_data, *submesh_data ) );
  auto* jac_data = cs->getMfemJacobianData();
  EXPECT_NE( jac_data, nullptr );
  if ( jac_data == nullptr ) {
    return {};
  }

  jac_data->UpdateJacobianXfer();
  return { cs, mesh_data, submesh_data, jac_data };
}

template <typename Fn>
void WithJacobianData( int order, Fn&& fn )
{
  // Build the same two-cube contact configuration used elsewhere in this file, then
  // run the callback with a ready-to-use MfemJacobianData object.
  const int ref_levels = 0;
  const int nel_per_dir = std::pow( 2, ref_levels );

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

  const int dim = mesh.SpaceDimension();
  const int cs_id = NextCouplingSchemeId();

  if ( order > 1 ) {
    mesh.SetCurvature( order );
    auto* nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
    ASSERT_NE( nodes, nullptr );
    mfem::ParGridFunction coords( nodes->ParFESpace() );
    coords = *nodes;
    auto ctx = CreateJacobianTestContext( mesh, coords, cs_id, 2 );
    ASSERT_NE( ctx.jac_data, nullptr );
    fn( ctx );
    // The callback may allocate global Tribol/MFEM state; tear it down before the
    // next iteration re-registers another coupling scheme.
    tribol::finalize();
    return;
  }

  mfem::H1_FECollection fe_coll( order, dim );
  mfem::ParFiniteElementSpace par_fe_space( &mesh, &fe_coll, dim );
  mfem::ParGridFunction coords( &par_fe_space );
  mesh.GetNodes( coords );
  auto ctx = CreateJacobianTestContext( mesh, coords, cs_id );
  ASSERT_NE( ctx.jac_data, nullptr );
  fn( ctx );
  tribol::finalize();
}

const mfem::FiniteElementSpace& RedecompFESpaceForBlock( const tribol::MfemMeshData& mesh_data,
                                                         const tribol::MfemSubmeshData& submesh_data,
                                                         tribol::BlockSpace space )
{
  // Map a Tribol BlockSpace to the redecomp FE space that defines its element-id domain and element DOF layout.
  // Primary (mortar/nonmortar) contributions live on the "response" redecomp space; dual (LM) lives on the "gap" space.
  switch ( space ) {
    case tribol::BlockSpace::MORTAR:
    case tribol::BlockSpace::NONMORTAR:
      return *mesh_data.GetRedecompResponse().FESpace();
    case tribol::BlockSpace::LAGRANGE_MULTIPLIER:
      return *submesh_data.GetRedecompGap().FESpace();
    default:
      ADD_FAILURE() << "Unsupported block space.";
      return *mesh_data.GetRedecompResponse().FESpace();
  }
}

const tribol::Array1D<int>& ElemMapForBlock( const tribol::MfemMeshData& mesh_data, tribol::BlockSpace space )
{
  // Map a Tribol BlockSpace to the corresponding Tribol->redecomp element map.
  // Mortar uses mesh1 element ids (elem_map1); nonmortar and LM use mesh2 ids (elem_map2).
  switch ( space ) {
    case tribol::BlockSpace::MORTAR:
      return mesh_data.GetElemMap1();
    case tribol::BlockSpace::NONMORTAR:
    case tribol::BlockSpace::LAGRANGE_MULTIPLIER:
      return mesh_data.GetElemMap2();
    default:
      ADD_FAILURE() << "Unsupported block space.";
      return mesh_data.GetElemMap1();
  }
}

tribol::PackedPairJacobianContribs MakeConstantContribution( const tribol::MfemMeshData& mesh_data,
                                                             const tribol::MfemSubmeshData& submesh_data,
                                                             tribol::BlockSpace row_space, tribol::BlockSpace col_space,
                                                             double value )
{
  // Use one local element pair and fill its dense contribution with a constant so
  // assembled values stay easy to reason about in the tests below.
  const bool use_lor = ( mesh_data.GetLORMesh() != nullptr );
  auto surface_fes_for = [&]( tribol::BlockSpace space ) -> const mfem::ParFiniteElementSpace& {
    switch ( space ) {
      case tribol::BlockSpace::MORTAR:
      case tribol::BlockSpace::NONMORTAR:
        return use_lor ? *mesh_data.GetLORMeshFESpace() : mesh_data.GetSubmeshFESpace();
      case tribol::BlockSpace::LAGRANGE_MULTIPLIER:
        return use_lor ? *submesh_data.GetLORMeshFESpace() : submesh_data.GetSubmeshFESpace();
      default:
        ADD_FAILURE() << "Unsupported block space.";
        return use_lor ? *mesh_data.GetLORMeshFESpace() : mesh_data.GetSubmeshFESpace();
    }
  };

  auto elem_map_for = [&]( tribol::BlockSpace space ) -> const tribol::Array1D<int>& {
    switch ( space ) {
      case tribol::BlockSpace::MORTAR:
        return mesh_data.GetElemMap1();
      case tribol::BlockSpace::NONMORTAR:
      case tribol::BlockSpace::LAGRANGE_MULTIPLIER:
        return mesh_data.GetElemMap2();
      default:
        ADD_FAILURE() << "Unsupported block space.";
        return mesh_data.GetElemMap1();
    }
  };

  tribol::PackedPairJacobianContribs contrib( surface_fes_for( row_space ), surface_fes_for( col_space ),
                                              RedecompFESpaceForBlock( mesh_data, submesh_data, row_space ),
                                              RedecompFESpaceForBlock( mesh_data, submesh_data, col_space ),
                                              elem_map_for( row_space ), elem_map_for( col_space ) );

  const auto& row_map = ElemMapForBlock( mesh_data, row_space );
  const auto& col_map = ElemMapForBlock( mesh_data, col_space );
  if ( row_map.size() == 0 || col_map.size() == 0 ) {
    return contrib;
  }

  const int row_redecomp_elem = row_map[0];
  const int col_redecomp_elem = col_map[0];

  mfem::Array<int> row_dofs;
  mfem::Array<int> col_dofs;
  RedecompFESpaceForBlock( mesh_data, submesh_data, row_space ).GetElementVDofs( row_redecomp_elem, row_dofs );
  RedecompFESpaceForBlock( mesh_data, submesh_data, col_space ).GetElementVDofs( col_redecomp_elem, col_dofs );

  std::vector<double> values( static_cast<size_t>( row_dofs.Size() * col_dofs.Size() ), value );
  contrib.reserve( 1, static_cast<int>( values.size() ) );
  contrib.append( 0, 0, values.data(), static_cast<int>( values.size() ) );
  return contrib;
}

}  // namespace

TEST_F( MfemJacobianTest, computed_element_data_append_tracks_offsets )
{
  tribol::PackedPairJacobianContribs contrib;
  // Verifies that PackedPairJacobianContribs stores a variable-length list of element Jacobians
  // using a single flat value buffer and per-entry offsets.
  //
  // Pass conditions:
  // - Appending entries preserves (row_elem_id, col_elem_id) in order.
  // - value_offsets points at the correct start of each entry in jacobian_data.
  // - jacobian_data is the concatenation of all appended values.
  // - Optional metadata pointers remain null when the default constructor is used.
  //
  // reserve() uses a per-entry value count; this test appends blocks of sizes 2 and 3.
  contrib.reserve( 2, 3 );

  const double first[] = { 1.0, 2.0 };
  const double second[] = { 3.0, 4.0, 5.0 };

  contrib.append( 7, 8, first, 2 );
  contrib.append( 9, 10, second, 3 );

  EXPECT_EQ( contrib.row_surface_fes, nullptr );
  EXPECT_EQ( contrib.col_surface_fes, nullptr );
  EXPECT_EQ( contrib.row_elem_map, nullptr );
  EXPECT_EQ( contrib.col_elem_map, nullptr );
  EXPECT_EQ( contrib.numEntries(), 2 );
  ASSERT_EQ( contrib.row_elem_ids.size(), 2 );
  ASSERT_EQ( contrib.col_elem_ids.size(), 2 );
  ASSERT_EQ( contrib.value_offsets.size(), 2 );
  ASSERT_EQ( contrib.jacobian_data.size(), 5 );

  EXPECT_EQ( contrib.row_elem_ids[0], 7 );
  EXPECT_EQ( contrib.row_elem_ids[1], 9 );
  EXPECT_EQ( contrib.col_elem_ids[0], 8 );
  EXPECT_EQ( contrib.col_elem_ids[1], 10 );
  EXPECT_EQ( contrib.value_offsets[0], 0 );
  EXPECT_EQ( contrib.value_offsets[1], 2 );
  EXPECT_DOUBLE_EQ( contrib.jacobian_data[0], 1.0 );
  EXPECT_DOUBLE_EQ( contrib.jacobian_data[1], 2.0 );
  EXPECT_DOUBLE_EQ( contrib.jacobian_data[2], 3.0 );
  EXPECT_DOUBLE_EQ( contrib.jacobian_data[3], 4.0 );
  EXPECT_DOUBLE_EQ( contrib.jacobian_data[4], 5.0 );
}

namespace {

tribol::MethodData BuildSyntheticMethodData( const tribol::MfemMeshData& mesh_data,
                                             const tribol::MfemSubmeshData& submesh_data )
{
  // Populate one entry in every Tribol block so the public block-Jacobian
  // wrapper can be checked independently of the contact-physics assembly path.
  tribol::MethodData method_data;
  method_data.reserveBlockJ(
      { tribol::BlockSpace::MORTAR, tribol::BlockSpace::NONMORTAR, tribol::BlockSpace::LAGRANGE_MULTIPLIER }, 1 );

  const bool has_local_pair = mesh_data.GetElemMap1().size() > 0 && mesh_data.GetElemMap2().size() > 0;
  if ( !has_local_pair ) {
    return method_data;
  }

  auto& block_J = method_data.getBlockJ();
  auto& block_elem_ids = const_cast<tribol::ArrayT<tribol::ArrayT<int>>&>( method_data.getBlockJElementIds() );

  block_elem_ids[static_cast<int>( tribol::BlockSpace::MORTAR )].push_back( 0 );
  block_elem_ids[static_cast<int>( tribol::BlockSpace::NONMORTAR )].push_back( 0 );
  block_elem_ids[static_cast<int>( tribol::BlockSpace::LAGRANGE_MULTIPLIER )].push_back( 0 );

  const tribol::BlockSpace spaces[] = { tribol::BlockSpace::MORTAR, tribol::BlockSpace::NONMORTAR,
                                        tribol::BlockSpace::LAGRANGE_MULTIPLIER };
  const double values[3][3] = { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 0.0 } };

  for ( int i = 0; i < 3; ++i ) {
    for ( int j = 0; j < 3; ++j ) {
      const auto& row_fes = RedecompFESpaceForBlock( mesh_data, submesh_data, spaces[i] );
      const auto& col_fes = RedecompFESpaceForBlock( mesh_data, submesh_data, spaces[j] );

      mfem::Array<int> row_dofs;
      mfem::Array<int> col_dofs;
      row_fes.GetElementVDofs( ElemMapForBlock( mesh_data, spaces[i] )[0], row_dofs );
      col_fes.GetElementVDofs( ElemMapForBlock( mesh_data, spaces[j] )[0], col_dofs );

      mfem::DenseMatrix mat = ConstantDenseMatrix( row_dofs.Size(), col_dofs.Size(), values[i][j] );
      block_J( static_cast<int>( spaces[i] ), static_cast<int>( spaces[j] ) ).push_back( mat );
    }
  }

  return method_data;
}

std::vector<int> BuildInactiveDualTdofs( const tribol::MfemMeshData& mesh_data,
                                         const tribol::MfemSubmeshData& submesh_data )
{
  // Mirror the production single-mortar logic so the test checks the exact tdof set
  // that should receive identity entries in empty dual-dual blocks.
  const auto& submesh_fe_space = submesh_data.GetSubmeshFESpace();
  const auto& submesh = mesh_data.GetSubmesh();

  mfem::Array<int> attr_marker( submesh.attributes.Max() );
  attr_marker = 0;
  for ( auto nonmortar_attr : mesh_data.GetBoundaryAttribs2() ) {
    attr_marker[nonmortar_attr - 1] = 1;
  }

  mfem::Array<int> mortar_dof_marker( submesh_fe_space.GetVSize() );
  mortar_dof_marker = 1;
  for ( int e = 0; e < submesh.GetNE(); ++e ) {
    if ( attr_marker[submesh_fe_space.GetAttribute( e ) - 1] ) {
      mfem::Array<int> vdofs;
      submesh_fe_space.GetElementVDofs( e, vdofs );
      for ( int d = 0; d < vdofs.Size(); ++d ) {
        int k = vdofs[d];
        if ( k < 0 ) {
          k = -1 - k;
        }
        mortar_dof_marker[k] = 0;
      }
    }
  }

  mfem::Array<int> mortar_tdof_marker( submesh_fe_space.GetTrueVSize() );
  submesh_fe_space.GetRestrictionMatrix()->BooleanMult( mortar_dof_marker, mortar_tdof_marker );
  mfem::Array<int> mortar_tdof_list;
  mfem::FiniteElementSpace::MarkerToList( mortar_tdof_marker, mortar_tdof_list );

  std::vector<int> expected( mortar_tdof_list.Size() );
  for ( int i = 0; i < mortar_tdof_list.Size(); ++i ) {
    expected[static_cast<size_t>( i )] = mortar_tdof_list[i];
  }
  return expected;
}

void ExpectParMatricesNear( const mfem::HypreParMatrix& actual, const shared::ParSparseMat& expected,
                            double tol = 1e-12 )
{
  // Compare two parallel sparse matrices by subtracting and checking the maximum absolute entry.
  // This avoids relying on Hypre internal equality checks and keeps the failure signal simple.
  shared::ParSparseMatView lhs( const_cast<mfem::HypreParMatrix*>( &actual ) );
  shared::ParSparseMatView rhs( const_cast<mfem::HypreParMatrix*>( &expected.get() ) );
  auto diff = lhs - rhs;
  EXPECT_LT( MaxAbsMatrixEntry( shared::ParSparseMatView( &diff.get() ) ), tol );
}

}  // namespace

TEST_F( MfemJacobianTest, lor_or_submesh_jacobian_and_solver_composition_smoke )
{
  // Smoke-test the Jacobian transfer API end-to-end on a minimal synthetic contribution set.
  //
  // Pass conditions (when any rank owns a surface element pair):
  // - AssembleLorOrSubmeshJacobian() produces a non-empty sparse matrix on the surface space.
  // - The solver-visible Jacobian assembled via the explicit transfer chain is also non-empty.
  //
  // Pass conditions (when there are no local surface element pairs on a rank):
  // - Assembly still succeeds and produces correctly-sized (possibly empty) matrices.
  for ( const int order : { 1, 2 } ) {
    SCOPED_TRACE( ::testing::Message() << "order=" << order );
    WithJacobianData( order, []( const JacobianTestContext& ctx ) {
      auto* mesh_data = ctx.mesh_data;
      auto* jac_data = ctx.jac_data;

      std::vector<tribol::PackedPairJacobianContribs> contributions;

      if ( mesh_data->GetMesh1NE() > 0 ) {
        // Hex element has 8 nodes
        int num_dofs_per_elem = 8 * mesh_data->GetParentCoords().ParFESpace()->GetVDim();
        int mat_size = num_dofs_per_elem * num_dofs_per_elem;

        const bool use_lor = ( mesh_data->GetLORMesh() != nullptr );
        auto& primary_surface_fes = use_lor ? *mesh_data->GetLORMeshFESpace() : mesh_data->GetSubmeshFESpace();
        tribol::PackedPairJacobianContribs contrib(
            primary_surface_fes, primary_surface_fes, *mesh_data->GetRedecompResponse().FESpace(),
            *mesh_data->GetRedecompResponse().FESpace(), mesh_data->GetElemMap1(), mesh_data->GetElemMap1() );
        std::vector<double> values( static_cast<size_t>( mat_size ), 1.0 );
        contrib.reserve( 1, mat_size );
        contrib.append( 0, 0, values.data(), mat_size );

        contributions.push_back( contrib );
      } else {
        // Provide metadata even when there are no local elements so assembly can
        // still build correctly-sized empty matrices on all ranks.
        const bool use_lor = ( mesh_data->GetLORMesh() != nullptr );
        auto& primary_surface_fes = use_lor ? *mesh_data->GetLORMeshFESpace() : mesh_data->GetSubmeshFESpace();
        contributions.emplace_back(
            primary_surface_fes, primary_surface_fes, *mesh_data->GetRedecompResponse().FESpace(),
            *mesh_data->GetRedecompResponse().FESpace(), mesh_data->GetElemMap1(), mesh_data->GetElemMap1() );
      }

      auto lor_or_submesh_jacobian = jac_data->AssembleLorOrSubmeshJacobian( contributions );
      auto ParJ = AssembleSolverBlockJacobian( ctx, contributions, SolverBlock::Primary, SolverBlock::Primary );

      int local_pairs = 0;
      for ( const auto& contrib : contributions ) {
        local_pairs += contrib.numEntries();
      }
      int global_pairs = 0;
      MPI_Allreduce( &local_pairs, &global_pairs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

      if ( global_pairs > 0 ) {
        EXPECT_GT( lor_or_submesh_jacobian->NNZ(), 0 );
        EXPECT_GT( ParJ->NNZ(), 0 );
      } else {
        int n_ranks;
        MPI_Comm_size( MPI_COMM_WORLD, &n_ranks );
        if ( n_ranks == 1 ) {
          FAIL() << "No surface elements found on mesh 1 in serial run.";
        }
      }
    } );
  }
}

TEST_F( MfemJacobianTest, logical_jacobian_primary_dual_assembles_in_ho_and_lor )
{
  // Assembles a solver-visible primary->dual Jacobian block from a single constant element contribution.
  //
  // Pass conditions:
  // - The assembled matrix has dimensions (primary HO true-dofs) x (dual submesh true-dofs).
  // - If any rank owns an element pair, the resulting sparse matrix has nonzero entries.
  // - This holds for both HO-only and HO->LOR->submesh transfer paths (depending on configuration).
  for ( const int order : { 1, 2 } ) {
    SCOPED_TRACE( ::testing::Message() << "order=" << order );
    WithJacobianData( order, [&]( const JacobianTestContext& ctx ) {
      std::vector<tribol::PackedPairJacobianContribs> contributions{
          MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::MORTAR,
                                    tribol::BlockSpace::LAGRANGE_MULTIPLIER, 2.0 ) };

      auto mat = AssembleSolverBlockJacobian( ctx, contributions, SolverBlock::Primary, SolverBlock::Dual );

      EXPECT_EQ( mat->Height(), ctx.mesh_data->GetParentCoords().ParFESpace()->GetTrueVSize() );
      EXPECT_EQ( mat->Width(), ctx.submesh_data->GetSubmeshFESpace().GetTrueVSize() );

      const int local_pairs = contributions[0].row_elem_ids.size();
      int global_pairs = 0;
      MPI_Allreduce( &local_pairs, &global_pairs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
      if ( global_pairs > 0 ) {
        EXPECT_GT( mat->NNZ(), 0 );
      }
    } );
  }
}

TEST_F( MfemJacobianTest, logical_jacobian_dual_primary_assembles_in_ho_and_lor )
{
  // Assembles a solver-visible dual->primary Jacobian block from a single constant element contribution.
  //
  // Pass conditions:
  // - The assembled matrix has dimensions (dual submesh true-dofs) x (primary HO true-dofs).
  // - If any rank owns an element pair, the resulting sparse matrix has nonzero entries.
  // - This holds for both HO-only and HO->LOR->submesh transfer paths (depending on configuration).
  for ( const int order : { 1, 2 } ) {
    SCOPED_TRACE( ::testing::Message() << "order=" << order );
    WithJacobianData( order, [&]( const JacobianTestContext& ctx ) {
      std::vector<tribol::PackedPairJacobianContribs> contributions{
          MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::LAGRANGE_MULTIPLIER,
                                    tribol::BlockSpace::MORTAR, 2.5 ) };

      auto mat = AssembleSolverBlockJacobian( ctx, contributions, SolverBlock::Dual, SolverBlock::Primary );

      EXPECT_EQ( mat->Height(), ctx.submesh_data->GetSubmeshFESpace().GetTrueVSize() );
      EXPECT_EQ( mat->Width(), ctx.mesh_data->GetParentCoords().ParFESpace()->GetTrueVSize() );

      const int local_pairs = contributions[0].row_elem_ids.size();
      int global_pairs = 0;
      MPI_Allreduce( &local_pairs, &global_pairs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
      if ( global_pairs > 0 ) {
        EXPECT_GT( mat->NNZ(), 0 );
      }
    } );
  }
}

TEST_F( MfemJacobianTest, logical_jacobian_dual_dual_inactive_dofs_form_identity )
{
  // Verifies the single-mortar solver compatibility behavior for empty dual-dual blocks.
  //
  // In single-mortar mode, the solver expects inactive dual true-dofs (those not on the nonmortar side)
  // to form an identity block, even when there are no explicit dual-dual contact contributions.
  //
  // Pass conditions:
  // - The assembled dual-dual block is square on the dual (submesh) true-dof space.
  // - The diagonal is exactly 1.0 on the inactive dual tdofs and 0.0 elsewhere.
  WithJacobianData( 1, [&]( const JacobianTestContext& ctx ) {
    const bool use_lor = ( ctx.mesh_data->GetLORMesh() != nullptr );
    auto& dual_surface_fes = use_lor ? *ctx.submesh_data->GetLORMeshFESpace() : ctx.submesh_data->GetSubmeshFESpace();
    std::vector<tribol::PackedPairJacobianContribs> contributions{ tribol::PackedPairJacobianContribs(
        dual_surface_fes, dual_surface_fes, *ctx.submesh_data->GetRedecompGap().FESpace(),
        *ctx.submesh_data->GetRedecompGap().FESpace(), ctx.mesh_data->GetElemMap2(), ctx.mesh_data->GetElemMap2() ) };

    auto mat = AssembleSolverBlockJacobian( ctx, contributions, SolverBlock::Dual, SolverBlock::Dual );

    EXPECT_EQ( mat->Height(), ctx.submesh_data->GetSubmeshFESpace().GetTrueVSize() );
    EXPECT_EQ( mat->Width(), ctx.submesh_data->GetSubmeshFESpace().GetTrueVSize() );

    auto expected_inactive = BuildInactiveDualTdofs( *ctx.mesh_data, *ctx.submesh_data );
    std::vector<int> is_inactive( static_cast<size_t>( ctx.submesh_data->GetSubmeshFESpace().GetTrueVSize() ), 0 );
    for ( int tdof : expected_inactive ) {
      is_inactive[static_cast<size_t>( tdof )] = 1;
    }

    // Avoid copying mfem::HypreParMatrix (unsafe with Hypre GPU memory-location tracking).
    const mfem::HypreParMatrix& hypre = mat.get();
    for ( int i = 0; i < hypre.NumRows(); ++i ) {
      const double expected = is_inactive[static_cast<size_t>( i )] ? 1.0 : 0.0;
      EXPECT_DOUBLE_EQ( LocalDiagonalEntry( hypre, i ), expected );
    }
  } );
}

TEST_F( MfemJacobianTest, logical_jacobian_primary_primary_aggregates_mortar_and_nonmortar_contributions )
{
  // Verifies that mortar/nonmortar contribution partitions are summed into one solver-visible
  // primary-primary Jacobian block.
  //
  // Pass conditions:
  // - Assembling all four mortar/nonmortar contribution pairs together equals the sum of assembling
  //   each contribution separately (entrywise, up to tolerance).
  // - If any rank owns an element pair, the combined matrix is non-empty.
  for ( const int order : { 1, 2 } ) {
    SCOPED_TRACE( ::testing::Message() << "order=" << order );
    WithJacobianData( order, [&]( const JacobianTestContext& ctx ) {
      const auto mm = MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::MORTAR,
                                                tribol::BlockSpace::MORTAR, 1.0 );
      const auto mn = MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::MORTAR,
                                                tribol::BlockSpace::NONMORTAR, 2.0 );
      const auto nm = MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::NONMORTAR,
                                                tribol::BlockSpace::MORTAR, 3.0 );
      const auto nn = MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::NONMORTAR,
                                                tribol::BlockSpace::NONMORTAR, 4.0 );

      auto combined =
          AssembleSolverBlockJacobian( ctx, { mm, mn, nm, nn }, SolverBlock::Primary, SolverBlock::Primary );
      auto expected = AssembleSolverBlockJacobian( ctx, { mm }, SolverBlock::Primary, SolverBlock::Primary );
      expected += AssembleSolverBlockJacobian( ctx, { mn }, SolverBlock::Primary, SolverBlock::Primary );
      expected += AssembleSolverBlockJacobian( ctx, { nm }, SolverBlock::Primary, SolverBlock::Primary );
      expected += AssembleSolverBlockJacobian( ctx, { nn }, SolverBlock::Primary, SolverBlock::Primary );

      const int local_pairs =
          mm.row_elem_ids.size() + mn.row_elem_ids.size() + nm.row_elem_ids.size() + nn.row_elem_ids.size();
      int global_pairs = 0;
      MPI_Allreduce( &local_pairs, &global_pairs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
      if ( global_pairs > 0 ) {
        EXPECT_GT( combined->NNZ(), 0 );
      }

      auto diff = shared::ParSparseMatView( &combined.get() ) - shared::ParSparseMatView( &expected.get() );
      EXPECT_LT( MaxAbsMatrixEntry( shared::ParSparseMatView( &diff.get() ) ), 1e-12 );
    } );
  }
}

TEST_F( MfemJacobianTest, mfem_block_jacobian_preserves_block_values_and_layout )
{
  // Validates tribol::getMfemBlockJacobian(cs_id) independently of contact physics by feeding
  // it synthetic MethodData with distinct constant values in each Tribol block.
  //
  // Pass conditions:
  // - The returned mfem::BlockOperator has the expected 2x2 solver block layout (primary/dual).
  // - Each block has the correct dimensions.
  // - Each block's values match the expected solver-visible assembly for the corresponding inputs.
  WithJacobianData( 1, [&]( const JacobianTestContext& ctx ) {
    ASSERT_NE( ctx.cs, nullptr );
    ctx.cs->allocateMethodData();
    ASSERT_NE( ctx.cs->getMethodData(), nullptr );
    *ctx.cs->getMethodData() = BuildSyntheticMethodData( *ctx.mesh_data, *ctx.submesh_data );
    ctx.cs->getEnforcementOptions().lm_implicit_options.sparse_mode = tribol::SparseMode::MFEM_ELEMENT_DENSE;

    auto block_J = tribol::getMfemBlockJacobian( ctx.cs->getId() );

    auto expected_00 = AssembleSolverBlockJacobian(
        ctx,
        { MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::MORTAR,
                                    tribol::BlockSpace::MORTAR, 1.0 ),
          MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::MORTAR,
                                    tribol::BlockSpace::NONMORTAR, 2.0 ),
          MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::NONMORTAR,
                                    tribol::BlockSpace::MORTAR, 4.0 ),
          MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::NONMORTAR,
                                    tribol::BlockSpace::NONMORTAR, 5.0 ) },
        SolverBlock::Primary, SolverBlock::Primary );

    auto expected_01 = AssembleSolverBlockJacobian(
        ctx,
        { MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::MORTAR,
                                    tribol::BlockSpace::LAGRANGE_MULTIPLIER, 3.0 ),
          MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::NONMORTAR,
                                    tribol::BlockSpace::LAGRANGE_MULTIPLIER, 6.0 ) },
        SolverBlock::Primary, SolverBlock::Dual );

    auto expected_10 = AssembleSolverBlockJacobian(
        ctx,
        { MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::LAGRANGE_MULTIPLIER,
                                    tribol::BlockSpace::MORTAR, 7.0 ),
          MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::LAGRANGE_MULTIPLIER,
                                    tribol::BlockSpace::NONMORTAR, 8.0 ) },
        SolverBlock::Dual, SolverBlock::Primary );

    auto expected_11 = AssembleSolverBlockJacobian(
        ctx,
        { MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::LAGRANGE_MULTIPLIER,
                                    tribol::BlockSpace::LAGRANGE_MULTIPLIER, 0.0 ) },
        SolverBlock::Dual, SolverBlock::Dual );

    const auto* block_00 = dynamic_cast<const mfem::HypreParMatrix*>( &block_J->GetBlock( 0, 0 ) );
    const auto* block_01 = dynamic_cast<const mfem::HypreParMatrix*>( &block_J->GetBlock( 0, 1 ) );
    const auto* block_10 = dynamic_cast<const mfem::HypreParMatrix*>( &block_J->GetBlock( 1, 0 ) );
    const auto* block_11 = dynamic_cast<const mfem::HypreParMatrix*>( &block_J->GetBlock( 1, 1 ) );

    ASSERT_NE( block_00, nullptr );
    ASSERT_NE( block_01, nullptr );
    ASSERT_NE( block_10, nullptr );
    ASSERT_NE( block_11, nullptr );

    EXPECT_EQ( block_00->Height(), expected_00->Height() );
    EXPECT_EQ( block_00->Width(), expected_00->Width() );
    EXPECT_EQ( block_01->Height(), expected_01->Height() );
    EXPECT_EQ( block_01->Width(), expected_01->Width() );
    EXPECT_EQ( block_10->Height(), expected_10->Height() );
    EXPECT_EQ( block_10->Width(), expected_10->Width() );
    EXPECT_EQ( block_11->Height(), expected_11->Height() );
    EXPECT_EQ( block_11->Width(), expected_11->Width() );

    ExpectParMatricesNear( *block_00, expected_00 );
    ExpectParMatricesNear( *block_01, expected_01 );
    ExpectParMatricesNear( *block_10, expected_10 );
    ExpectParMatricesNear( *block_11, expected_11 );
  } );
}

class MfemLorTransferParamTest : public MfemJacobianTest, public testing::WithParamInterface<LorTransferParams> {};

TEST_P( MfemLorTransferParamTest, lor_transfer_matches_mfem )
{
  // Compares Tribol's assembled HO->LOR transfer matrix against MFEM's ForwardOperator()
  // for a sweep of (polynomial order, LOR factor, field kind).
  //
  // Pass conditions:
  // - The assembled transfer produced by MfemJacobianData matches MFEM's transfer action
  //   for representative input vectors (checked by CompareHoToLorTransfers()).
  //
  // Note: this is intended to catch regressions in the assembled transfer build; it does not
  // validate any contact-physics Jacobian assembly.
  const auto p = GetParam();

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

  if ( p.lor_factor != mesh_data->GetLORFactor() ) {
    tribol::setMfemLORFactor( cs_id, p.lor_factor );
  }

  tribol::updateMfemParallelDecomposition( n_ranks, true );

  auto* submesh_data = cs->getMfemSubmeshData();
  ASSERT_NE( submesh_data, nullptr );

  cs->setMfemJacobianData( std::make_unique<tribol::MfemJacobianData>( *mesh_data, *submesh_data ) );
  auto* jac_data = cs->getMfemJacobianData();
  ASSERT_NE( jac_data, nullptr );
  jac_data->UpdateJacobianXfer();

  const mfem::ParFiniteElementSpace* ho_fes = nullptr;
  const mfem::ParFiniteElementSpace* lor_fes = nullptr;
  std::unique_ptr<shared::ParSparseMat> T;

  if ( p.field == FieldKind::Displacement ) {
    ho_fes = &mesh_data->GetSubmeshFESpace();
    lor_fes = mesh_data->GetLORMeshFESpace();
    const auto* mat = jac_data->GetDisplacementHoToLorTransferMat();
    ASSERT_NE( mat, nullptr );
    T = std::make_unique<shared::ParSparseMat>( mat->Assemble() );
  } else {
    ho_fes = &submesh_data->GetSubmeshFESpace();
    lor_fes = submesh_data->GetLORMeshFESpace();
    const auto* mat = jac_data->GetLagrangeMultiplierHoToLorTransferMat();
    ASSERT_NE( mat, nullptr );
    T = std::make_unique<shared::ParSparseMat>( mat->Assemble() );
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

INSTANTIATE_TEST_SUITE_P( MfemLorTransfer, MfemLorTransferParamTest,
                          testing::Values( LorTransferParams{ 2, 2, FieldKind::Displacement },
                                           LorTransferParams{ 2, 3, FieldKind::Displacement },  // lor_factor > order
                                           LorTransferParams{ 2, 4, FieldKind::Displacement },  // lor_factor > order
                                           LorTransferParams{ 3, 2, FieldKind::Displacement },  // lor_factor < order
                                           LorTransferParams{ 3, 3, FieldKind::Displacement },
                                           LorTransferParams{ 3, 4, FieldKind::Displacement },  // lor_factor > order
                                           LorTransferParams{ 2, 2, FieldKind::LagrangeMultiplier },
                                           LorTransferParams{ 2, 3, FieldKind::LagrangeMultiplier },
                                           LorTransferParams{ 3, 3, FieldKind::LagrangeMultiplier },
                                           LorTransferParams{ 3, 4, FieldKind::LagrangeMultiplier } ),
                          ParamsToString );

int main( int argc, char* argv[] )
{
  // GTest + MFEM/Tribol integration test entrypoint. Explicit MPI init/finalize is required because the test suite
  // runs in MPI (including 2-rank tests in CI).
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
