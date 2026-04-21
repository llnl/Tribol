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

// RAII helper for toggling the LOR fallback environment variable inside one test.
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
  static int cs_id_next = 100;
  return cs_id_next++;
}

std::string ParamsToString( const testing::TestParamInfo<LorTransferParams>& info )
{
  const auto& p = info.param;
  const char* field = ( p.field == FieldKind::Displacement ) ? "disp" : "lm";
  return "p" + std::to_string( p.order ) + "_lor" + std::to_string( p.lor_factor ) + "_" + field;
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
  HYPRE_ParCSRMatrix csr = const_cast<mfem::HypreParMatrix&>( mat );
  auto* parcsr = (hypre_ParCSRMatrix*)csr;
  auto* diag = hypre_ParCSRMatrixDiag( parcsr );
  auto* I = hypre_CSRMatrixI( diag );
  auto* J = hypre_CSRMatrixJ( diag );
  auto* data = hypre_CSRMatrixData( diag );

  if ( I == nullptr || J == nullptr || data == nullptr ) {
    return 0.0;
  }

  for ( int jj = I[local_row]; jj < I[local_row + 1]; ++jj ) {
    if ( J[jj] == local_row ) {
      return data[jj];
    }
  }
  return 0.0;
}

mfem::DenseMatrix ConstantDenseMatrix( int rows, int cols, double value )
{
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

  cs->setMfemJacobianData(
      std::make_unique<tribol::MfemJacobianData>( *mesh_data, *submesh_data, cs->getContactMethod() ) );
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

tribol::ComputedElementData MakeConstantContribution( const tribol::MfemMeshData& mesh_data,
                                                      const tribol::MfemSubmeshData& submesh_data,
                                                      tribol::BlockSpace row_space, tribol::BlockSpace col_space,
                                                      double value )
{
  // Use one local element pair and fill its dense contribution with a constant so
  // assembled values stay easy to reason about in the tests below.
  tribol::ComputedElementData contrib( row_space, col_space );

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
  tribol::ComputedElementData contrib( tribol::BlockSpace::MORTAR, tribol::BlockSpace::NONMORTAR );
  contrib.reserve( 2, 5 );

  const double first[] = { 1.0, 2.0 };
  const double second[] = { 3.0, 4.0, 5.0 };

  contrib.append( 7, 8, first, 2 );
  contrib.append( 9, 10, second, 3 );

  EXPECT_EQ( contrib.row_space, tribol::BlockSpace::MORTAR );
  EXPECT_EQ( contrib.col_space, tribol::BlockSpace::NONMORTAR );
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

std::unique_ptr<mfem::HypreParMatrix> CloneHypre( const shared::ParSparseMat& mat )
{
  return std::make_unique<mfem::HypreParMatrix>( mat.get() );
}

void ExpectParMatricesNear( const mfem::HypreParMatrix& actual, const shared::ParSparseMat& expected,
                            double tol = 1e-12 )
{
  shared::ParSparseMatView lhs( const_cast<mfem::HypreParMatrix*>( &actual ) );
  shared::ParSparseMatView rhs( const_cast<mfem::HypreParMatrix*>( &expected.get() ) );
  auto diff = lhs - rhs;
  EXPECT_LT( MaxAbsMatrixEntry( shared::ParSparseMatView( &diff.get() ) ), tol );
}

}  // namespace

TEST_F( MfemJacobianTest, direct_jacobian_assembly )
{
  // The assembled convenience API should still match the new logical-operator path
  // in both linear and LOR cases.
  for ( const int order : { 1, 2 } ) {
    SCOPED_TRACE( ::testing::Message() << "order=" << order );
    WithJacobianData( order, []( const JacobianTestContext& ctx ) {
      auto* mesh_data = ctx.mesh_data;
      auto* jac_data = ctx.jac_data;

      std::vector<tribol::ComputedElementData> contributions;

      if ( mesh_data->GetMesh1NE() > 0 ) {
        // Hex element has 8 nodes
        int num_dofs_per_elem = 8 * mesh_data->GetParentCoords().ParFESpace()->GetVDim();
        int mat_size = num_dofs_per_elem * num_dofs_per_elem;

        tribol::ComputedElementData contrib( tribol::BlockSpace::MORTAR, tribol::BlockSpace::MORTAR );
        std::vector<double> values( static_cast<size_t>( mat_size ), 1.0 );
        contrib.reserve( 1, mat_size );
        contrib.append( 0, 0, values.data(), mat_size );

        contributions.push_back( contrib );
      }

      auto ParJ = jac_data->GetMfemJacobian( contributions );
      auto block_op = jac_data->BuildJacobianBlockOp( contributions );
      auto ParJ_from_op = block_op->Assemble();

      int local_contrib = contributions.size();
      int global_contrib = 0;
      MPI_Allreduce( &local_contrib, &global_contrib, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

      if ( global_contrib > 0 ) {
        EXPECT_GT( ParJ->NNZ(), 0 );
        auto diff = shared::ParSparseMatView( &ParJ.get() ) - shared::ParSparseMatView( &ParJ_from_op.get() );
        EXPECT_LT( MaxAbsMatrixEntry( shared::ParSparseMatView( &diff.get() ) ), 1e-12 );
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
  // Exercise the primary->dual branch in both the direct HO path and the LOR-mapped path.
  for ( const int order : { 1, 2 } ) {
    SCOPED_TRACE( ::testing::Message() << "order=" << order );
    WithJacobianData( order, [&]( const JacobianTestContext& ctx ) {
      std::vector<tribol::ComputedElementData> contributions{
          MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::MORTAR,
                                    tribol::BlockSpace::LAGRANGE_MULTIPLIER, 2.0 ) };

      auto block_op = ctx.jac_data->BuildJacobianBlockOp( contributions );
      auto mat = block_op->Assemble();

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
  // Exercise the dual->primary branch in both the direct HO path and the LOR-mapped path.
  for ( const int order : { 1, 2 } ) {
    SCOPED_TRACE( ::testing::Message() << "order=" << order );
    WithJacobianData( order, [&]( const JacobianTestContext& ctx ) {
      std::vector<tribol::ComputedElementData> contributions{
          MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::LAGRANGE_MULTIPLIER,
                                    tribol::BlockSpace::MORTAR, 2.5 ) };

      auto block_op = ctx.jac_data->BuildJacobianBlockOp( contributions );
      auto mat = block_op->Assemble();

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
  // Empty dual-dual blocks still need the inactive-LM identity rows used by the solver.
  WithJacobianData( 1, [&]( const JacobianTestContext& ctx ) {
    std::vector<tribol::ComputedElementData> contributions( 1 );
    contributions[0].row_space = tribol::BlockSpace::LAGRANGE_MULTIPLIER;
    contributions[0].col_space = tribol::BlockSpace::LAGRANGE_MULTIPLIER;

    auto block_op = ctx.jac_data->BuildJacobianBlockOp( contributions );
    auto mat = block_op->Assemble();

    EXPECT_EQ( mat->Height(), ctx.submesh_data->GetSubmeshFESpace().GetTrueVSize() );
    EXPECT_EQ( mat->Width(), ctx.submesh_data->GetSubmeshFESpace().GetTrueVSize() );

    auto expected_inactive = BuildInactiveDualTdofs( *ctx.mesh_data, *ctx.submesh_data );
    std::vector<int> is_inactive( static_cast<size_t>( ctx.submesh_data->GetSubmeshFESpace().GetTrueVSize() ), 0 );
    for ( int tdof : expected_inactive ) {
      is_inactive[static_cast<size_t>( tdof )] = 1;
    }

    const auto hypre = CloneHypre( mat );
    for ( int i = 0; i < hypre->NumRows(); ++i ) {
      const double expected = is_inactive[static_cast<size_t>( i )] ? 1.0 : 0.0;
      EXPECT_DOUBLE_EQ( LocalDiagonalEntry( *hypre, i ), expected );
    }
  } );
}

TEST_F( MfemJacobianTest, logical_jacobian_primary_primary_aggregates_mortar_and_nonmortar_contributions )
{
  // Mortar and nonmortar partitions are internal contribution channels; they should
  // add into one solver-visible primary-primary block.
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

      auto combined = ctx.jac_data->BuildJacobianBlockOp( { mm, mn, nm, nn } )->Assemble();
      auto expected = ctx.jac_data->BuildJacobianBlockOp( { mm } )->Assemble();
      expected += ctx.jac_data->BuildJacobianBlockOp( { mn } )->Assemble();
      expected += ctx.jac_data->BuildJacobianBlockOp( { nm } )->Assemble();
      expected += ctx.jac_data->BuildJacobianBlockOp( { nn } )->Assemble();

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
  // Check the public block-assembly wrapper independently of contact physics by
  // feeding it synthetic MethodData with distinct values in every Tribol block.
  WithJacobianData( 1, [&]( const JacobianTestContext& ctx ) {
    ASSERT_NE( ctx.cs, nullptr );
    ctx.cs->allocateMethodData();
    ASSERT_NE( ctx.cs->getMethodData(), nullptr );
    *ctx.cs->getMethodData() = BuildSyntheticMethodData( *ctx.mesh_data, *ctx.submesh_data );
    ctx.cs->getEnforcementOptions().lm_implicit_options.sparse_mode = tribol::SparseMode::MFEM_ELEMENT_DENSE;

    auto block_J = tribol::getMfemBlockJacobian( ctx.cs->getId() );

    auto expected_00 =
        ctx.jac_data
            ->BuildJacobianBlockOp(
                { MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::MORTAR,
                                            tribol::BlockSpace::MORTAR, 1.0 ),
                  MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::MORTAR,
                                            tribol::BlockSpace::NONMORTAR, 2.0 ),
                  MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::NONMORTAR,
                                            tribol::BlockSpace::MORTAR, 4.0 ),
                  MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::NONMORTAR,
                                            tribol::BlockSpace::NONMORTAR, 5.0 ) } )
            ->Assemble();
    auto expected_01 =
        ctx.jac_data
            ->BuildJacobianBlockOp(
                { MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::MORTAR,
                                            tribol::BlockSpace::LAGRANGE_MULTIPLIER, 3.0 ),
                  MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::NONMORTAR,
                                            tribol::BlockSpace::LAGRANGE_MULTIPLIER, 6.0 ) } )
            ->Assemble();
    auto expected_10 =
        ctx.jac_data
            ->BuildJacobianBlockOp(
                { MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::LAGRANGE_MULTIPLIER,
                                            tribol::BlockSpace::MORTAR, 7.0 ),
                  MakeConstantContribution( *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::LAGRANGE_MULTIPLIER,
                                            tribol::BlockSpace::NONMORTAR, 8.0 ) } )
            ->Assemble();
    auto expected_11 = ctx.jac_data
                           ->BuildJacobianBlockOp( { MakeConstantContribution(
                               *ctx.mesh_data, *ctx.submesh_data, tribol::BlockSpace::LAGRANGE_MULTIPLIER,
                               tribol::BlockSpace::LAGRANGE_MULTIPLIER, 0.0 ) } )
                           ->Assemble();

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
  // Sweep polynomial order, LOR factor, and field kind against MFEM's forward
  // transfer operator.
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
