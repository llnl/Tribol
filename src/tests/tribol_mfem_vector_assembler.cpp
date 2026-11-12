// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check host MFEM pair and nodal assembly with prescribed residuals, nodal quantities, and nonsymmetric
 * Jacobians, without contact physics. Nonmatching two- and three-edge surfaces exercise shared nodes, scalar submesh
 * transfers, three independent fields with signed values, and all matrix blocks. Tests run on one and two ranks, with
 * contact data on rank zero. Batch tests check accumulation after host calls, repeated batches, and an empty batch on
 * Sequential and OpenMP backends. Physics and Enzyme derivatives are checked by the formulation tests.
 * Separate force and matrix components check shared-node accumulation and combine their views in one batch kernel.
 * Independent scalar/coordinate matrix assemblers check both rectangular orientations, host and batch packing,
 * invalid layouts, and empty batches on the same nonmatching surfaces.
 */
#include <cmath>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/assembly/MfemVectorAssembler.hpp"
#include "tribol/assembly/PairEvaluationAssembly.hpp"
#include "tribol/assembly/MfemMatrixAssembler.hpp"

namespace {
using namespace tribol;
constexpr int dimension = 2;       ///< Physical components in this two-dimensional patch.
constexpr int nodes_per_edge = 2;  ///< Finite-element nodes on a linear edge.
constexpr int pair_size = 2;       ///< Elements in a pair.
constexpr int pair_dofs = dimension * nodes_per_edge * pair_size;  ///< Coordinate entries per pair.
constexpr RealT tolerance = 1.0e-10;  ///< Roundoff allowance for MPI sums and matrix-vector products.

/**
 * @brief Assign distinct residual-row values from physical position and component.
 * @param x Node x coordinate.
 * @param y Node y coordinate.
 * @param component Spatial component.
 * @return Row value for the prescribed residual and outer-product Jacobian.
 */
RealT rowValue( RealT x, RealT y, int component ) { return 1.0 + x + 2.0 * y + 3.0 * component; }

/**
 * @brief Assign column values different from the row values, so transposing the Jacobian changes its action.
 * @param x Node x coordinate.
 * @param y Node y coordinate.
 * @param component Spatial component.
 * @return Column value for the prescribed outer-product Jacobian.
 */
RealT columnValue( RealT x, RealT y, int component ) { return -2.0 + 3.0 * x - y - 4.0 * component; }

/** @brief Provide parent and Redecomp transfers without selecting a contact formulation. */
class MfemVectorAssemblerTest : public testing::Test {
 protected:
  /** @brief Build nonmatching linear surfaces and concentrate their Redecomp elements on rank zero. */
  void SetUp() override
  {
    mesh_ = std::make_unique<mfem::ParMesh>( static_cast<const mfem::ParMesh&>( shared::ParMeshBuilder(
        MPI_COMM_WORLD,
        shared::MeshBuilder::Unify( { shared::MeshBuilder::SquareMesh( 2, 1 ), shared::MeshBuilder::SquareMesh( 3, 1 )
                                                                                   .translate( { 0.0, 0.99 } )
                                                                                   .updateBdrAttrib( 1, 5 )
                                                                                   .updateBdrAttrib( 3, 6 ) } ) ) ) );
    fec_ = std::make_unique<mfem::H1_FECollection>( 1, dimension );
    fes_ = std::make_unique<mfem::ParFiniteElementSpace>( mesh_.get(), fec_.get(), dimension, mfem::Ordering::byVDIM );
    coords_ = std::make_unique<mfem::ParGridFunction>( fes_.get() );
    mesh_->GetNodes( *coords_ );
    registerMfemCouplingScheme( 0, 0, 1, *mesh_, *coords_, { 3 }, { 5 }, SURFACE_TO_SURFACE, NO_CASE, COMMON_PLANE,
                                FRICTIONLESS, PENALTY, BINNING_GRID, ExecutionMode::Sequential );
    setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
    updateMfemParallelDecomposition( 1, true );
    auto& cs = scheme();
    ASSERT_TRUE( cs.setMeshPointers() );
    cs.setMfemJacobianData( std::make_unique<MfemJacobianData>( *cs.getMfemMeshData() ) );
    cs.getMfemJacobianData()->UpdateJacobianXfer();
  }
  /** @brief Destroy coupling transfers before the MFEM mesh and fields they reference. */
  void TearDown() override { finalize(); }
  /** @brief Return the registered scheme. @return Scheme zero. */
  CouplingScheme& scheme() { return CouplingSchemeManager::getInstance().at( 0 ); }
  /** @brief Compute a global vector norm. @param v Local entries. @return Distributed Euclidean norm. */
  RealT norm( const mfem::Vector& v ) { return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, v, v ) ); }

  /** @brief Create scalar dual fields and their matrix transfers on the existing decomposition. */
  void prepareDualData()
  {
    auto& cs = scheme();
    auto& mesh_data = *cs.getMfemMeshData();
    auto submesh_data =
        std::make_unique<MfemSubmeshData>( mesh_data.GetSubmesh(), mesh_data.GetLORMesh(),
                                           std::make_unique<mfem::H1_FECollection>( 1, dimension ), 1, false );
    submesh_data->UpdateMfemSubmeshData( mesh_data.GetRedecompMesh(), true );
    cs.setMfemSubmeshData( std::move( submesh_data ) );
    cs.setMfemJacobianData( std::make_unique<MfemJacobianData>( mesh_data, *cs.getMfemSubmeshData() ) );
    cs.getMfemJacobianData()->UpdateJacobianXfer();
  }

  /** @brief List every pair of the nonmatching contact edges. @return Owned host pair list. */
  ArrayT<InterfacePair> pairs()
  {
    const auto mesh1 = scheme().getMesh1().getView(), mesh2 = scheme().getMesh2().getView();
    const IndexT count = mesh1.numberOfElements() * mesh2.numberOfElements();
    ArrayT<InterfacePair> result( count, count, mesh1.getAllocatorId() );
    IndexT index = 0;
    for ( IndexT element1 = 0; element1 < mesh1.numberOfElements(); ++element1 ) {
      for ( IndexT element2 = 0; element2 < mesh2.numberOfElements(); ++element2 ) {
        result[index++] = { element1, element2 };
      }
    }
    return result;
  }

  std::unique_ptr<mfem::ParMesh> mesh_;               ///< Parent patch mesh.
  std::unique_ptr<mfem::H1_FECollection> fec_;        ///< Linear coordinate basis.
  std::unique_ptr<mfem::ParFiniteElementSpace> fes_;  ///< Parent vector space.
  std::unique_ptr<mfem::ParGridFunction> coords_;     ///< Parent node coordinates.
};

/** @brief Share scalar-vector/matrix configuration and batch execution across nodal assembly tests. */
class TestNodalAssembly {
 public:
  /** @brief Configure independent components for each field. @param data Displacement transfers.
   * @param dual Scalar transfers. @param jacobian Matrix transfers. @param mesh1 Field geometry.
   * @param mesh2 Second coordinate geometry. @param field_count Independent fields. @param capacity Expected pairs. */
  TestNodalAssembly( const MfemMeshData& data, const MfemSubmeshData& dual, const MfemJacobianData& jacobian,
                     MeshData::Viewer mesh1, MeshData::Viewer mesh2, IndexT field_count, IndexT capacity )
  {
    vectors_.reserve( field_count );
    matrices_.reserve( field_count );
    for ( IndexT field = 0; field < field_count; ++field ) {
      vectors_.push_back( detail::makeNodalVectorAssembler( dual, mesh1 ) );
      matrices_.push_back( detail::makeNodalCoordinateMatrixAssembler( data, dual, jacobian, mesh1, mesh2, capacity ) );
    }
  }
  /** @brief Add host field values and derivatives. @param pair Geometry IDs. @param values Scalar values.
   * @param gradients Coordinate derivatives. */
  void addPair( InterfacePair pair, ArrayViewT<const RealT, 2> values, ArrayViewT<const RealT, 3> gradients )
  {
    for ( IndexT field = 0; field < static_cast<IndexT>( vectors_.size() ); ++field ) {
      vectors_[field].addPairFromReader( pair, [&]( IndexT row ) { return values( field, row ); } );
      matrices_[field].addPairFromReader(
          pair, [&]( IndexT row, IndexT column ) { return gradients( field, row, column ); } );
    }
  }
  /** @brief Compose batch accumulation. @tparam Fields Field count. @tparam Nodes Rows per field.
   * @tparam Dofs Pair coordinates. @tparam Values Scalar reader. @tparam Jacobians Matrix reader.
   * @tparam Exec Backend. @param pairs Geometry pairs. @param values Scalar reader. @param jacobians Matrix reader. */
  template <ExecutionMode Exec, IndexT Fields, IndexT Nodes, IndexT Dofs, typename Values, typename Jacobians>
  void addPairs( const ArrayT<InterfacePair>& pairs, Values values, Jacobians jacobians )
  {
    detail::addNodalValuesAndJacobians<Exec, Fields, Nodes, Dofs>( pairs, vectors_, matrices_, values, jacobians );
  }
  /** @brief Return independent field results. @return Owned vectors and matrices. */
  std::vector<AssembledField> assemble() const
  {
    std::vector<AssembledField> result;
    for ( IndexT field = 0; field < static_cast<IndexT>( vectors_.size() ); ++field ) {
      result.push_back( { vectors_[field].assemble(), matrices_[field].assemble() } );
    }
    return result;
  }

 private:
  std::vector<MfemVectorAssembler> vectors_;   ///< Scalar components, one per field.
  std::vector<MfemMatrixAssembler> matrices_;  ///< Coordinate Jacobian components, one per field.
};

// A nonsymmetric outer product has a known action that checks row/column ordering and every pair block independently.
TEST_F( MfemVectorAssemblerTest, assembles_prescribed_residual_and_nonsymmetric_jacobian )
{
  mfem::ParGridFunction direction( fes_.get() );
  mfem::VectorFunctionCoefficient direction_coefficient( dimension, []( const mfem::Vector& x, mfem::Vector& value ) {
    value.SetSize( dimension );
    value[0] = 2.0 + x[0] - 0.5 * x[1];
    value[1] = -1.0 + 0.25 * x[0] + 2.0 * x[1];
  } );
  direction.ProjectCoefficient( direction_coefficient );
  mfem::HypreParVector true_direction( fes_.get() );
  direction.GetTrueDofs( true_direction );

  // Every edge is paired with every edge on the other surface. Count each element's column contribution once;
  // each row element then receives its own column contribution once per opposite element plus that surface's sum.
  MdArray<int, pair_size> local_counts{}, counts{};
  MdArray<RealT, pair_size> local_column_sums{}, column_sums{};
  for ( int boundary = 0; boundary < mesh_->GetNBE(); ++boundary ) {
    const int attribute = mesh_->GetBdrAttribute( boundary );
    if ( attribute != 3 && attribute != 5 ) {
      continue;
    }
    const int side = attribute == 3 ? 0 : 1;
    ++local_counts[side];
    mfem::Array<int> nodes;
    fes_->GetBdrElementDofs( boundary, nodes );
    for ( const int node : nodes ) {
      const RealT x = ( *coords_ )[fes_->DofToVDof( node, 0 )];
      const RealT y = ( *coords_ )[fes_->DofToVDof( node, 1 )];
      for ( int component = 0; component < dimension; ++component ) {
        local_column_sums[side] += columnValue( x, y, component ) * direction[fes_->DofToVDof( node, component )];
      }
    }
  }
  MPI_Allreduce( local_counts.data(), counts.data(), pair_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( local_column_sums.data(), column_sums.data(), pair_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  mfem::ParLinearForm expected_residual( fes_.get() ), expected_action( fes_.get() );
  expected_residual = 0.0;
  expected_action = 0.0;
  for ( int boundary = 0; boundary < mesh_->GetNBE(); ++boundary ) {
    const int attribute = mesh_->GetBdrAttribute( boundary );
    if ( attribute != 3 && attribute != 5 ) {
      continue;
    }
    const int side = attribute == 3 ? 0 : 1;
    mfem::Array<int> nodes;
    fes_->GetBdrElementDofs( boundary, nodes );
    RealT element_column_sum = 0.0;
    for ( const int node : nodes ) {
      const RealT x = ( *coords_ )[fes_->DofToVDof( node, 0 )];
      const RealT y = ( *coords_ )[fes_->DofToVDof( node, 1 )];
      for ( int component = 0; component < dimension; ++component ) {
        element_column_sum += columnValue( x, y, component ) * direction[fes_->DofToVDof( node, component )];
      }
    }
    const RealT pair_column_sum = counts[1 - side] * element_column_sum + column_sums[1 - side];
    for ( const int node : nodes ) {
      const RealT x = ( *coords_ )[fes_->DofToVDof( node, 0 )];
      const RealT y = ( *coords_ )[fes_->DofToVDof( node, 1 )];
      for ( int component = 0; component < dimension; ++component ) {
        const int dof = fes_->DofToVDof( node, component );
        expected_residual[dof] += rowValue( x, y, component ) * counts[1 - side];
        expected_action[dof] += rowValue( x, y, component ) * pair_column_sum;
      }
    }
  }
  std::unique_ptr<mfem::HypreParVector> residual_reference( expected_residual.ParallelAssemble() );
  std::unique_ptr<mfem::HypreParVector> action_reference( expected_action.ParallelAssemble() );

  auto& cs = scheme();
  const auto mesh1 = cs.getMesh1().getView(), mesh2 = cs.getMesh2().getView();
  ContactFormulationResult result;
  {
    const IndexT pair_count = mesh1.numberOfElements() * mesh2.numberOfElements();
    auto assembler = detail::makeDisplacementVectorAssembler( *cs.getMfemMeshData(), mesh1, mesh2 );
    auto matrix = detail::makeCoordinateMatrixAssembler( *cs.getMfemMeshData(), *cs.getMfemJacobianData(), mesh1, mesh2,
                                                         pair_count );
    MdArray<RealT, pair_dofs> residual{}, columns{};
    MdArray<RealT, pair_dofs, pair_dofs> jacobian{};
    for ( IndexT element1 = 0; element1 < mesh1.numberOfElements(); ++element1 ) {
      for ( IndexT element2 = 0; element2 < mesh2.numberOfElements(); ++element2 ) {
        std::array<MdArray<RealT, nodes_per_edge, dimension>, pair_size> coords;
        mesh1.getFaceCoords( element1, coords[0].data() );
        mesh2.getFaceCoords( element2, coords[1].data() );
        for ( int side = 0; side < pair_size; ++side ) {
          for ( int node = 0; node < nodes_per_edge; ++node ) {
            for ( int component = 0; component < dimension; ++component ) {
              const int dof = ( side * nodes_per_edge + node ) * dimension + component;
              residual[dof] = rowValue( coords[side]( node, 0 ), coords[side]( node, 1 ), component );
              columns[dof] = columnValue( coords[side]( node, 0 ), coords[side]( node, 1 ), component );
            }
          }
        }
        for ( int row = 0; row < pair_dofs; ++row ) {
          for ( int column = 0; column < pair_dofs; ++column ) {
            jacobian( row, column ) = residual[row] * columns[column];
          }
        }
        assembler.addPair( { element1, element2 }, { residual.data(), residual.size() } );
        matrix.addPair( { element1, element2 }, { jacobian.data(), { pair_dofs, pair_dofs } } );
      }
    }
    result.force.reset( assembler.assemble().release() );
    result.df_dx.reset( matrix.assemble().release() );
    residual[0] = 0.0;
    jacobian[0] = 0.0;
  }
  ASSERT_NE( result.force, nullptr );
  ASSERT_NE( result.df_dx, nullptr );
  EXPECT_FALSE( result.energy.has_value() );
  EXPECT_FALSE( result.time_step.has_value() );
  EXPECT_EQ( result.gap, nullptr );
  EXPECT_EQ( result.pressure, nullptr );
  EXPECT_EQ( result.dg_dx, nullptr );
  EXPECT_EQ( result.df_dp, nullptr );
  auto error = *result.force;
  error -= *residual_reference;
  EXPECT_LT( norm( error ), tolerance );
  mfem::Vector action( fes_->GetTrueVSize() );
  result.df_dx->Mult( true_direction, action );
  action -= *action_reference;
  EXPECT_LT( norm( action ), tolerance );
  int rank = 0, ranks = 0;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &ranks );
  if ( ranks == 2 && rank == 1 ) {
    EXPECT_EQ( mesh1.numberOfElements() + mesh2.numberOfElements(), 0 );
  }
}

// Empty assembly must return usable zero outputs, and rejected input shapes must not add any partial contribution.
TEST_F( MfemVectorAssemblerTest, returns_zero_outputs_for_empty_or_rejected_input )
{
  auto& cs = scheme();
  const auto mesh1 = cs.getMesh1().getView(), mesh2 = cs.getMesh2().getView();
  auto assembler = detail::makeDisplacementVectorAssembler( *cs.getMfemMeshData(), mesh1, mesh2 );
  auto matrix =
      detail::makeCoordinateMatrixAssembler( *cs.getMfemMeshData(), *cs.getMfemJacobianData(), mesh1, mesh2, 0 );
  MdArray<RealT, pair_dofs> residual{};
  MdArray<RealT, pair_dofs, pair_dofs> jacobian{};
  residual[0] = 1.0;
  EXPECT_THROW( assembler.addPair( { 0, 0 }, { residual.data(), pair_dofs - 1 } ), std::invalid_argument );
  EXPECT_THROW( matrix.addPair( { 0, 0 }, { jacobian.data(), { pair_dofs - 1, pair_dofs } } ), std::invalid_argument );
  const auto force = assembler.assemble();
  const auto df_dx = matrix.assemble();
  EXPECT_EQ( force.size(), fes_->GetTrueVSize() );
  EXPECT_EQ( norm( force.get() ), 0.0 );
  mfem::Vector direction( fes_->GetTrueVSize() ), action( fes_->GetTrueVSize() );
  direction = 1.0;
  df_dx->Mult( direction, action );
  EXPECT_EQ( norm( action ), 0.0 );
}

// Three signed scalar fields retain their values and gradients without normalization or contact activation.
TEST_F( MfemVectorAssemblerTest, assembles_scalar_fields_and_dual_transfers )
{
  prepareDualData();
  auto& cs = scheme();
  const auto mesh1 = cs.getMesh1().getView(), mesh2 = cs.getMesh2().getView();
  auto& dual_data = *cs.getMfemSubmeshData();
  const auto& dual_fes = dual_data.GetSubmeshFESpace();
  shared::ParVector field( const_cast<mfem::ParFiniteElementSpace*>( &dual_fes ) );
  field.fill( 2.5 );
  const auto redecomp_field = dual_data.TransferTrueDofsToRedecomp( field.get() );
  for ( int dof = 0; dof < redecomp_field.Size(); ++dof ) {
    EXPECT_NEAR( redecomp_field[dof], 2.5, tolerance );
  }
  mfem::Vector wrong_size( dual_fes.GetTrueVSize() + 1 );
  EXPECT_THROW( dual_data.TransferTrueDofsToRedecomp( wrong_size ), std::invalid_argument );

  constexpr IndexT field_count = 3;
  const MdArray<RealT, field_count> factors{ 1.0, -2.0, 3.0 };
  const auto pair_list = pairs();
  const int local_pairs = pair_list.size();
  int global_pairs = 0;
  MPI_Allreduce( &local_pairs, &global_pairs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  std::vector<AssembledField> result;
  {
    TestNodalAssembly assembler( *cs.getMfemMeshData(), dual_data, *cs.getMfemJacobianData(), mesh1, mesh2, field_count,
                                 pair_list.size() );
    MdArray<RealT, field_count, nodes_per_edge> values{};
    MdArray<RealT, field_count, nodes_per_edge, pair_dofs> gradients{};
    for ( IndexT field_index = 0; field_index < field_count; ++field_index ) {
      for ( IndexT node = 0; node < nodes_per_edge; ++node ) {
        values( field_index, node ) = factors[field_index] * ( node + 1.0 );
        for ( IndexT column = 0; column < pair_dofs; ++column ) {
          gradients( field_index, node, column ) = values( field_index, node ) * ( column + 1.0 );
        }
      }
    }
    for ( const auto pair : pair_list ) {
      assembler.addPair( pair, { values.data(), { field_count, nodes_per_edge } },
                         { gradients.data(), { field_count, nodes_per_edge, pair_dofs } } );
    }
    values[0] = 0.0;
    gradients[0] = 0.0;
    result = assembler.assemble();
  }
  ASSERT_EQ( result.size(), field_count );
  mfem::Vector direction( fes_->GetTrueVSize() ), action( dual_fes.GetTrueVSize() );
  direction = 1.0;
  // Local node weights sum to three, and the coordinate-column weights sum to 8*9/2.
  constexpr RealT node_weight_sum = 3.0;
  constexpr RealT column_weight_sum = pair_dofs * ( pair_dofs + 1 ) / 2.0;
  for ( IndexT field_index = 0; field_index < field_count; ++field_index ) {
    SCOPED_TRACE( field_index );
    const RealT local_sum = result[field_index].values.get().Sum();
    RealT global_sum = 0.0;
    MPI_Allreduce( &local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    const RealT expected = global_pairs * node_weight_sum * factors[field_index];
    EXPECT_NEAR( global_sum, expected, tolerance );
    result[field_index].dvalues_dx->Mult( direction, action );
    const RealT local_action = action.Sum();
    RealT global_action = 0.0;
    MPI_Allreduce( &local_action, &global_action, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    EXPECT_NEAR( global_action, expected * column_weight_sum, tolerance );
  }
}

// Scalar vector accumulation needs no Jacobian data. Invalid input sizes and view block counts add nothing.
TEST_F( MfemVectorAssemblerTest, scalar_values_need_no_jacobian_data )
{
  prepareDualData();
  auto& cs = scheme();
  cs.setMfemJacobianData( nullptr );
  const auto& dual = *cs.getMfemSubmeshData();
  const auto mesh1 = cs.getMesh1().getView();
  auto assembler = detail::makeNodalVectorAssembler( dual, mesh1 );
  MdArray<RealT, nodes_per_edge> values{};
  EXPECT_THROW( assembler.addPair( { 0, 0 }, { values.data(), nodes_per_edge - 1 } ), std::invalid_argument );
  EXPECT_THROW( ( assembler.view<ExecutionMode::Sequential, 2>() ), std::invalid_argument );
  const auto zero = assembler.assemble();
  EXPECT_EQ( norm( zero.get() ), 0.0 );
  const auto pair_list = pairs();
  assembler.addPairs<ExecutionMode::Sequential, nodes_per_edge, 1>(
      pair_list, [] TRIBOL_HOST_DEVICE( IndexT, IndexT row ) { return row + 1.0; } );
  const auto result = assembler.assemble();
  const RealT local_sum = result.get().Sum();
  RealT global_sum = 0.0;
  const int local_pairs = pair_list.size();
  int global_pairs = 0;
  MPI_Allreduce( &local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_pairs, &global_pairs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( global_sum, 3.0 * global_pairs, tolerance );
}

/** @brief Exercise host batch kernels with each enabled host execution mode. */
class MfemBatchAssemblerTest : public MfemVectorAssemblerTest, public testing::WithParamInterface<ExecutionMode> {};

// Rectangular packing must transfer scalar and coordinate spaces in either orientation, including nonmatching element
// maps and a rank with no local pairs. Host and execution-space batches must accumulate the same nonsymmetric entries.
TEST_P( MfemBatchAssemblerTest, assembles_rectangular_matrices_in_both_orientations )
{
  dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    prepareDualData();
    auto& cs = scheme();
    const auto& data = *cs.getMfemMeshData();
    const auto& dual = *cs.getMfemSubmeshData();
    const auto& transfer = *cs.getMfemJacobianData();
    const auto pair_list = pairs();
    const auto& map1 = data.GetElemMap1();
    const auto& map2 = data.GetElemMap2();
    const MfemMatrixSpace scalar{ transfer.SubmeshPath(), dual.GetRedecompFESpace() };
    const MfemMatrixSpace coordinate{ transfer.ParentPath(), *data.GetRedecompResponse().FESpace() };
    constexpr IndexT element_dofs = nodes_per_edge * dimension;
    const std::vector<MfemMatrixBlock> forward_blocks{
        { 0, 0, nodes_per_edge, nodes_per_edge, 0, 0, map1, map1 },
        { 0, 1, nodes_per_edge, nodes_per_edge, 0, element_dofs, map1, map2 } };
    const std::vector<MfemMatrixBlock> reverse_blocks{
        { 0, 0, nodes_per_edge, nodes_per_edge, 0, 0, map1, map1 },
        { 1, 0, nodes_per_edge, nodes_per_edge, element_dofs, 0, map2, map1 } };
    MfemMatrixAssembler forward( scalar, coordinate, forward_blocks, 0 );
    MfemMatrixAssembler reverse( coordinate, scalar, reverse_blocks, 0 );
    MfemMatrixAssembler reference( scalar, coordinate, forward_blocks, 0 );
    const ArrayT<InterfacePair> empty( 0, 0, pair_list.getAllocatorID() );

    // Rejected shapes and scratch from a different assembler must leave the accumulated matrix empty.
    MdArray<RealT, 1, 1> wrong_shape{};
    EXPECT_THROW( forward.addPair( { 0, 0 }, { wrong_shape.data(), { 1, 1 } } ), std::invalid_argument );
    EXPECT_THROW( ( forward.makeBatch<3, element_dofs, 2>( 0, pair_list.getAllocatorID() ) ), std::invalid_argument );
    auto foreign_batch = reverse.makeBatch<element_dofs, nodes_per_edge, 2>( 0, pair_list.getAllocatorID() );
    EXPECT_THROW( forward.addBatch( foreign_batch ), std::invalid_argument );
    auto invalid_blocks = forward_blocks;
    invalid_blocks[0].row_geometry = 2;
    EXPECT_THROW( MfemMatrixAssembler( scalar, coordinate, invalid_blocks, 0 ), std::invalid_argument );
    EXPECT_THROW( MfemMatrixAssembler( scalar, coordinate, forward_blocks, -1 ), std::invalid_argument );
    const auto zero = forward.assemble();
    mfem::Vector coordinate_direction( fes_->GetTrueVSize() ), scalar_action( zero->Height() );
    coordinate_direction = 1.0;
    zero->Mult( coordinate_direction, scalar_action );
    EXPECT_EQ( norm( scalar_action ), 0.0 );

    for ( IndexT batch = 0; batch < 2; ++batch ) {
      const RealT scale = batch + 1.0;
      const auto read_forward = [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT node, IndexT coordinate ) {
        return scale * ( node + 1.0 + ( coordinate + 2.0 ) * ( pair + 1.0 ) );
      };
      const auto read_reverse = [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT coordinate, IndexT node ) {
        return read_forward( pair, node, coordinate );
      };
      for ( IndexT pair = 0; pair < pair_list.size(); ++pair ) {
        MdArray<RealT, nodes_per_edge, pair_dofs> matrix{};
        MdArray<RealT, pair_dofs, nodes_per_edge> transposed{};
        for ( IndexT node = 0; node < nodes_per_edge; ++node ) {
          for ( IndexT column = 0; column < pair_dofs; ++column ) {
            const RealT entry = read_forward( pair, node, column );
            matrix( node, column ) = entry;
            transposed( column, node ) = entry;
          }
        }
        reference.addPair( pair_list[pair], { matrix.data(), { nodes_per_edge, pair_dofs } } );
        if ( batch == 0 ) {
          forward.addPair( pair_list[pair], { matrix.data(), { nodes_per_edge, pair_dofs } } );
          reverse.addPair( pair_list[pair], { transposed.data(), { pair_dofs, nodes_per_edge } } );
        }
      }
      forward.addPairs<Exec, nodes_per_edge, element_dofs, 2>( empty, read_forward );
      reverse.addPairs<Exec, element_dofs, nodes_per_edge, 2>( empty, read_reverse );
      if ( batch != 0 ) {
        forward.addPairs<Exec, nodes_per_edge, element_dofs, 2>( pair_list, read_forward );
        reverse.addPairs<Exec, element_dofs, nodes_per_edge, 2>( pair_list, read_reverse );
      }
    }
    const auto expected_forward = reference.assemble();
    const auto expected_reverse = expected_forward.transpose();
    const auto actual_forward = forward.assemble();
    const auto actual_reverse = reverse.assemble();
    EXPECT_EQ( actual_forward->Height(), dual.GetSubmeshFESpace().GetTrueVSize() );
    EXPECT_EQ( actual_forward->Width(), fes_->GetTrueVSize() );
    EXPECT_EQ( actual_reverse->Height(), fes_->GetTrueVSize() );
    EXPECT_EQ( actual_reverse->Width(), dual.GetSubmeshFESpace().GetTrueVSize() );
    mfem::Vector expected_scalar( actual_forward->Height() );
    for ( int dof = 0; dof < coordinate_direction.Size(); ++dof ) {
      coordinate_direction[dof] = 1.0 + 0.1 * dof;
    }
    expected_forward->Mult( coordinate_direction, expected_scalar );
    actual_forward->Mult( coordinate_direction, scalar_action );
    scalar_action -= expected_scalar;
    EXPECT_LT( norm( scalar_action ), tolerance );
    mfem::Vector scalar_direction( actual_reverse->Width() ), actual_force( actual_reverse->Height() ),
        expected_force( actual_reverse->Height() );
    for ( int dof = 0; dof < scalar_direction.Size(); ++dof ) {
      scalar_direction[dof] = 1.0 + 0.2 * dof;
    }
    expected_reverse->Mult( scalar_direction, expected_force );
    actual_reverse->Mult( scalar_direction, actual_force );
    EXPECT_GT( norm( expected_force ), 0.0 );
    actual_force -= expected_force;
    EXPECT_LT( norm( actual_force ), tolerance );
  } );
}

// Distinct batch entries and an intervening empty batch must preserve all four matrix blocks and accumulated forces.
TEST_P( MfemBatchAssemblerTest, accumulates_residuals_and_jacobians_across_batches )
{
  dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    auto& cs = scheme();
    const auto& data = *cs.getMfemMeshData();
    const auto& transfers = *cs.getMfemJacobianData();
    const auto mesh1 = cs.getMesh1().getView(), mesh2 = cs.getMesh2().getView();
    const auto pair_list = pairs();
    auto reference = detail::makeDisplacementVectorAssembler( data, mesh1, mesh2 );
    auto composed = detail::makeDisplacementVectorAssembler( data, mesh1, mesh2 );
    auto force_only = detail::makeDisplacementVectorAssembler( data, mesh1, mesh2 );
    auto reference_matrix = detail::makeCoordinateMatrixAssembler( data, transfers, mesh1, mesh2, 0 );
    auto composed_matrix = detail::makeCoordinateMatrixAssembler( data, transfers, mesh1, mesh2, 0 );
    auto matrix_only = detail::makeCoordinateMatrixAssembler( data, transfers, mesh1, mesh2, 0 );
    const auto initial_force = force_only.assemble();
    EXPECT_EQ( norm( initial_force.get() ), 0.0 );
    const ArrayT<InterfacePair> empty( 0, 0, pair_list.getAllocatorID() );
    const auto zero_force = [] TRIBOL_HOST_DEVICE( IndexT, IndexT ) { return 0.0; };
    EXPECT_THROW( ( force_only.addPairs<Exec, pair_dofs - 1, 2>( empty, zero_force ) ), std::invalid_argument );
    constexpr IndexT element_dofs = pair_dofs / 2;
    for ( int batch = 0; batch < 3; ++batch ) {
      const RealT scale = batch + 1.0;
      MdArray<RealT, pair_dofs> residual{};
      MdArray<RealT, pair_dofs, pair_dofs> jacobian{};
      for ( IndexT pair = 0; pair < pair_list.size(); ++pair ) {
        for ( int row = 0; row < pair_dofs; ++row ) {
          residual[row] = scale * ( row + 1.0 );
          for ( int column = 0; column < pair_dofs; ++column ) {
            jacobian( row, column ) = scale * ( row + 1.0 ) * ( column + 2.0 );
          }
        }
        reference.addPair( pair_list[pair], { residual.data(), pair_dofs } );
        reference_matrix.addPair( pair_list[pair], { jacobian.data(), { pair_dofs, pair_dofs } } );
        if ( batch == 0 ) {
          composed.addPair( pair_list[pair], { residual.data(), pair_dofs } );
          composed_matrix.addPair( pair_list[pair], { jacobian.data(), { pair_dofs, pair_dofs } } );
          matrix_only.addPair( pair_list[pair], { jacobian.data(), { pair_dofs, pair_dofs } } );
        }
      }
      const auto read_force = [=] TRIBOL_HOST_DEVICE( IndexT, IndexT row ) { return scale * ( row + 1.0 ); };
      const auto read_jacobian = [=] TRIBOL_HOST_DEVICE( IndexT, IndexT row, IndexT column ) {
        return scale * ( row + 1.0 ) * ( column + 2.0 );
      };
      force_only.addPairs<Exec, pair_dofs, 2>( empty, read_force );
      force_only.addPairs<Exec, pair_dofs, 2>( pair_list, read_force );
      if ( batch != 0 ) {
        matrix_only.addPairs<Exec, element_dofs, element_dofs, 4>( empty, read_jacobian );
        matrix_only.addPairs<Exec, element_dofs, element_dofs, 4>( pair_list, read_jacobian );
        composed.addPairs<Exec, pair_dofs, 2>( empty, read_force );
        composed_matrix.addPairs<Exec, element_dofs, element_dofs, 4>( empty, read_jacobian );
        auto matrix_batch =
            composed_matrix.makeBatch<element_dofs, element_dofs, 4>( pair_list.size(), pair_list.getAllocatorID() );
        const auto matrix_view = matrix_batch.view();
        const auto force_view = composed.view<Exec, 2>();
        const auto pairs_view = pair_list.view();
        forAllExec( GetParam(), pair_list.size(), [=] TRIBOL_HOST_DEVICE( IndexT pair ) {
          force_view.addPair( pair, pairs_view[pair], read_force );
          matrix_view.packPair( pair, pairs_view[pair], read_jacobian );
        } );
        composed_matrix.addBatch( matrix_batch );
      }
    }
    const auto expected = reference.assemble();
    auto force_result = force_only.assemble();
    force_result.get().Add( -1.0, expected.get() );
    EXPECT_LT( norm( force_result.get() ), tolerance );
    EXPECT_EQ( norm( initial_force.get() ), 0.0 );
    EXPECT_GT( norm( expected.get() ), 0.0 );
    auto result = composed.assemble();
    result.get().Add( -1.0, expected.get() );
    EXPECT_LT( norm( result.get() ), tolerance );
    mfem::Vector direction( fes_->GetTrueVSize() ), expected_action( fes_->GetTrueVSize() ),
        action( fes_->GetTrueVSize() );
    direction = 1.0;
    reference_matrix.assemble()->Mult( direction, expected_action );
    composed_matrix.assemble()->Mult( direction, action );
    action -= expected_action;
    EXPECT_LT( norm( action ), tolerance );
    matrix_only.assemble()->Mult( direction, action );
    action -= expected_action;
    EXPECT_LT( norm( action ), tolerance );
  } );
}

// Three independent fields and their rectangular Jacobians retain host contributions and every subsequent batch.
TEST_P( MfemBatchAssemblerTest, accumulates_scalar_fields_and_gradients_across_batches )
{
  dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    prepareDualData();
    auto& cs = scheme();
    const auto mesh1 = cs.getMesh1().getView(), mesh2 = cs.getMesh2().getView();
    const auto pair_list = pairs();
    constexpr IndexT field_count = 3;
    TestNodalAssembly reference( *cs.getMfemMeshData(), *cs.getMfemSubmeshData(), *cs.getMfemJacobianData(), mesh1,
                                 mesh2, field_count, 0 );
    TestNodalAssembly batched( *cs.getMfemMeshData(), *cs.getMfemSubmeshData(), *cs.getMfemJacobianData(), mesh1, mesh2,
                               field_count, 0 );
    const ArrayT<InterfacePair> empty( 0, 0, pair_list.getAllocatorID() );
    for ( int batch = 0; batch < 3; ++batch ) {
      const RealT scale = batch + 1.0;
      MdArray<RealT, field_count, nodes_per_edge> values{};
      MdArray<RealT, field_count, nodes_per_edge, pair_dofs> gradients{};
      for ( IndexT field = 0; field < field_count; ++field ) {
        for ( IndexT node = 0; node < nodes_per_edge; ++node ) {
          values( field, node ) = scale * ( field + 1.0 ) * ( node + 1.0 );
          for ( IndexT column = 0; column < pair_dofs; ++column ) {
            gradients( field, node, column ) = values( field, node ) * ( column + 2.0 );
          }
        }
      }
      for ( const auto pair : pair_list ) {
        reference.addPair( pair, { values.data(), { field_count, nodes_per_edge } },
                           { gradients.data(), { field_count, nodes_per_edge, pair_dofs } } );
        if ( batch == 0 ) {
          batched.addPair( pair, { values.data(), { field_count, nodes_per_edge } },
                           { gradients.data(), { field_count, nodes_per_edge, pair_dofs } } );
        }
      }
      if ( batch != 0 ) {
        const auto read_value = [=] TRIBOL_HOST_DEVICE( IndexT, IndexT field, IndexT node ) {
          return scale * ( field + 1.0 ) * ( node + 1.0 );
        };
        const auto read_gradient = [=] TRIBOL_HOST_DEVICE( IndexT, IndexT field, IndexT node, IndexT column ) {
          return scale * ( field + 1.0 ) * ( node + 1.0 ) * ( column + 2.0 );
        };
        batched.addPairs<Exec, field_count, nodes_per_edge, pair_dofs>( empty, read_value, read_gradient );
        batched.addPairs<Exec, field_count, nodes_per_edge, pair_dofs>( pair_list, read_value, read_gradient );
      }
    }
    const auto expected = reference.assemble();
    const auto result = batched.assemble();
    ASSERT_EQ( result.size(), field_count );
    mfem::Vector direction( fes_->GetTrueVSize() );
    direction = 1.0;
    const int dual_size = cs.getMfemSubmeshData()->GetSubmeshFESpace().GetTrueVSize();
    mfem::Vector expected_action( dual_size ), action( dual_size );
    for ( IndexT field = 0; field < field_count; ++field ) {
      SCOPED_TRACE( field );
      EXPECT_GT( norm( expected[field].values.get() ), 0.0 );
      auto error = result[field].values.get();
      error -= expected[field].values.get();
      EXPECT_LT( norm( error ), tolerance );
      expected[field].dvalues_dx->Mult( direction, expected_action );
      result[field].dvalues_dx->Mult( direction, action );
      action -= expected_action;
      EXPECT_LT( norm( action ), tolerance );
    }
  } );
}

INSTANTIATE_TEST_SUITE_P( host_backends, MfemBatchAssemblerTest,
                          testing::Values( ExecutionMode::Sequential
#ifdef TRIBOL_USE_OPENMP
                                           ,
                                           ExecutionMode::OpenMP
#endif
                                           ) );

}  // namespace

/**
 * @brief Run assembly tests with MPI and logging initialized.
 * @param argc Argument count.
 * @param argv Arguments.
 * @return Test status.
 */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
