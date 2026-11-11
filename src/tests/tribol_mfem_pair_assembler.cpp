// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check host MFEM pair and nodal assembly with prescribed residuals, nodal quantities, and nonsymmetric
 * Jacobians, without contact physics. Nonmatching two- and three-edge surfaces exercise shared nodes, dual submesh
 * transfers, positive-gap clamping, and all matrix blocks. Tests run on one and two ranks, with contact data on rank
 * zero. Batch tests check accumulation after host calls, repeated batches, and an empty batch on Sequential and
 * OpenMP backends. Physics and Enzyme derivatives are checked by the formulation tests.
 */
#include <cmath>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MfemNodalAssembler.hpp"
#include "tribol/mesh/MfemPairAssembler.hpp"

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
class MfemPairAssemblerTest : public testing::Test {
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
  /** @brief Destroy coupling transfers before their borrowed MFEM mesh and fields. */
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

// A nonsymmetric outer product has a known action that checks row/column ordering and every pair block independently.
TEST_F( MfemPairAssemblerTest, assembles_prescribed_residual_and_nonsymmetric_jacobian )
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
    MfemPairAssembler assembler( *cs.getMfemMeshData(), *cs.getMfemJacobianData(), mesh1, mesh2, pair_count );
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
        assembler.addPair( { element1, element2 }, { residual.data(), residual.size() },
                           { jacobian.data(), { pair_dofs, pair_dofs } } );
      }
    }
    result = assembler.assemble();
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
TEST_F( MfemPairAssemblerTest, returns_zero_outputs_for_empty_or_rejected_input )
{
  auto& cs = scheme();
  MfemPairAssembler assembler( *cs.getMfemMeshData(), *cs.getMfemJacobianData(), cs.getMesh1().getView(),
                               cs.getMesh2().getView(), 0 );
  if ( cs.getMesh1().numberOfElements() > 0 && cs.getMesh2().numberOfElements() > 0 ) {
    MdArray<RealT, pair_dofs> residual{};
    MdArray<RealT, pair_dofs, pair_dofs> jacobian{};
    residual[0] = 1.0;
    EXPECT_THROW( assembler.addPair( { 0, 0 }, { residual.data(), pair_dofs - 1 },
                                     { jacobian.data(), { pair_dofs, pair_dofs } } ),
                  std::invalid_argument );
    EXPECT_THROW( assembler.addPair( { 0, 0 }, { residual.data(), pair_dofs },
                                     { jacobian.data(), { pair_dofs - 1, pair_dofs } } ),
                  std::invalid_argument );
  }
  auto result = assembler.assemble();
  ASSERT_NE( result.force, nullptr );
  ASSERT_NE( result.df_dx, nullptr );
  EXPECT_EQ( norm( *result.force ), 0.0 );
  mfem::Vector direction( fes_->GetTrueVSize() ), action( fes_->GetTrueVSize() );
  direction = 1.0;
  result.df_dx->Mult( direction, action );
  EXPECT_EQ( norm( action ), 0.0 );
}

// Nodal assembly must transfer gaps, areas, and gradients to the dual submesh, clamp positive gaps and eliminate their
// Jacobian rows when requested, transfer dual true DOFs back to Redecomp, and return the correct dual values.
TEST_F( MfemPairAssemblerTest, assembles_nodal_quantities_and_dual_transfers )
{
  auto& cs = scheme();
  prepareDualData();
  auto& mesh_data = *cs.getMfemMeshData();
  auto& dual_data = *cs.getMfemSubmeshData();
  auto& dual_fes = dual_data.GetSubmeshFESpace();
  dual_data.GetSubmeshPressure() = 2.5;

  const auto mesh1 = cs.getMesh1().getView(), mesh2 = cs.getMesh2().getView();
  const IndexT pair_count = mesh1.numberOfElements() * mesh2.numberOfElements();
  MfemNodalAssembler assembler( mesh_data, dual_data, *cs.getMfemJacobianData(), mesh1, mesh2, pair_count );

  const auto pressure_true = assembler.getSubmeshPressureTrueDofs();
  for ( int i = 0; i < pressure_true.size(); ++i ) {
    EXPECT_NEAR( pressure_true[i], 2.5, tolerance );
  }
  const auto redecomp_pressure = assembler.transferToRedecomp( pressure_true.get() );
  for ( int i = 0; i < redecomp_pressure.Size(); ++i ) {
    EXPECT_NEAR( redecomp_pressure[i], 2.5, tolerance );
  }
  const auto diag_mat = assembler.makeSubmeshDiagonalMatrix( pressure_true.get() );
  mfem::Vector diag_out( dual_fes.GetTrueVSize() );
  diag_mat->Mult( pressure_true.get(), diag_out );
  for ( int i = 0; i < diag_out.Size(); ++i ) {
    EXPECT_NEAR( diag_out[i], 6.25, tolerance );
  }

  // Prescribe nodal gaps on mesh1 (attribute 3) with mixed signs (x > 0.5 positive, x <= 0.5 negative) so clamping
  // eliminates only the positive node while preserving negative nodes and all tributary areas.
  for ( IndexT element1 = 0; element1 < mesh1.numberOfElements(); ++element1 ) {
    for ( IndexT element2 = 0; element2 < mesh2.numberOfElements(); ++element2 ) {
      std::array<MdArray<RealT, nodes_per_edge, dimension>, pair_size> coords;
      mesh1.getFaceCoords( element1, coords[0].data() );
      mesh2.getFaceCoords( element2, coords[1].data() );

      MdArray<RealT, nodes_per_edge> gaps{}, areas{};
      MdArray<RealT, nodes_per_edge, pair_dofs> gap_grads{}, area_grads{};
      MdArray<RealT, pair_dofs> columns{};

      for ( int side = 0; side < pair_size; ++side ) {
        for ( int node = 0; node < nodes_per_edge; ++node ) {
          for ( int component = 0; component < dimension; ++component ) {
            const int dof = ( side * nodes_per_edge + node ) * dimension + component;
            columns[dof] = columnValue( coords[side]( node, 0 ), coords[side]( node, 1 ), component );
          }
        }
      }

      for ( int node = 0; node < nodes_per_edge; ++node ) {
        const RealT x = coords[0]( node, 0 );
        gaps[node] = x - 0.6;
        areas[node] = 1.5 + x;
        for ( int col = 0; col < pair_dofs; ++col ) {
          gap_grads( node, col ) = ( 1.0 + x ) * columns[col];
          area_grads( node, col ) = ( 2.0 + x ) * columns[col];
        }
      }

      assembler.addPairQuantities(
          { element1, element2 }, { gaps.data(), nodes_per_edge }, { areas.data(), nodes_per_edge },
          { gap_grads.data(), { nodes_per_edge, pair_dofs } }, { area_grads.data(), { nodes_per_edge, pair_dofs } } );
    }
  }

  const auto unclamped = assembler.assembleQuantities( false );
  const auto clamped = assembler.assembleQuantities( true );

  mfem::Vector parent_ones( fes_->GetTrueVSize() );
  parent_ones = 1.0;
  mfem::Vector unclamped_dg( dual_fes.GetTrueVSize() ), clamped_dg( dual_fes.GetTrueVSize() );
  mfem::Vector unclamped_da( dual_fes.GetTrueVSize() ), clamped_da( dual_fes.GetTrueVSize() );
  unclamped.weighted_gap_jacobian->Mult( parent_ones, unclamped_dg );
  clamped.weighted_gap_jacobian->Mult( parent_ones, clamped_dg );
  unclamped.area_jacobian->Mult( parent_ones, unclamped_da );
  clamped.area_jacobian->Mult( parent_ones, clamped_da );

  int clamped_positive_count = 0;
  int retained_negative_count = 0;
  for ( int i = 0; i < unclamped.weighted_gap.size(); ++i ) {
    EXPECT_NEAR( unclamped.area[i], clamped.area[i], tolerance );
    EXPECT_NEAR( unclamped_da[i], clamped_da[i], tolerance );
    if ( unclamped.weighted_gap[i] > 0.0 ) {
      ++clamped_positive_count;
      EXPECT_EQ( clamped.weighted_gap[i], 0.0 );
      EXPECT_EQ( clamped.normalized_gap[i], 0.0 );
      EXPECT_EQ( clamped_dg[i], 0.0 );
    } else if ( unclamped.weighted_gap[i] < 0.0 ) {
      ++retained_negative_count;
      EXPECT_NEAR( clamped.weighted_gap[i], unclamped.weighted_gap[i], tolerance );
      EXPECT_NEAR( clamped.normalized_gap[i], unclamped.weighted_gap[i] / unclamped.area[i], tolerance );
      EXPECT_NEAR( clamped_dg[i], unclamped_dg[i], tolerance );
    }
  }
  int global_positive = 0, global_negative = 0;
  MPI_Allreduce( &clamped_positive_count, &global_positive, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &retained_negative_count, &global_negative, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( global_positive, 0 );
  EXPECT_GT( global_negative, 0 );
}

/** @brief Exercise host batch kernels with each enabled host execution mode. */
class MfemBatchAssemblerTest : public MfemPairAssemblerTest, public testing::WithParamInterface<ExecutionMode> {};

// Distinct batch entries and an intervening empty batch must preserve all four matrix blocks and accumulated forces.
TEST_P( MfemBatchAssemblerTest, accumulates_residuals_and_jacobians_across_batches )
{
  auto& cs = scheme();
  const auto mesh1 = cs.getMesh1().getView(), mesh2 = cs.getMesh2().getView();
  const auto pair_list = pairs();
  MfemPairAssembler reference( *cs.getMfemMeshData(), *cs.getMfemJacobianData(), mesh1, mesh2, 0 );
  MfemPairAssembler batched( *cs.getMfemMeshData(), *cs.getMfemJacobianData(), mesh1, mesh2, 0 );
  const ArrayT<InterfacePair> empty( 0, 0, pair_list.getAllocatorID() );
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
      reference.addPair( pair_list[pair], { residual.data(), pair_dofs },
                         { jacobian.data(), { pair_dofs, pair_dofs } } );
      if ( batch == 0 ) {
        batched.addPair( pair_list[pair], { residual.data(), pair_dofs },
                         { jacobian.data(), { pair_dofs, pair_dofs } } );
      }
    }
    if ( batch != 0 ) {
      const auto read_residual = [=] TRIBOL_HOST_DEVICE( IndexT, IndexT row ) { return scale * ( row + 1.0 ); };
      const auto read_jacobian = [=] TRIBOL_HOST_DEVICE( IndexT, IndexT row, IndexT column ) {
        return scale * ( row + 1.0 ) * ( column + 2.0 );
      };
      batched.addPairs<pair_dofs>( GetParam(), empty, read_residual, read_jacobian );
      batched.addPairs<pair_dofs>( GetParam(), pair_list, read_residual, read_jacobian );
    }
  }
  auto expected = reference.assemble();
  auto result = batched.assemble();
  EXPECT_GT( norm( *expected.force ), 0.0 );
  result.force->Add( -1.0, *expected.force );
  EXPECT_LT( norm( *result.force ), tolerance );
  mfem::Vector direction( fes_->GetTrueVSize() ), expected_action( fes_->GetTrueVSize() ),
      action( fes_->GetTrueVSize() );
  direction = 1.0;
  expected.df_dx->Mult( direction, expected_action );
  result.df_dx->Mult( direction, action );
  action -= expected_action;
  EXPECT_LT( norm( action ), tolerance );
}

// Weighted gaps, areas, and both gradient matrices must retain host contributions and every subsequent batch.
TEST_P( MfemBatchAssemblerTest, accumulates_nodal_values_and_gradients_across_batches )
{
  prepareDualData();
  auto& cs = scheme();
  const auto mesh1 = cs.getMesh1().getView(), mesh2 = cs.getMesh2().getView();
  const auto pair_list = pairs();
  MfemNodalAssembler reference( *cs.getMfemMeshData(), *cs.getMfemSubmeshData(), *cs.getMfemJacobianData(), mesh1,
                                mesh2, 0 );
  MfemNodalAssembler batched( *cs.getMfemMeshData(), *cs.getMfemSubmeshData(), *cs.getMfemJacobianData(), mesh1, mesh2,
                              0 );
  const ArrayT<InterfacePair> empty( 0, 0, pair_list.getAllocatorID() );
  for ( int batch = 0; batch < 3; ++batch ) {
    const RealT scale = batch + 1.0;
    MdArray<RealT, nodes_per_edge> gaps{}, areas{};
    MdArray<RealT, nodes_per_edge, pair_dofs> gap_gradients{}, area_gradients{};
    for ( int node = 0; node < nodes_per_edge; ++node ) {
      gaps[node] = -scale * ( node + 1.0 );
      areas[node] = scale * ( node + 2.0 );
      for ( int column = 0; column < pair_dofs; ++column ) {
        gap_gradients( node, column ) = -scale * ( node + 1.0 ) * ( column + 2.0 );
        area_gradients( node, column ) = scale * ( node + 2.0 ) * ( column + 1.0 );
      }
    }
    for ( const auto pair : pair_list ) {
      reference.addPairQuantities( pair, { gaps.data(), nodes_per_edge }, { areas.data(), nodes_per_edge },
                                   { gap_gradients.data(), { nodes_per_edge, pair_dofs } },
                                   { area_gradients.data(), { nodes_per_edge, pair_dofs } } );
      if ( batch == 0 ) {
        batched.addPairQuantities( pair, { gaps.data(), nodes_per_edge }, { areas.data(), nodes_per_edge },
                                   { gap_gradients.data(), { nodes_per_edge, pair_dofs } },
                                   { area_gradients.data(), { nodes_per_edge, pair_dofs } } );
      }
    }
    if ( batch != 0 ) {
      const auto read_value = [=] TRIBOL_HOST_DEVICE( IndexT, IndexT value ) {
        return value < nodes_per_edge ? -scale * ( value + 1.0 ) : scale * ( value - nodes_per_edge + 2.0 );
      };
      const auto read_gradient = [=] TRIBOL_HOST_DEVICE( IndexT, IndexT value, IndexT column ) {
        return value < nodes_per_edge ? -scale * ( value + 1.0 ) * ( column + 2.0 )
                                      : scale * ( value - nodes_per_edge + 2.0 ) * ( column + 1.0 );
      };
      batched.addPairQuantities<nodes_per_edge, pair_dofs>( GetParam(), empty, read_value, read_gradient );
      batched.addPairQuantities<nodes_per_edge, pair_dofs>( GetParam(), pair_list, read_value, read_gradient );
    }
  }
  const auto expected = reference.assembleQuantities( false );
  const auto result = batched.assembleQuantities( false );
  EXPECT_GT( norm( expected.weighted_gap.get() ), 0.0 );
  auto gap_error = result.weighted_gap.get();
  gap_error -= expected.weighted_gap.get();
  EXPECT_LT( norm( gap_error ), tolerance );
  auto area_error = result.area.get();
  area_error -= expected.area.get();
  EXPECT_LT( norm( area_error ), tolerance );
  mfem::Vector direction( fes_->GetTrueVSize() );
  direction = 1.0;
  const int dual_size = cs.getMfemSubmeshData()->GetSubmeshFESpace().GetTrueVSize();
  mfem::Vector expected_action( dual_size ), action( dual_size );
  expected.weighted_gap_jacobian->Mult( direction, expected_action );
  result.weighted_gap_jacobian->Mult( direction, action );
  action -= expected_action;
  EXPECT_LT( norm( action ), tolerance );
  expected.area_jacobian->Mult( direction, expected_action );
  result.area_jacobian->Mult( direction, action );
  action -= expected_action;
  EXPECT_LT( norm( action ), tolerance );
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
