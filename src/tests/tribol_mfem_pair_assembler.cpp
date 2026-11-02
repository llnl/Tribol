// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check host MFEM pair and nodal assembly with prescribed residuals, nodal quantities, and nonsymmetric
 * Jacobians, without contact physics. Nonmatching two- and three-edge surfaces exercise shared nodes, dual submesh
 * transfers, positive-gap clamping, and all matrix blocks. Tests run on one and two ranks, with contact data on rank
 * zero. Nodal expression tests check Enzyme partials after shared-node assembly; formulation tests check the full
 * coordinate derivatives.
 */
#include <cmath>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MfemNodalAssembler.hpp"
#include "tribol/mesh/MfemNodalEnergy.hpp"
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

  /** @brief Prepare a scalar dual submesh and refresh the displacement/dual Jacobian transfer. */
  void enableDualTransfers()
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
    if ( attribute != 3 && attribute != 5 ) continue;
    const int side = attribute == 3 ? 0 : 1;
    ++local_counts[side];
    mfem::Array<int> nodes;
    fes_->GetBdrElementDofs( boundary, nodes );
    for ( const int node : nodes ) {
      const RealT x = ( *coords_ )[fes_->DofToVDof( node, 0 )];
      const RealT y = ( *coords_ )[fes_->DofToVDof( node, 1 )];
      for ( int component = 0; component < dimension; ++component )
        local_column_sums[side] += columnValue( x, y, component ) * direction[fes_->DofToVDof( node, component )];
    }
  }
  MPI_Allreduce( local_counts.data(), counts.data(), pair_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( local_column_sums.data(), column_sums.data(), pair_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  mfem::ParLinearForm expected_residual( fes_.get() ), expected_action( fes_.get() );
  expected_residual = 0.0;
  expected_action = 0.0;
  for ( int boundary = 0; boundary < mesh_->GetNBE(); ++boundary ) {
    const int attribute = mesh_->GetBdrAttribute( boundary );
    if ( attribute != 3 && attribute != 5 ) continue;
    const int side = attribute == 3 ? 0 : 1;
    mfem::Array<int> nodes;
    fes_->GetBdrElementDofs( boundary, nodes );
    RealT element_column_sum = 0.0;
    for ( const int node : nodes ) {
      const RealT x = ( *coords_ )[fes_->DofToVDof( node, 0 )];
      const RealT y = ( *coords_ )[fes_->DofToVDof( node, 1 )];
      for ( int component = 0; component < dimension; ++component )
        element_column_sum += columnValue( x, y, component ) * direction[fes_->DofToVDof( node, component )];
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
        for ( int side = 0; side < pair_size; ++side )
          for ( int node = 0; node < nodes_per_edge; ++node )
            for ( int component = 0; component < dimension; ++component ) {
              const int dof = ( side * nodes_per_edge + node ) * dimension + component;
              residual[dof] = rowValue( coords[side]( node, 0 ), coords[side]( node, 1 ), component );
              columns[dof] = columnValue( coords[side]( node, 0 ), coords[side]( node, 1 ), component );
            }
        for ( int row = 0; row < pair_dofs; ++row )
          for ( int column = 0; column < pair_dofs; ++column )
            jacobian( row, column ) = residual[row] * columns[column];
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
  if ( ranks == 2 && rank == 1 ) EXPECT_EQ( mesh1.numberOfElements() + mesh2.numberOfElements(), 0 );
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
// Jacobian rows when requested, transfer dual true DOFs back to Redecomp, and assemble node-weighted pair Hessians.
TEST_F( MfemPairAssemblerTest, assembles_nodal_quantities_dual_transfers_and_weighted_hessians )
{
  auto& cs = scheme();
  auto& mesh_data = *cs.getMfemMeshData();
  enableDualTransfers();

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

      MdArray<RealT, nodes_per_edge> gaps{}, areas{}, gap_weights{}, area_weights{};
      MdArray<RealT, nodes_per_edge, pair_dofs> gap_grads{}, area_grads{};
      MdArray<RealT, nodes_per_edge, pair_dofs, pair_dofs> gap_hessians{}, area_hessians{};
      MdArray<RealT, pair_dofs> rows{}, columns{};

      for ( int side = 0; side < pair_size; ++side )
        for ( int node = 0; node < nodes_per_edge; ++node )
          for ( int component = 0; component < dimension; ++component ) {
            const int dof = ( side * nodes_per_edge + node ) * dimension + component;
            rows[dof] = rowValue( coords[side]( node, 0 ), coords[side]( node, 1 ), component );
            columns[dof] = columnValue( coords[side]( node, 0 ), coords[side]( node, 1 ), component );
          }

      for ( int node = 0; node < nodes_per_edge; ++node ) {
        const RealT x = coords[0]( node, 0 );
        gaps[node] = x - 0.6;
        areas[node] = 1.5 + x;
        gap_weights[node] = 2.0;
        area_weights[node] = -0.5;
        for ( int col = 0; col < pair_dofs; ++col ) {
          gap_grads( node, col ) = ( 1.0 + x ) * columns[col];
          area_grads( node, col ) = ( 2.0 + x ) * columns[col];
          for ( int row = 0; row < pair_dofs; ++row ) {
            gap_hessians( node, row, col ) = rows[row] * columns[col];
            area_hessians( node, row, col ) = 2.0 * rows[row] * columns[col];
          }
        }
      }

      assembler.addPairQuantities(
          { element1, element2 }, { gaps.data(), nodes_per_edge }, { areas.data(), nodes_per_edge },
          { gap_grads.data(), { nodes_per_edge, pair_dofs } }, { area_grads.data(), { nodes_per_edge, pair_dofs } } );
      assembler.addPairHessians( { element1, element2 }, { gap_weights.data(), nodes_per_edge },
                                 { gap_hessians.data(), { nodes_per_edge, pair_dofs, pair_dofs } },
                                 { area_weights.data(), nodes_per_edge },
                                 { area_hessians.data(), { nodes_per_edge, pair_dofs, pair_dofs } } );
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

  // Each pair's weighted Hessian sum over the two nodes of element1 is 2 * (2.0 * 1.0 + (-0.5) * 2.0) = 2.0 times
  // rows * columns^T, so the assembled Hessian equals twice the outer-product Jacobian from MfemPairAssembler.
  MfemPairAssembler reference_assembler( mesh_data, *cs.getMfemJacobianData(), mesh1, mesh2, pair_count );
  for ( IndexT element1 = 0; element1 < mesh1.numberOfElements(); ++element1 ) {
    for ( IndexT element2 = 0; element2 < mesh2.numberOfElements(); ++element2 ) {
      std::array<MdArray<RealT, nodes_per_edge, dimension>, pair_size> coords;
      mesh1.getFaceCoords( element1, coords[0].data() );
      mesh2.getFaceCoords( element2, coords[1].data() );
      MdArray<RealT, pair_dofs> rows{}, columns{};
      MdArray<RealT, pair_dofs, pair_dofs> jacobian{};
      for ( int side = 0; side < pair_size; ++side )
        for ( int node = 0; node < nodes_per_edge; ++node )
          for ( int component = 0; component < dimension; ++component ) {
            const int dof = ( side * nodes_per_edge + node ) * dimension + component;
            rows[dof] = rowValue( coords[side]( node, 0 ), coords[side]( node, 1 ), component );
            columns[dof] = columnValue( coords[side]( node, 0 ), coords[side]( node, 1 ), component );
          }
      for ( int row = 0; row < pair_dofs; ++row )
        for ( int col = 0; col < pair_dofs; ++col ) jacobian( row, col ) = 2.0 * rows[row] * columns[col];
      reference_assembler.addPairJacobian( { element1, element2 }, { jacobian.data(), { pair_dofs, pair_dofs } } );
    }
  }
  const auto hessian_mat = assembler.assembleHessians();
  const auto reference_mat = reference_assembler.assembleJacobian();
  mfem::Vector hessian_action( fes_->GetTrueVSize() ), reference_action( fes_->GetTrueVSize() );
  hessian_mat->Mult( parent_ones, hessian_action );
  reference_mat->Mult( parent_ones, reference_action );
  hessian_action -= reference_action;
  EXPECT_LT( norm( hessian_action ), tolerance );

  // Verify chain-rule residual and outer-Hessian assembly from prescribed nodal derivatives.
  auto derivatives = assembler.makeDerivativeData( true, true );
  derivatives.d_energy_d_gap.fill( 3.0 );
  derivatives.d_energy_d_area.fill( -1.5 );
  derivatives.d2_energy_dg2.fill( 4.0 );
  derivatives.d2_energy_dg_da.fill( -0.5 );
  derivatives.d2_energy_da2.fill( 2.0 );

  const auto chain_force = assembler.assembleForce( unclamped, derivatives );
  const auto expected_force = ( derivatives.d_energy_d_gap * unclamped.weighted_gap_jacobian ) +
                              ( derivatives.d_energy_d_area * unclamped.area_jacobian );
  auto force_diff = chain_force.get();
  force_diff -= expected_force.get();
  EXPECT_LT( norm( force_diff ), tolerance );

  const auto full_jacobian = assembler.assembleJacobian( unclamped, derivatives );
  mfem::Vector full_action( fes_->GetTrueVSize() );
  full_jacobian->Mult( parent_ones, full_action );
  // Action of (dg^T * 4 * dg + dg^T * (-0.5) * dA + dA^T * (-0.5) * dg + dA^T * 2 * dA) * 1 on parent_ones:
  mfem::Vector expected_outer_dual_g( unclamped_dg ), expected_outer_dual_a( unclamped_da );
  expected_outer_dual_g *= 4.0;
  expected_outer_dual_g.Add( -0.5, unclamped_da );
  expected_outer_dual_a *= 2.0;
  expected_outer_dual_a.Add( -0.5, unclamped_dg );
  mfem::Vector expected_full_action( reference_action ), temp_parent( fes_->GetTrueVSize() );
  unclamped.weighted_gap_jacobian->MultTranspose( expected_outer_dual_g, temp_parent );
  expected_full_action += temp_parent;
  unclamped.area_jacobian->MultTranspose( expected_outer_dual_a, temp_parent );
  expected_full_action += temp_parent;
  full_action -= expected_full_action;
  EXPECT_LT( norm( full_action ), tolerance );
}
#ifdef TRIBOL_USE_ENZYME
// A shared node receives positive and negative pair contributions. Activation must use their assembled sum, and
// Enzyme partials must supply the gap/area cross terms needed by the displacement Jacobian.
TEST_F( MfemPairAssemblerTest, differentiates_assembled_nodal_integrals_before_coordinate_chain_rule )
{
  enableDualTransfers();
  auto& cs = scheme();
  auto& mesh_data = *cs.getMfemMeshData();
  auto& dual_data = *cs.getMfemSubmeshData();
  dual_data.GetSubmeshPressure() = -3.0;
  const auto mesh1 = cs.getMesh1().getView(), mesh2 = cs.getMesh2().getView();
  const IndexT pair_count = mesh1.numberOfElements() * mesh2.numberOfElements();
  MfemNodalAssembler assembler( mesh_data, dual_data, *cs.getMfemJacobianData(), mesh1, mesh2, pair_count );
  for ( IndexT element1 = 0; element1 < mesh1.numberOfElements(); ++element1 ) {
    for ( IndexT element2 = 0; element2 < mesh2.numberOfElements(); ++element2 ) {
      // The three mesh2 edges give +0.1, -0.2, -0.2 per mesh1 element node, with total area 3.0.
      const RealT gap = element2 == 0 ? 0.1 : -0.2;
      const MdArray<RealT, nodes_per_edge> gaps{ gap, gap }, areas{ 1.0, 1.0 };
      const MdArray<RealT, nodes_per_edge, pair_dofs> gradients{};
      assembler.addPairQuantities(
          { element1, element2 }, { gaps.data(), nodes_per_edge }, { areas.data(), nodes_per_edge },
          { gradients.data(), { nodes_per_edge, pair_dofs } }, { gradients.data(), { nodes_per_edge, pair_dofs } } );
    }
  }
  const auto quantities = assembler.assembleQuantities( false );
  constexpr RealT stiffness = 4.0;           // Pressure is -0.4 for assembled gap/area ratio -0.1.
  constexpr RealT area_tolerance = 1.0e-14;  // Match the formulation's support threshold.
  const auto g = min( nodal_weighted_gap, 0.0 );
  const auto energy = 0.5 * stiffness * sum( g, g, 1.0 / nodal_area );
  const auto derivatives = evaluateNodalEnergy( energy, quantities, assembler, area_tolerance );
  EXPECT_TRUE( derivatives.has_area_derivatives );
  EXPECT_TRUE( derivatives.has_outer_hessian );
  RealT local_expected_energy = 0.0;
  for ( int node = 0; node < quantities.area.size(); ++node ) {
    const RealT area = quantities.area[node];
    if ( area > area_tolerance ) {
      EXPECT_NEAR( quantities.weighted_gap[node] / area, -0.1, tolerance );
      local_expected_energy += 0.02 * area;
      EXPECT_NEAR( derivatives.pressure[node], -0.4, tolerance );
      EXPECT_NEAR( derivatives.d_energy_d_gap[node], -0.4, tolerance );
      EXPECT_NEAR( derivatives.d_energy_d_area[node], -0.02, tolerance );
      EXPECT_NEAR( derivatives.d2_energy_dg2[node], stiffness / area, tolerance );
      EXPECT_NEAR( derivatives.d2_energy_dg_da[node], 0.4 / area, tolerance );
      EXPECT_NEAR( derivatives.d2_energy_da2[node], 0.04 / area, tolerance );
    } else {
      EXPECT_EQ( derivatives.pressure[node], 0.0 );
      EXPECT_EQ( derivatives.d_energy_d_area[node], 0.0 );
      EXPECT_EQ( derivatives.d2_energy_dg2[node], 0.0 );
      EXPECT_EQ( derivatives.d2_energy_dg_da[node], 0.0 );
      EXPECT_EQ( derivatives.d2_energy_da2[node], 0.0 );
    }
  }
  RealT expected_energy = 0.0;
  MPI_Allreduce( &local_expected_energy, &expected_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( expected_energy, 0.0 );
  EXPECT_NEAR( derivatives.energy, expected_energy, tolerance );

  const auto coefficients = assembler.getSubmeshPressureTrueDofs();
  const IndependentNodalField pressure{ { coefficients.get().GetData(), coefficients.size() } };
  const auto lm_energy = dot( pressure, nodal_weighted_gap );
  const auto lm_derivatives = evaluateNodalEnergy( lm_energy, quantities, assembler, area_tolerance );
  EXPECT_FALSE( lm_derivatives.has_area_derivatives );
  EXPECT_FALSE( lm_derivatives.has_outer_hessian );
  for ( int node = 0; node < quantities.area.size(); ++node ) EXPECT_EQ( lm_derivatives.pressure[node], -3.0 );
  EXPECT_THROW( evaluateNodalEnergy( energy, quantities, assembler, -1.0 ), std::invalid_argument );
  const ArrayT<RealT> wrong_size( coefficients.size() + 1, coefficients.size() + 1 );
  const auto invalid = dot( IndependentNodalField{ wrong_size.view() }, nodal_weighted_gap );
  EXPECT_THROW( evaluateNodalEnergy( invalid, quantities, assembler, area_tolerance ), std::invalid_argument );
}
#endif

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
