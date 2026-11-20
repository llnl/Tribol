// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Verify MFEM residual and Jacobian assembly from prescribed per-pair data.
 *
 * These tests isolate assembly from contact physics on nonmatching edge meshes. They cover shared-node accumulation,
 * scalar fields, matrix block layouts and orientations, combined assembly views, host execution backends, and one- and
 * two-rank configurations.
 */
#include <cmath>
#include "gtest/gtest.h"
#include "SlicErrorTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/assembly/MfemVectorAssembler.hpp"
#include "tribol/assembly/AssemblePairs.hpp"
#include "tribol/assembly/MfemMatrixAssembler.hpp"

namespace {
using namespace tribol;
constexpr int spatial_dim = 2;     ///< Spatial dimension shared by the pair-layout templates in this file.
constexpr int nodes_per_edge = 2;  ///< Finite-element nodes on a linear edge.
constexpr int pair_size = 2;       ///< Elements in a pair.
constexpr int pair_dofs = spatial_dim * nodes_per_edge * pair_size;  ///< Coordinate entries per pair.
constexpr RealT assembly_tolerance = 1.0e-10;  ///< Roundoff allowance for MPI sums and matrix-vector products.

/**
 * @brief Assign distinct residual-row values from physical position and spatial coordinate index.
 * @param x Node x coordinate.
 * @param y Node y coordinate.
 * @param spatial_dim Spatial coordinate index.
 * @return Row value for the prescribed residual and outer-product Jacobian.
 */
RealT rowValue( RealT x, RealT y, int spatial_dim ) { return 1.0 + x + 2.0 * y + 3.0 * spatial_dim; }

/**
 * @brief Assign column values different from the row values, so transposing the Jacobian changes its action.
 * @param x Node x coordinate.
 * @param y Node y coordinate.
 * @param spatial_dim Spatial coordinate index.
 * @return Column value for the prescribed outer-product Jacobian.
 */
RealT columnValue( RealT x, RealT y, int spatial_dim ) { return -2.0 + 3.0 * x - y - 4.0 * spatial_dim; }

// These named callables keep device-capable assembly operations outside GoogleTest's generated TestBody() methods and
// the generic lambdas used to select an execution mode. NVCC rejects extended host-device lambdas in either context:
// TestBody() is private, and a generic lambda cannot contain another extended lambda. Each object below stores the same
// captured test data and performs the same calculation that a local lambda would otherwise express.
template <typename Values>
struct StoredResidualReader {
  Values values;

  TRIBOL_HOST_DEVICE RealT operator()( IndexT pair, IndexT row ) const { return values[pair * pair_dofs + row]; }
};

template <typename Values>
struct StoredJacobianReader {
  Values values;

  TRIBOL_HOST_DEVICE RealT operator()( IndexT pair, IndexT row, IndexT column ) const
  {
    return values[( pair * pair_dofs + row ) * pair_dofs + column];
  }
};

struct ZeroResidualReader {
  TRIBOL_HOST_DEVICE RealT operator()( IndexT, IndexT ) const { return 0.0; }
};

template <typename Factors>
struct ScaledNodalValueReader {
  Factors factors;

  TRIBOL_HOST_DEVICE RealT operator()( IndexT, IndexT field, IndexT node ) const
  {
    return factors[field] * ( node + 1.0 );
  }
};

template <typename Factors>
struct ScaledNodalGradientReader {
  Factors factors;

  TRIBOL_HOST_DEVICE RealT operator()( IndexT, IndexT field, IndexT node, IndexT column ) const
  {
    return factors[field] * ( node + 1.0 ) * ( column + 1.0 );
  }
};

struct ScalarRowReader {
  TRIBOL_HOST_DEVICE RealT operator()( IndexT, IndexT row ) const { return row + 1.0; }
};

struct ForwardMatrixReader {
  TRIBOL_HOST_DEVICE RealT operator()( IndexT pair, IndexT node, IndexT coordinate ) const
  {
    return node + 1.0 + ( coordinate + 2.0 ) * ( pair + 1.0 );
  }
};

struct ReverseMatrixReader {
  TRIBOL_HOST_DEVICE RealT operator()( IndexT pair, IndexT coordinate, IndexT node ) const
  {
    return ForwardMatrixReader{}( pair, node, coordinate );
  }
};

struct ResidualRowReader {
  TRIBOL_HOST_DEVICE RealT operator()( IndexT, IndexT row ) const { return row + 1.0; }
};

struct JacobianEntryReader {
  TRIBOL_HOST_DEVICE RealT operator()( IndexT, IndexT row, IndexT column ) const
  {
    return ( row + 1.0 ) * ( column + 2.0 );
  }
};

template <typename ResidualView, typename MatrixView, typename PairsView>
struct ComposedAssemblyWriter {
  ResidualView residual;
  MatrixView matrix;
  PairsView pairs;

  TRIBOL_HOST_DEVICE void operator()( IndexT pair ) const
  {
    residual.scatterAddPair( pair, pairs[pair], ResidualRowReader{} );
    matrix.packPair( pair, pairs[pair], JacobianEntryReader{} );
  }
};

struct FieldValueReader {
  TRIBOL_HOST_DEVICE RealT operator()( IndexT, IndexT field, IndexT node ) const
  {
    return ( field + 1.0 ) * ( node + 1.0 );
  }
};

struct FieldGradientReader {
  TRIBOL_HOST_DEVICE RealT operator()( IndexT, IndexT field, IndexT node, IndexT column ) const
  {
    return ( field + 1.0 ) * ( node + 1.0 ) * ( column + 2.0 );
  }
};

/** @brief Provide parent and Redecomp transfers without selecting a contact formulation. */
class MfemResidualAndJacobianAssemblyTest : public testing::Test {
 protected:
  static constexpr IndexT coupling_scheme_id = 0;

  /** @brief Build nonmatching linear surfaces and concentrate their Redecomp elements on rank zero. */
  void SetUp() override
  {
    constexpr int lower_boundary_edge_count = 2;
    constexpr int upper_boundary_edge_count = 3;
    constexpr int vertical_element_count = 1;
    constexpr int upper_contact_boundary_attribute = 5;
    constexpr int upper_noncontact_boundary_attribute = 6;
    // The lower unit square has two edges across its top boundary. The upper square has three edges across its bottom
    // boundary and begins at y=0.99, so the nonmatching contact surfaces overlap by 0.01. All assembly tests reuse this
    // geometry but prescribe their own pair values instead of evaluating contact physics.
    parent_mesh_ = std::make_unique<mfem::ParMesh>( static_cast<const mfem::ParMesh&>( shared::ParMeshBuilder(
        MPI_COMM_WORLD, shared::MeshBuilder::Unify(
                            { shared::MeshBuilder::SquareMesh( lower_boundary_edge_count, vertical_element_count ),
                              shared::MeshBuilder::SquareMesh( upper_boundary_edge_count, vertical_element_count )
                                  .translate( { 0.0, 0.99 } )
                                  .updateBdrAttrib( 1, upper_contact_boundary_attribute )
                                  .updateBdrAttrib( 3, upper_noncontact_boundary_attribute ) } ) ) ) );
    constexpr int linear_order = 1;
    coordinate_collection_ = std::make_unique<mfem::H1_FECollection>( linear_order, spatial_dim );
    coordinate_space_ = std::make_unique<mfem::ParFiniteElementSpace>( parent_mesh_.get(), coordinate_collection_.get(),
                                                                       spatial_dim, mfem::Ordering::byVDIM );
    coordinates_ = std::make_unique<mfem::ParGridFunction>( coordinate_space_.get() );
    parent_mesh_->GetNodes( *coordinates_ );
    constexpr IndexT lower_mesh_id = 0;
    constexpr IndexT upper_mesh_id = 1;
    constexpr int lower_contact_boundary_attribute = 3;
    registerMfemCouplingScheme( coupling_scheme_id, lower_mesh_id, upper_mesh_id, *parent_mesh_, *coordinates_,
                                { lower_contact_boundary_attribute }, { upper_contact_boundary_attribute },
                                SURFACE_TO_SURFACE, NO_CASE, COMMON_PLANE, FRICTIONLESS, PENALTY, BINNING_GRID,
                                ExecutionMode::Sequential );
    constexpr RealT lower_mesh_penalty = 2.0;
    constexpr RealT upper_mesh_penalty = 6.0;
    setMfemKinematicConstantPenalty( coupling_scheme_id, lower_mesh_penalty, upper_mesh_penalty );
    constexpr int contact_rank_count = 1;
    updateMfemParallelDecomposition( contact_rank_count, true );
    auto& coupling_scheme = scheme();
    ASSERT_TRUE( coupling_scheme.setMeshPointers() );
    coupling_scheme.setMfemJacobianData( std::make_unique<MfemJacobianData>( *coupling_scheme.getMfemMeshData() ) );
    coupling_scheme.getMfemJacobianData()->UpdateJacobianXfer();
  }
  /** @brief Destroy coupling transfers before the MFEM mesh and fields they reference. */
  void TearDown() override { finalize(); }
  /** @brief Return the registered scheme. @return Scheme zero. */
  CouplingScheme& scheme() { return CouplingSchemeManager::getInstance().at( coupling_scheme_id ); }
  /** @brief Compute a global vector norm. @param v Local entries. @return Distributed Euclidean norm. */
  RealT norm( const mfem::Vector& v ) { return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, v, v ) ); }

  /** @brief Create scalar dual fields and their matrix transfers on the existing decomposition. */
  void prepareDualData()
  {
    auto& coupling_scheme = scheme();
    auto& mesh_data = *coupling_scheme.getMfemMeshData();
    auto submesh_data =
        std::make_unique<MfemSubmeshData>( mesh_data.GetSubmesh(), mesh_data.GetLORMesh(),
                                           std::make_unique<mfem::H1_FECollection>( 1, spatial_dim ), 1, false );
    submesh_data->UpdateMfemSubmeshData( mesh_data.GetRedecompMesh(), true );
    coupling_scheme.setMfemSubmeshData( std::move( submesh_data ) );
    coupling_scheme.setMfemJacobianData(
        std::make_unique<MfemJacobianData>( mesh_data, *coupling_scheme.getMfemSubmeshData() ) );
    coupling_scheme.getMfemJacobianData()->UpdateJacobianXfer();
  }

  /** @brief List every pair of the nonmatching contact edges. @return Owned host pair list. */
  ArrayT<InterfacePair> pairs()
  {
    const auto mesh1 = scheme().getMesh1().getView(), mesh2 = scheme().getMesh2().getView();
    const IndexT num_pairs = mesh1.numberOfElements() * mesh2.numberOfElements();
    ArrayT<InterfacePair> element_pairs( num_pairs, num_pairs, mesh1.getAllocatorId() );
    IndexT index = 0;
    for ( IndexT element1 = 0; element1 < mesh1.numberOfElements(); ++element1 ) {
      for ( IndexT element2 = 0; element2 < mesh2.numberOfElements(); ++element2 ) {
        element_pairs[index++] = { element1, element2 };
      }
    }
    return element_pairs;
  }

  std::unique_ptr<mfem::ParMesh> parent_mesh_;                     ///< Parent patch mesh.
  std::unique_ptr<mfem::H1_FECollection> coordinate_collection_;   ///< Linear coordinate basis.
  std::unique_ptr<mfem::ParFiniteElementSpace> coordinate_space_;  ///< Parent coordinate space.
  std::unique_ptr<mfem::ParGridFunction> coordinates_;             ///< Parent node coordinates.
};

// A nonsymmetric outer product has a known action that checks row/column ordering and every pair block independently.
TEST_F( MfemResidualAndJacobianAssemblyTest, assembles_prescribed_residual_and_nonsymmetric_jacobian )
{
  mfem::ParGridFunction direction( coordinate_space_.get() );
  mfem::VectorFunctionCoefficient direction_coefficient( spatial_dim, []( const mfem::Vector& x, mfem::Vector& value ) {
    value.SetSize( spatial_dim );
    value[0] = 2.0 + x[0] - 0.5 * x[1];
    value[1] = -1.0 + 0.25 * x[0] + 2.0 * x[1];
  } );
  direction.ProjectCoefficient( direction_coefficient );
  mfem::HypreParVector true_direction( coordinate_space_.get() );
  direction.GetTrueDofs( true_direction );

  // Every edge is paired with every edge on the other surface. Count each element's column contribution once;
  // each row element then receives its own column contribution once per opposite element plus that surface's sum.
  MdArray<int, pair_size> num_local_elements{}, num_elements{};
  MdArray<RealT, pair_size> local_column_sums{}, column_sums{};
  for ( int boundary = 0; boundary < parent_mesh_->GetNBE(); ++boundary ) {
    const int attribute = parent_mesh_->GetBdrAttribute( boundary );
    if ( attribute != 3 && attribute != 5 ) {
      continue;
    }
    const int side = attribute == 3 ? 0 : 1;
    ++num_local_elements[side];
    mfem::Array<int> nodes;
    coordinate_space_->GetBdrElementDofs( boundary, nodes );
    for ( const int node : nodes ) {
      const RealT x = ( *coordinates_ )[coordinate_space_->DofToVDof( node, 0 )];
      const RealT y = ( *coordinates_ )[coordinate_space_->DofToVDof( node, 1 )];
      for ( int component = 0; component < spatial_dim; ++component ) {
        local_column_sums[side] +=
            columnValue( x, y, component ) * direction[coordinate_space_->DofToVDof( node, component )];
      }
    }
  }
  MPI_Allreduce( num_local_elements.data(), num_elements.data(), pair_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( local_column_sums.data(), column_sums.data(), pair_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  mfem::ParLinearForm expected_residual( coordinate_space_.get() ), expected_action( coordinate_space_.get() );
  expected_residual = 0.0;
  expected_action = 0.0;
  for ( int boundary = 0; boundary < parent_mesh_->GetNBE(); ++boundary ) {
    const int attribute = parent_mesh_->GetBdrAttribute( boundary );
    if ( attribute != 3 && attribute != 5 ) {
      continue;
    }
    const int side = attribute == 3 ? 0 : 1;
    mfem::Array<int> nodes;
    coordinate_space_->GetBdrElementDofs( boundary, nodes );
    RealT element_column_sum = 0.0;
    for ( const int node : nodes ) {
      const RealT x = ( *coordinates_ )[coordinate_space_->DofToVDof( node, 0 )];
      const RealT y = ( *coordinates_ )[coordinate_space_->DofToVDof( node, 1 )];
      for ( int component = 0; component < spatial_dim; ++component ) {
        element_column_sum +=
            columnValue( x, y, component ) * direction[coordinate_space_->DofToVDof( node, component )];
      }
    }
    const RealT pair_column_sum = num_elements[1 - side] * element_column_sum + column_sums[1 - side];
    for ( const int node : nodes ) {
      const RealT x = ( *coordinates_ )[coordinate_space_->DofToVDof( node, 0 )];
      const RealT y = ( *coordinates_ )[coordinate_space_->DofToVDof( node, 1 )];
      for ( int component = 0; component < spatial_dim; ++component ) {
        const int dof = coordinate_space_->DofToVDof( node, component );
        expected_residual[dof] += rowValue( x, y, component ) * num_elements[1 - side];
        expected_action[dof] += rowValue( x, y, component ) * pair_column_sum;
      }
    }
  }
  std::unique_ptr<mfem::HypreParVector> residual_reference( expected_residual.ParallelAssemble() );
  std::unique_ptr<mfem::HypreParVector> action_reference( expected_action.ParallelAssemble() );

  auto& coupling_scheme = scheme();
  const auto lower_mesh = coupling_scheme.getMesh1().getView();
  const auto upper_mesh = coupling_scheme.getMesh2().getView();
  const auto pair_list = pairs();
  ContactFormulationResult assembled_result;
  {
    auto residual_assembler =
        tribol::detail::makeResidualVectorAssembler( *coupling_scheme.getMfemMeshData(), lower_mesh, upper_mesh );
    auto jacobian_assembler = tribol::detail::makeResidualJacobianAssembler(
        *coupling_scheme.getMfemMeshData(), *coupling_scheme.getMfemJacobianData(), lower_mesh, upper_mesh,
        pair_list.size(), pair_list.getAllocatorID() );
    ArrayT<RealT> residuals( pair_list.size() * pair_dofs, pair_list.size() * pair_dofs, pair_list.getAllocatorID() );
    ArrayT<RealT> jacobians( pair_list.size() * pair_dofs * pair_dofs, pair_list.size() * pair_dofs * pair_dofs,
                             pair_list.getAllocatorID() );
    for ( IndexT pair = 0; pair < pair_list.size(); ++pair ) {
      const auto element_pair = pair_list[pair];
      std::array<MdArray<RealT, nodes_per_edge, spatial_dim>, pair_size> pair_coordinates;
      lower_mesh.getFaceCoords( element_pair.element_id1_, pair_coordinates[0].data() );
      upper_mesh.getFaceCoords( element_pair.element_id2_, pair_coordinates[1].data() );
      MdArray<RealT, pair_dofs> columns{};
      for ( int side = 0; side < pair_size; ++side ) {
        for ( int node = 0; node < nodes_per_edge; ++node ) {
          for ( int component = 0; component < spatial_dim; ++component ) {
            const int dof = ( side * nodes_per_edge + node ) * spatial_dim + component;
            residuals[pair * pair_dofs + dof] =
                rowValue( pair_coordinates[side]( node, 0 ), pair_coordinates[side]( node, 1 ), component );
            columns[dof] =
                columnValue( pair_coordinates[side]( node, 0 ), pair_coordinates[side]( node, 1 ), component );
          }
        }
      }
      for ( int row = 0; row < pair_dofs; ++row ) {
        for ( int column = 0; column < pair_dofs; ++column ) {
          jacobians[( pair * pair_dofs + row ) * pair_dofs + column] =
              residuals[pair * pair_dofs + row] * columns[column];
        }
      }
    }
    const auto residuals_view = residuals.view();
    const auto jacobians_view = jacobians.view();
    residual_assembler.addPairs<ExecutionMode::Sequential, pair_dofs, 2>( pair_list,
                                                                          StoredResidualReader{ residuals_view } );
    constexpr IndexT size_geom_coords = pair_dofs / 2;
    jacobian_assembler.addPairs<ExecutionMode::Sequential, size_geom_coords, size_geom_coords, 4>(
        pair_list, StoredJacobianReader{ jacobians_view } );
    assembled_result.residual.reset( residual_assembler.assemble().release() );
    assembled_result.df_dx.reset( jacobian_assembler.assemble().release() );
  }
  ASSERT_NE( assembled_result.residual, nullptr );
  ASSERT_NE( assembled_result.df_dx, nullptr );
  EXPECT_FALSE( assembled_result.energy.has_value() );
  EXPECT_FALSE( assembled_result.time_step.has_value() );
  EXPECT_EQ( assembled_result.gap, nullptr );
  EXPECT_EQ( assembled_result.pressure, nullptr );
  EXPECT_EQ( assembled_result.weighted_gap_jacobian, nullptr );
  EXPECT_EQ( assembled_result.df_dp, nullptr );
  auto residual_difference = *assembled_result.residual;
  residual_difference -= *residual_reference;
  EXPECT_LT( norm( residual_difference ), assembly_tolerance );
  mfem::Vector jacobian_action( coordinate_space_->GetTrueVSize() );
  assembled_result.df_dx->Mult( true_direction, jacobian_action );
  jacobian_action -= *action_reference;
  EXPECT_LT( norm( jacobian_action ), assembly_tolerance );
  int rank = 0;
  int rank_count = 0;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &rank_count );
  constexpr int two_rank_run = 2;
  constexpr int rank_without_contact_elements = 1;
  if ( rank_count == two_rank_run && rank == rank_without_contact_elements ) {
    EXPECT_EQ( lower_mesh.numberOfElements() + upper_mesh.numberOfElements(), 0 );
  }
}

// Empty assembly must return usable zero outputs, and rejected layouts must not write partial contributions.
TEST_F( MfemResidualAndJacobianAssemblyTest, returns_zero_outputs_for_empty_or_rejected_input )
{
  auto& coupling_scheme = scheme();
  const auto lower_mesh = coupling_scheme.getMesh1().getView();
  const auto upper_mesh = coupling_scheme.getMesh2().getView();
  auto residual_assembler =
      tribol::detail::makeResidualVectorAssembler( *coupling_scheme.getMfemMeshData(), lower_mesh, upper_mesh );
  auto jacobian_assembler = tribol::detail::makeResidualJacobianAssembler(
      *coupling_scheme.getMfemMeshData(), *coupling_scheme.getMfemJacobianData(), lower_mesh, upper_mesh, 0,
      lower_mesh.getAllocatorId() );
  const ArrayT<InterfacePair> empty_pairs( 0, 0, lower_mesh.getAllocatorId() );
  EXPECT_SLIC_ERROR( (
      residual_assembler.addPairs<ExecutionMode::Sequential, pair_dofs - 1, 2>( empty_pairs, ZeroResidualReader{} ) ) );
  constexpr IndexT size_geom_coords = pair_dofs / 2;
  EXPECT_SLIC_ERROR( ( jacobian_assembler.view<size_geom_coords - 1, size_geom_coords, 4>() ) );
  const auto residual = residual_assembler.assemble();
  const auto df_dx = jacobian_assembler.assemble();
  EXPECT_EQ( residual.size(), coordinate_space_->GetTrueVSize() );
  EXPECT_EQ( norm( residual.get() ), 0.0 );
  mfem::Vector direction( coordinate_space_->GetTrueVSize() ), action( coordinate_space_->GetTrueVSize() );
  direction = 1.0;
  df_dx->Mult( direction, action );
  EXPECT_EQ( norm( action ), 0.0 );
}

// Three signed scalar fields retain their values and gradients without normalization or contact activation.
TEST_F( MfemResidualAndJacobianAssemblyTest, assembles_scalar_fields_and_dual_transfers )
{
  prepareDualData();
  auto& coupling_scheme = scheme();
  const auto lower_mesh = coupling_scheme.getMesh1().getView();
  const auto upper_mesh = coupling_scheme.getMesh2().getView();
  auto& dual_data = *coupling_scheme.getMfemSubmeshData();
  const auto& dual_fes = dual_data.GetSubmeshFESpace();
  shared::ParVector field( const_cast<mfem::ParFiniteElementSpace*>( &dual_fes ) );
  field.fill( 2.5 );
  const auto redecomp_field = dual_data.TransferTrueDofsToRedecomp( field.get() );
  for ( int dof = 0; dof < redecomp_field.Size(); ++dof ) {
    EXPECT_NEAR( redecomp_field[dof], 2.5, assembly_tolerance );
  }
  mfem::Vector wrong_size( dual_fes.GetTrueVSize() + 1 );
  EXPECT_SLIC_ERROR( dual_data.TransferTrueDofsToRedecomp( wrong_size ) );

  constexpr IndexT num_fields = 3;
  const MdArray<RealT, num_fields> factors{ 1.0, -2.0, 3.0 };
  const auto pair_list = pairs();
  int local_pairs = pair_list.size();
  int global_pairs = 0;
  MPI_Allreduce( &local_pairs, &global_pairs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  const auto assembled_fields =
      tribol::detail::assembleNodalValuesAndJacobians<ExecutionMode::Sequential, num_fields, nodes_per_edge, pair_dofs>(
          pair_list, lower_mesh, upper_mesh, *coupling_scheme.getMfemMeshData(), dual_data,
          *coupling_scheme.getMfemJacobianData(), ScaledNodalValueReader{ factors },
          ScaledNodalGradientReader{ factors } );
  ASSERT_EQ( assembled_fields.size(), num_fields );
  mfem::Vector direction( coordinate_space_->GetTrueVSize() ), action( dual_fes.GetTrueVSize() );
  direction = 1.0;
  // Local node weights sum to three, and the coordinate-column weights sum to 8*9/2.
  constexpr RealT node_weight_sum = 3.0;
  constexpr RealT column_weight_sum = pair_dofs * ( pair_dofs + 1 ) / 2.0;
  for ( IndexT field_index = 0; field_index < num_fields; ++field_index ) {
    SCOPED_TRACE( field_index );
    RealT local_sum = assembled_fields[field_index].values.get().Sum();
    RealT global_sum = 0.0;
    MPI_Allreduce( &local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    const RealT expected = global_pairs * node_weight_sum * factors[field_index];
    EXPECT_NEAR( global_sum, expected, assembly_tolerance );
    assembled_fields[field_index].jacobian->Mult( direction, action );
    RealT local_action = action.Sum();
    RealT global_action = 0.0;
    MPI_Allreduce( &local_action, &global_action, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    EXPECT_NEAR( global_action, expected * column_weight_sum, assembly_tolerance );
  }
}

// Scalar vector accumulation needs no Jacobian data. Invalid view block counts add nothing.
TEST_F( MfemResidualAndJacobianAssemblyTest, scalar_values_need_no_jacobian_data )
{
  prepareDualData();
  auto& coupling_scheme = scheme();
  coupling_scheme.setMfemJacobianData( nullptr );
  const auto& dual_data = *coupling_scheme.getMfemSubmeshData();
  const auto lower_mesh = coupling_scheme.getMesh1().getView();
  auto scalar_assembler = tribol::detail::makeScalarFieldVectorAssembler( dual_data, lower_mesh );
  EXPECT_SLIC_ERROR( ( scalar_assembler.view<ExecutionMode::Sequential, 2>() ) );
  const auto zero = scalar_assembler.assemble();
  EXPECT_EQ( norm( zero.get() ), 0.0 );
  const auto pair_list = pairs();
  scalar_assembler.addPairs<ExecutionMode::Sequential, nodes_per_edge, 1>( pair_list, ScalarRowReader{} );
  const auto assembled_values = scalar_assembler.assemble();
  RealT local_sum = assembled_values.get().Sum();
  RealT global_sum = 0.0;
  int local_pairs = pair_list.size();
  int global_pairs = 0;
  MPI_Allreduce( &local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_pairs, &global_pairs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( global_sum, 3.0 * global_pairs, assembly_tolerance );
}

/** @brief Exercise matrix assembly with each enabled host execution mode. */
class MfemMatrixAssemblyTest : public MfemResidualAndJacobianAssemblyTest,
                               public testing::WithParamInterface<ExecutionMode> {};

// Rectangular packing must transfer scalar and coordinate spaces in either orientation, including nonmatching element
// maps and a rank with no local pairs. Host and execution-space packing must produce the same nonsymmetric entries.
TEST_P( MfemMatrixAssemblyTest, assembles_rectangular_matrices_in_both_orientations )
{
  dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    prepareDualData();
    auto& coupling_scheme = scheme();
    const auto& mfem_mesh_data = *coupling_scheme.getMfemMeshData();
    const auto& dual_data = *coupling_scheme.getMfemSubmeshData();
    const auto& jacobian_data = *coupling_scheme.getMfemJacobianData();
    const auto pair_list = pairs();
    const auto& lower_element_map = mfem_mesh_data.GetElemMap1();
    const auto& upper_element_map = mfem_mesh_data.GetElemMap2();
    const MfemMatrixSpace scalar_space{ jacobian_data.SubmeshPath(), dual_data.GetRedecompFESpace() };
    const MfemMatrixSpace coordinate_space{ jacobian_data.ParentPath(),
                                            *mfem_mesh_data.GetRedecompResponse().FESpace() };
    constexpr IndexT size_geom_coords = nodes_per_edge * spatial_dim;
    const std::vector<MfemMatrixBlock> forward_blocks{
        { 0, 0, nodes_per_edge, nodes_per_edge, 0, 0, lower_element_map, lower_element_map },
        { 0, 1, nodes_per_edge, nodes_per_edge, 0, size_geom_coords, lower_element_map, upper_element_map } };
    const std::vector<MfemMatrixBlock> reverse_blocks{
        { 0, 0, nodes_per_edge, nodes_per_edge, 0, 0, lower_element_map, lower_element_map },
        { 1, 0, nodes_per_edge, nodes_per_edge, size_geom_coords, 0, upper_element_map, lower_element_map } };
    const int allocator_id = pair_list.getAllocatorID();
    MfemMatrixAssembler forward_assembler( scalar_space, coordinate_space, forward_blocks, pair_list.size(),
                                           allocator_id );
    MfemMatrixAssembler reverse_assembler( coordinate_space, scalar_space, reverse_blocks, pair_list.size(),
                                           allocator_id );
    MfemMatrixAssembler reference_assembler( scalar_space, coordinate_space, forward_blocks, pair_list.size(),
                                             allocator_id );
    MfemMatrixAssembler empty_forward_assembler( scalar_space, coordinate_space, forward_blocks, 0, allocator_id );
    const ArrayT<InterfacePair> wrong_pair_count( pair_list.size() + 1, pair_list.size() + 1, allocator_id );

    EXPECT_SLIC_ERROR( ( forward_assembler.view<3, size_geom_coords, 2>() ) );
    auto invalid_blocks = forward_blocks;
    invalid_blocks[0].row_geometry = 2;
    EXPECT_SLIC_ERROR( MfemMatrixAssembler( scalar_space, coordinate_space, invalid_blocks, 0, allocator_id ) );
    EXPECT_SLIC_ERROR( MfemMatrixAssembler( scalar_space, coordinate_space, forward_blocks, -1, allocator_id ) );
    const auto zero = empty_forward_assembler.assemble();
    mfem::Vector coordinate_direction( coordinate_space_->GetTrueVSize() ), scalar_action( zero->Height() );
    coordinate_direction = 1.0;
    zero->Mult( coordinate_direction, scalar_action );
    EXPECT_EQ( norm( scalar_action ), 0.0 );

    const ForwardMatrixReader read_forward;
    const ReverseMatrixReader read_reverse;
    EXPECT_SLIC_ERROR(
        ( forward_assembler.addPairs<Exec, nodes_per_edge, size_geom_coords, 2>( wrong_pair_count, read_forward ) ) );
    reference_assembler.addPairs<ExecutionMode::Sequential, nodes_per_edge, size_geom_coords, 2>( pair_list,
                                                                                                  read_forward );
    forward_assembler.addPairs<Exec, nodes_per_edge, size_geom_coords, 2>( pair_list, read_forward );
    reverse_assembler.addPairs<Exec, size_geom_coords, nodes_per_edge, 2>( pair_list, read_reverse );
    const auto expected_forward = reference_assembler.assemble();
    const auto expected_reverse = expected_forward.transpose();
    const auto actual_forward = forward_assembler.assemble();
    const auto actual_reverse = reverse_assembler.assemble();
    EXPECT_EQ( actual_forward->Height(), dual_data.GetSubmeshFESpace().GetTrueVSize() );
    EXPECT_EQ( actual_forward->Width(), coordinate_space_->GetTrueVSize() );
    EXPECT_EQ( actual_reverse->Height(), coordinate_space_->GetTrueVSize() );
    EXPECT_EQ( actual_reverse->Width(), dual_data.GetSubmeshFESpace().GetTrueVSize() );
    mfem::Vector expected_scalar( actual_forward->Height() );
    for ( int dof = 0; dof < coordinate_direction.Size(); ++dof ) {
      coordinate_direction[dof] = 1.0 + 0.1 * dof;
    }
    expected_forward->Mult( coordinate_direction, expected_scalar );
    actual_forward->Mult( coordinate_direction, scalar_action );
    scalar_action -= expected_scalar;
    EXPECT_LT( norm( scalar_action ), assembly_tolerance );
    mfem::Vector scalar_direction( actual_reverse->Width() ), actual_coordinate_action( actual_reverse->Height() ),
        expected_coordinate_action( actual_reverse->Height() );
    for ( int dof = 0; dof < scalar_direction.Size(); ++dof ) {
      scalar_direction[dof] = 1.0 + 0.2 * dof;
    }
    expected_reverse->Mult( scalar_direction, expected_coordinate_action );
    actual_reverse->Mult( scalar_direction, actual_coordinate_action );
    EXPECT_GT( norm( expected_coordinate_action ), 0.0 );
    actual_coordinate_action -= expected_coordinate_action;
    EXPECT_LT( norm( actual_coordinate_action ), assembly_tolerance );
  } );
}

// Separate and composed kernels must preserve all four matrix blocks and shared-node residual accumulation.
TEST_P( MfemMatrixAssemblyTest, composes_residual_and_jacobian_assembly_views )
{
  dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    auto& coupling_scheme = scheme();
    const auto& mfem_mesh_data = *coupling_scheme.getMfemMeshData();
    const auto& jacobian_data = *coupling_scheme.getMfemJacobianData();
    const auto lower_mesh = coupling_scheme.getMesh1().getView();
    const auto upper_mesh = coupling_scheme.getMesh2().getView();
    const auto pair_list = pairs();
    auto reference_residual_assembler =
        tribol::detail::makeResidualVectorAssembler( mfem_mesh_data, lower_mesh, upper_mesh );
    auto composed_residual_assembler =
        tribol::detail::makeResidualVectorAssembler( mfem_mesh_data, lower_mesh, upper_mesh );
    auto residual_only_assembler =
        tribol::detail::makeResidualVectorAssembler( mfem_mesh_data, lower_mesh, upper_mesh );
    const int allocator_id = pair_list.getAllocatorID();
    auto reference_jacobian_assembler = tribol::detail::makeResidualJacobianAssembler(
        mfem_mesh_data, jacobian_data, lower_mesh, upper_mesh, pair_list.size(), allocator_id );
    auto composed_jacobian_assembler = tribol::detail::makeResidualJacobianAssembler(
        mfem_mesh_data, jacobian_data, lower_mesh, upper_mesh, pair_list.size(), allocator_id );
    auto jacobian_only_assembler = tribol::detail::makeResidualJacobianAssembler(
        mfem_mesh_data, jacobian_data, lower_mesh, upper_mesh, pair_list.size(), allocator_id );
    const auto initial_residual = residual_only_assembler.assemble();
    EXPECT_EQ( norm( initial_residual.get() ), 0.0 );
    EXPECT_SLIC_ERROR(
        ( residual_only_assembler.addPairs<Exec, pair_dofs - 1, 2>( pair_list, ZeroResidualReader{} ) ) );
    constexpr IndexT size_geom_coords = pair_dofs / 2;
    const ResidualRowReader read_residual;
    const JacobianEntryReader read_jacobian;
    reference_residual_assembler.addPairs<ExecutionMode::Sequential, pair_dofs, 2>( pair_list, read_residual );
    reference_jacobian_assembler.addPairs<ExecutionMode::Sequential, size_geom_coords, size_geom_coords, 4>(
        pair_list, read_jacobian );
    residual_only_assembler.addPairs<Exec, pair_dofs, 2>( pair_list, read_residual );
    jacobian_only_assembler.addPairs<Exec, size_geom_coords, size_geom_coords, 4>( pair_list, read_jacobian );
    const auto matrix_view = composed_jacobian_assembler.view<size_geom_coords, size_geom_coords, 4>();
    const auto residual_view = composed_residual_assembler.view<Exec, 2>();
    const auto pairs_view = pair_list.view();
    forAllExec<Exec>( pair_list.size(), ComposedAssemblyWriter{ residual_view, matrix_view, pairs_view } );
    const auto expected_residual = reference_residual_assembler.assemble();
    auto residual_only_result = residual_only_assembler.assemble();
    residual_only_result.get().Add( -1.0, expected_residual.get() );
    EXPECT_LT( norm( residual_only_result.get() ), assembly_tolerance );
    EXPECT_EQ( norm( initial_residual.get() ), 0.0 );
    EXPECT_GT( norm( expected_residual.get() ), 0.0 );
    auto composed_residual_result = composed_residual_assembler.assemble();
    composed_residual_result.get().Add( -1.0, expected_residual.get() );
    EXPECT_LT( norm( composed_residual_result.get() ), assembly_tolerance );
    mfem::Vector direction( coordinate_space_->GetTrueVSize() ), expected_action( coordinate_space_->GetTrueVSize() ),
        action( coordinate_space_->GetTrueVSize() );
    direction = 1.0;
    reference_jacobian_assembler.assemble()->Mult( direction, expected_action );
    composed_jacobian_assembler.assemble()->Mult( direction, action );
    action -= expected_action;
    EXPECT_LT( norm( action ), assembly_tolerance );
    jacobian_only_assembler.assemble()->Mult( direction, action );
    action -= expected_action;
    EXPECT_LT( norm( action ), assembly_tolerance );
  } );
}

// Three independent fields and their rectangular Jacobians must agree across host execution backends.
TEST_P( MfemMatrixAssemblyTest, assembles_scalar_fields_and_gradients )
{
  dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    prepareDualData();
    auto& coupling_scheme = scheme();
    const auto lower_mesh = coupling_scheme.getMesh1().getView();
    const auto upper_mesh = coupling_scheme.getMesh2().getView();
    const auto pair_list = pairs();
    constexpr IndexT num_fields = 3;
    const FieldValueReader read_value;
    const FieldGradientReader read_gradient;
    const auto sequential_fields =
        tribol::detail::assembleNodalValuesAndJacobians<ExecutionMode::Sequential, num_fields, nodes_per_edge,
                                                        pair_dofs>(
            pair_list, lower_mesh, upper_mesh, *coupling_scheme.getMfemMeshData(),
            *coupling_scheme.getMfemSubmeshData(), *coupling_scheme.getMfemJacobianData(), read_value, read_gradient );
    const auto backend_fields =
        tribol::detail::assembleNodalValuesAndJacobians<Exec, num_fields, nodes_per_edge, pair_dofs>(
            pair_list, lower_mesh, upper_mesh, *coupling_scheme.getMfemMeshData(),
            *coupling_scheme.getMfemSubmeshData(), *coupling_scheme.getMfemJacobianData(), read_value, read_gradient );
    ASSERT_EQ( backend_fields.size(), num_fields );
    mfem::Vector direction( coordinate_space_->GetTrueVSize() );
    direction = 1.0;
    const int dual_size = coupling_scheme.getMfemSubmeshData()->GetSubmeshFESpace().GetTrueVSize();
    mfem::Vector expected_action( dual_size ), action( dual_size );
    for ( IndexT field = 0; field < num_fields; ++field ) {
      SCOPED_TRACE( field );
      EXPECT_GT( norm( sequential_fields[field].values.get() ), 0.0 );
      auto value_difference = backend_fields[field].values.get();
      value_difference -= sequential_fields[field].values.get();
      EXPECT_LT( norm( value_difference ), assembly_tolerance );
      sequential_fields[field].jacobian->Mult( direction, expected_action );
      backend_fields[field].jacobian->Mult( direction, action );
      action -= expected_action;
      EXPECT_LT( norm( action ), assembly_tolerance );
    }
  } );
}

INSTANTIATE_TEST_SUITE_P( host_backends, MfemMatrixAssemblyTest,
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
  const int test_status = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_status;
}
