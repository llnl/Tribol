// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check host MFEM assembly with prescribed residuals and nonsymmetric pair Jacobians, without contact physics.
 * Nonmatching two- and three-edge surfaces exercise shared nodes and all four matrix blocks. Tests run on one and
 * two ranks, with contact data on rank zero. Physics and Enzyme derivatives are checked by the formulation tests.
 */
#include <cmath>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
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
