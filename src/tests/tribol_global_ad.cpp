// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check distributed AD vector/matrix arithmetic and sparse coordinate derivatives without Enzyme.
 * Analytical vector checks cover arithmetic, activation, supported reciprocals, and grouped field curvature.
 * Finite differences check nonlinear matrix-vector products, transposes, matrix products, and mixed second derivatives
 * with off-rank entries. Every case also runs with all mesh elements on rank zero, leaving another rank empty.
 * Contact formulation regressions cover geometry derivatives and Redecomp assembly.
 * Contact-result conversions check owned snapshots and errors for absent outputs or residual curvature.
 */
#include <cmath>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/mesh/GlobalAD.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/physics/ContactFormulation.hpp"

namespace {
using namespace tribol;
using Vector = shared::ParVector;
using Matrix = shared::ParSparseMat;
constexpr RealT tolerance = 1.0e-10;  ///< Roundoff allowance for distributed algebra and reductions.

/** @brief Provide a distributed scalar space, optionally leaving all other ranks empty. */
class GlobalADTest : public testing::TestWithParam<bool> {
 protected:
  /** @brief Build two quadrilaterals and distinct positive true-DOF values spanning open and compressed gaps. */
  void SetUp() override
  {
    auto serial = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0 );
    mfem::Array<int> partition( serial.GetNE() );
    partition = 0;
    mesh_ = std::make_unique<mfem::ParMesh>( MPI_COMM_WORLD, serial, GetParam() ? partition.GetData() : nullptr );
    basis_ = std::make_unique<mfem::H1_FECollection>( 1, mesh_->Dimension() );
    space_ = std::make_unique<mfem::ParFiniteElementSpace>( mesh_.get(), basis_.get() );
    values_ = Vector( space_.get() );
    for ( int i = 0; i < values_.size(); ++i ) values_[i] = 0.45 + 0.13 * ( space_->GetMyTDofOffset() + i + 1 );
  }
  /** @brief Allocate a constant field. @param value Fill value. @return Owned vector. */
  Vector filled( RealT value ) const
  {
    Vector result( space_.get() );
    result.fill( value );
    return result;
  }
  /** @brief Make a field-space diagonal. @param values Local diagonal values. @return Owned sparse matrix. */
  Matrix diagonalMatrix( const Vector& values ) const
  {
    return Matrix::diagonalMatrix( space_->GetComm(), space_->GlobalTrueVSize(), space_->GetTrueDofOffsets(),
                                   values.get() );
  }
  /** @brief Build a nonsymmetric matrix with neighbor coupling, including off-rank columns. @return Owned matrix. */
  Matrix neighborMatrix() const
  {
    mfem::Vector velocity( mesh_->Dimension() );
    velocity[0] = 1.0;
    velocity[1] = 0.5;
    mfem::VectorConstantCoefficient coefficient( velocity );
    mfem::ParBilinearForm form( space_.get() );
    form.AddDomainIntegrator( new mfem::MassIntegrator );
    form.AddDomainIntegrator( new mfem::ConvectionIntegrator( coefficient ) );
    form.Assemble();
    form.Finalize();
    return Matrix( form.ParallelAssemble() );
  }
  /** @brief Compute the maximum true-DOF difference across ranks. @param left First vector. @param right Second vector.
   * @return Global maximum absolute difference. */
  RealT difference( const Vector& left, const Vector& right ) const
  {
    RealT local = 0.0, global = 0.0;
    for ( int i = 0; i < left.size(); ++i ) local = std::max( local, std::abs( left[i] - right[i] ) );
    MPI_Allreduce( &local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    return global;
  }
  std::unique_ptr<mfem::ParMesh> mesh_;                 ///< Distributed mesh, possibly empty on some ranks.
  std::unique_ptr<mfem::H1_FECollection> basis_;        ///< Scalar finite-element basis.
  std::unique_ptr<mfem::ParFiniteElementSpace> space_;  ///< Value and independent-coordinate space.
  Vector values_;                                       ///< Independent coordinate snapshot in true-DOF ordering.
};

// Derivatives of x - 1 + 3/(x+1) - x/2 + 1/(x+2) check vector quotients and both derivative orders.
TEST_P( GlobalADTest, differentiates_distributed_vector_arithmetic )
{
  const auto x = ADVector::variable( *space_, values_ );
  const auto y = ( x * x + 2.0 ) / ( x + 1.0 ) - x / 2.0 + 1.0 / ( x + 2.0 );
  const auto energy = sum( y );
  const auto result = energy.differentiate();
  const auto ones = filled( 1.0 );
  const auto jacobian = y.jacobian();
  const auto action = jacobian * ones;
  const auto second = result.hessian * ones;
  RealT local = 0.0, expected = 0.0;
  for ( int i = 0; i < values_.size(); ++i ) {
    const RealT v = values_[i], a = v + 1.0, b = v + 2.0;
    local += v - 1.0 + 3.0 / a - v / 2.0 + 1.0 / b;
    const RealT first = 0.5 - 3.0 / ( a * a ) - 1.0 / ( b * b );
    const RealT curvature = 6.0 / ( a * a * a ) + 2.0 / ( b * b * b );
    EXPECT_NEAR( result.gradient[i], first, tolerance );
    EXPECT_NEAR( action[i], first, tolerance );
    EXPECT_NEAR( second[i], curvature, tolerance );
  }
  MPI_Allreduce( &local, &expected, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( energy.value(), expected, tolerance );
  const auto weighted = y.weightedHessian( ones ) * ones;
  EXPECT_LT( difference( second, weighted ), tolerance );
  const auto independent_gradient = energy.gradient();
  EXPECT_LT( difference( independent_gradient, result.gradient ), tolerance );
  const auto linear = 5.0 + ( 2.0 * energy - energy ) / 2.0;
  EXPECT_NEAR( linear.value(), 5.0 + energy.value() / 2.0, tolerance );
  EXPECT_LT( difference( linear.gradient(), result.gradient * 0.5 ), tolerance );
}

// Gap and area share one curvature call. Open gaps and unsupported areas must have zero pressure and derivatives.
TEST_P( GlobalADTest, shares_field_curvature_for_nodal_penalty_energy )
{
  Vector gap = filled( 0.0 ), area = filled( 0.0 ), gap_diagonal = filled( 0.0 ), area_diagonal = filled( 0.5 );
  for ( int i = 0; i < values_.size(); ++i ) {
    const RealT u = values_[i];
    gap[i] = u * u - 1.0;
    area[i] = 2.0 + 0.5 * u;
    gap_diagonal[i] = 2.0 * u;
    if ( space_->GetMyTDofOffset() + i == 0 ) area[i] = 0.0;
  }
  const auto jg = diagonalMatrix( gap_diagonal ), ja = diagonalMatrix( area_diagonal );
  int curvature_calls = 0;
  const DifferentiableNodalFields fields( *space_, *space_, { { gap, jg }, { area, ja } },
                                          [&]( const NodalEnergyGradient& weights ) {
                                            ++curvature_calls;
                                            auto diagonal = filled( 0.0 );
                                            if ( const auto found = weights.find( 0 ); found != weights.end() )
                                              for ( int i = 0; i < diagonal.size(); ++i )
                                                diagonal[i] = 2.0 * found->second[i];
                                            return diagonalMatrix( diagonal );
                                          } );
  const auto variables = makeADFields( fields );
  constexpr RealT stiffness = 4.0;
  const auto energy = EnergyMortarNodal{}.penaltyEnergy( variables[0], variables[1], stiffness );
  EXPECT_EQ( curvature_calls, 0 );
  const auto gradient = energy.gradient();
  EXPECT_EQ( curvature_calls, 0 );
  auto result = energy.differentiate();
  EXPECT_EQ( curvature_calls, 1 );
  EXPECT_TRUE( result.hasPartial( variables[0] ) );
  const auto& pressure = result.partial( variables[0] );
  const auto hessian_action = result.hessian * filled( 1.0 );
  RealT local = 0.0, expected_energy = 0.0;
  for ( int i = 0; i < values_.size(); ++i ) {
    const RealT u = values_[i], g = gap[i], a = area[i];
    RealT p = 0.0, force = 0.0, tangent = 0.0;
    if ( g < 0.0 && a > 0.0 ) {
      p = stiffness * g / a;
      force = 2.0 * u * p - 0.25 * stiffness * g * g / ( a * a );
      tangent = 2.0 * p + 4.0 * stiffness * u * u / a - 2.0 * stiffness * g * u / ( a * a ) +
                0.25 * stiffness * g * g / ( a * a * a );
      local += 0.5 * stiffness * g * g / a;
    }
    EXPECT_NEAR( pressure[i], p, tolerance );
    EXPECT_NEAR( gradient[i], force, tolerance );
    EXPECT_NEAR( result.gradient[i], force, tolerance );
    EXPECT_NEAR( hessian_action[i], tangent, tolerance );
  }
  MPI_Allreduce( &local, &expected_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( result.value, expected_energy, tolerance );
  auto owned_pressure = result.releasePartial( variables[0] );
  EXPECT_FALSE( result.hasPartial( variables[0] ) );
  EXPECT_THROW( result.releasePartial( variables[0] ), std::out_of_range );
  const auto changed = ( 2.0 * energy ).differentiate();
  EXPECT_EQ( curvature_calls, 2 );
  EXPECT_LT( difference( changed.gradient, result.gradient * 2.0 ), tolerance );
}

// Equality retains the compressive tangent, while fixed multiplier values retain only the supplied field curvature.
TEST_P( GlobalADTest, preserves_zero_gap_activation_and_independent_multipliers )
{
  const auto x = ADVector::variable( *space_, values_ );
  const auto zero = x - x;
  const auto g = x - ADVector::constant( *space_, *space_, values_ );
  const auto clipped = min( g, 0.0 );
  const auto energy = 0.5 * dot( clipped, clipped );
  const auto result = energy.differentiate();
  const auto ones = filled( 1.0 );
  EXPECT_LT( difference( result.gradient, filled( 0.0 ) ), tolerance );
  EXPECT_LT( difference( result.hessian * ones, ones ), tolerance );
  EXPECT_LT( difference( min( 0.0, g ).jacobian() * ones, filled( 0.0 ) ), tolerance );
  EXPECT_LT( difference( max( g, 0.0 ).jacobian() * ones, ones ), tolerance );
  EXPECT_LT( difference( zero.jacobian() * ones, filled( 0.0 ) ), tolerance );
  const auto multiplier = ADVector::constant( *space_, *space_, filled( -2.0 ) );
  const auto lm = dot( multiplier, x * x ).differentiate();
  EXPECT_LT( difference( lm.gradient, values_ * ( -4.0 ) ), tolerance );
  EXPECT_LT( difference( lm.hessian * ones, filled( -4.0 ) ), tolerance );
}

// A(x) = (diag(x²)B + diag(x)/4 - B)/2 has matrix curvature and nonsymmetric off-rank entries.
// Finite differences of the direct vector formulas check every global Hessian column and both matrix actions.
TEST_P( GlobalADTest, differentiates_active_matrix_products_and_transposes )
{
  const auto matrix = ADMatrix::constant( *space_, *space_, neighborMatrix() );
  const auto build = [&]( const Vector& values ) {
    const auto x = ADVector::variable( *space_, values );
    const auto a = ( diagonal( x * x ) * matrix + 0.25 * diagonal( x ) - matrix ) / 2.0;
    const auto y = a * x, z = a.transpose() * x;
    return 0.5 * dot( y, y ) + 0.2 * dot( z, z );
  };
  const auto numerical_energy = [&]( const Vector& values ) {
    const auto square = values.multiply( values );
    const auto bu = matrix.values() * values;
    auto y = square.multiply( bu ) + 0.25 * square - bu;
    y *= 0.5;
    auto z = square.multiply( values ) * matrix.values() + 0.25 * square - values * matrix.values();
    z *= 0.5;
    return 0.5 * y.dot( y ) + 0.2 * z.dot( z );
  };
  const auto energy = build( values_ );
  const auto result = energy.differentiate();
  EXPECT_NEAR( energy.value(), numerical_energy( values_ ), tolerance );
  constexpr RealT step = 1.0e-6;  // Perturb smooth matrix coefficients without crossing any domain boundary.
  constexpr RealT difference_tolerance = 1.0e-6;  // Allow finite-difference and distributed algebra roundoff.
  for ( HYPRE_BigInt column = 0; column < space_->GlobalTrueVSize(); ++column ) {
    auto direction = filled( 0.0 );
    const auto offset = space_->GetMyTDofOffset();
    if ( column >= offset && column < offset + values_.size() ) direction[column - offset] = 1.0;
    const auto plus = values_ + step * direction, minus = values_ - step * direction;
    const RealT reference = ( numerical_energy( plus ) - numerical_energy( minus ) ) / ( 2.0 * step );
    EXPECT_NEAR( result.gradient.dot( direction ), reference, difference_tolerance );
    const auto finite_column = ( build( plus ).gradient() - build( minus ).gradient() ) * ( 1.0 / ( 2.0 * step ) );
    EXPECT_LT( difference( result.hessian * direction, finite_column ), difference_tolerance );
  }
}

// Custom actions for A=diag(x²) must include matrix curvature and both mixed terms in A*x=x³.
TEST_P( GlobalADTest, uses_supplied_matrix_action_derivatives )
{
  int curvature_calls = 0;
  const auto action = [&]( const Vector& fixed ) {
    auto values = filled( 0.0 ), derivative = filled( 0.0 );
    for ( int i = 0; i < values.size(); ++i ) {
      const RealT u = values_[i];
      values[i] = u * u * fixed[i];
      derivative[i] = 2.0 * u * fixed[i];
    }
    return ADVector::fromDerivatives( *space_, *space_, std::move( values ), diagonalMatrix( derivative ),
                                      [&, fixed = Vector( fixed )]( const Vector& weights ) {
                                        ++curvature_calls;
                                        return diagonalMatrix( 2.0 * fixed.multiply( weights ) );
                                      } );
  };
  const auto x = ADVector::variable( *space_, values_ );
  const auto matrix =
      ADMatrix::fromActions( *space_, *space_, *space_, diagonalMatrix( values_.multiply( values_ ) ), action, action );
  const auto energy = sum( matrix * x );
  const auto gradient = energy.gradient();
  EXPECT_EQ( curvature_calls, 0 );
  const auto derivatives = energy.differentiate();
  EXPECT_EQ( curvature_calls, 1 );
  const auto hessian_action = derivatives.hessian * filled( 1.0 );
  for ( int i = 0; i < values_.size(); ++i ) {
    EXPECT_NEAR( gradient[i], 3.0 * values_[i] * values_[i], tolerance );
    EXPECT_NEAR( derivatives.gradient[i], gradient[i], tolerance );
    EXPECT_NEAR( hessian_action[i], 6.0 * values_[i], tolerance );
  }
}

// Rectangular matrix actions must preserve distinct value/coordinate partitions and return all transpose contributions.
TEST_P( GlobalADTest, differentiates_rectangular_constant_matrix_actions )
{
  mfem::ParFiniteElementSpace coordinates( mesh_.get(), basis_.get(), 2 );
  Vector values( &coordinates );
  values.fill( 1.0 );
  mfem::SparseMatrix local( space_->GetTrueVSize(), coordinates.GetTrueVSize() );
  for ( int row = 0; row < space_->GetTrueVSize(); ++row ) {
    local.Add( row, 2 * row, 1.0 );
    local.Add( row, 2 * row + 1, 2.0 );
  }
  local.Finalize();
  Matrix matrix( space_->GetComm(), space_->GlobalTrueVSize(), coordinates.GlobalTrueVSize(),
                 space_->GetTrueDofOffsets(), coordinates.GetTrueDofOffsets(), std::move( local ) );
  const auto a = ADMatrix::constant( *space_, coordinates, std::move( matrix ) );
  const auto x = ADVector::variable( coordinates, values );
  const auto y = a * x;
  const auto result = ( 0.5 * dot( y, y ) ).differentiate();
  const auto action = result.hessian * values;
  for ( int row = 0; row < space_->GetTrueVSize(); ++row ) {
    EXPECT_NEAR( y.values()[row], 3.0, tolerance );
    EXPECT_NEAR( result.gradient[2 * row], 3.0, tolerance );
    EXPECT_NEAR( result.gradient[2 * row + 1], 6.0, tolerance );
    EXPECT_NEAR( action[2 * row], 3.0, tolerance );
    EXPECT_NEAR( action[2 * row + 1], 6.0, tolerance );
  }
}

// A supplied global gradient and Hessian must combine with arithmetic-derived contributions without losing either.
TEST_P( GlobalADTest, combines_custom_scalar_coordinate_derivatives )
{
  auto gradient = filled( 0.0 ), curvature = filled( 0.0 );
  RealT local = 0.0, global = 0.0;
  for ( int i = 0; i < values_.size(); ++i ) {
    const RealT u = values_[i];
    local += u * u * u;
    gradient[i] = 3.0 * u * u;
    curvature[i] = 6.0 * u;
  }
  MPI_Allreduce( &local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  const auto custom = ADScalar::fromDerivatives( *space_, global, std::move( gradient ), diagonalMatrix( curvature ) );
  const auto x = ADVector::variable( *space_, values_ );
  const auto result = ( 2.0 * custom + 0.5 * dot( x, x ) - custom ).differentiate();
  const auto action = result.hessian * filled( 1.0 );
  for ( int i = 0; i < values_.size(); ++i ) {
    const RealT u = values_[i];
    EXPECT_NEAR( result.gradient[i], 3.0 * u * u + u, tolerance );
    EXPECT_NEAR( action[i], 6.0 * u + 1.0, tolerance );
  }
  EXPECT_NEAR( result.value, global + 0.5 * values_.dot( values_ ), tolerance );
}

// An assembled energy remains differentiable after its original outputs are changed and destroyed.
TEST_P( GlobalADTest, composes_owned_contact_energy )
{
  const auto energy = [&] {
    ContactFormulationResult result;
    result.energy = 0.5 * values_.dot( values_ );
    auto gradient = values_;
    auto hessian = diagonalMatrix( filled( 1.0 ) );
    result.force.reset( gradient.release() );
    result.df_dx.reset( hessian.release() );
    const auto snapshot = result.makeADEnergy( *space_ );
    *result.force = 0.0;
    *result.df_dx *= 0.0;
    return snapshot;
  }();
  const auto x = ADVector::variable( *space_, values_ );
  const auto combined = ( 3.0 * energy + dot( x, x ) ).differentiate();
  EXPECT_NEAR( combined.value, 2.5 * values_.dot( values_ ), tolerance );
  EXPECT_LT( difference( combined.gradient, values_ * 5.0 ), tolerance );
  EXPECT_LT( difference( combined.hessian * filled( 1.0 ), filled( 5.0 ) ), tolerance );
}

// Residuals expose supplied first derivatives; second derivatives require an explicit curvature provider.
TEST_P( GlobalADTest, composes_owned_contact_residual_with_optional_curvature )
{
  ContactFormulationResult result;
  EXPECT_THROW( result.makeADEnergy( *space_ ), std::logic_error );
  EXPECT_THROW( result.makeADResidual( *space_ ), std::logic_error );
  auto residual_values = filled( 0.0 );
  for ( int i = 0; i < values_.size(); ++i ) residual_values[i] = values_[i] * values_[i];
  auto jacobian = diagonalMatrix( values_ * 2.0 );
  result.force.reset( residual_values.release() );
  EXPECT_THROW( result.makeADResidual( *space_ ), std::logic_error );
  result.df_dx.reset( jacobian.release() );
  EXPECT_THROW( result.makeADEnergy( *space_ ), std::logic_error );
  const auto first_order = result.makeADResidual( *space_ );
  const auto second_order =
      result.makeADResidual( *space_, [&]( const Vector& weights ) { return diagonalMatrix( weights * 2.0 ); } );
  result = {};
  EXPECT_LT( difference( sum( first_order ).gradient(), values_ * 2.0 ), tolerance );
  EXPECT_THROW( sum( first_order ).hessian(), std::logic_error );
  const auto combined = ( 0.5 * dot( second_order, second_order ) ).differentiate();
  auto expected_gradient = filled( 0.0 ), expected_action = filled( 0.0 );
  for ( int i = 0; i < values_.size(); ++i ) {
    const RealT u = values_[i];
    EXPECT_NEAR( first_order.values()[i], u * u, tolerance );
    expected_gradient[i] = 2.0 * u * u * u;
    expected_action[i] = 6.0 * u * u;
  }
  EXPECT_LT( difference( combined.gradient, expected_gradient ), tolerance );
  EXPECT_LT( difference( combined.hessian * filled( 1.0 ), expected_action ), tolerance );
}

// Undefined division and missing curvature must fail explicitly; first derivatives remain usable without curvature.
TEST_P( GlobalADTest, rejects_invalid_operations_and_preserves_owned_derivative_results )
{
  const auto x = ADVector::variable( *space_, values_ );
  const auto zero = ADVector::constant( *space_, *space_, filled( 0.0 ) );
  EXPECT_THROW( x / zero, std::domain_error );
  EXPECT_THROW( ADMatrix::fromActions( *space_, *space_, *space_, diagonalMatrix( filled( 1.0 ) ), {}, {} ),
                std::invalid_argument );
  EXPECT_THROW( x / 0.0, std::domain_error );
  EXPECT_THROW( reciprocal( x, -1.0 ), std::invalid_argument );
  mfem::ParFiniteElementSpace other( mesh_.get(), basis_.get() );
  Vector other_values( &other );
  other_values.fill( 0.0 );
  const auto different = ADVector::variable( other, std::move( other_values ) );
  EXPECT_THROW( x + different, std::invalid_argument );
  const auto first_only = ADVector::fromDerivatives( *space_, *space_, values_, diagonalMatrix( filled( 1.0 ) ) );
  EXPECT_LT( difference( sum( first_only ).gradient(), filled( 1.0 ) ), tolerance );
  EXPECT_THROW( sum( first_only ).hessian(), std::logic_error );
  const auto owned = [&] {
    const auto temporary = ADVector::variable( *space_, values_ );
    return dot( temporary, temporary ).differentiate();
  }();
  EXPECT_LT( difference( owned.gradient, values_ * 2.0 ), tolerance );
  EXPECT_LT( difference( owned.hessian * filled( 1.0 ), filled( 2.0 ) ), tolerance );
}
INSTANTIATE_TEST_SUITE_P( Partitions, GlobalADTest, testing::Values( false, true ) );
}  // namespace

/** @brief Run global AD tests with MPI and logging. @param argc Argument count. @param argv Arguments. @return Status.
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
