// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
// SPDX-License-Identifier: (MIT)
/**
 * @file
 * @brief Verify Puso-Laursen mortar quantities, direct contact equations, accepted-step history, and MFEM Jacobians.
 * Local tests distinguish current unit normals from reference-scaled normals and check contact-law branches.
 * Nonmatching triangle and quad patches exercise shared-node assembly, changing pairs, all four Jacobian blocks,
 * parent ordering, empty Redecomp ranks, and generic MFEM residual/traction access. The same executable runs on one
 * and two MPI ranks.
 */
#include <cmath>
#include <limits>
#include <stdexcept>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/Mortar.hpp"

namespace {
using namespace tribol;
using detail::CoulombNodeInput;
using detail::evaluateCoulombNode;

// Stretching the surface changes corner cross-product magnitudes but must preserve each unit corner normal.
TEST( CoulombNormals, current_unit_normals_and_derivatives )
{
  MdArray<RealT, 12> x{ 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.2, 0.0 };
  auto result = detail::computeCornerNormals( x.data(), 4 );
  for ( int i = 0; i < 4; ++i ) {
    RealT norm = 0.0;
    for ( int d = 0; d < 3; ++d ) norm += result.values[d * 4 + i] * result.values[d * 4 + i];
    EXPECT_NEAR( norm, 1.0, 1.0e-14 );
  }
  MdArray<RealT, 12> stretched{};
  for ( int i = 0; i < 12; ++i ) stretched[i] = 2.0 * x[i];
  auto scaled = detail::computeCornerNormals( stretched.data(), 4 );
  for ( int i = 0; i < 12; ++i ) EXPECT_NEAR( result.values[i], scaled.values[i], 1.0e-14 );
  constexpr RealT h = 1.0e-6;
  for ( int col = 0; col < 12; ++col ) {
    x[col] += h;
    auto plus = detail::computeCornerNormals( x.data(), 4 );
    x[col] -= 2.0 * h;
    auto minus = detail::computeCornerNormals( x.data(), 4 );
    x[col] += h;
    for ( int row = 0; row < 12; ++row )
      EXPECT_NEAR( result.derivative( row, col ), ( plus.values[row] - minus.values[row] ) / ( 2.0 * h ), 1.0e-8 );
  }
  MdArray<RealT, 9> collapsed{};
  EXPECT_THROW( detail::computeCornerNormals( collapsed.data(), 3 ), std::invalid_argument );
}

// Equation 20 must equal the weight differences in equation 18 when the accepted gap vector is assembled explicitly.
TEST( CoulombMortarWeights, slip_equations_agree )
{
  MdArray<RealT, 12> old1{ 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0 };
  MdArray<RealT, 12> old2{ 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  MdArray<RealT, 12> now1{};
  for ( int i = 0; i < 12; ++i ) now1[i] = old1[i];
  for ( int i = 0; i < 4; ++i ) now1[i] += 0.15;
  auto before = detail::computeMortarPairWeights( old1.data(), 4, old2.data(), 4, 1.0e-8 );
  auto after = detail::computeMortarPairWeights( now1.data(), 4, old2.data(), 4, 1.0e-8 );
  for ( int i = 0; i < 4; ++i )
    for ( int d = 0; d < 2; ++d ) {
      RealT accepted = 0.0, eq18 = 0.0, eq20 = 0.0;
      for ( int j = 0; j < 4; ++j ) {
        accepted += before.nonmortar[i * 4 + j] * old1[d * 4 + j] - before.mortar[i * 4 + j] * old2[d * 4 + j];
        eq18 += ( after.nonmortar[i * 4 + j] - before.nonmortar[i * 4 + j] ) * old1[d * 4 + j] -
                ( after.mortar[i * 4 + j] - before.mortar[i * 4 + j] ) * old2[d * 4 + j];
        eq20 += after.nonmortar[i * 4 + j] * old1[d * 4 + j] - after.mortar[i * 4 + j] * old2[d * 4 + j];
      }
      EXPECT_NEAR( eq18, eq20 - accepted, 1.0e-14 );
      if ( d == 0 ) EXPECT_LT( eq18, 0.0 );
    }
}

// Open, stick, slip, reversed slip, and zero friction must solve the paper's discrete contact conditions.
TEST( CoulombLaw, admissible_roots_and_branch_derivatives )
{
  CoulombFrictionOptions options{ 0.5, 2.0, 3.0, 1.0e-14 };
  CoulombNodeInput input;
  input.normal_sum = MdArray<RealT, 3>{ 0.0, 0.0, 2.0 };
  input.area = 1.0;
  input.gap_vector = MdArray<RealT, 3>{ 0.0, 0.0, -0.1 };
  auto open = evaluateCoulombNode( input, options );
  EXPECT_EQ( open.state, FrictionState::Open );
  for ( int d = 0; d < 3; ++d ) EXPECT_DOUBLE_EQ( open.constraint[d], 0.0 );
  input.gap_vector = MdArray<RealT, 3>{ 0.0, 0.0, 0.0 };
  input.traction = MdArray<RealT, 3>{ 0.2, 0.0, 2.0 };
  auto stick = evaluateCoulombNode( input, options );
  EXPECT_EQ( stick.state, FrictionState::Stick );
  for ( int d = 0; d < 3; ++d ) EXPECT_NEAR( stick.constraint[d], 0.0, 1.0e-14 );
  for ( RealT sign : { -1.0, 1.0 } ) {
    input.traction = MdArray<RealT, 3>{ sign, 0.0, 2.0 };
    input.slip_vector = MdArray<RealT, 3>{ sign * 0.4, 0.0, 0.0 };
    auto slip = evaluateCoulombNode( input, options );
    EXPECT_EQ( slip.state, FrictionState::Slip );
    for ( int d = 0; d < 3; ++d ) EXPECT_NEAR( slip.constraint[d], 0.0, 1.0e-14 );
    EXPECT_GT( slip.pressure, 0.0 );
    EXPECT_GT( input.traction[0] * slip.slip[0], 0.0 );
  }
  options.coefficient = 0.0;
  input.traction = MdArray<RealT, 3>{ 0.0, 0.0, 2.0 };
  auto frictionless = evaluateCoulombNode( input, options );
  for ( int d = 0; d < 3; ++d ) EXPECT_NEAR( frictionless.constraint[d], 0.0, 1.0e-14 );
  input.area = 0.0;
  input.traction = MdArray<RealT, 3>{ 1.0, 2.0, 3.0 };
  auto absent = evaluateCoulombNode( input, options );
  for ( int d = 0; d < 3; ++d ) {
    EXPECT_DOUBLE_EQ( absent.constraint[d], input.traction[d] );
    for ( int k = 0; k < 3; ++k ) EXPECT_DOUBLE_EQ( absent.derivative( d, 9 + k ), d == k ? 1.0 : 0.0 );
  }
  input.area = 1.0;
  options.coefficient = 0.5;
  input.traction = MdArray<RealT, 3>{ 1.0, 0.0, 2.0 };
  input.slip_vector = MdArray<RealT, 3>{ 0.0, 0.0, 0.0 };
  auto boundary = evaluateCoulombNode( input, options );
  EXPECT_EQ( boundary.state, FrictionState::Stick );
  EXPECT_NEAR( boundary.derivative( 0, 9 ), 0.0, 1.0e-14 );
  input.gap_vector = MdArray<RealT, 3>{ 0.0, 0.0, -1.0 };
  boundary = evaluateCoulombNode( input, options );
  EXPECT_EQ( boundary.state, FrictionState::Open );
  EXPECT_DOUBLE_EQ( boundary.derivative( 2, 5 ), 0.0 );
  options.normal_scale = 0.0;
  EXPECT_THROW( evaluateCoulombNode( input, options ), std::invalid_argument );
}

// Differentiate independent normal, gap, accepted-coordinate integral, traction, and area inputs on smooth branches.
TEST( CoulombLaw, all_local_partials_match_finite_differences )
{
  CoulombFrictionOptions options{ 0.2, 2.0, 3.0, 1.0e-14 };
  CoulombNodeInput input;
  input.normal_sum = MdArray<RealT, 3>{ 0.1, 0.2, 1.7 };
  input.gap_vector = MdArray<RealT, 3>{ 0.2, -0.1, 0.01 };
  input.slip_vector = MdArray<RealT, 3>{ 0.4, -0.07, 0.1 };
  input.traction = MdArray<RealT, 3>{ 0.25, -0.2, 2.0 };
  input.area = 0.3;
  auto result = evaluateCoulombNode( input, options, 0.02 );
  constexpr RealT h = 1.0e-6;
  for ( int col = 0; col < 13; ++col ) {
    auto plus = input, minus = input;
    auto entry = []( CoulombNodeInput& a, int i ) -> RealT& {
      if ( i < 3 ) return a.normal_sum[i];
      if ( i < 6 ) return a.gap_vector[i - 3];
      if ( i < 9 ) return a.slip_vector[i - 6];
      if ( i < 12 ) return a.traction[i - 9];
      return a.area;
    };
    entry( plus, col ) += h;
    entry( minus, col ) -= h;
    auto p = evaluateCoulombNode( plus, options, 0.02, false ), m = evaluateCoulombNode( minus, options, 0.02, false );
    for ( int row = 0; row < 3; ++row )
      EXPECT_NEAR( result.derivative( row, col ), ( p.constraint[row] - m.constraint[row] ) / ( 2.0 * h ), 1.0e-8 );
  }
}

/** @brief Share mesh registration, field transfer, and directional-derivative checks for both face shapes. */
class MfemCoulombTest : public testing::TestWithParam<mfem::Element::Type> {
 protected:
  /** @brief Register a two-versus-three-element-per-direction contact patch. */
  void SetUp() override
  {
    mesh_ = std::make_unique<mfem::ParMesh>( static_cast<const mfem::ParMesh&>( shared::ParMeshBuilder(
        MPI_COMM_WORLD,
        shared::MeshBuilder::Unify( { shared::MeshBuilder::CubeMesh( 2, 2, 1, GetParam() ).updateBdrAttrib( 6, 7 ),
                                      shared::MeshBuilder::CubeMesh( 3, 3, 1, GetParam() )
                                          .scale( { 0.9, 0.9, 1.0 } )
                                          .translate( { 0.04, 0.06, 1.0 } )
                                          .updateBdrAttrib( 1, 8 )
                                          .updateAttrib( 1, 2 ) } ) ) ) );
    fec_ = std::make_unique<mfem::H1_FECollection>( 1, 3 );
    fes_ = std::make_unique<mfem::ParFiniteElementSpace>( mesh_.get(), fec_.get(), 3, mfem::Ordering::byVDIM );
    coordinates_ = std::make_unique<mfem::ParGridFunction>( fes_.get() );
    mesh_->GetNodes( *coordinates_ );
    initial_ = *coordinates_;
    upper_.SetSize( fes_->GetNDofs() );
    upper_ = 0;
    for ( int e = 0; e < mesh_->GetNE(); ++e )
      if ( mesh_->GetAttribute( e ) == 2 ) {
        mfem::Array<int> nodes;
        fes_->GetElementDofs( e, nodes );
        for ( int node : nodes ) upper_[node] = 1;
      }
    ExecutionMode execution = ExecutionMode::Sequential;
    BinningMethod binning = BINNING_GRID;
#ifdef TRIBOL_USE_OPENMP
    if ( GetParam() == mfem::Element::HEXAHEDRON ) {
      execution = ExecutionMode::OpenMP;
      binning = BINNING_BVH;
    }
#endif
    registerMfemCouplingScheme( 0, 0, 1, *mesh_, *coordinates_, { 7 }, { 8 }, SURFACE_TO_SURFACE, NO_CASE,
                                SINGLE_MORTAR, FRICTION_COULOMB, LAGRANGE_MULTIPLIER, binning, execution );
    setCoulombFrictionOptions( 0, { 0.5, 2.0, 3.0, 1.0e-14 } );
    updateMfemParallelDecomposition( 1, true );
  }
  /** @brief Release borrowed coupling data before parent mesh storage. */
  void TearDown() override { finalize(); }
  /** @brief Set Cartesian multipliers only on the nonmortar side.
   * @param x First traction component. @param y Second component. @param z Third component. */
  void setTraction( RealT x, RealT y, RealT z )
  {
    auto& traction = getMfemContactTraction( 0 );
    traction = 0.0;
    auto* fes = traction.ParFESpace();
    for ( int e = 0; e < fes->GetNE(); ++e )
      if ( fes->GetAttribute( e ) == 8 ) {
        mfem::Array<int> nodes;
        fes->GetElementDofs( e, nodes );
        for ( int node : nodes ) {
          traction[fes->DofToVDof( node, 0 )] = x;
          traction[fes->DofToVDof( node, 1 )] = y;
          traction[fes->DofToVDof( node, 2 )] = z;
        }
      }
    mfem::Vector true_values;
    traction.GetTrueDofs( true_values );
    traction.SetFromTrueDofs( true_values );
  }
  /** @brief Change only the upper body's coordinates from the reference configuration.
   * @param dx Tangential shift. @param dz Normal shift. @param warp Coefficient of the bilinear z perturbation. */
  void moveUpper( RealT dx, RealT dz, RealT warp = 0.0 )
  {
    *coordinates_ = initial_;
    for ( int i = 0; i < fes_->GetNDofs(); ++i )
      if ( upper_[i] ) {
        ( *coordinates_ )[fes_->DofToVDof( i, 0 )] += dx;
        ( *coordinates_ )[fes_->DofToVDof( i, 2 )] +=
            dz + warp * initial_[fes_->DofToVDof( i, 0 )] * initial_[fes_->DofToVDof( i, 1 )];
      }
  }
  /** @brief Rebuild Redecomp on one rank and evaluate the current trial. */
  void evaluate()
  {
    updateMfemParallelDecomposition( 1, true );
    RealT dt = 1.0;
    ASSERT_EQ( update( 0, 0.0, dt ), 0 );
  }
  /** @brief Calculate a global norm. @param v Owned true-DOF entries. @return Euclidean norm. */
  RealT norm( const mfem::Vector& v ) { return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, v, v ) ); }
  /** @brief Compare a matrix action with a centered residual difference.
   * @param plus Positive-perturbation residual. @param minus Negative-perturbation residual.
   * @param predicted Jacobian action. @param h Perturbation size. */
  void compareAction( const mfem::Vector& plus, const mfem::Vector& minus, const mfem::Vector& predicted, RealT h )
  {
    mfem::Vector error( plus );
    error -= minus;
    error /= 2.0 * h;
    error -= predicted;
    EXPECT_LT( norm( error ) / ( 1.0 + norm( predicted ) ), 2.0e-6 );
  }
  std::unique_ptr<mfem::ParMesh> mesh_;                 ///< Borrowed reference volume mesh.
  std::unique_ptr<mfem::H1_FECollection> fec_;          ///< Linear coordinate basis.
  std::unique_ptr<mfem::ParFiniteElementSpace> fes_;    ///< Parent coordinate space, with interleaved components.
  std::unique_ptr<mfem::ParGridFunction> coordinates_;  ///< Caller-owned current coordinates.
  mfem::Vector initial_;                                ///< Reference coordinates for repeatable trials.
  mfem::Array<int> upper_;                              ///< Parent scalar DOFs belonging to the upper body.
};

// A constant pressure on a nonmatching patch must give equal and opposite total forces and zero contact residual.
TEST_P( MfemCoulombTest, constant_pressure_and_rigid_motion )
{
  // A nonzero accepted gap vector makes omitting equation 20's history subtraction visible after rotation.
  constexpr RealT initial_separation = 0.01;
  moveUpper( 0.0, initial_separation );
  setResidualGap( 0, initial_separation );
  updateMfemParallelDecomposition( 1, true );
  mfem::Vector accepted_coordinates( *coordinates_ );
  setTraction( 0.0, 0.0, -2.0 );
  beginContactStep( 0 );
  evaluate();
  auto result = getMfemContactResidual( 0 );
  EXPECT_LT( norm( *result.constraint ), 1.0e-12 );
  EXPECT_LT( norm( result.slip ), 1.0e-12 );
  mfem::ParGridFunction constant( fes_.get() );
  for ( int d = 0; d < 3; ++d ) {
    constant = 0.0;
    for ( int i = 0; i < fes_->GetNDofs(); ++i ) constant[fes_->DofToVDof( i, d )] = 1.0;
    mfem::Vector true_constant;
    constant.GetTrueDofs( true_constant );
    EXPECT_NEAR( mfem::InnerProduct( MPI_COMM_WORLD, true_constant, *result.force ), 0.0, 1.0e-12 );
  }
  constant = 0.0;
  for ( int i = 0; i < fes_->GetNDofs(); ++i )
    if ( upper_[i] ) constant[fes_->DofToVDof( i, 2 )] = 1.0;
  mfem::Vector upper_selector;
  constant.GetTrueDofs( upper_selector );
  constexpr RealT pressure = 2.0, contact_area = 0.9 * 0.9;
  EXPECT_NEAR( mfem::InnerProduct( MPI_COMM_WORLD, upper_selector, *result.force ), -pressure * contact_area, 1.0e-12 );
  RealT angle = 0.3, c = std::cos( angle ), s = std::sin( angle );
  for ( int i = 0; i < fes_->GetNDofs(); ++i ) {
    RealT x = accepted_coordinates[fes_->DofToVDof( i, 0 )], z = accepted_coordinates[fes_->DofToVDof( i, 2 )];
    ( *coordinates_ )[fes_->DofToVDof( i, 0 )] = c * x + s * z + 0.2;
    ( *coordinates_ )[fes_->DofToVDof( i, 1 )] = accepted_coordinates[fes_->DofToVDof( i, 1 )] + 0.1;
    ( *coordinates_ )[fes_->DofToVDof( i, 2 )] = -s * x + c * z - 0.1;
  }
  setTraction( -2.0 * s, 0.0, -2.0 * c );
  evaluate();
  auto rotated = getMfemContactResidual( 0 );
  EXPECT_LT( norm( rotated.slip ), 1.0e-11 );
  EXPECT_LT( norm( *rotated.constraint ), 1.0e-11 );
  EXPECT_LT( norm( *result.constraint ), 1.0e-12 );
  commitContactStep( 0 );
}

// Geometry perturbations must include changing overlap weights and neighboring normals; traction derivatives include
// the physical multiplier block and the identity equations on unused nodes.
TEST_P( MfemCoulombTest, all_four_jacobian_blocks )
{
  setTraction( -0.1, 0.03, -2.0 );
  beginContactStep( 0 );
  moveUpper( 0.07, -0.005, 0.01 );
  evaluate();
  for ( bool sliding : { false, true } ) {
    if ( sliding ) {
      setTraction( -1.2, 0.03, -2.0 );
      evaluate();
    }
    auto op = getMfemBlockJacobian( 0 );
    mfem::Vector base_coordinates( *coordinates_ );
    mfem::Vector dx( fes_->GetTrueVSize() );
    for ( int i = 0; i < dx.Size(); ++i ) dx[i] = 0.1 * std::sin( static_cast<RealT>( i + 1 ) );
    mfem::ParGridFunction local_dx( fes_.get() );
    local_dx.SetFromTrueDofs( dx );
    mfem::Vector force_prediction( op->GetBlock( 0, 0 ).Height() ),
        constraint_prediction( op->GetBlock( 1, 0 ).Height() );
    op->GetBlock( 0, 0 ).Mult( dx, force_prediction );
    op->GetBlock( 1, 0 ).Mult( dx, constraint_prediction );
    constexpr RealT h = 1.0e-6;
    *coordinates_ = base_coordinates;
    coordinates_->Add( h, local_dx );
    evaluate();
    auto plus = getMfemContactResidual( 0 );
    *coordinates_ = base_coordinates;
    coordinates_->Add( -h, local_dx );
    evaluate();
    auto minus = getMfemContactResidual( 0 );
    compareAction( *plus.force, *minus.force, force_prediction, h );
    compareAction( *plus.constraint, *minus.constraint, constraint_prediction, h );
    *coordinates_ = base_coordinates;
    auto& traction = getMfemContactTraction( 0 );
    mfem::Vector base_traction;
    traction.GetTrueDofs( base_traction );
    mfem::Vector dl( base_traction.Size() );
    for ( int i = 0; i < dl.Size(); ++i ) dl[i] = 0.1 * std::cos( static_cast<RealT>( i + 1 ) );
    op->GetBlock( 0, 1 ).Mult( dl, force_prediction );
    op->GetBlock( 1, 1 ).Mult( dl, constraint_prediction );
    mfem::Vector perturbed( base_traction );
    perturbed.Add( h, dl );
    traction.SetFromTrueDofs( perturbed );
    evaluate();
    plus = getMfemContactResidual( 0 );
    perturbed = base_traction;
    perturbed.Add( -h, dl );
    traction.SetFromTrueDofs( perturbed );
    evaluate();
    minus = getMfemContactResidual( 0 );
    compareAction( *plus.force, *minus.force, force_prediction, h );
    compareAction( *plus.constraint, *minus.constraint, constraint_prediction, h );
    traction.SetFromTrueDofs( base_traction );
    evaluate();
  }
  rollbackContactStep( 0 );
}

// Rejecting a trial step must restore multiplier input, preserve returned arrays, and allow a clean retry.
TEST_P( MfemCoulombTest, rollback_retry_and_failure )
{
  setTraction( 0.0, 0.0, -2.0 );
  mfem::Vector accepted( getMfemContactTraction( 0 ) );
  beginContactStep( 0 );
  EXPECT_THROW( beginContactStep( 0 ), std::logic_error );
  EXPECT_THROW( setCoulombFrictionOptions( 0, { 0.4, 2.0, 3.0, 1.0e-14 } ), std::logic_error );
  moveUpper( 0.4, -0.01 );
  setTraction( -1.0, 0.0, -2.0 );
  evaluate();
  auto saved = getMfemContactResidual( 0 );
  // The force-only getter must agree with the coupled result and return independent storage.
  auto force = getMfemContactForce( 0 );
  force -= *saved.force;
  EXPECT_DOUBLE_EQ( norm( force ), 0.0 );
  mfem::Vector legacy;
  EXPECT_THROW( getMfemPressure( 0 ), std::logic_error );
  EXPECT_THROW( getMfemGap( 0, legacy ), std::logic_error );
  EXPECT_THROW( getMfemResponse( 0, legacy ), std::logic_error );
  RealT saved_norm = norm( saved.slip );
  EXPECT_GT( saved_norm, 0.0 );
  evaluate();
  auto repeated = getMfemContactResidual( 0 );
  repeated.slip -= saved.slip;
  EXPECT_LT( norm( repeated.slip ), 1.0e-13 );
  rollbackContactStep( 0 );
  mfem::Vector difference( getMfemContactTraction( 0 ) );
  difference -= accepted;
  EXPECT_DOUBLE_EQ( norm( difference ), 0.0 );
  EXPECT_THROW( getMfemContactResidual( 0 ), std::invalid_argument );
  EXPECT_THROW( getMfemContactForce( 0 ), std::invalid_argument );
  *coordinates_ = initial_;
  updateMfemParallelDecomposition( 1, true );
  beginContactStep( 0 );
  evaluate();
  auto retry = getMfemContactResidual( 0 );
  EXPECT_LT( norm( retry.slip ), 1.0e-12 );
  EXPECT_DOUBLE_EQ( norm( saved.slip ), saved_norm );
  getMfemContactTraction( 0 ) = 0.0;
  EXPECT_THROW( commitContactStep( 0 ), std::invalid_argument );
  rollbackContactStep( 0 );
}

// Complete separation removes overlap and must enforce zero traction on every multiplier degree of freedom.
TEST_P( MfemCoulombTest, no_overlap_is_a_valid_calculation )
{
  setTraction( 0.1, 0.2, -2.0 );
  beginContactStep( 0 );
  moveUpper( 5.0, 0.0 );
  evaluate();
  auto result = getMfemContactResidual( 0 );
  EXPECT_DOUBLE_EQ( norm( *result.force ), 0.0 );
  mfem::Vector traction;
  getMfemContactTraction( 0 ).GetTrueDofs( traction );
  traction -= *result.constraint;
  EXPECT_DOUBLE_EQ( norm( traction ), 0.0 );
  for ( int state : result.state ) EXPECT_EQ( state, static_cast<int>( FrictionState::Open ) );
  rollbackContactStep( 0 );
}

// A failed local nodal calculation must invalidate trial outputs on every rank and leave rollback usable.
TEST_P( MfemCoulombTest, nonfinite_input_invalidates_trial_results )
{
  setTraction( 0.0, 0.0, -2.0 );
  beginContactStep( 0 );
  evaluate();
  auto& traction = getMfemContactTraction( 0 );
  int rank = 0;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 && traction.Size() > 0 ) traction[0] = std::numeric_limits<RealT>::quiet_NaN();
  RealT dt = 1.0;
  EXPECT_THROW( update( 0, 0.0, dt ), std::invalid_argument );
  EXPECT_THROW( getMfemContactResidual( 0 ), std::invalid_argument );
  rollbackContactStep( 0 );
}

// Repartitioning contact elements across all available ranks must leave assembled integrals and equations unchanged.
TEST_P( MfemCoulombTest, redecomp_migration_preserves_outputs )
{
  setTraction( -0.1, 0.0, -2.0 );
  beginContactStep( 0 );
  moveUpper( 0.13, -0.01, 0.01 );
  evaluate();
  auto before = getMfemContactResidual( 0 );
  updateMfemParallelDecomposition( 0, true );
  RealT dt = 1.0;
  ASSERT_EQ( update( 0, 0.0, dt ), 0 );
  auto after = getMfemContactResidual( 0 );
  *after.force -= *before.force;
  *after.constraint -= *before.constraint;
  after.slip -= before.slip;
  EXPECT_LT( norm( *after.force ), 1.0e-12 );
  EXPECT_LT( norm( *after.constraint ), 1.0e-12 );
  EXPECT_LT( norm( after.slip ), 1.0e-12 );
  rollbackContactStep( 0 );
}

// Returned vectors and matrices must own their arrays when the scheme and its transfer data are destroyed.
TEST_P( MfemCoulombTest, returned_outputs_survive_scheme_destruction )
{
  setTraction( 0.0, 0.0, -2.0 );
  beginContactStep( 0 );
  evaluate();
  auto result = getMfemContactResidual( 0 );
  auto jacobian = getMfemBlockJacobian( 0 );
  RealT before = norm( *result.force );
  finalize();
  EXPECT_DOUBLE_EQ( norm( *result.force ), before );
  mfem::Vector direction( jacobian->GetBlock( 1, 1 ).Width() ), action( jacobian->GetBlock( 1, 1 ).Height() );
  direction = 1.0;
  jacobian->GetBlock( 1, 1 ).Mult( direction, action );
  EXPECT_TRUE( std::isfinite( norm( action ) ) );
}

// Changing the normal offset must require a new evaluation and must leave the accepted unshifted slip history intact.
TEST_P( MfemCoulombTest, residual_gap_changes_invalidate_trial_outputs )
{
  setTraction( 0.0, 0.0, -2.0 );
  beginContactStep( 0 );
  evaluate();
  auto previous = getMfemContactResidual( 0 );
  constexpr RealT offset = 0.01, overlap_area = 0.9 * 0.9;
  setResidualGap( 0, offset );
  EXPECT_THROW( getMfemContactResidual( 0 ), std::invalid_argument );
  EXPECT_THROW( commitContactStep( 0 ), std::invalid_argument );
  evaluate();
  auto updated = getMfemContactResidual( 0 );
  RealT local_sum = updated.gap.Sum(), sum = 0.0;
  MPI_Allreduce( &local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( sum, offset * overlap_area, 1.0e-12 );
  EXPECT_LT( norm( updated.slip ), 1.0e-12 );
  EXPECT_NEAR( previous.gap.Normlinf(), 0.0, 1.0e-12 );
  rollbackContactStep( 0 );
}

// Moving a scheme must preserve history without retaining a reference to its previous object address.
TEST_P( MfemCoulombTest, history_survives_scheme_moves )
{
  setTraction( 0.0, 0.0, -2.0 );
  beginContactStep( 0 );
  moveUpper( 0.1, -0.01 );
  evaluate();
  auto before = getMfemContactResidual( 0 );
  auto& registered = CouplingSchemeManager::getInstance().at( 0 );
  CouplingScheme moved( std::move( registered ) );
  auto moved_result = moved.getCoulombMortar()->residual( moved );
  *moved_result.constraint -= *before.constraint;
  EXPECT_DOUBLE_EQ( norm( *moved_result.constraint ), 0.0 );
  registered = std::move( moved );
  auto restored = getMfemContactResidual( 0 );
  restored.slip -= before.slip;
  EXPECT_DOUBLE_EQ( norm( restored.slip ), 0.0 );
  rollbackContactStep( 0 );
}

INSTANTIATE_TEST_SUITE_P( surface_shapes, MfemCoulombTest,
                          testing::Values( mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON ) );
}  // namespace
/** @brief Run local and MPI mortar tests. @param argc Argument count. @param argv Arguments. @return Test status. */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
