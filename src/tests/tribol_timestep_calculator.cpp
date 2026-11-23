// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Unit tests for individual timestep criteria, composable TimestepCalculator selection, and portable ReduceMin
 * reductions across enabled execution backends.
 */

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "axom/slic.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/Reductions.hpp"
#include "tribol/physics/timestep/TimestepCalculator.hpp"

namespace {

using tribol::ArrayT;
using tribol::ArrayViewT;
using tribol::CurrentGapTimestepCriterion;
using tribol::ExecutionMode;
using tribol::GershgorinCflTimestepCriterion;
using tribol::IndexT;
using tribol::MemorySpace;
using tribol::PairSpringMassCflTimestepCriterion;
using tribol::RealT;
using tribol::ReduceMin;
using tribol::TimestepCalculator;
using tribol::TimestepCriteria;
using tribol::TimestepGershgorinPointData;
using tribol::TimestepPairSpringMassData;
using tribol::TimestepPointData;
using tribol::TimestepSettings;
using tribol::VelocityProjectionTimestepCriterion;

// Helper creating two opposing horizontal surfaces in 2D with configurable gap, velocity, and thickness.
TimestepPointData makeOpposing2DPoint( RealT y1, RealT y2, RealT vy1, RealT vy2, RealT thickness1, RealT thickness2,
                                       bool in_contact )
{
  TimestepPointData p;
  p.point1[0] = 0.5;
  p.point1[1] = y1;
  p.point2[0] = 0.5;
  p.point2[1] = y2;
  p.vel1[1] = vy1;
  p.vel2[1] = vy2;
  // Geometry1 outward normal is +y; geometry2 outward normal is -y; contact_normal = unit(n2 - n1) = -y.
  p.contact_normal[1] = -1.0;
  p.thickness1 = thickness1;
  p.thickness2 = thickness2;
  p.in_contact = in_contact;
  return p;
}

// CurrentGapTimestepCriterion triggers only when in contact, approaching (relative closing speed > 0), and current
// penetration exceeds pen_frac * min(thickness1, thickness2).
TEST( TimestepCalculatorTest, current_gap_criterion_triggers_on_large_approaching_gap )
{
  TimestepSettings settings;
  settings.pen_frac = 0.3;

  // Penetration = -(point1 - point2) . contact_normal = 1.1 - 1.0 = 0.1 > 0.3 * 0.25 = 0.075.
  // Relative closing speed = -(vy1 - vy2) * (-1) = 2.0 - (-1.0) = 3.0.
  const auto approaching_large = makeOpposing2DPoint( 1.1, 1.0, 2.0, -1.0, 0.25, 0.25, true );
  const RealT dt = CurrentGapTimestepCriterion::evaluate( approaching_large, settings );
  EXPECT_NEAR( dt, 0.3 * 0.25 / 3.0, 1.0e-12 );

  // Rigid-body translation (equal velocities -> zero relative closing speed) produces no vote.
  const auto rigid_translation = makeOpposing2DPoint( 1.1, 1.0, 5.0, 5.0, 0.25, 0.25, true );
  EXPECT_TRUE( std::isinf( CurrentGapTimestepCriterion::evaluate( rigid_translation, settings ) ) );

  // Separating velocities produce no vote even when the current gap exceeds max_delta.
  const auto separating_large = makeOpposing2DPoint( 1.1, 1.0, -2.0, 1.0, 0.25, 0.25, true );
  EXPECT_TRUE( std::isinf( CurrentGapTimestepCriterion::evaluate( separating_large, settings ) ) );

  // Small gap below max_delta produces no vote.
  const auto small_gap = makeOpposing2DPoint( 1.02, 1.0, 2.0, -1.0, 0.25, 0.25, true );
  EXPECT_TRUE( std::isinf( CurrentGapTimestepCriterion::evaluate( small_gap, settings ) ) );

  // Not in contact produces no vote.
  const auto not_in_contact = makeOpposing2DPoint( 1.1, 1.0, 2.0, -1.0, 0.25, 0.25, false );
  EXPECT_TRUE( std::isinf( CurrentGapTimestepCriterion::evaluate( not_in_contact, settings ) ) );
}

// VelocityProjectionTimestepCriterion triggers when projected interpenetration over current_dt exceeds max_delta.
TEST( TimestepCalculatorTest, velocity_projection_criterion_uses_current_dt )
{
  TimestepSettings settings;
  settings.pen_frac = 0.3;

  // Separated by 0.1, approaching with vy1 = 5.0, vy2 = -5.0 -> relative closing speed = 10.0,
  // projected overlap over dt = 1.0 is 9.9 > 0.075.
  const auto fast_approach = makeOpposing2DPoint( 0.9, 1.0, 5.0, -5.0, 0.25, 0.25, false );
  const RealT dt = VelocityProjectionTimestepCriterion::evaluate( fast_approach, settings, 1.0 );
  EXPECT_NEAR( dt, 0.3 * 0.25 / 10.0, 1.0e-12 );

  // With a much smaller current_dt = 0.005, projected gap remains open so no vote is emitted.
  EXPECT_TRUE( std::isinf( VelocityProjectionTimestepCriterion::evaluate( fast_approach, settings, 0.005 ) ) );

  // Without a positive current_dt, velocity projection does not vote.
  EXPECT_TRUE( std::isinf( VelocityProjectionTimestepCriterion::evaluate( fast_approach, settings, 0.0 ) ) );
}

// PairSpringMassCflTimestepCriterion matches C_stab / sqrt(k_pen * A * (inv_m1 + inv_m2)), including rigid faces.
TEST( TimestepCalculatorTest, pair_spring_mass_cfl_matches_analytical_formula )
{
  TimestepSettings settings;
  settings.explicit_integrator_stability_factor = 2.0;

  // Asymmetric deformable faces: m1 = 4.0 (inv_m1 = 0.25), m2 = 12.0 (inv_m2 = 1/12), k_pen = 3.0, A = 2.0.
  TimestepPairSpringMassData pair;
  pair.inv_mass1 = 0.25;
  pair.inv_mass2 = 1.0 / 12.0;
  pair.penalty_stiffness_per_area = 3.0;
  pair.overlap_measure = 2.0;
  pair.in_contact = true;

  // inv_m1 + inv_m2 = 1/3, k_pen * A * (1/3) = 2.0, dt = 2.0 / sqrt(2.0) = sqrt(2.0).
  EXPECT_NEAR( PairSpringMassCflTimestepCriterion::evaluate( pair, settings ), std::sqrt( 2.0 ), 1.0e-12 );

  // Rigid-on-deformable (inv_mass2 = 0.0) reduces to C_stab / sqrt(k_pen * A * inv_m1).
  pair.inv_mass2 = 0.0;
  EXPECT_NEAR( PairSpringMassCflTimestepCriterion::evaluate( pair, settings ), 2.0 / std::sqrt( 3.0 * 2.0 * 0.25 ),
               1.0e-12 );

  // Rigid-on-rigid produces no CFL vote.
  pair.inv_mass1 = 0.0;
  EXPECT_TRUE( std::isinf( PairSpringMassCflTimestepCriterion::evaluate( pair, settings ) ) );

  // Non-contacting pair produces no CFL vote.
  pair.inv_mass1 = 0.25;
  pair.in_contact = false;
  EXPECT_TRUE( std::isinf( PairSpringMassCflTimestepCriterion::evaluate( pair, settings ) ) );
}

// GershgorinCflTimestepCriterion matches analytical stiffness/damping bounds with multi-point accumulation, signed
// higher-order basis values, and constrained (zero inverse mass) DOFs.
TEST( TimestepCalculatorTest, gershgorin_cfl_handles_signed_basis_damping_and_fixed_dofs )
{
  TimestepSettings settings;
  settings.explicit_integrator_stability_factor = 1.8;

  ArrayT<RealT> stiffness_sums( 6, 6 );
  ArrayT<RealT> damping_sums( 6, 6 );
  stiffness_sums.fill( 0.0 );
  damping_sums.fill( 0.0 );
  ArrayT<int> invalid_mass( { 0 } );

  // 2D normal = (0, -1), signed basis on surface 0: { 0.75, -0.25 }, on surface 1: { 0.5, 0.5 }.
  // Surface 0 node 0 has inv_my = 4.0 (sqrt = 2.0), node 1 has inv_my = 0.0 (constrained DOF).
  // Surface 1 node 0 has inv_my = 1.0 (sqrt = 1.0), node 1 has inv_my = 1.0 (sqrt = 1.0).
  TimestepGershgorinPointData pt;
  pt.integration_weight = 0.5;
  pt.penalty_stiffness = 10.0;
  pt.normal_rate_penalty = 2.0;
  pt.tangential_viscous_coeff = 0.0;
  pt.contact_normal[0] = 0.0;
  pt.contact_normal[1] = -1.0;
  pt.spatial_dim = 2;
  pt.size_basis[0] = 2;
  pt.size_basis[1] = 2;
  pt.basis_values[0][0] = 0.75;
  pt.basis_values[0][1] = -0.25;
  pt.basis_values[1][0] = 0.5;
  pt.basis_values[1][1] = 0.5;
  pt.inverse_mass[0][0][0] = 4.0;
  pt.inverse_mass[0][0][1] = 4.0;
  pt.inverse_mass[0][1][0] = 0.0;
  pt.inverse_mass[0][1][1] = 0.0;
  pt.inverse_mass[1][0][0] = 1.0;
  pt.inverse_mass[1][0][1] = 1.0;
  pt.inverse_mass[1][1][0] = 1.0;
  pt.inverse_mass[1][1][1] = 1.0;
  // Assign distinct DOF slots for y-components: 0, 1, 2, 3 (x-components unused since normal[0] == 0).
  pt.dof_indices[0][0][0] = 4;
  pt.dof_indices[0][0][1] = 0;
  pt.dof_indices[0][1][0] = 4;
  pt.dof_indices[0][1][1] = 1;
  pt.dof_indices[1][0][0] = 5;
  pt.dof_indices[1][0][1] = 2;
  pt.dof_indices[1][1][0] = 5;
  pt.dof_indices[1][1][1] = 3;

  // Accumulate twice to verify multi-point accumulation into shared DOFs.
  GershgorinCflTimestepCriterion::accumulatePoint( pt, stiffness_sums.view(), damping_sums.view(),
                                                   invalid_mass.view() );
  GershgorinCflTimestepCriterion::accumulatePoint( pt, stiffness_sums.view(), damping_sums.view(),
                                                   invalid_mass.view() );
  EXPECT_EQ( invalid_mass[0], 0 );
  // Constrained DOF (slot 1) remains zero.
  EXPECT_EQ( stiffness_sums[1], 0.0 );

  // normal_abs_sum = |0.75|*1*2 + |-0.25|*1*0 + |0.5|*1*1 + |0.5|*1*1 = 1.5 + 0 + 0.5 + 0.5 = 2.5.
  // Max DOF is slot 0: 2 * (0.5 * 10.0) * (0.75 * 2.0) * 2.5 = 10.0 * 1.5 * 2.5 = 37.5.
  // Damping max DOF is slot 0: 2 * (0.5 * 2.0) * (0.75 * 2.0) * 2.5 = 2.0 * 1.5 * 2.5 = 7.5.
  RealT max_k = 0.0;
  RealT max_c = 0.0;
  for ( IndexT i = 0; i < stiffness_sums.size(); ++i ) {
    max_k = std::max( max_k, stiffness_sums[i] );
    max_c = std::max( max_c, damping_sums[i] );
  }
  EXPECT_NEAR( max_k, 37.5, 1.0e-12 );
  EXPECT_NEAR( max_c, 7.5, 1.0e-12 );

  const RealT dt_cfl = GershgorinCflTimestepCriterion::finalizeFromBounds( max_k, max_c, settings );
  const RealT expected_omega = std::sqrt( 37.5 + 0.25 * 7.5 * 7.5 ) + 0.5 * 7.5;
  EXPECT_NEAR( dt_cfl, 1.8 / expected_omega, 1.0e-12 );
}

// TimestepCalculator respects runtime TimestepCriteria boolean toggles independently.
TEST( TimestepCalculatorTest, runtime_criteria_toggles_control_active_checks )
{
  const auto point = makeOpposing2DPoint( 1.1, 1.0, 2.0, -1.0, 0.25, 0.25, true );
  TimestepPairSpringMassData pair;
  pair.inv_mass1 = 0.25;
  pair.inv_mass2 = 0.25;
  pair.penalty_stiffness_per_area = 100.0;
  pair.overlap_measure = 1.0;
  pair.in_contact = true;

  TimestepSettings settings;
  settings.pen_frac = 0.3;
  settings.explicit_integrator_stability_factor = 2.0;

  // All disabled -> no votes.
  settings.active_criteria = { false, false, false, false };
  {
    TimestepCalculator calc( settings );
    EXPECT_FALSE( calc.anyEnabled() );
    EXPECT_TRUE( std::isinf( calc.evaluatePoint( point, 1.0 ) ) );
    EXPECT_TRUE( std::isinf( calc.evaluatePairSpringMassCfl( pair ) ) );
    EXPECT_TRUE( std::isinf( calc.finalizeGershgorin( 100.0, 0.0 ) ) );
  }

  // Only pair_spring_mass_cfl enabled -> point check inactive, pair CFL active.
  settings.active_criteria = { false, false, false, true };
  {
    TimestepCalculator calc( settings );
    EXPECT_TRUE( std::isinf( calc.evaluatePoint( point, 1.0 ) ) );
    EXPECT_NEAR( calc.evaluatePairSpringMassCfl( pair ), 2.0 / std::sqrt( 50.0 ), 1.0e-12 );
  }

  // Only current_gap enabled -> point check active, pair CFL inactive.
  settings.active_criteria = { true, false, false, false };
  {
    TimestepCalculator calc( settings );
    EXPECT_NEAR( calc.evaluatePoint( point, 1.0 ), 0.3 * 0.25 / 3.0, 1.0e-12 );
    EXPECT_TRUE( std::isinf( calc.evaluatePairSpringMassCfl( pair ) ) );
  }
}

}  // namespace

namespace test_detail {

template <ExecutionMode Exec>
RealT reducePairSpringMassCfl( const ArrayViewT<const RealT>& inv_masses_view, const TimestepCalculator& calculator )
{
  ReduceMin<Exec, RealT> min_reducer( axom::numeric_limits<RealT>::infinity() );
  tribol::forAllExec<Exec>( inv_masses_view.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    TimestepPairSpringMassData pair;
    pair.inv_mass1 = inv_masses_view[i];
    pair.inv_mass2 = inv_masses_view[i];
    pair.penalty_stiffness_per_area = 2.0;
    pair.overlap_measure = 1.0;
    pair.in_contact = true;
    const RealT dt = calculator.evaluatePairSpringMassCfl( pair );
    if ( std::isfinite( dt ) ) {
      min_reducer.min( dt );
    }
  } );
  return min_reducer.get();
}

}  // namespace test_detail

namespace {

class TimestepCalculatorExecutionTest : public tribol::detail::ExecutionTest {};

// Portable execution test verifying ReduceMin and TimestepCalculator inside forAllExec<Exec> on every enabled backend.
TEST_P( TimestepCalculatorExecutionTest, reduces_pair_cfl_across_execution_backends )
{
  const auto inv_masses = copyToExecutionSpace<RealT>( { 0.25, 1.0, 4.0, 0.5 } );
  const auto inv_masses_view = inv_masses.view();

  TimestepSettings settings;
  settings.active_criteria = { false, false, false, true };
  settings.explicit_integrator_stability_factor = 2.0;
  const TimestepCalculator calculator( settings );

  const RealT min_dt = tribol::dispatchExecutionMode( GetParam(), [&]<ExecutionMode Exec>() {
    return test_detail::reducePairSpringMassCfl<Exec>( inv_masses_view, calculator );
  } );

  // Largest inv_mass = 4.0 -> stiffness_bound = 2.0 * 1.0 * 8.0 = 16.0 -> dt = 2.0 / 4.0 = 0.5.
  EXPECT_NEAR( min_dt, 0.5, 1.0e-12 );
}

INSTANTIATE_TEST_SUITE_P( execution_modes, TimestepCalculatorExecutionTest, tribol::detail::enabledExecutionModes() );

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
