#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "axom/slic.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

namespace {

struct ProjectionResult {
  int return_code;
  bool converged;
  bool complementarity_converged;
  tribol::IndexT constraints;
  int iterations;
  double initial_residual;
  double final_residual;
  double final_primal_residual;
  double primal_tolerance;
  double max_endpoint_violation;
  double reported_energy_change;
  double spring_force;
  double damping_force;
  double guard_force;
  tribol::IndexT guard_constraints;
  double stored_energy;
  double max_penetration_fraction;
  double direct_energy_change;
  double impulse_force_error;
  double correction_norm;
  tribol::IndexT operator_velocity_dofs;
  tribol::IndexT operator_rank;
  double operator_minimum_eigenvalue;
  double operator_maximum_eigenvalue;
  double operator_condition_estimate;
  double operator_jacobi_contraction;
  int al_outer_iterations;
  int al_subproblem_iterations;
  int al_incomplete_subproblems;
  tribol::IndexT al_warm_start_rows;
  double al_history_force_norm;
  double al_multiplier_update_norm;
  double applied_force;
  double applied_tensile_force;
  double response_norm;
  tribol::IndexT penalty_al_damping_qpts;
  double penalty_al_damping_force;
  double penalty_al_damping_dissipation_rate;
  double penalty_al_maximum_damping_pressure;
  double penalty_stability_timestep;
  tribol::IndexT penalty_stability_active_rows;
  tribol::IndexT penalty_stability_predicted_rows;
  double history_total_force;
  double history_minimum_pressure;
  double history_maximum_pressure;
  double expected_area_scaled_history_force_norm;
  double expected_overlap_limited_history_force_norm;
  double unscaled_history_force_norm;
  tribol::IndexT contact_quadrature_points;
  double quadrature_average_gap_violation;
  double quadrature_maximum_gap_violation;
  tribol::IndexT parent_trace_gap_rows;
  tribol::IndexT parent_trace_geometric_gap_rows;
  double parent_trace_current_average_gap_violation;
  double parent_trace_current_maximum_gap_violation;
  double parent_trace_endpoint_average_gap_violation;
  double parent_trace_endpoint_maximum_gap_violation;
  double parent_trace_geometric_average_gap_violation;
  double parent_trace_geometric_maximum_gap_violation;
};

ProjectionResult RunProjectionCase( double upper_offset, double upper_velocity, double dt, int max_iterations = 250,
                                    double relative_tolerance = 1.e-10, double primal_relative_tolerance = 1.e-6,
                                    int lor_factor = 2, int quadrature_order = 3, double absolute_tolerance = 1.e-12,
                                    tribol::ContactMethod method = tribol::COMMON_PLANE,
                                    double position_velocity_scale = 1.,
                                    double projection_base_upper_velocity = 0.,
                                    tribol::ImpulseProjectionContactResponse contact_response =
                                        tribol::PROJECTION_RESPONSE_EXACT,
                                    double damping_ratio = 1.2, double max_penetration_fraction = 0.02,
                                    double al_augmentation_scale = 100., int al_max_iterations = 8,
                                    int al_fixed_iterations = 0,
                                    tribol::AugmentedLagrangianFailurePolicy al_failure_policy =
                                        tribol::AL_ACCEPT_FEASIBLE,
                                    int updates = 1, bool rollback_after_first_update = false,
                                    tribol::EnforcementMethod enforcement_method = tribol::IMPULSE_PROJECTION,
                                    bool penalty_augmented_lagrangian = false,
                                    int penalty_al_max_iterations = 2, int penalty_al_fixed_iterations = 2,
                                    double penalty_al_relative_tolerance = 1.e-6,
                                    double penalty_al_absolute_tolerance = 1.e-12,
                                    double penalty_al_relaxation = 1.,
                                    double penalty_al_spatial_smoothing = 0.,
                                    bool nonuniform_upper_velocity = false,
                                    double upper_horizontal_offset = 0.,
                                    double second_update_horizontal_offset =
                                        std::numeric_limits<double>::quiet_NaN(),
                                    double penalty_al_unloading_relaxation = -1.,
                                    double penalty_al_direction_deadband = 1.e-3,
                                    double second_update_vertical_offset =
                                        std::numeric_limits<double>::quiet_NaN(),
                                    tribol::PenaltyAugmentedLagrangianFormulation penalty_al_formulation =
                                        tribol::PENALTY_AL_SURFACE_COMPLIANCE,
                                    double penalty_al_loading_time_constant = 0.,
                                    double penalty_al_unloading_time_constant = -1.,
                                    double penalty_al_activation_gap_fraction = 0.,
                                    bool evaluate_two_rk_stages = false,
                                    double penalty_al_fixed_active_set_time = -1.,
                                    double initial_time = 0.,
                                    double penalty_al_normal_damping_ratio = 0.,
                                    double penalty_al_target_gap_fraction = 0.,
                                    double penalty_scale = 1.,
                                    double penalty_al_inner_relaxation = 1.,
                                    double duplicate_quadrature_cell_history_fraction = 0.,
                                    double penalty_al_continuation_factor = 1.,
                                    double penalty_al_continuation_max_scale = 1.,
                                    double projection_depenetration_fraction = 0.,
                                    double projection_maximum_force = -1.,
                                    double projection_gap_tolerance = 0.,
                                    double projection_maximum_gap = -1.,
                                    bool projection_positive_lor_basis = false,
                                    double projection_mass_inverse_scale = 0. )
{
  const std::set<int> surface1{ 3 };
  const std::set<int> surface2{ 5 };
  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
    shared::MeshBuilder::SquareMesh(1, 1)
      .updateBdrAttrib(1, 1)
      .updateBdrAttrib(2, 2)
      .updateBdrAttrib(3, 3)
      .updateBdrAttrib(4, 4),
    shared::MeshBuilder::SquareMesh(1, 1)
      .translate({upper_horizontal_offset, upper_offset})
      .updateAttrib(1, 2)
      .updateBdrAttrib(1, 5)
      .updateBdrAttrib(2, 6)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(4, 8)
  }));
  // clang-format on

  mesh.SetCurvature( 2 );
  auto* nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
  EXPECT_NE( nodes, nullptr );
  mfem::ParGridFunction coords( nodes->ParFESpace() );
  coords = *nodes;
  const double second_update_horizontal_shift = std::isfinite( second_update_horizontal_offset )
                                                    ? second_update_horizontal_offset - upper_horizontal_offset
                                                    : 0.;
  const double second_update_vertical_shift = std::isfinite( second_update_vertical_offset )
                                                  ? second_update_vertical_offset - upper_offset
                                                  : 0.;
  mfem::VectorFunctionCoefficient unchanged_coordinate_coefficient(
      2, []( const mfem::Vector& x, mfem::Vector& value ) {
        value.SetSize( x.Size() );
        value = x;
      } );
  mfem::VectorFunctionCoefficient shifted_upper_coordinate_coefficient(
      2, [second_update_horizontal_shift,
          second_update_vertical_shift]( const mfem::Vector& x, mfem::Vector& value ) {
        value.SetSize( x.Size() );
        value = x;
        value[0] += second_update_horizontal_shift;
        value[1] += second_update_vertical_shift;
      } );
  mfem::Array<int> coordinate_attributes( { 1, 2 } );
  mfem::Array<mfem::VectorCoefficient*> coordinate_coefficients(
      { &unchanged_coordinate_coefficient, &shifted_upper_coordinate_coefficient } );
  mfem::PWVectorCoefficient second_update_coordinate_coefficient(
      2, coordinate_attributes, coordinate_coefficients );
  mfem::ParGridFunction velocity( nodes->ParFESpace() );
  velocity = 0.;
  mfem::Vector upper_velocity_vector( { 0., upper_velocity } );
  mfem::VectorConstantCoefficient upper_velocity_coefficient( upper_velocity_vector );
  mfem::VectorFunctionCoefficient nonuniform_upper_velocity_coefficient(
      2, [upper_velocity]( const mfem::Vector& x, mfem::Vector& value ) {
        value[0] = 0.;
        value[1] = upper_velocity * ( 0.25 + 1.5 * x[0] );
      } );
  mfem::Array<int> moving_attributes( { 2 } );
  mfem::Array<mfem::VectorCoefficient*> velocity_coefficients(
      { nonuniform_upper_velocity
            ? static_cast<mfem::VectorCoefficient*>( &nonuniform_upper_velocity_coefficient )
            : static_cast<mfem::VectorCoefficient*>( &upper_velocity_coefficient ) } );
  mfem::PWVectorCoefficient velocity_coefficient( 2, moving_attributes, velocity_coefficients );
  velocity.ProjectCoefficient( velocity_coefficient );
  mfem::ParGridFunction projection_base_velocity( nodes->ParFESpace() );
  projection_base_velocity = 0.;
  mfem::Vector projection_base_velocity_vector( { 0., projection_base_upper_velocity } );
  mfem::VectorConstantCoefficient projection_base_velocity_coefficient( projection_base_velocity_vector );
  mfem::Array<mfem::VectorCoefficient*> projection_base_velocity_coefficients(
      { &projection_base_velocity_coefficient } );
  mfem::PWVectorCoefficient projection_base_coefficient( 2, moving_attributes,
                                                         projection_base_velocity_coefficients );
  projection_base_velocity.ProjectCoefficient( projection_base_coefficient );
  mfem::ParGridFunction inverse_mass( nodes->ParFESpace() );
  mfem::Vector lower_inverse_mass_vector( { 1., 1. } );
  mfem::Vector upper_inverse_mass_vector( { 0.25, 0.25 } );
  mfem::VectorConstantCoefficient lower_inverse_mass_coefficient( lower_inverse_mass_vector );
  mfem::VectorConstantCoefficient upper_inverse_mass_coefficient( upper_inverse_mass_vector );
  mfem::Array<int> material_attributes( { 1, 2 } );
  mfem::Array<mfem::VectorCoefficient*> inverse_mass_coefficients(
      { &lower_inverse_mass_coefficient, &upper_inverse_mass_coefficient } );
  mfem::PWVectorCoefficient inverse_mass_coefficient( 2, material_attributes, inverse_mass_coefficients );
  inverse_mass.ProjectCoefficient( inverse_mass_coefficient );

  constexpr int coupling_scheme_id = 720;
  constexpr int mesh1_id = 1440;
  constexpr int mesh2_id = 1441;
  tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, surface1, surface2,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, method, tribol::FRICTIONLESS,
                                      enforcement_method, tribol::BINNING_GRID,
                                      tribol::ExecutionMode::Sequential );
  tribol::setMfemLORFactor( coupling_scheme_id, lor_factor );
  tribol::setMfemSurfaceBasis( coupling_scheme_id, tribol::MfemSurfaceBasis::PARENT );
  tribol::registerMfemVelocity( coupling_scheme_id, velocity );
  if ( position_velocity_scale < 1. ) {
    tribol::registerMfemProjectionBaseVelocity( coupling_scheme_id, projection_base_velocity );
    tribol::setImpulseProjectionKinematics( coupling_scheme_id, position_velocity_scale );
  }
  tribol::registerMfemInverseMass( coupling_scheme_id, inverse_mass );
  if ( projection_mass_inverse_scale > 0. ) {
    mfem::Vector inverse_mass_values( inverse_mass );
    tribol::registerMfemVelocityMassInverse(
        coupling_scheme_id,
        [inverse_mass_values, projection_mass_inverse_scale]( const mfem::Vector& impulse,
                                                              mfem::Vector& velocity_increment ) {
          if ( impulse.Size() != inverse_mass_values.Size() ) {
            return false;
          }
          velocity_increment.SetSize( impulse.Size() );
          const double* impulse_values = impulse.HostRead();
          const double* inverse_values = inverse_mass_values.HostRead();
          double* increment_values = velocity_increment.HostWrite();
          for ( int i = 0; i < impulse.Size(); ++i ) {
            increment_values[i] =
                projection_mass_inverse_scale * inverse_values[i] * impulse_values[i];
          }
          return true;
        } );
  }
  tribol::setCommonPlaneIntegrationOptions( coupling_scheme_id, tribol::MULTI_POINT, quadrature_order );
  if ( enforcement_method == tribol::IMPULSE_PROJECTION ) {
    tribol::setImpulseProjectionOptions( coupling_scheme_id, max_iterations, relative_tolerance,
                                         absolute_tolerance, 1., primal_relative_tolerance );
    tribol::setImpulseProjectionDepenetration( coupling_scheme_id,
                                               projection_depenetration_fraction );
    tribol::setImpulseProjectionGapTolerance( coupling_scheme_id,
                                              projection_gap_tolerance );
    tribol::setImpulseProjectionPositiveLORBasis( coupling_scheme_id,
                                                  projection_positive_lor_basis );
    tribol::setImpulseProjectionMaximumGap( coupling_scheme_id,
                                            projection_maximum_gap );
    tribol::setImpulseProjectionMaximumForce( coupling_scheme_id,
                                              projection_maximum_force );
  } else {
    tribol::setPenaltyOptions( coupling_scheme_id, tribol::KINEMATIC,
                               tribol::KINEMATIC_CONSTANT, tribol::NO_RATE_PENALTY );
  }
  if ( enforcement_method == tribol::PENALTY ||
       ( enforcement_method == tribol::IMPULSE_PROJECTION &&
         contact_response != tribol::PROJECTION_RESPONSE_EXACT ) ) {
    tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, 10., 10. );
    tribol::setMfemKinematicPenaltyScale( coupling_scheme_id, penalty_scale,
                                          penalty_scale );
  }
  if ( enforcement_method == tribol::IMPULSE_PROJECTION ) {
    tribol::setParentTraceMortarOptions( coupling_scheme_id, 30., contact_response, damping_ratio,
                                         max_penetration_fraction );
  }
  if ( method == tribol::PARENT_TRACE_MORTAR ) {
    if ( contact_response == tribol::PROJECTION_RESPONSE_AUGMENTED_LAGRANGIAN ) {
      tribol::setAugmentedLagrangianOptions( coupling_scheme_id, al_augmentation_scale, al_max_iterations,
                                             al_fixed_iterations, al_failure_policy );
    }
    if ( penalty_augmented_lagrangian ) {
      tribol::setPenaltyAugmentedLagrangianFormulation( coupling_scheme_id,
                                                        penalty_al_formulation );
      tribol::setPenaltyAugmentedLagrangianContinuation(
          coupling_scheme_id, penalty_al_continuation_factor,
          penalty_al_continuation_max_scale );
      tribol::setPenaltyAugmentedLagrangianOptions(
          coupling_scheme_id, penalty_al_max_iterations, penalty_al_fixed_iterations,
          penalty_al_relative_tolerance, penalty_al_absolute_tolerance, penalty_al_relaxation,
          penalty_al_spatial_smoothing, penalty_al_unloading_relaxation,
          penalty_al_direction_deadband, penalty_al_loading_time_constant,
          penalty_al_unloading_time_constant, penalty_al_activation_gap_fraction,
          penalty_al_normal_damping_ratio, penalty_al_target_gap_fraction );
      tribol::setPenaltyAugmentedLagrangianInnerRelaxation(
          coupling_scheme_id, penalty_al_inner_relaxation );
      tribol::setPenaltyAugmentedLagrangianFixedActiveSetTime(
          coupling_scheme_id, penalty_al_fixed_active_set_time );
    }
  }
  tribol::updateMfemElemThickness( coupling_scheme_id );
  tribol::enableTimestepVote( coupling_scheme_id, false );
  tribol::updateMfemParallelDecomposition( 1, true );

  int return_code = 0;
  const bool uses_augmented_lagrangian =
      penalty_augmented_lagrangian ||
      contact_response == tribol::PROJECTION_RESPONSE_AUGMENTED_LAGRANGIAN;
  const auto coupling_scheme =
      tribol::CouplingSchemeManager::getInstance().findData( coupling_scheme_id );
  EXPECT_NE( coupling_scheme, nullptr );
  std::vector<tribol::ParentTraceMultiplierState> previous_history;
  for ( int update = 0; update < updates; ++update ) {
    if ( update == updates - 1 && update > 0 && coupling_scheme != nullptr ) {
      previous_history = coupling_scheme->getParentTraceMultiplierWarmStart();
    }
    if ( update == 1 && ( std::isfinite( second_update_horizontal_offset ) ||
                          std::isfinite( second_update_vertical_offset ) ) ) {
      coords.ProjectCoefficient( second_update_coordinate_coefficient );
      tribol::updateMfemParallelDecomposition( 1, true );
    }
    if ( uses_augmented_lagrangian ) {
      tribol::beginAugmentedLagrangianStep( coupling_scheme_id );
    }
    if ( update > 0 ) {
      velocity.ProjectCoefficient( velocity_coefficient );
    }
    if ( evaluate_two_rk_stages ) {
      double first_stage_timestep_vote = 0.5 * dt;
      return_code = tribol::update( update, initial_time + update * dt, 0.5 * dt,
                                    first_stage_timestep_vote );
    }
    double timestep_vote = dt;
    return_code = tribol::update( update, initial_time + update * dt, dt, timestep_vote );
    if ( uses_augmented_lagrangian ) {
      tribol::commitAugmentedLagrangianStep( coupling_scheme_id );
      if ( update == 0 && rollback_after_first_update ) {
        tribol::rollbackAugmentedLagrangianStep( coupling_scheme_id );
      } else if ( update == 0 && duplicate_quadrature_cell_history_fraction > 0. &&
                  coupling_scheme != nullptr ) {
        std::vector<tribol::QuadratureCellMultiplierState> duplicated_history =
            coupling_scheme->getQuadratureCellMultiplierWarmStart();
        EXPECT_FALSE( duplicated_history.empty() );
        if ( !duplicated_history.empty() ) {
          auto duplicate = duplicated_history.front();
          const double interval_length =
              duplicate.parent_coordinate_max - duplicate.parent_coordinate_min;
          duplicate.parent_coordinate_min =
              duplicate.parent_coordinate_max -
              duplicate_quadrature_cell_history_fraction * interval_length;
          duplicate.measure *= duplicate_quadrature_cell_history_fraction;
          duplicated_history.push_back( duplicate );
          coupling_scheme->setQuadratureCellMultiplierStage( std::move( duplicated_history ) );
          tribol::commitAugmentedLagrangianStep( coupling_scheme_id );
        }
      }
    }
  }
  mfem::Vector correction( coords.Size() );
  correction = 0.;
  if ( return_code == 0 && enforcement_method == tribol::IMPULSE_PROJECTION ) {
    tribol::getMfemVelocityCorrection( coupling_scheme_id, correction );
  }
  mfem::Vector force( coords.Size() );
  force = 0.;
  tribol::getMfemResponse( coupling_scheme_id, force );
  mfem::Vector impulse_error( force );
  impulse_error *= dt;
  mfem::Vector mass_times_correction( correction.Size() );
  const double* inverse_mass_data = inverse_mass.HostRead();
  const double* correction_data = correction.HostRead();
  double* mass_times_correction_data = mass_times_correction.HostWrite();
  double direct_energy_change = 0.;
  const double* velocity_data = velocity.HostRead();
  for ( int i = 0; i < correction.Size(); ++i ) {
    EXPECT_GT( inverse_mass_data[i], 0. );
    mass_times_correction_data[i] = correction_data[i] / inverse_mass_data[i];
    direct_energy_change +=
        velocity_data[i] * mass_times_correction_data[i] + 0.5 * correction_data[i] * mass_times_correction_data[i];
  }
  impulse_error -= mass_times_correction;

  double history_total_force = 0.;
  double history_minimum_pressure = std::numeric_limits<double>::infinity();
  double history_maximum_pressure = 0.;
  double expected_area_scaled_history_force_norm_squared = 0.;
  double expected_overlap_limited_history_force_norm_squared = 0.;
  double unscaled_history_force_norm_squared = 0.;
  if ( coupling_scheme != nullptr ) {
    for ( const auto& state : coupling_scheme->getParentTraceMultiplierWarmStart() ) {
      if ( penalty_al_fixed_active_set_time < 0. ||
           initial_time + ( updates - 1 ) * dt < penalty_al_fixed_active_set_time ) {
        EXPECT_GE( state.force, 0. );
      }
      EXPECT_GT( state.tributary_area, 0. );
      history_total_force += state.force;
      const double pressure = state.force / state.tributary_area;
      history_minimum_pressure = std::min( history_minimum_pressure, pressure );
      history_maximum_pressure = std::max( history_maximum_pressure, pressure );
      for ( const auto& previous_state : previous_history ) {
        if ( previous_state.parent_dof != state.parent_dof ||
             previous_state.patch != state.patch || previous_state.tributary_area <= 0. ) {
          continue;
        }
        const double area_scaled_force =
            previous_state.force * state.tributary_area / previous_state.tributary_area;
        const double overlap_limited_force =
            previous_state.force *
            std::min( state.tributary_area, previous_state.tributary_area ) /
            previous_state.tributary_area;
        expected_area_scaled_history_force_norm_squared += area_scaled_force * area_scaled_force;
        expected_overlap_limited_history_force_norm_squared +=
            overlap_limited_force * overlap_limited_force;
        unscaled_history_force_norm_squared += previous_state.force * previous_state.force;
        break;
      }
    }
    for ( const auto& state : coupling_scheme->getQuadratureCellMultiplierWarmStart() ) {
      EXPECT_GE( state.pressure, 0. );
      EXPECT_GT( state.measure, 0. );
      history_total_force += state.pressure * state.measure;
      history_minimum_pressure = std::min( history_minimum_pressure, state.pressure );
      history_maximum_pressure = std::max( history_maximum_pressure, state.pressure );
    }
  }
  if ( !std::isfinite( history_minimum_pressure ) ) {
    history_minimum_pressure = 0.;
  }

  const ProjectionResult result{ return_code,
                                 tribol::getProjectionConverged( coupling_scheme_id ),
                                 tribol::getProjectionComplementarityConverged( coupling_scheme_id ),
                                 tribol::getNumProjectionConstraints( coupling_scheme_id ),
                                 tribol::getProjectionIterations( coupling_scheme_id ),
                                 tribol::getProjectionInitialResidual( coupling_scheme_id ),
                                 tribol::getProjectionFinalResidual( coupling_scheme_id ),
                                 tribol::getProjectionFinalPrimalResidual( coupling_scheme_id ),
                                 tribol::getProjectionPrimalTolerance( coupling_scheme_id ),
                                 tribol::getProjectionMaxEndpointViolation( coupling_scheme_id ),
                                 tribol::getProjectionEnergyChange( coupling_scheme_id ),
                                 tribol::getCompliantProjectionSpringForce( coupling_scheme_id ),
                                 tribol::getCompliantProjectionDampingForce( coupling_scheme_id ),
                                 tribol::getCompliantProjectionGuardForce( coupling_scheme_id ),
                                 tribol::getCompliantProjectionGuardConstraints( coupling_scheme_id ),
                                 tribol::getCompliantProjectionStoredEnergy( coupling_scheme_id ),
                                 tribol::getCompliantProjectionMaxPenetrationFraction( coupling_scheme_id ),
                                 direct_energy_change,
                                 impulse_error.Norml2(),
                                 correction.Norml2(),
                                 tribol::getProjectionOperatorVelocityDofs( coupling_scheme_id ),
                                 tribol::getProjectionOperatorRank( coupling_scheme_id ),
                                 tribol::getProjectionOperatorMinimumEigenvalue( coupling_scheme_id ),
                                 tribol::getProjectionOperatorMaximumEigenvalue( coupling_scheme_id ),
                                 tribol::getProjectionOperatorConditionEstimate( coupling_scheme_id ),
                                 tribol::getProjectionOperatorJacobiContraction( coupling_scheme_id ),
                                 tribol::getAugmentedLagrangianOuterIterations( coupling_scheme_id ),
                                 tribol::getAugmentedLagrangianSubproblemIterations( coupling_scheme_id ),
                                 tribol::getAugmentedLagrangianIncompleteSubproblems( coupling_scheme_id ),
                                 tribol::getAugmentedLagrangianWarmStartRows( coupling_scheme_id ),
                                 tribol::getAugmentedLagrangianHistoryForceNorm( coupling_scheme_id ),
                                 tribol::getAugmentedLagrangianMultiplierUpdateNorm( coupling_scheme_id ),
                                 tribol::getIntegratedAppliedForce( coupling_scheme_id ),
                                 tribol::getIntegratedAppliedTensileForce( coupling_scheme_id ),
                                 force.Norml2(),
                                 tribol::getNumPenaltyAugmentedLagrangianDampingQuadraturePoints(
                                     coupling_scheme_id ),
                                 tribol::getIntegratedPenaltyAugmentedLagrangianDampingForce(
                                     coupling_scheme_id ),
                                 tribol::getPenaltyAugmentedLagrangianDampingDissipationRate(
                                     coupling_scheme_id ),
                                 tribol::getMaximumPenaltyAugmentedLagrangianDampingPressure(
                                     coupling_scheme_id ),
                                 tribol::getPenaltyStabilityTimestep( coupling_scheme_id ),
                                 tribol::getNumPenaltyStabilityActiveRows( coupling_scheme_id ),
                                 tribol::getNumPenaltyStabilityPredictedRows( coupling_scheme_id ),
                                 history_total_force,
                                 history_minimum_pressure,
                                 history_maximum_pressure,
                                 std::sqrt( expected_area_scaled_history_force_norm_squared ),
                                 std::sqrt( expected_overlap_limited_history_force_norm_squared ),
                                 std::sqrt( unscaled_history_force_norm_squared ),
                                 tribol::getNumContactQuadraturePoints( coupling_scheme_id ),
                                 tribol::getAverageGapViolation( coupling_scheme_id ),
                                 tribol::getMaxGapViolation( coupling_scheme_id ),
                                 tribol::getNumParentTraceGapRows( coupling_scheme_id ),
                                 tribol::getNumParentTraceGeometricGapRows( coupling_scheme_id ),
                                 tribol::getAverageParentTraceCurrentGapViolation( coupling_scheme_id ),
                                 tribol::getMaxParentTraceCurrentGapViolation( coupling_scheme_id ),
                                 tribol::getAverageParentTraceEndpointGapViolation( coupling_scheme_id ),
                                 tribol::getMaxParentTraceEndpointGapViolation( coupling_scheme_id ),
                                 tribol::getAverageParentTraceGeometricGapViolation( coupling_scheme_id ),
                                 tribol::getMaxParentTraceGeometricGapViolation( coupling_scheme_id ) };
  tribol::finalize();
  return result;
}

ProjectionResult RunPenaltyAugmentedLagrangianCase(
    double upper_offset, double upper_velocity, double dt, int al_iterations, int updates = 1,
    bool rollback_after_first_update = false, bool augmented_lagrangian = true, double spatial_smoothing = 0.,
    bool nonuniform_upper_velocity = false, double upper_horizontal_offset = 0.,
    double second_update_horizontal_offset = std::numeric_limits<double>::quiet_NaN(), double loading_relaxation = 1.,
    double unloading_relaxation = -1., double direction_deadband = 1.e-3,
    double second_update_vertical_offset = std::numeric_limits<double>::quiet_NaN(),
    tribol::PenaltyAugmentedLagrangianFormulation formulation = tribol::PENALTY_AL_SURFACE_COMPLIANCE,
    double loading_time_constant = 0., double unloading_time_constant = -1., double activation_gap_fraction = 0.,
    bool evaluate_two_rk_stages = false, double fixed_active_set_time = -1., double initial_time = 0.,
    double normal_damping_ratio = 0., double target_gap_fraction = 0., double penalty_scale = 1.,
    double inner_relaxation = 1., double duplicate_quadrature_cell_history_fraction = 0.,
    double continuation_factor = 1., double continuation_max_scale = 1.,
    double relative_tolerance = 1.e-8, double absolute_tolerance = 1.e-12 )
{
  const bool compliance_corrected =
      formulation == tribol::PENALTY_AL_QUADRATURE_CELL_MEMORY_COMPLIANCE ||
      formulation == tribol::PENALTY_AL_PARENT_TRACE_COMPLIANCE ||
      formulation == tribol::PENALTY_AL_QUADRATURE_HYBRID_COMPLIANCE;
  const int fixed_iterations = compliance_corrected ? 0 : al_iterations;
  return RunProjectionCase(
      upper_offset, upper_velocity, dt, 250, 1.e-10, 1.e-6, 2, 3, 1.e-12, tribol::PARENT_TRACE_MORTAR, 1., 0.,
      tribol::PROJECTION_RESPONSE_EXACT, 1.2, 0.02, 100., 8, 0, tribol::AL_ACCEPT_FEASIBLE, updates,
      rollback_after_first_update, tribol::PENALTY, augmented_lagrangian, al_iterations, fixed_iterations,
      relative_tolerance,
      absolute_tolerance, loading_relaxation, spatial_smoothing, nonuniform_upper_velocity, upper_horizontal_offset,
      second_update_horizontal_offset, unloading_relaxation, direction_deadband, second_update_vertical_offset,
      formulation, loading_time_constant, unloading_time_constant, activation_gap_fraction, evaluate_two_rk_stages,
      fixed_active_set_time, initial_time, normal_damping_ratio, target_gap_fraction, penalty_scale,
      inner_relaxation, duplicate_quadrature_cell_history_fraction, continuation_factor,
      continuation_max_scale );
}

ProjectionResult RunComplianceCorrectedPenaltyAugmentedLagrangianCase( double upper_offset, double upper_velocity,
                                                                       double dt, int max_iterations,
                                                                       double target_gap_fraction = 0.,
                                                                       double normal_damping_ratio = 0.,
                                                                       double penalty_scale = 1. )
{
  return RunPenaltyAugmentedLagrangianCase( upper_offset, upper_velocity, dt, max_iterations, 1, false, true, 0., false,
                                            0., std::numeric_limits<double>::quiet_NaN(), 1., 1., 1.e-3,
                                            std::numeric_limits<double>::quiet_NaN(),
                                            tribol::PENALTY_AL_QUADRATURE_HYBRID_COMPLIANCE, 0., 0., 0., false, -1., 0.,
                                            normal_damping_ratio, target_gap_fraction, penalty_scale );
}

ProjectionResult RunFixedIterationQuadratureHybridCase( int fixed_iterations, double inner_relaxation,
                                                         double history_relaxation = 1. )
{
  return RunPenaltyAugmentedLagrangianCase(
      0.99, 0., 1.e-2, fixed_iterations, 1, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), history_relaxation, history_relaxation, 1.e-3,
      std::numeric_limits<double>::quiet_NaN(), tribol::PENALTY_AL_QUADRATURE_HYBRID,
      0., -1., 0., false, -1., 0., 0., 0., 1., inner_relaxation );
}

ProjectionResult RunQuadratureLocalPenaltyAugmentedLagrangianCase(
    int fixed_iterations, double inner_relaxation, double penalty_scale = 1., int updates = 1,
    double upper_horizontal_offset = 0.,
    double second_update_horizontal_offset = std::numeric_limits<double>::quiet_NaN(),
    bool evaluate_two_rk_stages = false, double upper_velocity = 0.,
    double normal_damping_ratio = 0. )
{
  return RunPenaltyAugmentedLagrangianCase(
      0.99, upper_velocity, 1.e-2, fixed_iterations, updates, false, true, 0., false,
      upper_horizontal_offset, second_update_horizontal_offset, 1., -1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(), tribol::PENALTY_AL_QUADRATURE_LOCAL,
      0., -1., 0., evaluate_two_rk_stages, -1., 0., normal_damping_ratio, 0., penalty_scale,
      inner_relaxation );
}

ProjectionResult RunQuadratureCellMemoryPenaltyAugmentedLagrangianCase(
    double history_relaxation, int updates = 1, double upper_horizontal_offset = 0.,
    double second_update_horizontal_offset = std::numeric_limits<double>::quiet_NaN(),
    bool evaluate_two_rk_stages = false, bool rollback_after_first_update = false,
    double duplicate_history_fraction = 0. )
{
  return RunPenaltyAugmentedLagrangianCase(
      0.99, 0., 1.e-2, 1, updates, rollback_after_first_update, true, 0., false,
      upper_horizontal_offset, second_update_horizontal_offset, history_relaxation,
      history_relaxation, 1.e-3, std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_CELL_MEMORY, 0., -1., 0., evaluate_two_rk_stages,
      -1., 0., 0., 0., 1., 1., duplicate_history_fraction );
}

ProjectionResult RunComplianceCorrectedQuadratureCellMemoryCase(
    int max_iterations, int updates = 1, double penalty_scale = 1.,
    double relaxation = 1., double target_gap_fraction = 0.,
    double continuation_factor = 1., double continuation_max_scale = 1.,
    double absolute_tolerance = 1.e-12, double relative_tolerance = 1.e-8 )
{
  return RunPenaltyAugmentedLagrangianCase(
      0.99, -0.1, 0.5, max_iterations, updates, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), relaxation, relaxation, 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_CELL_MEMORY_COMPLIANCE, 0., -1., 0., false,
      -1., 0., 0., target_gap_fraction, penalty_scale, 1., 0.,
      continuation_factor, continuation_max_scale, relative_tolerance,
      absolute_tolerance );
}

ProjectionResult RunComplianceCorrectedParentTraceCase(
    int max_iterations, int updates = 1, double penalty_scale = 1.,
    double relaxation = 1., double target_gap_fraction = 0. )
{
  return RunPenaltyAugmentedLagrangianCase(
      0.99, -0.1, 0.5, max_iterations, updates, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), relaxation, relaxation, 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_PARENT_TRACE_COMPLIANCE, 0., -1., 0., false,
      -1., 0., 0., target_gap_fraction, penalty_scale );
}

struct ParentTracePenaltyResult {
  int return_code;
  tribol::IndexT constraints;
  double integrated_force;
  double maximum_gap_violation;
  double response_norm;
  double net_y_force;
  double upper_y_force;
  double stability_timestep;
  tribol::IndexT stability_active_rows;
  tribol::IndexT stability_predicted_rows;
  double minimum_impact_time;
};

ParentTracePenaltyResult RunParentTracePenaltyCase( double upper_offset, double penalty_scale,
                                                    bool reverse_surfaces = false,
                                                    double upper_velocity = 0.,
                                                    double stage_dt = 1.e-3,
                                                    bool register_velocity = true,
                                                    tribol::ContactMethod method = tribol::PARENT_TRACE_MORTAR,
                                                    int quadrature_order = 6 )
{
  const std::set<int> lower_surface{ 3 };
  const std::set<int> upper_surface{ 5 };
  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
    shared::MeshBuilder::SquareMesh(1, 1)
      .updateBdrAttrib(1, 1)
      .updateBdrAttrib(2, 2)
      .updateBdrAttrib(3, 3)
      .updateBdrAttrib(4, 4),
    shared::MeshBuilder::SquareMesh(1, 1)
      .translate({0.0, upper_offset})
      .updateAttrib(1, 2)
      .updateBdrAttrib(1, 5)
      .updateBdrAttrib(2, 6)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(4, 8)
  }));
  // clang-format on

  mesh.SetCurvature( 2 );
  auto* nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
  EXPECT_NE( nodes, nullptr );
  mfem::ParGridFunction coords( nodes->ParFESpace() );
  coords = *nodes;
  mfem::ParGridFunction velocity( nodes->ParFESpace() );
  velocity = 0.;
  mfem::Vector upper_velocity_vector( { 0., upper_velocity } );
  mfem::VectorConstantCoefficient upper_velocity_coefficient( upper_velocity_vector );
  mfem::Array<int> moving_attributes( { 2 } );
  mfem::Array<mfem::VectorCoefficient*> velocity_coefficients( { &upper_velocity_coefficient } );
  mfem::PWVectorCoefficient velocity_coefficient( 2, moving_attributes, velocity_coefficients );
  velocity.ProjectCoefficient( velocity_coefficient );
  mfem::ParGridFunction inverse_mass( nodes->ParFESpace() );
  inverse_mass = 1.;

  constexpr int coupling_scheme_id = 721;
  constexpr int mesh1_id = 1442;
  constexpr int mesh2_id = 1443;
  const auto& surface1 = reverse_surfaces ? upper_surface : lower_surface;
  const auto& surface2 = reverse_surfaces ? lower_surface : upper_surface;
  tribol::registerMfemCouplingScheme(
      coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, surface1, surface2,
      tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, method,
      tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
      tribol::ExecutionMode::Sequential );
  tribol::setMfemSurfaceBasis( coupling_scheme_id, tribol::MfemSurfaceBasis::PARENT );
  if ( register_velocity ) {
    tribol::registerMfemVelocity( coupling_scheme_id, velocity );
  }
  tribol::registerMfemInverseMass( coupling_scheme_id, inverse_mass );
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, 10., 10. );
  tribol::setMfemKinematicPenaltyScale( coupling_scheme_id, penalty_scale, penalty_scale );
  tribol::setCommonPlaneIntegrationOptions( coupling_scheme_id, tribol::MULTI_POINT,
                                            quadrature_order );
  if ( method == tribol::PARENT_TRACE_MORTAR ) {
    tribol::setParentTraceMortarOptions( coupling_scheme_id, 30.,
                                         tribol::PROJECTION_RESPONSE_COMPLIANT, 1.2, 0.02 );
  }
  tribol::updateMfemElemThickness( coupling_scheme_id );
  tribol::enableTimestepVote( coupling_scheme_id, false );
  tribol::updateMfemParallelDecomposition( 1, true );

  double timestep_vote = stage_dt;
  const int return_code = tribol::update( 0, 0., stage_dt, timestep_vote );
  mfem::Vector force( coords.Size() );
  force = 0.;
  tribol::getMfemResponse( coupling_scheme_id, force );

  mfem::Vector y_direction_vector( { 0., 1. } );
  mfem::VectorConstantCoefficient y_direction_coefficient( y_direction_vector );
  mfem::ParGridFunction y_direction( nodes->ParFESpace() );
  y_direction.ProjectCoefficient( y_direction_coefficient );
  mfem::Array<int> upper_attributes( { 2 } );
  mfem::Array<mfem::VectorCoefficient*> upper_coefficients( { &y_direction_coefficient } );
  mfem::PWVectorCoefficient upper_direction_coefficient( 2, upper_attributes, upper_coefficients );
  mfem::ParGridFunction upper_direction( nodes->ParFESpace() );
  upper_direction = 0.;
  upper_direction.ProjectCoefficient( upper_direction_coefficient );

  double net_y_force = 0.;
  double upper_y_force = 0.;
  const double* force_data = force.HostRead();
  const double* y_direction_data = y_direction.HostRead();
  const double* upper_direction_data = upper_direction.HostRead();
  for ( int i = 0; i < force.Size(); ++i ) {
    net_y_force += force_data[i] * y_direction_data[i];
    upper_y_force += force_data[i] * upper_direction_data[i];
  }

  const ParentTracePenaltyResult result{
    return_code,
    tribol::getNumContactQuadraturePoints( coupling_scheme_id ),
    tribol::getIntegratedAppliedForce( coupling_scheme_id ),
    tribol::getMaxGapViolation( coupling_scheme_id ),
    force.Norml2(),
    net_y_force,
    upper_y_force,
    tribol::getPenaltyStabilityTimestep( coupling_scheme_id ),
    tribol::getNumPenaltyStabilityActiveRows( coupling_scheme_id ),
    tribol::getNumPenaltyStabilityPredictedRows( coupling_scheme_id ),
    tribol::getPenaltyStabilityMinimumImpactTime( coupling_scheme_id )
  };
  tribol::finalize();
  return result;
}

TEST( MfemParentTracePenalty, UsesIndependentTraceRows )
{
  const ParentTracePenaltyResult baseline = RunParentTracePenaltyCase( 0.99, 1. );
  const ParentTracePenaltyResult doubled = RunParentTracePenaltyCase( 0.99, 2. );
  const ParentTracePenaltyResult reversed = RunParentTracePenaltyCase( 0.99, 1., true );
  const ParentTracePenaltyResult separated = RunParentTracePenaltyCase( 1.01, 1. );

  EXPECT_EQ( baseline.return_code, 0 );
  EXPECT_EQ( baseline.constraints, 3 );
  EXPECT_GT( baseline.integrated_force, 0. );
  EXPECT_GT( baseline.maximum_gap_violation, 0. );
  EXPECT_GT( baseline.response_norm, 0. );
  EXPECT_TRUE( std::isfinite( baseline.stability_timestep ) );
  EXPECT_EQ( baseline.stability_active_rows, 3 );
  EXPECT_EQ( baseline.stability_predicted_rows, 0 );
  EXPECT_EQ( baseline.minimum_impact_time, 0. );
  EXPECT_NEAR( baseline.net_y_force, 0., 1.e-12 * baseline.integrated_force );
  EXPECT_NEAR( std::abs( baseline.upper_y_force ), baseline.integrated_force,
               1.e-11 * baseline.integrated_force );

  EXPECT_EQ( doubled.return_code, 0 );
  EXPECT_NEAR( doubled.integrated_force, 2. * baseline.integrated_force,
               1.e-11 * baseline.integrated_force );
  EXPECT_NEAR( doubled.stability_timestep, baseline.stability_timestep / std::sqrt( 2. ),
               1.e-11 * baseline.stability_timestep );

  EXPECT_EQ( reversed.return_code, 0 );
  EXPECT_EQ( reversed.constraints, baseline.constraints );
  EXPECT_NEAR( reversed.integrated_force, baseline.integrated_force,
               1.e-11 * baseline.integrated_force );
  EXPECT_NEAR( reversed.net_y_force, 0., 1.e-12 * reversed.integrated_force );

  EXPECT_EQ( separated.return_code, 0 );
  EXPECT_EQ( separated.integrated_force, 0. );
  EXPECT_EQ( separated.response_norm, 0. );
  EXPECT_TRUE( std::isinf( separated.stability_timestep ) );
  EXPECT_EQ( separated.stability_active_rows, 0 );
  EXPECT_EQ( separated.stability_predicted_rows, 0 );
  EXPECT_TRUE( std::isinf( separated.minimum_impact_time ) );
}

TEST( MfemParentTracePenalty, StabilityVotePredictsActivationWithoutApplyingForce )
{
  const ParentTracePenaltyResult outside_stage =
      RunParentTracePenaltyCase( 1.01, 1., false, -1., 1.e-3 );
  const ParentTracePenaltyResult predicted =
      RunParentTracePenaltyCase( 1.0005, 1., false, -1., 1.e-3 );
  const ParentTracePenaltyResult doubled =
      RunParentTracePenaltyCase( 1.0005, 2., false, -1., 1.e-3 );
  const ParentTracePenaltyResult departing =
      RunParentTracePenaltyCase( 1.0005, 1., false, 1., 1.e-3 );

  EXPECT_EQ( outside_stage.return_code, 0 );
  EXPECT_EQ( outside_stage.integrated_force, 0. );
  EXPECT_EQ( outside_stage.response_norm, 0. );
  EXPECT_TRUE( std::isinf( outside_stage.stability_timestep ) );
  EXPECT_EQ( outside_stage.stability_active_rows, 0 );
  EXPECT_EQ( outside_stage.stability_predicted_rows, 0 );
  EXPECT_NEAR( outside_stage.minimum_impact_time, 1.e-2, 1.e-12 );

  EXPECT_EQ( predicted.return_code, 0 );
  EXPECT_EQ( predicted.integrated_force, 0. );
  EXPECT_EQ( predicted.response_norm, 0. );
  EXPECT_TRUE( std::isfinite( predicted.stability_timestep ) );
  EXPECT_EQ( predicted.stability_active_rows, 0 );
  EXPECT_EQ( predicted.stability_predicted_rows, 3 );
  EXPECT_NEAR( predicted.minimum_impact_time, 5.e-4, 1.e-12 );

  EXPECT_EQ( doubled.return_code, 0 );
  EXPECT_EQ( doubled.integrated_force, 0. );
  EXPECT_EQ( doubled.response_norm, 0. );
  EXPECT_EQ( doubled.stability_active_rows, 0 );
  EXPECT_EQ( doubled.stability_predicted_rows, 3 );
  EXPECT_NEAR( doubled.stability_timestep, predicted.stability_timestep / std::sqrt( 2. ),
               1.e-11 * predicted.stability_timestep );

  EXPECT_EQ( departing.return_code, 0 );
  EXPECT_EQ( departing.integrated_force, 0. );
  EXPECT_EQ( departing.response_norm, 0. );
  EXPECT_TRUE( std::isinf( departing.stability_timestep ) );
  EXPECT_EQ( departing.stability_active_rows, 0 );
  EXPECT_EQ( departing.stability_predicted_rows, 0 );
  EXPECT_TRUE( std::isinf( departing.minimum_impact_time ) );
}

TEST( MfemParentTracePenalty, ActiveStabilityVoteDoesNotRequireVelocity )
{
  const ParentTracePenaltyResult result =
      RunParentTracePenaltyCase( 0.99, 1., false, 0., 1.e-3, false );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_GT( result.integrated_force, 0. );
  EXPECT_GT( result.response_norm, 0. );
  EXPECT_TRUE( std::isfinite( result.stability_timestep ) );
  EXPECT_EQ( result.stability_active_rows, 3 );
  EXPECT_EQ( result.stability_predicted_rows, 0 );
  EXPECT_EQ( result.minimum_impact_time, 0. );
}

TEST( MfemImpulseProjection, CoupledJacobiConvergesAndPreservesImpulseMomentum )
{
  const ProjectionResult result = RunProjectionCase( 0.99, -1., 1.e-3 );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_GT( result.iterations, 0 );
  EXPECT_GT( result.initial_residual, 0. );
  EXPECT_LT( result.final_residual, 1.e-9 * result.initial_residual );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_NEAR( result.impulse_force_error, 0., 1.e-11 );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
  EXPECT_NEAR( result.reported_energy_change, result.direct_energy_change, 1.e-10 );
}

TEST( MfemImpulseProjection, RegisteredMassInverseControlsOperatorAndVelocityUpdate )
{
  const ProjectionResult baseline = RunProjectionCase( 0.99, -1., 1.e-3 );
  const ProjectionResult doubled_inverse = RunProjectionCase(
      0.99, -1., 1.e-3, 250, 1.e-10, 1.e-6, 2, 3, 1.e-12,
      tribol::COMMON_PLANE, 1., 0., tribol::PROJECTION_RESPONSE_EXACT,
      1.2, 0.02, 100., 8, 0, tribol::AL_ACCEPT_FEASIBLE, 1, false,
      tribol::IMPULSE_PROJECTION, false, 2, 2, 1.e-6, 1.e-12, 1., 0., false,
      0., std::numeric_limits<double>::quiet_NaN(), -1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(), tribol::PENALTY_AL_SURFACE_COMPLIANCE,
      0., -1., 0., false, -1., 0., 0., 0., 1., 1., 0., 1., 1., 0., -1., 0., -1., false, 2. );

  EXPECT_EQ( baseline.return_code, 0 );
  EXPECT_EQ( doubled_inverse.return_code, 0 );
  EXPECT_TRUE( doubled_inverse.converged );
  EXPECT_NEAR( doubled_inverse.correction_norm, baseline.correction_norm,
               1.e-10 * baseline.correction_norm );
  EXPECT_NEAR( doubled_inverse.response_norm, 0.5 * baseline.response_norm,
               1.e-10 * baseline.response_norm );
  EXPECT_NEAR( doubled_inverse.max_endpoint_violation, baseline.max_endpoint_violation,
               1.e-10 );
  EXPECT_NEAR( doubled_inverse.reported_energy_change,
               0.5 * baseline.reported_energy_change, 1.e-10 );
}

TEST( MfemImpulseProjection, SeparatedSurfacesStopAtEndpointGap )
{
  const ProjectionResult result = RunProjectionCase( 1.005, -10., 1.e-3 );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_GT( result.initial_residual, 0. );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_LT( result.max_endpoint_violation, 1.e-10 );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, NegativeGapToleranceAppliesSpeculativeClearance )
{
  const ProjectionResult baseline = RunProjectionCase( 1.005, -10., 1.e-3 );
  const ProjectionResult clearance = RunProjectionCase(
      1.005, -10., 1.e-3, 250, 1.e-10, 1.e-6, 2, 3, 1.e-12,
      tribol::PARENT_TRACE_MORTAR, 1., 0., tribol::PROJECTION_RESPONSE_EXACT,
      1.2, 0.02, 100., 8, 0, tribol::AL_ACCEPT_FEASIBLE, 1, false,
      tribol::IMPULSE_PROJECTION, false, 2, 2, 1.e-6, 1.e-12, 1., 0., false,
      0., std::numeric_limits<double>::quiet_NaN(), -1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(), tribol::PENALTY_AL_SURFACE_COMPLIANCE,
      0., -1., 0., false, -1., 0., 0., 0., 1., 1., 0., 1., 1., 0., -1., -0.002 );

  EXPECT_EQ( clearance.return_code, 0 );
  EXPECT_TRUE( clearance.converged );
  EXPECT_GT( clearance.correction_norm, baseline.correction_norm );
  EXPECT_LE( clearance.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, PositiveLORTraceBasisAppliesEnergySafeProjection )
{
  const ProjectionResult result = RunProjectionCase(
      1.005, -10., 1.e-3, 250, 1.e-10, 1.e-6, 2, 3, 1.e-12,
      tribol::PARENT_TRACE_MORTAR, 1., 0., tribol::PROJECTION_RESPONSE_EXACT,
      1.2, 0.02, 100., 8, 0, tribol::AL_ACCEPT_FEASIBLE, 1, false,
      tribol::IMPULSE_PROJECTION, false, 2, 2, 1.e-6, 1.e-12, 1., 0., false,
      0., std::numeric_limits<double>::quiet_NaN(), -1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(), tribol::PENALTY_AL_SURFACE_COMPLIANCE,
      0., -1., 0., false, -1., 0., 0., 0., 1., 1., 0., 1., 1., 0., -1., 0., -1., true );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
  EXPECT_GT( result.contact_quadrature_points, result.constraints );
}

TEST( MfemImpulseProjection, DepenetrationReducesOverlapWithoutPositiveWork )
{
  const auto run_case = []( double depenetration_fraction,
                            double maximum_force = -1.,
                            tribol::ContactMethod method = tribol::COMMON_PLANE ) {
    return RunProjectionCase(
        0.99, -1., 1.e-3, 250, 1.e-10, 1.e-6, 2, 3, 1.e-12,
        method, 1., 0., tribol::PROJECTION_RESPONSE_EXACT,
        1.2, 0.02, 100., 8, 0, tribol::AL_ACCEPT_FEASIBLE, 1, false,
        tribol::IMPULSE_PROJECTION, false, 2, 2, 1.e-6, 1.e-12, 1., 0., false,
        0., std::numeric_limits<double>::quiet_NaN(), -1., 1.e-3,
        std::numeric_limits<double>::quiet_NaN(), tribol::PENALTY_AL_SURFACE_COMPLIANCE,
        0., -1., 0., false, -1., 0., 0., 0., 1., 1., 0., 1., 1.,
        depenetration_fraction, maximum_force );
  };
  const ProjectionResult baseline = run_case( 0. );
  const ProjectionResult corrected = run_case( 0.2 );
  const ProjectionResult force_limited = run_case( 0.2, 1.e-3 );
  const ProjectionResult parent_corrected =
      run_case( 0.2, -1., tribol::PARENT_TRACE_MORTAR );
  const ProjectionResult parent_force_limited =
      run_case( 0.2, 1.e-3, tribol::PARENT_TRACE_MORTAR );

  EXPECT_EQ( baseline.return_code, 0 );
  EXPECT_EQ( corrected.return_code, 0 );
  EXPECT_GT( baseline.max_endpoint_violation, 0. );
  EXPECT_LT( corrected.max_endpoint_violation, baseline.max_endpoint_violation );
  EXPECT_GT( corrected.correction_norm, baseline.correction_norm );
  EXPECT_LE( corrected.reported_energy_change, 1.e-12 );
  EXPECT_NEAR( corrected.reported_energy_change, corrected.direct_energy_change, 1.e-10 );
  EXPECT_EQ( force_limited.return_code, 0 );
  EXPECT_GT( force_limited.max_endpoint_violation,
             corrected.max_endpoint_violation );
  EXPECT_LT( force_limited.correction_norm, corrected.correction_norm );
  EXPECT_LE( force_limited.reported_energy_change, 1.e-12 );
  EXPECT_EQ( parent_corrected.return_code, 0 );
  EXPECT_EQ( parent_force_limited.return_code, 0 );
  EXPECT_GT( parent_force_limited.max_endpoint_violation,
             parent_corrected.max_endpoint_violation );
  EXPECT_LT( parent_force_limited.correction_norm,
             parent_corrected.correction_norm );
  EXPECT_LE( parent_force_limited.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, GapToleranceRejectsUnresolvedEndpointPenetration )
{
  const ProjectionResult residual_gap = RunProjectionCase(
      0.996, -10., 1.e-3, 250, 1.e-10, 1.e-6, 2, 3, 1.e-12,
      tribol::PARENT_TRACE_MORTAR, 1., 0., tribol::PROJECTION_RESPONSE_EXACT,
      1.2, 0.02, 100., 8, 0, tribol::AL_ACCEPT_FEASIBLE, 1, false,
      tribol::IMPULSE_PROJECTION, false, 2, 2, 1.e-6, 1.e-12, 1., 0., false,
      0., std::numeric_limits<double>::quiet_NaN(), -1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(), tribol::PENALTY_AL_SURFACE_COMPLIANCE,
      0., -1., 0., false, -1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 1.e-3, 0.005, 0.005 );

  EXPECT_NE( residual_gap.return_code, 0 );
  EXPECT_GT( residual_gap.max_endpoint_violation, 0.005 );
  EXPECT_LE( residual_gap.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, MaximumGapRejectsNonWorseningOverLimitEndpoint )
{
  const ProjectionResult residual_gap = RunProjectionCase(
      0.996, 0., 1.e-3, 250, 1.e-10, 1.e-6, 2, 3, 1.e-12,
      tribol::PARENT_TRACE_MORTAR, 1., 0., tribol::PROJECTION_RESPONSE_EXACT,
      1.2, 0.02, 100., 8, 0, tribol::AL_ACCEPT_FEASIBLE, 1, false,
      tribol::IMPULSE_PROJECTION, false, 2, 2, 1.e-6, 1.e-12, 1., 0., false,
      0., std::numeric_limits<double>::quiet_NaN(), -1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(), tribol::PENALTY_AL_SURFACE_COMPLIANCE,
      0., -1., 0., false, -1., 0., 0., 0., 1., 1., 0., 1., 1., 0., -1., 0.005, 0.0023 );

  EXPECT_NE( residual_gap.return_code, 0 );
  EXPECT_GT( residual_gap.max_endpoint_violation, 0.0023 );
  EXPECT_LE( residual_gap.max_endpoint_violation, 0.004 + 1.e-12 );
  EXPECT_LE( residual_gap.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, RedundantConstraintsAcceptPrimalSafeIterate )
{
  const ProjectionResult result = RunProjectionCase( 1.005, -10., 1.e-3, 250, 0., 1.e-6, 8, 10, 1.e-30 );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_FALSE( result.complementarity_converged );
  EXPECT_GT( result.constraints, 20 );
  EXPECT_GT( result.iterations, 0 );
  EXPECT_LT( result.iterations, 250 );
  EXPECT_LE( result.final_primal_residual, result.primal_tolerance );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_NEAR( result.impulse_force_error, 0., 1.e-11 );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
  EXPECT_NEAR( result.reported_energy_change, result.direct_energy_change, 1.e-10 );
}

TEST( MfemImpulseProjection, RedundantConstraintsConvergeWithIndependentActiveBasis )
{
  constexpr int max_iterations = 250;
  const ProjectionResult result =
      RunProjectionCase( 1.005, -10., 1.e-3, max_iterations, 1.e-10, 1.e-8, 8, 10, 1.e-12 );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_TRUE( result.complementarity_converged );
  EXPECT_GT( result.constraints, result.operator_rank );
  EXPECT_LT( result.iterations, max_iterations );
  EXPECT_LE( result.final_primal_residual, result.primal_tolerance );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_NEAR( result.impulse_force_error, 0., 1.e-11 );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
  EXPECT_NEAR( result.reported_energy_change, result.direct_energy_change, 1.e-10 );
}

TEST( MfemImpulseProjection, OneSweepAcceptsRedundantPrimalSafeIterate )
{
  const ProjectionResult result = RunProjectionCase( 1.005, -10., 1.e-3, 1, 0., 1.e-16, 8, 10, 1.e-30 );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_FALSE( result.complementarity_converged );
  EXPECT_GT( result.constraints, 20 );
  EXPECT_EQ( result.iterations, 1 );
  EXPECT_LE( result.final_primal_residual, result.primal_tolerance );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_NEAR( result.impulse_force_error, 0., 1.e-11 );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, ParentTraceMortarUsesIndependentTraceRows )
{
  const ProjectionResult coarse =
      RunProjectionCase( 1.005, -10., 1.e-3, 250, 1.e-10, 1.e-8, 2, 3, 1.e-12, tribol::PARENT_TRACE_MORTAR );
  const ProjectionResult refined =
      RunProjectionCase( 1.005, -10., 1.e-3, 250, 1.e-10, 1.e-8, 8, 10, 1.e-12, tribol::PARENT_TRACE_MORTAR );
  EXPECT_EQ( coarse.return_code, 0 );
  EXPECT_EQ( refined.return_code, 0 );
  EXPECT_TRUE( coarse.converged );
  EXPECT_TRUE( refined.converged );
  EXPECT_TRUE( coarse.complementarity_converged );
  EXPECT_TRUE( refined.complementarity_converged );
  EXPECT_LT( refined.iterations, 250 );
  EXPECT_EQ( coarse.constraints, 3 );
  EXPECT_EQ( refined.constraints, coarse.constraints );
  EXPECT_EQ( coarse.operator_rank, coarse.constraints );
  EXPECT_EQ( refined.operator_rank, refined.constraints );
  EXPECT_GT( refined.correction_norm, 0. );
  EXPECT_NEAR( refined.impulse_force_error, 0., 1.e-11 );
  EXPECT_LE( refined.reported_energy_change, 1.e-12 );
  EXPECT_NEAR( refined.reported_energy_change, refined.direct_energy_change, 1.e-10 );
}

TEST( MfemImpulseProjection, ParentTraceMortarRk2AverageEnforcesEndpointGap )
{
  constexpr int max_iterations = 250;
  const ProjectionResult result = RunProjectionCase( 1.0001, -3., 1.e-3, max_iterations, 1.e-10, 1.e-8, 8, 10, 1.e-12,
                                                     tribol::PARENT_TRACE_MORTAR, 0.5, -1. );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_TRUE( result.complementarity_converged );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_LT( result.iterations, max_iterations );
  EXPECT_LT( result.max_endpoint_violation, 1.e-10 );
  EXPECT_EQ( result.operator_rank, result.constraints );
  EXPECT_NEAR( result.impulse_force_error, 0., 1.e-11 );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
  EXPECT_NEAR( result.reported_energy_change, result.direct_energy_change, 1.e-10 );
}

TEST( MfemImpulseProjection, ParentTraceMortarActiveSetPolishesWithinSmallIterationBudget )
{
  constexpr int max_iterations = 4;
  const ProjectionResult result =
      RunProjectionCase( 0.99, -1., 1.e-3, max_iterations, 1.e-12, 1.e-10, 8, 10, 1.e-14, tribol::PARENT_TRACE_MORTAR );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_TRUE( result.complementarity_converged );
  EXPECT_LE( result.iterations, max_iterations );
  EXPECT_EQ( result.operator_rank, result.constraints );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_NEAR( result.impulse_force_error, 0., 1.e-11 );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, ParentTraceMortarCompliantResponseIsEnergyStable )
{
  const ProjectionResult result =
      RunProjectionCase( 0.99, 0., 1.e-3, 250, 1.e-10, 1.e-8, 8, 10, 1.e-12,
                         tribol::PARENT_TRACE_MORTAR, 1., 0.,
                         tribol::PROJECTION_RESPONSE_COMPLIANT, 1.2, 0.02 );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_TRUE( result.complementarity_converged );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_GT( result.spring_force, 0. );
  EXPECT_GT( result.stored_energy, 0. );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, ParentTraceMortarCompliantGuardLimitsNewPenetration )
{
  const ProjectionResult result =
      RunProjectionCase( 1.005, -10., 1.e-3, 250, 1.e-10, 1.e-8, 8, 10, 1.e-12,
                         tribol::PARENT_TRACE_MORTAR, 1., 0.,
                         tribol::PROJECTION_RESPONSE_COMPLIANT, 1.2, 1.e-3 );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_TRUE( result.complementarity_converged );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_GT( result.guard_force, 0. );
  EXPECT_GT( result.guard_constraints, 0 );
  EXPECT_LE( result.max_penetration_fraction, 1.001e-3 );
  EXPECT_LE( result.max_endpoint_violation, 1.001e-3 );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, ParentTraceMortarAugmentedLagrangianConvergesToCompliantLaw )
{
  const ProjectionResult compliant =
      RunProjectionCase( 0.99, 0., 1.e-3, 250, 1.e-10, 1.e-8, 8, 10, 1.e-12,
                         tribol::PARENT_TRACE_MORTAR, 1., 0.,
                         tribol::PROJECTION_RESPONSE_COMPLIANT, 1.2, 0.02 );
  const ProjectionResult result =
      RunProjectionCase( 0.99, 0., 1.e-3, 250, 1.e-10, 1.e-8, 8, 10, 1.e-12,
                         tribol::PARENT_TRACE_MORTAR, 1., 0.,
                         tribol::PROJECTION_RESPONSE_AUGMENTED_LAGRANGIAN, 1.2, 0.02,
                         1.e8, 4, 0, tribol::AL_REPEAT_ON_NONCONVERGENCE );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_TRUE( result.complementarity_converged );
  EXPECT_GT( result.al_outer_iterations, 0 );
  EXPECT_LE( result.al_outer_iterations, 4 );
  EXPECT_GT( result.al_subproblem_iterations, 0 );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_GT( result.spring_force, 0. );
  EXPECT_NEAR( result.spring_force, compliant.spring_force,
               1.e-7 * std::max( 1., compliant.spring_force ) );
  EXPECT_NEAR( result.correction_norm, compliant.correction_norm,
               1.e-7 * std::max( 1., compliant.correction_norm ) );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, ParentTraceMortarAugmentedLagrangianWarmStartsFromForceHistory )
{
  const ProjectionResult cold_result =
      RunProjectionCase( 0.99, 0., 1.e-3, 250, 1.e-10, 1.e-8, 8, 10, 1.e-12,
                         tribol::PARENT_TRACE_MORTAR, 1., 0.,
                         tribol::PROJECTION_RESPONSE_AUGMENTED_LAGRANGIAN, 1.2, 0.02,
                         1.e8, 4, 0, tribol::AL_ACCEPT_FEASIBLE );
  const ProjectionResult result =
      RunProjectionCase( 0.99, 0., 1.e-3, 250, 1.e-10, 1.e-8, 8, 10, 1.e-12,
                         tribol::PARENT_TRACE_MORTAR, 1., 0.,
                         tribol::PROJECTION_RESPONSE_AUGMENTED_LAGRANGIAN, 1.2, 0.02,
                         1.e8, 4, 0, tribol::AL_ACCEPT_FEASIBLE, 2 );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_GT( result.al_warm_start_rows, 0 );
  EXPECT_GT( result.al_history_force_norm, 0. );
  EXPECT_NEAR( result.primal_tolerance, cold_result.primal_tolerance,
               1.e-2 * cold_result.primal_tolerance );
}

TEST( MfemImpulseProjection, ParentTraceMortarAugmentedLagrangianContinuesUntilPrimalConvergence )
{
  constexpr int al_max_iterations = 8;
  const ProjectionResult result =
      RunProjectionCase( 0.99, -1., 1.e-3, 250, 1.e-8, 1.e-12, 8, 10, 1.e-30,
                         tribol::PARENT_TRACE_MORTAR, 1., 0.,
                         tribol::PROJECTION_RESPONSE_AUGMENTED_LAGRANGIAN, 1.2, 0.02,
                         100., al_max_iterations, 0, tribol::AL_REPEAT_ON_NONCONVERGENCE );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_TRUE( result.complementarity_converged );
  EXPECT_GT( result.al_outer_iterations, 4 );
  EXPECT_LE( result.al_outer_iterations, al_max_iterations );
  EXPECT_LE( result.al_incomplete_subproblems, 1 );
  EXPECT_LE( result.final_primal_residual, result.primal_tolerance );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, ParentTraceMortarAugmentedLagrangianAcceptsCappedFiniteSubproblems )
{
  constexpr int al_max_iterations = 4;
  const ProjectionResult result =
      RunProjectionCase( 0.99, -1., 1.e-3, 1, 0., 1., 8, 10, 1.e-30,
                         tribol::PARENT_TRACE_MORTAR, 1., 0.,
                         tribol::PROJECTION_RESPONSE_AUGMENTED_LAGRANGIAN, 1.2, 0.02,
                         1., al_max_iterations, 0, tribol::AL_ACCEPT_FEASIBLE );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_FALSE( result.complementarity_converged );
  EXPECT_EQ( result.al_outer_iterations, al_max_iterations );
  EXPECT_EQ( result.al_subproblem_iterations, al_max_iterations );
  EXPECT_GT( result.al_incomplete_subproblems, 0 );
  EXPECT_LE( result.final_primal_residual, result.primal_tolerance );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
}

TEST( MfemImpulseProjection, ParentTraceMortarAugmentedLagrangianRepeatRejectsCappedSubproblems )
{
  constexpr int al_max_iterations = 4;
  const ProjectionResult result =
      RunProjectionCase( 0.99, -1., 1.e-3, 1, 0., 1., 8, 10, 1.e-30,
                         tribol::PARENT_TRACE_MORTAR, 1., 0.,
                         tribol::PROJECTION_RESPONSE_AUGMENTED_LAGRANGIAN, 1.2, 0.02,
                         1., al_max_iterations, 0, tribol::AL_REPEAT_ON_NONCONVERGENCE );
  EXPECT_NE( result.return_code, 0 );
  EXPECT_FALSE( result.converged );
  EXPECT_FALSE( result.complementarity_converged );
  EXPECT_EQ( result.al_outer_iterations, al_max_iterations );
  EXPECT_EQ( result.al_subproblem_iterations, al_max_iterations );
  EXPECT_GT( result.al_incomplete_subproblems, 0 );
}

TEST( MfemImpulseProjection, ParentTraceMortarAugmentedLagrangianRollbackDiscardsTrialHistory )
{
  const ProjectionResult result =
      RunProjectionCase( 0.99, 0., 1.e-3, 250, 1.e-10, 1.e-8, 8, 10, 1.e-12,
                         tribol::PARENT_TRACE_MORTAR, 1., 0.,
                         tribol::PROJECTION_RESPONSE_AUGMENTED_LAGRANGIAN, 1.2, 0.02,
                         1.e8, 4, 0, tribol::AL_ACCEPT_FEASIBLE, 2, true );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_EQ( result.al_warm_start_rows, 0 );
  EXPECT_EQ( result.al_history_force_norm, 0. );
}

TEST( MfemParentTracePenalty, AugmentedLagrangianUsesSurfaceCompliance )
{
  const ParentTracePenaltyResult baseline = RunParentTracePenaltyCase( 0.99, 1. );
  const ProjectionResult pure_penalty =
      RunPenaltyAugmentedLagrangianCase( 0.99, 0., 1.e-2, 1, 1, false, false );
  const ProjectionResult one_iteration =
      RunPenaltyAugmentedLagrangianCase( 0.99, 0., 1.e-2, 1 );
  const ProjectionResult two_iterations =
      RunPenaltyAugmentedLagrangianCase( 0.99, 0., 1.e-2, 2 );

  EXPECT_EQ( pure_penalty.return_code, 0 );
  EXPECT_EQ( one_iteration.return_code, 0 );
  EXPECT_EQ( two_iterations.return_code, 0 );
  EXPECT_GT( pure_penalty.applied_force, 0. );
  EXPECT_NEAR( pure_penalty.applied_force, baseline.integrated_force,
               1.e-11 * baseline.integrated_force );
  EXPECT_GT( one_iteration.applied_force, 0. );
  EXPECT_LT( one_iteration.applied_force, pure_penalty.applied_force );
  EXPECT_GT( two_iterations.applied_force, one_iteration.applied_force );
  EXPECT_LT( two_iterations.max_endpoint_violation,
             one_iteration.max_endpoint_violation );
  EXPECT_EQ( one_iteration.al_outer_iterations, 1 );
  EXPECT_EQ( two_iterations.al_outer_iterations, 2 );
  EXPECT_TRUE( std::isfinite( two_iterations.penalty_stability_timestep ) );
  EXPECT_NEAR( two_iterations.penalty_stability_timestep,
               pure_penalty.penalty_stability_timestep / std::sqrt( 2. ),
               1.e-11 * pure_penalty.penalty_stability_timestep );
}

TEST( MfemParentTracePenalty, AugmentedLagrangianPredictsClosingContact )
{
  const ProjectionResult pure_penalty =
      RunPenaltyAugmentedLagrangianCase( 1.01, -1., 2.e-2, 1, 1, false, false );
  const ProjectionResult augmented =
      RunPenaltyAugmentedLagrangianCase( 1.01, -1., 2.e-2, 1 );

  EXPECT_EQ( pure_penalty.return_code, 0 );
  EXPECT_EQ( augmented.return_code, 0 );
  EXPECT_EQ( pure_penalty.applied_force, 0. );
  EXPECT_GT( augmented.applied_force, 0. );
}

TEST( MfemParentTracePenalty, AugmentedLagrangianWarmStartsForceHistory )
{
  const ProjectionResult result =
      RunPenaltyAugmentedLagrangianCase( 0.99, 0., 1.e-2, 1, 2 );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_GT( result.al_warm_start_rows, 0 );
  EXPECT_GT( result.al_history_force_norm, 0. );
  EXPECT_GT( result.applied_force, 0. );
}

TEST( MfemParentTracePenalty, AugmentedLagrangianDropsZeroMeasureDualRows )
{
  const ProjectionResult result =
      RunPenaltyAugmentedLagrangianCase( 0.99, 0., 1.e-2, 1, 1, false, true,
                                         0., false, 0.75 );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_GT( result.constraints, 0 );
  EXPECT_LT( result.constraints, 3 );
  EXPECT_GT( result.applied_force, 0. );
}

TEST( MfemParentTracePenalty, AugmentedLagrangianWarmStartPreservesPressureWhenSupportShrinks )
{
  const ProjectionResult result =
      RunPenaltyAugmentedLagrangianCase( 0.99, 0., 1.e-2, 1, 2, false, true,
                                         0., false, 0., 0.75 );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_GT( result.al_warm_start_rows, 0 );
  EXPECT_GT( result.expected_area_scaled_history_force_norm, 0. );
  EXPECT_GT( std::abs( result.expected_area_scaled_history_force_norm -
                       result.unscaled_history_force_norm ),
             1.e-8 );
  EXPECT_NEAR( result.al_history_force_norm,
               result.expected_overlap_limited_history_force_norm,
               1.e-12 * result.expected_overlap_limited_history_force_norm );
}

TEST( MfemParentTracePenalty, AugmentedLagrangianWarmStartDoesNotAmplifyGrowingSupport )
{
  const ProjectionResult result =
      RunPenaltyAugmentedLagrangianCase( 0.99, 0., 1.e-2, 1, 2, false, true,
                                         0., false, 0.75, 0. );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_GT( result.al_warm_start_rows, 0 );
  EXPECT_GT( result.expected_area_scaled_history_force_norm,
             result.expected_overlap_limited_history_force_norm );
  EXPECT_NEAR( result.al_history_force_norm,
               result.expected_overlap_limited_history_force_norm,
               1.e-12 * result.expected_overlap_limited_history_force_norm );
  EXPECT_LE( result.al_history_force_norm,
             result.unscaled_history_force_norm +
                 1.e-12 * result.unscaled_history_force_norm );
}

TEST( MfemParentTracePenalty, AugmentedLagrangianRelaxesUnloadingFaster )
{
  constexpr double unset_offset = std::numeric_limits<double>::quiet_NaN();
  const ProjectionResult symmetric =
      RunPenaltyAugmentedLagrangianCase( 0.98, 0., 1.e-2, 1, 2, false, true,
                                         0., false, 0., unset_offset, 0.25, 0.25, 0., 1.005 );
  const ProjectionResult asymmetric =
      RunPenaltyAugmentedLagrangianCase( 0.98, 0., 1.e-2, 1, 2, false, true,
                                         0., false, 0., unset_offset, 0.25, 1., 0., 1.005 );

  EXPECT_EQ( symmetric.return_code, 0 );
  EXPECT_EQ( asymmetric.return_code, 0 );
  EXPECT_GT( symmetric.al_warm_start_rows, 0 );
  EXPECT_GT( asymmetric.al_warm_start_rows, 0 );
  EXPECT_NEAR( asymmetric.applied_force, symmetric.applied_force,
               1.e-12 * std::max( 1., symmetric.applied_force ) );
  EXPECT_LT( asymmetric.history_total_force, symmetric.history_total_force );
  EXPECT_GT( asymmetric.al_multiplier_update_norm,
             symmetric.al_multiplier_update_norm );
}

TEST( MfemParentTracePenalty, AugmentedLagrangianRollbackDiscardsTrialHistory )
{
  const ProjectionResult result =
      RunPenaltyAugmentedLagrangianCase( 0.99, 0., 1.e-2, 1, 2, true );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_EQ( result.al_warm_start_rows, 0 );
  EXPECT_EQ( result.al_history_force_norm, 0. );
}

TEST( MfemParentTracePenalty, AugmentedLagrangianSmoothsPersistentPressureConservatively )
{
  const ProjectionResult unsmoothed =
      RunPenaltyAugmentedLagrangianCase( 0.99, -1., 1.e-2, 1, 1, false, true, 0., true );
  const ProjectionResult smoothed =
      RunPenaltyAugmentedLagrangianCase( 0.99, -1., 1.e-2, 1, 1, false, true, 1., true );

  EXPECT_EQ( unsmoothed.return_code, 0 );
  EXPECT_EQ( smoothed.return_code, 0 );
  EXPECT_NEAR( smoothed.applied_force, unsmoothed.applied_force,
               1.e-12 * unsmoothed.applied_force );
  EXPECT_NEAR( smoothed.history_total_force, unsmoothed.history_total_force,
               1.e-12 * unsmoothed.history_total_force );
  EXPECT_GE( smoothed.history_minimum_pressure, 0. );
  EXPECT_GT( unsmoothed.history_maximum_pressure - unsmoothed.history_minimum_pressure, 0. );
  EXPECT_LT( smoothed.history_maximum_pressure - smoothed.history_minimum_pressure,
             unsmoothed.history_maximum_pressure - unsmoothed.history_minimum_pressure );
}

TEST( MfemParentTracePenalty, QuadratureHybridMatchesLocalPenaltyOnFirstUpdate )
{
  const ProjectionResult result = RunPenaltyAugmentedLagrangianCase(
      0.99, 0., 1.e-2, 1, 1, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), 1., 1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_HYBRID );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_EQ( result.al_outer_iterations, 1 );
  EXPECT_EQ( result.al_subproblem_iterations, 0 );
  EXPECT_GT( result.applied_force, 0. );
  EXPECT_NEAR( result.applied_force, 5.e-2, 1.e-12 );
  EXPECT_NEAR( result.history_total_force, result.applied_force, 1.e-12 );
}

TEST( MfemParentTracePenalty, QuadratureHybridFixedIterationsUseIndependentInnerRelaxation )
{
  const ProjectionResult one_iteration = RunFixedIterationQuadratureHybridCase( 1, 1., 0.25 );
  const ProjectionResult five_half_relaxed = RunFixedIterationQuadratureHybridCase( 5, 0.5, 0.25 );
  const ProjectionResult five_full_relaxed = RunFixedIterationQuadratureHybridCase( 5, 1., 0.25 );

  EXPECT_EQ( one_iteration.return_code, 0 );
  EXPECT_EQ( five_half_relaxed.return_code, 0 );
  EXPECT_EQ( five_full_relaxed.return_code, 0 );
  EXPECT_EQ( one_iteration.al_outer_iterations, 1 );
  EXPECT_EQ( five_half_relaxed.al_outer_iterations, 5 );
  EXPECT_EQ( five_full_relaxed.al_outer_iterations, 5 );
  EXPECT_NEAR( five_half_relaxed.applied_force, 3. * one_iteration.applied_force, 1.e-12 );
  EXPECT_NEAR( five_full_relaxed.applied_force, 5. * one_iteration.applied_force, 1.e-12 );
  EXPECT_NEAR( one_iteration.history_total_force, 0.25 * one_iteration.applied_force, 1.e-12 );
  EXPECT_NEAR( five_half_relaxed.history_total_force, 0.25 * five_half_relaxed.applied_force, 1.e-12 );
  EXPECT_NEAR( five_full_relaxed.history_total_force, 0.25 * five_full_relaxed.applied_force, 1.e-12 );
  EXPECT_NEAR( five_half_relaxed.penalty_stability_timestep,
               one_iteration.penalty_stability_timestep / std::sqrt( 3. ),
               1.e-12 * one_iteration.penalty_stability_timestep );
  EXPECT_NEAR( five_full_relaxed.penalty_stability_timestep,
               one_iteration.penalty_stability_timestep / std::sqrt( 5. ),
               1.e-12 * one_iteration.penalty_stability_timestep );
}

TEST( MfemParentTracePenalty, QuadratureLocalMatchesEquivalentQuadraturePenalty )
{
  const ParentTracePenaltyResult pure_five_penalty =
      RunParentTracePenaltyCase( 0.99, 5., false, 0., 1.e-2, true,
                                 tribol::COMMON_PLANE, 3 );
  const ParentTracePenaltyResult pure_three_penalty =
      RunParentTracePenaltyCase( 0.99, 3., false, 0., 1.e-2, true,
                                 tribol::COMMON_PLANE, 3 );
  const ProjectionResult unit_penalty =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase( 1, 1. );
  const ProjectionResult five_iteration =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase( 5, 1. );
  const ProjectionResult five_penalty =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase( 1, 1., 5. );
  const ProjectionResult five_half_relaxed =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase( 5, 0.5 );
  const ProjectionResult three_penalty =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase( 1, 1., 3. );

  EXPECT_EQ( pure_five_penalty.return_code, 0 );
  EXPECT_EQ( pure_three_penalty.return_code, 0 );
  EXPECT_EQ( unit_penalty.return_code, 0 );
  EXPECT_EQ( five_iteration.return_code, 0 );
  EXPECT_EQ( five_penalty.return_code, 0 );
  EXPECT_EQ( five_half_relaxed.return_code, 0 );
  EXPECT_EQ( three_penalty.return_code, 0 );
  EXPECT_NEAR( five_iteration.applied_force, five_penalty.applied_force,
               1.e-12 * five_penalty.applied_force );
  EXPECT_NEAR( five_iteration.response_norm, five_penalty.response_norm,
               1.e-12 * five_penalty.response_norm );
  EXPECT_NEAR( five_iteration.penalty_stability_timestep,
               five_penalty.penalty_stability_timestep,
               1.e-12 * five_penalty.penalty_stability_timestep );
  EXPECT_NEAR( five_iteration.applied_force, pure_five_penalty.integrated_force,
               1.e-12 * pure_five_penalty.integrated_force );
  EXPECT_NEAR( five_iteration.response_norm, pure_five_penalty.response_norm,
               1.e-12 * pure_five_penalty.response_norm );
  EXPECT_NEAR( five_iteration.penalty_stability_timestep,
               pure_five_penalty.stability_timestep,
               1.e-12 * pure_five_penalty.stability_timestep );
  EXPECT_NEAR( five_half_relaxed.applied_force, three_penalty.applied_force,
               1.e-12 * three_penalty.applied_force );
  EXPECT_NEAR( five_half_relaxed.response_norm, three_penalty.response_norm,
               1.e-12 * three_penalty.response_norm );
  EXPECT_NEAR( five_half_relaxed.penalty_stability_timestep,
               three_penalty.penalty_stability_timestep,
               1.e-12 * three_penalty.penalty_stability_timestep );
  EXPECT_NEAR( five_half_relaxed.applied_force, pure_three_penalty.integrated_force,
               1.e-12 * pure_three_penalty.integrated_force );
  EXPECT_NEAR( five_half_relaxed.response_norm, pure_three_penalty.response_norm,
               1.e-12 * pure_three_penalty.response_norm );
  EXPECT_NEAR( five_half_relaxed.penalty_stability_timestep,
               pure_three_penalty.stability_timestep,
               1.e-12 * pure_three_penalty.stability_timestep );
  EXPECT_NEAR( five_iteration.applied_force, 5. * unit_penalty.applied_force,
               1.e-12 * five_iteration.applied_force );
  EXPECT_NEAR( five_iteration.penalty_stability_timestep,
               unit_penalty.penalty_stability_timestep / std::sqrt( 5. ),
               1.e-12 * unit_penalty.penalty_stability_timestep );
}

TEST( MfemParentTracePenalty, QuadratureLocalDoesNotRetainMultiplierHistory )
{
  const ProjectionResult single_update =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase( 5, 0.5 );
  const ProjectionResult repeated_update =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase( 5, 0.5, 1., 2 );
  const ProjectionResult two_rk_stages =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase(
          5, 0.5, 1., 1, 0., std::numeric_limits<double>::quiet_NaN(), true );
  const ProjectionResult fresh_slid_geometry =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase( 5, 0.5, 1., 1, 0.2 );
  const ProjectionResult rebuilt_slid_geometry =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase( 5, 0.5, 1., 2, 0., 0.2 );

  EXPECT_EQ( single_update.return_code, 0 );
  EXPECT_EQ( repeated_update.return_code, 0 );
  EXPECT_EQ( two_rk_stages.return_code, 0 );
  EXPECT_EQ( fresh_slid_geometry.return_code, 0 );
  EXPECT_EQ( rebuilt_slid_geometry.return_code, 0 );
  EXPECT_EQ( single_update.al_warm_start_rows, 0 );
  EXPECT_EQ( repeated_update.al_warm_start_rows, 0 );
  EXPECT_EQ( two_rk_stages.al_warm_start_rows, 0 );
  EXPECT_EQ( rebuilt_slid_geometry.al_warm_start_rows, 0 );
  EXPECT_EQ( single_update.al_history_force_norm, 0. );
  EXPECT_EQ( repeated_update.al_history_force_norm, 0. );
  EXPECT_EQ( two_rk_stages.al_history_force_norm, 0. );
  EXPECT_EQ( rebuilt_slid_geometry.al_history_force_norm, 0. );
  EXPECT_EQ( single_update.history_total_force, 0. );
  EXPECT_EQ( repeated_update.history_total_force, 0. );
  EXPECT_EQ( two_rk_stages.history_total_force, 0. );
  EXPECT_EQ( rebuilt_slid_geometry.history_total_force, 0. );
  EXPECT_NEAR( repeated_update.applied_force, single_update.applied_force,
               1.e-12 * single_update.applied_force );
  EXPECT_NEAR( repeated_update.response_norm, single_update.response_norm,
               1.e-12 * single_update.response_norm );
  EXPECT_NEAR( two_rk_stages.applied_force, single_update.applied_force,
               1.e-12 * single_update.applied_force );
  EXPECT_NEAR( two_rk_stages.response_norm, single_update.response_norm,
               1.e-12 * single_update.response_norm );
  EXPECT_NEAR( rebuilt_slid_geometry.applied_force, fresh_slid_geometry.applied_force,
               1.e-12 * fresh_slid_geometry.applied_force );
  EXPECT_NEAR( rebuilt_slid_geometry.response_norm, fresh_slid_geometry.response_norm,
               1.e-12 * fresh_slid_geometry.response_norm );
}

TEST( MfemParentTracePenalty, QuadratureCellMemoryAccumulatesAcceptedHistory )
{
  const ProjectionResult pure_penalty =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase( 1, 1. );
  const ProjectionResult first_update =
      RunQuadratureCellMemoryPenaltyAugmentedLagrangianCase( 1. );
  const ProjectionResult second_update =
      RunQuadratureCellMemoryPenaltyAugmentedLagrangianCase( 1., 2 );

  EXPECT_EQ( pure_penalty.return_code, 0 );
  EXPECT_EQ( first_update.return_code, 0 );
  EXPECT_EQ( second_update.return_code, 0 );
  EXPECT_EQ( first_update.al_warm_start_rows, 0 );
  EXPECT_GT( second_update.al_warm_start_rows, 0 );
  EXPECT_NEAR( first_update.applied_force, pure_penalty.applied_force,
               1.e-12 * pure_penalty.applied_force );
  EXPECT_NEAR( first_update.history_total_force, first_update.applied_force,
               1.e-12 * first_update.applied_force );
  EXPECT_NEAR( second_update.applied_force, 2. * first_update.applied_force,
               1.e-12 * second_update.applied_force );
  EXPECT_NEAR( second_update.history_total_force, second_update.applied_force,
               1.e-12 * second_update.applied_force );
  EXPECT_NEAR( second_update.penalty_stability_timestep,
               first_update.penalty_stability_timestep,
               1.e-12 * first_update.penalty_stability_timestep );
}

TEST( MfemParentTracePenalty, QuadratureCellMemoryUsesAcceptedStepHistoryAcrossStages )
{
  const ProjectionResult single_stage =
      RunQuadratureCellMemoryPenaltyAugmentedLagrangianCase( 1. );
  const ProjectionResult two_stages =
      RunQuadratureCellMemoryPenaltyAugmentedLagrangianCase(
          1., 1, 0., std::numeric_limits<double>::quiet_NaN(), true );
  const ProjectionResult rollback =
      RunQuadratureCellMemoryPenaltyAugmentedLagrangianCase(
          1., 2, 0., std::numeric_limits<double>::quiet_NaN(), false, true );

  EXPECT_EQ( single_stage.return_code, 0 );
  EXPECT_EQ( two_stages.return_code, 0 );
  EXPECT_EQ( rollback.return_code, 0 );
  EXPECT_EQ( two_stages.al_warm_start_rows, 0 );
  EXPECT_EQ( rollback.al_warm_start_rows, 0 );
  EXPECT_NEAR( two_stages.applied_force, single_stage.applied_force,
               1.e-12 * single_stage.applied_force );
  EXPECT_NEAR( two_stages.history_total_force, single_stage.history_total_force,
               1.e-12 * single_stage.history_total_force );
  EXPECT_NEAR( rollback.applied_force, single_stage.applied_force,
               1.e-12 * single_stage.applied_force );
  EXPECT_NEAR( rollback.history_total_force, single_stage.history_total_force,
               1.e-12 * single_stage.history_total_force );
}

TEST( MfemParentTracePenalty, QuadratureCellMemoryProjectsPartialOverlapInParentCoordinates )
{
  const ProjectionResult fresh_full_overlap =
      RunQuadratureCellMemoryPenaltyAugmentedLagrangianCase( 1. );
  const ProjectionResult expanded_overlap =
      RunQuadratureCellMemoryPenaltyAugmentedLagrangianCase( 1., 2, 0.2, 0. );

  EXPECT_EQ( fresh_full_overlap.return_code, 0 );
  EXPECT_EQ( expanded_overlap.return_code, 0 );
  EXPECT_GT( expanded_overlap.al_warm_start_rows, 0 );
  EXPECT_NEAR( expanded_overlap.applied_force,
               1.8 * fresh_full_overlap.applied_force,
               1.e-11 * expanded_overlap.applied_force );
  EXPECT_NEAR( expanded_overlap.history_total_force,
               expanded_overlap.applied_force,
               1.e-11 * expanded_overlap.applied_force );
}

TEST( MfemParentTracePenalty, QuadratureCellMemoryAveragesOverlappingIntervals )
{
  const ProjectionResult first_update =
      RunQuadratureCellMemoryPenaltyAugmentedLagrangianCase( 1. );
  const ProjectionResult duplicated_boundary_sliver =
      RunQuadratureCellMemoryPenaltyAugmentedLagrangianCase(
          1., 2, 0., std::numeric_limits<double>::quiet_NaN(), false, false, 1.e-7 );

  EXPECT_EQ( first_update.return_code, 0 );
  EXPECT_EQ( duplicated_boundary_sliver.return_code, 0 );
  EXPECT_GT( duplicated_boundary_sliver.al_warm_start_rows, 0 );
  EXPECT_NEAR( duplicated_boundary_sliver.applied_force,
               2. * first_update.applied_force,
               1.e-11 * duplicated_boundary_sliver.applied_force );
  EXPECT_NEAR( duplicated_boundary_sliver.history_total_force,
               duplicated_boundary_sliver.applied_force,
               1.e-11 * duplicated_boundary_sliver.applied_force );

  const ProjectionResult overlapping_history =
      RunQuadratureCellMemoryPenaltyAugmentedLagrangianCase(
          1., 2, 0., std::numeric_limits<double>::quiet_NaN(), false, false, 0.25 );
  EXPECT_EQ( overlapping_history.return_code, 0 );
  EXPECT_NEAR( overlapping_history.applied_force,
               2. * first_update.applied_force,
               1.e-11 * overlapping_history.applied_force );
  EXPECT_NEAR( overlapping_history.history_total_force,
               overlapping_history.applied_force,
               1.e-11 * overlapping_history.applied_force );
}

TEST( MfemParentTracePenalty, ComplianceCorrectedQuadratureCellIterationCapReducesResidual )
{
  const ProjectionResult one_iteration =
      RunComplianceCorrectedQuadratureCellMemoryCase( 1 );
  const ProjectionResult four_iterations =
      RunComplianceCorrectedQuadratureCellMemoryCase( 4 );
  const ProjectionResult sixteen_iterations =
      RunComplianceCorrectedQuadratureCellMemoryCase( 16 );

  EXPECT_EQ( one_iteration.return_code, 0 );
  EXPECT_EQ( four_iterations.return_code, 0 );
  EXPECT_EQ( sixteen_iterations.return_code, 0 );
  EXPECT_EQ( one_iteration.al_outer_iterations, 1 );
  EXPECT_EQ( four_iterations.al_outer_iterations, 4 );
  EXPECT_LE( sixteen_iterations.al_outer_iterations, 16 );
  EXPECT_LT( four_iterations.final_residual, one_iteration.final_residual );
  EXPECT_LT( sixteen_iterations.final_residual, four_iterations.final_residual );
  EXPECT_LT( sixteen_iterations.final_primal_residual,
             one_iteration.final_primal_residual );
}

TEST( MfemParentTracePenalty, QuadratureCellUnitContinuationMatchesFixedPenalty )
{
  const ProjectionResult baseline =
      RunComplianceCorrectedQuadratureCellMemoryCase( 5 );
  const ProjectionResult unit_continuation =
      RunComplianceCorrectedQuadratureCellMemoryCase( 5, 1, 1., 1., 0., 1., 1. );

  EXPECT_EQ( baseline.return_code, 0 );
  EXPECT_EQ( unit_continuation.return_code, 0 );
  EXPECT_EQ( unit_continuation.al_outer_iterations, baseline.al_outer_iterations );
  EXPECT_NEAR( unit_continuation.final_residual, baseline.final_residual, 1.e-14 );
  EXPECT_NEAR( unit_continuation.final_primal_residual,
               baseline.final_primal_residual, 1.e-14 );
  EXPECT_NEAR( unit_continuation.applied_force, baseline.applied_force, 1.e-14 );
  EXPECT_NEAR( unit_continuation.penalty_stability_timestep,
               baseline.penalty_stability_timestep, 1.e-14 );
}

TEST( MfemParentTracePenalty, QuadratureCellContinuationReducesViolationAndTimestepVote )
{
  const ProjectionResult fixed_penalty =
      RunComplianceCorrectedQuadratureCellMemoryCase( 5 );
  const ProjectionResult continuation =
      RunComplianceCorrectedQuadratureCellMemoryCase( 5, 1, 1., 1., 0., 2., 16. );

  EXPECT_EQ( fixed_penalty.return_code, 0 );
  EXPECT_EQ( continuation.return_code, 0 );
  EXPECT_EQ( continuation.al_outer_iterations, 5 );
  EXPECT_TRUE( continuation.final_residual < fixed_penalty.final_residual ||
               continuation.final_primal_residual <
                   fixed_penalty.final_primal_residual );
  EXPECT_NEAR( continuation.penalty_stability_timestep,
               fixed_penalty.penalty_stability_timestep * std::sqrt( 5. / 16. ),
               1.e-11 * fixed_penalty.penalty_stability_timestep );
}

TEST( MfemParentTracePenalty, QuadratureCellContinuationAcceptsOverRelaxation )
{
  const ProjectionResult continuation =
      RunComplianceCorrectedQuadratureCellMemoryCase( 5, 1, 1., 1.5, 0., 2., 16. );

  EXPECT_EQ( continuation.return_code, 0 );
  EXPECT_GT( continuation.al_outer_iterations, 0 );
  EXPECT_TRUE( std::isfinite( continuation.final_residual ) );
  EXPECT_TRUE( std::isfinite( continuation.final_primal_residual ) );
  EXPECT_TRUE( std::isfinite( continuation.penalty_stability_timestep ) );
}

TEST( MfemParentTracePenalty, QuadratureCellGapToleranceRetainsPenetrationPenalty )
{
  const ProjectionResult result =
      RunComplianceCorrectedQuadratureCellMemoryCase( 20, 1, 1., 1., 0., 2., 20.,
                                                       1.e-1, 0. );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.complementarity_converged );
  EXPECT_EQ( result.al_outer_iterations, 1 );
  EXPECT_GT( result.applied_force, 0. );
  EXPECT_GT( result.al_multiplier_update_norm, 0. );
}

TEST( MfemParentTracePenalty, ComplianceCorrectedQuadratureCellReusesAcceptedHistory )
{
  const ProjectionResult first_step =
      RunComplianceCorrectedQuadratureCellMemoryCase( 2000 );
  const ProjectionResult second_step =
      RunComplianceCorrectedQuadratureCellMemoryCase( 2000, 2 );

  EXPECT_EQ( first_step.return_code, 0 );
  EXPECT_EQ( second_step.return_code, 0 );
  EXPECT_TRUE( first_step.complementarity_converged );
  EXPECT_TRUE( second_step.complementarity_converged );
  EXPECT_GT( second_step.al_warm_start_rows, 0 );
  EXPECT_LT( second_step.al_outer_iterations, first_step.al_outer_iterations );
  EXPECT_NEAR( second_step.applied_force, first_step.applied_force,
               1.e-8 * first_step.applied_force );
}

TEST( MfemParentTracePenalty, ComplianceCorrectedParentTraceIterationCapReducesResidual )
{
  const ProjectionResult one_iteration =
      RunComplianceCorrectedParentTraceCase( 1 );
  const ProjectionResult four_iterations =
      RunComplianceCorrectedParentTraceCase( 4 );
  const ProjectionResult sixteen_iterations =
      RunComplianceCorrectedParentTraceCase( 16 );

  EXPECT_EQ( one_iteration.return_code, 0 );
  EXPECT_EQ( four_iterations.return_code, 0 );
  EXPECT_EQ( sixteen_iterations.return_code, 0 );
  EXPECT_EQ( one_iteration.al_outer_iterations, 1 );
  EXPECT_LE( four_iterations.al_outer_iterations, 4 );
  EXPECT_LE( sixteen_iterations.al_outer_iterations, 16 );
  EXPECT_LT( four_iterations.final_residual, one_iteration.final_residual );
  EXPECT_LE( sixteen_iterations.final_residual, four_iterations.final_residual );
  EXPECT_LT( sixteen_iterations.final_primal_residual,
             one_iteration.final_primal_residual );
}

TEST( MfemParentTracePenalty, ComplianceCorrectedParentTraceReusesAcceptedHistory )
{
  const ProjectionResult first_step =
      RunComplianceCorrectedParentTraceCase( 2000 );
  const ProjectionResult second_step =
      RunComplianceCorrectedParentTraceCase( 2000, 2 );

  EXPECT_EQ( first_step.return_code, 0 );
  EXPECT_EQ( second_step.return_code, 0 );
  EXPECT_TRUE( first_step.complementarity_converged );
  EXPECT_TRUE( second_step.complementarity_converged );
  EXPECT_GT( second_step.al_warm_start_rows, 0 );
  EXPECT_LE( second_step.al_outer_iterations, first_step.al_outer_iterations );
  EXPECT_NEAR( second_step.applied_force, first_step.applied_force,
               1.e-8 * first_step.applied_force );
}

TEST( MfemParentTracePenalty, ComplianceCorrectedParentTraceReportsNodalGaps )
{
  const ProjectionResult result = RunComplianceCorrectedParentTraceCase( 2000 );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( std::isfinite( result.quadrature_average_gap_violation ) );
  EXPECT_TRUE( std::isfinite( result.quadrature_maximum_gap_violation ) );
  EXPECT_GT( result.quadrature_maximum_gap_violation, 0. );
  EXPECT_GT( result.parent_trace_gap_rows, 0 );
  EXPECT_GT( result.parent_trace_geometric_gap_rows, 0 );
  EXPECT_TRUE( std::isfinite( result.parent_trace_current_average_gap_violation ) );
  EXPECT_TRUE( std::isfinite( result.parent_trace_current_maximum_gap_violation ) );
  EXPECT_TRUE( std::isfinite( result.parent_trace_endpoint_average_gap_violation ) );
  EXPECT_TRUE( std::isfinite( result.parent_trace_endpoint_maximum_gap_violation ) );
  EXPECT_TRUE( std::isfinite( result.parent_trace_geometric_average_gap_violation ) );
  EXPECT_TRUE( std::isfinite( result.parent_trace_geometric_maximum_gap_violation ) );
  EXPECT_GT( result.parent_trace_current_maximum_gap_violation, 0. );
  EXPECT_GT( result.parent_trace_geometric_maximum_gap_violation, 0. );
  EXPECT_LT( result.parent_trace_endpoint_maximum_gap_violation,
             result.parent_trace_current_maximum_gap_violation );
}

TEST( MfemParentTracePenalty, QuadratureLocalDampingUsesEffectiveStiffness )
{
  const ProjectionResult five_iteration =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase(
          5, 1., 1., 1, 0., std::numeric_limits<double>::quiet_NaN(), false, -1., 0.5 );
  const ProjectionResult five_penalty =
      RunQuadratureLocalPenaltyAugmentedLagrangianCase(
          1, 1., 5., 1, 0., std::numeric_limits<double>::quiet_NaN(), false, -1., 0.5 );

  EXPECT_EQ( five_iteration.return_code, 0 );
  EXPECT_EQ( five_penalty.return_code, 0 );
  EXPECT_GT( five_iteration.penalty_al_damping_force, 0. );
  EXPECT_NEAR( five_iteration.applied_force, five_penalty.applied_force,
               1.e-12 * five_penalty.applied_force );
  EXPECT_NEAR( five_iteration.penalty_al_damping_force,
               five_penalty.penalty_al_damping_force,
               1.e-12 * five_penalty.penalty_al_damping_force );
  EXPECT_NEAR( five_iteration.penalty_al_damping_dissipation_rate,
               five_penalty.penalty_al_damping_dissipation_rate,
               1.e-12 * five_penalty.penalty_al_damping_dissipation_rate );
  EXPECT_NEAR( five_iteration.penalty_stability_timestep,
               five_penalty.penalty_stability_timestep,
               1.e-12 * five_penalty.penalty_stability_timestep );
}

TEST( MfemParentTracePenalty, QuadratureHybridRkStagesUseAcceptedStepHistory )
{
  const ProjectionResult single_stage = RunPenaltyAugmentedLagrangianCase(
      0.99, 0., 1.e-2, 1, 1, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), 1., 1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_HYBRID, 2.e-2, 2.e-2 );
  const ProjectionResult two_stages = RunPenaltyAugmentedLagrangianCase(
      0.99, 0., 1.e-2, 1, 1, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), 1., 1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_HYBRID, 2.e-2, 2.e-2, 0., true );

  EXPECT_EQ( single_stage.return_code, 0 );
  EXPECT_EQ( two_stages.return_code, 0 );
  EXPECT_NEAR( two_stages.applied_force, single_stage.applied_force,
               1.e-12 * single_stage.applied_force );
  EXPECT_NEAR( two_stages.history_total_force, single_stage.history_total_force,
               1.e-12 * single_stage.history_total_force );
  const double expected_relaxation = 1. - std::exp( -0.5 );
  EXPECT_NEAR( single_stage.history_total_force,
               expected_relaxation * single_stage.applied_force,
               1.e-12 * single_stage.applied_force );
}

TEST( MfemParentTracePenalty, QuadratureHybridC1ActivationRampsPressure )
{
  const ProjectionResult baseline = RunPenaltyAugmentedLagrangianCase(
      0.99, 0., 1.e-2, 1, 1, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), 1., 1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_HYBRID );
  const ProjectionResult regularized = RunPenaltyAugmentedLagrangianCase(
      0.99, 0., 1.e-2, 1, 1, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), 1., 1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_HYBRID, 0., 0., 0.1 );

  EXPECT_EQ( baseline.return_code, 0 );
  EXPECT_EQ( regularized.return_code, 0 );
  EXPECT_GT( regularized.applied_force, 0. );
  EXPECT_LT( regularized.applied_force, baseline.applied_force );
  EXPECT_NEAR( regularized.penalty_stability_timestep,
               baseline.penalty_stability_timestep * std::sqrt( 16. / 27. ),
               1.e-12 * baseline.penalty_stability_timestep );
}

TEST( MfemParentTracePenalty, QuadratureHybridClearsSeparatedMultiplierWithFullUnloading )
{
  const ProjectionResult result = RunPenaltyAugmentedLagrangianCase(
      0.99, 0., 1.e-2, 1, 2, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), 0.25, 1., 1.e-3, 1.02,
      tribol::PENALTY_AL_QUADRATURE_HYBRID );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_GT( result.al_warm_start_rows, 0 );
  EXPECT_NEAR( result.applied_force, 0., 1.e-14 );
  EXPECT_NEAR( result.history_total_force, 0., 1.e-14 );
}

TEST( MfemParentTracePenalty, ComplianceCorrectedHybridTargetsAllowablePenetration )
{
  const ProjectionResult zero_target = RunComplianceCorrectedPenaltyAugmentedLagrangianCase( 0.99, 0., 0.5, 2000 );
  const ProjectionResult finite_target =
      RunComplianceCorrectedPenaltyAugmentedLagrangianCase( 0.99, 0., 0.5, 2000, 0.005 );

  EXPECT_EQ( zero_target.return_code, 0 );
  EXPECT_EQ( finite_target.return_code, 0 );
  EXPECT_TRUE( zero_target.complementarity_converged );
  EXPECT_TRUE( finite_target.complementarity_converged );
  EXPECT_GT( zero_target.applied_force, finite_target.applied_force );
  EXPECT_GT( finite_target.applied_force, 0. );
}

TEST( MfemParentTracePenalty, ComplianceCorrectedHybridIterationCapReducesResidual )
{
  const ProjectionResult two_iterations = RunComplianceCorrectedPenaltyAugmentedLagrangianCase( 0.99, -1., 2.e-2, 2 );
  const ProjectionResult four_iterations = RunComplianceCorrectedPenaltyAugmentedLagrangianCase( 0.99, -1., 2.e-2, 4 );
  const ProjectionResult eight_iterations = RunComplianceCorrectedPenaltyAugmentedLagrangianCase( 0.99, -1., 2.e-2, 8 );

  EXPECT_EQ( two_iterations.return_code, 0 );
  EXPECT_EQ( four_iterations.return_code, 0 );
  EXPECT_EQ( eight_iterations.return_code, 0 );
  EXPECT_EQ( two_iterations.al_outer_iterations, 2 );
  EXPECT_EQ( four_iterations.al_outer_iterations, 4 );
  EXPECT_EQ( eight_iterations.al_outer_iterations, 8 );
  EXPECT_LT( four_iterations.final_residual, two_iterations.final_residual );
  EXPECT_LT( eight_iterations.final_residual, four_iterations.final_residual );
  EXPECT_LT( eight_iterations.final_primal_residual, two_iterations.final_primal_residual );
}

TEST( MfemParentTracePenalty, ComplianceCorrectedHybridDoesNotStoreDampingAsMultiplier )
{
  const ProjectionResult result = RunComplianceCorrectedPenaltyAugmentedLagrangianCase( 1.01, -0.1, 2.e-2, 2, 0., 0.5 );

  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.complementarity_converged );
  EXPECT_GT( result.penalty_al_damping_qpts, 0 );
  EXPECT_GT( result.penalty_al_damping_force, 0. );
  EXPECT_NEAR( result.applied_force, result.penalty_al_damping_force, 1.e-12 * result.applied_force );
  EXPECT_NEAR( result.history_total_force, 0., 1.e-14 );
}

TEST( MfemParentTracePenalty, ComplianceCorrectedHybridTargetIsPenaltyScaleIndependent )
{
  const ProjectionResult unit_penalty =
      RunComplianceCorrectedPenaltyAugmentedLagrangianCase( 0.99, -0.1, 0.2, 2000, 0.005, 0., 1. );
  const ProjectionResult quadruple_penalty =
      RunComplianceCorrectedPenaltyAugmentedLagrangianCase( 0.99, -0.1, 0.2, 2000, 0.005, 0., 4. );

  EXPECT_EQ( unit_penalty.return_code, 0 );
  EXPECT_EQ( quadruple_penalty.return_code, 0 );
  EXPECT_TRUE( unit_penalty.complementarity_converged );
  EXPECT_TRUE( quadruple_penalty.complementarity_converged );
  EXPECT_LE( unit_penalty.final_primal_residual, unit_penalty.primal_tolerance );
  EXPECT_LE( quadruple_penalty.final_primal_residual, quadruple_penalty.primal_tolerance );
  EXPECT_NEAR( unit_penalty.applied_force, quadruple_penalty.applied_force,
               1.e-4 * unit_penalty.applied_force );
}

TEST( MfemParentTracePenalty, QuadratureHybridNormalDampingActsOnlyWhileClosing )
{
  const ProjectionResult baseline = RunPenaltyAugmentedLagrangianCase(
      1.01, -1., 2.e-2, 1, 1, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), 1., 1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_HYBRID );
  const ProjectionResult closing = RunPenaltyAugmentedLagrangianCase(
      1.01, -1., 2.e-2, 1, 1, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), 1., 1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_HYBRID, 0., 0., 0., false, -1., 0., 0.5 );
  const ProjectionResult opening = RunPenaltyAugmentedLagrangianCase(
      1.01, 1., 2.e-2, 1, 1, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), 1., 1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_HYBRID, 0., 0., 0., false, -1., 0., 0.5 );

  EXPECT_EQ( baseline.return_code, 0 );
  EXPECT_EQ( closing.return_code, 0 );
  EXPECT_EQ( opening.return_code, 0 );
  EXPECT_NEAR( baseline.applied_force, 0., 1.e-14 );
  EXPECT_GT( closing.penalty_al_damping_qpts, 0 );
  EXPECT_GT( closing.penalty_al_damping_force, 0. );
  EXPECT_NEAR( closing.applied_force, closing.penalty_al_damping_force,
               1.e-12 * closing.penalty_al_damping_force );
  EXPECT_NEAR( closing.penalty_al_damping_dissipation_rate,
               closing.penalty_al_damping_force,
               1.e-12 * closing.penalty_al_damping_force );
  EXPECT_GT( closing.penalty_al_maximum_damping_pressure, 0. );
  EXPECT_NEAR( closing.history_total_force, 0., 1.e-14 );
  EXPECT_LT( closing.penalty_stability_timestep, baseline.penalty_stability_timestep );
  EXPECT_EQ( opening.penalty_al_damping_qpts, 0 );
  EXPECT_NEAR( opening.penalty_al_damping_force, 0., 1.e-14 );
  EXPECT_NEAR( opening.penalty_al_damping_dissipation_rate, 0., 1.e-14 );
  EXPECT_NEAR( opening.applied_force, 0., 1.e-14 );
}

TEST( MfemParentTracePenalty, QuadratureHybridFixedActiveSetIsBilateralAfterLock )
{
  const ProjectionResult before_lock = RunPenaltyAugmentedLagrangianCase(
      1.01, 0., 1.e-2, 1, 2, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), 1., 1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_HYBRID, 2.e-2, 2.e-2, 0., false, 1., 0. );
  const ProjectionResult after_lock = RunPenaltyAugmentedLagrangianCase(
      1.01, 0., 1.e-2, 1, 2, false, true, 0., false, 0.,
      std::numeric_limits<double>::quiet_NaN(), 1., 1., 1.e-3,
      std::numeric_limits<double>::quiet_NaN(),
      tribol::PENALTY_AL_QUADRATURE_HYBRID, 2.e-2, 2.e-2, 0., false, 0., 0. );

  EXPECT_EQ( before_lock.return_code, 0 );
  EXPECT_EQ( before_lock.applied_force, 0. );
  EXPECT_EQ( before_lock.applied_tensile_force, 0. );
  EXPECT_EQ( before_lock.history_total_force, 0. );
  EXPECT_EQ( before_lock.penalty_stability_active_rows, 0 );

  EXPECT_EQ( after_lock.return_code, 0 );
  EXPECT_EQ( after_lock.applied_force, 0. );
  EXPECT_GT( after_lock.applied_tensile_force, 0. );
  EXPECT_LT( after_lock.history_total_force, 0. );
  EXPECT_LT( after_lock.history_minimum_pressure, 0. );
  EXPECT_GT( after_lock.al_warm_start_rows, 0 );
  EXPECT_EQ( after_lock.penalty_stability_active_rows, after_lock.constraints );
  EXPECT_EQ( after_lock.penalty_stability_predicted_rows, 0 );
  EXPECT_TRUE( std::isfinite( after_lock.penalty_stability_timestep ) );
}

}  // namespace

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();
#endif

  mfem::Device device( "cpu" );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
