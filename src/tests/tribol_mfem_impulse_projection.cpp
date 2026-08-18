#include <cmath>
#include <set>

#include <gtest/gtest.h>

#include "axom/slic.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"

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
};

ProjectionResult RunProjectionCase( double upper_offset, double upper_velocity, double dt, int max_iterations = 250,
                                    double relative_tolerance = 1.e-10, double primal_relative_tolerance = 1.e-6,
                                    int lor_factor = 2, int quadrature_order = 3, double absolute_tolerance = 1.e-12,
                                    tribol::ContactMethod method = tribol::COMMON_PLANE,
                                    double position_velocity_scale = 1.,
                                    double projection_base_upper_velocity = 0.,
                                    tribol::ImpulseProjectionContactResponse contact_response =
                                        tribol::PROJECTION_RESPONSE_EXACT,
                                    double damping_ratio = 1.2, double max_penetration_fraction = 0.02 )
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
                                      tribol::IMPULSE_PROJECTION, tribol::BINNING_GRID,
                                      tribol::ExecutionMode::Sequential );
  tribol::setMfemLORFactor( coupling_scheme_id, lor_factor );
  tribol::setMfemSurfaceBasis( coupling_scheme_id, tribol::MfemSurfaceBasis::PARENT );
  tribol::registerMfemVelocity( coupling_scheme_id, velocity );
  if ( position_velocity_scale < 1. ) {
    tribol::registerMfemProjectionBaseVelocity( coupling_scheme_id, projection_base_velocity );
    tribol::setImpulseProjectionKinematics( coupling_scheme_id, position_velocity_scale );
  }
  tribol::registerMfemInverseMass( coupling_scheme_id, inverse_mass );
  tribol::setCommonPlaneIntegrationOptions( coupling_scheme_id, tribol::MULTI_POINT, quadrature_order );
  tribol::setImpulseProjectionOptions( coupling_scheme_id, max_iterations, relative_tolerance, absolute_tolerance, 1.,
                                       primal_relative_tolerance );
  if ( method == tribol::PARENT_TRACE_MORTAR ) {
    tribol::setParentTraceMortarOptions( coupling_scheme_id, 30., contact_response, damping_ratio,
                                         max_penetration_fraction );
    if ( contact_response == tribol::PROJECTION_RESPONSE_COMPLIANT ) {
      tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, 10., 10. );
      tribol::setMfemKinematicPenaltyScale( coupling_scheme_id, 1., 1. );
    }
  }
  tribol::updateMfemElemThickness( coupling_scheme_id );
  tribol::enableTimestepVote( coupling_scheme_id, false );
  tribol::updateMfemParallelDecomposition( 1, true );

  double timestep_vote = dt;
  const int return_code = tribol::update( 0, 0., dt, timestep_vote );
  mfem::Vector correction( coords.Size() );
  correction = 0.;
  if ( return_code == 0 ) {
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
                                 tribol::getProjectionOperatorJacobiContraction( coupling_scheme_id ) };
  tribol::finalize();
  return result;
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

TEST( MfemImpulseProjection, RedundantConstraintsAcceptPrimalSafeIterate )
{
  const ProjectionResult result = RunProjectionCase( 1.005, -10., 1.e-3, 250, 0., 1.e-6, 8, 10, 1.e-30 );
  EXPECT_EQ( result.return_code, 0 );
  EXPECT_TRUE( result.converged );
  EXPECT_FALSE( result.complementarity_converged );
  EXPECT_GT( result.constraints, 20 );
  EXPECT_EQ( result.iterations, 250 );
  EXPECT_LE( result.final_primal_residual, result.primal_tolerance );
  EXPECT_GT( result.correction_norm, 0. );
  EXPECT_NEAR( result.impulse_force_error, 0., 1.e-11 );
  EXPECT_LE( result.reported_energy_change, 1.e-12 );
  EXPECT_NEAR( result.reported_energy_change, result.direct_energy_change, 1.e-10 );
}

TEST( MfemImpulseProjection, RejectedSolveReportsRedundantConstraintSpectrum )
{
  const ProjectionResult result = RunProjectionCase( 1.005, -10., 1.e-3, 1, 0., 1.e-16, 8, 10, 1.e-30 );
  EXPECT_NE( result.return_code, 0 );
  EXPECT_FALSE( result.converged );
  EXPECT_GT( result.constraints, 20 );
  EXPECT_GT( result.operator_velocity_dofs, 0 );
  EXPECT_GT( result.operator_rank, 0 );
  EXPECT_LT( result.operator_rank, result.constraints );
  EXPECT_GT( result.operator_minimum_eigenvalue, 0. );
  EXPECT_GE( result.operator_maximum_eigenvalue, result.operator_minimum_eigenvalue );
  EXPECT_GT( result.operator_condition_estimate, 1. );
  EXPECT_GT( result.operator_jacobi_contraction, 0. );
  EXPECT_LE( result.operator_jacobi_contraction, 1. );
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
