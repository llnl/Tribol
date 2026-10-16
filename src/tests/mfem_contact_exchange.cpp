// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/config.hpp"

#include <gtest/gtest.h>

#include "mfem.hpp"

#include "shared/mesh/MeshBuilder.hpp"

#include "tribol/mesh/MfemContactData.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"

namespace tribol {
namespace {

void ProjectShiftedCoordinates( mfem::ParFiniteElementSpace& coordinate_space, double vertical_shift,
                                mfem::ParGridFunction& coordinates )
{
  const int space_dimension = coordinate_space.GetParMesh()->SpaceDimension();
  mfem::VectorFunctionCoefficient coordinate_coefficient(
      space_dimension, [vertical_shift]( const mfem::Vector& point, mfem::Vector& value ) {
        value = point;
        value[1] += vertical_shift;
      } );
  coordinates.ProjectCoefficient( coordinate_coefficient );
}

int GlobalPairCount( const MfemContactData& contact_data )
{
  const int local_pair_count = static_cast<int>( contact_data.PairSurface1Elements().size() );
  int global_pair_count = 0;
  MPI_Allreduce( &local_pair_count, &global_pair_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  return global_pair_count;
}

// Two coincident, distinct parent boundaries must produce the same global
// candidate count under every placement policy, with no work on excluded ranks.
TEST( MfemContactExchange, placement_policies_assign_each_candidate_once )
{
  mfem::Mesh serial_lower = mfem::Mesh::MakeCartesian2D( 6, 2, mfem::Element::QUADRILATERAL );
  mfem::Mesh serial_upper = mfem::Mesh::MakeCartesian2D( 6, 2, mfem::Element::QUADRILATERAL );
  mfem::ParMesh lower_mesh( MPI_COMM_WORLD, serial_lower );
  mfem::ParMesh upper_mesh( MPI_COMM_WORLD, serial_upper );
  mfem::H1_FECollection coordinate_collection( 2, 2 );
  mfem::ParFiniteElementSpace lower_space( &lower_mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParFiniteElementSpace upper_space( &upper_mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParGridFunction lower_coordinates( &lower_space );
  mfem::ParGridFunction upper_coordinates( &upper_space );
  ProjectShiftedCoordinates( lower_space, 0.0, lower_coordinates );
  ProjectShiftedCoordinates( upper_space, 1.0, upper_coordinates );

  constexpr std::array<MfemContactPlacement, 4> placements{
      MfemContactPlacement::AUTO, MfemContactPlacement::ANCHOR_OWNER, MfemContactPlacement::BALANCED_ANCHOR,
      MfemContactPlacement::SPATIAL_PARTITION };
  int reference_pair_count = -1;
  for ( const auto placement : placements ) {
    MfemContactOptions options;
    options.placement = placement;
    options.evaluation_rank_count = 2;
    options.refinement_factor = 2;
    options.motion_skin = 0.1;
    MfemContactData contact_data( MfemContactSurface( 60, lower_mesh, lower_coordinates, { 3 } ),
                                  MfemContactSurface( 61, upper_mesh, upper_coordinates, { 1 } ), options,
                                  COMMON_PLANE );
    contact_data.Update( 1.0e-8 );
    const int pair_count = GlobalPairCount( contact_data );
    EXPECT_GT( pair_count, 0 );
    if ( reference_pair_count < 0 ) {
      reference_pair_count = pair_count;
    } else {
      EXPECT_EQ( pair_count, reference_pair_count );
    }

    int rank = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank >= std::min( 2, contact_data.Options().evaluation_rank_count ) ) {
      EXPECT_TRUE( contact_data.PairSurface1Elements().empty() );
    }
  }
}

// Constant velocity and element data plus shifted reference coordinates have
// analytic patch values even when their parent H1 orders differ.
TEST( MfemContactExchange, optional_fields_can_use_independent_finite_element_spaces )
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D( 3, 2, mfem::Element::QUADRILATERAL );
  mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mfem::H1_FECollection coordinate_collection( 2, 2 );
  mfem::H1_FECollection velocity_collection( 1, 2 );
  mfem::H1_FECollection reference_collection( 3, 2 );
  mfem::ParFiniteElementSpace coordinate_space( &mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParFiniteElementSpace velocity_space( &mesh, &velocity_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParFiniteElementSpace reference_space( &mesh, &reference_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParGridFunction coordinates( &coordinate_space );
  mfem::ParGridFunction velocity( &velocity_space );
  mfem::ParGridFunction reference_coordinates( &reference_space );
  mfem::QuadratureSpace quadrature_space( &mesh, 0 );
  mfem::QuadratureFunction thickness( quadrature_space );
  mfem::QuadratureFunction modulus( quadrature_space );
  ProjectShiftedCoordinates( coordinate_space, 0.0, coordinates );
  ProjectShiftedCoordinates( reference_space, -0.25, reference_coordinates );
  mfem::Vector velocity_value{ 2.0, -3.0 };
  mfem::VectorConstantCoefficient velocity_coefficient( velocity_value );
  velocity.ProjectCoefficient( velocity_coefficient );
  thickness = 2.5;
  modulus = 7.0;

  auto first_surface = MfemContactSurface( 64, mesh, coordinates, { 3 } )
                           .SetVelocity( velocity )
                           .SetReferenceCoordinates( reference_coordinates )
                           .SetElementThickness( thickness )
                           .SetMaterialModulus( modulus );
  auto second_surface = MfemContactSurface( 65, mesh, coordinates, { 1 } )
                            .SetVelocity( velocity )
                            .SetReferenceCoordinates( reference_coordinates )
                            .SetElementThickness( thickness )
                            .SetMaterialModulus( modulus );
  MfemContactOptions options;
  options.refinement_factor = 2;
  options.motion_skin = 0.1;
  MfemContactData contact_data( std::move( first_surface ), std::move( second_surface ), options, COMMON_PLANE );
  contact_data.Update( 1.0 );

  for ( int surface = 0; surface < 2; ++surface ) {
    const auto& patch = contact_data.SurfacePatch( surface );
    ASSERT_TRUE( patch.has_velocity );
    ASSERT_TRUE( patch.has_reference_coordinates );
    for ( int node = 0; node < patch.NumNodes(); ++node ) {
      EXPECT_NEAR( patch.velocity[0][node], 2.0, 1.0e-13 );
      EXPECT_NEAR( patch.velocity[1][node], -3.0, 1.0e-13 );
      EXPECT_NEAR( patch.reference_coordinates[1][node], patch.coordinates[1][node] - 0.25, 1.0e-13 );
    }
    ASSERT_EQ( patch.element_thickness.Size(), patch.NumElements() );
    ASSERT_EQ( patch.material_modulus.Size(), patch.NumElements() );
    for ( int element = 0; element < patch.NumElements(); ++element ) {
      EXPECT_DOUBLE_EQ( patch.element_thickness[element], 2.5 );
      EXPECT_DOUBLE_EQ( patch.material_modulus[element], 7.0 );
    }
  }
}

// Eight collinear boundary segments have exactly seven adjacent candidates.
// Canonical self-contact ordering must remove self and reversed duplicates.
TEST( MfemContactExchange, self_contact_rejects_self_and_reversed_pairs )
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D( 8, 2, mfem::Element::QUADRILATERAL );
  mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mfem::H1_FECollection coordinate_collection( 1, 2 );
  mfem::ParFiniteElementSpace coordinate_space( &mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParGridFunction coordinates( &coordinate_space );
  ProjectShiftedCoordinates( coordinate_space, 0.0, coordinates );

  MfemContactOptions options;
  options.placement = MfemContactPlacement::AUTO;
  options.refinement_factor = 1;
  options.motion_skin = 0.1;
  MfemContactData contact_data( MfemContactSurface( 62, mesh, coordinates, { 3 } ),
                                MfemContactSurface( 63, mesh, coordinates, { 3 } ), options, COMMON_PLANE );
  contact_data.Update( 1.0e-8 );
  ASSERT_EQ( contact_data.PairSurface1Elements().size(), contact_data.PairSurface2Elements().size() );
  EXPECT_EQ( GlobalPairCount( contact_data ), 7 );
  const auto& first_patch = contact_data.SurfacePatch( 0 );
  const auto& second_patch = contact_data.SurfacePatch( 1 );
  for ( int pair = 0; pair < static_cast<int>( contact_data.PairSurface1Elements().size() ); ++pair ) {
    const int first_element = contact_data.PairSurface1Elements()[pair];
    const int second_element = contact_data.PairSurface2Elements()[pair];
    double squared_center_distance = 0.0;
    for ( int dimension = 0; dimension < first_patch.space_dimension; ++dimension ) {
      const auto element_center = []( const MfemContactPatchView& patch, int element, int component ) {
        double center = 0.0;
        for ( int vertex = 0; vertex < patch.VerticesPerElement(); ++vertex ) {
          const int node = patch.connectivity[element * patch.VerticesPerElement() + vertex];
          center += patch.coordinates[component][node];
        }
        return center / patch.VerticesPerElement();
      };
      const double difference = element_center( first_patch, first_element, dimension ) -
                                element_center( second_patch, second_element, dimension );
      squared_center_distance += difference * difference;
    }
    EXPECT_GT( squared_center_distance, 0.0 );
  }
}

// Separate high-order parent meshes exercise sparse face requests and the
// global adjoint identity between patch forces and parent true-DOF residuals.
TEST( MfemContactExchange, distinct_parent_meshes_preserve_work_and_request_only_contact_faces )
{
  mfem::Mesh serial_lower = mfem::Mesh::MakeCartesian2D( 4, 2, mfem::Element::QUADRILATERAL );
  mfem::Mesh serial_upper = mfem::Mesh::MakeCartesian2D( 4, 2, mfem::Element::QUADRILATERAL );
  mfem::ParMesh lower_mesh( MPI_COMM_WORLD, serial_lower );
  mfem::ParMesh upper_mesh( MPI_COMM_WORLD, serial_upper );
  mfem::H1_FECollection coordinate_collection( 3, 2 );
  mfem::ParFiniteElementSpace lower_space( &lower_mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParFiniteElementSpace upper_space( &upper_mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParGridFunction lower_coordinates( &lower_space );
  mfem::ParGridFunction upper_coordinates( &upper_space );
  ProjectShiftedCoordinates( lower_space, 0.0, lower_coordinates );
  ProjectShiftedCoordinates( upper_space, 1.0, upper_coordinates );

  // Cartesian meshes label the top and bottom boundaries 3 and 1. Selecting
  // only those opposing faces verifies that volume and unrelated boundary
  // elements never enter the exchange plan.
  MfemContactSurface lower_surface( 10, lower_mesh, lower_coordinates, { 3 } );
  MfemContactSurface upper_surface( 11, upper_mesh, upper_coordinates, { 1 } );
  MfemContactOptions options;
  options.placement = MfemContactPlacement::SPATIAL_PARTITION;
  options.refinement_factor = 3;
  options.motion_skin = 0.125;
  MfemContactData contact_data( std::move( lower_surface ), std::move( upper_surface ), options, COMMON_PLANE );

  contact_data.Update( 1.0e-8 );
  EXPECT_EQ( contact_data.PairSurface1Elements().size(), contact_data.PairSurface2Elements().size() );
  int local_pair_count = static_cast<int>( contact_data.PairSurface1Elements().size() );
  int global_pair_count = 0;
  MPI_Allreduce( &local_pair_count, &global_pair_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( global_pair_count, 0 );

  const auto sent_faces = contact_data.SentFaceCounts();
  int local_sent_faces = sent_faces[0] + sent_faces[1];
  int global_sent_faces = 0;
  MPI_Allreduce( &local_sent_faces, &global_sent_faces, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( global_sent_faces, 0 );
  EXPECT_LE( global_sent_faces, lower_mesh.GetGlobalNE() + upper_mesh.GetGlobalNE() );

  double local_patch_work = 0.0;
  for ( int surface = 0; surface < 2; ++surface ) {
    auto& patch = contact_data.SurfacePatch( surface );
    for ( int dimension = 0; dimension < patch.space_dimension; ++dimension ) {
      for ( int node = 0; node < patch.NumNodes(); ++node ) {
        patch.response[dimension][node] = 0.125 * ( 1 + surface + dimension ) + 0.01 * node;
      }
      local_patch_work += patch.coordinates[dimension] * patch.response[dimension];
    }
  }
  contact_data.CollectResiduals();

  // The patch forces are independent contributions. H^T must return them
  // additively, so their work equals the parent true-DOF residual work.
  mfem::Vector lower_true_coordinates;
  mfem::Vector upper_true_coordinates;
  lower_coordinates.GetTrueDofs( lower_true_coordinates );
  upper_coordinates.GetTrueDofs( upper_true_coordinates );
  const auto& contact_residuals = contact_data.ContactResiduals();
  ASSERT_EQ( contact_residuals.size(), 2 );
  double local_parent_work =
      lower_true_coordinates * contact_residuals[0].true_dof_values +
      upper_true_coordinates * contact_residuals[1].true_dof_values;
  double global_patch_work = 0.0;
  double global_parent_work = 0.0;
  MPI_Allreduce( &local_patch_work, &global_patch_work, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_parent_work, &global_parent_work, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( global_patch_work, global_parent_work, 1.0e-11 * std::max( 1.0, std::abs( global_patch_work ) ) );

  // A skin remains valid below the threshold and rebuilds exactly when the
  // sum of surface motions reaches it.
  ProjectShiftedCoordinates( upper_space, 1.0625, upper_coordinates );
  EXPECT_FALSE( contact_data.Update( 1.0e-8 ) );
  ProjectShiftedCoordinates( upper_space, 1.125, upper_coordinates );
  EXPECT_TRUE( contact_data.Update( 1.0e-8 ) );
}

// The motion check must observe every registered face, including faces that
// were outside the candidate patch at the previous search.
TEST( MfemContactExchange, uncached_faces_moving_into_contact_rebuild_the_patch )
{
  mfem::Mesh serial_lower = mfem::Mesh::MakeCartesian2D( 4, 2, mfem::Element::QUADRILATERAL );
  mfem::Mesh serial_upper = mfem::Mesh::MakeCartesian2D( 4, 2, mfem::Element::QUADRILATERAL );
  mfem::ParMesh lower_mesh( MPI_COMM_WORLD, serial_lower );
  mfem::ParMesh upper_mesh( MPI_COMM_WORLD, serial_upper );
  mfem::H1_FECollection coordinate_collection( 2, 2 );
  mfem::ParFiniteElementSpace lower_space( &lower_mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParFiniteElementSpace upper_space( &upper_mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParGridFunction lower_coordinates( &lower_space );
  mfem::ParGridFunction upper_coordinates( &upper_space );
  ProjectShiftedCoordinates( lower_space, 0.0, lower_coordinates );
  ProjectShiftedCoordinates( upper_space, 1.5, upper_coordinates );

  MfemContactOptions options;
  options.refinement_factor = 2;
  options.motion_skin = 0.2;
  MfemContactData contact_data( MfemContactSurface( 70, lower_mesh, lower_coordinates, { 3 } ),
                                MfemContactSurface( 71, upper_mesh, upper_coordinates, { 1 } ), options, COMMON_PLANE );
  EXPECT_TRUE( contact_data.Update( 1.0e-8 ) );
  EXPECT_EQ( GlobalPairCount( contact_data ), 0 );

  ProjectShiftedCoordinates( upper_space, 0.95, upper_coordinates );
  EXPECT_TRUE( contact_data.Update( 1.0e-8 ) );
  EXPECT_GT( GlobalPairCount( contact_data ), 0 );
}

// A parent-space update can change true-DOF counts without replacing the
// caller-owned MFEM objects. Result buffers must follow the rebuilt operator.
TEST( MfemContactExchange, mesh_refinement_resizes_contact_residuals )
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D( 2, 2, mfem::Element::QUADRILATERAL );
  mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mfem::H1_FECollection coordinate_collection( 2, 2 );
  mfem::ParFiniteElementSpace coordinate_space( &mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParGridFunction coordinates( &coordinate_space );
  ProjectShiftedCoordinates( coordinate_space, 0.0, coordinates );

  MfemContactOptions options;
  options.refinement_factor = 2;
  options.motion_skin = 0.2;
  MfemContactData contact_data( MfemContactSurface( 72, mesh, coordinates, { 3 } ),
                                MfemContactSurface( 73, mesh, coordinates, { 1 } ), options, COMMON_PLANE );
  contact_data.Update( 2.0 );
  const int original_true_size = coordinate_space.GetTrueVSize();

  mesh.UniformRefinement();
  coordinate_space.Update();
  coordinates.Update();
  ProjectShiftedCoordinates( coordinate_space, 0.0, coordinates );
  ASSERT_NE( coordinate_space.GetTrueVSize(), original_true_size );
  EXPECT_TRUE( contact_data.Update( 2.0 ) );
  ASSERT_EQ( contact_data.ContactResiduals().size(), 1 );
  EXPECT_EQ( contact_data.ContactResiduals().front().true_dof_values.Size(), coordinate_space.GetTrueVSize() );

  auto& patch = contact_data.SurfacePatch( 0 );
  for ( int dimension = 0; dimension < patch.space_dimension; ++dimension ) {
    patch.response[dimension] = 1.0;
  }
  contact_data.CollectResiduals();
  EXPECT_EQ( contact_data.ContactResiduals().front().true_dof_values.Size(), coordinate_space.GetTrueVSize() );
}

// A public update must gather geometry internally and return one true-DOF
// residual for each distinct parent finite element space.
TEST( MfemContactExchange, public_update_returns_solver_space_grouped_true_dof_residuals )
{
  mfem::Mesh serial_lower = mfem::Mesh::MakeCartesian2D( 3, 2, mfem::Element::QUADRILATERAL );
  mfem::Mesh serial_upper = mfem::Mesh::MakeCartesian2D( 3, 2, mfem::Element::QUADRILATERAL );
  mfem::ParMesh lower_mesh( MPI_COMM_WORLD, serial_lower );
  mfem::ParMesh upper_mesh( MPI_COMM_WORLD, serial_upper );
  mfem::H1_FECollection coordinate_collection( 2, 2 );
  mfem::ParFiniteElementSpace lower_space( &lower_mesh, &coordinate_collection, 2 );
  mfem::ParFiniteElementSpace upper_space( &upper_mesh, &coordinate_collection, 2 );
  mfem::ParGridFunction lower_coordinates( &lower_space );
  mfem::ParGridFunction upper_coordinates( &upper_space );
  ProjectShiftedCoordinates( lower_space, 0.0, lower_coordinates );
  ProjectShiftedCoordinates( upper_space, 0.95, upper_coordinates );

  MfemContactOptions options;
  options.refinement_factor = 2;
  options.motion_skin = 0.25;
  options.evaluation_rank_count = 1;
  registerMfemCouplingScheme( 20, MfemContactSurface( 20, lower_mesh, lower_coordinates, { 3 } ),
                              MfemContactSurface( 21, upper_mesh, upper_coordinates, { 1 } ), SURFACE_TO_SURFACE,
                              NO_SLIDING, COMMON_PLANE, FRICTIONLESS, PENALTY, options );
  setMfemKinematicConstantPenalty( 20, 100.0, 100.0 );
  RealT time_step = 1.0;
  EXPECT_EQ( update( 0, 0.0, time_step ), 0 );

  const auto& contact_residuals = getMfemContactResiduals( 20 );
  ASSERT_EQ( contact_residuals.size(), 2 );
  double local_residual_norm =
      contact_residuals[0].true_dof_values.Norml2() + contact_residuals[1].true_dof_values.Norml2();
  double global_residual_norm = 0.0;
  MPI_Allreduce( &local_residual_norm, &global_residual_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( global_residual_norm, 0.0 );

  // Diagnostic meshes are created only when requested; normal updates retain
  // the mesh-free patch arrays and cached operators.
  auto first_patch_mesh = materializeMfemContactPatch( 20, 0 );
  int local_materialized_elements = first_patch_mesh->GetNE();
  int global_materialized_elements = 0;
  MPI_Allreduce( &local_materialized_elements, &global_materialized_elements, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( global_materialized_elements, 0 );
  finalize();
}

// Constant multiplier pressure has a known size and exercises true-DOF gather
// and gap transpose without exposing a local-DOF GridFunction.
TEST( MfemContactExchange, multiplier_pressure_and_gap_use_true_dof_vectors )
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D( 3, 3, mfem::Element::QUADRILATERAL );
  mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mfem::H1_FECollection coordinate_collection( 2, 2 );
  mfem::ParFiniteElementSpace coordinate_space( &mesh, &coordinate_collection, 2 );
  mfem::ParGridFunction coordinates( &coordinate_space );
  ProjectShiftedCoordinates( coordinate_space, 0.0, coordinates );

  MfemContactOptions options;
  options.refinement_factor = 2;
  options.motion_skin = 0.25;
  MfemContactData contact_data( MfemContactSurface( 30, mesh, coordinates, { 3 } ),
                                MfemContactSurface( 31, mesh, coordinates, { 1 } ), options, SINGLE_MORTAR,
                                LAGRANGE_MULTIPLIER );
  contact_data.Update( 2.0 );
  auto* multiplier = contact_data.MultiplierSpace();
  ASSERT_NE( multiplier, nullptr );
  for ( int true_dof = 0; true_dof < multiplier->Pressure().Size(); ++true_dof ) {
    multiplier->Pressure()[true_dof] = 1.0 + 0.125 * true_dof;
  }
  contact_data.Update( 2.0 );
  for ( int node = 0; node < multiplier->PatchGap().Size(); ++node ) {
    multiplier->PatchGap()[node] = 0.25 + 0.03125 * node;
  }
  const double local_patch_work = multiplier->PatchPressure() * multiplier->PatchGap();
  contact_data.CollectResiduals();
  const double local_true_dof_work = multiplier->Pressure() * multiplier->Gap();
  double global_patch_work = 0.0;
  double global_true_dof_work = 0.0;
  MPI_Allreduce( &local_patch_work, &global_patch_work, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_true_dof_work, &global_true_dof_work, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( global_patch_work, global_true_dof_work, 1.0e-12 * std::max( 1.0, std::abs( global_patch_work ) ) );
  ASSERT_EQ( contact_data.ContactResiduals().size(), 2 );
  EXPECT_EQ( contact_data.ContactResiduals().back().space, &multiplier->Space() );
}

// Aligned mortar with Lagrange-multiplier enforcement needs the same
// solver-visible multiplier residual as single mortar.
TEST( MfemContactExchange, aligned_mortar_creates_multiplier_residual )
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D( 2, 2, mfem::Element::QUADRILATERAL );
  mfem::ParMesh mesh( MPI_COMM_WORLD, serial_mesh );
  mfem::H1_FECollection coordinate_collection( 2, 2 );
  mfem::ParFiniteElementSpace coordinate_space( &mesh, &coordinate_collection, 2 );
  mfem::ParGridFunction coordinates( &coordinate_space );
  ProjectShiftedCoordinates( coordinate_space, 0.0, coordinates );

  MfemContactData contact_data( MfemContactSurface( 32, mesh, coordinates, { 3 } ),
                                MfemContactSurface( 33, mesh, coordinates, { 1 } ), {}, ALIGNED_MORTAR,
                                LAGRANGE_MULTIPLIER );
  const auto* multiplier = contact_data.MultiplierSpace();
  ASSERT_NE( multiplier, nullptr );
  ASSERT_EQ( contact_data.ContactResiduals().size(), 2 );
  EXPECT_EQ( contact_data.ContactResiduals().back().space, &multiplier->Space() );
}

// Surfaces sharing one displacement space plus one multiplier space must
// produce a deterministic 2x2 block layout with both coupling blocks present.
TEST( MfemContactExchange, multiplier_jacobian_uses_solver_space_grouped_patch_operators )
{
  mfem::ParMesh mesh = shared::ParMeshBuilder(
      MPI_COMM_WORLD,
      shared::MeshBuilder::Unify(
          { shared::MeshBuilder::CubeMesh( 2, 1, 1 ),
            shared::MeshBuilder::CubeMesh( 2, 1, 1 )
                .translate( { 0.0, 0.0, 0.95 } )
                .updateBdrAttrib( 1, 7 )
                .updateBdrAttrib( 6, 8 ) } ) );
  mfem::H1_FECollection coordinate_collection( 2, 3 );
  mfem::ParFiniteElementSpace coordinate_space( &mesh, &coordinate_collection, 3 );
  mfem::ParGridFunction coordinates( &coordinate_space );
  ProjectShiftedCoordinates( coordinate_space, 0.0, coordinates );

  MfemContactOptions options;
  options.refinement_factor = 2;
  options.motion_skin = 0.25;
  options.execution_mode = ExecutionMode::Sequential;
  registerMfemCouplingScheme( 40, MfemContactSurface( 40, mesh, coordinates, { 7 } ),
                              MfemContactSurface( 41, mesh, coordinates, { 6 } ), SURFACE_TO_SURFACE, NO_SLIDING,
                              SINGLE_MORTAR, FRICTIONLESS, LAGRANGE_MULTIPLIER, options );
  setResidualGap( 40, 2.0 );
  setLagrangeMultiplierOptions( 40, ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN );
  getMfemMultiplierPressure( 40 ) = 1.0;

  RealT time_step = 1.0;
  EXPECT_EQ( update( 0, 0.0, time_step ), 0 );
  auto jacobian = getMfemContactJacobian( 40 );

  // Both physical surfaces reference one parent FE space, so their
  // contributions share residual block zero and the multiplier remains last.
  ASSERT_EQ( getMfemContactResiduals( 40 ).size(), 2 );
  EXPECT_EQ( jacobian->NumRowBlocks(), 2 );
  EXPECT_EQ( jacobian->NumColBlocks(), 2 );
  EXPECT_FALSE( jacobian->IsZeroBlock( 0, 1 ) );
  EXPECT_FALSE( jacobian->IsZeroBlock( 1, 0 ) );
  finalize();
}

#ifdef TRIBOL_USE_ENZYME
// When a rebuilt patch has no candidates, the formulation must discard the
// previous cycle's element IDs before evaluating forces.
TEST( MfemContactExchange, energy_mortar_clears_pairs_when_contact_disappears )
{
  mfem::Mesh serial_lower = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL );
  mfem::Mesh serial_upper = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL );
  mfem::ParMesh lower_mesh( MPI_COMM_WORLD, serial_lower );
  mfem::ParMesh upper_mesh( MPI_COMM_WORLD, serial_upper );
  mfem::H1_FECollection coordinate_collection( 1, 2 );
  mfem::ParFiniteElementSpace lower_space( &lower_mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParFiniteElementSpace upper_space( &upper_mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParGridFunction lower_coordinates( &lower_space );
  mfem::ParGridFunction upper_coordinates( &upper_space );
  ProjectShiftedCoordinates( lower_space, 0.0, lower_coordinates );
  ProjectShiftedCoordinates( upper_space, 0.95, upper_coordinates );

  MfemContactOptions options;
  options.execution_mode = ExecutionMode::Sequential;
  options.refinement_factor = 1;
  options.motion_skin = 0.2;
  registerMfemCouplingScheme( 51, MfemContactSurface( 52, lower_mesh, lower_coordinates, { 3 } ),
                              MfemContactSurface( 53, upper_mesh, upper_coordinates, { 1 } ), SURFACE_TO_SURFACE,
                              NO_SLIDING, ENERGY_MORTAR, FRICTIONLESS, PENALTY, options );
  setMfemKinematicConstantPenalty( 51, 100.0, 100.0 );

  RealT time_step = 1.0;
  EXPECT_EQ( update( 0, 0.0, time_step ), 0 );
  double local_contact_norm = 0.0;
  for ( const auto& contact_residual : getMfemContactResiduals( 51 ) ) {
    local_contact_norm += contact_residual.true_dof_values.Norml2();
  }
  double global_contact_norm = 0.0;
  MPI_Allreduce( &local_contact_norm, &global_contact_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( global_contact_norm, 0.0 );

  ProjectShiftedCoordinates( upper_space, 1.5, upper_coordinates );
  EXPECT_EQ( update( 1, 0.0, time_step ), 0 );
  double local_separated_norm = 0.0;
  for ( const auto& contact_residual : getMfemContactResiduals( 51 ) ) {
    local_separated_norm += contact_residual.true_dof_values.Norml2();
  }
  double global_separated_norm = 0.0;
  MPI_Allreduce( &local_separated_norm, &global_separated_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_DOUBLE_EQ( global_separated_norm, 0.0 );
  finalize();
}

// Central finite differences of the complete parent true-DOF residual verify
// the grouped H^T A H tangent, including both mortar/nonmortar patch blocks.
TEST( MfemContactExchange, energy_mortar_uses_patch_residuals_and_grouped_jacobian )
{
  mfem::ParMesh mesh = shared::ParMeshBuilder(
      MPI_COMM_WORLD,
      shared::MeshBuilder::Unify(
          { shared::MeshBuilder::SquareMesh( 1, 1 ).updateBdrAttrib( 3, 3 ),
            shared::MeshBuilder::SquareMesh( 1, 1 ).translate( { 0.0, 0.95 } ).updateBdrAttrib( 1, 5 ) } ) );
  mfem::H1_FECollection coordinate_collection( 1, 2 );
  mfem::ParFiniteElementSpace coordinate_space( &mesh, &coordinate_collection, 2, mfem::Ordering::byVDIM );
  mfem::ParGridFunction coordinates( &coordinate_space );
  mesh.GetNodes( coordinates );

  MfemContactOptions options;
  options.execution_mode = ExecutionMode::Sequential;
  options.refinement_factor = 1;
  options.motion_skin = 0.25;
  registerMfemCouplingScheme( 50, MfemContactSurface( 50, mesh, coordinates, { 5 } ),
                              MfemContactSurface( 51, mesh, coordinates, { 3 } ), SURFACE_TO_SURFACE, NO_SLIDING,
                              ENERGY_MORTAR, FRICTIONLESS, PENALTY, options );
  setMfemKinematicConstantPenalty( 50, 100.0, 100.0 );

  RealT time_step = 1.0;
  EXPECT_EQ( update( 0, 0.0, time_step ), 0 );
  const auto& contact_residuals = getMfemContactResiduals( 50 );
  ASSERT_EQ( contact_residuals.size(), 1 );
  double local_residual_norm = contact_residuals.front().true_dof_values.Norml2();
  double global_residual_norm = 0.0;
  MPI_Allreduce( &local_residual_norm, &global_residual_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( global_residual_norm, 0.0 );

  auto jacobian = getMfemContactJacobian( 50 );
  ASSERT_EQ( jacobian->NumRowBlocks(), 1 );
  EXPECT_FALSE( jacobian->IsZeroBlock( 0, 0 ) );

  mfem::Vector base_coordinates;
  coordinates.GetTrueDofs( base_coordinates );
  mfem::Vector direction( base_coordinates.Size() );
  for ( int true_dof = 0; true_dof < direction.Size(); ++true_dof ) {
    direction[true_dof] = 0.25 + 0.03125 * ( true_dof % 7 );
  }
  mfem::Vector jacobian_action( direction.Size() );
  jacobian->Mult( direction, jacobian_action );

  constexpr double perturbation = 1.0e-7;
  mfem::Vector perturbed_coordinates( base_coordinates );
  perturbed_coordinates.Add( perturbation, direction );
  coordinates.SetFromTrueDofs( perturbed_coordinates );
  EXPECT_EQ( update( 1, 0.0, time_step ), 0 );
  mfem::Vector plus_residual( getMfemContactResiduals( 50 ).front().true_dof_values );

  perturbed_coordinates = base_coordinates;
  perturbed_coordinates.Add( -perturbation, direction );
  coordinates.SetFromTrueDofs( perturbed_coordinates );
  EXPECT_EQ( update( 2, 0.0, time_step ), 0 );
  mfem::Vector finite_difference( getMfemContactResiduals( 50 ).front().true_dof_values );
  finite_difference *= -1.0;
  finite_difference += plus_residual;
  finite_difference *= 0.5 / perturbation;
  finite_difference -= jacobian_action;

  const double local_difference = finite_difference * finite_difference;
  const double local_action = jacobian_action * jacobian_action;
  plus_residual -= getMfemContactResiduals( 50 ).front().true_dof_values;
  plus_residual *= 0.5 / perturbation;
  const double local_finite_difference = plus_residual * plus_residual;
  const double local_pairing = plus_residual * jacobian_action;
  double global_difference = 0.0;
  double global_action = 0.0;
  double global_finite_difference = 0.0;
  double global_pairing = 0.0;
  MPI_Allreduce( &local_difference, &global_difference, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_action, &global_action, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_finite_difference, &global_finite_difference, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_pairing, &global_pairing, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_LT( std::sqrt( global_difference ), 2.0e-5 * std::max( 1.0, std::sqrt( global_action ) ) )
      << "finite-difference norm=" << std::sqrt( global_finite_difference )
      << ", Jacobian norm=" << std::sqrt( global_action ) << ", pairing=" << global_pairing;

  coordinates.SetFromTrueDofs( base_coordinates );
  finalize();
}
#endif

}  // namespace
}  // namespace tribol

#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
