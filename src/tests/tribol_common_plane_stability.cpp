// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/physics/CommonPlane.hpp"

namespace {

using tribol::IndexT;
using tribol::RealT;

/**
 * @brief Description of one synthetic CommonPlane quadrature row.
 */
struct StabilityRow {
  IndexT contact_pair_id;          ///< Face-pair slot that owns the row
  IndexT first_face_id;            ///< First contact-surface face identifier
  IndexT second_face_id;           ///< Second contact-surface face identifier
  RealT first_basis_values[2];     ///< First-face linear basis values
  RealT second_basis_values[2];    ///< Second-face linear basis values
  RealT integration_weight;        ///< Physical quadrature weight
  RealT penalty_stiffness;         ///< Normal penalty stiffness per unit measure
  RealT rate_penalty_coefficient;  ///< Normal rate coefficient per unit measure
  RealT tangential_coefficient;    ///< Tangential viscous coefficient per unit measure
};

/**
 * @brief Fixture that creates two two-segment contact surfaces.
 *
 * Each surface has two elements sharing node 1. Tests populate the production
 * row batch directly so the stability operator can be compared with small
 * analytic matrices independently of contact search and clipping.
 */
class CommonPlaneStabilityTest : public testing::Test {
 protected:
  /** Number of spatial dimensions in the synthetic contact problem. */
  static constexpr int spatial_dimension = 2;

  /** Number of nodes on each synthetic contact surface. */
  static constexpr int number_of_nodes = 3;

  /** Number of faces on each synthetic contact surface. */
  static constexpr int number_of_faces = 2;

  /** First surface connectivity; the faces share node 1. */
  IndexT first_connectivity_[2 * number_of_faces] = { 0, 1, 1, 2 };

  /** Second surface connectivity; the faces share node 1. */
  IndexT second_connectivity_[2 * number_of_faces] = { 0, 1, 1, 2 };

  /** First surface x coordinates. */
  RealT first_x_[number_of_nodes] = { 0.0, 1.0, 2.0 };

  /** First surface y coordinates. */
  RealT first_y_[number_of_nodes] = { 0.0, 0.0, 0.0 };

  /** Second surface x coordinates. */
  RealT second_x_[number_of_nodes] = { 0.0, 1.0, 2.0 };

  /** Second surface y coordinates. */
  RealT second_y_[number_of_nodes] = { -0.1, -0.1, -0.1 };

  /** Zero x velocity used by both surfaces. */
  RealT zero_velocity_x_[number_of_nodes] = { 0.0, 0.0, 0.0 };

  /** Zero y velocity used by both surfaces. */
  RealT zero_velocity_y_[number_of_nodes] = { 0.0, 0.0, 0.0 };

  /** First surface x response storage. */
  RealT first_response_x_[number_of_nodes] = { 0.0, 0.0, 0.0 };

  /** First surface y response storage. */
  RealT first_response_y_[number_of_nodes] = { 0.0, 0.0, 0.0 };

  /** Second surface x response storage. */
  RealT second_response_x_[number_of_nodes] = { 0.0, 0.0, 0.0 };

  /** Second surface y response storage. */
  RealT second_response_y_[number_of_nodes] = { 0.0, 0.0, 0.0 };

  /** Unit inverse masses for unconstrained vector degrees of freedom. */
  RealT unit_inverse_mass_[number_of_nodes] = { 1.0, 1.0, 1.0 };

  /** Zero inverse masses for constrained vector degrees of freedom. */
  RealT zero_inverse_mass_[number_of_nodes] = { 0.0, 0.0, 0.0 };

  /** Per-face viscous coefficients used when tangential damping is enabled. */
  RealT viscous_coefficients_[number_of_faces] = { 1.0, 1.0 };

  /** Register the two meshes and their force and velocity fields. */
  void SetUp() override
  {
    tribol::registerMesh( 0, number_of_faces, number_of_nodes, first_connectivity_, tribol::LINEAR_EDGE, first_x_,
                          first_y_, nullptr, tribol::MemorySpace::Host );
    tribol::registerMesh( 1, number_of_faces, number_of_nodes, second_connectivity_, tribol::LINEAR_EDGE, second_x_,
                          second_y_, nullptr, tribol::MemorySpace::Host );
    tribol::registerNodalResponse( 0, first_response_x_, first_response_y_ );
    tribol::registerNodalResponse( 1, second_response_x_, second_response_y_ );
    tribol::registerNodalVelocities( 0, zero_velocity_x_, zero_velocity_y_ );
    tribol::registerNodalVelocities( 1, zero_velocity_x_, zero_velocity_y_ );
    tribol::setKinematicConstantPenalty( 0, 1.0 );
    tribol::setKinematicConstantPenalty( 1, 1.0 );
  }

  /** Clear all globally registered Tribol data after each test. */
  void TearDown() override { tribol::finalize(); }

  /**
   * @brief Register inverse masses used by a normal-contact-only test.
   *
   * Only y motion contributes because the synthetic row normal is `(0,1)`.
   */
  void RegisterNormalInverseMass()
  {
    tribol::registerNodalInverseMass( 0, zero_inverse_mass_, unit_inverse_mass_ );
    tribol::registerNodalInverseMass( 1, zero_inverse_mass_, unit_inverse_mass_ );
  }

  /** Register unit inverse mass for both vector components. */
  void RegisterVectorInverseMass()
  {
    tribol::registerNodalInverseMass( 0, unit_inverse_mass_, unit_inverse_mass_ );
    tribol::registerNodalInverseMass( 1, unit_inverse_mass_, unit_inverse_mass_ );
  }

  /**
   * @brief Create and initialize a CommonPlane penalty coupling scheme.
   *
   * @param constraint_type Normal penalty terms represented by the rows
   * @param contact_model Contact model represented by the rows
   * @return Initialized coupling scheme
   */
  tribol::CouplingScheme& CreateCouplingScheme( tribol::PenaltyConstraintType constraint_type,
                                                tribol::ContactModel contact_model = tribol::FRICTIONLESS )
  {
    constexpr IndexT coupling_scheme_id = 0;
    tribol::registerCouplingScheme( coupling_scheme_id, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                                    tribol::COMMON_PLANE, contact_model, tribol::PENALTY, tribol::BINNING_GRID,
                                    tribol::ExecutionMode::Sequential );
    tribol::setPenaltyOptions(
        coupling_scheme_id, constraint_type, tribol::KINEMATIC_CONSTANT,
        constraint_type == tribol::KINEMATIC_AND_RATE ? tribol::RATE_CONSTANT : tribol::NO_RATE_PENALTY );
    if ( constraint_type == tribol::KINEMATIC_AND_RATE ) {
      tribol::setRateConstantPenalty( 0, 1.0 );
      tribol::setRateConstantPenalty( 1, 1.0 );
    }
    if ( contact_model == tribol::VISCOUS_TANGENTIAL ) {
      tribol::registerRealElementField( 0, tribol::VISCOUS_DAMPING_COEFF, viscous_coefficients_ );
      tribol::registerRealElementField( 1, tribol::VISCOUS_DAMPING_COEFF, viscous_coefficients_ );
    }
    tribol::setExplicitIntegratorStabilityFactor( coupling_scheme_id, 2.0 );
    tribol::CouplingScheme& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
    EXPECT_TRUE( coupling_scheme.init() );
    return coupling_scheme;
  }

  /**
   * @brief Populate the production row batch from compact synthetic row descriptions.
   *
   * @param coupling_scheme Coupling scheme that owns the row storage
   * @param row_descriptions Rows to place in their deterministic pair slots
   * @param number_of_rows Number of row descriptions
   */
  void SetRows( tribol::CouplingScheme& coupling_scheme, const StabilityRow* row_descriptions, int number_of_rows )
  {
    auto* common_plane_data = static_cast<tribol::CommonPlaneContactData*>( coupling_scheme.getMethodData() );
    common_plane_data->resize( number_of_rows, spatial_dimension, coupling_scheme.getAllocatorId() );
    tribol::CommonPlaneContactData::Viewer rows = common_plane_data->getView();

    for ( int row_index = 0; row_index < number_of_rows; ++row_index ) {
      const StabilityRow& description = row_descriptions[row_index];
      const IndexT row_id = rows.pairRowOffset( description.contact_pair_id );
      rows.pair_row_counts[description.contact_pair_id] = 1;
      rows.row_is_valid[row_id] = 1;
      rows.row_is_active[row_id] = 1;
      rows.contact_pair_ids[row_id] = description.contact_pair_id;
      rows.first_face_ids[row_id] = description.first_face_id;
      rows.second_face_ids[row_id] = description.second_face_id;
      rows.first_basis_counts[row_id] = 2;
      rows.second_basis_counts[row_id] = 2;
      rows.row_uses_parent_fields[row_id] = 0;
      rows.normals( row_id, 0 ) = 0.0;
      rows.normals( row_id, 1 ) = 1.0;
      for ( int basis_index = 0; basis_index < 2; ++basis_index ) {
        rows.first_basis_values( row_id, basis_index ) = description.first_basis_values[basis_index];
        rows.second_basis_values( row_id, basis_index ) = description.second_basis_values[basis_index];
      }
      rows.integration_weights[row_id] = description.integration_weight;
      rows.penalty_stiffnesses[row_id] = description.penalty_stiffness;
      rows.rate_penalty_coefficients[row_id] = description.rate_penalty_coefficient;
      rows.tangential_viscous_coefficients[row_id] = description.tangential_coefficient;
      rows.normal_velocity_gaps[row_id] = -1.0;
    }
  }
};

/** Verify the row-sum bound and the central-difference stability threshold for one row. */
TEST_F( CommonPlaneStabilityTest, ExactTwoBodyFrequencyAndCentralDifferenceLimit )
{
  // A single row with one active node on each body gives the mass-normalized
  // matrix 8*[1,-1]^T*[1,-1]. Its exact largest eigenvalue and row-sum bound
  // are both 16, so the central-difference factor 2 gives dt=0.5.
  RegisterNormalInverseMass();
  tribol::CouplingScheme& coupling_scheme = CreateCouplingScheme( tribol::KINEMATIC );
  const StabilityRow row = { 0, 0, 0, { 1.0, 0.0 }, { 1.0, 0.0 }, 1.0, 8.0, 0.0, 0.0 };
  SetRows( coupling_scheme, &row, 1 );

  RealT timestep = 10.0;
  ASSERT_EQ( tribol::ComputeCommonPlanePenaltyStabilityTimeStep( &coupling_scheme, timestep ), 0 );
  EXPECT_DOUBLE_EQ( coupling_scheme.getExplicitPenaltyStiffnessBound(), 16.0 );
  EXPECT_DOUBLE_EQ( coupling_scheme.getExplicitPenaltyDampingBound(), 0.0 );
  EXPECT_DOUBLE_EQ( timestep, 0.5 );

  // The undamped central-difference recurrence stays bounded below the vote
  // and grows above it, tying the API factor to a concrete explicit method.
  auto maximum_displacement = []( RealT integration_timestep ) {
    constexpr RealT exact_eigenvalue = 16.0;
    RealT previous_displacement = 1.0;
    RealT current_displacement = 1.0 - 0.5 * exact_eigenvalue * integration_timestep * integration_timestep;
    RealT maximum_absolute_displacement = std::abs( previous_displacement );
    for ( int step = 0; step < 100; ++step ) {
      const RealT next_displacement =
          2.0 * current_displacement - previous_displacement -
          exact_eigenvalue * integration_timestep * integration_timestep * current_displacement;
      maximum_absolute_displacement = std::max( maximum_absolute_displacement, std::abs( next_displacement ) );
      previous_displacement = current_displacement;
      current_displacement = next_displacement;
    }
    return maximum_absolute_displacement;
  };
  EXPECT_LT( maximum_displacement( 0.99 * timestep ), 1.01 );
  EXPECT_GT( maximum_displacement( 1.01 * timestep ), 10.0 );
}

/** Verify that the absolute row-sum bound remains conservative for signed basis values. */
TEST_F( CommonPlaneStabilityTest, SignedBasisBoundContainsExactSpectrum )
{
  // This extrapolatory row represents signed higher-order trace values. For
  // unit inverse masses, its exact nonzero eigenvalue is alpha*c^T*c=8.25;
  // the absolute row-sum bound is alpha*max(|c|)*sum(|c|)=10.5.
  RegisterNormalInverseMass();
  tribol::CouplingScheme& coupling_scheme = CreateCouplingScheme( tribol::KINEMATIC );
  const StabilityRow row = { 0, 0, 0, { 1.25, -0.25 }, { -0.5, 1.5 }, 1.0, 2.0, 0.0, 0.0 };
  SetRows( coupling_scheme, &row, 1 );

  RealT timestep = 10.0;
  ASSERT_EQ( tribol::ComputeCommonPlanePenaltyStabilityTimeStep( &coupling_scheme, timestep ), 0 );
  constexpr RealT exact_largest_eigenvalue = 8.25;
  EXPECT_DOUBLE_EQ( coupling_scheme.getExplicitPenaltyStiffnessBound(), 10.5 );
  EXPECT_GE( coupling_scheme.getExplicitPenaltyStiffnessBound(), exact_largest_eigenvalue );
  EXPECT_DOUBLE_EQ( timestep, 2.0 / std::sqrt( 10.5 ) );
}

/** Verify coupled accumulation for shared degrees of freedom and row-order invariance. */
TEST_F( CommonPlaneStabilityTest, AccumulatesSharedDofsAndIsIndependentOfRowOrder )
{
  // Both rows act on node 1 of each surface. Their stiffness scales add before
  // the maximum is taken, producing 2*(2+3)=10 instead of the unsafe
  // separately-voted maximum 2*3=6.
  RegisterNormalInverseMass();
  tribol::CouplingScheme& coupling_scheme = CreateCouplingScheme( tribol::KINEMATIC );
  const StabilityRow forward_rows[2] = { { 0, 0, 0, { 0.0, 1.0 }, { 0.0, 1.0 }, 1.0, 2.0, 0.0, 0.0 },
                                         { 1, 1, 1, { 1.0, 0.0 }, { 1.0, 0.0 }, 1.0, 3.0, 0.0, 0.0 } };
  SetRows( coupling_scheme, forward_rows, 2 );
  RealT forward_timestep = 10.0;
  ASSERT_EQ( tribol::ComputeCommonPlanePenaltyStabilityTimeStep( &coupling_scheme, forward_timestep ), 0 );
  EXPECT_DOUBLE_EQ( coupling_scheme.getExplicitPenaltyStiffnessBound(), 10.0 );

  const StabilityRow reverse_rows[2] = { { 0, 1, 1, { 1.0, 0.0 }, { 1.0, 0.0 }, 1.0, 3.0, 0.0, 0.0 },
                                         { 1, 0, 0, { 0.0, 1.0 }, { 0.0, 1.0 }, 1.0, 2.0, 0.0, 0.0 } };
  SetRows( coupling_scheme, reverse_rows, 2 );
  RealT reverse_timestep = 10.0;
  ASSERT_EQ( tribol::ComputeCommonPlanePenaltyStabilityTimeStep( &coupling_scheme, reverse_timestep ), 0 );
  EXPECT_DOUBLE_EQ( coupling_scheme.getExplicitPenaltyStiffnessBound(), 10.0 );
  EXPECT_DOUBLE_EQ( reverse_timestep, forward_timestep );
}

/** Verify that normal rate penalty and tangential viscous damping both limit the vote. */
TEST_F( CommonPlaneStabilityTest, IncludesNormalAndTangentialDamping )
{
  // Unit normal and tangential inverse masses give an exact row-sum bound of
  // 2*coefficient for each two-body damping block. With coefficients 3 and 4,
  // the combined bound is 8 because the normal and tangential blocks act on
  // orthogonal components and their per-DOF maxima do not add.
  RegisterVectorInverseMass();
  tribol::CouplingScheme& coupling_scheme =
      CreateCouplingScheme( tribol::KINEMATIC_AND_RATE, tribol::VISCOUS_TANGENTIAL );
  const StabilityRow row = { 0, 0, 0, { 1.0, 0.0 }, { 1.0, 0.0 }, 1.0, 8.0, 3.0, 4.0 };
  SetRows( coupling_scheme, &row, 1 );

  RealT timestep = 10.0;
  ASSERT_EQ( tribol::ComputeCommonPlanePenaltyStabilityTimeStep( &coupling_scheme, timestep ), 0 );
  constexpr RealT stiffness_bound = 16.0;
  constexpr RealT damping_bound = 8.0;
  const RealT expected_timestep =
      2.0 / ( std::sqrt( stiffness_bound + 0.25 * damping_bound * damping_bound ) + 0.5 * damping_bound );
  EXPECT_DOUBLE_EQ( coupling_scheme.getExplicitPenaltyStiffnessBound(), stiffness_bound );
  EXPECT_DOUBLE_EQ( coupling_scheme.getExplicitPenaltyDampingBound(), damping_bound );
  EXPECT_DOUBLE_EQ( timestep, expected_timestep );
}

/** Verify that an active penalty row cannot produce a vote without physical inverse mass. */
TEST_F( CommonPlaneStabilityTest, RejectsActiveRowsWithoutInverseMass )
{
  // Returning an optimistic timestep without the application's physical mass
  // would violate the explicit stability contract, so active rows fail.
  tribol::CouplingScheme& coupling_scheme = CreateCouplingScheme( tribol::KINEMATIC );
  const StabilityRow row = { 0, 0, 0, { 1.0, 0.0 }, { 1.0, 0.0 }, 1.0, 8.0, 0.0, 0.0 };
  SetRows( coupling_scheme, &row, 1 );

  RealT timestep = 10.0;
  EXPECT_NE( tribol::ComputeCommonPlanePenaltyStabilityTimeStep( &coupling_scheme, timestep ), 0 );
  EXPECT_LT( timestep, 0.0 );
}

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
