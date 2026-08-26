// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "gtest/gtest.h"

#include "tribol/interface/tribol.hpp"

#include <algorithm>
#include <vector>

namespace tribol {

namespace {

RealT matrixEntry( const CsrMatrix& matrix, IndexT row, IndexT column )
{
  for ( IndexT entry = matrix.rowOffsets()[row]; entry < matrix.rowOffsets()[row + 1]; ++entry ) {
    if ( matrix.columnIndices()[entry] == column ) {
      return matrix.values()[entry];
    }
  }
  return 0.0;
}

}  // namespace

TEST( EnergyMortarCsrMatrixTest, MatrixProductsPreserveCrossRowCouplings )
{
  auto matrix = CsrMatrix::fromEntries(
      2, 3, { { 0, 0, 1.0 }, { 0, 1, 2.0 }, { 1, 1, 3.0 }, { 1, 2, 4.0 } } );
  const auto product = matrix.transpose().multiply( matrix );

  EXPECT_DOUBLE_EQ( matrixEntry( product, 0, 1 ), 2.0 );
  EXPECT_DOUBLE_EQ( matrixEntry( product, 1, 0 ), 2.0 );
  EXPECT_DOUBLE_EQ( matrixEntry( product, 1, 2 ), 12.0 );
  EXPECT_DOUBLE_EQ( matrixEntry( product, 2, 1 ), 12.0 );
  EXPECT_DOUBLE_EQ( matrixEntry( product, 0, 2 ), 0.0 );
}

class NativeEnergyMortarTest : public testing::Test {
 protected:
  void SetUp() override
  {
    registerMesh( mortar_mesh_id, 1, 2, mortar_connectivity, LINEAR_EDGE, mortar_x, mortar_y, nullptr,
                  MemorySpace::Host );
    registerMesh( nonmortar_mesh_id, 1, 2, nonmortar_connectivity, LINEAR_EDGE, nonmortar_x, nonmortar_y, nullptr,
                  MemorySpace::Host );
  }

  void TearDown() override { finalize(); }

  void registerScheme( EnforcementMethod enforcement, EnforcementLocation location )
  {
    registerCouplingScheme( coupling_scheme_id, mortar_mesh_id, nonmortar_mesh_id, SURFACE_TO_SURFACE, NO_SLIDING,
                            ENERGY_MORTAR, FRICTIONLESS, enforcement, BINNING_CARTESIAN_PRODUCT,
                            ExecutionMode::Sequential );
    setEnforcementLocation( coupling_scheme_id, location );
    if ( enforcement == PENALTY ) {
      setPenaltyOptions( coupling_scheme_id, KINEMATIC, KINEMATIC_CONSTANT );
      setKinematicConstantPenalty( mortar_mesh_id, 3.0 );
      setKinematicConstantPenalty( nonmortar_mesh_id, 3.0 );
    }
  }

  static constexpr IndexT coupling_scheme_id = 0;
  static constexpr IndexT mortar_mesh_id = 10;
  static constexpr IndexT nonmortar_mesh_id = 20;
  IndexT mortar_connectivity[2] = { 0, 1 };
  IndexT nonmortar_connectivity[2] = { 1, 0 };
  RealT mortar_x[2] = { 0.2, 0.8 };
  RealT mortar_y[2] = { -0.1, -0.1 };
  RealT nonmortar_x[2] = { 0.0, 1.0 };
  RealT nonmortar_y[2] = { 0.0, 0.0 };
};

TEST_F( NativeEnergyMortarTest, OwnsRuleAndResults )
{
  registerScheme( PENALTY, EnforcementLocation::Nodal );
  const auto rule = computeIntegrationRule( coupling_scheme_id );
  ASSERT_EQ( rule.numberOfPairs(), 1u );
  ASSERT_EQ( rule.numberOfPoints(), 3u );

  const auto gaps = computeNodalGaps( coupling_scheme_id, rule );
  const auto forces = computeNodalForces( coupling_scheme_id, rule, gaps );
  EXPECT_EQ( gaps.weighted_gap.size(), 2u );
  EXPECT_EQ( forces.nonmortar_force.size(), 4u );
  EXPECT_EQ( forces.mortar_force.size(), 4u );
  EXPECT_EQ( forces.df_dx.nonmortar_mortar.rows(), 4 );
  EXPECT_EQ( forces.df_dx.nonmortar_mortar.columns(), 4 );
  EXPECT_GT( forces.energy, 0.0 );
  EXPECT_NEAR( forces.nonmortar_force[0] + forces.nonmortar_force[1] + forces.mortar_force[0] +
                   forces.mortar_force[1],
               0.0, 1.0e-12 );
  EXPECT_NEAR( forces.nonmortar_force[2] + forces.nonmortar_force[3] + forces.mortar_force[2] +
                   forces.mortar_force[3],
               0.0, 1.0e-12 );

  const auto saved_gap = gaps.weighted_gap;
  mortar_y[0] = -0.3;
  mortar_y[1] = -0.3;
  const auto reused_gaps = computeNodalGaps( coupling_scheme_id, rule );
  EXPECT_EQ( reused_gaps.weighted_gap, saved_gap );

  const auto refreshed_rule = computeIntegrationRule( coupling_scheme_id );
  const auto refreshed_gaps = computeNodalGaps( coupling_scheme_id, refreshed_rule );
  EXPECT_NE( refreshed_gaps.weighted_gap, saved_gap );
  EXPECT_EQ( gaps.weighted_gap, saved_gap );
}

TEST_F( NativeEnergyMortarTest, UsesCopiedLagrangeMultiplier )
{
  registerScheme( LAGRANGE_MULTIPLIER, EnforcementLocation::Nodal );
  std::vector<RealT> pressure{ 2.0, 3.0 };
  setContactPressure( coupling_scheme_id, pressure );
  pressure.assign( 2, 0.0 );

  const auto rule = computeIntegrationRule( coupling_scheme_id );
  const auto gaps = computeNodalGaps( coupling_scheme_id, rule );
  const auto forces = computeNodalForces( coupling_scheme_id, rule, gaps );
  EXPECT_EQ( forces.pressure, ( std::vector<RealT>{ 2.0, 3.0 } ) );
  EXPECT_TRUE( forces.df_dp.has_value() );
  EXPECT_FALSE( forces.df_dp->nonmortar_dual.empty() );
}

TEST_F( NativeEnergyMortarTest, QuadraturePointResultHasNoNodalGaps )
{
  registerScheme( PENALTY, EnforcementLocation::QuadraturePoint );
  const auto result = computeContactData( coupling_scheme_id );
  EXPECT_FALSE( result.gaps.has_value() );
  EXPECT_GT( result.forces.energy, 0.0 );
}

}  // namespace tribol

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
