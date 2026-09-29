// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortar.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <set>

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

#include "mfem.hpp"

#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

namespace tribol {
namespace {

EnergyMortar3DInput makePlanarInput( int source_corner, double target_z = -0.1 )
{
  EnergyMortar3DInput input;
  input.num_source_nodes = 4;
  input.num_source_faces = 1;
  input.source_corner = source_corner;
  input.source_face_nodes = { 0, 1, 2, 3 };
  input.source_star_connectivity[0] = 0;
  input.source_star_connectivity[1] = 1;
  input.source_star_connectivity[2] = 2;
  input.source_star_connectivity[3] = 3;

  const double source[4][3] = { { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 1.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0 } };
  const double target[4][3] = {
      { 0.15, 0.20, target_z }, { 0.15, 0.90, target_z }, { 0.85, 0.90, target_z }, { 0.85, 0.20, target_z } };
  for ( int node = 0; node < 4; ++node ) {
    for ( int component = 0; component < 3; ++component ) {
      input.coordinates[3 * node + component] = source[node][component];
      input.coordinates[3 * ( 4 + node ) + component] = target[node][component];
    }
  }
  return input;
}

EnergyMortarCalculator makeCalculator( double residual_gap = 0.0 )
{
  ContactParams params;
  params.del = 0.0;
  params.k = 2.0;
  params.N = 3;
  params.enzyme_quadrature = true;
  params.residual_gap = residual_gap;
  params.length_tol_ratio = 1.0e-10;
  return EnergyMortarCalculator( params );
}

TEST( EnergyMortar3D, PartitionOfUnityAndResidualGap )
{
  auto calculator = makeCalculator();
  double energy = 0.0;
  for ( int corner = 0; corner < 4; ++corner ) {
    energy += calculator.compute_penalty_energy_3d( makePlanarInput( corner ) );
  }
  EXPECT_NEAR( energy, 0.5 * 2.0 * 0.1 * 0.1 * 0.7 * 0.7, 1.0e-12 );

  auto residual_calculator = makeCalculator( 0.05 );
  double residual_energy = 0.0;
  for ( int corner = 0; corner < 4; ++corner ) {
    residual_energy += residual_calculator.compute_penalty_energy_3d( makePlanarInput( corner ) );
  }
  EXPECT_NEAR( residual_energy, 0.5 * 2.0 * 0.15 * 0.15 * 0.7 * 0.7, 1.0e-12 );
}

TEST( EnergyMortar3D, EmptyOverlapHasZeroResponse )
{
  auto input = makePlanarInput( 0 );
  for ( int node = 4; node < 8; ++node ) {
    input.coordinates[3 * node] += 2.0;
  }

  const auto result = makeCalculator().compute_penalty_data_3d( input );
  EXPECT_FALSE( result.has_overlap );
  EXPECT_FALSE( result.has_active_qp );
  EXPECT_DOUBLE_EQ( result.energy, 0.0 );
  for ( const auto value : result.force ) {
    EXPECT_DOUBLE_EQ( value, 0.0 );
  }
}

TEST( EnergyMortar3D, ExactlyAlignedEdgesExposeTopologyCusp )
{
  auto calculator = makeCalculator();
  const auto pair_energy = [&calculator]( double target_x_shift ) {
    double energy = 0.0;
    for ( int corner = 0; corner < 4; ++corner ) {
      auto input = makePlanarInput( corner );
      const double target[4][2] = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 1.0 }, { 1.0, 0.0 } };
      for ( int node = 0; node < 4; ++node ) {
        input.coordinates[3 * ( 4 + node )] = target[node][0] + target_x_shift;
        input.coordinates[3 * ( 4 + node ) + 1] = target[node][1];
      }
      energy += calculator.compute_penalty_energy_3d( input );
    }
    return energy;
  };

  constexpr double energy_density = 0.5 * 2.0 * 0.1 * 0.1;
  const double aligned_energy = pair_energy( 0.0 );
  EXPECT_NEAR( aligned_energy, energy_density, 1.0e-12 );

  constexpr double shift = 1.0e-6;
  const double left_slope = ( aligned_energy - pair_energy( -shift ) ) / shift;
  const double right_slope = ( pair_energy( shift ) - aligned_energy ) / shift;
  EXPECT_NEAR( left_slope, energy_density, 1.0e-9 );
  EXPECT_NEAR( right_slope, -energy_density, 1.0e-9 );
}

TEST( EnergyMortar3D, GradientHessianAndObjectivity )
{
  auto input = makePlanarInput( 0 );
  const auto calculator = makeCalculator();
  const auto result = calculator.compute_penalty_data_3d( input );
  ASSERT_TRUE( result.has_overlap );
  ASSERT_TRUE( result.has_active_qp );

  const int num_dofs = 3 * ( input.num_source_nodes + EnergyMortar3DInput::target_nodes );
  ASSERT_EQ( result.force.size(), static_cast<std::size_t>( num_dofs ) );
  ASSERT_EQ( result.stiffness.size(), static_cast<std::size_t>( num_dofs * num_dofs ) );

  constexpr double gradient_step = 1.0e-6;
  for ( int dof = 0; dof < num_dofs; ++dof ) {
    auto plus = input;
    auto minus = input;
    plus.coordinates[dof] += gradient_step;
    minus.coordinates[dof] -= gradient_step;
    const double finite_difference =
        ( calculator.compute_penalty_energy_3d( plus ) - calculator.compute_penalty_energy_3d( minus ) ) /
        ( 2.0 * gradient_step );
    EXPECT_NEAR( result.force[dof], finite_difference, 2.0e-7 ) << "gradient dof " << dof;
  }

  for ( int row = 0; row < num_dofs; ++row ) {
    for ( int column = 0; column < num_dofs; ++column ) {
      EXPECT_NEAR( result.stiffness[row * num_dofs + column], result.stiffness[column * num_dofs + row], 2.0e-9 )
          << "Hessian entry (" << row << ", " << column << ")";
    }
  }

  for ( int component = 0; component < 3; ++component ) {
    double resultant = 0.0;
    for ( int node = 0; node < input.num_source_nodes + EnergyMortar3DInput::target_nodes; ++node ) {
      resultant += result.force[3 * node + component];
    }
    EXPECT_NEAR( resultant, 0.0, 2.0e-10 );
  }

  std::array<double, 3> rotational_derivative{};
  for ( int node = 0; node < input.num_source_nodes + EnergyMortar3DInput::target_nodes; ++node ) {
    const double x = input.coordinates[3 * node];
    const double y = input.coordinates[3 * node + 1];
    const double z = input.coordinates[3 * node + 2];
    const double force_x = result.force[3 * node];
    const double force_y = result.force[3 * node + 1];
    const double force_z = result.force[3 * node + 2];
    rotational_derivative[0] += y * force_z - z * force_y;
    rotational_derivative[1] += z * force_x - x * force_z;
    rotational_derivative[2] += x * force_y - y * force_x;
  }
  for ( int component = 0; component < 3; ++component ) {
    EXPECT_NEAR( rotational_derivative[component], 0.0, 2.0e-10 );
  }
}

TEST( EnergyMortar3D, WarpedQuadrilateralsRemainFinite )
{
  auto input = makePlanarInput( 2 );
  input.coordinates[3 * 2 + 2] = 0.04;
  input.coordinates[3 * ( 4 + 2 ) + 2] -= 0.03;

  const auto result = makeCalculator().compute_penalty_data_3d( input );
  EXPECT_TRUE( result.has_overlap );
  EXPECT_TRUE( result.has_active_qp );
  EXPECT_TRUE( std::isfinite( result.energy ) );
  for ( const auto value : result.force ) {
    EXPECT_TRUE( std::isfinite( value ) );
  }
  for ( const auto value : result.stiffness ) {
    EXPECT_TRUE( std::isfinite( value ) );
  }
}

TEST( EnergyMortar3D, DifferentiatesAdjacentSourceFaces )
{
  auto input = makePlanarInput( 1 );
  input.num_source_nodes = 6;
  input.num_source_faces = 2;
  input.source_star_connectivity[4] = 1;
  input.source_star_connectivity[5] = 4;
  input.source_star_connectivity[6] = 5;
  input.source_star_connectivity[7] = 2;

  input.coordinates[3 * 4] = 2.0;
  input.coordinates[3 * 4 + 1] = 0.0;
  input.coordinates[3 * 4 + 2] = 0.2;
  input.coordinates[3 * 5] = 2.0;
  input.coordinates[3 * 5 + 1] = 1.0;
  input.coordinates[3 * 5 + 2] = 0.2;

  const auto original_target = makePlanarInput( 1 );
  for ( int node = 0; node < 4; ++node ) {
    for ( int component = 0; component < 3; ++component ) {
      input.coordinates[3 * ( 6 + node ) + component] = original_target.coordinates[3 * ( 4 + node ) + component];
    }
  }

  const auto result = makeCalculator().compute_penalty_data_3d( input );
  ASSERT_TRUE( result.has_active_qp );
  double adjacent_force_norm = 0.0;
  for ( int node : { 4, 5 } ) {
    for ( int component = 0; component < 3; ++component ) {
      const double value = result.force[3 * node + component];
      adjacent_force_norm += value * value;
    }
  }
  EXPECT_GT( adjacent_force_norm, 0.0 );
}

class EnergyMortar3DAdapterTest : public testing::TestWithParam<EnforcementLocation> {
 protected:
  void TearDown() override { tribol::finalize(); }
};

TEST_P( EnergyMortar3DAdapterTest, AssemblesForceAndTangent )
{
  constexpr int coupling_scheme_id = 0;
  constexpr int mortar_mesh_id = 0;
  constexpr int nonmortar_mesh_id = 1;

  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::Unify( {
    shared::MeshBuilder::CubeMesh( 1, 1, 1 ),
    shared::MeshBuilder::CubeMesh( 1, 1, 1 )
      .translate( { 0.1, 0.1, 0.99 } )
      .updateAttrib( 1, 2 )
      .updateBdrAttrib( 1, 7 )
      .updateBdrAttrib( 6, 8 )
  } ) );
  // clang-format on

  auto coordinate_collection = mfem::H1_FECollection( 1, mesh.SpaceDimension() );
  auto coordinate_space =
      mfem::ParFiniteElementSpace( &mesh, &coordinate_collection, mesh.SpaceDimension(), mfem::Ordering::byVDIM );
  auto coordinates = mfem::ParGridFunction( &coordinate_space );
  mesh.GetNodes( coordinates );

  tribol::registerMfemCouplingScheme( coupling_scheme_id, mortar_mesh_id, nonmortar_mesh_id, mesh, coordinates, { 7 },
                                      { 6 }, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::ENERGY_MORTAR,
                                      tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                      tribol::ExecutionMode::Sequential );
  tribol::setEnforcementLocation( coupling_scheme_id, GetParam() );
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, 10.0, 10.0 );
  tribol::setBinningProximityScale( coupling_scheme_id, 2.0 );
  tribol::updateMfemParallelDecomposition( 0, true );

  RealT dt = 1.0;
  EXPECT_EQ( tribol::update( 1, 0.0, dt ), 0 );
  const auto force = tribol::getMfemContactForce( coupling_scheme_id );
  EXPECT_GT( force.Norml2(), 0.0 );
  auto tangent = tribol::getMfemDfDx( coupling_scheme_id );
  ASSERT_NE( tangent, nullptr );
  EXPECT_EQ( tangent->Height(), coordinate_space.GlobalTrueVSize() );
  EXPECT_EQ( tangent->Width(), coordinate_space.GlobalTrueVSize() );

  const auto* formulation = CouplingSchemeManager::getInstance().at( coupling_scheme_id ).getContactFormulation();
  ASSERT_NE( formulation, nullptr );
  EXPECT_GT( formulation->getEnergy(), 0.0 );
}

INSTANTIATE_TEST_SUITE_P( EnforcementLocations, EnergyMortar3DAdapterTest,
                          testing::Values( EnforcementLocation::Nodal, EnforcementLocation::QuadraturePoint ) );

}  // namespace
}  // namespace tribol

#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();
#endif

  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();

  tribol::finalize();
  MPI_Finalize();

  return result;
}
