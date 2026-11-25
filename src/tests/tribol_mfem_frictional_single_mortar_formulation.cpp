// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check MFEM frictional single mortar 3D vector-LM outputs, nodal normals, solver blocks, and Jacobians on one
 * and two MPI ranks, for both linear (hex and tet) and quadratic LOR meshes.
 */
#include <cmath>

#include "ExecutionTest.hpp"
#include "SlicErrorTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

namespace {
using namespace tribol;

constexpr IndexT scheme_id = 0;
constexpr IndexT mesh1_id = 0;
constexpr IndexT mesh2_id = 1;
constexpr int spatial_dim = 3;
constexpr int nonmortar_attribute = 6;
constexpr int mortar_attribute = 7;
constexpr int upper_top_attribute = 8;
constexpr int upper_body_attribute = 2;

/** @brief Register two nonmatching 3D contact surfaces with reference and previous-step coordinates. */
class FrictionalMortarCase {
 public:
  /**
   * @brief Build and register the frictional single mortar test meshes.
   * @param element_type Volume element type for both bodies.
   * @param order Parent coordinate order (1 for linear, 2 for quadratic LOR).
   * @param execution_mode Backend for contact calculations.
   * @param with_previous_coordinates Whether to register previous-step coordinates immediately.
   */
  FrictionalMortarCase( mfem::Element::Type element_type, int order, ExecutionMode execution_mode,
                        bool with_previous_coordinates = true )
      : mesh( makeMesh( element_type, order ) ),
        basis( order, spatial_dim ),
        coordinate_space( &mesh, &basis, spatial_dim, mfem::Ordering::byVDIM ),
        coordinates( &coordinate_space ),
        reference_coordinates( &coordinate_space ),
        previous_coordinates( &coordinate_space )
  {
    mesh.GetNodes( reference_coordinates );
    mfem::VectorFunctionCoefficient prev_deformed( spatial_dim, []( const mfem::Vector& point, mfem::Vector& value ) {
      value = point;
      value[0] -= 0.015 * point[2];
      value[1] += 0.01 * point[0];
      value[2] += 0.03 * point[0] * point[0] + 0.015 * point[0] * point[1];
    } );
    previous_coordinates.ProjectCoefficient( prev_deformed );

    mfem::VectorFunctionCoefficient curr_deformed( spatial_dim, []( const mfem::Vector& point, mfem::Vector& value ) {
      value = point;
      // Add both tangential sliding and curved normal deformation relative to previous_coordinates.
      value[0] += 0.025 * point[2] + 0.01 * point[1];
      value[1] -= 0.015 * point[0];
      value[2] += 0.04 * point[0] * point[0] + 0.02 * point[0] * point[1];
    } );
    coordinates.ProjectCoefficient( curr_deformed );

    registerMfemCouplingScheme( scheme_id, mesh1_id, mesh2_id, mesh, coordinates, { nonmortar_attribute },
                                { mortar_attribute }, SURFACE_TO_SURFACE, NO_CASE, SINGLE_MORTAR, FRICTION_COULOMB,
                                LAGRANGE_MULTIPLIER, BINNING_GRID, execution_mode );
    registerMfemReferenceCoords( scheme_id, reference_coordinates );
    if ( with_previous_coordinates ) {
      registerMfemPreviousCoords( scheme_id, previous_coordinates );
    }
  }

  ~FrictionalMortarCase() { finalize(); }

  /** @brief Vary 3D global Cartesian multipliers along mesh1 while keeping unused mortar-side values zero. */
  void initializeMultipliers() const
  {
    auto& multiplier = getMfemContactPressure( scheme_id );
    auto& fes = *getMfemPressure( scheme_id ).ParFESpace();
    mfem::ParGridFunction input( &fes );
    mfem::VectorFunctionCoefficient value( spatial_dim, []( const mfem::Vector& point, mfem::Vector& out ) {
      out.SetSize( spatial_dim );
      // The original mesh1 surface is at z=1.0; mesh2 starts below it at z=0.97.
      constexpr RealT between_surfaces = 0.985;
      if ( point[2] > between_surfaces ) {
        out[0] = 0.25 - 0.1 * point[1];
        out[1] = -0.2 + 0.15 * point[0];
        out[2] = 0.6 + 0.25 * point[0];
      } else {
        out = 0.0;
      }
    } );
    input.ProjectCoefficient( value );
    input.GetTrueDofs( multiplier );
  }

  /** @brief Evaluate contact on current coordinates. */
  void evaluate() const
  {
    updateMfemParallelDecomposition();
    RealT dt = 1.0;
    ASSERT_EQ( update( 0, 0.0, dt ), 0 );
  }

  /** @brief Construct a displacement perturbation changing overlap, slip, and surface normals. */
  mfem::Vector coordinateDirection()
  {
    mfem::ParGridFunction direction_field( &coordinate_space );
    mfem::VectorFunctionCoefficient direction( spatial_dim, []( const mfem::Vector& point, mfem::Vector& value ) {
      value.SetSize( spatial_dim );
      value[0] = 0.04 + 0.07 * point[1] * point[2];
      value[1] = -0.06 * point[0] + 0.02 * point[2];
      value[2] = 0.08 * point[0] * point[0] - 0.09 * point[1] * point[2];
    } );
    direction_field.ProjectCoefficient( direction );
    mfem::Vector true_direction;
    direction_field.GetTrueDofs( true_direction );
    return true_direction;
  }

  /** @brief Construct a 3D multiplier perturbation on non-mortar dual true DOFs. */
  mfem::Vector multiplierDirection() const
  {
    auto& fes = *getMfemPressure( scheme_id ).ParFESpace();
    mfem::ParGridFunction direction_field( &fes );
    mfem::VectorFunctionCoefficient direction( spatial_dim, []( const mfem::Vector& point, mfem::Vector& value ) {
      value.SetSize( spatial_dim );
      constexpr RealT between_surfaces = 0.985;
      if ( point[2] > between_surfaces ) {
        value[0] = 0.3 + 0.1 * point[0];
        value[1] = -0.25 + 0.2 * point[1];
        value[2] = 0.4 - 0.15 * point[0];
      } else {
        value = 0.0;
      }
    } );
    direction_field.ProjectCoefficient( direction );
    mfem::Vector true_direction;
    direction_field.GetTrueDofs( true_direction );
    return true_direction;
  }

  mfem::ParMesh mesh;                            ///< Parent mesh.
  mfem::H1_FECollection basis;                   ///< Parent coordinate basis.
  mfem::ParFiniteElementSpace coordinate_space;  ///< Parent displacement space.
  mfem::ParGridFunction coordinates;             ///< Current coordinates.
  mfem::ParGridFunction reference_coordinates;   ///< Undeformed coordinates.
  mfem::ParGridFunction previous_coordinates;    ///< Previous converged step coordinates.

 private:
  static mfem::ParMesh makeMesh( mfem::Element::Type element_type, int order )
  {
    auto parent = shared::MeshBuilder::Unify( {
        shared::MeshBuilder::CubeMesh( 3, 2, 1, element_type ),
        shared::MeshBuilder::CubeMesh( 2, 2, 1, element_type )
            .scale( { 0.44, 0.78, 1.0 } )
            .translate( { 0.12, 0.08, 0.97 } )
            .updateBdrAttrib( nonmortar_attribute, upper_top_attribute )
            .updateBdrAttrib( 1, mortar_attribute )
            .updateAttrib( 1, upper_body_attribute ),
    } );
    mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, std::move( parent ) );
    mesh.SetCurvature( order );
    return mesh;
  }
};

RealT differenceNorm( const mfem::Vector& first, const mfem::Vector& second )
{
  first.HostRead();
  second.HostRead();
  mfem::Vector difference( first );
  difference -= second;
  return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, difference, difference ) );
}

RealT parallelNorm( const mfem::Vector& values )
{
  values.HostRead();
  return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, values, values ) );
}

// Verify unit nodal normals on non-mortar dual true DOFs and zero normals on mortar-side placeholder DOFs.
void checkUnitNodalNormals( const mfem::HypreParVector& normals, const mfem::ParFiniteElementSpace& dual_fes )
{
  normals.HostRead();
  int num_unit_nodes = 0;
  for ( int node = 0; node < dual_fes.GetNDofs(); ++node ) {
    int tdof0 = dual_fes.GetLocalTDofNumber( dual_fes.DofToVDof( node, 0 ) );
    if ( tdof0 < 0 ) {
      continue;
    }
    int tdof1 = dual_fes.GetLocalTDofNumber( dual_fes.DofToVDof( node, 1 ) );
    int tdof2 = dual_fes.GetLocalTDofNumber( dual_fes.DofToVDof( node, 2 ) );
    const RealT mag = std::sqrt( normals[tdof0] * normals[tdof0] + normals[tdof1] * normals[tdof1] +
                                 normals[tdof2] * normals[tdof2] );
    if ( mag > 1.0e-12 ) {
      EXPECT_NEAR( mag, 1.0, 1.0e-12 );
      ++num_unit_nodes;
    } else {
      EXPECT_EQ( mag, 0.0 );
    }
  }
  MPI_Allreduce( MPI_IN_PLACE, &num_unit_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( num_unit_nodes, 0 );
}

// Verify frozen-mode outputs, block operator placeholder constraints, and (with Enzyme) exact directional derivatives
// for df_dx, df_dp, weighted_gap_jacobian, and dn_dx on both linear and quadratic LOR meshes.
void checkFrictionalMortarCase( mfem::Element::Type element_type, int order, ExecutionMode execution_mode )
{
  FrictionalMortarCase contact( element_type, order, execution_mode );
  setMfemUseSingleMortarFormulation( scheme_id, true, false );
  contact.initializeMultipliers();
  contact.evaluate();

  const auto& dual_fes = *getMfemPressure( scheme_id ).ParFESpace();
  ASSERT_EQ( dual_fes.GetVDim(), spatial_dim );

  const auto frozen_residual = getMfemContactResidual( scheme_id );
  const auto frozen_gap = getMfemContactGap( scheme_id );
  const auto frozen_normals = getMfemNodalNormals( scheme_id );
  EXPECT_GT( parallelNorm( frozen_residual ), 0.0 );
  EXPECT_GT( parallelNorm( frozen_gap ), 0.0 );
  checkUnitNodalNormals( frozen_normals, dual_fes );

  EXPECT_EQ( getMfemDfDx( scheme_id ), nullptr );
  EXPECT_EQ( getMfemDnDx( scheme_id ), nullptr );
  const auto frozen_df_dp = getMfemDfDp( scheme_id );
  const auto frozen_dg_dx = getMfemWeightedGapJacobian( scheme_id );
  ASSERT_NE( frozen_df_dp, nullptr );
  ASSERT_NE( frozen_dg_dx, nullptr );

  // In frozen mode, df_dp must equal the transpose of weighted_gap_jacobian.
  const auto coord_dir = contact.coordinateDirection();
  const auto mult_dir = contact.multiplierDirection();
  mfem::Vector dg_dx_dir( frozen_dg_dx->Height() );
  frozen_dg_dx->Mult( coord_dir, dg_dx_dir );
  mfem::Vector df_dp_t_dir( frozen_df_dp->Width() );
  frozen_df_dp->MultTranspose( coord_dir, df_dp_t_dir );
  EXPECT_LT( differenceNorm( dg_dx_dir, df_dp_t_dir ), 1.0e-12 );

  // Block operator must expose implicit zero (0,0), non-null (0,1)/(1,0), and constrained 3D placeholder (1,1).
  const auto frozen_block = getMfemBlockJacobian( scheme_id );
  ASSERT_NE( frozen_block, nullptr );
  EXPECT_TRUE( frozen_block->IsZeroBlock( 0, 0 ) );
  EXPECT_FALSE( frozen_block->IsZeroBlock( 0, 1 ) );
  EXPECT_FALSE( frozen_block->IsZeroBlock( 1, 0 ) );
  EXPECT_FALSE( frozen_block->IsZeroBlock( 1, 1 ) );

#ifdef TRIBOL_USE_ENZYME
  setMfemUseSingleMortarFormulation( scheme_id, true, true );
  contact.evaluate();

  const auto exact_residual = getMfemContactResidual( scheme_id );
  const auto exact_gap = getMfemContactGap( scheme_id );
  const auto exact_normals = getMfemNodalNormals( scheme_id );
  EXPECT_LT( differenceNorm( exact_residual, frozen_residual ), 1.0e-13 );
  EXPECT_LT( differenceNorm( exact_gap, frozen_gap ), 1.0e-13 );
  EXPECT_LT( differenceNorm( exact_normals, frozen_normals ), 1.0e-13 );

  const auto df_dx = getMfemDfDx( scheme_id );
  const auto df_dp = getMfemDfDp( scheme_id );
  const auto dg_dx = getMfemWeightedGapJacobian( scheme_id );
  const auto dn_dx = getMfemDnDx( scheme_id );
  ASSERT_NE( df_dx, nullptr );
  ASSERT_NE( df_dp, nullptr );
  ASSERT_NE( dg_dx, nullptr );
  ASSERT_NE( dn_dx, nullptr );

  // Directional finite differences with respect to current coordinates (holding previous_coordinates fixed).
  constexpr RealT step = 1.0e-6;
  mfem::Vector base_coords;
  contact.coordinates.GetTrueDofs( base_coords );

  mfem::Vector perturbed_coords( base_coords );
  perturbed_coords.Add( step, coord_dir );
  contact.coordinates.SetFromTrueDofs( perturbed_coords );
  contact.evaluate();
  const auto res_plus = getMfemContactResidual( scheme_id );
  const auto gap_plus = getMfemContactGap( scheme_id );
  const auto nrm_plus = getMfemNodalNormals( scheme_id );

  perturbed_coords = base_coords;
  perturbed_coords.Add( -step, coord_dir );
  contact.coordinates.SetFromTrueDofs( perturbed_coords );
  contact.evaluate();
  const auto res_minus = getMfemContactResidual( scheme_id );
  const auto gap_minus = getMfemContactGap( scheme_id );
  const auto nrm_minus = getMfemNodalNormals( scheme_id );

  contact.coordinates.SetFromTrueDofs( base_coords );

  auto finite_diff = []( const mfem::Vector& plus, const mfem::Vector& minus ) {
    mfem::Vector diff( plus );
    diff -= minus;
    diff *= 1.0 / ( 2.0 * step );
    return diff;
  };

  mfem::Vector expected_df( df_dx->Height() );
  df_dx->Mult( coord_dir, expected_df );
  EXPECT_LT( differenceNorm( finite_diff( res_plus, res_minus ), expected_df ), 1.0e-7 );

  mfem::Vector expected_dg( dg_dx->Height() );
  dg_dx->Mult( coord_dir, expected_dg );
  EXPECT_LT( differenceNorm( finite_diff( gap_plus, gap_minus ), expected_dg ), 1.0e-7 );

  mfem::Vector expected_dn( dn_dx->Height() );
  dn_dx->Mult( coord_dir, expected_dn );
  EXPECT_LT( differenceNorm( finite_diff( nrm_plus, nrm_minus ), expected_dn ), 1.0e-7 );

  // Directional finite difference with respect to 3D multipliers.
  auto& multiplier = getMfemContactPressure( scheme_id );
  const mfem::HypreParVector base_multiplier( multiplier );
  multiplier = base_multiplier;
  multiplier.Add( step, mult_dir );
  contact.evaluate();
  const auto res_p_plus = getMfemContactResidual( scheme_id );

  multiplier = base_multiplier;
  multiplier.Add( -step, mult_dir );
  contact.evaluate();
  const auto res_p_minus = getMfemContactResidual( scheme_id );
  multiplier = base_multiplier;

  mfem::Vector expected_df_dp( df_dp->Height() );
  df_dp->Mult( mult_dir, expected_df_dp );
  EXPECT_LT( differenceNorm( finite_diff( res_p_plus, res_p_minus ), expected_df_dp ), 1.0e-10 );
#else
  EXPECT_SLIC_ERROR( setMfemUseSingleMortarFormulation( scheme_id, true, true ) );
#endif
}

class MfemFrictionalSingleMortarTest : public detail::ExecutionTest {};

TEST_P( MfemFrictionalSingleMortarTest, linear_hex_frozen_and_exact_jacobians )
{
  checkFrictionalMortarCase( mfem::Element::HEXAHEDRON, 1, GetParam() );
}

TEST_P( MfemFrictionalSingleMortarTest, linear_tet_frozen_and_exact_jacobians )
{
  checkFrictionalMortarCase( mfem::Element::TETRAHEDRON, 1, GetParam() );
}

TEST_P( MfemFrictionalSingleMortarTest, quadratic_hex_lor_frozen_and_exact_jacobians )
{
  checkFrictionalMortarCase( mfem::Element::HEXAHEDRON, 2, GetParam() );
}

// Enabling frictional single mortar without registering previous-step coordinates must fail cleanly.
TEST( MfemFrictionalSingleMortarValidationTest, requires_previous_coordinates )
{
  FrictionalMortarCase contact( mfem::Element::HEXAHEDRON, 1, ExecutionMode::Sequential, false );
  EXPECT_SLIC_ERROR( setMfemUseSingleMortarFormulation( scheme_id, true, false ) );
}

INSTANTIATE_TEST_SUITE_P( backends, MfemFrictionalSingleMortarTest, detail::enabledExecutionModes() );
}  // namespace

int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
