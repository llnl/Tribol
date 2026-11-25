// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check MFEM single mortar outputs, solver blocks, and Jacobians on one and two MPI ranks.
 *
 * Nonmatching interfaces exercise assembly on both surfaces. Quadratic parent meshes exercise the LOR normal field
 * and its coordinate derivative. Tests also cover changing the decomposition and switching to and from legacy mortar.
 */
#include <cmath>

#include "ExecutionTest.hpp"
#include "SlicErrorTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/geom/LinearQuadGeometry.hpp"
#include "tribol/geom/MortarFaceMapping.hpp"
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

/** @brief Register two nonmatching contact surfaces and keep their coordinate fields alive. */
class MortarCase {
 public:
  /**
   * @brief Build and register the test meshes.
   * @param element_type Volume element type for both bodies.
   * @param order Parent coordinate order.
   * @param execution_mode Backend for contact calculations.
   * @param with_reference_coordinates Whether to register the reference coordinates immediately.
   * @param contact_case Registered contact case.
   * @param mortar_x_offset Translation of the mortar body along x.
   */
  MortarCase( mfem::Element::Type element_type, int order, ExecutionMode execution_mode,
              bool with_reference_coordinates = true, ContactCase contact_case = NO_CASE, RealT mortar_x_offset = 0.12 )
      : mesh( makeMesh( element_type, order, mortar_x_offset ) ),
        basis( order, spatial_dim ),
        coordinate_space( &mesh, &basis, spatial_dim, mfem::Ordering::byVDIM ),
        coordinates( &coordinate_space ),
        reference_coordinates( &coordinate_space )
  {
    mesh.GetNodes( reference_coordinates );
    mfem::VectorFunctionCoefficient deformed( spatial_dim, []( const mfem::Vector& point, mfem::Vector& value ) {
      value = point;
      // The rightmost non-mortar elements end outside the mortar projection. Their normals still affect shared nodes.
      value[2] += 0.04 * point[0] * point[0] + 0.02 * point[0] * point[1];
    } );
    coordinates.ProjectCoefficient( deformed );
    registerMfemCouplingScheme( scheme_id, mesh1_id, mesh2_id, mesh, coordinates, { nonmortar_attribute },
                                { mortar_attribute }, SURFACE_TO_SURFACE, contact_case, SINGLE_MORTAR, FRICTIONLESS,
                                LAGRANGE_MULTIPLIER, BINNING_GRID, execution_mode );
    if ( with_reference_coordinates ) {
      registerReferenceCoordinates();
    }
  }

  ~MortarCase() { finalize(); }

  /** @brief Register the undeformed coordinate field. */
  void registerReferenceCoordinates() { registerMfemReferenceCoords( scheme_id, reference_coordinates ); }

  /** @brief Get the registered scheme. @return Scheme valid for this test's lifetime. */
  CouplingScheme& scheme() const { return CouplingSchemeManager::getInstance().at( scheme_id ); }

  /** @brief Vary pressure along mesh1 while keeping the unused mortar-side pressures zero. */
  void initializePressure() const
  {
    auto& pressure = getMfemContactPressure( scheme_id );
    auto& fes = *getMfemPressure( scheme_id ).ParFESpace();
    mfem::ParGridFunction input( &fes );
    mfem::FunctionCoefficient value( []( const mfem::Vector& point ) {
      // The original mesh1 surface is at z=1.0. Mesh2 starts below it at z=0.97.
      constexpr RealT between_surfaces = 0.985;
      return point[2] > between_surfaces ? 0.6 + 0.25 * point[0] : 0.0;
    } );
    input.ProjectCoefficient( value );
    input.GetTrueDofs( pressure );
  }

  /** @brief Evaluate on current coordinates. @param num_redecomp_ranks Number of ranks to use for search.
   * @param force_redecomp Whether to partition again before evaluating.
   */
  void evaluate( int num_redecomp_ranks = 0, bool force_redecomp = false ) const
  {
    updateMfemParallelDecomposition( num_redecomp_ranks, force_redecomp );
    RealT dt = 1.0;
    ASSERT_EQ( update( 0, 0.0, dt ), 0 );
  }

  /** @brief Construct a displacement that changes both overlap and surface normals. @return Direction on true DOFs. */
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

  mfem::ParMesh mesh;                            ///< Parent mesh.
  mfem::H1_FECollection basis;                   ///< Parent coordinate basis.
  mfem::ParFiniteElementSpace coordinate_space;  ///< Parent displacement space.
  mfem::ParGridFunction coordinates;             ///< Current coordinates.
  mfem::ParGridFunction reference_coordinates;   ///< Undeformed coordinates.

 private:
  /**
   * @brief Create two volumes with offset and differently divided contact boundaries.
   * @param element_type Volume element type.
   * @param order Mesh coordinate order.
   * @param mortar_x_offset Translation of the mortar body along x.
   * @return Distributed parent mesh.
   */
  static mfem::ParMesh makeMesh( mfem::Element::Type element_type, int order, RealT mortar_x_offset )
  {
    const int num_elements1_x = 3;
    const int num_elements2_x = 2;
    const int num_elements_y = 2;
    const int num_elements_z = 1;
    auto parent = shared::MeshBuilder::Unify( {
        shared::MeshBuilder::CubeMesh( num_elements1_x, num_elements_y, num_elements_z, element_type ),
        shared::MeshBuilder::CubeMesh( num_elements2_x, num_elements_y, num_elements_z, element_type )
            .scale( { 0.44, 0.78, 1.0 } )
            .translate( { mortar_x_offset, 0.08, 0.97 } )
            .updateBdrAttrib( nonmortar_attribute, upper_top_attribute )
            .updateBdrAttrib( 1, mortar_attribute )
            .updateAttrib( 1, upper_body_attribute ),
    } );
    mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, std::move( parent ) );
    mesh.SetCurvature( order );
    return mesh;
  }
};

/**
 * @brief Measure the difference between true-DOF vectors across all ranks.
 * @param first First distributed vector.
 * @param second Second distributed vector with the same DOF layout.
 * @return Global Euclidean norm of the difference.
 */
RealT differenceNorm( const mfem::Vector& first, const mfem::Vector& second )
{
  first.HostRead();
  second.HostRead();
  mfem::Vector difference( first );
  difference -= second;
  return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, difference, difference ) );
}

/**
 * @brief Measure the global Euclidean norm of a true-DOF vector.
 * @param values Local true-DOF values on each rank.
 * @return Norm on all ranks, including ranks with no supported DOFs.
 */
RealT parallelNorm( const mfem::Vector& values )
{
  values.HostRead();
  return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, values, values ) );
}

using QuadCoords = MdArray<RealT, LinearQuadGeometry::num_nodes, spatial_dim>;

/**
 * @brief Copy current element coordinates to the host, including when the mesh is on a device.
 * @param mesh Current mesh of linear quadrilaterals.
 * @param execution_mode Backend that can read the mesh.
 * @return Host array in local element order.
 */
Array1D<QuadCoords, MemorySpace::Host> copyQuadCoordsToHost( MeshData::Viewer mesh, ExecutionMode execution_mode )
{
  const IndexT num_elements = mesh.numberOfElements();
  Array1D<QuadCoords> coords( num_elements, num_elements, mesh.getAllocatorId() );
  const auto output = coords.view();
  forAllExec( execution_mode, num_elements, [mesh, output] TRIBOL_HOST_DEVICE( IndexT element ) {
    output[element] = mesh.getFaceCoords<LinearQuadGeometry::Coords>( element );
  } );
  return Array1D<QuadCoords, MemorySpace::Host>( coords );
}

/**
 * @brief Check whether the local Redecomp mesh holds an overlapping pair split between two owners.
 * @param scheme Current quadrilateral meshes and their Redecomp element maps.
 * @return True if one element of an overlapping pair is owned here and the other is a ghost.
 */
bool hasCrossRankOverlap( CouplingScheme& scheme )
{
  const auto mesh1 = scheme.getMesh1().getView();
  const auto mesh2 = scheme.getMesh2().getView();
  if ( mesh1.numberOfElements() == 0 || mesh2.numberOfElements() == 0 ) {
    return false;
  }

  auto& data = *scheme.getMfemMeshData();
  const auto& redecomp = data.GetRedecompMesh();
  Array1D<bool, MemorySpace::Host> is_ghost( redecomp.GetNE() );
  is_ghost.fill( false );
  for ( const auto& rank_ghosts : redecomp.getRedecompToParentGhostElems() ) {
    for ( const auto element : rank_ghosts ) {
      is_ghost[element] = true;
    }
  }

  // The test also runs with device meshes. Inspect the same element maps and coordinates on the host.
  const Array1D<int, MemorySpace::Host> map1( data.GetElemMap1() ), map2( data.GetElemMap2() );
  const auto coords1 = copyQuadCoordsToHost( mesh1, scheme.getExecutionMode() );
  const auto coords2 = copyQuadCoordsToHost( mesh2, scheme.getExecutionMode() );
  const detail::MortarFaceMapping<LinearQuadGeometry> mapping{ scheme.getParameters().overlap_area_frac };
  for ( IndexT element1 = 0; element1 < map1.size(); ++element1 ) {
    const LinearQuadGeometry geometry1( coords1[element1].view() );
    for ( IndexT element2 = 0; element2 < map2.size(); ++element2 ) {
      if ( is_ghost[map1[element1]] != is_ghost[map2[element2]] ) {
        const auto overlap =
            mapping.makeIntegrationSurface( geometry1, LinearQuadGeometry( coords2[element2].view() ) );
        if ( overlap.valid && overlap.num_vertices > 0 ) {
          return true;
        }
      }
    }
  }
  return false;
}

class MfemSingleMortarTest : public detail::ExecutionTest {};

// Moving the same pairs between Redecomp ranks must preserve assembled values and both coupling matrix actions.
TEST_P( MfemSingleMortarTest, approximate_assembly_and_solver_blocks )
{
  // The offset places the mortar element centroids on either side of the x bisection through mesh1.
  constexpr RealT mortar_x_offset = 0.22;
  MortarCase contact( mfem::Element::HEXAHEDRON, 1, GetParam(), true, NO_CASE, mortar_x_offset );
  setMfemUseSingleMortarFormulation( scheme_id, true, false );
  auto* multiplier = &getMfemContactPressure( scheme_id );
  EXPECT_EQ( multiplier->Norml2(), 0.0 );
  contact.initializePressure();
  EXPECT_FALSE( contact.scheme().hasContactFormulation() );

  int num_ranks = 0;
  MPI_Comm_size( MPI_COMM_WORLD, &num_ranks );

  contact.evaluate( 1, true );
  const auto first_residual = getMfemContactResidual( scheme_id );
  const auto first_gap = getMfemContactGap( scheme_id );
  const auto first_gap_dx = getMfemWeightedGapJacobian( scheme_id );
  const auto first_df_dp = getMfemDfDp( scheme_id );
  ASSERT_NE( first_gap_dx, nullptr );
  ASSERT_NE( first_df_dp, nullptr );
  ASSERT_GT( parallelNorm( first_residual ), 0.0 );
  ASSERT_GT( parallelNorm( first_gap ), 0.0 );
  EXPECT_EQ( getMfemDfDx( scheme_id ), nullptr );
  if ( num_ranks > 1 ) {
    const auto& redecomp = contact.scheme().getMfemMeshData()->GetRedecompMesh();
    first_residual.HostRead();
    int empty_rank_received_residual = redecomp.GetNE() == 0 && first_residual.Norml2() > 0.0;
    MPI_Allreduce( MPI_IN_PLACE, &empty_rank_received_residual, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
    EXPECT_EQ( empty_rank_received_residual, 1 );
  }

  const auto coordinate_direction = contact.coordinateDirection();
  mfem::Vector first_gap_action( multiplier->Size() ), pressure_action( first_residual.Size() );
  first_gap_dx->Mult( coordinate_direction, first_gap_action );
  first_df_dp->Mult( *multiplier, pressure_action );
  constexpr RealT assembly_tolerance = 1.0e-12;
  EXPECT_LT( differenceNorm( first_residual, pressure_action ), assembly_tolerance );
  mfem::Vector transposed_action( multiplier->Size() );
  first_df_dp->MultTranspose( coordinate_direction, transposed_action );
  EXPECT_LT( differenceNorm( first_gap_action, transposed_action ), assembly_tolerance );

  contact.evaluate( num_ranks, true );
  if ( num_ranks > 1 ) {
    int cross_rank_overlap = hasCrossRankOverlap( contact.scheme() );
    MPI_Allreduce( MPI_IN_PLACE, &cross_rank_overlap, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
    EXPECT_EQ( cross_rank_overlap, 1 );
  }
  EXPECT_EQ( &getMfemContactPressure( scheme_id ), multiplier );
  EXPECT_LT( differenceNorm( first_residual, getMfemContactResidual( scheme_id ) ), assembly_tolerance );
  EXPECT_LT( differenceNorm( first_gap, getMfemContactGap( scheme_id ) ), assembly_tolerance );
  mfem::Vector gap_action( multiplier->Size() );
  getMfemWeightedGapJacobian( scheme_id )->Mult( coordinate_direction, gap_action );
  EXPECT_LT( differenceNorm( first_gap_action, gap_action ), assembly_tolerance );
  mfem::Vector rebuilt_pressure_action( first_residual.Size() );
  getMfemDfDp( scheme_id )->Mult( *multiplier, rebuilt_pressure_action );
  EXPECT_LT( differenceNorm( pressure_action, rebuilt_pressure_action ), assembly_tolerance );

  const auto block = getMfemBlockJacobian( scheme_id );
  ASSERT_NE( block, nullptr );
  EXPECT_TRUE( block->IsZeroBlock( 0, 0 ) );
  EXPECT_FALSE( block->IsZeroBlock( 0, 1 ) );
  EXPECT_FALSE( block->IsZeroBlock( 1, 0 ) );
  EXPECT_FALSE( block->IsZeroBlock( 1, 1 ) );
  mfem::Vector unused_pressure( multiplier->Size() );
  block->GetBlock( 1, 1 ).Mult( *multiplier, unused_pressure );
  EXPECT_EQ( unused_pressure.Norml2(), 0.0 );
  mfem::Vector ones( multiplier->Size() );
  ones = 1.0;
  block->GetBlock( 1, 1 ).Mult( ones, unused_pressure );
  unused_pressure.HostRead();
  multiplier->HostRead();
  for ( int dof = 0; dof < multiplier->Size(); ++dof ) {
    EXPECT_EQ( unused_pressure[dof], ( *multiplier )[dof] == 0.0 ? 1.0 : 0.0 );
  }

  const auto gaps_only = contact.scheme().getContactFormulation()->computeNodalGaps(
      contact.scheme().getMesh1().getView(), contact.scheme().getMesh2().getView() );
  ASSERT_NE( gaps_only.gap, nullptr );
  ASSERT_NE( gaps_only.weighted_gap_jacobian, nullptr );
  EXPECT_EQ( gaps_only.residual, nullptr );
  EXPECT_EQ( gaps_only.pressure, nullptr );
  EXPECT_EQ( gaps_only.df_dp, nullptr );
  EXPECT_LT( differenceNorm( first_gap, *gaps_only.gap ), assembly_tolerance );
}

// Moving only the far edge of an element outside the overlap changes normals at its shared nodes.
TEST_P( MfemSingleMortarTest, normals_include_elements_outside_the_overlap )
{
  MortarCase contact( mfem::Element::HEXAHEDRON, 1, GetParam() );
  setMfemUseSingleMortarFormulation( scheme_id, true, false );
  contact.initializePressure();
  contact.evaluate();
  const auto original = getMfemContactResidual( scheme_id );
  mfem::ParGridFunction motion( &contact.coordinate_space );
  mfem::VectorFunctionCoefficient lift( spatial_dim, []( const mfem::Vector& point, mfem::Vector& value ) {
    value.SetSize( spatial_dim );
    value = 0.0;
    // Only mesh1 reaches x=1.0, and the mortar surface ends before the last column of mesh1 elements.
    constexpr RealT minimum_x = 0.9;
    constexpr RealT displacement = 0.04;
    if ( point[0] > minimum_x ) {
      value[2] = displacement;
    }
  } );
  motion.ProjectCoefficient( lift );
  contact.coordinates += motion;
  contact.evaluate();
  constexpr RealT minimum_residual_change = 1.0e-6;
  EXPECT_GT( differenceNorm( original, getMfemContactResidual( scheme_id ) ), minimum_residual_change );
}

// The requested quadrature changes the residual. Rejecting every overlap still returns sized zero outputs.
TEST_P( MfemSingleMortarTest, quadrature_order_and_empty_overlap )
{
  MortarCase contact( mfem::Element::HEXAHEDRON, 1, GetParam() );
  setMfemUseSingleMortarFormulation( scheme_id, true, false );
  contact.initializePressure();
  contact.evaluate();
  const auto initial = getMfemContactResidual( scheme_id );

  constexpr int centroid_rule_order = 0;
  setQuadratureOrder( scheme_id, centroid_rule_order );
  contact.evaluate();
  constexpr RealT minimum_quadrature_difference = 1.0e-8;
  EXPECT_GT( differenceNorm( initial, getMfemContactResidual( scheme_id ) ), minimum_quadrature_difference );

  constexpr RealT fraction_above_full_overlap = 1.1;
  setContactAreaFrac( scheme_id, fraction_above_full_overlap );
  contact.evaluate();
  const auto empty_residual = getMfemContactResidual( scheme_id );
  const auto empty_gap = getMfemContactGap( scheme_id );
  EXPECT_EQ( empty_residual.Size(), initial.Size() );
  EXPECT_EQ( empty_gap.Size(), getMfemContactPressure( scheme_id ).Size() );
  EXPECT_EQ( parallelNorm( empty_residual ), 0.0 );
  EXPECT_EQ( parallelNorm( empty_gap ), 0.0 );
  EXPECT_NE( getMfemBlockJacobian( scheme_id ), nullptr );
}

// A concave mortar element gives invalid projections while the other elements still contribute to the residual.
TEST_P( MfemSingleMortarTest, reports_invalid_projections_and_keeps_valid_contributions )
{
  MortarCase contact( mfem::Element::HEXAHEDRON, 1, GetParam() );
  setMfemUseSingleMortarFormulation( scheme_id, true, false );
  contact.initializePressure();
  mfem::ParGridFunction motion( &contact.coordinate_space );
  mfem::VectorFunctionCoefficient fold( spatial_dim, []( const mfem::Vector& point, mfem::Vector& value ) {
    value.SetSize( spatial_dim );
    value = 0.0;
    const MdArray<RealT, spatial_dim> corner{ 0.56, 0.08, 0.97 };
    constexpr RealT corner_tolerance = 1.0e-12;
    if ( std::abs( point[0] - corner[0] ) < corner_tolerance && std::abs( point[1] - corner[1] ) < corner_tolerance &&
         std::abs( point[2] - corner[2] ) < corner_tolerance ) {
      // Move this vertex inside the triangle formed by the other three vertices of its surface element.
      value[0] = -0.17;
      value[1] = 0.24;
    }
  } );
  motion.ProjectCoefficient( fold );
  contact.coordinates += motion;

  testing::internal::CaptureStdout();
  contact.evaluate( 1, true );
  const auto output = testing::internal::GetCapturedStdout();
  constexpr auto warning_text = "pair evaluations with invalid projected geometry";
  const auto warning = output.find( warning_text );
  int rank = 0;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  if ( rank == 0 ) {
    EXPECT_NE( warning, std::string::npos );
    EXPECT_EQ( output.rfind( warning_text ), warning );
  } else {
    EXPECT_EQ( warning, std::string::npos );
  }
  const auto residual = getMfemContactResidual( scheme_id );
  EXPECT_GT( parallelNorm( residual ), 0.0 );
}

// Changing the LOR factor after assembly replaces the linear surface data and keeps the multiplier space.
TEST_P( MfemSingleMortarTest, changing_lor_factor_rebuilds_transfers )
{
  constexpr int coordinate_order = 2;
  MortarCase contact( mfem::Element::HEXAHEDRON, coordinate_order, GetParam() );
  setMfemUseSingleMortarFormulation( scheme_id, true, false );
  contact.initializePressure();
  auto* multiplier = &getMfemContactPressure( scheme_id );
  contact.evaluate( 1, true );
  const auto original = getMfemContactResidual( scheme_id );
  const auto original_pressure = parallelNorm( *multiplier );

  constexpr int refined_factor = 3;
  setMfemLORFactor( scheme_id, refined_factor );
  EXPECT_FALSE( contact.scheme().hasContactFormulation() );
  contact.evaluate();
  EXPECT_EQ( contact.scheme().getMfemMeshData()->GetLORFactor(), refined_factor );
  EXPECT_EQ( &getMfemContactPressure( scheme_id ), multiplier );
  EXPECT_EQ( parallelNorm( *multiplier ), original_pressure );
  const auto residual = getMfemContactResidual( scheme_id );
  EXPECT_EQ( residual.Size(), original.Size() );
  EXPECT_GT( parallelNorm( residual ), 0.0 );
}

// Selecting the formulation requires reference coordinates and NO_CASE. Disabling restores the legacy pressure input.
TEST_P( MfemSingleMortarTest, selection_and_legacy_transition )
{
  {
    MortarCase unsupported( mfem::Element::HEXAHEDRON, 1, GetParam(), true, NO_SLIDING );
    EXPECT_SLIC_ERROR( setMfemUseSingleMortarFormulation( scheme_id, true, false ) );
    EXPECT_FALSE( unsupported.scheme().usesSingleMortarFormulation() );
  }
  MortarCase contact( mfem::Element::HEXAHEDRON, 1, GetParam(), false );
  EXPECT_SLIC_ERROR( setMfemUseSingleMortarFormulation( scheme_id, true, false ) );
  EXPECT_FALSE( contact.scheme().usesSingleMortarFormulation() );
  EXPECT_FALSE( contact.scheme().hasMfemJacobianData() );
  contact.registerReferenceCoordinates();
  setLagrangeMultiplierOptions( scheme_id, ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN );
  updateMfemParallelDecomposition( 1, true );
  constexpr RealT legacy_pressure = 0.5;
  getMfemPressure( scheme_id ) = legacy_pressure;

  setMfemUseSingleMortarFormulation( scheme_id, true, false );
#ifndef TRIBOL_USE_ENZYME
  EXPECT_SLIC_ERROR( setMfemUseSingleMortarFormulation( scheme_id, true, true ) );
  EXPECT_FALSE( contact.scheme().getParameters().single_mortar_exact_jacobian );
#endif
  EXPECT_SLIC_ERROR( (void)getMfemBlockJacobian( scheme_id ) );
  EXPECT_SLIC_ERROR( setEnforcementLocation( scheme_id, EnforcementLocation::QuadraturePoint ) );
  EXPECT_SLIC_ERROR( setQuadratureOrder( scheme_id, -1 ) );
  contact.initializePressure();
  contact.evaluate( 1, true );
  const auto saved = getMfemContactResidual( scheme_id );
  auto& scheme = contact.scheme();
  EXPECT_SLIC_ERROR( (void)scheme.getContactFormulation()->computeNodalResiduals( scheme.getMesh1().getView(),
                                                                                  scheme.getMesh2().getView(), {} ) );
  setMfemUseSingleMortarFormulation( scheme_id, false );
  EXPECT_FALSE( contact.scheme().usesSingleMortarFormulation() );
  EXPECT_FALSE( contact.scheme().hasContactFormulation() );
  EXPECT_TRUE( contact.scheme().hasMfemJacobianData() );
  EXPECT_DOUBLE_EQ( getMfemPressure( scheme_id ).Max(), legacy_pressure );
  EXPECT_GT( parallelNorm( saved ), 0.0 );
  if ( GetParam() == ExecutionMode::Sequential ) {
    contact.evaluate( 1, true );
    EXPECT_NE( getMfemBlockJacobian( scheme_id ), nullptr );
  }
}

#ifdef TRIBOL_USE_ENZYME
// Centered differences of the assembled result include changes in overlap and contributions from neighboring elements.
TEST_P( MfemSingleMortarTest, exact_jacobians_on_triangle_quad_and_lor_surfaces )
{
  struct Configuration {
    mfem::Element::Type element_type;
    int order;
  };
  for ( const auto configuration :
        { Configuration{ mfem::Element::HEXAHEDRON, 1 }, Configuration{ mfem::Element::TETRAHEDRON, 1 },
          Configuration{ mfem::Element::HEXAHEDRON, 2 }, Configuration{ mfem::Element::TETRAHEDRON, 2 } } ) {
    SCOPED_TRACE( testing::Message() << "Element " << configuration.element_type << ", order " << configuration.order );
    MortarCase contact( configuration.element_type, configuration.order, GetParam() );
    setMfemUseSingleMortarFormulation( scheme_id, true, false );
    contact.initializePressure();
    auto* multiplier = &getMfemContactPressure( scheme_id );
    contact.evaluate();
    const auto approximate_gap = getMfemContactGap( scheme_id );
    const auto approximate_residual = getMfemContactResidual( scheme_id );
    const auto approximate_df_dp = getMfemDfDp( scheme_id );

    setMfemUseSingleMortarFormulation( scheme_id, true );
    EXPECT_EQ( &getMfemContactPressure( scheme_id ), multiplier );
    contact.evaluate();
    const auto gap = getMfemContactGap( scheme_id );
    const auto residual = getMfemContactResidual( scheme_id );
    const auto df_dx = getMfemDfDx( scheme_id );
    const auto dg_dx = getMfemWeightedGapJacobian( scheme_id );
    const auto df_dp = getMfemDfDp( scheme_id );
    ASSERT_NE( df_dx, nullptr );
    ASSERT_NE( dg_dx, nullptr );
    ASSERT_NE( df_dp, nullptr );
    EXPECT_FALSE( getMfemBlockJacobian( scheme_id )->IsZeroBlock( 0, 0 ) );
    constexpr RealT assembly_tolerance = 1.0e-12;
    EXPECT_LT( differenceNorm( approximate_gap, gap ), assembly_tolerance );
    EXPECT_LT( differenceNorm( approximate_residual, residual ), assembly_tolerance );

    const auto gaps_only = contact.scheme().getContactFormulation()->computeNodalGaps(
        contact.scheme().getMesh1().getView(), contact.scheme().getMesh2().getView() );
    EXPECT_LT( differenceNorm( gap, *gaps_only.gap ), assembly_tolerance );

    const auto direction = contact.coordinateDirection();
    mfem::Vector expected_residual( residual.Size() ), expected_gap( gap.Size() ), gap_only_action( gap.Size() );
    df_dx->Mult( direction, expected_residual );
    dg_dx->Mult( direction, expected_gap );
    gaps_only.weighted_gap_jacobian->Mult( direction, gap_only_action );
    EXPECT_LT( differenceNorm( expected_gap, gap_only_action ), assembly_tolerance );

    constexpr RealT step = 1.0e-6;
    mfem::Vector original_coords;
    contact.coordinates.GetTrueDofs( original_coords );
    mfem::Vector perturbed_coords( original_coords );
    perturbed_coords.Add( step, direction );
    contact.coordinates.SetFromTrueDofs( perturbed_coords );
    contact.evaluate();
    const auto residual_plus = getMfemContactResidual( scheme_id );
    const auto gap_plus = getMfemContactGap( scheme_id );
    mfem::Vector residual_difference( residual_plus );
    mfem::Vector gap_difference( gap_plus );

    perturbed_coords = original_coords;
    perturbed_coords.Add( -step, direction );
    contact.coordinates.SetFromTrueDofs( perturbed_coords );
    contact.evaluate();
    residual_difference -= getMfemContactResidual( scheme_id );
    gap_difference -= getMfemContactGap( scheme_id );
    residual_difference /= 2.0 * step;
    gap_difference /= 2.0 * step;
    constexpr RealT derivative_tolerance = 1.0e-7;
    EXPECT_LT( differenceNorm( residual_difference, expected_residual ), derivative_tolerance );
    EXPECT_LT( differenceNorm( gap_difference, expected_gap ), derivative_tolerance );
    contact.coordinates.SetFromTrueDofs( original_coords );

    const mfem::Vector original_pressure( *multiplier );
    const mfem::Vector pressure_direction( original_pressure );
    mfem::Vector expected_pressure_action( residual.Size() ), approximate_pressure_action( residual.Size() );
    df_dp->Mult( pressure_direction, expected_pressure_action );
    approximate_df_dp->Mult( pressure_direction, approximate_pressure_action );
    EXPECT_LT( differenceNorm( expected_pressure_action, approximate_pressure_action ), assembly_tolerance );
    multiplier->Add( step, pressure_direction );
    contact.evaluate();
    const auto pressure_plus = getMfemContactResidual( scheme_id );
    mfem::Vector pressure_difference( pressure_plus );
    multiplier->Add( -2.0 * step, pressure_direction );
    contact.evaluate();
    pressure_difference -= getMfemContactResidual( scheme_id );
    pressure_difference /= 2.0 * step;
    EXPECT_LT( differenceNorm( pressure_difference, expected_pressure_action ), derivative_tolerance );
  }
}
#endif

INSTANTIATE_TEST_SUITE_P( backends, MfemSingleMortarTest, detail::enabledExecutionModes() );
}  // namespace

int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
#ifdef TRIBOL_USE_CUDA
  mfem::Device device( "cuda" );
#elif defined( TRIBOL_USE_HIP )
  mfem::Device device( "hip" );
#endif
  const int status = RUN_ALL_TESTS();
  MPI_Finalize();
  return status;
}
