// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check explicit MFEM Common Plane selection, nonmatching constant-traction patches, and parent Jacobians.
 * The same tests run on one and two ranks; decomposition onto rank zero exercises a rank without contact elements.
 * Direct calls check output availability and validity after destroying the formulation.
 * Local quadrature and full pair derivatives are checked in tribol_common_plane_formulation.
 */
#include <cmath>
#include <stdexcept>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/PairContactFormulation.hpp"
#include "tribol/integ/QuadratureRuleFactory.hpp"
#include "tribol/search/CoarseSearchFactory.hpp"
#include "tribol/search/ProjectedEdgeSearch.hpp"

namespace {
using namespace tribol;
/** @brief Register a two-versus-three-element contact patch for each supported rule. */
class MfemCommonPlaneTest : public testing::TestWithParam<int> {
 protected:
  /** @brief Create separated bodies with a uniform 0.01 penetration along nonmatching boundary meshes. */
  void SetUp() override
  {
    mesh_ = std::make_unique<mfem::ParMesh>( static_cast<const mfem::ParMesh&>( shared::ParMeshBuilder(
        MPI_COMM_WORLD,
        shared::MeshBuilder::Unify( { shared::MeshBuilder::SquareMesh( 2, 1 ), shared::MeshBuilder::SquareMesh( 3, 1 )
                                                                                   .translate( { 0.0, 0.99 } )
                                                                                   .updateBdrAttrib( 1, 5 )
                                                                                   .updateBdrAttrib( 3, 6 )
                                                                                   .updateAttrib( 1, 2 ) } ) ) ) );
    fec_ = std::make_unique<mfem::H1_FECollection>( 1, 2 );
    fes_ = std::make_unique<mfem::ParFiniteElementSpace>( mesh_.get(), fec_.get(), 2, mfem::Ordering::byVDIM );
    coords_ = std::make_unique<mfem::ParGridFunction>( fes_.get() );
    mesh_->GetNodes( *coords_ );
    registerMfemCouplingScheme( 0, 0, 1, *mesh_, *coords_, { 3 }, { 5 }, SURFACE_TO_SURFACE, NO_CASE, COMMON_PLANE,
                                FRICTIONLESS, PENALTY, BINNING_GRID, ExecutionMode::Sequential );
    setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
  }
  /** @brief Release coupling data before destroying the borrowed parent mesh and coordinate field. */
  void TearDown() override { finalize(); }
  /** @brief Enable the formulation and select the requested reference degree. */
  void enable()
  {
    setMfemUseCommonPlaneFormulation( 0, true );
    setQuadratureOrder( 0, GetParam() );
  }
  /** @brief Refresh geometry on one contact rank and compute the current residual. */
  void evaluate()
  {
    updateMfemParallelDecomposition( 1, true );
    RealT dt = 1.0;
    ASSERT_EQ( update( 0, 0.0, dt ), 0 );
  }
  /** @brief Return the registered coupling scheme. @return Scheme zero, valid for this fixture's lifetime. */
  CouplingScheme& scheme() { return CouplingSchemeManager::getInstance().at( 0 ); }
  /** @brief Calculate a distributed vector norm. @param v Distributed true-DOF vector. @return Global Euclidean norm.
   */
  RealT norm( const mfem::Vector& v ) { return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, v, v ) ); }
  std::unique_ptr<mfem::ParMesh> mesh_;               ///< Borrowed by the registered coupling scheme.
  std::unique_ptr<mfem::H1_FECollection> fec_;        ///< Linear coordinate basis.
  std::unique_ptr<mfem::ParFiniteElementSpace> fes_;  ///< Parent vector-coordinate space.
  std::unique_ptr<mfem::ParGridFunction> coords_;     ///< Current parent geometry.
};

// Nonmatching pair assembly must reproduce uniform analytic boundary traction without any pressure or gap field.
TEST_P( MfemCommonPlaneTest, matches_nonmatching_constant_traction_patch )
{
  EXPECT_FALSE( scheme().hasContactFormulation() );
  enable();
  evaluate();
  EXPECT_EQ( scheme().getMfemSubmeshData(), nullptr );
  ASSERT_TRUE( scheme().hasMfemJacobianData() );
  EXPECT_EQ( scheme().getMfemJacobianData()->ParentPath().final_fes, fes_.get() );
  for ( auto* mesh : { &scheme().getMesh1(), &scheme().getMesh2() } ) {
    EXPECT_FALSE( mesh->getNodalFields().m_is_node_gap_set );
    EXPECT_FALSE( mesh->getNodalFields().m_is_node_pressure_set );
  }
  const auto residual = getMfemContactForce( 0 );
  mfem::Vector traction1( 2 ), traction2( 2 );
  traction1 = 0.0;
  traction2 = 0.0;
  constexpr RealT pressure = 0.015;  // Series stiffness 1.5 times uniform penetration 0.01.
  traction1[1] = pressure;
  traction2[1] = -pressure;
  mfem::VectorConstantCoefficient coefficient1( traction1 ), coefficient2( traction2 );
  mfem::Array<int> marker1( mesh_->bdr_attributes.Max() ), marker2( mesh_->bdr_attributes.Max() );
  marker1 = 0;
  marker2 = 0;
  marker1[3 - 1] = 1;
  marker2[5 - 1] = 1;
  mfem::ParLinearForm expected( fes_.get() );
  expected.AddBoundaryIntegrator( new mfem::VectorBoundaryLFIntegrator( coefficient1 ), marker1 );
  expected.AddBoundaryIntegrator( new mfem::VectorBoundaryLFIntegrator( coefficient2 ), marker2 );
  expected.Assemble();
  std::unique_ptr<mfem::HypreParVector> exact( expected.ParallelAssemble() );
  auto error = residual;
  error -= *exact;
  EXPECT_LT( norm( error ), 1.0e-11 );
  int rank = 0, size = 0;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  if ( size == 2 && rank == 1 )
    EXPECT_EQ( scheme().getMesh1().numberOfElements() + scheme().getMesh2().numberOfElements(), 0 );
  EXPECT_FALSE( scheme().getContactFormulationResult().energy.has_value() );
  EXPECT_THROW( scheme().getContactFormulation()->computeNodalGaps( scheme().getMesh1().getView(),
                                                                    scheme().getMesh2().getView() ),
                std::logic_error );
}

// The same force definition must accept arithmetic averaging instead of springs in series, for each Gauss count.
TEST_P( MfemCommonPlaneTest, accepts_average_penalty_stiffness )
{
  enable();
  evaluate();
  const auto baseline = getMfemContactForce( 0 );
  auto alternate =
      createWithQuadratureRule( GetParam(), [&]<typename BaseRule>() -> std::unique_ptr<ContactFormulation> {
        using Method = CommonPlanePairMethod<BaseRule, AveragePenaltyStiffness>;
        auto coarse =
            createCoarseSearch( scheme().spatialDimension(), scheme().getBinningMethod(), scheme().getExecutionMode(),
                                scheme().getEffectiveBinningProximityScale(), 0.0 );
        auto refined = std::make_unique<ProjectedEdgeSearch>( scheme().getExecutionMode(), false );
        return std::make_unique<PairContactFormulation<Method>>(
            *scheme().getMfemMeshData(), *scheme().getMfemJacobianData(), std::move( coarse ), std::move( refined ),
            Method{ CommonPlaneQuadratureRule<BaseRule>{}, CommonPlaneResidual{ OffsetGap{ CommonPlaneGap{}, 0.0 } },
                    ScaledMeshPenaltyInputs<AveragePenaltyStiffness>{} } );
      } );
  auto result = alternate->computeNodalForces( scheme().getMesh1().getView(), scheme().getMesh2().getView() );
  ASSERT_NE( result.force, nullptr );
  EXPECT_FALSE( result.energy.has_value() );
  // The configured stiffnesses 2 and 6 have arithmetic mean 4 and series stiffness 1.5.
  constexpr RealT stiffness_ratio = 4.0 / 1.5;
  result.force->Add( -stiffness_ratio, baseline );
  EXPECT_LT( norm( *result.force ), 1.0e-11 );
}

// Each direct call returns its own force and matrix, without changing earlier results or the coupling-scheme cache.
TEST_P( MfemCommonPlaneTest, returns_owned_outputs_without_retaining_results )
{
  enable();
  updateMfemParallelDecomposition( 1, true );
  const auto& formulation = *scheme().getContactFormulation();
  auto first = formulation.computeNodalForces( scheme().getMesh1().getView(), scheme().getMesh2().getView() );
  ASSERT_NE( first.force, nullptr );
  ASSERT_NE( first.df_dx, nullptr );
  EXPECT_FALSE( first.energy.has_value() );
  EXPECT_FALSE( first.time_step.has_value() );
  EXPECT_EQ( first.gap, nullptr );
  EXPECT_EQ( first.pressure, nullptr );
  EXPECT_EQ( first.dg_dx, nullptr );
  EXPECT_EQ( first.df_dp, nullptr );
  EXPECT_EQ( scheme().getContactFormulationResult().force, nullptr );
  const mfem::HypreParVector original( *first.force );
  ASSERT_GT( norm( original ), 0.0 );

  setPenaltyScale( 0, 2.0 );
  setPenaltyScale( 1, 2.0 );
  auto second = scheme().getContactFormulation()->computeNodalForces( scheme().getMesh1().getView(),
                                                                      scheme().getMesh2().getView() );
  auto error = *second.force;
  error.Add( -2.0, original );
  EXPECT_LT( norm( error ), 1.0e-11 );
  scheme().setContactFormulation( nullptr );
  error = *first.force;
  error -= original;
  EXPECT_EQ( norm( error ), 0.0 );
  mfem::Vector direction( first.df_dx->Width() ), action( first.df_dx->Height() );
  direction = 1.0;
  first.df_dx->Mult( direction, action );
  EXPECT_TRUE( std::isfinite( norm( action ) ) );
}

// Parent Jacobians differentiate assembled residuals through nonmatching pair maps and MPI transfers.
TEST_P( MfemCommonPlaneTest, assembled_jacobian_matches_finite_difference )
{
  // Shift only the upper body so projection endpoints do not coincide. Rotating either edge under perturbation
  // then tests normal and overlap derivatives without switching which endpoints limit the overlap.
  mfem::Array<int> upper_nodes( fes_->GetNDofs() );
  upper_nodes = 0;
  for ( int element = 0; element < mesh_->GetNE(); ++element ) {
    if ( mesh_->GetAttribute( element ) != 2 ) continue;
    mfem::Array<int> nodes;
    fes_->GetElementDofs( element, nodes );
    for ( const auto node : nodes ) upper_nodes[node] = 1;
  }
  for ( int node = 0; node < upper_nodes.Size(); ++node )
    if ( upper_nodes[node] ) ( *coords_ )[node] += 0.13;
  enable();
  evaluate();
  auto jacobian = getMfemDfDx( 0 );
  ASSERT_NE( jacobian, nullptr );
  const mfem::ParGridFunction original( *coords_ );
  mfem::VectorFunctionCoefficient coefficient( 2, []( const mfem::Vector& x, mfem::Vector& v ) {
    v.SetSize( 2 );
    v[0] = 0.03 * x[0] + 0.01 * x[1];
    v[1] = 0.02 * x[0] - 0.04 * x[1];
  } );
  mfem::ParGridFunction direction( fes_.get() );
  direction.ProjectCoefficient( coefficient );
  mfem::Vector true_direction;
  direction.GetTrueDofs( true_direction );
  mfem::Vector predicted( true_direction.Size() );
  jacobian->Mult( true_direction, predicted );
  constexpr RealT step = 1.0e-6;
  *coords_ = original;
  coords_->Add( step, direction );
  evaluate();
  const auto plus = getMfemContactForce( 0 );
  *coords_ = original;
  coords_->Add( -step, direction );
  evaluate();
  auto difference = getMfemContactForce( 0 );
  difference.Neg();
  difference += plus;
  difference /= 2.0 * step;
  difference -= predicted;
  EXPECT_LT( norm( difference ), 1.0e-7 * ( 1.0 + norm( predicted ) ) );
  *coords_ = original;
}

// Enabling after legacy evaluation, changing penalties, rebuilding, and disabling must preserve transfers and signs.
TEST_P( MfemCommonPlaneTest, rebuilds_and_restores_legacy_response )
{
  evaluate();
  scheme().setFixedBinning( true );  // Re-enabling legacy contact must discard a previously cached pair list.

  enable();
  evaluate();
  const auto initial = getMfemContactForce( 0 );
  setMfemKinematicConstantPenalty( 0, 4.0, 12.0 );
  evaluate();
  auto scaled = getMfemContactForce( 0 );
  auto error = scaled;
  error.Add( -2.0, initial );
  EXPECT_LT( norm( error ), 1.0e-11 );
  EXPECT_THROW( setEnforcementLocation( 0, EnforcementLocation::Nodal ), std::invalid_argument );
  EXPECT_THROW( setMfemLORFactor( 0, 2 ), std::invalid_argument );
  EXPECT_EQ( scheme().getParameters().enforcement_location, EnforcementLocation::QuadraturePoint );
  setMfemUseCommonPlaneFormulation( 0, false );
  EXPECT_FALSE( scheme().hasContactFormulation() );
  EXPECT_FALSE( scheme().hasMfemJacobianData() );
  EXPECT_FALSE( scheme().hasFixedBinning() );
  EXPECT_FALSE( scheme().isBinned() );
  evaluate();
  mfem::Vector legacy( fes_->GetVSize() );
  legacy = 0.0;
  getMfemResponse( 0, legacy );
  mfem::Vector true_legacy( fes_->GetTrueVSize() );
  fes_->GetProlongationMatrix()->MultTranspose( legacy, true_legacy );
  scaled += true_legacy;
  EXPECT_LT( norm( scaled ), 1.0e-11 );
  setMfemUseCommonPlaneFormulation( 0, true );
  evaluate();
  EXPECT_GT( norm( getMfemContactForce( 0 ) ), 0.0 );
}

// Rejected configuration must leave the legacy selection and its resources intact.
TEST_P( MfemCommonPlaneTest, rejects_unsupported_settings_before_selection )
{
  const auto check = [&]() {
    EXPECT_THROW( setMfemUseCommonPlaneFormulation( 0, true ), std::invalid_argument );
    EXPECT_FALSE( scheme().getParameters().use_common_plane_formulation );
    EXPECT_FALSE( scheme().hasContactFormulation() );
    EXPECT_FALSE( scheme().hasMfemJacobianData() );
  };
  EXPECT_THROW( setMfemUseCommonPlaneFormulation( 1, true ), std::invalid_argument );
  setMfemKinematicConstantPenalty( 0, 0.0, 6.0 );
  check();
  setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
  scheme().getParameters().enforcement_location = EnforcementLocation::Nodal;
  check();
  scheme().getParameters().enforcement_location = EnforcementLocation::QuadraturePoint;
  auto& penalty = scheme().getEnforcementOptions().penalty_options;
  penalty.kinematic_calculation = KINEMATIC_ELEMENT;
  check();
  penalty.kinematic_calculation = KINEMATIC_CONSTANT;
  penalty.rate_calc_set = true;
  penalty.rate_calculation = RATE_CONSTANT;
  check();
  penalty.rate_calc_set = false;
  EXPECT_THROW( setQuadratureOrder( 0, 3 ), std::invalid_argument );
  enable();
  for ( int order = 0; order <= 5; ++order ) {
    setQuadratureOrder( 0, order );
    auto* formulation = scheme().getContactFormulation();
    if ( order <= 1 )
      EXPECT_NE(
          dynamic_cast<PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<1>>>*>( formulation ),
          nullptr );
    else if ( order <= 3 )
      EXPECT_NE(
          dynamic_cast<PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<2>>>*>( formulation ),
          nullptr );
    else
      EXPECT_NE(
          dynamic_cast<PairContactFormulation<CommonPlanePairMethod<EdgeQuadratureRuleGenerator<3>>>*>( formulation ),
          nullptr );
  }
  EXPECT_THROW( setQuadratureOrder( 0, 6 ), std::invalid_argument );
  EXPECT_EQ( scheme().getParameters().quadrature_order, 5 );
}
// The opt-in setter must reject unsupported contact models, enforcement, and tied cases before constructing resources.
TEST( CommonPlaneSelection, rejects_unsupported_methods_models_and_cases )
{
  struct Configuration {
    ContactMethod method;           ///< Contact method selected during registration.
    ContactModel model;             ///< Traction model selected during registration.
    EnforcementMethod enforcement;  ///< Constraint enforcement selected during registration.
    ContactCase contact_case;       ///< Pair interaction case.
  };
  const Configuration cases[] = { { COMMON_PLANE, VISCOUS_TANGENTIAL, PENALTY, NO_CASE },
                                  { COMMON_PLANE, FRICTIONLESS, LAGRANGE_MULTIPLIER, NO_CASE },
                                  { COMMON_PLANE, FRICTIONLESS, PENALTY, TIED_NORMAL },
                                  { COMMON_PLANE, FRICTIONLESS, PENALTY, TIED_FULL },
                                  { SINGLE_MORTAR, FRICTIONLESS, PENALTY, NO_CASE } };
  for ( const auto test : cases ) {
    mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( 1, 1 ) );
    mfem::H1_FECollection fec( 1, 2 );
    mfem::ParFiniteElementSpace fes( &mesh, &fec, 2 );
    mfem::ParGridFunction coordinates( &fes );
    mesh.GetNodes( coordinates );
    registerMfemCouplingScheme( 0, 0, 1, mesh, coordinates, { 1 }, { 3 }, SURFACE_TO_SURFACE, test.contact_case,
                                test.method, test.model, test.enforcement, BINNING_GRID, ExecutionMode::Sequential );
    setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
    EXPECT_THROW( setMfemUseCommonPlaneFormulation( 0, true ), std::invalid_argument );
    const auto& cs = CouplingSchemeManager::getInstance().at( 0 );
    EXPECT_FALSE( cs.hasContactFormulation() );
    EXPECT_FALSE( cs.hasMfemJacobianData() );
    finalize();
  }
}

// Curved coordinate spaces, actual LOR subdivision, and three-dimensional parents are rejected before any selection.
TEST( CommonPlaneSelection, rejects_higher_order_and_three_dimensional_geometry )
{
  for ( const int dimension : { 2, 3 } ) {
    const int order = dimension == 2 ? 2 : 1;
    mfem::ParMesh mesh =
        shared::ParMeshBuilder( MPI_COMM_WORLD, dimension == 2 ? shared::MeshBuilder::SquareMesh( 1, 1 )
                                                               : shared::MeshBuilder::CubeMesh( 1, 1, 1 ) );
    mfem::H1_FECollection fec( order, dimension );
    mfem::ParFiniteElementSpace fes( &mesh, &fec, dimension );
    mfem::ParGridFunction coordinates( &fes );
    mesh.GetNodes( coordinates );
    registerMfemCouplingScheme( 0, 0, 1, mesh, coordinates, { 1 }, { 3 }, SURFACE_TO_SURFACE, NO_CASE, COMMON_PLANE,
                                FRICTIONLESS, PENALTY, BINNING_GRID, ExecutionMode::Sequential );
    setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
    EXPECT_THROW( setMfemUseCommonPlaneFormulation( 0, true ), std::invalid_argument );
    auto& cs = CouplingSchemeManager::getInstance().at( 0 );
    EXPECT_FALSE( cs.hasContactFormulation() );
    EXPECT_FALSE( cs.hasMfemJacobianData() );
    if ( dimension == 2 ) {
      setMfemLORFactor( 0, 2 );
      EXPECT_THROW( setMfemUseCommonPlaneFormulation( 0, true ), std::invalid_argument );
    }
    finalize();
  }
}

INSTANTIATE_TEST_SUITE_P( quadrature, MfemCommonPlaneTest, testing::Values( 1, 3, 5 ) );
}  // namespace

/** @brief Run MFEM integration tests with MPI and SLIC initialized. @param argc Argument count. @param argv Arguments.
 * @return Test status. */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
