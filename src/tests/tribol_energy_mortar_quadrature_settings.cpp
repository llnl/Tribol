// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Tests factory selection and changes to Energy Mortar quadrature order for both enforcement locations.
 * Direct gap and force calls check returned outputs and repeated calculations without a decomposition refresh.
 * One- and two-rank runs also check parent-only storage, switching locations, and changed penalties; decomposition
 * onto one rank covers a contact rank with no elements. Directional differences check global energy and coordinate
 * derivatives. Signed nodal gaps check penalty activation separately from scalar field assembly. Local quadrature
 * accuracy is checked in tribol_pair_quadrature; gap and force derivatives have separate tests. Registration may
 * precede penalty configuration, but no penalty formulation is constructed until the settings are complete. Each
 * formulation retains its construction stiffness until replaced.
 * Direct force, Jacobian, and energy scaling checks use Sequential and OpenMP when enabled, with host mesh storage.
 * Scalar force-only helpers check active and empty pair batches without computing or returning a force Jacobian.
 * Public registration checks host execution modes and verifies that updates leave legacy face data unallocated.
 * Touching nonmatching surfaces retain the penalty tangent at zero gap, including zero nodal Hessian weights.
 */
#include <gtest/gtest.h>
#include <cmath>
#include <array>
#include <optional>
#include <limits>
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/ContactFormulationFactory.hpp"
#include "tribol/physics/EnergyMortar.hpp"
#include "tribol/physics/EnergyMortarEvaluation.hpp"
#include "tribol/physics/PenaltyStiffness.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/integ/QuadratureRuleGeneratorFactory.hpp"
#include "tribol/search/CoarseSearchFactory.hpp"
#include "tribol/search/ProjectedEdgeSearch.hpp"

namespace {
/** @brief Register an MFEM coupling scheme for each enforcement location and quadrature order. */
class EnergyMortarQuadratureSettingsTest : public testing::TestWithParam<std::tuple<tribol::EnforcementLocation, int>> {
 protected:
  /**
   * @brief Read the requested reference polynomial degree from the test parameters.
   *
   * @return Selected quadrature order.
   */
  int quadratureOrder() const { return std::get<1>( GetParam() ); }
  /** @brief Create parent mesh and coordinate fields after releasing earlier coupling schemes.
   * @param mesh_builder Parent body geometry and element attributes. */
  void createParentFields( shared::MeshBuilder mesh_builder )
  {
    coords_.reset();
    fes_.reset();
    fec_.reset();
    mesh_ = std::make_unique<mfem::ParMesh>(
        static_cast<const mfem::ParMesh&>( shared::ParMeshBuilder( MPI_COMM_WORLD, std::move( mesh_builder ) ) ) );
    fec_ = std::make_unique<mfem::H1_FECollection>( 1, 2 );
    fes_ = std::make_unique<mfem::ParFiniteElementSpace>( mesh_.get(), fec_.get(), 2 );
    coords_ = std::make_unique<mfem::ParGridFunction>( fes_.get() );
    mesh_->GetNodes( *coords_ );
  }
  /**
   * @brief Create the parent fields and register the scheme before penalty settings are supplied.
   * @param execution_mode Requested host execution mode, including Dynamic selection.
   */
  void registerScheme( tribol::ExecutionMode execution_mode = tribol::ExecutionMode::Sequential )
  {
    createParentFields( shared::MeshBuilder::SquareMesh( 2, 1 ) );
    tribol::registerMfemCouplingScheme( 0, 0, 1, *mesh_, *coords_, { 1 }, { 3 }, tribol::SURFACE_TO_SURFACE,
                                        tribol::NO_SLIDING, tribol::ENERGY_MORTAR, tribol::FRICTIONLESS,
                                        tribol::PENALTY, tribol::BINNING_GRID, execution_mode );
    EXPECT_EQ( tribol::CouplingSchemeManager::getInstance().findData( 0 )->getParameters().quadrature_order, 5 );
  }
  /** @brief Register the scheme and configure the requested quadrature and penalty settings. */
  void SetUp() override
  {
    registerScheme();
    tribol::setEnforcementLocation( 0, std::get<0>( GetParam() ) );
    tribol::setQuadratureOrder( 0, quadratureOrder() );
    tribol::setMfemKinematicConstantPenalty( 0, 100.0, 100.0 );
    ASSERT_NE( tribol::CouplingSchemeManager::getInstance().findData( 0 )->getContactFormulation(), nullptr );
  }
  /**
   * @brief Release coupling-scheme state after each test.
   */
  void TearDown() override { tribol::finalize(); }

  std::unique_ptr<mfem::ParMesh> mesh_;               ///< Owned parent mesh for registration.
  std::unique_ptr<mfem::H1_FECollection> fec_;        ///< Owned linear finite-element collection.
  std::unique_ptr<mfem::ParFiniteElementSpace> fes_;  ///< Owned vector space for parent-mesh coordinates.
  std::unique_ptr<mfem::ParGridFunction> coords_;     ///< Owned registered mesh-coordinate field.
};

// Public registration and update must accept host backends for both enforcement locations and each Gauss rule.
TEST_P( EnergyMortarQuadratureSettingsTest, public_registration_supports_host_execution_modes )
{
  using namespace tribol;
  std::optional<RealT> reference_energy;
  for ( const auto execution_mode : { ExecutionMode::Sequential, ExecutionMode::Dynamic
#ifdef TRIBOL_USE_OPENMP
                                      ,
                                      ExecutionMode::OpenMP
#endif
        } ) {
    SCOPED_TRACE( static_cast<int>( execution_mode ) );
    finalize();
    registerScheme( execution_mode );
    setEnforcementLocation( 0, std::get<0>( GetParam() ) );
    setQuadratureOrder( 0, quadratureOrder() );
    setMfemKinematicConstantPenalty( 0, 100.0, 100.0 );
    updateMfemParallelDecomposition( 1, true );
    auto& cs = CouplingSchemeManager::getInstance().at( 0 );
    const auto expected_mode = execution_mode == ExecutionMode::Dynamic
#ifdef TRIBOL_USE_OPENMP
                                   ? ExecutionMode::OpenMP
#else
                                   ? ExecutionMode::Sequential
#endif
                                   : execution_mode;
    EXPECT_EQ( cs.getExecutionMode(), expected_mode );
    RealT dt = 1.0;
    ASSERT_EQ( update( 0, 0.0, dt ), 0 );
    const auto& result = cs.getContactFormulationResult();
    ASSERT_NE( result.force, nullptr );
    ASSERT_NE( result.df_dx, nullptr );
    ASSERT_TRUE( result.energy.has_value() );
    EXPECT_GT( *result.energy, 0.0 );
    if ( reference_energy ) {
      EXPECT_NEAR( *result.energy, *reference_energy, 1.0e-11 );
    }
    reference_energy = result.energy;
    EXPECT_EQ( cs.getNumTotalNodes(), cs.getMesh1().numberOfNodes() );
    EXPECT_EQ( cs.getMethodData(), nullptr );
    for ( auto* mesh : { &cs.getMesh1(), &cs.getMesh2() } ) {
      EXPECT_TRUE( mesh->getView().getElementAreas().empty() );
      EXPECT_TRUE( mesh->getView().getElementCentroids().empty() );
      EXPECT_TRUE( mesh->getView().getElementNormals().empty() );
    }
  }
}

// Coincident nonmatching boundaries have zero force and zero Hessian weights, but a nonzero compressive tangent.
TEST_P( EnergyMortarQuadratureSettingsTest, zero_gap_keeps_penalty_tangent )
{
  using namespace tribol;
  finalize();
  createParentFields(
      shared::MeshBuilder::Unify( { shared::MeshBuilder::SquareMesh( 2, 1 ), shared::MeshBuilder::SquareMesh( 3, 1 )
                                                                                 .translate( { 0.0, 1.0 } )
                                                                                 .updateBdrAttrib( 1, 5 )
                                                                                 .updateBdrAttrib( 3, 6 )
                                                                                 .updateAttrib( 1, 2 ) } ) );
  registerMfemCouplingScheme( 0, 0, 1, *mesh_, *coords_, { 3 }, { 5 }, SURFACE_TO_SURFACE, NO_CASE, ENERGY_MORTAR,
                              FRICTIONLESS, PENALTY, BINNING_GRID, ExecutionMode::Sequential );
  setEnforcementLocation( 0, std::get<0>( GetParam() ) );
  setQuadratureOrder( 0, quadratureOrder() );
  setMfemKinematicConstantPenalty( 0, 4.0, 4.0 );
  updateMfemParallelDecomposition( 1, true );
  RealT dt = 1.0;
  ASSERT_EQ( update( 0, 0.0, dt ), 0 );
  const auto& result = CouplingSchemeManager::getInstance().at( 0 ).getContactFormulationResult();
  ASSERT_TRUE( result.energy.has_value() );
  ASSERT_NE( result.force, nullptr );
  ASSERT_NE( result.df_dx, nullptr );
  EXPECT_NEAR( *result.energy, 0.0, 1.0e-12 );
  EXPECT_NEAR( mfem::InnerProduct( MPI_COMM_WORLD, *result.force, *result.force ), 0.0, 1.0e-12 );
  mfem::Vector direction( result.df_dx->Width() ), action( result.df_dx->Height() );
  for ( int dof = 0; dof < direction.Size(); ++dof ) {
    direction[dof] = dof + 1.0;
  }
  result.df_dx->Mult( direction, action );
  EXPECT_GT( mfem::InnerProduct( MPI_COMM_WORLD, action, action ), 1.0e-12 );
}

// Supported orders configure either enforcement location; rejected orders must leave the selected object intact.
TEST_P( EnergyMortarQuadratureSettingsTest, configures_and_rebuilds_formulation )
{
  auto* cs = tribol::CouplingSchemeManager::getInstance().findData( 0 );
  const auto location = std::get<0>( GetParam() );
  const auto check = [&]( int order ) {
    tribol::setQuadratureOrder( 0, order );
    auto* formulation = cs->getContactFormulation();
    ASSERT_NE( formulation, nullptr );
    EXPECT_EQ( cs->getParameters().enforcement_location, location );
    EXPECT_EQ( cs->getParameters().quadrature_order, order );
  };
  check( quadratureOrder() );
  for ( int order = 0; order <= 5; ++order ) {
    check( order );
  }
  for ( int invalid : { -1, 6 } ) {
    auto* before = cs->getContactFormulation();
    EXPECT_THROW( tribol::setQuadratureOrder( 0, invalid ), std::invalid_argument );
    EXPECT_EQ( cs->getParameters().quadrature_order, 5 );
    EXPECT_EQ( cs->getContactFormulation(), before );
    cs->getParameters().quadrature_order = invalid;
    EXPECT_THROW( tribol::createContactFormulation( cs ), std::invalid_argument );
    cs->getParameters().quadrature_order = 5;
  }
  cs->setContactFormulation( nullptr );
  tribol::setQuadratureOrder( 0, 2 );
  EXPECT_FALSE( cs->hasContactFormulation() );
  cs->updateContactFormulation();
  EXPECT_TRUE( cs->hasContactFormulation() );
  EXPECT_EQ( cs->getParameters().enforcement_location, location );
}

// Gap and force calls must work independently, initialize only their outputs, and leave earlier results unchanged.
TEST_P( EnergyMortarQuadratureSettingsTest, returns_owned_outputs_for_each_calculation )
{
  tribol::updateMfemParallelDecomposition();
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  const auto& formulation = *cs.getContactFormulation();
  const auto mesh1 = cs.getMesh1().getView();
  const auto mesh2 = cs.getMesh2().getView();
  mfem::Vector legacy_gap;
  EXPECT_THROW( tribol::getMfemGap( 0, legacy_gap ), std::logic_error );
  if ( std::get<0>( GetParam() ) == tribol::EnforcementLocation::Nodal ) {
    auto gaps = formulation.computeNodalGaps( mesh1, mesh2 );
    ASSERT_NE( gaps.gap, nullptr );
    ASSERT_NE( gaps.dg_dx, nullptr );
    EXPECT_EQ( gaps.force, nullptr );
    EXPECT_EQ( gaps.pressure, nullptr );
    EXPECT_EQ( gaps.df_dx, nullptr );
    EXPECT_EQ( gaps.df_dp, nullptr );
    EXPECT_FALSE( gaps.energy.has_value() );
    EXPECT_FALSE( gaps.time_step.has_value() );
    const mfem::HypreParVector original( *gaps.gap );
    auto repeated = formulation.computeNodalGaps( mesh1, mesh2 );
    repeated.gap->Add( -1.0, original );
    EXPECT_LT( repeated.gap->Norml2(), 1.0e-12 );
    EXPECT_EQ( gaps.gap->Norml2(), original.Norml2() );
  } else {
    EXPECT_THROW( formulation.computeNodalGaps( mesh1, mesh2 ), std::logic_error );
  }

  auto first = formulation.computeNodalForces( mesh1, mesh2 );
  ASSERT_NE( first.force, nullptr );
  ASSERT_NE( first.df_dx, nullptr );
  ASSERT_TRUE( first.energy.has_value() );
  EXPECT_FALSE( first.time_step.has_value() );
  EXPECT_EQ( first.df_dp, nullptr );
  const bool nodal = std::get<0>( GetParam() ) == tribol::EnforcementLocation::Nodal;
  EXPECT_EQ( first.gap != nullptr, nodal );
  EXPECT_EQ( first.pressure != nullptr, nodal );
  EXPECT_EQ( first.dg_dx != nullptr, nodal );
  const mfem::HypreParVector original( *first.force );
  ASSERT_GT( original.Norml2(), 0.0 );
  auto second = formulation.computeNodalForces( mesh1, mesh2 );
  second.force->Add( -1.0, original );
  EXPECT_LT( second.force->Norml2(), 1.0e-12 );
  EXPECT_NEAR( *second.energy, *first.energy, 1.0e-12 );
  EXPECT_EQ( cs.getContactFormulationResult().force, nullptr );
  cs.setContactFormulation( nullptr );
  auto difference = *first.force;
  difference -= original;
  EXPECT_EQ( difference.Norml2(), 0.0 );
  mfem::Vector direction( first.df_dx->Width() ), action( first.df_dx->Height() );
  direction = 1.0;
  first.df_dx->Mult( direction, action );
  EXPECT_TRUE( std::isfinite( action.Norml2() ) );
}

// Springs in series must scale force, Jacobian, and energy on each host backend for either location and Gauss count.
TEST_P( EnergyMortarQuadratureSettingsTest, accepts_series_penalty_stiffness )
{
  using namespace tribol;
  setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
  updateMfemParallelDecomposition( 1, true );
  auto& cs = CouplingSchemeManager::getInstance().at( 0 );
  const auto baseline =
      cs.getContactFormulation()->computeNodalForces( cs.getMesh1().getView(), cs.getMesh2().getView() );
  mfem::Vector direction( baseline.df_dx->Width() ), reference_action( baseline.df_dx->Height() );
  for ( int dof = 0; dof < direction.Size(); ++dof ) {
    direction[dof] = 1.0 + 0.1 * dof;
  }
  baseline.df_dx->Mult( direction, reference_action );
  for ( const auto execution_mode : { ExecutionMode::Sequential
#ifdef TRIBOL_USE_OPENMP
                                      ,
                                      ExecutionMode::OpenMP
#endif
        } ) {
    SCOPED_TRACE( static_cast<int>( execution_mode ) );
    auto alternate = dispatchExecutionMode( execution_mode, [&]<ExecutionMode Exec>() {
      return createWithQuadratureRuleGenerator(
          quadratureOrder(), SmoothedProjectedEdgeMapping{},
          [&]( auto rule_generator ) -> std::unique_ptr<ContactFormulation> {
            auto coarse = createCoarseSearch( cs.spatialDimension(), cs.getBinningMethod(), execution_mode,
                                              cs.getEffectiveBinningProximityScale(), 0.0 );
            auto refined = std::make_unique<ProjectedEdgeSearch>( execution_mode, false );
            PairListBuilder pair_builder{ std::move( coarse ), std::move( refined ) };
            if ( std::get<0>( GetParam() ) == EnforcementLocation::Nodal ) {
              EnergyMortarNodalEnergy<decltype( rule_generator ), OffsetGap<OpposingNormalGap>> pair_function{
                  std::move( rule_generator ), OffsetGap{ OpposingNormalGap{}, 0.0 } };
              return std::make_unique<NodalPenaltyEnergyMortar<Exec, decltype( pair_function )>>(
                  *cs.getMfemMeshData(), *cs.getMfemSubmeshData(), *cs.getMfemJacobianData(),
                  std::move( pair_function ), PenaltyEnforcement{ SeriesPenaltyStiffness{}( 2.0, 6.0 ) },
                  std::move( pair_builder ) );
            }
            EnergyMortarEnergy<decltype( rule_generator ), OffsetGap<OpposingNormalGap>> pair_function{
                std::move( rule_generator ), SeriesPenaltyStiffness{}( 2.0, 6.0 ),
                OffsetGap{ OpposingNormalGap{}, 0.0 } };
            return std::make_unique<QuadraturePointEnergyMortar<Exec, decltype( pair_function )>>(
                *cs.getMfemMeshData(), *cs.getMfemJacobianData(), std::move( pair_function ),
                std::move( pair_builder ) );
          } );
    } );
    auto result = alternate->computeNodalForces( cs.getMesh1().getView(), cs.getMesh2().getView() );
    ASSERT_NE( result.force, nullptr );
    ASSERT_NE( result.df_dx, nullptr );
    ASSERT_TRUE( result.energy.has_value() );
    // The stiffnesses 2 and 6 give series stiffness 1.5 instead of the default arithmetic mean 4.
    constexpr RealT stiffness_ratio = 1.5 / 4.0;
    result.force->Add( -stiffness_ratio, *baseline.force );
    EXPECT_LT( result.force->Norml2(), 1.0e-11 );
    EXPECT_NEAR( *result.energy, stiffness_ratio * *baseline.energy, 1.0e-11 );
    mfem::Vector action( result.df_dx->Height() );
    result.df_dx->Mult( direction, action );
    action.Add( -stiffness_ratio, reference_action );
    EXPECT_LT( action.Norml2(), 1.0e-11 );
  }
}

// Scalar force-only evaluation must match the energy/Jacobian path while leaving matrix outputs absent.
TEST_P( EnergyMortarQuadratureSettingsTest, quadrature_force_only_matches_energy_and_force )
{
  using namespace tribol;
  if ( std::get<0>( GetParam() ) != EnforcementLocation::QuadraturePoint ) {
    GTEST_SKIP() << "This check requires quadrature-point energy.";
  }
  updateMfemParallelDecomposition( 1, true );
  auto& cs = CouplingSchemeManager::getInstance().at( 0 );
  const auto mesh1 = cs.getMesh1().getView();
  const auto mesh2 = cs.getMesh2().getView();
  const auto baseline = cs.getContactFormulation()->computeNodalForces( mesh1, mesh2 );
  ASSERT_TRUE( baseline.energy.has_value() );
  ASSERT_GT( *baseline.energy, 0.0 );
  struct ForceOnlyResults {
    ContactFormulationResult empty;   ///< Result for an empty accepted-pair list.
    ContactFormulationResult active;  ///< Result for the searched contact pairs.
  };
  for ( const auto mode : { ExecutionMode::Sequential
#ifdef TRIBOL_USE_OPENMP
                            ,
                            ExecutionMode::OpenMP
#endif
        } ) {
    SCOPED_TRACE( static_cast<int>( mode ) );
    auto results = dispatchExecutionMode( mode, [&]<ExecutionMode Exec>() {
      return createWithQuadratureRuleGenerator(
          quadratureOrder(), SmoothedProjectedEdgeMapping{}, [&]( auto rule_generator ) -> ForceOnlyResults {
            EnergyMortarEnergy<decltype( rule_generator ), OffsetGap<OpposingNormalGap>> pair_function{
                std::move( rule_generator ), 100.0, OffsetGap{ OpposingNormalGap{}, 0.0 } };
            const ArrayT<InterfacePair> empty_pairs( 0, 0, cs.getMesh1().getAllocatorId() );
            auto empty =
                evaluateAndAssemblePairEnergy<Exec>( empty_pairs, mesh1, mesh2, *cs.getMfemMeshData(), pair_function );
            auto coarse = createCoarseSearch( cs.spatialDimension(), cs.getBinningMethod(), mode,
                                              cs.getEffectiveBinningProximityScale(), 0.0 );
            auto refined = std::make_unique<ProjectedEdgeSearch>( mode, false );
            const auto pairs = PairListBuilder{ std::move( coarse ), std::move( refined ) }.build( mesh1, mesh2 );
            auto active =
                evaluateAndAssemblePairEnergy<Exec>( pairs, mesh1, mesh2, *cs.getMfemMeshData(), pair_function );
            return { std::move( empty ), std::move( active ) };
          } );
    } );
    ASSERT_NE( results.empty.force, nullptr );
    ASSERT_TRUE( results.empty.energy.has_value() );
    EXPECT_EQ( *results.empty.energy, 0.0 );
    EXPECT_EQ( results.empty.force->Norml2(), 0.0 );
    EXPECT_EQ( results.empty.df_dx, nullptr );
    auto& result = results.active;
    ASSERT_NE( result.force, nullptr );
    ASSERT_TRUE( result.energy.has_value() );
    EXPECT_NEAR( *result.energy, *baseline.energy, 1.0e-11 );
    result.force->Add( -1.0, *baseline.force );
    EXPECT_LT( result.force->Norml2(), 1.0e-11 );
    EXPECT_EQ( result.df_dx, nullptr );
    EXPECT_EQ( result.gap, nullptr );
    EXPECT_EQ( result.pressure, nullptr );
  }
}

// Unsupported locations must leave the selected formulation and its transfer fields unchanged.
TEST_P( EnergyMortarQuadratureSettingsTest, rejects_invalid_location_without_rebuilding )
{
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  auto* formulation = cs.getContactFormulation();
  auto* fields = cs.getMfemSubmeshData();
  const auto location = cs.getParameters().enforcement_location;
  constexpr auto invalid_location = static_cast<tribol::EnforcementLocation>( -1 );
  EXPECT_THROW( tribol::setEnforcementLocation( 0, invalid_location ), std::invalid_argument );
  EXPECT_EQ( cs.getParameters().enforcement_location, location );
  EXPECT_EQ( cs.getContactFormulation(), formulation );
  EXPECT_EQ( cs.getMfemSubmeshData(), fields );
}

// An unconfigured penalty must report an error before a pair formulation can calculate forces.
TEST_P( EnergyMortarQuadratureSettingsTest, rejects_unconfigured_pair_penalty )
{
  tribol::setEnforcementLocation( 0, tribol::EnforcementLocation::QuadraturePoint );
  tribol::updateMfemParallelDecomposition( 1, true );
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  cs.getEnforcementOptions().penalty_options.kinematic_calc_set = false;
  cs.updateContactFormulation();
  EXPECT_FALSE( cs.hasContactFormulation() );
  tribol::RealT dt = 1.0;
  EXPECT_THROW( tribol::update( 0, 0.0, dt ), std::logic_error );
  EXPECT_FALSE( cs.hasMfemSubmeshData() );
}

// Existing penalty calculations retain their construction stiffness even if registered constants change or disappear.
TEST_P( EnergyMortarQuadratureSettingsTest, keeps_penalty_fixed_until_formulation_is_rebuilt )
{
  tribol::updateMfemParallelDecomposition( 1, true );
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  auto& mesh_data = *cs.getMfemMeshData();
  const auto calculate = [&] {
    return cs.getContactFormulation()->computeNodalForces( cs.getMesh1().getView(), cs.getMesh2().getView() );
  };
  const auto initial = calculate();
  const auto check_unchanged = [&] {
    auto result = calculate();
    result.force->Add( -1.0, *initial.force );
    EXPECT_LT( result.force->Norml2(), 1.0e-12 );
    EXPECT_EQ( result.energy, initial.energy );
  };
  mesh_data.ClearAllPenaltyData();
  check_unchanged();
  EXPECT_EQ( tribol::createContactFormulation( &cs ), nullptr );
  mesh_data.SetMesh1KinematicConstantPenalty( 100.0 );
  check_unchanged();
  EXPECT_EQ( tribol::createContactFormulation( &cs ), nullptr );
  mesh_data.ClearAllPenaltyData();
  mesh_data.SetMesh2KinematicConstantPenalty( 100.0 );
  check_unchanged();
  EXPECT_EQ( tribol::createContactFormulation( &cs ), nullptr );
  mesh_data.SetMesh1KinematicConstantPenalty( 300.0 );
  check_unchanged();
  // The public setter builds a new formulation using the new mean stiffness, without refreshing the decomposition.
  tribol::setMfemKinematicConstantPenalty( 0, 300.0, 100.0 );
  auto rebuilt = calculate();
  rebuilt.force->Add( -2.0, *initial.force );
  EXPECT_LT( rebuilt.force->Norml2(), 1.0e-12 );
  EXPECT_NEAR( *rebuilt.energy, 2.0 * *initial.energy, 1.0e-12 );
}

/** @brief Start tests with a coupling scheme registered before its penalty settings are supplied. */
class EnergyMortarRegistrationTest : public EnergyMortarQuadratureSettingsTest {
 protected:
  /** @brief Register the scheme while leaving penalty selection and constants unset. */
  void SetUp() override { registerScheme(); }
};

// The penalty setter constructs a complete formulation before decomposition refresh and update().
TEST_P( EnergyMortarRegistrationTest, constructs_formulation_before_decomposition )
{
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  EXPECT_FALSE( cs.hasContactFormulation() );
  tribol::setEnforcementLocation( 0, std::get<0>( GetParam() ) );
  tribol::setQuadratureOrder( 0, quadratureOrder() );
  EXPECT_FALSE( cs.hasContactFormulation() );
  tribol::RealT dt = 1.0;
  EXPECT_THROW( tribol::update( 0, 0.0, dt ), std::logic_error );
  tribol::setMfemKinematicConstantPenalty( 0, 100.0, 100.0 );
  ASSERT_TRUE( cs.hasContactFormulation() );
  const auto location = std::get<0>( GetParam() );
  EXPECT_EQ( cs.getParameters().enforcement_location, location );
  EXPECT_EQ( cs.getParameters().quadrature_order, quadratureOrder() );
  tribol::updateMfemParallelDecomposition( 1, true );
  ASSERT_EQ( tribol::update( 0, 0.0, dt ), 0 );
  EXPECT_GT( tribol::getMfemContactForce( 0 ).Norml2(), 0.0 );
}

// Configuring after decomposition also initializes the correct field rows and requires no further refresh.
TEST_P( EnergyMortarRegistrationTest, constructs_formulation_on_existing_decomposition )
{
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  tribol::setEnforcementLocation( 0, std::get<0>( GetParam() ) );
  tribol::setQuadratureOrder( 0, quadratureOrder() );
  tribol::updateMfemParallelDecomposition( 1, true );
  EXPECT_FALSE( cs.hasContactFormulation() );
  tribol::setMfemKinematicConstantPenalty( 0, 100.0, 100.0 );
  ASSERT_TRUE( cs.hasContactFormulation() );
  const bool nodal = std::get<0>( GetParam() ) == tribol::EnforcementLocation::Nodal;
  EXPECT_EQ( cs.getMesh1().getNodalFields().m_is_node_gap_set, nodal );
  EXPECT_FALSE( cs.getMesh2().getNodalFields().m_is_node_gap_set );
  const auto result =
      cs.getContactFormulation()->computeNodalForces( cs.getMesh1().getView(), cs.getMesh2().getView() );
  ASSERT_NE( result.force, nullptr );
  ASSERT_TRUE( result.energy.has_value() );
  EXPECT_GT( result.force->Norml2(), 0.0 );
}

// Invalid fixed stiffness is rejected before a penalty enforcement value can be constructed.
TEST( EnergyMortarPenalty, rejects_invalid_fixed_stiffness )
{
  EXPECT_THROW( tribol::PenaltyEnforcement{ 0.0 }, std::invalid_argument );
  EXPECT_THROW( tribol::PenaltyEnforcement{ -1.0 }, std::invalid_argument );
  EXPECT_THROW( tribol::PenaltyEnforcement{ std::numeric_limits<tribol::RealT>::infinity() }, std::invalid_argument );
  EXPECT_EQ( tribol::PenaltyEnforcement{ 2.0 }.stiffness(), 2.0 );
}

INSTANTIATE_TEST_SUITE_P( enforcement, EnergyMortarRegistrationTest,
                          testing::Combine( testing::Values( tribol::EnforcementLocation::Nodal,
                                                             tribol::EnforcementLocation::QuadraturePoint ),
                                            testing::Values( 1, 3, 5 ) ) );

// Parent-only pair enforcement must never allocate or register pressure/gap fields, including after decomposition.
TEST_P( EnergyMortarQuadratureSettingsTest, creates_only_the_required_transfer_fields )
{
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  const bool nodal = std::get<0>( GetParam() ) == tribol::EnforcementLocation::Nodal;
  EXPECT_EQ( cs.hasMfemSubmeshData(), nodal );
  ASSERT_TRUE( cs.hasMfemJacobianData() );
  tribol::updateMfemParallelDecomposition( 1, true );
  EXPECT_EQ( cs.hasMfemSubmeshData(), nodal );
  EXPECT_EQ( cs.getMfemJacobianData()->ParentPath().final_fes, fes_.get() );
  EXPECT_EQ( cs.getMesh1().getNodalFields().m_is_node_gap_set, nodal );
  EXPECT_EQ( cs.getMesh1().getNodalFields().m_is_node_pressure_set, nodal );
  EXPECT_FALSE( cs.getMesh2().getNodalFields().m_is_node_gap_set );
  EXPECT_FALSE( cs.getMesh2().getNodalFields().m_is_node_pressure_set );
}

// Switching locations must initialize new transfers immediately and reproduce each location's previous outputs.
TEST_P( EnergyMortarQuadratureSettingsTest, switches_locations_without_rebuilding_the_decomposition )
{
  tribol::updateMfemParallelDecomposition( 1, true );
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  std::array<std::unique_ptr<mfem::HypreParVector>, 2> previous_forces;
  std::array<std::optional<tribol::RealT>, 2> previous_energies;
  for ( const auto location : { tribol::EnforcementLocation::Nodal, tribol::EnforcementLocation::QuadraturePoint,
                                tribol::EnforcementLocation::Nodal, tribol::EnforcementLocation::QuadraturePoint } ) {
    tribol::setEnforcementLocation( 0, location );
    const bool nodal = location == tribol::EnforcementLocation::Nodal;
    EXPECT_EQ( cs.hasMfemSubmeshData(), nodal );
    EXPECT_EQ( cs.getContactFormulationResult().force, nullptr );
    EXPECT_EQ( cs.getMesh1().getNodalFields().m_is_node_gap_set, nodal );
    EXPECT_EQ( cs.getMesh1().getNodalFields().m_is_node_pressure_set, nodal );
    auto result = cs.getContactFormulation()->computeNodalForces( cs.getMesh1().getView(), cs.getMesh2().getView() );
    ASSERT_NE( result.force, nullptr );
    ASSERT_NE( result.df_dx, nullptr );
    ASSERT_TRUE( result.energy.has_value() );
    EXPECT_EQ( result.gap != nullptr, nodal );
    EXPECT_EQ( result.pressure != nullptr, nodal );
    EXPECT_EQ( result.dg_dx != nullptr, nodal );
    const int index = nodal ? 0 : 1;
    if ( previous_forces[index] ) {
      auto difference = *result.force;
      difference -= *previous_forces[index];
      EXPECT_LT( difference.Norml2(), 1.0e-12 );
      EXPECT_NEAR( *result.energy, *previous_energies[index], 1.0e-12 );
    } else {
      previous_forces[index] = std::make_unique<mfem::HypreParVector>( *result.force );
      previous_energies[index] = result.energy;
    }
    mfem::Vector direction( result.df_dx->Width() ), action( result.df_dx->Height() );
    direction = 1.0;
    result.df_dx->Mult( direction, action );
    EXPECT_TRUE( std::isfinite( action.Norml2() ) );
  }
}

// Updating the two penalties must use their arithmetic mean without requiring geometry transfer to run again.
TEST_P( EnergyMortarQuadratureSettingsTest, uses_updated_penalties_without_redecomposition )
{
  tribol::updateMfemParallelDecomposition( 1, true );
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
  auto initial = cs.getContactFormulation()->computeNodalForces( cs.getMesh1().getView(), cs.getMesh2().getView() );
  // Retain only parent-space outputs while rebuilding the formulation.
  const auto force = *initial.force;
  const auto energy = *initial.energy;
  initial = {};
  tribol::setMfemKinematicConstantPenalty( 0, 100.0, 300.0 );
  auto updated = cs.getContactFormulation()->computeNodalForces( cs.getMesh1().getView(), cs.getMesh2().getView() );
  updated.force->Add( -2.0, force );
  EXPECT_LT( updated.force->Norml2(), 1.0e-11 );
  EXPECT_NEAR( *updated.energy, 2.0 * energy, 1.0e-11 );
}

// Penalty activation must remove positive weighted gaps and their derivative rows; LM gaps remain signed and weighted.
TEST_P( EnergyMortarQuadratureSettingsTest, nodal_activation_preserves_only_compressive_gap_rows )
{
  using namespace tribol;
  if ( std::get<0>( GetParam() ) != EnforcementLocation::Nodal ) {
    GTEST_SKIP() << "This check requires nodal enforcement.";
  }
  // The top boundary crosses the bottom boundary, producing both opening and penetration along the same interface.
  for ( int node = 0; node < fes_->GetNDofs(); ++node ) {
    if ( ( *coords_ )[fes_->DofToVDof( node, 1 )] > 0.5 ) {
      const RealT x = ( *coords_ )[fes_->DofToVDof( node, 0 )];
      ( *coords_ )[fes_->DofToVDof( node, 1 )] = 0.2 - 0.4 * x;
    }
  }
  updateMfemParallelDecomposition( 1, true );
  auto& cs = CouplingSchemeManager::getInstance().at( 0 );
  const auto penalty = cs.getContactFormulation()->computeNodalGaps( cs.getMesh1().getView(), cs.getMesh2().getView() );
  const auto weighted =
      createWithQuadratureRuleGenerator( quadratureOrder(), SmoothedProjectedEdgeMapping{}, [&]( auto rule_generator ) {
        auto coarse = createCoarseSearch( cs.spatialDimension(), cs.getBinningMethod(), cs.getExecutionMode(),
                                          cs.getEffectiveBinningProximityScale(), 0.0 );
        auto refined = std::make_unique<ProjectedEdgeSearch>( cs.getExecutionMode(), false );
        PairListBuilder pair_builder{ std::move( coarse ), std::move( refined ) };
        EnergyMortarNodalEnergy<decltype( rule_generator ), OffsetGap<OpposingNormalGap>> pair_function{
            std::move( rule_generator ), OffsetGap{ OpposingNormalGap{}, 0.0 } };
        NodalLagrangeMultiplierEnergyMortar<ExecutionMode::Sequential, decltype( pair_function )> lm(
            *cs.getMfemMeshData(), *cs.getMfemSubmeshData(), *cs.getMfemJacobianData(), std::move( pair_function ),
            std::move( pair_builder ) );
        return lm.computeNodalGaps( cs.getMesh1().getView(), cs.getMesh2().getView() );
      } );
  mfem::Vector direction( fes_->GetTrueVSize() ), penalty_action( penalty.gap->Size() ),
      weighted_action( weighted.gap->Size() );
  direction = 0.0;
  for ( int dof = 0; dof < direction.Size(); ++dof ) {
    direction[dof] = dof + 1.0;
  }
  penalty.dg_dx->Mult( direction, penalty_action );
  weighted.dg_dx->Mult( direction, weighted_action );
  int local_open = 0, local_closed = 0;
  for ( int node = 0; node < weighted.gap->Size(); ++node ) {
    if ( ( *weighted.gap )[node] > 1.0e-12 ) {
      ++local_open;
      EXPECT_EQ( ( *penalty.gap )[node], 0.0 );
      EXPECT_NEAR( penalty_action[node], 0.0, 1.0e-12 );
    } else if ( ( *weighted.gap )[node] < -1.0e-12 ) {
      ++local_closed;
      EXPECT_LT( ( *penalty.gap )[node], 0.0 );
      EXPECT_NEAR( penalty_action[node], weighted_action[node], 1.0e-12 );
    }
  }
  int global_open = 0, global_closed = 0;
  MPI_Allreduce( &local_open, &global_open, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_closed, &global_closed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( global_open, 0 );
  EXPECT_GT( global_closed, 0 );
}

// Assembled gradients and Hessians must match finite differences as gap, area, and edge length vary.
TEST_P( EnergyMortarQuadratureSettingsTest, assembled_force_and_jacobian_match_directional_finite_difference )
{
  tribol::setResidualGap( 0, 1.2 );
  mfem::ParGridFunction base_coords( *coords_ );
  mfem::ParGridFunction direction_gf( fes_.get() );
  mfem::VectorFunctionCoefficient direction_coeff( 2, []( const mfem::Vector& x, mfem::Vector& v ) {
    v.SetSize( 2 );
    v[0] = 0.3 * x[0] * ( 1.0 - x[0] ) + 0.1 * x[1];
    v[1] = -0.2 + 0.15 * x[0] - 0.1 * x[1];
  } );
  direction_gf.ProjectCoefficient( direction_coeff );
  mfem::HypreParVector direction_true( fes_.get() );
  direction_gf.GetTrueDofs( direction_true );

  auto evaluate_at = [&]( tribol::RealT eps ) {
    *coords_ = base_coords;
    coords_->Add( eps, direction_gf );
    tribol::updateMfemParallelDecomposition( 1, true );
    auto& cs = tribol::CouplingSchemeManager::getInstance().at( 0 );
    auto out = cs.getContactFormulation()->computeNodalForces( cs.getMesh1().getView(), cs.getMesh2().getView() );
    return out;
  };

  const auto base = evaluate_at( 0.0 );
  constexpr tribol::RealT step = 1.0e-6;
  const auto plus = evaluate_at( step );
  const auto minus = evaluate_at( -step );

  const tribol::RealT directional_force = mfem::InnerProduct( MPI_COMM_WORLD, *base.force, direction_true );
  const tribol::RealT fd_energy = ( *plus.energy - *minus.energy ) / ( 2.0 * step );
  EXPECT_NEAR( directional_force, fd_energy, 1.0e-5 );

  mfem::Vector jv( fes_.get()->GetTrueVSize() );
  base.df_dx->Mult( direction_true, jv );
  mfem::Vector fd_force( *plus.force );
  fd_force -= *minus.force;
  fd_force *= 1.0 / ( 2.0 * step );
  jv -= fd_force;
  const tribol::RealT jv_err = std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, jv, jv ) );
  EXPECT_LT( jv_err, 2.0e-4 );
}

/** @brief Host edges used to register an unsupported contact method. */
struct Edges {
  std::array<tribol::IndexT, 2> connectivity{ 0, 1 };  ///< Endpoint connectivity shared by both test segments.
  std::array<tribol::RealT, 2> x1{ 0.0, 2.0 };         ///< First-edge endpoint x coordinates.
  std::array<tribol::RealT, 2> y1{ 0.0, 0.0 };         ///< First-edge endpoint y coordinates.
  std::array<tribol::RealT, 2> x2{ 2.0, 0.0 };         ///< Oppositely ordered second-edge endpoint x coordinates.
  std::array<tribol::RealT, 2> y2{ 0.1, 0.1 };         ///< Target endpoint y coordinates.
  /// @brief First mesh containing one edge and using this object's coordinate and connectivity arrays.
  tribol::MeshData mesh1{
      0,         1,         2,       connectivity.data(),      tribol::EDGE, tribol::LINEAR_LAGRANGE,
      x1.data(), y1.data(), nullptr, tribol::MemorySpace::Host };
  /// @brief Second mesh containing one edge and using this object's coordinate and connectivity arrays.
  tribol::MeshData mesh2{
      1,         1,         2,       connectivity.data(),      tribol::EDGE, tribol::LINEAR_LAGRANGE,
      x2.data(), y2.data(), nullptr, tribol::MemorySpace::Host };
};

// Unsupported methods and missing schemes must reject the setting without modifying configuration.
TEST( ContactFormulation, rejects_order_setting_for_unsupported_schemes )
{
  EXPECT_THROW( tribol::setQuadratureOrder( 0, 2 ), std::invalid_argument );
  Edges edges;
  tribol::registerMesh( 0, 1, edges.x1.size(), edges.connectivity.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE,
                        edges.x1.data(), edges.y1.data() );
  tribol::registerMesh( 1, 1, edges.x2.size(), edges.connectivity.data(), tribol::EDGE, tribol::LINEAR_LAGRANGE,
                        edges.x2.data(), edges.y2.data() );
  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );
  EXPECT_THROW( tribol::setQuadratureOrder( 0, 2 ), std::invalid_argument );
  EXPECT_EQ( tribol::CouplingSchemeManager::getInstance().findData( 0 )->getParameters().quadrature_order, 5 );
  tribol::finalize();
}

/** @brief Exercise both enforcement locations with one, two, and three Gauss points. */
INSTANTIATE_TEST_SUITE_P( enforcement, EnergyMortarQuadratureSettingsTest,
                          testing::Combine( testing::Values( tribol::EnforcementLocation::Nodal,
                                                             tribol::EnforcementLocation::QuadraturePoint ),
                                            testing::Values( 1, 3, 5 ) ) );

}  // namespace

/**
 * @brief Initialize MPI and logging, run formulation tests, and finalize MPI.
 *
 * @param [in,out] argc Command-line argument count; MPI and Google Test may consume arguments.
 * @param [in,out] argv Command-line argument strings.
 * @return Google Test exit status.
 */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
