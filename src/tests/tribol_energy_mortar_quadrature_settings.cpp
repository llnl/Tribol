// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Tests factory selection and changes to Energy Mortar quadrature order for both enforcement locations.
 * Direct gap and residual calls check returned outputs and repeated calculations without a decomposition refresh.
 * One- and two-rank runs also check parent-only storage, switching locations, and changed penalties; decomposition
 * onto one rank covers a contact rank with no elements. Directional differences check global energy and coordinate
 * derivatives. Signed nodal gaps check penalty activation separately from scalar field assembly. Local quadrature
 * accuracy is checked in tribol_pair_quadrature; gap and residual derivatives have separate tests. Registration may
 * precede penalty configuration; update() constructs one formulation after the settings are complete. Each formulation
 * retains its construction stiffness until invalidated.
 * Direct residual, Jacobian, and energy scaling checks use Sequential and OpenMP when enabled, with host mesh storage.
 * Residual-only helpers check active and empty pair lists without computing or returning a residual Jacobian.
 * Public registration checks host execution modes and verifies that updates leave legacy face data unallocated.
 * Touching nonmatching surfaces retain the compressive-side derivative at zero gap, including zero nodal Hessian
 * weights.
 */
#include <gtest/gtest.h>
#include "SlicErrorTest.hpp"
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
#include "tribol/physics/integrals/PairIntegrals.hpp"
#include "tribol/physics/NodalEnergyMortar.hpp"
#include "tribol/physics/QuadraturePointEnergyMortar.hpp"
#include "tribol/physics/ContactFormulationFactory.hpp"
#include "tribol/enforcement/PenaltyStiffness.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/integ/QuadratureRuleGeneratorFactory.hpp"
#include "tribol/search/CoarseSearchFactory.hpp"
#include "tribol/search/ProjectedEdgeSearch.hpp"

namespace {
constexpr tribol::IndexT coupling_scheme_id = 0;      ///< Coupling scheme owned by each test fixture.
constexpr tribol::IndexT first_mesh_id = 0;           ///< First registered contact mesh.
constexpr tribol::IndexT second_mesh_id = 1;          ///< Second registered contact mesh.
constexpr int contact_rank_count = 1;                 ///< Decompose contact onto one rank, including in two-rank tests.
constexpr tribol::RealT configured_penalty = 100.0;   ///< Default equal penalty assigned to both test surfaces.
constexpr tribol::RealT first_series_penalty = 2.0;   ///< First stiffness in series-penalty checks.
constexpr tribol::RealT second_series_penalty = 6.0;  ///< Second stiffness in series-penalty checks.
constexpr int one_point_polynomial_order = tribol::EdgeQuadratureRuleGenerator<1>::polynomial_order;
constexpr int two_point_polynomial_order = tribol::EdgeQuadratureRuleGenerator<2>::polynomial_order;
constexpr int three_point_polynomial_order = tribol::EdgeQuadratureRuleGenerator<3>::polynomial_order;

/**
 * @brief Check whether a factory result contains the requested Energy Mortar Gauss rule.
 * @tparam NumPoints Number of Gauss points in the expected concrete formulation.
 * @param formulation Factory-created formulation.
 * @param location Configured enforcement location.
 * @return True when the concrete formulation contains the expected rule generator and enforcement type.
 */
template <int NumPoints>
bool usesEnergyMortarRule( const tribol::ContactFormulation* formulation, tribol::EnforcementLocation location )
{
  using namespace tribol;
  using RuleGenerator =
      PairQuadratureRuleGenerator<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>;
  if ( location == EnforcementLocation::Nodal ) {
    using Expected = NodalEnergyMortar<ExecutionMode::Sequential, RuleGenerator, OffsetGap<OpposingNormalGap>,
                                       NodalPenaltyEnforcement>;
    return dynamic_cast<const Expected*>( formulation ) != nullptr;
  }
  using Expected = QuadraturePointEnergyMortar<ExecutionMode::Sequential, RuleGenerator, OffsetGap<OpposingNormalGap>>;
  return dynamic_cast<const Expected*>( formulation ) != nullptr;
}

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
    tribol::registerMfemCouplingScheme( coupling_scheme_id, first_mesh_id, second_mesh_id, *mesh_, *coords_, { 1 },
                                        { 3 }, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::ENERGY_MORTAR,
                                        tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID, execution_mode );
    EXPECT_EQ(
        tribol::CouplingSchemeManager::getInstance().findData( coupling_scheme_id )->getParameters().quadrature_order,
        tribol::maximum_supported_quadrature_order );
  }
  /** @brief Register the scheme and configure the requested quadrature and penalty settings. */
  void SetUp() override
  {
    registerScheme();
    tribol::setEnforcementLocation( coupling_scheme_id, std::get<0>( GetParam() ) );
    tribol::setQuadratureOrder( coupling_scheme_id, quadratureOrder() );
    tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, configured_penalty, configured_penalty );
    auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
    ASSERT_FALSE( cs.hasContactFormulation() );
    ASSERT_TRUE( cs.init() );
    ASSERT_TRUE( cs.hasContactFormulation() );
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

// The formulation factory must instantiate the exact Gauss rule requested for both enforcement locations.
TEST_P( EnergyMortarQuadratureSettingsTest, factory_selects_configured_rule_generator )
{
  const auto location = std::get<0>( GetParam() );
  const auto* formulation =
      tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id ).getContactFormulation();
  ASSERT_NE( formulation, nullptr );
  switch ( quadratureOrder() ) {
    case one_point_polynomial_order:
      EXPECT_TRUE( usesEnergyMortarRule<1>( formulation, location ) );
      break;
    case two_point_polynomial_order:
      EXPECT_TRUE( usesEnergyMortarRule<2>( formulation, location ) );
      break;
    case three_point_polynomial_order:
      EXPECT_TRUE( usesEnergyMortarRule<3>( formulation, location ) );
      break;
    default:
      FAIL() << "Unexpected quadrature order " << quadratureOrder();
  }
}

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
    setEnforcementLocation( coupling_scheme_id, std::get<0>( GetParam() ) );
    setQuadratureOrder( coupling_scheme_id, quadratureOrder() );
    setMfemKinematicConstantPenalty( coupling_scheme_id, configured_penalty, configured_penalty );
    updateMfemParallelDecomposition( contact_rank_count, true );
    auto& cs = CouplingSchemeManager::getInstance().at( coupling_scheme_id );
    const auto expected_mode = execution_mode == ExecutionMode::Dynamic
#ifdef TRIBOL_USE_OPENMP
                                   ? ExecutionMode::OpenMP
#else
                                   ? ExecutionMode::Sequential
#endif
                                   : execution_mode;
    EXPECT_EQ( cs.getExecutionMode(), expected_mode );
    RealT dt = 1.0;
    ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
    const auto& result = cs.getContactFormulationResult();
    ASSERT_NE( result.residual, nullptr );
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

// Coincident nonmatching boundaries have zero residual and Hessian weights but a nonzero compressive-side derivative.
TEST_P( EnergyMortarQuadratureSettingsTest, zero_gap_keeps_compressive_derivative )
{
  using namespace tribol;
  finalize();
  createParentFields(
      shared::MeshBuilder::Unify( { shared::MeshBuilder::SquareMesh( 2, 1 ), shared::MeshBuilder::SquareMesh( 3, 1 )
                                                                                 .translate( { 0.0, 1.0 } )
                                                                                 .updateBdrAttrib( 1, 5 )
                                                                                 .updateBdrAttrib( 3, 6 )
                                                                                 .updateAttrib( 1, 2 ) } ) );
  registerMfemCouplingScheme( coupling_scheme_id, first_mesh_id, second_mesh_id, *mesh_, *coords_, { 3 }, { 5 },
                              SURFACE_TO_SURFACE, NO_CASE, ENERGY_MORTAR, FRICTIONLESS, PENALTY, BINNING_GRID,
                              ExecutionMode::Sequential );
  constexpr RealT zero_gap_penalty = 4.0;
  setEnforcementLocation( coupling_scheme_id, std::get<0>( GetParam() ) );
  setQuadratureOrder( coupling_scheme_id, quadratureOrder() );
  setMfemKinematicConstantPenalty( coupling_scheme_id, zero_gap_penalty, zero_gap_penalty );
  updateMfemParallelDecomposition( contact_rank_count, true );
  RealT dt = 1.0;
  ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
  const auto& result = CouplingSchemeManager::getInstance().at( coupling_scheme_id ).getContactFormulationResult();
  ASSERT_TRUE( result.energy.has_value() );
  ASSERT_NE( result.residual, nullptr );
  ASSERT_NE( result.df_dx, nullptr );
  EXPECT_NEAR( *result.energy, 0.0, 1.0e-12 );
  EXPECT_NEAR( mfem::InnerProduct( MPI_COMM_WORLD, *result.residual, *result.residual ), 0.0, 1.0e-12 );
  mfem::Vector direction( result.df_dx->Width() ), action( result.df_dx->Height() );
  for ( int dof = 0; dof < direction.Size(); ++dof ) {
    direction[dof] = dof + 1.0;
  }
  result.df_dx->Mult( direction, action );
  EXPECT_GT( mfem::InnerProduct( MPI_COMM_WORLD, action, action ), 1.0e-12 );
}

// Supported orders invalidate the formulation until init(); rejected orders leave the current object intact.
TEST_P( EnergyMortarQuadratureSettingsTest, configures_and_defers_formulation_rebuild )
{
  auto* cs = tribol::CouplingSchemeManager::getInstance().findData( coupling_scheme_id );
  const auto location = std::get<0>( GetParam() );
  const auto check = [&]( int order ) {
    const int previous_order = cs->getParameters().quadrature_order;
    auto* previous_formulation = cs->getContactFormulation();
    tribol::setQuadratureOrder( coupling_scheme_id, order );
    EXPECT_EQ( cs->hasContactFormulation(), previous_order == order );
    if ( previous_order == order ) {
      EXPECT_EQ( cs->getContactFormulation(), previous_formulation );
    }
    EXPECT_EQ( cs->getParameters().enforcement_location, location );
    EXPECT_EQ( cs->getParameters().quadrature_order, order );
    ASSERT_TRUE( cs->init() );
    ASSERT_TRUE( cs->hasContactFormulation() );
  };
  check( quadratureOrder() );
  for ( int order = tribol::minimum_supported_quadrature_order; order <= tribol::maximum_supported_quadrature_order;
        ++order ) {
    check( order );
  }
  for ( int invalid :
        { tribol::minimum_supported_quadrature_order - 1, tribol::maximum_supported_quadrature_order + 1 } ) {
    auto* before = cs->getContactFormulation();
    EXPECT_SLIC_ERROR( tribol::setQuadratureOrder( coupling_scheme_id, invalid ) );
    EXPECT_EQ( cs->getParameters().quadrature_order, tribol::maximum_supported_quadrature_order );
    EXPECT_EQ( cs->getContactFormulation(), before );
    cs->getParameters().quadrature_order = invalid;
    EXPECT_SLIC_ERROR( tribol::createContactFormulation( cs ) );
    cs->getParameters().quadrature_order = tribol::maximum_supported_quadrature_order;
  }
  cs->setContactFormulation( nullptr );
  constexpr int alternate_valid_order = 2;
  tribol::setQuadratureOrder( coupling_scheme_id, alternate_valid_order );
  EXPECT_FALSE( cs->hasContactFormulation() );
  ASSERT_TRUE( cs->init() );
  EXPECT_TRUE( cs->hasContactFormulation() );
  EXPECT_EQ( cs->getParameters().enforcement_location, location );
}

// Gap and residual calls must work independently, initialize only their outputs, and leave earlier results unchanged.
TEST_P( EnergyMortarQuadratureSettingsTest, returns_owned_outputs_for_each_calculation )
{
  tribol::updateMfemParallelDecomposition();
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  const auto& formulation = *cs.getContactFormulation();
  const auto mesh1 = cs.getMesh1().getView();
  const auto mesh2 = cs.getMesh2().getView();
  mfem::Vector legacy_gap;
  EXPECT_SLIC_ERROR( tribol::getMfemGap( coupling_scheme_id, legacy_gap ) );
  if ( std::get<0>( GetParam() ) == tribol::EnforcementLocation::Nodal ) {
    auto gaps = formulation.computeNodalGaps( mesh1, mesh2 );
    ASSERT_NE( gaps.gap, nullptr );
    ASSERT_NE( gaps.weighted_gap_jacobian, nullptr );
    EXPECT_EQ( gaps.residual, nullptr );
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
    EXPECT_SLIC_ERROR( formulation.computeNodalGaps( mesh1, mesh2 ) );
  }

  auto first = formulation.computeNodalResiduals( mesh1, mesh2, {} );
  ASSERT_NE( first.residual, nullptr );
  ASSERT_NE( first.df_dx, nullptr );
  ASSERT_TRUE( first.energy.has_value() );
  EXPECT_FALSE( first.time_step.has_value() );
  EXPECT_EQ( first.df_dp, nullptr );
  const bool nodal = std::get<0>( GetParam() ) == tribol::EnforcementLocation::Nodal;
  EXPECT_EQ( first.gap != nullptr, nodal );
  EXPECT_EQ( first.pressure != nullptr, nodal );
  EXPECT_EQ( first.weighted_gap_jacobian != nullptr, nodal );
  const mfem::HypreParVector original( *first.residual );
  ASSERT_GT( original.Norml2(), 0.0 );
  auto second = formulation.computeNodalResiduals( mesh1, mesh2, {} );
  second.residual->Add( -1.0, original );
  EXPECT_LT( second.residual->Norml2(), 1.0e-12 );
  EXPECT_NEAR( *second.energy, *first.energy, 1.0e-12 );
  EXPECT_EQ( cs.getContactFormulationResult().residual, nullptr );
  cs.setContactFormulation( nullptr );
  auto difference = *first.residual;
  difference -= original;
  EXPECT_EQ( difference.Norml2(), 0.0 );
  mfem::Vector direction( first.df_dx->Width() ), action( first.df_dx->Height() );
  direction = 1.0;
  first.df_dx->Mult( direction, action );
  EXPECT_TRUE( std::isfinite( action.Norml2() ) );
}

// Springs in series must scale the residual, Jacobian, and energy for either location and each host Gauss rule.
TEST_P( EnergyMortarQuadratureSettingsTest, accepts_series_penalty_stiffness )
{
  using namespace tribol;
  setMfemKinematicConstantPenalty( coupling_scheme_id, first_series_penalty, second_series_penalty );
  updateMfemParallelDecomposition( contact_rank_count, true );
  auto& cs = CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  ASSERT_TRUE( cs.init() );
  const auto baseline =
      cs.getContactFormulation()->computeNodalResiduals( cs.getMesh1().getView(), cs.getMesh2().getView(), {} );
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
            auto coarse = createCoarseSearch( cs.spatialDim(), cs.getBinningMethod(), execution_mode,
                                              cs.getEffectiveBinningProximityScale(), 0.0 );
            auto refined = std::make_unique<ProjectedEdgeSearch>( execution_mode, false );
            PairListBuilder pair_builder{ std::move( coarse ), std::move( refined ) };
            if ( std::get<0>( GetParam() ) == EnforcementLocation::Nodal ) {
              return std::make_unique<NodalEnergyMortar<Exec, decltype( rule_generator ), OffsetGap<OpposingNormalGap>,
                                                        NodalPenaltyEnforcement>>(
                  *cs.getMfemMeshData(), *cs.getMfemSubmeshData(), *cs.getMfemJacobianData(),
                  std::move( rule_generator ), OffsetGap{ OpposingNormalGap{}, 0.0 },
                  NodalPenaltyEnforcement{ SeriesPenaltyStiffness{}( first_series_penalty, second_series_penalty ) },
                  std::move( pair_builder ) );
            }
            return std::make_unique<
                QuadraturePointEnergyMortar<Exec, decltype( rule_generator ), OffsetGap<OpposingNormalGap>>>(
                *cs.getMfemMeshData(), *cs.getMfemJacobianData(), std::move( rule_generator ),
                SeriesPenaltyStiffness{}( first_series_penalty, second_series_penalty ),
                OffsetGap{ OpposingNormalGap{}, 0.0 }, std::move( pair_builder ) );
          } );
    } );
    auto result = alternate->computeNodalResiduals( cs.getMesh1().getView(), cs.getMesh2().getView(), {} );
    ASSERT_NE( result.residual, nullptr );
    ASSERT_NE( result.df_dx, nullptr );
    ASSERT_TRUE( result.energy.has_value() );
    // The stiffnesses 2 and 6 give series stiffness 1.5 instead of the default arithmetic mean 4.
    constexpr RealT stiffness_ratio = 1.5 / 4.0;
    result.residual->Add( -stiffness_ratio, *baseline.residual );
    EXPECT_LT( result.residual->Norml2(), 1.0e-11 );
    EXPECT_NEAR( *result.energy, stiffness_ratio * *baseline.energy, 1.0e-11 );
    mfem::Vector action( result.df_dx->Height() );
    result.df_dx->Mult( direction, action );
    action.Add( -stiffness_ratio, reference_action );
    EXPECT_LT( action.Norml2(), 1.0e-11 );
  }
}

// Unsupported locations must leave the selected formulation and its transfer fields unchanged.
TEST_P( EnergyMortarQuadratureSettingsTest, rejects_invalid_location_without_rebuilding )
{
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  auto* formulation = cs.getContactFormulation();
  auto* fields = cs.getMfemSubmeshData();
  const auto location = cs.getParameters().enforcement_location;
  constexpr auto invalid_location = static_cast<tribol::EnforcementLocation>( -1 );
  EXPECT_SLIC_ERROR( tribol::setEnforcementLocation( coupling_scheme_id, invalid_location ) );
  EXPECT_EQ( cs.getParameters().enforcement_location, location );
  EXPECT_EQ( cs.getContactFormulation(), formulation );
  EXPECT_EQ( cs.getMfemSubmeshData(), fields );
}

// An unconfigured penalty must report an error before a pair formulation can calculate a residual.
TEST_P( EnergyMortarQuadratureSettingsTest, rejects_unconfigured_pair_penalty )
{
  tribol::setEnforcementLocation( coupling_scheme_id, tribol::EnforcementLocation::QuadraturePoint );
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  cs.getEnforcementOptions().penalty_options.kinematic_calc_set = false;
  cs.invalidateContactFormulation();
  EXPECT_FALSE( cs.hasContactFormulation() );
  tribol::RealT dt = 1.0;
  EXPECT_SLIC_ERROR( tribol::update( coupling_scheme_id, 0.0, dt ) );
  EXPECT_FALSE( cs.hasMfemSubmeshData() );
}

// Existing penalty calculations retain their construction stiffness even if registered constants change or disappear.
TEST_P( EnergyMortarQuadratureSettingsTest, keeps_penalty_fixed_until_formulation_is_rebuilt )
{
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  auto& mesh_data = *cs.getMfemMeshData();
  const auto calculate = [&] {
    return cs.getContactFormulation()->computeNodalResiduals( cs.getMesh1().getView(), cs.getMesh2().getView(), {} );
  };
  const auto initial = calculate();
  const auto check_unchanged = [&] {
    auto result = calculate();
    result.residual->Add( -1.0, *initial.residual );
    EXPECT_LT( result.residual->Norml2(), 1.0e-12 );
    EXPECT_EQ( result.energy, initial.energy );
  };
  mesh_data.ClearAllPenaltyData();
  check_unchanged();
  EXPECT_EQ( tribol::createContactFormulation( &cs ), nullptr );
  mesh_data.SetMesh1KinematicConstantPenalty( configured_penalty );
  check_unchanged();
  EXPECT_EQ( tribol::createContactFormulation( &cs ), nullptr );
  mesh_data.ClearAllPenaltyData();
  mesh_data.SetMesh2KinematicConstantPenalty( configured_penalty );
  check_unchanged();
  EXPECT_EQ( tribol::createContactFormulation( &cs ), nullptr );
  mesh_data.SetMesh1KinematicConstantPenalty( 300.0 );
  check_unchanged();
  // The public setter invalidates the old formulation; init() builds the replacement without refreshing decomposition.
  constexpr tribol::RealT updated_penalty = 300.0;
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, updated_penalty, configured_penalty );
  EXPECT_FALSE( cs.hasContactFormulation() );
  ASSERT_TRUE( cs.init() );
  auto rebuilt = calculate();
  rebuilt.residual->Add( -2.0, *initial.residual );
  EXPECT_LT( rebuilt.residual->Norml2(), 1.0e-12 );
  EXPECT_NEAR( *rebuilt.energy, 2.0 * *initial.energy, 1.0e-12 );
}

/** @brief Start tests with a coupling scheme registered before its penalty settings are supplied. */
class EnergyMortarRegistrationTest : public EnergyMortarQuadratureSettingsTest {
 protected:
  /** @brief Register the scheme while leaving penalty selection and constants unset. */
  void SetUp() override { registerScheme(); }
};

// Complete penalty settings remain pending until update() initializes the formulation.
TEST_P( EnergyMortarRegistrationTest, constructs_formulation_during_update )
{
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  EXPECT_FALSE( cs.hasContactFormulation() );
  tribol::setEnforcementLocation( coupling_scheme_id, std::get<0>( GetParam() ) );
  tribol::setQuadratureOrder( coupling_scheme_id, quadratureOrder() );
  EXPECT_FALSE( cs.hasContactFormulation() );
  tribol::RealT dt = 1.0;
  EXPECT_SLIC_ERROR( tribol::update( coupling_scheme_id, 0.0, dt ) );
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, configured_penalty, configured_penalty );
  ASSERT_FALSE( cs.hasContactFormulation() );
  const auto location = std::get<0>( GetParam() );
  EXPECT_EQ( cs.getParameters().enforcement_location, location );
  EXPECT_EQ( cs.getParameters().quadrature_order, quadratureOrder() );
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  ASSERT_EQ( tribol::update( coupling_scheme_id, 0.0, dt ), 0 );
  ASSERT_TRUE( cs.hasContactFormulation() );
  EXPECT_GT( tribol::getMfemContactResidual( 0 ).Norml2(), 0.0 );
}

// init() can construct the formulation and its field rows on an existing decomposition without refreshing geometry.
TEST_P( EnergyMortarRegistrationTest, constructs_formulation_on_existing_decomposition )
{
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  tribol::setEnforcementLocation( coupling_scheme_id, std::get<0>( GetParam() ) );
  tribol::setQuadratureOrder( coupling_scheme_id, quadratureOrder() );
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  EXPECT_FALSE( cs.hasContactFormulation() );
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, configured_penalty, configured_penalty );
  ASSERT_FALSE( cs.hasContactFormulation() );
  ASSERT_TRUE( cs.init() );
  ASSERT_TRUE( cs.hasContactFormulation() );
  const bool nodal = std::get<0>( GetParam() ) == tribol::EnforcementLocation::Nodal;
  EXPECT_EQ( cs.getMesh1().getNodalFields().m_is_node_gap_set, nodal );
  EXPECT_FALSE( cs.getMesh2().getNodalFields().m_is_node_gap_set );
  const auto result =
      cs.getContactFormulation()->computeNodalResiduals( cs.getMesh1().getView(), cs.getMesh2().getView(), {} );
  ASSERT_NE( result.residual, nullptr );
  ASSERT_TRUE( result.energy.has_value() );
  EXPECT_GT( result.residual->Norml2(), 0.0 );
}

// Rejected mesh stiffnesses must leave both the installed formulation and its registered inputs unchanged.
TEST_P( EnergyMortarQuadratureSettingsTest, rejects_invalid_mesh_penalties_without_changing_scheme )
{
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  auto* formulation = cs.getContactFormulation();
  const auto& mesh_data = *cs.getMfemMeshData();
  const tribol::RealT mesh1_penalty = *mesh_data.GetMesh1KinematicConstantPenalty();
  const tribol::RealT mesh2_penalty = *mesh_data.GetMesh2KinematicConstantPenalty();
  const auto check_unchanged = [&] {
    EXPECT_EQ( cs.getContactFormulation(), formulation );
    EXPECT_EQ( *mesh_data.GetMesh1KinematicConstantPenalty(), mesh1_penalty );
    EXPECT_EQ( *mesh_data.GetMesh2KinematicConstantPenalty(), mesh2_penalty );
  };
  for ( const tribol::RealT invalid : { -10.0, 0.0, std::numeric_limits<tribol::RealT>::quiet_NaN(),
                                        std::numeric_limits<tribol::RealT>::infinity() } ) {
    SCOPED_TRACE( invalid );
    constexpr tribol::RealT valid_penalty = 20.0;
    EXPECT_SLIC_ERROR( tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, invalid, valid_penalty ) );
    check_unchanged();
    EXPECT_SLIC_ERROR( tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, valid_penalty, invalid ) );
    check_unchanged();
  }
  const tribol::RealT largest_penalty = std::numeric_limits<tribol::RealT>::max();
  EXPECT_SLIC_ERROR( tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, largest_penalty, largest_penalty ) );
  check_unchanged();
  EXPECT_NO_THROW(
      tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, first_series_penalty, second_series_penalty ) );
}

// Invalid offsets must be rejected before the search and gap definitions capture them.
TEST_P( EnergyMortarQuadratureSettingsTest, rejects_invalid_residual_gaps_without_rebuilding )
{
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  auto* formulation = cs.getContactFormulation();
  const tribol::RealT residual_gap = cs.getParameters().residual_gap;
  for ( const tribol::RealT invalid :
        { -1.0, -std::numeric_limits<tribol::RealT>::infinity(), std::numeric_limits<tribol::RealT>::infinity(),
          std::numeric_limits<tribol::RealT>::quiet_NaN() } ) {
    SCOPED_TRACE( invalid );
    EXPECT_SLIC_ERROR( tribol::setResidualGap( coupling_scheme_id, invalid ) );
    EXPECT_EQ( cs.getParameters().residual_gap, residual_gap );
    EXPECT_EQ( cs.getContactFormulation(), formulation );
  }
}

// Penalty data on an LM scheme must not make quadrature-point enforcement appear valid.
TEST( EnergyMortarConfiguration, rejects_quadrature_point_lm_without_releasing_nodal_state )
{
  struct FinalizeAfterTest {
    ~FinalizeAfterTest() { tribol::finalize(); }
  };
  tribol::finalize();
  mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( 2, 1 ) );
  mfem::H1_FECollection fec( 1, 2 );
  mfem::ParFiniteElementSpace fes( &mesh, &fec, 2 );
  mfem::ParGridFunction coordinates( &fes );
  mesh.GetNodes( coordinates );
  [[maybe_unused]] FinalizeAfterTest finalize_after_test;
  tribol::registerMfemCouplingScheme(
      0, 0, 1, mesh, coordinates, { 1 }, { 3 }, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::ENERGY_MORTAR,
      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, configured_penalty, configured_penalty );
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  ASSERT_FALSE( cs.hasContactFormulation() );
  auto* multiplier = &tribol::getMfemContactPressure( 0 );
  ASSERT_FALSE( cs.hasContactFormulation() );
  ASSERT_TRUE( cs.init() );
  EXPECT_EQ( &tribol::getMfemContactPressure( 0 ), multiplier );
  *multiplier = 2.0;
  const tribol::RealT multiplier_norm = multiplier->Norml2();
  constexpr int minimum_three_point_order = two_point_polynomial_order + 1;
  tribol::setQuadratureOrder( coupling_scheme_id, minimum_three_point_order );
  EXPECT_FALSE( cs.hasContactFormulation() );
  EXPECT_EQ( &tribol::getMfemContactPressure( 0 ), multiplier );
  EXPECT_EQ( multiplier->Norml2(), multiplier_norm );
  ASSERT_TRUE( cs.init() );
  EXPECT_EQ( &tribol::getMfemContactPressure( 0 ), multiplier );
  EXPECT_EQ( multiplier->Norml2(), multiplier_norm );
  auto* formulation = cs.getContactFormulation();
  auto* submesh_data = cs.getMfemSubmeshData();
  auto* jacobian_data = cs.getMfemJacobianData();

  EXPECT_SLIC_ERROR(
      tribol::setEnforcementLocation( coupling_scheme_id, tribol::EnforcementLocation::QuadraturePoint ) );
  EXPECT_EQ( cs.getParameters().enforcement_location, tribol::EnforcementLocation::Nodal );
  EXPECT_EQ( cs.getContactFormulation(), formulation );
  EXPECT_EQ( cs.getMfemSubmeshData(), submesh_data );
  EXPECT_EQ( cs.getMfemJacobianData(), jacobian_data );
  EXPECT_EQ( &tribol::getMfemContactPressure( 0 ), multiplier );
}

// LM residuals must read the multiplier passed in ContactFormulationInputs, not the submesh pressure field.
TEST( EnergyMortarConfiguration, lm_residuals_use_supplied_multiplier )
{
  struct FinalizeAfterTest {
    ~FinalizeAfterTest() { tribol::finalize(); }
  };
  tribol::finalize();
  mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( 2, 1 ) );
  mfem::H1_FECollection fec( 1, 2 );
  mfem::ParFiniteElementSpace fes( &mesh, &fec, 2 );
  mfem::ParGridFunction coordinates( &fes );
  mesh.GetNodes( coordinates );
  [[maybe_unused]] FinalizeAfterTest finalize_after_test;
  tribol::registerMfemCouplingScheme(
      0, 0, 1, mesh, coordinates, { 1 }, { 3 }, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::ENERGY_MORTAR,
      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  ASSERT_TRUE( cs.init() );
  const auto mesh1 = cs.getMesh1().getView();
  const auto mesh2 = cs.getMesh2().getView();
  const auto* formulation = cs.getContactFormulation();
  ASSERT_NE( formulation, nullptr );

  EXPECT_SLIC_ERROR( formulation->computeNodalResiduals( mesh1, mesh2, {} ) );

  constexpr tribol::RealT multiplier_value = 2.0;
  // A different value in the submesh pressure field must not affect the calculation.
  constexpr tribol::RealT ignored_pressure_value = 5.0;
  auto& multiplier = tribol::getMfemContactPressure( 0 );
  multiplier = multiplier_value;
  tribol::getMfemPressure( 0 ) = ignored_pressure_value;
  tribol::ContactFormulationInputs inputs;
  inputs.multiplier = &multiplier;
  const auto result = formulation->computeNodalResiduals( mesh1, mesh2, inputs );
  ASSERT_NE( result.pressure, nullptr );
  ASSERT_NE( result.gap, nullptr );
  ASSERT_TRUE( result.energy.has_value() );
  mfem::Vector pressure_error( *result.pressure );
  pressure_error -= multiplier;
  EXPECT_EQ( pressure_error.Normlinf(), 0.0 );
  const tribol::RealT weighted_gap_sum = mfem::InnerProduct( MPI_COMM_WORLD, *result.gap, *result.gap );
  ASSERT_GT( weighted_gap_sum, 0.0 ) << "The configuration must produce nonzero weighted gaps.";
  EXPECT_NEAR( *result.energy, mfem::InnerProduct( MPI_COMM_WORLD, multiplier, *result.gap ), 1.0e-12 );
}

INSTANTIATE_TEST_SUITE_P( enforcement, EnergyMortarRegistrationTest,
                          testing::Combine( testing::Values( tribol::EnforcementLocation::Nodal,
                                                             tribol::EnforcementLocation::QuadraturePoint ),
                                            testing::Values( 1, 3, 5 ) ) );

// Parent-only pair enforcement must never allocate or register pressure/gap fields, including after decomposition.
TEST_P( EnergyMortarQuadratureSettingsTest, creates_only_the_required_transfer_fields )
{
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  const bool nodal = std::get<0>( GetParam() ) == tribol::EnforcementLocation::Nodal;
  EXPECT_EQ( cs.hasMfemSubmeshData(), nodal );
  ASSERT_TRUE( cs.hasMfemJacobianData() );
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  EXPECT_EQ( cs.hasMfemSubmeshData(), nodal );
  EXPECT_EQ( cs.getMfemJacobianData()->ParentPath().final_fes, fes_.get() );
  EXPECT_EQ( cs.getMesh1().getNodalFields().m_is_node_gap_set, nodal );
  EXPECT_EQ( cs.getMesh1().getNodalFields().m_is_node_pressure_set, nodal );
  EXPECT_FALSE( cs.getMesh2().getNodalFields().m_is_node_gap_set );
  EXPECT_FALSE( cs.getMesh2().getNodalFields().m_is_node_pressure_set );
}

// Switching locations leaves one pending rebuild; init() creates the new transfers without rebuilding decomposition.
TEST_P( EnergyMortarQuadratureSettingsTest, switches_locations_without_rebuilding_the_decomposition )
{
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  std::array<std::unique_ptr<mfem::HypreParVector>, 2> previous_residuals;
  std::array<std::optional<tribol::RealT>, 2> previous_energies;
  for ( const auto location : { tribol::EnforcementLocation::Nodal, tribol::EnforcementLocation::QuadraturePoint,
                                tribol::EnforcementLocation::Nodal, tribol::EnforcementLocation::QuadraturePoint } ) {
    const auto previous_location = cs.getParameters().enforcement_location;
    auto* previous_formulation = cs.getContactFormulation();
    tribol::setEnforcementLocation( coupling_scheme_id, location );
    const bool nodal = location == tribol::EnforcementLocation::Nodal;
    if ( previous_location == location ) {
      EXPECT_EQ( cs.getContactFormulation(), previous_formulation );
    } else {
      EXPECT_FALSE( cs.hasContactFormulation() );
    }
    ASSERT_TRUE( cs.init() );
    EXPECT_EQ( cs.hasMfemSubmeshData(), nodal );
    EXPECT_EQ( cs.getContactFormulationResult().residual, nullptr );
    EXPECT_EQ( cs.getMesh1().getNodalFields().m_is_node_gap_set, nodal );
    EXPECT_EQ( cs.getMesh1().getNodalFields().m_is_node_pressure_set, nodal );
    auto result =
        cs.getContactFormulation()->computeNodalResiduals( cs.getMesh1().getView(), cs.getMesh2().getView(), {} );
    ASSERT_NE( result.residual, nullptr );
    ASSERT_NE( result.df_dx, nullptr );
    ASSERT_TRUE( result.energy.has_value() );
    EXPECT_EQ( result.gap != nullptr, nodal );
    EXPECT_EQ( result.pressure != nullptr, nodal );
    EXPECT_EQ( result.weighted_gap_jacobian != nullptr, nodal );
    const int index = nodal ? 0 : 1;
    if ( previous_residuals[index] ) {
      auto difference = *result.residual;
      difference -= *previous_residuals[index];
      EXPECT_LT( difference.Norml2(), 1.0e-12 );
      EXPECT_NEAR( *result.energy, *previous_energies[index], 1.0e-12 );
    } else {
      previous_residuals[index] = std::make_unique<mfem::HypreParVector>( *result.residual );
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
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  auto initial =
      cs.getContactFormulation()->computeNodalResiduals( cs.getMesh1().getView(), cs.getMesh2().getView(), {} );
  // Retain only parent-space outputs while rebuilding the formulation.
  const auto residual = *initial.residual;
  const auto energy = *initial.energy;
  initial = {};
  constexpr tribol::RealT updated_penalty = 300.0;
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, configured_penalty, updated_penalty );
  EXPECT_FALSE( cs.hasContactFormulation() );
  ASSERT_TRUE( cs.init() );
  auto updated =
      cs.getContactFormulation()->computeNodalResiduals( cs.getMesh1().getView(), cs.getMesh2().getView(), {} );
  updated.residual->Add( -2.0, residual );
  EXPECT_LT( updated.residual->Norml2(), 1.0e-11 );
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
  updateMfemParallelDecomposition( contact_rank_count, true );
  auto& cs = CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  const auto penalty = cs.getContactFormulation()->computeNodalGaps( cs.getMesh1().getView(), cs.getMesh2().getView() );
  const auto weighted =
      createWithQuadratureRuleGenerator( quadratureOrder(), SmoothedProjectedEdgeMapping{}, [&]( auto rule_generator ) {
        auto coarse = createCoarseSearch( cs.spatialDim(), cs.getBinningMethod(), cs.getExecutionMode(),
                                          cs.getEffectiveBinningProximityScale(), 0.0 );
        auto refined = std::make_unique<ProjectedEdgeSearch>( cs.getExecutionMode(), false );
        PairListBuilder pair_builder{ std::move( coarse ), std::move( refined ) };
        NodalEnergyMortar<ExecutionMode::Sequential, decltype( rule_generator ), OffsetGap<OpposingNormalGap>,
                          NodalLagrangeMultiplierEnforcement>
            lm( *cs.getMfemMeshData(), *cs.getMfemSubmeshData(), *cs.getMfemJacobianData(), std::move( rule_generator ),
                OffsetGap{ OpposingNormalGap{}, 0.0 }, NodalLagrangeMultiplierEnforcement{},
                std::move( pair_builder ) );
        return lm.computeNodalGaps( cs.getMesh1().getView(), cs.getMesh2().getView() );
      } );
  mfem::Vector direction( fes_->GetTrueVSize() ), penalty_action( penalty.gap->Size() ),
      weighted_action( weighted.gap->Size() );
  direction = 0.0;
  for ( int dof = 0; dof < direction.Size(); ++dof ) {
    direction[dof] = dof + 1.0;
  }
  penalty.weighted_gap_jacobian->Mult( direction, penalty_action );
  weighted.weighted_gap_jacobian->Mult( direction, weighted_action );
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
TEST_P( EnergyMortarQuadratureSettingsTest, assembled_residual_and_jacobian_match_directional_finite_difference )
{
  constexpr tribol::RealT residual_gap = 1.2;
  tribol::setResidualGap( coupling_scheme_id, residual_gap );
  auto& cs = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  EXPECT_FALSE( cs.hasContactFormulation() );
  ASSERT_TRUE( cs.init() );
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
    tribol::updateMfemParallelDecomposition( contact_rank_count, true );
    auto out =
        cs.getContactFormulation()->computeNodalResiduals( cs.getMesh1().getView(), cs.getMesh2().getView(), {} );
    return out;
  };

  const auto base = evaluate_at( 0.0 );
  constexpr tribol::RealT step = 1.0e-6;
  const auto plus = evaluate_at( step );
  const auto minus = evaluate_at( -step );

  const tribol::RealT directional_residual = mfem::InnerProduct( MPI_COMM_WORLD, *base.residual, direction_true );
  const tribol::RealT fd_energy = ( *plus.energy - *minus.energy ) / ( 2.0 * step );
  EXPECT_NEAR( directional_residual, fd_energy, 1.0e-5 );

  mfem::Vector jv( fes_.get()->GetTrueVSize() );
  base.df_dx->Mult( direction_true, jv );
  mfem::Vector fd_residual( *plus.residual );
  fd_residual -= *minus.residual;
  fd_residual *= 1.0 / ( 2.0 * step );
  jv -= fd_residual;
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
  constexpr int valid_quadrature_order = 2;
  EXPECT_SLIC_ERROR( tribol::setQuadratureOrder( coupling_scheme_id, valid_quadrature_order ) );
  Edges edges;
  tribol::registerMesh( first_mesh_id, 1, edges.x1.size(), edges.connectivity.data(), tribol::EDGE,
                        tribol::LINEAR_LAGRANGE, edges.x1.data(), edges.y1.data() );
  tribol::registerMesh( second_mesh_id, 1, edges.x2.size(), edges.connectivity.data(), tribol::EDGE,
                        tribol::LINEAR_LAGRANGE, edges.x2.data(), edges.y2.data() );
  tribol::registerCouplingScheme( coupling_scheme_id, first_mesh_id, second_mesh_id, tribol::SURFACE_TO_SURFACE,
                                  tribol::NO_CASE, tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                  tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
  EXPECT_SLIC_ERROR( tribol::setQuadratureOrder( coupling_scheme_id, valid_quadrature_order ) );
  EXPECT_EQ(
      tribol::CouplingSchemeManager::getInstance().findData( coupling_scheme_id )->getParameters().quadrature_order,
      tribol::maximum_supported_quadrature_order );
  tribol::finalize();
}

/** @brief Exercise both enforcement locations with one, two, and three Gauss points. */
INSTANTIATE_TEST_SUITE_P( enforcement, EnergyMortarQuadratureSettingsTest,
                          testing::Combine( testing::Values( tribol::EnforcementLocation::Nodal,
                                                             tribol::EnforcementLocation::QuadraturePoint ),
                                            testing::Values( one_point_polynomial_order, two_point_polynomial_order,
                                                             three_point_polynomial_order ) ) );

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
