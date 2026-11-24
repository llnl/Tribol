// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Verify public Energy Mortar configuration and formulation rebuilding with MFEM meshes.
 *
 * Tests cover both enforcement locations, every supported quadrature order, and enabled host execution backends. They
 * check formulation selection, delayed configuration, transfer data, penalty changes, contact activation, and the
 * assembled energy, residual, and Jacobian on one and two MPI ranks.
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
class MfemEnergyMortarFormulationTest : public testing::TestWithParam<std::tuple<tribol::EnforcementLocation, int>> {
 protected:
  static constexpr tribol::IndexT coupling_scheme_id = 0;

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
    coordinates_.reset();
    coordinate_space_.reset();
    coordinate_collection_.reset();
    parent_mesh_ = std::make_unique<mfem::ParMesh>(
        static_cast<const mfem::ParMesh&>( shared::ParMeshBuilder( MPI_COMM_WORLD, std::move( mesh_builder ) ) ) );
    constexpr int spatial_dim = 2;
    constexpr int linear_order = 1;
    coordinate_collection_ = std::make_unique<mfem::H1_FECollection>( linear_order, spatial_dim );
    coordinate_space_ =
        std::make_unique<mfem::ParFiniteElementSpace>( parent_mesh_.get(), coordinate_collection_.get(), spatial_dim );
    coordinates_ = std::make_unique<mfem::ParGridFunction>( coordinate_space_.get() );
    parent_mesh_->GetNodes( *coordinates_ );
  }
  /**
   * @brief Create the parent fields and register the scheme before penalty settings are supplied.
   * @param execution_mode Requested host execution mode, including Dynamic selection.
   */
  void registerScheme( tribol::ExecutionMode execution_mode = tribol::ExecutionMode::Sequential )
  {
    constexpr int horizontal_element_count = 2;
    constexpr int vertical_element_count = 1;
    // The unit square is two elements wide. Its two-edge bottom and top boundaries are registered as the opposing
    // contact surfaces, giving every configuration the same simple aligned mesh.
    createParentFields( shared::MeshBuilder::SquareMesh( horizontal_element_count, vertical_element_count ) );
    constexpr tribol::IndexT lower_mesh_id = 0;
    constexpr tribol::IndexT upper_mesh_id = 1;
    constexpr int lower_contact_boundary_attribute = 1;
    constexpr int upper_contact_boundary_attribute = 3;
    tribol::registerMfemCouplingScheme( coupling_scheme_id, lower_mesh_id, upper_mesh_id, *parent_mesh_, *coordinates_,
                                        { lower_contact_boundary_attribute }, { upper_contact_boundary_attribute },
                                        tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::ENERGY_MORTAR,
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
    constexpr tribol::RealT configured_penalty = 100.0;
    tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, configured_penalty, configured_penalty );
    auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
    ASSERT_FALSE( coupling_scheme.hasContactFormulation() );
    ASSERT_TRUE( coupling_scheme.init() );
    ASSERT_TRUE( coupling_scheme.hasContactFormulation() );
  }
  /**
   * @brief Release coupling-scheme state after each test.
   */
  void TearDown() override { tribol::finalize(); }

  std::unique_ptr<mfem::ParMesh> parent_mesh_;                     ///< Owned parent mesh for registration.
  std::unique_ptr<mfem::H1_FECollection> coordinate_collection_;   ///< Owned linear coordinate basis.
  std::unique_ptr<mfem::ParFiniteElementSpace> coordinate_space_;  ///< Parent-mesh coordinate space.
  std::unique_ptr<mfem::ParGridFunction> coordinates_;             ///< Registered parent-mesh coordinates.
};

// This test ensures every supported enforcement location and quadrature order creates the expected concrete Energy
// Mortar formulation.
TEST_P( MfemEnergyMortarFormulationTest, factory_selects_configured_rule_generator )
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

// This test ensures every valid Energy Mortar host configuration can be registered, evaluated, and produces the same
// energy regardless of the selected host backend.
TEST_P( MfemEnergyMortarFormulationTest, public_registration_supports_host_execution_modes )
{
  using namespace tribol;
  constexpr RealT configured_penalty = 100.0;
  constexpr int contact_rank_count = 1;
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
    auto& coupling_scheme = CouplingSchemeManager::getInstance().at( coupling_scheme_id );
    const auto expected_mode = execution_mode == ExecutionMode::Dynamic
#ifdef TRIBOL_USE_OPENMP
                                   ? ExecutionMode::OpenMP
#else
                                   ? ExecutionMode::Sequential
#endif
                                   : execution_mode;
    EXPECT_EQ( coupling_scheme.getExecutionMode(), expected_mode );
    RealT dt = 1.0;
    ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
    const auto& formulation_result = coupling_scheme.getContactFormulationResult();
    ASSERT_NE( formulation_result.residual, nullptr );
    ASSERT_NE( formulation_result.df_dx, nullptr );
    ASSERT_TRUE( formulation_result.energy.has_value() );
    EXPECT_GT( *formulation_result.energy, 0.0 );
    if ( reference_energy ) {
      EXPECT_NEAR( *formulation_result.energy, *reference_energy, 1.0e-11 );
    }
    reference_energy = formulation_result.energy;
    EXPECT_EQ( coupling_scheme.getNumTotalNodes(), coupling_scheme.getMesh1().numberOfNodes() );
    EXPECT_EQ( coupling_scheme.getMethodData(), nullptr );
    for ( auto* contact_mesh : { &coupling_scheme.getMesh1(), &coupling_scheme.getMesh2() } ) {
      EXPECT_TRUE( contact_mesh->getView().getElementAreas().empty() );
      EXPECT_TRUE( contact_mesh->getView().getElementCentroids().empty() );
      EXPECT_TRUE( contact_mesh->getView().getElementNormals().empty() );
    }
  }
}

// Coincident nonmatching boundaries have zero residual and Hessian weights but a nonzero compressive-side derivative.
TEST_P( MfemEnergyMortarFormulationTest, zero_gap_keeps_compressive_derivative )
{
  using namespace tribol;
  finalize();
  // The lower square has two edges on its top boundary. The upper square has three edges on its bottom boundary and is
  // translated so both boundaries lie at y=1, creating coincident nonmatching contact surfaces.
  createParentFields(
      shared::MeshBuilder::Unify( { shared::MeshBuilder::SquareMesh( 2, 1 ), shared::MeshBuilder::SquareMesh( 3, 1 )
                                                                                 .translate( { 0.0, 1.0 } )
                                                                                 .updateBdrAttrib( 1, 5 )
                                                                                 .updateBdrAttrib( 3, 6 )
                                                                                 .updateAttrib( 1, 2 ) } ) );
  constexpr IndexT lower_mesh_id = 0;
  constexpr IndexT upper_mesh_id = 1;
  constexpr int lower_contact_boundary_attribute = 3;
  constexpr int upper_contact_boundary_attribute = 5;
  registerMfemCouplingScheme( coupling_scheme_id, lower_mesh_id, upper_mesh_id, *parent_mesh_, *coordinates_,
                              { lower_contact_boundary_attribute }, { upper_contact_boundary_attribute },
                              SURFACE_TO_SURFACE, NO_CASE, ENERGY_MORTAR, FRICTIONLESS, PENALTY, BINNING_GRID,
                              ExecutionMode::Sequential );
  constexpr RealT zero_gap_penalty = 4.0;
  setEnforcementLocation( coupling_scheme_id, std::get<0>( GetParam() ) );
  setQuadratureOrder( coupling_scheme_id, quadratureOrder() );
  setMfemKinematicConstantPenalty( coupling_scheme_id, zero_gap_penalty, zero_gap_penalty );
  constexpr int contact_rank_count = 1;
  updateMfemParallelDecomposition( contact_rank_count, true );
  RealT dt = 1.0;
  ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
  const auto& zero_gap_result =
      CouplingSchemeManager::getInstance().at( coupling_scheme_id ).getContactFormulationResult();
  ASSERT_TRUE( zero_gap_result.energy.has_value() );
  ASSERT_NE( zero_gap_result.residual, nullptr );
  ASSERT_NE( zero_gap_result.df_dx, nullptr );
  EXPECT_NEAR( *zero_gap_result.energy, 0.0, 1.0e-12 );
  EXPECT_NEAR( mfem::InnerProduct( MPI_COMM_WORLD, *zero_gap_result.residual, *zero_gap_result.residual ), 0.0,
               1.0e-12 );
  mfem::Vector direction( zero_gap_result.df_dx->Width() ), action( zero_gap_result.df_dx->Height() );
  for ( int dof = 0; dof < direction.Size(); ++dof ) {
    direction[dof] = dof + 1.0;
  }
  zero_gap_result.df_dx->Mult( direction, action );
  EXPECT_GT( mfem::InnerProduct( MPI_COMM_WORLD, action, action ), 1.0e-12 );
}

// Supported orders invalidate the formulation until init(); rejected orders leave the current object intact.
TEST_P( MfemEnergyMortarFormulationTest, configures_and_defers_formulation_rebuild )
{
  auto* coupling_scheme = tribol::CouplingSchemeManager::getInstance().findData( coupling_scheme_id );
  const auto location = std::get<0>( GetParam() );
  const auto set_and_check_order = [&]( int order ) {
    const int previous_order = coupling_scheme->getParameters().quadrature_order;
    auto* previous_formulation = coupling_scheme->getContactFormulation();
    tribol::setQuadratureOrder( coupling_scheme_id, order );
    EXPECT_EQ( coupling_scheme->hasContactFormulation(), previous_order == order );
    if ( previous_order == order ) {
      EXPECT_EQ( coupling_scheme->getContactFormulation(), previous_formulation );
    }
    EXPECT_EQ( coupling_scheme->getParameters().enforcement_location, location );
    EXPECT_EQ( coupling_scheme->getParameters().quadrature_order, order );
    ASSERT_TRUE( coupling_scheme->init() );
    ASSERT_TRUE( coupling_scheme->hasContactFormulation() );
  };
  set_and_check_order( quadratureOrder() );
  for ( int order = tribol::minimum_supported_quadrature_order; order <= tribol::maximum_supported_quadrature_order;
        ++order ) {
    set_and_check_order( order );
  }
  for ( int invalid :
        { tribol::minimum_supported_quadrature_order - 1, tribol::maximum_supported_quadrature_order + 1 } ) {
    auto* retained_formulation = coupling_scheme->getContactFormulation();
    EXPECT_SLIC_ERROR( tribol::setQuadratureOrder( coupling_scheme_id, invalid ) );
    EXPECT_EQ( coupling_scheme->getParameters().quadrature_order, tribol::maximum_supported_quadrature_order );
    EXPECT_EQ( coupling_scheme->getContactFormulation(), retained_formulation );
    coupling_scheme->getParameters().quadrature_order = invalid;
    EXPECT_SLIC_ERROR( tribol::createContactFormulation( coupling_scheme ) );
    coupling_scheme->getParameters().quadrature_order = tribol::maximum_supported_quadrature_order;
  }
  coupling_scheme->setContactFormulation( nullptr );
  constexpr int alternate_valid_order = 2;
  tribol::setQuadratureOrder( coupling_scheme_id, alternate_valid_order );
  EXPECT_FALSE( coupling_scheme->hasContactFormulation() );
  ASSERT_TRUE( coupling_scheme->init() );
  EXPECT_TRUE( coupling_scheme->hasContactFormulation() );
  EXPECT_EQ( coupling_scheme->getParameters().enforcement_location, location );
}

// Gap and residual calls must work independently, initialize only their outputs, and leave earlier results unchanged.
TEST_P( MfemEnergyMortarFormulationTest, returns_owned_outputs_for_each_calculation )
{
  tribol::updateMfemParallelDecomposition();
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  const auto& formulation = *coupling_scheme.getContactFormulation();
  const auto lower_mesh = coupling_scheme.getMesh1().getView();
  const auto upper_mesh = coupling_scheme.getMesh2().getView();
  mfem::Vector legacy_gap;
  EXPECT_SLIC_ERROR( tribol::getMfemGap( coupling_scheme_id, legacy_gap ) );
  if ( std::get<0>( GetParam() ) == tribol::EnforcementLocation::Nodal ) {
    auto retained_gap_result = formulation.computeNodalGaps( lower_mesh, upper_mesh );
    ASSERT_NE( retained_gap_result.gap, nullptr );
    ASSERT_NE( retained_gap_result.weighted_gap_jacobian, nullptr );
    EXPECT_EQ( retained_gap_result.residual, nullptr );
    EXPECT_EQ( retained_gap_result.pressure, nullptr );
    EXPECT_EQ( retained_gap_result.df_dx, nullptr );
    EXPECT_EQ( retained_gap_result.df_dp, nullptr );
    EXPECT_FALSE( retained_gap_result.energy.has_value() );
    EXPECT_FALSE( retained_gap_result.time_step.has_value() );
    const mfem::HypreParVector retained_gap_copy( *retained_gap_result.gap );
    auto subsequent_gap_result = formulation.computeNodalGaps( lower_mesh, upper_mesh );
    subsequent_gap_result.gap->Add( -1.0, retained_gap_copy );
    EXPECT_LT( subsequent_gap_result.gap->Norml2(), 1.0e-12 );
    EXPECT_EQ( retained_gap_result.gap->Norml2(), retained_gap_copy.Norml2() );
  } else {
    EXPECT_SLIC_ERROR( formulation.computeNodalGaps( lower_mesh, upper_mesh ) );
  }

  auto retained_residual_result = formulation.computeNodalResiduals( lower_mesh, upper_mesh, {} );
  ASSERT_NE( retained_residual_result.residual, nullptr );
  ASSERT_NE( retained_residual_result.df_dx, nullptr );
  ASSERT_TRUE( retained_residual_result.energy.has_value() );
  EXPECT_FALSE( retained_residual_result.time_step.has_value() );
  EXPECT_EQ( retained_residual_result.df_dp, nullptr );
  const bool nodal = std::get<0>( GetParam() ) == tribol::EnforcementLocation::Nodal;
  EXPECT_EQ( retained_residual_result.gap != nullptr, nodal );
  EXPECT_EQ( retained_residual_result.pressure != nullptr, nodal );
  EXPECT_EQ( retained_residual_result.weighted_gap_jacobian != nullptr, nodal );
  const mfem::HypreParVector retained_residual_copy( *retained_residual_result.residual );
  ASSERT_GT( retained_residual_copy.Norml2(), 0.0 );
  auto subsequent_residual_result = formulation.computeNodalResiduals( lower_mesh, upper_mesh, {} );
  subsequent_residual_result.residual->Add( -1.0, retained_residual_copy );
  EXPECT_LT( subsequent_residual_result.residual->Norml2(), 1.0e-12 );
  EXPECT_NEAR( *subsequent_residual_result.energy, *retained_residual_result.energy, 1.0e-12 );
  EXPECT_EQ( coupling_scheme.getContactFormulationResult().residual, nullptr );
  coupling_scheme.setContactFormulation( nullptr );
  auto retained_residual_difference = *retained_residual_result.residual;
  retained_residual_difference -= retained_residual_copy;
  EXPECT_EQ( retained_residual_difference.Norml2(), 0.0 );
  mfem::Vector direction( retained_residual_result.df_dx->Width() );
  mfem::Vector action( retained_residual_result.df_dx->Height() );
  direction = 1.0;
  retained_residual_result.df_dx->Mult( direction, action );
  EXPECT_TRUE( std::isfinite( action.Norml2() ) );
}

// Springs in series must scale the residual, Jacobian, and energy for either location and each host Gauss rule.
TEST_P( MfemEnergyMortarFormulationTest, accepts_series_penalty_stiffness )
{
  using namespace tribol;
  constexpr RealT first_series_penalty = 2.0;
  constexpr RealT second_series_penalty = 6.0;
  setMfemKinematicConstantPenalty( coupling_scheme_id, first_series_penalty, second_series_penalty );
  constexpr int contact_rank_count = 1;
  updateMfemParallelDecomposition( contact_rank_count, true );
  auto& coupling_scheme = CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  ASSERT_TRUE( coupling_scheme.init() );
  const auto series_penalty_result = coupling_scheme.getContactFormulation()->computeNodalResiduals(
      coupling_scheme.getMesh1().getView(), coupling_scheme.getMesh2().getView(), {} );
  mfem::Vector direction( series_penalty_result.df_dx->Width() );
  mfem::Vector series_penalty_jacobian_action( series_penalty_result.df_dx->Height() );
  for ( int dof = 0; dof < direction.Size(); ++dof ) {
    direction[dof] = 1.0 + 0.1 * dof;
  }
  series_penalty_result.df_dx->Mult( direction, series_penalty_jacobian_action );
  for ( const auto execution_mode : { ExecutionMode::Sequential
#ifdef TRIBOL_USE_OPENMP
                                      ,
                                      ExecutionMode::OpenMP
#endif
        } ) {
    SCOPED_TRACE( static_cast<int>( execution_mode ) );
    auto average_penalty_formulation = dispatchExecutionMode( execution_mode, [&]<ExecutionMode Exec>() {
      return createWithQuadratureRuleGenerator(
          quadratureOrder(), SmoothedProjectedEdgeMapping{},
          [&]( auto rule_generator ) -> std::unique_ptr<ContactFormulation> {
            auto coarse_search =
                createCoarseSearch( coupling_scheme.spatialDim(), coupling_scheme.getBinningMethod(), execution_mode,
                                    coupling_scheme.getEffectiveBinningProximityScale(), 0.0 );
            auto projected_edge_search = std::make_unique<ProjectedEdgeSearch>( execution_mode, false );
            PairListBuilder pair_builder{ std::move( coarse_search ), std::move( projected_edge_search ) };
            if ( std::get<0>( GetParam() ) == EnforcementLocation::Nodal ) {
              return std::make_unique<NodalEnergyMortar<Exec, decltype( rule_generator ), OffsetGap<OpposingNormalGap>,
                                                        NodalPenaltyEnforcement>>(
                  *coupling_scheme.getMfemMeshData(), *coupling_scheme.getMfemSubmeshData(),
                  *coupling_scheme.getMfemJacobianData(), std::move( rule_generator ),
                  OffsetGap{ OpposingNormalGap{}, 0.0 },
                  NodalPenaltyEnforcement{ SeriesPenaltyStiffness{}( first_series_penalty, second_series_penalty ) },
                  std::move( pair_builder ) );
            }
            return std::make_unique<
                QuadraturePointEnergyMortar<Exec, decltype( rule_generator ), OffsetGap<OpposingNormalGap>>>(
                *coupling_scheme.getMfemMeshData(), *coupling_scheme.getMfemJacobianData(), std::move( rule_generator ),
                SeriesPenaltyStiffness{}( first_series_penalty, second_series_penalty ),
                OffsetGap{ OpposingNormalGap{}, 0.0 }, std::move( pair_builder ) );
          } );
    } );
    auto average_penalty_result = average_penalty_formulation->computeNodalResiduals(
        coupling_scheme.getMesh1().getView(), coupling_scheme.getMesh2().getView(), {} );
    ASSERT_NE( average_penalty_result.residual, nullptr );
    ASSERT_NE( average_penalty_result.df_dx, nullptr );
    ASSERT_TRUE( average_penalty_result.energy.has_value() );
    // The stiffnesses 2 and 6 give series stiffness 1.5 instead of the default arithmetic mean 4.
    constexpr RealT stiffness_ratio = 1.5 / 4.0;
    average_penalty_result.residual->Add( -stiffness_ratio, *series_penalty_result.residual );
    EXPECT_LT( average_penalty_result.residual->Norml2(), 1.0e-11 );
    EXPECT_NEAR( *average_penalty_result.energy, stiffness_ratio * *series_penalty_result.energy, 1.0e-11 );
    mfem::Vector average_penalty_jacobian_action( average_penalty_result.df_dx->Height() );
    average_penalty_result.df_dx->Mult( direction, average_penalty_jacobian_action );
    average_penalty_jacobian_action.Add( -stiffness_ratio, series_penalty_jacobian_action );
    EXPECT_LT( average_penalty_jacobian_action.Norml2(), 1.0e-11 );
  }
}

// Unsupported locations must leave the selected formulation and its transfer fields unchanged.
TEST_P( MfemEnergyMortarFormulationTest, rejects_invalid_location_without_rebuilding )
{
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  auto* formulation = coupling_scheme.getContactFormulation();
  auto* submesh_data = coupling_scheme.getMfemSubmeshData();
  const auto location = coupling_scheme.getParameters().enforcement_location;
  constexpr auto invalid_location = static_cast<tribol::EnforcementLocation>( -1 );
  EXPECT_SLIC_ERROR( tribol::setEnforcementLocation( coupling_scheme_id, invalid_location ) );
  EXPECT_EQ( coupling_scheme.getParameters().enforcement_location, location );
  EXPECT_EQ( coupling_scheme.getContactFormulation(), formulation );
  EXPECT_EQ( coupling_scheme.getMfemSubmeshData(), submesh_data );
}

// An unconfigured penalty must report an error before a pair formulation can calculate a residual.
TEST_P( MfemEnergyMortarFormulationTest, rejects_unconfigured_pair_penalty )
{
  tribol::setEnforcementLocation( coupling_scheme_id, tribol::EnforcementLocation::QuadraturePoint );
  constexpr int contact_rank_count = 1;
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  coupling_scheme.getEnforcementOptions().penalty_options.kinematic_calc_set = false;
  coupling_scheme.invalidateContactFormulation();
  EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
  tribol::RealT dt = 1.0;
  EXPECT_SLIC_ERROR( tribol::update( coupling_scheme_id, 0.0, dt ) );
  EXPECT_FALSE( coupling_scheme.hasMfemSubmeshData() );
}

// Existing penalty calculations retain their construction stiffness even if registered constants change or disappear.
TEST_P( MfemEnergyMortarFormulationTest, keeps_penalty_fixed_until_formulation_is_rebuilt )
{
  constexpr int contact_rank_count = 1;
  constexpr tribol::RealT configured_penalty = 100.0;
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  auto& mesh_data = *coupling_scheme.getMfemMeshData();
  const auto compute_residual = [&] {
    return coupling_scheme.getContactFormulation()->computeNodalResiduals( coupling_scheme.getMesh1().getView(),
                                                                           coupling_scheme.getMesh2().getView(), {} );
  };
  const auto original_penalty_result = compute_residual();
  const auto check_unchanged = [&] {
    auto unchanged_penalty_result = compute_residual();
    unchanged_penalty_result.residual->Add( -1.0, *original_penalty_result.residual );
    EXPECT_LT( unchanged_penalty_result.residual->Norml2(), 1.0e-12 );
    EXPECT_EQ( unchanged_penalty_result.energy, original_penalty_result.energy );
  };
  mesh_data.ClearAllPenaltyData();
  check_unchanged();
  EXPECT_EQ( tribol::createContactFormulation( &coupling_scheme ), nullptr );
  mesh_data.SetMesh1KinematicConstantPenalty( configured_penalty );
  check_unchanged();
  EXPECT_EQ( tribol::createContactFormulation( &coupling_scheme ), nullptr );
  mesh_data.ClearAllPenaltyData();
  mesh_data.SetMesh2KinematicConstantPenalty( configured_penalty );
  check_unchanged();
  EXPECT_EQ( tribol::createContactFormulation( &coupling_scheme ), nullptr );
  mesh_data.SetMesh1KinematicConstantPenalty( 300.0 );
  check_unchanged();
  // The public setter invalidates the old formulation; init() builds the replacement without refreshing decomposition.
  constexpr tribol::RealT updated_penalty = 300.0;
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, updated_penalty, configured_penalty );
  EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
  ASSERT_TRUE( coupling_scheme.init() );
  auto rebuilt_penalty_result = compute_residual();
  rebuilt_penalty_result.residual->Add( -2.0, *original_penalty_result.residual );
  EXPECT_LT( rebuilt_penalty_result.residual->Norml2(), 1.0e-12 );
  EXPECT_NEAR( *rebuilt_penalty_result.energy, 2.0 * *original_penalty_result.energy, 1.0e-12 );
}

/** @brief Start tests with a coupling scheme registered before its penalty settings are supplied. */
class MfemEnergyMortarRegistrationTest : public MfemEnergyMortarFormulationTest {
 protected:
  /** @brief Register the scheme while leaving penalty selection and constants unset. */
  void SetUp() override { registerScheme(); }
};

// Complete penalty settings remain pending until update() initializes the formulation.
TEST_P( MfemEnergyMortarRegistrationTest, constructs_formulation_during_update )
{
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
  tribol::setEnforcementLocation( coupling_scheme_id, std::get<0>( GetParam() ) );
  tribol::setQuadratureOrder( coupling_scheme_id, quadratureOrder() );
  EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
  tribol::RealT dt = 1.0;
  EXPECT_SLIC_ERROR( tribol::update( coupling_scheme_id, 0.0, dt ) );
  constexpr tribol::RealT configured_penalty = 100.0;
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, configured_penalty, configured_penalty );
  ASSERT_FALSE( coupling_scheme.hasContactFormulation() );
  const auto location = std::get<0>( GetParam() );
  EXPECT_EQ( coupling_scheme.getParameters().enforcement_location, location );
  EXPECT_EQ( coupling_scheme.getParameters().quadrature_order, quadratureOrder() );
  constexpr int contact_rank_count = 1;
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  ASSERT_EQ( tribol::update( coupling_scheme_id, 0.0, dt ), 0 );
  ASSERT_TRUE( coupling_scheme.hasContactFormulation() );
  EXPECT_GT( tribol::getMfemContactResidual( 0 ).Norml2(), 0.0 );
}

// init() can construct the formulation and its field rows on an existing decomposition without refreshing geometry.
TEST_P( MfemEnergyMortarRegistrationTest, constructs_formulation_on_existing_decomposition )
{
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  tribol::setEnforcementLocation( coupling_scheme_id, std::get<0>( GetParam() ) );
  tribol::setQuadratureOrder( coupling_scheme_id, quadratureOrder() );
  constexpr int contact_rank_count = 1;
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
  constexpr tribol::RealT configured_penalty = 100.0;
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, configured_penalty, configured_penalty );
  ASSERT_FALSE( coupling_scheme.hasContactFormulation() );
  ASSERT_TRUE( coupling_scheme.init() );
  ASSERT_TRUE( coupling_scheme.hasContactFormulation() );
  EXPECT_FALSE( coupling_scheme.getMesh1().getNodalFields().m_is_node_gap_set );
  EXPECT_FALSE( coupling_scheme.getMesh2().getNodalFields().m_is_node_gap_set );
  const auto formulation_result = coupling_scheme.getContactFormulation()->computeNodalResiduals(
      coupling_scheme.getMesh1().getView(), coupling_scheme.getMesh2().getView(), {} );
  ASSERT_NE( formulation_result.residual, nullptr );
  ASSERT_TRUE( formulation_result.energy.has_value() );
  EXPECT_GT( formulation_result.residual->Norml2(), 0.0 );
}

// Rejected mesh stiffnesses must leave both the installed formulation and its registered inputs unchanged.
TEST_P( MfemEnergyMortarFormulationTest, rejects_invalid_mesh_penalties_without_changing_scheme )
{
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  auto* formulation = coupling_scheme.getContactFormulation();
  const auto& mesh_data = *coupling_scheme.getMfemMeshData();
  const tribol::RealT mesh1_penalty = *mesh_data.GetMesh1KinematicConstantPenalty();
  const tribol::RealT mesh2_penalty = *mesh_data.GetMesh2KinematicConstantPenalty();
  const auto check_unchanged = [&] {
    EXPECT_EQ( coupling_scheme.getContactFormulation(), formulation );
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
  constexpr tribol::RealT first_series_penalty = 2.0;
  constexpr tribol::RealT second_series_penalty = 6.0;
  EXPECT_NO_THROW(
      tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, first_series_penalty, second_series_penalty ) );
}

// Invalid offsets must be rejected before the search and gap definitions capture them.
TEST_P( MfemEnergyMortarFormulationTest, rejects_invalid_residual_gaps_without_rebuilding )
{
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  auto* formulation = coupling_scheme.getContactFormulation();
  const tribol::RealT residual_gap = coupling_scheme.getParameters().residual_gap;
  for ( const tribol::RealT invalid :
        { -1.0, -std::numeric_limits<tribol::RealT>::infinity(), std::numeric_limits<tribol::RealT>::infinity(),
          std::numeric_limits<tribol::RealT>::quiet_NaN() } ) {
    SCOPED_TRACE( invalid );
    EXPECT_SLIC_ERROR( tribol::setResidualGap( coupling_scheme_id, invalid ) );
    EXPECT_EQ( coupling_scheme.getParameters().residual_gap, residual_gap );
    EXPECT_EQ( coupling_scheme.getContactFormulation(), formulation );
  }
}

// Penalty data on an LM scheme must not make quadrature-point enforcement appear valid.
TEST( MfemEnergyMortarConfiguration, rejects_quadrature_point_lm_without_releasing_nodal_state )
{
  struct FinalizeAfterTest {
    ~FinalizeAfterTest() { tribol::finalize(); }
  };
  tribol::finalize();
  // The two-edge bottom and top boundaries of one unit square provide the nodal multiplier field used by this
  // configuration test; their physical separation is not part of the assertion.
  mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( 2, 1 ) );
  mfem::H1_FECollection coordinate_collection( 1, 2 );
  mfem::ParFiniteElementSpace coordinate_space( &mesh, &coordinate_collection, 2 );
  mfem::ParGridFunction coordinates( &coordinate_space );
  mesh.GetNodes( coordinates );
  [[maybe_unused]] FinalizeAfterTest finalize_after_test;
  constexpr tribol::IndexT coupling_scheme_id = 0;
  tribol::registerMfemCouplingScheme(
      0, 0, 1, mesh, coordinates, { 1 }, { 3 }, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::ENERGY_MORTAR,
      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
  constexpr tribol::RealT configured_penalty = 100.0;
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, configured_penalty, configured_penalty );
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  ASSERT_FALSE( coupling_scheme.hasContactFormulation() );
  auto* multiplier = &tribol::getMfemContactPressure( coupling_scheme_id );
  ASSERT_FALSE( coupling_scheme.hasContactFormulation() );
  ASSERT_TRUE( coupling_scheme.init() );
  EXPECT_EQ( &tribol::getMfemContactPressure( coupling_scheme_id ), multiplier );
  *multiplier = 2.0;
  const tribol::RealT multiplier_norm = multiplier->Norml2();
  constexpr int minimum_three_point_order = two_point_polynomial_order + 1;
  tribol::setQuadratureOrder( coupling_scheme_id, minimum_three_point_order );
  EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
  EXPECT_EQ( &tribol::getMfemContactPressure( coupling_scheme_id ), multiplier );
  EXPECT_EQ( multiplier->Norml2(), multiplier_norm );
  ASSERT_TRUE( coupling_scheme.init() );
  EXPECT_EQ( &tribol::getMfemContactPressure( coupling_scheme_id ), multiplier );
  EXPECT_EQ( multiplier->Norml2(), multiplier_norm );
  auto* formulation = coupling_scheme.getContactFormulation();
  auto* submesh_data = coupling_scheme.getMfemSubmeshData();
  auto* jacobian_data = coupling_scheme.getMfemJacobianData();

  EXPECT_SLIC_ERROR(
      tribol::setEnforcementLocation( coupling_scheme_id, tribol::EnforcementLocation::QuadraturePoint ) );
  EXPECT_EQ( coupling_scheme.getParameters().enforcement_location, tribol::EnforcementLocation::Nodal );
  EXPECT_EQ( coupling_scheme.getContactFormulation(), formulation );
  EXPECT_EQ( coupling_scheme.getMfemSubmeshData(), submesh_data );
  EXPECT_EQ( coupling_scheme.getMfemJacobianData(), jacobian_data );
  EXPECT_EQ( &tribol::getMfemContactPressure( coupling_scheme_id ), multiplier );
}

// LM residuals must read the multiplier passed in ContactFormulationInputs, not the submesh pressure field.
TEST( MfemEnergyMortarConfiguration, lm_residuals_use_supplied_multiplier )
{
  struct FinalizeAfterTest {
    ~FinalizeAfterTest() { tribol::finalize(); }
  };
  tribol::finalize();
  // As above, the unit square supplies two aligned edges on each registered boundary. This test changes only which
  // multiplier field is passed to the formulation.
  mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( 2, 1 ) );
  mfem::H1_FECollection coordinate_collection( 1, 2 );
  mfem::ParFiniteElementSpace coordinate_space( &mesh, &coordinate_collection, 2 );
  mfem::ParGridFunction coordinates( &coordinate_space );
  mesh.GetNodes( coordinates );
  [[maybe_unused]] FinalizeAfterTest finalize_after_test;
  constexpr tribol::IndexT coupling_scheme_id = 0;
  tribol::registerMfemCouplingScheme(
      0, 0, 1, mesh, coordinates, { 1 }, { 3 }, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::ENERGY_MORTAR,
      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
  constexpr int contact_rank_count = 1;
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  ASSERT_TRUE( coupling_scheme.init() );
  const auto lower_mesh = coupling_scheme.getMesh1().getView();
  const auto upper_mesh = coupling_scheme.getMesh2().getView();
  const auto* formulation = coupling_scheme.getContactFormulation();
  ASSERT_NE( formulation, nullptr );

  EXPECT_SLIC_ERROR( formulation->computeNodalResiduals( lower_mesh, upper_mesh, {} ) );

  constexpr tribol::RealT multiplier_value = 2.0;
  // A different value in the submesh pressure field must not affect the calculation.
  constexpr tribol::RealT ignored_pressure_value = 5.0;
  auto& multiplier = tribol::getMfemContactPressure( coupling_scheme_id );
  multiplier = multiplier_value;
  tribol::getMfemPressure( coupling_scheme_id ) = ignored_pressure_value;
  tribol::ContactFormulationInputs inputs;
  inputs.multiplier = &multiplier;
  const auto multiplier_result = formulation->computeNodalResiduals( lower_mesh, upper_mesh, inputs );
  ASSERT_NE( multiplier_result.pressure, nullptr );
  ASSERT_NE( multiplier_result.gap, nullptr );
  ASSERT_TRUE( multiplier_result.energy.has_value() );
  mfem::Vector pressure_error( *multiplier_result.pressure );
  pressure_error -= multiplier;
  EXPECT_EQ( pressure_error.Normlinf(), 0.0 );
  const tribol::RealT weighted_gap_sum =
      mfem::InnerProduct( MPI_COMM_WORLD, *multiplier_result.gap, *multiplier_result.gap );
  ASSERT_GT( weighted_gap_sum, 0.0 ) << "The configuration must produce nonzero weighted gaps.";
  EXPECT_NEAR( *multiplier_result.energy, mfem::InnerProduct( MPI_COMM_WORLD, multiplier, *multiplier_result.gap ),
               1.0e-12 );
}

// Energy Mortar uses scalar transfer data for nodal assembly but returns gap and pressure through its result.
TEST_P( MfemEnergyMortarFormulationTest, creates_only_the_required_transfer_fields )
{
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  const bool nodal = std::get<0>( GetParam() ) == tribol::EnforcementLocation::Nodal;
  EXPECT_EQ( coupling_scheme.hasMfemSubmeshData(), nodal );
  ASSERT_TRUE( coupling_scheme.hasMfemJacobianData() );
  constexpr int contact_rank_count = 1;
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  EXPECT_EQ( coupling_scheme.hasMfemSubmeshData(), nodal );
  EXPECT_EQ( coupling_scheme.getMfemJacobianData()->ParentPath().final_fes, coordinate_space_.get() );
  EXPECT_FALSE( coupling_scheme.getMesh1().getNodalFields().m_is_node_gap_set );
  EXPECT_FALSE( coupling_scheme.getMesh1().getNodalFields().m_is_node_pressure_set );
  EXPECT_FALSE( coupling_scheme.getMesh2().getNodalFields().m_is_node_gap_set );
  EXPECT_FALSE( coupling_scheme.getMesh2().getNodalFields().m_is_node_pressure_set );
}

// Switching locations leaves one pending rebuild; init() creates the new transfers without rebuilding decomposition.
TEST_P( MfemEnergyMortarFormulationTest, switches_locations_without_rebuilding_the_decomposition )
{
  constexpr int contact_rank_count = 1;
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  std::array<std::unique_ptr<mfem::HypreParVector>, 2> previous_residuals;
  std::array<std::optional<tribol::RealT>, 2> previous_energies;
  for ( const auto location : { tribol::EnforcementLocation::Nodal, tribol::EnforcementLocation::QuadraturePoint,
                                tribol::EnforcementLocation::Nodal, tribol::EnforcementLocation::QuadraturePoint } ) {
    const auto previous_location = coupling_scheme.getParameters().enforcement_location;
    auto* previous_formulation = coupling_scheme.getContactFormulation();
    tribol::setEnforcementLocation( coupling_scheme_id, location );
    const bool nodal = location == tribol::EnforcementLocation::Nodal;
    if ( previous_location == location ) {
      EXPECT_EQ( coupling_scheme.getContactFormulation(), previous_formulation );
    } else {
      EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
    }
    ASSERT_TRUE( coupling_scheme.init() );
    EXPECT_EQ( coupling_scheme.hasMfemSubmeshData(), nodal );
    EXPECT_EQ( coupling_scheme.getContactFormulationResult().residual, nullptr );
    EXPECT_FALSE( coupling_scheme.getMesh1().getNodalFields().m_is_node_gap_set );
    EXPECT_FALSE( coupling_scheme.getMesh1().getNodalFields().m_is_node_pressure_set );
    auto location_result = coupling_scheme.getContactFormulation()->computeNodalResiduals(
        coupling_scheme.getMesh1().getView(), coupling_scheme.getMesh2().getView(), {} );
    ASSERT_NE( location_result.residual, nullptr );
    ASSERT_NE( location_result.df_dx, nullptr );
    ASSERT_TRUE( location_result.energy.has_value() );
    EXPECT_EQ( location_result.gap != nullptr, nodal );
    EXPECT_EQ( location_result.pressure != nullptr, nodal );
    EXPECT_EQ( location_result.weighted_gap_jacobian != nullptr, nodal );
    const int index = nodal ? 0 : 1;
    if ( previous_residuals[index] ) {
      auto residual_difference = *location_result.residual;
      residual_difference -= *previous_residuals[index];
      EXPECT_LT( residual_difference.Norml2(), 1.0e-12 );
      EXPECT_NEAR( *location_result.energy, *previous_energies[index], 1.0e-12 );
    } else {
      previous_residuals[index] = std::make_unique<mfem::HypreParVector>( *location_result.residual );
      previous_energies[index] = location_result.energy;
    }
    mfem::Vector direction( location_result.df_dx->Width() ), jacobian_action( location_result.df_dx->Height() );
    direction = 1.0;
    location_result.df_dx->Mult( direction, jacobian_action );
    EXPECT_TRUE( std::isfinite( jacobian_action.Norml2() ) );
  }
}

// Updating the two penalties must use their arithmetic mean without requiring geometry transfer to run again.
TEST_P( MfemEnergyMortarFormulationTest, uses_updated_penalties_without_redecomposition )
{
  constexpr int contact_rank_count = 1;
  tribol::updateMfemParallelDecomposition( contact_rank_count, true );
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  auto original_penalty_result = coupling_scheme.getContactFormulation()->computeNodalResiduals(
      coupling_scheme.getMesh1().getView(), coupling_scheme.getMesh2().getView(), {} );
  // Retain only parent-space outputs while rebuilding the formulation.
  const auto original_residual = *original_penalty_result.residual;
  const auto original_energy = *original_penalty_result.energy;
  original_penalty_result = {};
  constexpr tribol::RealT updated_penalty = 300.0;
  constexpr tribol::RealT configured_penalty = 100.0;
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, configured_penalty, updated_penalty );
  EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
  ASSERT_TRUE( coupling_scheme.init() );
  auto updated_penalty_result = coupling_scheme.getContactFormulation()->computeNodalResiduals(
      coupling_scheme.getMesh1().getView(), coupling_scheme.getMesh2().getView(), {} );
  updated_penalty_result.residual->Add( -2.0, original_residual );
  EXPECT_LT( updated_penalty_result.residual->Norml2(), 1.0e-11 );
  EXPECT_NEAR( *updated_penalty_result.energy, 2.0 * original_energy, 1.0e-11 );
}

// Penalty activation must remove positive weighted gaps and their derivative rows; LM gaps remain signed and weighted.
TEST_P( MfemEnergyMortarFormulationTest, nodal_activation_preserves_only_compressive_gap_rows )
{
  using namespace tribol;
  if ( std::get<0>( GetParam() ) != EnforcementLocation::Nodal ) {
    GTEST_SKIP() << "This check requires nodal enforcement.";
  }
  // The top boundary crosses the bottom boundary, producing both opening and penetration along the same interface.
  for ( int node = 0; node < coordinate_space_->GetNDofs(); ++node ) {
    if ( ( *coordinates_ )[coordinate_space_->DofToVDof( node, 1 )] > 0.5 ) {
      const RealT x = ( *coordinates_ )[coordinate_space_->DofToVDof( node, 0 )];
      ( *coordinates_ )[coordinate_space_->DofToVDof( node, 1 )] = 0.2 - 0.4 * x;
    }
  }
  constexpr int contact_rank_count = 1;
  updateMfemParallelDecomposition( contact_rank_count, true );
  auto& coupling_scheme = CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  const auto penalty_gap_result = coupling_scheme.getContactFormulation()->computeNodalGaps(
      coupling_scheme.getMesh1().getView(), coupling_scheme.getMesh2().getView() );
  const auto signed_gap_result =
      createWithQuadratureRuleGenerator( quadratureOrder(), SmoothedProjectedEdgeMapping{}, [&]( auto rule_generator ) {
        auto coarse_search = createCoarseSearch( coupling_scheme.spatialDim(), coupling_scheme.getBinningMethod(),
                                                 coupling_scheme.getExecutionMode(),
                                                 coupling_scheme.getEffectiveBinningProximityScale(), 0.0 );
        auto refined_search = std::make_unique<ProjectedEdgeSearch>( coupling_scheme.getExecutionMode(), false );
        PairListBuilder pair_builder{ std::move( coarse_search ), std::move( refined_search ) };
        NodalEnergyMortar<ExecutionMode::Sequential, decltype( rule_generator ), OffsetGap<OpposingNormalGap>,
                          NodalLagrangeMultiplierEnforcement>
            lagrange_multiplier_formulation( *coupling_scheme.getMfemMeshData(), *coupling_scheme.getMfemSubmeshData(),
                                             *coupling_scheme.getMfemJacobianData(), std::move( rule_generator ),
                                             OffsetGap{ OpposingNormalGap{}, 0.0 },
                                             NodalLagrangeMultiplierEnforcement{}, std::move( pair_builder ) );
        return lagrange_multiplier_formulation.computeNodalGaps( coupling_scheme.getMesh1().getView(),
                                                                 coupling_scheme.getMesh2().getView() );
      } );
  mfem::Vector direction( coordinate_space_->GetTrueVSize() ), penalty_action( penalty_gap_result.gap->Size() ),
      signed_gap_action( signed_gap_result.gap->Size() );
  direction = 0.0;
  for ( int dof = 0; dof < direction.Size(); ++dof ) {
    direction[dof] = dof + 1.0;
  }
  penalty_gap_result.weighted_gap_jacobian->Mult( direction, penalty_action );
  signed_gap_result.weighted_gap_jacobian->Mult( direction, signed_gap_action );
  int local_open = 0, local_closed = 0;
  for ( int node = 0; node < signed_gap_result.gap->Size(); ++node ) {
    if ( ( *signed_gap_result.gap )[node] > 1.0e-12 ) {
      ++local_open;
      EXPECT_EQ( ( *penalty_gap_result.gap )[node], 0.0 );
      EXPECT_NEAR( penalty_action[node], 0.0, 1.0e-12 );
    } else if ( ( *signed_gap_result.gap )[node] < -1.0e-12 ) {
      ++local_closed;
      EXPECT_LT( ( *penalty_gap_result.gap )[node], 0.0 );
      EXPECT_NEAR( penalty_action[node], signed_gap_action[node], 1.0e-12 );
    }
  }
  int global_open = 0, global_closed = 0;
  MPI_Allreduce( &local_open, &global_open, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_closed, &global_closed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( global_open, 0 );
  EXPECT_GT( global_closed, 0 );
}

// Assembled gradients and Hessians must match finite differences as gap, area, and edge length vary.
TEST_P( MfemEnergyMortarFormulationTest, assembled_residual_and_jacobian_match_directional_finite_difference )
{
  constexpr tribol::RealT residual_gap = 1.2;
  tribol::setResidualGap( coupling_scheme_id, residual_gap );
  auto& coupling_scheme = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
  ASSERT_TRUE( coupling_scheme.init() );
  mfem::ParGridFunction base_coordinates( *coordinates_ );
  mfem::ParGridFunction coordinate_direction( coordinate_space_.get() );
  mfem::VectorFunctionCoefficient direction_coeff( 2, []( const mfem::Vector& x, mfem::Vector& v ) {
    v.SetSize( 2 );
    v[0] = 0.3 * x[0] * ( 1.0 - x[0] ) + 0.1 * x[1];
    v[1] = -0.2 + 0.15 * x[0] - 0.1 * x[1];
  } );
  coordinate_direction.ProjectCoefficient( direction_coeff );
  mfem::HypreParVector coordinate_true_direction( coordinate_space_.get() );
  coordinate_direction.GetTrueDofs( coordinate_true_direction );

  auto evaluate_at = [&]( tribol::RealT displacement_scale ) {
    *coordinates_ = base_coordinates;
    coordinates_->Add( displacement_scale, coordinate_direction );
    constexpr int contact_rank_count = 1;
    tribol::updateMfemParallelDecomposition( contact_rank_count, true );
    return coupling_scheme.getContactFormulation()->computeNodalResiduals( coupling_scheme.getMesh1().getView(),
                                                                           coupling_scheme.getMesh2().getView(), {} );
  };

  const auto base_result = evaluate_at( 0.0 );
  constexpr tribol::RealT step = 1.0e-6;
  const auto forward_result = evaluate_at( step );
  const auto backward_result = evaluate_at( -step );

  const tribol::RealT directional_residual =
      mfem::InnerProduct( MPI_COMM_WORLD, *base_result.residual, coordinate_true_direction );
  const tribol::RealT finite_difference_energy = ( *forward_result.energy - *backward_result.energy ) / ( 2.0 * step );
  EXPECT_NEAR( directional_residual, finite_difference_energy, 1.0e-5 );

  mfem::Vector jacobian_action( coordinate_space_->GetTrueVSize() );
  base_result.df_dx->Mult( coordinate_true_direction, jacobian_action );
  mfem::Vector finite_difference_residual( *forward_result.residual );
  finite_difference_residual -= *backward_result.residual;
  finite_difference_residual *= 1.0 / ( 2.0 * step );
  jacobian_action -= finite_difference_residual;
  const tribol::RealT jacobian_error =
      std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, jacobian_action, jacobian_action ) );
  EXPECT_LT( jacobian_error, 2.0e-4 );
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
  constexpr tribol::IndexT coupling_scheme_id = 0;
  constexpr int valid_quadrature_order = 2;
  EXPECT_SLIC_ERROR( tribol::setQuadratureOrder( coupling_scheme_id, valid_quadrature_order ) );
  Edges edges;
  constexpr tribol::IndexT first_mesh_id = 0;
  constexpr tribol::IndexT second_mesh_id = 1;
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
INSTANTIATE_TEST_SUITE_P( enforcement, MfemEnergyMortarFormulationTest,
                          testing::Combine( testing::Values( tribol::EnforcementLocation::Nodal,
                                                             tribol::EnforcementLocation::QuadraturePoint ),
                                            testing::Values( one_point_polynomial_order, two_point_polynomial_order,
                                                             three_point_polynomial_order ) ) );

/** @brief Exercise delayed registration with both enforcement locations and each supported quadrature rule. */
INSTANTIATE_TEST_SUITE_P( enforcement, MfemEnergyMortarRegistrationTest,
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
  const int test_status = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_status;
}
