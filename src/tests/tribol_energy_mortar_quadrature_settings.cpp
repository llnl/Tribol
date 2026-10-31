// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Tests factory selection and changes to Energy Mortar quadrature order for both enforcement locations.
 * Direct gap and force calls check returned outputs and repeated calculations without a decomposition refresh.
 * One- and two-rank runs also check parent-only storage, switching locations, and changed penalties; decomposition
 * onto one rank covers a contact rank with no elements.
 * Local quadrature accuracy is checked in tribol_pair_quadrature; gap and force derivatives have separate tests.
 */
#include <gtest/gtest.h>
#include <cmath>
#include <array>
#include <optional>
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/ContactFormulationFactory.hpp"
#include "tribol/physics/EnergyMortarAdapter.hpp"
#include "tribol/physics/PairContactFormulation.hpp"
#include "tribol/integ/QuadratureRuleFactory.hpp"
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
  /**
   * @brief Register the MFEM coupling scheme and configure its enforcement location and quadrature order.
   */
  void SetUp() override
  {
    mesh_ = std::make_unique<mfem::ParMesh>( static_cast<const mfem::ParMesh&>(
        shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( 2, 1 ) ) ) );
    fec_ = std::make_unique<mfem::H1_FECollection>( 1, 2 );
    fes_ = std::make_unique<mfem::ParFiniteElementSpace>( mesh_.get(), fec_.get(), 2 );
    coords_ = std::make_unique<mfem::ParGridFunction>( fes_.get() );
    mesh_->GetNodes( *coords_ );
    tribol::registerMfemCouplingScheme( 0, 0, 1, *mesh_, *coords_, { 1 }, { 3 }, tribol::SURFACE_TO_SURFACE,
                                        tribol::NO_SLIDING, tribol::ENERGY_MORTAR, tribol::FRICTIONLESS,
                                        tribol::PENALTY, tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
    EXPECT_EQ( tribol::CouplingSchemeManager::getInstance().findData( 0 )->getParameters().quadrature_order, 5 );
    tribol::setEnforcementLocation( 0, std::get<0>( GetParam() ) );
    tribol::setQuadratureOrder( 0, quadratureOrder() );
    tribol::setMfemKinematicConstantPenalty( 0, 100, 100 );
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

/**
 * @brief Check the selected adapter specialization without exposing generated quadrature as a public API.
 * @tparam Count Expected Gauss point count.
 * @param formulation Factory-created contact formulation.
 * @param location Selected enforcement location.
 * @return Whether the adapter has the expected rule and enforcement types.
 */
template <int Count>
bool usesPointCount( tribol::ContactFormulation* formulation, tribol::EnforcementLocation location )
{
  using Rule =
      tribol::PairQuadratureRule<tribol::SmoothedProjectedEdgeMapping, tribol::EdgeQuadratureRuleGenerator<Count>>;
  if ( location == tribol::EnforcementLocation::Nodal ) {
    return dynamic_cast<tribol::EnergyMortarAdapter<Rule>*>( formulation ) != nullptr;
  }
  return dynamic_cast<tribol::PairContactFormulation<tribol::EnergyMortarPairMethod<Rule>>*>( formulation ) != nullptr;
}

// The default uses three points; changing order selects the requested specialization for either enforcement location.
TEST_P( EnergyMortarQuadratureSettingsTest, selects_and_rebuilds_adapter )
{
  auto* cs = tribol::CouplingSchemeManager::getInstance().findData( 0 );
  const auto location = std::get<0>( GetParam() );
  const auto check = [&]( int order ) {
    tribol::setQuadratureOrder( 0, order );
    auto* formulation = cs->getContactFormulation();
    ASSERT_NE( formulation, nullptr );
    if ( order <= 1 )
      EXPECT_TRUE( usesPointCount<1>( formulation, location ) );
    else if ( order <= 3 )
      EXPECT_TRUE( usesPointCount<2>( formulation, location ) );
    else
      EXPECT_TRUE( usesPointCount<3>( formulation, location ) );
  };
  check( quadratureOrder() );
  for ( int order = 0; order <= 5; ++order ) check( order );
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
  EXPECT_TRUE( usesPointCount<2>( cs->getContactFormulation(), location ) );
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

// Changing only the combination to springs in series must scale pair energy and force for every Gauss count.
TEST_P( EnergyMortarQuadratureSettingsTest, accepts_series_penalty_stiffness )
{
  using namespace tribol;
  setEnforcementLocation( 0, EnforcementLocation::QuadraturePoint );
  setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
  updateMfemParallelDecomposition( 1, true );
  auto& cs = CouplingSchemeManager::getInstance().at( 0 );
  const auto baseline =
      cs.getContactFormulation()->computeNodalForces( cs.getMesh1().getView(), cs.getMesh2().getView() );
  auto alternate = createWithQuadratureRule(
      quadratureOrder(), SmoothedProjectedEdgeMapping{}, [&]( auto rule ) -> std::unique_ptr<ContactFormulation> {
        using Method = EnergyMortarPairMethod<decltype( rule ), SeriesPenaltyStiffness>;
        auto coarse = createCoarseSearch( cs.spatialDimension(), cs.getBinningMethod(), cs.getExecutionMode(),
                                          cs.getEffectiveBinningProximityScale(), 0.0 );
        auto refined = std::make_unique<ProjectedEdgeSearch>( cs.getExecutionMode(), false );
        return std::make_unique<PairContactFormulation<Method>>(
            *cs.getMfemMeshData(), *cs.getMfemJacobianData(), std::move( coarse ), std::move( refined ),
            Method{ std::move( rule ), 0.0, RegisteredPenaltyInputs<SeriesPenaltyStiffness>{ true } } );
      } );
  auto result = alternate->computeNodalForces( cs.getMesh1().getView(), cs.getMesh2().getView() );
  ASSERT_NE( result.force, nullptr );
  ASSERT_TRUE( result.energy.has_value() );
  // The stiffnesses 2 and 6 give series stiffness 1.5 instead of the default arithmetic mean 4.
  constexpr RealT stiffness_ratio = 1.5 / 4.0;
  result.force->Add( -stiffness_ratio, *baseline.force );
  EXPECT_LT( result.force->Norml2(), 1.0e-11 );
  EXPECT_NEAR( *result.energy, stiffness_ratio * *baseline.energy, 1.0e-11 );
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
  EXPECT_THROW( cs.getContactFormulation()->computeNodalForces( cs.getMesh1().getView(), cs.getMesh2().getView() ),
                std::invalid_argument );
  EXPECT_FALSE( cs.hasMfemSubmeshData() );
}

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

/** @brief Host edges used to register an unsupported contact method. */
struct Edges {
  std::array<tribol::IndexT, 2> connectivity{ 0, 1 };  ///< Endpoint connectivity shared by both test segments.
  std::array<tribol::RealT, 2> x1{ 0.0, 2.0 };         ///< Source endpoint x coordinates.
  std::array<tribol::RealT, 2> y1{ 0.0, 0.0 };         ///< Source endpoint y coordinates.
  std::array<tribol::RealT, 2> x2{ 2.0, 0.0 };         ///< Oppositely ordered target endpoint x coordinates.
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
