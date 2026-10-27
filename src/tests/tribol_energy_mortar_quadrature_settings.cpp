// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Tests factory selection and changes to Energy Mortar quadrature order for both enforcement locations.
 * Local quadrature accuracy is checked in tribol_pair_quadrature; gap and force derivatives have separate tests.
 */
#include <gtest/gtest.h>
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/ContactFormulationFactory.hpp"
#include "tribol/physics/EnergyMortarAdapter.hpp"

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
        shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( 1, 1 ) ) ) );
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
  using Rule = tribol::EdgeQuadratureRule<Count, tribol::SmoothedEdgeProjection>;
  if ( location == tribol::EnforcementLocation::Nodal ) {
    return dynamic_cast<tribol::EnergyMortarAdapter<tribol::Nodal, Rule>*>( formulation ) != nullptr;
  }
  return dynamic_cast<tribol::EnergyMortarAdapter<tribol::QuadraturePoint, Rule>*>( formulation ) != nullptr;
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

/** @brief Host edges used to register an unsupported contact method. */
struct Edges {
  std::array<tribol::IndexT, 2> connectivity{ 0, 1 };  ///< Endpoint connectivity shared by both test segments.
  std::array<tribol::RealT, 2> x1{ 0.0, 2.0 };         ///< Source endpoint x coordinates.
  std::array<tribol::RealT, 2> y1{ 0.0, 0.0 };         ///< Source endpoint y coordinates.
  std::array<tribol::RealT, 2> x2{ 2.0, 0.0 };         ///< Oppositely ordered target endpoint x coordinates.
  std::array<tribol::RealT, 2> y2{ 0.1, 0.1 };         ///< Target endpoint y coordinates.
  /// @brief Source mesh borrowing the arrays in this object.
  tribol::MeshData mesh1{
      0,         1,         2,       connectivity.data(),      tribol::EDGE, tribol::LINEAR_LAGRANGE,
      x1.data(), y1.data(), nullptr, tribol::MemorySpace::Host };
  /// @brief Target mesh borrowing the arrays in this object.
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
