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

constexpr tribol::IndexT coupling_scheme_id = 0;
constexpr tribol::IndexT first_mesh_id = 0;
constexpr tribol::IndexT second_mesh_id = 1;
constexpr int minimum_supported_order = 0;
constexpr int highest_order_with_one_point = tribol::EdgeQuadratureRuleGenerator<1>::polynomial_order;
constexpr int highest_order_with_two_points = tribol::EdgeQuadratureRuleGenerator<2>::polynomial_order;
constexpr int maximum_supported_order = tribol::EdgeQuadratureRuleGenerator<3>::polynomial_order;
constexpr int minimum_order_for_two_point_rule = highest_order_with_one_point + 1;

/** @brief Register an MFEM coupling scheme for each enforcement location and quadrature order. */
class EnergyMortarQuadratureSettingsTest : public testing::TestWithParam<std::tuple<tribol::EnforcementLocation, int>> {
 protected:
  /**
   * @brief Read the enforcement location from the test parameters.
   * @return Selected enforcement location.
   */
  tribol::EnforcementLocation enforcementLocation() const { return std::get<0>( GetParam() ); }
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
    constexpr int elements_per_direction = 1;
    constexpr int linear_basis_order = 1;
    constexpr int spatial_dimension = 2;
    constexpr tribol::RealT kinematic_penalty_stiffness = 100.0;
    // MFEM labels the unit square's bottom and top boundaries 1 and 3. This test only needs two valid surfaces for
    // coupling-scheme registration; it does not evaluate their contact geometry.
    constexpr int bottom_boundary_attribute = 1;
    constexpr int top_boundary_attribute = 3;
    mesh_ = std::make_unique<mfem::ParMesh>( static_cast<const mfem::ParMesh&>( shared::ParMeshBuilder(
        MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( elements_per_direction, elements_per_direction ) ) ) );
    fec_ = std::make_unique<mfem::H1_FECollection>( linear_basis_order, spatial_dimension );
    fes_ = std::make_unique<mfem::ParFiniteElementSpace>( mesh_.get(), fec_.get(), spatial_dimension );
    coords_ = std::make_unique<mfem::ParGridFunction>( fes_.get() );
    mesh_->GetNodes( *coords_ );
    tribol::registerMfemCouplingScheme(
        coupling_scheme_id, first_mesh_id, second_mesh_id, *mesh_, *coords_, { bottom_boundary_attribute },
        { top_boundary_attribute }, tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::ENERGY_MORTAR,
        tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
    EXPECT_EQ(
        tribol::CouplingSchemeManager::getInstance().findData( coupling_scheme_id )->getParameters().quadrature_order,
        maximum_supported_order );
    tribol::setEnforcementLocation( coupling_scheme_id, enforcementLocation() );
    tribol::setQuadratureOrder( coupling_scheme_id, quadratureOrder() );
    tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, kinematic_penalty_stiffness,
                                             kinematic_penalty_stiffness );
    ASSERT_NE( tribol::CouplingSchemeManager::getInstance().findData( coupling_scheme_id )->getContactFormulation(),
               nullptr );
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
 * @brief Check whether the factory created the Energy Mortar adapter for the expected location and point count.
 * @tparam PointCount Integration-surface points in the expected pair quadrature rule.
 * @param formulation Factory-created contact formulation to inspect.
 * @param location Expected enforcement location.
 * @return Whether the formulation has the expected adapter specialization.
 */
template <int PointCount>
bool isEnergyMortarAdapterWithPointCount( tribol::ContactFormulation* formulation,
                                          tribol::EnforcementLocation location )
{
  using Rule =
      tribol::PairQuadratureRule<tribol::SmoothedProjectedEdgeMapping, tribol::EdgeQuadratureRuleGenerator<PointCount>>;
  if ( location == tribol::EnforcementLocation::Nodal ) {
    return dynamic_cast<tribol::EnergyMortarAdapter<tribol::Nodal, Rule>*>( formulation ) != nullptr;
  }
  return dynamic_cast<tribol::EnergyMortarAdapter<tribol::QuadraturePoint, Rule>*>( formulation ) != nullptr;
}

// The default uses three points; changing order selects the requested specialization for either enforcement location.
TEST_P( EnergyMortarQuadratureSettingsTest, selects_and_rebuilds_adapter )
{
  auto* cs = tribol::CouplingSchemeManager::getInstance().findData( coupling_scheme_id );
  const auto location = enforcementLocation();
  const auto check = [&]( int order ) {
    tribol::setQuadratureOrder( coupling_scheme_id, order );
    auto* formulation = cs->getContactFormulation();
    ASSERT_NE( formulation, nullptr );
    if ( order <= highest_order_with_one_point )
      EXPECT_TRUE( isEnergyMortarAdapterWithPointCount<1>( formulation, location ) );
    else if ( order <= highest_order_with_two_points )
      EXPECT_TRUE( isEnergyMortarAdapterWithPointCount<2>( formulation, location ) );
    else
      EXPECT_TRUE( isEnergyMortarAdapterWithPointCount<3>( formulation, location ) );
  };
  check( quadratureOrder() );
  for ( int order = minimum_supported_order; order <= maximum_supported_order; ++order ) check( order );
  for ( int invalid_order : { minimum_supported_order - 1, maximum_supported_order + 1 } ) {
    auto* formulation_before_invalid_order = cs->getContactFormulation();
    EXPECT_THROW( tribol::setQuadratureOrder( coupling_scheme_id, invalid_order ), std::invalid_argument );
    EXPECT_EQ( cs->getParameters().quadrature_order, maximum_supported_order );
    EXPECT_EQ( cs->getContactFormulation(), formulation_before_invalid_order );
    cs->getParameters().quadrature_order = invalid_order;
    EXPECT_THROW( tribol::createContactFormulation( cs ), std::invalid_argument );
    cs->getParameters().quadrature_order = maximum_supported_order;
  }
  cs->setContactFormulation( nullptr );
  tribol::setQuadratureOrder( coupling_scheme_id, minimum_order_for_two_point_rule );
  EXPECT_FALSE( cs->hasContactFormulation() );
  cs->updateContactFormulation();
  EXPECT_TRUE( isEnergyMortarAdapterWithPointCount<2>( cs->getContactFormulation(), location ) );
}

// setQuadratureOrder must reject missing registration and contact methods other than Energy Mortar without changing
// the stored order.
TEST( ContactFormulation, rejects_order_setting_without_registered_energy_mortar )
{
  constexpr tribol::IndexT edge_element_count = 1;

  EXPECT_THROW( tribol::setQuadratureOrder( coupling_scheme_id, minimum_order_for_two_point_rule ),
                std::invalid_argument );

  // Register Common Plane with two parallel, oppositely oriented edges 0.1 apart. This test only checks contact-method
  // validation; it does not evaluate the edge geometry for contact.
  constexpr tribol::RealT edge_separation = 0.1;
  std::array<tribol::IndexT, 2> edge_connectivity{ 0, 1 };
  std::array<tribol::RealT, 2> first_edge_x{ 0.0, 2.0 };
  std::array<tribol::RealT, 2> first_edge_y{ 0.0, 0.0 };
  std::array<tribol::RealT, 2> second_edge_x{ 2.0, 0.0 };
  std::array<tribol::RealT, 2> second_edge_y{ edge_separation, edge_separation };

  tribol::registerMesh( first_mesh_id, edge_element_count, first_edge_x.size(), edge_connectivity.data(), tribol::EDGE,
                        tribol::LINEAR_LAGRANGE, first_edge_x.data(), first_edge_y.data() );
  tribol::registerMesh( second_mesh_id, edge_element_count, second_edge_x.size(), edge_connectivity.data(),
                        tribol::EDGE, tribol::LINEAR_LAGRANGE, second_edge_x.data(), second_edge_y.data() );
  tribol::registerCouplingScheme( coupling_scheme_id, first_mesh_id, second_mesh_id, tribol::SURFACE_TO_SURFACE,
                                  tribol::NO_CASE, tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                  tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
  // This order is valid for Energy Mortar, so the Common Plane contact method alone must cause rejection.
  EXPECT_THROW( tribol::setQuadratureOrder( coupling_scheme_id, minimum_order_for_two_point_rule ),
                std::invalid_argument );
  EXPECT_EQ(
      tribol::CouplingSchemeManager::getInstance().findData( coupling_scheme_id )->getParameters().quadrature_order,
      maximum_supported_order );
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
