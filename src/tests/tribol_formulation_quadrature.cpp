// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <array>
#include <cmath>
#include <memory>
#include <vector>
#include <tuple>
#include "tribol/physics/ContactFormulationFactory.hpp"

#include <gtest/gtest.h>
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/EnergyMortar.hpp"

// Checks factory-selected quadrature orders, physical integrals, and independent rule generation through
// ContactFormulation. Per-pair numerical formulas, bounded writes, and capacity errors are covered in
// tribol_pair_quadrature.
namespace {

class UnsupportedFormulation : public tribol::ContactFormulation {
 public:
  void updateQuadrature( tribol::MeshData::Viewer, tribol::MeshData::Viewer ) override {}
  void updateNodalGaps( tribol::MeshData::Viewer, tribol::MeshData::Viewer ) override {}
  void updateNodalForces( tribol::MeshData::Viewer, tribol::MeshData::Viewer ) override {}
  bool hasTimeStepCalculation() override { return false; }
  tribol::RealT getEnergy() const override { return 0.0; }
};

// A formulation that does not implement rule generation must report that the operation is unsupported.
TEST( ContactFormulation, unsupported_rule_generation_throws )
{
  const tribol::RealT x[] = { 0.0, 1.0 }, y[] = { 0.0, 0.0 };
  const tribol::IndexT conn[] = { 0, 1 };
  tribol::MeshData mesh( 0, 1, 2, conn, tribol::EDGE, tribol::LINEAR_LAGRANGE, x, y, nullptr,
                         tribol::MemorySpace::Host );
  UnsupportedFormulation method;
  const tribol::ContactFormulation& base = method;
  EXPECT_THROW( base.generateQuadrature( mesh.getView(), mesh.getView() ), std::logic_error );
}

// Each test supplies independent mesh views to an existing MFEM-backed formulation. No contact update is required.
class FormulationQuadratureTest : public testing::TestWithParam<std::tuple<tribol::EnforcementLocation, int>> {
 protected:
  int quadratureOrder() const { return std::get<1>( GetParam() ); }
  int pointCount() const { return ( quadratureOrder() + 2 ) / 2; }

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
    formulation_ = tribol::CouplingSchemeManager::getInstance().findData( 0 )->getContactFormulation();
    ASSERT_NE( formulation_, nullptr );
  }
  void TearDown() override { tribol::finalize(); }

  std::unique_ptr<mfem::ParMesh> mesh_;
  std::unique_ptr<mfem::H1_FECollection> fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> fes_;
  std::unique_ptr<mfem::ParGridFunction> coords_;
  tribol::ContactFormulation* formulation_;
};

struct Edges {
  std::array<tribol::IndexT, 2> connectivity{ 0, 1 };
  std::array<tribol::RealT, 2> x1{ 0.0, 2.0 };
  std::array<tribol::RealT, 2> y1{ 0.0, 0.0 };
  std::array<tribol::RealT, 2> x2{ 2.0, 0.0 };
  std::array<tribol::RealT, 2> y2{ 0.1, 0.1 };
  tribol::MeshData mesh1{
      0,         1,         2,       connectivity.data(),      tribol::EDGE, tribol::LINEAR_LAGRANGE,
      x1.data(), y1.data(), nullptr, tribol::MemorySpace::Host };
  tribol::MeshData mesh2{
      1,         1,         2,       connectivity.data(),      tribol::EDGE, tribol::LINEAR_LAGRANGE,
      x2.data(), y2.data(), nullptr, tribol::MemorySpace::Host };
};

// Each selected quadrature order must reproduce the calculator integrals and map points onto the mortar edge.
TEST_P( FormulationQuadratureTest, matches_element_integrals_and_physical_projection )
{
  // Del = 0.1 in the formulation factory. Exercise full, partial, and support-only overlap with unequal edge lengths.
  for ( const auto interval : { std::array<double, 2>{ 2.0, 0.0 }, { 1.5, 0.5 }, { 2.15, 2.05 }, { 1.8, 0.2 } } ) {
    Edges edges;
    edges.x2 = interval;
    if ( interval[0] == 1.8 ) edges.y2 = { 0.3, 0.1 };
    auto rule = formulation_->generateQuadrature( edges.mesh1.getView(), edges.mesh2.getView() );
    EXPECT_EQ( rule.getParentDimension(), 1 );
    EXPECT_EQ( rule.getAllocatorId(), edges.mesh1.getAllocatorId() );
    auto view = rule.getView<1>();
    ASSERT_EQ( view.numberOfPairs(), 1 );
    ASSERT_EQ( view.numberOfQuadraturePoints( 0 ), pointCount() );
    EXPECT_EQ( view.getInterfacePair( 0 ).element_id1_, 0 );
    EXPECT_EQ( view.getInterfacePair( 0 ).element_id2_, 0 );
    const tribol::ContactParams params{ 1000.0, true, 0.0 };
    double expected_gaps[2], expected_area[2];
    const auto calculate = [&]<int Count>() {
      const tribol::EdgeQuadratureRule<Count> quadrature_rule{};
      tribol::EnergyMortarCalculator<tribol::EdgeQuadratureRule<Count>> calculator( params, quadrature_rule );
      calculator.compute_gtilde_and_area( { 0, 0 }, edges.mesh1.getView(), edges.mesh2.getView(), expected_gaps,
                                          expected_area );
    };
    switch ( pointCount() ) {
      case 1:
        calculate.template operator()<1>();
        break;
      case 2:
        calculate.template operator()<2>();
        break;
      case 3:
        calculate.template operator()<3>();
        break;
    }
    double area[2] = {}, gaps[2] = {};
    for ( int q = 0; q < pointCount(); ++q ) {
      auto point = view.getQuadraturePoint( 0, q );
      const double xi = point.getMesh1ParentCoordinates()[0];
      const double eta = point.getMesh2ParentCoordinates()[0];
      const double xA = ( 1 - xi ) * edges.x1[0] / 2 + ( 1 + xi ) * edges.x1[1] / 2;
      const double xB = ( 1 - eta ) * edges.x2[0] / 2 + ( 1 + eta ) * edges.x2[1] / 2;
      const double yB = ( 1 - eta ) * edges.y2[0] / 2 + ( 1 + eta ) * edges.y2[1] / 2;
      const double bx = edges.x2[1] - edges.x2[0], by = edges.y2[1] - edges.y2[0];
      const double length = std::sqrt( bx * bx + by * by );
      const double nx = by / length, ny = -bx / length;
      EXPECT_NEAR( ( xA - xB ) * bx - yB * by, 0.0, 1.e-12 );
      EXPECT_GT( point.getWeight(), 0 );
      const double gap = -( ( xA - xB ) * nx - yB * ny ) * ( -ny );
      area[0] += ( 1 - xi ) / 2 * point.getWeight();
      area[1] += ( 1 + xi ) / 2 * point.getWeight();
      gaps[0] += ( 1 - xi ) / 2 * gap * point.getWeight();
      gaps[1] += ( 1 + xi ) / 2 * gap * point.getWeight();
    }
    for ( int node = 0; node < 2; ++node ) {
      EXPECT_NEAR( area[node], expected_area[node], 1.e-12 );
      EXPECT_NEAR( gaps[node], expected_gaps[node], 1.e-12 );
    }
  }
}

// A length-two edge has a unit Jacobian; smoothing shortens its full integration interval to [-0.95, 0.95].
TEST_P( FormulationQuadratureTest, analytic_smoothed_full_overlap )
{
  Edges edges;
  const auto rule = formulation_->generateQuadrature( edges.mesh1.getView(), edges.mesh2.getView() );
  const auto view = rule.getView<1>();
  ASSERT_EQ( view.numberOfPairs(), 1 );
  double weight = 0.0, moment = 0.0;
  // Smoothing maps the full parent interval to [-0.95, 0.95]. Edge length is 2, so J = 1.
  std::vector<double> gauss;
  if ( pointCount() == 1 ) gauss = { 0.0 };
  if ( pointCount() == 2 ) gauss = { -1.0 / std::sqrt( 3.0 ), 1.0 / std::sqrt( 3.0 ) };
  if ( pointCount() == 3 ) gauss = { -std::sqrt( 3.0 / 5.0 ), 0.0, std::sqrt( 3.0 / 5.0 ) };
  for ( int q = 0; q < pointCount(); ++q ) {
    auto point = view.getQuadraturePoint( 0, q );
    EXPECT_NEAR( point.getMesh1ParentCoordinates()[0], 0.95 * gauss[q], 1.e-12 );
    EXPECT_NEAR( point.getMesh2ParentCoordinates()[0], -0.95 * gauss[q], 1.e-12 );
    weight += point.getWeight();
    moment += point.getWeight() * ( 1 + point.getMesh1ParentCoordinates()[0] );
  }
  EXPECT_NEAR( weight, 1.9, 1.e-12 );
  EXPECT_NEAR( moment, 1.9, 1.e-12 );
}

// Changed geometry must change new results without changing or invalidating a previously returned rule.
TEST_P( FormulationQuadratureTest, fresh_search_and_independent_owned_results )
{
  Edges edges;
  tribol::InterfaceQuadratureData retained;
  retained = formulation_->generateQuadrature( edges.mesh1.getView(), edges.mesh2.getView() );
  edges.x2 = { 4.0, 3.0 };
  auto empty = formulation_->generateQuadrature( edges.mesh1.getView(), edges.mesh2.getView() );
  EXPECT_EQ( empty.getView().numberOfPairs(), 0 );
  EXPECT_EQ( empty.getPairOffsets()[0], 0 );
  edges.x2 = { 1.5, 0.5 };
  auto changed = formulation_->generateQuadrature( edges.mesh1.getView(), edges.mesh2.getView() );
  EXPECT_EQ( changed.getView().numberOfPairs(), 1 );
  EXPECT_NE( changed.getView().getQuadraturePoint( 0, 0 ).getWeight(),
             retained.getView().getQuadraturePoint( 0, 0 ).getWeight() );
  tribol::finalize();
  formulation_ = nullptr;
  EXPECT_EQ( retained.getView().numberOfQuadraturePoints(), pointCount() );
}

// Empty meshes and pairs rejected by refined search must produce an empty quadrature rule.
TEST_P( FormulationQuadratureTest, empty_and_rejected_pairs )
{
  Edges edges;
  edges.x2 = { 0.0, 2.0 };  // Co-oriented normals.
  EXPECT_EQ(
      formulation_->generateQuadrature( edges.mesh1.getView(), edges.mesh2.getView() ).getInterfacePairs().size(), 0 );
  edges.x2 = { 1.0, 1.0 };  // Degenerate projected edge.
  EXPECT_EQ(
      formulation_->generateQuadrature( edges.mesh1.getView(), edges.mesh2.getView() ).getInterfacePairs().size(), 0 );
  tribol::MeshData empty( 2, 0, 0, nullptr, tribol::EDGE, tribol::LINEAR_LAGRANGE, edges.x1.data(), edges.y1.data(),
                          nullptr, tribol::MemorySpace::Host );
  auto rule = formulation_->generateQuadrature( empty.getView(), edges.mesh1.getView() );
  EXPECT_EQ( rule.getView().numberOfPairs(), 0 );
  EXPECT_EQ( rule.getParentDimension(), 1 );
}

// Generating rules from unrelated mesh views must leave the formulation force and energy state unchanged.
TEST_P( FormulationQuadratureTest, generation_does_not_seed_or_modify_force_state )
{
  Edges edges;
  auto first = formulation_->generateQuadrature( edges.mesh1.getView(), edges.mesh2.getView() );
  ASSERT_EQ( first.getView().numberOfPairs(), 1 );
  // Evaluate the real MFEM surfaces after generation on unrelated mesh views.
  tribol::updateMfemParallelDecomposition();
  double dt = 1;
  tribol::update( 1, 0, dt );
  const double energy = formulation_->getEnergy();
  const auto& force = formulation_->getMfemForce();
  std::vector<double> saved_force( force.HostRead(), force.HostRead() + force.Size() );
  auto second = formulation_->generateQuadrature( edges.mesh1.getView(), edges.mesh2.getView() );
  EXPECT_EQ( second.getView().numberOfPairs(), 1 );
  EXPECT_DOUBLE_EQ( formulation_->getEnergy(), energy );
  const auto& after = formulation_->getMfemForce();
  ASSERT_EQ( after.Size(), static_cast<int>( saved_force.size() ) );
  for ( int i = 0; i < after.Size(); ++i ) {
    EXPECT_DOUBLE_EQ( after.HostRead()[i], saved_force[i] );
  }
}

// Linear-edge rule generation must report unsupported quadratic elements instead of treating their nodes as endpoints.
TEST_P( FormulationQuadratureTest, rejects_unsupported_geometry )
{
  Edges edges;
  const tribol::IndexT conn[] = { 0, 1, 2 };
  const tribol::RealT x[] = { 0.0, 1.0, 0.5 }, y[] = { 0.0, 0.0, 0.2 };
  tribol::MeshData quadratic( 2, 1, 3, conn, tribol::EDGE, tribol::QUADRATIC_LAGRANGE, x, y, nullptr,
                              tribol::MemorySpace::Host );
  EXPECT_THROW( formulation_->generateQuadrature( quadratic.getView(), edges.mesh2.getView() ), std::invalid_argument );
}

// Changing the setting rebuilds the formulation, while invalid values leave its configuration unchanged.
TEST_P( FormulationQuadratureTest, updates_quadrature_order_and_rejects_invalid_settings )
{
  Edges edges;
  auto* cs = tribol::CouplingSchemeManager::getInstance().findData( 0 );
  // Even and odd requested degrees select the smallest available rule that meets the requested accuracy.
  const std::pair<int, int> order_and_point_count[] = { { 0, 1 }, { 1, 1 }, { 2, 2 }, { 3, 2 }, { 4, 3 }, { 5, 3 } };
  for ( const auto [order, expected_points] : order_and_point_count ) {
    tribol::setQuadratureOrder( 0, order );
    const auto rule = cs->getContactFormulation()->generateQuadrature( edges.mesh1.getView(), edges.mesh2.getView() );
    EXPECT_EQ( rule.getView().numberOfQuadraturePoints( 0 ), expected_points ) << "requested degree=" << order;
  }
  for ( const int invalid : { -1, 6 } ) {
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
  formulation_ = cs->getContactFormulation();
  const auto rebuilt = formulation_->generateQuadrature( edges.mesh1.getView(), edges.mesh2.getView() );
  EXPECT_EQ( rebuilt.getView().numberOfQuadraturePoints( 0 ), 2 );
}

// A missing scheme or a method that does not use this rule must reject the setting without changing its parameters.
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

INSTANTIATE_TEST_SUITE_P( enforcement, FormulationQuadratureTest,
                          testing::Combine( testing::Values( tribol::EnforcementLocation::Nodal,
                                                             tribol::EnforcementLocation::QuadraturePoint ),
                                            testing::Values( 1, 3, 5 ) ) );

}  // namespace

int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
