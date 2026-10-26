// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/EnergyMortar.hpp"

namespace {

class UnsupportedFormulation : public tribol::ContactFormulation {
 public:
  void updateIntegrationRule( tribol::MeshData::Viewer, tribol::MeshData::Viewer ) override {}
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
  EXPECT_THROW( base.generateIntegrationRule( mesh.getView(), mesh.getView() ), std::logic_error );
}

// Each test supplies independent mesh views to an existing MFEM-backed formulation. No contact update is required.
class IntegrationRuleTest : public testing::TestWithParam<tribol::EnforcementLocation> {
 protected:
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
    tribol::setEnforcementLocation( 0, GetParam() );
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
  std::array<tribol::RealT, 2> x1{ 0, 2 };
  std::array<tribol::RealT, 2> y1{ 0, 0 };
  std::array<tribol::RealT, 2> x2{ 2, 0 };
  std::array<tribol::RealT, 2> y2{ 0.1, 0.1 };
  tribol::MeshData mesh1{
      17,        1,         2,       connectivity.data(),      tribol::EDGE, tribol::LINEAR_LAGRANGE,
      x1.data(), y1.data(), nullptr, tribol::MemorySpace::Host };
  tribol::MeshData mesh2{
      29,        1,         2,       connectivity.data(),      tribol::EDGE, tribol::LINEAR_LAGRANGE,
      x2.data(), y2.data(), nullptr, tribol::MemorySpace::Host };
};

TEST_P( IntegrationRuleTest, matches_element_integrals_and_physical_projection )
{
  // Del = 0.1 in the formulation factory. Exercise full, partial, and support-only overlap with unequal edge lengths.
  for ( const auto interval : { std::array<double, 2>{ 2, 0 }, { 1.5, 0.5 }, { 2.15, 2.05 }, { 1.8, 0.2 } } ) {
    Edges edges;
    edges.x2 = interval;
    if ( interval[0] == 1.8 ) edges.y2 = { 0.3, 0.1 };
    auto rule = formulation_->generateIntegrationRule( edges.mesh1.getView(), edges.mesh2.getView() );
    EXPECT_EQ( rule.getParentDimension(), 1 );
    EXPECT_EQ( rule.getAllocatorId(), edges.mesh1.getAllocatorId() );
    auto view = rule.getView<1>();
    ASSERT_EQ( view.numberOfPairs(), 1 );
    ASSERT_EQ( view.numberOfQuadraturePoints( 0 ), 3 );
    EXPECT_EQ( view.getInterfacePair( 0 ).element_id1_, 0 );
    EXPECT_EQ( view.getInterfacePair( 0 ).element_id2_, 0 );
    tribol::ContactParams params{ 0.1, 1000, 3, true, 0 };
    tribol::EnergyMortarCalculator calculator( params );
    double expected_gaps[2], expected_area[2];
    calculator.compute_gtilde_and_area( { 0, 0 }, edges.mesh1.getView(), edges.mesh2.getView(), expected_gaps,
                                        expected_area );
    double area[2] = {}, gaps[2] = {};
    for ( int q = 0; q < 3; ++q ) {
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

TEST_P( IntegrationRuleTest, analytic_smoothed_full_overlap )
{
  Edges edges;
  const auto rule = formulation_->generateIntegrationRule( edges.mesh1.getView(), edges.mesh2.getView() );
  const auto view = rule.getView<1>();
  ASSERT_EQ( view.numberOfPairs(), 1 );
  double weight = 0, moment = 0;
  // Smoothing maps the full parent interval to [-0.95, 0.95]. Edge length is 2, so J = 1.
  const double gauss[] = { -std::sqrt( 3.0 / 5.0 ), 0, std::sqrt( 3.0 / 5.0 ) };
  for ( int q = 0; q < 3; ++q ) {
    auto point = view.getQuadraturePoint( 0, q );
    EXPECT_NEAR( point.getMesh1ParentCoordinates()[0], 0.95 * gauss[q], 1.e-12 );
    EXPECT_NEAR( point.getMesh2ParentCoordinates()[0], -0.95 * gauss[q], 1.e-12 );
    weight += point.getWeight();
    moment += point.getWeight() * ( 1 + point.getMesh1ParentCoordinates()[0] );
  }
  EXPECT_NEAR( weight, 1.9, 1.e-12 );
  EXPECT_NEAR( moment, 1.9, 1.e-12 );
}

TEST_P( IntegrationRuleTest, fresh_search_and_independent_owned_results )
{
  Edges edges;
  tribol::InterfaceQuadratureData retained;
  retained = formulation_->generateIntegrationRule( edges.mesh1.getView(), edges.mesh2.getView() );
  edges.x2 = { 11, 10 };
  auto empty = formulation_->generateIntegrationRule( edges.mesh1.getView(), edges.mesh2.getView() );
  EXPECT_EQ( empty.getView().numberOfPairs(), 0 );
  EXPECT_EQ( empty.getPairOffsets()[0], 0 );
  edges.x2 = { 1.5, 0.5 };
  auto changed = formulation_->generateIntegrationRule( edges.mesh1.getView(), edges.mesh2.getView() );
  EXPECT_EQ( changed.getView().numberOfPairs(), 1 );
  EXPECT_NE( changed.getView().getQuadraturePoint( 0, 0 ).getWeight(),
             retained.getView().getQuadraturePoint( 0, 0 ).getWeight() );
  tribol::finalize();
  formulation_ = nullptr;
  EXPECT_EQ( retained.getView().numberOfQuadraturePoints(), 3 );
}

TEST_P( IntegrationRuleTest, empty_and_rejected_pairs )
{
  Edges edges;
  edges.x2 = { 0, 2 };  // Co-oriented normals.
  EXPECT_EQ(
      formulation_->generateIntegrationRule( edges.mesh1.getView(), edges.mesh2.getView() ).getInterfacePairs().size(),
      0 );
  edges.x2 = { 1, 1 };  // Degenerate projected edge.
  EXPECT_EQ(
      formulation_->generateIntegrationRule( edges.mesh1.getView(), edges.mesh2.getView() ).getInterfacePairs().size(),
      0 );
  tribol::MeshData empty( 41, 0, 0, nullptr, tribol::EDGE, tribol::LINEAR_LAGRANGE, edges.x1.data(), edges.y1.data(),
                          nullptr, tribol::MemorySpace::Host );
  auto rule = formulation_->generateIntegrationRule( empty.getView(), edges.mesh1.getView() );
  EXPECT_EQ( rule.getView().numberOfPairs(), 0 );
  EXPECT_EQ( rule.getParentDimension(), 1 );
}

TEST_P( IntegrationRuleTest, generation_does_not_seed_or_modify_force_state )
{
  Edges edges;
  auto first = formulation_->generateIntegrationRule( edges.mesh1.getView(), edges.mesh2.getView() );
  ASSERT_EQ( first.getView().numberOfPairs(), 1 );
  // Evaluate the real MFEM surfaces after generation on unrelated mesh views.
  tribol::updateMfemParallelDecomposition();
  double dt = 1;
  tribol::update( 1, 0, dt );
  const double energy = formulation_->getEnergy();
  const auto& force = formulation_->getMfemForce();
  std::vector<double> saved_force( force.HostRead(), force.HostRead() + force.Size() );
  auto second = formulation_->generateIntegrationRule( edges.mesh1.getView(), edges.mesh2.getView() );
  EXPECT_EQ( second.getView().numberOfPairs(), 1 );
  EXPECT_DOUBLE_EQ( formulation_->getEnergy(), energy );
  const auto& after = formulation_->getMfemForce();
  ASSERT_EQ( after.Size(), static_cast<int>( saved_force.size() ) );
  for ( int i = 0; i < after.Size(); ++i ) {
    EXPECT_DOUBLE_EQ( after.HostRead()[i], saved_force[i] );
  }
}

TEST_P( IntegrationRuleTest, rejects_unsupported_geometry )
{
  Edges edges;
  const tribol::IndexT conn[] = { 0, 1, 2 };
  const tribol::RealT x[] = { 0, 1, 0.5 }, y[] = { 0, 0, 0.2 };
  tribol::MeshData quadratic( 42, 1, 3, conn, tribol::EDGE, tribol::QUADRATIC_LAGRANGE, x, y, nullptr,
                              tribol::MemorySpace::Host );
  EXPECT_THROW( formulation_->generateIntegrationRule( quadratic.getView(), edges.mesh2.getView() ),
                std::invalid_argument );
}

INSTANTIATE_TEST_SUITE_P( enforcement, IntegrationRuleTest,
                          testing::Values( tribol::EnforcementLocation::Nodal,
                                           tribol::EnforcementLocation::QuadraturePoint ) );

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
