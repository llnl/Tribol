// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check host contact-domain traversal independently of quadrature, Enzyme, and MFEM assembly.
 * Tests verify supplied pair order, packed coordinates, repeated traversal, unequal element node counts, empty domains,
 * and layout validation. Disjoint pairs remain in the domain because quadrature and the integrand determine contact.
 */
#include <array>
#include <gtest/gtest.h>
#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/geom/LinearEdgeGeometry.hpp"
#include "tribol/integ/ContactDomain.hpp"

namespace {
using namespace tribol;
using EdgeDomain = ContactDomain<LinearEdgeGeometry, LinearEdgeGeometry>;

/** @brief Three-node coordinate layout used only to check gathering, without claiming quadratic contact support. */
struct ThreeNodeGeometry {
  static constexpr int spatial_dimension = 2;                ///< Physical coordinate components.
  using Coords = MdSpan<const RealT, 3, spatial_dimension>;  ///< Three coordinate nodes.
};

// Distinct edge coordinates expose element/node/component ordering; disjoint and repeated pairs must be visited too.
TEST( ContactDomain, visits_supplied_pairs_in_order_with_packed_coordinates )
{
  const IndexT connectivity[]{ 0, 1, 1, 2 };
  const RealT x1[]{ 0.0, 1.0, 2.0 }, y1[]{ 0.0, 0.1, 0.2 };
  const RealT x2[]{ 10.0, 11.0, 12.0 }, y2[]{ 1.0, 1.1, 1.2 };
  MeshData mesh1( 0, 2, 3, connectivity, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 2, 3, connectivity, EDGE, LINEAR_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
  const ArrayT<InterfacePair> pairs{ { 1, 0 }, { 0, 1 }, { 1, 0 } };
  const std::array<EdgeDomain::Coordinates, 3> expected{
      EdgeDomain::Coordinates{ 1.0, 0.1, 2.0, 0.2, 10.0, 1.0, 11.0, 1.1 },
      EdgeDomain::Coordinates{ 0.0, 0.0, 1.0, 0.1, 11.0, 1.1, 12.0, 1.2 },
      EdgeDomain::Coordinates{ 1.0, 0.1, 2.0, 0.2, 10.0, 1.0, 11.0, 1.1 } };
  const EdgeDomain domain{ mesh1.getView(), mesh2.getView(), pairs.view() };
  EXPECT_EQ( domain.size(), pairs.size() );
  // A second operation can traverse the same domain without searching or storing geometry from the first operation.
  for ( int traversal = 0; traversal < 2; ++traversal ) {
    IndexT visited = 0;
    domain.forEachPair( [&]( InterfacePair pair, EdgeDomain::CoordinateView coordinates ) {
      ASSERT_LT( visited, pairs.size() );
      EXPECT_EQ( pair.element_id1_, pairs[visited].element_id1_ );
      EXPECT_EQ( pair.element_id2_, pairs[visited].element_id2_ );
      for ( IndexT dof = 0; dof < coordinates.size(); ++dof ) EXPECT_EQ( coordinates[dof], expected[visited][dof] );
      ++visited;
    } );
    EXPECT_EQ( visited, pairs.size() );
  }
}

// The second element starts after the first element's coordinate count, even when the two layouts differ.
TEST( ContactDomain, gathers_different_element_coordinate_counts )
{
  const IndexT conn1[]{ 0, 1 }, conn2[]{ 0, 1, 2 };
  const RealT x1[]{ 0.0, 1.0 }, y1[]{ 0.0, 0.0 };
  const RealT x2[]{ 0.0, 0.5, 1.0 }, y2[]{ 0.1, 0.2, 0.1 };
  MeshData mesh1( 0, 1, 2, conn1, EDGE, LINEAR_LAGRANGE, x1, y1, nullptr, MemorySpace::Host );
  MeshData mesh2( 1, 1, 3, conn2, EDGE, QUADRATIC_LAGRANGE, x2, y2, nullptr, MemorySpace::Host );
  using Domain = ContactDomain<LinearEdgeGeometry, ThreeNodeGeometry>;
  const ArrayT<InterfacePair> pairs{ { 0, 0 } };
  const Domain domain{ mesh1.getView(), mesh2.getView(), pairs.view() };
  const Domain::Coordinates expected{ 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.5, 0.2, 1.0, 0.1 };
  IndexT visited = 0;
  domain.forEachPair( [&]( InterfacePair, Domain::CoordinateView coordinates ) {
    for ( IndexT dof = 0; dof < coordinates.size(); ++dof ) EXPECT_EQ( coordinates[dof], expected[dof] );
    ++visited;
  } );
  EXPECT_EQ( visited, domain.size() );
  EXPECT_THROW( ( EdgeDomain{ mesh1.getView(), mesh2.getView(), pairs.view() } ), std::invalid_argument );
}

// An empty domain must not gather coordinates or invoke the visitor, including on ranks without local contact.
TEST( ContactDomain, empty_domain_does_not_call_visitor )
{
  const IndexT connectivity[]{ 0, 1 };
  const RealT x[]{ 0.0, 1.0 }, y[]{ 0.0, 0.0 };
  MeshData mesh( 0, 1, 2, connectivity, EDGE, LINEAR_LAGRANGE, x, y, nullptr, MemorySpace::Host );
  const EdgeDomain domain{ mesh.getView(), mesh.getView(), {} };
  EXPECT_EQ( domain.size(), 0 );
  domain.forEachPair( []( InterfacePair, EdgeDomain::CoordinateView ) { FAIL() << "Empty domain visited a pair."; } );
}
}  // namespace

/**
 * @brief Run host traversal tests with logging initialized.
 * @param argc Argument count.
 * @param argv Arguments.
 * @return Test status.
 */
int main( int argc, char** argv )
{
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
