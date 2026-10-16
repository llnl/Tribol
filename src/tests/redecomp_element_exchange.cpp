// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/config.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "mfem.hpp"

#include "redecomp/ElementExchangeOperator.hpp"
#include "redecomp/ContactPatchOperator.hpp"
#include "redecomp/DistributedBoundaryRestriction.hpp"
#include "redecomp/DistributedElementRestriction.hpp"
#include "redecomp/ElementFieldExchange.hpp"
#include "redecomp/PatchNodalLayoutOperator.hpp"
#include "redecomp/VirtualRefinementOperator.hpp"

namespace redecomp {
namespace {

std::shared_ptr<ElementExchangePlan> BuildRingExchangePlan()
{
  int rank = 0;
  int num_ranks = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_ranks );

  // Variable block sizes ensure the plan uses element offsets rather than
  // assuming every finite element contributes the same number of values.
  mfem::Array<int> source_offsets{ 0, rank + 1, rank + 3 };
  std::vector<mfem::Array<int>> elements_by_destination( static_cast<size_t>( num_ranks ) );
  elements_by_destination[static_cast<size_t>( ( rank + 1 ) % num_ranks )].Append( 0 );
  elements_by_destination[static_cast<size_t>( ( rank + num_ranks - 1 ) % num_ranks )].Append( 1 );
  return std::make_shared<ElementExchangePlan>( MPI_COMM_WORLD, std::move( source_offsets ),
                                                std::move( elements_by_destination ) );
}

// A ring exchange with rank-dependent block sizes detects fixed-size packing
// assumptions and verifies returned source provenance.
TEST( ElementExchangeOperator, gathers_requested_variable_size_blocks )
{
  int rank = 0;
  int num_ranks = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_ranks );

  auto plan = BuildRingExchangePlan();
  ElementExchangeOperator exchange( plan );
  mfem::Vector source( exchange.Width() );
  for ( int i = 0; i < source.Size(); ++i ) {
    source[i] = 100.0 * rank + i;
  }
  mfem::Vector destination( exchange.Height() );
  exchange.Mult( source, destination );

  const int previous_rank = ( rank + num_ranks - 1 ) % num_ranks;
  const int next_rank = ( rank + 1 ) % num_ranks;
  const auto& source_ranks = plan->DestinationSourceRanks();
  const auto& source_elements = plan->DestinationSourceElements();
  const auto& destination_offsets = plan->DestinationElementOffsets();
  for ( int block = 0; block < plan->NumDestinationElements(); ++block ) {
    const int source_rank = source_ranks[block];
    const int source_element = source_elements[block];
    EXPECT_TRUE( source_rank == previous_rank || source_rank == next_rank );
    const int source_begin = source_element == 0 ? 0 : source_rank + 1;
    for ( int i = destination_offsets[block]; i < destination_offsets[block + 1]; ++i ) {
      EXPECT_DOUBLE_EQ( destination[i], 100.0 * source_rank + source_begin + i - destination_offsets[block] );
    }
  }
}

// Independent destination contributions must satisfy the global communication
// adjoint identity when they are summed back to source blocks.
TEST( ElementExchangeOperator, transpose_is_the_additive_adjoint )
{
  auto plan = BuildRingExchangePlan();
  ElementExchangeOperator exchange( plan );
  mfem::Vector source( exchange.Width() );
  mfem::Vector destination_dual( exchange.Height() );
  for ( int i = 0; i < source.Size(); ++i ) {
    source[i] = 0.25 + i;
  }
  for ( int i = 0; i < destination_dual.Size(); ++i ) {
    destination_dual[i] = 1.5 - 0.125 * i;
  }

  mfem::Vector destination( exchange.Height() );
  mfem::Vector source_dual( exchange.Width() );
  exchange.Mult( source, destination );
  exchange.MultTranspose( destination_dual, source_dual );

  double local_forward_pairing = destination * destination_dual;
  double local_transpose_pairing = source * source_dual;
  double global_forward_pairing = 0.0;
  double global_transpose_pairing = 0.0;
  MPI_Allreduce( &local_forward_pairing, &global_forward_pairing, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_transpose_pairing, &global_transpose_pairing, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( global_forward_pairing, global_transpose_pairing,
               1.0e-13 * std::max( 1.0, std::abs( global_forward_pairing ) ) );
}

// Lazy explicit assembly must reproduce the matrix-free exchange action.
TEST( ElementExchangeOperator, sparse_matrix_matches_matrix_free_action )
{
  auto plan = BuildRingExchangePlan();
  ElementExchangeOperator exchange( plan );
  mfem::Vector source( exchange.Width() );
  for ( int i = 0; i < source.Size(); ++i ) {
    source[i] = 2.0 + i;
  }

  mfem::Vector expected( exchange.Height() );
  mfem::Vector assembled_result( exchange.Height() );
  exchange.Mult( source, expected );
  auto matrix = exchange.Assemble();
  matrix.get().Mult( source, assembled_result );
  assembled_result -= expected;
  EXPECT_LT( assembled_result.Norml2(), 1.0e-13 );
}

// Only rank zero owns source data and only the last rank receives it, covering
// empty source and destination partitions in one collective schedule.
TEST( ElementExchangeOperator, supports_empty_source_and_destination_ranks )
{
  int rank = 0;
  int num_ranks = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_ranks );

  const bool owns_element = rank == 0;
  mfem::Array<int> source_offsets = owns_element ? mfem::Array<int>{ 0, 2 } : mfem::Array<int>{ 0 };
  std::vector<mfem::Array<int>> elements_by_destination( static_cast<size_t>( num_ranks ) );
  if ( owns_element ) {
    elements_by_destination[static_cast<size_t>( num_ranks - 1 )].Append( 0 );
  }

  auto plan = std::make_shared<ElementExchangePlan>( MPI_COMM_WORLD, std::move( source_offsets ),
                                                     std::move( elements_by_destination ) );
  ElementExchangeOperator exchange( plan );
  mfem::Vector source( exchange.Width() );
  source = 3.0;
  mfem::Vector destination( exchange.Height() );
  exchange.Mult( source, destination );
  if ( rank == num_ranks - 1 ) {
    ASSERT_EQ( destination.Size(), 2 );
    EXPECT_DOUBLE_EQ( destination[0], 3.0 );
    EXPECT_DOUBLE_EQ( destination[1], 3.0 );
  } else {
    EXPECT_EQ( destination.Size(), 0 );
  }
}

// Variable quadrature blocks retain provenance and satisfy the global primal/
// dual work identity without being interpreted as nodal values.
TEST( ElementFieldExchange, preserves_quadrature_blocks_and_additive_work )
{
  int rank = 0;
  int num_ranks = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_ranks );
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D( 3, 2, mfem::Element::QUADRILATERAL );
  mfem::ParMesh parallel_mesh( MPI_COMM_WORLD, serial_mesh );
  mfem::QuadratureSpace quadrature_space( &parallel_mesh, 3 );
  mfem::QuadratureFunction source( quadrature_space, 2 );
  for ( int value = 0; value < source.Size(); ++value ) {
    source[value] = 1.0 + 0.25 * rank + 0.015625 * value;
  }

  std::vector<mfem::Array<int>> elements_by_destination( static_cast<size_t>( num_ranks ) );
  auto& requested_elements = elements_by_destination[static_cast<size_t>( ( rank + 1 ) % num_ranks )];
  for ( int element = 0; element < parallel_mesh.GetNE(); ++element ) {
    requested_elements.Append( element );
  }
  if ( parallel_mesh.GetNE() > 0 ) {
    // Two selected boundary faces can share one parent volume element. Field
    // exchange must preserve both face occurrences and add both returned duals.
    requested_elements.Append( 0 );
  }
  ElementFieldExchange exchange( source, std::move( elements_by_destination ) );
  auto destination = exchange.Gather( source );
  EXPECT_EQ( destination.element_offsets.Last(), destination.values.Size() );
  EXPECT_EQ( destination.source_ranks.Size(), destination.element_offsets.Size() - 1 );
  EXPECT_EQ( destination.source_elements.Size(), destination.element_offsets.Size() - 1 );

  ExchangedElementField destination_dual = destination;
  for ( int value = 0; value < destination_dual.values.Size(); ++value ) {
    destination_dual.values[value] = 0.5 + 0.0078125 * value;
  }
  mfem::QuadratureFunction source_dual( quadrature_space, 2 );
  source_dual = 0.0;
  exchange.AddToSource( destination_dual, source_dual );
  const double local_destination_work = destination.values * destination_dual.values;
  const double local_source_work = source * source_dual;
  double global_destination_work = 0.0;
  double global_source_work = 0.0;
  MPI_Allreduce( &local_destination_work, &global_destination_work, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_source_work, &global_source_work, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( global_destination_work, global_source_work,
               1.0e-12 * std::max( 1.0, std::abs( global_destination_work ) ) );
}

// Repeated child vertices must agree in the primal action and receive each
// unique patch-node dual through exactly one designated occurrence.
TEST( PatchNodalLayoutOperator, selects_one_occurrence_and_returns_each_dual_once )
{
  mfem::Array<int> occurrence_to_node{ 0, 1, 2, 1, 3, 2 };
  PatchNodalLayoutOperator layout( std::move( occurrence_to_node ), 4, 2 );
  mfem::Vector repeated_values( layout.Width() );
  for ( int component = 0; component < layout.VectorDimension(); ++component ) {
    for ( int occurrence = 0; occurrence < layout.NumOccurrences(); ++occurrence ) {
      const int node = layout.OccurrenceToNode()[occurrence];
      repeated_values[occurrence + component * layout.NumOccurrences()] = 10.0 * component + node;
    }
  }
  mfem::Vector nodal_values( layout.Height() );
  layout.Mult( repeated_values, nodal_values );

  mfem::Vector nodal_dual( layout.Height() );
  for ( int i = 0; i < nodal_dual.Size(); ++i ) {
    nodal_dual[i] = 0.25 + i;
  }
  mfem::Vector repeated_dual( layout.Width() );
  layout.MultTranspose( nodal_dual, repeated_dual );

  EXPECT_DOUBLE_EQ( repeated_values * repeated_dual, nodal_values * nodal_dual );
  for ( int component = 0; component < layout.VectorDimension(); ++component ) {
    for ( int occurrence = 0; occurrence < layout.NumOccurrences(); ++occurrence ) {
      const int node = layout.OccurrenceToNode()[occurrence];
      const double expected =
          layout.DesignatedOccurrences()[node] == occurrence ? nodal_dual[node + component * layout.NumNodes()] : 0.0;
      EXPECT_DOUBLE_EQ( repeated_dual[occurrence + component * layout.NumOccurrences()], expected );
    }
  }
}

enum class SpaceFamily
{
  H1,
  L2,
  HCurl,
  HDiv
};

class DistributedRestrictionTest : public testing::TestWithParam<SpaceFamily> {};

// H1, L2, H(curl), and H(div) volume restrictions exercise MFEM constraints,
// signs, orientations, vector ordering, and additive MPI return.
TEST_P( DistributedRestrictionTest, preserves_the_global_adjoint_identity )
{
  int rank = 0;
  int num_ranks = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_ranks );

  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL );
  mfem::ParMesh parallel_mesh( MPI_COMM_WORLD, serial_mesh );
  std::unique_ptr<mfem::FiniteElementCollection> collection;
  switch ( GetParam() ) {
    case SpaceFamily::H1:
      collection = std::make_unique<mfem::H1_FECollection>( 2, 2 );
      break;
    case SpaceFamily::L2:
      collection = std::make_unique<mfem::L2_FECollection>( 1, 2 );
      break;
    case SpaceFamily::HCurl:
      collection = std::make_unique<mfem::ND_FECollection>( 1, 2 );
      break;
    case SpaceFamily::HDiv:
      collection = std::make_unique<mfem::RT_FECollection>( 0, 2 );
      break;
  }
  mfem::ParFiniteElementSpace space( &parallel_mesh, collection.get() );

  mfem::Array<int> selected_elements( parallel_mesh.GetNE() );
  for ( int element = 0; element < selected_elements.Size(); ++element ) {
    selected_elements[element] = element;
  }
  std::vector<mfem::Array<int>> elements_by_destination( static_cast<size_t>( num_ranks ) );
  auto& next_rank_elements = elements_by_destination[static_cast<size_t>( ( rank + 1 ) % num_ranks )];
  for ( int element = 0; element < selected_elements.Size(); ++element ) {
    next_rank_elements.Append( element );
  }

  DistributedElementRestriction restriction( space, std::move( selected_elements ),
                                             std::move( elements_by_destination ) );
  mfem::Vector true_values( restriction.Width() );
  mfem::Vector element_dual( restriction.Height() );
  for ( int i = 0; i < true_values.Size(); ++i ) {
    true_values[i] = 1.0 + 0.25 * rank + 0.125 * i;
  }
  for ( int i = 0; i < element_dual.Size(); ++i ) {
    element_dual[i] = 0.5 - 0.0625 * i;
  }

  mfem::Vector element_values( restriction.Height() );
  mfem::Vector true_dual( restriction.Width() );
  restriction.Mult( true_values, element_values );
  restriction.MultTranspose( element_dual, true_dual );

  double local_forward_pairing = element_values * element_dual;
  double local_transpose_pairing = true_values * true_dual;
  double global_forward_pairing = 0.0;
  double global_transpose_pairing = 0.0;
  MPI_Allreduce( &local_forward_pairing, &global_forward_pairing, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_transpose_pairing, &global_transpose_pairing, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( global_forward_pairing, global_transpose_pairing,
               1.0e-12 * std::max( 1.0, std::abs( global_forward_pairing ) ) );
}

INSTANTIATE_TEST_SUITE_P( FiniteElementFamilies, DistributedRestrictionTest,
                          testing::Values( SpaceFamily::H1, SpaceFamily::L2, SpaceFamily::HCurl, SpaceFamily::HDiv ) );

class DistributedBoundaryRestrictionTest : public testing::TestWithParam<SpaceFamily> {};

// Boundary E-vector restrictions verify the same global adjoint identity for
// every supported MFEM family and signed boundary orientation.
TEST_P( DistributedBoundaryRestrictionTest, preserves_the_global_adjoint_identity )
{
  int rank = 0;
  int num_ranks = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_ranks );

  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D( 2, 2, mfem::Element::QUADRILATERAL );
  mfem::ParMesh parallel_mesh( MPI_COMM_WORLD, serial_mesh );
  std::unique_ptr<mfem::FiniteElementCollection> collection;
  switch ( GetParam() ) {
    case SpaceFamily::H1:
      collection = std::make_unique<mfem::H1_FECollection>( 3, 2 );
      break;
    case SpaceFamily::L2:
      collection = std::make_unique<mfem::L2_FECollection>( 2, 2 );
      break;
    case SpaceFamily::HCurl:
      collection = std::make_unique<mfem::ND_FECollection>( 2, 2 );
      break;
    case SpaceFamily::HDiv:
      collection = std::make_unique<mfem::RT_FECollection>( 1, 2 );
      break;
  }
  mfem::ParFiniteElementSpace space( &parallel_mesh, collection.get() );

  mfem::Array<int> selected_boundary_elements( parallel_mesh.GetNBE() );
  std::vector<mfem::Array<int>> elements_by_destination( static_cast<size_t>( num_ranks ) );
  auto& next_rank_elements = elements_by_destination[static_cast<size_t>( ( rank + 1 ) % num_ranks )];
  for ( int boundary_element = 0; boundary_element < selected_boundary_elements.Size(); ++boundary_element ) {
    selected_boundary_elements[boundary_element] = boundary_element;
    next_rank_elements.Append( boundary_element );
  }

  DistributedBoundaryRestriction restriction( space, std::move( selected_boundary_elements ),
                                              std::move( elements_by_destination ) );
  mfem::Vector true_values( restriction.Width() );
  mfem::Vector boundary_dual( restriction.Height() );
  for ( int i = 0; i < true_values.Size(); ++i ) {
    true_values[i] = 0.75 + 0.25 * rank + 0.0625 * i;
  }
  for ( int i = 0; i < boundary_dual.Size(); ++i ) {
    boundary_dual[i] = 1.25 - 0.03125 * i;
  }

  mfem::Vector boundary_values( restriction.Height() );
  mfem::Vector true_dual( restriction.Width() );
  restriction.Mult( true_values, boundary_values );
  restriction.MultTranspose( boundary_dual, true_dual );

  const double local_forward_pairing = boundary_values * boundary_dual;
  const double local_transpose_pairing = true_values * true_dual;
  double global_forward_pairing = 0.0;
  double global_transpose_pairing = 0.0;
  MPI_Allreduce( &local_forward_pairing, &global_forward_pairing, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_transpose_pairing, &global_transpose_pairing, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( global_forward_pairing, global_transpose_pairing,
               1.0e-12 * std::max( 1.0, std::abs( global_forward_pairing ) ) );
}

INSTANTIATE_TEST_SUITE_P( FiniteElementFamilies, DistributedBoundaryRestrictionTest,
                          testing::Values( SpaceFamily::H1, SpaceFamily::L2, SpaceFamily::HCurl, SpaceFamily::HDiv ) );

// The complete H=C B X R composition must preserve work between parent true
// DOFs and deduplicated patch nodes.
TEST( ContactPatchOperator, composed_action_preserves_work )
{
  int rank = 0;
  int num_ranks = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &num_ranks );

  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D( 2, 1, mfem::Element::QUADRILATERAL );
  mfem::ParMesh parallel_mesh( MPI_COMM_WORLD, serial_mesh );
  mfem::H1_FECollection collection( 3, 2 );
  mfem::ParFiniteElementSpace space( &parallel_mesh, &collection, 2, mfem::Ordering::byNODES );
  mfem::Array<int> selected_boundary_elements( parallel_mesh.GetNBE() );
  std::vector<mfem::Array<int>> elements_by_destination( static_cast<size_t>( num_ranks ) );
  auto& destination_elements = elements_by_destination[static_cast<size_t>( ( rank + 1 ) % num_ranks )];
  for ( int boundary_element = 0; boundary_element < selected_boundary_elements.Size(); ++boundary_element ) {
    selected_boundary_elements[boundary_element] = boundary_element;
    destination_elements.Append( boundary_element );
  }

  constexpr int refinement_factor = 3;
  int num_destination_faces = 0;
  const int source_rank = ( rank + num_ranks - 1 ) % num_ranks;
  const int num_selected_faces = selected_boundary_elements.Size();
  MPI_Sendrecv( &num_selected_faces, 1, MPI_INT, ( rank + 1 ) % num_ranks, 0, &num_destination_faces, 1, MPI_INT,
                source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
  std::vector<VirtualFaceSpecification> destination_faces(
      static_cast<size_t>( num_destination_faces ),
      VirtualFaceSpecification{ mfem::Geometry::SEGMENT, 3, refinement_factor } );
  const int occurrences_per_face = refinement_factor * 2;
  mfem::Array<int> occurrence_to_node( num_destination_faces * occurrences_per_face );
  for ( int occurrence = 0; occurrence < occurrence_to_node.Size(); ++occurrence ) {
    occurrence_to_node[occurrence] = occurrence;
  }
  const int num_patch_nodes = occurrence_to_node.Size();
  ContactPatchOperator patch_operator( space, std::move( selected_boundary_elements ),
                                       std::move( elements_by_destination ), std::move( destination_faces ),
                                       std::move( occurrence_to_node ), num_patch_nodes );

  mfem::Vector source( patch_operator.Width() );
  mfem::Vector patch_dual( patch_operator.Height() );
  for ( int i = 0; i < source.Size(); ++i ) {
    source[i] = 1.0 + 0.125 * rank + 0.03125 * i;
  }
  for ( int i = 0; i < patch_dual.Size(); ++i ) {
    patch_dual[i] = 0.75 - 0.0078125 * i;
  }
  mfem::Vector patch_values( patch_operator.Height() );
  mfem::Vector source_dual( patch_operator.Width() );
  patch_operator.Mult( source, patch_values );
  patch_operator.MultTranspose( patch_dual, source_dual );

  const double local_forward_pairing = patch_values * patch_dual;
  const double local_transpose_pairing = source * source_dual;
  double global_forward_pairing = 0.0;
  double global_transpose_pairing = 0.0;
  MPI_Allreduce( &local_forward_pairing, &global_forward_pairing, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  MPI_Allreduce( &local_transpose_pairing, &global_transpose_pairing, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( global_forward_pairing, global_transpose_pairing,
               1.0e-12 * std::max( 1.0, std::abs( global_forward_pairing ) ) );

  auto assembled = patch_operator.Assemble();
  mfem::Vector assembled_values( patch_operator.Height() );
  assembled.get().Mult( source, assembled_values );
  assembled_values -= patch_values;
  EXPECT_LT( assembled_values.Norml2(), 1.0e-12 );
}

struct VirtualRefinementCase {
  int dimension;
  mfem::Geometry::Type geometry;
  int order;
  int factor;
};

class VirtualRefinementTest : public testing::TestWithParam<VirtualRefinementCase> {};

// Analytic polynomial values at child vertices validate segment, triangle, and
// quadrilateral reference templates; the transpose checks discrete work.
TEST_P( VirtualRefinementTest, evaluates_fields_and_preserves_work )
{
  const auto test_case = GetParam();
  mfem::H1_FECollection collection( test_case.order, test_case.dimension );
  const mfem::FiniteElement* element = collection.FiniteElementForGeometry( test_case.geometry );
  ASSERT_NE( element, nullptr );
  constexpr int vector_dimension = 2;
  VirtualRefinementOperator refinement( *element, vector_dimension, test_case.factor );

  mfem::Vector source( refinement.Width() );
  const auto& source_nodes = element->GetNodes();
  for ( int component = 0; component < vector_dimension; ++component ) {
    for ( int dof = 0; dof < element->GetDof(); ++dof ) {
      const auto& point = source_nodes.IntPoint( dof );
      source[dof + element->GetDof() * component] =
          1.0 + component + 2.0 * point.x + ( test_case.dimension > 1 ? 3.0 * point.y : 0.0 );
    }
  }

  mfem::Vector refined( refinement.Height() );
  refinement.Mult( source, refined );
  const auto& child_points = refinement.ChildVertexReferencePoints();
  for ( int component = 0; component < vector_dimension; ++component ) {
    for ( int point_index = 0; point_index < child_points.GetNPoints(); ++point_index ) {
      const auto& point = child_points.IntPoint( point_index );
      const double expected = 1.0 + component + 2.0 * point.x + ( test_case.dimension > 1 ? 3.0 * point.y : 0.0 );
      EXPECT_NEAR( refined[point_index + child_points.GetNPoints() * component], expected, 1.0e-13 );
    }
  }

  mfem::Vector child_dual( refinement.Height() );
  for ( int i = 0; i < child_dual.Size(); ++i ) {
    child_dual[i] = 0.5 + 0.03125 * i;
  }
  mfem::Vector source_dual( refinement.Width() );
  refinement.MultTranspose( child_dual, source_dual );
  EXPECT_NEAR( refined * child_dual, source * source_dual,
               1.0e-13 * std::max( 1.0, std::abs( refined * child_dual ) ) );
}

INSTANTIATE_TEST_SUITE_P( SupportedSurfaceGeometries, VirtualRefinementTest,
                          testing::Values( VirtualRefinementCase{ 1, mfem::Geometry::SEGMENT, 1, 2 },
                                           VirtualRefinementCase{ 1, mfem::Geometry::SEGMENT, 4, 3 },
                                           VirtualRefinementCase{ 2, mfem::Geometry::TRIANGLE, 2, 3 },
                                           VirtualRefinementCase{ 2, mfem::Geometry::TRIANGLE, 3, 4 },
                                           VirtualRefinementCase{ 2, mfem::Geometry::SQUARE, 4, 3 },
                                           VirtualRefinementCase{ 2, mfem::Geometry::SQUARE, 5, 2 } ) );

}  // namespace
}  // namespace redecomp

#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
