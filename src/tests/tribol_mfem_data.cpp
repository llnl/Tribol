// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/config.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>

#include <gtest/gtest.h>

#include "mfem.hpp"

#include "shared/mesh/MeshBuilder.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MfemData.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

namespace {

constexpr tribol::IndexT first_mesh_id = 0;
constexpr tribol::IndexT second_mesh_id = 1;
constexpr int elements_per_direction = 2;
constexpr int first_volume_attribute = 1;
constexpr int second_volume_attribute = 2;
constexpr int lower_boundary_attribute = 1;
constexpr int first_contact_attribute = 6;
constexpr int second_contact_attribute = 7;
constexpr int unused_upper_boundary_attribute = 8;
constexpr double contact_surface_offset = 1.0;
constexpr double binning_proximity_scale = 1.0;
constexpr bool forces_redecomp_rebuild = true;
constexpr bool pressure_uses_device = false;
constexpr bool transfers_fields_through_lor = false;
constexpr int scalar_field_dimension = 1;
constexpr int scalar_mpi_value_count = 1;
constexpr int low_order = 1;
constexpr int no_lor_factor = 0;
constexpr int quadratic_order = 2;
constexpr int cubic_order = 3;
constexpr int linear_quadrilateral_node_count = 4;
constexpr int quadratic_quadrilateral_node_count = 9;
constexpr int x_component = 0;
constexpr int y_component = 1;
constexpr int z_component = 2;
constexpr int scalar_component = 0;
constexpr double velocity_x_offset = 1.0;
constexpr double velocity_y_offset = 2.0;
constexpr double velocity_z_offset = 3.0;
constexpr double pressure_offset = 4.0;
constexpr double pressure_y_weight = 2.0;
constexpr double comparison_tolerance = 1.0e-12;
constexpr double minimum_comparison_norm = 1.0;
constexpr double zero = 0.0;

mfem::ParMesh MakeContactMesh()
{
  // Each contact surface has several elements so both MPI ranks exercise higher-order DOFs owned by different parent
  // partitions.
  // clang-format off
  return shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
    shared::MeshBuilder::CubeMesh(elements_per_direction, elements_per_direction, elements_per_direction),
    shared::MeshBuilder::CubeMesh(elements_per_direction, elements_per_direction, elements_per_direction)
      .translate({zero, zero, contact_surface_offset})
      .updateAttrib(first_volume_attribute, second_volume_attribute)
      .updateBdrAttrib(lower_boundary_attribute, second_contact_attribute)
      .updateBdrAttrib(first_contact_attribute, unused_upper_boundary_attribute)
  }));
  // clang-format on
}

void ExpectVectorsNear( mfem::Vector actual, const mfem::Vector& expected )
{
  actual -= expected;
  EXPECT_LT( actual.Norml2() / std::max( minimum_comparison_norm, expected.Norml2() ), comparison_tolerance );
}

void ExpectReturnedDualMatchesField( mfem::Vector& returned_dual, mfem::ParGridFunction& expected_field )
{
  // MFEM stores dual fields with only owned shared DOFs populated. Parallel assembly sums that local storage into the
  // true-DOF values used for the comparison.
  mfem::ParLinearForm returned_form( expected_field.ParFESpace(), returned_dual.GetData() );
  mfem::Vector returned_true_dofs( expected_field.ParFESpace()->GetTrueVSize() );
  mfem::Vector expected_true_dofs( expected_field.ParFESpace()->GetTrueVSize() );
  returned_form.ParallelAssemble( returned_true_dofs );
  expected_field.GetTrueDofs( expected_true_dofs );
  ExpectVectorsNear( returned_true_dofs, expected_true_dofs );
}

void ExpectConnectivityMatchesFiniteElementSpace( tribol::MfemMeshData& mesh_data,
                                                  const mfem::FiniteElementSpace& redecomp_fes,
                                                  const std::set<int>& registered_attributes, int registered_elements,
                                                  const tribol::IndexT* connectivity )
{
  const auto& redecomp_mesh = mesh_data.GetRedecompMesh();
  int registered_element{};
  for ( int redecomp_element{}; redecomp_element < redecomp_mesh.GetNE(); ++redecomp_element ) {
    if ( registered_attributes.count( redecomp_mesh.GetAttribute( redecomp_element ) ) == 0 ) {
      continue;
    }

    ASSERT_LT( registered_element, registered_elements );
    mfem::Array<int> element_dofs;
    redecomp_fes.GetElementDofs( redecomp_element, element_dofs );
    ASSERT_EQ( element_dofs.Size(), mesh_data.GetNodesPerElement() );
    for ( int local_node{}; local_node < element_dofs.Size(); ++local_node ) {
      const auto connectivity_node = connectivity[registered_element * element_dofs.Size() + local_node];
      EXPECT_EQ( connectivity_node, mfem::FiniteElementSpace::DecodeDof( element_dofs[local_node] ) );
      EXPECT_GE( connectivity_node, 0 );
      EXPECT_LT( connectivity_node, mesh_data.GetNumNodes() );
    }
    ++registered_element;
  }
  EXPECT_EQ( registered_element, registered_elements );
}

class HigherOrderMfemDataTest : public testing::TestWithParam<int> {};

TEST_P( HigherOrderMfemDataTest, default_transfer_uses_lor )
{
  const int order = GetParam();
  auto mesh = MakeContactMesh();
  mesh.SetCurvature( order );
  auto* mesh_nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
  ASSERT_NE( mesh_nodes, nullptr );
  mfem::ParGridFunction coordinates( mesh_nodes->ParFESpace() );
  coordinates = *mesh_nodes;

  tribol::MfemMeshData mesh_data( first_mesh_id, second_mesh_id, mesh, coordinates,
                                  std::set<int>{ first_contact_attribute }, std::set<int>{ second_contact_attribute },
                                  tribol::ExecutionMode::Sequential, tribol::MemorySpace::Host );
  ASSERT_NE( mesh_data.GetLORMesh(), nullptr );
  EXPECT_EQ( mesh_data.GetLORFactor(), order );

  int rank_count{};
  MPI_Comm_size( MPI_COMM_WORLD, &rank_count );
  mesh_data.UpdateMfemMeshData( binning_proximity_scale, rank_count, forces_redecomp_rebuild );
  EXPECT_EQ( mesh_data.GetRedecompResponse().FESpace()->FEColl()->GetOrder(), low_order );
  EXPECT_EQ( mesh_data.GetFaceOrder(), tribol::LINEAR_LAGRANGE );
  EXPECT_EQ( mesh_data.GetNodesPerElement(), linear_quadrilateral_node_count );
  EXPECT_EQ( mesh_data.GetNumNodes(), mesh_data.GetRedecompResponse().FESpace()->GetNDofs() );

  auto pressure_collection = std::make_unique<mfem::H1_FECollection>( order, mesh.SpaceDimension() );
  tribol::MfemSubmeshData submesh_data( mesh_data.GetSubmesh(), mesh_data.GetLORMesh(),
                                        std::move( pressure_collection ), scalar_field_dimension,
                                        pressure_uses_device );
  submesh_data.UpdateMfemSubmeshData( mesh_data.GetRedecompMesh() );
  EXPECT_EQ( submesh_data.GetRedecompGap().FESpace()->FEColl()->GetOrder(), low_order );
}

TEST_P( HigherOrderMfemDataTest, direct_transfer_retains_higher_order_fields )
{
  const int order = GetParam();
  auto mesh = MakeContactMesh();
  mesh.SetCurvature( order );
  auto* mesh_nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
  ASSERT_NE( mesh_nodes, nullptr );
  mfem::ParGridFunction coordinates( mesh_nodes->ParFESpace() );
  coordinates = *mesh_nodes;

  mfem::VectorFunctionCoefficient velocity_coefficient(
      mesh.SpaceDimension(), []( const mfem::Vector& point, mfem::Vector& value ) {
        value.SetSize( point.Size() );
        value[x_component] = velocity_x_offset + point[x_component] * point[x_component] + point[y_component];
        value[y_component] = velocity_y_offset + point[y_component] * point[y_component] + point[z_component];
        value[z_component] = velocity_z_offset + point[z_component] * point[z_component] + point[x_component];
      } );
  mfem::ParGridFunction velocity( coordinates.ParFESpace() );
  velocity.ProjectCoefficient( velocity_coefficient );

  tribol::MfemMeshData mesh_data( first_mesh_id, second_mesh_id, mesh, coordinates,
                                  std::set<int>{ first_contact_attribute }, std::set<int>{ second_contact_attribute },
                                  tribol::ExecutionMode::Sequential, tribol::MemorySpace::Host,
                                  transfers_fields_through_lor );
  mesh_data.SetParentVelocity( velocity );
  EXPECT_EQ( mesh_data.GetLORMesh(), nullptr );
  EXPECT_EQ( mesh_data.GetLORFactor(), no_lor_factor );

  int rank_count{};
  MPI_Comm_size( MPI_COMM_WORLD, &rank_count );
  mesh_data.UpdateMfemMeshData( binning_proximity_scale, rank_count, forces_redecomp_rebuild );

  auto& redecomp_mesh = mesh_data.GetRedecompMesh();
  const auto* redecomp_nodes = dynamic_cast<const mfem::GridFunction*>( redecomp_mesh.GetNodes() );
  ASSERT_NE( redecomp_nodes, nullptr );
  EXPECT_EQ( redecomp_nodes->FESpace()->FEColl()->GetOrder(), order );

  auto* redecomp_fes = const_cast<mfem::FiniteElementSpace*>( mesh_data.GetRedecompResponse().FESpace() );
  ASSERT_NE( redecomp_fes, nullptr );
  EXPECT_EQ( redecomp_fes->FEColl()->GetOrder(), order );
  EXPECT_EQ( redecomp_fes->GetVDim(), mesh.SpaceDimension() );
  EXPECT_GT( redecomp_fes->GetNDofs(), redecomp_mesh.GetNV() );
  EXPECT_EQ( mesh_data.GetNumNodes(), redecomp_fes->GetNDofs() );

  if ( order == quadratic_order ) {
    EXPECT_EQ( mesh_data.GetFaceOrder(), tribol::QUADRATIC_LAGRANGE );
    EXPECT_EQ( mesh_data.GetNodesPerElement(), quadratic_quadrilateral_node_count );

    ExpectConnectivityMatchesFiniteElementSpace( mesh_data, *redecomp_fes, std::set<int>{ first_contact_attribute },
                                                 mesh_data.GetMesh1NE(), mesh_data.GetMesh1Conn() );
    ExpectConnectivityMatchesFiniteElementSpace( mesh_data, *redecomp_fes, std::set<int>{ second_contact_attribute },
                                                 mesh_data.GetMesh2NE(), mesh_data.GetMesh2Conn() );

    // Quadratic connectivity must include MFEM's edge and face-interior DOFs rather than stopping after geometric
    // vertices. Finding any non-vertex DOF also guards against accidentally returning GetElementVertices().
    int includes_higher_order_node{};
    for ( int element{}; element < mesh_data.GetMesh1NE(); ++element ) {
      for ( int local_node{}; local_node < mesh_data.GetNodesPerElement(); ++local_node ) {
        includes_higher_order_node |= static_cast<int>(
            mesh_data.GetMesh1Conn()[element * mesh_data.GetNodesPerElement() + local_node] >= redecomp_mesh.GetNV() );
      }
    }
    for ( int element{}; element < mesh_data.GetMesh2NE(); ++element ) {
      for ( int local_node{}; local_node < mesh_data.GetNodesPerElement(); ++local_node ) {
        includes_higher_order_node |= static_cast<int>(
            mesh_data.GetMesh2Conn()[element * mesh_data.GetNodesPerElement() + local_node] >= redecomp_mesh.GetNV() );
      }
    }
    MPI_Allreduce( MPI_IN_PLACE, &includes_higher_order_node, scalar_mpi_value_count, MPI_INT, MPI_MAX,
                   MPI_COMM_WORLD );
    EXPECT_TRUE( includes_higher_order_node );

    const auto coordinate_components = mesh_data.GetRedecompCoordsPtrs();
    ASSERT_EQ( redecomp_nodes->FESpace()->GetNDofs(), mesh_data.GetNumNodes() );
    for ( int component{}; component < redecomp_nodes->FESpace()->GetVDim(); ++component ) {
      ASSERT_NE( coordinate_components[component], nullptr );
      for ( int node{}; node < mesh_data.GetNumNodes(); ++node ) {
        EXPECT_NEAR( coordinate_components[component][node],
                     ( *redecomp_nodes )[redecomp_nodes->FESpace()->DofToVDof( node, component )],
                     comparison_tolerance );
      }
    }
  } else {
    EXPECT_EQ( mesh_data.GetFaceOrder(), tribol::CUBIC_LAGRANGE );
  }

  // Comparing against an independent coefficient projection checks every higher-order DOF, including edge, face, and
  // element-interior DOFs.
  mfem::GridFunction expected_velocity( redecomp_fes );
  expected_velocity.ProjectCoefficient( velocity_coefficient );
  const auto velocity_components = mesh_data.GetRedecompVelocityPtrs();
  for ( int component{}; component < redecomp_fes->GetVDim(); ++component ) {
    ASSERT_NE( velocity_components[component], nullptr );
    for ( int dof{}; dof < redecomp_fes->GetNDofs(); ++dof ) {
      EXPECT_NEAR( velocity_components[component][dof], expected_velocity[redecomp_fes->DofToVDof( dof, component )],
                   comparison_tolerance );
    }
  }

  // Reusing a transferred primal field as an already-assembled redecomp response verifies the supported replicated-dual
  // return path.
  auto response_components = mesh_data.GetRedecompResponsePtrs();
  for ( int component{}; component < redecomp_fes->GetVDim(); ++component ) {
    for ( int dof{}; dof < redecomp_fes->GetNDofs(); ++dof ) {
      response_components[component][dof] = expected_velocity[redecomp_fes->DofToVDof( dof, component )];
    }
  }
  mfem::Vector parent_response( velocity.Size() );
  parent_response = zero;
  mesh_data.GetParentResponse( parent_response );
  EXPECT_EQ( parent_response.Size(), velocity.Size() );
  double parent_response_norm = parent_response.Norml2();
  MPI_Allreduce( MPI_IN_PLACE, &parent_response_norm, scalar_mpi_value_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_GT( parent_response_norm, zero );

  auto pressure_collection = std::make_unique<mfem::H1_FECollection>( order, mesh.SpaceDimension() );
  tribol::MfemSubmeshData submesh_data( mesh_data.GetSubmesh(), mesh_data.GetLORMesh(),
                                        std::move( pressure_collection ), scalar_field_dimension,
                                        pressure_uses_device );
  mfem::FunctionCoefficient pressure_coefficient( []( const mfem::Vector& point ) {
    return pressure_offset + point[x_component] * point[x_component] +
           pressure_y_weight * point[y_component] * point[y_component] + point[z_component];
  } );
  auto& submesh_pressure = submesh_data.GetSubmeshPressure();
  submesh_pressure.ProjectCoefficient( pressure_coefficient );
  submesh_data.UpdateMfemSubmeshData( redecomp_mesh );

  auto* pressure_redecomp_fes = const_cast<mfem::FiniteElementSpace*>( submesh_data.GetRedecompGap().FESpace() );
  ASSERT_NE( pressure_redecomp_fes, nullptr );
  EXPECT_EQ( pressure_redecomp_fes->FEColl()->GetOrder(), order );
  EXPECT_EQ( pressure_redecomp_fes->GetVDim(), scalar_field_dimension );
  EXPECT_GT( pressure_redecomp_fes->GetNDofs(), redecomp_mesh.GetNV() );

  mfem::GridFunction expected_pressure( pressure_redecomp_fes );
  expected_pressure.ProjectCoefficient( pressure_coefficient );
  const auto pressure_components = submesh_data.GetRedecompPressurePtrs();
  ASSERT_NE( pressure_components[scalar_component], nullptr );
  for ( int dof{}; dof < pressure_redecomp_fes->GetNDofs(); ++dof ) {
    EXPECT_NEAR( pressure_components[scalar_component][dof], expected_pressure[dof], comparison_tolerance );
  }

  submesh_data.GetRedecompGap() = expected_pressure;
  mfem::Vector submesh_gap;
  submesh_data.GetSubmeshGap( submesh_gap );
  ExpectReturnedDualMatchesField( submesh_gap, submesh_pressure );
}

INSTANTIATE_TEST_SUITE_P( HigherOrder, HigherOrderMfemDataTest, testing::Values( quadratic_order, cubic_order ) );

}  // namespace

#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();
#endif

  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
