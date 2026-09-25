// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <array>
#include <cmath>
#include <set>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

// Tribol includes
#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/utils/TestUtils.hpp"

// Shared includes
#include "shared/mesh/MeshBuilder.hpp"

// Redecomp includes
#include "redecomp/redecomp.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// MFEM includes
#include "mfem.hpp"

// Axom includes
#include "axom/CLI11.hpp"
#include "axom/slic.hpp"

namespace {

/** Check one supported face mapping and its common invalid-input paths. */
void checkParentReferenceMapping(
    tribol::InterfaceElementType face_element_type, int number_of_face_vertices,
    const std::array<tribol::RealT, 2 * tribol::ParentFaceData::max_lor_face_vertices>& parent_reference_vertices,
    const std::array<tribol::RealT, 2>& face_reference_coordinates,
    const std::array<tribol::RealT, 2>& expected_parent_reference_coordinates,
    const std::array<tribol::RealT, 2>& invalid_face_reference_coordinates )
{
  constexpr tribol::IndexT number_of_faces = 1;
  constexpr tribol::IndexT maximum_number_of_face_vertices = 4;
  const int reference_dimension = face_element_type == tribol::LINEAR_EDGE ? 1 : 2;
  const tribol::IndexT connectivity[maximum_number_of_face_vertices] = { 0, 1, 2, 3 };
  const tribol::RealT x[maximum_number_of_face_vertices] = { 0.0, 1.0, 1.0, 0.0 };
  const tribol::RealT y[maximum_number_of_face_vertices] = { 0.0, 0.0, 1.0, 1.0 };
  const tribol::RealT z[maximum_number_of_face_vertices] = { 0.0, 0.0, 0.0, 0.0 };

  tribol::MeshData mesh_data( 0, number_of_faces, number_of_face_vertices, connectivity, face_element_type, x, y,
                              face_element_type == tribol::LINEAR_EDGE ? nullptr : z, tribol::MemorySpace::Host );
  tribol::Array1D<int, tribol::MemorySpace::Host> parent_face_orders( number_of_faces );
  tribol::Array1D<int, tribol::MemorySpace::Host> reference_vertex_counts( number_of_faces );
  tribol::Array2D<tribol::RealT, tribol::MemorySpace::Host> parent_reference_vertex_coordinates(
      number_of_faces,
      tribol::ParentFaceData::max_lor_face_vertices * tribol::ParentFaceData::max_reference_dimension );
  parent_face_orders[0] = 2;
  reference_vertex_counts[0] = number_of_face_vertices;
  for ( int coordinate_index = 0; coordinate_index < tribol::ParentFaceData::max_lor_face_vertices *
                                                         tribol::ParentFaceData::max_reference_dimension;
        ++coordinate_index ) {
    parent_reference_vertex_coordinates( 0, coordinate_index ) = parent_reference_vertices[coordinate_index];
  }

  tribol::ParentFaceData parent_face_data;
  parent_face_data.m_parent_face_orders = tribol::Array1DView<const int>( parent_face_orders );
  parent_face_data.m_reference_vertex_counts = tribol::Array1DView<const int>( reference_vertex_counts );
  parent_face_data.m_parent_reference_vertex_coordinates =
      tribol::Array2DView<const tribol::RealT>( parent_reference_vertex_coordinates );
  mesh_data.setParentFaceData( parent_face_data );
  const tribol::MeshData::Viewer mesh_view = mesh_data.getView();

  ASSERT_TRUE( parent_face_data.isValid() );
  ASSERT_TRUE( mesh_view.hasParentFaceData() );

  tribol::RealT mapped_parent_reference_coordinates[tribol::ParentFaceData::max_reference_dimension] = { 0.0, 0.0 };
  ASSERT_TRUE(
      mesh_view.mapToParentReference( 0, face_reference_coordinates.data(), mapped_parent_reference_coordinates ) );
  for ( int coordinate_component = 0; coordinate_component < reference_dimension; ++coordinate_component ) {
    EXPECT_NEAR( mapped_parent_reference_coordinates[coordinate_component],
                 expected_parent_reference_coordinates[coordinate_component], 1.e-12 );
  }

  EXPECT_FALSE(
      mesh_view.mapToParentReference( -1, face_reference_coordinates.data(), mapped_parent_reference_coordinates ) );
  EXPECT_FALSE( mesh_view.mapToParentReference( number_of_faces, face_reference_coordinates.data(),
                                                mapped_parent_reference_coordinates ) );
  EXPECT_FALSE( mesh_view.mapToParentReference( 0, nullptr, mapped_parent_reference_coordinates ) );
  EXPECT_FALSE( mesh_view.mapToParentReference( 0, face_reference_coordinates.data(), nullptr ) );
  EXPECT_FALSE( mesh_view.mapToParentReference( 0, invalid_face_reference_coordinates.data(),
                                                mapped_parent_reference_coordinates ) );

  const tribol::RealT negative_first_coordinate[2] = { -0.25, 0.0 };
  EXPECT_FALSE( mesh_view.mapToParentReference( 0, negative_first_coordinate, mapped_parent_reference_coordinates ) );

  // A face element type and vertex count must describe the same reference element.
  reference_vertex_counts[0] = number_of_face_vertices - 1;
  EXPECT_FALSE(
      mesh_view.mapToParentReference( 0, face_reference_coordinates.data(), mapped_parent_reference_coordinates ) );
}

/** Return the affine native-face reference-to-physical map for one boundary attribute. */
std::array<tribol::RealT, 6> getParentFaceCoordinateMap( mfem::ParMesh& mesh, int boundary_attribute )
{
  std::array<tribol::RealT, 6> local_coordinate_map = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  int local_parent_face_count = 0;

  for ( int boundary_element_id = 0; boundary_element_id < mesh.GetNBE(); ++boundary_element_id ) {
    if ( mesh.GetBdrAttribute( boundary_element_id ) != boundary_attribute ) {
      continue;
    }

    ++local_parent_face_count;
    mfem::ElementTransformation* parent_face_transformation = mesh.GetBdrElementTransformation( boundary_element_id );
    EXPECT_NE( parent_face_transformation, nullptr );
    if ( parent_face_transformation == nullptr ) {
      continue;
    }

    // Three reference points determine the affine x-y map for these planar unit-cube faces.
    const std::array<std::array<tribol::RealT, 2>, 3> reference_points = { std::array<tribol::RealT, 2>{ 0.0, 0.0 },
                                                                           std::array<tribol::RealT, 2>{ 1.0, 0.0 },
                                                                           std::array<tribol::RealT, 2>{ 0.0, 1.0 } };
    std::array<std::array<tribol::RealT, 2>, 3> physical_points;
    for ( int point_index = 0; point_index < 3; ++point_index ) {
      mfem::IntegrationPoint reference_point;
      reference_point.x = reference_points[point_index][0];
      reference_point.y = reference_points[point_index][1];
      mfem::Vector physical_point( mesh.SpaceDimension() );
      parent_face_transformation->Transform( reference_point, physical_point );
      physical_points[point_index][0] = physical_point[0];
      physical_points[point_index][1] = physical_point[1];
    }

    local_coordinate_map[0] = physical_points[0][0];
    local_coordinate_map[1] = physical_points[0][1];
    local_coordinate_map[2] = physical_points[1][0] - physical_points[0][0];
    local_coordinate_map[3] = physical_points[1][1] - physical_points[0][1];
    local_coordinate_map[4] = physical_points[2][0] - physical_points[0][0];
    local_coordinate_map[5] = physical_points[2][1] - physical_points[0][1];
  }

  int global_parent_face_count = 0;
  MPI_Allreduce( &local_parent_face_count, &global_parent_face_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_EQ( global_parent_face_count, 1 );

  std::array<tribol::RealT, 6> global_coordinate_map = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
  MPI_Allreduce( local_coordinate_map.data(), global_coordinate_map.data(), global_coordinate_map.size(), MPI_DOUBLE,
                 MPI_SUM, MPI_COMM_WORLD );
  return global_coordinate_map;
}

}  // namespace

/**
 * @brief This tests the Tribol MFEM interface running a small common plane explicit contact example using a central
 * difference explicit time integration scheme.
 *
 * Both the element penalty and a constant penalty are tested, with the constant penalty tuned to match the element
 * penalty for this case.  As a result, the test comparisons are the same for both penalty types.
 *
 */
class MfemCommonPlaneTest : public testing::TestWithParam<std::tuple<int, tribol::KinematicPenaltyCalculation>> {
 protected:
  tribol::RealT max_disp_;
  void SetUp() override
  {
    // number of times to uniformly refine the serial mesh before constructing the
    // parallel mesh
    int ref_levels = 2;
    // polynomial order of the finite element discretization
    int order = std::get<0>( GetParam() );
    // initial separation between the two blocks
    double initial_sep = -0.001;
    // initial velocity
    tribol::RealT initial_v = 0.0;
    // timestep size
    tribol::RealT dt = 0.001;
    // end time
    tribol::RealT t_end = 0.01;
    // material density
    tribol::RealT rho = 1000.0;
    // lame parameter
    tribol::RealT lambda = 100000.0;
    // lame parameter (shear modulus)
    tribol::RealT mu = 100000.0;
    // kinematic constant penalty stiffness equivalent to the element-wise calculation,
    // which is bulk-modulus over element thickness.
    tribol::RealT p_kine = ( lambda + 2.0 / 3.0 * mu ) / ( 1.0 / std::pow( 2.0, ref_levels ) );

    // fixed options
    // boundary element attributes of contact surface 1
    auto contact_surf_1 = std::set<int>( { 6 } );
    // boundary element attributes of contact surface 2
    auto contact_surf_2 = std::set<int>( { 7 } );
    // boundary element attributes of fixed surface (points on z = 0, all t)
    auto fixed_attrs = std::set<int>( { 1 } );
    // element attribute corresponding to volume elements where an initial
    // velocity will be applied
    auto moving_attrs = std::set<int>( { 2 } );

#if defined( TRIBOL_USE_CUDA )
    tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Cuda;
#elif defined( TRIBOL_USE_HIP )
    tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Hip;
#else
    tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Sequential;
#endif

    // read mesh
    // clang-format off
    mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::Unify( {
      shared::MeshBuilder::CubeMesh( 1, 1, 1 ),
      shared::MeshBuilder::CubeMesh( 1, 1, 1 )
        .translate( { 0.0, 0.0, 1.0 + initial_sep } )
        .updateAttrib( 1, 2 )    // changes attribute in all volume elements from 1 to 2 so it doesn't clash with the 
                                 // first (bottom) mesh
        .updateBdrAttrib( 1, 7 ) // set the bottom surface to boundary attribute 7
        .updateBdrAttrib( 6, 8 ) // set the top surface to boundary attribute 8
    } ).refine( ref_levels ) );
    // clang-format on

    // grid function for higher-order nodes
    auto fe_coll = mfem::H1_FECollection( order, mesh.SpaceDimension() );
    auto par_fe_space = mfem::ParFiniteElementSpace( &mesh, &fe_coll, mesh.SpaceDimension() );
    auto coords = mfem::ParGridFunction( &par_fe_space );
    if ( order > 1 ) {
      mesh.SetNodalGridFunction( &coords, false );
    } else {
      mesh.GetNodes( coords );
    }

    mfem::ParGridFunction ref_coords{ coords };

    // grid function for displacement
    mfem::ParGridFunction displacement{ &par_fe_space };
    displacement = 0.0;

    // grid function for velocity
    mfem::ParGridFunction velocity{ &par_fe_space };
    velocity = 0.0;

    // set initial velocity
    mfem::Vector init_velocity_vector( { 0.0, 0.0, -std::abs( initial_v ) } );
    mfem::VectorConstantCoefficient init_velocity_coeff( init_velocity_vector );
    mfem::Array<int> moving_attrs_array;
    mfem::Array<mfem::VectorCoefficient*> init_velocity_coeff_array;
    moving_attrs_array.Reserve( moving_attrs.size() );
    init_velocity_coeff_array.Reserve( moving_attrs.size() );
    for ( auto moving_attr : moving_attrs ) {
      moving_attrs_array.Append( moving_attr );
      init_velocity_coeff_array.Append( &init_velocity_coeff );
    }
    mfem::PWVectorCoefficient initial_v_coeff( mesh.SpaceDimension(), moving_attrs_array, init_velocity_coeff_array );
    velocity.ProjectCoefficient( initial_v_coeff );

    // recover dirichlet bc tdof list
    mfem::Array<int> ess_vdof_list;
    {
      mfem::Array<int> ess_vdof_marker;
      mfem::Array<int> ess_bdr( mesh.bdr_attributes.Max() );
      ess_bdr = 0;
      for ( auto fixed_attr : fixed_attrs ) {
        ess_bdr[fixed_attr - 1] = 1;
      }
      par_fe_space.GetEssentialVDofs( ess_bdr, ess_vdof_marker );
      mfem::FiniteElementSpace::MarkerToList( ess_vdof_marker, ess_vdof_list );
    }

    // set up mfem elasticity bilinear form
    mfem::ConstantCoefficient rho_coeff{ rho };
    mfem::ConstantCoefficient lambda_coeff{ lambda };
    mfem::ConstantCoefficient mu_coeff{ mu };
    mfem_ext::ExplicitMechanics op{ par_fe_space, rho_coeff, lambda_coeff, mu_coeff };

    // set up time integrator
    mfem_ext::CentralDiffSolver solver{ ess_vdof_list };
    solver.Init( op );

    // set up tribol
    int coupling_scheme_id = 0;
    int mesh1_id = 0;
    int mesh2_id = 1;
    tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, contact_surf_1,
                                        contact_surf_2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                                        tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                        tribol::BINNING_BVH, exec_mode );
    tribol::registerMfemVelocity( 0, velocity );
    if ( std::get<1>( GetParam() ) == tribol::KINEMATIC_CONSTANT ) {
      tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, p_kine, p_kine );
    } else {
      mfem::Vector bulk_moduli_by_bdry_attrib( mesh.bdr_attributes.Max() );
      bulk_moduli_by_bdry_attrib = lambda + 2.0 / 3.0 * mu;
      mfem::PWConstCoefficient mat_coeff( bulk_moduli_by_bdry_attrib );
      tribol::setMfemKinematicElementPenalty( coupling_scheme_id, mat_coeff );
    }

    int cycle{ 0 };
    for ( tribol::RealT t{ 0.0 }; t < t_end; t += dt ) {
      // build new parallel decomposed redecomp mesh and update grid functions
      // on each mesh
      tribol::updateMfemParallelDecomposition();
      tribol::update( cycle, t, dt );
      op.f_ext = 0.0;
      tribol::getMfemResponse( 0, op.f_ext );

      op.SetTime( t );
      solver.Step( displacement, velocity, t, dt );

      coords.Set( 1.0, ref_coords );
      coords += displacement;
      if ( order == 1 ) {
        coords.HostRead();
        mesh.SetVertices( coords );
      }

      ++cycle;
    }

    max_disp_ = displacement.Max();
    MPI_Allreduce( MPI_IN_PLACE, &max_disp_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

    tribol::finalize();
  }
};

TEST_P( MfemCommonPlaneTest, common_plane )
{
  // make sure there is some contact response after 10 cycles
  EXPECT_GT( max_disp_, 1.0e-4 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( tribol, MfemCommonPlaneTest,
                          testing::Values( std::make_tuple( 1, tribol::KINEMATIC_CONSTANT ),
                                           std::make_tuple( 1, tribol::KINEMATIC_ELEMENT ),
                                           std::make_tuple( 2, tribol::KINEMATIC_CONSTANT ),
                                           std::make_tuple( 2, tribol::KINEMATIC_ELEMENT ),
                                           std::make_tuple( 3, tribol::KINEMATIC_CONSTANT ),
                                           std::make_tuple( 4, tribol::KINEMATIC_CONSTANT ) ) );

/** Verify identity mappings for every supported contact face element type. */
TEST( MfemCommonPlaneParentFaceData, MapsSupportedFaceElementTypesAndRejectsInvalidInput )
{
  // A no-LOR edge maps directly to the same native parent reference interval.
  checkParentReferenceMapping( tribol::LINEAR_EDGE, 2, { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, { 0.25, 0.0 },
                               { 0.25, 0.0 }, { 1.25, 0.0 } );

  // A no-LOR triangle maps directly to the same native parent reference triangle.
  checkParentReferenceMapping( tribol::LINEAR_TRIANGLE, 3, { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0 }, { 0.25, 0.25 },
                               { 0.25, 0.25 }, { 0.75, 0.5 } );

  // A no-LOR quadrilateral maps directly to the same native parent reference square.
  checkParentReferenceMapping( tribol::LINEAR_QUAD, 4, { 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 }, { 0.25, 0.75 },
                               { 0.25, 0.75 }, { 0.5, 1.25 } );

  // This lower-left child represents one face from a factor-three LOR mesh.
  checkParentReferenceMapping( tribol::LINEAR_QUAD, 4,
                               { 0.0, 0.0, 1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0 }, { 0.5, 0.5 },
                               { 1.0 / 6.0, 1.0 / 6.0 }, { 0.5, 1.25 } );
}

/** Verify that parent-face validity requires compatible array dimensions. */
TEST( MfemCommonPlaneParentFaceData, ValidatesParentFaceArrayDimensions )
{
  constexpr tribol::IndexT number_of_parent_faces = 2;
  constexpr int parent_reference_coordinate_count =
      tribol::ParentFaceData::max_lor_face_vertices * tribol::ParentFaceData::max_reference_dimension;
  tribol::Array1D<int, tribol::MemorySpace::Host> parent_face_orders( number_of_parent_faces );
  tribol::Array1D<int, tribol::MemorySpace::Host> one_reference_vertex_count( 1 );
  tribol::Array1D<int, tribol::MemorySpace::Host> reference_vertex_counts( number_of_parent_faces );
  tribol::Array2D<tribol::RealT, tribol::MemorySpace::Host> one_coordinate_row( 1, parent_reference_coordinate_count );
  tribol::Array2D<tribol::RealT, tribol::MemorySpace::Host> narrow_coordinate_rows(
      number_of_parent_faces, parent_reference_coordinate_count - 1 );
  tribol::Array2D<tribol::RealT, tribol::MemorySpace::Host> compatible_coordinate_rows(
      number_of_parent_faces, parent_reference_coordinate_count );

  tribol::ParentFaceData parent_face_data;
  parent_face_data.m_parent_face_orders = tribol::Array1DView<const int>( parent_face_orders );
  parent_face_data.m_reference_vertex_counts = tribol::Array1DView<const int>( one_reference_vertex_count );
  parent_face_data.m_parent_reference_vertex_coordinates =
      tribol::Array2DView<const tribol::RealT>( one_coordinate_row );
  EXPECT_FALSE( parent_face_data.isValid() );

  parent_face_data.m_reference_vertex_counts = tribol::Array1DView<const int>( reference_vertex_counts );
  EXPECT_FALSE( parent_face_data.isValid() );

  parent_face_data.m_parent_reference_vertex_coordinates =
      tribol::Array2DView<const tribol::RealT>( narrow_coordinate_rows );
  EXPECT_FALSE( parent_face_data.isValid() );

  parent_face_data.m_parent_reference_vertex_coordinates =
      tribol::Array2DView<const tribol::RealT>( compatible_coordinate_rows );
  EXPECT_TRUE( parent_face_data.isValid() );

  // Parent-face rows must also match the number of faces in the registered mesh.
  const tribol::IndexT connectivity[4] = { 0, 1, 2, 3 };
  const tribol::RealT x[4] = { 0.0, 1.0, 1.0, 0.0 };
  const tribol::RealT y[4] = { 0.0, 0.0, 1.0, 1.0 };
  const tribol::RealT z[4] = { 0.0, 0.0, 0.0, 0.0 };
  tribol::MeshData mesh_data( 0, 1, 4, connectivity, tribol::LINEAR_QUAD, x, y, z, tribol::MemorySpace::Host );
  mesh_data.setParentFaceData( parent_face_data );
  EXPECT_FALSE( mesh_data.getView().hasParentFaceData() );
}

/** Verify native parent-reference mapping after LOR refinement and redecomposition. */
class MfemCommonPlaneParentFaceDataTest : public testing::TestWithParam<std::tuple<int, int>> {};

TEST_P( MfemCommonPlaneParentFaceDataTest, MapsRedecomposedQuadrilateralFacesToParentFace )
{
  const int parent_order = std::get<0>( GetParam() );
  const int lor_factor = std::get<1>( GetParam() );
  constexpr double initial_separation = -0.001;
  constexpr tribol::IndexT coupling_scheme_id = 0;
  constexpr tribol::IndexT first_mesh_id = 0;
  constexpr tribol::IndexT second_mesh_id = 1;

  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::Unify( {
    shared::MeshBuilder::CubeMesh( 1, 1, 1 ),
    shared::MeshBuilder::CubeMesh( 1, 1, 1 )
      .translate( { 0.0, 0.0, 1.0 + initial_separation } )
      .updateAttrib( 1, 2 )
      .updateBdrAttrib( 1, 7 )
      .updateBdrAttrib( 6, 8 )
  } ) );
  // clang-format on

  // Elevate the linear mesh while preserving its physical coordinates. Merely
  // attaching an uninitialized high-order grid function would make the native
  // parent and LOR geometry represent different surfaces.
  mesh.SetCurvature( parent_order );
  auto* coordinate_nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
  ASSERT_NE( coordinate_nodes, nullptr );
  mfem::ParGridFunction coordinates( coordinate_nodes->ParFESpace() );
  coordinates = *coordinate_nodes;

  // These independently generated affine maps convert each physical LOR face
  // center back to the corresponding native parent-face reference point.
  const std::array<tribol::RealT, 6> first_parent_coordinate_map = getParentFaceCoordinateMap( mesh, 6 );
  const std::array<tribol::RealT, 6> second_parent_coordinate_map = getParentFaceCoordinateMap( mesh, 7 );

#if defined( TRIBOL_USE_CUDA )
  constexpr tribol::ExecutionMode execution_mode = tribol::ExecutionMode::Cuda;
#elif defined( TRIBOL_USE_HIP )
  constexpr tribol::ExecutionMode execution_mode = tribol::ExecutionMode::Hip;
#else
  constexpr tribol::ExecutionMode execution_mode = tribol::ExecutionMode::Sequential;
#endif

  tribol::registerMfemCouplingScheme( coupling_scheme_id, first_mesh_id, second_mesh_id, mesh, coordinates, { 6 },
                                      { 7 }, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                      tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_BVH, execution_mode );
  if ( lor_factor > 1 ) {
    tribol::setMfemLORFactor( coupling_scheme_id, lor_factor );
  }
  tribol::updateMfemParallelDecomposition( 0, true );
  tribol::MfemMeshData* mfem_data =
      tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id ).getMfemMeshData();
  ASSERT_NE( mfem_data, nullptr );

  for ( const tribol::IndexT mesh_id : { first_mesh_id, second_mesh_id } ) {
    tribol::MeshData& mesh_data = tribol::MeshManager::getInstance().at( mesh_id );
    const tribol::MeshData::Viewer mesh_view = mesh_data.getView();
    const tribol::IndexT number_of_faces = mesh_view.numberOfElements();
    EXPECT_TRUE( number_of_faces == 0 || mesh_view.hasParentFaceData() );

    const std::array<tribol::RealT, 6>& parent_coordinate_map =
        mesh_id == first_mesh_id ? first_parent_coordinate_map : second_parent_coordinate_map;
    const tribol::RealT physical_origin_x = parent_coordinate_map[0];
    const tribol::RealT physical_origin_y = parent_coordinate_map[1];
    const tribol::RealT physical_x_per_first_reference_coordinate = parent_coordinate_map[2];
    const tribol::RealT physical_y_per_first_reference_coordinate = parent_coordinate_map[3];
    const tribol::RealT physical_x_per_second_reference_coordinate = parent_coordinate_map[4];
    const tribol::RealT physical_y_per_second_reference_coordinate = parent_coordinate_map[5];
    const tribol::RealT coordinate_map_determinant =
        physical_x_per_first_reference_coordinate * physical_y_per_second_reference_coordinate -
        physical_x_per_second_reference_coordinate * physical_y_per_first_reference_coordinate;
    ASSERT_GT( std::abs( coordinate_map_determinant ), 1.e-12 );

    // Evaluate each redecomposed face center with MFEM, independently of the
    // transferred parent-reference mapping stored by Tribol.
    const tribol::Array1D<int>& surface_element_map =
        mesh_id == first_mesh_id ? mfem_data->GetElemMap1() : mfem_data->GetElemMap2();
    const tribol::Array1D<int, tribol::MemorySpace::Host> host_surface_element_map( surface_element_map );
    ASSERT_EQ( host_surface_element_map.size(), number_of_faces );
    tribol::Array2D<tribol::RealT, tribol::MemorySpace::Host> host_expected_parent_centers( number_of_faces, 2 );
    mfem::Array<int> redecomp_face_vertex_ids;
    for ( tribol::IndexT face_id = 0; face_id < number_of_faces; ++face_id ) {
      mfem_data->GetRedecompMesh().GetElementVertices( host_surface_element_map[face_id], redecomp_face_vertex_ids );
      ASSERT_EQ( redecomp_face_vertex_ids.Size(), 4 );
      tribol::RealT physical_face_center[2] = { 0.0, 0.0 };
      for ( int vertex_index = 0; vertex_index < redecomp_face_vertex_ids.Size(); ++vertex_index ) {
        const double* physical_vertex =
            mfem_data->GetRedecompMesh().GetVertex( redecomp_face_vertex_ids[vertex_index] );
        physical_face_center[0] += 0.25 * physical_vertex[0];
        physical_face_center[1] += 0.25 * physical_vertex[1];
      }
      const tribol::RealT physical_x_offset = physical_face_center[0] - physical_origin_x;
      const tribol::RealT physical_y_offset = physical_face_center[1] - physical_origin_y;
      host_expected_parent_centers( face_id, 0 ) = ( physical_x_offset * physical_y_per_second_reference_coordinate -
                                                     physical_x_per_second_reference_coordinate * physical_y_offset ) /
                                                   coordinate_map_determinant;
      host_expected_parent_centers( face_id, 1 ) = ( physical_x_per_first_reference_coordinate * physical_y_offset -
                                                     physical_x_offset * physical_y_per_first_reference_coordinate ) /
                                                   coordinate_map_determinant;
    }
    tribol::Array2D<tribol::RealT> expected_parent_centers( host_expected_parent_centers, mesh_view.getAllocatorId() );

    // Run every mapping check in the selected execution space. A zero result
    // means the transferred child vertices, independently mapped center, and
    // rejection of an exterior child point all satisfy the contract.
    tribol::Array1D<int> face_validation_results( number_of_faces, number_of_faces, mesh_view.getAllocatorId() );
    tribol::Array1D<int> parent_region_indices( number_of_faces, number_of_faces, mesh_view.getAllocatorId() );
    tribol::Array2D<tribol::RealT> mapped_parent_centers( { number_of_faces, 2 }, mesh_view.getAllocatorId() );
    face_validation_results.fill( 0 );
    parent_region_indices.fill( -1 );
    tribol::Array1DView<int> face_validation_results_view( face_validation_results );
    tribol::Array1DView<int> parent_region_indices_view( parent_region_indices );
    tribol::Array2DView<tribol::RealT> mapped_parent_centers_view( mapped_parent_centers );
    tribol::Array2DView<tribol::RealT> expected_parent_centers_view( expected_parent_centers );
    tribol::forAllExec( execution_mode, number_of_faces, [=] TRIBOL_HOST_DEVICE( tribol::IndexT face_id ) {
      constexpr tribol::RealT comparison_tolerance = 1.e-12;
      const tribol::ParentFaceData& parent_face_data = mesh_view.getParentFaceData();
      int validation_result = 0;
      validation_result |= parent_face_data.m_parent_face_orders[face_id] != parent_order ? 1 : 0;
      validation_result |= parent_face_data.m_reference_vertex_counts[face_id] != 4 ? 2 : 0;

      tribol::RealT minimum_parent_coordinate[2] = { 1.0, 1.0 };
      tribol::RealT maximum_parent_coordinate[2] = { 0.0, 0.0 };
      for ( int vertex_index = 0; vertex_index < 4; ++vertex_index ) {
        for ( int coordinate_component = 0; coordinate_component < 2; ++coordinate_component ) {
          const int coordinate_index =
              vertex_index * tribol::ParentFaceData::max_reference_dimension + coordinate_component;
          const tribol::RealT parent_coordinate =
              parent_face_data.m_parent_reference_vertex_coordinates( face_id, coordinate_index );
          validation_result |=
              parent_coordinate < -comparison_tolerance || parent_coordinate > 1.0 + comparison_tolerance ? 4 : 0;
          minimum_parent_coordinate[coordinate_component] =
              parent_coordinate < minimum_parent_coordinate[coordinate_component]
                  ? parent_coordinate
                  : minimum_parent_coordinate[coordinate_component];
          maximum_parent_coordinate[coordinate_component] =
              parent_coordinate > maximum_parent_coordinate[coordinate_component]
                  ? parent_coordinate
                  : maximum_parent_coordinate[coordinate_component];
        }
      }

      const tribol::RealT expected_child_reference_width = 1.0 / lor_factor;
      for ( int coordinate_component = 0; coordinate_component < 2; ++coordinate_component ) {
        const tribol::RealT child_reference_width =
            maximum_parent_coordinate[coordinate_component] - minimum_parent_coordinate[coordinate_component];
        validation_result |= child_reference_width < expected_child_reference_width - comparison_tolerance ||
                                     child_reference_width > expected_child_reference_width + comparison_tolerance
                                 ? 8
                                 : 0;
      }

      const tribol::RealT lor_center[2] = { 0.5, 0.5 };
      tribol::RealT mapped_parent_center[2] = { 0.0, 0.0 };
      validation_result |= !mesh_view.mapToParentReference( face_id, lor_center, mapped_parent_center ) ? 16 : 0;
      for ( int coordinate_component = 0; coordinate_component < 2; ++coordinate_component ) {
        mapped_parent_centers_view( face_id, coordinate_component ) = mapped_parent_center[coordinate_component];
        const tribol::RealT expected_parent_coordinate = expected_parent_centers_view( face_id, coordinate_component );
        const tribol::RealT center_difference = mapped_parent_center[coordinate_component] - expected_parent_coordinate;
        validation_result |=
            center_difference < -comparison_tolerance || center_difference > comparison_tolerance ? 64 : 0;
      }

      const int first_parent_region_coordinate =
          static_cast<int>( expected_parent_centers_view( face_id, 0 ) * lor_factor );
      const int second_parent_region_coordinate =
          static_cast<int>( expected_parent_centers_view( face_id, 1 ) * lor_factor );
      const bool valid_parent_region =
          first_parent_region_coordinate >= 0 && first_parent_region_coordinate < lor_factor &&
          second_parent_region_coordinate >= 0 && second_parent_region_coordinate < lor_factor;
      validation_result |= valid_parent_region ? 0 : 128;
      if ( valid_parent_region ) {
        parent_region_indices_view[face_id] =
            second_parent_region_coordinate * lor_factor + first_parent_region_coordinate;
      }

      // Evaluate the native parent basis at the mapped point. Partition of unity
      // and reproduction of the LOR-face center prove that the device-side
      // basis representation follows MFEM's native parent-node ordering.
      tribol::RealT parent_basis_values[tribol::ParentFaceData::max_parent_face_nodes] = { 0.0 };
      validation_result |=
          !mesh_view.evaluateParentFaceBasis( face_id, mapped_parent_center, parent_basis_values ) ? 1024 : 0;
      tribol::RealT basis_value_sum = 0.0;
      const int number_of_parent_nodes = parent_face_data.m_parent_node_counts[face_id];
      for ( int parent_node = 0; parent_node < number_of_parent_nodes; ++parent_node ) {
        basis_value_sum += parent_basis_values[parent_node];
      }
      validation_result |= std::abs( basis_value_sum - 1.0 ) > comparison_tolerance ? 2048 : 0;

      tribol::RealT parent_position[3] = { 0.0, 0.0, 0.0 };
      mesh_view.evaluateParentFaceFields( face_id, parent_basis_values, parent_position, nullptr );

      // The two cube boundaries have opposite MFEM face orientations. Their
      // known affine coordinate fields provide an independent reproduction
      // check without relying on Tribol's derived face-centroid storage.
      const tribol::RealT expected_parent_position[3] = {
          mesh_id == first_mesh_id ? mapped_parent_center[0] : mapped_parent_center[1],
          mesh_id == first_mesh_id ? mapped_parent_center[1] : mapped_parent_center[0],
          mesh_id == first_mesh_id ? 1.0 : 1.0 + initial_separation };
      for ( int coordinate_component = 0; coordinate_component < mesh_view.spatialDimension();
            ++coordinate_component ) {
        validation_result |= std::abs( parent_position[coordinate_component] -
                                       expected_parent_position[coordinate_component] ) > comparison_tolerance
                                 ? 4096
                                 : 0;
      }

      const tribol::RealT exterior_lor_point[2] = { 1.25, 0.5 };
      validation_result |=
          mesh_view.mapToParentReference( face_id, exterior_lor_point, mapped_parent_center ) ? 8192 : 0;
      face_validation_results_view[face_id] = validation_result;
    } );

    tribol::ArrayT<int, 1, tribol::MemorySpace::Host> host_validation_results( face_validation_results );
    tribol::ArrayT<int, 1, tribol::MemorySpace::Host> host_parent_region_indices( parent_region_indices );
    tribol::Array2D<tribol::RealT, tribol::MemorySpace::Host> host_mapped_parent_centers( mapped_parent_centers );
    std::vector<int> local_parent_region_counts( lor_factor * lor_factor, 0 );
    for ( tribol::IndexT face_id = 0; face_id < number_of_faces; ++face_id ) {
      EXPECT_EQ( host_validation_results[face_id], 0 )
          << "parent-face mapping validation failed for face " << face_id << "; mapped center ("
          << host_mapped_parent_centers( face_id, 0 ) << ", " << host_mapped_parent_centers( face_id, 1 )
          << "), expected center (" << host_expected_parent_centers( face_id, 0 ) << ", "
          << host_expected_parent_centers( face_id, 1 ) << "), parent coordinate map (" << physical_origin_x << ", "
          << physical_origin_y << ", " << physical_x_per_first_reference_coordinate << ", "
          << physical_y_per_first_reference_coordinate << ", " << physical_x_per_second_reference_coordinate << ", "
          << physical_y_per_second_reference_coordinate << ")";
      const int parent_region_index = host_parent_region_indices[face_id];
      if ( parent_region_index >= 0 && parent_region_index < static_cast<int>( local_parent_region_counts.size() ) ) {
        ++local_parent_region_counts[parent_region_index];
      }
    }

    int local_face_count = number_of_faces;
    int global_face_count = 0;
    MPI_Allreduce( &local_face_count, &global_face_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    EXPECT_EQ( global_face_count, lor_factor * lor_factor ) << "unexpected face count for mesh " << mesh_id;

    std::vector<int> global_parent_region_counts( lor_factor * lor_factor, 0 );
    MPI_Allreduce( local_parent_region_counts.data(), global_parent_region_counts.data(),
                   static_cast<int>( global_parent_region_counts.size() ), MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    for ( int parent_region_index = 0; parent_region_index < lor_factor * lor_factor; ++parent_region_index ) {
      EXPECT_EQ( global_parent_region_counts[parent_region_index], 1 )
          << "unexpected coverage for parent region " << parent_region_index << " on mesh " << mesh_id;
    }
  }

  tribol::finalize();
  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( tribol, MfemCommonPlaneParentFaceDataTest,
                          testing::Values( std::make_tuple( 1, 1 ), std::make_tuple( 2, 2 ),
                                           std::make_tuple( 2, 3 ) ) );

//------------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

#if defined( TRIBOL_USE_CUDA )
  std::string device_str( "cuda" );
#elif defined( TRIBOL_USE_HIP )
  std::string device_str( "hip" );
#elif defined( TRIBOL_USE_OPENMP )
  std::string device_str( "omp" );
#else
  std::string device_str( "cpu" );
#endif

  mfem::Device device( device_str );
  device.Print();

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
