// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <array>
#include <cmath>
#include <set>

#include <gtest/gtest.h>

// Tribol includes
#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MfemData.hpp"

// Shared includes
#include "shared/mesh/MeshBuilder.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// MFEM includes
#include "mfem.hpp"

/**
 * @brief This tests the Tribol MFEM interface's ability to skip expensive RedecompMesh
 * recreation when nodal displacement change is below a certain threshold.
 */
class MfemRedecompSkipTest : public testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F( MfemRedecompSkipTest, test_skip_redecomp )
{
  tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Sequential;

  // 1. Initialize simplified problem: Two unit cubes separated by a small gap
  // Mesh is refined once, so element size is 0.5.
  // clang-format off
  double initial_sep = 0.1;
  mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::Unify( {
    shared::MeshBuilder::CubeMesh( 1, 1, 1 ),
    shared::MeshBuilder::CubeMesh( 1, 1, 1 )
      .translate( { 0.0, 0.0, 1.0 + initial_sep } )
      .updateAttrib( 1, 2 )
      .updateBdrAttrib( 1, 7 ) // Bottom of top cube -> 7
      .updateBdrAttrib( 6, 8 ) // set the top surface to boundary attribute 8 (prevent multiple 6)
  } ).refine( 1 ) );
  // clang-format on

  auto fe_coll = mfem::H1_FECollection( 1, mesh.SpaceDimension() );
  auto par_fe_space = mfem::ParFiniteElementSpace( &mesh, &fe_coll, mesh.SpaceDimension() );
  auto coords = mfem::ParGridFunction( &par_fe_space );
  mesh.GetNodes( coords );

  int coupling_scheme_id = 0;
  // Top of bottom cube: 6, Bottom of top cube: 7
  tribol::registerMfemCouplingScheme( coupling_scheme_id, 0, 1, mesh, coords, { 6 }, { 7 }, tribol::SURFACE_TO_SURFACE,
                                      tribol::NO_CASE, tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                      tribol::BINNING_BVH, exec_mode );

  // Set threshold explicitly to 0.2
  tribol::setMfemRedecompTriggerDisplacement( coupling_scheme_id, 0.2 );
  tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, 1.0e5, 1.0e5 );

  // 2. Initial redecomp
  tribol::updateMfemParallelDecomposition();
  auto* mfem_data = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id ).getMfemMeshData();
  const mfem::Mesh* redecomp_mesh_ptr_1 = &mfem_data->GetRedecompMesh();

  // Verify we can compute forces (should be zero as gap > 0)
  double dt = 0.1;
  tribol::update( 0, 0.0, dt );
  mfem::Vector force_1( coords.Size() );
  force_1 = 0.0;
  tribol::getMfemResponse( coupling_scheme_id, force_1 );

  // 3. Small shift (0.1 < 0.2)
  // Shift ALL nodes by 0.1. This preserves relative gap, so forces should remain same (zero).
  coords += 0.1;

  tribol::updateMfemParallelDecomposition();
  const mfem::Mesh* redecomp_mesh_ptr_2 = &mfem_data->GetRedecompMesh();

  // Expect no rebuild
  EXPECT_EQ( redecomp_mesh_ptr_1, redecomp_mesh_ptr_2 );

  tribol::update( 1, 0.1, dt );
  mfem::Vector force_2( coords.Size() );
  force_2 = 0.0;
  tribol::getMfemResponse( coupling_scheme_id, force_2 );
  // Forces should still be zero (invariant)
  EXPECT_LT( force_2.Norml2(), 1.0e-10 );

  // 4. Large shift (accumulated 0.3 > 0.2)
  // Shift ALL nodes by another 0.2 (total 0.3)
  coords += 0.2;

  tribol::updateMfemParallelDecomposition();
  const mfem::Mesh* redecomp_mesh_ptr_3 = &mfem_data->GetRedecompMesh();

  // Expect rebuild
  EXPECT_NE( redecomp_mesh_ptr_1, redecomp_mesh_ptr_3 );

  tribol::update( 2, 0.2, dt );
  mfem::Vector force_3( coords.Size() );
  force_3 = 0.0;
  tribol::getMfemResponse( coupling_scheme_id, force_3 );
  // Forces should still be zero (invariant)
  EXPECT_LT( force_3.Norml2(), 1.0e-10 );

  tribol::finalize();
}

TEST_F( MfemRedecompSkipTest, element_thickness_reference_coords )
{
  constexpr tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Sequential;
  constexpr int z_component = 2;
  constexpr tribol::RealT top_cube_offset = 1.1;
  constexpr tribol::RealT current_z_scale = 2.0;
  constexpr tribol::RealT material_modulus = 1.0;
  constexpr tribol::RealT tol = 1.0e-12;
  constexpr int coupling_scheme_id = 0;
  constexpr int mesh1_id = 0;
  constexpr int mesh2_id = 1;

  // Check that element thickness is calculated from reference coordinates when they are registered, and from current
  // coordinates otherwise.
  const auto get_thickness = []( const char* scenario, bool deform_current_coords, bool use_reference_coords ) {
    SCOPED_TRACE( scenario );

    // clang-format off
    mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::Unify( {
      shared::MeshBuilder::CubeMesh( 1, 1, 1 ),
      shared::MeshBuilder::CubeMesh( 1, 1, 1 )
        .translate( { 0.0, 0.0, top_cube_offset } )
        .updateAttrib( 1, 2 )
        .updateBdrAttrib( 1, 7 )
        .updateBdrAttrib( 6, 8 )
    } ) );
    // clang-format on
    // This is a pair of unit cubes separated in z. The top face of the lower cube keeps boundary attribute 6, while the
    // bottom face of the upper cube is changed to attribute 7 so those two surfaces can be registered as the contact
    // surfaces. The upper cube volume attribute is changed to 2, and its top boundary is changed to 8, to keep those
    // attributes distinct from the lower cube.

    auto fe_coll = mfem::H1_FECollection( 1, mesh.SpaceDimension() );
    auto par_fe_space = mfem::ParFiniteElementSpace( &mesh, &fe_coll, mesh.SpaceDimension() );
    auto coords = mfem::ParGridFunction( &par_fe_space );
    mesh.GetNodes( coords );
    auto ref_coords = mfem::ParGridFunction( coords );

    mfem::VectorFunctionCoefficient original_coords_coeff( mesh.SpaceDimension(),
                                                           []( const mfem::Vector& x, mfem::Vector& y ) {
                                                             y.SetSize( x.Size() );
                                                             y = x;
                                                           } );
    EXPECT_NEAR( ref_coords.ComputeL2Error( original_coords_coeff ), 0.0, tol );

    mfem::VectorFunctionCoefficient current_coords_coeff( mesh.SpaceDimension(),
                                                          []( const mfem::Vector& x, mfem::Vector& y ) {
                                                            y.SetSize( x.Size() );
                                                            y = x;
                                                            y[z_component] = current_z_scale * x[z_component];
                                                          } );
    if ( deform_current_coords ) {
      coords.ProjectCoefficient( current_coords_coeff );
      EXPECT_NEAR( coords.ComputeL2Error( current_coords_coeff ), 0.0, tol );
    } else {
      EXPECT_NEAR( coords.ComputeL2Error( original_coords_coeff ), 0.0, tol );
    }

    tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, { 6 }, { 7 },
                                        tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                        tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_BVH, exec_mode );
    if ( use_reference_coords ) {
      tribol::registerMfemReferenceCoords( coupling_scheme_id, ref_coords );
    }

    mfem::Vector bulk_moduli_by_bdry_attrib( mesh.bdr_attributes.Max() );
    bulk_moduli_by_bdry_attrib = material_modulus;
    mfem::PWConstCoefficient mat_coeff( bulk_moduli_by_bdry_attrib );
    tribol::setMfemKinematicElementPenalty( coupling_scheme_id, mat_coeff );
    tribol::updateMfemParallelDecomposition( 0, true );

    auto* mfem_data = tribol::CouplingSchemeManager::getInstance().at( coupling_scheme_id ).getMfemMeshData();
    if ( mfem_data == nullptr ) {
      ADD_FAILURE() << "MFEM mesh data was not created.";
      tribol::finalize();
      return std::array<tribol::RealT, 2>{ 0.0, 0.0 };
    }
    EXPECT_GT( mfem_data->GetMesh1NE(), 0 );
    EXPECT_GT( mfem_data->GetMesh2NE(), 0 );
    const std::array<tribol::RealT, 2> thickness = { mfem_data->GetRedecompElemThickness1()[0],
                                                     mfem_data->GetRedecompElemThickness2()[0] };

    tribol::finalize();
    return thickness;
  };

  constexpr bool use_original_current_coords = false;
  constexpr bool use_deformed_current_coords = true;
  constexpr bool no_reference_coords = false;
  constexpr bool has_reference_coords = true;

  const auto original = get_thickness( "original", use_original_current_coords, no_reference_coords );
  const auto current = get_thickness( "current", use_deformed_current_coords, no_reference_coords );
  const auto reference = get_thickness( "reference", use_deformed_current_coords, has_reference_coords );

  EXPECT_NEAR( reference[0], original[0], tol );
  EXPECT_NEAR( reference[1], original[1], tol );
  EXPECT_TRUE( std::abs( current[0] - original[0] ) > tol || std::abs( current[1] - original[1] ) > tol );
}

int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
