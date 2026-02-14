// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/config.hpp"

#include <gtest/gtest.h>

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

#include "mfem.hpp"

#include "axom/slic.hpp"

#include "shared/mesh/MeshBuilder.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MfemData.hpp"

class MfemJacobianTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F( MfemJacobianTest, direct_jacobian_assembly )
{
  int n_ranks;
  MPI_Comm_size( MPI_COMM_WORLD, &n_ranks );

  // Setup simple two-cube mesh
  int ref_levels = 0;
  int nel_per_dir = std::pow( 2, ref_levels );

  auto mortar_attrs = std::set<int>( { 4 } );
  auto nonmortar_attrs = std::set<int>( { 5 } );

  // clang-format off
  mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(1, 3)
      .updateBdrAttrib(4, 7) // Mortar
      .updateBdrAttrib(5, 1)
      .updateBdrAttrib(6, 4),
    shared::MeshBuilder::CubeMesh(nel_per_dir, nel_per_dir, nel_per_dir)
      .translate({0.0, 0.0, 0.99}) // Slight overlap
      .updateBdrAttrib(1, 8)
      .updateBdrAttrib(3, 7)
      .updateBdrAttrib(4, 7)
      .updateBdrAttrib(5, 1) // Nonmortar
      .updateBdrAttrib(8, 5)
  }));
  // clang-format on

  int dim = mesh.SpaceDimension();
  int order = 1;
  mfem::H1_FECollection fe_coll( order, dim );
  mfem::ParFiniteElementSpace par_fe_space( &mesh, &fe_coll, dim );
  mfem::ParGridFunction coords( &par_fe_space );
  mesh.GetNodes( coords );

  // Register coupling scheme
  int cs_id = 0;
  int mesh1_id = 0;
  int mesh2_id = 1;
  tribol::registerMfemCouplingScheme( cs_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs, nonmortar_attrs,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::SINGLE_MORTAR,
                                      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );

  // Build internal MfemData
  tribol::updateMfemParallelDecomposition();

  // Access CouplingScheme and MfemJacobianData
  auto& cs_manager = tribol::CouplingSchemeManager::getInstance();
  auto* cs = cs_manager.findData( cs_id );
  ASSERT_NE( cs, nullptr );

  auto* jac_data = cs->getMfemJacobianData();
  if ( jac_data == nullptr ) {
    auto* mesh_data = cs->getMfemMeshData();
    auto* submesh_data = cs->getMfemSubmeshData();
    ASSERT_NE( mesh_data, nullptr );
    ASSERT_NE( submesh_data, nullptr );

    cs->setMfemJacobianData(
        std::make_unique<tribol::MfemJacobianData>( *mesh_data, *submesh_data, cs->getContactMethod() ) );
    jac_data = cs->getMfemJacobianData();
  }
  ASSERT_NE( jac_data, nullptr );

  jac_data->UpdateJacobianXfer();

  // Synthesize ComputedElementData for mortar-mortar block
  std::vector<tribol::ComputedElementData> contributions;

  int num_dofs_per_elem = 8 * dim;  // Hex element, 8 nodes
  int mat_size = num_dofs_per_elem * num_dofs_per_elem;

  tribol::ComputedElementData contrib;
  contrib.row_space = tribol::BlockSpace::MORTAR;
  contrib.col_space = tribol::BlockSpace::MORTAR;

  auto* mesh_data = cs->getMfemMeshData();

  if ( mesh_data && mesh_data->GetMesh1NE() > 0 ) {
    contrib.row_elem_ids.push_back( 0 );
    contrib.col_elem_ids.push_back( 0 );

    contrib.jacobian_data.resize( mat_size );
    for ( int i = 0; i < mat_size; ++i ) contrib.jacobian_data[i] = 1.0;

    contrib.jacobian_offsets.push_back( 0 );

    contributions.push_back( contrib );
  }

  // Assemble Jacobian
  auto ParJ = jac_data->GetMfemJacobian( contributions );

  // Verify non-zeros if contributions were added
  int local_contrib = contributions.size();
  int global_contrib = 0;
  MPI_Allreduce( &local_contrib, &global_contrib, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

  if ( global_contrib > 0 ) {
    EXPECT_GT( ParJ->NNZ(), 0 );
  } else {
    if ( n_ranks == 1 ) {
      FAIL() << "No surface elements found on mesh 1 in serial run.";
    }
  }
}

int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();
#endif

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  tribol::finalize();
  MPI_Finalize();

  return result;
}
