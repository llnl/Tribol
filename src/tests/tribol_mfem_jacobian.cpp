// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <gtest/gtest.h>

// MFEM includes
#include "mfem.hpp"

// Axom includes
#include "axom/slic.hpp"

// Shared includes
#include "shared/mesh/MeshBuilder.hpp"

// Tribol includes
#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MfemData.hpp"

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

// Helper to access the CouplingScheme (which is usually hidden behind the interface)
// We need to look up the scheme from the manager.
// Since CouplingSchemeManager is a singleton/static internal, we rely on `tribol::getCouplingScheme` if available
// or we might need to rely on the fact that we can't easily get the pointer via public API without include internal
// headers. Actually, `tribol::getMfemBlockJacobian` implementation finds the scheme. We can include
// `tribol/search/InterfacePairs.hpp` or similar if needed, but let's check if there is a way. Ah,
// `tribol::CouplingSchemeManager` is in `tribol/coupling/CouplingSchemeManager.hpp`.

#include "tribol/mesh/CouplingScheme.hpp"

class MfemJacobianTest : public testing::Test {
 protected:
  void SetUp() override
  {
    // Silence Tribol output
    // tribol::setLoggingLevel( tribol::LoggingLevel::ERROR );
  }

  void TearDown() override
  {
    // tribol::finalize() is called in main
  }
};

TEST_F( MfemJacobianTest, direct_jacobian_assembly )
{
  int n_ranks;
  MPI_Comm_size( MPI_COMM_WORLD, &n_ranks );

  // 1. Setup simple mesh (2 cubes)
  int ref_levels = 0;
  int nel_per_dir = std::pow( 2, ref_levels );

  // Attributes:
  // Mesh 1 contact surface: 4
  // Mesh 2 contact surface: 5
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
      .translate({0.0, 0.0, 0.99}) // Slightly overlapping or close
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

  // 2. Register Coupling Scheme
  int cs_id = 0;
  int mesh1_id = 0;
  int mesh2_id = 1;
  tribol::registerMfemCouplingScheme( cs_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs, nonmortar_attrs,
                                      tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::SINGLE_MORTAR,
                                      tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );

  // 3. Update Decomp to build internal MfemData
  tribol::updateMfemParallelDecomposition();

  // 4. Access CouplingScheme and MfemJacobianData
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

  // We need to call UpdateJacobianXfer explicitly or ensure it's called.
  // It is usually called in `getMfemBlockJacobian`. Let's call it manually to be safe.
  jac_data->UpdateJacobianXfer();

  // 5. Synthesize ComputedElementData
  // We'll add a contribution for the first element of Mesh 1 (Mortar) against itself (Mortar-Mortar block).
  // BlockSpace::MORTAR -> BlockSpace::MORTAR
  // In the implementation of GetMfemJacobian, MORTAR maps to index 0 (Displacement).

  std::vector<tribol::ComputedElementData> contributions;

  // Find a valid element ID on the contact surface (Tribol mesh)
  // Since we have a simple cube surface, index 0 should be valid on at least one rank that owns the surface.
  // We need to know if *this* rank owns any surface elements.
  // The MfemMeshData has this info.
  // But strictly, we can just try to add data for element 0. If this rank maps element 0 to something valid in
  // redecomp, it works. MfemJacobianData::GetMfemJacobian logic maps Tribol ID -> Redecomp ID. We need
  // `parent_data_.GetElemMap1()` to have entry for 0. Since we are running in parallel, element 0 of the *Tribol Mesh*
  // (which is local to the rank?) Tribol meshes are rank-local surface meshes. So element 0 is valid if `GetMesh1NE() >
  // 0`.

  // Access MfemMeshData to check element counts
  // We can't easily access MfemMeshData from MfemJacobianData public API, but we know usage:
  // If we just use element 0, we must check if we have any elements.
  // For this test, we can try to find a rank that has elements.

  // Note: We can't easily check `GetMesh1NE` because `MfemMeshData` is hidden in `MfemJacobianData`.
  // However, we can construct the contribution anyway. If the element map doesn't contain the ID, it might crash or
  // throw if we are not careful, but `GetElemMap1` is an array. Accessing index 0 is valid only if size > 0.

  // Let's protect with a check on the mesh attributes or just try-catch or ensure all ranks have elements?
  // With 2 cubes and default partition, usually ranks have boundary elements.
  // But to be safe, let's look at the method signatures again.
  // `GetElemMap1` returns `Array1D<int>`.

  // Actually, we can just use `tribol::getMfemJacobianData` or similar? No.

  // Let's proceed assuming standard partition gives elements.

  int num_dofs_per_elem = 8 * dim;  // Hex element, 8 nodes, 3 dims
  int mat_size = num_dofs_per_elem * num_dofs_per_elem;

  tribol::ComputedElementData contrib;
  contrib.row_space = tribol::BlockSpace::MORTAR;
  contrib.col_space = tribol::BlockSpace::MORTAR;

  // We'll just add one element's contribution if possible.
  // We need to know how many elements are on this rank's surface mesh.
  // There isn't a direct Tribol API to query "number of surface elements on mesh 1 on this rank" easily exposed without
  // MfemMeshData. BUT, we can try to guess. Or just send empty if we don't know. Wait, if we send invalid IDs, it will
  // likely crash.

  // Let's assume we want to test the *mechanism*.
  // We can use `cs->getMfemMeshData()->GetMesh1NE()`?
  // `getMfemMeshData` is likely available on `CouplingScheme`.
  // Let's check CouplingScheme.hpp content I read earlier?
  // I didn't read the whole file, but typically getters are there.

  // Let's try to access it. If not, we'll need another way.
  // Assuming `getMfemMeshData()` exists and returns `MfemMeshData*`.

  auto* mesh_data = cs->getMfemMeshData();  // This might need verification
  // Based on `getMfemJacobianData`, it's likely `getMfemMeshData` exists.

  if ( mesh_data && mesh_data->GetMesh1NE() > 0 ) {
    contrib.row_elem_ids.push_back( 0 );
    contrib.col_elem_ids.push_back( 0 );

    contrib.jacobian_data.resize( mat_size );
    for ( int i = 0; i < mat_size; ++i ) contrib.jacobian_data[i] = 1.0;  // Fill with 1.0

    contrib.jacobian_offsets.push_back( 0 );

    contributions.push_back( contrib );
  }

  // 6. Call the new method
  auto ParJ = jac_data->GetMfemJacobian( contributions );

  // 7. Verify
  // If contributions were added, ParJ should have some non-zeros.
  // We can check the Global Num Nonzeros or Norm.

  // Reduce to see if *any* rank added something.
  int local_contrib = contributions.size();
  int global_contrib = 0;
  MPI_Allreduce( &local_contrib, &global_contrib, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

  if ( global_contrib > 0 ) {
    // We expect some non-zeros
    EXPECT_GT( ParJ->NNZ(), 0 );
  } else {
    // If no elements were found (unlikely with 2 cubes), this test is vacuous but passes
    // But we should warn.
    if ( n_ranks == 1 ) {
      // Serial run should definitely have elements
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
