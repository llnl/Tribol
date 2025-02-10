// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/interface/tribol.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// Axom includes
#include "axom/slic.hpp"

// gtest includes
#include "gtest/gtest.h"

// C/C++ includes
#include <vector>

/*!
 * Test fixture class with some setup necessary to use the
 * CouplingSchemeManager class
 */
class CouplingSchemeManagerTest : public ::testing::Test {
 public:
  // Simple function to generate a functioning coupling scheme
  tribol::CouplingScheme generateCouplingScheme()
  {
    // Note: We are not testing anything about coupling schemes in this file.
    //       Contact parameters don't matter for this setup either.
    return tribol::CouplingScheme( 0, mesh_id[0], mesh_id[1], 0, 0, 0, 0, 0, tribol::DEFAULT_BINNING_METHOD );
  }

 protected:
  void SetUp() override
  {
    // Register two "dummy" meshes to be used by coupling scheme

    mesh_id[0] = 0;
    tribol::registerMesh( mesh_id[0], 1, 4, &connectivity[0], 3, &x[0], &y[0], &z[0], tribol::MemorySpace::Host );

    mesh_id[1] = 1;
    tribol::registerMesh( mesh_id[1], 1, 4, &connectivity[0], 3, &x[0], &y[0], &z[0], tribol::MemorySpace::Host );
  }

  void TearDown() override
  {
    tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();

    csManager.clear();

    EXPECT_EQ( 0, csManager.size() );
  }

 protected:
  tribol::IndexT mesh_id[2];

  tribol::RealT x[4]{ 0., 1., 1., 0. };
  tribol::RealT y[4]{ 0., 0., 1., 1. };
  tribol::RealT z[4]{ 0., 0., 0., 0. };

  tribol::IndexT connectivity[4]{ 0, 1, 2, 3 };
};

TEST_F( CouplingSchemeManagerTest, initially_empty )
{
  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();

  EXPECT_EQ( 0, csManager.size() );
}

TEST_F( CouplingSchemeManagerTest, add_remove_couplings )
{
  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();

  EXPECT_EQ( 0, csManager.size() );

  // Adds a first coupling
  {
    constexpr tribol::IndexT cs_id = 0;
    csManager.addData( cs_id, this->generateCouplingScheme() );
    constexpr int expectedNumCouplings = 1;
    EXPECT_EQ( expectedNumCouplings, csManager.size() );

    EXPECT_NE( nullptr, csManager.findData( cs_id ) );
  }

  // Adds a second coupling
  {
    constexpr tribol::IndexT cs_id = 1;
    csManager.addData( cs_id, this->generateCouplingScheme() );
    const int expectedNumCouplings = 2;
    EXPECT_EQ( expectedNumCouplings, csManager.size() );

    EXPECT_NE( nullptr, csManager.findData( cs_id ) );
  }
}

TEST_F( CouplingSchemeManagerTest, add_couplings_at_index )
{
  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();

  EXPECT_EQ( 0, csManager.size() );

  // Add some couplings
  csManager.addData( 0, this->generateCouplingScheme() );
  csManager.addData( 1, this->generateCouplingScheme() );
  EXPECT_EQ( 2, csManager.size() );

  // Adds a coupling at index 4 -- note, we skipped indices 2 and 3
  {
    constexpr tribol::IndexT cs_id = 4;
    csManager.addData( cs_id, this->generateCouplingScheme() );

    constexpr size_t expectedNumCouplings = 3;
    EXPECT_EQ( expectedNumCouplings, csManager.size() );

    EXPECT_NE( nullptr, csManager.findData( cs_id ) );
  }

  // After growing the manager's array, there are some nullptrs slots
  {
    // The existing slots should still be there
    for ( auto i : std::vector<int>{ 0, 1, 4 } ) {
      EXPECT_NE( nullptr, csManager.findData( i ) );
    }

    // The extra slots should be filled with zeros
    for ( auto i : std::vector<int>{ 2, 3 } ) {
      EXPECT_EQ( nullptr, csManager.findData( i ) );
    }
  }

  // Remove coupling at index 1
  {
    constexpr tribol::IndexT cs_id = 1;
    csManager.erase( cs_id );

    EXPECT_EQ( nullptr, csManager.findData( cs_id ) );

    // num couplings is now 2
    constexpr size_t expectedNumCouplings = 2;
    EXPECT_EQ( expectedNumCouplings, csManager.size() );
  }

  // Replace coupling at index 4
  {
    constexpr tribol::IndexT cs_id = 4;
    csManager.addData( cs_id, this->generateCouplingScheme() );

    constexpr size_t expectedNumCouplings = 2;
    EXPECT_EQ( expectedNumCouplings, csManager.size() );

    EXPECT_NE( nullptr, csManager.findData( cs_id ) );
  }
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
