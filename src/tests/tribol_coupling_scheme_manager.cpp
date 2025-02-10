// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/types.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/CouplingSchemeManager.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/interface/tribol.hpp"

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
  tribol::CouplingScheme* generateCouplingScheme()
  {
    // Note: We are not testing anything about coupling schemes in this file.
    //       Contact parameters don't matter for this setup either.
    return new tribol::CouplingScheme( 0, meshId[0], meshId[1], 0, 0, 0, 0, 0, tribol::DEFAULT_BINNING_METHOD );
  }

 protected:
  void SetUp() override
  {
    // Register two "dummy" meshes to be used by coupling scheme

    meshId[0] = 0;
    tribol::registerMesh( meshId[0], 1, 4, &connectivity[0], 3, &x[0], &y[0], &z[0] );

    meshId[1] = 1;
    tribol::registerMesh( meshId[1], 1, 4, &connectivity[0], 3, &x[0], &y[0], &z[0] );
  }

  void TearDown() override
  {
    tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();

    csManager.clearAllCouplings();

    EXPECT_EQ( 0, csManager.getNumberOfCouplings() );
  }

 protected:
  int meshId[2];

  tribol::real x[4]{ 0., 1., 1., 0. };
  tribol::real y[4]{ 0., 0., 1., 1. };
  tribol::real z[4]{ 0., 0., 0., 0. };

  tribol::IndexType connectivity[4]{ 0, 1, 2, 3 };
};

TEST_F( CouplingSchemeManagerTest, initially_empty )
{
  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();

  EXPECT_EQ( 0, csManager.getNumberOfCouplings() );
}

TEST_F( CouplingSchemeManagerTest, add_remove_couplings )
{
  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();

  EXPECT_EQ( 0, csManager.getNumberOfCouplings() );

  // Adds a first coupling
  {
    int index = csManager.addCoupling( this->generateCouplingScheme() );
    const int expectedNumCouplings = 1;
    EXPECT_EQ( expectedNumCouplings, csManager.getNumberOfCouplings() );

    const int expectedIndex = 0;
    EXPECT_EQ( expectedIndex, index );

    EXPECT_TRUE( csManager.hasCoupling( index ) );
    EXPECT_NE( nullptr, csManager.getCoupling( index ) );
  }

  // Adds a second coupling
  {
    int index = csManager.addCoupling( this->generateCouplingScheme() );
    const int expectedNumCouplings = 2;
    EXPECT_EQ( expectedNumCouplings, csManager.getNumberOfCouplings() );

    const int expectedIndex = 1;
    EXPECT_EQ( expectedIndex, index );

    EXPECT_TRUE( csManager.hasCoupling( index ) );
    EXPECT_NE( nullptr, csManager.getCoupling( index ) );
  }
}

TEST_F( CouplingSchemeManagerTest, add_couplings_at_index )
{
  tribol::CouplingSchemeManager& csManager = tribol::CouplingSchemeManager::getInstance();

  EXPECT_EQ( 0, csManager.getNumberOfCouplings() );

  // Add some couplings
  csManager.addCoupling( this->generateCouplingScheme() );
  csManager.addCoupling( this->generateCouplingScheme() );
  EXPECT_EQ( 2, csManager.getNumberOfCouplings() );

  // Adds a coupling at index 4 -- note, we skipped indices 2 and 3
  {
    const int csIndex = 4;
    int index = csManager.addCoupling( csIndex, this->generateCouplingScheme() );

    const int expectedNumCouplings = 5;
    EXPECT_EQ( expectedNumCouplings, csManager.getNumberOfCouplings() );

    const int expectedIndex = csIndex;
    EXPECT_EQ( expectedIndex, index );

    EXPECT_TRUE( csManager.hasCoupling( index ) );
    EXPECT_NE( nullptr, csManager.getCoupling( index ) );
  }

  // After growing the manager's array, there are some nullptrs slots
  {
    // The existing slots should still be there
    for ( auto i : std::vector<int>{ 0, 1, 4 } ) {
      EXPECT_TRUE( csManager.hasCoupling( i ) );
    }

    // The extra slots should be filled with zeros
    for ( auto i : std::vector<int>{ 2, 3 } ) {
      EXPECT_FALSE( csManager.hasCoupling( i ) );
      EXPECT_EQ( nullptr, csManager.getCoupling( i ) );
    }
  }

  // Remove coupling at index 1
  {
    const int csIndex = 1;
    csManager.removeCoupling( csIndex );

    EXPECT_FALSE( csManager.hasCoupling( csIndex ) );

    // num couplings is still 5 since index 4 was filled
    const int expectedNumCouplings = 5;
    EXPECT_EQ( expectedNumCouplings, csManager.getNumberOfCouplings() );
  }

  // Replace coupling at index 4
  {
    const int csIndex = 4;
    int index = csManager.addCoupling( csIndex, this->generateCouplingScheme() );

    const int expectedNumCouplings = 5;
    EXPECT_EQ( expectedNumCouplings, csManager.getNumberOfCouplings() );

    const int expectedIndex = csIndex;
    EXPECT_EQ( expectedIndex, index );

    EXPECT_TRUE( csManager.hasCoupling( index ) );
  }
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
