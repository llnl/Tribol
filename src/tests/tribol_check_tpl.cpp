// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Axom includes
#include "axom/config.hpp"
#include "axom/core.hpp"
#include "axom/slic.hpp"

// C/C++ includes
#include <string>
#include <sstream>

// gtest includes
#include "gtest/gtest.h"

//------------------------------------------------------------------------------
// UNIT TESTS
//------------------------------------------------------------------------------
TEST( tribol_check_tpl, check_axom )
{
  EXPECT_TRUE( AXOM_VERSION_MAJOR >= 0 );
  EXPECT_TRUE( AXOM_VERSION_MINOR >= 0 );
  EXPECT_TRUE( AXOM_VERSION_PATCH >= 0 );

  std::string axom_version = AXOM_VERSION_FULL;
  EXPECT_TRUE( !axom_version.empty() );

  std::ostringstream oss;
  oss << "v" << AXOM_VERSION_MAJOR << "." << AXOM_VERSION_MINOR << "." << AXOM_VERSION_PATCH;

  EXPECT_EQ( axom_version, oss.str() );
}

TEST( tribol_check_tpl, print_axom_about ) { axom::about(); }

//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}
