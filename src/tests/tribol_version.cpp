// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/config.hpp"

// gtest includes
#include "gtest/gtest.h"

TEST( VersionTest, ensure_nonempty_string ) { EXPECT_EQ( std::string( TRIBOL_VERSION_FULL ).empty(), false ); }

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

  result = RUN_ALL_TESTS();

  return result;
}
