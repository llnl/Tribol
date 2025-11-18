// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

//-----------------------------------------------------------------------------
//
// file: caliper_smoke.cpp
//
//-----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include "caliper/cali-manager.h"
#include "caliper/cali.h"

void testFunction()
{
  CALI_CXX_MARK_FUNCTION;
  std::cout << "Writing output..." << std::endl;
}

TEST( caliper_smoke, basic_use ) { testFunction(); }

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );

  cali::ConfigManager mgr;
  mgr.add( "runtime-report,spot" );
  mgr.start();

  auto result = RUN_ALL_TESTS();

  mgr.stop();
  mgr.flush();

  return result;
}
