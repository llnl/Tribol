// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check average and series penalty combinations on every enabled execution backend, without MFEM or Enzyme.
 * Equal, unequal, exchanged, and scaled input pairs distinguish the two formulas. MFEM formulation tests check that
 * each contact method can select the other combination while preserving its geometry and derivative calculation.
 */
#include <limits>
#include <gtest/gtest.h>
#include "axom/slic/core/SimpleLogger.hpp"
#include "ExecutionTest.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/physics/PenaltyStiffness.hpp"

namespace {
using namespace tribol;

/** @brief Run stiffness calculations in memory accessible to each enabled execution backend. */
class PenaltyStiffnessTest : public tribol::detail::ExecutionTest {};

// Unequal inputs distinguish averaging from springs in series; swapping and scaling must preserve each formula.
TEST_P( PenaltyStiffnessTest, combines_supplied_stiffnesses )
{
  struct Case {
    RealT stiffness1;        ///< First supplied stiffness.
    RealT stiffness2;        ///< Second supplied stiffness.
    RealT expected_average;  ///< Arithmetic mean, supplied independently of the implementation.
    RealT expected_series;   ///< Equivalent stiffness of the two springs in series.
  };
  struct Result {
    RealT average;  ///< Calculated arithmetic mean.
    RealT series;   ///< Calculated stiffness of springs in series.
  };
  const auto cases = copyToExecutionSpace<Case>( { { 4.0, 4.0, 4.0, 2.0 },
                                                   { 2.0, 6.0, 4.0, 1.5 },
                                                   { 6.0, 2.0, 4.0, 1.5 },
                                                   { 4.0, 12.0, 8.0, 3.0 },
                                                   { 1.0, 99.0, 50.0, 0.99 } } );
  ArrayT<Result> results( cases.size(), cases.size(), allocatorId() );
  const auto input = cases.view();
  const auto output = results.view();
  forAllExec( GetParam(), cases.size(), [=] TRIBOL_HOST_DEVICE( IndexT i ) {
    output[i] = { AveragePenaltyStiffness{}( input[i].stiffness1, input[i].stiffness2 ),
                  SeriesPenaltyStiffness{}( input[i].stiffness1, input[i].stiffness2 ) };
  } );
  const ArrayT<Case, 1, MemorySpace::Host> host_cases( cases );
  const ArrayT<Result, 1, MemorySpace::Host> host( results );
  constexpr RealT tolerance = std::numeric_limits<RealT>::epsilon();
  for ( IndexT i = 0; i < host.size(); ++i ) {
    SCOPED_TRACE( i );
    EXPECT_NEAR( host[i].average, host_cases[i].expected_average, tolerance );
    EXPECT_NEAR( host[i].series, host_cases[i].expected_series, tolerance );
  }
}
INSTANTIATE_TEST_SUITE_P( execution, PenaltyStiffnessTest, tribol::detail::enabledExecutionModes() );
}  // namespace

/** @brief Run portable penalty tests. @param argc Argument count. @param argv Arguments. @return Test status. */
int main( int argc, char** argv )
{
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
