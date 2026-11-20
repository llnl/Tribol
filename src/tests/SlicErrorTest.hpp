// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Provide a Google Test assertion for expected SLIC errors without using death tests.
 */
#ifndef SRC_TESTS_SLICERRORTEST_HPP_
#define SRC_TESTS_SLICERRORTEST_HPP_

#include <gtest/gtest.h>

#include "axom/slic/interface/slic.hpp"

/**
 * @brief Expect a statement to report a SLIC error.
 *
 * While the statement runs, axom::slic::ScopedAbortToThrow turns the SLIC abort into an
 * axom::slic::SlicAbortException, so the test continues after the expected error. The previous abort behavior is
 * restored afterwards. Wrap statements containing unparenthesized commas, such as template arguments, in parentheses.
 * SLIC must be initialized, for example by an axom::slic::SimpleLogger in main().
 *
 * @param statement Statement expected to report a SLIC error.
 */
#define EXPECT_SLIC_ERROR( statement )                         \
  do {                                                         \
    axom::slic::ScopedAbortToThrow tribol_slic_abort_to_throw; \
    EXPECT_THROW( statement, axom::slic::SlicAbortException ); \
  } while ( false )

#endif  // SRC_TESTS_SLICERRORTEST_HPP_
