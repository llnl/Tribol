// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <type_traits>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/common/MdArray.hpp"

namespace {

TEST( MdArray, owns_fixed_extent_row_major_data )
{
  // MdArray must own only its fixed-size elements while exposing the same compile-time shape and row-major layout as
  // MdSpan.
  using Array = tribol::MdArray<double, 2, 3>;
  using Row = tribol::MdArray<double, 3>;
  using RowView = tribol::MdSpan<const double, 3>;
  static_assert( sizeof( Array ) == 6 * sizeof( double ) );
  static_assert( std::is_trivially_copyable_v<Array> );
  static_assert( Array::rank() == 2 );
  static_assert( Array::size() == 6 );
  static_assert( Array::extent<0>() == 2 );
  static_assert( Array::extent<1>() == 3 );
  // Constructor constraints make incomplete or oversized rows fail at compile time instead of producing malformed
  // multidimensional storage.
  static_assert( std::is_constructible_v<Row, double, double, double> );
  static_assert( !std::is_constructible_v<Row, double, double> );
  static_assert( !std::is_constructible_v<Row, double, double, double, double> );
  static_assert( std::is_constructible_v<Array, Row, Row> );
  static_assert( std::is_constructible_v<Array, RowView, RowView> );
  static_assert( !std::is_constructible_v<Array, Row> );
  static_assert( !std::is_constructible_v<Array, Row, Row, Row> );
  static_assert( !std::is_constructible_v<Array, tribol::MdArray<double, 2>, tribol::MdArray<double, 2>> );
  static_assert( !std::is_constructible_v<Array, tribol::MdSpan<const double, 2>, tribol::MdSpan<const double, 2>> );

  // Row objects and row views must both initialize the same contiguous row-major representation.
  Array values{ Row{ 0.0, 1.0, 2.0 }, Row{ 3.0, 4.0, 5.0 } };
  const double first_row[3] = { 6.0, 7.0, 8.0 };
  const double second_row[3] = { 9.0, 10.0, 11.0 };
  const Array values_from_views{ RowView{ first_row }, RowView{ second_row } };

  static_assert( std::is_same_v<decltype( values.view() ), tribol::MdSpan<double, 2, 3>> );
  EXPECT_EQ( values( 1, 2 ), 5.0 );
  EXPECT_EQ( values_from_views( 0, 2 ), 8.0 );
  EXPECT_EQ( values_from_views( 1, 0 ), 9.0 );

  // Const ownership must yield a read-only view so callers cannot discard constness through view conversion.
  const Array& read_only_values = values;
  static_assert( std::is_same_v<decltype( read_only_values.view() ), tribol::MdSpan<const double, 2, 3>> );
  EXPECT_EQ( read_only_values.view()( 1, 0 ), 3.0 );

  // MdArray has value semantics, so modifying a copy must not alias the original stack storage.
  auto copied_values = values;
  copied_values( 0, 1 ) = 8.0;
  EXPECT_EQ( values( 0, 1 ), 1.0 );
  EXPECT_EQ( copied_values( 0, 1 ), 8.0 );
}

// Rank-one arrays must support scalar multiplication on either side and inner products with arrays or views.
TEST( MdArray, supports_rank_one_scalar_multiply_and_dot )
{
  const tribol::MdArray<double, 3> a{ 1.5, -2.0, 4.0 };
  const auto right_scaled = a * 2.0;
  const auto left_scaled = -3.0 * a;
  EXPECT_DOUBLE_EQ( right_scaled[0], 3.0 );
  EXPECT_DOUBLE_EQ( right_scaled[1], -4.0 );
  EXPECT_DOUBLE_EQ( right_scaled[2], 8.0 );
  EXPECT_DOUBLE_EQ( left_scaled[0], -4.5 );
  EXPECT_DOUBLE_EQ( left_scaled[1], 6.0 );
  EXPECT_DOUBLE_EQ( left_scaled[2], -12.0 );

  const tribol::MdArray<double, 3> b{ 2.0, 0.5, -1.0 };
  EXPECT_DOUBLE_EQ( tribol::dot( a, b ), -2.0 );
  EXPECT_DOUBLE_EQ( tribol::dot( a.view(), b ), -2.0 );
  EXPECT_DOUBLE_EQ( tribol::dot( a, b.view() ), -2.0 );
  EXPECT_DOUBLE_EQ( tribol::dot( a.view(), b.view() ), -2.0 );

  const auto clamped = tribol::min( a, 0.0 );
  EXPECT_DOUBLE_EQ( clamped[0], 0.0 );
  EXPECT_DOUBLE_EQ( clamped[1], -2.0 );
  EXPECT_DOUBLE_EQ( clamped[2], 0.0 );

  const tribol::MdArray<double, 3> areas{ 0.5, 0.0, 2.0 };
  const auto inv_areas = 1.0 / areas;
  EXPECT_DOUBLE_EQ( inv_areas[0], 2.0 );
  EXPECT_DOUBLE_EQ( inv_areas[1], 0.0 );
  EXPECT_DOUBLE_EQ( inv_areas[2], 0.5 );
  EXPECT_DOUBLE_EQ( tribol::sum( clamped, clamped, inv_areas ), 0.0 );

  const tribol::MdArray<double, 2> g_tilde{ -0.2, -0.4 };
  const tribol::MdArray<double, 2> node_areas{ 0.5, 0.8 };
  EXPECT_NEAR( 0.5 * 10.0 * tribol::sum( g_tilde, g_tilde, 1.0 / node_areas ), 0.5 * 10.0 * ( 0.04 / 0.5 + 0.16 / 0.8 ),
               1.0e-15 );
}

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
