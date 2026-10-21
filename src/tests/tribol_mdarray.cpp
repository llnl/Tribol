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
  using Array = tribol::MdArray<double, 2, 3>;
  using Row = tribol::MdArray<double, 3>;
  using RowView = tribol::MdSpan<const double, 3>;
  static_assert( sizeof( Array ) == 6 * sizeof( double ) );
  static_assert( std::is_trivially_copyable_v<Array> );
  static_assert( Array::rank() == 2 );
  static_assert( Array::size() == 6 );
  static_assert( Array::extent<0>() == 2 );
  static_assert( Array::extent<1>() == 3 );
  static_assert( std::is_constructible_v<Row, double, double, double> );
  static_assert( !std::is_constructible_v<Row, double, double> );
  static_assert( !std::is_constructible_v<Row, double, double, double, double> );
  static_assert( std::is_constructible_v<Array, Row, Row> );
  static_assert( std::is_constructible_v<Array, RowView, RowView> );
  static_assert( !std::is_constructible_v<Array, Row> );
  static_assert( !std::is_constructible_v<Array, Row, Row, Row> );
  static_assert( !std::is_constructible_v<Array, tribol::MdArray<double, 2>, tribol::MdArray<double, 2>> );
  static_assert( !std::is_constructible_v<Array, tribol::MdSpan<const double, 2>, tribol::MdSpan<const double, 2>> );

  Array values{ Row{ 0.0, 1.0, 2.0 }, Row{ 3.0, 4.0, 5.0 } };
  const double first_row[3] = { 6.0, 7.0, 8.0 };
  const double second_row[3] = { 9.0, 10.0, 11.0 };
  const Array values_from_views{ RowView{ first_row }, RowView{ second_row } };

  static_assert( std::is_same_v<decltype( values.view() ), tribol::MdSpan<double, 2, 3>> );
  EXPECT_EQ( values( 1, 2 ), 5.0 );
  EXPECT_EQ( values_from_views( 0, 2 ), 8.0 );
  EXPECT_EQ( values_from_views( 1, 0 ), 9.0 );

  const Array& read_only_values = values;
  static_assert( std::is_same_v<decltype( read_only_values.view() ), tribol::MdSpan<const double, 2, 3>> );
  EXPECT_EQ( read_only_values.view()( 1, 0 ), 3.0 );

  auto copied_values = values;
  copied_values( 0, 1 ) = 8.0;
  EXPECT_EQ( values( 0, 1 ), 1.0 );
  EXPECT_EQ( copied_values( 0, 1 ), 8.0 );
}

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
