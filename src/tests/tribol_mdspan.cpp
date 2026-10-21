// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <array>
#include <type_traits>

#include <gtest/gtest.h>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/common/MdSpan.hpp"

namespace {

TEST( MdSpan, provides_a_fixed_extent_row_major_view )
{
  using MatrixView = tribol::MdSpan<double, 2, 3>;
  using VectorView = tribol::MdSpan<double, 2>;
  using ReadOnlyVectorView = tribol::MdSpan<const double, 2>;
  using ScalarView = tribol::MdSpan<double, 1>;
  using ReadOnlyScalarView = tribol::MdSpan<const double, 1>;

  static_assert( sizeof( MatrixView ) == sizeof( double* ) );
  static_assert( std::is_trivially_copyable_v<MatrixView> );
  static_assert( std::is_constructible_v<VectorView, double( & )[2]> );
  static_assert( !std::is_constructible_v<VectorView, double( & )[3]> );
  static_assert( std::is_constructible_v<VectorView, std::array<double, 2>&> );
  static_assert( std::is_constructible_v<ReadOnlyVectorView, const std::array<double, 2>&> );
  static_assert( !std::is_constructible_v<VectorView, const std::array<double, 2>&> );
  static_assert( !std::is_constructible_v<VectorView, std::array<double, 3>&> );
  static_assert( !std::is_constructible_v<VectorView, std::array<double, 2>&&> );
  static_assert( !std::is_constructible_v<ReadOnlyVectorView, const std::array<double, 2>&&> );
  static_assert( std::is_constructible_v<ScalarView, double&> );
  static_assert( std::is_constructible_v<ReadOnlyScalarView, const double&> );
  static_assert( !std::is_constructible_v<ScalarView, const double&> );
  static_assert( !std::is_constructible_v<ScalarView, double&&> );
  static_assert( !std::is_constructible_v<VectorView, double&> );

  double values[6] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
  MatrixView values_by_row( values );
  tribol::MdSpan<const double, 2, 3> read_only_values( values_by_row );

  std::array<double, 2> vector_values{ 6.0, 7.0 };
  const std::array<double, 2> read_only_vector_values{ 8.0, 9.0 };
  VectorView vector_view( vector_values );
  ReadOnlyVectorView read_only_vector_view( read_only_vector_values );

  double scalar_value = 10.0;
  const double read_only_scalar_value = 11.0;
  ScalarView scalar_view( scalar_value );
  ReadOnlyScalarView read_only_scalar_view( read_only_scalar_value );

  static_assert( decltype( values_by_row )::rank() == 2 );
  static_assert( decltype( values_by_row )::size() == 6 );
  static_assert( decltype( values_by_row )::extent<0>() == 2 );
  static_assert( decltype( values_by_row )::extent<1>() == 3 );
  EXPECT_EQ( values_by_row( 1, 2 ), values[5] );
  EXPECT_EQ( read_only_values[3], values[3] );
  EXPECT_EQ( read_only_vector_view[1], read_only_vector_values[1] );
  EXPECT_EQ( read_only_scalar_view[0], read_only_scalar_value );

  values_by_row( 0, 1 ) = 8.0;
  vector_view[1] = 12.0;
  scalar_view[0] = 13.0;
  EXPECT_EQ( values[1], 8.0 );
  EXPECT_EQ( vector_values[1], 12.0 );
  EXPECT_EQ( scalar_value, 13.0 );
}

}  // namespace

int main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  return RUN_ALL_TESTS();
}
