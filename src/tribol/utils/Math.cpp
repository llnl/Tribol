// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/utils/Math.hpp"

namespace tribol {

TRIBOL_HOST_DEVICE RealT magnitude( RealT const vx, RealT const vy, RealT const vz )
{
  return sqrt( vx * vx + vy * vy + vz * vz );
}

//------------------------------------------------------------------------------
RealT magnitude( RealT const* const v, int const dim )
{
  RealT mag = 0.;
  for ( int i = 0; i < dim; ++i ) {
    mag += v[i] * v[i];
  }
  return sqrt( mag );
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE RealT dotProd( RealT const* const v, RealT const* const w, int const dim )
{
  RealT z = 0;
  for ( int i = 0; i < dim; ++i ) {
    z += v[i] * w[i];
  }

  return z;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE RealT dotProd( RealT const aX, RealT const aY, RealT const aZ, RealT const bX, RealT const bY,
                                  RealT const bZ )
{
  return aX * bX + aY * bY + aZ * bZ;
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE RealT magCrossProd( RealT const a[3], RealT const b[3] )
{
  RealT vi = a[1] * b[2] - a[2] * b[1];
  RealT vj = a[2] * b[0] - a[0] * b[2];
  RealT vk = a[0] * b[1] - a[1] * b[0];

  return magnitude( vi, vj, vk );
}

//------------------------------------------------------------------------------
TRIBOL_HOST_DEVICE void crossProd( RealT const aX, RealT const aY, RealT const aZ, RealT const bX, RealT const bY,
                                   RealT const bZ, RealT& prodX, RealT& prodY, RealT& prodZ )
{
  prodX = aY * bZ - aZ * bY;
  prodY = aZ * bX - aX * bZ;
  prodZ = aX * bY - aY * bX;
}

//------------------------------------------------------------------------------
int binary_search( const int* const array, const int n, const int val )
{
  if ( n == 0 ) {
    SLIC_DEBUG( "binary_search: n = 0 return infeasible index." );
    return -1;
  } else if ( n == 1 && val == array[0] ) {
    return 0;
  } else if ( n == 1 && val != array[0] ) {
    SLIC_DEBUG( "binary_search: val is not equal to array[0] for n = 1." );
    return -1;
  }

  int L = 0;
  int R = n - 1;
  while ( L <= R ) {
    int m = ( L + R ) / 2;
    if ( array[m] < val ) {
      L = m + 1;
    } else if ( array[m] > val ) {
      R = m - 1;
    } else {
      return m;
    }
  }

  SLIC_DEBUG( "binary_search: could not locate value in provided array." );
  return -1;
}

//------------------------------------------------------------------------------
template <typename T>
void swap_val( T* xp, T* yp )
{
  int temp = *xp;
  *xp = *yp;
  *yp = temp;
}

//------------------------------------------------------------------------------
template <typename T>
void bubble_sort( T* array, int n )
{
  int i, j;
  for ( i = 0; i < n - 1; ++i ) {
    for ( j = 0; j < n - i - 1; ++j ) {
      if ( array[j] > array[j + 1] ) {
        swap_val( &array[j], &array[j + 1] );
      }
    }
  }
}

template void bubble_sort( int* array, int n );
template void bubble_sort( long* array, int n );
template void bubble_sort( long long* array, int n );

//------------------------------------------------------------------------------
RealT abs_val_diff( RealT val1, RealT val2 ) { return std::abs( val1 - val2 ); }
//------------------------------------------------------------------------------
void allocRealArray( RealT** arr, int length, RealT init_val )
{
  SLIC_ERROR_IF( length == 0, "allocRealArray: please specify nonzero length " << "for array allocation." );

  *arr = new RealT[length];
  initRealArray( *arr, length, init_val );
}

//------------------------------------------------------------------------------
void allocRealArray( RealT** arr, const int length, const RealT* const data )
{
  SLIC_ERROR_IF( length == 0, "allocRealArray: please specify nonzero length " << "for array allocation." );

  if ( data == nullptr ) {
    SLIC_ERROR( "allocRealArray: input data pointer not set." );
  }

  *arr = new RealT[length];

  for ( int i = 0; i < length; ++i ) {
    ( *arr )[i] = data[i];
  }

  return;
}

//------------------------------------------------------------------------------
void allocIntArray( int** arr, int length, int init_val )
{
  SLIC_ERROR_IF( length == 0, "allocIntArray: please specify nonzero length " << "for array allocation." );

  *arr = new int[length];
  initIntArray( *arr, length, init_val );
}

//------------------------------------------------------------------------------
void allocIntArray( int** arr, const int length, const int* const data )
{
  SLIC_ERROR_IF( length == 0, "allocIntArray: please specify nonzero length " << "for array allocation." );

  if ( data == nullptr ) {
    SLIC_ERROR( "allocIntArray: input data pointer not set." );
  }

  *arr = new int[length];

  for ( int i = 0; i < length; ++i ) {
    ( *arr )[i] = data[i];
  }

  return;
}

//------------------------------------------------------------------------------
template <typename T>
void allocArray( T** arr, int length, T init_val )
{
  SLIC_ERROR_IF( length == 0, "allocIntArray: please specify nonzero length " << "for array allocation." );

  *arr = new T[length];
  initArray( *arr, length, init_val );
}

template void allocArray( IndexT** arr, int length, IndexT init_val );

//------------------------------------------------------------------------------
void allocBoolArray( bool** arr, int length, bool init_val )
{
  SLIC_ERROR_IF( length == 0, "allocBoolArray: please specify nonzero length " << "for array allocation." );

  *arr = new bool[length];
  initBoolArray( *arr, length, init_val );
}

//------------------------------------------------------------------------------
template <typename T>
TRIBOL_HOST_DEVICE void initArray( T* arr, int length, T init_val )
{
#ifdef TRIBOL_USE_HOST
  SLIC_ERROR_IF( arr == nullptr, "initIntArray(): " << "input pointer to array is null." );
#endif
  for ( int i = 0; i < length; ++i ) {
    arr[i] = init_val;
  }
}

//------------------------------------------------------------------------------

}  // namespace tribol
