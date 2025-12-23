// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_UTILS_MATH_HPP_
#define SRC_TRIBOL_UTILS_MATH_HPP_

// C++ includes
#include <cmath>

#include "axom/slic.hpp"

#include "tribol/common/BasicTypes.hpp"

namespace tribol {

/// returns the magnitude of a 3-vector
TRIBOL_HOST_DEVICE RealT magnitude( RealT const vx,  ///< [in] x-component of the input vector
                                    RealT const vy,  ///< [in] y-component of the input vector
                                    RealT const vz   ///< [in] z-component of the input vector
);

/// returns the magnitude of a 2-vector
TRIBOL_HOST_DEVICE inline RealT magnitude( RealT const vx,  ///< [in] x-component of the input vector
                                           RealT const vy   ///< [in] y-component of the input vector
)
{
  return sqrt( vx * vx + vy * vy );
}

/// returns the dot product of two vectors
TRIBOL_HOST_DEVICE RealT dotProd( RealT const* const v,  ///< [in] first vector
                                  RealT const* const w,  ///< [in] second vector
                                  int const dim          ///< [in] dimension of the vectors
);

/// returns the dot product of two 3-vectors with component-wise input
TRIBOL_HOST_DEVICE RealT dotProd( RealT const aX,  ///< [in] x-component of first vector
                                  RealT const aY,  ///< [in] y-component of first vector
                                  RealT const aZ,  ///< [in] z-component of first vector
                                  RealT const bX,  ///< [in] x-component of second vector
                                  RealT const bY,  ///< [in] y-component of second vector
                                  RealT const bZ   ///< [in] z-component of second vector
);

/// returns the magnitude of the cross product of two 3-vectors
TRIBOL_HOST_DEVICE RealT magCrossProd( RealT const a[3],  ///< [in] array of components of first 3-vector
                                       RealT const b[3]   ///< [in] array of components of second 3-vector
);

/// computes and returns the constituent cross product terms of two 3-vectors with component-wise input
TRIBOL_HOST_DEVICE void crossProd( RealT const aX,  ///< [in] x-component of first vector
                                   RealT const aY,  ///< [in] y-component of first vector
                                   RealT const aZ,  ///< [in] z-component of first vector
                                   RealT const bX,  ///< [in] x-component of second vector
                                   RealT const bY,  ///< [in] y-component of second vector
                                   RealT const bZ,  ///< [in] z-component of second vector
                                   RealT& prodX,    ///< [in,out] j x k (i-component) product term
                                   RealT& prodY,    ///< [in,out] i x k (j-component) product term
                                   RealT& prodZ     ///< [in,out] i x j (k-component) product term
);

/// binary search algorithm on presorted array
int binary_search( const int* const array,  ///< [in] pointer to array of integer values
                   const int n,             ///< [in] size of array
                   const int val            ///< [in] value in array whose index is sought
);

/// routine to swap values between two arrays
template <typename T>
void swap_val( T* xp,  ///< [in] Pointer to value to be swapped
               T* yp   ///< [out] Pointer to new value
);

/// bubble sort elements of one array in increasing order
template <typename T>
void bubble_sort( T* array,  ///< [in] Input array to be sorted
                  int n      ///< [in] Size of array
);

/// compute the absolute value of the difference between two values
RealT abs_val_diff( RealT val1,  ///< [in] first value
                    RealT val2   ///< [in] second value
);

/// allocate and initialize an array of reals
void allocRealArray( RealT** arr, int length, RealT init_val );

/// allocate an array of reals and initialize with a pointer to data
void allocRealArray( RealT** arr, const int length, const RealT* const data );

/// allocate and initialize an array of integers
void allocIntArray( int** arr, int length, int init_val );

/// allocate an array of integers and initialize with a pointer to data
void allocIntArray( int** arr, const int length, const int* const data );

/// allocate and initialize an array of type T
template <typename T>
void allocArray( T** arr, int length, T init_val );

/// allocate and initialize an array of booleans
void allocBoolArray( bool** arr, int length, bool init_val );

/// initialize a array of reals
TRIBOL_HOST_DEVICE inline void initRealArray( RealT* arr, int length, RealT init_val )
{
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
  SLIC_ERROR_IF( arr == nullptr, "initRealArray(): " << "input pointer to array is null." );
#endif

  for ( int i = 0; i < length; ++i ) {
    arr[i] = init_val;
  }
}

/// initialize a array of integers
TRIBOL_HOST_DEVICE inline void initIntArray( int* arr, int length, int init_val )
{
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
  SLIC_ERROR_IF( arr == nullptr, "initIntArray(): " << "input pointer to array is null." );
#endif
  for ( int i = 0; i < length; ++i ) {
    arr[i] = init_val;
  }
}

/// initialize a array of type T
template <typename T>
TRIBOL_HOST_DEVICE void initArray( T* arr, int length, T init_val );

/// initialize a array of booleans
TRIBOL_HOST_DEVICE inline void initBoolArray( bool* arr, int length, bool init_val )
{
#if defined( TRIBOL_USE_HOST ) && !defined( TRIBOL_USE_ENZYME )
  SLIC_ERROR_IF( arr == nullptr, "initBoolArray(): " << "input pointer to array is null." );
#endif
  for ( int i = 0; i < length; ++i ) {
    arr[i] = init_val;
  }
}

}  // namespace tribol

#endif /* SRC_TRIBOL_UTILS_MATH_HPP_ */
