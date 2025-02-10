// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_UTILS_MATH_HPP_
#define SRC_UTILS_MATH_HPP_

#include "tribol/types.hpp"

namespace tribol {

/// returns the magnitude of a 3-vector
real magnitude( real const vx,  ///< [in] x-component of the input vector
                real const vy,  ///< [in] y-component of the input vector
                real const vz   ///< [in] z-component of the input vector
);

/// returns the magnitude of a 2-vector
real magnitude( real const vx,  ///< [in] x-component of the input vector
                real const vy   ///< [in] y-component of the input vector
);

/// returns the dot product of two vectors
real dotProd( real const* const v,  ///< [in] first vector
              real const* const w,  ///< [in] second vector
              int const dim         ///< [in] dimension of the vectors
);

/// returns the dot product of two 3-vectors with component-wise input
real dotProd( real const aX,  ///< [in] x-component of first vector
              real const aY,  ///< [in] y-component of first vector
              real const aZ,  ///< [in] z-component of first vector
              real const bX,  ///< [in] x-component of second vector
              real const bY,  ///< [in] y-component of second vector
              real const bZ   ///< [in] z-component of second vector
);

/// returns the magnitude of the cross product of two 3-vectors
real magCrossProd( real const a[3],  ///< [in] array of components of first 3-vector
                   real const b[3]   ///< [in] array of components of second 3-vector
);

/// computes and returns the constituent cross product terms of two 3-vectors with component-wise input
void crossProd( real const aX,  ///< [in] x-component of first vector
                real const aY,  ///< [in] y-component of first vector
                real const aZ,  ///< [in] z-component of first vector
                real const bX,  ///< [in] x-component of second vector
                real const bY,  ///< [in] y-component of second vector
                real const bZ,  ///< [in] z-component of second vector
                real& prodX,    ///< [in,out] j x k (i-component) product term
                real& prodY,    ///< [in,out] i x k (j-component) product term
                real& prodZ     ///< [in,out] i x j (k-component) product term
);

/// binary search algorithm on presorted array
int binary_search( const int* const array,  ///< [in] pointer to array of integer values
                   const int n,             ///< [in] size of array
                   const int val            ///< [in] value in array whose index is sought
);

/// routine to swap values between two arrays
void swap_val( int* xp,  ///< [in] Pointer to value to be swapped
               int* yp   ///< [out] Pointer to new value
);

/// bubble sort elements of one array in increasing order
void bubble_sort( int* const array,  ///< [in] Input array of integers
                  const int n        ///< [in] Size of array
);

/// allocate and initialize an array of reals
void allocRealArray( real** arr, int length, real init_val );

/// allocate an array of reals and initialize with a pointer to data
void allocRealArray( real** arr, const int length, const real* const data );

/// allocate and initialize an array of integers
void allocIntArray( int** arr, int length, int init_val );

/// allocate an array of integers and initialize with a pointer to data
void allocIntArray( int** arr, const int length, const int* const data );

/// allocate and initialize an array of booleans
void allocBoolArray( bool** arr, int length, bool init_val );

/// initialize a array of reals
void initRealArray( real* arr, int length, real init_val );

/// initialize a array of integers
void initIntArray( int* arr, int length, int init_val );

/// initialize a array of booleans
void initBoolArray( bool* arr, int length, bool init_val );

}  // namespace tribol

#endif /* SRC_UTILS_MATH_HPP_ */
