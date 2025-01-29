// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/utils/FiniteDifference.hpp"

#include <vector>

namespace tribol {

namespace util {

template <typename FN>
void FiniteDiff( double* x, int x_size, int f_size, double* dx, FN&& f, const double* x_dir, double delta )
{
  std::vector<double> y( f_size );
  std::vector<double> y_shift( f_size );
  f( x, y.data() );
  for ( int i{}; i < x_size; ++i ) {
    auto shift = delta;
    if ( x_dir ) {
      shift *= x_dir[i];
    }
    x[i] += shift;
    f( x, y_shift.data() );
    for ( int j{}; j < f_size; ++j ) {
      dx[i * f_size + j] = ( y_shift[j] - y[j] ) / shift;
    }
    x[i] -= shift;
  }
}

}  // namespace util

}  // namespace tribol
