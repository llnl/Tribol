// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_UTILS_FINITEDIFFERENCE_HPP_
#define SRC_UTILS_FINITEDIFFERENCE_HPP_

namespace tribol {

namespace util {

template <typename FN>
void FiniteDiff(double* x, int x_size, int f_size, double* dx, FN&& f, double* x_dir = nullptr, double delta = 1.0e-7);

}  // namespace util

}  // namespace tribol

#endif /* SRC_UTILS_FINITEDIFFERENCE_HPP_ */
