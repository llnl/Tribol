// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef TRIBOL_COMMON_ENZYME_HPP_
#define TRIBOL_COMMON_ENZYME_HPP_

#include "tribol/config.hpp"

#ifdef TRIBOL_USE_ENZYME
/*
 * Variables prefixed with enzyme_* or function types prefixed with __enzyme_*,
 * are variables which will get preprocessed in the LLVM intermediate
 * representation when the Enzyme LLVM plugin is loaded. See the Enzyme
 * documentation (https://enzyme.mit.edu) for more information.
 */

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

extern int enzyme_runtime_activity;

template <typename return_type, typename... Args>
return_type __enzyme_autodiff(Args...);

template <typename return_type, typename... Args>
return_type __enzyme_fwddiff(Args...);
#endif

#endif /* TRIBOL_COMMON_ENZYME_HPP_ */
