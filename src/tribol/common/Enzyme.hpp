// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_COMMON_ENZYME_HPP_
#define SRC_TRIBOL_COMMON_ENZYME_HPP_

// Tribol config include
#include "tribol/config.hpp"

// Tribol includes
#include "tribol/common/BasicTypes.hpp"

#ifdef TRIBOL_USE_ENZYME
/*
 * Enzyme is an LLVM-based automatic differentiation (AD) tool that enables
 * efficient computation of derivatives directly from code at compile time.
 *
 * Variables prefixed with enzyme_* or function types prefixed with __enzyme_*,
 * are variables which will get preprocessed in the LLVM intermediate
 * representation when the Enzyme LLVM plugin is loaded.
 *
 * For more details, see the Enzyme documentation: https://enzyme.mit.edu
 */

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

extern int enzyme_runtime_activity;

// Reverse mode autodiff
template <typename return_type, typename... Args>
TRIBOL_HOST_DEVICE return_type __enzyme_autodiff( Args... );

// Forward mode autodiff
template <typename return_type, typename... Args>
TRIBOL_HOST_DEVICE return_type __enzyme_fwddiff( Args... );

// Redefine ENZYME_ACTIVE based on whether Enzyme is being used
#undef ENZYME_ACTIVE
#define ENZYME_ACTIVE 1
// Redefine ENZYME_LOGGING to BASIC when Enzyme is being used. SLIC does not work with ClangEnzyme.
#undef ENZYME_LOGGING
#define ENZYME_LOGGING BASIC
#endif

#endif /* SRC_TRIBOL_COMMON_ENZYME_HPP_ */
