// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Runtime interface for evaluating and assembling a batch of pair residuals. */
#ifndef SRC_TRIBOL_ASSEMBLY_RESIDUALEVALUATION_HPP_
#define SRC_TRIBOL_ASSEMBLY_RESIDUALEVALUATION_HPP_

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/physics/ContactFormulation.hpp"

namespace tribol {

/**
 * @brief Evaluate pair residuals and assemble their selected force and Jacobian outputs.
 *
 * Implementations own a fixed pair function and reference the transfer data needed by their calculation. Transfer
 * data must outlive the evaluation. Mesh views are supplied on each call and are not retained.
 */
class ResidualEvaluation {
 public:
  /** @brief Destroy the evaluation without affecting previously returned outputs. */
  virtual ~ResidualEvaluation() = default;

  /**
   * @brief Evaluate the supplied pairs, check quadrature, and assemble the configured outputs.
   * @param pairs Accepted element pairs in execution-accessible memory; not retained.
   * @param mesh1 Current first mesh with execution-accessible coordinates and connectivity; not retained.
   * @param mesh2 Current second mesh with execution-accessible coordinates and connectivity; not retained.
   * @return Owned parent true-DOF force and, when configured, df_dx. Other outputs are absent. Empty or inactive
   * pairs produce allocated zero outputs. Scratch arrays remain alive until evaluation and assembly finish.
   * @throws std::invalid_argument Pair geometry or assembly dimensions are invalid; no result is returned.
   * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded.
   */
  virtual ContactFormulationResult evaluateAndAssemble( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                        MeshData::Viewer mesh2 ) const = 0;
};

}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_ASSEMBLY_RESIDUALEVALUATION_HPP_
