// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Runtime interface for evaluating and assembling a batch of contact pairs. */
#ifndef SRC_TRIBOL_ASSEMBLY_CONTACTEVALUATION_HPP_
#define SRC_TRIBOL_ASSEMBLY_CONTACTEVALUATION_HPP_

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/physics/ContactFormulation.hpp"

namespace tribol {

/**
 * @brief Evaluate contact pairs and assemble the outputs provided by the selected calculation.
 *
 * Implementations own a fixed pair function and reference the transfer data needed by their calculation. Transfer
 * data must outlive the evaluation. Mesh views are supplied on each call and are not retained.
 */
class ContactEvaluation {
 public:
  /** @brief Destroy the evaluation without affecting previously returned outputs. */
  virtual ~ContactEvaluation() = default;

  /**
   * @brief Evaluate the supplied pairs, check quadrature, and assemble the configured outputs.
   * @param pairs Accepted element pairs in execution-accessible memory; not retained.
   * @param mesh1 Current first mesh with execution-accessible coordinates and connectivity; not retained.
   * @param mesh2 Current second mesh with execution-accessible coordinates and connectivity; not retained.
   * @return Caller-owned force and any energy, nodal fields, and derivatives provided by the calculation. Unavailable
   * outputs are absent. Empty pairs contribute zero force and energy; nodal fields follow the selected enforcement.
   * Scratch arrays remain alive until evaluation and assembly finish.
   * @throws std::invalid_argument Pair geometry or assembly dimensions are invalid; no result is returned.
   * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded.
   */
  virtual ContactFormulationResult evaluateAndAssemble( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                        MeshData::Viewer mesh2 ) const = 0;

  /**
   * @brief Evaluate and assemble nodal gaps when defined by the selected calculation.
   * @param pairs Accepted element pairs in execution-accessible memory; not retained.
   * @param mesh1 Current first mesh; not retained.
   * @param mesh2 Current second mesh; not retained.
   * @return Caller-owned gap fields and their available derivatives. Normalization and activation follow the
   * concrete nodal enforcement; other outputs are absent.
   * @throws std::logic_error The calculation does not define nodal gaps.
   * @throws std::invalid_argument Pair geometry or assembly dimensions are invalid.
   * @throws std::length_error Quadrature capacity or the supported assembly index range is exceeded.
   */
  virtual ContactFormulationResult evaluateAndAssembleGaps( [[maybe_unused]] const ArrayT<InterfacePair>& pairs,
                                                            [[maybe_unused]] MeshData::Viewer mesh1,
                                                            [[maybe_unused]] MeshData::Viewer mesh2 ) const
  {
    throw std::logic_error( "This contact evaluation does not define a nodal gap field." );
  }
};

}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_ASSEMBLY_CONTACTEVALUATION_HPP_
