// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_CONTACTFORMULATIONFACTORY_HPP_
#define SRC_TRIBOL_PHYSICS_CONTACTFORMULATIONFACTORY_HPP_

#include "tribol/physics/ContactFormulation.hpp"
#include <memory>

namespace tribol {

// Forward declaration
class CouplingScheme;

/**
 * @brief Factory function to create a ContactFormulation based on the CouplingScheme settings.
 *
 * @param cs Pointer to the CouplingScheme
 * @return The created formulation, or nullptr if no formulation applies or Energy Mortar penalty settings are
 * incomplete.
 * @throws std::invalid_argument A supplied formulation setting or effective penalty is invalid.
 */
std::unique_ptr<ContactFormulation> createContactFormulation( CouplingScheme* cs );

namespace detail {
/**
 * @brief Validate the supported Common Plane formulation configuration before selecting or rebuilding it.
 * @param cs Registered coupling scheme; no configuration is changed.
 * @param compute_jacobian Whether the requested formulation will calculate df_dx.
 * @throws std::invalid_argument The method, geometry, penalty, or execution configuration is unsupported.
 * @throws std::runtime_error Redecomp is unavailable, or Jacobian calculation was requested without Enzyme.
 */
void validateCommonPlaneFormulation( CouplingScheme& cs, bool compute_jacobian );
}  // namespace detail

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_CONTACTFORMULATIONFACTORY_HPP_ */
