// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
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
 * @return std::unique_ptr<ContactFormulation> The created formulation, or nullptr if no formulation applies.
 */
std::unique_ptr<ContactFormulation> createContactFormulation( CouplingScheme* cs );

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_CONTACTFORMULATIONFACTORY_HPP_ */
