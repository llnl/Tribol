// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
// SPDX-License-Identifier: (MIT)
/** @file @brief Owned MFEM contact residuals and method-dependent diagnostics. */
#ifndef SRC_TRIBOL_PHYSICS_CONTACTRESIDUAL_HPP_
#define SRC_TRIBOL_PHYSICS_CONTACTRESIDUAL_HPP_

#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include "mfem.hpp"
#include <memory>

namespace tribol {
/**
 * @brief Own assembled contact residuals and the diagnostics supplied by the selected method.
 *
 * force uses parent displacement true DOFs. constraint uses multiplier true DOFs: scalar for frictionless nodal
 * mortar and Cartesian for Coulomb mortar. Penalty methods have no multiplier equation and return a null constraint.
 * Unavailable diagnostics have length zero; an allocated zero residual is an available calculation.
 *
 * Scalar diagnostics use locally owned scalar contact-submesh true DOFs. Coulomb mortar supplies compression-positive
 * pressure, an admissible gap <= 0, and area-weighted gap/slip with units length cubed. Other methods retain their
 * documented gap and pressure conventions. Only Coulomb mortar currently supplies slip, normal, and state.
 * Outputs remain valid after subsequent evaluations or scheme destruction while the associated spaces and MPI
 * communicator live. A successful calculation with a nonzero residual does not establish nonlinear convergence.
 */
struct ContactResidual {
  std::unique_ptr<mfem::HypreParVector> force;       ///< Contact contribution to displacement equilibrium.
  std::unique_ptr<mfem::HypreParVector> constraint;  ///< Multiplier equations, or null without multiplier enforcement.
  mfem::Vector gap;                                  ///< Normal gap in the selected method's convention.
  mfem::Vector pressure;                             ///< Normal traction in the selected method's convention.
  mfem::Vector slip;       ///< Coulomb weighted slip increment in Cartesian multiplier true-DOF order.
  mfem::Vector normal;     ///< Coulomb current unit normals in multiplier true-DOF order; zero without overlap.
  mfem::Array<int> state;  ///< Coulomb FrictionState values, for diagnostics only.
};
}  // namespace tribol
#endif
#endif
