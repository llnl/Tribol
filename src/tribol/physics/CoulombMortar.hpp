// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
// SPDX-License-Identifier: (MIT)
/** @file @brief Accepted-step data and MFEM assembly for Coulomb mortar. */
#ifndef SRC_TRIBOL_PHYSICS_COULOMBMORTAR_HPP_
#define SRC_TRIBOL_PHYSICS_COULOMBMORTAR_HPP_
#include "tribol/physics/CoulombFriction.hpp"
#include "tribol/physics/ContactResidual.hpp"
#ifdef BUILD_REDECOMP
namespace tribol {
class CouplingScheme;
namespace detail {
/** @brief Own accepted-step data separately from current mesh views and trial calculations. */
class CoulombMortar {
 public:
  /** @brief Record a registered scheme's parent spaces. @param scheme Scheme owning the fixed transfer spaces. */
  explicit CoulombMortar( CouplingScheme& scheme );
  /** @brief Release outputs and history before their MFEM spaces are released. */
  ~CoulombMortar();
  /** @brief Set validated options outside a load step. @param options Friction options. */
  void setOptions( const CoulombFrictionOptions& options );
  /** @brief Snapshot current accepted coordinates, multiplier, and assembled gap vectors.
   * @param scheme Current owner of the fixed MFEM spaces. */
  void begin( CouplingScheme& scheme );
  /** @brief Promote the latest evaluated inputs; the caller is responsible for convergence acceptance.
   * @param scheme Current owner of the fixed MFEM spaces. */
  void commit( CouplingScheme& scheme );
  /** @brief Discard trial outputs and restore accepted multiplier input; caller restores coordinates.
   * @param scheme Current owner of the fixed MFEM spaces. */
  void rollback( CouplingScheme& scheme );
  /** @brief Calculate residuals and all four Jacobian blocks; failures invalidate trial outputs only.
   * @param scheme Current geometry, configuration, and field transfers; no scheme reference is retained. */
  void evaluate( CouplingScheme& scheme );
  /** @brief Return a deep copy of the latest successful residual.
   * @param scheme Current configuration and input fields. @return Owned residual and diagnostics. */
  ContactResidual residual( CouplingScheme& scheme ) const;
  /** @brief Return only the displacement residual contribution as an owning copy.
   * @param scheme Current configuration and input fields. @return Contact residual on parent true DOFs. */
  mfem::HypreParVector force( CouplingScheme& scheme ) const;
  /** @brief Return copies of all four blocks.
   * @param scheme Current configuration and input fields. @return Owned nonsymmetric displacement/multiplier operator.
   */
  std::unique_ptr<mfem::BlockOperator> jacobian( CouplingScheme& scheme ) const;
  /** @brief Reject parent topology/space/partition changes before rebuilding Redecomp fields.
   * @param scheme Current parent mesh and spaces. */
  void checkParentMesh( CouplingScheme& scheme ) const;
  /** @brief Invalidate trial outputs after changing Redecomp data or input field access. */
  void invalidate();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;  ///< Owned options, stable-space history, and latest successful outputs.
};
}  // namespace detail
}  // namespace tribol
#endif
#endif
