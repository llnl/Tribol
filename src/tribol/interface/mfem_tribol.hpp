// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTERFACE_MFEM_TRIBOL_HPP_
#define SRC_TRIBOL_INTERFACE_MFEM_TRIBOL_HPP_

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP

#include <memory>
#include <vector>

#include "mfem.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/interface/MfemContact.hpp"
#include "tribol/mesh/MfemContactData.hpp"

namespace tribol {

/**
 * @brief Register contact between two caller-owned MFEM parent-mesh surfaces
 *
 * Registration creates the coupling scheme but does not search or exchange
 * faces. The first tribol::update() builds the patch; later updates reuse it
 * until its motion skin or another patch-rebuild condition is reached. All MFEM
 * objects referenced by the surface descriptors must outlive the scheme.
 */
void registerMfemCouplingScheme( IndexT cs_id, MfemContactSurface surface_1, MfemContactSurface surface_2,
                                 ContactMode contact_mode, ContactCase contact_case, ContactMethod contact_method,
                                 ContactModel contact_model, EnforcementMethod enforcement_method,
                                 const MfemContactOptions& options = {} );

/** @brief Set constant kinematic penalty stiffnesses for the two registered surfaces */
void setMfemKinematicConstantPenalty( IndexT cs_id, RealT surface_1_penalty, RealT surface_2_penalty );

/** @brief Set viscous damping coefficients for the two registered surfaces */
void setMfemViscousDampingCoeff( IndexT cs_id, RealT surface_1_coefficient, RealT surface_2_coefficient );

/** @brief Force the contact search and exchange plan to rebuild during the next update */
void forceMfemContactPatchRebuildOnNextUpdate( IndexT cs_id );

/**
 * @brief Return one true-DOF contact residual per unique solver finite element space
 *
 * The returned reference remains owned by the coupling scheme and is
 * overwritten by each successful update(). Surfaces sharing the same
 * ParFiniteElementSpace share one residual; a multiplier residual is last.
 */
const std::vector<MfemContactResidual>& getMfemContactResiduals( IndexT cs_id );

/**
 * @brief Assemble and return an owning Jacobian in contact-residual order
 *
 * Call after update() has produced element Jacobian contributions. Explicit H
 * matrices are assembled lazily; repeated calls reuse them until patch
 * rebuilding. The caller owns the returned BlockOperator.
 */
std::unique_ptr<mfem::BlockOperator> getMfemContactJacobian( IndexT cs_id );

/** @brief Return the solver-visible multiplier finite element space */
const mfem::ParFiniteElementSpace& getMfemMultiplierFESpace( IndexT cs_id );

/** @brief Return mutable, scheme-owned canonical multiplier pressure true DOFs */
mfem::Vector& getMfemMultiplierPressure( IndexT cs_id );

/** @brief Return scheme-owned canonical multiplier gap true DOFs from the latest update */
const mfem::Vector& getMfemMultiplierGap( IndexT cs_id );

/**
 * @brief Materialize one cached linear contact-patch surface for diagnostics
 *
 * @param surface Zero for the first registered surface or one for the second
 * @return Caller-owned serial mesh containing this rank's evaluation patch
 */
std::unique_ptr<mfem::Mesh> materializeMfemContactPatch( IndexT cs_id, int surface );

/** @brief Write both cached contact-patch surfaces without retaining diagnostic meshes */
void saveMfemContactPatch( IndexT cs_id, int output_id );

}  // namespace tribol

#endif  // BUILD_REDECOMP

#endif  // SRC_TRIBOL_INTERFACE_MFEM_TRIBOL_HPP_
