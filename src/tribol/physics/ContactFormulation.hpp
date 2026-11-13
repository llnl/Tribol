// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Contact calculations and their caller-owned outputs.
 */
#ifndef SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_

#include "tribol/config.hpp"
#include "tribol/mesh/MeshData.hpp"
#include <memory>
#include <optional>
#include <stdexcept>
#ifdef BUILD_REDECOMP
#include "mfem.hpp"
#endif

namespace tribol {

/**
 * @brief Own the outputs produced by a contact calculation.
 *
 * An empty optional or null pointer means that output was not calculated. A computed zero energy or an allocated
 * zero vector or matrix is an available result. Outputs remain valid after another calculation or destruction of the
 * formulation; the associated MFEM spaces and MPI communicator must remain alive.
 */
struct ContactFormulationResult {
  std::optional<RealT> energy;     ///< Globally summed contact energy, when the method defines one.
  std::optional<RealT> time_step;  ///< Maximum allowable timestep, when calculated.
#ifdef BUILD_REDECOMP
  std::unique_ptr<mfem::HypreParVector> force;     ///< Contact residual on parent displacement true DOFs.
  std::unique_ptr<mfem::HypreParVector> gap;       ///< Normalized penalty gaps or weighted LM gaps on dual true DOFs.
  std::unique_ptr<mfem::HypreParVector> pressure;  ///< Calculated penalty pressure or the LM values used in assembly.
  std::unique_ptr<mfem::HypreParMatrix> df_dx;     ///< Derivative of the contact residual with respect to coordinates.
  std::unique_ptr<mfem::HypreParMatrix> weighted_gap_jacobian;  ///< Weighted-gap coordinate Jacobian; penalty opening
                                                                ///< rows are zeroed, without normalization.
  std::unique_ptr<mfem::HypreParMatrix> df_dp;  ///< Derivative of the contact residual with respect to LM values.
#endif
};

/**
 * @brief Calculate contact outputs without retaining geometry, intermediate data, or results between calls.
 *
 * Implementations retain method and search configuration and reference mesh/field transfer data. The transfer data
 * must remain valid for the formulation's lifetime. Each calculation obtains its prerequisites from the current inputs,
 * independently of previous calls. Mortar formulations use mesh1 as the non-mortar integration surface and mesh2 as the
 * mortar projection surface. Mesh arrays must remain alive for the duration of the call.
 */
class ContactFormulation {
 public:
  /** @brief Destroy the formulation without affecting previously returned outputs. */
  virtual ~ContactFormulation() = default;

  /**
   * @brief Calculate nodal gaps and any related outputs provided by the method.
   * @param mesh1 Current first mesh in coupling-scheme order; not retained.
   * @param mesh2 Current second mesh in coupling-scheme order; not retained.
   * @return Caller-owned outputs; uncalculated quantities are absent.
   * @throws std::logic_error The method does not define a nodal gap field.
   */
  virtual ContactFormulationResult computeNodalGaps( [[maybe_unused]] MeshData::Viewer mesh1,
                                                     [[maybe_unused]] MeshData::Viewer mesh2 ) const
  {
    throw std::logic_error( "This contact formulation does not define nodal gaps." );
  }

  /**
   * @brief Calculate nodal contact residuals and any related outputs provided by the method.
   * @param mesh1 Current first mesh in coupling-scheme order; not retained.
   * @param mesh2 Current second mesh in coupling-scheme order; not retained.
   * @return Caller-owned outputs; uncalculated quantities are absent.
   */
  virtual ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const = 0;
};

}  // namespace tribol
#endif /* SRC_TRIBOL_PHYSICS_CONTACTFORMULATION_HPP_ */
