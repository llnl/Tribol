// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_TRANSFER_GRIDFNTRANSFER_HPP_
#define SRC_REDECOMP_TRANSFER_GRIDFNTRANSFER_HPP_

#include "mfem.hpp"

namespace redecomp {

/**
 * @brief Interface for transferring primal fields and dual forms between a parent mesh and a redecomp mesh
 */
class GridFnTransfer {
 public:
  /**
   * @brief Replicate parent primal values to every required redecomp copy
   *
   * @param src A parent ParGridFunction to be transferred to corresponding
   * redecomp GridFunction (dst)
   * @param dst A redecomp GridFunction which receives values from a parent
   * ParGridFunction (src)
   */
  virtual void TransferToSerial( const mfem::ParGridFunction& src, mfem::GridFunction& dst ) const = 0;

  /**
   * @brief Select one deterministic redecomp copy for each parent primal value
   *
   * @param src A redecomp GridFunction to be transferred to corresponding
   * parent ParGridFunction (dst)
   * @param dst A parent ParGridFunction which receives values from a redecomp
   * GridFunction (src)
   */
  virtual void TransferToParallel( const mfem::GridFunction& src, mfem::ParGridFunction& dst ) const = 0;

  /**
   * @brief Collapse an already-assembled redecomp dual replicated across eligible copies
   *
   * @param src Redecomp linear form whose return-eligible copies agree
   * @param dst Parent parallel linear form receiving one designated copy
   */
  virtual void CollapseReplicatedDualToParallel( const mfem::LinearForm& src, mfem::ParLinearForm& dst ) const = 0;

  /**
   * @brief Destroy the GridFnTransfer object
   */
  virtual ~GridFnTransfer() = default;
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_TRANSFER_GRIDFNTRANSFER_HPP_ */
