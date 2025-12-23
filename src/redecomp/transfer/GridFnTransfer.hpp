// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_TRANSFER_GRIDFNTRANSFER_HPP_
#define SRC_REDECOMP_TRANSFER_GRIDFNTRANSFER_HPP_

#include "mfem.hpp"

namespace redecomp {

/**
 * @brief GridFnTransfer interface base class
 */
class GridFnTransfer {
 public:
  /**
   * @brief Transfers nodal values from src to dst
   *
   * @param src A parent ParGridFunction to be transferred to corresponding
   * redecomp GridFunction (dst)
   * @param dst A redecomp GridFunction which receives values from a parent
   * ParGridFunction (src)
   */
  virtual void TransferToSerial( const mfem::ParGridFunction& src, mfem::GridFunction& dst ) const = 0;

  /**
   * @brief Transfers nodal values from src to dst
   *
   * @param src A redecomp GridFunction to be transferred to corresponding
   * parent ParGridFunction (dst)
   * @param dst A parent ParGridFunction which receives values from a redecomp
   * GridFunction (src)
   */
  virtual void TransferToParallel( const mfem::GridFunction& src, mfem::ParGridFunction& dst ) const = 0;

  /**
   * @brief Destroy the GridFnTransfer object
   */
  virtual ~GridFnTransfer() = default;
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_TRANSFER_GRIDFNTRANSFER_HPP_ */
