// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_TRANSFER_TRANSFERBYELEMENTS_HPP_
#define SRC_REDECOMP_TRANSFER_TRANSFERBYELEMENTS_HPP_

#include "redecomp/transfer/GridFnTransfer.hpp"

namespace redecomp {

/**
 * @brief GridFnTransfer method using elements
 *
 * The RedecompMesh holds a list of parent elements to send to RedecompMesh and
 * a list of RedecompMesh elements to send to parent.  Since this method
 * transfers degree-of-freedom values element by element, the element lists
 * stored in Redecomp suffice.  The related TransferByNodes method constructs
 * lists of degrees of freedom to transfer from parent to Redecomp and vice
 * versa and, therefore, may be faster for repeated transfers of H1 fields.
 */
class TransferByElements : public GridFnTransfer {
 public:
  /**
   * @brief Copies parent-based mfem::ParGridFunction values to a
   * RedecompMesh-based mfem::GridFunction
   *
   * @param src A parent ParGridFunction to be copied to corresponding redecomp
   * GridFunction (dst)
   * @param dst A redecomp GridFunction which receives values from a parent
   * ParGridFunction (src)
   */
  void TransferToSerial( const mfem::ParGridFunction& src, mfem::GridFunction& dst ) const override;

  /**
   * @brief Copies RedecompMesh-based mfem::GridFunction values to a
   * parent-based mfem::ParGridFunction
   *
   * @param src A redecomp GridFunction to be copied to corresponding parent
   * ParGridFunction (dst)
   * @param dst A parent ParGridFunction which receives values from a redecomp
   * GridFunction (src)
   */
  void TransferToParallel( const mfem::GridFunction& src, mfem::ParGridFunction& dst ) const override;
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_TRANSFER_TRANSFERBYELEMENTS_HPP_ */
