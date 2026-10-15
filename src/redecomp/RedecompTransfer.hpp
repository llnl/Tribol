// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_REDECOMPTRANSFER_HPP_
#define SRC_REDECOMP_REDECOMPTRANSFER_HPP_

#include "mfem.hpp"

#include "redecomp/transfer/GridFnTransfer.hpp"

namespace redecomp {

/**
 * @brief Transfer primal fields and dual forms between parent and redecomp meshes
 *
 * Primal transfers replicate or select coefficient values. Replicated-dual collapse selects one already-assembled
 * copy without multiplying the physical response.
 */
class RedecompTransfer {
 public:
  /**
   * @brief Construct a new RedecompTransfer object
   *
   * @param gf_transfer A pointer to a custom GridFnTransfer object
   */
  RedecompTransfer( std::unique_ptr<const GridFnTransfer> gf_transfer );

  /**
   * @brief Construct a new RedecompTransfer object with nodal GridFunction
   * transfer
   *
   * @note Nodal GridFunction transfer may be appropriate for repeated transfers
   * of H1 fields. Otherwise, the default constructor is usually sufficient. See
   * TransferByNodes.hpp for more details.
   *
   * @param parent_fes A ParFiniteElementSpace built on parent
   * @param redecomp_fes A FiniteElementSpace built on a RedecompMesh
   */
  RedecompTransfer( const mfem::ParFiniteElementSpace& parent_fes, const mfem::FiniteElementSpace& redecomp_fes );

  /**
   * @brief Construct a new RedecompTransfer object with element GridFunction transfer
   */
  RedecompTransfer();

  /**
   * @brief Replicate parent primal values to every required redecomp copy
   *
   * @param src A parent ParGridFunction to be copied to corresponding redecomp
   * GridFunction (dst)
   * @param dst A redecomp GridFunction which receives values from a parent
   * ParGridFunction (src)
   */
  void TransferToSerial( const mfem::ParGridFunction& src, mfem::GridFunction& dst ) const;

  /**
   * @brief Select one deterministic redecomp copy for each parent primal value
   *
   * @param src A redecomp GridFunction to be copied to corresponding parent
   * ParGridFunction (dst)
   * @param dst A parent ParGridFunction which receives values from a redecomp
   * GridFunction (src)
   */
  void TransferToParallel( const mfem::GridFunction& src, mfem::ParGridFunction& dst ) const;

  /**
   * @brief Collapse an already-assembled dual replicated across return-eligible redecomp copies
   *
   * @param src Redecomp linear form with matching return-eligible copies
   * @param dst Parent parallel linear form receiving one designated copy
   */
  void CollapseReplicatedDualToParallel( const mfem::LinearForm& src, mfem::ParLinearForm& dst ) const;

  /**
   * @brief Replicate parent quadrature values to every required redecomp element copy
   *
   * @param src A parent QuadratureFunction to be copied to corresponding
   *  redecomp QuadratureFunction (dst)
   * @param dst A redecomp QuadratureFunction which receives values from a
   *  parent QuadratureFunction (src)
   */
  void TransferToSerial( const mfem::QuadratureFunction& src, mfem::QuadratureFunction& dst ) const;

  /**
   * @brief Select the non-ghost redecomp element copy for each parent quadrature value
   *
   * @param src A redecomp QuadratureFunction to be copied to corresponding
   * parent QuadratureFunction (dst)
   * @param dst A parent QuadratureFunction which receives values from a
   * redecomp GridFunction (src)
   */
  void TransferToParallel( const mfem::QuadratureFunction& src, mfem::QuadratureFunction& dst ) const;

 private:
  /**
   * @brief Grid function transfer object
   */
  std::unique_ptr<const GridFnTransfer> gf_transfer_;
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_REDECOMPTRANSFER_HPP_ */
