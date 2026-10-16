// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_ELEMENTEXCHANGEOPERATOR_HPP_
#define SRC_REDECOMP_ELEMENTEXCHANGEOPERATOR_HPP_

#include <memory>

#include "mfem.hpp"

#include "redecomp/ElementExchangePlan.hpp"
#include "shared/math/ParSparseMat.hpp"

namespace redecomp {

/**
 * @brief MPI operator that gathers source element blocks and additively returns element duals
 *
 * The forward action copies every requested source block to its destination.
 * The transpose action returns independent destination contributions and sums
 * them into the corresponding source block.
 */
class ElementExchangeOperator : public mfem::Operator {
 public:
  explicit ElementExchangeOperator( std::shared_ptr<const ElementExchangePlan> plan );

  void Mult( const mfem::Vector& source, mfem::Vector& destination ) const override;
  void MultTranspose( const mfem::Vector& destination_dual, mfem::Vector& source_dual ) const override;
  void AddMultTranspose( const mfem::Vector& destination_dual, mfem::Vector& source_dual,
                         const mfem::real_t scale = 1.0 ) const override;

  shared::ParSparseMat Assemble() const;

  const ElementExchangePlan& Plan() const { return *plan_; }

 private:
  void PackSourceBlocks( const mfem::Vector& source, mfem::Vector& send_buffer ) const;
  void AddReturnedBlocks( const mfem::Vector& receive_buffer, mfem::Vector& source_dual, mfem::real_t scale ) const;

  std::shared_ptr<const ElementExchangePlan> plan_;
  mutable mfem::Vector send_buffer_;
  mutable mfem::Vector receive_buffer_;
};

}  // namespace redecomp

#endif  // SRC_REDECOMP_ELEMENTEXCHANGEOPERATOR_HPP_
