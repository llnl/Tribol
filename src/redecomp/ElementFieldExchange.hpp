// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_ELEMENTFIELDEXCHANGE_HPP_
#define SRC_REDECOMP_ELEMENTFIELDEXCHANGE_HPP_

#include <memory>
#include <vector>

#include "mfem.hpp"

#include "redecomp/ElementExchangeOperator.hpp"

namespace redecomp {

/** @brief Mesh-free destination storage for exchanged element or quadrature blocks */
struct ExchangedElementField {
  mfem::Vector values;               ///< Concatenated destination blocks
  mfem::Array<int> element_offsets;  ///< Half-open value range for each destination block
  mfem::Array<int> source_ranks;     ///< Parent-mesh owner of each destination block
  mfem::Array<int> source_elements;  ///< Parent-mesh element on source_ranks
};

/**
 * @brief Exchange parent-mesh QuadratureFunction blocks without constructing a destination mesh
 *
 * The constructor snapshots only the source QuadratureSpace layout and the
 * immutable element request schedule. The source QuadratureFunction and its
 * ParMesh remain caller-owned and must outlive this object. Gather() preserves
 * each element block's value ordering, size, source rank, and source element;
 * callers interpret quadrature points through the original source space.
 * AddToSource() additively applies the exact communication transpose; it does
 * not average shared or repeated requests.
 */
class ElementFieldExchange {
 public:
  /** @brief Collectively build a reusable request schedule from the source layout */
  ElementFieldExchange( const mfem::QuadratureFunction& source_layout,
                        std::vector<mfem::Array<int>> source_elements_by_destination );

  /** @brief Gather current source values and overwrite a newly owned destination buffer */
  ExchangedElementField Gather( const mfem::QuadratureFunction& source ) const;

  /** @brief Add scaled destination dual blocks to the caller-owned parent QuadratureFunction */
  void AddToSource( const ExchangedElementField& destination_dual, mfem::QuadratureFunction& source_dual,
                    mfem::real_t scale = 1.0 ) const;

  const ElementExchangePlan& Plan() const { return exchange_.Plan(); }

 private:
  const mfem::QuadratureSpaceBase* source_space_;
  int vector_dimension_;
  std::shared_ptr<const ElementExchangePlan> plan_;
  ElementExchangeOperator exchange_;
  mutable mfem::Vector returned_source_dual_;
};

}  // namespace redecomp

#endif  // SRC_REDECOMP_ELEMENTFIELDEXCHANGE_HPP_
