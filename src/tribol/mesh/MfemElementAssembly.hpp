// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Shared implementation for transferring packed element blocks to MFEM assembly storage. */
#ifndef SRC_TRIBOL_MESH_MFEMELEMENTASSEMBLY_HPP_
#define SRC_TRIBOL_MESH_MFEMELEMENTASSEMBLY_HPP_
#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <limits>
#include <stdexcept>
#include "tribol/mesh/MfemData.hpp"
#include "tribol/physics/PairContribution.hpp"
namespace tribol {
namespace detail {

/**
 * @brief Copy packed element blocks to the host and append them after any earlier contributions.
 * @tparam RowDofs Dense block row count.
 * @tparam ColumnDofs Dense block column count.
 * @param destination Accumulated host blocks; existing entries and value offsets are preserved.
 * @param blocks Execution-space blocks to copy; not retained.
 * @throws std::length_error Accumulated entries or values exceed the supported index range; nothing is appended.
 */
template <IndexT RowDofs, IndexT ColumnDofs>
void appendElementBlocks( PackedPairJacobianContribs& destination,
                          const ElementBlockBuffer<RowDofs, ColumnDofs>& blocks )
{
  const auto max_size = std::numeric_limits<IndexT>::max();
  if ( blocks.rowElementIds().size() > max_size - destination.row_elem_ids.size() ||
       blocks.values().size() > max_size - destination.jacobian_data.size() ) {
    throw std::length_error( "Accumulated element blocks exceed the supported index range." );
  }
  if ( blocks.rowElementIds().size() == 0 ) {
    return;
  }
  const Array1D<IndexT, MemorySpace::Host> rows( blocks.rowElementIds() );
  const Array1D<IndexT, MemorySpace::Host> columns( blocks.columnElementIds() );
  Array1D<IndexT, MemorySpace::Host> offsets( blocks.valueOffsets() );
  const Array1D<RealT, MemorySpace::Host> values( blocks.values() );
  const IndexT value_offset = destination.jacobian_data.size();
  for ( IndexT& offset : offsets ) {
    offset += value_offset;
  }
  destination.row_elem_ids.append( rows.view() );
  destination.col_elem_ids.append( columns.view() );
  destination.value_offsets.append( offsets.view() );
  destination.jacobian_data.append( values.view() );
}

}  // namespace detail

}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_MESH_MFEMELEMENTASSEMBLY_HPP_
