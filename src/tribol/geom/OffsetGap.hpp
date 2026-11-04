// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Add a fixed residual gap to a geometric gap definition by subtracting it from the calculated separation.
 */
#ifndef SRC_TRIBOL_GEOM_OFFSETGAP_HPP_
#define SRC_TRIBOL_GEOM_OFFSETGAP_HPP_

#include <utility>
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Enzyme.hpp"

namespace tribol {
/**
 * @brief Own a geometric gap and the fixed offset subtracted from its value.
 *
 * Geometry arguments remain active in differentiation; the owned offset is fixed configuration. The underlying gap's
 * geometry preconditions apply. No geometry or calculated values are retained.
 * @tparam GapDefinition Callable returning a geometric gap from a point and any prepared geometry arguments.
 */
template <typename GapDefinition>
struct OffsetGap {
  GapDefinition gap;          ///< Owned geometric gap definition.
  RealT residual_gap{ 0.0 };  ///< Fixed offset subtracted from the geometric gap; must be finite.
  /**
   * @brief Evaluate the geometric gap and subtract the configured residual gap.
   * @tparam Arguments Point and optional prepared geometry argument types.
   * @param arguments Geometry arguments forwarded to the underlying gap.
   * @return Shifted gap, preserving the underlying gap's failure/precondition contract.
   */
  template <typename... Arguments>
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT operator()( Arguments&&... arguments ) const
  {
    return gap( std::forward<Arguments>( arguments )... ) - residual_gap;
  }
};
}  // namespace tribol
#endif  // SRC_TRIBOL_GEOM_OFFSETGAP_HPP_
