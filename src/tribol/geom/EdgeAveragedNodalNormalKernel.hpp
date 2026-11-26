// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Compute the contributions of one linear surface element to the edge-averaged nodal normal field. */
#ifndef SRC_TRIBOL_GEOM_EDGEAVERAGEDNODALNORMALKERNEL_HPP_
#define SRC_TRIBOL_GEOM_EDGEAVERAGEDNODALNORMALKERNEL_HPP_

#include <cmath>

#include "axom/primal/geometry/Vector.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"

namespace tribol::detail {

/**
 * @brief Compute current normal contributions using reference edge lengths and corner angles.
 * @tparam NumNodes Number of nodes on the linear surface element.
 */
template <IndexT NumNodes>
struct EdgeAveragedNodalNormalKernel {
  static constexpr IndexT spatial_dim = 3;                       ///< Number of spatial coordinates.
  static constexpr IndexT size_coords = NumNodes * spatial_dim;  ///< Number of coordinate values.
  using Coords = MdSpan<const RealT, NumNodes, spatial_dim>;     ///< Read-only coordinates [node][spatial_dim].
  using NodalVectors = MdArray<RealT, NumNodes, spatial_dim>;    ///< Owned nodal coordinates or normals [node][dim].
  using Jacobian = MdArray<RealT, size_coords, size_coords>;     ///< Normal rows and coordinate columns.

  /**
   * @brief Divide each current corner cross product by the magnitude of its reference cross product.
   * @param coords Current element coordinates in boundary element order.
   * @param reference_coords Reference coordinates in the same order.
   * @return Contributions before summing at shared nodes. A degenerate reference corner contributes zero.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static NodalVectors evaluate( Coords coords, Coords reference_coords )
  {
    NodalVectors normals{};
    evaluateInto( coords.data(), reference_coords.data(), normals.data() );
    return normals;
  }

#ifdef TRIBOL_USE_ENZYME
  /**
   * @brief Differentiate the element's contributions while holding its reference coordinates fixed.
   * @param coords Current element coordinates.
   * @param reference_coords Reference element coordinates.
   * @return Normal contribution Jacobian, with zero rows at degenerate reference corners.
   */
  TRIBOL_HOST_DEVICE static Jacobian differentiate( Coords coords, Coords reference_coords )
  {
    Jacobian jacobian{};
    NodalVectors values{}, coord_seed{}, derivatives{};
    for ( IndexT column = 0; column < size_coords; ++column ) {
      coord_seed[column] = 1.0;
      __enzyme_fwddiff<void>( (void*)evaluateInto, TRIBOL_ENZYME_DUP, coords.data(), coord_seed.data(),
                              TRIBOL_ENZYME_CONST, reference_coords.data(), TRIBOL_ENZYME_DUP, values.data(),
                              derivatives.data() );
      for ( IndexT row = 0; row < size_coords; ++row ) {
        jacobian( row, column ) = derivatives[row];
      }
      coord_seed[column] = 0.0;
    }
    return jacobian;
  }
#endif

 private:
  using Vector = axom::primal::Vector<RealT, spatial_dim>;
  /// Minimum sine of the angle between reference edges for a usable cross product.
  static constexpr RealT minimum_reference_corner_sine = 1.0e-14;

  /**
   * @brief Write scaled cross products to caller-provided storage for differentiation.
   * @param coords Current coordinates in node order.
   * @param reference_coords Reference coordinates in node order.
   * @param normals Output normal contributions with the same layout as coordinates.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void evaluateInto( const RealT* coords, const RealT* reference_coords,
                                                                    RealT* normals )
  {
    const Coords current( coords, size_coords );
    const Coords reference( reference_coords, size_coords );
    MdSpan<RealT, NumNodes, spatial_dim> values( normals, size_coords );
    for ( IndexT node = 0; node < NumNodes; ++node ) {
      const IndexT next = ( node + 1 ) % NumNodes;
      const IndexT previous = ( node + NumNodes - 1 ) % NumNodes;
      Vector edge_next{}, edge_previous{}, ref_next{}, ref_previous{};
      for ( IndexT d = 0; d < spatial_dim; ++d ) {
        edge_next[d] = current( next, d ) - current( node, d );
        edge_previous[d] = current( previous, d ) - current( node, d );
        ref_next[d] = reference( next, d ) - reference( node, d );
        ref_previous[d] = reference( previous, d ) - reference( node, d );
      }
      const auto reference_normal = Vector::cross_product( ref_next, ref_previous );
      const RealT magnitude = std::sqrt( reference_normal.squared_norm() );
      const RealT edge_length_product = std::sqrt( ref_next.squared_norm() * ref_previous.squared_norm() );
      const bool valid = std::isfinite( magnitude ) && magnitude > minimum_reference_corner_sine * edge_length_product;
      const auto current_normal = Vector::cross_product( edge_next, edge_previous );
      for ( IndexT d = 0; d < spatial_dim; ++d ) {
        values( node, d ) = valid ? current_normal[d] / magnitude : 0.0;
      }
    }
  }
};

}  // namespace tribol::detail

#endif  // SRC_TRIBOL_GEOM_EDGEAVERAGEDNODALNORMALKERNEL_HPP_
