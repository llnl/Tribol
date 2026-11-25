// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Current three-node quadratic Lagrange edge geometry. */
#ifndef SRC_TRIBOL_GEOM_QUADRATICEDGEGEOMETRY_HPP_
#define SRC_TRIBOL_GEOM_QUADRATICEDGEGEOMETRY_HPP_

#include <cmath>

#include "axom/primal/geometry/Point.hpp"
#include "axom/primal/geometry/Vector.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/BasisFunction.hpp"

namespace tribol {

/** @brief Own a current quadratic edge in endpoint, endpoint, midpoint node order. */
class QuadraticEdgeGeometry {
 public:
  using Basis = BasisFunction<EDGE, QUADRATIC_LAGRANGE>;        ///< Basis defining interpolation and node ordering.
  static constexpr int num_nodes = Basis::size_basis;           ///< Number of finite-element nodes.
  static constexpr int parametric_dim = Basis::parametric_dim;  ///< Number of parametric coordinates.
  static constexpr int spatial_dim = Basis::spatial_dim;        ///< Number of physical coordinates.
  using Point = axom::primal::Point<RealT, spatial_dim>;        ///< Physical point on the edge.
  using Vector = axom::primal::Vector<RealT, spatial_dim>;      ///< Physical tangent or normal.
  using Coords = MdSpan<const RealT, num_nodes, spatial_dim>;   ///< Input coordinates [node][component].

  /** @brief Copy the three current finite-element node coordinates. @param coords Current nodal coordinates. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE explicit QuadraticEdgeGeometry( Coords coords )
  {
    for ( IndexT node = 0; node < num_nodes; ++node ) {
      for ( IndexT d = 0; d < spatial_dim; ++d ) {
        coords_( node, d ) = coords( node, d );
      }
    }
  }

  /** @brief Evaluate the quadratic map. @param parametric_coords Element parametric coordinate in [-1, 1].
   * @return Current point. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Point position( MdSpan<const RealT, parametric_dim> parametric_coords ) const
  {
    MdArray<RealT, num_nodes> values{};
    Basis::eval( parametric_coords, values.view() );
    Point point;
    for ( IndexT d = 0; d < spatial_dim; ++d ) {
      point[d] = 0.0;
      for ( IndexT node = 0; node < num_nodes; ++node ) {
        point[d] += values[node] * coords_( node, d );
      }
    }
    return point;
  }

  /** @brief Evaluate the physical tangent dx/dxi. @param parametric_coords Element parametric coordinate.
   * @return Current tangent. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Vector tangent( MdSpan<const RealT, parametric_dim> parametric_coords ) const
  {
    MdArray<RealT, num_nodes, parametric_dim> derivatives{};
    Basis::evalParametricDerivatives( parametric_coords, derivatives.view() );
    Vector result;
    for ( IndexT d = 0; d < spatial_dim; ++d ) {
      result[d] = 0.0;
      for ( IndexT node = 0; node < num_nodes; ++node ) {
        result[d] += derivatives( node, 0 ) * coords_( node, d );
      }
    }
    return result;
  }

  /** @brief Coefficients of the quadratic map x(xi) = constant + linear * xi + quadratic * xi^2. */
  struct MonomialCoefficients {
    Vector constant;   ///< Position at xi = 0.
    Vector linear;     ///< Tangent dx/dxi at xi = 0.
    Vector quadratic;  ///< Half of the constant second derivative d2x/dxi2.
  };

  /**
   * @brief Convert the nodal coordinates to monomial coefficients in the element parametric coordinate.
   * @return Coefficients reproducing position() exactly.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MonomialCoefficients monomialCoefficients() const
  {
    // Node order follows Basis: endpoint at xi = -1, endpoint at xi = 1, then the midpoint at xi = 0.
    constexpr IndexT first_endpoint = 0;
    constexpr IndexT second_endpoint = 1;
    constexpr IndexT midpoint = 2;
    MonomialCoefficients coefficients;
    for ( IndexT d = 0; d < spatial_dim; ++d ) {
      const RealT first = coords_( first_endpoint, d );
      const RealT second = coords_( second_endpoint, d );
      const RealT middle = coords_( midpoint, d );
      coefficients.constant[d] = middle;
      coefficients.linear[d] = 0.5 * ( second - first );
      coefficients.quadratic[d] = 0.5 * ( first + second ) - middle;
    }
    return coefficients;
  }

  /** @brief Unit normal together with a validity flag. */
  struct UnitNormalResult {
    Vector normal{};     ///< Unit normal oriented (dy,-dx) when valid; zero otherwise.
    bool valid = false;  ///< Whether the local tangent has finite components and nonzero length.
  };

  /**
   * @brief Evaluate the directed unit normal when the local tangent is usable.
   * @param parametric_coords Element parametric coordinate.
   * @return Unit normal and whether the local tangent was usable.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE UnitNormalResult
  unitNormal( MdSpan<const RealT, parametric_dim> parametric_coords ) const
  {
    const auto local_tangent = tangent( parametric_coords );
    const RealT squared_length = local_tangent.dot( local_tangent );
    if ( !std::isfinite( local_tangent[0] ) || !std::isfinite( local_tangent[1] ) || !std::isfinite( squared_length ) ||
         squared_length <= 0.0 ) {
      return {};
    }
    const RealT inverse_length = 1.0 / std::sqrt( squared_length );
    return { Vector{ local_tangent[1] * inverse_length, -local_tangent[0] * inverse_length }, true };
  }

 private:
  MdArray<RealT, num_nodes, spatial_dim> coords_{};  ///< Owned current nodal coordinates.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_GEOM_QUADRATICEDGEGEOMETRY_HPP_
