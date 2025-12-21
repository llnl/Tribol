// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_HYPERPLANE_HPP_
#define SRC_TRIBOL_GEOM_HYPERPLANE_HPP_

// Tribol config include
#include "tribol/config.hpp"

// Axom includes
#include "axom/primal.hpp"

// Tribol includes
#include "tribol/common/BasicTypes.hpp"
#include "tribol/geom/Vector.hpp"

namespace tribol {

/**
 * @brief Represents a hyperplane (affine subspace) in N-dimensional space.
 *
 * The `Hyperplane` type models a geometric hyperplane defined by an origin point and a normal vector. It provides
 * basic operations such as projecting a point onto the hyperplane. The template parameter `_VectorT` allows using
 * different vector storage/backing types.
 *
 * @tparam _T Element numeric type for vector components.
 * @tparam _VectorT Vector type used for points and normals (defaults to `Vector<_T>`).
 */
template <typename _T, int _Dim, typename _VectorT = axom::primal::Vector<_T, _Dim>>
class Hyperplane {
 public:
  /// @brief Alias for the vector type used by this hyperplane.
  using VectorT_ = _VectorT;

  /// @brief Scalar value type for components (derived from the vector type).
  using ValueT_ = typename VectorT_::ValueT_;

  static_assert( std::is_same<_T, ValueT_>::value, "Hyperplane must be used with the same type as the vector type" );

  /**
   * @brief Construct a hyperplane from an origin point and a normal vector.
   *
   * @param origin A point on the hyperplane.
   * @param normal A normal vector to the hyperplane (must be non-zero and same dimension as origin).
   */
  TRIBOL_HOST_DEVICE Hyperplane( const VectorT_& origin, const VectorT_& normal ) : origin_( origin ), normal_( normal )
  {
    assert( normal_.norm() > 0.0 );
    assert( origin_.dim() == normal_.dim() );
  }
  /**
   * @brief Move-construct a hyperplane from rvalue origin and normal vectors.
   *
   * @param origin Rvalue reference to a point on the hyperplane (moved into the object).
   * @param normal Rvalue reference to the normal vector (moved into the object).
   */
  TRIBOL_HOST_DEVICE Hyperplane( VectorT_&& origin, VectorT_&& normal )
      : origin_( std::move( origin ) ), normal_( std::move( normal ) )
  {
    assert( normal_.norm() > 0.0 );
    assert( origin_.dim() == normal_.dim() );
  }

  /**
   * @brief Return the dimension of the hyperplane (same as underlying vectors).
   * @return The dimension (number of components) of the origin/normal vectors.
   */
  TRIBOL_HOST_DEVICE constexpr SizeT dim() const { return origin_.dim(); }

  /**
   * @brief Project a point orthogonally onto the hyperplane.
   *
   * The function computes the orthogonal projection of `point` onto the hyperplane defined by `origin_` and
   * `normal_`.
   *
   * @param point The point to project (must have same dimension as the hyperplane).
   * @return The projected point on the hyperplane.
   */
  TRIBOL_HOST_DEVICE VectorT_ projectPoint( const VectorT_& point ) const
  {
    assert( point.dim() == dim() );

    // Calculate the projection of the point onto the hyperplane
    VectorT_ diff = point - origin_;
    ValueT_ dist = diff.dot( normal_ ) / normal_.norm();
    return point - dist * normal_;
  }

 private:
  /// @brief A point that lies on the hyperplane.
  VectorT_ origin_;

  /// @brief The normal vector to the hyperplane (should be non-zero).
  VectorT_ normal_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_HYPERPLANE_HPP_ */
