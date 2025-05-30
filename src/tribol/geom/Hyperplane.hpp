// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_HYPERPLANE_HPP_
#define SRC_TRIBOL_GEOM_HYPERPLANE_HPP_

#include "tribol/geom/Vector.hpp"

namespace tribol {

template <typename T, typename VectorT = Vector<T>>
class Hyperplane {
 public:
  using value_type = typename VectorT::value_type;

  static_assert( std::is_same<T, value_type>::value, "Hyperplane must be used with the same type as the vector type" );

  TRIBOL_HOST_DEVICE Hyperplane( const VectorT& origin, const VectorT& normal ) : origin_( origin ), normal_( normal )
  {
    assert( normal_.norm() > 0.0 );
    assert( origin_.dim() == normal_.dim() );
  }
  TRIBOL_HOST_DEVICE Hyperplane( VectorT&& origin, VectorT&& normal )
      : origin_( std::move( origin ) ), normal_( std::move( normal ) )
  {
    assert( normal_.norm() > 0.0 );
    assert( origin_.dim() == normal_.dim() );
  }

  TRIBOL_HOST_DEVICE constexpr size_t dim() const { return origin_.dim(); }

  TRIBOL_HOST_DEVICE VectorT projectPoint( const VectorT& point ) const
  {
    assert( point.dim() == dim() );

    // Calculate the projection of the point onto the hyperplane
    VectorT diff = point - origin_;
    value_type dist = diff.dot( normal_ ) / normal_.norm();
    return point - dist * normal_;
  }

 private:
  VectorT origin_;
  VectorT normal_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_HYPERPLANE_HPP_ */
