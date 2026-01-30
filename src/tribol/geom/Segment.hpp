// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_SEGMENT_HPP_
#define SRC_TRIBOL_GEOM_SEGMENT_HPP_

#include "tribol/common/BasicTypes.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/primal/geometry/Vector.hpp"
#include <cmath>

namespace tribol {

/**
 * @brief A class representing a line segment in D dimensions.
 */
template <int Dim>
class Segment {
 public:
  using PointType = axom::primal::Point<RealT, Dim>;
  using VectorType = axom::primal::Vector<RealT, Dim>;

  /**
   * @brief Constructor
   * @param A Start point
   * @param B End point
   */
  TRIBOL_HOST_DEVICE Segment( const PointType& A, const PointType& B )
    : A_( A ), B_( B ), V_( A_, B_ )
  {
    len_sq_ = V_.squared_norm();
    len_ = V_.norm();
  }

  /**
   * @brief Constructor from interleaved coordinates
   * @param coords Pointer to coordinates [x0, y0, (z0), x1, y1, (z1), ...]
   */
  TRIBOL_HOST_DEVICE Segment( const RealT* coords )
    : Segment( PointType( coords ), PointType( coords + Dim ) )
  {
  }

  /**
   * @brief Returns length of the segment.
   */
  TRIBOL_HOST_DEVICE RealT length() const { return len_; }

  /**
   * @brief Returns squared length of the segment.
   */
  TRIBOL_HOST_DEVICE RealT lengthSq() const { return len_sq_; }

  /**
   * @brief Projects a point P onto the line defined by the segment.
   *
   * @param P The point to project
   * @return RealT Parameter t such that Proj(P) = A + t * (B-A)
   */
  TRIBOL_HOST_DEVICE RealT projectPoint( const PointType& P ) const
  {
    if ( len_sq_ < 1.0e-14 ) {
      return 0.0;
    }
    VectorType AP( A_, P );
    return V_.dot( AP ) / len_sq_;
  }

  /**
   * @brief Projects a point P onto the line defined by the segment along a direction N.
   *
   * @param P The point to project
   * @param N Projection direction
   * @return RealT Parameter t such that Proj(P) = A + t * V
   */
  TRIBOL_HOST_DEVICE RealT projectPoint( const PointType& P, const VectorType& N ) const
  {
    VectorType AP( A_, P );
    auto crossV_N = VectorType::cross_product( V_, N );
    RealT denom = crossV_N.squared_norm();

    if ( denom < 1.0e-14 ) {
      return 0.0;
    }

    auto crossAP_N = VectorType::cross_product( AP, N );
    return crossAP_N.dot( crossV_N ) / denom;
  }

  /**
   * @brief Computes the range [t_min, t_max] of the projection of another segment onto this line.
   *
   * @param other The other segment to project
   * @param t_min Output minimum parameter
   * @param t_max Output maximum parameter
   */
  TRIBOL_HOST_DEVICE void projectSegment( const Segment<Dim>& other, RealT& t_min, RealT& t_max ) const
  {
    RealT t1 = projectPoint( other.A_ );
    RealT t2 = projectPoint( other.B_ );
    t_min = ( t1 < t2 ) ? t1 : t2;
    t_max = ( t1 > t2 ) ? t1 : t2;
  }

  /**
   * @brief Computes the range [t_min, t_max] of the projection of another segment onto this line along a direction N.
   *
   * @param other The other segment to project
   * @param N Projection direction
   * @param t_min Output minimum parameter
   * @param t_max Output maximum parameter
   */
  TRIBOL_HOST_DEVICE void projectSegment( const Segment<Dim>& other, const VectorType& N, RealT& t_min, RealT& t_max ) const
  {
    RealT t1 = projectPoint( other.A_, N );
    RealT t2 = projectPoint( other.B_, N );
    t_min = ( t1 < t2 ) ? t1 : t2;
    t_max = ( t1 > t2 ) ? t1 : t2;
  }

  /**
   * @brief Evaluates the point on the line at parameter t.
   *
   * @param t Parameter
   * @return The point at A + t*V
   */
  TRIBOL_HOST_DEVICE PointType eval( RealT t ) const
  {
    return A_ + ( V_ * t );
  }

 private:
  PointType A_;
  PointType B_;
  VectorType V_;
  RealT len_sq_;
  RealT len_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_SEGMENT_HPP_ */