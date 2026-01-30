// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_POINTWISEGAPANDNORMAL_HPP_
#define SRC_TRIBOL_GEOM_POINTWISEGAPANDNORMAL_HPP_

#include "tribol/common/BasicTypes.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/primal/geometry/Vector.hpp"

#include <functional>

namespace tribol {

template <int Dim>
struct ConstantNormalEvaluator {
  using VectorType = axom::primal::Vector<RealT, Dim>;
  using ParamPointType = axom::primal::Point<RealT, Dim - 1>;

  VectorType m_normal;

  TRIBOL_HOST_DEVICE ConstantNormalEvaluator( const VectorType& n ) : m_normal( n ) {}

  TRIBOL_HOST_DEVICE void operator()( const ParamPointType& /*pt*/, VectorType& normal ) const
  {
    normal = m_normal;
  }
};

/**
 * @brief Abstract base class for pointwise gap and normal computation.
 * 
 * @tparam Dim Spatial dimension
 * @tparam NormalEvaluatorType Functor type for evaluating the normal on device
 */
template <int Dim, typename NormalEvaluatorType>
class PointwiseGapAndNormal {
 public:
  static_assert( Dim >= 1 && Dim <= 3, "PointwiseGapAndNormal: Dim must be 1, 2, or 3." );

  using VectorType = axom::primal::Vector<RealT, Dim>;
  using PointType = axom::primal::Point<RealT, Dim>;
  using ParamPointType = axom::primal::Point<RealT, Dim - 1>;

  using GapNormalEvaluator = std::function<void( const ParamPointType& param_pt, VectorType& gap, VectorType& normal )>;
  using GapEvaluator = std::function<void( const ParamPointType& param_pt, VectorType& gap )>;
  using NormalEvaluator = NormalEvaluatorType;

  virtual ~PointwiseGapAndNormal() = default;

  /**
   * @brief Computes the gap vector and normal for a specific pair and parameter point.
   *
   * @param pair The interface pair.
   * @param faceID The face ID (1 or 2) on which the parameter point lies.
   * @param param_pt The isoparametric coordinates on the specified face.
   * @param gap Output gap vector.
   * @param normal Output normal vector.
   */
  virtual void computeGapVectorAndNormal( const InterfacePair& pair, int faceID, const ParamPointType& param_pt,
                                          VectorType& gap, VectorType& normal ) const = 0;

  /**
   * @brief Computes only the gap vector for a specific pair and parameter point.
   *
   * @param pair The interface pair.
   * @param faceID The face ID (1 or 2) on which the parameter point lies.
   * @param param_pt The isoparametric coordinates on the specified face.
   * @param gap Output gap vector.
   */
  virtual void computeGapVector( const InterfacePair& pair, int faceID, const ParamPointType& param_pt,
                                 VectorType& gap ) const = 0;

  /**
   * @brief Computes only the normal for a specific pair and parameter point.
   *
   * @param pair The interface pair.
   * @param faceID The face ID (1 or 2) on which the parameter point lies.
   * @param param_pt The isoparametric coordinates on the specified face.
   * @param normal Output normal vector.
   */
  virtual void computeNormal( const InterfacePair& pair, int faceID, const ParamPointType& param_pt,
                              VectorType& normal ) const = 0;

  /**
   * @brief Returns a callable object that computes gap vector and normal for a specific pair.
   */
  virtual GapNormalEvaluator getGapNormalEvaluator( const InterfacePair& pair, int faceID ) const
  {
    return [this, pair, faceID]( const ParamPointType& param_pt, VectorType& gap, VectorType& normal ) {
      this->computeGapVectorAndNormal( pair, faceID, param_pt, gap, normal );
    };
  }

  /**
   * @brief Returns a callable object that computes only the gap vector for a specific pair.
   */
  virtual GapEvaluator getGapVectorEvaluator( const InterfacePair& pair, int faceID ) const
  {
    return [this, pair, faceID]( const ParamPointType& param_pt, VectorType& gap ) {
      this->computeGapVector( pair, faceID, param_pt, gap );
    };
  }

  /**
   * @brief Returns a callable object that computes only the normal for a specific pair.
   */
  virtual NormalEvaluator getNormalEvaluator( const InterfacePair& pair, int faceID ) const = 0;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_POINTWISEGAPANDNORMAL_HPP_ */