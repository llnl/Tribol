// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_CLOSESTPOINT2D_HPP_
#define SRC_TRIBOL_GEOM_CLOSESTPOINT2D_HPP_

#include "tribol/geom/PointwiseGapAndNormal.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/geom/Segment.hpp"

namespace tribol {

/**
 * @brief Computes pointwise gap and normal based on the closest point projection for 2D line segments.
 *
 * @tparam SourceFaceID The face ID (1 or 2) from which the projection originates.
 *                      We project FROM a point on SourceFaceID TO the other face.
 * @tparam IntegrationSurface Type representing the integration surface.
 */
template <int SourceFaceID, typename IntegrationSurface>
class ClosestPoint2D : public PointwiseGapAndNormal<2, ConstantNormalEvaluator<2>, IntegrationSurface> {
 public:
  static_assert( SourceFaceID == 1 || SourceFaceID == 2, "ClosestPoint2D: SourceFaceID must be 1 or 2." );

  using BaseClass = PointwiseGapAndNormal<2, ConstantNormalEvaluator<2>, IntegrationSurface>;
  using PointType = typename BaseClass::PointType;
  using VectorType = typename BaseClass::VectorType;
  using ParamPointType = typename BaseClass::ParamPointType;
  using NormalEvaluator = typename BaseClass::NormalEvaluator;

  /**
   * @brief Constructor
   * @param mesh1 Mesh data for face 1
   * @param mesh2 Mesh data for face 2
   */
  ClosestPoint2D( const MeshData& mesh1, const MeshData& mesh2 )
      : mesh1_( const_cast<MeshData&>( mesh1 ).getView() ), mesh2_( const_cast<MeshData&>( mesh2 ).getView() )
  {
  }

  TRIBOL_HOST_DEVICE void computeGapVectorAndNormal( const InterfacePair& pair, const IntegrationSurface& surface,
                                                     const ParamPointType& param_pt, VectorType& gap,
                                                     VectorType& normal ) const override
  {
    const auto& meshSource = ( SourceFaceID == 1 ) ? mesh1_ : mesh2_;
    const auto& meshTarget = ( SourceFaceID == 1 ) ? mesh2_ : mesh1_;

    IndexT elemSource = ( SourceFaceID == 1 ) ? pair.m_element_id1 : pair.m_element_id2;
    IndexT elemTarget = ( SourceFaceID == 1 ) ? pair.m_element_id2 : pair.m_element_id1;

    // Get Coordinates
    RealT sourceCoords[4];
    RealT targetCoords[4];

    meshSource.getFaceCoords( elemSource, sourceCoords );
    meshTarget.getFaceCoords( elemTarget, targetCoords );

    Segment<2> segSource( sourceCoords );
    Segment<2> segTarget( targetCoords );

    // Inline Normal Calculation
    // Normal = -N_target = (Ty, -Tx) where T = B - A
    auto P_tgt_A = segTarget.eval( -1.0 );
    auto P_tgt_B = segTarget.eval( 1.0 );
    VectorType T_tgt( P_tgt_A, P_tgt_B );

    normal[0] = T_tgt[1];
    normal[1] = -T_tgt[0];
    if ( normal.squared_norm() > 1.0e-28 ) {
      normal = normal.unitVector();
    } else {
      normal = VectorType( 0.0 );
    }

    PointType P_source, P_target;
    bool onSegment = false;
    constexpr RealT tol = 1.0e-12;

    int faceID = surface.getFaceID();

    if ( faceID == SourceFaceID ) {
      // Case 1: Point is on Source Face
      P_source = surface.eval( param_pt );

      // Project onto infinite line of target
      RealT xi_proj = segTarget.projectPoint( P_source );  // Returns xi in [-1, 1] (or outside)
      P_target = segTarget.eval( xi_proj );

      // Check if projection lands on target segment (xi_proj in [-1, 1])
      if ( xi_proj >= -1.0 - tol && xi_proj <= 1.0 + tol ) {
        onSegment = true;
      }
    } else {
      // Case 2: Point is on Target Face
      P_target = surface.eval( param_pt );

      auto P_src_A = segSource.eval( -1.0 );
      auto P_src_B = segSource.eval( 1.0 );
      VectorType V_src( P_src_A, P_src_B );

      RealT denom = V_src.dot( T_tgt );

      if ( std::abs( denom ) > 1.0e-14 ) {
        VectorType AP( P_src_A, P_target );
        RealT s = AP.dot( T_tgt ) / denom;  // s is t in [0, 1] on source
        RealT xi_src = 2.0 * s - 1.0;
        P_source = segSource.eval( xi_src );
        onSegment = true;
      } else {
        onSegment = false;
        P_source = P_target;
      }
    }

    if ( onSegment ) {
      gap = VectorType( P_source, P_target );
    } else {
      gap = VectorType( 0.0 );
    }
  }

  TRIBOL_HOST_DEVICE void computeGapVector( const InterfacePair& pair, const IntegrationSurface& surface,
                                            const ParamPointType& param_pt, VectorType& gap ) const override
  {
    VectorType normal;  // Unused
    computeGapVectorAndNormal( pair, surface, param_pt, gap, normal );
  }

  TRIBOL_HOST_DEVICE void computeNormal( const InterfacePair& pair, const IntegrationSurface& /*surface*/,
                                         const ParamPointType& /*param_pt*/, VectorType& normal ) const override
  {
    const auto& meshTarget = ( SourceFaceID == 1 ) ? mesh2_ : mesh1_;
    IndexT elemTarget = ( SourceFaceID == 1 ) ? pair.m_element_id2 : pair.m_element_id1;

    RealT targetCoords[4];
    meshTarget.getFaceCoords( elemTarget, targetCoords );
    Segment<2> segTarget( targetCoords );

    auto P_tgt_A = segTarget.eval( -1.0 );
    auto P_tgt_B = segTarget.eval( 1.0 );
    VectorType T_tgt( P_tgt_A, P_tgt_B );

    // Normal = -N_target = (Ty, -Tx)
    normal[0] = T_tgt[1];
    normal[1] = -T_tgt[0];
    if ( normal.squared_norm() > 1.0e-28 ) {
      normal = normal.unitVector();
    } else {
      normal = VectorType( 0.0 );
    }
  }

  TRIBOL_HOST_DEVICE NormalEvaluator getNormalEvaluator( const InterfacePair& pair,
                                                         const IntegrationSurface& surface ) const override
  {
    VectorType const_normal;
    this->computeNormal( pair, surface, ParamPointType(), const_normal );
    return NormalEvaluator( const_normal );
  }

 private:
  MeshData::Viewer mesh1_;
  MeshData::Viewer mesh2_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_CLOSESTPOINT2D_HPP_ */
