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
 */
template <int SourceFaceID>
class ClosestPoint2D : public PointwiseGapAndNormal<2, ConstantNormalEvaluator<2>> {
 public:
  static_assert( SourceFaceID == 1 || SourceFaceID == 2, "ClosestPoint2D: SourceFaceID must be 1 or 2." );

  using BaseClass = PointwiseGapAndNormal<2, ConstantNormalEvaluator<2>>;
  using PointType = typename BaseClass::PointType;
  using VectorType = typename BaseClass::VectorType;
  using ParamPointType = typename BaseClass::ParamPointType;
  using NormalEvaluator = typename BaseClass::NormalEvaluator;

  /**
   * @brief Constructor
   * @param mesh1 Mesh data for face 1
   * @param mesh2 Mesh data for face 2
   */
  ClosestPoint2D( const MeshData& mesh1, const MeshData& mesh2 ) : mesh1_( mesh1 ), mesh2_( mesh2 ) {}

  void computeGapVectorAndNormal( const InterfacePair& pair, int faceID, const ParamPointType& param_pt,
                                  VectorType& gap, VectorType& normal ) const override
  {
    const MeshData& meshSource = ( SourceFaceID == 1 ) ? mesh1_ : mesh2_;
    const MeshData& meshTarget = ( SourceFaceID == 1 ) ? mesh2_ : mesh1_;
    
    IndexT elemSource = ( SourceFaceID == 1 ) ? pair.m_element_id1 : pair.m_element_id2;
    IndexT elemTarget = ( SourceFaceID == 1 ) ? pair.m_element_id2 : pair.m_element_id1;

    // Get Viewers
    auto viewSource = const_cast<MeshData&>( meshSource ).getView();
    auto viewTarget = const_cast<MeshData&>( meshTarget ).getView();

    // Get Coordinates
    RealT sourceCoords[4];
    RealT targetCoords[4];
    
    viewSource.getFaceCoords( elemSource, sourceCoords );
    viewTarget.getFaceCoords( elemTarget, targetCoords );

    Segment<2> segSource( sourceCoords );
    Segment<2> segTarget( targetCoords );

    // Inline Normal Calculation
    // Normal = -N_target = (Ty, -Tx) where T = B - A
    auto P_tgt_A = segTarget.eval(0.0);
    auto P_tgt_B = segTarget.eval(1.0);
    VectorType T_tgt( P_tgt_A, P_tgt_B );
    
    normal[0] = T_tgt[1];
    normal[1] = -T_tgt[0];
    if ( normal.squared_norm() > 1.0e-28 ) {
      normal = normal.unitVector();
    } else {
      normal = VectorType(0.0); 
    }

    PointType P_source, P_target;
    RealT t_proj = 0.0;
    bool onSegment = false;
    constexpr RealT tol = 1.0e-12;

    if ( faceID == SourceFaceID ) {
      // Case 1: Point is on Source Face
      RealT t_src = param_pt[0];
      P_source = segSource.eval( t_src );

      // Project onto infinite line of target
      t_proj = segTarget.projectPoint( P_source ); // Unclamped
      P_target = segTarget.eval( t_proj );
      
      // Check if projection lands on target segment
      if ( t_proj >= -tol && t_proj <= 1.0 + tol ) {
        onSegment = true;
      }
    } else {
      // Case 2: Point is on Target Face
      RealT t_tgt = param_pt[0];
      P_target = segTarget.eval( t_tgt );
      
      auto P_src_A = segSource.eval(0.0);
      auto P_src_B = segSource.eval(1.0);
      VectorType V_src( P_src_A, P_src_B );
      
      RealT denom = V_src.dot( T_tgt );
      
      if ( std::abs(denom) > 1.0e-14 ) {
        VectorType AP( P_src_A, P_target );
        RealT s = AP.dot( T_tgt ) / denom;
        P_source = segSource.eval( s );
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

  void computeGapVector( const InterfacePair& pair, int faceID, const ParamPointType& param_pt,
                         VectorType& gap ) const override
  {
    VectorType normal; // Unused
    computeGapVectorAndNormal( pair, faceID, param_pt, gap, normal );
  }

  void computeNormal( const InterfacePair& pair, int /*faceID*/, const ParamPointType& /*param_pt*/,
                      VectorType& normal ) const override
  {
    const MeshData& meshTarget = ( SourceFaceID == 1 ) ? mesh2_ : mesh1_;
    IndexT elemTarget = ( SourceFaceID == 1 ) ? pair.m_element_id2 : pair.m_element_id1;
    auto viewTarget = const_cast<MeshData&>( meshTarget ).getView();

    RealT targetCoords[4];
    viewTarget.getFaceCoords( elemTarget, targetCoords );
    Segment<2> segTarget( targetCoords );

    auto P_tgt_A = segTarget.eval(0.0);
    auto P_tgt_B = segTarget.eval(1.0);
    VectorType T_tgt( P_tgt_A, P_tgt_B );
    
    // Normal = -N_target = (Ty, -Tx)
    normal[0] = T_tgt[1];
    normal[1] = -T_tgt[0];
    if ( normal.squared_norm() > 1.0e-28 ) {
      normal = normal.unitVector();
    } else {
      normal = VectorType(0.0); 
    }
  }

  NormalEvaluator getNormalEvaluator( const InterfacePair& pair, int faceID ) const override
  {
    VectorType const_normal;
    this->computeNormal( pair, faceID, ParamPointType(), const_normal );
    return NormalEvaluator( const_normal );
  }

 private:
  const MeshData& mesh1_;
  const MeshData& mesh2_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_CLOSESTPOINT2D_HPP_ */