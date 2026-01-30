// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_GAUSS1DRULE_HPP_
#define SRC_TRIBOL_INTEG_GAUSS1DRULE_HPP_

#include "tribol/integ/IntegrationRule.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/geom/Segment.hpp"

#include <cmath>

namespace tribol {

/**
 * @brief 1D Gauss-Legendre integration rule defined on a line segment.
 *
 * The segment is defined by the end points of the specified Face (1 or 2).
 *
 * @tparam FaceID The face ID (1 or 2) used to define the integration line segment.
 * @tparam PointwiseGapAndNormal Policy type for gap and normal computation.
 * @tparam Dim Spatial dimension (must be 2 or 3).
 */
template <int FaceID, typename PointwiseGapAndNormal, int Dim>
class Gauss1DRule : public IntegrationRule<PointwiseGapAndNormal, Dim> {
 public:
  /**
   * @brief Constructor
   * @param mesh1 View of mesh 1
   * @param mesh2 View of mesh 2
   * @param order Integration order
   */
  Gauss1DRule( const MeshData& mesh1, const MeshData& mesh2, int order, ExecutionMode exec_mode )
      : mesh1_( mesh1 ), mesh2_( mesh2 ), order_( order ), exec_mode_( exec_mode )
  {
  }

  /**
   * @brief Returns the computed integration points.
   */
  const ArrayT<IntegrationPoints>& getRule() const override { return integ_data_; }

  /**
   * @brief Returns the active interface pairs.
   */
  const ArrayT<InterfacePair>& getPairs() const override { return pairs_; }

  // Overrides from IntegrationRule
  void findPairsInContact( ArrayT<InterfacePair>&& pairs, int check_level, PointwiseGapAndNormal& gap_method ) override
  {
    pairs_ = std::move( pairs );

    if ( check_level == 0 ) {
      integ_data_.resize( pairs_.size() );
      return;
    }

    // Check Level 1: Filter pairs based on segment projection overlap
    int num_pairs = pairs_.size();
    int allocId = pairs_.getAllocatorID();

    // Pass 1: Determine active pairs and count them
    ArrayT<int> counts( 1, 1, allocId );
    counts.fill( 0 );

    ArrayT<int> keep_flags( num_pairs, num_pairs, allocId );

    auto pairs_view = pairs_.view();
    auto mesh1_view = const_cast<MeshData&>( mesh1_ ).getView();
    auto mesh2_view = const_cast<MeshData&>( mesh2_ ).getView();
    auto counts_view = counts.view();
    auto keep_view = keep_flags.view();

    forAllExec( exec_mode_, num_pairs,
                [pairs_view, mesh1_view, mesh2_view, counts_view, keep_view] TRIBOL_HOST_DEVICE( IndexT i ) {
                  auto& pair = pairs_view[i];

                  // Determine "Face A" (defines the line) and "Face B" (projects onto it)
                  IndexT elemA = ( FaceID == 1 ) ? pair.m_element_id1 : pair.m_element_id2;
                  IndexT elemB = ( FaceID == 1 ) ? pair.m_element_id2 : pair.m_element_id1;
                  auto& meshA = ( FaceID == 1 ) ? mesh1_view : mesh2_view;
                  auto& meshB = ( FaceID == 1 ) ? mesh2_view : mesh1_view;

                  // Get Coordinates
                  RealT stackedA[2 * 3];
                  meshA.getFaceCoords( elemA, stackedA );
                  Segment<Dim> segA( stackedA );

                  RealT stackedB[2 * 3];
                  meshB.getFaceCoords( elemB, stackedB );
                  Segment<Dim> segB( stackedB );

                  if ( segA.lengthSq() < 1.0e-14 ) {
                    keep_view[i] = 0;
                    return;
                  }

                  RealT tB_min, tB_max;
                  segA.projectSegment( segB, tB_min, tB_max );

                  RealT t_start = ( 0.0 > tB_min ) ? 0.0 : tB_min;
                  RealT t_end = ( 1.0 < tB_max ) ? 1.0 : tB_max;

                  if ( t_end <= t_start ) {
                    keep_view[i] = 0;
                  } else {
                    keep_view[i] = 1;
#ifdef TRIBOL_USE_RAJA
                    RAJA::atomicInc<RAJA::auto_atomic>( &counts_view[0] );
#else
                    counts_view[0]++;
#endif
                  }
                } );

    // Get number of active pairs on host
    ArrayT<int, 1, MemorySpace::Host> counts_host( counts );
    int num_active = counts_host[0];

    // Pass 2: Compact active pairs
    ArrayT<InterfacePair> active_pairs( num_active, num_active, allocId );
    auto active_view = active_pairs.view();

    // Reset counter
    counts.fill( 0 );

    forAllExec( exec_mode_, num_pairs,
                [pairs_view, keep_view, counts_view, active_view] TRIBOL_HOST_DEVICE( IndexT i ) {
                  if ( keep_view[i] ) {
#ifdef TRIBOL_USE_RAJA
                    auto idx = RAJA::atomicInc<RAJA::auto_atomic>( &counts_view[0] );
#else
                    auto idx = counts_view[0]++;
#endif
                    active_view[idx] = pairs_view[i];
                  }
                } );

    // Update pairs and resize integ data
    pairs_ = std::move( active_pairs );
    integ_data_.resize( pairs_.size() );
  }

  void updateRule( PointwiseGapAndNormal& gap_method ) override
  {
    // Precompute Gauss points/weights for reference [-1, 1]
    std::vector<RealT> gp( order_ );
    std::vector<RealT> gw( order_ );
    computeGaussRule( order_, gp.data(), gw.data() );

    // Get viewers for coordinate access
    auto mesh1_view = const_cast<MeshData&>( mesh1_ ).getView();
    auto mesh2_view = const_cast<MeshData&>( mesh2_ ).getView();

    int num_pairs = pairs_.size();
    for ( int i = 0; i < num_pairs; ++i ) {
      auto& pair = pairs_[i];
      auto& data = integ_data_[i];

      data.pair_ = pair;

      IndexT elemA = ( FaceID == 1 ) ? pair.m_element_id1 : pair.m_element_id2;
      IndexT elemB = ( FaceID == 1 ) ? pair.m_element_id2 : pair.m_element_id1;
      auto& meshA = ( FaceID == 1 ) ? mesh1_view : mesh2_view;
      auto& meshB = ( FaceID == 1 ) ? mesh2_view : mesh1_view;

      // Get Coordinates
      RealT stackedA[2 * 3];
      meshA.getFaceCoords( elemA, stackedA );
      Segment<Dim> segA( stackedA );

      RealT stackedB[2 * 3];
      meshB.getFaceCoords( elemB, stackedB );
      Segment<Dim> segB( stackedB );

      if ( segA.lengthSq() < 1.0e-14 ) {
        data.points_.resize( 0 );
        continue;
      }

      RealT tB_min, tB_max;
      segA.projectSegment( segB, tB_min, tB_max );

      RealT t_start = ( 0.0 > tB_min ) ? 0.0 : tB_min;
      RealT t_end = ( 1.0 < tB_max ) ? 1.0 : tB_max;

      if ( t_end <= t_start ) {
        data.points_.resize( 0 );
        continue;
      }

      RealT jacobian = 0.5 * ( t_end - t_start );
      RealT physical_jacobian = segA.length() * jacobian;

      data.points_.resize( order_ );

      for ( int k = 0; k < order_; ++k ) {
        auto& ip = data.points_[k];
        ip.point1_.resize( Dim );
        ip.point2_.resize( Dim );

        RealT xi = gp[k];
        RealT t = 0.5 * ( ( t_end - t_start ) * xi + ( t_end + t_start ) );

        // Physical coordinate on the integration segment (Face A)
        RealT P_A[3] = { 0.0, 0.0, 0.0 };
        auto P_A_Pt = segA.eval( t );
        for ( int d = 0; d < Dim; ++d ) {
          P_A[d] = P_A_Pt[d];
          ip.point1_[d] = P_A_Pt[d];
        }

        // Project to opposite face (Face B)
        RealT projectedB[3] = { 0.0, 0.0, 0.0 };
        gap_method.projectPointToOppositeSurface( pair, FaceID, P_A, projectedB );

        for ( int d = 0; d < Dim; ++d ) {
          ip.point2_[d] = projectedB[d];
        }

        ip.weight_ = gw[k] * physical_jacobian;
      }
    }
  }

 private:
  const MeshData& mesh1_;
  const MeshData& mesh2_;
  int order_;
  ExecutionMode exec_mode_;
  ArrayT<InterfacePair> pairs_;
  ArrayT<IntegrationPoints> integ_data_;

  static void computeGaussRule( int n, RealT* p, RealT* w )
  {
    if ( n == 1 ) {
      p[0] = 0.0;
      w[0] = 2.0;
    } else if ( n == 2 ) {
      RealT val = 1.0 / std::sqrt( 3.0 );
      p[0] = -val;
      w[0] = 1.0;
      p[1] = val;
      w[1] = 1.0;
    } else if ( n == 3 ) {
      RealT val = std::sqrt( 0.6 );
      p[0] = -val;
      w[0] = 5.0 / 9.0;
      p[1] = 0.0;
      w[1] = 8.0 / 9.0;
      p[2] = val;
      w[2] = 5.0 / 9.0;
    } else {
      // n=4
      RealT v1 = std::sqrt( ( 3.0 - 2.0 * std::sqrt( 1.2 ) ) / 7.0 );
      RealT v2 = std::sqrt( ( 3.0 + 2.0 * std::sqrt( 1.2 ) ) / 7.0 );
      RealT w1 = ( 18.0 + std::sqrt( 30.0 ) ) / 36.0;
      RealT w2 = ( 18.0 - std::sqrt( 30.0 ) ) / 36.0;
      p[0] = -v2;
      w[0] = w2;
      p[1] = -v1;
      w[1] = w1;
      p[2] = v1;
      w[2] = w1;
      p[3] = v2;
      w[3] = w2;
    }
  }
};

}  // namespace tribol

#endif /* SRC_TRIBOL_INTEG_GAUSS1DRULE_HPP_ */