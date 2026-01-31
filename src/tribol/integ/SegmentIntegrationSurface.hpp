// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_SEGMENTINTEGRATIONSURFACE_HPP_
#define SRC_TRIBOL_INTEG_SEGMENTINTEGRATIONSURFACE_HPP_

#include "tribol/integ/IntegrationSurface.hpp"
#include "tribol/geom/Segment.hpp"

namespace tribol {

/**
 * @brief An IntegrationSurface defined by a line segment.
 * 
 * @tparam Dim Spatial dimension
 */
template <int Dim>
class SegmentIntegrationSurface : public IntegrationSurface<Dim> {
 public:
  using BaseClass = IntegrationSurface<Dim>;
  using PointType = typename BaseClass::PointType;
  using ParamPointType = typename BaseClass::ParamPointType;

  /**
   * @brief Constructor
   * @param faceID The face ID (0, 1, or 2) associated with this surface.
   * @param segment The geometric segment defining the surface.
   */
  TRIBOL_HOST_DEVICE SegmentIntegrationSurface( int faceID, const Segment<Dim>& segment )
      : face_id_( faceID ), segment_( segment )
  {
  }

  TRIBOL_HOST_DEVICE int getFaceID() const override { return face_id_; }

  TRIBOL_HOST_DEVICE PointType eval( const ParamPointType& param_pt ) const override
  {
    // Segment::eval takes a RealT parameter xi in [-1, 1].
    // ParamPointType is Point<RealT, Dim-1> in [0, 1].
    // For Dim=2, Dim-1=1. So param_pt has 1 component.
    RealT t = param_pt[0];
    RealT xi = 2.0 * t - 1.0;
    return segment_.eval( xi );
  }

 private:
  int face_id_;
  Segment<Dim> segment_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_SEGMENTINTEGRATIONSURFACE_HPP_ */