// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_INTEGRATIONSURFACE_HPP_
#define SRC_TRIBOL_INTEG_INTEGRATIONSURFACE_HPP_

#include "tribol/common/BasicTypes.hpp"
#include "axom/primal/geometry/Point.hpp"

namespace tribol {

/**
 * @brief Base class for an integration surface used in pointwise gap and normal computations.
 * 
 * @tparam Dim Spatial dimension
 */
template <int Dim>
class IntegrationSurface {
 public:
  using PointType = axom::primal::Point<RealT, Dim>;
  using ParamPointType = axom::primal::Point<RealT, Dim - 1>;

  TRIBOL_HOST_DEVICE virtual ~IntegrationSurface() = default;

  /**
   * @brief Returns the face ID (1 or 2) associated with this surface.
   * 
   * @return int 1 or 2 if the surface corresponds to that face in a given InterfacePair, 
   *             0 otherwise.
   */
  TRIBOL_HOST_DEVICE virtual int getFaceID() const = 0;

  /**
   * @brief Evaluates the physical coordinates of a point on the surface given its parameter-space coordinates.
   * 
   * @param param_pt Parameter-space coordinates
   * @return PointType Physical-space coordinates
   */
  TRIBOL_HOST_DEVICE virtual PointType eval( const ParamPointType& param_pt ) const = 0;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_INTEG_INTEGRATIONSURFACE_HPP_ */