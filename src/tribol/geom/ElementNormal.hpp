// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_ELEMENTNORMAL_HPP_
#define SRC_TRIBOL_GEOM_ELEMENTNORMAL_HPP_

#include "tribol/common/BasicTypes.hpp"

namespace tribol {

/**
 * @brief Virtual base class to define the interface for element normal calculations
 */
class ElementNormal {
 public:
  /**
   * @brief Destructor
   */
  TRIBOL_HOST_DEVICE virtual ~ElementNormal() {}

  /**
   * @brief Interface for computing an element normal in derived classes
   *
   * @param [in] x Nodal coordinates for the element (stored by nodes, i.e. [x0, x1, x2, y0, y1, y2, z0, z1, z2])
   * @param [in] c Centroid for the element (length = spatial dimension)
   * @param [out] n Unit vector in the normal direction (length = spatial dimension)
   * @param [in] num_nodes Number of nodes in the element
   * @param [out] area Area of the element
   * @return Is face data OK?  true = yes; false = no
   */
  TRIBOL_HOST_DEVICE virtual bool Compute( const RealT* x, const RealT* c, RealT* n, int num_nodes,
                                           RealT& area ) const = 0;
};

/**
 * @brief Computes element normal as the average of normal directions of triangular pallets fanned from the element
 * centroid. This approach applies to arbitrary polygonal elements.
 */
class PalletAvgNormal : public ElementNormal {
 public:
  /**
   * @brief Computes element normal as the average of normal directions of triangular pallets fanned from the element
   * centroid
   *
   * @param [in] x Nodal coordinates for the element (stored by nodes, i.e. [x0, x1, x2, y0, y1, y2, z0, z1, z2])
   * @param [in] c Centroid for the element (length = spatial dimension)
   * @param [out] n Unit vector in the normal direction (length = spatial dimension)
   * @param [in] num_nodes Number of nodes in the element
   * @param [out] area Area of the element
   * @return Is face data OK?  true = yes; false = no
   */
  TRIBOL_HOST_DEVICE bool Compute( const RealT* x, const RealT* c, RealT* n, int num_nodes,
                                   RealT& area ) const override;
};

/**
 * @brief Computes element normal at the origin of the isoparametric element (triangle or quadrilateral)
 */
class ElementCentroidNormal : public ElementNormal {
 public:
  /**
   * @brief Computes element normal at the origin of the isoparametric element
   *
   * @param [in] x Nodal coordinates for the element (stored by nodes, i.e. [x0, x1, x2, y0, y1, y2, z0, z1, z2])
   * @param [in] c Centroid for the element (length = spatial dimension)
   * @param [out] n Unit vector in the normal direction (length = spatial dimension)
   * @param [in] num_nodes Number of nodes in the element (either 3 or 4)
   * @param [out] area Area of the element
   * @return Is face data OK?  true = yes; false = no
   */
  TRIBOL_HOST_DEVICE bool Compute( const RealT* x, const RealT* c, RealT* n, int num_nodes,
                                   RealT& area ) const override;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_ELEMENTNORMAL_HPP_ */
