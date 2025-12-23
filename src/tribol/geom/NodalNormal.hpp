// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_NODALNORMAL_HPP_
#define SRC_TRIBOL_GEOM_NODALNORMAL_HPP_

#include "tribol/config.hpp"

#include "tribol/mesh/MeshData.hpp"

namespace tribol {

class MethodData;

/**
 * @brief Virtual base class to define the interface for nodal normal calculations
 */
class NodalNormal {
 public:
  /**
   * @brief Destructor
   */
  virtual ~NodalNormal() {}

  /**
   * @brief Interface for computing and storing nodal normals on a given mesh
   *
   * @param mesh Mesh data
   * @param jacobian_data Method data for storing Jacobian contributions (optional)
   */
  virtual void Compute( MeshData& mesh, MethodData* jacobian_data = nullptr ) = 0;
};

/**
 * @brief Computes nodal normals as the average of connected element normals
 */
class ElementAvgNodalNormal : public NodalNormal {
 public:
  /**
   * @brief Computes nodal normals as the average of connected element normals
   *
   * @param mesh Mesh data
   * @param jacobian_data Method data for storing Jacobian contributions (optional)
   */
  void Compute( MeshData& mesh, MethodData* jacobian_data = nullptr ) override;
};

/**
 * @brief Computes nodal normals by computing the normal on all connected elements at the nodal location and averaging
 *        the normals
 */
class EdgeAvgNodalNormal : public NodalNormal {
 public:
  /**
   * @brief Computes nodal normals by computing the normal on all connected elements at the nodal location and averaging
   *        the normals
   *
   * @param mesh Mesh data
   * @param jacobian_data Method data for storing Jacobian contributions (optional)
   */
  void Compute( MeshData& mesh, MethodData* jacobian_data = nullptr ) override;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_NODALNORMAL_HPP_ */
