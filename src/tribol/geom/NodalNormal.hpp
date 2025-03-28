// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_GEOM_NODALNORMAL_HPP_
#define SRC_GEOM_NODALNORMAL_HPP_

#include "tribol/config.hpp"

#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"

namespace tribol {

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
class ElementAvgNormal : public NodalNormal {
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
 * @brief Computes nodal normals by averaging normal evaluated at the node of all connected elements
 */
class VertexAvgNormal : public NodalNormal {
 public:
  /**
   * @brief Computes nodal normals by averaging normal evaluated at the node of all connected elements
   *
   * @param mesh Mesh data
   * @param jacobian_data Method data for storing Jacobian contributions (optional)
   */
  void Compute( MeshData& mesh, MethodData* jacobian_data = nullptr ) override;
};

}  // namespace tribol

#endif /* SRC_GEOM_NODALNORMAL_HPP_ */
