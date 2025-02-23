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
   */
  virtual void Compute( MeshData& mesh ) = 0;

  /**
   * @brief Gets the Jacobian data for the nodal normals
   *
   * @pre Must call NodalNormal::Compute() first
   *
   * @return MethodData reference storing Jacobian data
   */
  MethodData& getJacobianData() { return elem_jacobians_; }

  /// @overload
  const MethodData& getJacobianData() const { return elem_jacobians_; }

 private:
  /**
   * @brief Stores Jacobian contributions of the nodal normals
   */
  MethodData elem_jacobians_;
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
   */
  void Compute( MeshData& mesh ) override;
};

#ifdef TRIBOL_USE_ENZYME

/**
 * @brief Computes nodal normals by averaging normal evaluated at the node of all connected elements
 */
class VertexAvgNormal : public NodalNormal {
 public:
  /**
   * @brief Constructor
   *
   * @param compute_deriv Computes the Jacobian if true
   */
  VertexAvgNormal( bool compute_deriv = true ) : compute_deriv_( compute_deriv ) {}

  /**
   * @brief Computes nodal normals by averaging normal evaluated at the node of all connected elements
   *
   * @param mesh Mesh data
   */
  void Compute( MeshData& mesh ) override;

 private:
  /**
   * @brief Computes the Jacobian if true
   */
  bool compute_deriv_;
};

// free functions for enzyme

/**
 * @brief Computes the normal direction at all nodes of the element
 *
 * @param [in] x Nodal coordinates for the element (stored by nodes, i.e. [x0, x1, x2, y0, y1, y2, z0, z1, z2])
 * @param [in] xref Reference nodal coordinates for the element (i.e. at t = 0) (stored by nodes)
 * @param [out] n Unit vectors giving the normal direction for each node (stored by nodes)
 * @param [in] num_nodes_per_elem Number of nodes in the element
 */
void ElementVertexAvgNormal( const RealT* x, const RealT* xref, RealT* n, int num_nodes_per_elem );

/**
 * @brief Computes the normal direction and Jacobian at all nodes of the element
 *
 * @param [in] x Nodal coordinates for the element (stored by nodes, i.e. [x0, x1, x2, y0, y1, y2, z0, z1, z2])
 * @param [in] xref Reference nodal coordinates for the element (i.e. at t = 0) (stored by nodes)
 * @param [out] n Unit vectors giving the normal direction for each node (stored by nodes)
 * @param [out] dndx Derivative of the unit normal vectors for each node (size = num_nodes_per_elem^2 x spatial dim^2)
 * @param [in] num_nodes_per_elem Number of nodes in the element
 */
void ElementVertexAvgNormalJacobian( const RealT* x, const RealT* xref, RealT* n, RealT* dndx, int num_nodes_per_elem );

#endif

}  // namespace tribol

#endif /* SRC_GEOM_NODALNORMAL_HPP_ */
