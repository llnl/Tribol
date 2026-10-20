// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_BASISFUNCTION_HPP_
#define SRC_TRIBOL_INTEG_BASISFUNCTION_HPP_

#include <cmath>

#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdSpan.hpp"
#include "tribol/common/Parameters.hpp"

namespace tribol {

/**
 * @brief Describes the dimensions and local-node ordering for a supported contact face element
 *
 * The local-node ordering is shared by MeshData connectivity and BasisFunction output arrays. Parent-node coordinates
 * use the normalized interval and square [-1, 1]. The linear triangle uses the standard simplex with vertices (0, 0),
 * (1, 0), and (0, 1). The spatial dimension is the ambient contact dimension: two for edges and three for triangles
 * and quadrilaterals.
 *
 * @tparam Element Contact face shape
 * @tparam Order Lagrange order represented by the element connectivity
 */
template <InterfaceElementType Element, FaceOrderType Order>
struct FaceElementTraits;

template <>
struct FaceElementTraits<LINEAR_EDGE, LINEAR> {
  static constexpr int node_count = 2;
  static constexpr int parent_dimension = 1;
  static constexpr int spatial_dimension = 2;

  /**
   * @brief Get one node's coordinate in the normalized parent segment
   *
   * @param local_node Local node index in MeshData connectivity order
   * @param parent_coordinates Output array with length parent_dimension
   *
   * @pre local_node is in [0, node_count)
   */
  TRIBOL_HOST_DEVICE static void GetParentNodeCoordinates( int local_node,
                                                           MdSpan<RealT, parent_dimension> parent_coordinates )
  {
    parent_coordinates[0] = local_node == 0 ? -1.0 : 1.0;
  }
};

template <>
struct FaceElementTraits<LINEAR_EDGE, QUADRATIC_LAGRANGE> {
  static constexpr int node_count = 3;
  static constexpr int parent_dimension = 1;
  static constexpr int spatial_dimension = 2;

  /**
   * @brief Get one node's coordinate in the normalized parent segment
   *
   * The two vertices precede the midpoint to match MFEM H1 local-node ordering.
   *
   * @param local_node Local node index in MeshData connectivity order
   * @param parent_coordinates Output array with length parent_dimension
   *
   * @pre local_node is in [0, node_count)
   */
  TRIBOL_HOST_DEVICE static void GetParentNodeCoordinates( int local_node,
                                                           MdSpan<RealT, parent_dimension> parent_coordinates )
  {
    parent_coordinates[0] = local_node == 0 ? -1.0 : ( local_node == 1 ? 1.0 : 0.0 );
  }
};

template <>
struct FaceElementTraits<LINEAR_TRIANGLE, LINEAR> {
  static constexpr int node_count = 3;
  static constexpr int parent_dimension = 2;
  static constexpr int spatial_dimension = 3;

  /**
   * @brief Get one vertex coordinate in the standard parent triangle
   *
   * @param local_node Local node index in MeshData connectivity order
   * @param parent_coordinates Output array with length parent_dimension
   *
   * @pre local_node is in [0, node_count)
   */
  TRIBOL_HOST_DEVICE static void GetParentNodeCoordinates( int local_node,
                                                           MdSpan<RealT, parent_dimension> parent_coordinates )
  {
    parent_coordinates[0] = local_node == 1 ? 1.0 : 0.0;
    parent_coordinates[1] = local_node == 2 ? 1.0 : 0.0;
  }
};

template <>
struct FaceElementTraits<LINEAR_QUAD, LINEAR> {
  static constexpr int node_count = 4;
  static constexpr int parent_dimension = 2;
  static constexpr int spatial_dimension = 3;

  /**
   * @brief Get one vertex coordinate in the normalized parent square
   *
   * @param local_node Local node index in MeshData connectivity order
   * @param parent_coordinates Output array with length parent_dimension
   *
   * @pre local_node is in [0, node_count)
   */
  TRIBOL_HOST_DEVICE static void GetParentNodeCoordinates( int local_node,
                                                           MdSpan<RealT, parent_dimension> parent_coordinates )
  {
    parent_coordinates[0] = local_node == 0 || local_node == 3 ? -1.0 : 1.0;
    parent_coordinates[1] = local_node < 2 ? -1.0 : 1.0;
  }
};

template <>
struct FaceElementTraits<LINEAR_QUAD, QUADRATIC_LAGRANGE> {
  static constexpr int node_count = 9;
  static constexpr int parent_dimension = 2;
  static constexpr int spatial_dimension = 3;

  /**
   * @brief Get one node coordinate in the normalized parent square
   *
   * The four vertices are followed by the midpoints of edges (0,1), (1,2), (2,3), and (3,0), then the center. This is
   * the local-node ordering used by MFEM H1 quadrilateral elements.
   *
   * @param local_node Local node index in MeshData connectivity order
   * @param parent_coordinates Output array with length parent_dimension
   *
   * @pre local_node is in [0, node_count)
   */
  TRIBOL_HOST_DEVICE static void GetParentNodeCoordinates( int local_node,
                                                           MdSpan<RealT, parent_dimension> parent_coordinates )
  {
    constexpr RealT parent_node_coordinates[node_count][parent_dimension] = {
        { -1.0, -1.0 }, { 1.0, -1.0 }, { 1.0, 1.0 },  { -1.0, 1.0 }, { 0.0, -1.0 },
        { 1.0, 0.0 },   { 0.0, 1.0 },  { -1.0, 0.0 }, { 0.0, 0.0 } };
    parent_coordinates[0] = parent_node_coordinates[local_node][0];
    parent_coordinates[1] = parent_node_coordinates[local_node][1];
  }
};

/**
 * @brief Get the number of local nodes for a supported face shape and order
 *
 * This runtime entry point uses the same FaceElementTraits specializations as BasisFunction and MeshData. Unsupported
 * combinations return zero so the caller can report an error in the appropriate execution context.
 */
TRIBOL_HOST_DEVICE inline int GetFaceNodeCount( InterfaceElementType element, FaceOrderType order )
{
  if ( element == LINEAR_EDGE ) {
    if ( order == LINEAR ) {
      return FaceElementTraits<LINEAR_EDGE, LINEAR>::node_count;
    }
    if ( order == QUADRATIC_LAGRANGE ) {
      return FaceElementTraits<LINEAR_EDGE, QUADRATIC_LAGRANGE>::node_count;
    }
  }
  if ( element == LINEAR_TRIANGLE && order == LINEAR ) {
    return FaceElementTraits<LINEAR_TRIANGLE, LINEAR>::node_count;
  }
  if ( element == LINEAR_QUAD ) {
    if ( order == LINEAR ) {
      return FaceElementTraits<LINEAR_QUAD, LINEAR>::node_count;
    }
    if ( order == QUADRATIC_LAGRANGE ) {
      return FaceElementTraits<LINEAR_QUAD, QUADRATIC_LAGRANGE>::node_count;
    }
  }
  return 0;
}

namespace detail {

TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void EvalLinearLagrange1D( RealT parent_coordinate,
                                                                          MdSpan<RealT, 2> values )
{
  values[0] = 0.5 * ( 1.0 - parent_coordinate );
  values[1] = 0.5 * ( 1.0 + parent_coordinate );
}

TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void EvalLinearLagrange1DDerivatives( MdSpan<RealT, 2> derivatives )
{
  derivatives[0] = -0.5;
  derivatives[1] = 0.5;
}

TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void EvalQuadraticLagrange1D( RealT parent_coordinate,
                                                                             MdSpan<RealT, 3> values )
{
  values[0] = 0.5 * parent_coordinate * ( parent_coordinate - 1.0 );
  values[1] = 0.5 * parent_coordinate * ( parent_coordinate + 1.0 );
  values[2] = 1.0 - parent_coordinate * parent_coordinate;
}

TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void EvalQuadraticLagrange1DDerivatives( RealT parent_coordinate,
                                                                                        MdSpan<RealT, 3> derivatives )
{
  derivatives[0] = parent_coordinate - 0.5;
  derivatives[1] = parent_coordinate + 0.5;
  derivatives[2] = -2.0 * parent_coordinate;
}

}  // namespace detail

/**
 * @brief Evaluates the complete Lagrange basis for one supported contact face element
 *
 * Basis values and derivatives use the local-node ordering defined by FaceElementTraits. Parent derivatives are stored
 * node-major as parent_derivatives[node * parent_dimension + parent_component].
 *
 * Supported specializations provide Eval(parent_coordinates, basis_values), where the output has node_count entries,
 * and EvalParentDerivatives(parent_coordinates, parent_derivatives), where the output has node_count * parent_dimension
 * entries. Both methods use caller-provided arrays so contact kernels can evaluate bases without allocation.
 */
template <InterfaceElementType Element, FaceOrderType Order>
struct BasisFunction;

template <>
struct BasisFunction<LINEAR_EDGE, LINEAR> : FaceElementTraits<LINEAR_EDGE, LINEAR> {
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void Eval( MdSpan<const RealT, parent_dimension> parent_coordinates,
                                                            MdSpan<RealT, node_count> basis_values )
  {
    detail::EvalLinearLagrange1D( parent_coordinates[0], basis_values );
  }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void EvalParentDerivatives(
      MdSpan<const RealT, parent_dimension>, MdSpan<RealT, node_count, parent_dimension> parent_derivatives )
  {
    parent_derivatives( 0, 0 ) = -0.5;
    parent_derivatives( 1, 0 ) = 0.5;
  }
};

template <>
struct BasisFunction<LINEAR_EDGE, QUADRATIC_LAGRANGE> : FaceElementTraits<LINEAR_EDGE, QUADRATIC_LAGRANGE> {
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void Eval( MdSpan<const RealT, parent_dimension> parent_coordinates,
                                                            MdSpan<RealT, node_count> basis_values )
  {
    detail::EvalQuadraticLagrange1D( parent_coordinates[0], basis_values );
  }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void EvalParentDerivatives(
      MdSpan<const RealT, parent_dimension> parent_coordinates,
      MdSpan<RealT, node_count, parent_dimension> parent_derivatives )
  {
    parent_derivatives( 0, 0 ) = parent_coordinates[0] - 0.5;
    parent_derivatives( 1, 0 ) = parent_coordinates[0] + 0.5;
    parent_derivatives( 2, 0 ) = -2.0 * parent_coordinates[0];
  }
};

template <>
struct BasisFunction<LINEAR_TRIANGLE, LINEAR> : FaceElementTraits<LINEAR_TRIANGLE, LINEAR> {
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void Eval( MdSpan<const RealT, parent_dimension> parent_coordinates,
                                                            MdSpan<RealT, node_count> basis_values )
  {
    basis_values[0] = 1.0 - parent_coordinates[0] - parent_coordinates[1];
    basis_values[1] = parent_coordinates[0];
    basis_values[2] = parent_coordinates[1];
  }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void EvalParentDerivatives(
      MdSpan<const RealT, parent_dimension>, MdSpan<RealT, node_count, parent_dimension> parent_derivatives )
  {
    parent_derivatives( 0, 0 ) = -1.0;
    parent_derivatives( 0, 1 ) = -1.0;
    parent_derivatives( 1, 0 ) = 1.0;
    parent_derivatives( 1, 1 ) = 0.0;
    parent_derivatives( 2, 0 ) = 0.0;
    parent_derivatives( 2, 1 ) = 1.0;
  }
};

template <>
struct BasisFunction<LINEAR_QUAD, LINEAR> : FaceElementTraits<LINEAR_QUAD, LINEAR> {
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void Eval( MdSpan<const RealT, parent_dimension> parent_coordinates,
                                                            MdSpan<RealT, node_count> basis_values )
  {
    RealT x_values[2];
    RealT y_values[2];
    detail::EvalLinearLagrange1D( parent_coordinates[0], x_values );
    detail::EvalLinearLagrange1D( parent_coordinates[1], y_values );

    // Map tensor products into the face-node ordering shared by MeshData instead of exposing tensor-array ordering.
    basis_values[0] = x_values[0] * y_values[0];
    basis_values[1] = x_values[1] * y_values[0];
    basis_values[2] = x_values[1] * y_values[1];
    basis_values[3] = x_values[0] * y_values[1];
  }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void EvalParentDerivatives(
      MdSpan<const RealT, parent_dimension> parent_coordinates,
      MdSpan<RealT, node_count, parent_dimension> parent_derivatives )
  {
    RealT x_values[2];
    RealT y_values[2];
    RealT derivatives[2];
    detail::EvalLinearLagrange1D( parent_coordinates[0], x_values );
    detail::EvalLinearLagrange1D( parent_coordinates[1], y_values );
    detail::EvalLinearLagrange1DDerivatives( derivatives );

    // Each adjacent pair stores the derivatives for one face node in the shared local-node ordering.
    parent_derivatives( 0, 0 ) = derivatives[0] * y_values[0];
    parent_derivatives( 0, 1 ) = x_values[0] * derivatives[0];
    parent_derivatives( 1, 0 ) = derivatives[1] * y_values[0];
    parent_derivatives( 1, 1 ) = x_values[1] * derivatives[0];
    parent_derivatives( 2, 0 ) = derivatives[1] * y_values[1];
    parent_derivatives( 2, 1 ) = x_values[1] * derivatives[1];
    parent_derivatives( 3, 0 ) = derivatives[0] * y_values[1];
    parent_derivatives( 3, 1 ) = x_values[0] * derivatives[1];
  }
};

template <>
struct BasisFunction<LINEAR_QUAD, QUADRATIC_LAGRANGE> : FaceElementTraits<LINEAR_QUAD, QUADRATIC_LAGRANGE> {
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void Eval( MdSpan<const RealT, parent_dimension> parent_coordinates,
                                                            MdSpan<RealT, node_count> basis_values )
  {
    RealT x_values[3];
    RealT y_values[3];
    detail::EvalQuadraticLagrange1D( parent_coordinates[0], x_values );
    detail::EvalQuadraticLagrange1D( parent_coordinates[1], y_values );

    // The one-dimensional factors are vertex 0, vertex 1, then midpoint. Map their tensor products into MFEM's
    // vertices, oriented edge midpoints, then center ordering used by MeshData.
    basis_values[0] = x_values[0] * y_values[0];
    basis_values[1] = x_values[1] * y_values[0];
    basis_values[2] = x_values[1] * y_values[1];
    basis_values[3] = x_values[0] * y_values[1];
    basis_values[4] = x_values[2] * y_values[0];
    basis_values[5] = x_values[1] * y_values[2];
    basis_values[6] = x_values[2] * y_values[1];
    basis_values[7] = x_values[0] * y_values[2];
    basis_values[8] = x_values[2] * y_values[2];
  }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void EvalParentDerivatives(
      MdSpan<const RealT, parent_dimension> parent_coordinates,
      MdSpan<RealT, node_count, parent_dimension> parent_derivatives )
  {
    RealT x_values[3];
    RealT y_values[3];
    RealT x_derivatives[3];
    RealT y_derivatives[3];
    detail::EvalQuadraticLagrange1D( parent_coordinates[0], x_values );
    detail::EvalQuadraticLagrange1D( parent_coordinates[1], y_values );
    detail::EvalQuadraticLagrange1DDerivatives( parent_coordinates[0], x_derivatives );
    detail::EvalQuadraticLagrange1DDerivatives( parent_coordinates[1], y_derivatives );

    // Apply the same face-node map as Eval() while storing each node's two parent derivatives next to each other.
    constexpr int x_factor_by_node[9] = { 0, 1, 1, 0, 2, 1, 2, 0, 2 };
    constexpr int y_factor_by_node[9] = { 0, 0, 1, 1, 0, 2, 1, 2, 2 };
    for ( int node = 0; node < node_count; ++node ) {
      parent_derivatives( node, 0 ) = x_derivatives[x_factor_by_node[node]] * y_values[y_factor_by_node[node]];
      parent_derivatives( node, 1 ) = x_values[x_factor_by_node[node]] * y_derivatives[y_factor_by_node[node]];
    }
  }
};

/**
 * @brief Map one parent-space point to physical space using precomputed basis values
 *
 * The coordinates view makes the node-major layout explicit as coordinates(node, physical_component).
 *
 * @param basis_values Basis values indexed by node
 * @param coordinates Nodal physical coordinates indexed by node, then physical component
 * @param physical_point Mapped point indexed by physical component
 */
template <typename Basis>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void MapToPhysical(
    MdSpan<const RealT, Basis::node_count> basis_values,
    MdSpan<const RealT, Basis::node_count, Basis::spatial_dimension> coordinates,
    MdSpan<RealT, Basis::spatial_dimension> physical_point )
{
  for ( int physical_component = 0; physical_component < Basis::spatial_dimension; ++physical_component ) {
    physical_point[physical_component] = 0.0;
    for ( int node = 0; node < Basis::node_count; ++node ) {
      physical_point[physical_component] += basis_values[node] * coordinates( node, physical_component );
    }
  }
}

/**
 * @brief Compute the physical mapping Jacobian from parent basis derivatives and nodal coordinates
 *
 * The Jacobian view makes its layout explicit as jacobian(physical_component, parent_component).
 *
 * @param parent_derivatives Parent derivatives indexed by node, then parent component
 * @param coordinates Nodal physical coordinates indexed by node, then physical component
 * @param jacobian Output mapping Jacobian indexed by physical component, then parent component
 */
template <typename Basis>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void ComputeMappingJacobian(
    MdSpan<const RealT, Basis::node_count, Basis::parent_dimension> parent_derivatives,
    MdSpan<const RealT, Basis::node_count, Basis::spatial_dimension> coordinates,
    MdSpan<RealT, Basis::spatial_dimension, Basis::parent_dimension> jacobian )
{
  for ( int physical_component = 0; physical_component < Basis::spatial_dimension; ++physical_component ) {
    for ( int parent_component = 0; parent_component < Basis::parent_dimension; ++parent_component ) {
      RealT derivative = 0.0;
      for ( int node = 0; node < Basis::node_count; ++node ) {
        derivative += coordinates( node, physical_component ) * parent_derivatives( node, parent_component );
      }
      jacobian( physical_component, parent_component ) = derivative;
    }
  }
}

/**
 * @brief Compute the line or surface measure that maps parent integration weights to physical space
 *
 * @param jacobian Mapping Jacobian stored by physical component, then parent component
 * @return sqrt(det(J^T J)), which is the line or surface integration weight
 */
template <typename Basis>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline RealT ComputeJacobianWeight(
    MdSpan<const RealT, Basis::spatial_dimension, Basis::parent_dimension> jacobian )
{
  if constexpr ( Basis::parent_dimension == 1 ) {
    RealT metric = 0.0;
    for ( int physical_component = 0; physical_component < Basis::spatial_dimension; ++physical_component ) {
      const RealT derivative = jacobian( physical_component, 0 );
      metric += derivative * derivative;
    }
    return std::sqrt( metric );
  } else {
    RealT metric_00 = 0.0;
    RealT metric_01 = 0.0;
    RealT metric_11 = 0.0;
    for ( int physical_component = 0; physical_component < Basis::spatial_dimension; ++physical_component ) {
      const RealT derivative_0 = jacobian( physical_component, 0 );
      const RealT derivative_1 = jacobian( physical_component, 1 );
      metric_00 += derivative_0 * derivative_0;
      metric_01 += derivative_0 * derivative_1;
      metric_11 += derivative_1 * derivative_1;
    }
    const RealT metric_determinant = metric_00 * metric_11 - metric_01 * metric_01;
    return metric_determinant > 0.0 ? std::sqrt( metric_determinant ) : 0.0;
  }
}

/**
 * @brief Transform parent basis derivatives into ambient tangential physical gradients
 *
 * The transformation J(J^T J)^-1 produces the unique tangent gradient that satisfies the chain rule. Physical
 * derivatives remain node-major as physical_derivatives(node, physical_component).
 *
 * @param jacobian Mapping Jacobian stored by physical component, then parent component
 * @param parent_derivatives Parent derivatives stored by node, then parent component
 * @param physical_derivatives Output ambient derivatives stored by node, then physical component
 *
 * @return true when the mapping Jacobian has full parent-space rank; false after zeroing output otherwise
 */
template <typename Basis>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline bool ComputePhysicalDerivatives(
    MdSpan<const RealT, Basis::spatial_dimension, Basis::parent_dimension> jacobian,
    MdSpan<const RealT, Basis::node_count, Basis::parent_dimension> parent_derivatives,
    MdSpan<RealT, Basis::node_count, Basis::spatial_dimension> physical_derivatives )
{
  for ( int node = 0; node < Basis::node_count; ++node ) {
    for ( int physical_component = 0; physical_component < Basis::spatial_dimension; ++physical_component ) {
      physical_derivatives( node, physical_component ) = 0.0;
    }
  }

  RealT inverse_metric[4] = { 0.0, 0.0, 0.0, 0.0 };
  if constexpr ( Basis::parent_dimension == 1 ) {
    RealT metric = 0.0;
    for ( int physical_component = 0; physical_component < Basis::spatial_dimension; ++physical_component ) {
      metric += jacobian( physical_component, 0 ) * jacobian( physical_component, 0 );
    }
    if ( metric <= 0.0 ) {
      return false;
    }
    inverse_metric[0] = 1.0 / metric;
  } else {
    RealT metric_00 = 0.0;
    RealT metric_01 = 0.0;
    RealT metric_11 = 0.0;
    for ( int physical_component = 0; physical_component < Basis::spatial_dimension; ++physical_component ) {
      const RealT derivative_0 = jacobian( physical_component, 0 );
      const RealT derivative_1 = jacobian( physical_component, 1 );
      metric_00 += derivative_0 * derivative_0;
      metric_01 += derivative_0 * derivative_1;
      metric_11 += derivative_1 * derivative_1;
    }
    const RealT metric_determinant = metric_00 * metric_11 - metric_01 * metric_01;
    if ( metric_determinant <= 0.0 ) {
      return false;
    }
    inverse_metric[0] = metric_11 / metric_determinant;
    inverse_metric[1] = -metric_01 / metric_determinant;
    inverse_metric[2] = -metric_01 / metric_determinant;
    inverse_metric[3] = metric_00 / metric_determinant;
  }

  for ( int node = 0; node < Basis::node_count; ++node ) {
    RealT metric_transformed_derivative[2] = { 0.0, 0.0 };
    for ( int parent_component = 0; parent_component < Basis::parent_dimension; ++parent_component ) {
      for ( int input_component = 0; input_component < Basis::parent_dimension; ++input_component ) {
        metric_transformed_derivative[parent_component] +=
            inverse_metric[Basis::parent_dimension * parent_component + input_component] *
            parent_derivatives( node, input_component );
      }
    }
    for ( int physical_component = 0; physical_component < Basis::spatial_dimension; ++physical_component ) {
      for ( int parent_component = 0; parent_component < Basis::parent_dimension; ++parent_component ) {
        physical_derivatives( node, physical_component ) +=
            jacobian( physical_component, parent_component ) * metric_transformed_derivative[parent_component];
      }
    }
  }
  return true;
}

}  // namespace tribol

#endif /* SRC_TRIBOL_INTEG_BASISFUNCTION_HPP_ */
