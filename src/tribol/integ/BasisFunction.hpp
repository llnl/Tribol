// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_BASISFUNCTION_HPP_
#define SRC_TRIBOL_INTEG_BASISFUNCTION_HPP_

#include <cmath>

#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/common/MdSpan.hpp"
#include "tribol/common/Parameters.hpp"

namespace tribol {

/**
 * @brief Describes the dimensions and local-node ordering for a supported contact face element
 *
 * The local-node ordering is shared by MeshData connectivity and BasisFunction output arrays. Parametric-node
 * coordinates
 * use the normalized interval and square [-1, 1]. The linear triangle uses the standard simplex with vertices (0, 0),
 * (1, 0), and (0, 1). The spatial dimension is the ambient contact dimension: two for edges and three for triangles
 * and quadrilaterals.
 *
 * getParametricNodeCoords() returns its small coordinate array for the common case. Its output-view overload allows a
 * caller to reuse existing storage when a return value is undesirable in a device or differentiated kernel.
 *
 * @tparam Element Contact face shape
 * @tparam Order Lagrange order represented by the element connectivity
 */
template <InterfaceElementType Element, FaceOrderType Order>
struct FaceElementTraits;

namespace detail {

/**
 * @brief Stores the parametric dimension and adds the owning coordinate return path shared by face element traits
 *
 * @tparam Derived FaceElementTraits specialization that writes its element-specific parametric coordinates
 * @tparam ParametricDim Number of coordinates used to parameterize the element
 */
template <typename Derived, int ParametricDim>
struct FaceElementTraitsBase {
  // Face element traits expose only compile-time data and static operations, so they do not create objects.
  FaceElementTraitsBase() = delete;

  static constexpr int parametric_dim = ParametricDim;
  using ParametricCoords = MdArray<RealT, parametric_dim>;

  /**
   * @brief Return one node's parametric coordinates in an owning fixed-size array
   *
   * Each traits specialization remains responsible for writing its coordinates into the supplied view. Keeping that
   * operation in the specialization allows callers to reuse storage, while this shared overload provides the concise
   * return-value interface.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static ParametricCoords getParametricNodeCoords( int local_node )
  {
    ParametricCoords parametric_coords;
    Derived::getParametricNodeCoords( local_node, parametric_coords.view() );
    return parametric_coords;
  }
};

}  // namespace detail

template <>
struct FaceElementTraits<EDGE, LINEAR_LAGRANGE>
    : detail::FaceElementTraitsBase<FaceElementTraits<EDGE, LINEAR_LAGRANGE>, 1> {
  using Base = detail::FaceElementTraitsBase<FaceElementTraits<EDGE, LINEAR_LAGRANGE>, 1>;
  using Base::getParametricNodeCoords;

  static constexpr int num_nodes = 2;
  static constexpr int spatial_dim = 2;

  /**
   * @brief Get one node's coordinate in the normalized parametric segment
   *
   * @param local_node Local node index in MeshData connectivity order
   * @param parametric_coords Output array with length parametric_dim
   *
   * @pre local_node is in [0, num_nodes)
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void getParametricNodeCoords(
      int local_node, MdSpan<RealT, parametric_dim> parametric_coords )
  {
    parametric_coords[0] = local_node == 0 ? -1.0 : 1.0;
  }
};

template <>
struct FaceElementTraits<EDGE, QUADRATIC_LAGRANGE>
    : detail::FaceElementTraitsBase<FaceElementTraits<EDGE, QUADRATIC_LAGRANGE>, 1> {
  using Base = detail::FaceElementTraitsBase<FaceElementTraits<EDGE, QUADRATIC_LAGRANGE>, 1>;
  using Base::getParametricNodeCoords;

  static constexpr int num_nodes = 3;
  static constexpr int spatial_dim = 2;

  /**
   * @brief Get one node's coordinate in the normalized parametric segment
   *
   * The two vertices precede the midpoint to match MFEM H1 local-node ordering.
   *
   * @param local_node Local node index in MeshData connectivity order
   * @param parametric_coords Output array with length parametric_dim
   *
   * @pre local_node is in [0, num_nodes)
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void getParametricNodeCoords(
      int local_node, MdSpan<RealT, parametric_dim> parametric_coords )
  {
    parametric_coords[0] = local_node == 0 ? -1.0 : ( local_node == 1 ? 1.0 : 0.0 );
  }
};

template <>
struct FaceElementTraits<TRIANGLE, LINEAR_LAGRANGE>
    : detail::FaceElementTraitsBase<FaceElementTraits<TRIANGLE, LINEAR_LAGRANGE>, 2> {
  using Base = detail::FaceElementTraitsBase<FaceElementTraits<TRIANGLE, LINEAR_LAGRANGE>, 2>;
  using Base::getParametricNodeCoords;

  static constexpr int num_nodes = 3;
  static constexpr int spatial_dim = 3;

  /**
   * @brief Get one vertex coordinate in the standard parametric triangle
   *
   * @param local_node Local node index in MeshData connectivity order
   * @param parametric_coords Output array with length parametric_dim
   *
   * @pre local_node is in [0, num_nodes)
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void getParametricNodeCoords(
      int local_node, MdSpan<RealT, parametric_dim> parametric_coords )
  {
    parametric_coords[0] = local_node == 1 ? 1.0 : 0.0;
    parametric_coords[1] = local_node == 2 ? 1.0 : 0.0;
  }
};

template <>
struct FaceElementTraits<QUAD, LINEAR_LAGRANGE>
    : detail::FaceElementTraitsBase<FaceElementTraits<QUAD, LINEAR_LAGRANGE>, 2> {
  using Base = detail::FaceElementTraitsBase<FaceElementTraits<QUAD, LINEAR_LAGRANGE>, 2>;
  using Base::getParametricNodeCoords;

  static constexpr int num_nodes = 4;
  static constexpr int spatial_dim = 3;

  /**
   * @brief Get one vertex coordinate in the normalized parametric square
   *
   * @param local_node Local node index in MeshData connectivity order
   * @param parametric_coords Output array with length parametric_dim
   *
   * @pre local_node is in [0, num_nodes)
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void getParametricNodeCoords(
      int local_node, MdSpan<RealT, parametric_dim> parametric_coords )
  {
    parametric_coords[0] = local_node == 0 || local_node == 3 ? -1.0 : 1.0;
    parametric_coords[1] = local_node < 2 ? -1.0 : 1.0;
  }
};

template <>
struct FaceElementTraits<QUAD, QUADRATIC_LAGRANGE>
    : detail::FaceElementTraitsBase<FaceElementTraits<QUAD, QUADRATIC_LAGRANGE>, 2> {
  using Base = detail::FaceElementTraitsBase<FaceElementTraits<QUAD, QUADRATIC_LAGRANGE>, 2>;
  using Base::getParametricNodeCoords;

  static constexpr int num_nodes = 9;
  static constexpr int spatial_dim = 3;

  /**
   * @brief Get one node coordinate in the normalized parametric square
   *
   * The four vertices are followed by the midpoints of edges (0,1), (1,2), (2,3), and (3,0), then the center. This is
   * the local-node ordering used by MFEM H1 quadrilateral elements.
   *
   * @param local_node Local node index in MeshData connectivity order
   * @param parametric_coords Output array with length parametric_dim
   *
   * @pre local_node is in [0, num_nodes)
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void getParametricNodeCoords(
      int local_node, MdSpan<RealT, parametric_dim> parametric_coords )
  {
    constexpr RealT parametric_node_coords[num_nodes][parametric_dim] = { { -1.0, -1.0 }, { 1.0, -1.0 }, { 1.0, 1.0 },
                                                                          { -1.0, 1.0 },  { 0.0, -1.0 }, { 1.0, 0.0 },
                                                                          { 0.0, 1.0 },   { -1.0, 0.0 }, { 0.0, 0.0 } };
    parametric_coords[0] = parametric_node_coords[local_node][0];
    parametric_coords[1] = parametric_node_coords[local_node][1];
  }
};

/**
 * @brief Get the number of local nodes for a supported face shape and order
 *
 * This runtime entry point uses the same FaceElementTraits specializations as BasisFunction and MeshData. Unsupported
 * combinations return zero so the caller can report an error in the appropriate execution context.
 */
TRIBOL_HOST_DEVICE inline int numberOfFaceNodes( InterfaceElementType element, FaceOrderType order )
{
  if ( element == EDGE ) {
    if ( order == LINEAR_LAGRANGE ) {
      return FaceElementTraits<EDGE, LINEAR_LAGRANGE>::num_nodes;
    }
    if ( order == QUADRATIC_LAGRANGE ) {
      return FaceElementTraits<EDGE, QUADRATIC_LAGRANGE>::num_nodes;
    }
  }
  if ( element == TRIANGLE && order == LINEAR_LAGRANGE ) {
    return FaceElementTraits<TRIANGLE, LINEAR_LAGRANGE>::num_nodes;
  }
  if ( element == QUAD ) {
    if ( order == LINEAR_LAGRANGE ) {
      return FaceElementTraits<QUAD, LINEAR_LAGRANGE>::num_nodes;
    }
    if ( order == QUADRATIC_LAGRANGE ) {
      return FaceElementTraits<QUAD, QUADRATIC_LAGRANGE>::num_nodes;
    }
  }
  return 0;
}

namespace detail {

TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void evalLinearLagrange1D( RealT parametric_coord,
                                                                          MdSpan<RealT, 2> values )
{
  values[0] = 0.5 * ( 1.0 - parametric_coord );
  values[1] = 0.5 * ( 1.0 + parametric_coord );
}

TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void evalLinearLagrange1DDerivatives( MdSpan<RealT, 2> derivatives )
{
  derivatives[0] = -0.5;
  derivatives[1] = 0.5;
}

TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void evalQuadraticLagrange1D( RealT parametric_coord,
                                                                             MdSpan<RealT, 3> values )
{
  values[0] = 0.5 * parametric_coord * ( parametric_coord - 1.0 );
  values[1] = 0.5 * parametric_coord * ( parametric_coord + 1.0 );
  values[2] = 1.0 - parametric_coord * parametric_coord;
}

TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void evalQuadraticLagrange1DDerivatives( RealT parametric_coord,
                                                                                        MdSpan<RealT, 3> derivatives )
{
  derivatives[0] = parametric_coord - 0.5;
  derivatives[1] = parametric_coord + 0.5;
  derivatives[2] = -2.0 * parametric_coord;
}

}  // namespace detail

/**
 * @brief Evaluates the complete Lagrange basis for one supported contact face element
 *
 * Basis values and derivatives use the local-node ordering defined by FaceElementTraits. Parametric derivatives are
 * stored
 * node-major as parametric_derivatives[node * parametric_dim + parametric_dim].
 *
 * Supported specializations provide eval(parametric_coords, basis_values), where the output has num_nodes entries,
 * and evalParametricDerivatives(parametric_coords, parametric_derivatives), where the output has num_nodes *
 * parametric_dim entries. Both methods use caller-provided arrays so contact kernels can evaluate bases without
 * allocation.
 */
template <InterfaceElementType Element, FaceOrderType Order>
struct BasisFunction;

template <>
struct BasisFunction<EDGE, LINEAR_LAGRANGE> : FaceElementTraits<EDGE, LINEAR_LAGRANGE> {
  static constexpr int size_basis =
      FaceElementTraits<EDGE, LINEAR_LAGRANGE>::num_nodes;  ///< Number of basis functions.

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void eval( MdSpan<const RealT, parametric_dim> parametric_coords,
                                                            MdSpan<RealT, size_basis> basis_values )
  {
    detail::evalLinearLagrange1D( parametric_coords[0], basis_values );
  }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void evalParametricDerivatives(
      MdSpan<const RealT, parametric_dim>, MdSpan<RealT, size_basis, parametric_dim> parametric_derivatives )
  {
    parametric_derivatives( 0, 0 ) = -0.5;
    parametric_derivatives( 1, 0 ) = 0.5;
  }
};

template <>
struct BasisFunction<EDGE, QUADRATIC_LAGRANGE> : FaceElementTraits<EDGE, QUADRATIC_LAGRANGE> {
  static constexpr int size_basis =
      FaceElementTraits<EDGE, QUADRATIC_LAGRANGE>::num_nodes;  ///< Number of basis functions.

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void eval( MdSpan<const RealT, parametric_dim> parametric_coords,
                                                            MdSpan<RealT, size_basis> basis_values )
  {
    detail::evalQuadraticLagrange1D( parametric_coords[0], basis_values );
  }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void evalParametricDerivatives(
      MdSpan<const RealT, parametric_dim> parametric_coords,
      MdSpan<RealT, size_basis, parametric_dim> parametric_derivatives )
  {
    parametric_derivatives( 0, 0 ) = parametric_coords[0] - 0.5;
    parametric_derivatives( 1, 0 ) = parametric_coords[0] + 0.5;
    parametric_derivatives( 2, 0 ) = -2.0 * parametric_coords[0];
  }
};

template <>
struct BasisFunction<TRIANGLE, LINEAR_LAGRANGE> : FaceElementTraits<TRIANGLE, LINEAR_LAGRANGE> {
  static constexpr int size_basis =
      FaceElementTraits<TRIANGLE, LINEAR_LAGRANGE>::num_nodes;  ///< Number of basis functions.

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void eval( MdSpan<const RealT, parametric_dim> parametric_coords,
                                                            MdSpan<RealT, size_basis> basis_values )
  {
    basis_values[0] = 1.0 - parametric_coords[0] - parametric_coords[1];
    basis_values[1] = parametric_coords[0];
    basis_values[2] = parametric_coords[1];
  }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void evalParametricDerivatives(
      MdSpan<const RealT, parametric_dim>, MdSpan<RealT, size_basis, parametric_dim> parametric_derivatives )
  {
    parametric_derivatives( 0, 0 ) = -1.0;
    parametric_derivatives( 0, 1 ) = -1.0;
    parametric_derivatives( 1, 0 ) = 1.0;
    parametric_derivatives( 1, 1 ) = 0.0;
    parametric_derivatives( 2, 0 ) = 0.0;
    parametric_derivatives( 2, 1 ) = 1.0;
  }
};

template <>
struct BasisFunction<QUAD, LINEAR_LAGRANGE> : FaceElementTraits<QUAD, LINEAR_LAGRANGE> {
  static constexpr int size_basis =
      FaceElementTraits<QUAD, LINEAR_LAGRANGE>::num_nodes;  ///< Number of basis functions.

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void eval( MdSpan<const RealT, parametric_dim> parametric_coords,
                                                            MdSpan<RealT, size_basis> basis_values )
  {
    RealT x_values[2];
    RealT y_values[2];
    detail::evalLinearLagrange1D( parametric_coords[0], x_values );
    detail::evalLinearLagrange1D( parametric_coords[1], y_values );

    // Map tensor products into the face-node ordering shared by MeshData instead of exposing tensor-array ordering.
    basis_values[0] = x_values[0] * y_values[0];
    basis_values[1] = x_values[1] * y_values[0];
    basis_values[2] = x_values[1] * y_values[1];
    basis_values[3] = x_values[0] * y_values[1];
  }

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void evalParametricDerivatives(
      MdSpan<const RealT, parametric_dim> parametric_coords,
      MdSpan<RealT, size_basis, parametric_dim> parametric_derivatives )
  {
    RealT x_values[2];
    RealT y_values[2];
    RealT derivatives[2];
    detail::evalLinearLagrange1D( parametric_coords[0], x_values );
    detail::evalLinearLagrange1D( parametric_coords[1], y_values );
    detail::evalLinearLagrange1DDerivatives( derivatives );

    // Each adjacent pair stores the derivatives for one face node in the shared local-node ordering.
    parametric_derivatives( 0, 0 ) = derivatives[0] * y_values[0];
    parametric_derivatives( 0, 1 ) = x_values[0] * derivatives[0];
    parametric_derivatives( 1, 0 ) = derivatives[1] * y_values[0];
    parametric_derivatives( 1, 1 ) = x_values[1] * derivatives[0];
    parametric_derivatives( 2, 0 ) = derivatives[1] * y_values[1];
    parametric_derivatives( 2, 1 ) = x_values[1] * derivatives[1];
    parametric_derivatives( 3, 0 ) = derivatives[0] * y_values[1];
    parametric_derivatives( 3, 1 ) = x_values[0] * derivatives[1];
  }
};

template <>
struct BasisFunction<QUAD, QUADRATIC_LAGRANGE> : FaceElementTraits<QUAD, QUADRATIC_LAGRANGE> {
  static constexpr int size_basis =
      FaceElementTraits<QUAD, QUADRATIC_LAGRANGE>::num_nodes;  ///< Number of basis functions.

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void eval( MdSpan<const RealT, parametric_dim> parametric_coords,
                                                            MdSpan<RealT, size_basis> basis_values )
  {
    RealT x_values[3];
    RealT y_values[3];
    detail::evalQuadraticLagrange1D( parametric_coords[0], x_values );
    detail::evalQuadraticLagrange1D( parametric_coords[1], y_values );

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

  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE static void evalParametricDerivatives(
      MdSpan<const RealT, parametric_dim> parametric_coords,
      MdSpan<RealT, size_basis, parametric_dim> parametric_derivatives )
  {
    RealT x_values[3];
    RealT y_values[3];
    RealT x_derivatives[3];
    RealT y_derivatives[3];
    detail::evalQuadraticLagrange1D( parametric_coords[0], x_values );
    detail::evalQuadraticLagrange1D( parametric_coords[1], y_values );
    detail::evalQuadraticLagrange1DDerivatives( parametric_coords[0], x_derivatives );
    detail::evalQuadraticLagrange1DDerivatives( parametric_coords[1], y_derivatives );

    // Apply the same face-node map as eval() while storing each node's two parametric derivatives next to each other.
    constexpr int x_factor_by_node[9] = { 0, 1, 1, 0, 2, 1, 2, 0, 2 };
    constexpr int y_factor_by_node[9] = { 0, 0, 1, 1, 0, 2, 1, 2, 2 };
    for ( int node = 0; node < size_basis; ++node ) {
      parametric_derivatives( node, 0 ) = x_derivatives[x_factor_by_node[node]] * y_values[y_factor_by_node[node]];
      parametric_derivatives( node, 1 ) = x_values[x_factor_by_node[node]] * y_derivatives[y_factor_by_node[node]];
    }
  }
};

/**
 * @brief Interpolate nodal values at one parametric-space point using precomputed basis values
 *
 * The nodal-values argument provides value access as nodal_values(node, value_index). MdSpan supplies contiguous
 * values, while a mesh-specific view can use the same interface to read separate nodal value arrays through element
 * connectivity without changing this finite-element operation.
 *
 * @tparam Basis Basis that defines the local node count
 * @tparam NumValuesPerNode Number of scalar values at each node
 * @tparam NodalValues Lightweight view that provides nodal_values(node, value_index)
 *
 * @param basis_values Basis values indexed by local node
 * @param nodal_values Values indexed by local node, then value index
 * @param interpolated_values Interpolated values indexed by value index
 */
template <typename Basis, int NumValuesPerNode, typename NodalValues>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void interpolateNodalValues(
    MdSpan<const RealT, Basis::size_basis> basis_values, NodalValues nodal_values,
    MdSpan<RealT, NumValuesPerNode> interpolated_values )
{
  for ( int value_index = 0; value_index < NumValuesPerNode; ++value_index ) {
    interpolated_values[value_index] = 0.0;
    for ( int node = 0; node < Basis::size_basis; ++node ) {
      interpolated_values[value_index] += basis_values[node] * nodal_values( node, value_index );
    }
  }
}

/**
 * @brief Map one parametric-space point to physical space using precomputed basis values
 *
 * The coordinates view makes the node-major layout explicit as coords(node, spatial_dim).
 *
 * @param basis_values Basis values indexed by node
 * @param coords Nodal physical coordinates indexed by node, then spatial dimension
 * @param physical_point Mapped point indexed by spatial dimension
 */
template <typename Basis>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void mapToPhysical(
    MdSpan<const RealT, Basis::size_basis> basis_values,
    MdSpan<const RealT, Basis::size_basis, Basis::spatial_dim> coords,
    MdSpan<RealT, Basis::spatial_dim> physical_point )
{
  interpolateNodalValues<Basis, Basis::spatial_dim>( basis_values, coords, physical_point );
}

/**
 * @brief Compute the physical mapping Jacobian from parametric basis derivatives and nodal coordinates
 *
 * The Jacobian view makes its layout explicit as jacobian(spatial_dim, parametric_dim).
 *
 * @param parametric_derivatives Parametric derivatives indexed by node, then parametric dimension
 * @param coords Nodal physical coordinates indexed by node, then spatial dimension
 * @param jacobian Output mapping Jacobian indexed by spatial dimension, then parametric dimension
 */
template <typename Basis>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void computeMappingJacobian(
    MdSpan<const RealT, Basis::size_basis, Basis::parametric_dim> parametric_derivatives,
    MdSpan<const RealT, Basis::size_basis, Basis::spatial_dim> coords,
    MdSpan<RealT, Basis::spatial_dim, Basis::parametric_dim> jacobian )
{
  for ( int spatial_dim = 0; spatial_dim < Basis::spatial_dim; ++spatial_dim ) {
    for ( int parametric_dim = 0; parametric_dim < Basis::parametric_dim; ++parametric_dim ) {
      RealT derivative = 0.0;
      for ( int node = 0; node < Basis::size_basis; ++node ) {
        derivative += coords( node, spatial_dim ) * parametric_derivatives( node, parametric_dim );
      }
      jacobian( spatial_dim, parametric_dim ) = derivative;
    }
  }
}

/**
 * @brief Compute the physical mapping Jacobian at one parametric-space point
 *
 * This overload evaluates the basis derivatives before forming the Jacobian. Callers that also need the derivatives can
 * use the lower-level overload to evaluate them once and reuse them.
 *
 * @param parametric_coords Coordinates of the evaluation point in parametric space
 * @param coords Nodal physical coordinates indexed by node, then spatial dimension
 * @param jacobian Output mapping Jacobian indexed by spatial dimension, then parametric dimension
 */
template <typename Basis>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline void computeMappingJacobian(
    MdSpan<const RealT, Basis::parametric_dim> parametric_coords,
    MdSpan<const RealT, Basis::size_basis, Basis::spatial_dim> coords,
    MdSpan<RealT, Basis::spatial_dim, Basis::parametric_dim> jacobian )
{
  MdArray<RealT, Basis::size_basis, Basis::parametric_dim> parametric_derivatives;
  Basis::evalParametricDerivatives( parametric_coords, parametric_derivatives.view() );
  computeMappingJacobian<Basis>( parametric_derivatives.view(), coords, jacobian );
}

/**
 * @brief Compute the line or surface measure that maps parametric integration weights to physical space
 *
 * For an edge, this measure maps a parametric differential to physical arc length as ds = sqrt(det(J^T J)) dxi. For a
 * face, it maps a parametric area differential to physical surface area as dA = sqrt(det(J^T J)) dxi deta.
 *
 * @param jacobian Mapping Jacobian stored by spatial dimension, then parametric dimension
 * @return sqrt(det(J^T J)), which is the line or surface integration weight
 */
template <typename Basis>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline RealT computeJacobianWeight(
    MdSpan<const RealT, Basis::spatial_dim, Basis::parametric_dim> jacobian )
{
  if constexpr ( Basis::parametric_dim == 1 ) {
    RealT metric = 0.0;
    for ( int spatial_dim = 0; spatial_dim < Basis::spatial_dim; ++spatial_dim ) {
      const RealT derivative = jacobian( spatial_dim, 0 );
      metric += derivative * derivative;
    }
    return std::sqrt( metric );
  } else {
    RealT metric_00 = 0.0;
    RealT metric_01 = 0.0;
    RealT metric_11 = 0.0;
    for ( int spatial_dim = 0; spatial_dim < Basis::spatial_dim; ++spatial_dim ) {
      const RealT derivative_0 = jacobian( spatial_dim, 0 );
      const RealT derivative_1 = jacobian( spatial_dim, 1 );
      metric_00 += derivative_0 * derivative_0;
      metric_01 += derivative_0 * derivative_1;
      metric_11 += derivative_1 * derivative_1;
    }
    const RealT metric_determinant = metric_00 * metric_11 - metric_01 * metric_01;
    return metric_determinant > 0.0 ? std::sqrt( metric_determinant ) : 0.0;
  }
}

/**
 * @brief Compute the line or surface integration weight at one parametric-space point
 *
 * This overload keeps the derivative and Jacobian storage local when a caller only needs the integration weight.
 *
 * @param parametric_coords Coordinates of the evaluation point in parametric space
 * @param coords Nodal physical coordinates indexed by node, then spatial dimension
 * @return sqrt(det(J^T J)) at the parametric-space point
 */
template <typename Basis>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline RealT computeJacobianWeight(
    MdSpan<const RealT, Basis::parametric_dim> parametric_coords,
    MdSpan<const RealT, Basis::size_basis, Basis::spatial_dim> coords )
{
  MdArray<RealT, Basis::spatial_dim, Basis::parametric_dim> jacobian;
  computeMappingJacobian<Basis>( parametric_coords, coords, jacobian.view() );
  return computeJacobianWeight<Basis>( jacobian.view() );
}

/**
 * @brief Compute the line integration weight at one scalar parametric coordinate
 *
 * Edge quadrature naturally supplies one scalar coordinate. This overload keeps the one-entry parametric-coordinate
 * storage
 * inside the basis utility instead of requiring every edge kernel to construct it.
 *
 * @param parametric_coord Coordinate of the evaluation point on the parametric edge
 * @param coords Nodal physical coordinates indexed by node, then spatial dimension
 * @return sqrt(det(J^T J)) at the parametric coordinate
 */
template <typename Basis>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline RealT computeJacobianWeight(
    RealT parametric_coord, MdSpan<const RealT, Basis::size_basis, Basis::spatial_dim> coords )
{
  static_assert( Basis::parametric_dim == 1,
                 "A scalar parametric coordinate requires a one-dimensional parametric element." );
  typename Basis::ParametricCoords parametric_coords;
  parametric_coords[0] = parametric_coord;
  return computeJacobianWeight<Basis>( parametric_coords.view(), coords );
}

/**
 * @brief Compute physical-space basis gradients tangent to the mapped element
 *
 * The mapped edge or face lies in the surrounding physical space. The transformation J(J^T J)^-1 produces the unique
 * gradient in that space that is tangent to the mapped element and satisfies the chain rule. Physical derivatives
 * remain node-major as physical_derivatives(node, spatial_dim).
 *
 * @param jacobian Mapping Jacobian stored by spatial dimension, then parametric dimension
 * @param parametric_derivatives Parametric derivatives stored by node, then parametric dimension
 * @param physical_derivatives Output tangent gradients stored by node, then spatial dimension
 *
 * @return true when the mapping Jacobian has full parametric-space rank; false after zeroing output otherwise
 */
template <typename Basis>
TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE inline bool computePhysicalDerivatives(
    MdSpan<const RealT, Basis::spatial_dim, Basis::parametric_dim> jacobian,
    MdSpan<const RealT, Basis::size_basis, Basis::parametric_dim> parametric_derivatives,
    MdSpan<RealT, Basis::size_basis, Basis::spatial_dim> physical_derivatives )
{
  for ( int node = 0; node < Basis::size_basis; ++node ) {
    for ( int spatial_dim = 0; spatial_dim < Basis::spatial_dim; ++spatial_dim ) {
      physical_derivatives( node, spatial_dim ) = 0.0;
    }
  }

  RealT inverse_metric[4] = { 0.0, 0.0, 0.0, 0.0 };
  if constexpr ( Basis::parametric_dim == 1 ) {
    RealT metric = 0.0;
    for ( int spatial_dim = 0; spatial_dim < Basis::spatial_dim; ++spatial_dim ) {
      metric += jacobian( spatial_dim, 0 ) * jacobian( spatial_dim, 0 );
    }
    if ( metric <= 0.0 ) {
      return false;
    }
    inverse_metric[0] = 1.0 / metric;
  } else {
    RealT metric_00 = 0.0;
    RealT metric_01 = 0.0;
    RealT metric_11 = 0.0;
    for ( int spatial_dim = 0; spatial_dim < Basis::spatial_dim; ++spatial_dim ) {
      const RealT derivative_0 = jacobian( spatial_dim, 0 );
      const RealT derivative_1 = jacobian( spatial_dim, 1 );
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

  for ( int node = 0; node < Basis::size_basis; ++node ) {
    RealT metric_transformed_derivative[2] = { 0.0, 0.0 };
    for ( int row = 0; row < Basis::parametric_dim; ++row ) {
      for ( int column = 0; column < Basis::parametric_dim; ++column ) {
        metric_transformed_derivative[row] +=
            inverse_metric[Basis::parametric_dim * row + column] * parametric_derivatives( node, column );
      }
    }
    for ( int spatial_dim = 0; spatial_dim < Basis::spatial_dim; ++spatial_dim ) {
      for ( int parametric_dim = 0; parametric_dim < Basis::parametric_dim; ++parametric_dim ) {
        physical_derivatives( node, spatial_dim ) +=
            jacobian( spatial_dim, parametric_dim ) * metric_transformed_derivative[parametric_dim];
      }
    }
  }
  return true;
}

}  // namespace tribol

#endif /* SRC_TRIBOL_INTEG_BASISFUNCTION_HPP_ */
