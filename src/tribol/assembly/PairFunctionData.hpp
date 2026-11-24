// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Quadrature-point and single-pair evaluation result types and the PairFunctionType concept.
 */
#ifndef SRC_TRIBOL_ASSEMBLY_PAIRFUNCTIONDATA_HPP_
#define SRC_TRIBOL_ASSEMBLY_PAIRFUNCTIONDATA_HPP_

#include <concepts>
#include <type_traits>
#include <utility>
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/** @brief Pair energy, quadrature status, and compressive-point activation. */
struct PairEnergyValue {
  static constexpr IndexT size_values = 1;  ///< Scalar pair energy.
  RealT energy = 0.0;                       ///< Integrated energy, zero for empty or inactive rules.
  bool has_active_points =
      false;  ///< Whether values or derivatives contribute after successful quadrature, including zero gap.
  QuadratureMetadata quadrature_metadata{};  ///< Failed quadrature output must not be consumed.

  /** @brief Return the scalar energy as a one-entry function value vector. @return Integrated pair energy. */
  TRIBOL_HOST_DEVICE MdArray<RealT, 1> values() const { return MdArray<RealT, 1>{ energy }; }

  /** @brief Return writable flattened value storage for a derivative provider. @return Address of energy. */
  TRIBOL_HOST_DEVICE RealT* data() { return &energy; }
};

/**
 * @brief Pair residual split into two element-local basis-function/value-index blocks, with quadrature status and
 * activation.
 * @tparam SizeGeom1Basis Number of basis functions associated with geometry1.
 * @tparam SizeGeom2Basis Number of basis functions associated with geometry2.
 * @tparam NumValuesPerBasis Number of residual values associated with each basis-function index.
 */
template <IndexT SizeGeom1Basis, IndexT SizeGeom2Basis, IndexT NumValuesPerBasis>
class PairResidualValue {
  static_assert( SizeGeom1Basis > 0 );
  static_assert( SizeGeom2Basis > 0 );
  static_assert( NumValuesPerBasis > 0 );

 public:
  static constexpr IndexT size_geom1_basis = SizeGeom1Basis;         ///< Geometry1 basis size.
  static constexpr IndexT size_geom2_basis = SizeGeom2Basis;         ///< Geometry2 basis size.
  static constexpr IndexT num_values_per_basis = NumValuesPerBasis;  ///< Residual values per basis function.
  static constexpr IndexT size_geom1_values = SizeGeom1Basis * NumValuesPerBasis;  ///< Geometry1 residual extent.
  static constexpr IndexT size_geom2_values = SizeGeom2Basis * NumValuesPerBasis;  ///< Geometry2 residual extent.
  static constexpr IndexT size_values = size_geom1_values + size_geom2_values;     ///< Total residual extent.

  /** @brief Construct a zero residual with default quadrature state and no active points. */
  TRIBOL_DEFAULT_HOST_DEVICE PairResidualValue() = default;

  QuadratureMetadata quadrature_metadata{};  ///< Failed quadrature output must not be consumed.
  bool has_active_points =
      false;  ///< Whether values or derivatives contribute after successful quadrature, including zero gap.

  /**
   * @brief Access one geometry1 residual entry.
   * @param basis_index Geometry1-local basis-function index.
   * @param value_index Residual value index at the selected basis function.
   * @return Mutable entry.
   */
  TRIBOL_HOST_DEVICE RealT& geometry1( IndexT basis_index, IndexT value_index )
  {
    return residual_[basis_index * NumValuesPerBasis + value_index];
  }

  /**
   * @brief Access one geometry1 residual entry.
   * @param basis_index Geometry1-local basis-function index.
   * @param value_index Residual value index at the selected basis function.
   * @return Read-only entry.
   */
  TRIBOL_HOST_DEVICE const RealT& geometry1( IndexT basis_index, IndexT value_index ) const
  {
    return residual_[basis_index * NumValuesPerBasis + value_index];
  }

  /**
   * @brief Access one geometry2 residual entry.
   * @param basis_index Geometry2-local basis-function index.
   * @param value_index Residual value index at the selected basis function.
   * @return Mutable entry.
   */
  TRIBOL_HOST_DEVICE RealT& geometry2( IndexT basis_index, IndexT value_index )
  {
    return residual_[size_geom1_values + basis_index * NumValuesPerBasis + value_index];
  }

  /**
   * @brief Access one geometry2 residual entry.
   * @param basis_index Geometry2-local basis-function index.
   * @param value_index Residual value index at the selected basis function.
   * @return Read-only entry.
   */
  TRIBOL_HOST_DEVICE const RealT& geometry2( IndexT basis_index, IndexT value_index ) const
  {
    return residual_[size_geom1_values + basis_index * NumValuesPerBasis + value_index];
  }

  /** @brief Return the residual as the function value vector. @return Geometry1 then geometry2
   * basis-function/value-index values. */
  TRIBOL_HOST_DEVICE MdArray<RealT, size_values> values() const { return residual_; }

  /** @brief Return writable flattened value storage for a derivative provider. @return Residual storage. */
  TRIBOL_HOST_DEVICE RealT* data() { return residual_.data(); }

 private:
  MdArray<RealT, size_values>
      residual_{};  ///< Geometry1 values followed by geometry2 values in basis-function/value-index order.
};

/** @brief Nodal weighted gaps and tributary areas with activation and quadrature status.
 * @tparam NumNodes Number of FE nodes carrying the nodal integrals.
 */
template <IndexT NumNodes>
class PairNodalGapValue {
  static_assert( NumNodes > 0 );

 public:
  static constexpr IndexT num_nodes = NumNodes;        ///< Number of FE nodes.
  static constexpr IndexT size_values = 2 * NumNodes;  ///< Weighted gaps followed by tributary areas.

  /** @brief Fields stored for each geometry1 node. */
  enum class Field : IndexT
  {
    WeightedGap,    ///< Basis-weighted gap integral.
    TributaryArea,  ///< Basis-weighted area integral.
  };

  static constexpr IndexT num_fields = 2;  ///< Number of nodal integral fields.

  /** @brief Construct zero nodal integrals with default quadrature state and no active points. */
  TRIBOL_DEFAULT_HOST_DEVICE PairNodalGapValue() = default;

  /**
   * @brief Return a field's flattened value index.
   * @param field Nodal integral field.
   * @param node Geometry1 element-local node.
   * @return Index into values().
   */
  TRIBOL_HOST_DEVICE static constexpr IndexT valueIndex( Field field, IndexT node )
  {
    return static_cast<IndexT>( field ) * num_nodes + node;
  }

  /** @brief Return a weighted-gap index. @param node Geometry1 local node. @return Index into values(). */
  TRIBOL_HOST_DEVICE static constexpr IndexT weightedGapIndex( IndexT node )
  {
    return valueIndex( Field::WeightedGap, node );
  }

  /** @brief Return a tributary-area index. @param node Geometry1 local node. @return Index into values(). */
  TRIBOL_HOST_DEVICE static constexpr IndexT tributaryAreaIndex( IndexT node )
  {
    return valueIndex( Field::TributaryArea, node );
  }

  /** @brief Access a weighted gap. @param node Geometry1 local node. @return Mutable nodal integral. */
  TRIBOL_HOST_DEVICE RealT& weightedGap( IndexT node ) { return values_[weightedGapIndex( node )]; }

  /** @brief Access a weighted gap. @param node Geometry1 local node. @return Read-only nodal integral. */
  TRIBOL_HOST_DEVICE const RealT& weightedGap( IndexT node ) const { return values_[weightedGapIndex( node )]; }

  /** @brief Access a tributary area. @param node Geometry1 local node. @return Mutable nodal integral. */
  TRIBOL_HOST_DEVICE RealT& tributaryArea( IndexT node ) { return values_[tributaryAreaIndex( node )]; }

  /** @brief Access a tributary area. @param node Geometry1 local node. @return Read-only nodal integral. */
  TRIBOL_HOST_DEVICE const RealT& tributaryArea( IndexT node ) const { return values_[tributaryAreaIndex( node )]; }

  /** @brief Return the function value vector. @return Weighted gaps followed by tributary areas in node order. */
  TRIBOL_HOST_DEVICE MdArray<RealT, size_values> values() const { return values_; }

  /** @brief Return writable flattened value storage for a derivative provider. @return Nodal value storage. */
  TRIBOL_HOST_DEVICE RealT* data() { return values_.data(); }

  /// Whether successful quadrature is nonempty, including opening gaps; this does not indicate contact activation.
  bool has_active_points = false;
  QuadratureMetadata quadrature_metadata{};  ///< Failed output must not be consumed.

 private:
  MdArray<RealT, size_values> values_{};  ///< Weighted gaps followed by tributary areas in local-node order.
};

/** @brief Result type returned by a pair function's evaluate() operation. @tparam PairFunction Pair function type. */
template <typename PairFunction>
using PairFunctionValue = std::remove_cvref_t<decltype( std::declval<const PairFunction&>().evaluate(
    std::declval<typename PairFunction::Coords1>(), std::declval<typename PairFunction::Coords2>() ) )>;

/**
 * @brief Require two fixed-size coordinate views and a const evaluation with values, quadrature status, and activation.
 *
 * Coords1 and Coords2 are read-only views whose sizes agree with size_geom1_coords and size_geom2_coords. Jacobian
 * columns follow geometry1 coordinates, then geometry2 coordinates. The const evaluation result supplies values(),
 * quadrature_metadata, and has_active_points. After successful quadrature, has_active_points indicates whether the
 * function's values or derivatives must be retained. Nodal integration can be active before contact activation.
 * These requirements check types and expressions; numerical correctness, host/device execution, and Enzyme
 * compatibility must be verified separately.
 * @tparam PairFunction Type evaluating a pair from two coordinate views.
 */
template <typename PairFunction>
concept PairFunctionType =
    requires {
      typename PairFunction::Coords1;
      typename PairFunction::Coords2;
      typename std::integral_constant<IndexT, PairFunction::size_values>;
      typename std::integral_constant<IndexT, PairFunction::size_geom1_coords>;
      typename std::integral_constant<IndexT, PairFunction::size_geom2_coords>;
      typename std::integral_constant<IndexT, PairFunction::size_coords>;
      requires( PairFunction::size_values > 0 && PairFunction::size_geom1_coords > 0 &&
                PairFunction::size_geom2_coords > 0 );
      requires( PairFunction::Coords1::size() == PairFunction::size_geom1_coords );
      requires( PairFunction::Coords2::size() == PairFunction::size_geom2_coords );
      requires( PairFunction::size_coords == PairFunction::size_geom1_coords + PairFunction::size_geom2_coords );
      requires std::same_as<typename PairFunction::Coords1::value_type, const RealT>;
      requires std::same_as<typename PairFunction::Coords2::value_type, const RealT>;
    } && requires( const PairFunction& pair_function, const RealT* data1, const RealT* data2,
                   typename PairFunction::Coords1 coords1, typename PairFunction::Coords2 coords2,
                   const PairFunctionValue<PairFunction>& pair_values_struct ) {
      {
        typename PairFunction::Coords1( data1, PairFunction::size_geom1_coords )
      } -> std::same_as<typename PairFunction::Coords1>;
      {
        typename PairFunction::Coords2( data2, PairFunction::size_geom2_coords )
      } -> std::same_as<typename PairFunction::Coords2>;
      { pair_values_struct.values() } -> std::same_as<MdArray<RealT, PairFunction::size_values>>;
      requires std::same_as<std::remove_cvref_t<decltype( pair_values_struct.quadrature_metadata )>,
                            QuadratureMetadata>;
      requires std::same_as<std::remove_cvref_t<decltype( pair_values_struct.has_active_points )>, bool>;
    };

/**
 * @brief Require writable flattened values in addition to the read-only pair-function interface.
 *
 * A derivative provider writes values produced during differentiation into the same domain result returned by
 * evaluate(). Numerical Enzyme compatibility must still be verified separately.
 * @tparam PairFunction Pair function type.
 */
template <typename PairFunction>
concept DifferentiablePairFunctionType =
    PairFunctionType<PairFunction> && requires( PairFunctionValue<PairFunction>& pair_value ) {
      typename std::integral_constant<IndexT, PairFunctionValue<PairFunction>::size_values>;
      requires( PairFunctionValue<PairFunction>::size_values == PairFunction::size_values );
      { pair_value.data() } -> std::same_as<RealT*>;
    };

}  // namespace tribol

#endif  // SRC_TRIBOL_ASSEMBLY_PAIRFUNCTIONDATA_HPP_
