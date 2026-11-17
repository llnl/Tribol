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
#include "tribol/common/MdArray.hpp"
#include "tribol/integ/QuadratureCommon.hpp"

namespace tribol {

/** @brief One quadrature-point quantity and whether its values or derivatives must be integrated.
 * @tparam T Scalar energy density or fixed-size vector traction.
 */
template <typename T>
struct QuadraturePointValue {
  T value{};            ///< Unweighted quantity; the integral applies the physical quadrature weight.
  bool active = false;  ///< Whether to retain this point, including a zero value with a nonzero tangent.
};

/** @brief Pair energy, quadrature status, and compressive-point activation. */
struct PairEnergyValue {
  RealT energy = 0.0;  ///< Integrated energy, zero for empty or inactive rules.
  bool has_active_points =
      false;  ///< Whether values or derivatives contribute after successful quadrature, including zero gap.
  QuadratureResult quadrature_result{};  ///< Failed quadrature output must not be consumed.

  /** @brief Return the scalar energy as a one-entry function value vector. @return Integrated pair energy. */
  TRIBOL_HOST_DEVICE MdArray<RealT, 1> values() const { return MdArray<RealT, 1>{ energy }; }
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
  TRIBOL_HOST_DEVICE PairResidualValue() = default;

  QuadratureResult quadrature_result{};  ///< Failed quadrature output must not be consumed.
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

 private:
  MdArray<RealT, size_values>
      residual_{};  ///< Geometry1 values followed by geometry2 values in basis-function/value-index order.
};

/**
 * @brief Require two fixed-size coordinate views and a const evaluation with values, quadrature status, and activation.
 *
 * Coords1 and Coords2 are read-only views whose sizes agree with size_geom1_coords and size_geom2_coords. Jacobian
 * columns follow geometry1 coordinates, then geometry2 coordinates. The const evaluation result supplies values(),
 * quadrature_result, and has_active_points. After successful quadrature, has_active_points indicates whether the
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
    } &&
    requires( const PairFunction& pair_function, const RealT* data1, const RealT* data2,
              typename PairFunction::Coords1 coords1, typename PairFunction::Coords2 coords2,
              const std::remove_cvref_t<decltype( pair_function.evaluate( coords1, coords2 ) )>& pair_values_struct ) {
      {
        typename PairFunction::Coords1( data1, PairFunction::size_geom1_coords )
      } -> std::same_as<typename PairFunction::Coords1>;
      {
        typename PairFunction::Coords2( data2, PairFunction::size_geom2_coords )
      } -> std::same_as<typename PairFunction::Coords2>;
      { pair_values_struct.values() } -> std::same_as<MdArray<RealT, PairFunction::size_values>>;
      requires std::same_as<std::remove_cvref_t<decltype( pair_values_struct.quadrature_result )>, QuadratureResult>;
      requires std::same_as<std::remove_cvref_t<decltype( pair_values_struct.has_active_points )>, bool>;
    };

}  // namespace tribol

#endif  // SRC_TRIBOL_ASSEMBLY_PAIRFUNCTIONDATA_HPP_
