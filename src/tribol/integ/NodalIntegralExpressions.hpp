// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Build nodal energies from named assembled gap and area integrals without allocating field storage.
 */
#ifndef SRC_TRIBOL_INTEG_NODALINTEGRALEXPRESSIONS_HPP_
#define SRC_TRIBOL_INTEG_NODALINTEGRALEXPRESSIONS_HPP_

#include <cmath>
#include <concepts>
#include <stdexcept>
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/MdArray.hpp"

namespace tribol {
/** @brief Identify the assembled integral occupying each component of a node's differentiation state. */
enum class NodalIntegralQuantity
{
  WeightedGap,  ///< Assembled basis-weighted gap.
  Area,         ///< Assembled basis-weighted measure.
  Count         ///< Number of state components.
};
/** @brief Number of named assembled quantities at one node. */
inline constexpr IndexT nodal_integral_count = static_cast<IndexT>( NodalIntegralQuantity::Count );
/** @brief Distinguish nodal field expressions from quadrature-point integrands and numerical arrays. */
struct NodalFieldExpressionTag {};

namespace detail {
/**
 * @brief Describe dependencies needed for nodal energy derivative storage and area normalization.
 * @tparam UsesArea Whether the field depends on the assembled tributary area.
 * @tparam NeedsArea Whether a reciprocal requires a nonzero tributary area.
 * @tparam Affine Whether each smooth branch is affine in the assembled state.
 * @tparam Depends Whether the field depends on the assembled state rather than fixed inputs.
 */
template <bool UsesArea, bool NeedsArea, bool Affine, bool Depends>
struct NodalFieldProperties : NodalFieldExpressionTag {
  static constexpr bool uses_area = UsesArea;            ///< Whether area partial derivatives are needed.
  static constexpr bool needs_nonzero_area = NeedsArea;  ///< Whether area normalization needs a support check.
  static constexpr bool is_affine = Affine;              ///< Whether smooth branches have zero Hessian.
  static constexpr bool depends_on_state = Depends;      ///< Whether gap or area is an active input.
};
}  // namespace detail

/**
 * @brief Refer to one named assembled integral, whose values and coordinate derivatives belong to the assembly context.
 *
 * This value owns no vectors or matrices. It selects the corresponding component when evaluated at a node, so repeated
 * references share the same differentiation variable. Integral evaluation order does not affect its identity.
 * @tparam Quantity Weighted gap or tributary area.
 */
template <NodalIntegralQuantity Quantity>
struct NodalIntegralField : detail::NodalFieldProperties<Quantity == NodalIntegralQuantity::Area, false, true, true> {
  static_assert( Quantity != NodalIntegralQuantity::Count );
  /**
   * @brief Read this integral from one assembled nodal state.
   * @param state Named quantities in NodalIntegralQuantity order.
   * @param node Local true-DOF index, unused for an assembled state component.
   * @return Assembled integral value.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT value( MdSpan<const RealT, nodal_integral_count> state,
                                                       [[maybe_unused]] IndexT node ) const
  {
    return state[static_cast<IndexT>( Quantity )];
  }
  /** @brief Accept any field length. @param count Number of local dual true DOFs. */
  void validateSize( [[maybe_unused]] IndexT count ) const {}
};
/** @brief Symbolic weighted gap integral on the assembled dual submesh. */
inline constexpr NodalIntegralField<NodalIntegralQuantity::WeightedGap> nodal_weighted_gap{};
/** @brief Symbolic tributary area integral on the assembled dual submesh. */
inline constexpr NodalIntegralField<NodalIntegralQuantity::Area> nodal_area{};

/**
 * @brief Borrow an independent scalar field, such as LM values, in local dual true-DOF order.
 *
 * Values are held fixed during coordinate differentiation. The array must remain accessible and unchanged while the
 * expression is evaluated. Matching the assembled field's DOF ordering and MPI partition is a caller precondition.
 */
struct IndependentNodalField : detail::NodalFieldProperties<false, false, true, false> {
  ArrayViewT<const RealT> values;  ///< Borrowed fixed nodal values; no view is retained by an assembled result.
  /** @brief Borrow independent coefficients. @param coefficients Fixed values in local dual true-DOF order. */
  TRIBOL_HOST_DEVICE explicit IndependentNodalField( ArrayViewT<const RealT> coefficients ) : values( coefficients ) {}
  /**
   * @brief Read a fixed field value at a node.
   * @param state Active assembled state, unused by an independent field.
   * @param node Valid local true-DOF index.
   * @return Fixed coefficient for this node.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT value( [[maybe_unused]] MdSpan<const RealT, nodal_integral_count> state,
                                                       IndexT node ) const
  {
    return values[node];
  }
  /**
   * @brief Require one coefficient per local dual true DOF.
   * @param count Expected local vector length.
   * @throws std::invalid_argument The supplied field has a different length.
   */
  void validateSize( IndexT count ) const
  {
    if ( values.size() != count )
      throw std::invalid_argument( "Independent nodal field size must match assembled integrals." );
  }
};

namespace detail {
/**
 * @brief Multiply a nodal field by a fixed scalar.
 * @tparam Field Nodal field expression.
 */
template <typename Field>
struct ScaledNodalField
    : NodalFieldProperties<Field::uses_area, Field::needs_nonzero_area, Field::is_affine, Field::depends_on_state> {
  Field field;  ///< Owned expression, potentially borrowing a fixed field.
  RealT scale;  ///< Inactive scalar multiplier.
  /** @brief Evaluate the scaled field. @param state Active nodal integrals. @param node Local true DOF.
   * @return Scaled value. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT value( MdSpan<const RealT, nodal_integral_count> state,
                                                       IndexT node ) const
  {
    return scale * field.value( state, node );
  }
  /** @brief Validate borrowed field lengths. @param count Local dual true-DOF count. */
  void validateSize( IndexT count ) const { field.validateSize( count ); }
};

/**
 * @brief Multiply two nodal fields at the same node.
 * @tparam Left First nodal field.
 * @tparam Right Second nodal field.
 */
template <typename Left, typename Right>
struct ProductNodalField
    : NodalFieldProperties<Left::uses_area || Right::uses_area, Left::needs_nonzero_area || Right::needs_nonzero_area,
                           Left::is_affine && Right::is_affine &&
                               !( Left::depends_on_state && Right::depends_on_state ),
                           Left::depends_on_state || Right::depends_on_state> {
  Left left;    ///< First owned expression.
  Right right;  ///< Second owned expression.
  /** @brief Evaluate the field product. @param state Active nodal integrals. @param node Local true DOF.
   * @return Product at the node. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT value( MdSpan<const RealT, nodal_integral_count> state,
                                                       IndexT node ) const
  {
    return left.value( state, node ) * right.value( state, node );
  }
  /** @brief Validate both fields. @param count Local dual true-DOF count. */
  void validateSize( IndexT count ) const
  {
    left.validateSize( count );
    right.validateSize( count );
  }
};

/**
 * @brief Clamp a nodal field after its integral has been assembled.
 * @tparam Field Underlying nodal expression.
 */
template <typename Field>
struct ClampedNodalField
    : NodalFieldProperties<Field::uses_area, Field::needs_nonzero_area, Field::is_affine, Field::depends_on_state> {
  Field field;        ///< Field being clamped.
  RealT upper_bound;  ///< Inactive maximum value.
  /**
   * @brief Retain the underlying branch at equality, including the compressive tangent at zero gap.
   * @param state Active nodal integrals.
   * @param node Local true DOF.
   * @return The field value at or below the bound, otherwise the bound.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT value( MdSpan<const RealT, nodal_integral_count> state,
                                                       IndexT node ) const
  {
    const RealT v = field.value( state, node );
    return v > upper_bound ? upper_bound : v;
  }
  /** @brief Validate the underlying field. @param count Local dual true-DOF count. */
  void validateSize( IndexT count ) const { field.validateSize( count ); }
};

/** @brief Divide a fixed numerator by the assembled tributary area. */
struct ReciprocalNodalArea : NodalFieldProperties<true, true, false, true> {
  RealT numerator;  ///< Inactive numerator.
  /**
   * @brief Evaluate the quotient on a node with supported area.
   * @param state Active nodal integrals; nonzero area is required.
   * @param node Local true DOF, unused.
   * @return Numerator divided by area; NodalEnergyAtNode checks support before this is called.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT value( MdSpan<const RealT, nodal_integral_count> state,
                                                       [[maybe_unused]] IndexT node ) const
  {
    return numerator / state[static_cast<IndexT>( NodalIntegralQuantity::Area )];
  }
  /** @brief Accept any field length. @param count Local dual true-DOF count. */
  void validateSize( [[maybe_unused]] IndexT count ) const {}
};

/**
 * @brief Bind one nodal energy expression for Enzyme differentiation with respect to its assembled integral values.
 * @tparam Field Product expression being reduced across nodes.
 */
template <typename Field>
struct NodalEnergyAtNode {
  static constexpr IndexT dof_count = nodal_integral_count;  ///< Weighted-gap and area variables.
  Field field;                                               ///< Inactive expression and borrowed fixed coefficients.
  IndexT node;                                               ///< Local dual true DOF supplying fixed coefficients.
  RealT area_tolerance;  ///< Maximum unsupported area magnitude for reciprocal-area expressions.
  /**
   * @brief Evaluate one node's energy, returning zero for unsupported reciprocal-area nodes before dividing.
   * @param state Active assembled nodal integrals.
   * @return Energy at this node; unsupported nodes have zero energy and zero first and second derivatives.
   */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT energy( MdSpan<const RealT, dof_count> state ) const
  {
    if constexpr ( Field::needs_nonzero_area ) {
      if ( std::abs( state[static_cast<IndexT>( NodalIntegralQuantity::Area )] ) <= area_tolerance ) return 0.0;
    }
    return field.value( state, node );
  }
};
}  // namespace detail

/**
 * @brief Defer a nodal contraction until values and coordinate derivatives have been assembled.
 * @tparam Field Field expression whose node values are summed for energy.
 */
template <typename Field>
struct NodalEnergyExpression {
  static constexpr bool has_area_derivatives = Field::uses_area;  ///< Whether area participates in the energy.
  static constexpr bool has_outer_hessian = !Field::is_affine;    ///< Whether smooth branches can have second partials.
  Field field;  ///< Owned expression; independent coefficient arrays remain borrowed.
  /**
   * @brief Bind a local node without generating geometry or allocating storage.
   * @param node Valid local dual true-DOF index.
   * @param area_tolerance Finite nonnegative support threshold, held fixed in differentiation.
   * @return Definition accepted by EnergyPairEvaluator; it reads named components rather than integral call order.
   */
  TRIBOL_HOST_DEVICE auto atNode( IndexT node, RealT area_tolerance ) const
  {
    return detail::NodalEnergyAtNode<Field>{ field, node, area_tolerance };
  }
  /** @brief Validate independent field lengths before evaluation. @param count Local dual true-DOF count. */
  void validateSize( IndexT count ) const { field.validateSize( count ); }
};

/**
 * @brief Scale a nodal field without evaluating its values.
 * @tparam Field Nodal field expression.
 * @param scale Inactive scalar multiplier.
 * @param field Field to scale.
 * @return Deferred scaled field.
 */
template <typename Field>
  requires std::derived_from<Field, NodalFieldExpressionTag>
TRIBOL_HOST_DEVICE constexpr auto operator*( RealT scale, const Field& field )
{
  return detail::ScaledNodalField<Field>{ {}, field, scale };
}

/**
 * @brief Scale a nodal field on the right.
 * @tparam Field Nodal field expression.
 * @param field Field to scale.
 * @param scale Inactive scalar multiplier.
 * @return Deferred scaled field.
 */
template <typename Field>
  requires std::derived_from<Field, NodalFieldExpressionTag>
TRIBOL_HOST_DEVICE constexpr auto operator*( const Field& field, RealT scale )
{
  return scale * field;
}

/**
 * @brief Form a product of fields sharing the same assembled nodal state.
 * @tparam Left First nodal expression.
 * @tparam Right Second nodal expression.
 * @param left First field.
 * @param right Second field.
 * @return Deferred field product.
 */
template <typename Left, typename Right>
  requires( std::derived_from<Left, NodalFieldExpressionTag> && std::derived_from<Right, NodalFieldExpressionTag> )
TRIBOL_HOST_DEVICE constexpr auto operator*( const Left& left, const Right& right )
{
  return detail::ProductNodalField<Left, Right>{ {}, left, right };
}

/**
 * @brief Clamp an assembled field, preserving its derivative branch at equality.
 * @tparam Field Nodal field expression.
 * @param field Field to clamp.
 * @param upper_bound Maximum value, typically zero for contact activation.
 * @return Deferred clamped field.
 */
template <typename Field>
  requires std::derived_from<Field, NodalFieldExpressionTag>
TRIBOL_HOST_DEVICE constexpr auto min( const Field& field, RealT upper_bound )
{
  return detail::ClampedNodalField<Field>{ {}, field, upper_bound };
}

/**
 * @brief Normalize by tributary area; evaluation checks the configured area tolerance before dividing.
 * @param numerator Inactive numerator.
 * @param area Named assembled area field.
 * @return Deferred reciprocal area.
 */
TRIBOL_HOST_DEVICE constexpr auto operator/( RealT numerator,
                                             [[maybe_unused]] NodalIntegralField<NodalIntegralQuantity::Area> area )
{
  return detail::ReciprocalNodalArea{ {}, numerator };
}

/**
 * @brief Build an energy by summing a product of fields over assembled nodes.
 * @tparam First First field type.
 * @tparam Rest Remaining field types.
 * @param first First field.
 * @param rest Remaining fields in the nodewise product.
 * @return Deferred energy contraction; no MPI reduction or assembly occurs here.
 */
template <typename First, typename... Rest>
  requires( std::derived_from<First, NodalFieldExpressionTag> &&
            ( std::derived_from<Rest, NodalFieldExpressionTag> && ... ) )
TRIBOL_HOST_DEVICE constexpr auto sum( const First& first, const Rest&... rest )
{
  auto product = ( first * ... * rest );
  return NodalEnergyExpression<decltype( product )>{ product };
}

/**
 * @brief Contract an independent field with an assembled integral, or two nodal fields.
 * @tparam Left First field type.
 * @tparam Right Second field type.
 * @param left First field.
 * @param right Second field.
 * @return Deferred energy contraction.
 */
template <typename Left, typename Right>
  requires( std::derived_from<Left, NodalFieldExpressionTag> && std::derived_from<Right, NodalFieldExpressionTag> )
TRIBOL_HOST_DEVICE constexpr auto dot( const Left& left, const Right& right )
{
  return sum( left, right );
}

/**
 * @brief Scale the completed energy expression before derivatives or assembly are evaluated.
 * @tparam Field Contracted field type.
 * @param scale Inactive scalar multiplier.
 * @param energy Energy expression.
 * @return Deferred scaled energy; gap and pressure output fields are not scaled as assembled results.
 */
template <typename Field>
TRIBOL_HOST_DEVICE constexpr auto operator*( RealT scale, const NodalEnergyExpression<Field>& energy )
{
  auto scaled = scale * energy.field;
  return NodalEnergyExpression<decltype( scaled )>{ scaled };
}
/**
 * @brief Scale an energy expression on the right.
 * @tparam Field Contracted field type.
 * @param energy Energy expression.
 * @param scale Inactive scalar multiplier.
 * @return Deferred scaled energy.
 */
template <typename Field>
TRIBOL_HOST_DEVICE constexpr auto operator*( const NodalEnergyExpression<Field>& energy, RealT scale )
{
  return scale * energy;
}

}  // namespace tribol
#endif  // SRC_TRIBOL_INTEG_NODALINTEGRALEXPRESSIONS_HPP_
