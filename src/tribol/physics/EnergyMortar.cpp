// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortar.hpp"
#include "tribol/physics/PairEvaluation.hpp"

#include <algorithm>

#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/PairIntegral.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

constexpr IndexT node_count = NodalContactData::node_count;

/// Define nodal weighted-gap and tributary-area integrals for one pair calculation.
/// @tparam QuadratureRule Element-pair quadrature generator.
template <typename QuadratureRule>
struct NodalIntegralDefinition {
  using Calculator = EnergyMortarCalculator<QuadratureRule>;  ///< Coordinate layout for this generator.
  static constexpr IndexT value_count = 2 * node_count;       ///< Gap nodes followed by area nodes.
  static constexpr IndexT dof_count = Calculator::dof_count;  ///< Active pair coordinate count.

  /// Prepare frozen integration-surface points once when quadrature differentiation is disabled.
  /// @param rule Quadrature generator borrowed for this call.
  /// @param coordinates Current pair coordinates used only to prepare frozen quadrature.
  /// @param parameters Residual-gap and quadrature-derivative settings.
  NodalIntegralDefinition( const QuadratureRule& rule, typename Calculator::CoordinateView coordinates,
                           const ContactParams& parameters )
      : quadrature_rule( rule ), residual_gap( parameters.residual_gap ), varying( parameters.enzyme_quadrature )
  {
    if ( !varying ) {
      frozen = rule.generateIntegrationSurfaceRule(
          { coordinates.data(), Calculator::element_dof_count },
          { coordinates.data() + Calculator::element_dof_count, Calculator::element_dof_count } );
      if ( frozen.result.status != QuadratureStatus::Success ) reportQuadratureFailure( frozen.result.status );
    }
  }

  /// Calculate both nodal gaps followed by both tributary areas from the active coordinates.
  /// @param coordinates Active coordinates in element/node/component order.
  /// @return Owned nodal integrals; invalid geometry throws before quadrature data is consumed.
  TRIBOL_ENZYME_INLINE MdArray<RealT, value_count> values( MdSpan<const RealT, dof_count> coordinates ) const
  {
    constexpr IndexT element_dofs = Calculator::element_dof_count;
    const LinearEdgeGeometry geometry1( { coordinates.data(), element_dofs } );
    const LinearEdgeGeometry geometry2( { coordinates.data() + element_dofs, element_dofs } );
    const auto quadrature =
        varying ? quadrature_rule.generate( geometry1, geometry2 )
                : quadrature_rule.mapIntegrationSurfaceRuleToElements( geometry1, geometry2, frozen.getView() );
    const auto integral = integratePair( geometry1, geometry2, quadrature, [&]( const auto& point ) {
      const auto normal2 = point.geometry2.normal();
      const RealT alignment = std::min( point.geometry1.normal().dot( normal2 ), RealT{ 0.0 } );
      const auto basis1 = point.template evalBasis1<EDGE, LINEAR_LAGRANGE>();
      const RealT gap =
          LinearEdgeGeometry::Vector( point.position1(), point.position2() ).dot( normal2 ) * alignment - residual_gap;
      return MdArray<RealT, value_count>{ basis1[0] * gap, basis1[1] * gap, basis1[0], basis1[1] };
    } );
    if ( integral.quadrature_result.status != QuadratureStatus::Success ) {
      reportQuadratureFailure( integral.quadrature_result.status );
    }
    return integral.value;
  }

  const QuadratureRule& quadrature_rule;  ///< Generator that outlives this call.
  RealT residual_gap;                     ///< Gap offset held constant during differentiation.
  bool varying;                           ///< Whether integration-surface points follow active geometry.
  typename QuadratureRule::IntegrationSurfaceRule frozen{};  ///< Reference points and weights when varying is false.
};

}  // namespace

template <typename QuadratureRule>
NodalContactData EnergyMortarCalculator<QuadratureRule>::computeNodalValues( CoordinateView coordinates ) const
{
  const NodalIntegralDefinition<QuadratureRule> definition{ quadrature_rule_, coordinates, p_ };
  const auto values = definition.values( coordinates );
  NodalContactData result{};
  for ( IndexT node = 0; node < node_count; ++node ) {
    result.g_tilde[node] = values[node];
    result.AI[node] = values[node + node_count];
  }
  return result;
}

template <typename QuadratureRule>
typename EnergyMortarCalculator<QuadratureRule>::NodalContactQuantities
EnergyMortarCalculator<QuadratureRule>::computeContactQuantities( CoordinateView coordinates ) const
{
  using Definition = NodalIntegralDefinition<QuadratureRule>;
  const Definition definition{ quadrature_rule_, coordinates, p_ };
  const auto values = definition.values( coordinates );
  NodalContactQuantities result{};
  for ( IndexT node = 0; node < node_count; ++node ) {
    result.g_tilde[node] = values[node];
    result.AI[node] = values[node + node_count];
  }
  // Empty rules have no nodal constraints to differentiate.
  if ( result.AI[0] == 0.0 && result.AI[1] == 0.0 ) return result;
  const auto evaluated = VectorPairEvaluator<Definition>::evaluateGradients( coordinates, definition, values.view() );
  for ( IndexT node = 0; node < node_count; ++node ) {
    for ( IndexT dof = 0; dof < dof_count; ++dof ) {
      result.gap_gradients( node, dof ) = evaluated.gradients( node, dof );
      result.area_gradients( node, dof ) = evaluated.gradients( node + node_count, dof );
    }
  }
  return result;
}

template <typename QuadratureRule>
typename EnergyMortarCalculator<QuadratureRule>::NodalContactHessians
EnergyMortarCalculator<QuadratureRule>::computeSecondDerivatives( CoordinateView coordinates, bool include_area ) const
{
  using Definition = NodalIntegralDefinition<QuadratureRule>;
  const Definition definition{ quadrature_rule_, coordinates, p_ };
  const auto hessians =
      include_area ? VectorPairEvaluator<Definition>::evaluateHessians( coordinates, definition )
                   : VectorPairEvaluator<Definition>::template evaluateHessians<node_count>( coordinates, definition );
  NodalContactHessians result{};
  for ( IndexT node = 0; node < node_count; ++node ) {
    for ( IndexT row = 0; row < dof_count; ++row ) {
      for ( IndexT column = 0; column < dof_count; ++column ) {
        result.gap_hessians( node, row, column ) = hessians( node, row, column );
        result.area_hessians( node, row, column ) = hessians( node + node_count, row, column );
      }
    }
  }
  return result;
}

template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>;
template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>>;
template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>>;

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
