// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortar.hpp"

#include <algorithm>

#include "tribol/common/Enzyme.hpp"
#include "tribol/integ/BasisFunction.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

using LinearEdgeBasis = BasisFunction<EDGE, LINEAR_LAGRANGE>;
constexpr IndexT node_count = NodalContactData::node_count;
constexpr IndexT output_count = 2 * node_count;

/// Inactive settings and optional frozen integration-surface points for one nodal calculation.
/// @tparam QuadratureRule Element-pair quadrature generator.
template <typename QuadratureRule>
struct KernelParams {
  using Calculator = EnergyMortarCalculator<QuadratureRule>;  ///< Coordinate layout for this generator.

  /// Prepare frozen points once, outside differentiation; varying quadrature is generated inside the kernel.
  /// @param rule Quadrature generator borrowed for this call.
  /// @param coordinates Current pair coordinates used only to prepare frozen quadrature.
  /// @param parameters Residual-gap and quadrature-derivative settings.
  KernelParams( const QuadratureRule& rule, typename Calculator::CoordinateView coordinates,
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

  const QuadratureRule& quadrature_rule;  ///< Generator that outlives this call.
  RealT residual_gap;                     ///< Gap offset held constant during differentiation.
  bool varying;                           ///< Whether integration-surface points follow active geometry.
  typename QuadratureRule::IntegrationSurfaceRule frozen{};  ///< Reference points and weights when varying is false.
};

/// Calculate both nodal gaps followed by both tributary areas, with current normals and physical weights.
/// @tparam QuadratureRule Element-pair quadrature generator.
/// @param coordinates Active coordinates in element/node/component order.
/// @param parameters_void Inactive settings, erased for Enzyme argument handling.
/// @param[out] values Initialized integrals; failed quadrature throws before its output is consumed.
template <typename QuadratureRule>
void nodalKernel( const RealT* coordinates, const void* parameters_void, RealT* values )
{
  const auto& parameters = *static_cast<const KernelParams<QuadratureRule>*>( parameters_void );
  constexpr IndexT element_dofs = EnergyMortarCalculator<QuadratureRule>::element_dof_count;
  const LinearEdgeGeometry::Coords coordinates1{ coordinates, element_dofs };
  const LinearEdgeGeometry::Coords coordinates2{ coordinates + element_dofs, element_dofs };
  const auto quadrature = parameters.varying ? parameters.quadrature_rule.generate( coordinates1, coordinates2 )
                                             : parameters.quadrature_rule.mapIntegrationSurfaceRuleToElements(
                                                   coordinates1, coordinates2, parameters.frozen.getView() );
  if ( quadrature.result.status != QuadratureStatus::Success ) reportQuadratureFailure( quadrature.result.status );
  for ( IndexT output = 0; output < output_count; ++output ) values[output] = 0.0;
  if ( quadrature.result.required_point_count == 0 ) return;

  const LinearEdgeGeometry geometry1( coordinates1 ), geometry2( coordinates2 );
  const auto normal2 = geometry2.normal();
  const RealT alignment = std::min( geometry1.normal().dot( normal2 ), RealT{ 0.0 } );
  for ( IndexT q = 0; q < quadrature.result.required_point_count; ++q ) {
    const RealT xi1 = quadrature.geometry1_coords( q, 0 );
    MdArray<RealT, node_count> basis{};
    LinearEdgeBasis::eval( { &xi1, LinearEdgeBasis::parametric_dimension }, basis.view() );
    const auto point1 = geometry1.position( xi1 );
    const auto point2 = geometry2.position( quadrature.geometry2_coords( q, 0 ) );
    const RealT gap = LinearEdgeGeometry::Vector( point1, point2 ).dot( normal2 ) * alignment - parameters.residual_gap;
    for ( IndexT node = 0; node < node_count; ++node ) {
      const RealT weight = quadrature.weights[q] * basis[node];
      values[node] += weight * gap;
      values[node + node_count] += weight;
    }
  }
}

/// Select one integral so reverse differentiation always has a fixed scalar unit seed.
/// @tparam QuadratureRule Element-pair quadrature generator.
/// @tparam Output Gap-node index followed by area-node index in the combined integral array.
/// @param coordinates Active pair coordinates.
/// @param parameters_void Inactive kernel settings.
/// @param[out] value Selected integral.
template <typename QuadratureRule, IndexT Output>
void scalarKernel( const RealT* coordinates, const void* parameters_void, RealT* value )
{
  MdArray<RealT, output_count> values{};
  nodalKernel<QuadratureRule>( coordinates, parameters_void, values.data() );
  *value = values[Output];
}

/// Calculate the coordinate gradient of one nodal integral for direct use or Hessian differentiation.
/// @tparam QuadratureRule Element-pair quadrature generator.
/// @tparam Output Selected nodal integral.
/// @param coordinates Active pair coordinates.
/// @param parameters_void Inactive kernel settings.
/// @param[out] gradient Initialized gradient in pair DOF order.
template <typename QuadratureRule, IndexT Output>
void gradientKernel( const RealT* coordinates, const void* parameters_void, RealT* gradient )
{
  constexpr IndexT dofs = EnergyMortarCalculator<QuadratureRule>::dof_count;
  std::fill_n( gradient, dofs, RealT{ 0.0 } );
  RealT value = 0.0, seed = 1.0;
  __enzyme_autodiff<void>( (void*)scalarKernel<QuadratureRule, Output>, TRIBOL_ENZYME_DUP, coordinates, gradient,
                           TRIBOL_ENZYME_CONST, parameters_void, TRIBOL_ENZYME_DUP, &value, &seed );
}

/// Form Hessian columns by forward differentiation of a scalar integral's gradient.
/// @tparam QuadratureRule Element-pair quadrature generator.
/// @tparam Output Selected nodal integral.
/// @param coordinates Current pair coordinates.
/// @param parameters Inactive kernel settings shared by all Hessians in this calculation.
/// @param[out] hessian Coordinate rows and columns in pair DOF order.
template <typename QuadratureRule, IndexT Output>
void computeHessian(
    typename EnergyMortarCalculator<QuadratureRule>::CoordinateView coordinates,
    const KernelParams<QuadratureRule>& parameters,
    MdSpan<RealT, EnergyMortarCalculator<QuadratureRule>::dof_count, EnergyMortarCalculator<QuadratureRule>::dof_count>
        hessian )
{
  constexpr IndexT dofs = EnergyMortarCalculator<QuadratureRule>::dof_count;
  for ( IndexT column = 0; column < dofs; ++column ) {
    MdArray<RealT, dofs> seed{}, gradient{}, derivative{};
    seed[column] = 1.0;
    __enzyme_fwddiff<void>( (void*)gradientKernel<QuadratureRule, Output>, TRIBOL_ENZYME_DUP, coordinates.data(),
                            seed.data(), TRIBOL_ENZYME_CONST, (const void*)&parameters, TRIBOL_ENZYME_DUP,
                            gradient.data(), derivative.data() );
    for ( IndexT row = 0; row < dofs; ++row ) hessian( row, column ) = derivative[row];
  }
}

}  // namespace

template <typename QuadratureRule>
NodalContactData EnergyMortarCalculator<QuadratureRule>::computeNodalValues( CoordinateView coordinates ) const
{
  const KernelParams<QuadratureRule> parameters{ quadrature_rule_, coordinates, p_ };
  MdArray<RealT, output_count> values{};
  nodalKernel<QuadratureRule>( coordinates.data(), &parameters, values.data() );
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
  const KernelParams<QuadratureRule> parameters{ quadrature_rule_, coordinates, p_ };
  NodalContactQuantities result{};
  MdArray<RealT, output_count> values{};
  nodalKernel<QuadratureRule>( coordinates.data(), &parameters, values.data() );
  for ( IndexT node = 0; node < node_count; ++node ) {
    result.g_tilde[node] = values[node];
    result.AI[node] = values[node + node_count];
  }
  // Empty rules have no nodal constraints to differentiate.
  if ( result.AI[0] == 0.0 && result.AI[1] == 0.0 ) return result;
  gradientKernel<QuadratureRule, 0>( coordinates.data(), &parameters, result.gap_gradients.data() );
  gradientKernel<QuadratureRule, 1>( coordinates.data(), &parameters, result.gap_gradients.data() + dof_count );
  gradientKernel<QuadratureRule, node_count>( coordinates.data(), &parameters, result.area_gradients.data() );
  gradientKernel<QuadratureRule, node_count + 1>( coordinates.data(), &parameters,
                                                  result.area_gradients.data() + dof_count );
  return result;
}

template <typename QuadratureRule>
typename EnergyMortarCalculator<QuadratureRule>::NodalContactHessians
EnergyMortarCalculator<QuadratureRule>::computeSecondDerivatives( CoordinateView coordinates, bool include_area ) const
{
  const KernelParams<QuadratureRule> parameters{ quadrature_rule_, coordinates, p_ };
  NodalContactHessians result{};
  computeHessian<QuadratureRule, 0>( coordinates, parameters, { result.gap_hessians.data(), dof_count * dof_count } );
  computeHessian<QuadratureRule, 1>( coordinates, parameters,
                                     { result.gap_hessians.data() + dof_count * dof_count, dof_count * dof_count } );
  if ( include_area ) {
    computeHessian<QuadratureRule, node_count>( coordinates, parameters,
                                                { result.area_hessians.data(), dof_count * dof_count } );
    computeHessian<QuadratureRule, node_count + 1>(
        coordinates, parameters, { result.area_hessians.data() + dof_count * dof_count, dof_count * dof_count } );
  }
  return result;
}

template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<1>>>;
template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<2>>>;
template class EnergyMortarCalculator<PairQuadratureRule<SmoothedProjectedEdgeMapping, EdgeQuadratureRuleGenerator<3>>>;

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
