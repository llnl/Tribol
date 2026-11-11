// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/** @file @brief Composed Common Plane contact formulation without changes to the legacy implementation. */
#ifndef SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_

#include "tribol/config.hpp"
#include "tribol/geom/CommonPlaneEdgeMapping.hpp"
#include "tribol/geom/CommonPlaneGap.hpp"
#include "tribol/geom/OffsetGap.hpp"
#include "tribol/integ/BasisFunction.hpp"
#include "tribol/integ/EdgeQuadratureRuleGenerator.hpp"
#include "tribol/integ/PairQuadratureRule.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/physics/ContactEnforcement.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/PairEvaluation.hpp"
#include "tribol/physics/PenaltyStiffness.hpp"
#include "tribol/search/PairListBuilder.hpp"

namespace tribol {

/** @brief Common-line pair quadrature. @tparam BaseRule Reference Gauss rule. */
template <typename BaseRule>
using CommonPlaneQuadratureRule = PairQuadratureRule<CommonPlaneEdgeMapping, BaseRule>;

/** @brief Residual and generation status for a pair of linear two-dimensional edges. */
using CommonPlaneResidualData = PairResidualValue<2 * LinearEdgeGeometry::Coords::size()>;

/** @brief Pair residual and its coordinate derivative. */
struct CommonPlanePairData : CommonPlaneResidualData {
  MdArray<RealT, coordinate_count, coordinate_count> jacobian{};  ///< Residual rows by coordinate columns.
};

/** @brief Configure Common Plane with an owned geometric or offset gap. @tparam GapDefinition Gap callable. */
template <typename GapDefinition = CommonPlaneGap>
struct CommonPlaneResidual {
  GapDefinition gap{};  ///< Owned geometric gap and any fixed offset.
};

/**
 * @brief Fixed-size Common Plane residual function used by Enzyme pair differentiation.
 * @tparam BaseRule Reference Gauss rule.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 */
template <typename BaseRule, typename GapDefinition = CommonPlaneGap>
class CommonPlanePairFunction {
 public:
  using Quadrature = CommonPlaneQuadratureRule<BaseRule>;                   ///< Pair quadrature type.
  using Physics = CommonPlaneResidual<GapDefinition>;                       ///< Fixed gap configuration.
  static constexpr IndexT first_dofs = LinearEdgeGeometry::Coords::size();  ///< First edge coordinate count.
  static constexpr IndexT dof_count = 2 * first_dofs;                       ///< Pair coordinate count.
  static constexpr IndexT value_count = dof_count;                          ///< Residual entry count.
  using PrimalResult = PairResidualValue<dof_count>;                        ///< Residual and quadrature state.

  /** @brief Store fixed configuration. @param rule Pair quadrature. @param penalty Stiffness. @param physics Gap. */
  TRIBOL_HOST_DEVICE CommonPlanePairFunction( Quadrature rule, RealT penalty, Physics physics = {} )
      : quadrature_( rule ), penalty_( penalty ), physics_( physics )
  {
  }

  /** @brief Evaluate one pair residual. @param coordinates Current pair coordinates. @return Residual and status. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE PrimalResult evaluate( MdSpan<const RealT, dof_count> coordinates ) const
  {
    const LinearEdgeGeometry geometry1( { coordinates.data(), first_dofs } );
    const LinearEdgeGeometry geometry2( { coordinates.data() + first_dofs, first_dofs } );
    const auto points = quadrature_.generate( geometry1, geometry2 );
    PrimalResult result{ {}, points.result, false };
    if ( points.result.status != QuadratureStatus::Success || points.result.required_point_count == 0 ) {
      return result;
    }
    const auto frame = CommonPlaneEdgeMapping::makeFrame( geometry1, geometry2 );
    using Basis = BasisFunction<EDGE, LINEAR_LAGRANGE>;
    for ( IndexT point = 0; point < points.result.required_point_count; ++point ) {
      const RealT gap = physics_.gap( geometry1.position( points.geometry1_coords( point, 0 ) ),
                                      geometry2.position( points.geometry2_coords( point, 0 ) ), frame );
      if ( gap > 0.0 ) {
        continue;
      }
      result.has_active_points = true;
      const RealT pressure = penalty_ * gap;
      MdArray<RealT, Basis::node_count> basis1{}, basis2{};
      Basis::eval( { &points.geometry1_coords( point, 0 ), Basis::parametric_dimension }, basis1.view() );
      Basis::eval( { &points.geometry2_coords( point, 0 ), Basis::parametric_dimension }, basis2.view() );
      for ( IndexT node = 0; node < Basis::node_count; ++node ) {
        for ( IndexT component = 0; component < LinearEdgeGeometry::spatial_dimension; ++component ) {
          const IndexT row = node * LinearEdgeGeometry::spatial_dimension + component;
          const RealT traction = points.weights[point] * pressure * frame.normal[component];
          result.residual[row] += basis1[node] * traction;
          result.residual[first_dofs + row] -= basis2[node] * traction;
        }
      }
    }
    return result;
  }

  /** @brief Vector entry for Enzyme. @param coordinates Current coordinates. @return Pair residual. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdArray<RealT, value_count> values(
      MdSpan<const RealT, dof_count> coordinates ) const
  {
    return evaluate( coordinates ).residual;
  }

 private:
  Quadrature quadrature_;  ///< Owned pair quadrature configuration.
  RealT penalty_;          ///< Effective pair stiffness.
  Physics physics_;        ///< Owned gap definition.
};

/**
 * @brief Evaluate Common Plane residuals from two coordinate arrays.
 * @tparam BaseRule Reference Gauss rule.
 * @tparam GapDefinition Geometric or offset gap.
 * @param coordinates1 First edge coordinates.
 * @param coordinates2 Second edge coordinates.
 * @param quadrature Pair quadrature.
 * @param penalty Effective stiffness.
 * @param gap Gap definition.
 * @return Pair residual and quadrature state.
 */
template <typename BaseRule, typename GapDefinition = CommonPlaneGap>
TRIBOL_HOST_DEVICE CommonPlaneResidualData computeCommonPlaneResidual(
    LinearEdgeGeometry::Coords coordinates1, LinearEdgeGeometry::Coords coordinates2,
    const CommonPlaneQuadratureRule<BaseRule>& quadrature, RealT penalty, GapDefinition gap = {} )
{
  constexpr IndexT first_dofs = LinearEdgeGeometry::Coords::size();
  MdArray<RealT, CommonPlaneResidualData::coordinate_count> coordinates{};
  for ( IndexT coordinate = 0; coordinate < first_dofs; ++coordinate ) {
    coordinates[coordinate] = coordinates1[coordinate];
    coordinates[first_dofs + coordinate] = coordinates2[coordinate];
  }
  return CommonPlanePairFunction<BaseRule, GapDefinition>{ quadrature, penalty, CommonPlaneResidual{ gap } }.evaluate(
      coordinates.view() );
}

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Evaluate and differentiate one Common Plane pair.
 * @tparam BaseRule Reference Gauss rule.
 * @tparam GapDefinition Geometric or offset gap.
 * @param coordinates1 First edge coordinates.
 * @param coordinates2 Second edge coordinates.
 * @param quadrature Pair quadrature.
 * @param penalty Effective stiffness.
 * @param gap Gap definition.
 * @return Pair residual and Jacobian.
 */
template <typename BaseRule, typename GapDefinition = CommonPlaneGap>
CommonPlanePairData computeCommonPlanePair( LinearEdgeGeometry::Coords coordinates1,
                                            LinearEdgeGeometry::Coords coordinates2,
                                            const CommonPlaneQuadratureRule<BaseRule>& quadrature, RealT penalty,
                                            GapDefinition gap = {} )
{
  using Function = CommonPlanePairFunction<BaseRule, GapDefinition>;
  MdArray<RealT, Function::dof_count> coordinates{};
  for ( IndexT coordinate = 0; coordinate < Function::first_dofs; ++coordinate ) {
    coordinates[coordinate] = coordinates1[coordinate];
    coordinates[Function::first_dofs + coordinate] = coordinates2[coordinate];
  }
  const Function function{ quadrature, penalty, CommonPlaneResidual{ gap } };
  const auto primal = function.evaluate( coordinates.view() );
  if ( primal.quadrature_result.status != QuadratureStatus::Success ) {
    reportQuadratureFailure( primal.quadrature_result.status );
  }
  const auto derivatives = EnzymePairDerivative<Function>::valuesAndJacobian( coordinates.view(), function );
  CommonPlanePairData result;
  result.quadrature_result = primal.quadrature_result;
  result.has_active_points = primal.has_active_points;
  result.residual = derivatives.values;
  result.jacobian = derivatives.gradients;
  return result;
}
#endif

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
/**
 * @brief Common Plane formulation composed from pair search, differentiation, buffers, and MFEM assembly.
 * @tparam BaseRule Reference Gauss rule.
 * @tparam PenaltyStiffness Combination of scaled mesh penalties.
 */
template <typename BaseRule, typename PenaltyStiffness = SeriesPenaltyStiffness>
class CommonPlaneFormulation final : public ContactFormulation {
 public:
  using PairFunction = CommonPlanePairFunction<BaseRule, OffsetGap<CommonPlaneGap>>;  ///< Pair-local residual.

  /**
   * @brief Borrow transfer data and own search and formulation configuration.
   * @param mesh_data Parent displacement transfer data.
   * @param jacobian_data Parent Jacobian transfer data.
   * @param execution_mode Pair evaluation and numerical assembly execution mode.
   * @param pair_builder Owned pair search composition.
   * @param quadrature Owned common-line quadrature.
   * @param gap Offset common-plane gap.
   * @param penalty_inputs Current scaled mesh penalty reader.
   */
  CommonPlaneFormulation( MfemMeshData& mesh_data, MfemJacobianData& jacobian_data, ExecutionMode execution_mode,
                          PairListBuilder pair_builder, typename PairFunction::Quadrature quadrature,
                          OffsetGap<CommonPlaneGap> gap, ScaledMeshPenaltyInputs<PenaltyStiffness> penalty_inputs = {} )
      : mesh_data_( mesh_data ),
        jacobian_data_( jacobian_data ),
        execution_mode_( execution_mode ),
        pair_builder_( std::move( pair_builder ) ),
        quadrature_( std::move( quadrature ) ),
        gap_( std::move( gap ) ),
        penalty_inputs_( std::move( penalty_inputs ) )
  {
  }

  /** @brief Compute Common Plane force and Jacobian. @param mesh1 First mesh. @param mesh2 Second mesh.
   * @return Caller-owned force and Jacobian. */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

 private:
  MfemMeshData& mesh_data_;                                   ///< Borrowed displacement transfer data.
  MfemJacobianData& jacobian_data_;                           ///< Borrowed Jacobian transfer data.
  ExecutionMode execution_mode_;                              ///< Pair kernel execution policy.
  PairListBuilder pair_builder_;                              ///< Owned search composition.
  typename PairFunction::Quadrature quadrature_;              ///< Owned quadrature configuration.
  OffsetGap<CommonPlaneGap> gap_;                             ///< Owned gap offset.
  ScaledMeshPenaltyInputs<PenaltyStiffness> penalty_inputs_;  ///< Current penalty reader.
};
#endif

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
