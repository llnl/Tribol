// Copyright (c) Lawrence Livermore National Security, LLC and
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
#include "tribol/assembly/PairEvaluation.hpp"
#include "tribol/search/PairListBuilder.hpp"

namespace tribol {

/**
 * @brief Fixed-size Common Plane residual function for direct evaluation and optional Enzyme differentiation.
 * @tparam BaseRule Reference Gauss rule.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 */
template <typename BaseRule, typename GapDefinition = CommonPlaneGap>
class CommonPlanePairFunction {
 public:
  using Quadrature = PairQuadratureRule<CommonPlaneEdgeMapping, BaseRule>;  ///< Pair quadrature type.
  static constexpr IndexT first_dofs = LinearEdgeGeometry::Coords::size();  ///< First edge coordinate count.
  static constexpr IndexT dof_count = 2 * first_dofs;                       ///< Pair coordinate count.
  static constexpr IndexT value_count = dof_count;                          ///< Residual entry count.
  using PrimalResult = PairResidualValue<dof_count>;                        ///< Residual and quadrature state.

  /** @brief Store fixed configuration. @param rule Pair quadrature. @param penalty Stiffness. @param gap Gap callable.
   */
  TRIBOL_HOST_DEVICE CommonPlanePairFunction( Quadrature rule, RealT penalty, GapDefinition gap = {} )
      : quadrature_( rule ), penalty_( penalty ), gap_( gap )
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
    const auto line = CommonPlaneEdgeMapping::computeCommonPlaneLine( geometry1, geometry2 );
    // The linear coordinate maps make the weighted parent coordinates the overlap centroid for every base rule.
    RealT measure = 0.0;
    RealT center1 = 0.0;
    RealT center2 = 0.0;
    for ( IndexT point = 0; point < points.result.required_point_count; ++point ) {
      measure += points.weights[point];
      center1 += points.weights[point] * points.geometry1_coords( point, 0 );
      center2 += points.weights[point] * points.geometry2_coords( point, 0 );
    }
    const RealT gap = gap_( geometry1.position( center1 / measure ), geometry2.position( center2 / measure ), line );
    if ( gap > 0.0 ) {
      return result;
    }
    result.has_active_points = true;
    const RealT pressure = penalty_ * gap;
    using Basis = BasisFunction<EDGE, LINEAR_LAGRANGE>;
    for ( IndexT point = 0; point < points.result.required_point_count; ++point ) {
      MdArray<RealT, Basis::node_count> basis1{}, basis2{};
      Basis::eval( { &points.geometry1_coords( point, 0 ), Basis::parametric_dimension }, basis1.view() );
      Basis::eval( { &points.geometry2_coords( point, 0 ), Basis::parametric_dimension }, basis2.view() );
      for ( IndexT node = 0; node < Basis::node_count; ++node ) {
        for ( IndexT component = 0; component < LinearEdgeGeometry::spatial_dimension; ++component ) {
          const IndexT row = node * LinearEdgeGeometry::spatial_dimension + component;
          const RealT traction = points.weights[point] * pressure * line.normal[component];
          result.residual[row] += basis1[node] * traction;
          result.residual[first_dofs + row] -= basis2[node] * traction;
        }
      }
    }
    return result;
  }

 private:
  Quadrature quadrature_;  ///< Owned pair quadrature configuration.
  RealT penalty_;          ///< Effective pair stiffness.
  GapDefinition gap_;      ///< Owned gap definition.
};

#ifdef BUILD_REDECOMP
/**
 * @brief Common Plane pair search and force assembly, with optional Enzyme Jacobians.
 * @tparam BaseRule Reference Gauss rule.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 */
template <typename BaseRule, ExecutionMode Exec>
class CommonPlaneFormulation final : public ContactFormulation {
 public:
  /**
   * @brief Borrow transfer data and own search and formulation configuration.
   * @param mesh_data Parent displacement transfer data.
   * @param jacobian_data Parent Jacobian transfer data.
   * @param pair_builder Owned pair search composition.
   * @param quadrature Owned common-line quadrature.
   * @param gap Offset common-plane gap.
   * @param penalty Effective stiffness, fixed for this formulation's lifetime.
   * @throws std::invalid_argument Stiffness is nonpositive or nonfinite; no formulation is constructed.
   * @throws std::runtime_error Enzyme support is unavailable; no formulation is constructed.
   */
  CommonPlaneFormulation( MfemMeshData& mesh_data, MfemJacobianData& jacobian_data, PairListBuilder pair_builder,
                          PairQuadratureRule<CommonPlaneEdgeMapping, BaseRule> quadrature,
                          OffsetGap<CommonPlaneGap> gap, RealT penalty )
      : CommonPlaneFormulation( mesh_data, &jacobian_data, std::move( pair_builder ), std::move( quadrature ),
                                std::move( gap ), penalty )
  {
  }

  /**
   * @brief Construct Common Plane with optional coordinate Jacobian calculation.
   * @param mesh_data Borrowed displacement transfer data.
   * @param jacobian_data Borrowed Jacobian transfer data, or nullptr to calculate forces only.
   * @param pair_builder Owned pair search composition.
   * @param quadrature Owned common-line quadrature.
   * @param gap Owned gap definition.
   * @param penalty Fixed effective stiffness.
   * @throws std::invalid_argument Stiffness is nonpositive or nonfinite.
   * @throws std::runtime_error Jacobian calculation was requested without Enzyme support.
   */
  CommonPlaneFormulation( MfemMeshData& mesh_data, const MfemJacobianData* jacobian_data, PairListBuilder pair_builder,
                          PairQuadratureRule<CommonPlaneEdgeMapping, BaseRule> quadrature,
                          OffsetGap<CommonPlaneGap> gap, RealT penalty )
      : mesh_data_( mesh_data ),
        jacobian_data_( jacobian_data ),
        pair_builder_( std::move( pair_builder ) ),
        function_( std::move( quadrature ), detail::checkedPenaltyStiffness( penalty ), std::move( gap ) )
  {
#ifndef TRIBOL_USE_ENZYME
    if ( jacobian_data_ ) {
      throw std::runtime_error( "Common Plane Jacobian calculation requires Enzyme." );
    }
#endif
  }

  /** @brief Compute Common Plane force and the optional Jacobian. @param mesh1 First mesh. @param mesh2 Second mesh.
   * @return Owned force; df_dx is absent when Jacobian calculation was disabled at construction. */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override;

 private:
  MfemMeshData& mesh_data_;                ///< Borrowed displacement transfer data.
  const MfemJacobianData* jacobian_data_;  ///< Borrowed Jacobian transfers, or nullptr for force-only calculation.
  PairListBuilder pair_builder_;           ///< Owned search composition.
  /// Owned pair quadrature, fixed stiffness, and gap definition.
  const CommonPlanePairFunction<BaseRule, OffsetGap<CommonPlaneGap>> function_;
};
#endif

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
