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
#include "tribol/integ/PairQuadratureRuleGenerator.hpp"
#include "tribol/assembly/PairEvaluationAssembly.hpp"
#include "tribol/assembly/PairEvaluation.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/search/PairListBuilder.hpp"
#include <utility>

namespace tribol {

/**
 * @brief Fixed-size Common Plane residual function for direct evaluation and optional Enzyme differentiation.
 * @tparam BaseRuleGenerator Reference Gauss rule generator.
 * @tparam GapDefinition Geometric or offset common-plane gap.
 */
template <typename BaseRuleGenerator, typename GapDefinition = CommonPlaneGap>
class CommonPlaneResidualPairFunction {
 public:
  using Geometry1 = LinearEdgeGeometry;                         ///< Geometry of the first directed edge.
  using Geometry2 = LinearEdgeGeometry;                         ///< Geometry of the second directed edge.
  using ResidualBasis = BasisFunction<EDGE, LINEAR_LAGRANGE>;   ///< Basis defining element residual DOFs.
  using Coords1 = typename Geometry1::Coords;                   ///< Geometry1 coordinate view.
  using Coords2 = typename Geometry2::Coords;                   ///< Geometry2 coordinate view.
  static constexpr IndexT size_geom1_coords = Coords1::size();  ///< Geometry1 coordinate count.
  static constexpr IndexT size_geom2_coords = Coords2::size();  ///< Geometry2 coordinate count.
  static constexpr IndexT size_coords = size_geom1_coords + size_geom2_coords;  ///< Combined coordinate count.
  using Result = PairResidualValue<ResidualBasis::size_basis, ResidualBasis::size_basis,
                                   Geometry1::spatial_dim>;  ///< Pair-local residual and state.
  static_assert( Geometry1::spatial_dim == Geometry2::spatial_dim );
  static constexpr IndexT size_values = Result::size_values;  ///< Residual entry count.

  /** @brief Store fixed configuration. @param rule_generator Pair quadrature rule generator. @param penalty Stiffness.
   * @param gap Gap callable.
   */
  TRIBOL_HOST_DEVICE CommonPlaneResidualPairFunction(
      PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, BaseRuleGenerator> rule_generator, RealT penalty,
      GapDefinition gap = {} )
      : rule_generator_( rule_generator ), penalty_( penalty ), gap_( gap )
  {
  }

  /** @brief Evaluate one pair residual. @param coords1 First current edge coordinates. @param coords2 Second current
   * edge coordinates.
   * A nonpositive overlap-centroid gap sets has_active_points; zero gap retains the compressive tangent.
   * @return Residual and status; failed, empty, and open pairs return zero residual and inactive state. */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE Result evaluate( Coords1 coords1, Coords2 coords2 ) const
  {
    const Geometry1 geometry1( coords1 );
    const Geometry2 geometry2( coords2 );
    const auto points = rule_generator_.generate( geometry1, geometry2 );
    Result pair_residual_data;
    pair_residual_data.quadrature_result = points.result;
    if ( points.result.status != QuadratureStatus::Success || points.result.num_required_points == 0 ) {
      return pair_residual_data;
    }
    const auto line = CommonPlaneEdgeMapping::computeCommonPlaneLine( geometry1, geometry2 );
    // The linear coordinate maps make the weighted parent coordinates the overlap centroid for every base rule.
    RealT measure = 0.0;
    RealT center1 = 0.0;
    RealT center2 = 0.0;
    for ( IndexT point = 0; point < points.result.num_required_points; ++point ) {
      measure += points.weights[point];
      center1 += points.weights[point] * points.geom1_parametric_coords( point, 0 );
      center2 += points.weights[point] * points.geom2_parametric_coords( point, 0 );
    }
    const RealT gap = gap_( geometry1.position( center1 / measure ), geometry2.position( center2 / measure ), line );
    if ( gap > 0.0 ) {
      return pair_residual_data;
    }
    pair_residual_data.has_active_points = true;
    const RealT pressure = penalty_ * gap;
    for ( IndexT point = 0; point < points.result.num_required_points; ++point ) {
      MdArray<RealT, ResidualBasis::size_basis> basis1{}, basis2{};
      ResidualBasis::eval( { &points.geom1_parametric_coords( point, 0 ), ResidualBasis::parametric_dim },
                           basis1.view() );
      ResidualBasis::eval( { &points.geom2_parametric_coords( point, 0 ), ResidualBasis::parametric_dim },
                           basis2.view() );
      for ( IndexT node = 0; node < ResidualBasis::size_basis; ++node ) {
        for ( IndexT spatial_dim = 0; spatial_dim < LinearEdgeGeometry::spatial_dim; ++spatial_dim ) {
          const RealT traction = points.weights[point] * pressure * line.normal[spatial_dim];
          pair_residual_data.geometry1( node, spatial_dim ) += basis1[node] * traction;
          pair_residual_data.geometry2( node, spatial_dim ) -= basis2[node] * traction;
        }
      }
    }
    return pair_residual_data;
  }

 private:
  PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, BaseRuleGenerator>
      rule_generator_;  ///< Owned pair quadrature rule generator.
  RealT penalty_;       ///< Effective pair stiffness.
  GapDefinition gap_;   ///< Owned gap definition.
};

#ifdef BUILD_REDECOMP
/**
 * @brief Common Plane pair search followed by residual-only evaluation and force assembly.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairFunction Fixed Common Plane residual function.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
class CommonPlaneResidualFormulation final : public ContactFormulation {
 public:
  /**
   * @brief Own the residual function and pair search and reference displacement transfer data.
   * @param mesh_data Displacement transfer data; must outlive this formulation.
   * @param pair_function Configured residual function, retained by value.
   * @param pair_builder Owned pair search composition.
   */
  CommonPlaneResidualFormulation( const MfemMeshData& mesh_data, PairFunction pair_function,
                                  PairListBuilder pair_builder )
      : mesh_data_( mesh_data ),
        pair_function_( std::move( pair_function ) ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Compute the Common Plane force without calculating a Jacobian.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned force; df_dx and other outputs are absent.
   */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return evaluateAndAssemblePairResidual<Exec>( pairs, mesh1, mesh2, mesh_data_, pair_function_ );
  }

 private:
  const MfemMeshData& mesh_data_;     ///< Displacement transfer data referenced for this formulation's lifetime.
  const PairFunction pair_function_;  ///< Owned, fixed Common Plane residual function.
  PairListBuilder pair_builder_;      ///< Owned search composition.
};

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Common Plane pair search followed by residual and coordinate-Jacobian evaluation and assembly.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairFunction Fixed Common Plane residual function.
 */
template <ExecutionMode Exec, PairFunctionType PairFunction>
class CommonPlaneResidualJacobianFormulation final : public ContactFormulation {
 public:
  /**
   * @brief Own the residual function and pair search and reference vector and matrix transfer data.
   * @param mesh_data Displacement transfer data; must outlive this formulation.
   * @param jacobian_data Coordinate-matrix transfer data; must outlive this formulation.
   * @param pair_function Configured residual function, retained by value.
   * @param pair_builder Owned pair search composition.
   */
  CommonPlaneResidualJacobianFormulation( const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                                          PairFunction pair_function, PairListBuilder pair_builder )
      : mesh_data_( mesh_data ),
        jacobian_data_( jacobian_data ),
        pair_function_( std::move( pair_function ) ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Compute the Common Plane force and its coordinate Jacobian.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned force and df_dx; other outputs are absent.
   */
  ContactFormulationResult computeNodalForces( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    return evaluateAndAssemblePairResidualJacobian<Exec>( pairs, mesh1, mesh2, mesh_data_, jacobian_data_,
                                                          pair_function_ );
  }

 private:
  const MfemMeshData& mesh_data_;          ///< Displacement transfer data referenced for this formulation's lifetime.
  const MfemJacobianData& jacobian_data_;  ///< Matrix transfer data referenced for this formulation's lifetime.
  const PairFunction pair_function_;       ///< Owned, fixed Common Plane residual function.
  PairListBuilder pair_builder_;           ///< Owned search composition.
};
#endif  // TRIBOL_USE_ENZYME
#endif

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_COMMONPLANEFORMULATION_HPP_
