// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Nodal Energy Mortar integrals, penalty and Lagrange multiplier enforcement, and contact formulations.
 */
#ifndef SRC_TRIBOL_PHYSICS_NODALENERGYMORTAR_HPP_
#define SRC_TRIBOL_PHYSICS_NODALENERGYMORTAR_HPP_

#include "tribol/config.hpp"
#include "axom/slic/interface/slic.hpp"
#include "axom/slic/interface/slic_macros.hpp"
#include "tribol/enforcement/PenaltyEnergy.hpp"
#include "tribol/geom/OpposingNormalGap.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"
#include "tribol/integ/BasisFunction.hpp"

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )
#include "tribol/assembly/AssemblePairs.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/search/PairListBuilder.hpp"
#include <utility>
#include "tribol/enforcement/PenaltyStiffness.hpp"
#endif

namespace tribol {

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )

namespace detail {
/** @brief Energy Mortar's assembled weighted gaps and tributary areas, before activation. */
struct EnergyMortarNodalQuantities {
  AssembledField weighted_gap;    ///< Weighted gaps and their coordinate Jacobian.
  AssembledField tributary_area;  ///< Tributary areas and their coordinate Jacobian.
};

/**
 * @brief Evaluate and assemble weighted gaps, tributary areas, and their coordinate Jacobians.
 * @tparam PairFunction Pair function returning weighted gaps followed by areas.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 * @param pairs Current accepted pair list.
 * @param mesh1 Current first mesh.
 * @param mesh2 Current second mesh.
 * @param pair_function Fixed pair-function configuration.
 * @param mesh_data Current displacement transfer data.
 * @param submesh_data Scalar field transfer data.
 * @param jacobian_data Parent and scalar-field Jacobian transfer data.
 * @return Unmodified weighted gaps, areas, and their coordinate Jacobians, including allocated zeros for empty pairs.
 * @note Reports a SLIC error if geometry or assembly dimensions are invalid.
 * @note Reports a SLIC error if quadrature capacity or the assembly index range is exceeded.
 */
template <ExecutionMode Exec, typename PairFunction>
EnergyMortarNodalQuantities evaluateAndAssembleNodalQuantities( const ArrayT<InterfacePair>& pairs,
                                                                MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                                const PairFunction& pair_function,
                                                                const MfemMeshData& mesh_data,
                                                                const MfemSubmeshData& submesh_data,
                                                                const MfemJacobianData& jacobian_data )
{
  using PairNodalGap = typename PairFunction::PairNodalGap;
  constexpr IndexT num_nodes = PairFunction::num_nodes;
  constexpr IndexT size_coords = PairFunction::size_coords;
  static_assert( PairFunction::size_values == 2 * num_nodes );
  auto buffer = evaluatePairValuesAndJacobians<Exec>( pairs, mesh1, mesh2, pair_function );
  const auto contributions_view = buffer.view();
  checkPairQuadratureMetadata<Exec>( buffer.quadratureMetadata() );
  constexpr IndexT num_fields = PairNodalGap::num_fields;
  auto fields = detail::assembleNodalValuesAndJacobians<Exec, num_fields, num_nodes, size_coords>(
      pairs, mesh1, mesh2, mesh_data, submesh_data, jacobian_data,
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT field, IndexT node ) {
        const auto integral_field = static_cast<typename PairNodalGap::Field>( field );
        return contributions_view.value( pair, PairNodalGap::valueIndex( integral_field, node ) );
      },
      [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT field, IndexT node, IndexT coordinate ) {
        const auto integral_field = static_cast<typename PairNodalGap::Field>( field );
        return contributions_view.jacobian( pair, PairNodalGap::valueIndex( integral_field, node ), coordinate );
      } );
  return { std::move( fields[0] ), std::move( fields[1] ) };
}

/**
 * @brief Assemble the coordinate Hessian of fixed nodal weights contracted with pair gap and area integrals.
 * @tparam PairFunction Pair function returning weighted gaps followed by areas.
 * @tparam WeightReader Device-callable node reader returning gap and area weights in that order.
 * @tparam Exec Concrete execution mode for evaluation and numerical assembly.
 * @param pairs Current accepted pair list.
 * @param mesh1 Current first mesh, whose scalar finite-element nodes carry the weights.
 * @param mesh2 Current second mesh.
 * @param pair_function Fixed pair-function configuration.
 * @param mesh_data Current displacement transfer data.
 * @param jacobian_data Current Jacobian transfer data.
 * @param read_weights Reader indexed by scalar submesh node.
 * @return Coordinate Hessian assembled on parent true DOFs.
 * @note Reports a SLIC error if geometry or assembly dimensions are invalid.
 * @note Reports a SLIC error if pair buffers or matrix storage exceed the supported index range.
 */
template <ExecutionMode Exec, typename PairFunction, typename WeightReader>
shared::ParSparseMat assembleWeightedNodalHessian( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                                   MeshData::Viewer mesh2, const PairFunction& pair_function,
                                                   const MfemMeshData& mesh_data, const MfemJacobianData& jacobian_data,
                                                   WeightReader read_weights )
{
  using PairNodalGap = typename PairFunction::PairNodalGap;
  constexpr IndexT num_nodes = PairFunction::num_nodes;
  static_assert( PairFunction::size_values == 2 * num_nodes );
  const IndexT num_pairs = pairs.size();
  const auto pairs_view = pairs.view();
  const auto connectivity = mesh1.getConnectivity();
  auto hessians = evaluatePairWeightedJacobiansAndHessians<Exec>(
      pairs, mesh1, mesh2, pair_function, [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
        MdArray<RealT, PairFunction::size_values> weights{};
        const auto nodes = connectivity( pairs_view[pair_index].element_id1_ );
        for ( IndexT node = 0; node < num_nodes; ++node ) {
          const auto nodal_weights = read_weights( nodes[node] );
          weights[PairNodalGap::weightedGapIndex( node )] = nodal_weights[0];
          weights[PairNodalGap::tributaryAreaIndex( node )] = nodal_weights[1];
        }
        return weights;
      } );
  const auto hessian_view = hessians.view();
  auto assembler = detail::makeResidualJacobianAssembler( mesh_data, jacobian_data, mesh1, mesh2, num_pairs,
                                                          pairs.getAllocatorID() );
  assembler.addPairs<Exec, PairFunction::size_coords / 2, PairFunction::size_coords / 2, 4>(
      pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair, IndexT row, IndexT column ) {
        return hessian_view.hessian( pair, row, column );
      } );
  return assembler.assemble();
}

}  // namespace detail

/** @brief Penalty enforcement applied after nodal Energy Mortar gap and area assembly. */
struct NodalPenaltyEnforcement {
  /**
   * @brief Store a validated penalty stiffness.
   * @param penalty_value Positive, finite penalty stiffness.
   * @note Reports a SLIC error if the stiffness is nonpositive or nonfinite.
   */
  explicit NodalPenaltyEnforcement( RealT penalty_value ) : penalty( detail::checkedPenaltyStiffness( penalty_value ) )
  {
  }

  /**
   * @brief Accept inputs because penalty enforcement does not require caller-supplied fields.
   * @param inputs Unused caller-supplied fields.
   * @param submesh_data Unused scalar transfer data.
   * @return Always true.
   */
  bool hasValidInputs( [[maybe_unused]] const ContactFormulationInputs& inputs,
                       [[maybe_unused]] const MfemSubmeshData& submesh_data ) const
  {
    return true;
  }

  /**
   * @brief Convert assembled nodal quantities to normalized compressive gaps.
   * @param quantities Assembled weighted gaps and areas, consumed for the result.
   * @return Owned gap and weighted-gap Jacobian; other outputs are absent.
   */
  ContactFormulationResult computeGaps( detail::EnergyMortarNodalQuantities& quantities ) const;

  /**
   * @brief Apply penalty enforcement and calculate residuals and derivatives.
   * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
   * @tparam PairFunction Pair function returning weighted-gap integrals followed by tributary areas.
   * @param pairs Accepted element pairs; not retained.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @param mesh_data Displacement transfer data used for assembly.
   * @param submesh_data Pressure and gap transfer data used for assembly.
   * @param jacobian_data Parent and scalar-field Jacobian transfer data used for assembly.
   * @param pair_function Configured nodal-integral pair function.
   * @param quantities Assembled weighted gaps and areas, consumed for the result.
   * @param inputs Unused caller-supplied fields.
   * @return Owned gaps, pressure, energy, residual, weighted-gap Jacobian, and df_dx; df_dp is absent.
   */
  template <ExecutionMode Exec, PairFunctionType PairFunction>
  ContactFormulationResult computeResidual( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                            MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
                                            const MfemSubmeshData& submesh_data, const MfemJacobianData& jacobian_data,
                                            const PairFunction& pair_function,
                                            detail::EnergyMortarNodalQuantities& quantities,
                                            [[maybe_unused]] const ContactFormulationInputs& inputs ) const
  {
    const auto make_vector = [&] {
      const auto memory_space = isOnDevice( Exec ) ? shared::ParVectorMemory::Device : shared::ParVectorMemory::Host;
      shared::ParVector vector( memory_space,
                                const_cast<mfem::ParFiniteElementSpace*>( &submesh_data.GetSubmeshFESpace() ) );
      vector.fill( 0.0 );
      return vector;
    };
    auto gap_weights = make_vector();
    auto area_weights = make_vector();
    auto d2energy_dweighted_gap2 = make_vector();
    auto d2energy_dweighted_gap_dtributary_area = make_vector();
    auto d2energy_dtributary_area2 = make_vector();
    const int allocator_id = pairs.getAllocatorID();
    Array1D<RealT> nodal_energy( gap_weights.size(), gap_weights.size(), allocator_id );
    const bool use_device = isOnDevice( Exec );
    const auto* weighted_gap = quantities.weighted_gap.values.get().Read( use_device );
    const auto* area = quantities.tributary_area.values.get().Read( use_device );
    auto* gap_weight = gap_weights.get().Write( use_device );
    auto* area_weight = area_weights.get().Write( use_device );
    auto* gap_gap_data = d2energy_dweighted_gap2.get().Write( use_device );
    auto* gap_area_data = d2energy_dweighted_gap_dtributary_area.get().Write( use_device );
    auto* area_area_data = d2energy_dtributary_area2.get().Write( use_device );
    auto* nodal_energy_data = nodal_energy.data();
    const RealT penalty_stiffness = penalty;
    forAllExec<Exec>( gap_weights.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) {
      const auto partials =
          EnergyMortarNodalPenaltyPartials::evaluate( weighted_gap[node], area[node], penalty_stiffness );
      nodal_energy_data[node] = partials.energy;
      gap_weight[node] = partials.denergy_dweighted_gap;
      area_weight[node] = partials.denergy_dtributary_area;
      gap_gap_data[node] = partials.d2energy_dweighted_gap2;
      gap_area_data[node] = partials.d2energy_dweighted_gap_dtributary_area;
      area_area_data[node] = partials.d2energy_dtributary_area2;
    } );
    const RealT energy = GlobalScalarAssembler::sum<Exec>(
        gap_weights.size(), [=] TRIBOL_HOST_DEVICE( IndexT node ) { return nodal_energy_data[node]; },
        gap_weights.get().GetComm() );
    auto assembled_residual = gap_weights * quantities.weighted_gap.jacobian;
    assembled_residual += area_weights * quantities.tributary_area.jacobian;

    const auto redecomp_gap_weights = submesh_data.TransferTrueDofsToRedecomp( gap_weights.get() );
    const auto redecomp_area_weights = submesh_data.TransferTrueDofsToRedecomp( area_weights.get() );
    const auto* gap_data = redecomp_gap_weights.Read( use_device );
    const auto* area_data = redecomp_area_weights.Read( use_device );
    auto jacobian = detail::assembleWeightedNodalHessian<Exec>(
        pairs, mesh1, mesh2, pair_function, mesh_data, jacobian_data,
        [=] TRIBOL_HOST_DEVICE( IndexT node ) { return MdArray<RealT, 2>{ gap_data[node], area_data[node] }; } );
    const auto make_diagonal = [&]( const shared::ParVector& values ) {
      const auto& fes = submesh_data.GetSubmeshFESpace();
      return shared::ParSparseMat::diagonalMatrix( fes.GetComm(), fes.GlobalTrueVSize(), fes.GetTrueDofOffsets(),
                                                   values.get() );
    };
    const auto gg = make_diagonal( d2energy_dweighted_gap2 );
    const auto ga = make_diagonal( d2energy_dweighted_gap_dtributary_area );
    const auto aa = make_diagonal( d2energy_dtributary_area2 );
    jacobian += shared::ParSparseMat::rap( gg, quantities.weighted_gap.jacobian );
    const auto mixed =
        shared::ParSparseMat::rap( quantities.weighted_gap.jacobian, ga, quantities.tributary_area.jacobian );
    jacobian += mixed;
    jacobian += mixed.transpose();
    jacobian += shared::ParSparseMat::rap( aa, quantities.tributary_area.jacobian );

    auto result = computeGaps( quantities );
    result.energy = energy;
    result.residual.reset( assembled_residual.release() );
    result.df_dx.reset( jacobian.release() );
    result.pressure.reset( gap_weights.release() );
    return result;
  }

  RealT penalty;  ///< Fixed positive, finite penalty stiffness.
};

/** @brief Lagrange multiplier enforcement applied after nodal Energy Mortar gap and area assembly. */
struct NodalLagrangeMultiplierEnforcement {
  /**
   * @brief Check that caller-supplied multiplier values match the dual true DOFs.
   * @param inputs Caller-supplied fields containing required multiplier values.
   * @param submesh_data Scalar transfer data defining the dual true DOFs.
   * @return True when multiplier values are present with the required size.
   * @note Reports a SLIC error when the multiplier is absent or has the wrong size.
   */
  bool hasValidInputs( const ContactFormulationInputs& inputs, const MfemSubmeshData& submesh_data ) const
  {
    if ( !inputs.multiplier ) {
      SLIC_ERROR( "Lagrange multiplier Energy Mortar requires LM values in ContactFormulationInputs." );
      return false;
    }
    if ( inputs.multiplier->Size() != submesh_data.GetSubmeshFESpace().GetTrueVSize() ) {
      SLIC_ERROR( "LM values must match dual true DOFs." );
      return false;
    }
    return true;
  }

  /**
   * @brief Convert assembled nodal quantities to signed weighted gaps.
   * @param quantities Assembled weighted gaps and areas, consumed for the result.
   * @return Owned gap and weighted-gap Jacobian; other outputs are absent.
   */
  ContactFormulationResult computeGaps( detail::EnergyMortarNodalQuantities& quantities ) const;

  /**
   * @brief Apply Lagrange multiplier enforcement and calculate residuals and derivatives.
   * @tparam Exec Concrete execution mode for evaluation and assembly kernels.
   * @tparam PairFunction Pair function returning weighted-gap integrals followed by tributary areas.
   * @param pairs Accepted element pairs; not retained.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @param mesh_data Displacement transfer data used for assembly.
   * @param submesh_data Pressure and gap transfer data used for assembly.
   * @param jacobian_data Parent and scalar-field Jacobian transfer data used for assembly.
   * @param pair_function Configured nodal-integral pair function.
   * @param quantities Assembled weighted gaps and areas, consumed for the result.
   * @param inputs Caller-supplied fields containing validated multiplier values.
   * @return Owned gaps, pressure, energy, residual, weighted-gap Jacobian, df_dx, and df_dp.
   */
  template <ExecutionMode Exec, PairFunctionType PairFunction>
  ContactFormulationResult computeResidual( const ArrayT<InterfacePair>& pairs, MeshData::Viewer mesh1,
                                            MeshData::Viewer mesh2, const MfemMeshData& mesh_data,
                                            const MfemSubmeshData& submesh_data, const MfemJacobianData& jacobian_data,
                                            const PairFunction& pair_function,
                                            detail::EnergyMortarNodalQuantities& quantities,
                                            const ContactFormulationInputs& inputs ) const
  {
    shared::ParVector pressure( *inputs.multiplier );
    const RealT energy = pressure.dot( quantities.weighted_gap.values );
    auto assembled_residual = pressure * quantities.weighted_gap.jacobian;
    const auto redecomp_pressure = submesh_data.TransferTrueDofsToRedecomp( pressure.get() );
    const auto* pressure_data = redecomp_pressure.Read( isOnDevice( Exec ) );
    auto jacobian = detail::assembleWeightedNodalHessian<Exec>(
        pairs, mesh1, mesh2, pair_function, mesh_data, jacobian_data,
        [=] TRIBOL_HOST_DEVICE( IndexT node ) { return MdArray<RealT, 2>{ pressure_data[node], 0.0 }; } );
    auto df_dp = quantities.weighted_gap.jacobian.transpose();

    auto result = computeGaps( quantities );
    result.energy = energy;
    result.residual.reset( assembled_residual.release() );
    result.df_dx.reset( jacobian.release() );
    result.df_dp.reset( df_dp.release() );
    result.pressure.reset( pressure.release() );
    return result;
  }
};

/**
 * @brief Nodal Energy Mortar with enforcement selected by a policy object.
 *
 * Uses PairNodalGapIntegral with a linear Lagrange basis on geometry1, then applies enforcement after weighted gaps and
 * tributary areas are assembled.
 * @tparam Exec Concrete execution mode for search, evaluation, and assembly kernels.
 * @tparam PairRuleGenerator Generates quadrature points and physical weights for the current edge pair.
 * @tparam GapDefinition Geometric or offset alignment-corrected gap.
 * @tparam Enforcement Converts assembled nodal quantities to gaps, residuals, and derivatives.
 */
template <ExecutionMode Exec, typename PairRuleGenerator, typename GapDefinition, typename Enforcement>
class NodalEnergyMortar final : public ContactFormulation {
 public:
  /**
   * @brief Construct the nodal gap integral, own the enforcement and search, and reference required transfer data.
   * @param mesh_data Displacement transfer data; must outlive this formulation.
   * @param submesh_data Pressure and gap transfer data; must outlive this formulation.
   * @param jacobian_data Parent and scalar-field Jacobian transfer data; must outlive this formulation.
   * @param rule_generator Configured pair quadrature generator, retained by value.
   * @param gap Configured alignment-corrected gap, retained by value.
   * @param enforcement Configured nodal enforcement, retained by value.
   * @param pair_builder Owned pair search.
   */
  NodalEnergyMortar( const MfemMeshData& mesh_data, const MfemSubmeshData& submesh_data,
                     const MfemJacobianData& jacobian_data, PairRuleGenerator rule_generator, GapDefinition gap,
                     Enforcement enforcement, PairListBuilder pair_builder )
      : mesh_data_( mesh_data ),
        submesh_data_( submesh_data ),
        jacobian_data_( jacobian_data ),
        pair_integral_( std::move( rule_generator ), std::move( gap ) ),
        enforcement_( std::move( enforcement ) ),
        pair_builder_( std::move( pair_builder ) )
  {
  }

  /**
   * @brief Calculate nodal gaps and their coordinate Jacobian using the selected enforcement.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @return Owned gap and weighted-gap Jacobian; other outputs are absent.
   */
  ContactFormulationResult computeNodalGaps( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) const override
  {
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    auto quantities = detail::evaluateAndAssembleNodalQuantities<Exec>( pairs, mesh1, mesh2, pair_integral_, mesh_data_,
                                                                        submesh_data_, jacobian_data_ );
    return enforcement_.computeGaps( quantities );
  }

  /**
   * @brief Calculate nodal energy, gaps, pressure, residual, and derivatives using the selected enforcement.
   * @param mesh1 Current first mesh.
   * @param mesh2 Current second mesh.
   * @param inputs Caller-supplied fields required by the selected enforcement.
   * @return Owned formulation outputs selected by the enforcement.
   */
  ContactFormulationResult computeNodalResiduals( MeshData::Viewer mesh1, MeshData::Viewer mesh2,
                                                  const ContactFormulationInputs& inputs ) const override
  {
    if ( !enforcement_.hasValidInputs( inputs, submesh_data_ ) ) {
      return {};
    }
    const auto pairs = pair_builder_.build( mesh1, mesh2 );
    auto quantities = detail::evaluateAndAssembleNodalQuantities<Exec>( pairs, mesh1, mesh2, pair_integral_, mesh_data_,
                                                                        submesh_data_, jacobian_data_ );
    return enforcement_.template computeResidual<Exec>( pairs, mesh1, mesh2, mesh_data_, submesh_data_, jacobian_data_,
                                                        pair_integral_, quantities, inputs );
  }

 private:
  const MfemMeshData& mesh_data_;          ///< Displacement transfer data referenced for this formulation's lifetime.
  const MfemSubmeshData& submesh_data_;    ///< Scalar transfer data referenced for this formulation's lifetime.
  const MfemJacobianData& jacobian_data_;  ///< Matrix transfer data referenced for this formulation's lifetime.
  const PairNodalGapIntegral<PairRuleGenerator, GapDefinition, BasisFunction<EDGE, LINEAR_LAGRANGE>>
      pair_integral_;              ///< Integrates weighted gaps and tributary areas before contact enforcement.
  const Enforcement enforcement_;  ///< Owns enforcement-specific validation and output assembly.
  PairListBuilder pair_builder_;   ///< Owned pair search.
};

#endif

}  // namespace tribol
#endif  // SRC_TRIBOL_PHYSICS_NODALENERGYMORTAR_HPP_
