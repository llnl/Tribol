#ifndef TRIBOL_EVALUATION_VARIATIONALEVALUATOR_INL_
#define TRIBOL_EVALUATION_VARIATIONALEVALUATOR_INL_

namespace variational_detail {

inline constexpr int maximumLocalCoordinates = 2 * basis::maximumLinearNodes * 3;

template <typename Scalar>
using LocalGradient = linearization_detail::StaticGradient<Scalar, maximumLocalCoordinates>;

template <typename Scalar>
struct PairKinematics {
  std::array<Scalar, basis::maximumLinearNodes> weighted_gap{};
  std::array<Scalar, basis::maximumLinearNodes> area{};
  std::array<Scalar, integration::maximumQuadraturePoints> gap{};
  std::array<Scalar, integration::maximumQuadraturePoints> weight{};
  int mortar_nodes{};
  int quadrature_points{};
  bool valid{};
};

template <typename Scalar>
[[nodiscard]] inline SurfaceMeshViewT<LocalGradient<Scalar>> localSurface(
    const SurfaceMeshViewT<Scalar>& source, Index element, int side,
    std::array<LocalGradient<Scalar>, basis::maximumLinearNodes * 3>& coordinates, std::array<Index, 2>& offsets,
    std::array<Index, basis::maximumLinearNodes>& connectivity, std::array<ElementTopology, 1>& topology,
    std::array<int, 1>& attributes )
{
  const Index begin = source.element_offsets[element];
  const Index nodes = source.element_offsets[element + 1] - begin;
  offsets = { 0, nodes };
  topology[0] = source.topologies[element];
  attributes[0] = source.attributes.empty() ? 0 : source.attributes[element];
  for ( Index local_node = 0; local_node < nodes; ++local_node ) {
    connectivity[local_node] = local_node;
    const Index global_node = source.connectivity[begin + local_node];
    for ( int component = 0; component < source.dimension; ++component ) {
      const int local_coordinate = ( side * basis::maximumLinearNodes + local_node ) * source.dimension + component;
      coordinates[static_cast<std::size_t>( local_node * source.dimension + component )] =
          LocalGradient<Scalar>::variable( source.coordinates( global_node, component ), local_coordinate );
    }
  }
  return {
      .dimension = source.dimension,
      .coordinates = { { coordinates.data(), nodes * source.dimension },
                       nodes,
                       source.dimension,
                       FieldLayout::Interleaved },
      .element_offsets = { offsets.data(), 2 },
      .connectivity = { connectivity.data(), nodes },
      .topologies = { topology.data(), 1 },
      .attributes = { attributes.data(), 1 },
  };
}

template <typename MethodType, typename Scalar>
[[nodiscard]] inline PairKinematics<LocalGradient<Scalar>> pairKinematics(
    const SurfacePairViewT<Scalar>& surfaces, ElementPair pair, const typename MethodType::Parameters& parameters )
{
  using Gradient = LocalGradient<Scalar>;
  using Geometry = typename MethodType::geometry_policy;
  using Integration = typename MethodType::integration_policy;
  using Constraint = typename MethodType::constraint_policy;
  std::array<Gradient, basis::maximumLinearNodes * 3> mortar_coordinates{};
  std::array<Gradient, basis::maximumLinearNodes * 3> nonmortar_coordinates{};
  std::array<Index, 2> mortar_offsets{};
  std::array<Index, 2> nonmortar_offsets{};
  std::array<Index, basis::maximumLinearNodes> mortar_connectivity{};
  std::array<Index, basis::maximumLinearNodes> nonmortar_connectivity{};
  std::array<ElementTopology, 1> mortar_topology{};
  std::array<ElementTopology, 1> nonmortar_topology{};
  std::array<int, 1> mortar_attributes{};
  std::array<int, 1> nonmortar_attributes{};
  const SurfacePairViewT<Gradient> local{
      localSurface( surfaces.mortar, pair.mortar_element, 0, mortar_coordinates, mortar_offsets, mortar_connectivity,
                    mortar_topology, mortar_attributes ),
      localSurface( surfaces.nonmortar, pair.nonmortar_element, 1, nonmortar_coordinates, nonmortar_offsets,
                    nonmortar_connectivity, nonmortar_topology, nonmortar_attributes ),
  };
  const auto patch = policy_evaluator_detail::makePatch<Geometry, Integration>( local, { 0, 0 }, parameters.geometry,
                                                                                parameters.integration );
  PairKinematics<Gradient> result;
  if ( !patch.valid ) {
    return result;
  }
  const auto quadrature = policy_evaluator_detail::makeQuadrature<Integration>( patch );
  result.mortar_nodes = static_cast<int>( mortar_offsets[1] );
  result.quadrature_points = quadrature.size;
  result.valid = true;
  for ( int point = 0; point < quadrature.size; ++point ) {
    const auto& quadrature_point = quadrature[point];
    const auto sample = constraint::stageNormalConstraint<Constraint>(
        local, { 0, 0 }, quadrature_point.mortar_position, quadrature_point.nonmortar_position, quadrature_point.normal,
        quadrature_point.weight );
    const Gradient effective_gap = sample.gap - parameters.constraint.activation.residual_gap;
    result.gap[point] = effective_gap;
    result.weight[point] = sample.weight;
    for ( int node = 0; node < sample.mortar_test.size; ++node ) {
      result.weighted_gap[node] += sample.weight * sample.mortar_test[node] * effective_gap;
      result.area[node] += sample.weight * sample.mortar_test[node];
    }
  }
  return result;
}

template <typename Scalar>
inline void scatterGradient( const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
                             const LocalGradient<Scalar>& energy, ContactResidualViewT<Scalar> residual )
{
  const SurfaceMeshViewT<Scalar> meshes[2] = { surfaces.mortar, surfaces.nonmortar };
  const Index elements[2] = { pair.mortar_element, pair.nonmortar_element };
  FieldView<Scalar> fields[2] = { residual.mortar, residual.nonmortar };
  for ( int side = 0; side < 2; ++side ) {
    if ( !policy_evaluator_detail::hasField( fields[side] ) ) {
      continue;
    }
    const Index begin = meshes[side].element_offsets[elements[side]];
    const Index nodes = meshes[side].element_offsets[elements[side] + 1] - begin;
    for ( Index local_node = 0; local_node < nodes; ++local_node ) {
      const Index global_node = meshes[side].connectivity[begin + local_node];
      for ( int component = 0; component < meshes[side].dimension; ++component ) {
        const int local_coordinate =
            ( side * basis::maximumLinearNodes + local_node ) * meshes[side].dimension + component;
        fields[side]( global_node, component ) += energy.gradient[local_coordinate];
      }
    }
  }
}

template <typename Scalar>
[[nodiscard]] inline Scalar externalPressure( const ContactStateViewT<Scalar>& state, Index node, Scalar gap )
{
  Scalar pressure = state.external_pressure[node];
  if ( !state.external_pressure_tangent.empty() ) {
    pressure += state.external_pressure_tangent[node] * ( gap - Scalar{ linearization_detail::primal( gap ) } );
  }
  return pressure;
}

template <typename Scalar>
[[nodiscard]] inline Scalar externalPotential( const ContactStateViewT<Scalar>& state, Index node, Scalar gap )
{
  Scalar potential{};
  if ( !state.external_potential_density.empty() ) {
    potential = state.external_potential_density[node] +
                state.external_pressure[node] * ( gap - Scalar{ linearization_detail::primal( gap ) } );
  }
  return potential;
}

template <typename MethodType, typename Scalar>
[[nodiscard]] inline LocalGradient<Scalar> pairEnergy( const SurfacePairViewT<Scalar>& surfaces, ElementPair pair,
                                                       const typename MethodType::Parameters& parameters,
                                                       const ContactStateViewT<Scalar>& state,
                                                       ContactOutputViewT<Scalar> output )
{
  using Constraint = typename MethodType::constraint_policy;
  using Enforcement = typename MethodType::enforcement_policy;
  using Gradient = LocalGradient<Scalar>;
  const auto values = pairKinematics<MethodType>( surfaces, pair, parameters );
  Gradient energy{};
  if ( !values.valid ) {
    return energy;
  }
  if constexpr ( std::same_as<Constraint, constraint::QuadraturePoint> ) {
    const Scalar stiffness =
        pointwise_penalty_detail::penaltyStiffness<Enforcement>( parameters.enforcement, state, pair );
    for ( int point = 0; point < values.quadrature_points; ++point ) {
      const Gradient gap =
          linearization_detail::primal( values.gap[point] ) <= parameters.constraint.activation.gap_tolerance
              ? values.gap[point]
              : Gradient{};
      energy += 0.5 * values.weight[point] * Gradient{ stiffness } * gap * gap;
    }
  } else {
    for ( int local_node = 0; local_node < values.mortar_nodes; ++local_node ) {
      const Index node = policy_evaluator_detail::globalNode( surfaces.mortar, pair.mortar_element, local_node );
      if constexpr ( detail::is_penalty_v<Enforcement> ) {
        const Scalar stiffness =
            pointwise_penalty_detail::penaltyStiffness<Enforcement>( parameters.enforcement, state, pair );
        const Scalar gap = output.gap[node];
        const Scalar pressure = output.pressure[node];
        const Scalar active_gap =
            linearization_detail::primal( gap ) <= parameters.constraint.activation.gap_tolerance ? gap : Scalar{};
        const Scalar potential = 0.5 * stiffness * active_gap * active_gap;
        energy += Gradient{ pressure } * values.weighted_gap[local_node] +
                  Gradient{ potential - pressure * gap } * values.area[local_node];
      } else if constexpr ( std::same_as<Enforcement, enforcement::LagrangeMultiplier> ) {
        energy += Gradient{ state.multiplier[node] } * values.weighted_gap[local_node];
      } else {
        const Scalar gap = output.gap[node];
        const Scalar pressure = output.pressure[node];
        const Scalar potential = externalPotential( state, node, gap );
        energy += Gradient{ pressure } * values.weighted_gap[local_node] +
                  Gradient{ potential - pressure * gap } * values.area[local_node];
      }
    }
  }
  return energy;
}

}  // namespace variational_detail

template <SupportedMethod MethodType, typename Scalar>
void validateVariationalEvaluation( const SurfacePairViewT<Scalar>& surfaces, const ContactStateViewT<Scalar>& state,
                                    ContactOutputViewT<Scalar> output )
{
  using Constraint = typename MethodType::constraint_policy;
  policy_evaluator_detail::validateEvaluation<MethodType>( surfaces, state, output );
  if constexpr ( detail::is_nodal_constraint_v<Constraint> ) {
    if ( surfaces.mortar.numberOfNodes() > 0 && ( output.gap.empty() || output.weighted_gap.empty() ||
                                                  output.tributary_area.empty() || output.pressure.empty() ) ) {
      throw std::invalid_argument(
          "Nodal variational contact requires gap, weighted-gap, area, and pressure work arrays." );
    }
  }
}

template <SupportedMethod MethodType, typename Scalar>
EvaluationSummary stageVariationalKinematics( const SurfacePairViewT<Scalar>& surfaces,
                                              ArrayView<const ElementPair> interactions,
                                              const typename MethodType::Parameters& parameters,
                                              const ContactStateViewT<Scalar>& state,
                                              ContactOutputViewT<Scalar> output )
{
  using Geometry = typename MethodType::geometry_policy;
  using Integration = typename MethodType::integration_policy;
  using Constraint = typename MethodType::constraint_policy;
  using Enforcement = typename MethodType::enforcement_policy;
  EvaluationSummary summary;
  Index quadrature_offset{};
  for ( const ElementPair pair : interactions ) {
    const auto patch = policy_evaluator_detail::makePatch<Geometry, Integration>( surfaces, pair, parameters.geometry,
                                                                                  parameters.integration );
    if ( !patch.valid ) {
      continue;
    }
    const auto quadrature = policy_evaluator_detail::makeQuadrature<Integration>( patch );
    const Scalar stiffness = [&]() {
      if constexpr ( detail::is_penalty_v<Enforcement> ) {
        return pointwise_penalty_detail::penaltyStiffness<Enforcement>( parameters.enforcement, state, pair );
      } else {
        return Scalar{};
      }
    }();
    for ( int point = 0; point < quadrature.size; ++point ) {
      const auto& quadrature_point = quadrature[point];
      const auto sample = constraint::stageNormalConstraint<Constraint>(
          surfaces, pair, quadrature_point.mortar_position, quadrature_point.nonmortar_position,
          quadrature_point.normal, quadrature_point.weight );
      const Scalar effective_gap = sample.gap - parameters.constraint.activation.residual_gap;
      policy_evaluator_detail::addDiagnostics( surfaces, pair, sample, effective_gap, output );
      Scalar quadrature_pressure{};
      if constexpr ( std::same_as<Constraint, constraint::QuadraturePoint> ) {
        quadrature_pressure = stiffness * ( linearization_detail::primal( effective_gap ) <=
                                                    parameters.constraint.activation.gap_tolerance
                                                ? effective_gap
                                                : Scalar{} );
        summary.energy +=
            linearization_detail::primal( 0.5 * sample.weight * quadrature_pressure * quadrature_pressure / stiffness );
      } else if constexpr ( std::same_as<Enforcement, enforcement::LagrangeMultiplier> ) {
        if ( !output.quadrature_pressure.empty() ) {
          for ( int local_node = 0; local_node < sample.mortar_trial.size; ++local_node ) {
            const Index node = policy_evaluator_detail::globalNode( surfaces.mortar, pair.mortar_element, local_node );
            quadrature_pressure += sample.mortar_trial[local_node] * state.multiplier[node];
          }
        }
      } else if constexpr ( std::same_as<Enforcement, enforcement::ExternalPressure> ) {
        if ( !output.quadrature_pressure.empty() ) {
          for ( int local_node = 0; local_node < sample.mortar_trial.size; ++local_node ) {
            const Index node = policy_evaluator_detail::globalNode( surfaces.mortar, pair.mortar_element, local_node );
            quadrature_pressure += sample.mortar_trial[local_node] * state.external_pressure[node];
          }
        }
      }
      if ( !output.quadrature_gap.empty() ) {
        output.quadrature_gap[quadrature_offset] = effective_gap;
      }
      if ( !output.quadrature_pressure.empty() ) {
        output.quadrature_pressure[quadrature_offset] = quadrature_pressure;
      }
      if constexpr ( std::same_as<Constraint, constraint::QuadraturePoint> ) {
        if ( !output.pressure.empty() ) {
          for ( int local_node = 0; local_node < sample.mortar_test.size; ++local_node ) {
            const Index node = policy_evaluator_detail::globalNode( surfaces.mortar, pair.mortar_element, local_node );
            output.pressure[node] += sample.weight * sample.mortar_test[local_node] * quadrature_pressure;
          }
        }
      }
      ++quadrature_offset;
    }
    ++summary.active_interactions;
  }
  summary.quadrature_points = quadrature_offset;
  return summary;
}

template <SupportedMethod MethodType, typename Scalar>
void finalizeVariationalKinematics( const SurfacePairViewT<Scalar>& surfaces,
                                    const typename MethodType::Parameters& parameters,
                                    const ContactStateViewT<Scalar>& state, ContactOutputViewT<Scalar> output,
                                    EvaluationSummary& summary )
{
  using Constraint = typename MethodType::constraint_policy;
  using Enforcement = typename MethodType::enforcement_policy;
  policy_evaluator_detail::finalizeGap( output );
  if constexpr ( detail::is_nodal_constraint_v<Constraint> ) {
    for ( Index node = 0; node < surfaces.mortar.numberOfNodes(); ++node ) {
      if ( linearization_detail::absolute( linearization_detail::primal( output.tributary_area[node] ) ) <= 1.0e-28 ) {
        continue;
      }
      if constexpr ( detail::is_penalty_v<Enforcement> ) {
        const Scalar stiffness = parameters.enforcement.stiffness.value;
        const Scalar gap = output.gap[node];
        const Scalar active_gap =
            linearization_detail::primal( gap ) <= parameters.constraint.activation.gap_tolerance ? gap : Scalar{};
        output.pressure[node] = stiffness * active_gap;
        summary.energy +=
            linearization_detail::primal( 0.5 * output.tributary_area[node] * stiffness * active_gap * active_gap );
      } else if constexpr ( std::same_as<Enforcement, enforcement::LagrangeMultiplier> ) {
        output.pressure[node] = state.multiplier[node];
        output.residual.constraint[node] += output.weighted_gap[node];
      } else {
        const Scalar gap = output.gap[node];
        output.pressure[node] = variational_detail::externalPressure( state, node, gap );
        summary.energy += linearization_detail::primal( variational_detail::externalPotential( state, node, gap ) *
                                                        output.tributary_area[node] );
      }
    }
  } else if ( !output.pressure.empty() ) {
    for ( Index node = 0; node < output.pressure.size(); ++node ) {
      output.pressure[node] =
          linearization_detail::absolute( linearization_detail::primal( output.tributary_area[node] ) ) > 1.0e-28
              ? output.pressure[node] / output.tributary_area[node]
              : Scalar{};
    }
  }
}

template <SupportedMethod MethodType, typename Scalar>
void addVariationalForces( const SurfacePairViewT<Scalar>& surfaces, ArrayView<const ElementPair> interactions,
                           const typename MethodType::Parameters& parameters, const ContactStateViewT<Scalar>& state,
                           ContactOutputViewT<Scalar> output )
{
  for ( const ElementPair pair : interactions ) {
    const auto energy = variational_detail::pairEnergy<MethodType>( surfaces, pair, parameters, state, output );
    variational_detail::scatterGradient( surfaces, pair, energy, output.residual );
  }
}

template <SupportedMethod MethodType>
void finalizeVariationalTimestep( const typename MethodType::Parameters& parameters, EvaluationSummary& summary )
{
  using Enforcement = typename MethodType::enforcement_policy;
  if constexpr ( detail::is_penalty_v<Enforcement> ) {
    const Real stiffness = parameters.enforcement.stiffness.value;
    if ( stiffness > 0.0 ) {
      summary.timestep_vote = 1.0 / std::sqrt( stiffness );
    }
  }
}

template <SupportedMethod MethodType, typename Scalar>
EvaluationSummary addVariationalResidual( const SurfacePairViewT<Scalar>& surfaces,
                                          ArrayView<const ElementPair> interactions,
                                          const typename MethodType::Parameters& parameters,
                                          const ContactStateViewT<Scalar>& state, ContactOutputViewT<Scalar> output )
{
  validateVariationalEvaluation<MethodType>( surfaces, state, output );
  auto summary = stageVariationalKinematics<MethodType>( surfaces, interactions, parameters, state, output );
  finalizeVariationalKinematics<MethodType>( surfaces, parameters, state, output, summary );
  addVariationalForces<MethodType>( surfaces, interactions, parameters, state, output );
  finalizeVariationalTimestep<MethodType>( parameters, summary );
  return summary;
}

#endif
