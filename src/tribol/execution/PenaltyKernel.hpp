#ifndef TRIBOL_EXECUTION_PENALTYKERNEL_HPP_
#define TRIBOL_EXECUTION_PENALTYKERNEL_HPP_

#include "tribol/core/ArrayView.hpp"
#include "tribol/enforcement/PenaltyLaw.hpp"
#include "tribol/execution/Execution.hpp"
#include "tribol/integration/InteractionQuadrature.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <type_traits>

namespace tribol::execution {

struct PenaltyContribution {
  std::array<Real, 3> mortar_force{};
  Real effective_gap{};
  Real pressure{};
  Real energy{};
  bool active{};

  friend constexpr bool operator==( const PenaltyContribution&, const PenaltyContribution& ) = default;
};

[[nodiscard]] TRIBOL_HOST_DEVICE inline PenaltyContribution evaluatePointwisePenaltyPatch(
    const InteractionPatch& patch, Real stiffness, constraint::GapActivationParameters activation = {} )
{
  PenaltyContribution result;
  if ( !patch.valid ) {
    return result;
  }
  Real gap{};
  for ( int component = 0; component < 3; ++component ) {
    gap += ( patch.mortar_centroid[component] - patch.nonmortar_centroid[component] ) * patch.normal[component];
  }
  const auto response = enforcement::evaluateNormalPenalty(
      enforcement::NormalPenaltyInput{ .gap = gap, .pressure_stiffness = stiffness, .activation = activation } );
  result.effective_gap = response.effective_gap;
  result.pressure = response.pressure;
  for ( int component = 0; component < 3; ++component ) {
    result.mortar_force[component] = patch.measure * result.pressure * patch.normal[component];
  }
  result.energy = patch.measure * response.potential_density;
  result.active = true;
  return result;
}

template <typename Integration = integration::Polygon<2>>
[[nodiscard]] TRIBOL_HOST_DEVICE inline PenaltyContribution evaluatePenaltyPatch( const InteractionPatch& patch,
                                                                                  Real stiffness,
                                                                                  Integration rule = {} )
{
  PenaltyContribution result;
  result.active = patch.valid;
  const auto quadrature = integration::interactionQuadrature( patch, rule );
  for ( int point_index = 0; point_index < quadrature.size; ++point_index ) {
    const auto& point = quadrature[point_index];
    Real gap{};
    for ( int component = 0; component < 3; ++component ) {
      gap += ( point.mortar_position[component] - point.nonmortar_position[component] ) * point.normal[component];
    }
    const auto response = enforcement::evaluateNormalPenalty(
        enforcement::NormalPenaltyInput{ .gap = gap, .pressure_stiffness = stiffness } );
    for ( int component = 0; component < 3; ++component ) {
      result.mortar_force[component] += point.weight * response.pressure * point.normal[component];
    }
    result.energy += point.weight * response.potential_density;
  }
  return result;
}

template <typename Integration = integration::Polygon<2>, Policy Execution>
  requires( std::same_as<Execution, Sequential> || std::same_as<Execution, Deterministic> )
inline void evaluatePenaltyPatches( ArrayView<const InteractionPatch> patches, Real stiffness,
                                    ArrayView<PenaltyContribution> contributions, Execution, Integration rule = {} )
{
  if ( stiffness < 0.0 || contributions.size() != patches.size() || !patches.isStructurallyValid() ||
       !contributions.isStructurallyValid() ) {
    throw std::invalid_argument( "Penalty patch execution requires nonnegative stiffness and matching valid views." );
  }
  for ( Index patch = 0; patch < patches.size(); ++patch ) {
    contributions[patch] = evaluatePenaltyPatch( patches[patch], stiffness, rule );
  }
}

static_assert( std::is_trivially_copyable_v<PenaltyContribution> );

}  // namespace tribol::execution

#endif
