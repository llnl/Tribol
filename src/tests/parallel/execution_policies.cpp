#include "tribol/Tribol.hpp"

#include <algorithm>
#include <array>
#include <cmath>

// Requirements: PAR-003

namespace {

tribol::InteractionPatch makePatch( tribol::Real gap )
{
  tribol::InteractionPatch patch;
  patch.normal = { 0.0, 0.0, -1.0 };
  patch.integration_vertices[0] = { 0.0, 0.0, -0.5 * gap };
  patch.integration_vertices[1] = { 1.0, 0.0, -0.5 * gap };
  patch.integration_vertices[2] = { 1.0, 1.0, -0.5 * gap };
  patch.integration_vertices[3] = { 0.0, 1.0, -0.5 * gap };
  for ( int node = 0; node < 4; ++node ) {
    patch.mortar_vertices[node] = patch.integration_vertices[node];
    patch.nonmortar_vertices[node] = patch.integration_vertices[node];
    patch.mortar_vertices[node][2] = 0.0;
    patch.nonmortar_vertices[node][2] = -gap;
  }
  patch.mortar_centroid = { 0.5, 0.5, 0.0 };
  patch.nonmortar_centroid = { 0.5, 0.5, -gap };
  patch.measure = 1.0;
  patch.vertex_count = 4;
  patch.manifold_dimension = 2;
  patch.valid = true;
  return patch;
}

tribol::SurfaceMeshView makeSegment( const std::array<tribol::Real, 4>& coordinates )
{
  static constexpr std::array<tribol::Index, 2> offsets{ 0, 2 };
  static constexpr std::array<tribol::Index, 2> connectivity{ 0, 1 };
  static constexpr std::array<tribol::ElementTopology, 1> topologies{ tribol::ElementTopology::Segment };
  static constexpr std::array<int, 1> attributes{ 1 };
  return {
      .dimension = 2,
      .coordinates = { { coordinates.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .element_offsets = { offsets.data(), 2 },
      .connectivity = { connectivity.data(), 2 },
      .topologies = { topologies.data(), 1 },
      .attributes = { attributes.data(), 1 },
  };
}

bool fullContactParity()
{
  constexpr std::array<tribol::Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<tribol::Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  const tribol::SurfacePairView surfaces{ makeSegment( mortar_coordinates ), makeSegment( nonmortar_coordinates ) };
  tribol::Contact<>::Options sequential_options;
  sequential_options.search.expansion = 0.2;
  using DeterministicContact =
      tribol::Contact<tribol::DefaultMethod, tribol::search::CartesianProduct, tribol::execution::Deterministic>;
  DeterministicContact::Options deterministic_options;
  deterministic_options.search.expansion = sequential_options.search.expansion;
  tribol::Contact<> sequential_contact( surfaces, sequential_options );
  DeterministicContact deterministic_contact( surfaces, deterministic_options );
  sequential_contact.updateInteractions();
  deterministic_contact.updateInteractions();
  const auto sequential_result = sequential_contact.evaluate();
  const auto deterministic_result = deterministic_contact.evaluate();
  return sequential_result.summary.energy == deterministic_result.summary.energy &&
         std::equal( sequential_result.mortar_force.values.begin(), sequential_result.mortar_force.values.end(),
                     deterministic_result.mortar_force.values.begin() ) &&
         std::equal( sequential_result.nonmortar_force.values.begin(), sequential_result.nonmortar_force.values.end(),
                     deterministic_result.nonmortar_force.values.begin() );
}

}  // namespace

int main()
{
  std::array<tribol::InteractionPatch, 3> patches{ makePatch( 0.1 ), makePatch( 0.2 ), makePatch( -0.1 ) };
  std::array<tribol::execution::PenaltyContribution, 3> sequential{};
  std::array<tribol::execution::PenaltyContribution, 3> deterministic{};
  tribol::execution::evaluatePenaltyPatches( { patches.data(), 3 }, 10.0, { sequential.data(), 3 },
                                             tribol::execution::Sequential{} );
  tribol::execution::evaluatePenaltyPatches( { patches.data(), 3 }, 10.0, { deterministic.data(), 3 },
                                             tribol::execution::Deterministic{} );
  return fullContactParity() && sequential == deterministic && std::abs( sequential[0].energy - 0.05 ) < 1.0e-12 &&
                 std::abs( sequential[1].energy - 0.2 ) < 1.0e-12 && sequential[2].energy == 0.0
             ? 0
             : 1;
}
