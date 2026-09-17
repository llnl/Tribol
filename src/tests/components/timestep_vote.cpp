#include "tribol/Tribol.hpp"

#include <array>
#include <cmath>

// Requirements: TIME-001

namespace {

using namespace tribol;

SurfaceMeshView makeSegment( const std::array<Real, 4>& coordinates, const std::array<Index, 2>& connectivity )
{
  static constexpr std::array<Index, 2> offsets{ 0, 2 };
  static constexpr std::array<ElementTopology, 1> topologies{ ElementTopology::Segment };
  static constexpr std::array<int, 1> attributes{ 1 };
  return {
      .dimension = 2,
      .coordinates = { { coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .element_offsets = { offsets.data(), 2 },
      .connectivity = { connectivity.data(), 2 },
      .topologies = { topologies.data(), 1 },
      .attributes = { attributes.data(), 1 },
  };
}

Real vote( Real penetration, Real mortar_speed, Real nonmortar_speed )
{
  constexpr std::array<Index, 2> mortar_connectivity{ 1, 0 };
  constexpr std::array<Index, 2> nonmortar_connectivity{ 0, 1 };
  const std::array<Real, 4> mortar_coordinates{ 0.0, penetration, 1.0, penetration };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  const SurfacePairView surfaces{ makeSegment( mortar_coordinates, mortar_connectivity ),
                                  makeSegment( nonmortar_coordinates, nonmortar_connectivity ) };
  Contact<>::Options options;
  options.search.expansion = 0.25;
  options.timestep = { .enabled = true, .current_step = 1.0, .penetration_fraction = 0.3 };
  Contact<> contact( surfaces, options );
  contact.updateInteractions();
  const std::array<Real, 4> mortar_velocity{ 0.0, mortar_speed, 0.0, mortar_speed };
  const std::array<Real, 4> nonmortar_velocity{ 0.0, nonmortar_speed, 0.0, nonmortar_speed };
  constexpr std::array<Real, 1> thickness{ 0.1 };
  const ContactStateView state{
      .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .mortar_element_thickness = { thickness.data(), 1 },
      .nonmortar_element_thickness = { thickness.data(), 1 },
  };
  return contact.evaluate( state ).summary.timestep_vote;
}

bool legacyKinematicContract()
{
  const Real excessive_penetration = vote( 0.1, 0.3, -0.3 );
  const Real projected_penetration = vote( 0.01, 0.1, -0.1 );
  const Real separating = vote( 0.01, -0.1, 0.1 );
  return std::abs( excessive_penetration - 0.1 ) < 1.0e-10 && std::abs( projected_penetration - 0.3 ) < 1.0e-10 &&
         std::abs( separating - 1.0 ) < 1.0e-12;
}

bool disabledVoteIsUnbounded()
{
  constexpr std::array<Index, 2> connectivity{ 0, 1 };
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  Contact<> contact(
      { makeSegment( mortar_coordinates, connectivity ), makeSegment( nonmortar_coordinates, connectivity ) },
      { .search = { .expansion = 0.2 } } );
  contact.updateInteractions();
  return std::isinf( contact.evaluate().summary.timestep_vote );
}

}  // namespace

int main() { return legacyKinematicContract() && disabledVoteIsUnbounded() ? 0 : 1; }
