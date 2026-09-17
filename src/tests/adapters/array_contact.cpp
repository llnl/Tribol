#include "tribol/adapters/array/ArrayContact.hpp"

#include <array>
#include <cmath>

// Requirements: ADAPTER-001, API-001, PHYS-001

int main()
{
  constexpr std::array<tribol::Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<tribol::Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  constexpr std::array<tribol::Index, 2> connectivity{ 0, 1 };
  tribol::Contact<>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 10.0;
  auto contact = tribol::array::makeContact(
      tribol::array::UniformSurface(
          2, tribol::ElementTopology::Segment,
          { mortar_coordinates.data(), static_cast<tribol::Index>( mortar_coordinates.size() ) },
          { connectivity.data(), static_cast<tribol::Index>( connectivity.size() ) } ),
      tribol::array::UniformSurface(
          2, tribol::ElementTopology::Segment,
          { nonmortar_coordinates.data(), static_cast<tribol::Index>( nonmortar_coordinates.size() ) },
          { connectivity.data(), static_cast<tribol::Index>( connectivity.size() ) } ),
      options );
  contact.updateInteractions();
  const auto result = contact.evaluate();
  tribol::Real total{};
  for ( tribol::Index node = 0; node < result.mortar_force.entities; ++node ) {
    total += result.mortar_force( node, 1 ) + result.nonmortar_force( node, 1 );
  }
  return result.summary.active_interactions == 1 && std::abs( result.summary.energy - 0.025 ) < 1.0e-12 &&
                 std::abs( total ) < 1.0e-12
             ? 0
             : 1;
}
