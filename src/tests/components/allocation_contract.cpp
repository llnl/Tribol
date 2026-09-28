#include "tribol/Tribol.hpp"
#include "tests/AllocationCounter.hpp"

#include <array>

// Requirements: CORE-003, DIFF-001

int main()
{
  using namespace tribol;
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  constexpr std::array<Index, 2> offsets{ 0, 2 };
  constexpr std::array<Index, 2> connectivity{ 0, 1 };
  constexpr std::array<ElementTopology, 1> topologies{ ElementTopology::Segment };
  constexpr std::array<int, 1> attributes{ 1 };
  const auto make_surface = [&]( const std::array<Real, 4>& coordinates ) {
    return SurfaceMeshView{
        .dimension = 2,
        .coordinates = { { coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .element_offsets = { offsets.data(), 2 },
        .connectivity = { connectivity.data(), 2 },
        .topologies = { topologies.data(), 1 },
        .attributes = { attributes.data(), 1 },
    };
  };
  Contact<>::Options options;
  options.search.expansion = 0.2;
  Contact<> contact( { make_surface( mortar_coordinates ), make_surface( nonmortar_coordinates ) }, options );
  contact.updateInteractions();

  std::array<Real, 4> mortar_direction{};
  std::array<Real, 4> nonmortar_direction{ 0.0, 1.0, 0.0, 1.0 };
  std::array<Real, 4> mortar_derivative{};
  std::array<Real, 4> nonmortar_derivative{};
  const ContactDirectionView direction{
      .mortar = { { mortar_direction.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_direction.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  const ContactResidualView derivative{
      .mortar = { { mortar_derivative.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_derivative.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };

  tribol::test::resetAllocationCount();
  tribol::test::startAllocationCounting();
  const auto result = contact.evaluate();
  contact.applyCoordinateDerivative( {}, direction, derivative );
  const auto jacobian = contact.assembleCoordinateJacobian( {} );
  tribol::test::stopAllocationCounting();

  return tribol::test::allocationCount() == 0 && result.summary.active_interactions == 1 && jacobian.rows == 8 &&
                 jacobian.columns == 8
             ? 0
             : 1;
}
