#include "tribol/Tribol.hpp"

#include <array>
#include <cmath>

// Requirements: CONS-001, FORM-001

namespace {

using namespace tribol;

SurfaceMeshView makeQuadrilateral( const std::array<Real, 12>& coordinates )
{
  static constexpr std::array<Index, 2> offsets{ 0, 4 };
  static constexpr std::array<Index, 4> connectivity{ 0, 1, 2, 3 };
  static constexpr std::array<ElementTopology, 1> topologies{ ElementTopology::Quadrilateral };
  static constexpr std::array<int, 1> attributes{ 1 };
  return {
      .dimension = 3,
      .coordinates =
          FieldView<const Real>{ ArrayView<const Real>{ coordinates.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .element_offsets = ArrayView<const Index>{ offsets.data(), 2 },
      .connectivity = ArrayView<const Index>{ connectivity.data(), 4 },
      .topologies = ArrayView<const ElementTopology>{ topologies.data(), 1 },
      .attributes = ArrayView<const int>{ attributes.data(), 1 },
  };
}

bool quadratureAndBasisContract()
{
  constexpr std::array<Real, 12> coordinates{ 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  const auto surface = makeQuadrilateral( coordinates );
  const auto quadrature = integration::faceQuadrature( surface, 0, integration::Face<2>{} );
  Real area{};
  std::array<std::array<Real, 4>, 4> pairing{};
  for ( int point = 0; point < quadrature.size; ++point ) {
    area += quadrature[point].weight;
    const auto primal = basis::primalShape( ElementTopology::Quadrilateral, quadrature[point].reference );
    const auto dual = basis::dualShape( ElementTopology::Quadrilateral, quadrature[point].reference );
    Real partition{};
    for ( int node = 0; node < 4; ++node ) {
      partition += primal[node];
      for ( int trial = 0; trial < 4; ++trial ) {
        pairing[node][trial] += quadrature[point].weight * dual[node] * primal[trial];
      }
    }
    if ( std::abs( partition - 1.0 ) > 1.0e-12 ) {
      return false;
    }
  }
  if ( std::abs( area - 2.0 ) > 1.0e-12 ) {
    return false;
  }
  for ( int test = 0; test < 4; ++test ) {
    for ( int trial = 0; trial < 4; ++trial ) {
      const Real expected = test == trial ? 0.5 : 0.0;
      if ( std::abs( pairing[test][trial] - expected ) > 1.0e-12 ) {
        return false;
      }
    }
  }
  return true;
}

bool normalConstraintContract()
{
  constexpr std::array<Real, 12> mortar_coordinates{ 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 12> nonmortar_coordinates{ 0.0, 0.0, -0.2, 2.0, 0.0, -0.2,
                                                        2.0, 1.0, -0.2, 0.0, 1.0, -0.2 };
  const SurfacePairView surfaces{ makeQuadrilateral( mortar_coordinates ), makeQuadrilateral( nonmortar_coordinates ) };
  constexpr std::array<Real, 3> mortar_point{ 1.0, 0.5, 0.0 };
  constexpr std::array<Real, 3> nonmortar_point{ 1.0, 0.5, -0.2 };
  constexpr std::array<Real, 3> normal{ 0.0, 0.0, -1.0 };
  const auto sample = constraint::stageNormalConstraint<constraint::Nodal<basis::Dual>>(
      surfaces, { 0, 0 }, mortar_point, nonmortar_point, normal, 2.0 );
  std::array<Real, 4> weighted_gap{};
  constraint::addWeightedGap( sample, { weighted_gap.data(), 4 } );
  Real sum{};
  for ( Real value : weighted_gap ) {
    sum += value;
  }
  return std::abs( sample.gap + 0.2 ) < 1.0e-12 && std::abs( sum + 0.4 ) < 1.0e-12;
}

}  // namespace

int main() { return quadratureAndBasisContract() && normalConstraintContract() ? 0 : 1; }
