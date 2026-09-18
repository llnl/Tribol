#include "tribol/Tribol.hpp"

#include <array>
#include <cmath>

// Requirements: CORE-002, GEOM-001, PHYS-001, PHYS-003

namespace {

using namespace tribol;

SurfaceMeshView makeSurface( const std::array<Real, 12>& coordinates, const std::array<Index, 4>& connectivity )
{
  static constexpr std::array<Index, 2> offsets{ 0, 4 };
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

bool projectedQuadrilateralContract()
{
  constexpr std::array<Real, 12> mortar_coordinates{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 12> nonmortar_coordinates{ 0.0, 0.0, -0.1, 0.0, 1.0, -0.1,
                                                        1.0, 1.0, -0.1, 1.0, 0.0, -0.1 };
  constexpr std::array<Index, 4> connectivity{ 0, 1, 2, 3 };
  const SurfacePairView surfaces{ makeSurface( mortar_coordinates, connectivity ),
                                  makeSurface( nonmortar_coordinates, connectivity ) };
  geometry::ProjectedOverlap<normal::MeanPlane>::Parameters parameters;
  const auto overlap = projectedOverlap<normal::MeanPlane>( surfaces, { 0, 0 }, parameters );
  const Real tolerance = 1.0e-12;
  return overlap.valid && std::abs( overlap.measure - 1.0 ) < tolerance && std::abs( overlap.gap + 0.1 ) < tolerance &&
         std::abs( overlap.normal[2] + 1.0 ) < tolerance;
}

bool overlapPatchContract()
{
  constexpr std::array<Real, 12> mortar_coordinates{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 12> nonmortar_coordinates{ 0.25, 0.0, -0.1, 0.25, 1.0, -0.1,
                                                        1.25, 1.0, -0.1, 1.25, 0.0, -0.1 };
  constexpr std::array<Index, 4> connectivity{ 0, 1, 2, 3 };
  const SurfacePairView surfaces{ makeSurface( mortar_coordinates, connectivity ),
                                  makeSurface( nonmortar_coordinates, connectivity ) };
  const auto patch = projectedOverlapPatch<normal::MortarSurface>(
      surfaces, { 0, 0 }, geometry::ProjectedOverlap<normal::MortarSurface>::Parameters{} );
  geometry::ProjectedOverlap<normal::MortarSurface>::Parameters rejected_parameters;
  rejected_parameters.minimum_overlap_fraction = 0.8;
  const auto rejected = projectedOverlapPatch<normal::MortarSurface>( surfaces, { 0, 0 }, rejected_parameters );
  const auto quadrature = integration::interactionQuadrature( patch, integration::Polygon<2>{} );
  const auto higher_quadrature = integration::interactionQuadrature( patch, integration::Polygon<4>{} );
  Real measure{};
  Real weighted_gap{};
  for ( int point = 0; point < quadrature.size; ++point ) {
    measure += quadrature[point].weight;
    for ( int component = 0; component < 3; ++component ) {
      weighted_gap +=
          quadrature[point].weight *
          ( quadrature[point].mortar_position[component] - quadrature[point].nonmortar_position[component] ) *
          quadrature[point].normal[component];
    }
  }
  return patch.valid && !rejected.valid && patch.vertex_count == 4 && quadrature.size == 6 &&
         higher_quadrature.size == 12 && std::abs( measure - 0.75 ) < 1.0e-12 &&
         std::abs( weighted_gap + 0.075 ) < 1.0e-12;
}

bool conformingPatchContract()
{
  constexpr std::array<Real, 12> mortar_coordinates{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 12> nonmortar_coordinates{ 1.0, 1.0, -0.1, 0.0, 1.0, -0.1,
                                                        0.0, 0.0, -0.1, 1.0, 0.0, -0.1 };
  constexpr std::array<Index, 4> connectivity{ 0, 1, 2, 3 };
  const SurfacePairView surfaces{ makeSurface( mortar_coordinates, connectivity ),
                                  makeSurface( nonmortar_coordinates, connectivity ) };
  const auto patch = conformingOverlapPatch( surfaces, { 0, 0 }, {} );
  const auto geometry = conformingOverlap( surfaces, { 0, 0 }, {} );
  return patch.valid && patch.vertex_count == 4 && std::abs( patch.measure - 1.0 ) < 1.0e-12 && geometry.valid &&
         std::abs( geometry.gap + 0.1 ) < 1.0e-12;
}

bool pointwiseForceContract()
{
  constexpr std::array<Real, 12> mortar_coordinates{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 12> nonmortar_coordinates{ 0.0, 0.0, -0.1, 0.0, 1.0, -0.1,
                                                        1.0, 1.0, -0.1, 1.0, 0.0, -0.1 };
  constexpr std::array<Index, 4> connectivity{ 0, 1, 2, 3 };
  const SurfacePairView surfaces{ makeSurface( mortar_coordinates, connectivity ),
                                  makeSurface( nonmortar_coordinates, connectivity ) };
  Contact<>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 10.0;
  Contact<> contact( surfaces, options );
  contact.updateInteractions();

  std::array<Real, 12> mortar_residual{};
  std::array<Real, 12> nonmortar_residual{};
  const ContactResidualView residual{
      .mortar = FieldView<Real>{ ArrayView<Real>{ mortar_residual.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .nonmortar = FieldView<Real>{ ArrayView<Real>{ nonmortar_residual.data(), 12 }, 4, 3, FieldLayout::Interleaved },
  };
  const auto summary = contact.addResidual( {}, residual );
  std::array<Real, 3> total{};
  for ( int node = 0; node < 4; ++node ) {
    for ( int component = 0; component < 3; ++component ) {
      total[component] += mortar_residual[3 * node + component] + nonmortar_residual[3 * node + component];
    }
  }
  const Real tolerance = 1.0e-12;
  return summary.active_interactions == 1 && std::abs( summary.energy - 0.025 ) < tolerance &&
         std::abs( mortar_residual[2] - 0.125 ) < tolerance && std::abs( nonmortar_residual[2] + 0.125 ) < tolerance &&
         std::abs( total[0] ) < tolerance && std::abs( total[1] ) < tolerance && std::abs( total[2] ) < tolerance;
}

}  // namespace

int main()
{
  return projectedQuadrilateralContract() && overlapPatchContract() && conformingPatchContract() &&
                 pointwiseForceContract()
             ? 0
             : 1;
}
