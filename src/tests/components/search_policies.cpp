#include "tribol/Tribol.hpp"

#include <array>

// Requirements: SEARCH-001, SEARCH-002

namespace {

using namespace tribol;

struct SurfaceStorage {
  std::array<Real, 16> coordinates{};
  std::array<Index, 5> offsets{ 0, 2, 4, 6, 8 };
  std::array<Index, 8> connectivity{ 0, 1, 2, 3, 4, 5, 6, 7 };
  std::array<ElementTopology, 4> topologies{ ElementTopology::Segment, ElementTopology::Segment,
                                             ElementTopology::Segment, ElementTopology::Segment };
  std::array<int, 4> attributes{ 1, 1, 1, 1 };

  SurfaceMeshView view() const
  {
    return {
        .dimension = 2,
        .coordinates =
            FieldView<const Real>{ ArrayView<const Real>{ coordinates.data(), 16 }, 8, 2, FieldLayout::Interleaved },
        .element_offsets = ArrayView<const Index>{ offsets.data(), 5 },
        .connectivity = ArrayView<const Index>{ connectivity.data(), 8 },
        .topologies = ArrayView<const ElementTopology>{ topologies.data(), 4 },
        .attributes = ArrayView<const int>{ attributes.data(), 4 },
    };
  }
};

bool equivalentBroadPhases()
{
  const SurfaceStorage mortar{
      .coordinates = { 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0 } };
  const SurfaceStorage nonmortar{
      .coordinates = { 0.5, 0.05, 1.5, 0.05, 2.5, 0.05, 3.5, 0.05, 8.0, 0.05, 9.0, 0.05, 6.5, 0.05, 7.5, 0.05 } };
  const SurfacePairView surfaces{ mortar.view(), nonmortar.view() };
  CandidatePairs cartesian;
  CandidatePairs grid;
  CandidatePairs bvh;
  search::CartesianProduct( { .expansion = 0.1 } ).findCandidates( surfaces, cartesian );
  search::Grid( { .expansion = 0.1, .cell_size = 0.75 } ).findCandidates( surfaces, grid );
  search::Bvh( { .expansion = 0.1, .leaf_size = 1 } ).findCandidates( surfaces, bvh );
  cartesian.canonicalize();
  if ( cartesian.size() != grid.size() || cartesian.size() != bvh.size() || cartesian.size() != 3 ) {
    return false;
  }
  for ( Index pair = 0; pair < cartesian.size(); ++pair ) {
    if ( !( cartesian.view()[pair] == grid.view()[pair] ) || !( cartesian.view()[pair] == bvh.view()[pair] ) ) {
      return false;
    }
  }
  return true;
}

bool selfContactFiltering()
{
  SurfaceStorage surface{
      .coordinates = { 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0, 1.5, 0.0, 5.0, 0.0, 6.0, 0.0 },
      .connectivity = { 0, 1, 1, 3, 4, 5, 6, 7 } };
  CandidatePairs filtered;
  search::SelfContact<search::Grid>( { .broad_phase = { .cell_size = 0.5 }, .exclude_adjacent = true } )
      .findCandidates( surface.view(), filtered );
  if ( filtered.size() != 2 ) {
    return false;
  }
  return filtered.view()[0] == ElementPair{ 0, 2 } && filtered.view()[1] == ElementPair{ 1, 2 };
}

bool contactSelfSearch()
{
  SurfaceStorage surface{
      .coordinates = { 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.5, -0.05, 1.5, -0.05, 5.0, 0.0, 6.0, 0.0 },
      .connectivity = { 0, 1, 1, 3, 4, 5, 6, 7 } };
  Contact<DefaultMethod, search::Bvh>::Options options;
  options.search.expansion = 0.1;
  options.search.leaf_size = 1;
  Contact<DefaultMethod, search::Bvh> contact( surface.view(), options );
  contact.updateInteractions();
  return contact.interactions().size() == 2;
}

}  // namespace

int main() { return equivalentBroadPhases() && selfContactFiltering() && contactSelfSearch() ? 0 : 1; }
