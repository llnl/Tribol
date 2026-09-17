#include "tribol/Tribol.hpp"

#include <array>

// Requirements: SEARCH-001, SEARCH-002, SEARCH-003, SEARCH-004

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

bool suppliedPairContract()
{
  const SurfaceStorage mortar{
      .coordinates = { 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0 } };
  const SurfaceStorage nonmortar{
      .coordinates = { 0.5, 0.05, 1.5, 0.05, 2.5, 0.05, 3.5, 0.05, 8.0, 0.05, 9.0, 0.05, 6.5, 0.05, 7.5, 0.05 } };
  const SurfacePairView surfaces{ mortar.view(), nonmortar.view() };
  Contact<DefaultMethod, search::Supplied>::Options options;
  options.search.pairs = { { 2, 0 }, { 0, 2 }, { 2, 0 } };
  Contact<DefaultMethod, search::Supplied> contact( surfaces, options );
  contact.updateInteractions();
  if ( contact.interactions().size() != 2 || contact.interactions()[0] != ElementPair{ 0, 2 } ||
       contact.interactions()[1] != ElementPair{ 2, 0 } ) {
    return false;
  }
  constexpr std::array<ElementPair, 2> replacement{ ElementPair{ 1, 1 }, ElementPair{ 3, 2 } };
  contact.setInteractions( { replacement.data(), 2 } );
  if ( contact.interactions().size() != 2 || contact.interactions()[0] != replacement[0] ||
       contact.interactions()[1] != replacement[1] ) {
    return false;
  }
  try {
    constexpr std::array<ElementPair, 1> invalid{ ElementPair{ 4, 0 } };
    contact.setInteractions( { invalid.data(), 1 } );
  } catch ( const std::out_of_range& ) {
    return true;
  }
  return false;
}

bool relativeProximityMatchesLegacyFormula()
{
  struct SegmentStorage {
    std::array<Real, 4> coordinates;
    std::array<Index, 2> offsets{ 0, 2 };
    std::array<Index, 2> connectivity{ 0, 1 };
    std::array<ElementTopology, 1> topologies{ ElementTopology::Segment };
    std::array<int, 1> attributes{ 1 };

    SurfaceMeshView view() const
    {
      return { .dimension = 2,
               .coordinates = { { coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
               .element_offsets = { offsets.data(), 2 },
               .connectivity = { connectivity.data(), 2 },
               .topologies = { topologies.data(), 1 },
               .attributes = { attributes.data(), 1 } };
    }
  };

  const SegmentStorage mortar{ { 0.0, 0.0, 1.0, 0.0 } };
  const SegmentStorage touching{ { 0.0, 8.0, 1.0, 8.0 } };
  const SegmentStorage outside{ { 0.0, 8.01, 1.0, 8.01 } };
  auto has_one_pair = [&]( const SurfaceMeshView& nonmortar, const auto& policy ) {
    CandidatePairs candidates;
    policy.findCandidates( { mortar.view(), nonmortar }, candidates );
    return candidates.size() == 1;
  };
  auto has_no_pairs = [&]( const SurfaceMeshView& nonmortar, const auto& policy ) {
    CandidatePairs candidates;
    policy.findCandidates( { mortar.view(), nonmortar }, candidates );
    return candidates.empty();
  };

  const search::CartesianProduct cartesian( { .proximity_scale = search::legacyProximityScale } );
  const search::Grid grid( { .proximity_scale = search::legacyProximityScale, .cell_size = 1.0 } );
  const search::Bvh bvh( { .proximity_scale = search::legacyProximityScale, .leaf_size = 1 } );
  return has_one_pair( touching.view(), cartesian ) && has_one_pair( touching.view(), grid ) &&
         has_one_pair( touching.view(), bvh ) && has_no_pairs( outside.view(), cartesian ) &&
         has_no_pairs( outside.view(), grid ) && has_no_pairs( outside.view(), bvh );
}

}  // namespace

int main()
{
  return equivalentBroadPhases() && selfContactFiltering() && contactSelfSearch() && suppliedPairContract() &&
                 relativeProximityMatchesLegacyFormula()
             ? 0
             : 1;
}
