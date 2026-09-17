#include "tribol/Tribol.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

// Requirements: SEARCH-004, LEGACY-003

namespace {

using namespace tribol;

struct SelfContactQuads {
  std::array<Real, 24> coordinates{};
  std::array<Index, 3> offsets{ 0, 4, 8 };
  std::array<Index, 8> connectivity{ 0, 1, 2, 3, 4, 7, 6, 5 };
  std::array<ElementTopology, 2> topologies{ ElementTopology::Quadrilateral, ElementTopology::Quadrilateral };
  std::array<int, 2> attributes{ 1, 1 };

  explicit SelfContactQuads( Real gap )
      : coordinates{ 0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 1.0, 0.0,  0.0, 1.0, 0.0,
                     0.0, 0.0, -gap, 1.0, 0.0, -gap, 1.0, 1.0, -gap, 0.0, 1.0, -gap }
  {
  }

  SurfaceMeshView view() const
  {
    return { .dimension = 3,
             .coordinates = { { coordinates.data(), 24 }, 8, 3, FieldLayout::Interleaved },
             .element_offsets = { offsets.data(), 3 },
             .connectivity = { connectivity.data(), 8 },
             .topologies = { topologies.data(), 2 },
             .attributes = { attributes.data(), 2 } };
  }
};

EvaluationSummary evaluate( Real gap, Real residual_gap = 0.0 )
{
  const SelfContactQuads surface( gap );
  Contact<DefaultMethod, search::Supplied>::Options options;
  options.search.pairs = { { 0, 1 } };
  options.method.constraint.activation.residual_gap = residual_gap;
  Contact<DefaultMethod, search::Supplied> contact( surface.view(), options );
  contact.updateInteractions();
  constexpr std::array<Real, 2> thickness{ 1.0, 1.0 };
  return contact
      .evaluate( { .mortar_element_thickness = { thickness.data(), 2 },
                   .nonmortar_element_thickness = { thickness.data(), 2 } } )
      .summary;
}

bool legacyMaximumPenetrationRule()
{
  const auto shallow = evaluate( 0.90 ).active_interactions;
  const auto deep = evaluate( 0.951 ).active_interactions;
  const auto shifted = evaluate( 0.90, 0.06 ).active_interactions;
  if ( shallow != 1 || deep != 0 || shifted != 0 ) {
    std::cerr << "unexpected active counts: shallow=" << shallow << ", deep=" << deep << ", shifted=" << shifted
              << '\n';
  }
  return shallow == 1 && deep == 0 && shifted == 0;
}

bool thicknessIsRequired()
{
  const SelfContactQuads surface( 0.5 );
  Contact<DefaultMethod, search::Supplied>::Options options;
  options.search.pairs = { { 0, 1 } };
  Contact<DefaultMethod, search::Supplied> contact( surface.view(), options );
  contact.updateInteractions();
  try {
    (void)contact.evaluate();
  } catch ( const std::invalid_argument& ) {
    return true;
  }
  return false;
}

bool limitCanBeDisabled()
{
  const SelfContactQuads surface( 1.1 );
  Contact<DefaultMethod, search::Supplied>::Options options;
  options.search.pairs = { { 0, 1 } };
  options.reject_excessive_self_penetration = false;
  Contact<DefaultMethod, search::Supplied> contact( surface.view(), options );
  contact.updateInteractions();
  return contact.evaluate().summary.active_interactions == 1;
}

}  // namespace

int main()
{
  if ( !legacyMaximumPenetrationRule() ) {
    return 1;
  }
  if ( !thicknessIsRequired() ) {
    return 2;
  }
  return limitCanBeDisabled() ? 0 : 3;
}
