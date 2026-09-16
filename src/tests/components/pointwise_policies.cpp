#include "tribol/Tribol.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

// Requirements: ENF-001, ENF-002, RESP-001, RESP-002, OUTPUT-001, PHYS-001

namespace {

using namespace tribol;

template <typename Stiffness = stiffness::Constant, typename Rate = rate::None,
          typename Response = response::Frictionless>
using PointwiseMethod =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::Penalty<Stiffness, Rate>, Response, formulation::PointwiseTraction, linearization::Exact>;

SurfaceMeshView makeSegment( const std::array<Real, 4>& coordinates )
{
  static constexpr std::array<Index, 2> offsets{ 0, 2 };
  static constexpr std::array<Index, 2> connectivity{ 0, 1 };
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

SurfacePairView surfaces()
{
  static constexpr std::array<Real, 4> mortar{ 0.0, 0.0, 1.0, 0.0 };
  static constexpr std::array<Real, 4> nonmortar{ 0.0, -0.1, 1.0, -0.1 };
  return { makeSegment( mortar ), makeSegment( nonmortar ) };
}

template <typename MethodType>
EvaluationSummary evaluate( typename Contact<MethodType>::Options options, const ContactStateView& state,
                            std::array<Real, 4>& mortar_force, std::array<Real, 4>& nonmortar_force )
{
  options.search.expansion = 0.2;
  Contact<MethodType> contact( surfaces(), options );
  contact.updateInteractions();
  return contact.addResidual( state,
                              { .mortar = { { mortar_force.data(), 4 }, 2, 2, FieldLayout::Interleaved },
                                .nonmortar = { { nonmortar_force.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );
}

bool stiffnessAndRateContract()
{
  using MethodType = PointwiseMethod<stiffness::Material, rate::Percentage>;
  Contact<MethodType>::Options options;
  options.method.enforcement.stiffness.scale = 2.0;
  options.method.enforcement.rate.ratio = 0.1;
  constexpr std::array<Real, 1> thickness{ 0.5 };
  constexpr std::array<Real, 1> modulus{ 5.0 };
  constexpr std::array<Real, 4> mortar_velocity{ 0.0, 1.0, 0.0, 1.0 };
  constexpr std::array<Real, 4> nonmortar_velocity{ 0.0, 0.0, 0.0, 0.0 };
  std::array<Real, 4> mortar_force{};
  std::array<Real, 4> nonmortar_force{};
  const auto summary = evaluate<MethodType>(
      options,
      { .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .mortar_element_thickness = { thickness.data(), 1 },
        .nonmortar_element_thickness = { thickness.data(), 1 },
        .mortar_material_modulus = { modulus.data(), 1 },
        .nonmortar_material_modulus = { modulus.data(), 1 } },
      mortar_force, nonmortar_force );
  return summary.active_interactions == 1 && std::abs( summary.timestep_vote - std::sqrt( 0.05 ) ) < 1.0e-12 &&
         std::abs( mortar_force[1] - 2.0 ) < 1.0e-12;
}

bool viscousAndTiedContract()
{
  using Viscous = PointwiseMethod<stiffness::Constant, rate::None, response::ViscousTangential>;
  Contact<Viscous>::Options viscous_options;
  viscous_options.method.response.damping = 4.0;
  constexpr std::array<Real, 4> mortar_velocity{ 1.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_velocity{};
  std::array<Real, 4> viscous_mortar{};
  std::array<Real, 4> viscous_nonmortar{};
  evaluate<Viscous>( viscous_options,
                     { .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
                       .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved } },
                     viscous_mortar, viscous_nonmortar );

  using Tied = PointwiseMethod<stiffness::Constant, rate::None, response::TiedFull>;
  Contact<Tied>::Options tied_options;
  tied_options.method.enforcement.stiffness.value = 10.0;
  constexpr std::array<Real, 4> mortar_reference{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_reference{ -0.2, 0.0, 0.8, 0.0 };
  std::array<Real, 4> tied_mortar{};
  std::array<Real, 4> tied_nonmortar{};
  evaluate<Tied>(
      tied_options,
      { .mortar_reference_coordinates = { { mortar_reference.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar_reference_coordinates = { { nonmortar_reference.data(), 4 }, 2, 2, FieldLayout::Interleaved } },
      tied_mortar, tied_nonmortar );
  return std::abs( viscous_mortar[0] - 2.0 ) < 1.0e-12 && std::abs( tied_mortar[0] + 1.0 ) < 1.0e-12 &&
         std::abs( tied_mortar[1] - 0.5 ) < 1.0e-12;
}

bool invalidParametersAreRejected()
{
  Contact<>::Options options;
  options.method.enforcement.stiffness.value = -1.0;
  try {
    Contact<> contact( surfaces(), options );
    (void)contact;
  } catch ( const std::invalid_argument& ) {
    return true;
  }
  return false;
}

}  // namespace

int main() { return stiffnessAndRateContract() && viscousAndTiedContract() && invalidParametersAreRejected() ? 0 : 1; }
