#include "tribol/Tribol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

// Requirements: DIFF-007

namespace {

using namespace tribol;

template <typename Stiffness = stiffness::Constant, typename Rate = rate::None,
          typename Response = response::Frictionless>
using PointwiseMethod =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::Penalty<Stiffness, Rate>, Response, formulation::PointwiseTraction, linearization::Exact>;

using MaterialRateViscous = PointwiseMethod<stiffness::Material, rate::Percentage, response::ViscousTangential>;
using TiedFull = PointwiseMethod<stiffness::Constant, rate::None, response::TiedFull>;

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
  static constexpr std::array<Real, 4> nonmortar{ 0.1, -0.12, 0.9, -0.08 };
  return { makeSegment( mortar ), makeSegment( nonmortar ) };
}

struct Residual {
  std::array<Real, 4> mortar{};
  std::array<Real, 4> nonmortar{};
};

template <SupportedMethod MethodType>
Residual evaluate( Contact<MethodType>& contact, const ContactStateView& state )
{
  Residual residual;
  contact.addResidual( state, { .mortar = { { residual.mortar.data(), 4 }, 2, 2, FieldLayout::Interleaved },
                                .nonmortar = { { residual.nonmortar.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );
  return residual;
}

template <SupportedMethod MethodType>
Residual exactAction( Contact<MethodType>& contact, const ContactStateView& state,
                      const ContactStateDirectionView& direction )
{
  Residual derivative;
  contact.applyDerivative( state, { .state = direction },
                           { .mortar = { { derivative.mortar.data(), 4 }, 2, 2, FieldLayout::Interleaved },
                             .nonmortar = { { derivative.nonmortar.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );
  return derivative;
}

bool close( const Residual& exact, const Residual& plus, const Residual& minus, const char* field )
{
  constexpr Real step = 1.0e-6;
  constexpr Real absolute_tolerance = 2.0e-8;
  constexpr Real relative_tolerance = 2.0e-6;
  for ( std::size_t entry = 0; entry < exact.mortar.size(); ++entry ) {
    const Real mortar_fd = ( plus.mortar[entry] - minus.mortar[entry] ) / ( 2.0 * step );
    const Real nonmortar_fd = ( plus.nonmortar[entry] - minus.nonmortar[entry] ) / ( 2.0 * step );
    const auto matches = [&]( Real value, Real finite_difference, const char* side ) {
      const Real scale = std::max( std::abs( value ), std::abs( finite_difference ) );
      if ( !std::isfinite( value ) || !std::isfinite( finite_difference ) ||
           std::abs( value - finite_difference ) > absolute_tolerance + relative_tolerance * scale ) {
        std::cerr << field << ' ' << side << " entry=" << entry << " exact=" << value
                  << " finite_difference=" << finite_difference << '\n';
        return false;
      }
      return true;
    };
    if ( !matches( exact.mortar[entry], mortar_fd, "mortar" ) ||
         !matches( exact.nonmortar[entry], nonmortar_fd, "nonmortar" ) ) {
      return false;
    }
  }
  return true;
}

template <std::size_t Size>
std::array<Real, Size> perturb( const std::array<Real, Size>& values, const std::array<Real, Size>& direction,
                                Real scale )
{
  std::array<Real, Size> result{};
  for ( std::size_t entry = 0; entry < Size; ++entry ) {
    result[entry] = values[entry] + scale * direction[entry];
  }
  return result;
}

Contact<MaterialRateViscous> makeMaterialContact()
{
  Contact<MaterialRateViscous>::Options options;
  options.search.expansion = 0.3;
  options.method.enforcement.stiffness.scale = 1.7;
  options.method.enforcement.rate.ratio = 0.35;
  options.method.response.damping = 0.6;
  return Contact<MaterialRateViscous>( surfaces(), options );
}

ContactStateView materialState( const std::array<Real, 4>& mortar_velocity,
                                const std::array<Real, 4>& nonmortar_velocity,
                                const std::array<Real, 1>& mortar_thickness,
                                const std::array<Real, 1>& nonmortar_thickness,
                                const std::array<Real, 1>& mortar_modulus,
                                const std::array<Real, 1>& nonmortar_modulus )
{
  return {
      .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .mortar_element_thickness = { mortar_thickness.data(), 1 },
      .nonmortar_element_thickness = { nonmortar_thickness.data(), 1 },
      .mortar_material_modulus = { mortar_modulus.data(), 1 },
      .nonmortar_material_modulus = { nonmortar_modulus.data(), 1 },
  };
}

bool velocityDerivativeMatchesFiniteDifference()
{
  auto contact = makeMaterialContact();
  contact.updateInteractions();
  constexpr std::array<Real, 4> mortar_velocity{ 0.4, 0.3, 0.1, 0.2 };
  constexpr std::array<Real, 4> nonmortar_velocity{ -0.2, -0.1, 0.0, -0.15 };
  constexpr std::array<Real, 4> mortar_direction{ 0.3, -0.2, -0.1, 0.4 };
  constexpr std::array<Real, 4> nonmortar_direction{ -0.2, 0.1, 0.25, -0.3 };
  constexpr std::array<Real, 1> mortar_thickness{ 0.4 };
  constexpr std::array<Real, 1> nonmortar_thickness{ 0.25 };
  constexpr std::array<Real, 1> mortar_modulus{ 3.0 };
  constexpr std::array<Real, 1> nonmortar_modulus{ 5.0 };
  const auto state = materialState( mortar_velocity, nonmortar_velocity, mortar_thickness, nonmortar_thickness,
                                    mortar_modulus, nonmortar_modulus );
  const auto exact =
      exactAction( contact, state,
                   { .mortar_velocity = { { mortar_direction.data(), 4 }, 2, 2, FieldLayout::Interleaved },
                     .nonmortar_velocity = { { nonmortar_direction.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );
  constexpr Real step = 1.0e-6;
  const auto plus_mortar = perturb( mortar_velocity, mortar_direction, step );
  const auto plus_nonmortar = perturb( nonmortar_velocity, nonmortar_direction, step );
  const auto minus_mortar = perturb( mortar_velocity, mortar_direction, -step );
  const auto minus_nonmortar = perturb( nonmortar_velocity, nonmortar_direction, -step );
  return close( exact,
                evaluate( contact, materialState( plus_mortar, plus_nonmortar, mortar_thickness, nonmortar_thickness,
                                                  mortar_modulus, nonmortar_modulus ) ),
                evaluate( contact, materialState( minus_mortar, minus_nonmortar, mortar_thickness, nonmortar_thickness,
                                                  mortar_modulus, nonmortar_modulus ) ),
                "velocity" );
}

bool materialDerivativeMatchesFiniteDifference( bool perturb_thickness )
{
  auto contact = makeMaterialContact();
  contact.updateInteractions();
  constexpr std::array<Real, 4> mortar_velocity{ 0.4, 0.3, 0.1, 0.2 };
  constexpr std::array<Real, 4> nonmortar_velocity{ -0.2, -0.1, 0.0, -0.15 };
  constexpr std::array<Real, 1> mortar_thickness{ 0.4 };
  constexpr std::array<Real, 1> nonmortar_thickness{ 0.25 };
  constexpr std::array<Real, 1> mortar_modulus{ 3.0 };
  constexpr std::array<Real, 1> nonmortar_modulus{ 5.0 };
  constexpr std::array<Real, 1> mortar_direction{ 0.17 };
  constexpr std::array<Real, 1> nonmortar_direction{ -0.11 };
  const auto state = materialState( mortar_velocity, nonmortar_velocity, mortar_thickness, nonmortar_thickness,
                                    mortar_modulus, nonmortar_modulus );
  ContactStateDirectionView direction;
  if ( perturb_thickness ) {
    direction.mortar_element_thickness = { mortar_direction.data(), 1 };
    direction.nonmortar_element_thickness = { nonmortar_direction.data(), 1 };
  } else {
    direction.mortar_material_modulus = { mortar_direction.data(), 1 };
    direction.nonmortar_material_modulus = { nonmortar_direction.data(), 1 };
  }
  const auto exact = exactAction( contact, state, direction );
  constexpr Real step = 1.0e-6;
  const auto plus_mortar = perturb( perturb_thickness ? mortar_thickness : mortar_modulus, mortar_direction, step );
  const auto plus_nonmortar =
      perturb( perturb_thickness ? nonmortar_thickness : nonmortar_modulus, nonmortar_direction, step );
  const auto minus_mortar = perturb( perturb_thickness ? mortar_thickness : mortar_modulus, mortar_direction, -step );
  const auto minus_nonmortar =
      perturb( perturb_thickness ? nonmortar_thickness : nonmortar_modulus, nonmortar_direction, -step );
  const auto plus_state = perturb_thickness ? materialState( mortar_velocity, nonmortar_velocity, plus_mortar,
                                                             plus_nonmortar, mortar_modulus, nonmortar_modulus )
                                            : materialState( mortar_velocity, nonmortar_velocity, mortar_thickness,
                                                             nonmortar_thickness, plus_mortar, plus_nonmortar );
  const auto minus_state = perturb_thickness ? materialState( mortar_velocity, nonmortar_velocity, minus_mortar,
                                                              minus_nonmortar, mortar_modulus, nonmortar_modulus )
                                             : materialState( mortar_velocity, nonmortar_velocity, mortar_thickness,
                                                              nonmortar_thickness, minus_mortar, minus_nonmortar );
  return close( exact, evaluate( contact, plus_state ), evaluate( contact, minus_state ),
                perturb_thickness ? "thickness" : "modulus" );
}

bool referenceDerivativeMatchesFiniteDifference()
{
  Contact<TiedFull>::Options options;
  options.search.expansion = 0.3;
  options.method.enforcement.stiffness.value = 2.5;
  Contact<TiedFull> contact( surfaces(), options );
  contact.updateInteractions();
  constexpr std::array<Real, 4> mortar_reference{ 0.0, 0.02, 1.0, -0.01 };
  constexpr std::array<Real, 4> nonmortar_reference{ 0.08, -0.06, 0.92, -0.04 };
  constexpr std::array<Real, 4> mortar_direction{ 0.2, -0.1, -0.3, 0.15 };
  constexpr std::array<Real, 4> nonmortar_direction{ -0.1, 0.25, 0.2, -0.2 };
  const auto state = ContactStateView{
      .mortar_reference_coordinates = { { mortar_reference.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar_reference_coordinates = { { nonmortar_reference.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  const auto exact = exactAction(
      contact, state,
      { .mortar_reference_coordinates = { { mortar_direction.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar_reference_coordinates = { { nonmortar_direction.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );
  constexpr Real step = 1.0e-6;
  const auto plus_mortar = perturb( mortar_reference, mortar_direction, step );
  const auto plus_nonmortar = perturb( nonmortar_reference, nonmortar_direction, step );
  const auto minus_mortar = perturb( mortar_reference, mortar_direction, -step );
  const auto minus_nonmortar = perturb( nonmortar_reference, nonmortar_direction, -step );
  return close(
      exact,
      evaluate(
          contact,
          ContactStateView{
              .mortar_reference_coordinates = { { plus_mortar.data(), 4 }, 2, 2, FieldLayout::Interleaved },
              .nonmortar_reference_coordinates = { { plus_nonmortar.data(), 4 }, 2, 2, FieldLayout::Interleaved } } ),
      evaluate(
          contact,
          ContactStateView{
              .mortar_reference_coordinates = { { minus_mortar.data(), 4 }, 2, 2, FieldLayout::Interleaved },
              .nonmortar_reference_coordinates = { { minus_nonmortar.data(), 4 }, 2, 2, FieldLayout::Interleaved } } ),
      "reference" );
}

}  // namespace

int main()
{
  return velocityDerivativeMatchesFiniteDifference() && materialDerivativeMatchesFiniteDifference( true ) &&
                 materialDerivativeMatchesFiniteDifference( false ) && referenceDerivativeMatchesFiniteDifference()
             ? 0
             : 1;
}
