#include "tribol/Tribol.hpp"

#include <array>
#include <cmath>

// Requirements: LEGACY-001, ENF-001, ENF-002, RESP-001, RESP-002

namespace {

using namespace tribol;

template <typename Stiffness = stiffness::Constant, typename Rate = rate::None,
          typename Response = response::Frictionless>
using PointwiseMethod =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::Penalty<Stiffness, Rate>, Response, formulation::PointwiseTraction, linearization::Exact>;

struct SegmentPair {
  std::array<Real, 4> mortar{ 0.0, 0.0, 1.0, 0.0 };
  std::array<Real, 4> nonmortar{ 0.0, -0.1, 1.0, -0.1 };

  SurfaceMeshView surface( const std::array<Real, 4>& coordinates ) const
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

  SurfacePairView view() const { return { surface( mortar ), surface( nonmortar ) }; }
};

template <typename MethodType>
ContactResultView evaluate( const SegmentPair& pair, typename Contact<MethodType>::Options options,
                            const ContactStateView& state, Contact<MethodType>& contact )
{
  (void)pair;
  (void)options;
  contact.updateInteractions();
  return contact.evaluate( state );
}

Real sumComponent( FieldView<const Real> field, int component )
{
  Real result{};
  for ( Index node = 0; node < field.entities; ++node ) {
    result += field( node, component );
  }
  return result;
}

bool constantPenaltyMatchesLegacySeriesSpring()
{
  const SegmentPair pair;
  Contact<>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 0.75;
  Contact<> contact( pair.view(), options );
  contact.updateInteractions();
  const auto result = contact.evaluate();
  const Real expected_pressure = ( 0.75 * 0.75 / ( 0.75 + 0.75 ) ) * -0.1;
  return std::abs( sumComponent( result.mortar_force, 1 ) + expected_pressure ) < 1.0e-13 &&
         std::abs( sumComponent( result.nonmortar_force, 1 ) - expected_pressure ) < 1.0e-13;
}

bool materialPenaltyMatchesLegacySeriesSpring()
{
  using MethodType = PointwiseMethod<stiffness::Material>;
  const SegmentPair pair;
  Contact<MethodType>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.scale = 0.75;
  constexpr std::array<Real, 1> mortar_thickness{ 0.25 };
  constexpr std::array<Real, 1> nonmortar_thickness{ 0.2 };
  constexpr std::array<Real, 1> mortar_modulus{ 1.0 };
  constexpr std::array<Real, 1> nonmortar_modulus{ 2.0 };
  Contact<MethodType> contact( pair.view(), options );
  contact.updateInteractions();
  const auto result = contact.evaluate( { .mortar_element_thickness = { mortar_thickness.data(), 1 },
                                          .nonmortar_element_thickness = { nonmortar_thickness.data(), 1 },
                                          .mortar_material_modulus = { mortar_modulus.data(), 1 },
                                          .nonmortar_material_modulus = { nonmortar_modulus.data(), 1 } } );
  const Real mortar_stiffness = mortar_modulus[0] / mortar_thickness[0];
  const Real nonmortar_stiffness = nonmortar_modulus[0] / nonmortar_thickness[0];
  const Real equivalent = 0.75 * mortar_stiffness * nonmortar_stiffness / ( mortar_stiffness + nonmortar_stiffness );
  return std::abs( sumComponent( result.mortar_force, 1 ) - equivalent * 0.1 ) < 1.0e-13;
}

bool ratePenaltiesMatchLegacyPressure()
{
  const SegmentPair pair;
  constexpr std::array<Real, 4> mortar_velocity{ 0.0, 1.0, 0.0, 1.0 };
  constexpr std::array<Real, 4> nonmortar_velocity{ 0.0, -1.0, 0.0, -1.0 };
  const ContactStateView state{
      .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };

  using ConstantRate = PointwiseMethod<stiffness::Constant, rate::Constant>;
  Contact<ConstantRate>::Options constant_options;
  constant_options.search.expansion = 0.2;
  constant_options.method.enforcement.stiffness.value = 0.0;
  constant_options.method.enforcement.rate.value = 1.0;
  Contact<ConstantRate> constant_contact( pair.view(), constant_options );
  constant_contact.updateInteractions();
  const auto constant_result = constant_contact.evaluate( state );

  using PercentageRate = PointwiseMethod<stiffness::Constant, rate::Percentage>;
  Contact<PercentageRate>::Options percentage_options;
  percentage_options.search.expansion = 0.2;
  percentage_options.method.enforcement.stiffness.value = 2.0;
  percentage_options.method.enforcement.rate.ratio = 0.25;
  Contact<PercentageRate> percentage_contact( pair.view(), percentage_options );
  percentage_contact.updateInteractions();
  const auto percentage_result = percentage_contact.evaluate( state );

  return std::abs( sumComponent( constant_result.mortar_force, 1 ) - 2.0 ) < 1.0e-13 &&
         std::abs( sumComponent( percentage_result.mortar_force, 1 ) - 0.6 ) < 1.0e-13;
}

bool viscousAndTiedResponsesMatchLegacy()
{
  const SegmentPair pair;
  constexpr std::array<Real, 4> mortar_velocity{ 2.0, 0.0, 2.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_velocity{ -2.0, 0.0, -2.0, 0.0 };
  using Viscous = PointwiseMethod<stiffness::Constant, rate::None, response::ViscousTangential>;
  Contact<Viscous>::Options viscous_options;
  viscous_options.search.expansion = 0.2;
  viscous_options.method.response.damping = 0.5;
  Contact<Viscous> viscous_contact( pair.view(), viscous_options );
  viscous_contact.updateInteractions();
  const auto viscous_result = viscous_contact.evaluate(
      { .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );

  using Tied = PointwiseMethod<stiffness::Constant, rate::None, response::TiedNormal>;
  Contact<Tied>::Options tied_options;
  tied_options.search.expansion = 0.2;
  tied_options.method.enforcement.stiffness.value = 2.0;
  constexpr std::array<Real, 4> mortar_reference{ 0.0, -0.05, 1.0, -0.05 };
  constexpr std::array<Real, 4> nonmortar_reference{ 0.0, -0.05, 1.0, -0.05 };
  Contact<Tied> tied_contact( pair.view(), tied_options );
  tied_contact.updateInteractions();
  const auto tied_result = tied_contact.evaluate(
      { .mortar_reference_coordinates = { { mortar_reference.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar_reference_coordinates = { { nonmortar_reference.data(), 4 }, 2, 2, FieldLayout::Interleaved } } );

  return std::abs( sumComponent( viscous_result.mortar_force, 0 ) - 2.0 ) < 1.0e-13 &&
         std::abs( sumComponent( tied_result.mortar_force, 1 ) - 0.1 ) < 1.0e-13;
}

bool residualGapAndCentroidWeightsMatchLegacy()
{
  SegmentPair pair;
  pair.mortar = { 0.0, 0.0, 2.0, 0.0 };
  pair.nonmortar = { 0.0, 0.05, 1.0, 0.05 };
  Contact<>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 2.0;
  options.method.constraint.activation.residual_gap = 0.1;
  Contact<> contact( pair.view(), options );
  contact.updateInteractions();
  const auto result = contact.evaluate();
  return std::abs( result.mortar_force( 0, 1 ) - 0.0375 ) < 1.0e-13 &&
         std::abs( result.mortar_force( 1, 1 ) - 0.0125 ) < 1.0e-13 &&
         std::abs( result.nonmortar_force( 0, 1 ) + 0.025 ) < 1.0e-13 &&
         std::abs( result.nonmortar_force( 1, 1 ) + 0.025 ) < 1.0e-13;
}

}  // namespace

int main()
{
  return constantPenaltyMatchesLegacySeriesSpring() && materialPenaltyMatchesLegacySeriesSpring() &&
                 ratePenaltiesMatchLegacyPressure() && viscousAndTiedResponsesMatchLegacy() &&
                 residualGapAndCentroidWeightsMatchLegacy()
             ? 0
             : 1;
}
