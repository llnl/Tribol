#include "tests/spec/SupportedMethods.hpp"

#include "tribol/Tribol.hpp"

#include <array>
#include <cmath>
#include <type_traits>

// Requirements: CORE-002, DIFF-001, PHYS-001, PHYS-003, RESP-001, RESP-002

namespace {

using namespace tribol;
using namespace tribol::test::spec;

struct Patch {
  int dimension{};
  Index nodes{};
  ElementTopology topology{};
  int normal_component{};
  std::array<Real, 12> mortar_coordinates{};
  std::array<Real, 12> nonmortar_coordinates{};
  std::array<Index, 2> offsets{};
  std::array<Index, 4> connectivity{};
  std::array<ElementTopology, 1> topologies{};
  std::array<int, 1> attributes{ 1 };

  [[nodiscard]] SurfaceMeshView surface( const std::array<Real, 12>& coordinates ) const
  {
    return {
        .dimension = dimension,
        .coordinates = { { coordinates.data(), nodes * dimension }, nodes, dimension, FieldLayout::Interleaved },
        .element_offsets = { offsets.data(), 2 },
        .connectivity = { connectivity.data(), nodes },
        .topologies = { topologies.data(), 1 },
        .attributes = { attributes.data(), 1 },
    };
  }

  [[nodiscard]] SurfacePairView surfaces() const
  {
    return { surface( mortar_coordinates ), surface( nonmortar_coordinates ) };
  }
};

Patch segmentPatch()
{
  Patch patch;
  patch.dimension = 2;
  patch.nodes = 2;
  patch.topology = ElementTopology::Segment;
  patch.normal_component = 1;
  patch.mortar_coordinates = { 1.0, 0.0, 0.0, 0.0 };
  patch.nonmortar_coordinates = { 0.0, -0.1, 1.0, -0.1 };
  patch.offsets = { 0, 2 };
  patch.connectivity = { 0, 1, 0, 0 };
  patch.topologies = { patch.topology };
  return patch;
}

Patch trianglePatch()
{
  Patch patch;
  patch.dimension = 3;
  patch.nodes = 3;
  patch.topology = ElementTopology::Triangle;
  patch.normal_component = 2;
  patch.mortar_coordinates = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
  patch.nonmortar_coordinates = { 0.0, 0.0, -0.1, 0.0, 1.0, -0.1, 1.0, 0.0, -0.1 };
  patch.offsets = { 0, 3 };
  patch.connectivity = { 0, 1, 2, 0 };
  patch.topologies = { patch.topology };
  return patch;
}

Patch quadrilateralPatch()
{
  Patch patch;
  patch.dimension = 3;
  patch.nodes = 4;
  patch.topology = ElementTopology::Quadrilateral;
  patch.normal_component = 2;
  patch.mortar_coordinates = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  patch.nonmortar_coordinates = { 0.0, 0.0, -0.1, 0.0, 1.0, -0.1, 1.0, 1.0, -0.1, 1.0, 0.0, -0.1 };
  patch.offsets = { 0, 4 };
  patch.connectivity = { 0, 1, 2, 3 };
  patch.topologies = { patch.topology };
  return patch;
}

template <SupportedMethod MethodType>
bool exercise( const Patch& patch )
{
  typename Contact<MethodType>::Options options;
  options.search.expansion = 0.2;
  Contact<MethodType> contact( patch.surfaces(), options );
  contact.updateInteractions();

  std::array<Real, 12> mortar_velocity{};
  std::array<Real, 12> nonmortar_velocity{};
  std::array<Real, 12> mortar_reference = patch.mortar_coordinates;
  std::array<Real, 12> nonmortar_reference = patch.nonmortar_coordinates;
  for ( Index node = 0; node < patch.nodes; ++node ) {
    nonmortar_reference[static_cast<std::size_t>( node * patch.dimension + patch.normal_component )] -= 0.1;
  }
  constexpr std::array<Real, 1> thickness{ 1.0 };
  constexpr std::array<Real, 1> modulus{ 10.0 };
  constexpr std::array<Real, 4> multiplier{ -1.0, -1.0, -1.0, -1.0 };
  constexpr std::array<Real, 4> pressure{ -1.0, -1.0, -1.0, -1.0 };
  const ContactStateView state{
      .mortar_velocity = { { mortar_velocity.data(), patch.nodes * patch.dimension },
                           patch.nodes,
                           patch.dimension,
                           FieldLayout::Interleaved },
      .nonmortar_velocity = { { nonmortar_velocity.data(), patch.nodes * patch.dimension },
                              patch.nodes,
                              patch.dimension,
                              FieldLayout::Interleaved },
      .mortar_reference_coordinates = { { mortar_reference.data(), patch.nodes * patch.dimension },
                                        patch.nodes,
                                        patch.dimension,
                                        FieldLayout::Interleaved },
      .nonmortar_reference_coordinates = { { nonmortar_reference.data(), patch.nodes * patch.dimension },
                                           patch.nodes,
                                           patch.dimension,
                                           FieldLayout::Interleaved },
      .mortar_element_thickness = { thickness.data(), 1 },
      .nonmortar_element_thickness = { thickness.data(), 1 },
      .mortar_material_modulus = { modulus.data(), 1 },
      .nonmortar_material_modulus = { modulus.data(), 1 },
      .multiplier = { multiplier.data(), patch.nodes },
      .external_pressure = { pressure.data(), patch.nodes },
  };
  const auto result = contact.evaluate( state );
  if ( result.summary.active_interactions != 1 || result.summary.quadrature_points < 1 ||
       !std::isfinite( result.summary.energy ) || std::isnan( result.summary.timestep_vote ) ) {
    return false;
  }

  Real force_norm{};
  for ( int component = 0; component < patch.dimension; ++component ) {
    Real balance{};
    for ( Index node = 0; node < patch.nodes; ++node ) {
      const Real mortar = result.mortar_force( node, component );
      const Real nonmortar = result.nonmortar_force( node, component );
      if ( !std::isfinite( mortar ) || !std::isfinite( nonmortar ) ) {
        return false;
      }
      balance += mortar + nonmortar;
      force_norm += mortar * mortar + nonmortar * nonmortar;
    }
    if ( std::abs( balance ) > 1.0e-11 ) {
      return false;
    }
  }
  if constexpr ( MethodTraits<MethodType>::capabilities.produces_diagnostic_weights ) {
    Real weight_sum{};
    for ( Real value : result.mortar_weights ) {
      weight_sum += value;
    }
    return weight_sum > 0.0 && force_norm == 0.0;
  } else {
    return force_norm > 0.0;
  }
}

template <SupportedMethod... Methods>
bool exercisePointwiseMethods()
{
  const auto segment = segmentPatch();
  const auto triangle = trianglePatch();
  const auto quadrilateral = quadrilateralPatch();
  return ( ( exercise<Methods>( segment ) && exercise<Methods>( triangle ) && exercise<Methods>( quadrilateral ) ) &&
           ... );
}

template <SupportedMethod... Methods>
bool exerciseFaceMethods()
{
  const auto triangle = trianglePatch();
  const auto quadrilateral = quadrilateralPatch();
  return ( ( exercise<Methods>( triangle ) && exercise<Methods>( quadrilateral ) ) && ... );
}

template <SupportedMethod... Methods>
bool exerciseVariationalMethods()
{
  return exercisePointwiseMethods<Methods...>();
}

}  // namespace

int main()
{
  return exercisePointwiseMethods<PointwisePenalty, PointwiseMaterialRateViscous, PointwiseTiedNormal,
                                  PointwiseTiedFull>() &&
                 exerciseFaceMethods<ProjectedMultiplier, ConformingMultiplier, DiagnosticWeights>() &&
                 exerciseVariationalMethods<VariationalNodalPenalty, VariationalQuadraturePenalty,
                                            VariationalMultiplier, VariationalExternalPressure>()
             ? 0
             : 1;
}
