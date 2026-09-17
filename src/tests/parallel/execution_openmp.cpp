#include "tests/spec/SupportedMethods.hpp"

#include "tribol/Tribol.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <omp.h>

// Requirements: PAR-004

namespace {

using namespace tribol;
using namespace tribol::test::spec;

struct RepeatedPatch {
  static constexpr Index elements = 64;
  static constexpr Index nodes = 4;
  static constexpr int dimension = 3;
  std::vector<Index> offsets;
  std::vector<Index> connectivity;
  std::vector<ElementTopology> topologies;
  std::vector<int> attributes;
  std::vector<ElementPair> pairs;
  std::vector<Real> thickness;
  std::vector<Real> modulus;
  std::array<Real, 12> mortar{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  std::array<Real, 12> nonmortar{ 0.0, 0.0, -0.1, 0.0, 1.0, -0.1, 1.0, 1.0, -0.1, 1.0, 0.0, -0.1 };

  RepeatedPatch()
      : offsets( static_cast<std::size_t>( elements + 1 ) ),
        connectivity( static_cast<std::size_t>( 4 * elements ) ),
        topologies( static_cast<std::size_t>( elements ), ElementTopology::Quadrilateral ),
        attributes( static_cast<std::size_t>( elements ), 1 ),
        pairs( static_cast<std::size_t>( elements ) ),
        thickness( static_cast<std::size_t>( elements ), 1.0 ),
        modulus( static_cast<std::size_t>( elements ), 10.0 )
  {
    for ( Index element = 0; element < elements; ++element ) {
      offsets[static_cast<std::size_t>( element )] = 4 * element;
      pairs[static_cast<std::size_t>( element )] = { element, element };
      for ( Index node = 0; node < 4; ++node ) {
        connectivity[static_cast<std::size_t>( 4 * element + node )] = node;
      }
    }
    offsets.back() = 4 * elements;
  }

  SurfaceMeshView surface( const std::array<Real, 12>& coordinates ) const
  {
    return { .dimension = 3,
             .coordinates = { { coordinates.data(), 12 }, 4, 3, FieldLayout::Interleaved },
             .element_offsets = { offsets.data(), static_cast<Index>( offsets.size() ) },
             .connectivity = { connectivity.data(), static_cast<Index>( connectivity.size() ) },
             .topologies = { topologies.data(), elements },
             .attributes = { attributes.data(), elements } };
  }

  SurfacePairView surfaces() const { return { surface( mortar ), surface( nonmortar ) }; }
};

struct RepeatedSegmentPatch {
  static constexpr Index elements = 64;
  static constexpr Index nodes = 2;
  static constexpr int dimension = 2;
  std::vector<Index> offsets;
  std::vector<Index> connectivity;
  std::vector<ElementTopology> topologies;
  std::vector<int> attributes;
  std::vector<ElementPair> pairs;
  std::vector<Real> thickness;
  std::vector<Real> modulus;
  std::array<Real, 4> mortar{ 0.0, 0.0, 1.0, 0.0 };
  std::array<Real, 4> nonmortar{ 0.0, -0.1, 1.0, -0.1 };

  RepeatedSegmentPatch()
      : offsets( static_cast<std::size_t>( elements + 1 ) ),
        connectivity( static_cast<std::size_t>( 2 * elements ) ),
        topologies( static_cast<std::size_t>( elements ), ElementTopology::Segment ),
        attributes( static_cast<std::size_t>( elements ), 1 ),
        pairs( static_cast<std::size_t>( elements ) ),
        thickness( static_cast<std::size_t>( elements ), 1.0 ),
        modulus( static_cast<std::size_t>( elements ), 10.0 )
  {
    for ( Index element = 0; element < elements; ++element ) {
      offsets[static_cast<std::size_t>( element )] = 2 * element;
      pairs[static_cast<std::size_t>( element )] = { element, element };
      connectivity[static_cast<std::size_t>( 2 * element )] = 0;
      connectivity[static_cast<std::size_t>( 2 * element + 1 )] = 1;
    }
    offsets.back() = 2 * elements;
  }

  SurfaceMeshView surface( const std::array<Real, 4>& coordinates ) const
  {
    return { .dimension = dimension,
             .coordinates = { { coordinates.data(), 4 }, nodes, dimension, FieldLayout::Interleaved },
             .element_offsets = { offsets.data(), static_cast<Index>( offsets.size() ) },
             .connectivity = { connectivity.data(), static_cast<Index>( connectivity.size() ) },
             .topologies = { topologies.data(), elements },
             .attributes = { attributes.data(), elements } };
  }

  SurfacePairView surfaces() const { return { surface( mortar ), surface( nonmortar ) }; }
};

bool close( ArrayView<const Real> left, ArrayView<const Real> right )
{
  if ( left.size() != right.size() ) {
    return false;
  }
  for ( Index entry = 0; entry < left.size(); ++entry ) {
    if ( std::abs( left[entry] - right[entry] ) > 1.0e-11 ) {
      return false;
    }
  }
  return true;
}

template <SupportedMethod MethodType, typename Options>
void configureMethod( Options& options )
{
  using Enforcement = typename MethodType::enforcement_policy;
  using Response = typename MethodType::response_policy;
  if constexpr ( detail::is_penalty_v<Enforcement> ) {
    using Stiffness = typename Enforcement::stiffness_policy;
    using Rate = typename Enforcement::rate_policy;
    if constexpr ( std::same_as<Stiffness, stiffness::Constant> ) {
      options.method.enforcement.stiffness.value = 10.0;
    } else {
      options.method.enforcement.stiffness.scale = 1.0;
    }
    if constexpr ( std::same_as<Rate, rate::Constant> ) {
      options.method.enforcement.rate.value = 0.25;
    } else if constexpr ( std::same_as<Rate, rate::Percentage> ) {
      options.method.enforcement.rate.ratio = 0.25;
    }
  }
  if constexpr ( std::same_as<Response, response::ViscousTangential> ) {
    options.method.response.damping = 0.5;
  }
}

template <SupportedMethod MethodType, typename Patch>
bool compareMethod( const Patch& patch )
{
  using SequentialContact = Contact<MethodType, search::Supplied, execution::Sequential>;
  using OpenMPContact = Contact<MethodType, search::Supplied, execution::OpenMP>;
  typename SequentialContact::Options sequential_options;
  typename OpenMPContact::Options openmp_options;
  sequential_options.search.pairs = patch.pairs;
  openmp_options.search.pairs = patch.pairs;
  configureMethod<MethodType>( sequential_options );
  configureMethod<MethodType>( openmp_options );
  SequentialContact sequential( patch.surfaces(), sequential_options );
  OpenMPContact parallel( patch.surfaces(), openmp_options );
  sequential.updateInteractions();
  parallel.updateInteractions();

  std::vector<Real> mortar_velocity( static_cast<std::size_t>( Patch::nodes * Patch::dimension ), 0.0 );
  std::vector<Real> nonmortar_velocity( static_cast<std::size_t>( Patch::nodes * Patch::dimension ), 0.0 );
  std::vector<Real> multiplier( static_cast<std::size_t>( Patch::nodes ), -1.0 );
  std::vector<Real> potential( static_cast<std::size_t>( Patch::nodes ), 0.05 );
  std::vector<Real> tangent( static_cast<std::size_t>( Patch::nodes ), 10.0 );
  for ( Index node = 0; node < Patch::nodes; ++node ) {
    nonmortar_velocity[static_cast<std::size_t>( node * Patch::dimension )] = 0.2;
    nonmortar_velocity[static_cast<std::size_t>( node * Patch::dimension + Patch::dimension - 1 )] = 0.3;
  }
  const ContactStateView state{
      .mortar_velocity = { { mortar_velocity.data(), static_cast<Index>( mortar_velocity.size() ) },
                           Patch::nodes,
                           Patch::dimension,
                           FieldLayout::Interleaved },
      .nonmortar_velocity = { { nonmortar_velocity.data(), static_cast<Index>( nonmortar_velocity.size() ) },
                              Patch::nodes,
                              Patch::dimension,
                              FieldLayout::Interleaved },
      .mortar_reference_coordinates = { { patch.mortar.data(), Patch::nodes * Patch::dimension },
                                        Patch::nodes,
                                        Patch::dimension,
                                        FieldLayout::Interleaved },
      .nonmortar_reference_coordinates = { { patch.nonmortar.data(), Patch::nodes * Patch::dimension },
                                           Patch::nodes,
                                           Patch::dimension,
                                           FieldLayout::Interleaved },
      .mortar_element_thickness = { patch.thickness.data(), Patch::elements },
      .nonmortar_element_thickness = { patch.thickness.data(), Patch::elements },
      .mortar_material_modulus = { patch.modulus.data(), Patch::elements },
      .nonmortar_material_modulus = { patch.modulus.data(), Patch::elements },
      .multiplier = { multiplier.data(), Patch::nodes },
      .external_potential_density = { potential.data(), Patch::nodes },
      .external_pressure = { multiplier.data(), Patch::nodes },
      .external_pressure_tangent = { tangent.data(), Patch::nodes },
  };
  const auto expected = sequential.evaluate( state );
  const auto actual = parallel.evaluate( state );
  bool matches = expected.summary.active_interactions == actual.summary.active_interactions &&
                 expected.summary.quadrature_points == actual.summary.quadrature_points &&
                 std::abs( expected.summary.energy - actual.summary.energy ) < 1.0e-11 &&
                 close( expected.mortar_force.values, actual.mortar_force.values ) &&
                 close( expected.nonmortar_force.values, actual.nonmortar_force.values ) &&
                 close( expected.constraint_residual, actual.constraint_residual ) &&
                 close( expected.gap, actual.gap ) && close( expected.weighted_gap, actual.weighted_gap ) &&
                 close( expected.tributary_area, actual.tributary_area ) && close( expected.pressure, actual.pressure );
  if constexpr ( std::same_as<typename MethodType::formulation_policy, formulation::Variational> &&
                 detail::is_nodal_constraint_v<typename MethodType::constraint_policy> ) {
    const auto expected_kinematics = sequential.evaluateNodalKinematics();
    const auto actual_kinematics = parallel.evaluateNodalKinematics();
    matches = matches && close( expected_kinematics.gap, actual_kinematics.gap ) &&
              close( expected_kinematics.weighted_gap, actual_kinematics.weighted_gap ) &&
              close( expected_kinematics.tributary_area, actual_kinematics.tributary_area );
  }
  return matches;
}

}  // namespace

int main()
{
  omp_set_dynamic( 0 );
  omp_set_num_threads( 4 );
  int threads{};
#pragma omp parallel
#pragma omp single
  threads = omp_get_num_threads();
  const RepeatedPatch patch;
  const RepeatedSegmentPatch segment_patch;
  return threads > 1 && compareMethod<PointwiseFamily<stiffness::Constant, rate::None>>( patch ) &&
                 compareMethod<PointwiseFamily<stiffness::Constant, rate::None, response::ViscousTangential>>(
                     patch ) &&
                 compareMethod<PointwiseFamily<stiffness::Constant, rate::Constant>>( patch ) &&
                 compareMethod<PointwiseFamily<stiffness::Constant, rate::Constant, response::ViscousTangential>>(
                     patch ) &&
                 compareMethod<PointwiseFamily<stiffness::Constant, rate::Percentage>>( patch ) &&
                 compareMethod<PointwiseFamily<stiffness::Constant, rate::Percentage, response::ViscousTangential>>(
                     patch ) &&
                 compareMethod<PointwiseFamily<stiffness::Material, rate::None>>( patch ) &&
                 compareMethod<PointwiseFamily<stiffness::Material, rate::None, response::ViscousTangential>>(
                     patch ) &&
                 compareMethod<PointwiseFamily<stiffness::Material, rate::Constant>>( patch ) &&
                 compareMethod<PointwiseFamily<stiffness::Material, rate::Constant, response::ViscousTangential>>(
                     patch ) &&
                 compareMethod<PointwiseFamily<stiffness::Material, rate::Percentage>>( patch ) &&
                 compareMethod<PointwiseMaterialRateViscous>( patch ) && compareMethod<PointwiseTiedNormal>( patch ) &&
                 compareMethod<PointwiseTiedFull>( patch ) && compareMethod<ProjectedMultiplier>( patch ) &&
                 compareMethod<ConformingMultiplier>( patch ) && compareMethod<DiagnosticWeights>( patch ) &&
                 compareMethod<VariationalNodalPenalty>( patch ) &&
                 compareMethod<VariationalQuadraturePenalty>( patch ) &&
                 compareMethod<VariationalMultiplier>( patch ) && compareMethod<VariationalExternalPressure>( patch ) &&
                 compareMethod<SmoothedVariationalNodalPenalty>( segment_patch ) &&
                 compareMethod<SmoothedVariationalQuadraturePenalty>( segment_patch )
             ? 0
             : 1;
}
