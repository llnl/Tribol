#include "tribol/execution/DevicePenalty.hpp"
#include "tribol/execution/DeviceRows.hpp"
#include "tribol/contact/Contact.hpp"

#include "../components/ConstraintRowFixture.hpp"
#include "../components/ParentTraceFixture.hpp"

#if defined( TRIBOL_TEST_USE_HIP )
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

// Requirements: API-003, CORE-003, DIFF-001, DIFF-002, OUTPUT-001, OUTPUT-003, PAR-001, PAR-006, PAR-008, SEARCH-001,
// TIME-001

namespace {

#if defined( TRIBOL_TEST_USE_HIP )
using DeviceExecution = tribol::execution::Hip;

bool deviceAvailable() { return tribol::execution::hipDeviceAvailable(); }

bool synchronizeAndGetMemoryInfo( std::size_t* free, std::size_t* total )
{
  return hipDeviceSynchronize() == hipSuccess && hipMemGetInfo( free, total ) == hipSuccess;
}
#else
using DeviceExecution = tribol::execution::Cuda;

bool deviceAvailable() { return tribol::execution::cudaDeviceAvailable(); }

bool synchronizeAndGetMemoryInfo( std::size_t* free, std::size_t* total )
{
  return cudaDeviceSynchronize() == cudaSuccess && cudaMemGetInfo( free, total ) == cudaSuccess;
}
#endif

using NativeExplicitMethod =
    tribol::Method<tribol::geometry::ProjectedOverlap<tribol::normal::MeanPlane>, tribol::integration::Automatic,
                   tribol::constraint::QuadraturePoint, tribol::enforcement::Penalty<>, tribol::response::Frictionless,
                   tribol::formulation::PointwiseTraction, tribol::linearization::Exact>;

tribol::InteractionPatch makePatch( tribol::Real gap, tribol::Real measure )
{
  tribol::InteractionPatch patch;
  patch.normal = { 0.0, 0.0, -1.0 };
  patch.integration_vertices[0] = { 0.0, 0.0, -0.5 * gap };
  patch.integration_vertices[1] = { measure, 0.0, -0.5 * gap };
  patch.mortar_vertices[0] = { 0.0, 0.0, 0.0 };
  patch.mortar_vertices[1] = { measure, 0.0, 0.0 };
  patch.nonmortar_vertices[0] = { 0.0, 0.0, -gap };
  patch.nonmortar_vertices[1] = { measure, 0.0, -gap };
  patch.mortar_centroid = { 0.5 * measure, 0.0, 0.0 };
  patch.nonmortar_centroid = { 0.5 * measure, 0.0, -gap };
  patch.measure = measure;
  patch.vertex_count = 2;
  patch.manifold_dimension = 1;
  patch.valid = true;
  return patch;
}

tribol::SurfaceMeshView makeSegment( const std::array<tribol::Real, 4>& coordinates )
{
  static constexpr std::array<tribol::Index, 2> offsets{ 0, 2 };
  static constexpr std::array<tribol::Index, 2> connectivity{ 0, 1 };
  static constexpr std::array<tribol::ElementTopology, 1> topologies{ tribol::ElementTopology::Segment };
  static constexpr std::array<int, 1> attributes{ 1 };
  return {
      .dimension = 2,
      .coordinates = { { coordinates.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .element_offsets = { offsets.data(), 2 },
      .connectivity = { connectivity.data(), 2 },
      .topologies = { topologies.data(), 1 },
      .attributes = { attributes.data(), 1 },
  };
}

tribol::SurfaceMeshView makeSegment( const std::array<tribol::Real, 4>& coordinates,
                                     const std::array<tribol::Index, 2>& connectivity )
{
  static constexpr std::array<tribol::Index, 2> offsets{ 0, 2 };
  static constexpr std::array<tribol::ElementTopology, 1> topologies{ tribol::ElementTopology::Segment };
  static constexpr std::array<int, 1> attributes{ 1 };
  return {
      .dimension = 2,
      .coordinates = { { coordinates.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .element_offsets = { offsets.data(), 2 },
      .connectivity = { connectivity.data(), 2 },
      .topologies = { topologies.data(), 1 },
      .attributes = { attributes.data(), 1 },
  };
}

bool arraysMatch( tribol::ArrayView<const tribol::Real> host, tribol::ArrayView<const tribol::Real> device,
                  tribol::Real tolerance = 1.0e-13 )
{
  if ( host.size() != device.size() ) {
    return false;
  }
  for ( tribol::Index value = 0; value < host.size(); ++value ) {
    if ( std::abs( host[value] - device[value] ) > tolerance ) {
      return false;
    }
  }
  return true;
}

bool rowDiagnosticsMatch( const tribol::diagnostics::ExplicitRowDiagnostics& left,
                          const tribol::diagnostics::ExplicitRowDiagnostics& right )
{
  if ( left.rows != right.rows || left.owned_rows != right.owned_rows || left.active_rows != right.active_rows ||
       left.nonfinite_rows != right.nonfinite_rows || std::abs( left.minimum_gap - right.minimum_gap ) > 1.0e-13 ||
       std::abs( left.maximum_penetration - right.maximum_penetration ) > 1.0e-13 ||
       std::abs( left.maximum_closing_rate - right.maximum_closing_rate ) > 1.0e-13 ||
       std::abs( left.maximum_traction - right.maximum_traction ) > 1.0e-13 ||
       std::abs( left.spring_energy - right.spring_energy ) > 1.0e-13 ||
       std::abs( left.dashpot_dissipation - right.dashpot_dissipation ) > 1.0e-13 ||
       std::abs( left.contact_work - right.contact_work ) > 1.0e-13 ) {
    return false;
  }
  for ( int component = 0; component < 3; ++component ) {
    if ( std::abs( left.mortar_force[component] - right.mortar_force[component] ) > 1.0e-13 ||
         std::abs( left.mortar_moment[component] - right.mortar_moment[component] ) > 1.0e-13 ||
         std::abs( left.impulse[component] - right.impulse[component] ) > 1.0e-13 ) {
      return false;
    }
  }
  return true;
}

bool diagnosticsMatch( const tribol::diagnostics::RankProductionDiagnostics& left,
                       const tribol::diagnostics::RankProductionDiagnostics& right )
{
  return left.rank == right.rank && left.ranks == right.ranks &&
         rowDiagnosticsMatch( left.local.rows, right.local.rows ) &&
         rowDiagnosticsMatch( left.owned.rows, right.owned.rows ) &&
         left.local.quadrature.samples == right.local.quadrature.samples &&
         left.local.quadrature.owned_samples == right.local.quadrature.owned_samples &&
         std::abs( left.local.quadrature.maximum_penetration - right.local.quadrature.maximum_penetration ) < 1.0e-13 &&
         std::abs( left.local.quadrature.maximum_closing_rate - right.local.quadrature.maximum_closing_rate ) <
             1.0e-13 &&
         left.owned.quadrature.samples == right.owned.quadrature.samples &&
         left.owned.quadrature.owned_samples == right.owned.quadrature.owned_samples &&
         std::abs( left.owned.quadrature.maximum_penetration - right.owned.quadrature.maximum_penetration ) < 1.0e-13 &&
         std::abs( left.owned.quadrature.maximum_closing_rate - right.owned.quadrature.maximum_closing_rate ) < 1.0e-13;
}

bool fullContactParity()
{
  std::array<tribol::Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  std::array<tribol::Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  const tribol::SurfacePairView surfaces{ makeSegment( mortar_coordinates ), makeSegment( nonmortar_coordinates ) };
  using HostContact = tribol::Contact<>;
  using DeviceContact = tribol::Contact<tribol::DefaultMethod, tribol::search::Bvh, DeviceExecution>;
  HostContact::Options host_options;
  host_options.search.expansion = 0.2;
  host_options.method.enforcement.stiffness.value = 12.5;
  host_options.method.constraint.activation.residual_gap = 0.025;
  host_options.method.constraint.activation.gap_tolerance = 0.01;
  DeviceContact::Options device_options;
  device_options.search.expansion = host_options.search.expansion;
  device_options.method.enforcement.stiffness.value = host_options.method.enforcement.stiffness.value;
  device_options.method.constraint.activation = host_options.method.constraint.activation;
  HostContact host( surfaces, host_options );
  DeviceContact device( surfaces, device_options );
  host.updateInteractions();
  device.updateInteractions();
  const auto host_result = host.evaluate();
  const auto device_result = device.evaluate();
  if ( host_result.summary.active_interactions != device_result.summary.active_interactions ||
       std::abs( host_result.summary.energy - device_result.summary.energy ) > 1.0e-13 ) {
    return false;
  }
  if ( !arraysMatch( host_result.mortar_force.values, device_result.mortar_force.values ) ||
       !arraysMatch( host_result.nonmortar_force.values, device_result.nonmortar_force.values ) ||
       !arraysMatch( host_result.gap, device_result.gap ) ||
       !arraysMatch( host_result.weighted_gap, device_result.weighted_gap ) ||
       !arraysMatch( host_result.tributary_area, device_result.tributary_area ) ||
       !arraysMatch( host_result.mortar_weights, device_result.mortar_weights ) ||
       !arraysMatch( host_result.mortar_mass_weights, device_result.mortar_mass_weights ) ||
       !arraysMatch( host_result.quadrature_gap, device_result.quadrature_gap ) ||
       !arraysMatch( host_result.quadrature_pressure, device_result.quadrature_pressure ) ||
       !arraysMatch( host_result.pressure, device_result.pressure ) ) {
    return false;
  }
  constexpr std::array<tribol::Real, 4> mortar_direction{ 0.0, 0.25, 0.0, 0.25 };
  constexpr std::array<tribol::Real, 4> nonmortar_direction{ 0.0, 1.0, 0.0, 1.0 };
  const tribol::ContactDirectionView direction{
      .mortar = { { mortar_direction.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .nonmortar = { { nonmortar_direction.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
  };
  std::array<tribol::Real, 4> host_mortar_derivative{};
  std::array<tribol::Real, 4> host_nonmortar_derivative{};
  std::array<tribol::Real, 4> device_mortar_derivative{};
  std::array<tribol::Real, 4> device_nonmortar_derivative{};
  host.applyCoordinateDerivative(
      {}, direction,
      { .mortar = { { host_mortar_derivative.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
        .nonmortar = { { host_nonmortar_derivative.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved } } );
  device.applyCoordinateDerivative(
      {}, direction,
      { .mortar = { { device_mortar_derivative.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
        .nonmortar = { { device_nonmortar_derivative.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved } } );
  for ( std::size_t value = 0; value < host_mortar_derivative.size(); ++value ) {
    if ( std::abs( host_mortar_derivative[value] - device_mortar_derivative[value] ) > 5.0e-8 ||
         std::abs( host_nonmortar_derivative[value] - device_nonmortar_derivative[value] ) > 5.0e-8 ) {
      return false;
    }
  }
  constexpr tribol::Real finite_difference_step = 1.0e-6;
  for ( std::size_t value = 0; value < mortar_coordinates.size(); ++value ) {
    mortar_coordinates[value] += finite_difference_step * mortar_direction[value];
    nonmortar_coordinates[value] += finite_difference_step * nonmortar_direction[value];
  }
  device.updateGeometry( surfaces );
  const auto positive = device.evaluate();
  const std::vector<tribol::Real> positive_mortar( positive.mortar_force.values.begin(),
                                                   positive.mortar_force.values.end() );
  const std::vector<tribol::Real> positive_nonmortar( positive.nonmortar_force.values.begin(),
                                                      positive.nonmortar_force.values.end() );
  for ( std::size_t value = 0; value < mortar_coordinates.size(); ++value ) {
    mortar_coordinates[value] -= 2.0 * finite_difference_step * mortar_direction[value];
    nonmortar_coordinates[value] -= 2.0 * finite_difference_step * nonmortar_direction[value];
  }
  device.updateGeometry( surfaces );
  const auto negative = device.evaluate();
  for ( std::size_t value = 0; value < device_mortar_derivative.size(); ++value ) {
    const tribol::Real mortar_finite_difference =
        ( positive_mortar[value] - negative.mortar_force.values[value] ) / ( 2.0 * finite_difference_step );
    const tribol::Real nonmortar_finite_difference =
        ( positive_nonmortar[value] - negative.nonmortar_force.values[value] ) / ( 2.0 * finite_difference_step );
    if ( std::abs( mortar_finite_difference - device_mortar_derivative[value] ) > 5.0e-7 ||
         std::abs( nonmortar_finite_difference - device_nonmortar_derivative[value] ) > 5.0e-7 ) {
      return false;
    }
    mortar_coordinates[value] += finite_difference_step * mortar_direction[value];
    nonmortar_coordinates[value] += finite_difference_step * nonmortar_direction[value];
  }
  device.updateGeometry( surfaces );
  const auto host_jacobian = host.assembleCoordinateJacobian( {} );
  const auto device_jacobian = device.assembleCoordinateJacobian( {} );
  if ( host_jacobian.rows != device_jacobian.rows || host_jacobian.columns != device_jacobian.columns ) {
    return false;
  }
  for ( tribol::Index value = 0; value < host_jacobian.values.size(); ++value ) {
    if ( std::abs( host_jacobian.values[value] - device_jacobian.values[value] ) > 5.0e-8 ) {
      return false;
    }
  }
  return true;
}

bool activationAndPenetrationParity()
{
  std::array<tribol::InteractionPatch, 1> patches{ makePatch( -0.01, 1.0 ) };
  std::array<tribol::execution::PenaltyContribution, 1> sequential{};
  std::array<tribol::execution::PenaltyContribution, 1> device_contributions{};
  tribol::constraint::GapActivationParameters activation;
  activation.residual_gap = 0.005;
  activation.gap_tolerance = 0.01;
  sequential[0] = tribol::execution::evaluatePointwisePenaltyPatch( patches[0], 12.5, activation );
  tribol::execution::evaluatePointwisePenaltyPatches( { patches.data(), 1 }, 12.5, { device_contributions.data(), 1 },
                                                      DeviceExecution{}, activation );
  if ( sequential != device_contributions || std::abs( device_contributions[0].effective_gap - 0.005 ) > 1.0e-13 ||
       device_contributions[0].pressure <= 0.0 ) {
    return false;
  }

  constexpr std::array<tribol::Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<tribol::Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  const tribol::SurfacePairView surfaces{ makeSegment( mortar_coordinates ), makeSegment( nonmortar_coordinates ) };
  using HostContact = tribol::Contact<>;
  using DeviceContact = tribol::Contact<tribol::DefaultMethod, tribol::search::Bvh, DeviceExecution>;
  HostContact::Options host_options;
  host_options.search.expansion = 0.2;
  host_options.method.constraint.activation.reject_excessive_penetration = true;
  DeviceContact::Options device_options;
  device_options.search.expansion = host_options.search.expansion;
  device_options.method.constraint.activation = host_options.method.constraint.activation;
  HostContact host( surfaces, host_options );
  DeviceContact device( surfaces, device_options );
  host.updateInteractions();
  device.updateInteractions();
  constexpr std::array<tribol::Real, 1> thickness{ 0.05 };
  const tribol::ContactStateView state{ .mortar_element_thickness = { thickness.data(), 1 },
                                        .nonmortar_element_thickness = { thickness.data(), 1 } };
  return host.evaluate( state ).summary.active_interactions == 0 &&
         device.evaluate( state ).summary.active_interactions == 0;
}

bool isDevicePointer( const void* pointer )
{
  if ( pointer == nullptr ) {
    return false;
  }
#if defined( TRIBOL_TEST_USE_HIP )
  hipPointerAttribute_t attributes{};
  if ( hipPointerGetAttributes( &attributes, pointer ) != hipSuccess ) {
    return false;
  }
#if HIP_VERSION_MAJOR >= 6
  return attributes.type == hipMemoryTypeDevice;
#else
  return attributes.memoryType == hipMemoryTypeDevice;
#endif
#else
  cudaPointerAttributes attributes{};
  return cudaPointerGetAttributes( &attributes, pointer ) == cudaSuccess && attributes.type == cudaMemoryTypeDevice;
#endif
}

bool requireDevicePointer( const char* name, const void* pointer )
{
  if ( isDevicePointer( pointer ) ) {
    return true;
  }
#if defined( TRIBOL_TEST_USE_HIP )
  hipPointerAttribute_t attributes{};
  const auto error = pointer == nullptr ? hipErrorInvalidValue : hipPointerGetAttributes( &attributes, pointer );
  std::cerr << name << " is not device-resident: pointer=" << pointer << " error=" << hipGetErrorString( error )
            << " type="
            << ( error == hipSuccess
#if HIP_VERSION_MAJOR >= 6
                     ? static_cast<int>( attributes.type )
#else
                     ? static_cast<int>( attributes.memoryType )
#endif
                     : -1 )
            << '\n';
#else
  cudaPointerAttributes attributes{};
  const auto error = pointer == nullptr ? cudaErrorInvalidValue : cudaPointerGetAttributes( &attributes, pointer );
  std::cerr << name << " is not device-resident: pointer=" << pointer << " error=" << cudaGetErrorString( error )
            << " type=" << ( error == cudaSuccess ? static_cast<int>( attributes.type ) : -1 ) << '\n';
#endif
  return false;
}

struct SharedSegmentPair {
  std::vector<tribol::Real> mortar_coordinates;
  std::vector<tribol::Real> nonmortar_coordinates;
  std::vector<tribol::Index> offsets;
  std::vector<tribol::Index> connectivity;
  std::vector<tribol::ElementTopology> topologies;
  std::vector<int> attributes;

  explicit SharedSegmentPair( tribol::Index elements )
  {
    mortar_coordinates.reserve( static_cast<std::size_t>( 2 * ( elements + 1 ) ) );
    nonmortar_coordinates.reserve( static_cast<std::size_t>( 2 * ( elements + 1 ) ) );
    offsets.reserve( static_cast<std::size_t>( elements + 1 ) );
    connectivity.reserve( static_cast<std::size_t>( 2 * elements ) );
    offsets.push_back( 0 );
    for ( tribol::Index node = 0; node <= elements; ++node ) {
      mortar_coordinates.push_back( static_cast<tribol::Real>( node ) );
      mortar_coordinates.push_back( 0.0 );
      nonmortar_coordinates.push_back( static_cast<tribol::Real>( node ) );
      nonmortar_coordinates.push_back( -0.1 );
    }
    for ( tribol::Index element = 0; element < elements; ++element ) {
      connectivity.push_back( element );
      connectivity.push_back( element + 1 );
      offsets.push_back( static_cast<tribol::Index>( connectivity.size() ) );
      topologies.push_back( tribol::ElementTopology::Segment );
      attributes.push_back( 1 );
    }
  }

  tribol::SurfaceMeshView surface( const std::vector<tribol::Real>& coordinates ) const
  {
    return {
        .dimension = 2,
        .coordinates = { { coordinates.data(), static_cast<tribol::Index>( coordinates.size() ) },
                         static_cast<tribol::Index>( coordinates.size() / 2 ),
                         2,
                         tribol::FieldLayout::Interleaved },
        .element_offsets = { offsets.data(), static_cast<tribol::Index>( offsets.size() ) },
        .connectivity = { connectivity.data(), static_cast<tribol::Index>( connectivity.size() ) },
        .topologies = { topologies.data(), static_cast<tribol::Index>( topologies.size() ) },
        .attributes = { attributes.data(), static_cast<tribol::Index>( attributes.size() ) },
    };
  }

  tribol::SurfacePairView view() const { return { surface( mortar_coordinates ), surface( nonmortar_coordinates ) }; }
};

struct RepeatedQuadrilateralPair {
  static constexpr tribol::Index elements = 32;
  std::array<tribol::Real, 12> mortar_coordinates{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  std::array<tribol::Real, 12> nonmortar_coordinates{ 0.0, 0.0, -0.1, 0.0, 1.0, -0.1, 1.0, 1.0, -0.1, 1.0, 0.0, -0.1 };
  std::vector<tribol::Index> offsets;
  std::vector<tribol::Index> connectivity;
  std::vector<tribol::ElementTopology> topologies;
  std::vector<int> attributes;
  std::vector<tribol::ElementPair> pairs;

  RepeatedQuadrilateralPair()
      : offsets( static_cast<std::size_t>( elements + 1 ) ),
        connectivity( static_cast<std::size_t>( 4 * elements ) ),
        topologies( static_cast<std::size_t>( elements ), tribol::ElementTopology::Quadrilateral ),
        attributes( static_cast<std::size_t>( elements ), 1 )
  {
    pairs.reserve( static_cast<std::size_t>( elements + 1 ) );
    for ( tribol::Index element = elements; element-- > 0; ) {
      offsets[static_cast<std::size_t>( element )] = 4 * element;
      pairs.push_back( { element, element } );
      for ( tribol::Index node = 0; node < 4; ++node ) {
        connectivity[static_cast<std::size_t>( 4 * element + node )] = node;
      }
    }
    offsets.back() = 4 * elements;
    pairs.push_back( { elements - 1, elements - 1 } );
  }

  tribol::SurfaceMeshView surface( const std::array<tribol::Real, 12>& coordinates ) const
  {
    return {
        .dimension = 3,
        .coordinates = { { coordinates.data(), 12 }, 4, 3, tribol::FieldLayout::Interleaved },
        .element_offsets = { offsets.data(), static_cast<tribol::Index>( offsets.size() ) },
        .connectivity = { connectivity.data(), static_cast<tribol::Index>( connectivity.size() ) },
        .topologies = { topologies.data(), elements },
        .attributes = { attributes.data(), elements },
    };
  }

  tribol::SurfacePairView view() const { return { surface( mortar_coordinates ), surface( nonmortar_coordinates ) }; }
};

bool deviceResidentBvhAndDeterministicScatter()
{
  SharedSegmentPair mesh( 64 );
  using HostContact = tribol::Contact<tribol::DefaultMethod, tribol::search::Bvh>;
  using DeviceContact = tribol::Contact<tribol::DefaultMethod, tribol::search::Bvh, DeviceExecution>;
  HostContact::Options host_options;
  DeviceContact::Options device_options;
  host_options.search.expansion = 0.2;
  device_options.search.expansion = host_options.search.expansion;
  host_options.method.enforcement.stiffness.value = 7.25;
  device_options.method.enforcement.stiffness.value = host_options.method.enforcement.stiffness.value;
  HostContact host( mesh.view(), host_options );
  DeviceContact device( mesh.view(), device_options );
  host.updateInteractions();
  device.updateInteractions();
  const auto host_pairs = host.interactions();
  const auto device_pairs = device.interactions();
  if ( host_pairs.size() != device_pairs.size() ) {
    std::cerr << "candidate count mismatch: host=" << host_pairs.size() << " device=" << device_pairs.size() << '\n';
    return false;
  }
  for ( tribol::Index pair = 0; pair < host_pairs.size(); ++pair ) {
    if ( host_pairs[pair] != device_pairs[pair] ) {
      std::cerr << "candidate mismatch at " << pair << ": host=(" << host_pairs[pair].mortar_element << ','
                << host_pairs[pair].nonmortar_element << ") device=(" << device_pairs[pair].mortar_element << ','
                << device_pairs[pair].nonmortar_element << ")\n";
      return false;
    }
  }

  const auto device_result = device.evaluateDevice();
  const auto pipeline = device.devicePipelineView();
  if ( !requireDevicePointer( "mortar coordinates", pipeline.surfaces.mortar.coordinates.values.data() ) ||
       !requireDevicePointer( "nonmortar connectivity", pipeline.surfaces.nonmortar.connectivity.data() ) ||
       !requireDevicePointer( "candidate pairs", pipeline.candidates.data() ) ||
       !requireDevicePointer( "interaction patches", pipeline.patches.data() ) ||
       !requireDevicePointer( "mortar force", device_result.mortar_force.values.data() ) ||
       !requireDevicePointer( "nonmortar force", device_result.nonmortar_force.values.data() ) ||
       !requireDevicePointer( "gap", device_result.gap.data() ) ) {
    return false;
  }

  const auto host_result = host.evaluate();
  const auto first = device.evaluate();
  std::vector<tribol::Real> first_mortar( first.mortar_force.values.begin(), first.mortar_force.values.end() );
  std::vector<tribol::Real> first_nonmortar( first.nonmortar_force.values.begin(), first.nonmortar_force.values.end() );
  const auto second = device.evaluate();
  if ( !arraysMatch( host_result.mortar_force.values, second.mortar_force.values, 2.0e-12 ) ||
       !arraysMatch( host_result.nonmortar_force.values, second.nonmortar_force.values, 2.0e-12 ) ||
       std::memcmp( first_mortar.data(), second.mortar_force.values.data(),
                    first_mortar.size() * sizeof( tribol::Real ) ) != 0 ||
       std::memcmp( first_nonmortar.data(), second.nonmortar_force.values.data(),
                    first_nonmortar.size() * sizeof( tribol::Real ) ) != 0 ) {
    std::cerr << "shared-node result parity or determinism mismatch\n";
    return false;
  }
  if ( device_result.summary.active_interactions != 64 || second.summary.active_interactions != 64 ) {
    std::cerr << "active interaction mismatch: device=" << device_result.summary.active_interactions
              << " repeated=" << second.summary.active_interactions << '\n';
    return false;
  }
  return true;
}

bool suppliedInteractionsGeometryAndAllocationContract()
{
  RepeatedQuadrilateralPair mesh;
  using HostContact = tribol::Contact<tribol::DefaultMethod, tribol::search::Bvh>;
  using DeviceContact = tribol::Contact<tribol::DefaultMethod, tribol::search::Bvh, DeviceExecution>;
  HostContact::Options host_options;
  DeviceContact::Options device_options;
  host_options.method.enforcement.stiffness.value = 4.5;
  device_options.method.enforcement.stiffness.value = host_options.method.enforcement.stiffness.value;
  HostContact host( mesh.view(), host_options );
  DeviceContact device( mesh.view(), device_options );
  const tribol::ArrayView<const tribol::ElementPair> supplied{ mesh.pairs.data(),
                                                               static_cast<tribol::Index>( mesh.pairs.size() ) };
  host.setInteractions( supplied );
  device.setInteractions( supplied );
  if ( host.interactions().size() != RepeatedQuadrilateralPair::elements ||
       device.interactions().size() != RepeatedQuadrilateralPair::elements ) {
    std::cerr << "supplied device candidates were not canonicalized\n";
    return false;
  }

  const auto first_device_result = device.evaluateDevice();
  const auto initial_pipeline = device.devicePipelineView();
  const void* mortar_coordinates = initial_pipeline.surfaces.mortar.coordinates.values.data();
  const void* candidates = initial_pipeline.candidates.data();
  const void* patches = initial_pipeline.patches.data();
  const void* mortar_force = first_device_result.mortar_force.values.data();
  std::size_t free_before{};
  std::size_t total_before{};
  if ( !synchronizeAndGetMemoryInfo( &free_before, &total_before ) ) {
    std::cerr << "could not establish the device allocation baseline\n";
    return false;
  }
  for ( int repeat = 0; repeat < 8; ++repeat ) {
    const auto repeated = device.evaluateDevice();
    if ( repeated.mortar_force.values.data() != mortar_force ) {
      std::cerr << "device result storage moved during evaluation\n";
      return false;
    }
  }
  std::size_t free_after{};
  std::size_t total_after{};
  if ( !synchronizeAndGetMemoryInfo( &free_after, &total_after ) || free_before != free_after ||
       total_before != total_after ) {
    std::cerr << "device evaluation changed the prepared allocation footprint\n";
    return false;
  }

  const auto host_result = host.evaluate();
  const auto downloaded = device.evaluate();
  if ( !arraysMatch( host_result.mortar_force.values, downloaded.mortar_force.values, 5.0e-12 ) ||
       !arraysMatch( host_result.nonmortar_force.values, downloaded.nonmortar_force.values, 5.0e-12 ) ||
       std::abs( host_result.summary.energy - downloaded.summary.energy ) > 5.0e-12 ) {
    std::cerr << "supplied-pair 3D device parity failed\n";
    return false;
  }

  std::array<tribol::Real, 12> mortar_residual{};
  std::array<tribol::Real, 12> nonmortar_residual{};
  for ( std::size_t value = 0; value < mortar_residual.size(); ++value ) {
    mortar_residual[value] = 10.0 + static_cast<tribol::Real>( value );
    nonmortar_residual[value] = -20.0 - static_cast<tribol::Real>( value );
  }
  const auto initial_mortar_residual = mortar_residual;
  const auto initial_nonmortar_residual = nonmortar_residual;
  tribol::FieldView<tribol::Real> mortar_field{
      { mortar_residual.data(), 12 }, 4, 3, tribol::FieldLayout::ComponentMajor };
  tribol::FieldView<tribol::Real> nonmortar_field{
      { nonmortar_residual.data(), 12 }, 4, 3, tribol::FieldLayout::ComponentMajor };
  const auto residual_summary = device.addResidual( {}, { .mortar = mortar_field, .nonmortar = nonmortar_field } );
  if ( residual_summary.active_interactions != RepeatedQuadrilateralPair::elements ) {
    return false;
  }
  for ( tribol::Index node = 0; node < 4; ++node ) {
    for ( int component = 0; component < 3; ++component ) {
      const std::size_t offset = static_cast<std::size_t>( component * 4 + node );
      if ( std::abs( mortar_field( node, component ) -
                     ( initial_mortar_residual[offset] + downloaded.mortar_force( node, component ) ) ) > 5.0e-12 ||
           std::abs( nonmortar_field( node, component ) -
                     ( initial_nonmortar_residual[offset] + downloaded.nonmortar_force( node, component ) ) ) >
               5.0e-12 ) {
        std::cerr << "device addResidual did not preserve component-major accumulation\n";
        return false;
      }
    }
  }

  for ( std::size_t coordinate = 2; coordinate < mesh.nonmortar_coordinates.size(); coordinate += 3 ) {
    mesh.nonmortar_coordinates[coordinate] = -0.05;
  }
  host.updateGeometry( mesh.view() );
  device.updateGeometry( mesh.view() );
  const auto updated_pipeline = device.devicePipelineView();
  if ( updated_pipeline.surfaces.mortar.coordinates.values.data() != mortar_coordinates ||
       updated_pipeline.candidates.data() != candidates || updated_pipeline.patches.data() != patches ||
       device.geometryVersion() == device.interactionGeometryVersion() ||
       updated_pipeline.result.geometry_version == device.geometryVersion() ||
       device.interactions().size() != RepeatedQuadrilateralPair::elements ) {
    std::cerr << "device geometry update did not preserve resident topology and candidates\n";
    return false;
  }
  const auto updated_host = host.evaluate();
  const auto updated_device = device.evaluate();
  return arraysMatch( updated_host.mortar_force.values, updated_device.mortar_force.values, 5.0e-12 ) &&
         arraysMatch( updated_host.nonmortar_force.values, updated_device.nonmortar_force.values, 5.0e-12 ) &&
         updated_device.summary.energy < downloaded.summary.energy;
}

bool timestepVoteParity()
{
  using HostContact = tribol::Contact<tribol::DefaultMethod, tribol::search::Bvh>;
  using DeviceContact = tribol::Contact<tribol::DefaultMethod, tribol::search::Bvh, DeviceExecution>;
  constexpr std::array<tribol::Index, 2> mortar_connectivity{ 1, 0 };
  constexpr std::array<tribol::Index, 2> nonmortar_connectivity{ 0, 1 };
  constexpr std::array<tribol::Real, 4> nonmortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<tribol::Real, 1> thickness{ 0.1 };
  struct VoteCase {
    tribol::Real penetration;
    tribol::Real mortar_speed;
    tribol::Real nonmortar_speed;
    tribol::Real expected;
  };
  constexpr std::array<VoteCase, 3> cases{ VoteCase{ 0.1, 0.3, -0.3, 0.1 }, VoteCase{ 0.01, 0.1, -0.1, 0.3 },
                                           VoteCase{ 0.01, -0.1, 0.1, 1.0 } };
  for ( const VoteCase& vote_case : cases ) {
    const std::array<tribol::Real, 4> mortar_coordinates{ 0.0, vote_case.penetration, 1.0, vote_case.penetration };
    const tribol::SurfacePairView surfaces{
        makeSegment( mortar_coordinates, mortar_connectivity ),
        makeSegment( nonmortar_coordinates, nonmortar_connectivity ),
    };
    HostContact::Options host_options;
    DeviceContact::Options device_options;
    host_options.search.expansion = 0.25;
    host_options.timestep = { .enabled = true, .current_step = 1.0, .penetration_fraction = 0.3 };
    device_options.search = host_options.search;
    device_options.timestep = host_options.timestep;
    HostContact host( surfaces, host_options );
    DeviceContact device( surfaces, device_options );
    host.updateInteractions();
    device.updateInteractions();
    const std::array<tribol::Real, 4> mortar_velocity{ 0.0, vote_case.mortar_speed, 0.0, vote_case.mortar_speed };
    const std::array<tribol::Real, 4> nonmortar_velocity{ 0.0, vote_case.nonmortar_speed, 0.0,
                                                          vote_case.nonmortar_speed };
    const tribol::ContactStateView state{
        .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
        .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
        .mortar_element_thickness = { thickness.data(), 1 },
        .nonmortar_element_thickness = { thickness.data(), 1 },
    };
    const auto host_vote = host.evaluate( state ).summary.timestep_vote;
    const auto device_vote = device.evaluateDevice( state ).summary.timestep_vote;
    if ( std::abs( host_vote - vote_case.expected ) > 1.0e-10 || std::abs( host_vote - device_vote ) > 1.0e-12 ) {
      std::cerr << "device timestep vote mismatch: host=" << host_vote << " device=" << device_vote << '\n';
      return false;
    }
  }

  constexpr std::array<tribol::Real, 4> mortar_coordinates{ 0.0, 0.01, 1.0, 0.01 };
  const tribol::SurfacePairView surfaces{ makeSegment( mortar_coordinates, mortar_connectivity ),
                                          makeSegment( nonmortar_coordinates, nonmortar_connectivity ) };
  DeviceContact::Options device_options;
  device_options.search.expansion = 0.25;
  device_options.timestep.enabled = true;
  device_options.timestep.current_step = 1.0e-10;
  device_options.timestep.minimum_current_step = 1.0e-8;
  DeviceContact device( surfaces, device_options );
  device.updateInteractions();
  return std::abs( device.evaluateDevice().summary.timestep_vote - device_options.timestep.current_step ) < 1.0e-20;
}

bool nativeRowPipelineParity()
{
  tribol::test::TwoRowFixture fixture;
  constexpr std::array<tribol::Real, 2> stiffness{ 10.0, 10.0 };
  constexpr std::array<tribol::Real, 2> rate{ 2.0, 2.0 };
  constexpr std::array<tribol::Real, 4> inverse_mass{ 0.0, 1.0, 0.0, 1.0 };
  const tribol::PenaltyRowInputView penalty{ { stiffness.data(), 2 }, { rate.data(), 2 } };
  const tribol::timestep::DiagonalInverseMassView mass{
      .mortar = { { inverse_mass.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .nonmortar = { { inverse_mass.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
  };
  const auto integrator = tribol::timestep::IntegratorStabilityLimits::centralDifference();

  std::array<tribol::Real, 4> host_mortar{};
  std::array<tribol::Real, 4> host_nonmortar{};
  std::array<tribol::Real, 2> host_effective_gap{};
  std::array<tribol::Real, 2> host_pressure{};
  std::array<tribol::Real, 2> host_row_force{};
  std::array<tribol::Real, 2> host_row_stiffness{};
  std::array<tribol::Real, 2> host_row_rate{};
  const auto host_summary = tribol::evaluateExplicitPenaltyRows(
      fixture.view(), penalty,
      { .mortar = { { host_mortar.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
        .nonmortar = { { host_nonmortar.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved } },
      { .effective_gap = { host_effective_gap.data(), 2 },
        .pressure = { host_pressure.data(), 2 },
        .row_force = { host_row_force.data(), 2 },
        .row_stiffness = { host_row_stiffness.data(), 2 },
        .row_rate_coefficient = { host_row_rate.data(), 2 } } );
  std::array<tribol::Real, 4> host_mortar_row_sums{};
  std::array<tribol::Real, 4> host_nonmortar_row_sums{};
  const auto host_stability = tribol::timestep::explicitPenaltyStabilityVote(
      fixture.view(),
      { .row_stiffness = { host_row_stiffness.data(), 2 },
        .row_rate_coefficient = { host_row_rate.data(), 2 },
        .inverse_mass = mass,
        .integrator = integrator },
      { .mortar_row_sums = { { host_mortar_row_sums.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
        .nonmortar_row_sums = { { host_nonmortar_row_sums.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved } } );

  tribol::execution::DeviceRowWorkspace<DeviceExecution> device;
  device.prepare( fixture.view(), tribol::execution::DeviceRowDofLayout::separate( 2, 2, 2 ) );
  const auto device_summary = device.evaluate( penalty, mass, integrator );
  const auto device_view = device.deviceResult( device_summary );
  if ( !requireDevicePointer( "native rows", device_view.rows.gaps.data() ) ||
       !requireDevicePointer( "native row force", device_view.mortar_force.values.data() ) ||
       !requireDevicePointer( "native row pressure", device_view.row_output.pressure.data() ) ) {
    return false;
  }
  const auto* force_pointer = device_view.mortar_force.values.data();
  const auto* row_pointer = device_view.rows.gaps.data();
  std::array<tribol::Real, 8> downloaded_force{};
  std::array<tribol::Real, 2> downloaded_effective_gap{};
  std::array<tribol::Real, 2> downloaded_pressure{};
  std::array<tribol::Real, 2> downloaded_row_force{};
  std::array<tribol::Real, 2> downloaded_stiffness{};
  std::array<tribol::Real, 2> downloaded_rate{};
  device.download( { downloaded_force.data(), 8 }, { .effective_gap = { downloaded_effective_gap.data(), 2 },
                                                     .pressure = { downloaded_pressure.data(), 2 },
                                                     .row_force = { downloaded_row_force.data(), 2 },
                                                     .row_stiffness = { downloaded_stiffness.data(), 2 },
                                                     .row_rate_coefficient = { downloaded_rate.data(), 2 } } );
  std::array<tribol::Real, 8> repeated_force{};
  const auto repeated_summary = device.evaluate( penalty, mass, integrator );
  device.download( { repeated_force.data(), 8 }, {} );
  const auto repeated_view = device.deviceResult( repeated_summary );
  for ( int value = 0; value < 4; ++value ) {
    if ( std::abs( downloaded_force[value] - host_mortar[value] ) > 1.0e-13 ||
         std::abs( downloaded_force[value + 4] - host_nonmortar[value] ) > 1.0e-13 ) {
      return false;
    }
  }
  return std::memcmp( downloaded_force.data(), repeated_force.data(), sizeof( downloaded_force ) ) == 0 &&
         force_pointer == repeated_view.mortar_force.values.data() && row_pointer == repeated_view.rows.gaps.data() &&
         std::abs( device_summary.explicit_response.energy - host_summary.energy ) < 1.0e-13 &&
         std::abs( device_summary.explicit_response.contact_power - host_summary.contact_power ) < 1.0e-13 &&
         std::abs( device_summary.stability.stiffness_frequency_squared_bound -
                   host_stability.stiffness_frequency_squared_bound ) < 1.0e-13 &&
         std::abs( device_summary.stability.rate_frequency_bound - host_stability.rate_frequency_bound ) < 1.0e-13 &&
         std::abs( device_summary.stability.timestep_vote - host_stability.timestep_vote ) < 1.0e-13;
}

bool deviceNativeRowGenerationParity()
{
  const auto mortar = tribol::test::segmentTrace( 0.0, 1.0, 0.0, false, 101, 0 );
  const auto nonmortar = tribol::test::segmentTrace( 0.0, 1.0, -0.1, true, 202, 0 );
  const tribol::ParentTracePairView traces{ mortar.view(), nonmortar.view() };
  const std::array<tribol::Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  const std::array<tribol::Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  const std::array<tribol::Real, 4> mortar_velocity{ 0.0, -0.2, 0.0, -0.2 };
  const std::array<tribol::Real, 4> nonmortar_velocity{};
  const tribol::ParentTraceFieldPairView fields{
      .mortar_coordinates = { { mortar_coordinates.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .nonmortar_coordinates = { { nonmortar_coordinates.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
  };
  const tribol::ElementPair candidate{ 0, 0 };
  tribol::execution::DeviceContactWorkspace<DeviceExecution> search;
  search.uploadSurfaces( { traces.mortar.geometry, traces.nonmortar.geometry } );
  search.setCandidates( { &candidate, 1 } );

  tribol::execution::DeviceNativeRowOptions options;
  options.quadrature.minimum_exactness = 3;
  options.include_normal_rates = true;
  const auto cells = tribol::buildOverlapCellBatch<tribol::normal::MeanPlane>( traces, { &candidate, 1 } );
  const auto host_rows = tribol::buildConstraintRowBatch(
      traces, cells.view(), fields,
      { .quadrature = options.quadrature, .include_normal_rates = options.include_normal_rates } );
  const auto rows = host_rows.view();
  std::vector<tribol::Real> stiffness( static_cast<std::size_t>( rows.numberOfRows() ), 10.0 );
  std::vector<tribol::Real> damping( static_cast<std::size_t>( rows.numberOfRows() ), 0.25 );
  const tribol::PenaltyRowInputView penalty{
      { stiffness.data(), rows.numberOfRows() }, { damping.data(), rows.numberOfRows() }, {} };
  std::array<tribol::Real, 4> host_mortar{};
  std::array<tribol::Real, 4> host_nonmortar{};
  std::vector<tribol::Real> host_effective_gap( stiffness.size() );
  std::vector<tribol::Real> host_pressure( stiffness.size() );
  std::vector<tribol::Real> host_row_force( stiffness.size() );
  std::vector<tribol::Real> host_row_stiffness( stiffness.size() );
  std::vector<tribol::Real> host_row_rate( stiffness.size() );
  const auto host_summary =
      tribol::evaluateExplicitPenaltyRows( rows, penalty,
                                           { { { host_mortar.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
                                             { { host_nonmortar.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved } },
                                           { { host_effective_gap.data(), rows.numberOfRows() },
                                             { host_pressure.data(), rows.numberOfRows() },
                                             { host_row_force.data(), rows.numberOfRows() },
                                             { host_row_stiffness.data(), rows.numberOfRows() },
                                             { host_row_rate.data(), rows.numberOfRows() } } );
  const std::array<tribol::Real, 4> inverse_mass_values{ 1.0, 1.0, 1.0, 1.0 };
  const tribol::timestep::DiagonalInverseMassView inverse_mass{
      { { inverse_mass_values.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      { { inverse_mass_values.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
  };
  const tribol::timestep::IntegratorStabilityLimits integrator{ 2.0, 1.0, 0.9 };

  tribol::execution::DeviceRowWorkspace<DeviceExecution> device;
  device.prepareNative( traces, fields, search.deviceCandidates(),
                        tribol::execution::DeviceRowDofLayout::separate( 2, 2, 2 ), options );
  const auto device_summary = device.evaluate( penalty, inverse_mass, integrator );
  const auto device_view = device.deviceResult( device_summary );
  std::array<tribol::Real, 8> force{};
  std::vector<tribol::Real> effective_gap( stiffness.size() );
  std::vector<tribol::Real> pressure( stiffness.size() );
  std::vector<tribol::Real> row_force( stiffness.size() );
  device.download( { force.data(), 8 }, { .effective_gap = { effective_gap.data(), rows.numberOfRows() },
                                          .pressure = { pressure.data(), rows.numberOfRows() },
                                          .row_force = { row_force.data(), rows.numberOfRows() } } );
  if ( device_view.rows.numberOfRows() != rows.numberOfRows() || device.devicePatches().size() != 1 ||
       !requireDevicePointer( "generated native rows", device_view.rows.gaps.data() ) ||
       !requireDevicePointer( "native overlap patches", device.devicePatches().data() ) ||
       std::abs( device_summary.explicit_response.energy - host_summary.energy ) > 1.0e-13 ) {
    return false;
  }
  for ( tribol::Index row = 0; row < rows.numberOfRows(); ++row ) {
    if ( std::abs( effective_gap[row] - host_effective_gap[row] ) > 1.0e-13 ||
         std::abs( pressure[row] - host_pressure[row] ) > 1.0e-13 ||
         std::abs( row_force[row] - host_row_force[row] ) > 1.0e-13 ) {
      return false;
    }
  }
  for ( int value = 0; value < 4; ++value ) {
    if ( std::abs( force[value] - host_mortar[value] ) > 1.0e-13 ||
         std::abs( force[value + 4] - host_nonmortar[value] ) > 1.0e-13 ) {
      return false;
    }
  }
  return true;
}

bool publicNativeDeviceContactParity()
{
  const auto mortar = tribol::test::segmentTrace( 0.0, 1.0, 0.0, false, 101, 0 );
  const auto nonmortar = tribol::test::segmentTrace( 0.0, 1.0, -0.1, true, 202, 0 );
  const tribol::ParentTracePairView traces{ mortar.view(), nonmortar.view() };
  const std::array<tribol::Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  const std::array<tribol::Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  const tribol::ParentTraceFieldPairView fields{
      .mortar_coordinates = { { mortar_coordinates.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .nonmortar_coordinates = { { nonmortar_coordinates.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
  };
  using HostContact = tribol::Contact<NativeExplicitMethod, tribol::search::Bvh>;
  using DeviceContact = tribol::Contact<NativeExplicitMethod, tribol::search::Bvh, DeviceExecution>;
  HostContact::Options host_options;
  DeviceContact::Options device_options;
  host_options.search.expansion = 0.2;
  host_options.method.integration.minimum_exactness = 3;
  host_options.method.constraint.activation.gap_tolerance = 1.0;
  host_options.method.enforcement.stiffness.value = 0.01;
  host_options.timestep = {
      .enabled = true, .current_step = 1.0, .penetration_fraction = 0.3, .velocity_tolerance = 0.0 };
  device_options.search = host_options.search;
  device_options.method.integration = host_options.method.integration;
  device_options.method.constraint = host_options.method.constraint;
  device_options.method.enforcement = host_options.method.enforcement;
  device_options.timestep = host_options.timestep;
  HostContact host( traces, fields, host_options );
  DeviceContact device( traces, fields, device_options );
  host.updateInteractions();
  device.updateInteractions();
  const std::array<tribol::Real, 4> inverse_mass_values{ 1.0, 1.0, 1.0, 1.0 };
  const tribol::timestep::DiagonalInverseMassView inverse_mass{
      { { inverse_mass_values.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      { { inverse_mass_values.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
  };
  const auto integrator = tribol::timestep::IntegratorStabilityLimits::centralDifference();
  constexpr std::array<tribol::Real, 4> mortar_closing_velocity{ 0.0, -0.1, 0.0, -0.1 };
  constexpr std::array<tribol::Real, 4> nonmortar_closing_velocity{ 0.0, 0.1, 0.0, 0.1 };
  constexpr std::array<tribol::Real, 1> thickness{ 0.1 };
  const tribol::ContactStateView closing_state{
      .mortar_velocity = { { mortar_closing_velocity.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .nonmortar_velocity = { { nonmortar_closing_velocity.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .mortar_element_thickness = { thickness.data(), 1 },
      .nonmortar_element_thickness = { thickness.data(), 1 },
  };
  const auto host_result = host.evaluate( closing_state, inverse_mass, integrator );
  const auto resident_result = device.evaluateDevice( closing_state, inverse_mass, integrator );
  const auto pipeline = device.devicePipelineView();
  if ( !requireDevicePointer( "public native force", resident_result.mortar_force.values.data() ) ||
       !requireDevicePointer( "public native candidates", pipeline.candidates.data() ) ||
       !requireDevicePointer( "public native patches", pipeline.patches.data() ) ||
       pipeline.result.quadrature_gap.data() != resident_result.quadrature_gap.data() ) {
    std::cerr << "native device residency mismatch\n";
    return false;
  }
  constexpr tribol::diagnostics::QuadratureSamplingOptions diagnostic_options{ .exactness = 8,
                                                                               .include_normal_rates = true };
  const auto host_diagnostics = host.productionDiagnostics( closing_state, 0.25, 0, 1, diagnostic_options );
  const auto device_diagnostics = device.productionDiagnostics( closing_state, 0.25, 0, 1, diagnostic_options );
  const auto device_samples = device.diagnosticQuadrature();
  if ( !diagnosticsMatch( host_diagnostics, device_diagnostics ) || device_samples.numberOfSamples() != 5 ||
       !device_samples.hasNormalRates() ||
       !requireDevicePointer( "diagnostic gaps", device_samples.physical_gaps.data() ) ||
       !requireDevicePointer( "diagnostic rates", device_samples.normal_rates.data() ) ) {
    std::cerr << "native device diagnostics mismatch\n";
    return false;
  }
  std::size_t free_before{};
  std::size_t total_before{};
  if ( !synchronizeAndGetMemoryInfo( &free_before, &total_before ) ) {
    std::cerr << "could not establish native kinematic allocation baseline\n";
    return false;
  }
  for ( int repeat = 0; repeat < 8; ++repeat ) {
    const auto repeated = device.evaluateDevice( closing_state, inverse_mass, integrator );
    const auto repeated_diagnostics = device.productionDiagnostics( closing_state, 0.25, 0, 1, diagnostic_options );
    if ( repeated.mortar_force.values.data() != resident_result.mortar_force.values.data() ||
         std::abs( repeated.summary.timestep_vote - 0.3 ) > 1.0e-13 ||
         !diagnosticsMatch( device_diagnostics, repeated_diagnostics ) ) {
      std::cerr << "repeated native kinematic vote mismatch: " << repeated.summary.timestep_vote << '\n';
      return false;
    }
  }
  std::size_t free_after{};
  std::size_t total_after{};
  if ( !synchronizeAndGetMemoryInfo( &free_after, &total_after ) || free_before != free_after ||
       total_before != total_after ) {
    std::cerr << "native kinematic evaluation changed the prepared allocation footprint\n";
    return false;
  }
  const auto device_result = device.evaluate( closing_state, inverse_mass, integrator );
  if ( host_result.summary.active_interactions == 0 ||
       device_result.summary.quadrature_points != host_result.summary.quadrature_points ||
       device_result.summary.active_interactions != host_result.summary.active_interactions ||
       std::abs( device_result.summary.energy - host_result.summary.energy ) > 1.0e-13 ||
       std::abs( device_result.summary.timestep_vote - host_result.summary.timestep_vote ) > 1.0e-13 ||
       std::abs( device_result.summary.timestep_vote - 0.3 ) > 1.0e-13 ) {
    std::cerr << "native contact summary mismatch: host vote=" << host_result.summary.timestep_vote
              << " device vote=" << device_result.summary.timestep_vote
              << " active=" << host_result.summary.active_interactions << '\n';
    return false;
  }
  for ( tribol::Index entity = 0; entity < host_result.mortar_force.entities; ++entity ) {
    for ( int component = 0; component < host_result.mortar_force.components; ++component ) {
      if ( std::abs( device_result.mortar_force( entity, component ) - host_result.mortar_force( entity, component ) ) >
               1.0e-13 ||
           std::abs( device_result.nonmortar_force( entity, component ) -
                     host_result.nonmortar_force( entity, component ) ) > 1.0e-13 ) {
        std::cerr << "native contact force mismatch\n";
        return false;
      }
    }
  }
  constexpr std::array<tribol::Real, 4> mortar_opening_velocity{ 0.0, 0.1, 0.0, 0.1 };
  constexpr std::array<tribol::Real, 4> nonmortar_opening_velocity{ 0.0, -0.1, 0.0, -0.1 };
  const tribol::ContactStateView opening_state{
      .mortar_velocity = { { mortar_opening_velocity.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .nonmortar_velocity = { { nonmortar_opening_velocity.data(), 4 }, 2, 2, tribol::FieldLayout::Interleaved },
      .mortar_element_thickness = { thickness.data(), 1 },
      .nonmortar_element_thickness = { thickness.data(), 1 },
  };
  const tribol::Real host_opening_vote = host.evaluate( opening_state, inverse_mass, integrator ).summary.timestep_vote;
  const tribol::Real device_opening_vote =
      device.evaluateDevice( opening_state, inverse_mass, integrator ).summary.timestep_vote;
  if ( std::abs( host_opening_vote - 1.0 ) > 1.0e-13 ||
       std::abs( device_opening_vote - host_opening_vote ) > 1.0e-13 ) {
    std::cerr << "native opening vote mismatch: host=" << host_opening_vote << " device=" << device_opening_vote
              << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main()
{
  if ( !deviceAvailable() ) {
    std::cerr << "A device-enabled Tribol build requires a visible device for PAR-001.\n";
    return 1;
  }
  std::array<tribol::InteractionPatch, 4> patches{ makePatch( 0.1, 1.0 ), makePatch( 0.2, 0.5 ), makePatch( -0.1, 2.0 ),
                                                   makePatch( 0.05, 3.0 ) };
  std::array<tribol::execution::PenaltyContribution, 4> sequential{};
  std::array<tribol::execution::PenaltyContribution, 4> device_contributions{};
  tribol::execution::evaluatePenaltyPatches( { patches.data(), 4 }, 12.5, { sequential.data(), 4 },
                                             tribol::execution::Sequential{} );
  tribol::execution::evaluatePenaltyPatches( { patches.data(), 4 }, 12.5, { device_contributions.data(), 4 },
                                             DeviceExecution{} );
  for ( std::size_t patch = 0; patch < patches.size(); ++patch ) {
    if ( std::abs( sequential[patch].energy - device_contributions[patch].energy ) > 1.0e-13 ) {
      return 1;
    }
    for ( int component = 0; component < 3; ++component ) {
      if ( std::abs( sequential[patch].mortar_force[component] - device_contributions[patch].mortar_force[component] ) >
           1.0e-13 ) {
        return 1;
      }
    }
  }
  if ( !fullContactParity() ) {
    std::cerr << "full device contact parity failed\n";
    return 1;
  }
  if ( !activationAndPenetrationParity() ) {
    std::cerr << "device activation/penetration parity failed\n";
    return 1;
  }
  if ( !deviceResidentBvhAndDeterministicScatter() ) {
    std::cerr << "device-resident BVH or deterministic scatter failed\n";
    return 1;
  }
  if ( !suppliedInteractionsGeometryAndAllocationContract() ) {
    std::cerr << "device supplied-pair, geometry-update, or allocation contract failed\n";
    return 1;
  }
  if ( !timestepVoteParity() ) {
    std::cerr << "device timestep-vote parity failed\n";
    return 1;
  }
  if ( !nativeRowPipelineParity() ) {
    std::cerr << "native row device pipeline parity failed\n";
    return 1;
  }
  if ( !deviceNativeRowGenerationParity() ) {
    std::cerr << "device-native row generation parity failed\n";
    return 1;
  }
  if ( !publicNativeDeviceContactParity() ) {
    std::cerr << "public native device contact parity failed\n";
    return 1;
  }
  return 0;
}
