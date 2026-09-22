#include "tribol/execution/DevicePenalty.hpp"
#include "tribol/contact/Contact.hpp"

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

// Requirements: API-003, CORE-003, DIFF-001, DIFF-002, PAR-001, PAR-006, SEARCH-001, SEARCH-003, TIME-001

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
  return 0;
}
