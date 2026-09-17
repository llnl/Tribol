#include "tribol/execution/CudaPenalty.hpp"
#include "tribol/contact/Contact.hpp"

#include <array>
#include <cmath>
#include <iostream>

// Requirements: DIFF-001, DIFF-002, PAR-001

namespace {

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
  constexpr std::array<tribol::Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<tribol::Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  const tribol::SurfacePairView surfaces{ makeSegment( mortar_coordinates ), makeSegment( nonmortar_coordinates ) };
  using HostContact = tribol::Contact<>;
  using DeviceContact =
      tribol::Contact<tribol::DefaultMethod, tribol::search::CartesianProduct, tribol::execution::Cuda>;
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
  std::array<tribol::execution::PenaltyContribution, 1> cuda{};
  tribol::constraint::GapActivationParameters activation;
  activation.residual_gap = 0.005;
  activation.gap_tolerance = 0.01;
  sequential[0] = tribol::execution::evaluatePointwisePenaltyPatch( patches[0], 12.5, activation );
  tribol::execution::evaluatePointwisePenaltyPatches( { patches.data(), 1 }, 12.5, { cuda.data(), 1 },
                                                      tribol::execution::Cuda{}, activation );
  if ( sequential != cuda || std::abs( cuda[0].effective_gap - 0.005 ) > 1.0e-13 || cuda[0].pressure <= 0.0 ) {
    return false;
  }

  constexpr std::array<tribol::Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<tribol::Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  const tribol::SurfacePairView surfaces{ makeSegment( mortar_coordinates ), makeSegment( nonmortar_coordinates ) };
  using HostContact = tribol::Contact<>;
  using DeviceContact =
      tribol::Contact<tribol::DefaultMethod, tribol::search::CartesianProduct, tribol::execution::Cuda>;
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

}  // namespace

int main()
{
  if ( !tribol::execution::cudaDeviceAvailable() ) {
    std::cerr << "A CUDA-enabled Tribol build requires a visible device for PAR-001.\n";
    return 1;
  }
  std::array<tribol::InteractionPatch, 4> patches{ makePatch( 0.1, 1.0 ), makePatch( 0.2, 0.5 ), makePatch( -0.1, 2.0 ),
                                                   makePatch( 0.05, 3.0 ) };
  std::array<tribol::execution::PenaltyContribution, 4> sequential{};
  std::array<tribol::execution::PenaltyContribution, 4> cuda{};
  tribol::execution::evaluatePenaltyPatches( { patches.data(), 4 }, 12.5, { sequential.data(), 4 },
                                             tribol::execution::Sequential{} );
  tribol::execution::evaluatePenaltyPatches( { patches.data(), 4 }, 12.5, { cuda.data(), 4 },
                                             tribol::execution::Cuda{} );
  for ( std::size_t patch = 0; patch < patches.size(); ++patch ) {
    if ( std::abs( sequential[patch].energy - cuda[patch].energy ) > 1.0e-13 ) {
      return 1;
    }
    for ( int component = 0; component < 3; ++component ) {
      if ( std::abs( sequential[patch].mortar_force[component] - cuda[patch].mortar_force[component] ) > 1.0e-13 ) {
        return 1;
      }
    }
  }
  return fullContactParity() && activationAndPenetrationParity() ? 0 : 1;
}
