#include "tribol/contact/Contact.hpp"

#if defined( TRIBOL_TEST_USE_HIP )
#include "tribol/execution/HipContact.hpp"
#else
#include "tribol/execution/CudaContact.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

// Requirements: CONS-002, CORE-004, OUTPUT-003, PAR-001, PAR-008, PHYS-001, PHYS-003, PHYS-004

namespace {

#if defined( TRIBOL_TEST_USE_HIP )
using DeviceExecution = tribol::execution::Hip;
bool deviceAvailable() { return tribol::execution::hipDeviceAvailable(); }
#else
using DeviceExecution = tribol::execution::Cuda;
bool deviceAvailable() { return tribol::execution::cudaDeviceAvailable(); }
#endif

using NativePenalty =
    tribol::Method<tribol::geometry::ProjectedOverlap<tribol::normal::MeanPlane>, tribol::integration::Automatic,
                   tribol::constraint::QuadraturePoint, tribol::enforcement::Penalty<>, tribol::response::Frictionless,
                   tribol::formulation::PointwiseTraction, tribol::linearization::Exact>;

bool close( tribol::Real actual, tribol::Real expected, tribol::Real tolerance = 2.0e-11 )
{
  return std::abs( actual - expected ) <= tolerance * std::max( { 1.0, std::abs( actual ), std::abs( expected ) } );
}

std::vector<tribol::Real> lagrangeCoefficients( int order )
{
  const int count = order + 1;
  std::vector<tribol::Real> coefficients( static_cast<std::size_t>( count * count ) );
  for ( int node = 0; node < count; ++node ) {
    std::vector<tribol::Real> polynomial( 1, 1.0 );
    const tribol::Real node_coordinate = static_cast<tribol::Real>( node ) / order;
    for ( int other = 0; other < count; ++other ) {
      if ( other == node ) {
        continue;
      }
      const tribol::Real other_coordinate = static_cast<tribol::Real>( other ) / order;
      const tribol::Real inverse_denominator = 1.0 / ( node_coordinate - other_coordinate );
      std::vector<tribol::Real> next( polynomial.size() + 1 );
      for ( std::size_t power = 0; power < polynomial.size(); ++power ) {
        next[power] -= other_coordinate * inverse_denominator * polynomial[power];
        next[power + 1] += inverse_denominator * polynomial[power];
      }
      polynomial = std::move( next );
    }
    std::copy( polynomial.begin(), polynomial.end(), coefficients.begin() + node * count );
  }
  return coefficients;
}

struct PolynomialTrace {
  int dimension{};
  tribol::ElementTopology topology{};
  std::vector<tribol::Real> geometry_coordinates;
  std::vector<tribol::Index> geometry_offsets;
  std::vector<tribol::Index> geometry_connectivity;
  std::vector<tribol::ElementTopology> geometry_topologies;
  std::vector<int> geometry_attributes;
  std::array<std::uint64_t, 1> face_ids{};
  std::array<int, 1> face_owners{};
  std::array<tribol::ElementTopology, 1> face_topologies{};
  std::array<int, 1> face_orders{};
  std::array<tribol::Index, 2> dof_offsets{};
  std::vector<tribol::Index> dofs;
  std::vector<tribol::Real> dof_signs;
  std::array<tribol::Index, 2> basis_offsets{};
  std::vector<tribol::Real> basis_coefficients;
  std::array<tribol::Index, 1> subfacet_faces{};
  std::vector<tribol::Real> parent_reference_coordinates;
  std::vector<tribol::Real> parent_coordinates;

  PolynomialTrace( int spatial_dimension, int order, tribol::Real height, bool reverse, std::uint64_t face_id )
      : dimension( spatial_dimension ),
        topology( spatial_dimension == 2 ? tribol::ElementTopology::Segment : tribol::ElementTopology::Quadrilateral ),
        geometry_offsets{ 0, spatial_dimension == 2 ? 2 : 4 },
        geometry_topologies{ topology },
        geometry_attributes{ 1 },
        face_ids{ face_id },
        face_topologies{ topology },
        face_orders{ order }
  {
    if ( dimension == 2 ) {
      geometry_coordinates = reverse ? std::vector<tribol::Real>{ 1.0, height, 0.0, height }
                                     : std::vector<tribol::Real>{ 0.0, height, 1.0, height };
      parent_reference_coordinates =
          reverse ? std::vector<tribol::Real>{ 1.0, 0.0 } : std::vector<tribol::Real>{ 0.0, 1.0 };
      geometry_connectivity = { 0, 1 };
    } else {
      geometry_coordinates =
          reverse ? std::vector<tribol::Real>{ 0.0, 0.0, height, 0.0, 1.0, height, 1.0, 1.0, height, 1.0, 0.0, height }
                  : std::vector<tribol::Real>{ 0.0, 0.0, height, 1.0, 0.0, height, 1.0, 1.0, height, 0.0, 1.0, height };
      parent_reference_coordinates = reverse ? std::vector<tribol::Real>{ 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0 }
                                             : std::vector<tribol::Real>{ 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 };
      geometry_connectivity = { 0, 1, 2, 3 };
    }
    const int dof_count = tribol::basis::parentDofCount( topology, order );
    dof_offsets = { 0, dof_count };
    basis_offsets = { 0, dof_count * dof_count };
    dofs.resize( static_cast<std::size_t>( dof_count ) );
    std::iota( dofs.begin(), dofs.end(), tribol::Index{} );
    dof_signs.assign( static_cast<std::size_t>( dof_count ), 1.0 );
    const auto segment_coefficients = lagrangeCoefficients( order );
    if ( dimension == 2 ) {
      basis_coefficients = segment_coefficients;
    } else {
      basis_coefficients.resize( static_cast<std::size_t>( dof_count * dof_count ) );
      const int width = order + 1;
      for ( int second_node = 0; second_node < width; ++second_node ) {
        for ( int first_node = 0; first_node < width; ++first_node ) {
          const int node = second_node * width + first_node;
          for ( int second_power = 0; second_power < width; ++second_power ) {
            for ( int first_power = 0; first_power < width; ++first_power ) {
              const int term = second_power * width + first_power;
              basis_coefficients[static_cast<std::size_t>( node * dof_count + term )] =
                  segment_coefficients[static_cast<std::size_t>( first_node * width + first_power )] *
                  segment_coefficients[static_cast<std::size_t>( second_node * width + second_power )];
            }
          }
        }
      }
    }
    parent_coordinates.resize( static_cast<std::size_t>( dimension * dof_count ) );
    for ( int node = 0; node < dof_count; ++node ) {
      const auto reference = tribol::basis::parentReferenceNode( topology, order, node );
      parent_coordinates[static_cast<std::size_t>( dimension * node )] = reference.first;
      parent_coordinates[static_cast<std::size_t>( dimension * node + 1 )] = dimension == 2 ? height : reference.second;
      if ( dimension == 3 ) {
        parent_coordinates[static_cast<std::size_t>( dimension * node + 2 )] = height;
      }
    }
  }

  [[nodiscard]] tribol::ParentTraceView view() const
  {
    const tribol::Index geometry_nodes = static_cast<tribol::Index>( geometry_connectivity.size() );
    return {
        .geometry =
            {
                .dimension = dimension,
                .coordinates = { { geometry_coordinates.data(),
                                   static_cast<tribol::Index>( geometry_coordinates.size() ) },
                                 geometry_nodes,
                                 dimension,
                                 tribol::FieldLayout::Interleaved },
                .element_offsets = { geometry_offsets.data(), static_cast<tribol::Index>( geometry_offsets.size() ) },
                .connectivity = { geometry_connectivity.data(),
                                  static_cast<tribol::Index>( geometry_connectivity.size() ) },
                .topologies = { geometry_topologies.data(), 1 },
                .attributes = { geometry_attributes.data(), 1 },
            },
        .parent_face_ids = { face_ids.data(), 1 },
        .parent_face_owners = { face_owners.data(), 1 },
        .parent_face_topologies = { face_topologies.data(), 1 },
        .parent_face_orders = { face_orders.data(), 1 },
        .parent_dof_offsets = { dof_offsets.data(), 2 },
        .parent_dofs = { dofs.data(), static_cast<tribol::Index>( dofs.size() ) },
        .parent_dof_signs = { dof_signs.data(), static_cast<tribol::Index>( dof_signs.size() ) },
        .parent_basis_offsets = { basis_offsets.data(), 2 },
        .parent_basis_coefficients = { basis_coefficients.data(),
                                       static_cast<tribol::Index>( basis_coefficients.size() ) },
        .subfacet_parent_faces = { subfacet_faces.data(), 1 },
        .subfacet_parent_reference_coordinates = { { parent_reference_coordinates.data(),
                                                     static_cast<tribol::Index>(
                                                         parent_reference_coordinates.size() ) },
                                                   geometry_nodes,
                                                   dimension - 1,
                                                   tribol::FieldLayout::Interleaved },
    };
  }

  [[nodiscard]] tribol::FieldView<const tribol::Real> field() const
  {
    return { { parent_coordinates.data(), static_cast<tribol::Index>( parent_coordinates.size() ) },
             static_cast<tribol::Index>( dofs.size() ),
             dimension,
             tribol::FieldLayout::Interleaved };
  }
};

bool diagnosticParity( const tribol::diagnostics::ProductionDiagnosticSummary& host,
                       const tribol::diagnostics::ProductionDiagnosticSummary& device )
{
  bool valid = host.rows.rows == device.rows.rows && host.rows.active_rows == device.rows.active_rows &&
               host.quadrature.samples == device.quadrature.samples &&
               close( host.rows.minimum_gap, device.rows.minimum_gap ) &&
               close( host.rows.maximum_traction, device.rows.maximum_traction ) &&
               close( host.rows.spring_energy, device.rows.spring_energy ) &&
               close( host.quadrature.maximum_penetration, device.quadrature.maximum_penetration );
  for ( int component = 0; component < 3; ++component ) {
    valid = valid && close( host.rows.mortar_force[component], device.rows.mortar_force[component] ) &&
            close( host.rows.mortar_moment[component], device.rows.mortar_moment[component] );
  }
  return valid;
}

bool orderContract( int dimension, int order )
{
  const tribol::ElementTopology topology =
      dimension == 2 ? tribol::ElementTopology::Segment : tribol::ElementTopology::Quadrilateral;
  PolynomialTrace mortar( dimension, order, 0.0, false, 101 );
  PolynomialTrace nonmortar( dimension, order, -0.1, true, 202 );
  const tribol::ParentTracePairView traces{ mortar.view(), nonmortar.view() };
  const tribol::ParentTraceFieldPairView fields{ .mortar_coordinates = mortar.field(),
                                                 .nonmortar_coordinates = nonmortar.field() };
  if ( !traces.isStructurallyValid() || traces.mortar.parent_face_topologies[0] != topology ) {
    return false;
  }

  using HostContact = tribol::Contact<NativePenalty, tribol::search::Bvh>;
  using DeviceContact = tribol::Contact<NativePenalty, tribol::search::Bvh, DeviceExecution>;
  HostContact::Options host_options;
  DeviceContact::Options device_options;
  host_options.search.expansion = 0.2;
  host_options.method.enforcement.stiffness.value = 10.0;
  device_options.search = host_options.search;
  device_options.method = host_options.method;
  HostContact host( traces, fields, host_options );
  DeviceContact device( traces, fields, device_options );
  host.updateInteractions();
  device.updateInteractions();

  const int dof_count = tribol::basis::parentDofCount( topology, order );
  std::vector<tribol::Real> inverse_mass_values( static_cast<std::size_t>( dimension * dof_count ), 1.0 );
  const tribol::timestep::DiagonalInverseMassView inverse_mass{
      .mortar = { { inverse_mass_values.data(), static_cast<tribol::Index>( inverse_mass_values.size() ) },
                  dof_count,
                  dimension,
                  tribol::FieldLayout::Interleaved },
      .nonmortar = { { inverse_mass_values.data(), static_cast<tribol::Index>( inverse_mass_values.size() ) },
                     dof_count,
                     dimension,
                     tribol::FieldLayout::Interleaved },
  };
  const auto integrator = tribol::timestep::IntegratorStabilityLimits::centralDifference();
  const auto host_result = host.evaluate( {}, inverse_mass, integrator );
  const auto device_result = device.evaluate( {}, inverse_mass, integrator );
  const auto host_diagnostics = host.productionDiagnostics( {}, 0.25, 0, 1, { .exactness = 8 } );
  const auto device_diagnostics = device.productionDiagnostics( {}, 0.25, 0, 1, { .exactness = 8 } );

  std::array<tribol::Real, 3> resultant{};
  tribol::Real mortar_normal_force{};
  bool valid = host_result.summary.active_interactions > 0 &&
               host_result.summary.quadrature_points == device_result.summary.quadrature_points &&
               close( host_result.summary.energy, 0.025 ) &&
               close( device_result.summary.energy, host_result.summary.energy ) &&
               close( device_result.summary.timestep_vote, host_result.summary.timestep_vote ) &&
               diagnosticParity( host_diagnostics.local, device_diagnostics.local );
  for ( tribol::Index entity = 0; entity < host_result.mortar_force.entities; ++entity ) {
    for ( int component = 0; component < dimension; ++component ) {
      valid =
          valid &&
          close( device_result.mortar_force( entity, component ), host_result.mortar_force( entity, component ) ) &&
          close( device_result.nonmortar_force( entity, component ), host_result.nonmortar_force( entity, component ) );
      resultant[static_cast<std::size_t>( component )] +=
          host_result.mortar_force( entity, component ) + host_result.nonmortar_force( entity, component );
    }
    mortar_normal_force += host_result.mortar_force( entity, dimension - 1 );
  }
  for ( int component = 0; component < dimension; ++component ) {
    valid = valid && close( resultant[static_cast<std::size_t>( component )], 0.0 );
  }
  return valid && close( std::abs( mortar_normal_force ), 0.5 );
}

}  // namespace

int main()
{
  if ( !deviceAvailable() ) {
    std::cerr << "A visible accelerator is required for native Q1-Q4 device conformance.\n";
    return 1;
  }
  for ( int dimension : { 2, 3 } ) {
    for ( int order = 1; order <= tribol::maximumParentTraceOrder; ++order ) {
      if ( !orderContract( dimension, order ) ) {
        std::cerr << dimension << "D native device parent-order contract failed at Q" << order << ".\n";
        return 1;
      }
    }
  }
  return 0;
}
