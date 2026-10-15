#include "tribol/adapters/array/ArrayContact.hpp"

#include <array>
#include <cmath>

// Requirements: ADAPTER-001, API-001, API-003, API-005, CORE-004, PHYS-001, TIME-002

namespace {

using namespace tribol;

using NativeExplicitMethod =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Automatic, constraint::QuadraturePoint,
           enforcement::Penalty<>, response::Frictionless, formulation::PointwiseTraction, linearization::Exact>;

bool close( Real left, Real right ) { return std::abs( left - right ) < 1.0e-12; }

bool defaultContactContract()
{
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  constexpr std::array<Index, 2> connectivity{ 0, 1 };
  Contact<>::Options options;
  options.search.expansion = 0.2;
  options.method.enforcement.stiffness.value = 10.0;
  auto contact = array::makeContact(
      array::SingleTopologySurface( 2, ElementTopology::Segment,
                                    { mortar_coordinates.data(), static_cast<Index>( mortar_coordinates.size() ) },
                                    { connectivity.data(), static_cast<Index>( connectivity.size() ) } ),
      array::SingleTopologySurface(
          2, ElementTopology::Segment,
          { nonmortar_coordinates.data(), static_cast<Index>( nonmortar_coordinates.size() ) },
          { connectivity.data(), static_cast<Index>( connectivity.size() ) } ),
      options );
  contact.updateInteractions();
  const auto result = contact.evaluate();
  Real total{};
  for ( Index node = 0; node < result.mortar_force.entities; ++node ) {
    total += result.mortar_force( node, 1 ) + result.nonmortar_force( node, 1 );
  }
  return result.summary.active_interactions == 1 && close( result.summary.energy, 0.025 ) && close( total, 0.0 );
}

bool identityParentTraceContract()
{
  constexpr std::array<Real, 12> coordinates{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Index, 4> connectivity{ 0, 1, 2, 3 };
  const array::SingleTopologySurface surface( 3, ElementTopology::Quadrilateral,
                                              { coordinates.data(), static_cast<Index>( coordinates.size() ) },
                                              { connectivity.data(), static_cast<Index>( connectivity.size() ) } );
  const auto trace = surface.parentTraceView();
  constexpr std::array<Real, 8> expected_reference{ 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 };
  if ( !trace.isStructurallyValid() || trace.geometry.coordinates.values.data() != coordinates.data() ||
       trace.geometry.connectivity.data() != connectivity.data() || trace.numberOfParentFaces() != 1 ||
       trace.numberOfSubfacets() != 1 || trace.parent_face_ids[0] != 0 || trace.parent_face_owners[0] != 0 ||
       trace.parent_face_orders[0] != 1 || trace.parent_dofs.data() != connectivity.data() ) {
    return false;
  }
  for ( Index dof = 0; dof < 4; ++dof ) {
    if ( trace.parent_dofs[dof] != connectivity[dof] || !close( trace.parent_dof_signs[dof], 1.0 ) ) {
      return false;
    }
  }
  for ( Index local_node = 0; local_node < 4; ++local_node ) {
    const basis::ReferencePoint reference{ trace.subfacet_parent_reference_coordinates( local_node, 0 ),
                                           trace.subfacet_parent_reference_coordinates( local_node, 1 ) };
    if ( !close( reference.first, expected_reference[static_cast<std::size_t>( 2 * local_node )] ) ||
         !close( reference.second, expected_reference[static_cast<std::size_t>( 2 * local_node + 1 )] ) ) {
      return false;
    }
    const auto point = basis::interpolateParentField( trace, 0, trace.geometry.coordinates, reference );
    for ( int component = 0; component < 3; ++component ) {
      if ( !close( point[component], trace.geometry.coordinates( connectivity[local_node], component ) ) ) {
        return false;
      }
    }
  }
  constexpr std::array<Real, 6> segment_coordinates{ 0.0, 0.0, 1.0, 0.0, 2.0, 0.0 };
  constexpr std::array<Index, 4> segment_connectivity{ 0, 1, 1, 2 };
  const array::SingleTopologySurface segments(
      2, ElementTopology::Segment, { segment_coordinates.data(), static_cast<Index>( segment_coordinates.size() ) },
      { segment_connectivity.data(), static_cast<Index>( segment_connectivity.size() ) } );
  const auto segment_trace = segments.parentTraceView();
  for ( Index subfacet = 0; subfacet < 2; ++subfacet ) {
    const Index begin = segment_trace.geometry.element_offsets[subfacet];
    for ( Index local_node = 0; local_node < 2; ++local_node ) {
      std::array<Real, 3> point{};
      const Index node = segment_trace.geometry.connectivity[begin + local_node];
      for ( int component = 0; component < 2; ++component ) {
        point[component] = segment_trace.geometry.coordinates( node, component );
      }
      const auto mapped = overlap_cell_detail::mapToParent( segment_trace, subfacet, point, 1.0e-12 );
      if ( mapped.status != GeometryStatus::Accepted || !close( mapped.parent_reference.first, local_node ) ) {
        return false;
      }
    }
  }
  return true;
}

bool nativeContactContract()
{
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, 0.1, 1.0, 0.1 };
  constexpr std::array<Real, 4> updated_nonmortar_coordinates{ 0.0, 0.2, 1.0, 0.2 };
  constexpr std::array<Index, 2> mortar_connectivity{ 0, 1 };
  constexpr std::array<Index, 2> nonmortar_connectivity{ 1, 0 };
  Contact<NativeExplicitMethod>::Options options;
  options.search.expansion = 0.25;
  options.method.enforcement.stiffness.value = 10.0;
  auto contact = array::makeContact<NativeExplicitMethod>(
      array::SingleTopologySurface( 2, ElementTopology::Segment,
                                    { mortar_coordinates.data(), static_cast<Index>( mortar_coordinates.size() ) },
                                    { mortar_connectivity.data(), static_cast<Index>( mortar_connectivity.size() ) } ),
      array::SingleTopologySurface(
          2, ElementTopology::Segment,
          { nonmortar_coordinates.data(), static_cast<Index>( nonmortar_coordinates.size() ) },
          { nonmortar_connectivity.data(), static_cast<Index>( nonmortar_connectivity.size() ) } ),
      options );
  contact.updateInteractions();

  constexpr std::array<Real, 4> inverse_mass_values{ 1.0, 1.0, 1.0, 1.0 };
  const timestep::DiagonalInverseMassView inverse_mass{
      .mortar = { { inverse_mass_values.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = { { inverse_mass_values.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  const auto result = contact.evaluate( {}, inverse_mass, timestep::IntegratorStabilityLimits::centralDifference() );
  const auto rows = contact.core().constraintRows();
  Real total{};
  for ( Index node = 0; node < result.mortar_force.entities; ++node ) {
    total += result.mortar_force( node, 1 ) + result.nonmortar_force( node, 1 );
  }
  if ( !rows.isStructurallyValid() || rows.numberOfRows() != 2 || result.summary.active_interactions != 2 ||
       result.summary.quadrature_points != 2 || !close( result.summary.energy, 0.025 ) || !close( total, 0.0 ) ||
       !std::isfinite( result.summary.timestep_vote ) || result.summary.timestep_vote <= 0.0 ) {
    return false;
  }

  contact.updateGeometry(
      { mortar_coordinates.data(), static_cast<Index>( mortar_coordinates.size() ) },
      { updated_nonmortar_coordinates.data(), static_cast<Index>( updated_nonmortar_coordinates.size() ) } );
  const auto updated = contact.evaluate( {}, inverse_mass, timestep::IntegratorStabilityLimits::centralDifference() );
  return updated.summary.active_interactions == 2 && updated.summary.quadrature_points == 2 &&
         close( updated.summary.energy, 0.1 );
}

bool nativeKinematicVoteContract()
{
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, 0.01, 1.0, 0.01 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Index, 2> mortar_connectivity{ 1, 0 };
  constexpr std::array<Index, 2> nonmortar_connectivity{ 0, 1 };
  Contact<NativeExplicitMethod>::Options options;
  options.search.expansion = 0.25;
  options.method.enforcement.stiffness.value = 0.01;
  options.timestep = { .enabled = true, .current_step = 1.0, .penetration_fraction = 0.3, .velocity_tolerance = 0.0 };
  auto contact = array::makeContact<NativeExplicitMethod>(
      array::SingleTopologySurface( 2, ElementTopology::Segment,
                                    { mortar_coordinates.data(), static_cast<Index>( mortar_coordinates.size() ) },
                                    { mortar_connectivity.data(), static_cast<Index>( mortar_connectivity.size() ) } ),
      array::SingleTopologySurface(
          2, ElementTopology::Segment,
          { nonmortar_coordinates.data(), static_cast<Index>( nonmortar_coordinates.size() ) },
          { nonmortar_connectivity.data(), static_cast<Index>( nonmortar_connectivity.size() ) } ),
      options );
  contact.updateInteractions();
  constexpr std::array<Real, 4> mortar_velocity{ 0.0, 0.1, 0.0, 0.1 };
  constexpr std::array<Real, 4> nonmortar_velocity{ 0.0, -0.1, 0.0, -0.1 };
  constexpr std::array<Real, 1> thickness{ 0.1 };
  constexpr std::array<Real, 4> inverse_mass_values{ 1.0, 1.0, 1.0, 1.0 };
  const ContactStateView state{
      .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .mortar_element_thickness = { thickness.data(), 1 },
      .nonmortar_element_thickness = { thickness.data(), 1 },
  };
  const timestep::DiagonalInverseMassView inverse_mass{
      .mortar = { { inverse_mass_values.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar = { { inverse_mass_values.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  const auto result = contact.evaluate( state, inverse_mass, timestep::IntegratorStabilityLimits::centralDifference() );
  return close( result.summary.timestep_vote, 0.3 );
}

}  // namespace

int main()
{
  return defaultContactContract() && identityParentTraceContract() && nativeContactContract() &&
                 nativeKinematicVoteContract()
             ? 0
             : 1;
}
