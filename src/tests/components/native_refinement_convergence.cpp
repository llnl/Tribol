#include "tribol/Tribol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

// Requirements: CONS-002, GEOM-004, PHYS-004

namespace {

using namespace tribol;

struct RefinedSegmentTrace {
  std::vector<Real> coordinates{};
  std::vector<Index> element_offsets{ 0 };
  std::vector<Index> connectivity{};
  std::vector<ElementTopology> geometry_topologies{};
  std::vector<int> geometry_attributes{};
  std::array<std::uint64_t, 1> face_ids{};
  std::array<int, 1> face_owners{};
  std::array<ElementTopology, 1> face_topologies{ ElementTopology::Segment };
  std::array<int, 1> face_orders{ 1 };
  std::array<Index, 2> dof_offsets{ 0, 2 };
  std::array<Index, 2> dofs{ 0, 1 };
  std::array<Real, 2> dof_signs{ 1.0, 1.0 };
  std::array<Index, 2> basis_offsets{ 0, 4 };
  std::array<Real, 4> basis_coefficients{ 1.0, -1.0, 0.0, 1.0 };
  std::vector<Index> subfacet_faces{};
  std::vector<Real> parent_reference_coordinates{};

  RefinedSegmentTrace( int subdivisions, Real height, bool reverse, std::uint64_t face_id )
  {
    face_ids[0] = face_id;
    for ( int subdivision = 0; subdivision < subdivisions; ++subdivision ) {
      const Real first = static_cast<Real>( subdivision ) / subdivisions;
      const Real second = static_cast<Real>( subdivision + 1 ) / subdivisions;
      const Real left = reverse ? second : first;
      const Real right = reverse ? first : second;
      const Index node = static_cast<Index>( coordinates.size() / 2 );
      coordinates.insert( coordinates.end(), { left, height, right, height } );
      connectivity.insert( connectivity.end(), { node, node + 1 } );
      element_offsets.push_back( static_cast<Index>( connectivity.size() ) );
      geometry_topologies.push_back( ElementTopology::Segment );
      geometry_attributes.push_back( 1 );
      subfacet_faces.push_back( 0 );
      parent_reference_coordinates.insert( parent_reference_coordinates.end(), { left, right } );
    }
  }

  [[nodiscard]] ParentTraceView view() const
  {
    const Index nodes = static_cast<Index>( coordinates.size() / 2 );
    const Index subfacets = static_cast<Index>( geometry_topologies.size() );
    return {
        .geometry =
            {
                .dimension = 2,
                .coordinates = { { coordinates.data(), static_cast<Index>( coordinates.size() ) },
                                 nodes,
                                 2,
                                 FieldLayout::Interleaved },
                .element_offsets = { element_offsets.data(), static_cast<Index>( element_offsets.size() ) },
                .connectivity = { connectivity.data(), static_cast<Index>( connectivity.size() ) },
                .topologies = { geometry_topologies.data(), subfacets },
                .attributes = { geometry_attributes.data(), subfacets },
            },
        .parent_face_ids = { face_ids.data(), 1 },
        .parent_face_owners = { face_owners.data(), 1 },
        .parent_face_topologies = { face_topologies.data(), 1 },
        .parent_face_orders = { face_orders.data(), 1 },
        .parent_dof_offsets = { dof_offsets.data(), 2 },
        .parent_dofs = { dofs.data(), 2 },
        .parent_dof_signs = { dof_signs.data(), 2 },
        .parent_basis_offsets = { basis_offsets.data(), 2 },
        .parent_basis_coefficients = { basis_coefficients.data(), 4 },
        .subfacet_parent_faces = { subfacet_faces.data(), subfacets },
        .subfacet_parent_reference_coordinates = { { parent_reference_coordinates.data(),
                                                     static_cast<Index>( parent_reference_coordinates.size() ) },
                                                   nodes,
                                                   1,
                                                   FieldLayout::Interleaved },
    };
  }
};

struct Metrics {
  std::array<Real, 4> mortar_force{};
  std::array<Real, 4> nonmortar_force{};
  diagnostics::ExplicitRowDiagnostics diagnostics{};
  Real measure{};
  Index rows{};
};

Metrics evaluateRefinement( int mortar_subdivisions, int nonmortar_subdivisions, bool reverse_candidates )
{
  const RefinedSegmentTrace mortar( mortar_subdivisions, -0.1, false, 101 );
  const RefinedSegmentTrace nonmortar( nonmortar_subdivisions, 0.0, true, 202 );
  std::vector<ElementPair> candidates;
  for ( int mortar_element = 0; mortar_element < mortar_subdivisions; ++mortar_element ) {
    for ( int nonmortar_element = 0; nonmortar_element < nonmortar_subdivisions; ++nonmortar_element ) {
      candidates.push_back( { mortar_element, nonmortar_element } );
    }
  }
  if ( reverse_candidates ) {
    std::reverse( candidates.begin(), candidates.end() );
  }
  const ParentTracePairView traces{ mortar.view(), nonmortar.view() };
  const auto cells = buildOverlapCellBatch<normal::MeanPlane>(
      traces, { candidates.data(), static_cast<Index>( candidates.size() ) } );
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> mortar_velocity{ 0.0, -0.1, 0.0, -0.1 };
  constexpr std::array<Real, 4> nonmortar_velocity{ 0.0, 0.1, 0.0, 0.1 };
  const ParentTraceFieldPairView fields{
      .mortar_coordinates = { { mortar_coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar_coordinates = { { nonmortar_coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  const auto row_storage = buildConstraintRowBatch( traces, cells.view(), fields );
  const auto rows = row_storage.view();
  std::vector<Real> stiffness( static_cast<std::size_t>( rows.numberOfRows() ), 10.0 );
  std::vector<Real> rate( static_cast<std::size_t>( rows.numberOfRows() ), 2.0 );
  std::vector<Real> effective_gap( static_cast<std::size_t>( rows.numberOfRows() ) );
  std::vector<Real> pressure( static_cast<std::size_t>( rows.numberOfRows() ) );
  std::vector<Real> row_force( static_cast<std::size_t>( rows.numberOfRows() ) );
  std::vector<Real> row_stiffness( static_cast<std::size_t>( rows.numberOfRows() ) );
  std::vector<Real> row_rate( static_cast<std::size_t>( rows.numberOfRows() ) );
  Metrics result;
  const PenaltyRowOutputView output{
      .effective_gap = { effective_gap.data(), rows.numberOfRows() },
      .pressure = { pressure.data(), rows.numberOfRows() },
      .row_force = { row_force.data(), rows.numberOfRows() },
      .row_stiffness = { row_stiffness.data(), rows.numberOfRows() },
      .row_rate_coefficient = { row_rate.data(), rows.numberOfRows() },
  };
  (void)evaluateExplicitPenaltyRows(
      rows,
      { .pressure_stiffness = { stiffness.data(), rows.numberOfRows() },
        .rate_coefficients = { rate.data(), rows.numberOfRows() } },
      { .mortar = { { result.mortar_force.data(), 4 }, 2, 2, FieldLayout::Interleaved },
        .nonmortar = { { result.nonmortar_force.data(), 4 }, 2, 2, FieldLayout::Interleaved } },
      output );
  result.diagnostics = diagnostics::summarizeExplicitRows( rows,
                                                           { .effective_gap = output.effective_gap,
                                                             .pressure = output.pressure,
                                                             .row_force = output.row_force,
                                                             .row_stiffness = output.row_stiffness,
                                                             .row_rate_coefficient = output.row_rate_coefficient },
                                                           0.25, 0 );
  result.rows = rows.numberOfRows();
  for ( Real weight : rows.integration_weights ) {
    result.measure += weight;
  }
  return result;
}

bool close( Real left, Real right ) { return std::abs( left - right ) < 2.0e-12; }

bool analyticContract( const Metrics& result )
{
  return result.rows > 0 && close( result.measure, 1.0 ) && close( result.mortar_force[0], 0.0 ) &&
         close( result.mortar_force[1], -0.7 ) && close( result.mortar_force[2], 0.0 ) &&
         close( result.mortar_force[3], -0.7 ) && close( result.nonmortar_force[0], 0.0 ) &&
         close( result.nonmortar_force[1], 0.7 ) && close( result.nonmortar_force[2], 0.0 ) &&
         close( result.nonmortar_force[3], 0.7 ) && close( result.diagnostics.minimum_gap, -0.1 ) &&
         close( result.diagnostics.maximum_penetration, 0.1 ) &&
         close( result.diagnostics.maximum_closing_rate, 0.2 ) && close( result.diagnostics.mortar_force[1], -1.4 ) &&
         close( result.diagnostics.mortar_moment[2], -0.7 ) && close( result.diagnostics.maximum_traction, 1.4 ) &&
         close( result.diagnostics.spring_energy, 0.05 ) && close( result.diagnostics.dashpot_dissipation, 0.02 ) &&
         close( result.diagnostics.contact_work, 0.07 );
}

bool samePhysics( const Metrics& left, const Metrics& right )
{
  for ( std::size_t value = 0; value < left.mortar_force.size(); ++value ) {
    if ( !close( left.mortar_force[value], right.mortar_force[value] ) ||
         !close( left.nonmortar_force[value], right.nonmortar_force[value] ) ) {
      return false;
    }
  }
  return close( left.measure, right.measure ) &&
         close( left.diagnostics.mortar_moment[2], right.diagnostics.mortar_moment[2] ) &&
         close( left.diagnostics.spring_energy, right.diagnostics.spring_energy ) &&
         close( left.diagnostics.dashpot_dissipation, right.diagnostics.dashpot_dissipation ) &&
         close( left.diagnostics.contact_work, right.diagnostics.contact_work );
}

void printMetrics( const char* label, const Metrics& result )
{
  std::cerr << label << " rows=" << result.rows << " measure=" << result.measure << " force=(" << result.mortar_force[0]
            << ',' << result.mortar_force[1] << ") moment=" << result.diagnostics.mortar_moment[2]
            << " gap=" << result.diagnostics.minimum_gap << " rate=" << result.diagnostics.maximum_closing_rate
            << " traction=" << result.diagnostics.maximum_traction << " spring=" << result.diagnostics.spring_energy
            << " dissipation=" << result.diagnostics.dashpot_dissipation << " work=" << result.diagnostics.contact_work
            << '\n';
}

}  // namespace

int main()
{
  const Metrics reference = evaluateRefinement( 1, 1, false );
  bool valid = analyticContract( reference );
  if ( !valid ) {
    printMetrics( "reference", reference );
  }
  for ( const auto refinement : { std::array<int, 2>{ 2, 3 }, { 3, 5 }, { 8, 5 } } ) {
    const Metrics ordered = evaluateRefinement( refinement[0], refinement[1], false );
    const Metrics reversed = evaluateRefinement( refinement[0], refinement[1], true );
    const bool refinement_valid =
        analyticContract( ordered ) && samePhysics( reference, ordered ) && samePhysics( ordered, reversed );
    if ( !refinement_valid ) {
      printMetrics( "ordered", ordered );
      printMetrics( "reversed", reversed );
    }
    valid = valid && refinement_valid;
  }
  if ( !valid ) {
    std::cerr << "nonmatching native refinement changed parent-space contact physics\n";
  }
  return valid ? 0 : 1;
}
