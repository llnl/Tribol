#include "tribol/Tribol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

// Requirements: CONS-002, GEOM-004

namespace {

using namespace tribol;

struct SegmentTraceStorage {
  std::vector<Real> coordinates{};
  std::vector<Index> element_offsets{};
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

  ParentTraceView view() const
  {
    const Index nodes = static_cast<Index>( coordinates.size() / 2 );
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
                .topologies = { geometry_topologies.data(), static_cast<Index>( geometry_topologies.size() ) },
                .attributes = { geometry_attributes.data(), static_cast<Index>( geometry_attributes.size() ) },
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
        .subfacet_parent_faces = { subfacet_faces.data(), static_cast<Index>( subfacet_faces.size() ) },
        .subfacet_parent_reference_coordinates = { { parent_reference_coordinates.data(),
                                                     static_cast<Index>( parent_reference_coordinates.size() ) },
                                                   nodes,
                                                   1,
                                                   FieldLayout::Interleaved },
    };
  }
};

SegmentTraceStorage segmentTrace( Real first, Real second, Real height, bool reverse, std::uint64_t id, int owner,
                                  int copies = 1 )
{
  SegmentTraceStorage storage;
  storage.face_ids[0] = id;
  storage.face_owners[0] = owner;
  for ( int copy = 0; copy < copies; ++copy ) {
    const Index node = static_cast<Index>( storage.coordinates.size() / 2 );
    storage.coordinates.insert( storage.coordinates.end(),
                                reverse ? std::initializer_list<Real>{ second, height, first, height }
                                        : std::initializer_list<Real>{ first, height, second, height } );
    storage.connectivity.insert( storage.connectivity.end(), { node, node + 1 } );
    storage.element_offsets.push_back( 2 * copy );
    storage.geometry_topologies.push_back( ElementTopology::Segment );
    storage.geometry_attributes.push_back( 1 );
    storage.subfacet_faces.push_back( 0 );
    storage.parent_reference_coordinates.insert(
        storage.parent_reference_coordinates.end(),
        reverse ? std::initializer_list<Real>{ 1.0, 0.0 } : std::initializer_list<Real>{ 0.0, 1.0 } );
  }
  storage.element_offsets.push_back( 2 * copies );
  return storage;
}

struct QuadrilateralTraceStorage {
  std::array<Real, 12> coordinates{};
  std::array<Index, 2> element_offsets{ 0, 4 };
  std::array<Index, 4> connectivity{ 0, 1, 2, 3 };
  std::array<ElementTopology, 1> geometry_topologies{ ElementTopology::Quadrilateral };
  std::array<int, 1> geometry_attributes{ 1 };
  std::array<std::uint64_t, 1> face_ids{};
  std::array<int, 1> face_owners{};
  std::array<ElementTopology, 1> face_topologies{ ElementTopology::Quadrilateral };
  std::array<int, 1> face_orders{ 1 };
  std::array<Index, 2> dof_offsets{ 0, 4 };
  std::array<Index, 4> dofs{ 0, 1, 2, 3 };
  std::array<Real, 4> dof_signs{ 1.0, 1.0, 1.0, 1.0 };
  std::array<Index, 2> basis_offsets{ 0, 16 };
  std::array<Real, 16> basis_coefficients{ 1.0, -1.0, -1.0, 1.0,  0.0, 1.0, 0.0, -1.0,
                                           0.0, 0.0,  1.0,  -1.0, 0.0, 0.0, 0.0, 1.0 };
  std::array<Index, 1> subfacet_faces{ 0 };
  std::array<Real, 8> parent_reference_coordinates{};

  ParentTraceView view() const
  {
    return {
        .geometry =
            {
                .dimension = 3,
                .coordinates = { { coordinates.data(), 12 }, 4, 3, FieldLayout::Interleaved },
                .element_offsets = { element_offsets.data(), 2 },
                .connectivity = { connectivity.data(), 4 },
                .topologies = { geometry_topologies.data(), 1 },
                .attributes = { geometry_attributes.data(), 1 },
            },
        .parent_face_ids = { face_ids.data(), 1 },
        .parent_face_owners = { face_owners.data(), 1 },
        .parent_face_topologies = { face_topologies.data(), 1 },
        .parent_face_orders = { face_orders.data(), 1 },
        .parent_dof_offsets = { dof_offsets.data(), 2 },
        .parent_dofs = { dofs.data(), 4 },
        .parent_dof_signs = { dof_signs.data(), 4 },
        .parent_basis_offsets = { basis_offsets.data(), 2 },
        .parent_basis_coefficients = { basis_coefficients.data(), 16 },
        .subfacet_parent_faces = { subfacet_faces.data(), 1 },
        .subfacet_parent_reference_coordinates = { { parent_reference_coordinates.data(), 8 },
                                                   4,
                                                   2,
                                                   FieldLayout::Interleaved },
    };
  }
};

QuadrilateralTraceStorage quadrilateralTrace( Real height, bool reverse, Real warp, std::uint64_t id, int owner )
{
  QuadrilateralTraceStorage storage;
  storage.face_ids[0] = id;
  storage.face_owners[0] = owner;
  if ( reverse ) {
    storage.coordinates = { 0.0, 0.0, height, 0.0, 1.0, height, 1.0, 1.0, height + warp, 1.0, 0.0, height };
    storage.parent_reference_coordinates = { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0 };
  } else {
    storage.coordinates = { 0.0, 0.0, height, 1.0, 0.0, height, 1.0, 1.0, height + warp, 0.0, 1.0, height };
    storage.parent_reference_coordinates = { 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 };
  }
  return storage;
}

bool acceptedProvenanceContract()
{
  const auto mortar = segmentTrace( 0.0, 1.0, 0.0, false, 101, 3 );
  const auto nonmortar = segmentTrace( 0.25, 1.25, -0.1, true, 202, 7 );
  const std::array<ElementPair, 1> candidates{ ElementPair{ 0, 0 } };
  const auto storage =
      buildOverlapCellBatch<normal::MeanPlane>( { mortar.view(), nonmortar.view() }, { candidates.data(), 1 }, {},
                                                { .contact_pair_id = 9, .first_cell_id = 41 } );
  const auto cells = storage.view();
  if ( !cells.isStructurallyValid() || cells.numberOfAttempts() != 1 || cells.numberOfVertices() != 2 ||
       !cells.isAccepted( 0 ) || cells.contact_pair_ids[0] != 9 || cells.cell_ids[0] != 41 ||
       cells.owner_ranks[0] != 3 || cells.mortar_parent_faces[0] != 0 || cells.nonmortar_parent_faces[0] != 0 ||
       std::abs( cells.measures[0] - 0.75 ) > 1.0e-12 ) {
    return false;
  }
  for ( Index vertex = 0; vertex < cells.numberOfVertices(); ++vertex ) {
    const Real mortar_reference = cells.mortar_parent_reference_coordinates( vertex, 0 );
    const Real nonmortar_reference = cells.nonmortar_parent_reference_coordinates( vertex, 0 );
    if ( mortar_reference < 0.25 - 1.0e-12 || mortar_reference > 1.0 + 1.0e-12 || nonmortar_reference < -1.0e-12 ||
         nonmortar_reference > 0.75 + 1.0e-12 ) {
      return false;
    }
  }
  return true;
}

bool explicitStatusContract()
{
  const auto mortar = segmentTrace( 0.0, 1.0, 0.0, false, 101, 0 );
  const auto separated = segmentTrace( 2.0, 3.0, -0.1, true, 202, 1 );
  const auto same_orientation = segmentTrace( 0.0, 1.0, -0.1, false, 202, 1 );
  auto degenerate = segmentTrace( 0.0, 0.0, 0.0, false, 303, 0 );
  const std::array<ElementPair, 1> candidate{ ElementPair{ 0, 0 } };
  const std::array<ElementPair, 1> invalid_candidate{ ElementPair{ 1, 0 } };
  const auto no_overlap_storage =
      buildOverlapCellBatch<normal::MeanPlane>( { mortar.view(), separated.view() }, { candidate.data(), 1 } );
  const auto orientation_storage =
      buildOverlapCellBatch<normal::MeanPlane>( { mortar.view(), same_orientation.view() }, { candidate.data(), 1 } );
  const auto degenerate_storage =
      buildOverlapCellBatch<normal::MeanPlane>( { degenerate.view(), separated.view() }, { candidate.data(), 1 } );
  const auto invalid_storage =
      buildOverlapCellBatch<normal::MeanPlane>( { mortar.view(), separated.view() }, { invalid_candidate.data(), 1 } );
  const auto no_overlap = no_overlap_storage.view();
  const auto orientation = orientation_storage.view();
  const auto degenerate_result = degenerate_storage.view();
  const auto invalid = invalid_storage.view();
  return no_overlap.statuses[0] == GeometryStatus::NoOverlap &&
         orientation.statuses[0] == GeometryStatus::InconsistentOrientation &&
         degenerate_result.statuses[0] == GeometryStatus::DegenerateOverlap &&
         invalid.statuses[0] == GeometryStatus::InvalidProvenance;
}

bool duplicateAndMappingContract()
{
  const auto mortar = segmentTrace( 0.0, 1.0, 0.0, false, 101, 0, 2 );
  const auto nonmortar = segmentTrace( 0.0, 1.0, -0.1, true, 202, 1, 2 );
  const std::array<ElementPair, 2> candidates{ ElementPair{ 1, 1 }, ElementPair{ 0, 0 } };
  const auto duplicate_storage =
      buildOverlapCellBatch<normal::MeanPlane>( { mortar.view(), nonmortar.view() }, { candidates.data(), 2 } );
  const auto duplicates = duplicate_storage.view();
  if ( duplicate_storage.numberOfAcceptedCells() != 1 || duplicates.statuses[0] != GeometryStatus::Accepted ||
       duplicates.statuses[1] != GeometryStatus::Duplicate || duplicates.cell_ids[0] != 0 ||
       duplicates.cell_ids[1] != 1 ) {
    return false;
  }

  auto invalid_reference = segmentTrace( 0.0, 1.0, 0.0, false, 101, 0 );
  invalid_reference.parent_reference_coordinates[0] = -0.25;
  const std::array<ElementPair, 1> candidate{ ElementPair{ 0, 0 } };
  const auto invalid_storage = buildOverlapCellBatch<normal::MeanPlane>( { invalid_reference.view(), nonmortar.view() },
                                                                         { candidate.data(), 1 } );
  return invalid_storage.view().statuses[0] == GeometryStatus::InvalidMapping;
}

bool threeDimensionalMappingContract()
{
  const auto mortar = quadrilateralTrace( 0.0, false, 0.0, 501, 2 );
  const auto nonmortar = quadrilateralTrace( -0.1, true, 0.0, 502, 4 );
  const std::array<ElementPair, 1> candidate{ ElementPair{ 0, 0 } };
  const auto storage =
      buildOverlapCellBatch<normal::MeanPlane>( { mortar.view(), nonmortar.view() }, { candidate.data(), 1 } );
  const auto cells = storage.view();
  if ( !cells.isStructurallyValid() || !cells.isAccepted( 0 ) || cells.numberOfVertices() != 4 ||
       std::abs( cells.measures[0] - 1.0 ) > 1.0e-12 ) {
    return false;
  }
  for ( Index vertex = 0; vertex < cells.numberOfVertices(); ++vertex ) {
    for ( int component = 0; component < 2; ++component ) {
      const Real mortar_reference = cells.mortar_parent_reference_coordinates( vertex, component );
      const Real nonmortar_reference = cells.nonmortar_parent_reference_coordinates( vertex, component );
      if ( mortar_reference < -1.0e-12 || mortar_reference > 1.0 + 1.0e-12 || nonmortar_reference < -1.0e-12 ||
           nonmortar_reference > 1.0 + 1.0e-12 ) {
        return false;
      }
    }
  }

  const auto warped_mortar = quadrilateralTrace( 0.0, false, 0.25, 601, 2 );
  const auto warped_nonmortar = quadrilateralTrace( -0.1, true, -0.25, 602, 4 );
  const auto warped_storage = buildOverlapCellBatch<normal::MeanPlane>(
      { warped_mortar.view(), warped_nonmortar.view() }, { candidate.data(), 1 } );
  return warped_storage.view().statuses[0] == GeometryStatus::ExcessiveMappingError;
}

bool variableRowContract()
{
  const auto mortar = segmentTrace( 0.0, 1.0, 0.0, false, 101, 3 );
  const auto nonmortar = segmentTrace( 0.25, 1.25, -0.1, true, 202, 7 );
  const std::array<ElementPair, 2> candidates{ ElementPair{ 1, 0 }, ElementPair{ 0, 0 } };
  const auto cell_storage = buildOverlapCellBatch<normal::MeanPlane>(
      { mortar.view(), nonmortar.view() }, { candidates.data(), 2 }, {}, { .contact_pair_id = 9 } );
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.25, -0.1, 1.25, -0.1 };
  constexpr std::array<Real, 4> mortar_velocity{ 0.0, 1.0, 0.0, 1.0 };
  constexpr std::array<Real, 4> nonmortar_velocity{};
  const ParentTraceFieldPairView fields{
      .mortar_coordinates = { { mortar_coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar_coordinates = { { nonmortar_coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .mortar_velocity = { { mortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar_velocity = { { nonmortar_velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  const auto row_storage = buildConstraintRowBatch( { mortar.view(), nonmortar.view() }, cell_storage.view(), fields,
                                                    { .first_row_id = 30 } );
  const auto rows = row_storage.view();
  if ( !rows.isStructurallyValid() || rows.numberOfCells() != 2 || rows.numberOfRows() != 2 ||
       rows.cell_row_offsets[0] != 0 || rows.cell_row_offsets[1] != 2 || rows.cell_row_offsets[2] != 2 ||
       rows.row_ids[0] != 30 || rows.row_ids[1] != 31 || !rows.hasNormalRates() ) {
    return false;
  }
  Real weight{};
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    weight += rows.integration_weights[row];
    Real mortar_partition{};
    for ( Index entry = rows.mortar_dof_offsets[row]; entry < rows.mortar_dof_offsets[row + 1]; ++entry ) {
      mortar_partition += rows.mortar_basis_values[entry];
    }
    Real nonmortar_partition{};
    for ( Index entry = rows.nonmortar_dof_offsets[row]; entry < rows.nonmortar_dof_offsets[row + 1]; ++entry ) {
      nonmortar_partition += rows.nonmortar_basis_values[entry];
    }
    if ( std::abs( mortar_partition - 1.0 ) > 1.0e-12 || std::abs( nonmortar_partition - 1.0 ) > 1.0e-12 ||
         std::abs( rows.gaps[row] - 0.1 ) > 1.0e-12 || std::abs( rows.normal_rates[row] - 1.0 ) > 1.0e-12 ) {
      return false;
    }
  }
  return std::abs( weight - 0.75 ) < 1.0e-12;
}

bool automaticQuadratureContract()
{
  const auto mortar = segmentTrace( 0.0, 1.0, 0.0, false, 101, 0 );
  const auto nonmortar = segmentTrace( 0.0, 1.0, -0.1, true, 202, 1 );
  const std::array<ElementPair, 1> candidate{ ElementPair{ 0, 0 } };
  const auto cell_storage =
      buildOverlapCellBatch<normal::MeanPlane>( { mortar.view(), nonmortar.view() }, { candidate.data(), 1 } );
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  const ParentTraceFieldPairView fields{
      .mortar_coordinates = { { mortar_coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar_coordinates = { { nonmortar_coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
  };
  ConstraintRowBuildOptions options;
  options.quadrature.minimum_exactness = 8;
  const auto row_storage =
      buildConstraintRowBatch( { mortar.view(), nonmortar.view() }, cell_storage.view(), fields, options );
  const auto rows = row_storage.view();
  Real integral{};
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    const Real coordinate = rows.mortar_parent_reference_coordinates( row, 0 );
    integral += rows.integration_weights[row] * std::pow( coordinate, 8 );
  }
  return rows.numberOfRows() == 5 && !rows.hasNormalRates() && std::abs( integral - 1.0 / 9.0 ) < 1.0e-13;
}

bool threeDimensionalRowContract()
{
  const auto mortar = quadrilateralTrace( 0.0, false, 0.0, 501, 2 );
  const auto nonmortar = quadrilateralTrace( -0.1, true, 0.0, 502, 4 );
  const std::array<ElementPair, 1> candidate{ ElementPair{ 0, 0 } };
  const auto cell_storage =
      buildOverlapCellBatch<normal::MeanPlane>( { mortar.view(), nonmortar.view() }, { candidate.data(), 1 } );
  constexpr std::array<Real, 12> mortar_coordinates{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0 };
  constexpr std::array<Real, 12> nonmortar_coordinates{ 0.0, 0.0, -0.1, 1.0, 0.0, -0.1,
                                                        0.0, 1.0, -0.1, 1.0, 1.0, -0.1 };
  const ParentTraceFieldPairView fields{
      .mortar_coordinates = { { mortar_coordinates.data(), 12 }, 4, 3, FieldLayout::Interleaved },
      .nonmortar_coordinates = { { nonmortar_coordinates.data(), 12 }, 4, 3, FieldLayout::Interleaved },
  };
  const auto row_storage = buildConstraintRowBatch( { mortar.view(), nonmortar.view() }, cell_storage.view(), fields );
  const auto rows = row_storage.view();
  Real weight{};
  Real moment{};
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    weight += rows.integration_weights[row];
    moment += rows.integration_weights[row] * rows.mortar_positions( row, 0 ) * rows.mortar_positions( row, 1 );
  }
  return rows.isStructurallyValid() && rows.numberOfRows() == 8 && std::abs( weight - 1.0 ) < 1.0e-12 &&
         std::abs( moment - 0.25 ) < 1.0e-12;
}

static_assert( std::is_trivially_copyable_v<OverlapCellBatchView> );
static_assert( std::is_trivially_copyable_v<ConstraintRowBatchView> );

}  // namespace

int main()
{
  const bool accepted = acceptedProvenanceContract();
  const bool statuses = explicitStatusContract();
  const bool duplicates = duplicateAndMappingContract();
  const bool three_dimensional = threeDimensionalMappingContract();
  const bool variable_rows = variableRowContract();
  const bool automatic_quadrature = automaticQuadratureContract();
  const bool three_dimensional_rows = threeDimensionalRowContract();
  if ( !accepted || !statuses || !duplicates || !three_dimensional || !variable_rows || !automatic_quadrature ||
       !three_dimensional_rows ) {
    std::cerr << "accepted=" << accepted << " statuses=" << statuses << " duplicates=" << duplicates
              << " three_dimensional=" << three_dimensional << " variable_rows=" << variable_rows
              << " automatic_quadrature=" << automatic_quadrature
              << " three_dimensional_rows=" << three_dimensional_rows << '\n';
    return 1;
  }
  return 0;
}
