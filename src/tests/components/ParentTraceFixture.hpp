#ifndef TRIBOL_TESTS_COMPONENTS_PARENTTRACEFIXTURE_HPP_
#define TRIBOL_TESTS_COMPONENTS_PARENTTRACEFIXTURE_HPP_

#include "tribol/core/ParentTraceView.hpp"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace tribol::test {

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

  [[nodiscard]] ParentTraceView view() const
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

[[nodiscard]] inline SegmentTraceStorage segmentTrace( Real first, Real second, Real height, bool reverse,
                                                       std::uint64_t id, int owner, int copies = 1 )
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

}  // namespace tribol::test

#endif
