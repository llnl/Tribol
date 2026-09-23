#ifndef TRIBOL_CORE_PARENTTRACEVIEW_HPP_
#define TRIBOL_CORE_PARENTTRACEVIEW_HPP_

#include "tribol/core/MeshView.hpp"

#include <cstdint>
#include <type_traits>

namespace tribol {

inline constexpr int maximumParentTraceOrder = 4;

[[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index parentFaceDofCount( ElementTopology topology, int order )
{
  if ( order < 1 || order > maximumParentTraceOrder ) {
    return 0;
  }
  switch ( topology ) {
    case ElementTopology::Segment:
      return order + 1;
    case ElementTopology::Triangle:
      return ( order + 1 ) * ( order + 2 ) / 2;
    case ElementTopology::Quadrilateral:
      return ( order + 1 ) * ( order + 1 );
  }
  return 0;
}

template <typename Scalar>
struct ParentTraceViewT {
  SurfaceMeshViewT<Scalar> geometry{};
  ArrayView<const std::uint64_t> parent_face_ids{};
  ArrayView<const int> parent_face_owners{};
  ArrayView<const ElementTopology> parent_face_topologies{};
  ArrayView<const int> parent_face_orders{};
  ArrayView<const Index> parent_dof_offsets{};
  ArrayView<const Index> parent_dofs{};
  ArrayView<const Index> subfacet_parent_faces{};
  FieldView<const Scalar> subfacet_parent_reference_coordinates{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index numberOfParentFaces() const { return parent_face_topologies.size(); }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index numberOfSubfacets() const { return geometry.numberOfElements(); }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr int parentReferenceDimension() const { return geometry.dimension - 1; }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool isStructurallyValid() const
  {
    const Index parent_faces = numberOfParentFaces();
    if ( !geometry.isStructurallyValid() || !parent_face_ids.isStructurallyValid() ||
         !parent_face_owners.isStructurallyValid() || !parent_face_topologies.isStructurallyValid() ||
         !parent_face_orders.isStructurallyValid() || !parent_dof_offsets.isStructurallyValid() ||
         !parent_dofs.isStructurallyValid() || !subfacet_parent_faces.isStructurallyValid() ||
         !subfacet_parent_reference_coordinates.isStructurallyValid() || parent_face_ids.size() != parent_faces ||
         parent_face_owners.size() != parent_faces || parent_face_orders.size() != parent_faces ||
         parent_dof_offsets.size() != parent_faces + 1 || parent_dof_offsets.empty() || parent_dof_offsets[0] != 0 ||
         parent_dof_offsets[parent_faces] != parent_dofs.size() ||
         subfacet_parent_faces.size() != numberOfSubfacets() ||
         subfacet_parent_reference_coordinates.entities != geometry.connectivity.size() ||
         subfacet_parent_reference_coordinates.components != parentReferenceDimension() ) {
      return false;
    }
    for ( Index face = 0; face < parent_faces; ++face ) {
      const ElementTopology topology = parent_face_topologies[face];
      const int order = parent_face_orders[face];
      const Index begin = parent_dof_offsets[face];
      const Index end = parent_dof_offsets[face + 1];
      const bool topology_matches_dimension =
          geometry.dimension == 2 ? topology == ElementTopology::Segment : topology != ElementTopology::Segment;
      if ( !topology_matches_dimension || begin < 0 || end < begin || end > parent_dofs.size() ||
           end - begin != parentFaceDofCount( topology, order ) ) {
        return false;
      }
      for ( Index offset = begin; offset < end; ++offset ) {
        if ( parent_dofs[offset] < 0 ) {
          return false;
        }
      }
    }
    for ( Index subfacet = 0; subfacet < numberOfSubfacets(); ++subfacet ) {
      const Index parent_face = subfacet_parent_faces[subfacet];
      const ElementTopology topology = geometry.topologies[subfacet];
      const bool topology_matches_dimension =
          geometry.dimension == 2 ? topology == ElementTopology::Segment : topology != ElementTopology::Segment;
      if ( parent_face < 0 || parent_face >= parent_faces || !topology_matches_dimension ) {
        return false;
      }
    }
    return true;
  }
};

using ParentTraceView = ParentTraceViewT<Real>;

template <typename Scalar>
struct ParentTracePairViewT {
  ParentTraceViewT<Scalar> mortar{};
  ParentTraceViewT<Scalar> nonmortar{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool isStructurallyValid() const
  {
    return mortar.isStructurallyValid() && nonmortar.isStructurallyValid() &&
           mortar.geometry.dimension == nonmortar.geometry.dimension;
  }
};

using ParentTracePairView = ParentTracePairViewT<Real>;

static_assert( std::is_trivially_copyable_v<ParentTraceView> );
static_assert( std::is_trivially_copyable_v<ParentTracePairView> );

}  // namespace tribol

#endif
