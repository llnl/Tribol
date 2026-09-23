#ifndef TRIBOL_CORE_OVERLAPCELLVIEW_HPP_
#define TRIBOL_CORE_OVERLAPCELLVIEW_HPP_

#include "tribol/core/MeshView.hpp"

#include <cstdint>
#include <type_traits>

namespace tribol {

inline constexpr int maximumOverlapVertices = 8;

enum class GeometryStatus : std::uint8_t
{
  Accepted,
  NoOverlap,
  InvalidProvenance,
  InvalidMapping,
  InconsistentOrientation,
  ExcessiveMappingError,
  DegenerateOverlap,
  Duplicate
};

template <typename Scalar>
struct OverlapCellBatchViewT {
  int dimension{};
  ArrayView<const std::uint64_t> contact_pair_ids{};
  ArrayView<const std::uint64_t> cell_ids{};
  ArrayView<const int> owner_ranks{};
  ArrayView<const Index> mortar_subfacets{};
  ArrayView<const Index> nonmortar_subfacets{};
  ArrayView<const Index> mortar_parent_faces{};
  ArrayView<const Index> nonmortar_parent_faces{};
  ArrayView<const GeometryStatus> statuses{};
  FieldView<const Scalar> normals{};
  ArrayView<const Scalar> measures{};
  ArrayView<const Index> vertex_offsets{};
  FieldView<const Scalar> integration_vertices{};
  FieldView<const Scalar> mortar_parent_reference_coordinates{};
  FieldView<const Scalar> nonmortar_parent_reference_coordinates{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index numberOfAttempts() const { return statuses.size(); }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index numberOfVertices() const { return integration_vertices.entities; }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool isAccepted( Index attempt ) const
  {
    return statuses[attempt] == GeometryStatus::Accepted;
  }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool isStructurallyValid() const
  {
    const Index attempts = numberOfAttempts();
    const int reference_dimension = dimension - 1;
    const bool supported_dimension = dimension == 2 || dimension == 3;
    const bool attempt_arrays =
        contact_pair_ids.isStructurallyValid() && cell_ids.isStructurallyValid() && owner_ranks.isStructurallyValid() &&
        mortar_subfacets.isStructurallyValid() && nonmortar_subfacets.isStructurallyValid() &&
        mortar_parent_faces.isStructurallyValid() && nonmortar_parent_faces.isStructurallyValid() &&
        statuses.isStructurallyValid() && measures.isStructurallyValid() && vertex_offsets.isStructurallyValid();
    const bool attempt_sizes = contact_pair_ids.size() == attempts && cell_ids.size() == attempts &&
                               owner_ranks.size() == attempts && mortar_subfacets.size() == attempts &&
                               nonmortar_subfacets.size() == attempts && mortar_parent_faces.size() == attempts &&
                               nonmortar_parent_faces.size() == attempts && measures.size() == attempts &&
                               vertex_offsets.size() == attempts + 1;
    const bool field_shapes = normals.isStructurallyValid() && normals.entities == attempts &&
                              normals.components == 3 && integration_vertices.isStructurallyValid() &&
                              integration_vertices.components == 3 &&
                              mortar_parent_reference_coordinates.isStructurallyValid() &&
                              mortar_parent_reference_coordinates.entities == numberOfVertices() &&
                              mortar_parent_reference_coordinates.components == reference_dimension &&
                              nonmortar_parent_reference_coordinates.isStructurallyValid() &&
                              nonmortar_parent_reference_coordinates.entities == numberOfVertices() &&
                              nonmortar_parent_reference_coordinates.components == reference_dimension;
    if ( !supported_dimension || !attempt_arrays || !attempt_sizes || !field_shapes || vertex_offsets[0] != 0 ||
         vertex_offsets[attempts] != numberOfVertices() ) {
      return false;
    }
    for ( Index attempt = 0; attempt < attempts; ++attempt ) {
      const Index vertices = vertex_offsets[attempt + 1] - vertex_offsets[attempt];
      if ( vertex_offsets[attempt] < 0 || vertices < 0 || vertices > maximumOverlapVertices ||
           ( isAccepted( attempt ) && vertices < dimension ) ) {
        return false;
      }
    }
    return true;
  }
};

using OverlapCellBatchView = OverlapCellBatchViewT<Real>;

static_assert( std::is_trivially_copyable_v<OverlapCellBatchView> );

}  // namespace tribol

#endif
