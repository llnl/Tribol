#ifndef TRIBOL_CORE_INTERACTIONPATCH_HPP_
#define TRIBOL_CORE_INTERACTIONPATCH_HPP_

#include "tribol/core/OverlapCellView.hpp"

#include <array>
#include <type_traits>

namespace tribol {

template <typename Scalar>
struct InteractionGeometryT {
  std::array<Scalar, 3> normal{};
  std::array<Scalar, 3> mortar_point{};
  std::array<Scalar, 3> nonmortar_point{};
  Scalar measure{};
  Scalar gap{};
  bool valid{};
};

using InteractionGeometry = InteractionGeometryT<Real>;

inline constexpr int maximumInteractionVertices = maximumOverlapVertices;

template <typename Scalar>
struct InteractionPatchT {
  std::array<Scalar, 3> normal{};
  std::array<std::array<Scalar, 3>, maximumInteractionVertices> integration_vertices{};
  std::array<std::array<Scalar, 3>, maximumInteractionVertices> mortar_vertices{};
  std::array<std::array<Scalar, 3>, maximumInteractionVertices> nonmortar_vertices{};
  std::array<Scalar, 3> mortar_centroid{};
  std::array<Scalar, 3> nonmortar_centroid{};
  Scalar measure{};
  int vertex_count{};
  int manifold_dimension{};
  GeometryStatus status{ GeometryStatus::NoOverlap };
  bool valid{};
};

using InteractionPatch = InteractionPatchT<Real>;

static_assert( std::is_trivially_copyable_v<InteractionPatch> );

}  // namespace tribol

#endif
