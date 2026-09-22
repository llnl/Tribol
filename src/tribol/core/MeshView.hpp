#ifndef TRIBOL_CORE_MESHVIEW_HPP_
#define TRIBOL_CORE_MESHVIEW_HPP_

#include "tribol/core/ArrayView.hpp"

#include <cstdint>
#include <type_traits>

namespace tribol {

enum class ElementTopology : std::uint8_t
{
  Segment,
  Triangle,
  Quadrilateral
};

enum class FieldLayout : std::uint8_t
{
  Interleaved,
  ComponentMajor
};

template <typename T>
struct FieldView {
  ArrayView<T> values{};
  Index entities{};
  int components{};
  FieldLayout layout{ FieldLayout::Interleaved };

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr T& operator()( Index entity, int component ) const
  {
    const Index offset = layout == FieldLayout::Interleaved ? entity * components + component
                                                            : static_cast<Index>( component ) * entities + entity;
    return values[offset];
  }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool isStructurallyValid() const
  {
    return entities >= 0 && components > 0 && values.isStructurallyValid() && values.size() == entities * components;
  }
};

template <typename Scalar>
struct SurfaceMeshViewT {
  int dimension{};
  FieldView<const Scalar> coordinates{};
  ArrayView<const Index> element_offsets{};
  ArrayView<const Index> connectivity{};
  ArrayView<const ElementTopology> topologies{};
  ArrayView<const int> attributes{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index numberOfNodes() const { return coordinates.entities; }
  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index numberOfElements() const { return topologies.size(); }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool isStructurallyValid() const
  {
    const bool supported_dimension = dimension == 2 || dimension == 3;
    const bool coordinate_shape = coordinates.components == dimension && coordinates.isStructurallyValid();
    const bool arrays_valid = element_offsets.isStructurallyValid() && connectivity.isStructurallyValid() &&
                              topologies.isStructurallyValid() && attributes.isStructurallyValid();
    const bool element_shape = numberOfElements() >= 0 && element_offsets.size() == numberOfElements() + 1 &&
                               attributes.size() == numberOfElements() && !element_offsets.empty();
    if ( !supported_dimension || !coordinate_shape || !arrays_valid || !element_shape || element_offsets[0] != 0 ||
         element_offsets[numberOfElements()] != connectivity.size() ) {
      return false;
    }
    for ( Index element = 0; element < numberOfElements(); ++element ) {
      const Index begin = element_offsets[element];
      const Index end = element_offsets[element + 1];
      if ( begin < 0 || end < begin || end > connectivity.size() ||
           end - begin != nodesPerElement( topologies[element] ) ) {
        return false;
      }
      for ( Index local_node = begin; local_node < end; ++local_node ) {
        if ( connectivity[local_node] < 0 || connectivity[local_node] >= numberOfNodes() ) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  [[nodiscard]] TRIBOL_HOST_DEVICE static constexpr Index nodesPerElement( ElementTopology topology )
  {
    switch ( topology ) {
      case ElementTopology::Segment:
        return 2;
      case ElementTopology::Triangle:
        return 3;
      case ElementTopology::Quadrilateral:
        return 4;
    }
    return 0;
  }
};

using SurfaceMeshView = SurfaceMeshViewT<Real>;

template <typename Scalar>
struct SurfacePairViewT {
  SurfaceMeshViewT<Scalar> mortar{};
  SurfaceMeshViewT<Scalar> nonmortar{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool isStructurallyValid() const
  {
    return mortar.isStructurallyValid() && nonmortar.isStructurallyValid() && mortar.dimension == nonmortar.dimension;
  }
};

using SurfacePairView = SurfacePairViewT<Real>;

static_assert( std::is_trivially_copyable_v<FieldView<const Real>> );
static_assert( std::is_trivially_copyable_v<SurfaceMeshView> );
static_assert( std::is_trivially_copyable_v<SurfacePairView> );

}  // namespace tribol

#endif
