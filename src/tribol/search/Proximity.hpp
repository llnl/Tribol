#ifndef TRIBOL_SEARCH_PROXIMITY_HPP_
#define TRIBOL_SEARCH_PROXIMITY_HPP_

#include "tribol/core/MeshView.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace tribol::search {

inline constexpr Real legacyProximityScale = 4.0;
inline constexpr Real minimumRecommendedProximityScale = 2.0;

namespace detail {

struct BoundingBox {
  std::array<Real, 3> minimum{};
  std::array<Real, 3> maximum{};
};

inline BoundingBox elementBounds( const SurfaceMeshView& mesh, Index element, Real expansion, Real proximity_scale )
{
  BoundingBox bounds;
  const Index begin = mesh.element_offsets[element];
  const Index end = mesh.element_offsets[element + 1];
  if ( begin == end ) {
    throw std::invalid_argument( "Surface elements must contain at least one node." );
  }

  Real longest_extent{};
  for ( int component = 0; component < mesh.dimension; ++component ) {
    const Real first = mesh.coordinates( mesh.connectivity[begin], component );
    bounds.minimum[component] = first;
    bounds.maximum[component] = first;
    for ( Index local_node = begin + 1; local_node < end; ++local_node ) {
      const Real value = mesh.coordinates( mesh.connectivity[local_node], component );
      bounds.minimum[component] = std::min( bounds.minimum[component], value );
      bounds.maximum[component] = std::max( bounds.maximum[component], value );
    }
    longest_extent = std::max( longest_extent, bounds.maximum[component] - bounds.minimum[component] );
  }
  const Real total_expansion = expansion + proximity_scale * longest_extent;
  for ( int component = 0; component < mesh.dimension; ++component ) {
    bounds.minimum[component] -= total_expansion;
    bounds.maximum[component] += total_expansion;
  }
  return bounds;
}

inline bool overlaps( const BoundingBox& left, const BoundingBox& right, int dimension )
{
  for ( int component = 0; component < dimension; ++component ) {
    if ( left.maximum[component] < right.minimum[component] || right.maximum[component] < left.minimum[component] ) {
      return false;
    }
  }
  return true;
}

inline void requireValidSurfaces( const SurfacePairView& surfaces )
{
  if ( !surfaces.isStructurallyValid() ) {
    throw std::invalid_argument( "Search requires a structurally valid surface pair." );
  }
}

inline Real center( const BoundingBox& bounds, int component )
{
  return 0.5 * ( bounds.minimum[component] + bounds.maximum[component] );
}

inline void validateProximity( Real expansion, Real proximity_scale )
{
  if ( expansion < 0.0 || proximity_scale < 0.0 ) {
    throw std::invalid_argument( "Search expansion and proximity scale cannot be negative." );
  }
}

}  // namespace detail
}  // namespace tribol::search

#endif
