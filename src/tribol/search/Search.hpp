#ifndef TRIBOL_SEARCH_SEARCH_HPP_
#define TRIBOL_SEARCH_SEARCH_HPP_

#include "tribol/core/ArrayView.hpp"
#include "tribol/core/MeshView.hpp"
#include "tribol/search/Proximity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
namespace tribol {
struct ElementPair {
  Index mortar_element{};
  Index nonmortar_element{};

  TRIBOL_HOST_DEVICE friend constexpr bool operator==( ElementPair, ElementPair ) = default;
};
class CandidatePairs {
 public:
  void clear() { pairs_.clear(); }
  void reserve( Index count ) { pairs_.reserve( static_cast<std::size_t>( count ) ); }
  void append( ElementPair pair ) { pairs_.push_back( pair ); }

  void canonicalize()
  {
    std::sort( pairs_.begin(), pairs_.end(), []( ElementPair left, ElementPair right ) {
      return left.mortar_element < right.mortar_element ||
             ( left.mortar_element == right.mortar_element && left.nonmortar_element < right.nonmortar_element );
    } );
    pairs_.erase( std::unique( pairs_.begin(), pairs_.end() ), pairs_.end() );
  }

  void filterSelfContact( const SurfaceMeshView& surface, bool exclude_adjacent )
  {
    auto output = pairs_.begin();
    for ( ElementPair pair : pairs_ ) {
      if ( pair.mortar_element >= pair.nonmortar_element ||
           ( exclude_adjacent && detailSharesNode( surface, pair.mortar_element, pair.nonmortar_element ) ) ) {
        continue;
      }
      *output++ = pair;
    }
    pairs_.erase( output, pairs_.end() );
    canonicalize();
  }

  [[nodiscard]] Index size() const { return static_cast<Index>( pairs_.size() ); }
  [[nodiscard]] bool empty() const { return pairs_.empty(); }
  [[nodiscard]] ArrayView<const ElementPair> view() const { return { pairs_.data(), size() }; }

 private:
  static bool detailSharesNode( const SurfaceMeshView& mesh, Index left_element, Index right_element )
  {
    const Index left_begin = mesh.element_offsets[left_element];
    const Index left_end = mesh.element_offsets[left_element + 1];
    const Index right_begin = mesh.element_offsets[right_element];
    const Index right_end = mesh.element_offsets[right_element + 1];
    for ( Index left = left_begin; left < left_end; ++left ) {
      for ( Index right = right_begin; right < right_end; ++right ) {
        if ( mesh.connectivity[left] == mesh.connectivity[right] ) {
          return true;
        }
      }
    }
    return false;
  }

  std::vector<ElementPair> pairs_;
};
template <typename T>
concept SearchPolicy = std::constructible_from<T, typename T::Parameters> &&
                       requires( const T& search, const SurfacePairView& surfaces, CandidatePairs& candidates ) {
                         typename T::Parameters;
                         { search.findCandidates( surfaces, candidates ) } -> std::same_as<void>;
                       };

namespace search {
namespace detail {

struct Cell {
  std::array<Index, 3> index{};

  friend bool operator==( const Cell&, const Cell& ) = default;
};

struct CellHash {
  std::size_t operator()( const Cell& cell ) const noexcept
  {
    std::size_t result = 0xcbf29ce484222325ULL;
    for ( const Index index : cell.index ) {
      result ^= std::hash<Index>{}( index ) + 0x9e3779b9 + ( result << 6U ) + ( result >> 2U );
    }
    return result;
  }
};

inline Index cellIndex( Real coordinate, Real origin, Real cell_size )
{
  return static_cast<Index>( std::floor( ( coordinate - origin ) / cell_size ) );
}

}  // namespace detail
class CartesianProduct {
 public:
  struct Parameters {
    Real expansion{};
    Real proximity_scale{};
  };

  CartesianProduct() : CartesianProduct( Parameters{} ) {}

  explicit CartesianProduct( Parameters parameters ) : parameters_( parameters )
  {
    detail::validateProximity( parameters_.expansion, parameters_.proximity_scale );
  }

  void findCandidates( const SurfacePairView& surfaces, CandidatePairs& candidates ) const
  {
    detail::requireValidSurfaces( surfaces );
    candidates.clear();
    candidates.reserve( surfaces.mortar.numberOfElements() * surfaces.nonmortar.numberOfElements() );
    for ( Index mortar = 0; mortar < surfaces.mortar.numberOfElements(); ++mortar ) {
      const auto mortar_bounds =
          detail::elementBounds( surfaces.mortar, mortar, parameters_.expansion, parameters_.proximity_scale );
      for ( Index nonmortar = 0; nonmortar < surfaces.nonmortar.numberOfElements(); ++nonmortar ) {
        const auto nonmortar_bounds =
            detail::elementBounds( surfaces.nonmortar, nonmortar, parameters_.expansion, parameters_.proximity_scale );
        if ( detail::overlaps( mortar_bounds, nonmortar_bounds, surfaces.mortar.dimension ) ) {
          candidates.append( { mortar, nonmortar } );
        }
      }
    }
  }

 private:
  Parameters parameters_;
};
class Grid {
 public:
  struct Parameters {
    Real expansion{};
    Real proximity_scale{};
    Real cell_size{};
  };

  Grid() : Grid( Parameters{} ) {}

  explicit Grid( Parameters parameters ) : parameters_( parameters )
  {
    detail::validateProximity( parameters_.expansion, parameters_.proximity_scale );
    if ( parameters_.cell_size < 0.0 ) {
      throw std::invalid_argument( "Grid search cell size cannot be negative." );
    }
  }

  void findCandidates( const SurfacePairView& surfaces, CandidatePairs& candidates ) const
  {
    detail::requireValidSurfaces( surfaces );
    candidates.clear();
    const auto mortar_count = surfaces.mortar.numberOfElements();
    const auto nonmortar_count = surfaces.nonmortar.numberOfElements();
    std::vector<detail::BoundingBox> mortar_bounds( static_cast<std::size_t>( mortar_count ) );
    std::vector<detail::BoundingBox> nonmortar_bounds( static_cast<std::size_t>( nonmortar_count ) );
    std::array<Real, 3> origin{};
    std::array<Real, 3> global_maximum{};
    for ( int component = 0; component < 3; ++component ) {
      origin[component] = 0.0;
      global_maximum[component] = 0.0;
    }
    bool first_bounds = true;
    auto include_bounds = [&]( const detail::BoundingBox& bounds ) {
      for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
        if ( first_bounds ) {
          origin[component] = bounds.minimum[component];
          global_maximum[component] = bounds.maximum[component];
        } else {
          origin[component] = std::min( origin[component], bounds.minimum[component] );
          global_maximum[component] = std::max( global_maximum[component], bounds.maximum[component] );
        }
      }
      first_bounds = false;
    };
    for ( Index element = 0; element < mortar_count; ++element ) {
      mortar_bounds[static_cast<std::size_t>( element )] =
          detail::elementBounds( surfaces.mortar, element, parameters_.expansion, parameters_.proximity_scale );
      include_bounds( mortar_bounds[static_cast<std::size_t>( element )] );
    }
    for ( Index element = 0; element < nonmortar_count; ++element ) {
      nonmortar_bounds[static_cast<std::size_t>( element )] =
          detail::elementBounds( surfaces.nonmortar, element, parameters_.expansion, parameters_.proximity_scale );
      include_bounds( nonmortar_bounds[static_cast<std::size_t>( element )] );
    }
    Real cell_size = parameters_.cell_size;
    if ( cell_size == 0.0 ) {
      Real longest_extent{};
      for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
        longest_extent = std::max( longest_extent, global_maximum[component] - origin[component] );
      }
      const auto count = std::max<Index>( mortar_count + nonmortar_count, 1 );
      cell_size = longest_extent > 0.0
                      ? longest_extent / std::pow( static_cast<Real>( count ), 1.0 / surfaces.mortar.dimension )
                      : 1.0;
    }

    std::unordered_map<detail::Cell, std::vector<Index>, detail::CellHash> cells;
    auto visit_cells = [&]( const detail::BoundingBox& bounds, const auto& visit ) {
      std::array<Index, 3> lower{};
      std::array<Index, 3> upper{};
      for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
        lower[component] = detail::cellIndex( bounds.minimum[component], origin[component], cell_size );
        upper[component] = detail::cellIndex( bounds.maximum[component], origin[component], cell_size );
      }
      for ( Index third = lower[2]; third <= upper[2]; ++third ) {
        for ( Index second = lower[1]; second <= upper[1]; ++second ) {
          for ( Index first = lower[0]; first <= upper[0]; ++first ) {
            visit( detail::Cell{ { first, second, third } } );
          }
        }
      }
    };
    for ( Index element = 0; element < mortar_count; ++element ) {
      visit_cells( mortar_bounds[static_cast<std::size_t>( element )],
                   [&]( detail::Cell cell ) { cells[cell].push_back( element ); } );
    }
    for ( Index nonmortar = 0; nonmortar < nonmortar_count; ++nonmortar ) {
      const auto& right = nonmortar_bounds[static_cast<std::size_t>( nonmortar )];
      visit_cells( right, [&]( detail::Cell cell ) {
        const auto found = cells.find( cell );
        if ( found == cells.end() ) {
          return;
        }
        for ( const Index mortar : found->second ) {
          if ( detail::overlaps( mortar_bounds[static_cast<std::size_t>( mortar )], right,
                                 surfaces.mortar.dimension ) ) {
            candidates.append( { mortar, nonmortar } );
          }
        }
      } );
    }
    candidates.canonicalize();
  }

 private:
  Parameters parameters_;
};

class Bvh {
 public:
  struct Parameters {
    Real expansion{};
    Real proximity_scale{};
    Index leaf_size{ 4 };
  };

  Bvh() : Bvh( Parameters{} ) {}

  explicit Bvh( Parameters parameters ) : parameters_( parameters )
  {
    detail::validateProximity( parameters_.expansion, parameters_.proximity_scale );
    if ( parameters_.leaf_size <= 0 ) {
      throw std::invalid_argument( "BVH search requires positive leaf size." );
    }
  }

  void findCandidates( const SurfacePairView& surfaces, CandidatePairs& candidates ) const
  {
    detail::requireValidSurfaces( surfaces );
    candidates.clear();
    if ( surfaces.mortar.numberOfElements() == 0 || surfaces.nonmortar.numberOfElements() == 0 ) {
      return;
    }
    std::vector<detail::BoundingBox> bounds( static_cast<std::size_t>( surfaces.mortar.numberOfElements() ) );
    std::vector<Index> order( bounds.size() );
    std::iota( order.begin(), order.end(), Index{} );
    for ( Index element = 0; element < surfaces.mortar.numberOfElements(); ++element ) {
      bounds[static_cast<std::size_t>( element )] =
          detail::elementBounds( surfaces.mortar, element, parameters_.expansion, parameters_.proximity_scale );
    }
    std::vector<Node> nodes;
    nodes.reserve( bounds.size() * 2U );
    const Index root = build( bounds, order, 0, static_cast<Index>( order.size() ), surfaces.mortar.dimension, nodes );
    for ( Index nonmortar = 0; nonmortar < surfaces.nonmortar.numberOfElements(); ++nonmortar ) {
      const auto query =
          detail::elementBounds( surfaces.nonmortar, nonmortar, parameters_.expansion, parameters_.proximity_scale );
      traverse( root, query, nonmortar, bounds, order, surfaces.mortar.dimension, nodes, candidates );
    }
    candidates.canonicalize();
  }

 private:
  struct Node {
    detail::BoundingBox bounds{};
    Index begin{};
    Index end{};
    Index left{ -1 };
    Index right{ -1 };

    [[nodiscard]] bool leaf() const { return left < 0; }
  };

  static detail::BoundingBox mergedBounds( const std::vector<detail::BoundingBox>& bounds,
                                           const std::vector<Index>& order, Index begin, Index end, int dimension )
  {
    auto result = bounds[static_cast<std::size_t>( order[static_cast<std::size_t>( begin )] )];
    for ( Index entry = begin + 1; entry < end; ++entry ) {
      const auto& next = bounds[static_cast<std::size_t>( order[static_cast<std::size_t>( entry )] )];
      for ( int component = 0; component < dimension; ++component ) {
        result.minimum[component] = std::min( result.minimum[component], next.minimum[component] );
        result.maximum[component] = std::max( result.maximum[component], next.maximum[component] );
      }
    }
    return result;
  }

  Index build( const std::vector<detail::BoundingBox>& bounds, std::vector<Index>& order, Index begin, Index end,
               int dimension, std::vector<Node>& nodes ) const
  {
    const Index node_index = static_cast<Index>( nodes.size() );
    nodes.push_back(
        Node{ .bounds = mergedBounds( bounds, order, begin, end, dimension ), .begin = begin, .end = end } );
    if ( end - begin <= parameters_.leaf_size ) {
      return node_index;
    }
    int split_component = 0;
    Real split_extent = nodes[static_cast<std::size_t>( node_index )].bounds.maximum[0] -
                        nodes[static_cast<std::size_t>( node_index )].bounds.minimum[0];
    for ( int component = 1; component < dimension; ++component ) {
      const Real extent = nodes[static_cast<std::size_t>( node_index )].bounds.maximum[component] -
                          nodes[static_cast<std::size_t>( node_index )].bounds.minimum[component];
      if ( extent > split_extent ) {
        split_component = component;
        split_extent = extent;
      }
    }
    const Index middle = begin + ( end - begin ) / 2;
    std::nth_element(
        order.begin() + begin, order.begin() + middle, order.begin() + end, [&]( Index left, Index right ) {
          const Real left_center = detail::center( bounds[static_cast<std::size_t>( left )], split_component );
          const Real right_center = detail::center( bounds[static_cast<std::size_t>( right )], split_component );
          return left_center < right_center || ( left_center == right_center && left < right );
        } );
    const Index left = build( bounds, order, begin, middle, dimension, nodes );
    const Index right = build( bounds, order, middle, end, dimension, nodes );
    nodes[static_cast<std::size_t>( node_index )].left = left;
    nodes[static_cast<std::size_t>( node_index )].right = right;
    return node_index;
  }

  static void traverse( Index node_index, const detail::BoundingBox& query, Index nonmortar,
                        const std::vector<detail::BoundingBox>& bounds, const std::vector<Index>& order, int dimension,
                        const std::vector<Node>& nodes, CandidatePairs& candidates )
  {
    const auto& node = nodes[static_cast<std::size_t>( node_index )];
    if ( !detail::overlaps( node.bounds, query, dimension ) ) {
      return;
    }
    if ( node.leaf() ) {
      for ( Index entry = node.begin; entry < node.end; ++entry ) {
        const Index mortar = order[static_cast<std::size_t>( entry )];
        if ( detail::overlaps( bounds[static_cast<std::size_t>( mortar )], query, dimension ) ) {
          candidates.append( { mortar, nonmortar } );
        }
      }
      return;
    }
    traverse( node.left, query, nonmortar, bounds, order, dimension, nodes, candidates );
    traverse( node.right, query, nonmortar, bounds, order, dimension, nodes, candidates );
  }

  Parameters parameters_;
};

class Supplied {
 public:
  struct Parameters {
    std::vector<ElementPair> pairs;
  };

  Supplied() = default;

  explicit Supplied( Parameters parameters ) : pairs_( std::move( parameters.pairs ) ) {}

  void findCandidates( const SurfacePairView& surfaces, CandidatePairs& candidates ) const
  {
    candidates.clear();
    candidates.reserve( static_cast<Index>( pairs_.size() ) );
    for ( const ElementPair pair : pairs_ ) {
      if ( pair.mortar_element < 0 || pair.mortar_element >= surfaces.mortar.numberOfElements() ||
           pair.nonmortar_element < 0 || pair.nonmortar_element >= surfaces.nonmortar.numberOfElements() ) {
        throw std::out_of_range( "A supplied contact pair references an element outside its surface." );
      }
      candidates.append( pair );
    }
    candidates.canonicalize();
  }

 private:
  std::vector<ElementPair> pairs_;
};

template <SearchPolicy BroadPhase = CartesianProduct>
class SelfContact {
 public:
  struct Parameters {
    typename BroadPhase::Parameters broad_phase{};
    bool exclude_adjacent{ true };
  };

  SelfContact() : SelfContact( Parameters{} ) {}

  explicit SelfContact( Parameters parameters )
      : broad_phase_( std::move( parameters.broad_phase ) ), exclude_adjacent_( parameters.exclude_adjacent )
  {
  }

  void findCandidates( const SurfaceMeshView& surface, CandidatePairs& candidates ) const
  {
    if ( !surface.isStructurallyValid() ) {
      throw std::invalid_argument( "Self-contact search requires a structurally valid surface." );
    }
    CandidatePairs directed;
    broad_phase_.findCandidates( { surface, surface }, directed );
    candidates.clear();
    for ( ElementPair pair : directed.view() ) {
      if ( pair.mortar_element >= pair.nonmortar_element ) {
        continue;
      }
      candidates.append( pair );
    }
    candidates.filterSelfContact( surface, exclude_adjacent_ );
  }

 private:
  BroadPhase broad_phase_;
  bool exclude_adjacent_{};
};

}  // namespace search

static_assert( SearchPolicy<search::CartesianProduct> );
static_assert( SearchPolicy<search::Grid> );
static_assert( SearchPolicy<search::Bvh> );
static_assert( SearchPolicy<search::Supplied> );

}  // namespace tribol

#endif
