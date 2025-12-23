// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_UTILS_BISECTREE_HPP_
#define SRC_REDECOMP_UTILS_BISECTREE_HPP_

#include <cmath>
#include <cstddef>

#include <utility>

#include "axom/core.hpp"

namespace redecomp {

/**
 * @brief Implements a nearly-complete binary tree designed for breadth-first traversal
 *
 * A nearly complete binary tree is filled from the left first, but may contain
 * multiple levels which are not completely filled.  Storage is through axom::Array.
 */
template <typename T>
class BisecTree {
 public:
  using BFSLevelIterator = typename axom::Array<axom::Array<T>>::ArrayIterator;
  using ConstBFSLevelIterator = typename axom::Array<axom::Array<T>>::ConstArrayIterator;
  using BFSNodeIterator = typename axom::Array<T>::ArrayIterator;
  using ConstBFSNodeIterator = typename axom::Array<T>::ConstArrayIterator;
  using BFSIterator = std::pair<BFSLevelIterator, BFSNodeIterator>;
  using ConstBFSIterator = std::pair<ConstBFSLevelIterator, ConstBFSNodeIterator>;

  BisecTree() = default;

  /**
   * @brief Construct a new BisecTree object
   *
   * @param n_parts Number of entries at the lowest level of the tree
   */
  BisecTree( size_t n_parts )
      : n_parts_{ n_parts },
        n_levels_{ static_cast<size_t>( ceil( log2( static_cast<double>( n_parts ) - 0.25 ) ) + 0.5 ) + 1 },
        all_nodes_( n_levels_, n_levels_ )
  {
    // populate the bisection tree with empty nodes
    auto n_nodes = n_parts_;
    for ( auto it = all_nodes_.end(); it != all_nodes_.begin(); ) {
      --it;
      *it = axom::Array<T>( n_nodes, n_nodes );
      n_nodes = ( n_nodes + 1 ) / 2;
    }
  }

  /**
   * @brief Returns the value at the given level and node
   *
   * @param level Tree level (0 = top, n_levels_ = bottom)
   * @param node Node index (0 = left-most node)
   * @return T& Value at given level and node
   */
  T& operator()( size_t level, size_t node ) { return all_nodes_[level][node]; }

  /**
   * @brief Returns the value at the bottom level and given node
   *
   * @param node Node index (0 = left-most node)
   * @return T& Value at given node
   */
  T& operator()( size_t node ) { return all_nodes_[n_levels_ - 1][node]; }

  /**
   * @brief Returns an iterator to the top level
   *
   * @return BFSLevelIterator pointing to the top level
   */
  BFSLevelIterator begin() { return all_nodes_.begin(); }

  /**
   * @brief Returns a const iterator to the top level
   *
   * @return ConstBFSLevelIterator pointing to the top level
   */
  ConstBFSLevelIterator begin() const { return all_nodes_.begin(); }

  /**
   * @brief Returns an iterator to the bottom level
   *
   * @return BFSLevelIterator pointing to the bottom level
   */
  BFSLevelIterator end() { return all_nodes_.end(); }

  /**
   * @brief Returns a const iterator to the bottom level
   *
   * @return ConstBFSLevelIterator pointing to the bottom level
   */
  ConstBFSLevelIterator end() const { return all_nodes_.end(); }

  /**
   * @brief Returns an iterator to the left-most node of the given level
   *
   * @param lvl_it Iterator pointing to a level in the BisecTree
   * @return BFSNodeIterator pointing to the left-most node of the level in lvl_it
   */
  BFSNodeIterator begin( BFSLevelIterator lvl_it ) { return ( *lvl_it ).begin(); }

  /**
   * @brief Returns a const iterator to the left-most node of the given level
   *
   * @param lvl_it Const iterator pointing to a level in the BisecTree
   * @return ConstBFSNodeIterator pointing to the left-most node of the level in lvl_it
   */
  ConstBFSNodeIterator begin( ConstBFSLevelIterator lvl_it ) const { return ( *lvl_it ).begin(); }

  /**
   * @brief Returns a iterator to one after the right-most node of the given level
   *
   * @param lvl_it Iterator pointing to a level in the BisecTree
   * @return BFSNodeIterator pointing to one after the right-most node of the level in lvl_it
   */
  BFSNodeIterator end( BFSLevelIterator lvl_it ) { return ( *lvl_it ).end(); }

  /**
   * @brief Returns a const iterator to one after the right-most node of the given level
   *
   * @param lvl_it Const iterator pointing to a level in the BisecTree
   * @return ConstBFSNodeIterator pointing to one after the right-most node of the level in lvl_it
   */
  ConstBFSNodeIterator end( ConstBFSLevelIterator lvl_it ) const { return ( *lvl_it ).end(); }

  /**
   * @brief Returns the offset of the given node_it to the beginning of the level
   *
   * @param lvl_it Iterator pointing to a level in the BisecTree
   * @param node_it Iterator pointing to a node in the level in lvl_it
   * @return Offset of node_it
   */
  size_t position( BFSLevelIterator lvl_it, BFSNodeIterator node_it ) { return node_it - ( *lvl_it ).begin(); }

  /**
   * @brief Returns the offset of the given node_it to the beginning of the level
   *
   * @param lvl_it Iterator pointing to a level in the BisecTree
   * @param node_it Iterator pointing to a node in the level in lvl_it
   * @return Offset of node_it
   */
  size_t position( ConstBFSLevelIterator lvl_it, ConstBFSNodeIterator node_it ) const
  {
    return node_it - ( *lvl_it ).begin();
  }

  /**
   * @brief Returns the offset of the given it to the beginning of the level
   *
   * @param it BFSIterator pointing to a level and node in the BisecTree
   * @return Offset of node_it
   */
  size_t position( const BFSIterator& it ) { return position( it.first, it.second ); }

  /**
   * @brief Returns the offset of the given it to the beginning of the level
   *
   * @param it ConstBFSIterator pointing to a level and node in the BisecTree
   * @return Offset of node_it
   */
  size_t position( const ConstBFSIterator& it ) const { return position( it.first, it.second ); }

  /**
   * @brief Returns iterator to the root node of the given level and node iterators
   *
   * @param lvl_it Iterator pointing to a level in the BisecTree
   * @param node_it Iterator pointing to a node in the level of lvl_it
   * @return BFSIterator pointing to the root node
   */
  BFSIterator root( BFSLevelIterator lvl_it, BFSNodeIterator node_it )
  {
    auto node_idx = position( lvl_it, node_it );
    --lvl_it;
    return std::make_pair( lvl_it, BFSNodeIterator( node_idx / 2, &( *lvl_it ) ) );
  }

  /**
   * @brief Returns const iterator to the root node of the given level and node const iterators
   *
   * @param lvl_it Const iterator pointing to a level in the BisecTree
   * @param node_it Const iterator pointing to a node in the level of lvl_it
   * @return ConstBFSIterator pointing to the root node
   */
  ConstBFSIterator root( ConstBFSLevelIterator lvl_it, ConstBFSNodeIterator node_it ) const
  {
    auto node_idx = position( lvl_it, node_it );
    --lvl_it;
    return std::make_pair( lvl_it, ConstBFSNodeIterator( node_idx / 2, &( *lvl_it ) ) );
  }

  /**
   * @brief Returns iterator to the root node of the given iterator
   *
   * @param it Iterator pointing to a level and node in the BisecTree
   * @return BFSIterator pointing to the root node
   */
  BFSIterator root( const BFSIterator& it ) { return root( it.first, it.second ); }

  /**
   * @brief Returns const iterator to the root node of the given const iterator
   *
   * @param it Const iterator pointing to a level and node in the BisecTree
   * @return ConstBFSIterator pointing to the root node
   */
  ConstBFSIterator root( const ConstBFSIterator& it ) const { return root( it.first, it.second ); }

  /**
   * @brief Returns iterator to the left node of the given level and node iterators
   *
   * @param lvl_it Iterator pointing to a level in the BisecTree
   * @param node_it Iterator pointing to a node in the level of lvl_it
   * @return BFSIterator pointing to the left node
   */
  BFSIterator left( BFSLevelIterator lvl_it, BFSNodeIterator node_it )
  {
    auto node_idx = position( lvl_it, node_it );
    ++lvl_it;
    return std::make_pair( lvl_it, BFSNodeIterator( node_idx * 2, &( *lvl_it ) ) );
  }

  /**
   * @brief Returns const iterator to the left node of the given level and node const iterators
   *
   * @param lvl_it Const iterator pointing to a level in the BisecTree
   * @param node_it Const iterator pointing to a node in the level of lvl_it
   * @return ConstBFSIterator pointing to the left node
   */
  ConstBFSIterator left( ConstBFSLevelIterator lvl_it, ConstBFSNodeIterator node_it ) const
  {
    auto node_idx = position( lvl_it, node_it );
    ++lvl_it;
    return std::make_pair( lvl_it, ConstBFSNodeIterator( node_idx * 2, &( *lvl_it ) ) );
  }

  /**
   * @brief Returns iterator to the left node of the given iterator
   *
   * @param it Iterator pointing to a level and node in the BisecTree
   * @return BFSIterator pointing to the left node
   */
  BFSIterator left( const BFSIterator& it ) { return left( it.first, it.second ); }

  /**
   * @brief Returns const iterator to the left node of the given const iterator
   *
   * @param it Const iterator pointing to a level and node in the BisecTree
   * @return ConstBFSIterator pointing to the left node
   */
  ConstBFSIterator left( const ConstBFSIterator& it ) const { return left( it.first, it.second ); }

  /**
   * @brief Returns iterator to the right node of the given level and node iterators
   *
   * @param lvl_it Iterator pointing to a level in the BisecTree
   * @param node_it Iterator pointing to a node in the level of lvl_it
   * @return BFSIterator pointing to the right node
   */
  BFSIterator right( BFSLevelIterator lvl_it, BFSNodeIterator node_it )
  {
    auto node_idx = position( lvl_it, node_it );
    ++lvl_it;
    return std::make_pair( lvl_it, BFSNodeIterator( node_idx * 2 + 1, &( *lvl_it ) ) );
  }

  /**
   * @brief Returns const iterator to the right node of the given level and node const iterators
   *
   * @param lvl_it Const iterator pointing to a level in the BisecTree
   * @param node_it Const iterator pointing to a node in the level of lvl_it
   * @return ConstBFSIterator pointing to the right node
   */
  ConstBFSIterator right( ConstBFSLevelIterator lvl_it, ConstBFSNodeIterator node_it ) const
  {
    auto node_idx = position( lvl_it, node_it );
    ++lvl_it;
    return std::make_pair( lvl_it, ConstBFSNodeIterator( node_idx * 2 + 1, &( *lvl_it ) ) );
  }

  /**
   * @brief Returns iterator to the right node of the given iterator
   *
   * @param it Iterator pointing to a level and node in the BisecTree
   * @return BFSIterator pointing to the right node
   */
  BFSIterator right( const BFSIterator& it ) { return right( it.first, it.second ); }

  /**
   * @brief Returns const iterator to the right node of the given const iterator
   *
   * @param it Const iterator pointing to a level and node in the BisecTree
   * @return ConstBFSIterator pointing to the right node
   */
  ConstBFSIterator right( const ConstBFSIterator& it ) const { return right( it.first, it.second ); }

  /**
   * @brief Returns number of nodes at the last level
   *
   * @return Number of nodes at the last level
   */
  size_t NumParts() const { return n_parts_; }

  /**
   * @brief Returns number of levels in the tree
   *
   * @return Number of levels in BisecTree
   */
  size_t NumLevels() const { return n_levels_; }

 private:
  /**
   * @brief Number of nodes at the last level
   */
  size_t n_parts_;

  /**
   * @brief Number of levels in the tree
   */
  size_t n_levels_;

  /**
   * @brief Tree data
   */
  axom::Array<axom::Array<T>> all_nodes_;
};

}  // end namespace redecomp

#endif /* SRC_REDECOMP_UTILS_BISECTREE_HPP_ */
