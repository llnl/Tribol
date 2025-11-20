// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "RCB.hpp"

#include "axom/slic.hpp"

#include "shared/infrastructure/Profiling.hpp"

#include "redecomp/common/TypeDefs.hpp"
#include "redecomp/utils/ArrayUtility.hpp"
#include "redecomp/utils/MPIUtility.hpp"

namespace redecomp {

template <int NDIMS>
RCB<NDIMS>::RCB( const MPI_Comm& comm, double max_out_of_balance, int n_try_new_axis )
    : PartitionMethod<NDIMS>{ comm }, max_out_of_balance_{ max_out_of_balance }, n_try_new_axis_{ n_try_new_axis }
{
}

template <int NDIMS>
std::vector<EntityIndexByRank> RCB<NDIMS>::generatePartitioning(
    int n_parts, const std::vector<axom::Array<Point<NDIMS>>>& coords_by_mesh, double ghost_len ) const
{
  TRIBOL_MARK_FUNCTION;
  auto partitioning = std::vector<EntityIndexByRank>();
  partitioning.reserve( coords_by_mesh.size() );

  // Build a partitioning using recursive coordinate bisection
  auto problem_tree = BuildProblemTree( n_parts, coords_by_mesh, ghost_len );

  // Ensures ranks with entities are evenly distributed throughout the partition. E.g. if 10 MPI ranks are allocated and
  // n_parts == 2, then put entities on rank 0 and rank 5.
  auto rank_jump_interval = this->getMPIUtility().NRanks() / n_parts;

  for ( const auto& coords : coords_by_mesh ) {
    // Build a list of coords that belong in each of the RCB entity parts
    auto ent_idx = MPIArray<int>( &this->getMPIUtility() );
    auto ent_ghost = MPIArray<bool>( &this->getMPIUtility() );
    for ( int i{ 0 }; i < n_parts; ++i ) {
      auto rank = i * rank_jump_interval;
      ent_idx[rank].reserve( coords.size() );
      ent_ghost[rank].reserve( coords.size() );
    }
    for ( int i{ 0 }; i < coords.size(); ++i ) {
      auto dest = DetermineDomain( problem_tree, coords[i] );
      auto rank = dest * rank_jump_interval;
      ent_idx[rank].push_back( i );
      ent_ghost[rank].push_back( false );
      const auto& neighbors = problem_tree( dest ).neighbor_bboxes_;
      for ( int j{ 0 }; j < neighbors.size(); ++j ) {
        if ( problem_tree( neighbors[j] ).ghost_bbox_.contains( coords[i] ) ) {
          auto neighbor_rank = neighbors[j] * rank_jump_interval;
          ent_idx[neighbor_rank].push_back( i );
          ent_ghost[neighbor_rank].push_back( true );
        }
      }
    }
    for ( int i{ 0 }; i < n_parts; ++i ) {
      ent_idx[i].shrink();
      ent_ghost[i].shrink();
    }

    partitioning.emplace_back( std::move( ent_idx ), std::move( ent_ghost ) );
  }

  return partitioning;
}

template <int NDIMS>
BisecTree<RCBInfo<NDIMS>> RCB<NDIMS>::BuildProblemTree( int n_parts,
                                                        const std::vector<axom::Array<Point<NDIMS>>>& coords_by_mesh,
                                                        double ghost_len ) const
{
  TRIBOL_MARK_FUNCTION;
  // subdivide the domain into n_parts pieces.  create a bisection tree of the
  // domain so we can focus on one piece at a time.
  auto problem_tree = BisecTree<RCBInfo<NDIMS>>( n_parts );

  // store the desired fraction of each piece in the lowest level of the tree
  {
    auto base_problem_frac = 1.0 / static_cast<double>( n_parts );
    auto lvl_it = --problem_tree.end();
    for ( auto node_it = problem_tree.begin( lvl_it ); node_it != problem_tree.end( lvl_it ); ++node_it ) {
      ( *node_it ).desired_frac_ = base_problem_frac;
    }

    // define desired_frac_ for each node by summing up the children
    // (reverse breadth-first traversal).  should sum to 1 for the root node.
    for ( ; lvl_it != problem_tree.begin(); ) {
      --lvl_it;
      for ( auto node_it = problem_tree.begin( lvl_it ); node_it != problem_tree.end( lvl_it ); ++node_it ) {
        // there always will be a left child
        ( *node_it ).desired_frac_ += ( *problem_tree.left( lvl_it, node_it ).second ).desired_frac_;
        // there may not be a right child
        auto right_it = problem_tree.right( lvl_it, node_it );
        if ( right_it.second != problem_tree.end( right_it.first ) ) {
          ( *node_it ).desired_frac_ += ( *right_it.second ).desired_frac_;
        }
      }
    }
  }

  // compute total number of entities in the domain
  auto total_ents = 0;
  for ( const auto& coords : coords_by_mesh ) {
    total_ents += coords.size();
  }
  total_ents = TotalEntities( total_ents );
  // construct an AABB of the whole domain in the root node
  ( *problem_tree.begin() )[0].bbox_ = DomainBoundingBox( coords_by_mesh );

  // start splitting.  overview of the nested loops:
  // 1) breadth-first traversal of the bisection tree
  // 2) coordinate-axis of the cut (in case the nodes are aligned along an axis)
  // 3) cut coordinate
  for ( auto lvl_it = problem_tree.begin(); lvl_it != --problem_tree.end(); ++lvl_it ) {
    for ( auto node_it = problem_tree.begin( lvl_it ); node_it != problem_tree.end( lvl_it ); ++node_it ) {
      auto left_it = problem_tree.left( lvl_it, node_it );
      auto right_it = problem_tree.right( lvl_it, node_it );
      // loop shortcut if no right child exists
      if ( right_it.second == problem_tree.end( right_it.first ) ) {
        ( *left_it.second ).bbox_ = ( *node_it ).bbox_;
        ( *left_it.second ).actual_frac_ = ( *node_it ).actual_frac_;
        ( *left_it.second ).ghost_bbox_ = ( *node_it ).ghost_bbox_;
        continue;
      }

      // compute proportion of node fractions
      auto left_frac = ( *left_it.second ).desired_frac_;
      auto left_prop = left_frac / ( left_frac + ( *right_it.second ).desired_frac_ );

      // rank coordinate axes by cut desirability (largest AABB length to smallest)
      auto bbox_range = ( *node_it ).bbox_.range();
      auto best_axes = ArrayUtility::IndexArray<int, NDIMS>();
      std::stable_sort( best_axes.begin(), best_axes.end(),
                        [&bbox_range]( int i, int j ) { return bbox_range[i] > bbox_range[j]; } );

      auto axis_ok = false;
      auto actual_left_prop = 0.0;
      auto node_max_out_of_balance = 1.0;
      auto total_node_ents = 0.0;
      for ( auto cut_ax : best_axes ) {
        auto bbox_min = ( *node_it ).bbox_.getMin();
        auto bbox_max = ( *node_it ).bbox_.getMax();
        auto left_max = bbox_max;
        auto right_min = bbox_min;
        // cut the axis by the proportion of node fractions
        auto min_coord = bbox_min[cut_ax];
        auto max_coord = bbox_max[cut_ax];
        auto cut_coord = min_coord * ( 1.0 - left_prop ) + max_coord * left_prop;

        // use bisection method to find a cut coordinate within the relative tolerance mesh_max_out_of_balance
        auto prev_actual_left_prop = 0.0;
        int same_prop_ct{ 0 };
        while ( true ) {
          left_max[cut_ax] = cut_coord;
          right_min[cut_ax] = cut_coord;
          ( *left_it.second ).bbox_ = BoundingBox<NDIMS>( bbox_min, left_max );
          ( *right_it.second ).bbox_ = BoundingBox<NDIMS>( right_min, bbox_max );

          auto n_ents = CountEntities( ( *left_it.second ).bbox_, ( *right_it.second ).bbox_, coords_by_mesh );
          total_node_ents = static_cast<double>( n_ents.first + n_ents.second );
          actual_left_prop = static_cast<double>( n_ents.first ) / total_node_ents;
          node_max_out_of_balance = 1.0;
          if ( total_node_ents > 0.0 ) {
            node_max_out_of_balance = std::max( max_out_of_balance_, 2.0 / total_node_ents );
          }
          if ( actual_left_prop < left_prop - node_max_out_of_balance ) {
            // cut coordinate needs to be increased
            min_coord = cut_coord;
          } else if ( actual_left_prop > left_prop + node_max_out_of_balance ) {
            // cut coordinate needs to be reduced
            max_coord = cut_coord;
          } else {
            // cut coordinate is ok if we get here
            axis_ok = true;
            ( *left_it.second ).actual_frac_ = static_cast<double>( n_ents.first ) / static_cast<double>( total_ents );
            ( *right_it.second ).actual_frac_ =
                static_cast<double>( n_ents.second ) / static_cast<double>( total_ents );
            // set ghost bounding boxes
            for ( int d{ 0 }; d < NDIMS; ++d ) {
              bbox_min[d] -= ghost_len;
              left_max[d] += ghost_len;
              right_min[d] -= ghost_len;
              bbox_max[d] += ghost_len;
            }
            ( *left_it.second ).ghost_bbox_ = BoundingBox<NDIMS>( bbox_min, left_max );
            ( *right_it.second ).ghost_bbox_ = BoundingBox<NDIMS>( right_min, bbox_max );
            break;
          }
          // check if we should try a different axis
          if ( prev_actual_left_prop == actual_left_prop ) ++same_prop_ct;
          if ( same_prop_ct == n_try_new_axis_ ) break;
          prev_actual_left_prop = actual_left_prop;

          // try a new cut coordinate
          cut_coord = 0.5 * min_coord + 0.5 * max_coord;
        }

        if ( axis_ok ) break;
      }
      // none of the axes worked.  issue a debug statement but continue.
      SLIC_DEBUG_ROOT_IF( !axis_ok,
                          axom::fmt::format( "RCB domain decomposition unsuccessful.\n"
                                             "  Max out of balance tolerance: {}\n"
                                             "  Total entities to split: {}\n"
                                             "  Proportion of entities on left of cut: {}\n"
                                             "  Desired proportion of entites on left of cut: {}\n",
                                             node_max_out_of_balance, total_node_ents, actual_left_prop, left_prop ) );
    }
  }

  // solve for neighboring bounding boxes (used for testing for ghost elements)
  // TODO: is there a better (non-O(n^2)) way to do this?
  for ( int i{ 0 }; i < n_parts; ++i ) {
    for ( int j{ 0 }; j < i; ++j ) {
      // shared root OR overlap
      if ( ( j == ( i - 1 ) && ( i + j + 1 ) / 2 % 2 == 1 ) ||
           problem_tree( i ).ghost_bbox_.intersectsWith( problem_tree( j ).ghost_bbox_ ) ) {
        problem_tree( i ).neighbor_bboxes_.push_back( j );
        problem_tree( j ).neighbor_bboxes_.push_back( i );
      }
    }
  }

  return problem_tree;
}

template <int NDIMS>
BoundingBox<NDIMS> RCB<NDIMS>::DomainBoundingBox( const std::vector<axom::Array<Point<NDIMS>>>& coords_by_mesh ) const
{
  auto on_rank_bbox = BoundingBox<NDIMS>();
  size_t num_coords{ 0 };
  for ( const auto& coords : coords_by_mesh ) {
    on_rank_bbox.addBox( BoundingBox<NDIMS>( coords.data(), coords.size() ) );
    num_coords += coords.size();
  }
  auto min_coord = on_rank_bbox.getMin();
  auto max_coord = on_rank_bbox.getMax();

  if ( num_coords == 0 ) {
    on_rank_bbox.clear();
  }

  // reduce to all ranks
  this->getMPIUtility().Allreduce( &min_coord.array(), MPI_MIN );
  this->getMPIUtility().Allreduce( &max_coord.array(), MPI_MAX );

  return BoundingBox<NDIMS>( min_coord, max_coord );
}

template <int NDIMS>
std::pair<int, int> RCB<NDIMS>::CountEntities( const BoundingBox<NDIMS>& left_bbox,
                                               const BoundingBox<NDIMS>& right_bbox,
                                               const std::vector<axom::Array<Point<NDIMS>>>& coords_by_mesh ) const
{
  auto n_ents = std::make_pair( 0, 0 );

  for ( const auto& coords : coords_by_mesh ) {
    for ( const auto& coord : coords ) {
      if ( left_bbox.contains( coord ) )
        n_ents.first += 1;
      else if ( right_bbox.contains( coord ) )
        n_ents.second += 1;
    }
  }

  // reduce to all ranks
  n_ents.first = this->getMPIUtility().AllreduceValue( n_ents.first, MPI_SUM );
  n_ents.second = this->getMPIUtility().AllreduceValue( n_ents.second, MPI_SUM );

  return n_ents;
}

template <int NDIMS>
int RCB<NDIMS>::DetermineDomain( const BisecTree<RCBInfo<NDIMS>>& problem_tree, const Point<NDIMS>& coord ) const
{
  auto it = std::make_pair( problem_tree.begin(), problem_tree.begin( problem_tree.begin() ) );
  for ( size_t i{ 0 }; i < ( problem_tree.NumLevels() - 1 ); ++i ) {
    auto left_it = problem_tree.left( it );
    if ( ( *left_it.second ).bbox_.contains( coord ) ) {
      it = left_it;
    } else {
      it = problem_tree.right( it );
    }
  }
  return problem_tree.position( it );
}

template <int NDIMS>
int RCB<NDIMS>::TotalEntities( int n_local_ents ) const
{
  return this->getMPIUtility().AllreduceValue( n_local_ents, MPI_SUM );
}

template class RCB<2>;
template class RCB<3>;

}  // end namespace redecomp
