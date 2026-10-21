// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_GRIDSEARCH_HPP_
#define SRC_TRIBOL_SEARCH_GRIDSEARCH_HPP_

#include <cmath>

#include "axom/primal.hpp"
#include "axom/slic.hpp"
#include "axom/spin.hpp"

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

/**
 * @brief Uses an implicit grid to find spatially proximate element pairs
 *
 * GridSearch performs only coarse spatial binning. It intentionally does not apply method-specific normal, overlap, or
 * contact checks. Callers decide which returned candidates are appropriate for their contact formulation.
 *
 * The search stores non-owning mesh views. The MeshData objects and their caller-owned arrays must remain valid for the
 * lifetime of this object. Each query builds and releases its grid and bounding boxes so moving coordinates cannot
 * leave stale spatial data on the search object. GridSearch is sequential because Axom's implicit grid uses
 * axom::SEQ_EXEC.
 *
 * @tparam SpatialDimension Spatial dimension of the mesh coordinates
 */
template <int SpatialDimension>
class GridSearch {
 public:
  using BoundingBox = axom::primal::BoundingBox<RealT, SpatialDimension>;
  using Point = axom::primal::Point<RealT, SpatialDimension>;
  using ImplicitGrid = axom::spin::ImplicitGrid<SpatialDimension, axom::SEQ_EXEC, int>;
  using SpaceVector = typename ImplicitGrid::SpaceVec;
  using SpatialBoundingBox = typename ImplicitGrid::SpatialBoundingBox;

  /**
   * @brief Store the meshes and proximity values used by each one-shot grid search
   *
   * @param mesh1 Elements inserted into the implicit grid
   * @param mesh2 Elements used to query the implicit grid
   * @param binning_proximity_scale Multiplier applied to each element's longest bounding-box dimension
   * @param residual_gap Additional physical distance added to each bounding-box expansion
   */
  GridSearch( MeshData::Viewer mesh1, MeshData::Viewer mesh2, RealT binning_proximity_scale, RealT residual_gap )
      : mesh1_( mesh1 ),
        mesh2_( mesh2 ),
        binning_proximity_scale_( binning_proximity_scale ),
        residual_gap_( residual_gap )
  {
    SLIC_ASSERT_MSG( mesh1_.getAllocatorId() == mesh2_.getAllocatorId(),
                     "GridSearch meshes must use the same allocator." );
  }

  /**
   * @brief Build an implicit grid and return the spatial candidates found for the second mesh
   *
   * Bounding-box candidates can still include self-pairs, shared-node pairs, co-oriented faces, and pairs that fail a
   * contact method's overlap checks. Keeping those decisions outside this class allows each formulation to apply the
   * geometry checks it requires.
   */
  ArrayT<InterfacePair> findCandidatePairs() const
  {
    constexpr RealT grid_boundary_scale = 1.0e-6;
    constexpr RealT grid_resolution_scale = 0.5;

    ArrayT<InterfacePair> candidate_pairs( 0, 0, mesh1_.getAllocatorId() );
    // Empty meshes have no candidates and cannot define a useful grid resolution.
    if ( mesh1_.numberOfElements() == 0 || mesh2_.numberOfElements() == 0 ) {
      return candidate_pairs;
    }

    // Search storage is local because moving meshes require the grid and bounds to be rebuilt for every query.
    ImplicitGrid grid;
    SpatialBoundingBox grid_bounding_box;
    ArrayT<SpatialBoundingBox> mesh1_bounding_boxes( 0, mesh1_.numberOfElements(), mesh1_.getAllocatorId() );
    for ( IndexT element = 0; element < mesh1_.numberOfElements(); ++element ) {
      mesh1_bounding_boxes.emplace_back( elementBoundingBox( mesh1_, element ) );
    }

    SpaceVector average_ranges;
    for ( IndexT element = 0; element < mesh1_.numberOfElements(); ++element ) {
      auto& bounding_box = mesh1_bounding_boxes[element];
      inflateBoundingBox( bounding_box );
      average_ranges += bounding_box.range();
      grid_bounding_box.addBox( bounding_box );
    }

    // Moving the grid boundary away from element boundaries avoids ambiguous cell insertion at roundoff scale.
    grid_bounding_box.scale( 1.0 + grid_boundary_scale );
    average_ranges /= static_cast<RealT>( mesh1_.numberOfElements() );

    typename ImplicitGrid::GridCell resolution;
    const SpaceVector grid_range = grid_bounding_box.range();
    for ( int dimension = 0; dimension < SpatialDimension; ++dimension ) {
      resolution[dimension] =
          static_cast<IndexT>( std::ceil( grid_resolution_scale * grid_range[dimension] / average_ranges[dimension] ) );
    }

    grid.initialize( grid_bounding_box, &resolution, mesh1_.numberOfElements() );
    for ( IndexT element = 0; element < mesh1_.numberOfElements(); ++element ) {
      grid.insert( mesh1_bounding_boxes[element], element );
    }

    SLIC_DEBUG( "Implicit Grid info: " << "\n Mesh 1 bounding box (inflated): " << grid_bounding_box << "\n Avg range: "
                                       << average_ranges << "\n Computed resolution: " << resolution );

    using Bitset = typename ImplicitGrid::BitsetType;
    for ( IndexT mesh2_element = 0; mesh2_element < mesh2_.numberOfElements(); ++mesh2_element ) {
      SpatialBoundingBox bounding_box = elementBoundingBox( mesh2_, mesh2_element );
      inflateBoundingBox( bounding_box );

      auto candidate_bits = grid.getCandidates( bounding_box );
      for ( IndexT mesh1_element = candidate_bits.find_first(); mesh1_element != Bitset::npos;
            mesh1_element = candidate_bits.find_next( mesh1_element ) ) {
        // A same-mesh search visits both pair directions. Retain one direction while leaving self-pair policy to the
        // method-specific filter.
        if ( mesh1_.meshId() == mesh2_.meshId() && mesh1_element < mesh2_element ) {
          continue;
        }
        candidate_pairs.emplace_back( InterfacePair{ mesh1_element, mesh2_element } );
      }
    }

    return candidate_pairs;
  }

 private:
  /** @brief Bound every finite-element node so quadratic edge and interior nodes participate in coarse search */
  static BoundingBox elementBoundingBox( const MeshData::Viewer& mesh, IndexT element )
  {
    BoundingBox bounding_box;
    for ( IndexT local_node = 0; local_node < mesh.numberOfNodesPerElement(); ++local_node ) {
      Point position;
      const IndexT node = mesh.getGlobalNodeId( element, local_node );
      for ( int dimension = 0; dimension < SpatialDimension; ++dimension ) {
        position[dimension] = mesh.getPosition()[dimension][node];
      }
      bounding_box.addPoint( position );
    }
    return bounding_box;
  }

  /** @brief Expand a box by the configured fraction of its longest dimension and the residual gap */
  void inflateBoundingBox( SpatialBoundingBox& bounding_box ) const
  {
    const int longest_dimension = bounding_box.getLongestDimension();
    const RealT expansion = binning_proximity_scale_ * bounding_box.range()[longest_dimension] + residual_gap_;
    bounding_box.expand( expansion );
  }

  MeshData::Viewer mesh1_;
  MeshData::Viewer mesh2_;
  RealT binning_proximity_scale_;
  RealT residual_gap_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_GRIDSEARCH_HPP_ */
