// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/search/InterfacePairFinder.hpp"

// Axom includes
#include "axom/slic.hpp"
#include "axom/primal.hpp"
#include "axom/spin.hpp"

// Tribol includes
#include "tribol/common/ExecModel.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/utils/Algorithm.hpp"

#include <cmath>
#include <memory>
#include <utility>

// Short aliases for frequently used Axom namespaces
namespace primal = axom::primal;
namespace spin = axom::spin;

namespace tribol {

/**
 * @brief Runtime-polymorphic interface for explicit coarse-search strategies
 *
 * Grid and BVH searches derive from this interface so the public dispatcher can select an implementation from runtime
 * configuration. Lazy Cartesian products bypass this interface because they do not allocate explicit pairs.
 */
class SearchBase {
 public:
  /** @brief Construct a coarse-search implementation. */
  SearchBase() = default;

  /** @brief Destroy a coarse-search implementation through the base class. */
  virtual ~SearchBase() = default;

  /**
   * @brief Find candidates from the first mesh for each second-mesh element
   *
   * @param [in] mesh1 Mesh whose elements provide candidate IDs.
   * @param [in] mesh2 Mesh whose elements are matched against @p mesh1.
   * @return Explicitly stored candidate element pairs.
   */
  virtual ArrayT<ElementPair> findInterfacePairs( const MeshData::Viewer& mesh1,
                                                  const MeshData::Viewer& mesh2 ) const = 0;
};

///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Finds coarse candidate pairs with an Axom implicit grid
 *
 * The search inserts inflated bounding boxes for the first mesh into a spatial index. It then queries the index with
 * each inflated second-mesh element box and stores every returned candidate pair.
 *
 * The spatial index is constructed and queried by findInterfacePairs(). Grid search currently executes sequentially.
 *
 * @tparam D Spatial dimension of the mesh coordinates.
 */
template <int D>
class GridSearch : public SearchBase {
 public:
  using BBox = primal::BoundingBox<RealT, D>;
  using PointT = primal::Point<RealT, D>;

  using ImplicitGridType = spin::ImplicitGrid<D, axom::SEQ_EXEC, int>;
  using SpaceVec = typename ImplicitGridType::SpaceVec;
  using SpatialBoundingBox = typename ImplicitGridType::SpatialBoundingBox;

  /**
   * @brief Construct a grid-search strategy
   *
   * @param [in] proximity_scale Element-size multiplier used to inflate each element box.
   * @param [in] allocator_id Allocator used for explicitly stored result pairs.
   */
  GridSearch( RealT proximity_scale, int allocator_id )
      : proximity_scale_( proximity_scale ), allocator_id_( allocator_id )
  {
  }

 private:
  /**
   * @brief Build and query an implicit grid for two meshes
   *
   * Empty input meshes leave the grid uninitialized because no query can produce a pair.
   *
   * @param [in] mesh1 Mesh whose element boxes are inserted into the grid.
   * @param [in] mesh2 Mesh whose element boxes query the grid.
   * @return Candidate pairs stored with the configured allocator.
   */
  ArrayT<ElementPair> findInterfacePairs( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const override
  {
    ArrayT<ElementPair> pairs( 0, 0, allocator_id_ );

    if ( mesh1.numberOfElements() == 0 || mesh2.numberOfElements() == 0 ) {
      return pairs;
    }

    // TODO: Determine whether this tolerance should scale with the mesh.
    const RealT bboxTolerance = 1e-6;

    // The caller supplies the effective scale, including any upstream LOR adjustment.
    auto e_binning_proximity_scale = proximity_scale_;

    // Cache first-mesh element boxes for grid construction and insertion.
    ArrayT<SpatialBoundingBox> mesh_bboxes1;
    mesh_bboxes1.reserve( mesh1.numberOfElements() );
    for ( int i = 0; i < mesh1.numberOfElements(); ++i ) {
      mesh_bboxes1.emplace_back( elementBoundingBox( mesh1, i ) );
    }

    // Estimate a grid resolution from the average inflated element-box extent:
    // 1. Inflate each box isotropically using its longest dimension.
    // 2. Average the box extents, assuming elements have comparable sizes.
    // 3. Choose approximately one cell per two average box widths.
    SpaceVec ranges;
    SpatialBoundingBox grid_bbox;
    for ( int i = 0; i < mesh1.numberOfElements(); ++i ) {
      auto& bbox = mesh_bboxes1[i];
      inflateBBox( bbox, e_binning_proximity_scale );

      ranges += bbox.range();

      // Accumulate the bounds of every indexed element.
      grid_bbox.addBox( bbox );
    }

    // Keep inserted boxes away from the grid boundary despite roundoff.
    grid_bbox.scale( 1 + bboxTolerance );

    ranges /= static_cast<double>( mesh1.numberOfElements() );

    // Compute each grid dimension from the corresponding average box extent.
    typename ImplicitGridType::GridCell resolution;
    SpaceVec bboxRange = grid_bbox.range();
    const RealT scaleFac = 0.5;  // TODO: Determine whether this factor should be mesh-dependent.
    for ( int i = 0; i < D; ++i ) {
      resolution[i] = static_cast<IndexT>( std::ceil( scaleFac * bboxRange[i] / ranges[i] ) );
    }

    // Initialize the spatial index and insert each first-mesh element box.
    ImplicitGridType grid;
    grid.initialize( grid_bbox, &resolution, mesh1.numberOfElements() );
    for ( int i = 0; i < mesh1.numberOfElements(); ++i ) {
      grid.insert( mesh_bboxes1[i], i );
    }

    // Report search-domain information when debug logging is enabled.
    SLIC_DEBUG( "Implicit Grid info: " << "\n Mesh 1 bounding box (inflated): " << grid_bbox
                                       << "\n Avg range: " << ranges << "\n Computed resolution: " << resolution );

    SpatialBoundingBox bbox2;
    for ( int i = 0; i < mesh2.numberOfElements(); ++i ) {
      bbox2.addBox( elementBoundingBox( mesh2, i ) );
    }

    SLIC_DEBUG( "Mesh 2 bounding box is: " << bbox2 );

    // Query first-mesh candidates for each second-mesh element.
    using BitsetType = typename ImplicitGridType::BitsetType;
    for ( int toIdx = 0; toIdx < mesh2.numberOfElements(); ++toIdx ) {
      SpatialBoundingBox bbox = elementBoundingBox( mesh2, toIdx );
      inflateBBox( bbox, e_binning_proximity_scale );

      auto candidateBits = grid.getCandidates( bbox );

      for ( IndexT fromIdx = candidateBits.find_first(); fromIdx != BitsetType::npos;
            fromIdx = candidateBits.find_next( fromIdx ) ) {
        // Retain only one ordering of same-mesh element pairs.
        if ( ( mesh1.meshId() == mesh2.meshId() ) && ( fromIdx < toIdx ) ) {
          continue;
        }

        // TODO: Reject candidates whose boxes share grid cells but do not overlap.

        pairs.emplace_back( fromIdx, toIdx );
      }
    }

    return pairs;
  }

  /**
   * @brief Compute the axis-aligned bounding box of one mesh element
   *
   * @param [in] mesh Mesh containing the element.
   * @param [in] eId Element index in @p mesh.
   * @return Minimal axis-aligned box containing the element nodes.
   */
  static BBox elementBoundingBox( const MeshData::Viewer& mesh, IndexT eId )
  {
    // These using directives support Axom releases before and after NumericArray moved into the axom::primal namespace
    // in Axom 0.10.0.
    using namespace axom;
    using namespace axom::primal;

    BBox box;

    for ( int i{ 0 }; i < mesh.numberOfNodesPerElement(); ++i ) {
      NumericArray<RealT, D> vert_array;
      auto vert_id = mesh.getGlobalNodeId( eId, i );
      for ( int d{ 0 }; d < D; ++d ) {
        vert_array[d] = mesh.getPosition()[d][vert_id];
      }
      box.addPoint( PointT( vert_array ) );
    }

    return box;
  }

  /**
   * @brief Expand a box isotropically based on its longest dimension
   *
   * @param [in,out] bbox Box to expand.
   * @param [in] range_multiplier Multiplier applied to the box's longest
   * extent to obtain the expansion distance.
   */
  static void inflateBBox( SpatialBoundingBox& bbox, RealT range_multiplier )
  {
    int d = bbox.getLongestDimension();
    const RealT expansionFac = range_multiplier * bbox.range()[d];
    bbox.expand( expansionFac );
  }

  /** Element-size multiplier used to inflate element boxes. */
  RealT proximity_scale_;

  /** Allocator used for explicitly stored result pairs. */
  int allocator_id_;
};

///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Finds coarse candidate pairs with an Axom bounding-volume hierarchy
 *
 * The search builds a BVH over expanded first-mesh element boxes and queries it
 * with expanded second-mesh element boxes. The BVH returns first-mesh element
 * indices grouped by second-mesh query element; findInterfacePairs() converts
 * that representation into explicit ElementPair values.
 *
 * @tparam D Spatial dimension of the mesh coordinates.
 * @tparam ExecSpace Axom execution space used to build and query the BVH.
 */
template <int D, class ExecSpace>
class BvhSearch : public SearchBase {
 public:
  using BVHT = axom::spin::BVH<D, ExecSpace, RealT>;
  using BoxT = typename BVHT::BoxType;
  using PointT = primal::Point<RealT, D>;
  using RayT = primal::Ray<RealT, D>;
  using VectorT = primal::Vector<RealT, D>;

  /**
   * @brief Construct a BVH-search strategy
   *
   * @param [in] execution_mode Runtime execution mode corresponding to
   * @p ExecSpace.
   * @param [in] allocator_id Allocator used for search work arrays and result
   * pairs.
   * @param [in] proximity_scale Element-size multiplier used to expand element
   * boxes along their face normals.
   */
  BvhSearch( ExecutionMode execution_mode, int allocator_id, RealT proximity_scale )
      : execution_mode_( execution_mode ), allocator_id_( allocator_id ), proximity_scale_( proximity_scale )
  {
  }

 private:
  /**
   * @brief Build and query a BVH for two meshes
   *
   * @param [in] mesh1 Mesh whose element boxes populate the BVH.
   * @param [in] mesh2 Mesh whose element boxes query the BVH.
   * @return Candidate pairs stored with the configured allocator.
   */
  ArrayT<ElementPair> findInterfacePairs( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const override
  {
    if ( mesh1.numberOfElements() == 0 || mesh2.numberOfElements() == 0 ) {
      return ArrayT<ElementPair>( 0, 0, allocator_id_ );
    }

    ArrayT<BoxT> boxes1( axom::ArrayOptions::Uninitialized{}, mesh1.numberOfElements(), mesh1.numberOfElements(),
                         allocator_id_ );
    ArrayT<BoxT> boxes2( axom::ArrayOptions::Uninitialized{}, mesh2.numberOfElements(), mesh2.numberOfElements(),
                         allocator_id_ );

    // The caller supplies the effective scale, including any upstream LOR adjustment.
    buildMeshBBoxes( boxes1, mesh1, proximity_scale_ );
    buildMeshBBoxes( boxes2, mesh2, proximity_scale_ );

    // Build the index over first-mesh element boxes.
    BVHT bvh;
    bvh.setAllocatorID( allocator_id_ );
    bvh.initialize( boxes1.view(), boxes1.size() );

    // Query all second-mesh boxes. Candidates contains flattened first-mesh
    // element IDs; offsets and counts map each query element to its candidates.
    ArrayT<IndexT> candidates( axom::ArrayOptions::Uninitialized{}, 0, 0, allocator_id_ );
    ArrayT<IndexT> offsets( axom::ArrayOptions::Uninitialized{}, mesh2.numberOfElements(), mesh2.numberOfElements(),
                            allocator_id_ );
    ArrayT<IndexT> counts( axom::ArrayOptions::Uninitialized{}, mesh2.numberOfElements(), mesh2.numberOfElements(),
                           allocator_id_ );
    auto counts_view = counts.view();
    auto offsets_view = offsets.view();
    bvh.findBoundingBoxes( offsets_view, counts_view, candidates, mesh2.numberOfElements(), boxes2.view() );

    // Convert the flattened BVH output into explicit element pairs in the
    // configured execution space.
    auto candidates_view = candidates.view();
    ArrayT<ElementPair> pairs( candidates.size(), candidates.size(), allocator_id_ );
    auto pairs_view = pairs.view();
    forAllExec<false>( execution_mode_, candidates.size(),
                       [candidates_view, offsets_view, counts_view, pairs_view] TRIBOL_HOST_DEVICE( IndexT i ) {
                         auto mesh1_elem = candidates_view[i];
                         auto mesh2_elem = algorithm::binarySearch( offsets_view, counts_view, i );
                         pairs_view[i] = ElementPair( mesh1_elem, mesh2_elem );
                       } );
    return pairs;
  }

  /**
   * @brief Build proximity-expanded element boxes for one mesh
   *
   * The unexpanded box contains the element nodes. The final box also contains
   * points offset from its centroid in both face-normal directions by the
   * scaled face radius.
   *
   * @param [out] boxes Preallocated output array with one box per element.
   * @param [in] mesh Mesh whose element boxes are built.
   * @param [in] binning_proximity Multiplier applied to each element's face
   * radius to obtain the normal expansion distance.
   */
  void buildMeshBBoxes( ArrayT<BoxT>& boxes, const MeshData::Viewer& mesh, RealT binning_proximity ) const
  {
    auto boxes_view = boxes.view();
    forAllExec<false>( execution_mode_, mesh.numberOfElements(),
                       [mesh, boxes_view, binning_proximity] TRIBOL_HOST_DEVICE( IndexT i ) {
                         BoxT box;
                         auto num_nodes_per_elem = mesh.numberOfNodesPerElement();
                         for ( IndexT j{ 0 }; j < num_nodes_per_elem; ++j ) {
                           IndexT node_id = mesh.getGlobalNodeId( i, j );
                           PointT pos;
                           for ( int d{ 0 }; d < D; ++d ) {
                             pos[d] = mesh.getPosition()[d][node_id];
                           }
                           box.addPoint( pos );
                         }

                         // Include proximity in both directions normal to the face.
                         RealT vnorm[3];
                         mesh.getFaceNormal( i, vnorm );
                         VectorT faceNormal( vnorm );
                         RealT faceRadius = mesh.getFaceRadius()[i];
                         expandBBoxNormal( box, faceNormal, binning_proximity * faceRadius );
                         boxes_view[i] = std::move( box );
                       } );
  }

  /**
   * @brief Expand a box in both directions along a face normal
   *
   * @param [in,out] bbox Box to expand.
   * @param [in] faceNormal Unit face normal defining the expansion direction.
   * @param [in] faceRadius Distance added in each normal direction.
   */
  TRIBOL_HOST_DEVICE static void expandBBoxNormal( BoxT& bbox, const VectorT& faceNormal, const RealT faceRadius )
  {
    PointT p0 = bbox.getCentroid();
    RayT outwardRay( p0, faceNormal );
    VectorT inwardNormal( faceNormal );
    inwardNormal *= -1.0;
    RayT inwardRay( p0, inwardNormal );
    PointT pout = outwardRay.at( faceRadius );
    PointT pin = inwardRay.at( faceRadius );
    bbox.addPoint( pout );
    bbox.addPoint( pin );
  }

  /** Runtime execution mode corresponding to @p ExecSpace. */
  ExecutionMode execution_mode_;

  /** Allocator for search work arrays and explicitly stored result pairs. */
  int allocator_id_;

  /** Element-size multiplier used for face-normal box expansion. */
  RealT proximity_scale_;
};

///////////////////////////////////////////////////////////////////////////////

InterfacePairFinder::InterfacePairFinder( BinningMethod binning_method, ExecutionMode execution_mode, int allocator_id,
                                          RealT proximity_scale )
    : binning_method_( binning_method ),
      execution_mode_( execution_mode ),
      allocator_id_( allocator_id ),
      proximity_scale_( proximity_scale )
{
  if ( isOnDevice( execution_mode_ ) && binning_method_ == BINNING_GRID ) {
    SLIC_WARNING_ROOT( "BINNING_GRID is not supported on GPU. Switching to BINNING_BVH." );
    binning_method_ = BINNING_BVH;
  }
}

ContactPairRange InterfacePairFinder::findInterfacePairs( MeshData& mesh1, MeshData& mesh2 ) const
{
  SLIC_DEBUG( "Searching for interface pairs" );
  const auto mesh1_view = mesh1.getView();
  const auto mesh2_view = mesh2.getView();
  const int dim = mesh1_view.spatialDimension();

  // Preserve the Cartesian product as a lazy range instead of allocating all
  // possible pairs.
  if ( binning_method_ == BINNING_CARTESIAN_PRODUCT ) {
    return CartesianPairView( mesh1_view.numberOfElements(), mesh2_view.numberOfElements(),
                              mesh1_view.meshId() == mesh2_view.meshId() );
  }

  std::unique_ptr<SearchBase> search;
  switch ( binning_method_ ) {
    case BINNING_CARTESIAN_PRODUCT:
      // Handled above to preserve the lazy representation.
      break;
    case BINNING_GRID:
      // Instantiate the grid implementation for the mesh dimension.
      switch ( dim ) {
        case 2:
          search = std::make_unique<GridSearch<2>>( proximity_scale_, allocator_id_ );
          break;
        case 3:
          search = std::make_unique<GridSearch<3>>( proximity_scale_, allocator_id_ );
          break;
        default:
          SLIC_ERROR_ROOT( "Invalid dimension: " << dim );
          break;
      }
      break;
    case BINNING_BVH:
      // Instantiate the BVH for both the mesh dimension and execution space.
      switch ( dim ) {
        case 2:
          switch ( execution_mode_ ) {
            case ( ExecutionMode::Sequential ):
              search =
                  std::make_unique<BvhSearch<2, axom::SEQ_EXEC>>( execution_mode_, allocator_id_, proximity_scale_ );
              break;
#ifdef TRIBOL_USE_OPENMP
            // TODO: Verify whether this instantiation still causes compiler hangs.
            case ( ExecutionMode::OpenMP ):
              search =
                  std::make_unique<BvhSearch<2, axom::OMP_EXEC>>( execution_mode_, allocator_id_, proximity_scale_ );
              break;
#endif
#ifdef TRIBOL_USE_CUDA
            case ( ExecutionMode::Cuda ):
              search = std::make_unique<BvhSearch<2, axom::CUDA_EXEC<TRIBOL_BLOCK_SIZE>>>(
                  execution_mode_, allocator_id_, proximity_scale_ );
              break;
#endif
#ifdef TRIBOL_USE_HIP
            case ( ExecutionMode::Hip ):
              search = std::make_unique<BvhSearch<2, axom::HIP_EXEC<TRIBOL_BLOCK_SIZE>>>(
                  execution_mode_, allocator_id_, proximity_scale_ );
              break;
#endif
            default:
              SLIC_ERROR_ROOT( "Invalid execution mode." );
              break;
          }
          break;
        case 3:
          switch ( execution_mode_ ) {
            case ( ExecutionMode::Sequential ):
              search =
                  std::make_unique<BvhSearch<3, axom::SEQ_EXEC>>( execution_mode_, allocator_id_, proximity_scale_ );
              break;
#ifdef TRIBOL_USE_OPENMP
            // TODO: Verify whether this instantiation still causes compiler hangs.
            case ( ExecutionMode::OpenMP ):
              search =
                  std::make_unique<BvhSearch<3, axom::OMP_EXEC>>( execution_mode_, allocator_id_, proximity_scale_ );
              break;
#endif
#ifdef TRIBOL_USE_CUDA
            case ( ExecutionMode::Cuda ):
              search = std::make_unique<BvhSearch<3, axom::CUDA_EXEC<TRIBOL_BLOCK_SIZE>>>(
                  execution_mode_, allocator_id_, proximity_scale_ );
              break;
#endif
#ifdef TRIBOL_USE_HIP
            case ( ExecutionMode::Hip ):
              search = std::make_unique<BvhSearch<3, axom::HIP_EXEC<TRIBOL_BLOCK_SIZE>>>(
                  execution_mode_, allocator_id_, proximity_scale_ );
              break;
#endif
            default:
              SLIC_ERROR_ROOT( "Invalid execution mode." );
              break;
          }
          break;
        default:
          SLIC_ERROR_ROOT( "Invalid dimension: " << dim );
          break;
      }
      break;
    default:
      SLIC_ERROR_ROOT( "Invalid binning method: " << binning_method_ );
      break;
  }
  if ( search != nullptr ) {
    return search->findInterfacePairs( mesh1_view, mesh2_view );
  }
  return ArrayT<ElementPair>( 0, 0, allocator_id_ );
}

}  // end namespace tribol
