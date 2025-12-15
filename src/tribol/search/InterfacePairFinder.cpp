// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/search/InterfacePairFinder.hpp"

#include "tribol/common/ExecModel.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/geom/GeomUtilities.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/utils/Algorithm.hpp"
#include "tribol/utils/Math.hpp"

#include "axom/slic.hpp"
#include "axom/primal.hpp"
#include "axom/spin.hpp"

// Define some namespace aliases to help with axom usage
namespace primal = axom::primal;
namespace spin = axom::spin;

namespace tribol {

/*!
 *  Perform geometry/proximity checks 1-4
 */
TRIBOL_HOST_DEVICE bool geomFilter( const CouplingScheme::Viewer& cs_view, IndexT element_id1, IndexT element_id2 )
{
  auto& mesh1 = cs_view.getMesh1View();
  auto& mesh2 = cs_view.getMesh2View();
  bool auto_contact_check = cs_view.getParameters().auto_contact_check;
  // we want binning proximity scaled by LOR factor on HO meshes, i.e. the effective binning proximity
  auto element_radius_multiplier = cs_view.getEffectiveBinningProximityScale();
  auto mode = cs_view.getContactMode();

  /// CHECK #1: Check to make sure the two face ids are not the same
  ///           and the two mesh ids are not the same.
  if ( ( mesh1.meshId() == mesh2.meshId() ) && ( element_id1 == element_id2 ) ) {
    return false;
  }

  int dim = mesh1.spatialDimension();

  /// CHECK #2: Auto-contact precludes faces that share a common
  ///           node(s). We want to preclude two adjacent faces from interacting
  ///           due to problematic configurations, such as corners where the
  ///           configuration and opposing normals appear to be in contact, but
  ///           are not.
  ///
  ///           Note: non-auto-contact coupling schemes should typically be amongst
  ///                 topologically disconnected surfaces unless it is known apriori that
  ///                 face-pairs with shared nodes can in fact contact.
  if ( auto_contact_check ) {
    for ( IndexT i{ 0 }; i < mesh1.numberOfNodesPerElement(); ++i ) {
      int node1 = mesh1.getGlobalNodeId( element_id1, i );
      for ( IndexT j{ 0 }; j < mesh2.numberOfNodesPerElement(); ++j ) {
        int node2 = mesh2.getGlobalNodeId( element_id2, j );
        if ( node1 == node2 ) {
          return false;
        }
      }
    }
  }

  /// CHECK #3: Check that face normals are opposing up to some tolerance.
  ///           This uses a hard coded normal tolerance for this check.
  RealT nrmlTol = -0.173648177;  // taken as cos(100) between face pair

  RealT nrmlCheck = 0.0;
  for ( int d{ 0 }; d < dim; ++d ) {
    nrmlCheck += mesh1.getElementNormals()[d][element_id1] * mesh2.getElementNormals()[d][element_id2];
  }

  // check normal projection against tolerance
  if ( nrmlCheck > nrmlTol ) {
    return false;
  }

  /// CHECK #4: Perform radius check, which involves seeing if
  ///           the distance between the two face vertex averaged
  ///           centroid is less than the sum of the two face radii
  ///           premultiplied by a binning scale factor.
  ///           The face radii are taken to be the magnitude of the
  ///           longest vector from that face's vertex averaged
  ///           centroid to one its nodes.
  RealT offset_tol = 0.05;
  if ( dim == 3 ) {
    RealT r1 = mesh1.getFaceRadius()[element_id1];
    RealT r2 = mesh2.getFaceRadius()[element_id2];

    // set maximum offset of face centroids for inclusion
    RealT distMax = element_radius_multiplier * ( r1 + r2 );  // default is sum of face radii

    // check if the contact mode is conforming, in which case the
    // faces are supposed to be aligned
    if ( mode == SURFACE_TO_SURFACE_CONFORMING ) {
      // use 5% of max face radius for conforming case as
      // tolerance on face offsets
      distMax *= offset_tol;
    }

    // compute the distance between the two face centroids
    RealT distX = mesh2.getElementCentroids()[0][element_id2] - mesh1.getElementCentroids()[0][element_id1];
    RealT distY = mesh2.getElementCentroids()[1][element_id2] - mesh1.getElementCentroids()[1][element_id1];
    RealT distZ = mesh2.getElementCentroids()[2][element_id2] - mesh1.getElementCentroids()[2][element_id1];

    RealT distMag = magnitude( distX, distY, distZ );

    if ( distMag > ( distMax ) ) {
      return false;
    }
  }  // end of dim == 3
  else if ( dim == 2 ) {
    // get 1/2 edge length off the mesh data
    RealT e1 = 0.5 * mesh1.getElementAreas()[element_id1];
    RealT e2 = 0.5 * mesh2.getElementAreas()[element_id2];

    RealT distMax = element_radius_multiplier * ( e1 + e2 );

    // check if the contact mode is conforming, in which case the
    // edges are supposed to be aligned
    if ( mode == SURFACE_TO_SURFACE_CONFORMING ) {
      // use 5% of max face radius for conforming case as
      // tolerance on face offsets
      distMax *= offset_tol;
    }

    // compute the distance between the two edge centroids
    RealT distX = mesh2.getElementCentroids()[0][element_id2] - mesh1.getElementCentroids()[0][element_id1];
    RealT distY = mesh2.getElementCentroids()[1][element_id2] - mesh1.getElementCentroids()[1][element_id1];

    RealT distMag = magnitude( distX, distY );

    // include faces where separation equals distMax
    if ( distMag > ( distMax ) ) {
      return false;
    }
  }  // end of dim == 2

  /// Check #5: Check to see if there is a positive area of overlap when both faces/edges
  ///           are projected onto an intermediate plane
  if ( cs_view.pruneMethodFacePair( element_id1, element_id2 ) ) {
    return false;
  }

  // if we made it here we passed all checks
  return true;

}  // end geomFilter()

/*!
 * \brief Base class to compute the candidate pairs for a coupling scheme
 *
 * \a initialize() must be called prior to \a findInterfacePairs()
 *
 */
class SearchBase {
 public:
  SearchBase() {};
  virtual ~SearchBase() {};
  /*!
   * Prepares the object for spatial searches
   */
  virtual void initialize() = 0;

  /*!
   * Find candidates in first mesh for each element in second mesh of coupling scheme.
   */
  virtual void findInterfacePairs() = 0;
};

///////////////////////////////////////////////////////////////////////////////

/*!
 * \brief Helper class to compute the candidate pairs for a coupling scheme
 *
 * A CartesianProduct search combines each element from the first mesh of
 * the coupling scheme with each element in the second mesh. A geometry filter
 * is then applied to each resulting element pair.  This is the slowest of all
 * pair-finding methods since ALL possible element pairs are considered, i.e.,
 * this is an exhaustive search.
 *
 * \tparam D The spatial dimension of the coupling scheme mesh vertices.
 */
template <int D>
class CartesianProduct : public SearchBase {
 public:
  /*!
   * Constructs a CartesianProduct instance over CouplingScheme \a couplingScheme
   * \pre couplingScheme is not null
   */
  CartesianProduct( CouplingScheme* couplingScheme ) : m_coupling_scheme( couplingScheme ) {}

  void initialize() override {}

  void findInterfacePairs() override
  {
    const auto mesh1 = m_coupling_scheme->getMesh1().getView();
    IndexT mesh1NumElems = mesh1.numberOfElements();

    const auto mesh2 = m_coupling_scheme->getMesh2().getView();
    IndexT mesh2NumElems = mesh2.numberOfElements();

    // Reserve memory for boolean array indicating which pairs are proximate
    IndexT maxNumPairs = mesh1NumElems * mesh2NumElems;
    bool is_symm = m_coupling_scheme->getMeshId1() == m_coupling_scheme->getMeshId2();
    if ( is_symm ) {
      // account for symmetry: the max number of pairs when the meshes are the
      // same is the upper triangular portion of the cartesian product pair
      // matrix
      maxNumPairs = mesh1NumElems * ( mesh1NumElems + 1 ) / 2;
    }
    ArrayT<bool> proximityArray( maxNumPairs, maxNumPairs, m_coupling_scheme->getAllocatorId() );
    bool* isProximate = proximityArray.data();

    // Allocate memory for a counter
    ArrayT<int> countArray( 1, 1, m_coupling_scheme->getAllocatorId() );
    int* pCount = countArray.data();

    const auto cs_view = m_coupling_scheme->getView();
    // count how many pairs are proximate
    forAllExec( m_coupling_scheme->getExecutionMode(), maxNumPairs,
                [cs_view, mesh1NumElems, mesh2NumElems, is_symm, isProximate, pCount] TRIBOL_HOST_DEVICE( IndexT i ) {
                  IndexT fromIdx = i / mesh2NumElems;
                  IndexT toIdx = i % mesh2NumElems;
                  if ( is_symm ) {
                    IndexT row = algorithm::symmMatrixRow( i, mesh1NumElems );
                    IndexT offset = row * ( row + 1 ) / 2;
                    fromIdx = row;
                    toIdx = i - offset;
                  }
                  isProximate[i] = geomFilter( cs_view, fromIdx, toIdx );
#ifdef TRIBOL_USE_RAJA
                  RAJA::atomicAdd<RAJA::auto_atomic>( pCount, static_cast<int>( isProximate[i] ) );
#else
                  if ( isProximate[i] ) {
                    ++( *pCount );
                  }
#endif
                } );

    ArrayT<int, 1, MemorySpace::Host> countArray_host( countArray );
    SLIC_INFO( "Found " << countArray_host[0] << " proximate pairs" );

    // allocate proximate pairs array
    auto& contactPairs = m_coupling_scheme->getInterfacePairs();
    contactPairs.resize( countArray_host[0] );

    countArray.fill( 0 );
    auto pairs_view = m_coupling_scheme->getInterfacePairs().view();
    // fill proximate pairs array
    forAllExec(
        m_coupling_scheme->getExecutionMode(), maxNumPairs,
        [isProximate, pCount, pairs_view, mesh1NumElems, mesh2NumElems, is_symm] TRIBOL_HOST_DEVICE( IndexT i ) {
          // Filtering removed this case
          if ( !isProximate[i] ) {
            return;
          }

          IndexT fromIdx = i / mesh2NumElems;
          IndexT toIdx = i % mesh2NumElems;
          if ( is_symm ) {
            IndexT row = algorithm::symmMatrixRow( i, mesh1NumElems );
            IndexT offset = row * ( row + 1 ) / 2;
            fromIdx = row;
            toIdx = i - offset;
          }

      // get unique index for the array
#ifdef TRIBOL_USE_RAJA
          auto idx = RAJA::atomicInc<RAJA::auto_atomic>( pCount );
#else
          auto idx = *pCount;
          ++( *pCount );
#endif

          pairs_view[idx] = InterfacePair( fromIdx, toIdx, true );
        } );

    SLIC_INFO( "Coupling scheme has " << contactPairs.size() << " pairs out of a maximum possible of " << maxNumPairs
                                      << "." );
  }

 private:
  CouplingScheme* m_coupling_scheme;
};  // End of CartesianProduct definition

///////////////////////////////////////////////////////////////////////////////

/*!
 * \brief Implicit Grid helper class to compute the candidate pairs for a coupling scheme
 *
 * A GridSearch indexes the elements from the first mesh of the coupling scheme
 * in a spatial index that requires element bounding boxes. Then, for each of
 * the elements in the second mesh, we find proximate faces and add them to the
 * coupling scheme's list of candidate pairs.
 *
 * The spatial index is generated in \a initialize()
 * and the search is performed in \a findInterfacePairs()
 *
 * \tparam D The spatial dimension of the coupling scheme mesh vertices.
 */
template <int D>
class GridSearch : public SearchBase {
 public:
  using BBox = primal::BoundingBox<RealT, D>;
  using PointT = primal::Point<RealT, D>;

  using ImplicitGridType = spin::ImplicitGrid<D, axom::SEQ_EXEC, int>;
  using SpacePoint = typename ImplicitGridType::SpacePoint;
  using SpaceVec = typename ImplicitGridType::SpaceVec;
  using SpatialBoundingBox = typename ImplicitGridType::SpatialBoundingBox;

  /*!
   * Constructs a GridSearch instance over CouplingScheme \a couplingScheme
   * \pre couplingScheme is not null
   */
  GridSearch( CouplingScheme* couplingScheme )
      : m_coupling_scheme( couplingScheme ),
        m_mesh1( m_coupling_scheme->getMesh1().getView() ),
        m_mesh2( m_coupling_scheme->getMesh2().getView() )
  {
  }

  /*!
   * Constructs spatial index over elements of coupling scheme's first mesh
   */
  void initialize() override
  {
    // TODO does this tolerance need to scale with the mesh?
    const RealT bboxTolerance = 1e-6;

    m_coupling_scheme->getInterfacePairs().clear();
    // we want binning proximity scaled by LOR factor on HO meshes, i.e. the effective binning proximity
    auto e_binning_proximity_scale = m_coupling_scheme->getEffectiveBinningProximityScale();

    // if either mesh is empty, don't initialize because...
    // 1) there won't be any pairs
    // 2) there is some division by the number of elements below
    if ( m_mesh1.numberOfElements() == 0 || m_mesh2.numberOfElements() == 0 ) {
      return;
    }

    // Find the bounding boxes of the elements in the first mesh
    // Store them in an array for efficient reuse
    m_gridBBox.clear();
    m_meshBBoxes1.reserve( m_mesh1.numberOfElements() );
    for ( int i = 0; i < m_mesh1.numberOfElements(); ++i ) {
      m_meshBBoxes1.emplace_back( elementBoundingBox( m_mesh1, i ) );
    }

    // Find an appropriate resolution for the spatial index grid
    //
    // (Note KW) This implementation is a bit ad-hoc
    // * Inflate bounding boxes by proximity scale * longest dimension to avoid zero-width dimensions
    // * Find the average extents (range) of the boxes Assumption is that elements are roughly the same size
    // * Grid resolution for each dimension is overall box width divided by half the average element width
    SpaceVec ranges;
    for ( int i = 0; i < m_mesh1.numberOfElements(); ++i ) {
      auto& bbox = m_meshBBoxes1[i];
      inflateBBox( bbox, e_binning_proximity_scale );

      ranges += bbox.range();

      // build up overall bounding box along the way
      m_gridBBox.addBox( bbox );
    }

    // inflate grid box slightly so elem bounding boxes are not on grid bdry
    m_gridBBox.scale( 1 + bboxTolerance );

    ranges /= static_cast<double>( m_mesh1.numberOfElements() );

    // Compute grid resolution from average bbox size
    typename ImplicitGridType::GridCell resolution;
    SpaceVec bboxRange = m_gridBBox.range();
    const RealT scaleFac = 0.5;  // TODO is this mesh dependent?
    for ( int i = 0; i < D; ++i ) {
      resolution[i] = static_cast<IndexT>( std::ceil( scaleFac * bboxRange[i] / ranges[i] ) );
    }

    // Next, initialize the ImplicitGrid
    m_grid.initialize( m_gridBBox, &resolution, m_mesh1.numberOfElements() );

    // Finally, insert the elements
    for ( int i = 0; i < m_mesh1.numberOfElements(); ++i ) {
      m_grid.insert( m_meshBBoxes1[i], i );
    }

    // Output some info for debugging
    if ( true ) {
      SLIC_DEBUG( "Implicit Grid info: " << "\n Mesh 1 bounding box (inflated): " << m_gridBBox
                                         << "\n Avg range: " << ranges << "\n Computed resolution: " << resolution );

      SpatialBoundingBox bbox2;
      for ( int i = 0; i < m_mesh2.numberOfElements(); ++i ) {
        bbox2.addBox( elementBoundingBox( m_mesh2, i ) );
      }

      SLIC_DEBUG( "Mesh 2 bounding box is: " << bbox2 );
    }
  };  // end initialize()

  /*!
   * Use the spatial index to find candidates in first mesh for each
   * element in second mesh of coupling scheme.
   */
  void findInterfacePairs() override
  {
    using BitsetType = typename ImplicitGridType::BitsetType;

    // Extract some mesh metadata from coupling scheme
    const auto mesh1 = m_coupling_scheme->getMesh1().getView();
    const auto mesh2 = m_coupling_scheme->getMesh2().getView();
    auto& contactPairs = m_coupling_scheme->getInterfacePairs();
    // we want binning proximity scaled by LOR factor on HO meshes, i.e. the effective binning proximity
    auto e_binning_proximity_scale = m_coupling_scheme->getEffectiveBinningProximityScale();

    // Find matches in first mesh (with index 'fromIdx')
    // with candidate elements in second mesh (with index 'toIdx')
    // int k = 0;  // Debug only
    for ( int toIdx = 0; toIdx < m_mesh2.numberOfElements(); ++toIdx ) {
      SpatialBoundingBox bbox = elementBoundingBox( m_mesh2, toIdx );
      inflateBBox( bbox, e_binning_proximity_scale );

      // Query the mesh
      auto candidateBits = m_grid.getCandidates( bbox );

      // Add candidates
      for ( IndexT fromIdx = candidateBits.find_first(); fromIdx != BitsetType::npos;
            fromIdx = candidateBits.find_next( fromIdx ) ) {
        // if meshId1 = meshId2, then check to make sure fromIdx < toIdx
        // so we don't double count
        if ( ( mesh1.meshId() == mesh2.meshId() ) && ( fromIdx < toIdx ) ) {
          continue;
        }

        // TODO: Add extra filter by bbox

        // Preliminary geometry/proximity checks, SRW
        bool contact = geomFilter( m_coupling_scheme->getView(), fromIdx, toIdx );

        if ( contact ) {
          contactPairs.emplace_back( fromIdx, toIdx, true );
          // SLIC_INFO("Interface pair " << k << " = " << toIdx << ", " << fromIdx);  // Debug only
          // ++k;  // Debug only
        }
      }
    }  // end of loop over candidates in second mesh

  }  // end findInterfacePairs()

 private:
  BBox elementBoundingBox( const MeshData::Viewer& mesh, IndexT eId )
  {
    // NOTE: namespace for NumericArray changed in axom 0.10.0. The using directives below allow Tribol to work with
    // older and newer versions of axom.
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
  /*!
   * Expands bounding box by range_multiplier * the longest dimension's range
   */
  void inflateBBox( SpatialBoundingBox& bbox, RealT range_multiplier )
  {
    int d = bbox.getLongestDimension();
    const RealT expansionFac = range_multiplier * bbox.range()[d];
    bbox.expand( expansionFac );
  }

  CouplingScheme* m_coupling_scheme;
  const MeshData::Viewer m_mesh1;
  const MeshData::Viewer m_mesh2;

  ImplicitGridType m_grid;
  SpatialBoundingBox m_gridBBox;
  ArrayT<SpatialBoundingBox> m_meshBBoxes1;

};  // End of GridSearch class definition

///////////////////////////////////////////////////////////////////////////////

/*!
 * \brief BVH helper class to compute the candidate pairs for a coupling scheme
 *
 * A BvhSearch constructs a BVH tree from the elements of the first mesh using
 * element bounding boxes. Then, for each of the elements in the second mesh, we
 * traverse the BVH tree, find proximate faces, and add them to the coupling
 * scheme's list of candidate pairs.
 *
 * The search is performed in \a findInterfacePairs()
 *
 * \tparam D The spatial dimension of the coupling scheme mesh vertices.
 */
template <int D, class ExecSpace>
class BvhSearch : public SearchBase {
 public:
  using BVHT = axom::spin::BVH<D, ExecSpace, RealT>;
  using BoxT = typename BVHT::BoxType;
  using PointT = primal::Point<RealT, D>;
  using RayT = primal::Ray<RealT, D>;
  using VectorT = primal::Vector<RealT, D>;
  using AtomicPolicy = typename axom::execution_space<ExecSpace>::atomic_policy;

  /*!
   * Constructs a BvhSearch instance over CouplingScheme \a couplingScheme
   * \pre couplingScheme is not null
   */
  BvhSearch( CouplingScheme* coupling_scheme )
      : m_coupling_scheme( coupling_scheme ),
        m_mesh1( m_coupling_scheme->getMesh1().getView() ),
        m_mesh2( m_coupling_scheme->getMesh2().getView() ),
        m_boxes1( axom::ArrayOptions::Uninitialized{}, m_mesh1.numberOfElements(), m_mesh1.numberOfElements(),
                  m_coupling_scheme->getAllocatorId() ),
        m_boxes2( axom::ArrayOptions::Uninitialized{}, m_mesh2.numberOfElements(), m_mesh2.numberOfElements(),
                  m_coupling_scheme->getAllocatorId() ),
        m_candidates( axom::ArrayOptions::Uninitialized{}, 0, 0, m_coupling_scheme->getAllocatorId() ),
        m_offsets( axom::ArrayOptions::Uninitialized{}, m_mesh2.numberOfElements(), m_mesh2.numberOfElements(),
                   m_coupling_scheme->getAllocatorId() ),
        m_counts( axom::ArrayOptions::Uninitialized{}, m_mesh2.numberOfElements(), m_mesh2.numberOfElements(),
                  m_coupling_scheme->getAllocatorId() )
  {
  }

  /*!
   * Allocate and fill bounding box arrays for each of the two meshes
   */
  void initialize() override
  {
    // we want binning proximity scaled by LOR factor on HO meshes, i.e. the effective binning proximity
    buildMeshBBoxes( m_boxes1, m_coupling_scheme->getMesh1().getView(),
                     m_coupling_scheme->getEffectiveBinningProximityScale() );
    // we want binning proximity scaled by LOR factor on HO meshes, i.e. the effective binning proximity
    buildMeshBBoxes( m_boxes2, m_coupling_scheme->getMesh2().getView(),
                     m_coupling_scheme->getEffectiveBinningProximityScale() );
  }  // end initialize()

  /*!
   * Use the BVH to find candidates in first mesh for each
   * element in second mesh of coupling scheme.
   */
  void findInterfacePairs() override
  {
    // Build the BVH
    BVHT bvh;
    bvh.setAllocatorID( m_coupling_scheme->getAllocatorId() );
    bvh.initialize( m_boxes1.view(), m_boxes1.size() );

    // Search for intersecting bounding boxes
    auto counts_view = m_counts.view();
    auto offsets_view = m_offsets.view();
    bvh.findBoundingBoxes( offsets_view, counts_view, m_candidates, m_mesh2.numberOfElements(), m_boxes2.view() );

    // Apply geom filter to check if intersecting bounding boxes are proximate
    // Change candidate value to -1 if geom filter checks are failed
    auto candidates_view = m_candidates.view();
    // array of size 1 to track the number of candidates in a way compatible
    // with device kernels
    ArrayT<IndexT> filtered_candidates_data( 1, 1, m_coupling_scheme->getAllocatorId() );
    auto filtered_candidates = filtered_candidates_data.view();
    const auto cs_view = m_coupling_scheme->getView();
    // count the number of filtered proximate pairs
    forAllExec(
        m_coupling_scheme->getExecutionMode(), m_candidates.size(),
        [cs_view, offsets_view, counts_view, candidates_view, filtered_candidates] TRIBOL_HOST_DEVICE( IndexT i ) {
          auto mesh1_elem = candidates_view[i];
          auto mesh2_elem = algorithm::binarySearch( offsets_view, counts_view, i );
          if ( geomFilter( cs_view, mesh1_elem, mesh2_elem ) ) {
#ifdef TRIBOL_USE_RAJA
            RAJA::atomicInc<AtomicPolicy>( filtered_candidates.data() );
#else
            ++filtered_candidates[0];
#endif
          } else {
            candidates_view[i] = -1;
          }
        } );

    ArrayT<IndexT, 1, MemorySpace::Host> filtered_candidates_host( filtered_candidates_data );
    m_coupling_scheme->getInterfacePairs().resize( filtered_candidates_host[0] );
    filtered_candidates_data.fill( 0 );

    auto pairs_view = m_coupling_scheme->getInterfacePairs().view();
    // add filtered pairs to interface pairs array
    forAllExec(
        m_coupling_scheme->getExecutionMode(), m_candidates.size(),
        [candidates_view, offsets_view, counts_view, filtered_candidates, pairs_view] TRIBOL_HOST_DEVICE( IndexT i ) {
          // Filtering removed this case
          if ( candidates_view[i] == -1 ) {
            return;
          }

          auto mesh1_elem = candidates_view[i];
          auto mesh2_elem = algorithm::binarySearch( offsets_view, counts_view, i );

      // get unique index for the array
#ifdef TRIBOL_USE_RAJA
          auto idx = RAJA::atomicInc<AtomicPolicy>( filtered_candidates.data() );
#else
          auto idx = filtered_candidates[0];
          ++filtered_candidates[0];
#endif

          pairs_view[idx] = InterfacePair( mesh1_elem, mesh2_elem, true );
        } );
  }  // end findInterfacePairs()

  void buildMeshBBoxes( ArrayT<BoxT>& boxes, const MeshData::Viewer& mesh, RealT binning_proximity )
  {
    auto boxes_view = boxes.view();
    forAllExec( m_coupling_scheme->getExecutionMode(), mesh.numberOfElements(),
                [this, mesh, boxes_view, binning_proximity] TRIBOL_HOST_DEVICE( IndexT i ) {
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
                  // Expand the bounding box in the face normal direction
                  RealT vnorm[3];
                  mesh.getFaceNormal( i, vnorm );
                  VectorT faceNormal( vnorm );
                  RealT faceRadius = mesh.getFaceRadius()[i];
                  expandBBoxNormal( box, faceNormal, binning_proximity * faceRadius );
                  boxes_view[i] = std::move( box );
                } );
  }

 private:
  /*!
   * Expands bounding box by projecting the face normal by a distance
   * equal to the effective face radius
   */
  TRIBOL_HOST_DEVICE void expandBBoxNormal( BoxT& bbox, const VectorT& faceNormal, const RealT faceRadius )
  {
    PointT p0 = bbox.getCentroid();
    RayT outwardRay( p0, faceNormal );
    VectorT inwardNormal( faceNormal );
    inwardNormal *= -1.0;  // this operation is available on device
    RayT inwardRay( p0, inwardNormal );
    PointT pout = outwardRay.at( faceRadius );
    PointT pin = inwardRay.at( faceRadius );
    bbox.addPoint( pout );
    bbox.addPoint( pin );
  }

  /*!
   * Isotropically expands bounding box by the effective face radius.
   */
  TRIBOL_HOST_DEVICE void inflateBBox( BoxT& bbox, const RealT faceRadius ) { bbox.expand( faceRadius ); }

  CouplingScheme* m_coupling_scheme;
  const MeshData::Viewer m_mesh1;
  const MeshData::Viewer m_mesh2;
  ArrayT<BoxT> m_boxes1;
  ArrayT<BoxT> m_boxes2;
  ArrayT<IndexT> m_candidates;
  ArrayT<IndexT> m_offsets;
  ArrayT<IndexT> m_counts;
};  // End of BvhSearch class definition

///////////////////////////////////////////////////////////////////////////////

InterfacePairFinder::InterfacePairFinder( CouplingScheme* cs ) : m_coupling_scheme( cs )
{
  SLIC_ASSERT_MSG( cs != nullptr, "Coupling scheme was invalid (null pointer)" );
  const int dim = m_coupling_scheme->spatialDimension();
  m_search = nullptr;

  if ( isOnDevice( cs->getExecutionMode() ) && cs->getBinningMethod() == BINNING_GRID ) {
    SLIC_WARNING_ROOT( "BINNING_GRID is not supported on GPU. Switching to BINNING_BVH." );
    cs->setBinningMethod( BINNING_BVH );
  }

  switch ( cs->getBinningMethod() ) {
    case BINNING_CARTESIAN_PRODUCT:
      switch ( dim ) {
        case 2:
          m_search = new CartesianProduct<2>( m_coupling_scheme );
          break;
        case 3:
          m_search = new CartesianProduct<3>( m_coupling_scheme );
          break;
        default:
          SLIC_ERROR_ROOT( "Invalid dimension: " << dim );
          break;
      }  // end of BINNING_CARTESIAN_PRODUCT dimension switch
      break;
    case BINNING_GRID:
      // The spatial grid is templated on the dimension
      switch ( dim ) {
        case 2:
          m_search = new GridSearch<2>( m_coupling_scheme );
          break;
        case 3:
          m_search = new GridSearch<3>( m_coupling_scheme );
          break;
        default:
          SLIC_ERROR_ROOT( "Invalid dimension: " << dim );
          break;
      }  // end of BINNING_GRID dimension switch
      break;
    case BINNING_BVH:
      // The BVH is templated on the dimension and execution space
      switch ( dim ) {
        case 2:
          switch ( cs->getExecutionMode() ) {
            case ( ExecutionMode::Sequential ):
              m_search = new BvhSearch<2, axom::SEQ_EXEC>( m_coupling_scheme );
              break;
#ifdef TRIBOL_USE_OPENMP
            case ( ExecutionMode::OpenMP ):  // This causes compiler to hang (EBC: Check if this is still true)
              m_search = new BvhSearch<2, axom::OMP_EXEC>( m_coupling_scheme );
              break;
#endif
#ifdef TRIBOL_USE_CUDA
            case ( ExecutionMode::Cuda ):
              m_search = new BvhSearch<2, axom::CUDA_EXEC<TRIBOL_BLOCK_SIZE>>( m_coupling_scheme );
              break;
#endif
#ifdef TRIBOL_USE_HIP
            case ( ExecutionMode::Hip ):
              m_search = new BvhSearch<2, axom::HIP_EXEC<TRIBOL_BLOCK_SIZE>>( m_coupling_scheme );
              break;
#endif
            default:
              SLIC_ERROR_ROOT( "Invalid execution mode." );
              break;
          }
          break;
        case 3:
          switch ( cs->getExecutionMode() ) {
            case ( ExecutionMode::Sequential ):
              m_search = new BvhSearch<3, axom::SEQ_EXEC>( m_coupling_scheme );
              break;
#ifdef TRIBOL_USE_OPENMP
            case ( ExecutionMode::OpenMP ):  // This causes compiler to hang (EBC: Check if this is still true)
              m_search = new BvhSearch<3, axom::OMP_EXEC>( m_coupling_scheme );
              break;
#endif
#ifdef TRIBOL_USE_CUDA
            case ( ExecutionMode::Cuda ):
              m_search = new BvhSearch<3, axom::CUDA_EXEC<TRIBOL_BLOCK_SIZE>>( m_coupling_scheme );
              break;
#endif
#ifdef TRIBOL_USE_HIP
            case ( ExecutionMode::Hip ):
              m_search = new BvhSearch<3, axom::HIP_EXEC<TRIBOL_BLOCK_SIZE>>( m_coupling_scheme );
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
      }  // end of BINNING_BVH dimension switch
      break;
    default:
      SLIC_ERROR_ROOT( "Invalid binning method: " << cs->getBinningMethod() );
      break;
  }  // end of binning method switch
}

InterfacePairFinder::~InterfacePairFinder()
{
  if ( m_search != nullptr ) {
    delete m_search;
  }
}

void InterfacePairFinder::initialize()
{
  SLIC_ASSERT( m_search != nullptr );
  m_search->initialize();
}

void InterfacePairFinder::findInterfacePairs()
{
  SLIC_DEBUG( "Searching for interface pairs" );
  m_search->findInterfacePairs();
  // set boolean on coupling scheme object indicating
  // that binning has occurred
  m_coupling_scheme->setBinned( true );
}

}  // end namespace tribol
