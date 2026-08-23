// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/search/InterfacePairFinder.hpp"

#include "axom/slic.hpp"

#include "tribol/common/LoopExec.hpp"
#include "tribol/search/BvhSearch.hpp"
#include "tribol/search/CartesianProductSearch.hpp"
#include "tribol/search/GridSearch.hpp"

namespace tribol {
namespace {

// Dispatch a BVH search to the execution space selected by the legacy API.
template <int Dimension>
ArrayT<ElementPair> findBvhPairs( ExecutionMode execution_mode, int allocator_id, RealT proximity_scale,
                                  const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  switch ( execution_mode ) {
    case ExecutionMode::Sequential:
      return BvhSearch<Dimension, axom::SEQ_EXEC>( proximity_scale, allocator_id ).findPairs( mesh1, mesh2 );
#ifdef TRIBOL_USE_OPENMP
    case ExecutionMode::OpenMP:
      return BvhSearch<Dimension, axom::OMP_EXEC>( proximity_scale, allocator_id ).findPairs( mesh1, mesh2 );
#endif
#ifdef TRIBOL_USE_CUDA
    case ExecutionMode::Cuda:
      return BvhSearch<Dimension, axom::CUDA_EXEC<TRIBOL_BLOCK_SIZE>>( proximity_scale, allocator_id )
          .findPairs( mesh1, mesh2 );
#endif
#ifdef TRIBOL_USE_HIP
    case ExecutionMode::Hip:
      return BvhSearch<Dimension, axom::HIP_EXEC<TRIBOL_BLOCK_SIZE>>( proximity_scale, allocator_id )
          .findPairs( mesh1, mesh2 );
#endif
    default:
      SLIC_ERROR_ROOT( "Invalid execution mode." );
      return ArrayT<ElementPair>( 0, 0, allocator_id );
  }
}

}  // namespace

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
  const int dimension = mesh1_view.spatialDimension();

  switch ( binning_method_ ) {
    case BINNING_CARTESIAN_PRODUCT:
      return CartesianProductSearch().findPairs( mesh1_view, mesh2_view );
    case BINNING_GRID:
      switch ( dimension ) {
        case 2:
          return GridSearch<2>( proximity_scale_, allocator_id_ ).findPairs( mesh1_view, mesh2_view );
        case 3:
          return GridSearch<3>( proximity_scale_, allocator_id_ ).findPairs( mesh1_view, mesh2_view );
        default:
          SLIC_ERROR_ROOT( "Invalid dimension: " << dimension );
          break;
      }
      break;
    case BINNING_BVH:
      switch ( dimension ) {
        case 2:
          return findBvhPairs<2>( execution_mode_, allocator_id_, proximity_scale_, mesh1_view, mesh2_view );
        case 3:
          return findBvhPairs<3>( execution_mode_, allocator_id_, proximity_scale_, mesh1_view, mesh2_view );
        default:
          SLIC_ERROR_ROOT( "Invalid dimension: " << dimension );
          break;
      }
      break;
    default:
      SLIC_ERROR_ROOT( "Invalid binning method: " << binning_method_ );
      break;
  }

  return ArrayT<ElementPair>( 0, 0, allocator_id_ );
}

}  // namespace tribol
