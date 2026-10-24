// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/search/CoarseSearchFactory.hpp"

#include "tribol/common/ExecModel.hpp"
#include "tribol/search/BvhSearch.hpp"
#include "tribol/search/GridSearch.hpp"

namespace tribol {

namespace {

/** @brief Select a coarse-search specialization for a compile-time spatial dimension */
template <int SpatialDimension>
std::unique_ptr<CoarseSearch> createCoarseSearchForDimension( BinningMethod binning_method,
                                                              ExecutionMode execution_mode,
                                                              RealT binning_proximity_scale, RealT residual_gap )
{
  if ( binning_method == BINNING_GRID && isOnDevice( execution_mode ) ) {
    SLIC_WARNING_ROOT( "BINNING_GRID is not supported on GPU. Switching coarse search to BINNING_BVH." );
    binning_method = BINNING_BVH;
  }

  switch ( binning_method ) {
    case BINNING_GRID:
      return std::make_unique<GridSearch<SpatialDimension>>( binning_proximity_scale, residual_gap );
    case BINNING_BVH:
      switch ( execution_mode ) {
        case ExecutionMode::Sequential:
          return std::make_unique<BvhSearch<SpatialDimension, ExecutionMode::Sequential>>( binning_proximity_scale,
                                                                                           residual_gap );
#ifdef TRIBOL_USE_OPENMP
        case ExecutionMode::OpenMP:
          return std::make_unique<BvhSearch<SpatialDimension, ExecutionMode::OpenMP>>( binning_proximity_scale,
                                                                                       residual_gap );
#endif
#ifdef TRIBOL_USE_CUDA
        case ExecutionMode::Cuda:
          return std::make_unique<BvhSearch<SpatialDimension, ExecutionMode::Cuda>>( binning_proximity_scale,
                                                                                     residual_gap );
#endif
#ifdef TRIBOL_USE_HIP
        case ExecutionMode::Hip:
          return std::make_unique<BvhSearch<SpatialDimension, ExecutionMode::Hip>>( binning_proximity_scale,
                                                                                    residual_gap );
#endif
        default:
          SLIC_ERROR_ROOT( "BVH search cannot run with the configured execution mode." );
          return nullptr;
      }
    case BINNING_CARTESIAN_PRODUCT:
      SLIC_ERROR_ROOT( "Coarse search does not support BINNING_CARTESIAN_PRODUCT." );
      return nullptr;
    default:
      SLIC_ERROR_ROOT( "Coarse search received an invalid binning method." );
      return nullptr;
  }
}

}  // namespace

std::unique_ptr<CoarseSearch> createCoarseSearch( int spatial_dimension, BinningMethod binning_method,
                                                  ExecutionMode execution_mode, RealT binning_proximity_scale,
                                                  RealT residual_gap )
{
  switch ( spatial_dimension ) {
    case 2:
      return createCoarseSearchForDimension<2>( binning_method, execution_mode, binning_proximity_scale, residual_gap );
    case 3:
      return createCoarseSearchForDimension<3>( binning_method, execution_mode, binning_proximity_scale, residual_gap );
    default:
      SLIC_ERROR_ROOT( "Invalid coarse-search spatial dimension: " << spatial_dimension );
      return nullptr;
  }
}

}  // namespace tribol
