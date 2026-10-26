// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_SEARCH_COARSESEARCHFACTORY_HPP_
#define SRC_TRIBOL_SEARCH_COARSESEARCHFACTORY_HPP_

#include <memory>

#include "tribol/common/Parameters.hpp"
#include "tribol/search/CoarseSearch.hpp"

namespace tribol {

/**
 * @brief Construct a two- or three-dimensional Grid or BVH coarse search
 *
 * Grid search requested on a device falls back to BVH. Cartesian-product binning is unsupported because its legacy
 * implementation fuses candidate generation with formulation-specific filtering.
 *
 * @param spatial_dimension Spatial dimension of the meshes (2 or 3)
 * @param binning_method Spatial binning algorithm
 * @param execution_mode Execution mode used by BVH; Grid runs sequentially
 * @param binning_proximity_scale Element-size multiplier used to expand coarse-search bounds
 * @param residual_gap Additional physical distance added to each bounding-box expansion
 */
std::unique_ptr<CoarseSearch> createCoarseSearch( int spatial_dimension, BinningMethod binning_method,
                                                  ExecutionMode execution_mode, RealT binning_proximity_scale,
                                                  RealT residual_gap );

}  // namespace tribol

#endif /* SRC_TRIBOL_SEARCH_COARSESEARCHFACTORY_HPP_ */
