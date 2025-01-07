// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_COMMON_TYPEDEFS_HPP_
#define SRC_REDECOMP_COMMON_TYPEDEFS_HPP_

#include <utility>
#include <cmath>

#include "axom/primal.hpp"

#include "redecomp/utils/MPIArray.hpp"

namespace redecomp {
constexpr double pi = M_PI;

template <int NDIMS>
using Point = axom::primal::Point<double, NDIMS>;

template <int NDIMS>
using BoundingBox = axom::primal::BoundingBox<double, NDIMS>;

/**
 * @brief In its usage in redecomp, the first pair entry contains a list of
 * entity (node or element) indices per rank. The second pair entry is sized
 * the same as the first pair entry, and if the entry is true, it denotes the
 * entity is in the ghost layer on a corresponding RedecompMesh. This data
 * structure lists entities that need to be sent to or are received from other
 * MPI ranks.
 */
using EntityIndexByRank = std::pair<MPIArray<int>, MPIArray<bool>>;
}  // end namespace redecomp

#endif /* SRC_REDECOMP_COMMON_TYPEDEFS_HPP_ */
