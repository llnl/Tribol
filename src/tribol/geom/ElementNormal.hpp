// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_GEOM_ELEMENTNORMAL_HPP_
#define SRC_GEOM_ELEMENTNORMAL_HPP_

#include <functional>

#include "tribol/common/BasicTypes.hpp"

namespace tribol {

class ElementNormal {
 public:
  TRIBOL_HOST_DEVICE virtual ~ElementNormal() {}
  TRIBOL_HOST_DEVICE virtual bool Compute( const RealT* x, const RealT* c, RealT* n, int num_nodes, RealT& area ) const = 0;
};

class PalletAvgNormal : public ElementNormal {
 public:
  TRIBOL_HOST_DEVICE bool Compute( const RealT* x, const RealT* c, RealT* n, int num_nodes, RealT& area ) const override;
};

class QuadCentroidNormal : public ElementNormal {
 public:
  TRIBOL_HOST_DEVICE bool Compute( const RealT* x, const RealT* c, RealT* n, int num_nodes, RealT& area ) const override;
};

}  // namespace tribol

#endif /* SRC_GEOM_ELEMENTNORMAL_HPP_ */
