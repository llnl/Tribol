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
  virtual ~ElementNormal() {}
  virtual std::function<bool( const RealT*, const RealT*, RealT*, int, RealT& )> NormalFunction() = 0;
};

class PalletAvgNormal : public ElementNormal {
 public:
  std::function<bool( const RealT*, const RealT*, RealT*, int, RealT& )> NormalFunction() override;
};

class QuadCentroidNormal : public ElementNormal {
 public:
  std::function<bool( const RealT*, const RealT*, RealT*, int, RealT& )> NormalFunction() override;
};

}  // namespace tribol

#endif /* SRC_GEOM_ELEMENTNORMAL_HPP_ */
