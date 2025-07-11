// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)
#ifndef SRC_TRIBOL_MESH_PROCESSINGUNIT_HPP_
#define SRC_TRIBOL_MESH_PROCESSINGUNIT_HPP_

// Tribol includes
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

template <typename BinningT, typename MethodT>
class ProcessingUnit {
 public:
  ProcessingUnit( const MeshData& mesh1, const MeshData& mesh2, const Parameters& params );
  auto findPairs();
  template <typename PairListT>
  void applyPhysics( const PairListT& pair_list );
  RealT computeTimestep();
  void writeOutput();

 private:
  BinningT pair_finder_;
  MethodT method_;
};  // end class ProcessingUnit

} /* namespace tribol */

#endif /* SRC_TRIBOL_MESH_PROCESSINGUNIT_HPP_ */
