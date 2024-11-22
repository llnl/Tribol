// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_GEOM_NORMAL_HPP_
#define SRC_GEOM_NORMAL_HPP_

#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"

namespace tribol
{

class NodalNormal
{
public:
  virtual ~NodalNormal() {}
  virtual void Compute(MeshData& mesh) = 0;
  MethodData& getJacobianData() { return elem_jacobians_; }
  const MethodData& getJacobianData() const { return elem_jacobians_; }
private:
  MethodData elem_jacobians_;
};

class ElementAvgNormal : public NodalNormal
{
  void Compute(MeshData& mesh) override;
};

#ifdef TRIBOL_USE_ENZYME

class VertexAvgNormal : public NodalNormal
{
public:
  VertexAvgNormal(bool compute_deriv = true) : compute_deriv_( compute_deriv ) {}
  void Compute(MeshData& mesh) override;
private:
  bool compute_deriv_;
};

// free functions for enzyme
void ElementVertexAvgNormal(const RealT* x, RealT* n, int num_nodes_per_elem);
void ElementVertexAvgNormalJacobian(const RealT* x, RealT* n, RealT* dndx, int num_nodes_per_elem);

#endif
  
}

#endif /* SRC_GEOM_NORMAL_HPP_ */
