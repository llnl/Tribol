// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_ELEMENTFORCEANDGAP_HPP_
#define SRC_TRIBOL_PHYSICS_ELEMENTFORCEANDGAP_HPP_

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/integ/IntegrationRule.hpp"

#include "mfem.hpp"

#include <vector>
#include <memory>

namespace tribol {

/**
 * @brief Base class for element-level force and gap policies.
 *
 * This class defines the interface for policies that compute contact gaps,
 * forces, and their derivatives at the element level. Concrete policies
 * (e.g., MortarPenalty, MortarLM) should implement these methods.
 *
 * @tparam PointwiseGapAndNormal Policy for pointwise gap and normal calculation.
 */
template <typename PointwiseGapAndNormal, int Dim>
class ElementForceAndGap {
 public:
  virtual ~ElementForceAndGap() = default;

  /**
   * @brief Computes the contact gap and its derivatives for an element pair.
   *
   * @param pair The interface pair.
   * @param integration_rule Integration rule used to compute points.
   * @param gap_method Pointwise gap and normal computation method.
   * @param coords1 Nodal coordinates of the first element (Mesh 1).
   * @param coords2 Nodal coordinates of the second element (Mesh 2).
   * @param gaps Output vector for nodal gaps on Mesh 2.
   * @param dg_dx1 Output matrix for gap derivatives w.r.t. Mesh 1 displacement.
   * @param dg_dx2 Output matrix for gap derivatives w.r.t. Mesh 2 displacement.
   */
  virtual void computeGap( const InterfacePair& pair, const IntegrationRule<PointwiseGapAndNormal, Dim>& integration_rule,
                           PointwiseGapAndNormal& gap_method, const std::vector<RealT>& coords1,
                           const std::vector<RealT>& coords2, mfem::Vector& gaps, mfem::DenseMatrix& dg_dx1,
                           mfem::DenseMatrix& dg_dx2 ) = 0;

  /**
   * @brief Computes the contact gap for an element pair without derivatives.
   *
   * @param pair The interface pair.
   * @param integration_rule Integration rule used to compute points.
   * @param gap_method Pointwise gap and normal computation method.
   * @param coords1 Nodal coordinates of the first element (Mesh 1).
   * @param coords2 Nodal coordinates of the second element (Mesh 2).
   * @param gaps Output vector for nodal gaps on Mesh 2.
   */
  virtual void computeGap( const InterfacePair& pair, const IntegrationRule<PointwiseGapAndNormal, Dim>& integration_rule,
                           PointwiseGapAndNormal& gap_method, const std::vector<RealT>& coords1,
                           const std::vector<RealT>& coords2, mfem::Vector& gaps ) = 0;

  /**
   * @brief Computes contact forces and their derivatives for an element pair.
   *
   * @param pair The interface pair.
   * @param integration_rule Integration rule used to compute points.
   * @param gap_method Pointwise gap and normal computation method.
   * @param coords1 Nodal coordinates of the first element (Mesh 1).
   * @param coords2 Nodal coordinates of the second element (Mesh 2).
   * @param pressures Nodal pressures (optional).
   * @param f1 Output vector for contact forces on Mesh 1.
   * @param f2 Output vector for contact forces on Mesh 2.
   * @param df1_dx1 Output matrix for f1 derivatives w.r.t. Mesh 1 displacement.
   * @param df1_dx2 Output matrix for f1 derivatives w.r.t. Mesh 2 displacement.
   * @param df2_dx1 Output matrix for f2 derivatives w.r.t. Mesh 1 displacement.
   * @param df2_dx2 Output matrix for f2 derivatives w.r.t. Mesh 2 displacement.
   * @param df1_dp Output matrix for f1 derivatives w.r.t. pressure.
   * @param df2_dp Output matrix for f2 derivatives w.r.t. pressure.
   * @param energy Output contact energy contribution of the pair.
   */
  virtual void computeForce( const InterfacePair& pair, const IntegrationRule<PointwiseGapAndNormal, Dim>& integration_rule,
                             PointwiseGapAndNormal& gap_method, const std::vector<RealT>& coords1,
                             const std::vector<RealT>& coords2, std::shared_ptr<ArrayViewT<RealT>> pressures,
                             mfem::Vector& f1, mfem::Vector& f2, mfem::DenseMatrix& df1_dx1, mfem::DenseMatrix& df1_dx2,
                             mfem::DenseMatrix& df2_dx1, mfem::DenseMatrix& df2_dx2, mfem::DenseMatrix& df1_dp,
                             mfem::DenseMatrix& df2_dp, RealT& energy ) = 0;

  /**
   * @brief Computes contact forces for an element pair without derivatives.
   *
   * @param pair The interface pair.
   * @param integration_rule Integration rule used to compute points.
   * @param gap_method Pointwise gap and normal computation method.
   * @param coords1 Nodal coordinates of the first element (Mesh 1).
   * @param coords2 Nodal coordinates of the second element (Mesh 2).
   * @param pressures Nodal pressures (optional).
   * @param f1 Output vector for contact forces on Mesh 1.
   * @param f2 Output vector for contact forces on Mesh 2.
   * @param energy Output contact energy contribution of the pair.
   */
  virtual void computeForce( const InterfacePair& pair, const IntegrationRule<PointwiseGapAndNormal, Dim>& integration_rule,
                             PointwiseGapAndNormal& gap_method, const std::vector<RealT>& coords1,
                             const std::vector<RealT>& coords2, std::shared_ptr<ArrayViewT<RealT>> pressures,
                             mfem::Vector& f1, mfem::Vector& f2, RealT& energy ) = 0;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_ELEMENTFORCEANDGAP_HPP_ */
