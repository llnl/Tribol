// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_NEWMETHODADAPTER_HPP_
#define SRC_TRIBOL_PHYSICS_NEWMETHODADAPTER_HPP_

#include "tribol/config.hpp"

#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/new_method.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/common/Parameters.hpp"

#include "mfem.hpp"

#include <memory>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

class NewMethodAdapter : public ContactFormulation {
 public:
  /**
   * @brief Constructor
   *
   * @param mfem_data Reference to Tribol's MFEM mesh data
   * @param k Penalty stiffness
   * @param delta Smoothing length
   * @param N Quadrature order
   */
  NewMethodAdapter( MfemSubmeshData& submesh_data, MfemJacobianData& jac_data, MeshData& mesh1, MeshData& mesh2,
                    double k, double delta, int N );

  virtual ~NewMethodAdapter() = default;

  // --- ContactFormulation Interface Implementation ---

  void setInterfacePairs( ArrayT<InterfacePair>&& pairs, int check_level ) override;

  void updateIntegrationRule() override;

  void updateNodalGaps() override;

  void updateNodalForces() override;

  RealT computeTimeStep() override;

  RealT getEnergy() const override { return energy_; }

#ifdef BUILD_REDECOMP
  const mfem::HypreParVector& getMfemForce() const override { return force_vec_.get(); }

  const mfem::HypreParVector& getMfemGap() const override { return gap_vec_.get(); }

  mfem::HypreParVector& getMfemPressure() override { return pressure_vec_.get(); }

  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx() const override;

  std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx() const override;

  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const override;
#endif

 private:
  // --- Member Variables ---

  double area_tol_{ 1.0e-14 };
  bool tied_contact_ = false;

  MfemSubmeshData& submesh_data_;
  MfemJacobianData& jac_data_;
  MeshData& mesh1_;
  MeshData& mesh2_;
  ContactParams params_;
  std::unique_ptr<ContactEvaluator> evaluator_;

  // Stored InterfacePairs
  ArrayT<InterfacePair> pairs_;

  // These store the assembled nodal values
  shared::ParVector g_tilde_vec_;
  shared::ParVector A_vec_;
  shared::ParVector gap_vec_;
  mutable shared::ParSparseMat dg_tilde_dx_;
  shared::ParSparseMat dA_dx_;

  shared::ParVector pressure_vec_;  // This holds p = k * g / A
  RealT energy_;
  shared::ParVector force_vec_;
  mutable shared::ParSparseMat df_dx_;
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_NEWMETHODADAPTER_HPP_ */
