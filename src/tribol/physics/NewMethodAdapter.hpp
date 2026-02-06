// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_NEWMETHODADAPTER_HPP_
#define SRC_TRIBOL_PHYSICS_NEWMETHODADAPTER_HPP_

#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/new_method.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/common/Parameters.hpp"

#include "mfem.hpp"

#include <memory>

namespace tribol {

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

  void getMfemForce( mfem::Vector& forces ) const override;

  void getMfemGap( mfem::Vector& gaps ) const override;

  mfem::ParGridFunction& getMfemPressure() override;

  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx() const override;

  std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx() const override;

  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const override;

 private:
  // --- Member Variables ---

  MfemSubmeshData& submesh_data_;
  MfemJacobianData& jac_data_;
  MeshData& mesh1_;
  MeshData& mesh2_;
  ContactParams params_;
  std::unique_ptr<ContactEvaluator> evaluator_;

  // Stored InterfacePairs
  ArrayT<InterfacePair> pairs_;

  // These store the assembled nodal values
  mfem::HypreParVector g_tilde_vec_;
  mfem::HypreParVector A_vec_;
  mutable ParSparseMat dg_tilde_dx_;
  ParSparseMat dA_dx_;

  mfem::HypreParVector pressure_vec_;  // This holds p = k * g / A
  RealT energy_;
  mfem::Vector force_vec_;
  mutable ParSparseMat df_dx_;

  // Pressure GridFunction wrapper (required by interface)
  // We wrap the pressure_vec_ in a ParGridFunction for return
  std::unique_ptr<mfem::ParGridFunction> pressure_gf_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_NEWMETHODADAPTER_HPP_ */
