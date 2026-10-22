// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_MODULARFORMULATION_HPP_
#define SRC_TRIBOL_PHYSICS_MODULARFORMULATION_HPP_

#include "tribol/config.hpp"

#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/common/ArrayTypes.hpp"

#include "mfem.hpp"

#include <memory>
#include <vector>
#include <array>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

class ModularFormulation : public ContactFormulation {
 public:
  /**
   * @brief Constructor
   *
   * @param submesh_data Reference to Tribol's MFEM submesh data
   * @param jac_data Reference to Tribol's MFEM Jacobian data
   * @param mesh1 Non-mortar mesh (swapped internally to mesh2_ for literature consistency)
   * @param mesh2 Mortar mesh (swapped internally to mesh1_ for literature consistency)
   * @param k Penalty stiffness
   * @param delta Smoothing length
   * @param N Quadrature order
   */
  ModularFormulation( MfemSubmeshData& submesh_data, MfemJacobianData& jac_data, MeshData& mesh1, MeshData& mesh2,
                      double k, double delta, int N );

  virtual ~ModularFormulation() = default;

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
  struct ContactParams {
    double del;
    double k;
    int N;
  };

  struct QuadPoints {
    std::vector<double> qp;
    std::vector<double> w;
  };

  struct NodalContactData {
    std::array<double, 2> AI;
    std::array<double, 2> g_tilde;
  };

  // --- Physics Member Functions ---

  void gtilde_and_area( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                        double gtilde[2], double area[2] ) const;

  void grad_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                    double dgt1_dx[8], double dgt2_dx[8] ) const;

  void grad_trib_area( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                       double dA1_dx[8], double dA2_dx[8] ) const;

  void d2_g2tilde( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                   double dgt1_dx[64], double dgt2_dx[64] ) const;

  void compute_d2A_d2u( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                        double dgt1_dx[64], double dgt2_dx[64] ) const;

  std::array<double, 2> bounds_from_projections( const std::array<double, 2>& proj ) const;

  std::array<double, 2> smooth_bounds( const std::array<double, 2>& bounds ) const;

  QuadPoints compute_quadrature( const std::array<double, 2>& xi_bounds ) const;

  std::array<double, 2> projections( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                     const MeshData::Viewer& mesh2 ) const;

  double gap( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
              double xiA ) const;

  NodalContactData compute_nodal_contact_data( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                               const MeshData::Viewer& mesh2 ) const;

  std::array<double, 2> compute_pressures( const NodalContactData& ncd ) const;

  // --- Member Variables ---

  double area_tol_{ 1.0e-14 };
  bool tied_contact_ = false;

  MfemSubmeshData& submesh_data_;
  MfemJacobianData& jac_data_;
  MeshData& mesh1_;
  MeshData& mesh2_;
  ContactParams params_;

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

#endif /* SRC_TRIBOL_PHYSICS_MODULARFORMULATION_HPP_ */
