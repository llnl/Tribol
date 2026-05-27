#pragma once

#include "tribol/config.hpp"
#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/EnergyAreaPenalty.hpp"
#include "tribol/physics/EnergyMortar.hpp"
#include "tribol/mesh/MfemData.hpp"
#include "tribol/common/Parameters.hpp"

#include "mfem.hpp"

#include <memory>
#include <utility>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

class EnergyAreaPenaltyAdapter : public ContactFormulation {
 public:
  EnergyAreaPenaltyAdapter( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data, MfemJacobianData& jac_data,
                            MeshData& mesh1, MeshData& mesh2, const EnergyMortarOptions& opts );

  virtual ~EnergyAreaPenaltyAdapter() = default;

  void setInterfacePairs( ArrayT<InterfacePair>&& pairs, int check_level ) override;
  void updateIntegrationRule() override;
  void updateNodalGaps() override;
  void updateNodalForces() override;
  RealT computeTimeStep() override;
  RealT getEnergy() const override { return energy_; }
  void updateMeshes( MeshData& mesh1, MeshData& mesh2 ) override;

#ifdef BUILD_REDECOMP
  const mfem::HypreParVector& getMfemForce() const override { return force_vec_.get(); }
  const mfem::HypreParVector& getMfemGap() const override { return gap_vec_.get(); }
  mfem::HypreParVector& getMfemPressure() override { return pressure_vec_.get(); }
  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx() const override;
  std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx() const override { return nullptr; }
  std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp() const override { return nullptr; }
#endif

 private:
  MfemMeshData& mesh_data_;
  MfemSubmeshData& submesh_data_;
  MfemJacobianData& jac_data_;

  MeshData* mesh1_ = nullptr;
  MeshData* mesh2_ = nullptr;

  EnergyMortarOptions params_;
  std::unique_ptr<EnergyAreaPenaltyCalculator> evaluator_;

  ArrayT<InterfacePair> pairs_;

  RealT energy_ = 0.0;

  shared::ParVector force_vec_;
  shared::ParVector gap_vec_;
  shared::ParVector pressure_vec_;
  mutable shared::ParSparseMat df_dx_;
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
