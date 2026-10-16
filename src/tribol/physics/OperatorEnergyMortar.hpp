// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_OPERATORENERGYMORTAR_HPP_
#define SRC_TRIBOL_PHYSICS_OPERATORENERGYMORTAR_HPP_

#include "tribol/config.hpp"

#if defined( TRIBOL_USE_ENZYME ) && defined( BUILD_REDECOMP )

#include <memory>

#include "tribol/physics/ContactFormulation.hpp"
#include "tribol/physics/EnergyMortar.hpp"

namespace tribol {

class CouplingScheme;
class MeshData;

/**
 * @brief Two-dimensional quadrature-point EnergyMortar over operator-managed linear-edge patches
 *
 * EnergyMortar integrates on Tribol's nonmortar (second) surface. Its effective
 * gap is the orientation-weighted kinematic gap minus residual_gap; separation
 * is positive, interpenetration is negative, and a zero effective gap remains
 * active so the tangent is retained at first contact. Patch forces are
 * independent evaluation-rank duals. MfemContactData returns them to parent
 * true DOFs after updateNodalForces() completes.
 */
class OperatorEnergyMortar : public ContactFormulation {
 public:
  OperatorEnergyMortar( CouplingScheme& coupling_scheme, double penalty_stiffness, double smoothing_length,
                        int quadrature_order, bool enzyme_quadrature, RealT residual_gap );

  void setInterfacePairs( ArrayT<InterfacePair>&& pairs, int check_level ) override;
  void updateIntegrationRule() override {}
  void updateNodalGaps() override {}
  void updateNodalForces() override;
  bool hasTimeStepCalculation() override { return false; }
  RealT computeTimeStep() override { return 1.0; }
  RealT getEnergy() const override { return energy_; }
  void updateMeshes( MeshData& first_surface, MeshData& second_surface ) override;
  void updateConstantPenaltyStiffness( double first_penalty, double second_penalty ) override;
  void setResidualGap( RealT residual_gap ) override;

 private:
  CouplingScheme& coupling_scheme_;
  MeshData* nonmortar_mesh_{ nullptr };
  MeshData* mortar_mesh_{ nullptr };
  ContactParams parameters_;
  std::unique_ptr<EnergyMortarCalculator> evaluator_;
  ArrayT<InterfacePair> pairs_;
  RealT energy_{ 0.0 };
};

}  // namespace tribol

#endif

#endif  // SRC_TRIBOL_PHYSICS_OPERATORENERGYMORTAR_HPP_
