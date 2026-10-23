// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_CONTACTFORMULATIONINPUT_HPP_
#define SRC_TRIBOL_PHYSICS_CONTACTFORMULATIONINPUT_HPP_

#include "tribol/common/ExecModel.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

/**
 * @brief Supplies current mesh and search data to one contact-formulation request
 *
 * Contact formulations receive current mesh views instead of retaining handles to CouplingScheme or MeshManager data.
 * The caller owns every array referenced by the mesh views and must keep those arrays alive for the duration of the
 * formulation request.
 */
class ContactFormulationInput {
 public:
  /**
   * @brief Store the current non-owning mesh views and pair-search settings
   *
   * @param mesh1 First mesh in Tribol coupling-scheme order
   * @param mesh2 Second mesh in Tribol coupling-scheme order
   * @param execution_mode Execution mode used by search and filtering loops
   * @param binning_method Coarse-search method selected for this request
   * @param effective_binning_proximity_scale Element-size multiplier used to expand coarse-search bounds
   * @param residual_gap Physical distance added to coarse-search bounds
   * @param checks_shared_nodes Whether auto-contact pairs sharing a finite-element node are rejected
   * @param reuses_fixed_pairs Whether a formulation may reuse a pair set initialized by an earlier request
   */
  ContactFormulationInput( MeshData::Viewer mesh1, MeshData::Viewer mesh2, ExecutionMode execution_mode,
                           BinningMethod binning_method, RealT effective_binning_proximity_scale, RealT residual_gap,
                           bool checks_shared_nodes, bool reuses_fixed_pairs )
      : mesh1_( mesh1 ),
        mesh2_( mesh2 ),
        execution_mode_( execution_mode ),
        binning_method_( binning_method ),
        effective_binning_proximity_scale_( effective_binning_proximity_scale ),
        residual_gap_( residual_gap ),
        checks_shared_nodes_( checks_shared_nodes ),
        reuses_fixed_pairs_( reuses_fixed_pairs )
  {
  }

  /** @brief Return the current first mesh in Tribol coupling-scheme order */
  const MeshData::Viewer& getMesh1() const { return mesh1_; }

  /** @brief Return the current second mesh in Tribol coupling-scheme order */
  const MeshData::Viewer& getMesh2() const { return mesh2_; }

  /** @brief Return the execution mode used by search and filtering loops */
  ExecutionMode getExecutionMode() const { return execution_mode_; }

  /** @brief Return the coarse-search method selected for this request */
  BinningMethod getBinningMethod() const { return binning_method_; }

  /** @brief Return the element-size multiplier used to expand coarse-search bounds */
  RealT getEffectiveBinningProximityScale() const { return effective_binning_proximity_scale_; }

  /** @brief Return the physical distance added to coarse-search bounds */
  RealT getResidualGap() const { return residual_gap_; }

  /** @brief Return whether auto-contact rejects pairs sharing a finite-element node */
  bool checksSharedNodes() const { return checks_shared_nodes_; }

  /** @brief Return whether an initialized pair set may be reused */
  bool reusesFixedPairs() const { return reuses_fixed_pairs_; }

 private:
  MeshData::Viewer mesh1_;
  MeshData::Viewer mesh2_;
  ExecutionMode execution_mode_;
  BinningMethod binning_method_;
  RealT effective_binning_proximity_scale_;
  RealT residual_gap_;
  bool checks_shared_nodes_;
  bool reuses_fixed_pairs_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_CONTACTFORMULATIONINPUT_HPP_ */
