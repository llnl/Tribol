// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_CONTACTFORMULATIONINPUT_HPP_
#define SRC_TRIBOL_PHYSICS_CONTACTFORMULATIONINPUT_HPP_

#include "tribol/mesh/MeshData.hpp"

namespace tribol {

/**
 * @brief Supplies the current meshes to one contact-formulation request
 *
 * Contact formulations receive current mesh views instead of retaining handles to CouplingScheme or MeshManager data.
 * The caller owns every array referenced by the mesh views and must keep those arrays alive for the duration of the
 * formulation request. Method choices and search settings belong to the constructed formulation rather than this
 * per-request input.
 */
class ContactFormulationInput {
 public:
  /**
   * @brief Store the current non-owning mesh views
   *
   * @param mesh1 First mesh in Tribol coupling-scheme order
   * @param mesh2 Second mesh in Tribol coupling-scheme order
   */
  ContactFormulationInput( MeshData::Viewer mesh1, MeshData::Viewer mesh2 ) : mesh1_( mesh1 ), mesh2_( mesh2 ) {}

  /** @brief Return the current first mesh in Tribol coupling-scheme order */
  const MeshData::Viewer& getMesh1() const { return mesh1_; }

  /** @brief Return the current second mesh in Tribol coupling-scheme order */
  const MeshData::Viewer& getMesh2() const { return mesh2_; }

 private:
  MeshData::Viewer mesh1_;
  MeshData::Viewer mesh2_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_CONTACTFORMULATIONINPUT_HPP_ */
