// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_MORTARELEMENTFORCE_HPP_
#define SRC_TRIBOL_PHYSICS_MORTARELEMENTFORCE_HPP_

#include "tribol/common/Parameters.hpp"
#include "mfem.hpp"
#include <vector>

namespace tribol {

/**
 * @brief Struct to encapsulate the mortar force computation.
 */
struct MortarElementForce
{
  // Input normals for element 1 (Nonmortar)
  std::vector<RealT> n1;

  // Output derivatives w.r.t normals (stored here as ElementForceAndGap interface doesn't support them)
  mfem::DenseMatrix df1_dn1;
  mfem::DenseMatrix df2_dn1;
  mfem::DenseMatrix dg1_dn1;

  /**
   * @brief Computes the frictionless mortar forces for a 3D quad element.
   *
   * @param [in] x1 Nodal coordinates for element 1 (Nonmortar).
   * @param [in] n1 Nodal unit normal vectors for element 1.
   * @param [in] p1 Nodal pressures for element 1.
   * @param [out] f1 Nodal forces for element 1.
   * @param [out] g1 Nodal gaps for element 1.
   * @param [in] size1 Number of nodes on element 1.
   * @param [in] x2 Nodal coordinates for element 2 (Mortar).
   * @param [out] f2 Nodal forces for element 2.
   * @param [in] size2 Number of nodes on element 2.
   */
  static void compute( const RealT* x1, const RealT* n1, const RealT* p1, RealT* f1, RealT* g1, int size1,
                       const RealT* x2, RealT* f2, int size2 );
};

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_MORTARELEMENTFORCE_HPP_ */
