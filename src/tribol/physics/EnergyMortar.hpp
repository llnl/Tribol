// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#pragma once
#include <vector>
#include <array>

#include "tribol/config.hpp"

#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

/*!
 * \brief Struct representing a quadrature rule (points and weights) for the Energy Mortar method.
 */
struct QuadPoints {
  int N{ 3 };                // Number of active quadrature points (up to 5)
  std::array<double, 5> qp;  // qp locations
  std::array<double, 5> w;   // weights
};

using Gparams = QuadPoints;

/*!
 * \brief Struct containing parameters for the Energy Mortar contact formulation.
 */
struct ContactParams {
  double del;              // Smoothing Parameter
  double k;                // Penalty
  int N;                   // Quadrature Points
  bool enzyme_quadrature;  // Determines how enzyming is performed (default = True)
};

/*!
 * \brief Provides smoothing operations for the Energy Mortar contact formulation.
 */
class ContactSmoothing {
 public:
  /*!
   * \brief Computes smoothed integration bounds from node projections.
   * \param [in] proj The node projection coordinates
   * \param [in] del The smoothing parameter delta
   * \return Computed integration bounds
   */
  static std::array<double, 2> boundsFromProjections( const std::array<double, 2>& proj, double del );

  /*!
   * \brief Applies further smoothing to the integration bounds.
   * \param [in] bounds The integration bounds
   * \param [in] del The smoothing parameter delta
   * \return Smoothed integration bounds
   */
  static std::array<double, 2> smoothBounds( const std::array<double, 2>& bounds, double del );
};

/*!
 * \brief Evaluates Energy Mortar contact quantities for a single interface pair.
 */
class EnergyMortarCalculator {
 public:
  /*!
   * \brief Constructor for EnergyMortarCalculator
   * \param [in] p Contact parameters
   */
  explicit EnergyMortarCalculator( const ContactParams& p )
      : p_( p ), smoother_() {}

  /*!
   * \brief Gets the number of quadrature points
   * \return Number of quadrature points
   */
  int getN() const { return p_.N; }

  /*!
   * \brief Computes the quadrature points and weights for given integration bounds.
   * \param [in] xi_bounds The integration bounds
   * \param [in] N The number of quadrature points
   * \return Computed quadrature points and weights
   */
  static QuadPoints computeQuadrature( const std::array<double, 2>& xi_bounds, int N );

  /*!
   * \brief Computes contact quantities: smoothed nodal gap integrals, tributary areas, and their gradients.
   * \param [in] pair Interface pair
   * \param [in] mesh1 Viewer for mesh 1
   * \param [in] mesh2 Viewer for mesh 2
   * \param [out] gtilde Computed gtilde values
   * \param [out] area Computed area values
   * \param [out] dgt1_dx Gradient of gtilde1 w.r.t coordinates
   * \param [out] dgt2_dx Gradient of gtilde2 w.r.t coordinates
   * \param [out] dA1_dx Gradient of area1 w.r.t coordinates
   * \param [out] dA2_dx Gradient of area2 w.r.t coordinates
   */
  void computeContactQuantities( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                 double gtilde[2], double area[2], double dgt1_dx[8], double dgt2_dx[8],
                                 double dA1_dx[8], double dA2_dx[8] ) const;

  /*!
   * \brief Computes second derivatives (Hessians) of the nodal smoothed gap integrals and tributary areas.
   * \param [in] pair Interface pair
   * \param [in] mesh1 Viewer for mesh 1
   * \param [in] mesh2 Viewer for mesh 2
   * \param [out] d2gt1_dx2 Hessian of gtilde1 w.r.t coordinates
   * \param [out] d2gt2_dx2 Hessian of gtilde2 w.r.t coordinates
   * \param [out] d2A1_dx2 Optional Hessian of area1 w.r.t coordinates (can be nullptr)
   * \param [out] d2A2_dx2 Optional Hessian of area2 w.r.t coordinates (can be nullptr)
   */
  void computeSecondDerivatives( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                 double d2gt1_dx2[64], double d2gt2_dx2[64], double d2A1_dx2[64] = nullptr,
                                 double d2A2_dx2[64] = nullptr ) const;

 private:
  ContactParams p_;
  ContactSmoothing smoother_;

  /*!
   * \brief Constructs Gparams for the given interface pair.
   * \param [in] pair Interface pair
   * \param [in] mesh1 Viewer for mesh 1
   * \param [in] mesh2 Viewer for mesh 2
   * \return Constructed Gparams
   */
  Gparams constructGparams( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                             const MeshData::Viewer& mesh2 ) const;

  /*!
   * \brief Computes projections of nodes for the interface pair.
   * \param [in] pair Interface pair
   * \param [in] mesh1 Viewer for mesh 1
   * \param [in] mesh2 Viewer for mesh 2
   * \return Computed node projections
   */
  std::array<double, 2> projections( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                      const MeshData::Viewer& mesh2 ) const;
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
