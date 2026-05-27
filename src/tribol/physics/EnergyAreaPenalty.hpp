#pragma once
#include <array>
#include <cmath>

#include "tribol/config.hpp"
#include "tribol/physics/EnergyMortarTypes.hpp"
#include "tribol/physics/EnergyMortar.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace energy_area_penalty_kernel_detail {

using namespace energy_mortar_kernel_detail;

// Energy kernel: E = 0.5 * ∫ H(gn)² dA
//
// This computes the area-integrated squared (ramped) gap for a single
// element pair.  Forces and Jacobians are obtained by Enzyme AD of this
// scalar kernel.
inline double energy_kernel( const double* x, const Gparams& gp,
                             double penalty_smoothing_del,
                             PenaltySmoothing penalty_smoothing )
{
  const double A0[2] = { x[0], x[1] };
  const double A1[2] = { x[2], x[3] };
  const double B0[2] = { x[4], x[5] };
  const double B1[2] = { x[6], x[7] };

  const double J = std::sqrt( ( A1[0] - A0[0] ) * ( A1[0] - A0[0] ) +
                              ( A1[1] - A0[1] ) * ( A1[1] - A0[1] ) );

  double nB[2];
  find_normal( B0, B1, nB );
  double nA[2];
  find_normal( A0, A1, nA );
  const double dot = nB[0] * nA[0] + nB[1] * nA[1];
  const double eta = ( dot < 0.0 ) ? -dot * dot : 0.0;

  double E = 0.0;
  for ( int i = 0; i < 3; ++i ) {
    const double xiA = gp.qp[i];
    const double w = gp.w[i];

    double x1[2];
    iso_map( A0, A1, xiA, x1 );
    double x2[2];
    find_intersection( B0, B1, x1, nB, x2 );

    const double dx = x1[0] - x2[0];
    const double dy = x1[1] - x2[1];
    const double gn = -( dx * nB[0] + dy * nB[1] );
    const double g = gn * eta;

    const auto ramp = penalty_ramp( g, penalty_smoothing_del, penalty_smoothing );
    E += 0.5 * w * ramp.value * ramp.value * J;
  }

  return E;
}

// Varying-quadrature version: computes quadrature from geometry
inline double energy_kernel_varying( const double* x, double del, int N,
                                     SmoothingType smoothing_type,
                                     double penalty_smoothing_del,
                                     PenaltySmoothing penalty_smoothing )
{
  const double A0[2] = { x[0], x[1] };
  const double A1[2] = { x[2], x[3] };
  const double B0[2] = { x[4], x[5] };
  const double B1[2] = { x[6], x[7] };

  double projs[2] = { 0.0, 0.0 };
  get_projections( A0, A1, B0, B1, projs );
  const std::array<double, 2> projections = { projs[0], projs[1] };
  const auto bounds = bounds_from_projections( projections, del );
  const auto xi_bounds = smooth_bounds( bounds, del, smoothing_type );
  const auto qp = compute_quadrature( xi_bounds, N );

  Gparams gp;
  for ( std::size_t i = 0; i < qp.qp.size(); ++i ) {
    gp.qp[i] = qp.qp[i];
    gp.w[i] = qp.w[i];
  }

  return energy_kernel( x, gp, penalty_smoothing_del, penalty_smoothing );
}

}  // namespace energy_area_penalty_kernel_detail

// Calculator class for the energy area penalty formulation
class EnergyAreaPenaltyCalculator {
 public:
  explicit EnergyAreaPenaltyCalculator( const EnergyMortarOptions& opts ) : opts_( opts ) {}

  // Compute scalar energy for one element pair
  double compute_energy( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                         const MeshData::Viewer& mesh2 ) const;

  // Compute gradient dE/dx[8] for one element pair
  void grad_energy( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                    const MeshData::Viewer& mesh2, double dE_dx[8] ) const;

  // Compute Hessian d²E/dx²[64] for one element pair
  void hessian_energy( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                       const MeshData::Viewer& mesh2, double d2E_dx2[64] ) const;

 private:
  EnergyMortarOptions opts_;
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
