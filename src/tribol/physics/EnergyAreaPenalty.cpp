#include "EnergyAreaPenalty.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

using namespace energy_area_penalty_kernel_detail;
using namespace energy_mortar_kernel_detail;

struct KernelParams {
  int N = 3;
  double del = 0.1;
  SmoothingType smoothing_type = SmoothingType::Hermite;
  double penalty_smoothing_del = 0.0;
  PenaltySmoothing penalty_smoothing = PenaltySmoothing::Hard;
};

KernelParams make_kernel_params( const EnergyMortarOptions& opts )
{
  return { opts.N, opts.del, opts.smoothing_type, opts.penalty_smoothing_del, opts.penalty_smoothing };
}

inline void endpoints( const MeshData::Viewer& mesh, int elem_id, double P0[2], double P1[2] )
{
  double P0_P1[4];
  mesh.getFaceCoords( elem_id, P0_P1 );
  P0[0] = P0_P1[0];
  P0[1] = P0_P1[1];
  P1[0] = P0_P1[2];
  P1[1] = P0_P1[3];
}

// Fixed-quadrature scalar wrapper for Enzyme
static void energy_kernel_wrapper( const double* x, const void* gp_void,
                                   const double* ps_del, const int* ps_type,
                                   double* out )
{
  const Gparams* gp = static_cast<const Gparams*>( gp_void );
  *out = energy_kernel( x, *gp, *ps_del, static_cast<PenaltySmoothing>( *ps_type ) );
}

// Gradient of fixed-quadrature energy kernel
void grad_energy_fixed( const double* x, const Gparams* gp,
                        double penalty_smoothing_del, PenaltySmoothing penalty_smoothing,
                        double* dE_dx )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;
  int ps_type = static_cast<int>( penalty_smoothing );

  __enzyme_autodiff<void>( (void*)energy_kernel_wrapper,
                           enzyme_dup, x, dx,
                           enzyme_const, (const void*)gp,
                           enzyme_const, &penalty_smoothing_del,
                           enzyme_const, &ps_type,
                           enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) dE_dx[i] = dx[i];
}

// Varying-quadrature scalar wrapper for Enzyme
static void energy_kernel_varying_wrapper( const double* x, const void* kp_void, double* out )
{
  const auto* kp = static_cast<const KernelParams*>( kp_void );
  *out = energy_kernel_varying( x, kp->del, kp->N, kp->smoothing_type,
                                kp->penalty_smoothing_del, kp->penalty_smoothing );
}

// Gradient of varying-quadrature energy kernel
void grad_energy_varying( const double* x, const KernelParams* kp, double* dE_dx )
{
  double dx[8] = { 0.0 };
  double out = 0.0;
  double dout = 1.0;

  __enzyme_autodiff<void>( (void*)energy_kernel_varying_wrapper,
                           enzyme_dup, x, dx,
                           enzyme_const, (const void*)kp,
                           enzyme_dup, &out, &dout );

  for ( int i = 0; i < 8; ++i ) dE_dx[i] = dx[i];
}

// Hessian of varying-quadrature energy kernel
void hessian_energy_varying( const double* x, const KernelParams* kp, double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_energy_varying,
                            enzyme_dup, x, dx,
                            enzyme_const, (const void*)kp,
                            enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}

// Hessian of fixed-quadrature energy kernel
void hessian_energy_fixed( const double* x, const Gparams* gp,
                           double penalty_smoothing_del, PenaltySmoothing penalty_smoothing,
                           double* H )
{
  for ( int col = 0; col < 8; ++col ) {
    double dx[8] = { 0.0 };
    dx[col] = 1.0;

    double grad[8] = { 0.0 };
    double dgrad[8] = { 0.0 };

    __enzyme_fwddiff<void>( (void*)grad_energy_fixed,
                            enzyme_dup, x, dx,
                            enzyme_const, (const void*)gp,
                            enzyme_const, penalty_smoothing_del,
                            enzyme_const, penalty_smoothing,
                            enzyme_dup, grad, dgrad );

    for ( int row = 0; row < 8; ++row ) H[row * 8 + col] = dgrad[row];
  }
}

}  // namespace

// ============================================================================
// EnergyAreaPenaltyCalculator implementations
// ============================================================================

double EnergyAreaPenaltyCalculator::compute_energy( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                    const MeshData::Viewer& mesh2 ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  const double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  if ( !opts_.enzyme_quadrature ) {
    Gparams gp;
    double projs[2];
    energy_mortar_kernel_detail::get_projections( A0, A1, B0, B1, projs );
    const std::array<double, 2> projections = { projs[0], projs[1] };
    const auto bounds = energy_mortar_kernel_detail::bounds_from_projections( projections, opts_.del );
    const auto xi_bounds = energy_mortar_kernel_detail::smooth_bounds( bounds, opts_.del, opts_.smoothing_type );
    const auto qp = energy_mortar_kernel_detail::compute_quadrature( xi_bounds, opts_.N );
    for ( std::size_t i = 0; i < qp.qp.size(); ++i ) {
      gp.qp[i] = qp.qp[i];
      gp.w[i] = qp.w[i];
    }
    return energy_kernel( x, gp, opts_.penalty_smoothing_del, opts_.penalty_smoothing );
  } else {
    return energy_kernel_varying( x, opts_.del, opts_.N, opts_.smoothing_type,
                                  opts_.penalty_smoothing_del, opts_.penalty_smoothing );
  }
}

void EnergyAreaPenaltyCalculator::grad_energy( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                               const MeshData::Viewer& mesh2, double dE_dx[8] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  const double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  if ( !opts_.enzyme_quadrature ) {
    Gparams gp;
    double projs[2];
    energy_mortar_kernel_detail::get_projections( A0, A1, B0, B1, projs );
    const std::array<double, 2> projections = { projs[0], projs[1] };
    const auto bounds = energy_mortar_kernel_detail::bounds_from_projections( projections, opts_.del );
    const auto xi_bounds = energy_mortar_kernel_detail::smooth_bounds( bounds, opts_.del, opts_.smoothing_type );
    const auto qp = energy_mortar_kernel_detail::compute_quadrature( xi_bounds, opts_.N );
    for ( std::size_t i = 0; i < qp.qp.size(); ++i ) {
      gp.qp[i] = qp.qp[i];
      gp.w[i] = qp.w[i];
    }
    grad_energy_fixed( x, &gp, opts_.penalty_smoothing_del, opts_.penalty_smoothing, dE_dx );
  } else {
    const auto kp = make_kernel_params( opts_ );
    grad_energy_varying( x, &kp, dE_dx );
  }
}

void EnergyAreaPenaltyCalculator::hessian_energy( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                  const MeshData::Viewer& mesh2, double d2E_dx2[64] ) const
{
  double A0[2], A1[2], B0[2], B1[2];
  endpoints( mesh1, pair.m_element_id1, A0, A1 );
  endpoints( mesh2, pair.m_element_id2, B0, B1 );

  const double x[8] = { A0[0], A0[1], A1[0], A1[1], B0[0], B0[1], B1[0], B1[1] };

  if ( !opts_.enzyme_quadrature ) {
    Gparams gp;
    double projs[2];
    energy_mortar_kernel_detail::get_projections( A0, A1, B0, B1, projs );
    const std::array<double, 2> projections = { projs[0], projs[1] };
    const auto bounds = energy_mortar_kernel_detail::bounds_from_projections( projections, opts_.del );
    const auto xi_bounds = energy_mortar_kernel_detail::smooth_bounds( bounds, opts_.del, opts_.smoothing_type );
    const auto qp = energy_mortar_kernel_detail::compute_quadrature( xi_bounds, opts_.N );
    for ( std::size_t i = 0; i < qp.qp.size(); ++i ) {
      gp.qp[i] = qp.qp[i];
      gp.w[i] = qp.w[i];
    }
    hessian_energy_fixed( x, &gp, opts_.penalty_smoothing_del, opts_.penalty_smoothing, d2E_dx2 );
  } else {
    const auto kp = make_kernel_params( opts_ );
    hessian_energy_varying( x, &kp, d2E_dx2 );
  }
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
