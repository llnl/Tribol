// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
// SPDX-License-Identifier: (MIT)
#include "tribol/physics/CoulombFriction.hpp"
#include "tribol/common/Enzyme.hpp"
#include <cmath>
#include <stdexcept>

#ifdef TRIBOL_USE_ENZYME
namespace tribol {
namespace {
/** @brief Evaluate one nodal law on validated arrays.
 * @param a Normal sum, gap vector, slip vector, traction, and area (13 entries).
 * @param options Constant friction parameters.
 * @param offset Normal gap offset in length units.
 * @param out Constraint, unit normal, slip, pressure, and weighted normal gap (11 entries).
 */
void nodeKernel( const RealT* a, const CoulombFrictionOptions* options, RealT offset, RealT* out )
{
  for ( int i = 0; i < 11; ++i ) out[i] = 0.0;
  if ( a[12] <= options->area_tolerance ) {
    for ( int d = 0; d < 3; ++d ) out[d] = a[9 + d];
    return;
  }
  RealT magnitude = std::sqrt( a[0] * a[0] + a[1] * a[1] + a[2] * a[2] );
  for ( int d = 0; d < 3; ++d ) out[3 + d] = a[d] / magnitude;
  RealT normal_slip = 0.0;
  for ( int d = 0; d < 3; ++d ) {
    out[9] += out[3 + d] * a[9 + d];
    out[10] += out[3 + d] * a[3 + d];
    normal_slip += out[3 + d] * a[6 + d];
  }
  out[10] += offset * a[12];
  RealT normal_trial = out[9] + options->normal_scale * out[10];
  RealT normal_residual = out[9] - ( normal_trial > 0.0 ? normal_trial : 0.0 );
  MdArray<RealT, 3> tangential{}, trial{};
  RealT trial_squared = 0.0;
  for ( int d = 0; d < 3; ++d ) {
    out[6 + d] = a[6 + d] - out[3 + d] * normal_slip;
    tangential[d] = a[9 + d] - out[3 + d] * out[9];
    trial[d] = tangential[d] + options->tangential_scale * out[6 + d];
    trial_squared += trial[d] * trial[d];
    out[d] = out[3 + d] * normal_residual + tangential[d];
  }
  RealT radius = options->coefficient * ( out[9] > 0.0 ? out[9] : 0.0 );
  // The zero-radius disk contains only zero; its derivative must not evaluate a direction at zero length.
  if ( radius <= 0.0 ) return;
  if ( trial_squared <= radius * radius ) {
    for ( int d = 0; d < 3; ++d ) out[d] -= trial[d];
  } else {
    RealT scale = radius / std::sqrt( trial_squared );
    for ( int d = 0; d < 3; ++d ) out[d] -= scale * trial[d];
  }
}

/** @brief Calculate current unit corner normals for Enzyme differentiation.
 * @param x Component-major coordinates of a nondegenerate face.
 * @param nodes Number of face nodes, three or four.
 * @param out Component-major unit corner normals; unused entries through index 11 are zeroed.
 */
void cornerKernel( const RealT* x, int nodes, RealT* out )
{
  for ( int i = 0; i < 12; ++i ) out[i] = 0.0;
  for ( int i = 0; i < nodes; ++i ) {
    int previous = ( i + nodes - 1 ) % nodes;
    int next = ( i + 1 ) % nodes;
    MdArray<RealT, 3> a{}, b{}, n{};
    for ( int d = 0; d < 3; ++d ) {
      a[d] = x[d * nodes + next] - x[d * nodes + i];
      b[d] = x[d * nodes + previous] - x[d * nodes + i];
    }
    n[0] = a[1] * b[2] - a[2] * b[1];
    n[1] = a[2] * b[0] - a[0] * b[2];
    n[2] = a[0] * b[1] - a[1] * b[0];
    RealT mag = std::sqrt( n[0] * n[0] + n[1] * n[1] + n[2] * n[2] );
    for ( int d = 0; d < 3; ++d ) out[d * nodes + i] = n[d] / mag;
  }
}
}  // namespace
namespace detail {
CoulombNodeResult evaluateCoulombNode( const CoulombNodeInput& input, const CoulombFrictionOptions& options,
                                       RealT residual_gap, bool differentiate )
{
  if ( !std::isfinite( options.coefficient ) || options.coefficient < 0.0 || !std::isfinite( options.normal_scale ) ||
       options.normal_scale <= 0.0 || !std::isfinite( options.tangential_scale ) || options.tangential_scale <= 0.0 ||
       !std::isfinite( options.area_tolerance ) || options.area_tolerance <= 0.0 || !std::isfinite( residual_gap ) )
    throw std::invalid_argument(
        "Coulomb friction requires finite nonnegative friction and positive scales/tolerance." );
  MdArray<RealT, 13> a{}, seed{};
  MdArray<RealT, 11> out{}, derivative{};
  for ( int d = 0; d < 3; ++d ) {
    a[d] = input.normal_sum[d];
    a[3 + d] = input.gap_vector[d];
    a[6 + d] = input.slip_vector[d];
    a[9 + d] = input.traction[d];
  }
  a[12] = input.area;
  for ( int i = 0; i < 13; ++i )
    if ( !std::isfinite( a[i] ) ) throw std::invalid_argument( "Nonfinite Coulomb nodal input." );
  if ( a[12] > options.area_tolerance && a[0] * a[0] + a[1] * a[1] + a[2] * a[2] <= 1.0e-28 )
    throw std::invalid_argument( "Overlapping mortar node has a degenerate averaged normal." );
  nodeKernel( a.data(), &options, residual_gap, out.data() );
  CoulombNodeResult result;
  for ( int d = 0; d < 3; ++d ) {
    result.constraint[d] = out[d];
    result.normal[d] = out[3 + d];
    result.slip[d] = out[6 + d];
  }
  result.pressure = out[9];
  result.gap = out[10];
  if ( a[12] > options.area_tolerance && result.pressure + options.normal_scale * result.gap > 0.0 ) {
    RealT trial_squared = 0.0;
    for ( int d = 0; d < 3; ++d ) {
      RealT trial = a[9 + d] - out[3 + d] * out[9] + options.tangential_scale * out[6 + d];
      trial_squared += trial * trial;
    }
    RealT radius = options.coefficient * ( out[9] > 0.0 ? out[9] : 0.0 );
    result.state = trial_squared <= radius * radius ? FrictionState::Stick : FrictionState::Slip;
  }
  if ( differentiate )
    for ( int col = 0; col < 13; ++col ) {
      seed[col] = 1.0;
      __enzyme_fwddiff<void>( (void*)nodeKernel, TRIBOL_ENZYME_DUP, a.data(), seed.data(), TRIBOL_ENZYME_CONST,
                              &options, TRIBOL_ENZYME_CONST, residual_gap, TRIBOL_ENZYME_DUP, out.data(),
                              derivative.data() );
      for ( int row = 0; row < 3; ++row ) result.derivative( row, col ) = derivative[row];
      seed[col] = 0.0;
    }
  return result;
}

CornerNormalResult computeCornerNormals( const RealT* coordinates, int nodes )
{
  if ( nodes != 3 && nodes != 4 ) throw std::invalid_argument( "Coulomb mortar requires TRI3 or QUAD4 faces." );
  for ( int i = 0; i < 3 * nodes; ++i )
    if ( !std::isfinite( coordinates[i] ) ) throw std::invalid_argument( "Nonfinite corner-normal coordinate." );
  CornerNormalResult result;
  cornerKernel( coordinates, nodes, result.values.data() );
  for ( int i = 0; i < 3 * nodes; ++i )
    if ( !std::isfinite( result.values[i] ) ) throw std::invalid_argument( "Degenerate mortar corner." );
  MdArray<RealT, 12> seed{}, derivative{};
  for ( int col = 0; col < 3 * nodes; ++col ) {
    seed[col] = 1.0;
    __enzyme_fwddiff<void>( (void*)cornerKernel, TRIBOL_ENZYME_DUP, coordinates, seed.data(), TRIBOL_ENZYME_CONST,
                            nodes, TRIBOL_ENZYME_DUP, result.values.data(), derivative.data() );
    for ( int row = 0; row < 3 * nodes; ++row ) result.derivative( row, col ) = derivative[row];
    seed[col] = 0.0;
  }
  return result;
}
}  // namespace detail
}  // namespace tribol
#endif
