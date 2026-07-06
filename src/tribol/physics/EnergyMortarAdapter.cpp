// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortarAdapter.hpp"
#include <axom/slic/interface/slic_macros.hpp>
#include "tribol/geom/NodalNormal.hpp"
#include "tribol/mesh/MfemData.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

struct H1DofInfo {
  int side;
  int node_index;
  int elem_id;
  int local_dof;
};

int localDofForNodeComponent( const MeshData::Viewer& mesh, int elem_id, int node_id, int component )
{
  for ( int i{ 0 }; i < mesh.numberOfNodesPerElement(); ++i ) {
    if ( mesh.getGlobalNodeId( elem_id, i ) == node_id ) {
      return component * mesh.numberOfNodesPerElement() + i;
    }
  }
  SLIC_ERROR_ROOT( "ENERGY_MORTAR H1 derivative stencil owner element does not contain the requested node." );
  return 0;
}

std::vector<H1DofInfo> buildH1DofInfo( const H1TotalDerivatives& h1, const MeshData::Viewer& mesh1,
                                       const MeshData::Viewer& mesh2 )
{
  std::vector<H1DofInfo> dofs;
  dofs.reserve( 2 * ( h1.num_mesh1_nodes + h1.num_mesh2_nodes ) );
  for ( int component{ 0 }; component < 2; ++component ) {
    for ( int i{ 0 }; i < h1.num_mesh1_nodes; ++i ) {
      const int node_id = h1.mesh1_nodes[i];
      const int elem_id = h1.mesh1_owner_elems[i];
      dofs.push_back( { 0, i, elem_id, localDofForNodeComponent( mesh1, elem_id, node_id, component ) } );
    }
  }
  for ( int component{ 0 }; component < 2; ++component ) {
    for ( int i{ 0 }; i < h1.num_mesh2_nodes; ++i ) {
      const int node_id = h1.mesh2_nodes[i];
      const int elem_id = h1.mesh2_owner_elems[i];
      dofs.push_back( { 1, i, elem_id, localDofForNodeComponent( mesh2, elem_id, node_id, component ) } );
    }
  }
  return dofs;
}

std::vector<H1DofInfo> buildH1DofInfo( const QuadraturePointPenaltyData& h1, const MeshData::Viewer& mesh1,
                                       const MeshData::Viewer& mesh2 )
{
  std::vector<H1DofInfo> dofs;
  dofs.reserve( 2 * ( h1.num_mesh1_nodes + h1.num_mesh2_nodes ) );
  for ( int component{ 0 }; component < 2; ++component ) {
    for ( int i{ 0 }; i < h1.num_mesh1_nodes; ++i ) {
      const int node_id = h1.mesh1_nodes[i];
      const int elem_id = h1.mesh1_owner_elems[i];
      dofs.push_back( { 0, i, elem_id, localDofForNodeComponent( mesh1, elem_id, node_id, component ) } );
    }
  }
  for ( int component{ 0 }; component < 2; ++component ) {
    for ( int i{ 0 }; i < h1.num_mesh2_nodes; ++i ) {
      const int node_id = h1.mesh2_nodes[i];
      const int elem_id = h1.mesh2_owner_elems[i];
      dofs.push_back( { 1, i, elem_id, localDofForNodeComponent( mesh2, elem_id, node_id, component ) } );
    }
  }
  return dofs;
}

void appendH1FirstDerivativeBlocks( const H1TotalDerivatives& h1, const MeshData::Viewer& mesh1,
                                    const MeshData::Viewer& mesh2, PackedPairJacobianContribs& nm_contribs,
                                    PackedPairJacobianContribs& m_contribs, int row_elem, bool area_derivative )
{
  std::map<int, std::array<double, 8>> nm_blocks;
  std::map<int, std::array<double, 8>> m_blocks;
  const auto dofs = buildH1DofInfo( h1, mesh1, mesh2 );
  const auto& deriv1 = area_derivative ? h1.dA1_dx : h1.dg1_dx;
  const auto& deriv2 = area_derivative ? h1.dA2_dx : h1.dg2_dx;

  for ( int j{ 0 }; j < static_cast<int>( dofs.size() ); ++j ) {
    auto& blocks = dofs[j].side == 0 ? nm_blocks : m_blocks;
    auto it = blocks.find( dofs[j].elem_id );
    if ( it == blocks.end() ) {
      it = blocks.emplace( dofs[j].elem_id, std::array<double, 8>{} ).first;
    }
    auto& block = it->second;
    block[dofs[j].local_dof * 2] += deriv1[j];
    block[dofs[j].local_dof * 2 + 1] += deriv2[j];
  }

  for ( const auto& [elem_id, block] : nm_blocks ) {
    nm_contribs.append( row_elem, elem_id, block.data(), static_cast<int>( block.size() ) );
  }
  for ( const auto& [elem_id, block] : m_blocks ) {
    m_contribs.append( row_elem, elem_id, block.data(), static_cast<int>( block.size() ) );
  }
}

void addH1HessianBlocks( const H1TotalDerivatives& h1, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                         double w1, double w2, const std::vector<double>& H1, const std::vector<double>& H2,
                         std::map<std::pair<int, int>, std::array<double, 16>>& nm_nm_blocks,
                         std::map<std::pair<int, int>, std::array<double, 16>>& nm_m_blocks,
                         std::map<std::pair<int, int>, std::array<double, 16>>& m_nm_blocks,
                         std::map<std::pair<int, int>, std::array<double, 16>>& m_m_blocks )
{
  const auto dofs = buildH1DofInfo( h1, mesh1, mesh2 );
  const int ndof = static_cast<int>( dofs.size() );
  for ( int row{ 0 }; row < ndof; ++row ) {
    for ( int col{ 0 }; col < ndof; ++col ) {
      const double val = w1 * H1[row * ndof + col] + w2 * H2[row * ndof + col];
      auto key = std::make_pair( dofs[row].elem_id, dofs[col].elem_id );
      std::array<double, 16>* block = nullptr;
      if ( dofs[row].side == 0 && dofs[col].side == 0 ) {
        auto it = nm_nm_blocks.find( key );
        if ( it == nm_nm_blocks.end() ) {
          it = nm_nm_blocks.emplace( key, std::array<double, 16>{} ).first;
        }
        block = &it->second;
      } else if ( dofs[row].side == 0 && dofs[col].side == 1 ) {
        auto it = nm_m_blocks.find( key );
        if ( it == nm_m_blocks.end() ) {
          it = nm_m_blocks.emplace( key, std::array<double, 16>{} ).first;
        }
        block = &it->second;
      } else if ( dofs[row].side == 1 && dofs[col].side == 0 ) {
        auto it = m_nm_blocks.find( key );
        if ( it == m_nm_blocks.end() ) {
          it = m_nm_blocks.emplace( key, std::array<double, 16>{} ).first;
        }
        block = &it->second;
      } else {
        auto it = m_m_blocks.find( key );
        if ( it == m_m_blocks.end() ) {
          it = m_m_blocks.emplace( key, std::array<double, 16>{} ).first;
        }
        block = &it->second;
      }
      ( *block )[dofs[row].local_dof + dofs[col].local_dof * 4] += val;
    }
  }
}

void addH1QuadraturePointPenaltyHessianBlocks( const QuadraturePointPenaltyData& h1, const MeshData::Viewer& mesh1,
                                               const MeshData::Viewer& mesh2,
                                               std::map<std::pair<int, int>, std::array<double, 16>>& nm_nm_blocks,
                                               std::map<std::pair<int, int>, std::array<double, 16>>& nm_m_blocks,
                                               std::map<std::pair<int, int>, std::array<double, 16>>& m_nm_blocks,
                                               std::map<std::pair<int, int>, std::array<double, 16>>& m_m_blocks )
{
  const auto dofs = buildH1DofInfo( h1, mesh1, mesh2 );
  const int ndof = static_cast<int>( dofs.size() );
  for ( int row{ 0 }; row < ndof; ++row ) {
    for ( int col{ 0 }; col < ndof; ++col ) {
      const double val = h1.h1_stiffness[row * ndof + col];
      auto key = std::make_pair( dofs[row].elem_id, dofs[col].elem_id );
      std::array<double, 16>* block = nullptr;
      if ( dofs[row].side == 0 && dofs[col].side == 0 ) {
        auto it = nm_nm_blocks.find( key );
        if ( it == nm_nm_blocks.end() ) {
          it = nm_nm_blocks.emplace( key, std::array<double, 16>{} ).first;
        }
        block = &it->second;
      } else if ( dofs[row].side == 0 && dofs[col].side == 1 ) {
        auto it = nm_m_blocks.find( key );
        if ( it == nm_m_blocks.end() ) {
          it = nm_m_blocks.emplace( key, std::array<double, 16>{} ).first;
        }
        block = &it->second;
      } else if ( dofs[row].side == 1 && dofs[col].side == 0 ) {
        auto it = m_nm_blocks.find( key );
        if ( it == m_nm_blocks.end() ) {
          it = m_nm_blocks.emplace( key, std::array<double, 16>{} ).first;
        }
        block = &it->second;
      } else {
        auto it = m_m_blocks.find( key );
        if ( it == m_m_blocks.end() ) {
          it = m_m_blocks.emplace( key, std::array<double, 16>{} ).first;
        }
        block = &it->second;
      }
      ( *block )[dofs[row].local_dof + dofs[col].local_dof * 4] += val;
    }
  }
}

void appendH1HessianBlockMaps( const std::map<std::pair<int, int>, std::array<double, 16>>& nm_nm_blocks,
                               const std::map<std::pair<int, int>, std::array<double, 16>>& nm_m_blocks,
                               const std::map<std::pair<int, int>, std::array<double, 16>>& m_nm_blocks,
                               const std::map<std::pair<int, int>, std::array<double, 16>>& m_m_blocks,
                               PackedPairJacobianContribs& df_nm_nm, PackedPairJacobianContribs& df_nm_m,
                               PackedPairJacobianContribs& df_m_nm, PackedPairJacobianContribs& df_m_m )
{
  for ( const auto& [key, block] : nm_nm_blocks ) {
    df_nm_nm.append( key.first, key.second, block.data(), static_cast<int>( block.size() ) );
  }
  for ( const auto& [key, block] : nm_m_blocks ) {
    df_nm_m.append( key.first, key.second, block.data(), static_cast<int>( block.size() ) );
  }
  for ( const auto& [key, block] : m_nm_blocks ) {
    df_m_nm.append( key.first, key.second, block.data(), static_cast<int>( block.size() ) );
  }
  for ( const auto& [key, block] : m_m_blocks ) {
    df_m_m.append( key.first, key.second, block.data(), static_cast<int>( block.size() ) );
  }
}

void copyNodalNormalsToGridFunction( MeshData& mesh, mfem::GridFunction& normal )
{
  auto mesh_view = mesh.getView();
  SLIC_ERROR_ROOT_IF( !mesh_view.hasNodalNormals(), "ENERGY_MORTAR H1 nodal normal field has not been computed." );
  const int scalar_size = normal.FESpace()->GetVSize() / normal.FESpace()->GetVDim();
  SLIC_ERROR_ROOT_IF( scalar_size < mesh_view.numberOfNodes(),
                      "ENERGY_MORTAR nodal normal output field is smaller than the redecomp mesh node count." );

  for ( int i{ 0 }; i < mesh_view.numberOfNodes(); ++i ) {
    for ( int d{ 0 }; d < mesh_view.spatialDimension(); ++d ) {
      normal( d * scalar_size + i ) = mesh_view.getNodalNormals()( d, i );
    }
  }
}

struct H1DebugConfig {
  bool enabled{ false };
  int call_filter{ 0 };
  int top_pairs{ 25 };
  std::string file{ "tribol_energy_mortar_h1_debug.csv" };
};

H1DebugConfig getH1DebugConfig()
{
  H1DebugConfig config;
  const char* enabled = std::getenv( "TRIBOL_EM_H1_DEBUG" );
  config.enabled = enabled != nullptr && std::string( enabled ) != "0";
  if ( const char* call_filter = std::getenv( "TRIBOL_EM_H1_DEBUG_CALL" ) ) {
    config.call_filter = std::atoi( call_filter );
  }
  if ( const char* top_pairs = std::getenv( "TRIBOL_EM_H1_DEBUG_TOP" ) ) {
    config.top_pairs = std::max( 1, std::atoi( top_pairs ) );
  }
  if ( const char* file = std::getenv( "TRIBOL_EM_H1_DEBUG_FILE" ) ) {
    config.file = file;
  }
  return config;
}

std::ofstream& h1DebugStream( const std::string& file )
{
  static std::ofstream stream;
  static bool initialized{ false };
  if ( !initialized ) {
    stream.open( file );
    stream << "call,orig_elem1,orig_elem2,internal_A_elem,internal_B_elem,rank,score,h1_proj_lo,h1_proj_hi,"
              "elem_proj_lo,elem_proj_hi,h1_smooth_lo,h1_smooth_hi,elem_smooth_lo,elem_smooth_hi,h1_len,elem_len,"
              "h1_g0,h1_g1,h1_A0,h1_A1,h1_gap0,h1_gap1,elem_g0,elem_g1,elem_A0,elem_A1,elem_gap0,elem_gap1,"
              "max_abs_dg,max_abs_dA,A_elem_dot_B_elem,A_node_dot_elem_min,B_node_dot_elem_min,A_node_spread,"
              "B_node_spread\n";
    stream << std::setprecision( 16 );
    initialized = true;
  }
  return stream;
}

double maxAbs( const std::vector<double>& values )
{
  double result = 0.0;
  for ( double value : values ) {
    result = std::max( result, std::abs( value ) );
  }
  return result;
}

double safeRatio( double numerator, double denominator )
{
  return std::abs( denominator ) > 1.0e-30 ? numerator / denominator : 0.0;
}

void faceNormal( const MeshData::Viewer& mesh, int elem_id, double normal[2] )
{
  double coords[4];
  mesh.getFaceCoords( elem_id, coords );
  const double dx = coords[2] - coords[0];
  const double dy = coords[3] - coords[1];
  const double len = std::sqrt( dx * dx + dy * dy );
  if ( len <= 1.0e-30 ) {
    normal[0] = 0.0;
    normal[1] = 0.0;
    return;
  }
  normal[0] = dy / len;
  normal[1] = -dx / len;
}

void endpointNormals( const MeshData::Viewer& mesh, int elem_id, double N0[2], double N1[2] )
{
  N0[0] = N0[1] = N1[0] = N1[1] = 0.0;
  if ( !mesh.hasNodalNormals() ) {
    return;
  }
  const auto conn = mesh.getConnectivity()( elem_id );
  N0[0] = mesh.getNodalNormals()( 0, conn[0] );
  N0[1] = mesh.getNodalNormals()( 1, conn[0] );
  N1[0] = mesh.getNodalNormals()( 0, conn[1] );
  N1[1] = mesh.getNodalNormals()( 1, conn[1] );
}

double dot2( const double* a, const double* b ) { return a[0] * b[0] + a[1] * b[1]; }

void normalizeDebugVector( double v[2] )
{
  const double mag = std::sqrt( dot2( v, v ) );
  if ( mag <= 1.0e-30 ) {
    return;
  }
  v[0] /= mag;
  v[1] /= mag;
}

struct H1DebugPair {
  InterfacePair original_pair;
  InterfacePair flipped_pair;
  double score{ 0.0 };
  std::array<double, 2> h1_proj{};
  std::array<double, 2> elem_proj{};
  std::array<double, 2> h1_smooth{};
  std::array<double, 2> elem_smooth{};
  std::array<double, 2> h1_g{};
  std::array<double, 2> h1_A{};
  std::array<double, 2> h1_gap{};
  std::array<double, 2> elem_g{};
  std::array<double, 2> elem_A{};
  std::array<double, 2> elem_gap{};
  double max_abs_dg{ 0.0 };
  double max_abs_dA{ 0.0 };
  double A_elem_dot_B_elem{ 0.0 };
  double A_node_dot_elem_min{ 0.0 };
  double B_node_dot_elem_min{ 0.0 };
  double A_node_spread{ 0.0 };
  double B_node_spread{ 0.0 };
};

void recordH1DebugPair( std::vector<H1DebugPair>& debug_pairs, const InterfacePair& original_pair,
                        const InterfacePair& flipped_pair, const H1TotalDerivatives& h1, const ContactParams& params,
                        const MeshData::Viewer& mesh1_view, const MeshData::Viewer& mesh2_view )
{
  ContactParams elem_params = params;
  elem_params.normal_mode = EnergyMortarNormalMode::ELEMENT_NORMAL;
  EnergyMortarCalculator elem_evaluator( elem_params );
  EnergyMortarCalculator h1_bounds_evaluator( params );

  H1DebugPair debug;
  debug.original_pair = original_pair;
  debug.flipped_pair = flipped_pair;
  debug.h1_proj = h1_bounds_evaluator.compute_projection_bounds( flipped_pair, mesh1_view, mesh2_view );
  debug.elem_proj = elem_evaluator.compute_projection_bounds( flipped_pair, mesh1_view, mesh2_view );
  const auto h1_bounds = ContactSmoothing::bounds_from_projections( debug.h1_proj, params.del );
  const auto elem_bounds = ContactSmoothing::bounds_from_projections( debug.elem_proj, params.del );
  debug.h1_smooth = params.projection_smoothing ? ContactSmoothing::smooth_bounds( h1_bounds, params.del ) : h1_bounds;
  debug.elem_smooth =
      params.projection_smoothing ? ContactSmoothing::smooth_bounds( elem_bounds, params.del ) : elem_bounds;

  debug.h1_g = h1.g_tilde;
  debug.h1_A = h1.area;
  debug.h1_gap = { safeRatio( h1.g_tilde[0], h1.area[0] ), safeRatio( h1.g_tilde[1], h1.area[1] ) };
  double elem_g[2] = { 0.0, 0.0 };
  double elem_A[2] = { 0.0, 0.0 };
  elem_evaluator.compute_gtilde_and_area( flipped_pair, mesh1_view, mesh2_view, elem_g, elem_A );
  debug.elem_g = { elem_g[0], elem_g[1] };
  debug.elem_A = { elem_A[0], elem_A[1] };
  debug.elem_gap = { safeRatio( elem_g[0], elem_A[0] ), safeRatio( elem_g[1], elem_A[1] ) };
  debug.max_abs_dg = std::max( maxAbs( h1.dg1_dx ), maxAbs( h1.dg2_dx ) );
  debug.max_abs_dA = std::max( maxAbs( h1.dA1_dx ), maxAbs( h1.dA2_dx ) );

  double nA_elem[2], nB_elem[2], nA0[2], nA1[2], nB0[2], nB1[2];
  faceNormal( mesh1_view, flipped_pair.m_element_id1, nA_elem );
  faceNormal( mesh2_view, flipped_pair.m_element_id2, nB_elem );
  endpointNormals( mesh1_view, flipped_pair.m_element_id1, nA0, nA1 );
  endpointNormals( mesh2_view, flipped_pair.m_element_id2, nB0, nB1 );
  normalizeDebugVector( nA0 );
  normalizeDebugVector( nA1 );
  normalizeDebugVector( nB0 );
  normalizeDebugVector( nB1 );
  debug.A_elem_dot_B_elem = dot2( nA_elem, nB_elem );
  debug.A_node_dot_elem_min = std::min( dot2( nA0, nA_elem ), dot2( nA1, nA_elem ) );
  debug.B_node_dot_elem_min = std::min( dot2( nB0, nB_elem ), dot2( nB1, nB_elem ) );
  debug.A_node_spread = dot2( nA0, nA1 );
  debug.B_node_spread = dot2( nB0, nB1 );

  const double gap_diff =
      std::max( std::abs( debug.h1_gap[0] - debug.elem_gap[0] ), std::abs( debug.h1_gap[1] - debug.elem_gap[1] ) );
  const double len_diff =
      std::abs( ( debug.h1_smooth[1] - debug.h1_smooth[0] ) - ( debug.elem_smooth[1] - debug.elem_smooth[0] ) );
  const double min_area = std::min( std::abs( debug.h1_A[0] ), std::abs( debug.h1_A[1] ) );
  debug.score = debug.max_abs_dg + debug.max_abs_dA + 100.0 * gap_diff + len_diff + safeRatio( 1.0e-8, min_area );

  debug_pairs.push_back( debug );
}

void flushH1DebugPairs( int call, const H1DebugConfig& config, std::vector<H1DebugPair>& debug_pairs )
{
  if ( !config.enabled || ( config.call_filter > 0 && config.call_filter != call ) ) {
    return;
  }
  std::sort( debug_pairs.begin(), debug_pairs.end(),
             []( const H1DebugPair& a, const H1DebugPair& b ) { return a.score > b.score; } );

  auto& stream = h1DebugStream( config.file );
  const int num_pairs = std::min( config.top_pairs, static_cast<int>( debug_pairs.size() ) );
  for ( int i{ 0 }; i < num_pairs; ++i ) {
    const auto& debug = debug_pairs[i];
    stream << call << "," << debug.original_pair.m_element_id1 << "," << debug.original_pair.m_element_id2 << ","
           << debug.flipped_pair.m_element_id1 << "," << debug.flipped_pair.m_element_id2 << "," << i << ","
           << debug.score << "," << debug.h1_proj[0] << "," << debug.h1_proj[1] << "," << debug.elem_proj[0] << ","
           << debug.elem_proj[1] << "," << debug.h1_smooth[0] << "," << debug.h1_smooth[1] << ","
           << debug.elem_smooth[0] << "," << debug.elem_smooth[1] << "," << debug.h1_smooth[1] - debug.h1_smooth[0]
           << "," << debug.elem_smooth[1] - debug.elem_smooth[0] << "," << debug.h1_g[0] << "," << debug.h1_g[1] << ","
           << debug.h1_A[0] << "," << debug.h1_A[1] << "," << debug.h1_gap[0] << "," << debug.h1_gap[1] << ","
           << debug.elem_g[0] << "," << debug.elem_g[1] << "," << debug.elem_A[0] << "," << debug.elem_A[1] << ","
           << debug.elem_gap[0] << "," << debug.elem_gap[1] << "," << debug.max_abs_dg << "," << debug.max_abs_dA << ","
           << debug.A_elem_dot_B_elem << "," << debug.A_node_dot_elem_min << "," << debug.B_node_dot_elem_min << ","
           << debug.A_node_spread << "," << debug.B_node_spread << "\n";
  }
  stream.flush();
}

struct ActiveSetWeight {
  double value{ 1.0 };
  double first_deriv{ 0.0 };
  double second_deriv{ 0.0 };
};

ActiveSetWeight h1ActiveSetWeight( double gap, double transition_gap )
{
  if ( transition_gap <= 0.0 || gap <= -transition_gap ) {
    return { 1.0, 0.0, 0.0 };
  }
  if ( gap >= transition_gap ) {
    return { 0.0, 0.0, 0.0 };
  }

  const double t = ( gap + transition_gap ) / ( 2.0 * transition_gap );
  const double smooth = 3.0 * t * t - 2.0 * t * t * t;
  const double smooth_prime = 3.0 * t * ( 1.0 - t ) / transition_gap;
  const double smooth_second = 3.0 * ( 1.0 - 2.0 * t ) / ( 2.0 * transition_gap * transition_gap );
  return { 1.0 - smooth, -smooth_prime, -smooth_second };
}

void zeroH1NodeContribution( H1TotalDerivatives& h1, int node )
{
  h1.g_tilde[node] = 0.0;
  h1.area[node] = 0.0;
  auto& dg = node == 0 ? h1.dg1_dx : h1.dg2_dx;
  auto& dA = node == 0 ? h1.dA1_dx : h1.dA2_dx;
  auto& d2g = node == 0 ? h1.d2g1_dx2 : h1.d2g2_dx2;
  auto& d2A = node == 0 ? h1.d2A1_dx2 : h1.d2A2_dx2;
  std::fill( dg.begin(), dg.end(), 0.0 );
  std::fill( dA.begin(), dA.end(), 0.0 );
  std::fill( d2g.begin(), d2g.end(), 0.0 );
  std::fill( d2A.begin(), d2A.end(), 0.0 );
}

void applyWeightToH1Quantity( double& value, std::vector<double>& deriv, std::vector<double>& hessian,
                              const ActiveSetWeight& weight, const std::vector<double>& gap_deriv,
                              const std::vector<double>& gap_hessian )
{
  const double original_value = value;
  const auto original_deriv = deriv;
  const auto original_hessian = hessian;
  const int ndof = static_cast<int>( deriv.size() );

  value = weight.value * original_value;
  for ( int i{ 0 }; i < ndof; ++i ) {
    deriv[i] = weight.value * original_deriv[i] + original_value * weight.first_deriv * gap_deriv[i];
  }

  if ( hessian.empty() ) {
    return;
  }

  for ( int i{ 0 }; i < ndof; ++i ) {
    for ( int j{ 0 }; j < ndof; ++j ) {
      const int idx = i * ndof + j;
      hessian[idx] = weight.value * original_hessian[idx] +
                     weight.first_deriv * ( original_deriv[i] * gap_deriv[j] + gap_deriv[i] * original_deriv[j] +
                                            original_value * gap_hessian[idx] ) +
                     original_value * weight.second_deriv * gap_deriv[i] * gap_deriv[j];
    }
  }
}

void applyH1ActiveSetSmoothing( H1TotalDerivatives& h1, double transition_gap, double area_tol )
{
  if ( transition_gap <= 0.0 ) {
    return;
  }

  for ( int node{ 0 }; node < 2; ++node ) {
    const double G = h1.g_tilde[node];
    const double A = h1.area[node];
    if ( A <= area_tol ) {
      zeroH1NodeContribution( h1, node );
      continue;
    }

    auto& dG = node == 0 ? h1.dg1_dx : h1.dg2_dx;
    auto& dA = node == 0 ? h1.dA1_dx : h1.dA2_dx;
    auto& d2G = node == 0 ? h1.d2g1_dx2 : h1.d2g2_dx2;
    auto& d2A = node == 0 ? h1.d2A1_dx2 : h1.d2A2_dx2;

    const int ndof = static_cast<int>( dG.size() );
    std::vector<double> gap_deriv( ndof, 0.0 );
    const double inv_A = 1.0 / A;
    const double inv_A2 = inv_A * inv_A;
    const double inv_A3 = inv_A2 * inv_A;
    for ( int i{ 0 }; i < ndof; ++i ) {
      gap_deriv[i] = dG[i] * inv_A - G * dA[i] * inv_A2;
    }

    std::vector<double> gap_hessian;
    if ( !d2G.empty() ) {
      gap_hessian.assign( d2G.size(), 0.0 );
      for ( int i{ 0 }; i < ndof; ++i ) {
        for ( int j{ 0 }; j < ndof; ++j ) {
          const int idx = i * ndof + j;
          gap_hessian[idx] = d2G[idx] * inv_A - G * d2A[idx] * inv_A2 - ( dG[i] * dA[j] + dA[i] * dG[j] ) * inv_A2 +
                             2.0 * G * dA[i] * dA[j] * inv_A3;
        }
      }
    }

    const ActiveSetWeight weight = h1ActiveSetWeight( G * inv_A, transition_gap );
    applyWeightToH1Quantity( h1.g_tilde[node], dG, d2G, weight, gap_deriv, gap_hessian );
    applyWeightToH1Quantity( h1.area[node], dA, d2A, weight, gap_deriv, gap_hessian );
  }
}

}  // namespace

EnergyMortarAdapter::EnergyMortarAdapter( MfemMeshData& mesh_data, MfemSubmeshData& submesh_data,
                                          MfemJacobianData& jac_data, double k, double delta, int N,
                                          bool enzyme_quadrature, bool fixed_integration_jacobian, bool use_penalty,
                                          EnergyMortarNormalMode normal_mode, bool projection_smoothing,
                                          double h1_active_set_smoothing_gap, double qp_derivative_blend_gap,
                                          double qp_derivative_blend_weight,
                                          EnergyMortarPenaltyMode penalty_mode,
                                          EnergyMortarNodalEnergyBasis nodal_energy_basis,
                                          bool nodal_energy_angle_smoothing, RealT residual_gap )
    // NOTE: mesh1 maps to mesh2_ and mesh2 maps to mesh1_. This is to keep consistent with mesh1_ being non-mortar and
    // mesh2_ being mortar as is typical in the literature, but different from Tribol convention.
    : use_penalty_( use_penalty ), mesh_data_( mesh_data ), submesh_data_( submesh_data ), jac_data_( jac_data )
{
  params_.k = k;
  params_.del = delta;
  params_.N = N;
  params_.enzyme_quadrature = enzyme_quadrature;
  params_.fixed_integration_jacobian = fixed_integration_jacobian;
  params_.normal_mode = normal_mode;
  params_.projection_smoothing = projection_smoothing;
  params_.h1_active_set_smoothing_gap = h1_active_set_smoothing_gap;
  params_.qp_derivative_blend_gap = qp_derivative_blend_gap;
  if ( qp_derivative_blend_weight > 1.0 ) {
    qp_derivative_blend_weight = 1.0;
  }
  params_.qp_derivative_blend_weight = qp_derivative_blend_weight;
  params_.penalty_mode = penalty_mode;
  params_.nodal_energy_basis = nodal_energy_basis;
  params_.nodal_energy_angle_smoothing = nodal_energy_angle_smoothing;
  params_.residual_gap = residual_gap;

  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );

  // Allocate the (pressure) true-dof vector early so host code can set it via tribol::getMfemTDofPressure() after the
  // formulation is created. In penalty mode this is overwritten in updateNodalForces(); in LM mode it is treated as the
  // Lagrange multiplier vector (lambda).
  pressure_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  pressure_vec_.fill( 0.0 );

  submesh_nodal_normal_.SetSpace( const_cast<mfem::ParFiniteElementSpace*>( &mesh_data_.GetSubmeshFESpace() ) );
  submesh_nodal_normal_ = 0.0;
}

void EnergyMortarAdapter::updateMeshes( MeshData& mesh1, MeshData& mesh2 )
{
  // Maintain the same "flipped" convention as the constructor.
  mesh1_ = &mesh2;
  mesh2_ = &mesh1;
}

void EnergyMortarAdapter::updatePenaltyParameters( bool use_penalty, double k )
{
  use_penalty_ = use_penalty;
  params_.k = k;
}

void EnergyMortarAdapter::updateEnergyMortarNormalMode( EnergyMortarNormalMode normal_mode, bool projection_smoothing )
{
  params_.normal_mode = normal_mode;
  params_.projection_smoothing = projection_smoothing;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
  redecomp_nodal_normal_.reset();
  submesh_nodal_normal_ = 0.0;
}

void EnergyMortarAdapter::updateEnergyMortarEnzymeQuadrature( bool enabled )
{
  params_.enzyme_quadrature = enabled;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
}

void EnergyMortarAdapter::updateEnergyMortarFixedIntegrationJacobian( bool enabled )
{
  params_.fixed_integration_jacobian = enabled;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
}

void EnergyMortarAdapter::updateEnergyMortarH1ActiveSetSmoothing( RealT gap_transition )
{
  params_.h1_active_set_smoothing_gap = gap_transition;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
}

void EnergyMortarAdapter::updateEnergyMortarQpDerivativeBlendGap( RealT gap_transition )
{
  params_.qp_derivative_blend_gap = gap_transition;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
}

void EnergyMortarAdapter::updateEnergyMortarQpDerivativeBlendWeight( RealT weight )
{
  if ( weight > 1.0 ) {
    weight = 1.0;
  }
  params_.qp_derivative_blend_weight = weight;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
}

void EnergyMortarAdapter::updateEnergyMortarPenaltyMode( EnergyMortarPenaltyMode mode )
{
  params_.penalty_mode = mode;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
}

void EnergyMortarAdapter::updateEnergyMortarNodalEnergyBasis( EnergyMortarNodalEnergyBasis basis )
{
  params_.nodal_energy_basis = basis;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
}

void EnergyMortarAdapter::updateEnergyMortarNodalEnergyAngleSmoothing( bool enabled )
{
  params_.nodal_energy_angle_smoothing = enabled;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
}

void EnergyMortarAdapter::updateResidualGap( RealT residual_gap )
{
  params_.residual_gap = residual_gap;
  evaluator_ = std::make_unique<EnergyMortarCalculator>( params_ );
}

const mfem::HypreParVector& EnergyMortarAdapter::getMfemGap() const
{
  // Penalty mode uses the normalized gap g = g_tilde / A. LM mode enforces the unnormalized constraint g_tilde = 0,
  // consistent with dg/dx returned by getMfemDgDx().
  return use_penalty_ ? gap_vec_.get() : g_tilde_vec_.get();
}

mfem::ParGridFunction* EnergyMortarAdapter::getMfemNodalNormal()
{
  if ( params_.normal_mode != EnergyMortarNormalMode::H1_NODAL_NORMAL || !redecomp_nodal_normal_ ) {
    return nullptr;
  }
  return &submesh_nodal_normal_;
}

void EnergyMortarAdapter::setInterfacePairs( ArrayT<InterfacePair>&& pairs, int /*check_level*/ )
{
  // TODO: Consider design and how this interacts with binning and CG
  pairs_ = std::move( pairs );
}

void EnergyMortarAdapter::updateIntegrationRule()
{
  SLIC_WARNING_ROOT( "Update integration rule not implemmented for any method" );
  // TODO: break out integration rule as a separate method
}

void EnergyMortarAdapter::updateNodalGaps()
{
  // NOTE: user should have called updateMfemParallelDecomposition() with updated coords before calling this

  // Tribol level data structures for storing gap, area, and derivatives
  auto& redecomp_gap = submesh_data_.GetRedecompGap();
  mfem::GridFunction redecomp_area( redecomp_gap.FESpace() );
  redecomp_area = 0.0;

  const bool use_lor = ( mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes = use_lor ? *mesh_data_.GetLORMeshFESpace() : mesh_data_.GetSubmeshFESpace();
  const auto& pressure_surface_fes = use_lor ? *submesh_data_.GetLORMeshFESpace() : submesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *mesh_data_.GetRedecompResponse().FESpace();
  const auto& pressure_redecomp_fes = *submesh_data_.GetRedecompGap().FESpace();
  const auto& mortar_elem_map = mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = mesh_data_.GetElemMap2();

  PackedPairJacobianContribs dg_lm_nm( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs dg_lm_m( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs dA_lm_nm( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs dA_lm_m( pressure_surface_fes, displacement_surface_fes, pressure_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );

  dg_lm_nm.reserve( pairs_.size(), 8 );
  dg_lm_m.reserve( pairs_.size(), 8 );
  dA_lm_nm.reserve( pairs_.size(), 8 );
  dA_lm_m.reserve( pairs_.size(), 8 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  SLIC_ERROR_ROOT_IF( mesh1_ == nullptr || mesh2_ == nullptr, "ENERGY_MORTAR meshes not set." );
  if ( params_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL ) {
    ReferenceScaledEdgeAvgNodalNormal2D normal_method;
    normal_method.Compute( *mesh1_ );
    normal_method.Compute( *mesh2_ );

    redecomp_nodal_normal_ = std::make_unique<mfem::GridFunction>(
        const_cast<mfem::FiniteElementSpace*>( mesh_data_.GetRedecompResponse().FESpace() ) );
    redecomp_nodal_normal_->UseDevice( false );
    ( *redecomp_nodal_normal_ ) = 0.0;
    copyNodalNormalsToGridFunction( *mesh1_, *redecomp_nodal_normal_ );
    copyNodalNormalsToGridFunction( *mesh2_, *redecomp_nodal_normal_ );
    mesh_data_.GetParentRedecompTransfer().RedecompToSubmesh( *redecomp_nodal_normal_, submesh_nodal_normal_ );
  }
  auto mesh1_view = mesh1_->getView();
  auto mesh2_view = mesh2_->getView();
  static int h1_debug_call{ 0 };
  const H1DebugConfig h1_debug_config = getH1DebugConfig();
  std::vector<H1DebugPair> h1_debug_pairs;
  if ( h1_debug_config.enabled && params_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL ) {
    ++h1_debug_call;
    h1_debug_pairs.reserve( pairs_.size() );
  }

  // Compute local contributions
  for ( const auto& pair : pairs_ ) {
    // These need to be flipped, since the pairs are determined with element 1 associated with mesh 1, and we flipped
    // the mesh numbers to be consistent with the literature and since the underlying method integrates on element 1
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    if ( params_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL ) {
      auto h1 = evaluator_->compute_h1_total_derivatives( flipped_pair, mesh1_view, mesh2_view, false );
      if ( h1_debug_config.enabled ) {
        recordH1DebugPair( h1_debug_pairs, pair, flipped_pair, h1, params_, mesh1_view, mesh2_view );
      }
      applyH1ActiveSetSmoothing( h1, params_.h1_active_set_smoothing_gap, area_tol_ );
      if ( h1.area[0] <= 0.0 && h1.area[1] <= 0.0 ) {
        continue;
      }

      auto A_conn = mesh1_view.getConnectivity()( elem1 );
      redecomp_gap( A_conn[0] ) += h1.g_tilde[0];
      redecomp_gap( A_conn[1] ) += h1.g_tilde[1];
      redecomp_area( A_conn[0] ) += h1.area[0];
      redecomp_area( A_conn[1] ) += h1.area[1];

      appendH1FirstDerivativeBlocks( h1, mesh1_view, mesh2_view, dg_lm_nm, dg_lm_m, elem1, false );
      appendH1FirstDerivativeBlocks( h1, mesh1_view, mesh2_view, dA_lm_nm, dA_lm_m, elem1, true );
      continue;
    }

    double g_tilde_elem[2];
    double A_elem[2];

    evaluator_->compute_gtilde_and_area( flipped_pair, mesh1_view, mesh2_view, g_tilde_elem, A_elem );

    if ( A_elem[0] <= 0.0 && A_elem[1] <= 0.0 ) {
      continue;
    }

    auto A_conn = mesh1_view.getConnectivity()( elem1 );

    // Add to nodes of Element A
    redecomp_gap( A_conn[0] ) += g_tilde_elem[0];
    redecomp_gap( A_conn[1] ) += g_tilde_elem[1];

    redecomp_area( A_conn[0] ) += A_elem[0];
    redecomp_area( A_conn[1] ) += A_elem[1];

    // compute g_tilde first derivative
    double dg_dx_node1[8];
    double dg_dx_node2[8];
    // TODO: make grad_gtilde return directly in dg_tilde_dx_blocks format
    evaluator_->grad_gtilde( flipped_pair, mesh1_view, mesh2_view, dg_dx_node1, dg_dx_node2 );
    double dg_tilde_dx_blocks[2][8];
    for ( int i{ 0 }; i < 4; ++i ) {
      dg_tilde_dx_blocks[0][i * 2] = dg_dx_node1[node_idx[i]];
      dg_tilde_dx_blocks[0][i * 2 + 1] = dg_dx_node2[node_idx[i]];
      dg_tilde_dx_blocks[1][i * 2] = dg_dx_node1[node_idx[i + 4]];
      dg_tilde_dx_blocks[1][i * 2 + 1] = dg_dx_node2[node_idx[i + 4]];
    }
    dg_lm_nm.append( elem1, elem1, dg_tilde_dx_blocks[0], 8 );
    dg_lm_m.append( elem1, elem2, dg_tilde_dx_blocks[1], 8 );

    double dA_dx_node1[8];
    double dA_dx_node2[8];
    // TODO: make grad_trib_area return directly in dA_dx_blocks format
    evaluator_->grad_trib_area( flipped_pair, mesh1_view, mesh2_view, dA_dx_node1, dA_dx_node2 );
    double dA_dx_blocks[2][8];
    for ( int i{ 0 }; i < 4; ++i ) {
      dA_dx_blocks[0][i * 2] = dA_dx_node1[node_idx[i]];
      dA_dx_blocks[0][i * 2 + 1] = dA_dx_node2[node_idx[i]];
      dA_dx_blocks[1][i * 2] = dA_dx_node1[node_idx[i + 4]];
      dA_dx_blocks[1][i * 2 + 1] = dA_dx_node2[node_idx[i + 4]];
    }
    dA_lm_nm.append( elem1, elem1, dA_dx_blocks[0], 8 );
    dA_lm_m.append( elem1, elem2, dA_dx_blocks[1], 8 );
  }
  flushH1DebugPairs( h1_debug_call, h1_debug_config, h1_debug_pairs );

  // Move gap and area to submesh level vectors
  mfem::ParLinearForm g_tilde_linear_form(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetSubmeshGap( g_tilde_linear_form );
  auto& P_submesh = *submesh_data_.GetSubmeshFESpace().GetProlongationMatrix();
  g_tilde_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  g_tilde_vec_.fill( 0.0 );
  P_submesh.MultTranspose( g_tilde_linear_form, g_tilde_vec_.get() );

  mfem::Array<int> rows_to_elim;
  if ( !tied_contact_ && use_penalty_ ) {
    rows_to_elim.Reserve( g_tilde_vec_.size() );
    for ( int i{ 0 }; i < g_tilde_vec_.size(); ++i ) {
      if ( g_tilde_vec_[i] > 0.0 ) {
        g_tilde_vec_[i] = 0.0;
        rows_to_elim.push_back( i );
      }
    }
  }

  mfem::ParLinearForm A_linear_form( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_data_.GetPressureTransfer().RedecompToSubmesh( redecomp_area, A_linear_form );
  A_vec_ = shared::ParVector( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  A_vec_.fill( 0.0 );
  P_submesh.MultTranspose( A_linear_form, A_vec_.get() );

  gap_vec_ = g_tilde_vec_.divide( A_vec_, area_tol_ );

  // Move gap and area derivatives to (pressure true-dof rows, displacement true-dof cols)
  std::vector<PackedPairJacobianContribs> dg_contribs;
  dg_contribs.reserve( 2 );
  dg_contribs.push_back( std::move( dg_lm_nm ) );
  dg_contribs.push_back( std::move( dg_lm_m ) );
  dg_tilde_dx_ = jac_data_.GetMfemJacobian( &submesh_data_.GetSubmeshFESpace(),
                                            mesh_data_.GetParentCoords().ParFESpace(), dg_contribs );
  if ( !tied_contact_ && use_penalty_ ) {
    // technically, we should do this on all the vectors/matrices below, but it looks like the mutliplication operators
    // below will zero them out anyway
    dg_tilde_dx_.eliminateRows( rows_to_elim );
  }

  std::vector<PackedPairJacobianContribs> dA_contribs;
  dA_contribs.reserve( 2 );
  dA_contribs.push_back( std::move( dA_lm_nm ) );
  dA_contribs.push_back( std::move( dA_lm_m ) );
  dA_dx_ = jac_data_.GetMfemJacobian( &submesh_data_.GetSubmeshFESpace(), mesh_data_.GetParentCoords().ParFESpace(),
                                      dA_contribs );
}

void EnergyMortarAdapter::updateQuadraturePointPenaltyForces()
{
  SLIC_ERROR_ROOT_IF( mesh1_ == nullptr || mesh2_ == nullptr, "ENERGY_MORTAR meshes not set." );

  const bool use_lor = ( mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes = use_lor ? *mesh_data_.GetLORMeshFESpace() : mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *mesh_data_.GetRedecompResponse().FESpace();
  const auto& mortar_elem_map = mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = mesh_data_.GetElemMap2();

  PackedPairJacobianContribs df_nm_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_nm_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs df_m_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, mortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_m_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                     displacement_redecomp_fes, mortar_elem_map, mortar_elem_map );

  df_nm_nm.reserve( pairs_.size(), 16 );
  df_nm_m.reserve( pairs_.size(), 16 );
  df_m_nm.reserve( pairs_.size(), 16 );
  df_m_m.reserve( pairs_.size(), 16 );

  mfem::GridFunction redecomp_force(
      const_cast<mfem::FiniteElementSpace*>( mesh_data_.GetRedecompResponse().FESpace() ) );
  redecomp_force = 0.0;
  const int scalar_size = redecomp_force.FESpace()->GetVSize() / redecomp_force.FESpace()->GetVDim();
  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  auto mesh1_view = mesh1_->getView();
  auto mesh2_view = mesh2_->getView();

  double local_energy = 0.0;
  for ( const auto& pair : pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    const auto qp_penalty = evaluator_->compute_quadrature_point_penalty_data( flipped_pair, mesh1_view, mesh2_view );
    if ( qp_penalty.energy == 0.0 ) {
      continue;
    }
    local_energy += qp_penalty.energy;

    if ( params_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL ) {
      for ( int i{ 0 }; i < qp_penalty.num_mesh1_nodes; ++i ) {
        const int node = qp_penalty.mesh1_nodes[i];
        redecomp_force( node ) += qp_penalty.h1_force[i];
        redecomp_force( scalar_size + node ) += qp_penalty.h1_force[qp_penalty.num_mesh1_nodes + i];
      }
      const int mesh2_offset = 2 * qp_penalty.num_mesh1_nodes;
      for ( int i{ 0 }; i < qp_penalty.num_mesh2_nodes; ++i ) {
        const int node = qp_penalty.mesh2_nodes[i];
        redecomp_force( node ) += qp_penalty.h1_force[mesh2_offset + i];
        redecomp_force( scalar_size + node ) += qp_penalty.h1_force[mesh2_offset + qp_penalty.num_mesh2_nodes + i];
      }

      std::map<std::pair<int, int>, std::array<double, 16>> nm_nm_blocks;
      std::map<std::pair<int, int>, std::array<double, 16>> nm_m_blocks;
      std::map<std::pair<int, int>, std::array<double, 16>> m_nm_blocks;
      std::map<std::pair<int, int>, std::array<double, 16>> m_m_blocks;
      addH1QuadraturePointPenaltyHessianBlocks( qp_penalty, mesh1_view, mesh2_view, nm_nm_blocks, nm_m_blocks,
                                                m_nm_blocks, m_m_blocks );
      appendH1HessianBlockMaps( nm_nm_blocks, nm_m_blocks, m_nm_blocks, m_m_blocks, df_nm_nm, df_nm_m, df_m_nm,
                                df_m_m );
      continue;
    }

    auto A_conn = mesh1_view.getConnectivity()( elem1 );
    auto B_conn = mesh2_view.getConnectivity()( elem2 );
    redecomp_force( A_conn[0] ) += qp_penalty.force[0];
    redecomp_force( scalar_size + A_conn[0] ) += qp_penalty.force[1];
    redecomp_force( A_conn[1] ) += qp_penalty.force[2];
    redecomp_force( scalar_size + A_conn[1] ) += qp_penalty.force[3];
    redecomp_force( B_conn[0] ) += qp_penalty.force[4];
    redecomp_force( scalar_size + B_conn[0] ) += qp_penalty.force[5];
    redecomp_force( B_conn[1] ) += qp_penalty.force[6];
    redecomp_force( scalar_size + B_conn[1] ) += qp_penalty.force[7];

    double df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            const auto idx = node_idx[l + i * 4] + node_idx[k + j * 4] * 8;
            df_dx_blocks[i][j][l + k * 4] = qp_penalty.stiffness[idx];
          }
        }
      }
    }

    df_nm_nm.append( elem1, elem1, df_dx_blocks[0][0], 16 );
    df_nm_m.append( elem1, elem2, df_dx_blocks[0][1], 16 );
    df_m_nm.append( elem2, elem1, df_dx_blocks[1][0], 16 );
    df_m_m.append( elem2, elem2, df_dx_blocks[1][1], 16 );
  }

  MPI_Allreduce( &local_energy, &energy_, 1, MPI_DOUBLE, MPI_SUM,
                 mesh_data_.GetParentCoords().ParFESpace()->GetComm() );

  mfem::Vector parent_force_dof( mesh_data_.GetParentCoords().ParFESpace()->GetVSize() );
  parent_force_dof = 0.0;
  mesh_data_.GetParentRedecompTransfer().RedecompToParent( redecomp_force, parent_force_dof );

  auto* parent_fes = const_cast<mfem::ParFiniteElementSpace*>( mesh_data_.GetParentCoords().ParFESpace() );
  force_vec_ = shared::ParVector( parent_fes );
  force_vec_.fill( 0.0 );
  parent_fes->GetProlongationMatrix()->MultTranspose( parent_force_dof, force_vec_.get() );

  std::vector<PackedPairJacobianContribs> df_contribs;
  df_contribs.reserve( 4 );
  df_contribs.push_back( std::move( df_nm_nm ) );
  df_contribs.push_back( std::move( df_nm_m ) );
  df_contribs.push_back( std::move( df_m_nm ) );
  df_contribs.push_back( std::move( df_m_m ) );
  df_dx_ = jac_data_.GetMfemJacobian( mesh_data_.GetParentCoords().ParFESpace(),
                                      mesh_data_.GetParentCoords().ParFESpace(), df_contribs );
}

void EnergyMortarAdapter::updateNodalForces()
{
  // NOTE: user should have called updateNodalGaps() with updated coords before calling this

  if ( use_penalty_ ) {
    // Penalty mode: p = k * (g_tilde / A)
    pressure_vec_ = params_.k * gap_vec_;
  } else {
    // LM mode: pressure_vec_ is treated as the Lagrange multiplier vector (lambda)
    SLIC_ERROR_ROOT_IF( submesh_data_.GetSubmeshFESpace().GetTrueVSize() != pressure_vec_.size(),
                        "LM vector is not initialized. Call tribol::update() once to initialize the formulation." );
    SLIC_ERROR_ROOT_IF( pressure_vec_.size() != g_tilde_vec_.size(),
                        "LM vector size mismatch with contact dofs (g_tilde)." );
  }

  if ( use_penalty_ && ( params_.penalty_mode == EnergyMortarPenaltyMode::QUADRATURE_POINT_GAP ||
                         params_.penalty_mode == EnergyMortarPenaltyMode::NODAL_ENERGY ) ) {
    updateQuadraturePointPenaltyForces();
    return;
  }

  energy_ = pressure_vec_.dot( g_tilde_vec_ );

  if ( !use_penalty_ ) {
    // -------------------------------------------------------------------------
    // LM mode: force = G^T * lambda and df/dx = lambda · d^2(g_tilde)/dx^2
    // -------------------------------------------------------------------------
    force_vec_ = pressure_vec_ * dg_tilde_dx_;

    mfem::GridFunction redecomp_lambda( submesh_data_.GetRedecompGap() );
    mfem::ParGridFunction submesh_lambda(
        const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
    submesh_lambda.SetFromTrueDofs( pressure_vec_.get() );
    submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_lambda, redecomp_lambda );

    df_dx_ = computeDfDxSecondDerivativesLM( redecomp_lambda );
    return;
  }

  // ---------------------------------------------------------------------------
  // Penalty mode: force and Jacobian include pressure/area coupling terms
  // ---------------------------------------------------------------------------
  auto k_over_a = params_.k * A_vec_.inverse( area_tol_ );
  auto p_over_a = pressure_vec_.divide( A_vec_, area_tol_ );

  shared::ParSparseMat dp_dx( dg_tilde_dx_.get() );
  dp_dx->ScaleRows( k_over_a.get() );
  shared::ParSparseMat dp_dx_temp( dA_dx_.get() );
  dp_dx_temp->ScaleRows( p_over_a.get() );
  dp_dx -= dp_dx_temp;

  force_vec_ = ( pressure_vec_ * dg_tilde_dx_ ) + ( g_tilde_vec_ * dp_dx );

  // TODO (EBC): Move transfer path-specific logic out of this file
  mfem::GridFunction redecomp_pressure( submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_pressure(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_pressure.SetFromTrueDofs( pressure_vec_.get() );
  submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_pressure, redecomp_pressure );

  mfem::GridFunction redecomp_g_tilde( submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_g_tilde(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_g_tilde.SetFromTrueDofs( g_tilde_vec_.get() );
  submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_g_tilde, redecomp_g_tilde );

  mfem::GridFunction redecomp_A( submesh_data_.GetRedecompGap() );
  mfem::ParGridFunction submesh_A( const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() ) );
  submesh_A.SetFromTrueDofs( A_vec_.get() );
  submesh_data_.GetPressureTransfer().SubmeshToRedecomp( submesh_A, redecomp_A );

  df_dx_ = computeDfDxSecondDerivativesPenalty( redecomp_pressure, redecomp_g_tilde, redecomp_A );

  auto pg2_over_asq = ( 2.0 * pressure_vec_ )
                          .multiplyInPlace( g_tilde_vec_ )
                          .divideInPlace( A_vec_, area_tol_ )
                          .divideInPlace( A_vec_, area_tol_ );

  auto& submesh_fes = submesh_data_.GetSubmeshFESpace();
  auto p_over_a_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                             submesh_fes.GetTrueDofOffsets(), p_over_a.get() );
  auto pg2_over_asq_diag = shared::ParSparseMat::diagonalMatrix( submesh_fes.GetComm(), submesh_fes.GlobalTrueVSize(),
                                                                 submesh_fes.GetTrueDofOffsets(), pg2_over_asq.get() );

  df_dx_ -= shared::ParSparseMat::rap( dg_tilde_dx_, p_over_a_diag, dA_dx_ );
  df_dx_ -= shared::ParSparseMat::rap( dA_dx_, p_over_a_diag, dg_tilde_dx_ );
  df_dx_ += shared::ParSparseMat::rap( dA_dx_, pg2_over_asq_diag, dg_tilde_dx_ );
  df_dx_ += dp_dx.transpose() * dg_tilde_dx_;
  df_dx_ += dg_tilde_dx_.transpose() * dp_dx;
}

RealT EnergyMortarAdapter::computeTimeStep()
{
  SLIC_INFO_ROOT( "computeTimestep() not implemented for EnergyMortar" );
  // TODO: implement timestep calculation
  return 1.0;
}

shared::ParSparseMat EnergyMortarAdapter::computeDfDxSecondDerivativesLM( const mfem::GridFunction& redecomp_lambda )
{
  const bool use_lor = ( mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes = use_lor ? *mesh_data_.GetLORMeshFESpace() : mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *mesh_data_.GetRedecompResponse().FESpace();
  const auto& mortar_elem_map = mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = mesh_data_.GetElemMap2();

  PackedPairJacobianContribs df_nm_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_nm_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs df_m_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, mortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_m_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                     displacement_redecomp_fes, mortar_elem_map, mortar_elem_map );

  df_nm_nm.reserve( pairs_.size(), 16 );
  df_nm_m.reserve( pairs_.size(), 16 );
  df_m_nm.reserve( pairs_.size(), 16 );
  df_m_m.reserve( pairs_.size(), 16 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  SLIC_ERROR_ROOT_IF( mesh1_ == nullptr || mesh2_ == nullptr, "ENERGY_MORTAR meshes not set." );
  auto mesh1_view = mesh1_->getView();
  auto mesh2_view = mesh2_->getView();

  for ( auto& pair : pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto node11 = mesh1_view.getConnectivity()( elem1, 0 );
    const auto node12 = mesh1_view.getConnectivity()( elem1, 1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    const RealT lambda1 = redecomp_lambda( node11 );
    const RealT lambda2 = redecomp_lambda( node12 );

    if ( params_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL ) {
      auto h1 = evaluator_->compute_h1_total_derivatives( flipped_pair, mesh1_view, mesh2_view );
      applyH1ActiveSetSmoothing( h1, params_.h1_active_set_smoothing_gap, area_tol_ );
      std::map<std::pair<int, int>, std::array<double, 16>> nm_nm_blocks;
      std::map<std::pair<int, int>, std::array<double, 16>> nm_m_blocks;
      std::map<std::pair<int, int>, std::array<double, 16>> m_nm_blocks;
      std::map<std::pair<int, int>, std::array<double, 16>> m_m_blocks;
      addH1HessianBlocks( h1, mesh1_view, mesh2_view, lambda1, lambda2, h1.d2g1_dx2, h1.d2g2_dx2, nm_nm_blocks,
                          nm_m_blocks, m_nm_blocks, m_m_blocks );
      appendH1HessianBlockMaps( nm_nm_blocks, nm_m_blocks, m_nm_blocks, m_m_blocks, df_nm_nm, df_nm_m, df_m_nm,
                                df_m_m );
      continue;
    }

    double d2g_dx2_node1[64];
    double d2g_dx2_node2[64];
    evaluator_->d2_g2tilde( flipped_pair, mesh1_view, mesh2_view, d2g_dx2_node1, d2g_dx2_node2 );

    double df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            const auto idx = node_idx[l + i * 4] + node_idx[k + j * 4] * 8;
            df_dx_blocks[i][j][l + k * 4] = lambda1 * d2g_dx2_node1[idx] + lambda2 * d2g_dx2_node2[idx];
          }
        }
      }
    }

    df_nm_nm.append( elem1, elem1, df_dx_blocks[0][0], 16 );
    df_nm_m.append( elem1, elem2, df_dx_blocks[0][1], 16 );
    df_m_nm.append( elem2, elem1, df_dx_blocks[1][0], 16 );
    df_m_m.append( elem2, elem2, df_dx_blocks[1][1], 16 );
  }

  std::vector<PackedPairJacobianContribs> df_contribs;
  df_contribs.reserve( 4 );
  df_contribs.push_back( std::move( df_nm_nm ) );
  df_contribs.push_back( std::move( df_nm_m ) );
  df_contribs.push_back( std::move( df_m_nm ) );
  df_contribs.push_back( std::move( df_m_m ) );
  return jac_data_.GetMfemJacobian( mesh_data_.GetParentCoords().ParFESpace(),
                                    mesh_data_.GetParentCoords().ParFESpace(), df_contribs );
}

shared::ParSparseMat EnergyMortarAdapter::computeDfDxSecondDerivativesPenalty(
    const mfem::GridFunction& redecomp_pressure, const mfem::GridFunction& redecomp_g_tilde,
    const mfem::GridFunction& redecomp_A )
{
  const bool use_lor = ( mesh_data_.GetLORMesh() != nullptr );
  const auto& displacement_surface_fes = use_lor ? *mesh_data_.GetLORMeshFESpace() : mesh_data_.GetSubmeshFESpace();
  const auto& displacement_redecomp_fes = *mesh_data_.GetRedecompResponse().FESpace();
  const auto& mortar_elem_map = mesh_data_.GetElemMap1();
  const auto& nonmortar_elem_map = mesh_data_.GetElemMap2();

  PackedPairJacobianContribs df_nm_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                       displacement_redecomp_fes, nonmortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_nm_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, nonmortar_elem_map, mortar_elem_map );
  PackedPairJacobianContribs df_m_nm( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                      displacement_redecomp_fes, mortar_elem_map, nonmortar_elem_map );
  PackedPairJacobianContribs df_m_m( displacement_surface_fes, displacement_surface_fes, displacement_redecomp_fes,
                                     displacement_redecomp_fes, mortar_elem_map, mortar_elem_map );

  df_nm_nm.reserve( pairs_.size(), 16 );
  df_nm_m.reserve( pairs_.size(), 16 );
  df_m_nm.reserve( pairs_.size(), 16 );
  df_m_m.reserve( pairs_.size(), 16 );

  const int node_idx[8] = { 0, 2, 1, 3, 4, 6, 5, 7 };

  SLIC_ERROR_ROOT_IF( mesh1_ == nullptr || mesh2_ == nullptr, "ENERGY_MORTAR meshes not set." );
  auto mesh1_view = mesh1_->getView();
  auto mesh2_view = mesh2_->getView();

  for ( auto& pair : pairs_ ) {
    InterfacePair flipped_pair( pair.m_element_id2, pair.m_element_id1 );
    const auto elem1 = static_cast<int>( flipped_pair.m_element_id1 );
    const auto node11 = mesh1_view.getConnectivity()( elem1, 0 );
    const auto node12 = mesh1_view.getConnectivity()( elem1, 1 );
    const auto elem2 = static_cast<int>( flipped_pair.m_element_id2 );

    const RealT pressure1 = 2.0 * redecomp_pressure( node11 );
    const RealT pressure2 = 2.0 * redecomp_pressure( node12 );

    if ( pressure1 == 0.0 && pressure2 == 0.0 ) {
      continue;
    }

    const RealT g_p_ainv1 = std::abs( redecomp_A( node11 ) ) > area_tol_
                                ? -redecomp_g_tilde( node11 ) * redecomp_pressure( node11 ) / redecomp_A( node11 )
                                : 0.0;
    const RealT g_p_ainv2 = std::abs( redecomp_A( node12 ) ) > area_tol_
                                ? -redecomp_g_tilde( node12 ) * redecomp_pressure( node12 ) / redecomp_A( node12 )
                                : 0.0;

    if ( params_.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL ) {
      auto h1 = evaluator_->compute_h1_total_derivatives( flipped_pair, mesh1_view, mesh2_view );
      applyH1ActiveSetSmoothing( h1, params_.h1_active_set_smoothing_gap, area_tol_ );
      std::map<std::pair<int, int>, std::array<double, 16>> nm_nm_blocks;
      std::map<std::pair<int, int>, std::array<double, 16>> nm_m_blocks;
      std::map<std::pair<int, int>, std::array<double, 16>> m_nm_blocks;
      std::map<std::pair<int, int>, std::array<double, 16>> m_m_blocks;
      addH1HessianBlocks( h1, mesh1_view, mesh2_view, pressure1, pressure2, h1.d2g1_dx2, h1.d2g2_dx2, nm_nm_blocks,
                          nm_m_blocks, m_nm_blocks, m_m_blocks );
      addH1HessianBlocks( h1, mesh1_view, mesh2_view, g_p_ainv1, g_p_ainv2, h1.d2A1_dx2, h1.d2A2_dx2, nm_nm_blocks,
                          nm_m_blocks, m_nm_blocks, m_m_blocks );
      appendH1HessianBlockMaps( nm_nm_blocks, nm_m_blocks, m_nm_blocks, m_m_blocks, df_nm_nm, df_nm_m, df_m_nm,
                                df_m_m );
      continue;
    }

    double d2g_dx2_node1[64];
    double d2g_dx2_node2[64];
    evaluator_->d2_g2tilde( flipped_pair, mesh1_view, mesh2_view, d2g_dx2_node1, d2g_dx2_node2 );

    double d2A_dx2_node1[64];
    double d2A_dx2_node2[64];
    evaluator_->compute_d2A_d2u( flipped_pair, mesh1_view, mesh2_view, d2A_dx2_node1, d2A_dx2_node2 );

    double df_dx_blocks[2][2][16];
    for ( int i{ 0 }; i < 2; ++i ) {
      for ( int j{ 0 }; j < 2; ++j ) {
        for ( int k{ 0 }; k < 4; ++k ) {
          for ( int l{ 0 }; l < 4; ++l ) {
            const auto idx = node_idx[l + i * 4] + node_idx[k + j * 4] * 8;
            df_dx_blocks[i][j][l + k * 4] = pressure1 * d2g_dx2_node1[idx] + pressure2 * d2g_dx2_node2[idx] +
                                            g_p_ainv1 * d2A_dx2_node1[idx] + g_p_ainv2 * d2A_dx2_node2[idx];
          }
        }
      }
    }

    df_nm_nm.append( elem1, elem1, df_dx_blocks[0][0], 16 );
    df_nm_m.append( elem1, elem2, df_dx_blocks[0][1], 16 );
    df_m_nm.append( elem2, elem1, df_dx_blocks[1][0], 16 );
    df_m_m.append( elem2, elem2, df_dx_blocks[1][1], 16 );
  }

  std::vector<PackedPairJacobianContribs> df_contribs;
  df_contribs.reserve( 4 );
  df_contribs.push_back( std::move( df_nm_nm ) );
  df_contribs.push_back( std::move( df_nm_m ) );
  df_contribs.push_back( std::move( df_m_nm ) );
  df_contribs.push_back( std::move( df_m_m ) );
  return jac_data_.GetMfemJacobian( mesh_data_.GetParentCoords().ParFESpace(),
                                    mesh_data_.GetParentCoords().ParFESpace(), df_contribs );
}

std::unique_ptr<mfem::HypreParMatrix> EnergyMortarAdapter::getMfemDfDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( df_dx_.release() );
}

std::unique_ptr<mfem::HypreParMatrix> EnergyMortarAdapter::getMfemDgDx() const
{
  return std::unique_ptr<mfem::HypreParMatrix>( dg_tilde_dx_.release() );
}

std::unique_ptr<mfem::HypreParMatrix> EnergyMortarAdapter::getMfemDfDp() const
{
  if ( use_penalty_ ) {
    return nullptr;
  }
  // TODO (EBC): figure out better lifetime for this. if you called getMfemDgDx() before this, then dg_tilde_dx_ will be
  // null.
  // LM mode: df/dlambda = (d g_tilde / dx)^T
  auto df_dlambda = dg_tilde_dx_.transpose();
  return std::unique_ptr<mfem::HypreParMatrix>( df_dlambda.release() );
}

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
