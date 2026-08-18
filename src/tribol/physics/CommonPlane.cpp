// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "CommonPlane.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

#include "axom/fmt.hpp"

#include "tribol/common/LoopExec.hpp"
#include "tribol/common/Atomics.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#ifdef BUILD_REDECOMP
#include "tribol/mesh/MfemData.hpp"
#endif
#include "tribol/geom/CompGeom.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/integ/Integration.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/Math.hpp"

namespace tribol {

namespace {

constexpr int max_dim = 3;
constexpr int max_nodes_per_face = 4;
constexpr int max_nodes_per_overlap = 10;
constexpr int parent_q2_num_nodes = 3;

TRIBOL_HOST_DEVICE inline void EvalParentQ2Basis( RealT xi, RealT* phi, RealT* dphi );

struct ProjectionConstraint {
  IndexT plane_index;
  IndexT element1;
  IndexT element2;
  RealT normal[2];
  RealT phi1[parent_q2_num_nodes];
  RealT phi2[parent_q2_num_nodes];
  RealT gap;
  RealT target_velocity;
  RealT diagonal;
  RealT quadrature_measure;
  RealT trial_velocity;
  RealT position_trial_velocity;
  RealT multiplier{ 0. };
};

struct ProjectionResiduals {
  RealT complementarity{ 0. };
  RealT primal{ 0. };
};

struct ProjectionOperatorDiagnostics {
  IndexT velocity_dofs{ 0 };
  IndexT rank{ 0 };
  RealT minimum_eigenvalue{ 0. };
  RealT maximum_eigenvalue{ 0. };
  RealT condition_estimate{ 0. };
  RealT jacobi_contraction{ 0. };
};

struct TraceProjectionContribution {
  IndexT plane_index;
  IndexT element1;
  IndexT element2;
  RealT normal[2];
  RealT phi1[parent_q2_num_nodes];
  RealT phi2[parent_q2_num_nodes];
  RealT plane_weight;
};

struct TraceProjectionConstraint {
  IndexT parent_dof;
  IndexT patch;
  std::vector<TraceProjectionContribution> contributions;
  RealT gap{ 0. };
  RealT target_velocity{ 0. };
  RealT diagonal{ 0. };
  RealT trial_velocity{ 0. };
  RealT position_trial_velocity{ 0. };
  RealT tributary_area{ 0. };
  RealT weighted_penalty_stiffness{ 0. };
  RealT minimum_thickness{ std::numeric_limits<RealT>::infinity() };
  RealT spring_stiffness{ 0. };
  RealT damping_coefficient{ 0. };
  RealT compliant_multiplier{ 0. };
  RealT guard_multiplier{ 0. };
  RealT multiplier{ 0. };
};

struct TraceProjectionSolveResult {
  bool valid{ true };
  bool complementarity_converged{ true };
  int iterations{ 0 };
  RealT initial_residual{ 0. };
  RealT final_residual{ 0. };
  RealT final_primal_residual{ 0. };
  RealT primal_tolerance{ 0. };
};

struct ParentTraceFaceKey {
  std::array<IndexT, parent_q2_num_nodes> dofs;

  bool operator<( const ParentTraceFaceKey& other ) const { return dofs < other.dofs; }
};

struct ParentTraceFace {
  ParentTraceFaceKey key;
  IndexT representative_element;
  std::array<IndexT, parent_q2_num_nodes> local_dofs;
  std::array<IndexT, parent_q2_num_nodes> rows;
  RealT normal[2];
  RealT inverse_mass[parent_q2_num_nodes][parent_q2_num_nodes];
};

RealT EvaluateProjectionVelocity( const ProjectionConstraint& constraint, const MeshData::Viewer& mesh1,
                                  const MeshData::Viewer& mesh2 )
{
  RealT velocity = 0.;
  for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
    for ( int d = 0; d < 2; ++d ) {
      velocity += constraint.normal[d] *
                  ( constraint.phi1[a] * mesh1.getParentVelocity( constraint.element1, a, d ) -
                    constraint.phi2[a] * mesh2.getParentVelocity( constraint.element2, a, d ) );
    }
  }
  return velocity;
}

RealT EvaluateProjectionBaseVelocity( const ProjectionConstraint& constraint, const MeshData::Viewer& mesh1,
                                      const MeshData::Viewer& mesh2 )
{
  RealT velocity = 0.;
  for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
    for ( int d = 0; d < 2; ++d ) {
      velocity +=
          constraint.normal[d] *
          ( constraint.phi1[a] * mesh1.getParentProjectionBaseVelocity( constraint.element1, a, d ) -
            constraint.phi2[a] * mesh2.getParentProjectionBaseVelocity( constraint.element2, a, d ) );
    }
  }
  return velocity;
}

void AccumulateProjectionImpulse( const ProjectionConstraint& constraint, RealT multiplier_increment,
                                  const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
    for ( int d = 0; d < 2; ++d ) {
      const RealT impulse = multiplier_increment * constraint.normal[d];
      mesh1.getParentResponse( constraint.element1, a, d ) += constraint.phi1[a] * impulse;
      mesh2.getParentResponse( constraint.element2, a, d ) -= constraint.phi2[a] * impulse;
    }
  }
}

RealT EvaluateProjectionVelocity( const TraceProjectionConstraint& constraint, const MeshData::Viewer& mesh1,
                                  const MeshData::Viewer& mesh2 )
{
  RealT velocity = 0.;
  for ( const auto& contribution : constraint.contributions ) {
    for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
      for ( int d = 0; d < 2; ++d ) {
        velocity += contribution.normal[d] *
                    ( contribution.phi1[a] * mesh1.getParentVelocity( contribution.element1, a, d ) -
                      contribution.phi2[a] * mesh2.getParentVelocity( contribution.element2, a, d ) );
      }
    }
  }
  return velocity;
}

RealT EvaluateProjectionBaseVelocity( const TraceProjectionConstraint& constraint, const MeshData::Viewer& mesh1,
                                      const MeshData::Viewer& mesh2 )
{
  RealT velocity = 0.;
  for ( const auto& contribution : constraint.contributions ) {
    for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
      for ( int d = 0; d < 2; ++d ) {
        velocity +=
            contribution.normal[d] *
            ( contribution.phi1[a] *
                  mesh1.getParentProjectionBaseVelocity( contribution.element1, a, d ) -
              contribution.phi2[a] *
                  mesh2.getParentProjectionBaseVelocity( contribution.element2, a, d ) );
      }
    }
  }
  return velocity;
}

void AccumulateProjectionImpulse( const TraceProjectionConstraint& constraint, RealT multiplier_increment,
                                  const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 )
{
  for ( const auto& contribution : constraint.contributions ) {
    for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
      for ( int d = 0; d < 2; ++d ) {
        const RealT impulse = multiplier_increment * contribution.normal[d];
        mesh1.getParentResponse( contribution.element1, a, d ) += contribution.phi1[a] * impulse;
        mesh2.getParentResponse( contribution.element2, a, d ) -= contribution.phi2[a] * impulse;
      }
    }
  }
}

ParentTraceFaceKey GetParentTraceFaceKey( const MeshData::Viewer& mesh, IndexT element )
{
  ParentTraceFaceKey key;
  for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
    key.dofs[a] = mesh.getParentDofId( element, a );
  }
  std::sort( key.dofs.begin(), key.dofs.end() );
  return key;
}

bool InvertThreeByThree( const RealT matrix[3][3], RealT inverse[3][3] )
{
  RealT matrix_scale = 0.;
  for ( int i = 0; i < 3; ++i ) {
    for ( int j = 0; j < 3; ++j ) {
      matrix_scale = std::max( matrix_scale, std::abs( matrix[i][j] ) );
    }
  }
  if ( !std::isfinite( matrix_scale ) || matrix_scale <= 0. ) {
    return false;
  }

  RealT augmented[3][6];
  for ( int i = 0; i < 3; ++i ) {
    for ( int j = 0; j < 3; ++j ) {
      augmented[i][j] = matrix[i][j] / matrix_scale;
      augmented[i][j + 3] = i == j ? 1. : 0.;
    }
  }
  for ( int pivot = 0; pivot < 3; ++pivot ) {
    int pivot_row = pivot;
    for ( int row = pivot + 1; row < 3; ++row ) {
      if ( std::abs( augmented[row][pivot] ) > std::abs( augmented[pivot_row][pivot] ) ) {
        pivot_row = row;
      }
    }
    const RealT scale = std::abs( augmented[pivot_row][pivot] );
    if ( !std::isfinite( scale ) || scale <= 100. * std::numeric_limits<RealT>::epsilon() ) {
      return false;
    }
    if ( pivot_row != pivot ) {
      for ( int column = 0; column < 6; ++column ) {
        std::swap( augmented[pivot][column], augmented[pivot_row][column] );
      }
    }
    const RealT diagonal = augmented[pivot][pivot];
    for ( int column = 0; column < 6; ++column ) {
      augmented[pivot][column] /= diagonal;
    }
    for ( int row = 0; row < 3; ++row ) {
      if ( row == pivot ) {
        continue;
      }
      const RealT factor = augmented[row][pivot];
      for ( int column = 0; column < 6; ++column ) {
        augmented[row][column] -= factor * augmented[pivot][column];
      }
    }
  }
  for ( int i = 0; i < 3; ++i ) {
    for ( int j = 0; j < 3; ++j ) {
      inverse[i][j] = augmented[i][j + 3] / matrix_scale;
    }
  }
  return true;
}

bool BuildParentTraceFace( const MeshData::Viewer& mortar_mesh, IndexT representative_element,
                           ParentTraceFace& face )
{
  face.key = GetParentTraceFaceKey( mortar_mesh, representative_element );
  face.representative_element = representative_element;
  face.rows.fill( -1 );
  for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
    face.local_dofs[a] = mortar_mesh.getParentDofId( representative_element, a );
  }

  RealT midpoint_phi[3];
  RealT midpoint_dphi[3];
  EvalParentQ2Basis( 0.5, midpoint_phi, midpoint_dphi );
  RealT tangent[2] = { 0., 0. };
  for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
    tangent[0] += midpoint_dphi[a] * mortar_mesh.getParentPosition( representative_element, a, 0 );
    tangent[1] += midpoint_dphi[a] * mortar_mesh.getParentPosition( representative_element, a, 1 );
  }
  const RealT tangent_norm = magnitude( tangent[0], tangent[1] );
  if ( !std::isfinite( tangent_norm ) || tangent_norm <= 1.e-14 ) {
    return false;
  }
  face.normal[0] = tangent[1] / tangent_norm;
  face.normal[1] = -tangent[0] / tangent_norm;
  RealT chord_normal[2];
  mortar_mesh.getFaceNormal( representative_element, chord_normal );
  if ( face.normal[0] * chord_normal[0] + face.normal[1] * chord_normal[1] < 0. ) {
    face.normal[0] = -face.normal[0];
    face.normal[1] = -face.normal[1];
  }

  RealT mass[3][3] = { { 0., 0., 0. }, { 0., 0., 0. }, { 0., 0., 0. } };
  RealT weights[max_segment_gauss_legendre_qpts] = { 0. };
  RealT coordinates[max_segment_gauss_legendre_qpts] = { 0. };
  const int num_points = GetCommonPlaneSegmentRule( 6, weights, coordinates );
  for ( int qp = 0; qp < num_points; ++qp ) {
    RealT phi[3];
    RealT dphi[3];
    EvalParentQ2Basis( coordinates[qp], phi, dphi );
    RealT derivative[2] = { 0., 0. };
    for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
      derivative[0] += dphi[a] * mortar_mesh.getParentPosition( representative_element, a, 0 );
      derivative[1] += dphi[a] * mortar_mesh.getParentPosition( representative_element, a, 1 );
    }
    const RealT jacobian = magnitude( derivative[0], derivative[1] );
    if ( !std::isfinite( jacobian ) || jacobian <= 0. ) {
      return false;
    }
    for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
      for ( int b = 0; b < parent_q2_num_nodes; ++b ) {
        mass[a][b] += weights[qp] * jacobian * phi[a] * phi[b];
      }
    }
  }
  return InvertThreeByThree( mass, face.inverse_mass );
}

#ifdef BUILD_REDECOMP
std::vector<RealT> ComputeSymmetricEigenvalues( mfem::DenseMatrix& matrix )
{
  const int size = matrix.Height();
  constexpr int max_sweeps = 50;
  for ( int sweep = 0; sweep < max_sweeps; ++sweep ) {
    RealT diagonal_scale = 0.;
    RealT maximum_off_diagonal = 0.;
    for ( int p = 0; p < size; ++p ) {
      diagonal_scale = std::max( diagonal_scale, std::abs( matrix( p, p ) ) );
      for ( int q = p + 1; q < size; ++q ) {
        maximum_off_diagonal = std::max( maximum_off_diagonal, std::abs( matrix( p, q ) ) );
      }
    }
    const RealT convergence_tolerance =
        100. * std::numeric_limits<RealT>::epsilon() * std::max( 1., diagonal_scale );
    if ( maximum_off_diagonal <= convergence_tolerance ) {
      break;
    }

    for ( int p = 0; p < size - 1; ++p ) {
      for ( int q = p + 1; q < size; ++q ) {
        const RealT off_diagonal = matrix( p, q );
        if ( std::abs( off_diagonal ) <= convergence_tolerance ) {
          continue;
        }
        const RealT diagonal_difference = matrix( q, q ) - matrix( p, p );
        const RealT tau = diagonal_difference / ( 2. * off_diagonal );
        const RealT tangent = tau >= 0. ? 1. / ( tau + std::sqrt( 1. + tau * tau ) )
                                        : -1. / ( -tau + std::sqrt( 1. + tau * tau ) );
        const RealT cosine = 1. / std::sqrt( 1. + tangent * tangent );
        const RealT sine = tangent * cosine;
        const RealT diagonal_p = matrix( p, p );
        const RealT diagonal_q = matrix( q, q );
        for ( int k = 0; k < size; ++k ) {
          if ( k == p || k == q ) {
            continue;
          }
          const RealT value_p = matrix( k, p );
          const RealT value_q = matrix( k, q );
          matrix( k, p ) = cosine * value_p - sine * value_q;
          matrix( p, k ) = matrix( k, p );
          matrix( k, q ) = sine * value_p + cosine * value_q;
          matrix( q, k ) = matrix( k, q );
        }
        matrix( p, p ) = cosine * cosine * diagonal_p - 2. * sine * cosine * off_diagonal +
                         sine * sine * diagonal_q;
        matrix( q, q ) = sine * sine * diagonal_p + 2. * sine * cosine * off_diagonal +
                         cosine * cosine * diagonal_q;
        matrix( p, q ) = 0.;
        matrix( q, p ) = 0.;
      }
    }
  }

  std::vector<RealT> eigenvalues( size );
  for ( int i = 0; i < size; ++i ) {
    eigenvalues[i] = matrix( i, i );
  }
  std::sort( eigenvalues.begin(), eigenvalues.end() );
  return eigenvalues;
}

ProjectionOperatorDiagnostics ComputeProjectionOperatorDiagnostics(
    const std::vector<ProjectionConstraint>& constraints, RealT relaxation, MfemMeshData& mfem_data )
{
  ProjectionOperatorDiagnostics diagnostics;
  const IndexT num_constraints = static_cast<IndexT>( constraints.size() );
  std::vector<IndexT> elements1( constraints.size() );
  std::vector<IndexT> elements2( constraints.size() );
  std::vector<RealT> normals( constraints.size() * 2 );
  std::vector<RealT> phi1( constraints.size() * parent_q2_num_nodes );
  std::vector<RealT> phi2( constraints.size() * parent_q2_num_nodes );
  for ( IndexT i = 0; i < num_constraints; ++i ) {
    const auto& constraint = constraints[i];
    elements1[i] = constraint.element1;
    elements2[i] = constraint.element2;
    for ( int d = 0; d < 2; ++d ) {
      normals[i * 2 + d] = constraint.normal[d];
    }
    for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
      phi1[i * parent_q2_num_nodes + a] = constraint.phi1[a];
      phi2[i * parent_q2_num_nodes + a] = constraint.phi2[a];
    }
  }

  mfem::DenseMatrix rows;
  if ( !mfem_data.AssembleParentQ2MassScaledConstraintRows( num_constraints, elements1.data(), elements2.data(),
                                                            normals.data(), phi1.data(), phi2.data(), rows ) ) {
    return diagnostics;
  }
  for ( IndexT i = 0; i < num_constraints; ++i ) {
    const RealT inverse_sqrt_diagonal = 1. / std::sqrt( constraints[i].diagonal );
    for ( int d = 0; d < rows.Height(); ++d ) {
      rows( d, i ) *= inverse_sqrt_diagonal;
    }
  }
  for ( int d = 0; d < rows.Height(); ++d ) {
    bool active = false;
    for ( IndexT i = 0; i < num_constraints; ++i ) {
      active = active || rows( d, i ) != 0.;
    }
    diagnostics.velocity_dofs += active ? 1 : 0;
  }

  mfem::DenseMatrix normalized_operator( num_constraints );
  mfem::MultAtB( rows, rows, normalized_operator );
  const std::vector<RealT> eigenvalues = ComputeSymmetricEigenvalues( normalized_operator );
  for ( IndexT i = 0; i < num_constraints; ++i ) {
    diagnostics.maximum_eigenvalue = std::max( diagnostics.maximum_eigenvalue, eigenvalues[i] );
  }
  const RealT rank_tolerance = 100. * std::numeric_limits<RealT>::epsilon() *
                               std::max( rows.Height(), static_cast<int>( num_constraints ) ) *
                               diagnostics.maximum_eigenvalue;
  diagnostics.minimum_eigenvalue = std::numeric_limits<RealT>::infinity();
  for ( IndexT i = 0; i < num_constraints; ++i ) {
    if ( eigenvalues[i] > rank_tolerance ) {
      ++diagnostics.rank;
      diagnostics.minimum_eigenvalue = std::min( diagnostics.minimum_eigenvalue, eigenvalues[i] );
      diagnostics.jacobi_contraction =
          std::max( diagnostics.jacobi_contraction, std::abs( 1. - relaxation * eigenvalues[i] ) );
    }
  }
  if ( diagnostics.rank > 0 ) {
    diagnostics.condition_estimate = diagnostics.maximum_eigenvalue / diagnostics.minimum_eigenvalue;
  } else {
    diagnostics.minimum_eigenvalue = 0.;
  }
  return diagnostics;
}

ProjectionOperatorDiagnostics ComputeProjectionOperatorDiagnostics( const mfem::DenseMatrix& mass_scaled_rows,
                                                                    const mfem::DenseMatrix& projection_operator,
                                                                    RealT relaxation )
{
  ProjectionOperatorDiagnostics diagnostics;
  const int num_constraints = projection_operator.Height();
  for ( int d = 0; d < mass_scaled_rows.Height(); ++d ) {
    bool active = false;
    for ( int i = 0; i < num_constraints; ++i ) {
      active = active || mass_scaled_rows( d, i ) != 0.;
    }
    diagnostics.velocity_dofs += active ? 1 : 0;
  }

  mfem::DenseMatrix normalized_operator( num_constraints );
  for ( int i = 0; i < num_constraints; ++i ) {
    for ( int j = 0; j < num_constraints; ++j ) {
      normalized_operator( i, j ) =
          projection_operator( i, j ) / std::sqrt( projection_operator( i, i ) * projection_operator( j, j ) );
    }
  }
  const std::vector<RealT> eigenvalues = ComputeSymmetricEigenvalues( normalized_operator );
  for ( const RealT eigenvalue : eigenvalues ) {
    diagnostics.maximum_eigenvalue = std::max( diagnostics.maximum_eigenvalue, eigenvalue );
  }
  const RealT rank_tolerance = 100. * std::numeric_limits<RealT>::epsilon() *
                               std::max( mass_scaled_rows.Height(), num_constraints ) *
                               diagnostics.maximum_eigenvalue;
  diagnostics.minimum_eigenvalue = std::numeric_limits<RealT>::infinity();
  for ( const RealT eigenvalue : eigenvalues ) {
    if ( eigenvalue > rank_tolerance ) {
      ++diagnostics.rank;
      diagnostics.minimum_eigenvalue = std::min( diagnostics.minimum_eigenvalue, eigenvalue );
      diagnostics.jacobi_contraction =
          std::max( diagnostics.jacobi_contraction, std::abs( 1. - relaxation * eigenvalue ) );
    }
  }
  if ( diagnostics.rank > 0 ) {
    diagnostics.condition_estimate = diagnostics.maximum_eigenvalue / diagnostics.minimum_eigenvalue;
  } else {
    diagnostics.minimum_eigenvalue = 0.;
  }
  return diagnostics;
}

bool PolishProjectionActiveSet( const mfem::DenseMatrix& projection_operator,
                                std::vector<TraceProjectionConstraint>& constraints, RealT residual_tolerance,
                                int max_iterations, int& iterations )
{
  const int num_constraints = static_cast<int>( constraints.size() );
  std::vector<RealT> free_velocity( constraints.size() );
  RealT multiplier_scale = 0.;
  for ( int i = 0; i < num_constraints; ++i ) {
    free_velocity[i] = constraints[i].trial_velocity - constraints[i].target_velocity;
    multiplier_scale = std::max( multiplier_scale, constraints[i].multiplier );
  }
  const RealT multiplier_tolerance = 1000. * std::numeric_limits<RealT>::epsilon() * std::max( 1., multiplier_scale );
  std::vector<bool> free_set( constraints.size(), false );
  for ( int i = 0; i < num_constraints; ++i ) {
    free_set[i] = constraints[i].multiplier > multiplier_tolerance;
  }

  while ( iterations < max_iterations ) {
    ++iterations;

    std::vector<RealT> residual( constraints.size() );
    for ( int i = 0; i < num_constraints; ++i ) {
      residual[i] = free_velocity[i];
      for ( int j = 0; j < num_constraints; ++j ) {
        residual[i] += projection_operator( i, j ) * constraints[j].multiplier;
      }
    }

    std::vector<int> free_indices;
    free_indices.reserve( constraints.size() );
    for ( int i = 0; i < num_constraints; ++i ) {
      if ( free_set[i] ) {
        free_indices.push_back( i );
      }
    }
    if ( free_indices.empty() ) {
      int most_violated = -1;
      RealT minimum_residual = 0.;
      for ( int i = 0; i < num_constraints; ++i ) {
        if ( residual[i] < minimum_residual ) {
          minimum_residual = residual[i];
          most_violated = i;
        }
      }
      if ( minimum_residual >= -residual_tolerance ) {
        return true;
      }
      free_set[most_violated] = true;
      continue;
    }

    const int free_size = static_cast<int>( free_indices.size() );
    mfem::DenseMatrix free_operator( free_size );
    mfem::Vector free_rhs( free_size );
    for ( int i = 0; i < free_size; ++i ) {
      free_rhs[i] = -free_velocity[free_indices[i]];
      for ( int j = 0; j < free_size; ++j ) {
        free_operator( i, j ) = projection_operator( free_indices[i], free_indices[j] );
      }
    }
    mfem::DenseMatrixInverse free_inverse( free_operator, true );
    mfem::Vector candidate_multipliers;
    free_inverse.Mult( free_rhs, candidate_multipliers );

    RealT minimum_candidate = 0.;
    for ( int i = 0; i < free_size; ++i ) {
      if ( !std::isfinite( candidate_multipliers[i] ) ) {
        return false;
      }
      minimum_candidate = std::min( minimum_candidate, candidate_multipliers[i] );
    }

    if ( minimum_candidate >= -multiplier_tolerance ) {
      for ( int i = 0; i < num_constraints; ++i ) {
        constraints[i].multiplier = 0.;
      }
      for ( int i = 0; i < free_size; ++i ) {
        const int constraint_index = free_indices[i];
        if ( candidate_multipliers[i] > multiplier_tolerance ) {
          constraints[constraint_index].multiplier = candidate_multipliers[i];
        } else {
          free_set[constraint_index] = false;
        }
      }
      int add_index = -1;
      RealT minimum_residual = 0.;
      for ( int i = 0; i < num_constraints; ++i ) {
        if ( free_set[i] ) {
          continue;
        }
        RealT inactive_residual = free_velocity[i];
        for ( int j = 0; j < num_constraints; ++j ) {
          inactive_residual += projection_operator( i, j ) * constraints[j].multiplier;
        }
        if ( inactive_residual < minimum_residual ) {
          minimum_residual = inactive_residual;
          add_index = i;
        }
      }
      if ( minimum_residual < -residual_tolerance ) {
        free_set[add_index] = true;
        continue;
      }
      return true;
    }

    RealT step_length = 1.;
    for ( int i = 0; i < free_size; ++i ) {
      const RealT multiplier = constraints[free_indices[i]].multiplier;
      const RealT direction = candidate_multipliers[i] - multiplier;
      if ( direction < 0. ) {
        step_length = std::min( step_length, -multiplier / direction );
      }
    }
    if ( !std::isfinite( step_length ) || step_length < 0. || step_length > 1. ) {
      return false;
    }
    for ( int i = 0; i < free_size; ++i ) {
      const int constraint_index = free_indices[i];
      const RealT multiplier = constraints[constraint_index].multiplier;
      constraints[constraint_index].multiplier = multiplier + step_length * ( candidate_multipliers[i] - multiplier );
      if ( constraints[constraint_index].multiplier <= multiplier_tolerance ) {
        constraints[constraint_index].multiplier = 0.;
        free_set[constraint_index] = false;
      }
    }
  }
  return true;
}

TraceProjectionSolveResult SolveTraceProjectionSystem( const mfem::DenseMatrix& projection_operator,
                                                        std::vector<TraceProjectionConstraint>& constraints,
                                                        const ImpulseProjectionOptions& projection_options )
{
  TraceProjectionSolveResult result;
  const int num_constraints = static_cast<int>( constraints.size() );
  auto compute_residual = [&]() {
    ProjectionResiduals residuals;
    for ( int i = 0; i < num_constraints; ++i ) {
      RealT projected_gap_rate = constraints[i].trial_velocity - constraints[i].target_velocity;
      for ( int j = 0; j < num_constraints; ++j ) {
        projected_gap_rate += projection_operator( i, j ) * constraints[j].multiplier;
      }
      residuals.complementarity =
          std::max( residuals.complementarity,
                    std::abs( std::min( constraints[i].diagonal * constraints[i].multiplier, projected_gap_rate ) ) );
      residuals.primal = std::max( residuals.primal, -projected_gap_rate );
    }
    residuals.primal = std::max( 0., residuals.primal );
    return residuals;
  };

  const ProjectionResiduals initial_residuals = compute_residual();
  result.initial_residual = initial_residuals.complementarity;
  result.final_residual = result.initial_residual;
  result.final_primal_residual = initial_residuals.primal;
  const RealT convergence_tolerance = projection_options.absolute_tolerance +
                                      projection_options.relative_tolerance * result.initial_residual;
  result.primal_tolerance = projection_options.absolute_tolerance +
                            projection_options.primal_relative_tolerance * initial_residuals.primal;
  result.complementarity_converged = result.final_residual <= convergence_tolerance;
  constexpr int projected_gauss_seidel_warmup_iterations = 3;
  const int warmup_iterations = std::min( projection_options.max_iterations,
                                          projected_gauss_seidel_warmup_iterations );
  for ( int iteration = 1; result.valid && !result.complementarity_converged &&
                           iteration <= warmup_iterations;
        ++iteration ) {
    for ( int i = 0; i < num_constraints; ++i ) {
      RealT projected_gap_rate = constraints[i].trial_velocity - constraints[i].target_velocity;
      for ( int j = 0; j < num_constraints; ++j ) {
        projected_gap_rate += projection_operator( i, j ) * constraints[j].multiplier;
      }
      constraints[i].multiplier =
          std::max( 0., constraints[i].multiplier -
                            projection_options.relaxation_scale * projected_gap_rate / constraints[i].diagonal );
      result.valid = result.valid && std::isfinite( constraints[i].multiplier );
    }
    const ProjectionResiduals residuals = compute_residual();
    result.final_residual = residuals.complementarity;
    result.final_primal_residual = residuals.primal;
    result.valid = result.valid && std::isfinite( result.final_residual ) &&
                   std::isfinite( result.final_primal_residual );
    result.complementarity_converged = result.valid && result.final_residual <= convergence_tolerance;
    result.iterations = iteration;
  }
  if ( result.valid && !result.complementarity_converged &&
       result.iterations < projection_options.max_iterations ) {
    int active_set_iterations = 0;
    result.valid = PolishProjectionActiveSet(
        projection_operator, constraints, std::min( convergence_tolerance, result.primal_tolerance ),
        projection_options.max_iterations - result.iterations, active_set_iterations );
    result.iterations += active_set_iterations;
    const ProjectionResiduals residuals = compute_residual();
    result.final_residual = residuals.complementarity;
    result.final_primal_residual = residuals.primal;
    result.valid = result.valid && std::isfinite( result.final_residual ) &&
                   std::isfinite( result.final_primal_residual );
    result.complementarity_converged = result.valid && result.final_residual <= convergence_tolerance;
  }
  return result;
}
#endif

TRIBOL_HOST_DEVICE inline RealT ComputeRatePenalty( const MeshData::Viewer& m1, const MeshData::Viewer& m2,
                                                    RealT element_penalty, RatePenaltyCalculation rate_calc )
{
  switch ( rate_calc ) {
    case NO_RATE_PENALTY: {
      return 0.;
    }
    case RATE_CONSTANT: {
      return 0.5 * ( m1.getElementData().m_rate_penalty_stiffness + m2.getElementData().m_rate_penalty_stiffness );
    }
    case RATE_PERCENT: {
      return element_penalty * 0.5 *
             ( m1.getElementData().m_rate_percent_stiffness + m2.getElementData().m_rate_percent_stiffness );
    }
    default:
      return 0.;
  }
}

TRIBOL_HOST_DEVICE inline bool Solve3x3( const RealT A[3][3], const RealT b[3], RealT x[3] )
{
  const RealT detA = A[0][0] * ( A[1][1] * A[2][2] - A[1][2] * A[2][1] ) -
                     A[0][1] * ( A[1][0] * A[2][2] - A[1][2] * A[2][0] ) +
                     A[0][2] * ( A[1][0] * A[2][1] - A[1][1] * A[2][0] );
  constexpr RealT det_tol = 1.e-15;
  if ( std::abs( detA ) <= det_tol ) {
    return false;
  }

  const RealT inv_detA = 1. / detA;
  x[0] = inv_detA * ( b[0] * ( A[1][1] * A[2][2] - A[1][2] * A[2][1] ) - A[0][1] * ( b[1] * A[2][2] - A[1][2] * b[2] ) +
                      A[0][2] * ( b[1] * A[2][1] - A[1][1] * b[2] ) );
  x[1] = inv_detA * ( A[0][0] * ( b[1] * A[2][2] - A[1][2] * b[2] ) - b[0] * ( A[1][0] * A[2][2] - A[1][2] * A[2][0] ) +
                      A[0][2] * ( A[1][0] * b[2] - b[1] * A[2][0] ) );
  x[2] = inv_detA * ( A[0][0] * ( A[1][1] * b[2] - b[1] * A[2][1] ) - A[0][1] * ( A[1][0] * b[2] - b[1] * A[2][0] ) +
                      b[0] * ( A[1][0] * A[2][1] - A[1][1] * A[2][0] ) );
  return true;
}

TRIBOL_HOST_DEVICE inline void AccumulateFaceInterpolation( const RealT* face_coords, const int num_nodes,
                                                            const RealT* phi, RealT x_face[3], const int value_dim = 0,
                                                            const RealT* nodal_vals = nullptr, RealT* values = nullptr )
{
  initRealArray( x_face, max_dim, 0. );
  if ( values != nullptr ) {
    initRealArray( values, value_dim, 0. );
  }

  for ( int a = 0; a < num_nodes; ++a ) {
    x_face[0] += face_coords[3 * a] * phi[a];
    x_face[1] += face_coords[3 * a + 1] * phi[a];
    x_face[2] += face_coords[3 * a + 2] * phi[a];

    if ( values != nullptr ) {
      for ( int i = 0; i < value_dim; ++i ) {
        values[i] += nodal_vals[i + a * value_dim] * phi[a];
      }
    }
  }
}

TRIBOL_HOST_DEVICE inline bool EvalLinearFaceAtProjectedPoint( const RealT* face_coords, const int num_nodes,
                                                               const RealT x_query[3], const RealT projection_dir[3],
                                                               RealT x_face[3], RealT* phi, const int value_dim = 0,
                                                               const RealT* nodal_vals = nullptr,
                                                               RealT* values = nullptr )
{
  // CommonPlane quadrature points lie on the overlap polygon. Evaluate the face
  // fields at the corresponding on-face point found by projecting along the
  // common-plane normal rather than by an off-surface closest-point inverse map.
  initRealArray( phi, num_nodes, 0. );

  if ( num_nodes == 3 ) {
    const RealT x0[3] = { face_coords[0], face_coords[1], face_coords[2] };
    const RealT e1[3] = { face_coords[3] - x0[0], face_coords[4] - x0[1], face_coords[5] - x0[2] };
    const RealT e2[3] = { face_coords[6] - x0[0], face_coords[7] - x0[1], face_coords[8] - x0[2] };

    RealT n_face[3];
    crossProd( e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], n_face[0], n_face[1], n_face[2] );

    const RealT denom =
        dotProd( n_face[0], n_face[1], n_face[2], projection_dir[0], projection_dir[1], projection_dir[2] );
    constexpr RealT parallel_tol = 1.e-14;
    if ( std::abs( denom ) <= parallel_tol ) {
      return false;
    }

    const RealT dx0 = x0[0] - x_query[0];
    const RealT dy0 = x0[1] - x_query[1];
    const RealT dz0 = x0[2] - x_query[2];
    const RealT step = dotProd( n_face[0], n_face[1], n_face[2], dx0, dy0, dz0 ) / denom;

    x_face[0] = x_query[0] + step * projection_dir[0];
    x_face[1] = x_query[1] + step * projection_dir[1];
    x_face[2] = x_query[2] + step * projection_dir[2];

    const RealT r[3] = { x_face[0] - x0[0], x_face[1] - x0[1], x_face[2] - x0[2] };
    const RealT gram11 = dotProd( e1[0], e1[1], e1[2], e1[0], e1[1], e1[2] );
    const RealT gram12 = dotProd( e1[0], e1[1], e1[2], e2[0], e2[1], e2[2] );
    const RealT gram22 = dotProd( e2[0], e2[1], e2[2], e2[0], e2[1], e2[2] );
    const RealT rhs1 = dotProd( e1[0], e1[1], e1[2], r[0], r[1], r[2] );
    const RealT rhs2 = dotProd( e2[0], e2[1], e2[2], r[0], r[1], r[2] );

    const RealT gram_det = gram11 * gram22 - gram12 * gram12;
    constexpr RealT gram_tol = 1.e-15;
    if ( std::abs( gram_det ) <= gram_tol ) {
      return false;
    }

    const RealT xi = ( gram22 * rhs1 - gram12 * rhs2 ) / gram_det;
    const RealT eta = ( gram11 * rhs2 - gram12 * rhs1 ) / gram_det;
    phi[0] = 1. - xi - eta;
    phi[1] = xi;
    phi[2] = eta;
  } else if ( num_nodes == 4 ) {
    constexpr int max_iter = 25;
    constexpr RealT step_tol = 1.e-12;
    constexpr RealT residual_tol = 1.e-12;
    constexpr RealT xi_tol = 1.e-8;

    RealT xi = 0.;
    RealT eta = 0.;
    RealT phi0[max_nodes_per_face] = { 0.25, 0.25, 0.25, 0.25 };
    RealT x_center[3];
    AccumulateFaceInterpolation( face_coords, num_nodes, phi0, x_center );
    RealT s = ( x_center[0] - x_query[0] ) * projection_dir[0] + ( x_center[1] - x_query[1] ) * projection_dir[1] +
              ( x_center[2] - x_query[2] ) * projection_dir[2];
    bool converged = false;

    for ( int iter = 0; iter < max_iter; ++iter ) {
      const RealT xi_node[4] = { 1., -1., -1., 1. };
      const RealT eta_node[4] = { 1., 1., -1., -1. };
      RealT dxdxi[3] = { 0., 0., 0. };
      RealT dxdeta[3] = { 0., 0., 0. };
      initRealArray( phi, num_nodes, 0. );
      initRealArray( x_face, max_dim, 0. );

      for ( int a = 0; a < num_nodes; ++a ) {
        phi[a] = 0.25 * ( 1. + xi_node[a] * xi ) * ( 1. + eta_node[a] * eta );
        const RealT dphi_dxi = 0.25 * xi_node[a] * ( 1. + eta_node[a] * eta );
        const RealT dphi_deta = 0.25 * eta_node[a] * ( 1. + xi_node[a] * xi );

        const RealT xa = face_coords[3 * a];
        const RealT ya = face_coords[3 * a + 1];
        const RealT za = face_coords[3 * a + 2];

        x_face[0] += xa * phi[a];
        x_face[1] += ya * phi[a];
        x_face[2] += za * phi[a];

        dxdxi[0] += xa * dphi_dxi;
        dxdxi[1] += ya * dphi_dxi;
        dxdxi[2] += za * dphi_dxi;

        dxdeta[0] += xa * dphi_deta;
        dxdeta[1] += ya * dphi_deta;
        dxdeta[2] += za * dphi_deta;
      }

      RealT residual[3] = { x_face[0] - x_query[0] - s * projection_dir[0],
                            x_face[1] - x_query[1] - s * projection_dir[1],
                            x_face[2] - x_query[2] - s * projection_dir[2] };
      const RealT residual_norm = magnitude( residual[0], residual[1], residual[2] );
      if ( residual_norm <= residual_tol ) {
        converged = true;
        break;
      }

      RealT J[3][3] = { { dxdxi[0], dxdeta[0], -projection_dir[0] },
                        { dxdxi[1], dxdeta[1], -projection_dir[1] },
                        { dxdxi[2], dxdeta[2], -projection_dir[2] } };
      RealT rhs[3] = { -residual[0], -residual[1], -residual[2] };
      RealT delta[3];
      if ( !Solve3x3( J, rhs, delta ) ) {
        return false;
      }

      xi += delta[0];
      eta += delta[1];
      s += delta[2];

      const RealT step_norm = magnitude( delta[0], delta[1], delta[2] );
      if ( step_norm <= step_tol ) {
        converged = true;
        break;
      }
    }

    if ( !converged || xi < -1. - xi_tol || xi > 1. + xi_tol || eta < -1. - xi_tol || eta > 1. + xi_tol ) {
      return false;
    }

    initRealArray( phi, num_nodes, 0. );
    phi[0] = 0.25 * ( 1. + xi ) * ( 1. + eta );
    phi[1] = 0.25 * ( 1. - xi ) * ( 1. + eta );
    phi[2] = 0.25 * ( 1. - xi ) * ( 1. - eta );
    phi[3] = 0.25 * ( 1. + xi ) * ( 1. - eta );
    AccumulateFaceInterpolation( face_coords, num_nodes, phi, x_face );

    const RealT line_residual = magnitude( x_face[0] - x_query[0], x_face[1] - x_query[1], x_face[2] - x_query[2] );
    const RealT normal_step = ( x_face[0] - x_query[0] ) * projection_dir[0] +
                              ( x_face[1] - x_query[1] ) * projection_dir[1] +
                              ( x_face[2] - x_query[2] ) * projection_dir[2];
    const RealT projection_residual = magnitude( x_face[0] - x_query[0] - normal_step * projection_dir[0],
                                                 x_face[1] - x_query[1] - normal_step * projection_dir[1],
                                                 x_face[2] - x_query[2] - normal_step * projection_dir[2] );
    if ( line_residual > 0. && projection_residual > residual_tol * line_residual ) {
      return false;
    }
  } else {
    return false;
  }

  if ( values != nullptr ) {
    initRealArray( values, value_dim, 0. );
    for ( int a = 0; a < num_nodes; ++a ) {
      for ( int i = 0; i < value_dim; ++i ) {
        values[i] += nodal_vals[i + a * value_dim] * phi[a];
      }
    }
  }

  return true;
}

TRIBOL_HOST_DEVICE inline bool EvalLinearEdgeAtProjectedPoint( const RealT* edge_coords, const RealT x_query[2],
                                                               const RealT projection_dir[2], RealT x_edge[3],
                                                               RealT* phi, const int value_dim = 0,
                                                               const RealT* nodal_vals = nullptr,
                                                               RealT* values = nullptr,
                                                               RealT* edge_parameter = nullptr )
{
  const RealT ax = edge_coords[0];
  const RealT ay = edge_coords[1];
  const RealT bx = edge_coords[2];
  const RealT by = edge_coords[3];
  const RealT ex = bx - ax;
  const RealT ey = by - ay;

  const RealT det = projection_dir[0] * ey - ex * projection_dir[1];
  constexpr RealT det_tol = 1.e-14;
  if ( std::abs( det ) <= det_tol ) {
    return false;
  }

  const RealT rhs_x = x_query[0] - ax;
  const RealT rhs_y = x_query[1] - ay;
  RealT edge_param = ( projection_dir[0] * rhs_y - projection_dir[1] * rhs_x ) / det;

  constexpr RealT edge_tol = 1.e-8;
  if ( edge_param < -edge_tol || edge_param > 1. + edge_tol ) {
    return false;
  }
  edge_param = std::max( 0., std::min( 1., edge_param ) );
  if ( edge_parameter != nullptr ) {
    *edge_parameter = edge_param;
  }

  phi[0] = 1. - edge_param;
  phi[1] = edge_param;

  x_edge[0] = ax + edge_param * ex;
  x_edge[1] = ay + edge_param * ey;
  x_edge[2] = 0.;

  if ( values != nullptr ) {
    initRealArray( values, value_dim, 0. );
    for ( int a = 0; a < 2; ++a ) {
      for ( int i = 0; i < value_dim; ++i ) {
        values[i] += nodal_vals[i + a * value_dim] * phi[a];
      }
    }
  }

  return true;
}

TRIBOL_HOST_DEVICE inline void EvalParentQ2Basis( RealT xi, RealT* phi, RealT* dphi )
{
  phi[0] = 2. * ( xi - 0.5 ) * ( xi - 1. );
  phi[1] = 2. * xi * ( xi - 0.5 );
  phi[2] = 4. * xi * ( 1. - xi );
  dphi[0] = 4. * xi - 3.;
  dphi[1] = 4. * xi - 1.;
  dphi[2] = 4. - 8. * xi;
}

TRIBOL_HOST_DEVICE inline void EvalParentLORBasis( RealT xi, RealT* phi )
{
  if ( xi <= 0.5 ) {
    phi[0] = 1. - 2. * xi;
    phi[1] = 0.;
    phi[2] = 2. * xi;
  } else {
    phi[0] = 0.;
    phi[1] = 2. * xi - 1.;
    phi[2] = 2. * ( 1. - xi );
  }
}

TRIBOL_HOST_DEVICE inline RealT ComputeProjectionPenaltyStiffnessPerArea(
    const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, IndexT element1, IndexT element2,
    const PenaltyEnforcementOptions& penalty_options )
{
  const RealT scale1 = mesh1.getElementData().m_penalty_scale;
  const RealT scale2 = mesh2.getElementData().m_penalty_scale;
  switch ( penalty_options.kinematic_calculation ) {
    case KINEMATIC_CONSTANT: {
      return ComputePenaltyStiffnessPerArea( scale1 * mesh1.getElementData().m_penalty_stiffness,
                                             scale2 * mesh2.getElementData().m_penalty_stiffness );
    }
    case KINEMATIC_ELEMENT: {
      const RealT thickness1 = mesh1.getElementData().m_thickness[element1] + penalty_options.tiny_length;
      const RealT thickness2 = mesh2.getElementData().m_thickness[element2] + penalty_options.tiny_length;
      if ( thickness1 <= 0. || thickness2 <= 0. ) {
        return 0.;
      }
      return ComputePenaltyStiffnessPerArea( scale1 * mesh1.getElementData().m_mat_mod[element1] / thickness1,
                                             scale2 * mesh2.getElementData().m_mat_mod[element2] / thickness2 );
    }
    default:
      return 0.;
  }
}

TRIBOL_HOST_DEVICE inline bool EvalParentQ2Edge( const MeshData::Viewer& mesh, IndexT element_id, RealT child_parameter,
                                                 bool use_velocity, RealT* phi, RealT* position, RealT* velocity,
                                                 RealT* outward_normal )
{
  const RealT xi0 = mesh.getParentReferenceCoordinate( element_id, 0 );
  const RealT xi1 = mesh.getParentReferenceCoordinate( element_id, 1 );
  const RealT xi = ( 1. - child_parameter ) * xi0 + child_parameter * xi1;
  RealT dphi[parent_q2_num_nodes];
  EvalParentQ2Basis( xi, phi, dphi );

  position[0] = 0.;
  position[1] = 0.;
  velocity[0] = 0.;
  velocity[1] = 0.;
  RealT tangent[2] = { 0., 0. };
  for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
    for ( int d = 0; d < 2; ++d ) {
      const RealT coordinate = mesh.getParentPosition( element_id, a, d );
      position[d] += phi[a] * coordinate;
      tangent[d] += dphi[a] * coordinate;
      if ( use_velocity ) {
        velocity[d] += phi[a] * mesh.getParentVelocity( element_id, a, d );
      }
    }
  }

  const RealT tangent_norm = magnitude( tangent[0], tangent[1] );
  constexpr RealT tangent_tol = 1.e-14;
  if ( tangent_norm <= tangent_tol ) {
    return false;
  }
  outward_normal[0] = tangent[1] / tangent_norm;
  outward_normal[1] = -tangent[0] / tangent_norm;
  RealT chord_normal[2];
  mesh.getFaceNormal( element_id, chord_normal );
  if ( outward_normal[0] * chord_normal[0] + outward_normal[1] * chord_normal[1] < 0. ) {
    outward_normal[0] = -outward_normal[0];
    outward_normal[1] = -outward_normal[1];
  }
  return true;
}

TRIBOL_HOST_DEVICE inline bool EvalParentQ2Pair( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                                 IndexT index1, IndexT index2, RealT child_parameter1,
                                                 RealT child_parameter2, bool use_velocity,
                                                 const RealT* fallback_normal, RealT* phi1, RealT* phi2,
                                                 RealT* position1, RealT* position2, RealT* velocity1, RealT* velocity2,
                                                 RealT* common_normal )
{
  RealT normal1[2];
  RealT normal2[2];
  const bool valid1 =
      EvalParentQ2Edge( mesh1, index1, child_parameter1, use_velocity, phi1, position1, velocity1, normal1 );
  const bool valid2 =
      EvalParentQ2Edge( mesh2, index2, child_parameter2, use_velocity, phi2, position2, velocity2, normal2 );
  const RealT average_x = valid1 && valid2 ? normal2[0] - normal1[0] : 0.;
  const RealT average_y = valid1 && valid2 ? normal2[1] - normal1[1] : 0.;
  const RealT average_norm = magnitude( average_x, average_y );
  constexpr RealT normal_tol = 1.e-14;
  const bool used_parent_normal = average_norm > normal_tol;
  if ( used_parent_normal ) {
    common_normal[0] = average_x / average_norm;
    common_normal[1] = average_y / average_norm;
    if ( common_normal[0] * fallback_normal[0] + common_normal[1] * fallback_normal[1] < 0. ) {
      common_normal[0] = -common_normal[0];
      common_normal[1] = -common_normal[1];
    }
  } else {
    common_normal[0] = fallback_normal[0];
    common_normal[1] = fallback_normal[1];
  }
  return used_parent_normal;
}

TRIBOL_HOST_DEVICE inline void AccumulateContactForce( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                                       const IndexT index1, const IndexT index2, const int dim,
                                                       const int num_nodes_per_face, const RealT force_x,
                                                       const RealT force_y, const RealT force_z, const RealT* phi1,
                                                       const RealT* phi2 )
{
  for ( IndexT a = 0; a < num_nodes_per_face; ++a ) {
    IndexT node0 = mesh1.getGlobalNodeId( index1, a );
    IndexT node1 = mesh2.getGlobalNodeId( index2, a );

    const RealT nodal_force_x1 = force_x * phi1[a];
    const RealT nodal_force_y1 = force_y * phi1[a];
    const RealT nodal_force_z1 = force_z * phi1[a];

    const RealT nodal_force_x2 = force_x * phi2[a];
    const RealT nodal_force_y2 = force_y * phi2[a];
    const RealT nodal_force_z2 = force_z * phi2[a];

    // accumulate contributions in host code's registered nodal force arrays
    tribol::atomicAdd( &mesh1.getResponse()[0][node0], -nodal_force_x1 );
    tribol::atomicAdd( &mesh2.getResponse()[0][node1], nodal_force_x2 );

    tribol::atomicAdd( &mesh1.getResponse()[1][node0], -nodal_force_y1 );
    tribol::atomicAdd( &mesh2.getResponse()[1][node1], nodal_force_y2 );

    // there is no z component for 2D
    if ( dim == 3 ) {
      tribol::atomicAdd( &mesh1.getResponse()[2][node0], -nodal_force_z1 );
      tribol::atomicAdd( &mesh2.getResponse()[2][node1], nodal_force_z2 );
    }
  }  // end switch on rate_calc
}

TRIBOL_HOST_DEVICE inline void AccumulateParentContactForce( const MeshData::Viewer& mesh1,
                                                             const MeshData::Viewer& mesh2, IndexT index1,
                                                             IndexT index2, int dim, const RealT* force,
                                                             const RealT* phi1, const RealT* phi2 )
{
  for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
    for ( int d = 0; d < dim; ++d ) {
      tribol::atomicAdd( &mesh1.getParentResponse( index1, a, d ), -force[d] * phi1[a] );
      tribol::atomicAdd( &mesh2.getParentResponse( index2, a, d ), force[d] * phi2[a] );
    }
  }
}

TRIBOL_HOST_DEVICE inline RealT ComputeInverseEffectiveMass( const MeshData::Viewer& mesh1,
                                                             const MeshData::Viewer& mesh2, IndexT index1,
                                                             IndexT index2, int dim, int num_nodes_per_face,
                                                             const RealT* normal,
                                                             const RealT* phi1, const RealT* phi2 )
{
  RealT inverse_effective_mass = 0.;
  for ( int a = 0; a < num_nodes_per_face; ++a ) {
    const IndexT node1 = mesh1.getGlobalNodeId( index1, a );
    const IndexT node2 = mesh2.getGlobalNodeId( index2, a );
    for ( int d = 0; d < dim; ++d ) {
      const RealT constraint1 = phi1[a] * normal[d];
      const RealT constraint2 = phi2[a] * normal[d];
      inverse_effective_mass += constraint1 * constraint1 * mesh1.getInverseMass( node1, d );
      inverse_effective_mass += constraint2 * constraint2 * mesh2.getInverseMass( node2, d );
    }
  }
  return inverse_effective_mass;
}

TRIBOL_HOST_DEVICE inline RealT ComputeParentInverseEffectiveMass( const MeshData::Viewer& mesh1,
                                                                   const MeshData::Viewer& mesh2, IndexT index1,
                                                                   IndexT index2, int dim, const RealT* normal,
                                                                   const RealT* phi1, const RealT* phi2 )
{
  RealT inverse_effective_mass = 0.;
  for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
    for ( int d = 0; d < dim; ++d ) {
      const RealT constraint1 = phi1[a] * normal[d];
      const RealT constraint2 = phi2[a] * normal[d];
      inverse_effective_mass += constraint1 * constraint1 * mesh1.getParentInverseMass( index1, a, d );
      inverse_effective_mass += constraint2 * constraint2 * mesh2.getParentInverseMass( index2, a, d );
    }
  }
  return inverse_effective_mass;
}

TRIBOL_HOST_DEVICE inline RealT ComputePredictorTargetVelocity( RealT gap, RealT dt )
{
  return dt > 0. && gap > 0. ? -gap / dt : 0.;
}

TRIBOL_HOST_DEVICE inline RealT ComputePredictorPressure( const MeshData::Viewer& mesh1,
                                                          const MeshData::Viewer& mesh2, IndexT index1,
                                                          IndexT index2, int dim, int num_nodes_per_face,
                                                          const RealT* normal, const RealT* phi1, const RealT* phi2,
                                                          RealT gap, RealT velocity_gap,
                                                          RealT quadrature_measure, RealT dt,
                                                          RealT relaxation )
{
  if ( dt <= 0. || quadrature_measure <= 0. ) {
    return 0.;
  }
  const RealT inverse_effective_mass =
      ComputeInverseEffectiveMass( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, normal, phi1, phi2 );
  if ( inverse_effective_mass <= 0. ) {
    return 0.;
  }

  const RealT target_velocity = ComputePredictorTargetVelocity( gap, dt );
  if ( velocity_gap >= target_velocity ) {
    return 0.;
  }

  const RealT impulse = relaxation * ( target_velocity - velocity_gap ) / inverse_effective_mass;
  return -impulse / ( quadrature_measure * dt );
}

TRIBOL_HOST_DEVICE inline RealT ComputeParentPredictorPressure( const MeshData::Viewer& mesh1,
                                                                const MeshData::Viewer& mesh2, IndexT index1,
                                                                IndexT index2, int dim, const RealT* normal,
                                                                const RealT* phi1, const RealT* phi2, RealT gap,
                                                                RealT velocity_gap, RealT quadrature_measure, RealT dt,
                                                                RealT relaxation )
{
  if ( dt <= 0. || quadrature_measure <= 0. ) {
    return 0.;
  }
  const RealT inverse_effective_mass =
      ComputeParentInverseEffectiveMass( mesh1, mesh2, index1, index2, dim, normal, phi1, phi2 );
  if ( inverse_effective_mass <= 0. ) {
    return 0.;
  }
  const RealT target_velocity = ComputePredictorTargetVelocity( gap, dt );
  if ( velocity_gap >= target_velocity ) {
    return 0.;
  }
  const RealT impulse = relaxation * ( target_velocity - velocity_gap ) / inverse_effective_mass;
  return -impulse / ( quadrature_measure * dt );
}

TRIBOL_HOST_DEVICE inline void AccumulateConstraintRowBounds(
    const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, IndexT index1, IndexT index2, int dim,
    int num_nodes_per_face, const RealT* normal, const RealT* phi1, const RealT* phi2, RealT quadrature_measure,
    RealT penalty_stiffness, bool include_predictor, ArrayViewT<RealT> predictor_rows1,
    ArrayViewT<RealT> predictor_rows2, ArrayViewT<RealT> stiffness_rows1, ArrayViewT<RealT> stiffness_rows2 )
{
  const RealT inverse_effective_mass =
      ComputeInverseEffectiveMass( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, normal, phi1, phi2 );
  if ( inverse_effective_mass <= 0. ) {
    return;
  }

  const RealT inv_sqrt_effective = 1. / sqrt( inverse_effective_mass );
  RealT predictor_l1 = 0.;
  RealT stiffness_l1 = 0.;
  for ( int a = 0; a < num_nodes_per_face; ++a ) {
    const IndexT node1 = mesh1.getGlobalNodeId( index1, a );
    const IndexT node2 = mesh2.getGlobalNodeId( index2, a );
    for ( int d = 0; d < dim; ++d ) {
      const RealT sqrt_inv_mass1 = sqrt( mesh1.getInverseMass( node1, d ) );
      const RealT sqrt_inv_mass2 = sqrt( mesh2.getInverseMass( node2, d ) );
      predictor_l1 += std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 * inv_sqrt_effective );
      predictor_l1 += std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 * inv_sqrt_effective );
      stiffness_l1 += std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 );
      stiffness_l1 += std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 );
    }
  }

  const RealT stiffness_scale = penalty_stiffness * quadrature_measure;
  for ( int a = 0; a < num_nodes_per_face; ++a ) {
    const IndexT node1 = mesh1.getGlobalNodeId( index1, a );
    const IndexT node2 = mesh2.getGlobalNodeId( index2, a );
    for ( int d = 0; d < dim; ++d ) {
      const RealT sqrt_inv_mass1 = sqrt( mesh1.getInverseMass( node1, d ) );
      const RealT sqrt_inv_mass2 = sqrt( mesh2.getInverseMass( node2, d ) );
      const RealT predictor_h1 = std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 * inv_sqrt_effective );
      const RealT predictor_h2 = std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 * inv_sqrt_effective );
      const RealT stiffness_h1 = std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 );
      const RealT stiffness_h2 = std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 );
      if ( include_predictor ) {
        tribol::atomicAdd( &predictor_rows1[node1 * dim + d], predictor_h1 * predictor_l1 );
        tribol::atomicAdd( &predictor_rows2[node2 * dim + d], predictor_h2 * predictor_l1 );
      }
      tribol::atomicAdd( &stiffness_rows1[node1 * dim + d], stiffness_scale * stiffness_h1 * stiffness_l1 );
      tribol::atomicAdd( &stiffness_rows2[node2 * dim + d], stiffness_scale * stiffness_h2 * stiffness_l1 );
    }
  }
}

TRIBOL_HOST_DEVICE inline void AccumulateParentConstraintRowBounds(
    const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2, IndexT index1, IndexT index2, int dim,
    const RealT* normal, const RealT* phi1, const RealT* phi2, RealT quadrature_measure, RealT penalty_stiffness,
    bool include_predictor, ArrayViewT<RealT> predictor_rows1, ArrayViewT<RealT> predictor_rows2,
    ArrayViewT<RealT> stiffness_rows1, ArrayViewT<RealT> stiffness_rows2 )
{
  const RealT inverse_effective_mass =
      ComputeParentInverseEffectiveMass( mesh1, mesh2, index1, index2, dim, normal, phi1, phi2 );
  if ( inverse_effective_mass <= 0. ) {
    return;
  }

  const RealT inv_sqrt_effective = 1. / sqrt( inverse_effective_mass );
  RealT predictor_l1 = 0.;
  RealT stiffness_l1 = 0.;
  for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
    for ( int d = 0; d < dim; ++d ) {
      const RealT sqrt_inv_mass1 = sqrt( mesh1.getParentInverseMass( index1, a, d ) );
      const RealT sqrt_inv_mass2 = sqrt( mesh2.getParentInverseMass( index2, a, d ) );
      predictor_l1 += std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 * inv_sqrt_effective );
      predictor_l1 += std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 * inv_sqrt_effective );
      stiffness_l1 += std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 );
      stiffness_l1 += std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 );
    }
  }

  const RealT stiffness_scale = penalty_stiffness * quadrature_measure;
  for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
    for ( int d = 0; d < dim; ++d ) {
      const RealT sqrt_inv_mass1 = sqrt( mesh1.getParentInverseMass( index1, a, d ) );
      const RealT sqrt_inv_mass2 = sqrt( mesh2.getParentInverseMass( index2, a, d ) );
      const RealT predictor_h1 = std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 * inv_sqrt_effective );
      const RealT predictor_h2 = std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 * inv_sqrt_effective );
      const RealT stiffness_h1 = std::abs( phi1[a] * normal[d] * sqrt_inv_mass1 );
      const RealT stiffness_h2 = std::abs( phi2[a] * normal[d] * sqrt_inv_mass2 );
      const IndexT row1 = ( index1 * parent_q2_num_nodes + a ) * dim + d;
      const IndexT row2 = ( index2 * parent_q2_num_nodes + a ) * dim + d;
      if ( include_predictor ) {
        tribol::atomicAdd( &predictor_rows1[row1], predictor_h1 * predictor_l1 );
        tribol::atomicAdd( &predictor_rows2[row2], predictor_h2 * predictor_l1 );
      }
      tribol::atomicAdd( &stiffness_rows1[row1], stiffness_scale * stiffness_h1 * stiffness_l1 );
      tribol::atomicAdd( &stiffness_rows2[row2], stiffness_scale * stiffness_h2 * stiffness_l1 );
    }
  }
}

TRIBOL_HOST_DEVICE inline void AccumulateForceDiagnostics(
    bool predictor_active, RealT quadrature_measure, RealT penalty_pressure, RealT predictor_pressure,
    RealT applied_pressure, RealT gap, RealT velocity_gap, ArrayViewT<int> diagnostic_counts,
    ArrayViewT<RealT> integrated_forces, ArrayViewT<RealT> contact_sums, ArrayViewT<RealT> contact_maxima )
{
  if ( predictor_active ) {
    tribol::atomicInc( &diagnostic_counts[0] );
    if ( predictor_pressure < penalty_pressure ) {
      tribol::atomicInc( &diagnostic_counts[1] );
    }
  }
  tribol::atomicInc( &diagnostic_counts[2] );

  const RealT penalty_force = penalty_pressure < 0. ? -quadrature_measure * penalty_pressure : 0.;
  const RealT predictor_force = predictor_pressure < 0. ? -quadrature_measure * predictor_pressure : 0.;
  const RealT applied_force = applied_pressure < 0. ? -quadrature_measure * applied_pressure : 0.;
  const RealT gap_violation = gap < 0. ? -gap : 0.;
  const RealT closing_gap_rate = velocity_gap < 0. ? -velocity_gap : 0.;
  tribol::atomicAdd( &integrated_forces[0], penalty_force );
  tribol::atomicAdd( &integrated_forces[1], predictor_force );
  tribol::atomicAdd( &integrated_forces[2], applied_force );
  tribol::atomicAdd( &contact_sums[0], gap_violation );
  tribol::atomicAdd( &contact_sums[1], closing_gap_rate );
  tribol::atomicMax( &contact_maxima[0], applied_force );
  tribol::atomicMax( &contact_maxima[1], gap_violation );
  tribol::atomicMax( &contact_maxima[2], closing_gap_rate );
}

}  // namespace

TRIBOL_HOST_DEVICE RealT ComputeGapRatePressure( CommonPlanePair& plane, const MeshData::Viewer& m1,
                                                 const MeshData::Viewer& m2, RealT element_penalty,
                                                 RatePenaltyCalculation rate_calc )
{
  auto fId1 = plane.getCpElementId1();
  auto fId2 = plane.getCpElementId2();

  const auto dim = plane.m_dim;
  const RealT rate_penalty = ComputeRatePenalty( m1, m2, element_penalty, rate_calc );
  if ( rate_penalty == 0. ) {
    return 0.;
  }

  // compute the velocity gap and pressure contribution
  StackArrayT<RealT, max_dim * max_nodes_per_face> x1;
  StackArrayT<RealT, max_dim * max_nodes_per_face> v1;
  auto numNodesPerFace1 = m1.numberOfNodesPerElement();
  plane.getFace1Coords( x1, numNodesPerFace1 );  // get avg face coords off the contact plane
  m1.getFaceVelocities( fId1, v1 );

  StackArrayT<RealT, max_dim * max_nodes_per_face> x2;
  StackArrayT<RealT, max_dim * max_nodes_per_face> v2;
  auto numNodesPerFace2 = m2.numberOfNodesPerElement();
  plane.getFace2Coords( x2, numNodesPerFace2 );  // get avg face coords off the contact plane
  m2.getFaceVelocities( fId2, v2 );

  //////////////////////////////////////////////////////////
  // compute velocity Galerkin approximation at projected //
  // overlap centroid                                     //
  //////////////////////////////////////////////////////////
  RealT vel_f1[max_dim];
  RealT vel_f2[max_dim];
  initRealArray( vel_f1, dim, 0. );
  initRealArray( vel_f2, dim, 0. );

  // interpolate nodal velocity at overlap centroid as projected
  // onto face 1
  RealT cXf1 = plane.m_cXf1;
  RealT cYf1 = plane.m_cYf1;
  RealT cZf1 = ( dim == 3 ) ? plane.m_cZf1 : 0.;
  GalerkinEvalOnPhysicalFace( x1, cXf1, cYf1, cZf1, numNodesPerFace1, dim, v1, vel_f1 );

  // interpolate nodal velocity at overlap centroid as projected
  // onto face 2
  RealT cXf2 = plane.m_cXf2;
  RealT cYf2 = plane.m_cYf2;
  RealT cZf2 = ( dim == 3 ) ? plane.m_cZf2 : 0.;
  GalerkinEvalOnPhysicalFace( x2, cXf2, cYf2, cZf2, numNodesPerFace2, dim, v2, vel_f2 );

  // compute velocity gap vector
  RealT velGap[max_dim];
  velGap[0] = vel_f1[0] - vel_f2[0];
  velGap[1] = vel_f1[1] - vel_f2[1];
  if ( dim == 3 ) {
    velGap[2] = vel_f1[2] - vel_f2[2];
  }

  // compute velocity gap scalar
  plane.m_velGap = 0.;
  plane.m_velGap += velGap[0] * plane.m_nX;
  plane.m_velGap += velGap[1] * plane.m_nY;
  if ( dim == 3 ) {
    plane.m_velGap += velGap[2] * plane.m_nZ;
  }

  // check the gap rate sense.
  // (v1-v2) * \nu < 0 : velocities lead to more interpenetration;
  // note, \nu is in direction of face_2 outward unit normal
  // TODO consider a velocity gap tolerance. Checking this against
  // 0. actually smoothed out contact behavior in contact problem 1
  // for certain percent rate penalties.
  if ( plane.m_velGap <= 0. )  // TODO do we want = or just <?
  {
    plane.m_ratePressure = plane.m_velGap * rate_penalty;
    return plane.m_ratePressure;
  }  // end if-check on velocity gap

  return 0.;

}  // end ComputeGapRatePressure()

//------------------------------------------------------------------------------
template <>
int ApplyNormal<COMMON_PLANE, PENALTY>( CouplingScheme* cs )
{
  ///////////////////////////////
  // loop over interface pairs //
  ///////////////////////////////
  ArrayT<int> err_data( { 0 }, cs->getAllocatorId() );
  ArrayViewT<int> err = err_data;
  ArrayT<bool> neg_thickness_data( { false }, cs->getAllocatorId() );
  ArrayViewT<bool> neg_thickness = neg_thickness_data;
  auto cs_view = cs->getView();
  const auto num_pairs = cs->getNumActivePairs();
  const auto& penalty_options = cs->getEnforcementOptions().penalty_options;
  const bool dissipative = penalty_options.constraint_type == KINEMATIC_AND_DISSIPATIVE;
  const bool multipoint = penalty_options.common_plane_rule == MULTI_POINT;

  if ( dissipative ) {
    SLIC_ERROR_ROOT_IF( !multipoint,
                        "Dissipative common-plane enforcement requires MULTI_POINT integration." );
    SLIC_ERROR_ROOT_IF( !cs->getMesh1().hasInverseMass() || !cs->getMesh2().hasInverseMass(),
                        "Dissipative common-plane enforcement requires registered inverse nodal masses." );
  }

  RealT predictor_relaxation = 1.;
  if ( cs->getMesh1().hasInverseMass() && cs->getMesh2().hasInverseMass() ) {
    const int dim = cs->spatialDimension();
    const bool parent_basis = cs->getMesh1().hasParentElementData();
    const IndexT row_count1 = parent_basis ? cs->getMesh1().numberOfElements() * parent_q2_num_nodes * dim
                                           : cs->getMesh1().numberOfNodes() * dim;
    const IndexT row_count2 = parent_basis ? cs->getMesh2().numberOfElements() * parent_q2_num_nodes * dim
                                           : cs->getMesh2().numberOfNodes() * dim;
    ArrayT<RealT> predictor_rows1( row_count1, std::max<IndexT>( row_count1, 1 ), cs->getAllocatorId() );
    ArrayT<RealT> predictor_rows2( row_count2, std::max<IndexT>( row_count2, 1 ), cs->getAllocatorId() );
    ArrayT<RealT> stiffness_rows1( row_count1, std::max<IndexT>( row_count1, 1 ), cs->getAllocatorId() );
    ArrayT<RealT> stiffness_rows2( row_count2, std::max<IndexT>( row_count2, 1 ), cs->getAllocatorId() );
    predictor_rows1.fill( 0. );
    predictor_rows2.fill( 0. );
    stiffness_rows1.fill( 0. );
    stiffness_rows2.fill( 0. );
    auto predictor_view1 = predictor_rows1.view();
    auto predictor_view2 = predictor_rows2.view();
    auto stiffness_view1 = stiffness_rows1.view();
    auto stiffness_view2 = stiffness_rows2.view();
    const RealT stage_dt = cs->getCurrentTimeStep();

    forAllExec( cs->getExecutionMode(), num_pairs,
                [cs_view, predictor_view1, predictor_view2, stiffness_view1, stiffness_view2, dissipative,
                 stage_dt] TRIBOL_HOST_DEVICE( IndexT i ) {
                  auto& plane = cs_view.getCompGeomView().getCommonPlane( i );
                  const auto& mesh1 = cs_view.getMesh1View();
                  const auto& mesh2 = cs_view.getMesh2View();
                  const IndexT index1 = plane.getCpElementId1();
                  const IndexT index2 = plane.getCpElementId2();
                  const int local_dim = cs_view.spatialDimension();
                  const int num_nodes_per_face = mesh1.numberOfNodesPerElement();
                  const auto& options = cs_view.getEnforcementOptions().penalty_options;

                  RealT penalty_stiffness = 0.;
                  const RealT scale1 = mesh1.getElementData().m_penalty_scale;
                  const RealT scale2 = mesh2.getElementData().m_penalty_scale;
                  if ( options.kinematic_calculation == KINEMATIC_CONSTANT ) {
                    penalty_stiffness = ComputePenaltyStiffnessPerArea(
                        scale1 * mesh1.getElementData().m_penalty_stiffness,
                        scale2 * mesh2.getElementData().m_penalty_stiffness );
                  } else if ( options.kinematic_calculation == KINEMATIC_ELEMENT ) {
                    const RealT thickness1 = mesh1.getElementData().m_thickness[index1] + options.tiny_length;
                    const RealT thickness2 = mesh2.getElementData().m_thickness[index2] + options.tiny_length;
                    if ( thickness1 <= 0. || thickness2 <= 0. ) {
                      return;
                    }
                    penalty_stiffness = ComputePenaltyStiffnessPerArea(
                        scale1 * mesh1.getElementData().m_mat_mod[index1] / thickness1,
                        scale2 * mesh2.getElementData().m_mat_mod[index2] / thickness2 );
                  }

                  StackArrayT<RealT, max_dim * max_nodes_per_face> face1;
                  StackArrayT<RealT, max_dim * max_nodes_per_face> face2;
                  StackArrayT<RealT, max_dim * max_nodes_per_face> velocity1;
                  StackArrayT<RealT, max_dim * max_nodes_per_face> velocity2;
                  mesh1.getFaceCoords( index1, face1 );
                  mesh2.getFaceCoords( index2, face2 );
                  if ( dissipative ) {
                    mesh1.getFaceVelocities( index1, velocity1 );
                    mesh2.getFaceVelocities( index2, velocity2 );
                  }

                  RealT normal[max_dim] = { plane.m_nX, plane.m_nY, plane.m_nZ };
                  RealT overlap_vertices[max_dim * max_nodes_per_overlap] = { 0. };
                  plane.getOverlapVertices( overlap_vertices );
                  const RealT gap_tolerance = cs_view.getGapTol( index1, index2 );

                  if ( options.common_plane_rule != MULTI_POINT ) {
                    if ( mesh1.hasParentElementData() ) {
                      return;
                    }
                    RealT projected_face1[max_dim * max_nodes_per_face] = { 0. };
                    RealT projected_face2[max_dim * max_nodes_per_face] = { 0. };
                    plane.getFace1Coords( projected_face1, num_nodes_per_face );
                    plane.getFace2Coords( projected_face2, num_nodes_per_face );
                    const int num_vertices = local_dim == 3 ? plane.m_numPolyVert : 2;
                    SurfaceContactElem contact_element( local_dim, projected_face1, projected_face2, overlap_vertices,
                                                        num_nodes_per_face, num_vertices, &mesh1, &mesh2, index1, index2 );
                    RealT phi1[max_nodes_per_face] = { 0. };
                    RealT phi2[max_nodes_per_face] = { 0. };
                    EvalWeakFormIntegralCommonPlane( contact_element, options.common_plane_rule,
                                                     options.common_plane_quadrature_order, phi1, phi2 );
                    if ( plane.m_gap <= gap_tolerance ) {
                      AccumulateConstraintRowBounds(
                          mesh1, mesh2, index1, index2, local_dim, num_nodes_per_face, normal, phi1, phi2, plane.m_area,
                          penalty_stiffness, false, predictor_view1, predictor_view2, stiffness_view1, stiffness_view2 );
                    }
                    return;
                  }

                  if ( local_dim == 3 ) {
                    RealT rule_weights[max_symmetric_triangle_qpts] = { 0. };
                    RealT rule_coordinates[2 * max_symmetric_triangle_qpts] = { 0. };
                    const int num_qpts =
                        GetCommonPlaneTriangleRule( options.common_plane_quadrature_order, rule_weights, rule_coordinates );
                    RealT face_coords1[max_dim * max_nodes_per_face] = { 0. };
                    RealT face_coords2[max_dim * max_nodes_per_face] = { 0. };
                    plane.getFace1Coords( face_coords1, num_nodes_per_face );
                    plane.getFace2Coords( face_coords2, num_nodes_per_face );
                    const int num_vertices = plane.m_numPolyVert;
                    SurfaceContactElem contact_element( local_dim, face_coords1, face_coords2, overlap_vertices,
                                                        num_nodes_per_face, num_vertices, &mesh1, &mesh2, index1, index2 );
                    RealT centroid[3];
                    GetCommonPlaneOverlapCentroid( contact_element, centroid );
                    for ( int vertex = 0; vertex < num_vertices; ++vertex ) {
                      const int next = vertex == num_vertices - 1 ? 0 : vertex + 1;
                      RealT triangle_x[3] = { overlap_vertices[3 * vertex], overlap_vertices[3 * next], centroid[0] };
                      RealT triangle_y[3] = { overlap_vertices[3 * vertex + 1], overlap_vertices[3 * next + 1],
                                              centroid[1] };
                      RealT triangle_z[3] = { overlap_vertices[3 * vertex + 2], overlap_vertices[3 * next + 2],
                                              centroid[2] };
                      const RealT area = Area3DTri( triangle_x, triangle_y, triangle_z );
                      if ( area <= 0. ) {
                        continue;
                      }
                      for ( int qp = 0; qp < num_qpts; ++qp ) {
                        const RealT xi = rule_coordinates[2 * qp];
                        const RealT eta = rule_coordinates[2 * qp + 1];
                        const RealT n0 = 1. - xi - eta;
                        RealT x_q[3] = { n0 * triangle_x[0] + xi * triangle_x[1] + eta * triangle_x[2],
                                         n0 * triangle_y[0] + xi * triangle_y[1] + eta * triangle_y[2],
                                         n0 * triangle_z[0] + xi * triangle_z[1] + eta * triangle_z[2] };
                        RealT phi1[max_nodes_per_face] = { 0. };
                        RealT phi2[max_nodes_per_face] = { 0. };
                        RealT x_face1[max_dim], x_face2[max_dim];
                        RealT vel1[max_dim] = { 0. }, vel2[max_dim] = { 0. };
                        if ( !EvalLinearFaceAtProjectedPoint(
                                 face1, num_nodes_per_face, x_q, normal, x_face1, phi1,
                                 dissipative ? local_dim : 0, dissipative ? &velocity1[0] : nullptr,
                                 dissipative ? vel1 : nullptr ) ||
                             !EvalLinearFaceAtProjectedPoint(
                                 face2, num_nodes_per_face, x_q, normal, x_face2, phi2,
                                 dissipative ? local_dim : 0, dissipative ? &velocity2[0] : nullptr,
                                 dissipative ? vel2 : nullptr ) ) {
                          continue;
                        }
                        const RealT gap = ( x_face1[0] - x_face2[0] ) * normal[0] +
                                          ( x_face1[1] - x_face2[1] ) * normal[1] +
                                          ( x_face1[2] - x_face2[2] ) * normal[2];
                        const RealT velocity_gap = ( vel1[0] - vel2[0] ) * normal[0] +
                                                   ( vel1[1] - vel2[1] ) * normal[1] +
                                                   ( vel1[2] - vel2[2] ) * normal[2];
                        const RealT target_velocity = ComputePredictorTargetVelocity( gap, stage_dt );
                        const bool predictor_active =
                            dissipative && stage_dt > 0. && velocity_gap < target_velocity;
                        if ( gap <= gap_tolerance || predictor_active ) {
                          AccumulateConstraintRowBounds(
                              mesh1, mesh2, index1, index2, local_dim, num_nodes_per_face, normal, phi1, phi2,
                              area * rule_weights[qp], penalty_stiffness, predictor_active, predictor_view1,
                              predictor_view2, stiffness_view1, stiffness_view2 );
                        }
                      }
                    }
                  } else {
                    RealT rule_weights[max_segment_gauss_legendre_qpts] = { 0. };
                    RealT rule_coordinates[max_segment_gauss_legendre_qpts] = { 0. };
                    const int num_qpts =
                        GetCommonPlaneSegmentRule( options.common_plane_quadrature_order, rule_weights, rule_coordinates );
                    const RealT x0 = overlap_vertices[0];
                    const RealT y0 = overlap_vertices[1];
                    const RealT x1 = overlap_vertices[2];
                    const RealT y1 = overlap_vertices[3];
                    const RealT length = magnitude( x1 - x0, y1 - y0 );
                    for ( int qp = 0; qp < num_qpts; ++qp ) {
                      const RealT s = rule_coordinates[qp];
                      RealT x_q[2] = { ( 1. - s ) * x0 + s * x1, ( 1. - s ) * y0 + s * y1 };
                      RealT phi1[max_nodes_per_face] = { 0. };
                      RealT phi2[max_nodes_per_face] = { 0. };
                      RealT x_face1[max_dim], x_face2[max_dim];
                      RealT vel1[max_dim] = { 0. }, vel2[max_dim] = { 0. };
                      RealT constraint_normal[2] = { normal[0], normal[1] };
                      if ( mesh1.hasParentElementData() ) {
                        RealT linear_phi1[2];
                        RealT linear_phi2[2];
                        RealT child_parameter1 = 0.;
                        RealT child_parameter2 = 0.;
                        if ( !EvalLinearEdgeAtProjectedPoint( face1, x_q, normal, x_face1, linear_phi1, 0, nullptr,
                                                              nullptr, &child_parameter1 ) ||
                             !EvalLinearEdgeAtProjectedPoint( face2, x_q, normal, x_face2, linear_phi2, 0, nullptr,
                                                              nullptr, &child_parameter2 ) ) {
                          continue;
                        }
                        EvalParentQ2Pair( mesh1, mesh2, index1, index2, child_parameter1, child_parameter2,
                                          dissipative, normal, phi1, phi2, x_face1, x_face2, vel1, vel2,
                                          constraint_normal );
                      } else if ( !EvalLinearEdgeAtProjectedPoint(
                                      face1, x_q, normal, x_face1, phi1, dissipative ? local_dim : 0,
                                      dissipative ? &velocity1[0] : nullptr, dissipative ? vel1 : nullptr ) ||
                                  !EvalLinearEdgeAtProjectedPoint(
                                      face2, x_q, normal, x_face2, phi2, dissipative ? local_dim : 0,
                                      dissipative ? &velocity2[0] : nullptr, dissipative ? vel2 : nullptr ) ) {
                        continue;
                      }
                      const RealT gap = ( x_face1[0] - x_face2[0] ) * constraint_normal[0] +
                                        ( x_face1[1] - x_face2[1] ) * constraint_normal[1];
                      const RealT velocity_gap = ( vel1[0] - vel2[0] ) * constraint_normal[0] +
                                                 ( vel1[1] - vel2[1] ) * constraint_normal[1];
                      const RealT target_velocity = ComputePredictorTargetVelocity( gap, stage_dt );
                      const bool predictor_active = dissipative && stage_dt > 0. && velocity_gap < target_velocity;
                      if ( gap <= gap_tolerance || predictor_active ) {
                        if ( mesh1.hasParentElementData() ) {
                          AccumulateParentConstraintRowBounds(
                              mesh1, mesh2, index1, index2, local_dim, constraint_normal, phi1, phi2,
                              length * rule_weights[qp], penalty_stiffness, predictor_active, predictor_view1,
                              predictor_view2, stiffness_view1, stiffness_view2 );
                        } else {
                          AccumulateConstraintRowBounds(
                              mesh1, mesh2, index1, index2, local_dim, num_nodes_per_face, constraint_normal, phi1, phi2,
                              length * rule_weights[qp], penalty_stiffness, predictor_active, predictor_view1,
                              predictor_view2, stiffness_view1, stiffness_view2 );
                        }
                      }
                    }
                  }
                } );

    RealT predictor_max = 0.;
    RealT stiffness_max = 0.;
    if ( parent_basis ) {
#ifdef BUILD_REDECOMP
      const auto maxima = cs->getMfemMeshData()->AssembleParentQ2RowMaxima(
          predictor_rows1.data(), predictor_rows2.data(), stiffness_rows1.data(), stiffness_rows2.data() );
      predictor_max = maxima.first;
      stiffness_max = maxima.second;
#else
      SLIC_ERROR_ROOT( "Parent surface-basis contact requires BUILD_REDECOMP." );
#endif
    } else {
      ArrayT<RealT> maxima_data( { 0., 0. }, cs->getAllocatorId() );
      auto maxima = maxima_data.view();
      forAllExec( cs->getExecutionMode(), predictor_rows1.size(),
                  [predictor_view1, stiffness_view1, maxima] TRIBOL_HOST_DEVICE( IndexT i ) {
                    tribol::atomicMax( &maxima[0], predictor_view1[i] );
                    tribol::atomicMax( &maxima[1], stiffness_view1[i] );
                  } );
      forAllExec( cs->getExecutionMode(), predictor_rows2.size(),
                  [predictor_view2, stiffness_view2, maxima] TRIBOL_HOST_DEVICE( IndexT i ) {
                    tribol::atomicMax( &maxima[0], predictor_view2[i] );
                    tribol::atomicMax( &maxima[1], stiffness_view2[i] );
                  } );
      ArrayT<RealT, 1, MemorySpace::Host> maxima_host( maxima_data );
#ifdef TRIBOL_USE_MPI
      int mpi_initialized = 0;
      MPI_Initialized( &mpi_initialized );
      if ( mpi_initialized ) {
        MPI_Allreduce( MPI_IN_PLACE, maxima_host.data(), 2, MPI_DOUBLE, MPI_SUM, cs->getProblemComm() );
      }
#endif
      predictor_max = maxima_host[0];
      stiffness_max = maxima_host[1];
    }
    const RealT predictor_bound = predictor_max > 0. ? predictor_max : 1.;
    predictor_relaxation = std::min( 1., penalty_options.predictor_relaxation_scale / predictor_bound );
    const RealT stiffness_bound = stiffness_max;
    const RealT stability_dt = stiffness_bound > 0.
        ? 2. * penalty_options.penalty_stability_scale / sqrt( stiffness_bound )
        : std::numeric_limits<RealT>::infinity();
    cs->setPredictorDiagnostics( predictor_bound, predictor_relaxation );
    cs->setPenaltyStabilityTimeStep( stability_dt );
  }

  const RealT stage_dt = cs->getCurrentTimeStep();
  ArrayT<int> diagnostic_counts_data( { 0, 0, 0 }, cs->getAllocatorId() );
  ArrayT<RealT> integrated_forces_data( { 0., 0., 0. }, cs->getAllocatorId() );
  ArrayT<RealT> contact_sums_data( { 0., 0. }, cs->getAllocatorId() );
  ArrayT<RealT> contact_maxima_data( { 0., 0., 0. }, cs->getAllocatorId() );
  ArrayT<int> parent_normal_fallback_data( { 0 }, cs->getAllocatorId() );
  auto diagnostic_counts = diagnostic_counts_data.view();
  auto integrated_forces = integrated_forces_data.view();
  auto contact_sums = contact_sums_data.view();
  auto contact_maxima = contact_maxima_data.view();
  auto parent_normal_fallbacks = parent_normal_fallback_data.view();
  forAllExec( cs->getExecutionMode(), num_pairs,
              [cs_view, err, neg_thickness, predictor_relaxation, stage_dt, diagnostic_counts,
               integrated_forces, contact_sums, contact_maxima, parent_normal_fallbacks] TRIBOL_HOST_DEVICE( IndexT i ) {
    auto& cg_view = cs_view.getCompGeomView();
    auto& plane = cg_view.getCommonPlane( i );

    auto& mesh1 = cs_view.getMesh1View();
    auto& mesh2 = cs_view.getMesh2View();

    // get pair indices
    IndexT index1 = plane.getCpElementId1();
    IndexT index2 = plane.getCpElementId2();

    RealT gap = plane.m_gap;
    RealT A = plane.m_area;  // face-pair overlap area

    //  don't proceed for gaps that don't violate the constraints. This check
    //  allows for numerically zero interpenetration.
    RealT gap_tol = cs_view.getGapTol( index1, index2 );

    // debug force sums
    // RealT dbg_sum_force1 {0.};
    // RealT dbg_sum_force2 {0.};
    /////////////////////////////////////////////
    // kinematic penalty stiffness calculation //
    /////////////////////////////////////////////
    RealT penalty_stiff_per_area{ 0. };
    auto& enforcement_options = cs_view.getEnforcementOptions();
    const PenaltyEnforcementOptions& pen_enfrc_options = enforcement_options.penalty_options;
    const bool use_multi_point = pen_enfrc_options.common_plane_rule == MULTI_POINT;
    if ( !use_multi_point && gap > gap_tol ) {
      // We are here if we have a pair that passes ALL geometric
      // filter checks, BUT does not actually violate this method's
      // gap constraint.
      plane.m_inContact = false;
      return;
    }

    RealT pen_scale1 = mesh1.getElementData().m_penalty_scale;
    RealT pen_scale2 = mesh2.getElementData().m_penalty_scale;
    switch ( pen_enfrc_options.kinematic_calculation ) {
      case KINEMATIC_CONSTANT: {
        // pre-multiply each spring stiffness by each mesh's penalty scale
        auto stiffness1 = pen_scale1 * mesh1.getElementData().m_penalty_stiffness;
        auto stiffness2 = pen_scale2 * mesh2.getElementData().m_penalty_stiffness;
        // compute the equivalent contact penalty spring stiffness per area
        penalty_stiff_per_area = ComputePenaltyStiffnessPerArea( stiffness1, stiffness2 );
        break;
      }
      case KINEMATIC_ELEMENT: {
        // add tiny_length to element thickness to avoid division by zero
        auto t1 = mesh1.getElementData().m_thickness[index1] + pen_enfrc_options.tiny_length;
        auto t2 = mesh2.getElementData().m_thickness[index2] + pen_enfrc_options.tiny_length;

        if ( t1 < 0. || t2 < 0. ) {
          neg_thickness[0] = true;
          err[0] = 1;
        }

        // compute each element spring stiffness. Pre-multiply the material modulus
        // (i.e. material stiffness) by each mesh's penalty scale
        auto stiffness1 = pen_scale1 * mesh1.getElementData().m_mat_mod[index1] / t1;
        auto stiffness2 = pen_scale2 * mesh2.getElementData().m_mat_mod[index2] / t2;
        // compute the equivalent contact penalty spring stiffness per area
        penalty_stiff_per_area = ComputePenaltyStiffnessPerArea( stiffness1, stiffness2 );
        break;
      }
      default:
        // no-op, quiet compiler
        break;
    }  // end switch on kinematic penalty calculation option

    ////////////////////////////////////////////////////
    // Compute contact pressure(s) on current overlap //
    ////////////////////////////////////////////////////

    // compute total pressure based on constraint type
    RealT totalPressure = 0.;
    plane.m_pressure = gap * penalty_stiff_per_area;  // kinematic contribution
    switch ( pen_enfrc_options.constraint_type ) {
      case KINEMATIC_AND_RATE: {
        // kinematic contribution
        totalPressure += plane.m_pressure;
        // add gap-rate contribution
        totalPressure +=
            ComputeGapRatePressure( plane, mesh1, mesh2, penalty_stiff_per_area, pen_enfrc_options.rate_calculation );
        break;
      }
      case KINEMATIC:
      case KINEMATIC_AND_DISSIPATIVE:
        // kinematic gap pressure contribution  only
        totalPressure += plane.m_pressure;
        break;
      default:
        // no-op
        break;
    }  // end switch on registered penalty enforcement option

    // debug prints. Comment out for now, but keep for future common plane
    // debugging
    //         SLIC_DEBUG("gap: " << gap);
    //         SLIC_DEBUG("area: " << A);
    //         SLIC_DEBUG("penalty stiffness: " << penalty_stiff_per_area);
    //         SLIC_DEBUG("pressure: " << cpManager.m_pressure[ cpID ]);

    ///////////////////////////////////////////
    // create surface contact element struct //
    ///////////////////////////////////////////

    // construct array of nodal coordinates
    RealT xf1[max_dim * max_nodes_per_face];
    RealT xf2[max_dim * max_nodes_per_face];
    RealT xVert[max_dim * max_nodes_per_overlap];
    int dim = cs_view.spatialDimension();
    int num_nodes_per_face = mesh1.numberOfNodesPerElement();
    initRealArray( xf1, dim * num_nodes_per_face, 0. );
    initRealArray( xf2, dim * num_nodes_per_face, 0. );
    // initialize assuming 2d
    auto xVert_size = 4;
    auto numPolyVert = 2;
    // update if we are in 3d
    if ( dim == 3 ) {
      numPolyVert = plane.m_numPolyVert;
      xVert_size = 3 * numPolyVert;
    }
    initRealArray( xVert, xVert_size, 0. );

    // get current configuration, physical coordinates of each face
    plane.getFace1Coords( &xf1[0], num_nodes_per_face );
    plane.getFace2Coords( &xf2[0], num_nodes_per_face );

    // construct array of polygon overlap vertex coordinates
    plane.getOverlapVertices( &xVert[0] );

    // instantiate surface contact element struct. Note, this is done with current
    // configuration face coordinates (i.e. NOT on the contact plane) and overlap
    // coordinates ON the contact plane. The surface contact element does not need
    // to be used this way, but the developer should do the book-keeping.
    SurfaceContactElem cntctElem( dim, xf1, xf2, xVert, num_nodes_per_face, numPolyVert, &mesh1, &mesh2, index1,
                                  index2 );

    // set SurfaceContactElem face normals and overlap normal
    RealT faceNormal1[max_dim];
    RealT faceNormal2[max_dim];
    RealT overlapNormal[max_dim];

    mesh1.getFaceNormal( index1, faceNormal1 );
    mesh2.getFaceNormal( index2, faceNormal2 );
    overlapNormal[0] = plane.m_nX;
    overlapNormal[1] = plane.m_nY;
    if ( dim == 3 ) {
      overlapNormal[2] = plane.m_nZ;
    }

    cntctElem.faceNormal1 = faceNormal1;
    cntctElem.faceNormal2 = faceNormal2;
    cntctElem.overlapNormal = overlapNormal;
    cntctElem.overlapArea = plane.m_area;

    if ( use_multi_point ) {
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf2;
      mesh1.getFaceCoords( index1, actual_xf1 );
      mesh2.getFaceCoords( index2, actual_xf2 );

      const bool use_rate = pen_enfrc_options.constraint_type == KINEMATIC_AND_RATE;
      const bool use_dissipative = pen_enfrc_options.constraint_type == KINEMATIC_AND_DISSIPATIVE;
      const bool use_velocity = use_rate || use_dissipative || ( mesh1.hasVelocity() && mesh2.hasVelocity() );
      const RealT rate_penalty =
          use_rate ? ComputeRatePenalty( mesh1, mesh2, penalty_stiff_per_area, pen_enfrc_options.rate_calculation )
                   : 0.;

      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf2;
      if ( use_velocity ) {
        mesh1.getFaceVelocities( index1, actual_vf1 );
        mesh2.getFaceVelocities( index2, actual_vf2 );
      }

      constexpr int max_qpts = max_symmetric_triangle_qpts;
      RealT rule_wts[max_qpts] = { 0. };
      RealT rule_coords[2 * max_qpts] = { 0. };
      bool has_contact = false;

      if ( dim == 3 ) {
        const int num_qpts =
            GetCommonPlaneTriangleRule( pen_enfrc_options.common_plane_quadrature_order, rule_wts, rule_coords );

        RealT centroid[3];
        GetCommonPlaneOverlapCentroid( cntctElem, centroid );

        RealT xTri[3];
        RealT yTri[3];
        RealT zTri[3];

        for ( int j = 0; j < numPolyVert; ++j ) {
          const int next = ( j == numPolyVert - 1 ) ? 0 : j + 1;
          xTri[0] = xVert[dim * j];
          yTri[0] = xVert[dim * j + 1];
          zTri[0] = xVert[dim * j + 2];
          xTri[1] = xVert[dim * next];
          yTri[1] = xVert[dim * next + 1];
          zTri[1] = xVert[dim * next + 2];
          xTri[2] = centroid[0];
          yTri[2] = centroid[1];
          zTri[2] = centroid[2];

          const RealT area = Area3DTri( xTri, yTri, zTri );
          if ( area <= 0. ) {
            continue;
          }

          for ( int qp = 0; qp < num_qpts; ++qp ) {
            const RealT xi = rule_coords[2 * qp];
            const RealT eta = rule_coords[2 * qp + 1];
            const RealT n0 = 1. - xi - eta;
            RealT x_q[3];
            x_q[0] = n0 * xTri[0] + xi * xTri[1] + eta * xTri[2];
            x_q[1] = n0 * yTri[0] + xi * yTri[1] + eta * yTri[2];
            x_q[2] = n0 * zTri[0] + xi * zTri[1] + eta * zTri[2];

            RealT phi_q1[max_nodes_per_face] = { 0., 0., 0., 0. };
            RealT phi_q2[max_nodes_per_face] = { 0., 0., 0., 0. };
            RealT x_qf1[max_dim];
            RealT x_qf2[max_dim];
            RealT vel_q1[max_dim] = { 0., 0., 0. };
            RealT vel_q2[max_dim] = { 0., 0., 0. };

            const bool mapped_face1 = EvalLinearFaceAtProjectedPoint(
                actual_xf1, num_nodes_per_face, x_q, overlapNormal, x_qf1, phi_q1, use_velocity ? dim : 0,
                use_velocity ? &actual_vf1[0] : nullptr, use_velocity ? vel_q1 : nullptr );
            const bool mapped_face2 = EvalLinearFaceAtProjectedPoint(
                actual_xf2, num_nodes_per_face, x_q, overlapNormal, x_qf2, phi_q2, use_velocity ? dim : 0,
                use_velocity ? &actual_vf2[0] : nullptr, use_velocity ? vel_q2 : nullptr );
            if ( !mapped_face1 || !mapped_face2 ) {
              continue;
            }

            RealT local_gap = ( x_qf1[0] - x_qf2[0] ) * overlapNormal[0] + ( x_qf1[1] - x_qf2[1] ) * overlapNormal[1] +
                              ( x_qf1[2] - x_qf2[2] ) * overlapNormal[2];
            RealT local_vel_gap = 0.;
            if ( use_velocity ) {
              local_vel_gap = ( vel_q1[0] - vel_q2[0] ) * overlapNormal[0] +
                              ( vel_q1[1] - vel_q2[1] ) * overlapNormal[1] +
                              ( vel_q1[2] - vel_q2[2] ) * overlapNormal[2];
            }
            const RealT target_velocity = ComputePredictorTargetVelocity( local_gap, stage_dt );
            const bool predictor_active = use_dissipative && stage_dt > 0. && local_vel_gap < target_velocity;
            if ( local_gap > gap_tol && !predictor_active ) {
              continue;
            }

            has_contact = true;

            const RealT penalty_pressure = local_gap <= gap_tol ? local_gap * penalty_stiff_per_area : 0.;
            RealT local_pressure = penalty_pressure;
            if ( use_rate && rate_penalty > 0. ) {
              if ( local_vel_gap <= 0. ) {
                local_pressure += local_vel_gap * rate_penalty;
              }
            }

            const RealT quadrature_measure = area * rule_wts[qp];
            RealT predictor_pressure = 0.;
            if ( use_dissipative ) {
              predictor_pressure = ComputePredictorPressure(
                  mesh1, mesh2, index1, index2, dim, num_nodes_per_face, overlapNormal, phi_q1, phi_q2, local_gap,
                  local_vel_gap, quadrature_measure, stage_dt, predictor_relaxation );
              local_pressure = std::min( local_pressure, predictor_pressure );
            }
            AccumulateForceDiagnostics( predictor_active, quadrature_measure, penalty_pressure, predictor_pressure,
                                        local_pressure, local_gap, local_vel_gap, diagnostic_counts, integrated_forces,
                                        contact_sums, contact_maxima );

            const RealT weighted_force = quadrature_measure * local_pressure;
            const RealT force_x = overlapNormal[0] * weighted_force;
            const RealT force_y = overlapNormal[1] * weighted_force;
            const RealT force_z = overlapNormal[2] * weighted_force;

            AccumulateContactForce( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, force_x, force_y, force_z,
                                    phi_q1, phi_q2 );
          }
        }
      } else {
        RealT segment_rule_wts[max_segment_gauss_legendre_qpts] = { 0. };
        RealT segment_rule_coords[max_segment_gauss_legendre_qpts] = { 0. };
        const int num_qpts = GetCommonPlaneSegmentRule( pen_enfrc_options.common_plane_quadrature_order,
                                                        segment_rule_wts, segment_rule_coords );
        const RealT x0 = xVert[0];
        const RealT y0 = xVert[1];
        const RealT x1 = xVert[2];
        const RealT y1 = xVert[3];
        const RealT length = magnitude( x1 - x0, y1 - y0 );

        for ( int qp = 0; qp < num_qpts; ++qp ) {
          const RealT s = segment_rule_coords[qp];
          const RealT one_minus_s = 1. - s;
          RealT x_q[2] = { one_minus_s * x0 + s * x1, one_minus_s * y0 + s * y1 };

          RealT phi_q1[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT phi_q2[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT x_qf1[max_dim];
          RealT x_qf2[max_dim];
          RealT vel_q1[max_dim] = { 0., 0., 0. };
          RealT vel_q2[max_dim] = { 0., 0., 0. };
          RealT constraint_normal[2] = { overlapNormal[0], overlapNormal[1] };
          if ( mesh1.hasParentElementData() ) {
            RealT linear_phi1[2];
            RealT linear_phi2[2];
            RealT child_parameter1 = 0.;
            RealT child_parameter2 = 0.;
            if ( !EvalLinearEdgeAtProjectedPoint( actual_xf1, x_q, overlapNormal, x_qf1, linear_phi1, 0, nullptr,
                                                  nullptr, &child_parameter1 ) ||
                 !EvalLinearEdgeAtProjectedPoint( actual_xf2, x_q, overlapNormal, x_qf2, linear_phi2, 0, nullptr,
                                                  nullptr, &child_parameter2 ) ) {
              continue;
            }
            const bool used_parent_normal = EvalParentQ2Pair(
                mesh1, mesh2, index1, index2, child_parameter1, child_parameter2, use_velocity, overlapNormal, phi_q1,
                phi_q2, x_qf1, x_qf2, vel_q1, vel_q2, constraint_normal );
            if ( !used_parent_normal ) {
              tribol::atomicInc( &parent_normal_fallbacks[0] );
            }
          } else {
            const bool mapped_face1 = EvalLinearEdgeAtProjectedPoint(
                actual_xf1, x_q, overlapNormal, x_qf1, phi_q1, use_velocity ? dim : 0,
                use_velocity ? &actual_vf1[0] : nullptr, use_velocity ? vel_q1 : nullptr );
            const bool mapped_face2 = EvalLinearEdgeAtProjectedPoint(
                actual_xf2, x_q, overlapNormal, x_qf2, phi_q2, use_velocity ? dim : 0,
                use_velocity ? &actual_vf2[0] : nullptr, use_velocity ? vel_q2 : nullptr );
            if ( !mapped_face1 || !mapped_face2 ) {
              continue;
            }
          }

          RealT local_gap = ( x_qf1[0] - x_qf2[0] ) * constraint_normal[0] +
                            ( x_qf1[1] - x_qf2[1] ) * constraint_normal[1];
          RealT local_vel_gap = 0.;
          if ( use_velocity ) {
            local_vel_gap = ( vel_q1[0] - vel_q2[0] ) * constraint_normal[0] +
                            ( vel_q1[1] - vel_q2[1] ) * constraint_normal[1];
          }
          const RealT target_velocity = ComputePredictorTargetVelocity( local_gap, stage_dt );
          const bool predictor_active = use_dissipative && stage_dt > 0. && local_vel_gap < target_velocity;
          if ( local_gap > gap_tol && !predictor_active ) {
            continue;
          }

          has_contact = true;

          const RealT penalty_pressure = local_gap <= gap_tol ? local_gap * penalty_stiff_per_area : 0.;
          RealT local_pressure = penalty_pressure;
          if ( use_rate && rate_penalty > 0. ) {
            if ( local_vel_gap <= 0. ) {
              local_pressure += local_vel_gap * rate_penalty;
            }
          }

          const RealT quadrature_measure = length * segment_rule_wts[qp];
          RealT predictor_pressure = 0.;
          if ( use_dissipative ) {
            predictor_pressure = mesh1.hasParentElementData()
                ? ComputeParentPredictorPressure( mesh1, mesh2, index1, index2, dim, constraint_normal, phi_q1, phi_q2,
                                                  local_gap, local_vel_gap, quadrature_measure, stage_dt,
                                                  predictor_relaxation )
                : ComputePredictorPressure( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, constraint_normal,
                                            phi_q1, phi_q2, local_gap, local_vel_gap, quadrature_measure, stage_dt,
                                            predictor_relaxation );
            local_pressure = std::min( local_pressure, predictor_pressure );
          }
          AccumulateForceDiagnostics( predictor_active, quadrature_measure, penalty_pressure, predictor_pressure,
                                      local_pressure, local_gap, local_vel_gap, diagnostic_counts, integrated_forces,
                                      contact_sums, contact_maxima );

          const RealT weighted_force = quadrature_measure * local_pressure;
          const RealT force[2] = { constraint_normal[0] * weighted_force,
                                   constraint_normal[1] * weighted_force };
          if ( mesh1.hasParentElementData() ) {
            AccumulateParentContactForce( mesh1, mesh2, index1, index2, dim, force, phi_q1, phi_q2 );
          } else {
            AccumulateContactForce( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, force[0], force[1], 0.,
                                    phi_q1, phi_q2 );
          }
        }
      }

      plane.m_inContact = has_contact;
      return;
    }

    // create arrays to hold nodal residual weak form integral evaluations
    RealT phi1[max_nodes_per_face];
    RealT phi2[max_nodes_per_face];
    initRealArray( phi1, num_nodes_per_face, 0. );
    initRealArray( phi2, num_nodes_per_face, 0. );

    ////////////////////////////////////////////////////////////////////////
    // Integration of contact integrals: integral of shape functions over //
    // contact overlap patch                                              //
    ////////////////////////////////////////////////////////////////////////
    EvalWeakFormIntegralCommonPlane( cntctElem, pen_enfrc_options.common_plane_rule,
                                     pen_enfrc_options.common_plane_quadrature_order, phi1, phi2 );

    ///////////////////////////////////////////////////////////////////////
    // Computation of full contact nodal force contributions             //
    // (i.e. premultiplication of contact integrals by normal component, //
    //  contact pressure, and overlap area)                              //
    ///////////////////////////////////////////////////////////////////////

    // RealT phi_sum_1 = 0.;
    // RealT phi_sum_2 = 0.;

    // compute contact force (spring force)
    AccumulateForceDiagnostics( false, A, plane.m_pressure, 0., totalPressure, gap, 0., diagnostic_counts,
                                integrated_forces, contact_sums, contact_maxima );
    RealT contact_force = totalPressure * A;

    RealT force_x = overlapNormal[0] * contact_force;
    RealT force_y = overlapNormal[1] * contact_force;
    RealT force_z = 0.;
    if ( dim == 3 ) {
      force_z = overlapNormal[2] * contact_force;
    }

    AccumulateContactForce( mesh1, mesh2, index1, index2, dim, num_nodes_per_face, force_x, force_y, force_z, phi1,
                            phi2 );

    // comment out debug logs; too much output during tests. Keep for easy
    // debugging if needed
    // SLIC_DEBUG("force sum, side 1, pair " << kp << ": " << -dbg_sum_force1 );
    // SLIC_DEBUG("force sum, side 2, pair " << kp << ": " << dbg_sum_force2 );
    // SLIC_DEBUG("phi 1 sum: " << phi_sum_1 );
    // SLIC_DEBUG("phi 2 sum: " << phi_sum_2 );
  } );

  ArrayT<int, 1, MemorySpace::Host> diagnostic_counts_host( diagnostic_counts_data );
  ArrayT<RealT, 1, MemorySpace::Host> integrated_forces_host( integrated_forces_data );
  ArrayT<RealT, 1, MemorySpace::Host> contact_sums_host( contact_sums_data );
  ArrayT<RealT, 1, MemorySpace::Host> contact_maxima_host( contact_maxima_data );
  ArrayT<int, 1, MemorySpace::Host> parent_normal_fallback_host( parent_normal_fallback_data );
#ifdef TRIBOL_USE_MPI
  int mpi_initialized = 0;
  MPI_Initialized( &mpi_initialized );
  if ( mpi_initialized ) {
    MPI_Allreduce( MPI_IN_PLACE, diagnostic_counts_host.data(), 3, MPI_INT, MPI_SUM, cs->getProblemComm() );
    MPI_Allreduce( MPI_IN_PLACE, integrated_forces_host.data(), 3, MPI_DOUBLE, MPI_SUM, cs->getProblemComm() );
    MPI_Allreduce( MPI_IN_PLACE, contact_sums_host.data(), 2, MPI_DOUBLE, MPI_SUM, cs->getProblemComm() );
    MPI_Allreduce( MPI_IN_PLACE, contact_maxima_host.data(), 3, MPI_DOUBLE, MPI_MAX, cs->getProblemComm() );
    MPI_Allreduce( MPI_IN_PLACE, parent_normal_fallback_host.data(), 1, MPI_INT, MPI_SUM, cs->getProblemComm() );
  }
#endif
  cs->setPredictorForceDiagnostics( diagnostic_counts_host[0], diagnostic_counts_host[1], integrated_forces_host[0],
                                    integrated_forces_host[1], integrated_forces_host[2] );
  cs->setContactPointDiagnostics( diagnostic_counts_host[2], contact_maxima_host[0], contact_sums_host[0],
                                  contact_maxima_host[1], contact_sums_host[1], contact_maxima_host[2] );
  SLIC_DEBUG_IF( parent_normal_fallback_host[0] > 0,
                 axom::fmt::format( "Parent Q2 common-plane normal fallbacks: {0}",
                                    parent_normal_fallback_host[0] ) );

  ArrayT<bool, 1, MemorySpace::Host> neg_thickness_host( neg_thickness_data );
  SLIC_DEBUG_IF( neg_thickness_host[0],
                 "ApplyNormal<COMMON_PLANE, PENALTY>: negative element thicknesses encountered." );

  ArrayT<int, 1, MemorySpace::Host> err_host( err_data );
  return err_host[0];

}  // end ApplyNormal<COMMON_PLANE, PENALTY>()

//------------------------------------------------------------------------------
template <>
int ApplyNormal<COMMON_PLANE, IMPULSE_PROJECTION>( CouplingScheme* cs )
{
#ifndef BUILD_REDECOMP
  SLIC_ERROR_ROOT( "Common-plane impulse projection requires BUILD_REDECOMP." );
  return 1;
#else
  SLIC_ERROR_ROOT_IF( cs->spatialDimension() != 2,
                      "Common-plane impulse projection currently supports only two-dimensional contact." );
  SLIC_ERROR_ROOT_IF( cs->getExecutionMode() != ExecutionMode::Sequential,
                      "Common-plane impulse projection currently supports sequential execution only." );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(), "Common-plane impulse projection requires MFEM mesh data." );
  SLIC_ERROR_ROOT_IF( cs->getMfemMeshData()->GetSurfaceBasis() != MfemSurfaceBasis::PARENT,
                      "Common-plane impulse projection requires the parent surface basis." );
  SLIC_ERROR_ROOT_IF( cs->getEnforcementOptions().penalty_options.common_plane_rule != MULTI_POINT,
                      "Common-plane impulse projection requires MULTI_POINT integration." );
  SLIC_ERROR_ROOT_IF( !cs->getMesh1().hasParentElementData() || !cs->getMesh2().hasParentElementData(),
                      "Common-plane impulse projection requires parent element data." );
  SLIC_ERROR_ROOT_IF( !cs->getMesh1().hasInverseMass() || !cs->getMesh2().hasInverseMass(),
                      "Common-plane impulse projection requires inverse lumped masses." );
  SLIC_ERROR_ROOT_IF( cs->getCurrentTimeStep() <= 0.,
                      "Common-plane impulse projection requires a positive stage timestep." );

#ifdef TRIBOL_USE_MPI
  int comm_size = 1;
  MPI_Comm_size( cs->getProblemComm(), &comm_size );
  SLIC_ERROR_ROOT_IF( comm_size != 1, "Common-plane impulse projection currently supports one MPI rank." );
#endif

  auto* mfem_data = cs->getMfemMeshData();
  mfem_data->BeginParentQ2Projection();
  auto cs_view = cs->getView();
  auto mesh1 = cs_view.getMesh1View();
  auto mesh2 = cs_view.getMesh2View();
  const RealT stage_dt = cs->getCurrentTimeStep();
  const auto& integration_options = cs->getEnforcementOptions().penalty_options;
  const auto& projection_options = cs->getEnforcementOptions().projection_options;
  const RealT position_velocity_scale = projection_options.position_velocity_scale;
  SLIC_ERROR_ROOT_IF( position_velocity_scale < 1. &&
                          ( !mesh1.hasParentProjectionBaseVelocity() ||
                            !mesh2.hasParentProjectionBaseVelocity() ),
                      "Common-plane impulse projection requires a base velocity when the position velocity scale "
                      "is less than one." );

  RealT rule_weights[max_segment_gauss_legendre_qpts] = { 0. };
  RealT rule_coordinates[max_segment_gauss_legendre_qpts] = { 0. };
  const int num_rule_points = GetCommonPlaneSegmentRule( integration_options.common_plane_quadrature_order,
                                                         rule_weights, rule_coordinates );
  std::vector<ProjectionConstraint> constraints;
  constraints.reserve( static_cast<std::size_t>( cs->getNumActivePairs() * num_rule_points ) );

  for ( IndexT i = 0; i < cs->getNumActivePairs(); ++i ) {
    auto& plane = cs_view.getCompGeomView().getCommonPlane( i );
    plane.m_inContact = false;
    plane.m_pressure = 0.;
    const IndexT element1 = plane.getCpElementId1();
    const IndexT element2 = plane.getCpElementId2();
    StackArrayT<RealT, max_dim * max_nodes_per_face> face1;
    StackArrayT<RealT, max_dim * max_nodes_per_face> face2;
    mesh1.getFaceCoords( element1, face1 );
    mesh2.getFaceCoords( element2, face2 );
    RealT overlap_vertices[max_dim * max_nodes_per_overlap] = { 0. };
    plane.getOverlapVertices( overlap_vertices );
    const RealT x0 = overlap_vertices[0];
    const RealT y0 = overlap_vertices[1];
    const RealT x1 = overlap_vertices[2];
    const RealT y1 = overlap_vertices[3];
    const RealT length = magnitude( x1 - x0, y1 - y0 );
    if ( !std::isfinite( length ) || length <= 0. ) {
      continue;
    }
    const RealT overlap_normal[2] = { plane.m_nX, plane.m_nY };
    bool pair_has_constraint = false;

    for ( int qp = 0; qp < num_rule_points; ++qp ) {
      const RealT coordinate = rule_coordinates[qp];
      const RealT x_q[2] = { ( 1. - coordinate ) * x0 + coordinate * x1,
                             ( 1. - coordinate ) * y0 + coordinate * y1 };
      RealT x_face1[max_dim];
      RealT x_face2[max_dim];
      RealT linear_phi1[2];
      RealT linear_phi2[2];
      RealT child_parameter1 = 0.;
      RealT child_parameter2 = 0.;
      if ( !EvalLinearEdgeAtProjectedPoint( face1, x_q, overlap_normal, x_face1, linear_phi1, 0, nullptr, nullptr,
                                            &child_parameter1 ) ||
           !EvalLinearEdgeAtProjectedPoint( face2, x_q, overlap_normal, x_face2, linear_phi2, 0, nullptr, nullptr,
                                            &child_parameter2 ) ) {
        continue;
      }

      ProjectionConstraint constraint{};
      constraint.plane_index = i;
      constraint.element1 = element1;
      constraint.element2 = element2;
      RealT velocity1[max_dim] = { 0., 0., 0. };
      RealT velocity2[max_dim] = { 0., 0., 0. };
      EvalParentQ2Pair( mesh1, mesh2, element1, element2, child_parameter1, child_parameter2, true, overlap_normal,
                        constraint.phi1, constraint.phi2, x_face1, x_face2, velocity1, velocity2,
                        constraint.normal );
      constraint.gap = ( x_face1[0] - x_face2[0] ) * constraint.normal[0] +
                       ( x_face1[1] - x_face2[1] ) * constraint.normal[1];
      constraint.quadrature_measure = length * rule_weights[qp];
      constraint.diagonal = ComputeParentInverseEffectiveMass( mesh1, mesh2, element1, element2, 2,
                                                               constraint.normal, constraint.phi1, constraint.phi2 );
      constraint.trial_velocity = EvaluateProjectionVelocity( constraint, mesh1, mesh2 );
      const RealT base_velocity = position_velocity_scale < 1.
                                      ? EvaluateProjectionBaseVelocity( constraint, mesh1, mesh2 )
                                      : constraint.trial_velocity;
      constraint.position_trial_velocity = ( 1. - position_velocity_scale ) * base_velocity +
                                           position_velocity_scale * constraint.trial_velocity;
      const RealT target_position_velocity = constraint.gap > 0. ? -constraint.gap / stage_dt : 0.;
      constraint.target_velocity =
          constraint.trial_velocity +
          ( target_position_velocity - constraint.position_trial_velocity ) / position_velocity_scale;
      if ( !std::isfinite( constraint.gap ) || !std::isfinite( constraint.target_velocity ) ||
           !std::isfinite( constraint.quadrature_measure ) || constraint.quadrature_measure <= 0. ||
           !std::isfinite( constraint.diagonal ) || constraint.diagonal <= 0. ||
           !std::isfinite( constraint.trial_velocity ) ) {
        mfem_data->ResetParentQ2Projection();
        cs->setProjectionDiagnostics( static_cast<IndexT>( constraints.size() ), 0, 0, false, false, 0., 0., 0., 0.,
                                      1., 1., 0., 0., 0., 0. );
        return 1;
      }
      constraints.push_back( constraint );
      pair_has_constraint = true;
    }
    plane.m_inContact = pair_has_constraint;
  }

  if ( constraints.empty() ) {
    cs->setProjectionDiagnostics( 0, 0, 0, true, true, 0., 0., 0., 0., 1., 1., 0., 0., 0., 0. );
    cs->setCompliantProjectionDiagnostics( 0., 0., 0., 0, 0., 0. );
    cs->setPredictorForceDiagnostics( 0, 0, 0., 0., 0. );
    cs->setContactPointDiagnostics( 0, 0., 0., 0., 0., 0. );
    return 0;
  }

  const IndexT row_count1 = mesh1.numberOfElements() * parent_q2_num_nodes * 2;
  const IndexT row_count2 = mesh2.numberOfElements() * parent_q2_num_nodes * 2;
  ArrayT<RealT> normalized_rows1( row_count1, std::max<IndexT>( row_count1, 1 ), cs->getAllocatorId() );
  ArrayT<RealT> normalized_rows2( row_count2, std::max<IndexT>( row_count2, 1 ), cs->getAllocatorId() );
  ArrayT<RealT> unused_rows1( row_count1, std::max<IndexT>( row_count1, 1 ), cs->getAllocatorId() );
  ArrayT<RealT> unused_rows2( row_count2, std::max<IndexT>( row_count2, 1 ), cs->getAllocatorId() );
  normalized_rows1.fill( 0. );
  normalized_rows2.fill( 0. );
  unused_rows1.fill( 0. );
  unused_rows2.fill( 0. );
  auto normalized_view1 = normalized_rows1.view();
  auto normalized_view2 = normalized_rows2.view();
  auto unused_view1 = unused_rows1.view();
  auto unused_view2 = unused_rows2.view();
  for ( const auto& constraint : constraints ) {
    AccumulateParentConstraintRowBounds( mesh1, mesh2, constraint.element1, constraint.element2, 2,
                                         constraint.normal, constraint.phi1, constraint.phi2,
                                         constraint.quadrature_measure, 0., true, normalized_view1,
                                         normalized_view2, unused_view1, unused_view2 );
  }
  const auto row_maxima = mfem_data->AssembleParentQ2RowMaxima(
      normalized_rows1.data(), normalized_rows2.data(), unused_rows1.data(), unused_rows2.data() );
  const RealT coupling_bound = row_maxima.first > 0. ? row_maxima.first : 1.;
  const RealT relaxation = std::min( 1., projection_options.relaxation_scale / coupling_bound );

  auto compute_residual = [&]() {
    ProjectionResiduals residuals;
    for ( const auto& constraint : constraints ) {
      const RealT velocity = EvaluateProjectionVelocity( constraint, mesh1, mesh2 );
      residuals.complementarity = std::max(
          residuals.complementarity,
          std::abs( std::min( constraint.diagonal * constraint.multiplier,
                              velocity - constraint.target_velocity ) ) );
      residuals.primal = std::max( residuals.primal, constraint.target_velocity - velocity );
    }
    residuals.primal = std::max( 0., residuals.primal );
    return residuals;
  };

  const ProjectionResiduals initial_residuals = compute_residual();
  const RealT initial_residual = initial_residuals.complementarity;
  RealT final_residual = initial_residual;
  RealT final_primal_residual = initial_residuals.primal;
  const RealT convergence_tolerance =
      projection_options.absolute_tolerance + projection_options.relative_tolerance * initial_residual;
  const RealT primal_tolerance =
      projection_options.absolute_tolerance + projection_options.primal_relative_tolerance * initial_residuals.primal;
  bool complementarity_converged = final_residual <= convergence_tolerance;
  bool valid = std::isfinite( initial_residual ) && std::isfinite( relaxation ) && relaxation > 0.;
  int iterations = 0;
  for ( int iteration = 1;
        valid && !complementarity_converged && iteration <= projection_options.max_iterations; ++iteration ) {
    for ( auto& constraint : constraints ) {
      const RealT velocity = EvaluateProjectionVelocity( constraint, mesh1, mesh2 );
      const RealT old_multiplier = constraint.multiplier;
      constraint.multiplier = std::max(
          0., old_multiplier + relaxation * ( constraint.target_velocity - velocity ) / constraint.diagonal );
      const RealT multiplier_increment = constraint.multiplier - old_multiplier;
      valid = valid && std::isfinite( constraint.multiplier ) && std::isfinite( multiplier_increment );
      AccumulateProjectionImpulse( constraint, multiplier_increment, mesh1, mesh2 );
    }
    valid = valid && mfem_data->ApplyParentQ2ProjectionImpulse();
    const ProjectionResiduals residuals = compute_residual();
    final_residual = residuals.complementarity;
    final_primal_residual = residuals.primal;
    valid = valid && std::isfinite( final_residual ) && std::isfinite( final_primal_residual );
    complementarity_converged = valid && final_residual <= convergence_tolerance;
    iterations = iteration;
  }

  IndexT active_multipliers = 0;
  RealT total_impulse = 0.;
  RealT maximum_force = 0.;
  RealT gap_violation_sum = 0.;
  RealT maximum_gap_violation = 0.;
  RealT closing_rate_sum = 0.;
  RealT maximum_closing_rate = 0.;
  RealT maximum_endpoint_violation = 0.;
  RealT energy_change = 0.;
  RealT energy_scale = 0.;
  std::vector<RealT> plane_impulses( static_cast<std::size_t>( cs->getNumActivePairs() ), 0. );
  for ( const auto& constraint : constraints ) {
    const RealT final_velocity = EvaluateProjectionVelocity( constraint, mesh1, mesh2 );
    const RealT position_final_velocity =
        constraint.position_trial_velocity +
        position_velocity_scale * ( final_velocity - constraint.trial_velocity );
    const RealT endpoint_gap = constraint.gap + stage_dt * position_final_velocity;
    const RealT gap_violation = std::max( 0., -constraint.gap );
    const RealT closing_rate = std::max( 0., -constraint.trial_velocity );
    const RealT energy_term = 0.5 * constraint.multiplier *
                              ( constraint.trial_velocity + final_velocity );
    if ( constraint.multiplier > 0. ) {
      ++active_multipliers;
    }
    total_impulse += constraint.multiplier;
    plane_impulses[constraint.plane_index] += constraint.multiplier;
    maximum_force = std::max( maximum_force, constraint.multiplier / stage_dt );
    gap_violation_sum += gap_violation;
    maximum_gap_violation = std::max( maximum_gap_violation, gap_violation );
    closing_rate_sum += closing_rate;
    maximum_closing_rate = std::max( maximum_closing_rate, closing_rate );
    maximum_endpoint_violation = std::max( maximum_endpoint_violation, std::max( 0., -endpoint_gap ) );
    energy_change += energy_term;
    energy_scale += std::abs( energy_term );
  }
  const RealT energy_tolerance = 100. * std::numeric_limits<RealT>::epsilon() * std::max( 1., energy_scale );
  const bool primal_converged = final_primal_residual <= primal_tolerance;
  const bool accepted = valid && primal_converged && std::isfinite( energy_change ) &&
                        energy_change <= energy_tolerance;

  if ( !accepted && !cs->hasProjectionOperatorDiagnostics() ) {
    const ProjectionOperatorDiagnostics operator_diagnostics =
        ComputeProjectionOperatorDiagnostics( constraints, relaxation, *mfem_data );
    cs->setProjectionOperatorDiagnostics(
        operator_diagnostics.velocity_dofs, operator_diagnostics.rank, operator_diagnostics.minimum_eigenvalue,
        operator_diagnostics.maximum_eigenvalue, operator_diagnostics.condition_estimate,
        operator_diagnostics.jacobi_contraction );
  }

  if ( accepted ) {
    for ( const auto& constraint : constraints ) {
      AccumulateProjectionImpulse( constraint, constraint.multiplier / stage_dt, mesh1, mesh2 );
    }
  } else {
    mfem_data->ResetParentQ2Projection();
  }

  for ( IndexT i = 0; i < cs->getNumActivePairs(); ++i ) {
    auto& plane = cs_view.getCompGeomView().getCommonPlane( i );
    const RealT impulse = accepted ? plane_impulses[i] : 0.;
    plane.m_pressure = plane.m_area > 0. ? -impulse / ( plane.m_area * stage_dt ) : 0.;
    plane.m_inContact = impulse > 0.;
  }

  const RealT equivalent_force = total_impulse / stage_dt;
  cs->setProjectionDiagnostics( static_cast<IndexT>( constraints.size() ), active_multipliers, iterations, accepted,
                                complementarity_converged, initial_residual, final_residual, final_primal_residual,
                                primal_tolerance, coupling_bound, relaxation, total_impulse, equivalent_force,
                                maximum_endpoint_violation, energy_change );
  cs->setPredictorForceDiagnostics( 0, 0, 0., 0., accepted ? equivalent_force : 0. );
  cs->setContactPointDiagnostics( static_cast<IndexT>( constraints.size() ), accepted ? maximum_force : 0.,
                                  gap_violation_sum, maximum_gap_violation, closing_rate_sum, maximum_closing_rate );
  return accepted ? 0 : 1;
#endif
}

//------------------------------------------------------------------------------
template <>
int ApplyNormal<PARENT_TRACE_MORTAR, IMPULSE_PROJECTION>( CouplingScheme* cs )
{
#ifndef BUILD_REDECOMP
  SLIC_ERROR_ROOT( "Parent-trace mortar impulse projection requires BUILD_REDECOMP." );
  return 1;
#else
  SLIC_ERROR_ROOT_IF( cs->spatialDimension() != 2,
                      "Parent-trace mortar impulse projection currently supports only two-dimensional contact." );
  SLIC_ERROR_ROOT_IF( cs->getExecutionMode() != ExecutionMode::Sequential,
                      "Parent-trace mortar impulse projection currently supports sequential execution only." );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(), "Parent-trace mortar impulse projection requires MFEM mesh data." );
  SLIC_ERROR_ROOT_IF( cs->getMfemMeshData()->GetSurfaceBasis() != MfemSurfaceBasis::PARENT,
                      "Parent-trace mortar impulse projection requires the parent surface basis." );
  SLIC_ERROR_ROOT_IF( cs->getEnforcementOptions().penalty_options.common_plane_rule != MULTI_POINT,
                      "Parent-trace mortar impulse projection requires MULTI_POINT integration." );
  SLIC_ERROR_ROOT_IF( !cs->getMesh1().hasParentElementData() || !cs->getMesh2().hasParentElementData(),
                      "Parent-trace mortar impulse projection requires parent element data." );
  SLIC_ERROR_ROOT_IF( !cs->getMesh1().hasInverseMass() || !cs->getMesh2().hasInverseMass(),
                      "Parent-trace mortar impulse projection requires inverse lumped masses." );
  SLIC_ERROR_ROOT_IF( cs->getCurrentTimeStep() <= 0.,
                      "Parent-trace mortar impulse projection requires a positive stage timestep." );

#ifdef TRIBOL_USE_MPI
  int comm_size = 1;
  MPI_Comm_size( cs->getProblemComm(), &comm_size );
  SLIC_ERROR_ROOT_IF( comm_size != 1, "Parent-trace mortar impulse projection currently supports one MPI rank." );
#endif

  auto* mfem_data = cs->getMfemMeshData();
  mfem_data->BeginParentQ2Projection();
  auto cs_view = cs->getView();
  auto mesh1 = cs_view.getMesh1View();
  auto mesh2 = cs_view.getMesh2View();
  SLIC_ERROR_ROOT_IF( !mesh2.hasParentDofIds(),
                      "Parent-trace mortar impulse projection requires parent trace DOF identifiers." );
  const RealT stage_dt = cs->getCurrentTimeStep();
  const auto& integration_options = cs->getEnforcementOptions().penalty_options;
  const auto& projection_options = cs->getEnforcementOptions().projection_options;
  const RealT position_velocity_scale = projection_options.position_velocity_scale;
  SLIC_ERROR_ROOT_IF( position_velocity_scale < 1. &&
                          ( !mesh1.hasParentProjectionBaseVelocity() ||
                            !mesh2.hasParentProjectionBaseVelocity() ),
                      "Parent-trace mortar impulse projection requires a base velocity when the position velocity "
                      "scale is less than one." );

  std::map<ParentTraceFaceKey, IndexT> face_indices;
  std::vector<ParentTraceFace> faces;
  std::map<IndexT, std::vector<std::pair<IndexT, IndexT>>> dof_incidence;
  for ( IndexT element = 0; element < mesh2.numberOfElements(); ++element ) {
    const ParentTraceFaceKey key = GetParentTraceFaceKey( mesh2, element );
    auto found = face_indices.find( key );
    if ( found != face_indices.end() ) {
      continue;
    }
    ParentTraceFace face;
    if ( !BuildParentTraceFace( mesh2, element, face ) ) {
      mfem_data->ResetParentQ2Projection();
      return 1;
    }
    const IndexT face_index = static_cast<IndexT>( faces.size() );
    face_indices.emplace( key, face_index );
    faces.push_back( face );
    for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
      dof_incidence[face.local_dofs[a]].push_back( { face_index, a } );
    }
  }

  std::vector<TraceProjectionConstraint> constraints;
  std::vector<IndexT> row_face_counts;
  const RealT patch_angle = projection_options.normal_patch_angle_degrees;
  const RealT patch_cosine = std::cos( patch_angle * std::acos( -1. ) / 180. );
  for ( const auto& dof_entry : dof_incidence ) {
    struct Patch {
      IndexT representative_face;
      std::vector<std::pair<IndexT, IndexT>> incidence;
    };
    std::vector<Patch> patches;
    for ( const auto& incidence : dof_entry.second ) {
      const auto& face = faces[incidence.first];
      IndexT patch_index = -1;
      for ( IndexT candidate = 0; candidate < static_cast<IndexT>( patches.size() ); ++candidate ) {
        const auto& representative = faces[patches[candidate].representative_face];
        const RealT normal_dot = face.normal[0] * representative.normal[0] +
                                 face.normal[1] * representative.normal[1];
        if ( normal_dot >= patch_cosine ) {
          patch_index = candidate;
          break;
        }
      }
      if ( patch_index < 0 ) {
        patches.push_back( { incidence.first, {} } );
        patch_index = static_cast<IndexT>( patches.size() - 1 );
      }
      patches[patch_index].incidence.push_back( incidence );
    }

    for ( IndexT patch = 0; patch < static_cast<IndexT>( patches.size() ); ++patch ) {
      const IndexT row = static_cast<IndexT>( constraints.size() );
      TraceProjectionConstraint constraint;
      constraint.parent_dof = dof_entry.first;
      constraint.patch = patch;
      constraints.push_back( std::move( constraint ) );
      row_face_counts.push_back( static_cast<IndexT>( patches[patch].incidence.size() ) );
      for ( const auto& incidence : patches[patch].incidence ) {
        faces[incidence.first].rows[incidence.second] = row;
      }
    }
  }

  RealT rule_weights[max_segment_gauss_legendre_qpts] = { 0. };
  RealT rule_coordinates[max_segment_gauss_legendre_qpts] = { 0. };
  const int num_rule_points = GetCommonPlaneSegmentRule( integration_options.common_plane_quadrature_order,
                                                         rule_weights, rule_coordinates );
  for ( IndexT i = 0; i < cs->getNumActivePairs(); ++i ) {
    auto& plane = cs_view.getCompGeomView().getCommonPlane( i );
    plane.m_inContact = false;
    plane.m_pressure = 0.;
    const IndexT element1 = plane.getCpElementId1();
    const IndexT element2 = plane.getCpElementId2();
    const auto face_found = face_indices.find( GetParentTraceFaceKey( mesh2, element2 ) );
    if ( face_found == face_indices.end() ) {
      continue;
    }
    const auto& mortar_face = faces[face_found->second];
    StackArrayT<RealT, max_dim * max_nodes_per_face> face1;
    StackArrayT<RealT, max_dim * max_nodes_per_face> face2;
    mesh1.getFaceCoords( element1, face1 );
    mesh2.getFaceCoords( element2, face2 );
    RealT overlap_vertices[max_dim * max_nodes_per_overlap] = { 0. };
    plane.getOverlapVertices( overlap_vertices );
    const RealT x0 = overlap_vertices[0];
    const RealT y0 = overlap_vertices[1];
    const RealT x1 = overlap_vertices[2];
    const RealT y1 = overlap_vertices[3];
    const RealT length = magnitude( x1 - x0, y1 - y0 );
    if ( !std::isfinite( length ) || length <= 0. ) {
      continue;
    }
    const RealT overlap_normal[2] = { plane.m_nX, plane.m_nY };

    for ( int qp = 0; qp < num_rule_points; ++qp ) {
      const RealT coordinate = rule_coordinates[qp];
      const RealT x_q[2] = { ( 1. - coordinate ) * x0 + coordinate * x1,
                             ( 1. - coordinate ) * y0 + coordinate * y1 };
      RealT x_face1[max_dim];
      RealT x_face2[max_dim];
      RealT linear_phi1[2];
      RealT linear_phi2[2];
      RealT child_parameter1 = 0.;
      RealT child_parameter2 = 0.;
      if ( !EvalLinearEdgeAtProjectedPoint( face1, x_q, overlap_normal, x_face1, linear_phi1, 0, nullptr, nullptr,
                                            &child_parameter1 ) ||
           !EvalLinearEdgeAtProjectedPoint( face2, x_q, overlap_normal, x_face2, linear_phi2, 0, nullptr, nullptr,
                                            &child_parameter2 ) ) {
        continue;
      }

      RealT phi1[parent_q2_num_nodes];
      RealT phi2[parent_q2_num_nodes];
      RealT velocity1[max_dim] = { 0., 0., 0. };
      RealT velocity2[max_dim] = { 0., 0., 0. };
      RealT common_normal[2];
      EvalParentQ2Pair( mesh1, mesh2, element1, element2, child_parameter1, child_parameter2, true, overlap_normal,
                        phi1, phi2, x_face1, x_face2, velocity1, velocity2, common_normal );
      const RealT gap = ( x_face1[0] - x_face2[0] ) * common_normal[0] +
                        ( x_face1[1] - x_face2[1] ) * common_normal[1];
      const RealT quadrature_measure = length * rule_weights[qp];
      const RealT parent_xi0 = mesh2.getParentReferenceCoordinate( element2, 0 );
      const RealT parent_xi1 = mesh2.getParentReferenceCoordinate( element2, 1 );
      const RealT parent_xi = ( 1. - child_parameter2 ) * parent_xi0 + child_parameter2 * parent_xi1;
      RealT lor_phi[parent_q2_num_nodes];
      EvalParentLORBasis( parent_xi, lor_phi );
      const bool use_compliant_response =
          projection_options.contact_response == PROJECTION_RESPONSE_COMPLIANT;
      const RealT penalty_stiffness = use_compliant_response
                                          ? ComputeProjectionPenaltyStiffnessPerArea(
                                                mesh1, mesh2, element1, element2, integration_options )
                                          : 0.;
      const RealT local_thickness =
          use_compliant_response
              ? std::min( mesh1.getElementData().m_thickness[element1],
                          mesh2.getElementData().m_thickness[element2] )
              : 0.;

      for ( int a = 0; a < parent_q2_num_nodes; ++a ) {
        const IndexT row = mortar_face.rows[a];
        if ( row < 0 ) {
          continue;
        }
        RealT dual_value = 0.;
        for ( int b = 0; b < parent_q2_num_nodes; ++b ) {
          dual_value += mortar_face.inverse_mass[a][b] * phi2[b];
        }
        dual_value /= static_cast<RealT>( row_face_counts[row] );
        const RealT coefficient = quadrature_measure * dual_value;
        if ( coefficient == 0. ) {
          continue;
        }
        TraceProjectionContribution contribution{};
        contribution.plane_index = i;
        contribution.element1 = element1;
        contribution.element2 = element2;
        contribution.normal[0] = common_normal[0];
        contribution.normal[1] = common_normal[1];
        contribution.plane_weight = coefficient;
        for ( int b = 0; b < parent_q2_num_nodes; ++b ) {
          contribution.phi1[b] = coefficient * phi1[b];
          contribution.phi2[b] = coefficient * phi2[b];
        }
        constraints[row].contributions.push_back( contribution );
        constraints[row].gap += coefficient * gap;
        const RealT tributary_measure = quadrature_measure * std::max( 0., lor_phi[a] );
        constraints[row].tributary_area += tributary_measure;
        constraints[row].weighted_penalty_stiffness += tributary_measure * penalty_stiffness;
        if ( tributary_measure > 0. ) {
          constraints[row].minimum_thickness =
              std::min( constraints[row].minimum_thickness, local_thickness );
        }
        plane.m_inContact = true;
      }
    }
  }

  constraints.erase( std::remove_if( constraints.begin(), constraints.end(), []( const auto& constraint ) {
                       return constraint.contributions.empty();
                     } ),
                     constraints.end() );
  if ( constraints.empty() ) {
    cs->setProjectionDiagnostics( 0, 0, 0, true, true, 0., 0., 0., 0., 1., 1., 0., 0., 0., 0. );
    cs->setCompliantProjectionDiagnostics( 0., 0., 0., 0, 0., 0. );
    cs->setPredictorForceDiagnostics( 0, 0, 0., 0., 0. );
    cs->setContactPointDiagnostics( 0, 0., 0., 0., 0., 0. );
    return 0;
  }

  const int num_constraints = static_cast<int>( constraints.size() );
  mfem::DenseMatrix mass_scaled_rows;
  for ( int i = 0; i < num_constraints; ++i ) {
    auto& constraint = constraints[i];
    constraint.trial_velocity = EvaluateProjectionVelocity( constraint, mesh1, mesh2 );
    const RealT base_velocity = position_velocity_scale < 1.
                                    ? EvaluateProjectionBaseVelocity( constraint, mesh1, mesh2 )
                                    : constraint.trial_velocity;
    constraint.position_trial_velocity = ( 1. - position_velocity_scale ) * base_velocity +
                                         position_velocity_scale * constraint.trial_velocity;
    const RealT target_position_velocity = constraint.gap > 0. ? -constraint.gap / stage_dt : 0.;
    constraint.target_velocity =
        constraint.trial_velocity +
        ( target_position_velocity - constraint.position_trial_velocity ) / position_velocity_scale;
    AccumulateProjectionImpulse( constraint, 1., mesh1, mesh2 );
    mfem::Vector row;
    if ( !mfem_data->AssembleParentQ2MassScaledResponseRow( row ) ) {
      mfem_data->ResetParentQ2Projection();
      return 1;
    }
    if ( i == 0 ) {
      mass_scaled_rows.SetSize( row.Size(), num_constraints );
      mass_scaled_rows = 0.;
    }
    for ( int d = 0; d < row.Size(); ++d ) {
      mass_scaled_rows( d, i ) = row[d];
    }
  }

  mfem::DenseMatrix projection_operator( num_constraints );
  mfem::MultAtB( mass_scaled_rows, mass_scaled_rows, projection_operator );
  mfem::DenseMatrix solve_operator( projection_operator );
  const bool use_compliant_response =
      projection_options.contact_response == PROJECTION_RESPONSE_COMPLIANT;
  bool valid = true;
  for ( int i = 0; i < num_constraints; ++i ) {
    auto& constraint = constraints[i];
    const RealT physical_diagonal = projection_operator( i, i );
    const bool valid_physical_diagonal = std::isfinite( physical_diagonal ) && physical_diagonal > 0.;
    valid = valid && valid_physical_diagonal && std::isfinite( constraint.gap ) &&
            std::isfinite( constraint.target_velocity ) && std::isfinite( constraint.trial_velocity );
    if ( use_compliant_response ) {
      const bool valid_contact_data = std::isfinite( constraint.tributary_area ) &&
                                      constraint.tributary_area > 0. &&
                                      std::isfinite( constraint.weighted_penalty_stiffness ) &&
                                      constraint.weighted_penalty_stiffness > 0. &&
                                      std::isfinite( constraint.minimum_thickness ) &&
                                      constraint.minimum_thickness > 0.;
      valid = valid && valid_contact_data;
      if ( valid_physical_diagonal && valid_contact_data ) {
        constraint.spring_stiffness = constraint.weighted_penalty_stiffness;
        const RealT effective_mass = 1. / physical_diagonal;
        constraint.damping_coefficient =
            2. * projection_options.damping_ratio *
            std::sqrt( constraint.spring_stiffness * effective_mass );
        const RealT effective_damping = constraint.damping_coefficient +
                                        constraint.spring_stiffness * position_velocity_scale * stage_dt;
        valid = valid && std::isfinite( effective_damping ) && effective_damping > 0.;
        if ( std::isfinite( effective_damping ) && effective_damping > 0. ) {
          solve_operator( i, i ) += 1. / ( stage_dt * effective_damping );
          const RealT trial_endpoint_gap =
              constraint.gap + stage_dt * constraint.position_trial_velocity;
          const RealT trial_penetration = std::max( 0., -trial_endpoint_gap );
          const bool contact_predicted = constraint.gap < 0. || trial_endpoint_gap < 0.;
          const RealT free_contact_residual =
              contact_predicted
                  ? ( constraint.damping_coefficient * constraint.trial_velocity -
                      constraint.spring_stiffness * trial_penetration ) /
                        effective_damping
                  : 0.;
          constraint.target_velocity = constraint.trial_velocity - free_contact_residual;
        }
      }
    }
    constraint.diagonal = solve_operator( i, i );
    valid = valid && std::isfinite( constraint.diagonal ) && constraint.diagonal > 0.;
  }

  RealT coupling_bound = 1.;
  for ( int i = 0; i < num_constraints; ++i ) {
    RealT row_sum = 0.;
    for ( int j = 0; j < num_constraints; ++j ) {
      row_sum += std::abs( solve_operator( i, j ) );
    }
    if ( constraints[i].diagonal > 0. ) {
      coupling_bound = std::max( coupling_bound, row_sum / constraints[i].diagonal );
    }
  }
  const bool operator_valid = valid;
  const RealT jacobi_relaxation = std::min( 1., projection_options.relaxation_scale / coupling_bound );
  const RealT relaxation = projection_options.relaxation_scale;
  valid = valid && std::isfinite( relaxation ) && relaxation > 0.;
  TraceProjectionSolveResult compliant_result;
  if ( valid ) {
    compliant_result = SolveTraceProjectionSystem( solve_operator, constraints, projection_options );
    valid = valid && compliant_result.valid;
  } else {
    compliant_result.valid = false;
    compliant_result.complementarity_converged = false;
  }

  std::vector<RealT> original_trial_velocity( constraints.size() );
  std::vector<RealT> original_position_trial_velocity( constraints.size() );
  for ( int i = 0; i < num_constraints; ++i ) {
    auto& constraint = constraints[i];
    original_trial_velocity[i] = constraint.trial_velocity;
    original_position_trial_velocity[i] = constraint.position_trial_velocity;
    constraint.compliant_multiplier = use_compliant_response ? constraint.multiplier : 0.;
    constraint.guard_multiplier = use_compliant_response ? 0. : constraint.multiplier;
  }

  TraceProjectionSolveResult guard_result;
  bool guard_required = false;
  if ( valid && use_compliant_response ) {
    for ( int i = 0; i < num_constraints; ++i ) {
      auto& constraint = constraints[i];
      RealT compliant_velocity = original_trial_velocity[i];
      for ( int j = 0; j < num_constraints; ++j ) {
        compliant_velocity += projection_operator( i, j ) * constraints[j].compliant_multiplier;
      }
      const RealT compliant_position_velocity =
          original_position_trial_velocity[i] +
          position_velocity_scale * ( compliant_velocity - original_trial_velocity[i] );
      const RealT compliant_endpoint_gap = constraint.gap + stage_dt * compliant_position_velocity;
      const RealT allowed_penetration =
          projection_options.max_penetration_fraction * constraint.minimum_thickness;
      const RealT minimum_endpoint_gap =
          constraint.gap < -allowed_penetration ? constraint.gap : -allowed_penetration;
      const RealT guard_tolerance = 100. * std::numeric_limits<RealT>::epsilon() *
                                    std::max( 1., std::abs( minimum_endpoint_gap ) );
      const bool constraint_requires_guard =
          compliant_endpoint_gap < minimum_endpoint_gap - guard_tolerance;
      guard_required = guard_required || constraint_requires_guard;
      constraint.trial_velocity = compliant_velocity;
      constraint.position_trial_velocity = compliant_position_velocity;
      const RealT target_position_velocity =
          constraint_requires_guard ? ( minimum_endpoint_gap - constraint.gap ) / stage_dt
                                    : compliant_position_velocity;
      constraint.target_velocity =
          compliant_velocity +
          ( target_position_velocity - compliant_position_velocity ) / position_velocity_scale;
      constraint.multiplier = 0.;
      constraint.diagonal = projection_operator( i, i );
    }
    if ( guard_required ) {
      guard_result = SolveTraceProjectionSystem( projection_operator, constraints, projection_options );
      valid = valid && guard_result.valid;
    }
    for ( int i = 0; i < num_constraints; ++i ) {
      auto& constraint = constraints[i];
      constraint.guard_multiplier = guard_required ? constraint.multiplier : 0.;
      constraint.multiplier = constraint.compliant_multiplier + constraint.guard_multiplier;
      constraint.trial_velocity = original_trial_velocity[i];
      constraint.position_trial_velocity = original_position_trial_velocity[i];
    }
  }

  const RealT initial_residual =
      std::max( compliant_result.initial_residual, guard_result.initial_residual );
  const RealT final_residual =
      std::max( compliant_result.final_residual, guard_result.final_residual );
  const RealT final_primal_residual =
      std::max( compliant_result.final_primal_residual, guard_result.final_primal_residual );
  const RealT primal_tolerance =
      std::max( compliant_result.primal_tolerance, guard_result.primal_tolerance );
  const bool complementarity_converged = compliant_result.complementarity_converged &&
                                         ( !guard_required || guard_result.complementarity_converged );
  const int iterations = compliant_result.iterations + guard_result.iterations;

  IndexT active_multipliers = 0;
  RealT total_impulse = 0.;
  RealT spring_force = 0.;
  RealT damping_force = 0.;
  RealT guard_force = 0.;
  IndexT guard_constraints = 0;
  RealT stored_energy = 0.;
  RealT maximum_penetration_fraction = 0.;
  RealT maximum_force = 0.;
  RealT gap_violation_sum = 0.;
  RealT maximum_gap_violation = 0.;
  RealT closing_rate_sum = 0.;
  RealT maximum_closing_rate = 0.;
  RealT maximum_endpoint_violation = 0.;
  RealT energy_change = 0.;
  RealT spring_energy_change = 0.;
  RealT energy_scale = 0.;
  std::vector<RealT> plane_impulses( static_cast<std::size_t>( cs->getNumActivePairs() ), 0. );
  for ( int i = 0; i < num_constraints; ++i ) {
    const auto& constraint = constraints[i];
    RealT final_velocity = constraint.trial_velocity;
    for ( int j = 0; j < num_constraints; ++j ) {
      final_velocity += projection_operator( i, j ) * constraints[j].multiplier;
    }
    const RealT position_final_velocity =
        constraint.position_trial_velocity +
        position_velocity_scale * ( final_velocity - constraint.trial_velocity );
    const RealT endpoint_gap = constraint.gap + stage_dt * position_final_velocity;
    const RealT gap_violation = std::max( 0., -constraint.gap );
    const RealT closing_rate = std::max( 0., -constraint.trial_velocity );
    const RealT energy_term = 0.5 * constraint.multiplier * ( constraint.trial_velocity + final_velocity );
    const RealT initial_penetration = std::max( 0., -constraint.gap );
    const RealT final_penetration = std::max( 0., -endpoint_gap );
    const RealT spring_energy_term =
        use_compliant_response
            ? 0.5 * constraint.spring_stiffness *
                  ( final_penetration * final_penetration - initial_penetration * initial_penetration )
            : 0.;
    if ( constraint.multiplier > 0. ) {
      ++active_multipliers;
    }
    total_impulse += constraint.multiplier;
    spring_force += use_compliant_response ? constraint.spring_stiffness * final_penetration : 0.;
    damping_force += use_compliant_response
                         ? std::max( 0., -constraint.damping_coefficient * final_velocity )
                         : 0.;
    guard_force += constraint.guard_multiplier / stage_dt;
    guard_constraints += constraint.guard_multiplier > 0. ? 1 : 0;
    stored_energy += use_compliant_response
                         ? 0.5 * constraint.spring_stiffness * final_penetration * final_penetration
                         : 0.;
    maximum_penetration_fraction =
        use_compliant_response && constraint.minimum_thickness > 0.
            ? std::max( maximum_penetration_fraction,
                        final_penetration / constraint.minimum_thickness )
            : maximum_penetration_fraction;
    maximum_force = std::max( maximum_force, constraint.multiplier / stage_dt );
    gap_violation_sum += gap_violation;
    maximum_gap_violation = std::max( maximum_gap_violation, gap_violation );
    closing_rate_sum += closing_rate;
    maximum_closing_rate = std::max( maximum_closing_rate, closing_rate );
    maximum_endpoint_violation = std::max( maximum_endpoint_violation, std::max( 0., -endpoint_gap ) );
    energy_change += energy_term;
    spring_energy_change += spring_energy_term;
    energy_scale += std::abs( energy_term ) + std::abs( spring_energy_term );
    for ( const auto& contribution : constraint.contributions ) {
      plane_impulses[contribution.plane_index] += constraint.multiplier * contribution.plane_weight;
    }
  }
  const RealT energy_tolerance = 100. * std::numeric_limits<RealT>::epsilon() * std::max( 1., energy_scale );
  const bool primal_converged = final_primal_residual <= primal_tolerance;
  const RealT validated_energy_change = energy_change + spring_energy_change;
  const bool accepted = valid && primal_converged && std::isfinite( validated_energy_change ) &&
                        validated_energy_change <= energy_tolerance;

  if ( operator_valid ) {
    const ProjectionOperatorDiagnostics operator_diagnostics =
        ComputeProjectionOperatorDiagnostics( mass_scaled_rows, projection_operator, jacobi_relaxation );
    cs->setProjectionOperatorDiagnostics(
        operator_diagnostics.velocity_dofs, operator_diagnostics.rank, operator_diagnostics.minimum_eigenvalue,
        operator_diagnostics.maximum_eigenvalue, operator_diagnostics.condition_estimate,
        operator_diagnostics.jacobi_contraction );
  }

  if ( accepted ) {
    for ( const auto& constraint : constraints ) {
      AccumulateProjectionImpulse( constraint, constraint.multiplier, mesh1, mesh2 );
    }
    valid = mfem_data->ApplyParentQ2ProjectionImpulse();
    if ( valid ) {
      for ( const auto& constraint : constraints ) {
        AccumulateProjectionImpulse( constraint, constraint.multiplier / stage_dt, mesh1, mesh2 );
      }
    } else {
      mfem_data->ResetParentQ2Projection();
    }
  } else {
    mfem_data->ResetParentQ2Projection();
  }

  const bool final_accepted = accepted && valid;
  for ( IndexT i = 0; i < cs->getNumActivePairs(); ++i ) {
    auto& plane = cs_view.getCompGeomView().getCommonPlane( i );
    const RealT impulse = final_accepted ? std::max( 0., plane_impulses[i] ) : 0.;
    plane.m_pressure = plane.m_area > 0. ? -impulse / ( plane.m_area * stage_dt ) : 0.;
    plane.m_inContact = impulse > 0.;
  }

  const RealT equivalent_force = total_impulse / stage_dt;
  cs->setProjectionDiagnostics( static_cast<IndexT>( constraints.size() ), active_multipliers, iterations,
                                final_accepted, complementarity_converged, initial_residual, final_residual,
                                final_primal_residual, primal_tolerance, coupling_bound, relaxation, total_impulse,
                                equivalent_force, maximum_endpoint_violation, validated_energy_change );
  cs->setCompliantProjectionDiagnostics(
      final_accepted && use_compliant_response ? spring_force : 0.,
      final_accepted && use_compliant_response ? damping_force : 0.,
      final_accepted && use_compliant_response ? guard_force : 0.,
      final_accepted && use_compliant_response ? guard_constraints : 0,
      final_accepted && use_compliant_response ? stored_energy : 0.,
      final_accepted && use_compliant_response ? maximum_penetration_fraction : 0. );
  cs->setPredictorForceDiagnostics( 0, 0, final_accepted ? spring_force : 0.,
                                    final_accepted && use_compliant_response ? damping_force + guard_force : 0.,
                                    final_accepted ? equivalent_force : 0. );
  cs->setContactPointDiagnostics( static_cast<IndexT>( constraints.size() ), final_accepted ? maximum_force : 0.,
                                  gap_violation_sum, maximum_gap_violation, closing_rate_sum, maximum_closing_rate );
  return final_accepted ? 0 : 1;
#endif
}

//------------------------------------------------------------------------------
template <>
int ApplyTangential<COMMON_PLANE, PENALTY, VISCOUS_TANGENTIAL>( CouplingScheme* cs )
{
  ///////////////////////////////
  // loop over interface pairs //
  ///////////////////////////////
  auto cs_view = cs->getView();
  const auto num_pairs = cs->getNumActivePairs();
  forAllExec( cs->getExecutionMode(), num_pairs, [cs_view] TRIBOL_HOST_DEVICE( IndexT i ) {
    auto& cg_view = cs_view.getCompGeomView();
    auto& plane = cg_view.getCommonPlane( i );

    if ( !plane.m_inContact ) {
      return;
    }

    const auto dim = plane.m_dim;
    auto& mesh1 = cs_view.getMesh1View();
    auto& mesh2 = cs_view.getMesh2View();
    const PenaltyEnforcementOptions& pen_enfrc_options = cs_view.getEnforcementOptions().penalty_options;

    // get pair indices
    IndexT index1 = plane.getCpElementId1();
    IndexT index2 = plane.getCpElementId2();

    const bool use_multi_point = pen_enfrc_options.common_plane_rule == MULTI_POINT;

    // compute the velocity gap and pressure contribution
    StackArrayT<RealT, max_dim * max_nodes_per_face> x1;
    StackArrayT<RealT, max_dim * max_nodes_per_face> v1;
    auto numNodesPerFace1 = mesh1.numberOfNodesPerElement();
    plane.getFace1Coords( x1, numNodesPerFace1 );  // get avg face coords off the contact plane
    mesh1.getFaceVelocities( index1, v1 );

    StackArrayT<RealT, max_dim * max_nodes_per_face> x2;
    StackArrayT<RealT, max_dim * max_nodes_per_face> v2;
    auto numNodesPerFace2 = mesh2.numberOfNodesPerElement();
    plane.getFace2Coords( x2, numNodesPerFace2 );  // get avg face coords off the contact plane
    mesh2.getFaceVelocities( index2, v2 );

    //////////////////////////////////////////////////////////
    // compute velocity Galerkin approximation at projected //
    // overlap centroid                                     //
    //////////////////////////////////////////////////////////
    RealT vel_f1[max_dim];
    RealT vel_f2[max_dim];
    initRealArray( vel_f1, dim, 0. );
    initRealArray( vel_f2, dim, 0. );

    // interpolate nodal velocity at overlap centroid as projected
    // onto face 1
    RealT cXf1 = plane.m_cXf1;
    RealT cYf1 = plane.m_cYf1;
    RealT cZf1 = ( dim == 3 ) ? plane.m_cZf1 : 0.;
    GalerkinEvalOnPhysicalFace( x1, cXf1, cYf1, cZf1, numNodesPerFace1, dim, v1, vel_f1 );

    // interpolate nodal velocity at overlap centroid as projected
    // onto face 2
    RealT cXf2 = plane.m_cXf2;
    RealT cYf2 = plane.m_cYf2;
    RealT cZf2 = ( dim == 3 ) ? plane.m_cZf2 : 0.;
    GalerkinEvalOnPhysicalFace( x2, cXf2, cYf2, cZf2, numNodesPerFace2, dim, v2, vel_f2 );

    // compute velocity gap vector
    RealT velGap[max_dim];
    velGap[0] = vel_f1[0] - vel_f2[0];
    velGap[1] = vel_f1[1] - vel_f2[1];
    if ( dim == 3 ) {
      velGap[2] = vel_f1[2] - vel_f2[2];
    }

    // subtract off the common-plane normal component of the velocity gap
    RealT velGap_dot_n = velGap[0] * plane.m_nX + velGap[1] * plane.m_nY;
    if ( dim == 3 ) {
      velGap_dot_n += velGap[2] * plane.m_nZ;
    }
    RealT velGapTan[max_dim];
    velGapTan[0] = velGap[0] - velGap_dot_n * plane.m_nX;
    velGapTan[1] = velGap[1] - velGap_dot_n * plane.m_nY;
    if ( dim == 3 ) {
      velGapTan[2] = velGap[2] - velGap_dot_n * plane.m_nZ;
    }

    // setup the contact element struct for purposes of evaluating basis functions on overlap
    // initialize assuming 2d
    RealT xVert[max_dim * max_nodes_per_overlap];
    auto xVert_size = 4;
    auto numPolyVert = 2;
    // update if we are in 3d
    if ( dim == 3 ) {
      numPolyVert = plane.m_numPolyVert;
      xVert_size = 3 * numPolyVert;
    }
    initRealArray( xVert, xVert_size, 0. );

    // construct array of polygon overlap vertex coordinates
    plane.getOverlapVertices( &xVert[0] );

    // instantiate surface contact element struct. Note, this is done with current
    // configuration face coordinates (i.e. NOT on the contact plane) and overlap
    // coordinates ON the contact plane. The surface contact element does not need
    // to be used this way, but the developer should do the book-keeping.
    SurfaceContactElem cntctElem( dim, x1, x2, xVert, numNodesPerFace1, numPolyVert, &mesh1, &mesh2, index1, index2 );

    // set SurfaceContactElem face normals and overlap normal
    RealT faceNormal1[max_dim];
    RealT faceNormal2[max_dim];
    RealT overlapNormal[max_dim];

    mesh1.getFaceNormal( index1, faceNormal1 );
    mesh2.getFaceNormal( index2, faceNormal2 );
    overlapNormal[0] = plane.m_nX;
    overlapNormal[1] = plane.m_nY;
    if ( dim == 3 ) {
      overlapNormal[2] = plane.m_nZ;
    }

    cntctElem.faceNormal1 = faceNormal1;
    cntctElem.faceNormal2 = faceNormal2;
    cntctElem.overlapNormal = overlapNormal;
    cntctElem.overlapArea = plane.m_area;

    if ( use_multi_point ) {
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_xf2;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf1;
      StackArrayT<RealT, max_dim * max_nodes_per_face> actual_vf2;
      mesh1.getFaceCoords( index1, actual_xf1 );
      mesh2.getFaceCoords( index2, actual_xf2 );
      mesh1.getFaceVelocities( index1, actual_vf1 );
      mesh2.getFaceVelocities( index2, actual_vf2 );

      constexpr int max_qpts = max_symmetric_triangle_qpts;
      RealT rule_wts[max_qpts] = { 0. };
      RealT rule_coords[2 * max_qpts] = { 0. };
      const RealT visc =
          0.5 * ( mesh1.getElementData().m_viscous_damping_coeff + mesh2.getElementData().m_viscous_damping_coeff );

      if ( dim == 3 ) {
        const int num_qpts =
            GetCommonPlaneTriangleRule( pen_enfrc_options.common_plane_quadrature_order, rule_wts, rule_coords );

        RealT centroid[3];
        GetCommonPlaneOverlapCentroid( cntctElem, centroid );

        RealT xTri[3];
        RealT yTri[3];
        RealT zTri[3];

        for ( int j = 0; j < numPolyVert; ++j ) {
          const int next = ( j == numPolyVert - 1 ) ? 0 : j + 1;
          xTri[0] = xVert[dim * j];
          yTri[0] = xVert[dim * j + 1];
          zTri[0] = xVert[dim * j + 2];
          xTri[1] = xVert[dim * next];
          yTri[1] = xVert[dim * next + 1];
          zTri[1] = xVert[dim * next + 2];
          xTri[2] = centroid[0];
          yTri[2] = centroid[1];
          zTri[2] = centroid[2];

          const RealT area = Area3DTri( xTri, yTri, zTri );
          if ( area <= 0. ) {
            continue;
          }

          for ( int qp = 0; qp < num_qpts; ++qp ) {
            const RealT xi = rule_coords[2 * qp];
            const RealT eta = rule_coords[2 * qp + 1];
            const RealT n0 = 1. - xi - eta;
            RealT x_q[3];
            x_q[0] = n0 * xTri[0] + xi * xTri[1] + eta * xTri[2];
            x_q[1] = n0 * yTri[0] + xi * yTri[1] + eta * yTri[2];
            x_q[2] = n0 * zTri[0] + xi * zTri[1] + eta * zTri[2];

            RealT phi_q1[max_nodes_per_face] = { 0., 0., 0., 0. };
            RealT phi_q2[max_nodes_per_face] = { 0., 0., 0., 0. };
            RealT x_qf1[max_dim];
            RealT x_qf2[max_dim];
            RealT vel_q1[max_dim];
            RealT vel_q2[max_dim];

            const bool mapped_face1 = EvalLinearFaceAtProjectedPoint( actual_xf1, numNodesPerFace1, x_q, overlapNormal,
                                                                      x_qf1, phi_q1, dim, &actual_vf1[0], vel_q1 );
            const bool mapped_face2 = EvalLinearFaceAtProjectedPoint( actual_xf2, numNodesPerFace2, x_q, overlapNormal,
                                                                      x_qf2, phi_q2, dim, &actual_vf2[0], vel_q2 );
            if ( !mapped_face1 || !mapped_face2 ) {
              continue;
            }

            RealT local_gap = ( x_qf1[0] - x_qf2[0] ) * overlapNormal[0] + ( x_qf1[1] - x_qf2[1] ) * overlapNormal[1] +
                              ( x_qf1[2] - x_qf2[2] ) * overlapNormal[2];
            RealT gap_tol = cs_view.getGapTol( index1, index2 );
            if ( local_gap > gap_tol ) {
              continue;
            }

            RealT velGap_q[max_dim];
            velGap_q[0] = vel_q1[0] - vel_q2[0];
            velGap_q[1] = vel_q1[1] - vel_q2[1];
            velGap_q[2] = vel_q1[2] - vel_q2[2];

            RealT velGap_dot_n =
                velGap_q[0] * overlapNormal[0] + velGap_q[1] * overlapNormal[1] + velGap_q[2] * overlapNormal[2];
            RealT velGapTan[max_dim];
            velGapTan[0] = velGap_q[0] - velGap_dot_n * overlapNormal[0];
            velGapTan[1] = velGap_q[1] - velGap_dot_n * overlapNormal[1];
            velGapTan[2] = velGap_q[2] - velGap_dot_n * overlapNormal[2];

            const RealT weighted_force = area * rule_wts[qp] * visc;
            const RealT force_x = weighted_force * velGapTan[0];
            const RealT force_y = weighted_force * velGapTan[1];
            const RealT force_z = weighted_force * velGapTan[2];

            AccumulateContactForce( mesh1, mesh2, index1, index2, dim, numNodesPerFace1, force_x, force_y, force_z,
                                    phi_q1, phi_q2 );
          }
        }
      } else {
        RealT segment_rule_wts[max_segment_gauss_legendre_qpts] = { 0. };
        RealT segment_rule_coords[max_segment_gauss_legendre_qpts] = { 0. };
        const int num_qpts = GetCommonPlaneSegmentRule( pen_enfrc_options.common_plane_quadrature_order,
                                                        segment_rule_wts, segment_rule_coords );
        const RealT x0 = xVert[0];
        const RealT y0 = xVert[1];
        const RealT x1_seg = xVert[2];
        const RealT y1_seg = xVert[3];
        const RealT length = magnitude( x1_seg - x0, y1_seg - y0 );

        for ( int qp = 0; qp < num_qpts; ++qp ) {
          const RealT s = segment_rule_coords[qp];
          const RealT one_minus_s = 1. - s;
          RealT x_q[2] = { one_minus_s * x0 + s * x1_seg, one_minus_s * y0 + s * y1_seg };

          RealT phi_q1[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT phi_q2[max_nodes_per_face] = { 0., 0., 0., 0. };
          RealT x_qf1[max_dim];
          RealT x_qf2[max_dim];
          RealT vel_q1[max_dim] = { 0., 0., 0. };
          RealT vel_q2[max_dim] = { 0., 0., 0. };

          const bool mapped_face1 = EvalLinearEdgeAtProjectedPoint( actual_xf1, x_q, overlapNormal, x_qf1, phi_q1, dim,
                                                                    &actual_vf1[0], vel_q1 );
          const bool mapped_face2 = EvalLinearEdgeAtProjectedPoint( actual_xf2, x_q, overlapNormal, x_qf2, phi_q2, dim,
                                                                    &actual_vf2[0], vel_q2 );
          if ( !mapped_face1 || !mapped_face2 ) {
            continue;
          }

          RealT local_gap = ( x_qf1[0] - x_qf2[0] ) * overlapNormal[0] + ( x_qf1[1] - x_qf2[1] ) * overlapNormal[1];
          RealT gap_tol = cs_view.getGapTol( index1, index2 );
          if ( local_gap > gap_tol ) {
            continue;
          }

          RealT velGap_q[max_dim];
          velGap_q[0] = vel_q1[0] - vel_q2[0];
          velGap_q[1] = vel_q1[1] - vel_q2[1];

          RealT velGap_dot_n = velGap_q[0] * overlapNormal[0] + velGap_q[1] * overlapNormal[1];
          RealT velGapTan_q[max_dim] = { velGap_q[0] - velGap_dot_n * overlapNormal[0],
                                         velGap_q[1] - velGap_dot_n * overlapNormal[1], 0. };

          const RealT weighted_force = length * segment_rule_wts[qp] * visc;
          const RealT force_x = weighted_force * velGapTan_q[0];
          const RealT force_y = weighted_force * velGapTan_q[1];

          AccumulateContactForce( mesh1, mesh2, index1, index2, dim, numNodesPerFace1, force_x, force_y, 0., phi_q1,
                                  phi_q2 );
        }
      }

      return;
    }

    // create arrays to hold nodal residual weak form integral evaluations
    RealT phi1[max_nodes_per_face];
    RealT phi2[max_nodes_per_face];
    initRealArray( phi1, numNodesPerFace1, 0. );
    initRealArray( phi2, numNodesPerFace2, 0. );

    ////////////////////////////////////////////////////////////////////////
    // Integration of contact integrals: integral of shape functions over //
    // contact overlap patch                                              //
    ////////////////////////////////////////////////////////////////////////
    EvalWeakFormIntegralCommonPlane( cntctElem, pen_enfrc_options.common_plane_rule,
                                     pen_enfrc_options.common_plane_quadrature_order, phi1, phi2 );

    /////////////////////////////////////////////////////
    // Computation of tangential viscous damping force //
    /////////////////////////////////////////////////////
    RealT visc =
        0.5 * ( mesh1.getElementData().m_viscous_damping_coeff + mesh2.getElementData().m_viscous_damping_coeff );
    RealT force_x = visc * velGapTan[0];
    RealT force_y = visc * velGapTan[1];
    RealT force_z = 0.;
    if ( dim == 3 ) {
      force_z = visc * velGapTan[2];
    }

    AccumulateContactForce( mesh1, mesh2, index1, index2, dim, numNodesPerFace1, force_x, force_y, force_z, phi1,
                            phi2 );
  } );

  return 0;
}
//------------------------------------------------------------------------------

}  // namespace tribol
