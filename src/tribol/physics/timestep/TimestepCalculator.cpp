// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/timestep/TimestepCalculator.hpp"

#ifdef BUILD_REDECOMP

#include <cstring>
#include <limits>

#include "tribol/mesh/CouplingScheme.hpp"

namespace tribol::detail {

MfemParentPenaltyCflData prepareMfemParentPenaltyCfl( CouplingScheme& scheme )
{
  auto& mesh_data = *scheme.getMfemMeshData();
  if ( !scheme.hasMfemJacobianData() ) {
    if ( mesh_data.GetLORMesh() ) {
      mesh_data.EnableLORJacobianTransfer();
    }
    scheme.setMfemJacobianData( std::make_unique<MfemJacobianData>( mesh_data ) );
  }
  auto& transfer_data = *scheme.getMfemJacobianData();
  if ( !transfer_data.HasParentPath() ) {
    transfer_data.UpdateJacobianXfer();
  }
  return { mesh_data, transfer_data };
}

std::optional<RealT> computeParentPenaltyRowSumBound( const mfem::HypreParMatrix& penalty_matrix,
                                                      const MfemMeshData& mesh_data )
{
  const auto* inverse_mass = mesh_data.GetParentInverseMass();
  const auto& parent_space = *mesh_data.GetParentCoords().ParFESpace();
  const auto* mass_space = inverse_mass ? inverse_mass->ParFESpace() : nullptr;
  const bool compatible_space =
      mass_space && mass_space->GetParMesh() == parent_space.GetParMesh() &&
      mass_space->GetVDim() == parent_space.GetVDim() && mass_space->GetOrdering() == parent_space.GetOrdering() &&
      mass_space->GetTrueVSize() == penalty_matrix.Width() && penalty_matrix.Width() == penalty_matrix.Height() &&
      std::strcmp( mass_space->FEColl()->Name(), parent_space.FEColl()->Name() ) == 0;

  mfem::Vector sqrt_inverse_mass;
  int invalid_mass = compatible_space ? 0 : 1;
  if ( compatible_space ) {
    // Read the registered values in true-DOF order, including zeros for fixed DOFs.
    inverse_mass->GetTrueDofs( sqrt_inverse_mass );
    auto* values = sqrt_inverse_mass.HostReadWrite();
    for ( int dof = 0; dof < sqrt_inverse_mass.Size(); ++dof ) {
      if ( !isValidInverseMass( values[dof] ) ) {
        invalid_mass = 1;
        break;
      }
      values[dof] = std::sqrt( values[dof] );
    }
  }
  MPI_Allreduce( MPI_IN_PLACE, &invalid_mass, 1, MPI_INT, MPI_MAX, penalty_matrix.GetComm() );
  if ( invalid_mass ) {
    SLIC_ERROR(
        "nodal_penalty_cfl requires finite, nonnegative inverse diagonal masses on the parent coordinate "
        "finite-element space. Correct the field registered via registerMfemInverseMass(), or set "
        "nodal_penalty_cfl = false via setTimestepCriteria()." );
    return std::nullopt;
  }

  // The signed matrix is already assembled. AbsMult sums |K_ij| * sqrt(inverse_mass_j), including off-rank columns.
  mfem::Vector row_sums( penalty_matrix.Height() );
  penalty_matrix.AbsMult( 1.0, sqrt_inverse_mass, 0.0, row_sums );
  const auto* sums = row_sums.HostRead();
  const auto* mass_scales = sqrt_inverse_mass.HostRead();
  RealT bound = 0.0;
  for ( int dof = 0; dof < row_sums.Size(); ++dof ) {
    const RealT row_sum = mass_scales[dof] * sums[dof];
    if ( !std::isfinite( row_sum ) ) {
      bound = std::numeric_limits<RealT>::infinity();
      break;
    }
    bound = std::max( bound, row_sum );
  }
  MPI_Allreduce( MPI_IN_PLACE, &bound, 1, MPI_DOUBLE, MPI_MAX, penalty_matrix.GetComm() );
  if ( !std::isfinite( bound ) ) {
    SLIC_ERROR(
        "nodal_penalty_cfl produced a nonfinite mass-scaled row sum on parent true DOFs. Check the penalty "
        "coefficients and inverse masses, or set nodal_penalty_cfl = false via setTimestepCriteria()." );
    return std::nullopt;
  }
  return bound;
}

}  // namespace tribol::detail

#endif  // BUILD_REDECOMP
