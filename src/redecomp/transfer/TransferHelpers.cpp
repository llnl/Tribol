// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "TransferHelpers.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "axom/slic.hpp"

namespace redecomp {
namespace transfer_helpers {

mfem::Array<int> ExpandScalarDofsToVDofs( const mfem::FiniteElementSpace& finite_element_space,
                                          const mfem::Array<int>& scalar_dofs )
{
  mfem::Array<int> vector_dofs;
  vector_dofs.Reserve( finite_element_space.GetVDim() * scalar_dofs.Size() );
  for ( auto scalar_dof : scalar_dofs ) {
    for ( int component{ 0 }; component < finite_element_space.GetVDim(); ++component ) {
      vector_dofs.Append( finite_element_space.DofToVDof( scalar_dof, component ) );
    }
  }
  return vector_dofs;
}

mfem::Vector GatherValuesFromVDofs( const mfem::Vector& source, const mfem::Array<int>& source_vdofs )
{
  mfem::Vector gathered_values;
  gathered_values.UseDevice( source.UseDevice() );
  if ( source_vdofs.Size() > 0 ) {
    source_vdofs.GetMemory().UseDevice( source.UseDevice() );
    gathered_values.SetSize( source_vdofs.Size() );
    source.GetSubVector( source_vdofs, gathered_values );
  }
  return gathered_values;
}

void AssignValuesToVDofs( const mfem::Vector& received_values, const mfem::Array<int>& destination_vdofs,
                          mfem::Vector& destination )
{
  if ( destination_vdofs.Size() == 0 ) {
    return;
  }
  destination_vdofs.GetMemory().UseDevice( destination.UseDevice() );
  received_values.Read( destination.UseDevice() );
  received_values.UseDevice( destination.UseDevice() );
  destination.SetSubVector( destination_vdofs, received_values );
}

void KeepOnlyOwnedParentVDofs( mfem::Vector& parent_values, const mfem::ParFiniteElementSpace& parent_fes )
{
  auto parent_data = parent_values.ReadWrite( parent_values.UseDevice() );
  auto parent_dof_offsets = mfem::Read( parent_fes.Dof_TrueDof_Matrix()->GetDiagMemoryI(), parent_fes.GetVSize() + 1,
                                       parent_values.UseDevice() );
  mfem::forall_switch( parent_values.UseDevice(), parent_fes.GetVSize(), [=] MFEM_HOST_DEVICE( int parent_vdof ) {
    if ( parent_dof_offsets[parent_vdof + 1] == parent_dof_offsets[parent_vdof] ) {
      parent_data[parent_vdof] = 0.0;
    }
  } );
}

void VerifyReplicatedValueMatchesSelected( const char* field_name, int parent_vdof, int selected_rank,
                                           double selected_value, int candidate_rank, double candidate_value )
{
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
                           std::max( { 1.0, std::abs( selected_value ), std::abs( candidate_value ) } );
  SLIC_ERROR_IF( std::abs( selected_value - candidate_value ) > tolerance,
                 axom::fmt::format( "Conflicting {} copies for parent local VDof {}: selected redecomp rank {} "
                                    "has value {}, while redecomp rank {} has value {}.",
                                    field_name, parent_vdof, selected_rank, selected_value, candidate_rank,
                                    candidate_value ) );
}

}  // namespace transfer_helpers
}  // namespace redecomp
