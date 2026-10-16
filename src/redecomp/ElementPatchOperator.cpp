// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/ElementPatchOperator.hpp"

#include <utility>

#include "redecomp/OperatorAssembly.hpp"

#include "axom/slic.hpp"

namespace redecomp {

ElementPatchOperator::ElementPatchOperator( const mfem::ParFiniteElementSpace& source_space,
                                            mfem::Array<int> selected_elements,
                                            std::vector<mfem::Array<int>> selected_elements_by_destination,
                                            std::vector<VirtualFaceSpecification> destination_faces,
                                            mfem::Array<int> occurrence_to_node, int num_patch_nodes )
    : mfem::Operator(),
      source_space_{ source_space },
      restriction_{ source_space, std::move( selected_elements ), std::move( selected_elements_by_destination ) },
      refinement_{ restriction_.Plan().DestinationElementOffsets(), std::move( destination_faces ),
                   source_space.GetVDim(), mfem::Ordering::byNODES },
      nodal_layout_{ std::move( occurrence_to_node ), num_patch_nodes, source_space.GetVDim() },
      exchanged_values_( restriction_.Height() ),
      repeated_values_( refinement_.Height() ),
      repeated_dual_( refinement_.Height() ),
      exchanged_dual_( restriction_.Height() ),
      returned_true_dual_( restriction_.Width() )
{
  height = nodal_layout_.Height();
  width = restriction_.Width();
}

void ElementPatchOperator::Mult( const mfem::Vector& source_true_dofs, mfem::Vector& patch_values ) const
{
  restriction_.Mult( source_true_dofs, exchanged_values_ );
  refinement_.Mult( exchanged_values_, repeated_values_ );
  nodal_layout_.Mult( repeated_values_, patch_values );
}

void ElementPatchOperator::MultTranspose( const mfem::Vector& patch_dual, mfem::Vector& source_true_dof_dual ) const
{
  source_true_dof_dual = 0.0;
  AddMultTranspose( patch_dual, source_true_dof_dual );
}

void ElementPatchOperator::AddMultTranspose( const mfem::Vector& patch_dual, mfem::Vector& source_true_dof_dual,
                                             mfem::real_t scale ) const
{
  nodal_layout_.MultTranspose( patch_dual, repeated_dual_ );
  refinement_.MultTranspose( repeated_dual_, exchanged_dual_ );
  restriction_.MultTranspose( exchanged_dual_, returned_true_dual_ );
  source_true_dof_dual.Add( scale, returned_true_dual_ );
}

shared::ParSparseMat ElementPatchOperator::Assemble() const
{
  auto* prolongation = source_space_.Dof_TrueDof_Matrix();
  SLIC_ERROR_ROOT_IF( prolongation == nullptr,
                      "Element patch assembly requires an MFEM parallel prolongation matrix." );
  shared::ParSparseMatView prolongation_view( prolongation );

  auto selection = AssembleBlockDiagonalOperator( restriction_.LocalRestriction().AssembleLocal(),
                                                   source_space_.GetComm() );
  auto selected_true_dofs = selection * prolongation_view;
  auto exchange = restriction_.Exchange().Assemble();
  auto exchanged_true_dofs = exchange * selected_true_dofs;
  auto refinement =
      AssembleBlockDiagonalOperator( AssembleLocalOperator( refinement_ ), source_space_.GetComm() );
  auto repeated_true_dofs = refinement * exchanged_true_dofs;
  auto nodal_layout =
      AssembleBlockDiagonalOperator( AssembleLocalOperator( nodal_layout_ ), source_space_.GetComm() );
  return nodal_layout * repeated_true_dofs;
}

}  // namespace redecomp
