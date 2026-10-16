// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/DistributedBoundaryRestriction.hpp"

namespace redecomp {

DistributedBoundaryRestriction::DistributedBoundaryRestriction(
    const mfem::ParFiniteElementSpace& source_space, mfem::Array<int> selected_boundary_elements,
    std::vector<mfem::Array<int>> selected_elements_by_destination )
    : mfem::Operator(),
      local_restriction_{ source_space, std::move( selected_boundary_elements ) },
      plan_{ std::make_shared<ElementExchangePlan>( source_space.GetComm(), local_restriction_.SelectedElementOffsets(),
                                                    std::move( selected_elements_by_destination ) ) },
      exchange_{ plan_ },
      selected_values_( local_restriction_.Height() ),
      selected_dual_( local_restriction_.Height() ),
      returned_true_dual_( local_restriction_.Width() )
{
  height = exchange_.Height();
  width = local_restriction_.Width();
}

void DistributedBoundaryRestriction::Mult( const mfem::Vector& true_dofs,
                                           mfem::Vector& destination_element_values ) const
{
  local_restriction_.Mult( true_dofs, selected_values_ );
  exchange_.Mult( selected_values_, destination_element_values );
}

void DistributedBoundaryRestriction::MultTranspose( const mfem::Vector& destination_element_dual,
                                                    mfem::Vector& true_dof_dual ) const
{
  true_dof_dual = 0.0;
  AddMultTranspose( destination_element_dual, true_dof_dual );
}

void DistributedBoundaryRestriction::AddMultTranspose( const mfem::Vector& destination_element_dual,
                                                       mfem::Vector& true_dof_dual, mfem::real_t scale ) const
{
  exchange_.MultTranspose( destination_element_dual, selected_dual_ );
  local_restriction_.MultTranspose( selected_dual_, returned_true_dual_ );
  true_dof_dual.Add( scale, returned_true_dual_ );
}

}  // namespace redecomp
