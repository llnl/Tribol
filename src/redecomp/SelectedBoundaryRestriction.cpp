// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/SelectedBoundaryRestriction.hpp"

#include <algorithm>
#include <utility>

#include "axom/slic.hpp"

namespace redecomp {

namespace {

const mfem::Operator& GetParallelProlongation( const mfem::ParFiniteElementSpace& source_space )
{
  const auto* prolongation = source_space.GetProlongationMatrix();
  SLIC_ERROR_ROOT_IF( prolongation == nullptr,
                      "SelectedBoundaryRestriction requires an MFEM parallel prolongation operator." );
  return *prolongation;
}

}  // namespace

SelectedBoundaryRestriction::SelectedBoundaryRestriction( const mfem::ParFiniteElementSpace& source_space,
                                                          mfem::Array<int> selected_boundary_elements )
    : mfem::Operator(),
      source_space_{ source_space },
      prolongation_{ GetParallelProlongation( source_space ) },
      selected_boundary_elements_{ std::move( selected_boundary_elements ) },
      local_values_( source_space.GetVSize() ),
      local_dual_( source_space.GetVSize() ),
      returned_true_dual_( source_space.GetTrueVSize() )
{
  BuildElementOffsets();
  height = selected_element_offsets_.Last();
  width = source_space_.GetTrueVSize();
}

void SelectedBoundaryRestriction::BuildElementOffsets()
{
  selected_element_offsets_.SetSize( selected_boundary_elements_.Size() + 1 );
  selected_element_offsets_[0] = 0;
  for ( int selected_element = 0; selected_element < selected_boundary_elements_.Size(); ++selected_element ) {
    const int boundary_element = selected_boundary_elements_[selected_element];
    SLIC_ERROR_ROOT_IF( boundary_element < 0 || boundary_element >= source_space_.GetNBE(),
                        "SelectedBoundaryRestriction contains an out-of-range boundary element." );
    source_space_.GetBdrElementVDofs( boundary_element, boundary_vdofs_ );
    selected_element_offsets_[selected_element + 1] =
        selected_element_offsets_[selected_element] + boundary_vdofs_.Size();
  }
}

void SelectedBoundaryRestriction::GatherBoundaryElement( int selected_element, mfem::Vector& element_values ) const
{
  const int boundary_element = selected_boundary_elements_[selected_element];
  mfem::DofTransformation orientation;
  source_space_.GetBdrElementVDofs( boundary_element, boundary_vdofs_, orientation );
  element_values.SetSize( boundary_vdofs_.Size() );
  local_values_.GetSubVector( boundary_vdofs_, element_values );

  // MFEM stores shared VDofs in a globally consistent orientation. Contact
  // evaluates each face in its native reference orientation instead.
  orientation.InvTransformPrimal( element_values );
}

void SelectedBoundaryRestriction::AddBoundaryElementDual( int selected_element, const mfem::Vector& element_dual ) const
{
  const int boundary_element = selected_boundary_elements_[selected_element];
  mfem::DofTransformation orientation;
  source_space_.GetBdrElementVDofs( boundary_element, boundary_vdofs_, orientation );
  element_buffer_ = element_dual;

  // This is the exact dual transform of GatherBoundaryElement(). Applying it
  // before signed VDof insertion preserves work across oriented face spaces.
  orientation.TransformDual( element_buffer_ );
  local_dual_.AddElementVector( boundary_vdofs_, element_buffer_ );
}

void SelectedBoundaryRestriction::Mult( const mfem::Vector& true_dofs, mfem::Vector& selected_element_values ) const
{
  SLIC_ERROR_ROOT_IF( true_dofs.Size() != Width(), "SelectedBoundaryRestriction true-DOF size mismatch." );
  SLIC_ERROR_ROOT_IF( selected_element_values.Size() != Height(),
                      "SelectedBoundaryRestriction boundary E-vector size mismatch." );
  prolongation_.Mult( true_dofs, local_values_ );
  for ( int selected_element = 0; selected_element < selected_boundary_elements_.Size(); ++selected_element ) {
    GatherBoundaryElement( selected_element, element_buffer_ );
    const int offset = selected_element_offsets_[selected_element];
    std::copy_n( element_buffer_.HostRead(), element_buffer_.Size(), selected_element_values.HostWrite() + offset );
  }
}

void SelectedBoundaryRestriction::MultTranspose( const mfem::Vector& selected_element_dual,
                                                 mfem::Vector& true_dof_dual ) const
{
  true_dof_dual = 0.0;
  AddMultTranspose( selected_element_dual, true_dof_dual );
}

void SelectedBoundaryRestriction::AddMultTranspose( const mfem::Vector& selected_element_dual,
                                                    mfem::Vector& true_dof_dual, mfem::real_t scale ) const
{
  SLIC_ERROR_ROOT_IF( selected_element_dual.Size() != Height(),
                      "SelectedBoundaryRestriction boundary E-vector dual size mismatch." );
  SLIC_ERROR_ROOT_IF( true_dof_dual.Size() != Width(), "SelectedBoundaryRestriction true-DOF dual size mismatch." );
  local_dual_ = 0.0;
  for ( int selected_element = 0; selected_element < selected_boundary_elements_.Size(); ++selected_element ) {
    const int offset = selected_element_offsets_[selected_element];
    const int size = selected_element_offsets_[selected_element + 1] - offset;
    mfem::Vector element_dual_view( const_cast<mfem::real_t*>( selected_element_dual.HostRead() ) + offset, size );
    AddBoundaryElementDual( selected_element, element_dual_view );
  }
  prolongation_.MultTranspose( local_dual_, returned_true_dual_ );
  true_dof_dual.Add( scale, returned_true_dual_ );
}

mfem::SparseMatrix SelectedBoundaryRestriction::AssembleLocal() const
{
  mfem::SparseMatrix matrix( Height(), source_space_.GetVSize() );
  for ( int selected_element = 0; selected_element < selected_boundary_elements_.Size(); ++selected_element ) {
    mfem::DofTransformation orientation;
    source_space_.GetBdrElementVDofs( selected_boundary_elements_[selected_element], boundary_vdofs_, orientation );
    const int row_begin = selected_element_offsets_[selected_element];
    for ( int element_row = 0; element_row < boundary_vdofs_.Size(); ++element_row ) {
      element_buffer_.SetSize( boundary_vdofs_.Size() );
      element_buffer_ = 0.0;
      element_buffer_[element_row] = 1.0;
      orientation.TransformDual( element_buffer_ );
      for ( int element_column = 0; element_column < boundary_vdofs_.Size(); ++element_column ) {
        const mfem::real_t value = element_buffer_[element_column];
        if ( value == 0.0 ) continue;
        const int signed_vdof = boundary_vdofs_[element_column];
        const int local_vdof = signed_vdof >= 0 ? signed_vdof : -1 - signed_vdof;
        matrix.Add( row_begin + element_row, local_vdof, signed_vdof >= 0 ? value : -value );
      }
    }
  }
  matrix.Finalize();
  return matrix;
}

}  // namespace redecomp
