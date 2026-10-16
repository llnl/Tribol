// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/SelectedElementRestriction.hpp"

#include <algorithm>

#include "axom/slic.hpp"

namespace redecomp {

namespace {

const mfem::Operator& ResolveProlongation( const mfem::ParFiniteElementSpace& source_space )
{
  const mfem::Operator* prolongation = source_space.GetProlongationMatrix();
  SLIC_ERROR_ROOT_IF( prolongation == nullptr, "SelectedElementRestriction requires an MFEM prolongation operator." );
  return *prolongation;
}

const mfem::ElementRestrictionOperator& ResolveElementRestriction( const mfem::ParFiniteElementSpace& source_space,
                                                                   mfem::ElementDofOrdering element_ordering )
{
  const auto* restriction = source_space.GetElementRestriction( element_ordering );
  SLIC_ERROR_ROOT_IF( restriction == nullptr,
                      "SelectedElementRestriction requires an MFEM element restriction operator." );
  return *restriction;
}

}  // namespace

SelectedElementRestriction::SelectedElementRestriction( const mfem::ParFiniteElementSpace& source_space,
                                                        mfem::Array<int> selected_elements,
                                                        mfem::ElementDofOrdering element_ordering )
    : mfem::Operator(),
      source_space_{ source_space },
      prolongation_{ ResolveProlongation( source_space ) },
      element_restriction_{ ResolveElementRestriction( source_space, element_ordering ) },
      selected_elements_{ std::move( selected_elements ) }
{
  ValidateUniformElementLayout();
  selected_element_offsets_.SetSize( selected_elements_.Size() + 1 );
  for ( int i = 0; i <= selected_elements_.Size(); ++i ) {
    selected_element_offsets_[i] = i * element_block_size_;
  }
  height = selected_element_offsets_.Last();
  width = source_space_.GetTrueVSize();

  local_values_.SetSize( source_space_.GetVSize() );
  all_element_values_.SetSize( element_restriction_.Height() );
  local_dual_.SetSize( source_space_.GetVSize() );
  all_element_dual_.SetSize( element_restriction_.Height() );
  returned_true_dual_.SetSize( source_space_.GetTrueVSize() );
}

void SelectedElementRestriction::ValidateUniformElementLayout()
{
  SLIC_ERROR_ROOT_IF( source_space_.GetNE() == 0 && selected_elements_.Size() != 0,
                      "SelectedElementRestriction cannot select elements from an empty space." );
  if ( source_space_.GetNE() == 0 ) {
    return;
  }

  element_block_size_ = source_space_.GetFE( 0 )->GetDof() * source_space_.GetVDim();
  for ( int element = 0; element < source_space_.GetNE(); ++element ) {
    SLIC_ERROR_ROOT_IF( source_space_.GetFE( element )->GetDof() * source_space_.GetVDim() != element_block_size_,
                        "SelectedElementRestriction currently requires a uniform element E-vector layout." );
  }
  for ( int element : selected_elements_ ) {
    SLIC_ERROR_ROOT_IF( element < 0 || element >= source_space_.GetNE(),
                        "SelectedElementRestriction contains an out-of-range element." );
  }
  SLIC_ERROR_ROOT_IF( element_restriction_.Height() != element_block_size_ * source_space_.GetNE(),
                      "MFEM element restriction layout does not match the selected element block layout." );
}

void SelectedElementRestriction::GatherSelectedBlocks( const mfem::Vector& all_element_values,
                                                       mfem::Vector& selected_values ) const
{
  const mfem::real_t* all_values = all_element_values.HostRead();
  mfem::real_t* selected = selected_values.HostWrite();
  for ( int selected_element = 0; selected_element < selected_elements_.Size(); ++selected_element ) {
    const int source_element = selected_elements_[selected_element];
    std::copy_n( all_values + source_element * element_block_size_, element_block_size_,
                 selected + selected_element_offsets_[selected_element] );
  }
}

void SelectedElementRestriction::InjectSelectedBlocks( const mfem::Vector& selected_values,
                                                       mfem::Vector& all_element_values, mfem::real_t scale ) const
{
  const mfem::real_t* selected = selected_values.HostRead();
  mfem::real_t* all_values = all_element_values.HostReadWrite();
  for ( int selected_element = 0; selected_element < selected_elements_.Size(); ++selected_element ) {
    const int destination_element = selected_elements_[selected_element];
    for ( int i = 0; i < element_block_size_; ++i ) {
      all_values[destination_element * element_block_size_ + i] +=
          scale * selected[selected_element_offsets_[selected_element] + i];
    }
  }
}

void SelectedElementRestriction::Mult( const mfem::Vector& true_dofs, mfem::Vector& selected_element_values ) const
{
  SLIC_ERROR_ROOT_IF( true_dofs.Size() != Width(), "SelectedElementRestriction true-DOF size mismatch." );
  SLIC_ERROR_ROOT_IF( selected_element_values.Size() != Height(),
                      "SelectedElementRestriction selected E-vector size mismatch." );
  prolongation_.Mult( true_dofs, local_values_ );
  element_restriction_.Mult( local_values_, all_element_values_ );
  GatherSelectedBlocks( all_element_values_, selected_element_values );
}

void SelectedElementRestriction::MultTranspose( const mfem::Vector& selected_element_dual,
                                                mfem::Vector& true_dof_dual ) const
{
  true_dof_dual = 0.0;
  AddMultTranspose( selected_element_dual, true_dof_dual );
}

void SelectedElementRestriction::AddMultTranspose( const mfem::Vector& selected_element_dual,
                                                   mfem::Vector& true_dof_dual, const mfem::real_t scale ) const
{
  SLIC_ERROR_ROOT_IF( selected_element_dual.Size() != Height(),
                      "SelectedElementRestriction selected dual size mismatch." );
  SLIC_ERROR_ROOT_IF( true_dof_dual.Size() != Width(), "SelectedElementRestriction true dual size mismatch." );

  all_element_dual_ = 0.0;
  InjectSelectedBlocks( selected_element_dual, all_element_dual_, 1.0 );
  element_restriction_.MultTranspose( all_element_dual_, local_dual_ );
  prolongation_.MultTranspose( local_dual_, returned_true_dual_ );
  true_dof_dual.Add( scale, returned_true_dual_ );
}

mfem::SparseMatrix SelectedElementRestriction::AssembleLocal() const
{
  mfem::SparseMatrix matrix( Height(), source_space_.GetVSize() );
  mfem::Array<int> element_vdofs;
  mfem::Vector element_dual;
  for ( int selected_element = 0; selected_element < selected_elements_.Size(); ++selected_element ) {
    mfem::DofTransformation orientation;
    source_space_.GetElementVDofs( selected_elements_[selected_element], element_vdofs, orientation );
    const int row_begin = selected_element_offsets_[selected_element];
    SLIC_ERROR_ROOT_IF( element_vdofs.Size() != element_block_size_,
                        "Selected element VDofs do not match its E-vector block size." );
    for ( int element_row = 0; element_row < element_vdofs.Size(); ++element_row ) {
      element_dual.SetSize( element_vdofs.Size() );
      element_dual = 0.0;
      element_dual[element_row] = 1.0;
      orientation.TransformDual( element_dual );
      for ( int element_column = 0; element_column < element_vdofs.Size(); ++element_column ) {
        const mfem::real_t value = element_dual[element_column];
        if ( value == 0.0 ) continue;
        const int signed_vdof = element_vdofs[element_column];
        const int local_vdof = signed_vdof >= 0 ? signed_vdof : -1 - signed_vdof;
        matrix.Add( row_begin + element_row, local_vdof, signed_vdof >= 0 ? value : -value );
      }
    }
  }
  matrix.Finalize();
  return matrix;
}

}  // namespace redecomp
