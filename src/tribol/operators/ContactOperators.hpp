#ifndef TRIBOL_OPERATORS_CONTACTOPERATORS_HPP_
#define TRIBOL_OPERATORS_CONTACTOPERATORS_HPP_

#include "tribol/core/ConstraintRowView.hpp"
#include "tribol/evaluation/State.hpp"

#include <stdexcept>

namespace tribol::operators {

namespace detail {

inline void requireVector( FieldView<const Real> field, int dimension, const char* message )
{
  if ( !field.isStructurallyValid() || field.components != dimension ) {
    throw std::invalid_argument( message );
  }
}

inline void requireVector( FieldView<Real> field, int dimension, const char* message )
{
  if ( !field.isStructurallyValid() || field.components != dimension ) {
    throw std::invalid_argument( message );
  }
}

template <typename Field>
inline void requireDofs( ArrayView<const Index> dofs, Field field, const char* message )
{
  for ( Index entry = 0; entry < dofs.size(); ++entry ) {
    if ( dofs[entry] < 0 || dofs[entry] >= field.entities ) {
      throw std::out_of_range( message );
    }
  }
}

inline Real sideAction( ConstraintRowBatchView rows, Index row, ArrayView<const Index> offsets,
                        ArrayView<const Index> dofs, ArrayView<const Real> basis_values, FieldView<const Real> values )
{
  Real result{};
  for ( Index entry = offsets[row]; entry < offsets[row + 1]; ++entry ) {
    for ( int component = 0; component < rows.dimension; ++component ) {
      result += basis_values[entry] * rows.normals( row, component ) * values( dofs[entry], component );
    }
  }
  return result;
}

inline void addTransposeSide( ConstraintRowBatchView rows, Index row, ArrayView<const Index> offsets,
                              ArrayView<const Index> dofs, ArrayView<const Real> basis_values, Real sign, Real value,
                              FieldView<Real> result )
{
  for ( Index entry = offsets[row]; entry < offsets[row + 1]; ++entry ) {
    for ( int component = 0; component < rows.dimension; ++component ) {
      result( dofs[entry], component ) += sign * basis_values[entry] * rows.normals( row, component ) * value;
    }
  }
}

inline void zero( FieldView<Real> values )
{
  for ( Index entry = 0; entry < values.values.size(); ++entry ) {
    values.values[entry] = 0.0;
  }
}

inline void validate( ConstraintRowBatchView rows, ContactDirectionView input )
{
  if ( !rows.isStructurallyValid() ) {
    throw std::invalid_argument( "Contact operators require a structurally valid row batch." );
  }
  requireVector( input.mortar, rows.dimension, "Contact operators require a valid mortar input." );
  requireVector( input.nonmortar, rows.dimension, "Contact operators require a valid nonmortar input." );
  requireDofs( rows.mortar_dofs, input.mortar, "A mortar row references an input DOF outside the field." );
  requireDofs( rows.nonmortar_dofs, input.nonmortar, "A nonmortar row references an input DOF outside the field." );
}

inline void validate( ConstraintRowBatchView rows, ContactResidualView output )
{
  if ( !rows.isStructurallyValid() ) {
    throw std::invalid_argument( "Contact operators require a structurally valid row batch." );
  }
  requireVector( output.mortar, rows.dimension, "Contact operators require a valid mortar output." );
  requireVector( output.nonmortar, rows.dimension, "Contact operators require a valid nonmortar output." );
  requireDofs( rows.mortar_dofs, output.mortar, "A mortar row references an output DOF outside the field." );
  requireDofs( rows.nonmortar_dofs, output.nonmortar, "A nonmortar row references an output DOF outside the field." );
}

}  // namespace detail

inline void applyB( ConstraintRowBatchView rows, ContactDirectionView input, ArrayView<Real> output )
{
  detail::validate( rows, input );
  if ( !output.isStructurallyValid() || output.size() != rows.numberOfRows() ) {
    throw std::invalid_argument( "B output must contain one value per constraint row." );
  }
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    output[row] = detail::sideAction( rows, row, rows.mortar_dof_offsets, rows.mortar_dofs, rows.mortar_basis_values,
                                      input.mortar ) -
                  detail::sideAction( rows, row, rows.nonmortar_dof_offsets, rows.nonmortar_dofs,
                                      rows.nonmortar_basis_values, input.nonmortar );
  }
}

inline void addBT( ConstraintRowBatchView rows, ArrayView<const Real> row_values, ContactResidualView output )
{
  detail::validate( rows, output );
  if ( !row_values.isStructurallyValid() || row_values.size() != rows.numberOfRows() ) {
    throw std::invalid_argument( "B^T input must contain one value per constraint row." );
  }
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    detail::addTransposeSide( rows, row, rows.mortar_dof_offsets, rows.mortar_dofs, rows.mortar_basis_values, 1.0,
                              row_values[row], output.mortar );
    detail::addTransposeSide( rows, row, rows.nonmortar_dof_offsets, rows.nonmortar_dofs, rows.nonmortar_basis_values,
                              -1.0, row_values[row], output.nonmortar );
  }
}

inline void applyBT( ConstraintRowBatchView rows, ArrayView<const Real> row_values, ContactResidualView output )
{
  detail::validate( rows, output );
  detail::zero( output.mortar );
  detail::zero( output.nonmortar );
  addBT( rows, row_values, output );
}

inline void applyContactTangent( ConstraintRowBatchView rows, ArrayView<const Real> row_stiffness,
                                 ContactDirectionView input, ArrayView<Real> row_workspace, ContactResidualView output )
{
  if ( !row_stiffness.isStructurallyValid() || row_stiffness.size() != rows.numberOfRows() ||
       !row_workspace.isStructurallyValid() || row_workspace.size() != rows.numberOfRows() ) {
    throw std::invalid_argument( "Contact tangent requires row-sized stiffness and workspace arrays." );
  }
  applyB( rows, input, row_workspace );
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    row_workspace[row] *= row_stiffness[row];
  }
  applyBT( rows, { row_workspace.data(), row_workspace.size() }, output );
}

}  // namespace tribol::operators

#endif
