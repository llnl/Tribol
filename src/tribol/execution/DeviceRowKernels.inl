struct DeviceRowResultLayout {
  Index nonmortar_force{};
  Index force_values{};
};

[[nodiscard]] TRIBOL_HOST_DEVICE Index rowForceKey( DeviceRowDofLayout layout, bool mortar, Index dof, int component )
{
  const Index side_offset = mortar || layout.shared_parent_dofs ? 0 : layout.mortar_dofs * layout.dimension;
  return side_offset + dof * layout.dimension + component;
}

[[nodiscard]] TRIBOL_HOST_DEVICE Index rowScatterOffset( ConstraintRowBatchView rows, Index row )
{
  return rows.dimension * ( rows.mortar_dof_offsets[row] + rows.nonmortar_dof_offsets[row] );
}

TRIBOL_HOST_DEVICE void evaluateDeviceRow( Index row, ConstraintRowBatchView rows, PenaltyRowInputView penalty,
                                           DeviceRowDofLayout layout, PenaltyRowOutputView output,
                                           ExplicitRowSummary* summaries, std::uint64_t* keys, Real* values )
{
  if ( row >= rows.numberOfRows() ) {
    return;
  }
  const auto evaluation = explicit_row_detail::evaluateRow( rows, penalty, row );
  output.effective_gap[row] = evaluation.response.effective_gap;
  output.pressure[row] = evaluation.response.pressure;
  output.row_force[row] = evaluation.integrated_force;
  output.row_stiffness[row] = evaluation.integrated_stiffness;
  output.row_rate_coefficient[row] = evaluation.integrated_rate_coefficient;
  summaries[row] = { .energy = evaluation.energy,
                     .contact_power = evaluation.contact_power,
                     .active_rows = evaluation.response.active() ? 1 : 0,
                     .active_stiffness_rows = evaluation.integrated_stiffness > 0.0 ? 1 : 0,
                     .active_rate_rows = evaluation.integrated_rate_coefficient > 0.0 ? 1 : 0,
                     .rows = 1 };

  Index scatter = rowScatterOffset( rows, row );
  for ( Index entry = rows.mortar_dof_offsets[row]; entry < rows.mortar_dof_offsets[row + 1]; ++entry ) {
    for ( int component = 0; component < rows.dimension; ++component ) {
      keys[scatter] = static_cast<std::uint64_t>( rowForceKey( layout, true, rows.mortar_dofs[entry], component ) );
      values[scatter] = rows.mortar_basis_values[entry] * evaluation.integrated_force * rows.normals( row, component );
      ++scatter;
    }
  }
  for ( Index entry = rows.nonmortar_dof_offsets[row]; entry < rows.nonmortar_dof_offsets[row + 1]; ++entry ) {
    for ( int component = 0; component < rows.dimension; ++component ) {
      keys[scatter] = static_cast<std::uint64_t>( rowForceKey( layout, false, rows.nonmortar_dofs[entry], component ) );
      values[scatter] =
          -rows.nonmortar_basis_values[entry] * evaluation.integrated_force * rows.normals( row, component );
      ++scatter;
    }
  }
}

[[nodiscard]] TRIBOL_HOST_DEVICE Real deviceAbsolute( Real value ) { return value < 0.0 ? -value : value; }

[[nodiscard]] TRIBOL_HOST_DEVICE Real rowMassScaledOneNorm( ConstraintRowBatchView rows, Index row,
                                                            DeviceRowDofLayout layout, const Real* inverse_mass )
{
  Real result{};
  for ( Index entry = rows.mortar_dof_offsets[row]; entry < rows.mortar_dof_offsets[row + 1]; ++entry ) {
    for ( int component = 0; component < rows.dimension; ++component ) {
      const Index key = rowForceKey( layout, true, rows.mortar_dofs[entry], component );
      result += deviceAbsolute( rows.mortar_basis_values[entry] * rows.normals( row, component ) ) *
                linearization_detail::squareRoot( inverse_mass[key] );
    }
  }
  for ( Index entry = rows.nonmortar_dof_offsets[row]; entry < rows.nonmortar_dof_offsets[row + 1]; ++entry ) {
    for ( int component = 0; component < rows.dimension; ++component ) {
      const Index key = rowForceKey( layout, false, rows.nonmortar_dofs[entry], component );
      result += deviceAbsolute( rows.nonmortar_basis_values[entry] * rows.normals( row, component ) ) *
                linearization_detail::squareRoot( inverse_mass[key] );
    }
  }
  return result;
}

[[nodiscard]] TRIBOL_HOST_DEVICE Index emitDeviceStabilitySide(
    ConstraintRowBatchView rows, Index row, DeviceRowDofLayout layout, bool mortar, ArrayView<const Index> offsets,
    ArrayView<const Index> dofs, ArrayView<const Real> basis_values, const Real* inverse_mass,
    const Real* row_stiffness, const Real* row_rate, Real row_norm, Index force_values, Index scatter,
    std::uint64_t* keys, Real* values )
{
  for ( Index entry = offsets[row]; entry < offsets[row + 1]; ++entry ) {
    for ( int component = 0; component < rows.dimension; ++component ) {
      const Index key = rowForceKey( layout, mortar, dofs[entry], component );
      const Real coefficient = deviceAbsolute( basis_values[entry] * rows.normals( row, component ) ) *
                               linearization_detail::squareRoot( inverse_mass[key] );
      keys[scatter] = static_cast<std::uint64_t>( key );
      values[scatter] = row_stiffness[row] * coefficient * row_norm;
      ++scatter;
      keys[scatter] = static_cast<std::uint64_t>( force_values + key );
      values[scatter] = row_rate[row] * coefficient * row_norm;
      ++scatter;
    }
  }
  return scatter;
}

TRIBOL_HOST_DEVICE void emitDeviceStabilityEntries( Index row, ConstraintRowBatchView rows, DeviceRowDofLayout layout,
                                                    const Real* inverse_mass, const Real* row_stiffness,
                                                    const Real* row_rate, Index force_values, std::uint64_t* keys,
                                                    Real* values )
{
  if ( row >= rows.numberOfRows() ) {
    return;
  }
  const Real row_norm = rowMassScaledOneNorm( rows, row, layout, inverse_mass );
  Index scatter = 2 * rowScatterOffset( rows, row );
  scatter = emitDeviceStabilitySide( rows, row, layout, true, rows.mortar_dof_offsets, rows.mortar_dofs,
                                     rows.mortar_basis_values, inverse_mass, row_stiffness, row_rate, row_norm,
                                     force_values, scatter, keys, values );
  static_cast<void>( emitDeviceStabilitySide( rows, row, layout, false, rows.nonmortar_dof_offsets, rows.nonmortar_dofs,
                                              rows.nonmortar_basis_values, inverse_mass, row_stiffness, row_rate,
                                              row_norm, force_values, scatter, keys, values ) );
}

struct ExplicitRowSummaryReduction {
  [[nodiscard]] TRIBOL_HOST_DEVICE ExplicitRowSummary operator()( ExplicitRowSummary left,
                                                                  ExplicitRowSummary right ) const
  {
    return {
        .energy = left.energy + right.energy,
        .contact_power = left.contact_power + right.contact_power,
        .active_rows = left.active_rows + right.active_rows,
        .active_stiffness_rows = left.active_stiffness_rows + right.active_stiffness_rows,
        .active_rate_rows = left.active_rate_rows + right.active_rate_rows,
        .rows = left.rows + right.rows,
    };
  }
};

struct MaximumReduction {
  [[nodiscard]] TRIBOL_HOST_DEVICE Real operator()( Real left, Real right ) const
  {
    return left > right ? left : right;
  }
};
