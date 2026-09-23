#ifndef TRIBOL_CONSTRAINT_CONSTRAINTROWASSEMBLY_HPP_
#define TRIBOL_CONSTRAINT_CONSTRAINTROWASSEMBLY_HPP_

#include "tribol/constraint/ConstraintRow.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tribol {

struct ConstraintRowSourceView {
  ConstraintRowBatchView rows{};
  ArrayView<const Index> mortar_parent_dof_map{};
  ArrayView<const Index> nonmortar_parent_dof_map{};
};

struct ConstraintRowAssemblyOptions {
  int dimension{};
  Index number_of_parent_dofs{};
  int owner_rank{ -1 };
};

class ConstraintRowAssembler {
 private:
  struct SelectedRow {
    Index source{};
    Index row{};
  };

  static void validateMap( ArrayView<const Index> source_dofs, ArrayView<const Index> dof_map, Index parent_dofs )
  {
    for ( Index source_dof : source_dofs ) {
      if ( source_dof < 0 || source_dof >= dof_map.size() ) {
        throw std::invalid_argument( "A coupled row DOF map does not cover every source DOF." );
      }
      if ( dof_map[source_dof] < 0 || dof_map[source_dof] >= parent_dofs ) {
        throw std::invalid_argument( "A coupled row DOF map references an invalid parent DOF." );
      }
    }
  }

  [[nodiscard]] static bool before( const SelectedRow& left, const SelectedRow& right,
                                    ArrayView<const ConstraintRowSourceView> sources )
  {
    const auto left_rows = sources[left.source].rows;
    const auto right_rows = sources[right.source].rows;
    if ( left_rows.contact_pair_ids[left.row] != right_rows.contact_pair_ids[right.row] ) {
      return left_rows.contact_pair_ids[left.row] < right_rows.contact_pair_ids[right.row];
    }
    if ( left_rows.row_ids[left.row] != right_rows.row_ids[right.row] ) {
      return left_rows.row_ids[left.row] < right_rows.row_ids[right.row];
    }
    if ( left.source != right.source ) {
      return left.source < right.source;
    }
    return left.row < right.row;
  }

  [[nodiscard]] static bool sameIdentity( const SelectedRow& left, const SelectedRow& right,
                                          ArrayView<const ConstraintRowSourceView> sources )
  {
    const auto left_rows = sources[left.source].rows;
    const auto right_rows = sources[right.source].rows;
    return left_rows.contact_pair_ids[left.row] == right_rows.contact_pair_ids[right.row] &&
           left_rows.row_ids[left.row] == right_rows.row_ids[right.row];
  }

 public:
  [[nodiscard]] static ConstraintRowBatch assemble( ArrayView<const ConstraintRowSourceView> sources,
                                                    ConstraintRowAssemblyOptions options )
  {
    if ( ( options.dimension != 2 && options.dimension != 3 ) || options.number_of_parent_dofs < 0 ||
         options.owner_rank < -1 ) {
      throw std::invalid_argument( "Invalid coupled constraint-row assembly options." );
    }
    bool has_velocity{};
    bool velocity_contract_known{};
    std::vector<SelectedRow> selected;
    for ( Index source_index = 0; source_index < sources.size(); ++source_index ) {
      const auto source = sources[source_index];
      if ( !source.rows.isStructurallyValid() || source.rows.dimension != options.dimension ) {
        throw std::invalid_argument( "Coupled constraint rows require valid sources with one dimension." );
      }
      validateMap( source.rows.mortar_dofs, source.mortar_parent_dof_map, options.number_of_parent_dofs );
      validateMap( source.rows.nonmortar_dofs, source.nonmortar_parent_dof_map, options.number_of_parent_dofs );
      if ( source.rows.numberOfRows() > 0 ) {
        if ( velocity_contract_known && has_velocity != source.rows.hasNormalRates() ) {
          throw std::invalid_argument( "Coupled constraint-row sources must agree on normal-rate availability." );
        }
        has_velocity = source.rows.hasNormalRates();
        velocity_contract_known = true;
      }
      for ( Index row = 0; row < source.rows.numberOfRows(); ++row ) {
        if ( options.owner_rank < 0 || source.rows.owner_ranks[row] == options.owner_rank ) {
          selected.push_back( { source_index, row } );
        }
      }
    }
    std::sort( selected.begin(), selected.end(), [sources]( const SelectedRow& left, const SelectedRow& right ) {
      return before( left, right, sources );
    } );
    selected.erase( std::unique( selected.begin(), selected.end(),
                                 [sources]( const SelectedRow& left, const SelectedRow& right ) {
                                   return sameIdentity( left, right, sources );
                                 } ),
                    selected.end() );

    std::vector<Index> cell_row_offsets( selected.size() + 1 );
    Index mortar_entries{};
    Index nonmortar_entries{};
    for ( std::size_t index = 0; index < selected.size(); ++index ) {
      cell_row_offsets[index + 1] = static_cast<Index>( index + 1 );
      const auto rows = sources[selected[index].source].rows;
      const Index row = selected[index].row;
      mortar_entries += rows.mortar_dof_offsets[row + 1] - rows.mortar_dof_offsets[row];
      nonmortar_entries += rows.nonmortar_dof_offsets[row + 1] - rows.nonmortar_dof_offsets[row];
    }
    ConstraintRowBatch result( options.dimension, std::move( cell_row_offsets ), has_velocity );
    result.reserve( static_cast<Index>( selected.size() ), mortar_entries, nonmortar_entries );
    for ( Index coupled_row = 0; coupled_row < static_cast<Index>( selected.size() ); ++coupled_row ) {
      const auto selection = selected[static_cast<std::size_t>( coupled_row )];
      const auto source = sources[selection.source];
      result.appendMappedRow( source.rows, selection.row, coupled_row, source.mortar_parent_dof_map,
                              source.nonmortar_parent_dof_map );
    }
    return result;
  }
};

[[nodiscard]] inline ConstraintRowBatch assembleOwnedConstraintRows( ArrayView<const ConstraintRowSourceView> sources,
                                                                     ConstraintRowAssemblyOptions options )
{
  return ConstraintRowAssembler::assemble( sources, options );
}

static_assert( std::is_trivially_copyable_v<ConstraintRowSourceView> );
static_assert( std::is_trivially_copyable_v<ConstraintRowAssemblyOptions> );

}  // namespace tribol

#endif
