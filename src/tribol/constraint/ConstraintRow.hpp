#ifndef TRIBOL_CONSTRAINT_CONSTRAINTROW_HPP_
#define TRIBOL_CONSTRAINT_CONSTRAINTROW_HPP_

#include "tribol/basis/ParentBasis.hpp"
#include "tribol/core/ConstraintRowView.hpp"
#include "tribol/core/OverlapCellView.hpp"
#include "tribol/integration/AutomaticQuadrature.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tribol {

class ConstraintRowAssembler;

struct ConstraintRowBuildOptions {
  integration::Automatic::Parameters quadrature{};
  std::uint64_t first_row_id{};
  bool include_normal_rates{};
};

class ConstraintRowBatch {
 public:
  ConstraintRowBatch( int dimension, std::vector<Index> cell_row_offsets, bool has_velocity )
      : dimension_( dimension ), cell_row_offsets_( std::move( cell_row_offsets ) ), has_velocity_( has_velocity )
  {
    mortar_dof_offsets_.push_back( 0 );
    nonmortar_dof_offsets_.push_back( 0 );
  }

  [[nodiscard]] Index numberOfRows() const { return static_cast<Index>( row_ids_.size() ); }

  void updateFields( ParentTraceFieldPairView fields )
  {
    FieldShapeValidator::require( fields.mortar_coordinates, dimension_, mortar_dofs_,
                                  "Constraint rows require valid mortar parent coordinates." );
    FieldShapeValidator::require( fields.nonmortar_coordinates, dimension_, nonmortar_dofs_,
                                  "Constraint rows require valid nonmortar parent coordinates." );
    if ( has_velocity_ ) {
      FieldShapeValidator::require( fields.mortar_velocity, dimension_, mortar_dofs_,
                                    "Constraint rows require valid mortar parent velocity." );
      FieldShapeValidator::require( fields.nonmortar_velocity, dimension_, nonmortar_dofs_,
                                    "Constraint rows require valid nonmortar parent velocity." );
    }
    for ( Index row = 0; row < numberOfRows(); ++row ) {
      const auto mortar_position = interpolateRowField( row, mortar_dof_offsets_, mortar_dofs_, mortar_basis_values_,
                                                        fields.mortar_coordinates );
      const auto nonmortar_position = interpolateRowField( row, nonmortar_dof_offsets_, nonmortar_dofs_,
                                                           nonmortar_basis_values_, fields.nonmortar_coordinates );
      for ( int component = 0; component < 3; ++component ) {
        mortar_positions_[static_cast<std::size_t>( 3 * row + component )] = mortar_position[component];
        nonmortar_positions_[static_cast<std::size_t>( 3 * row + component )] = nonmortar_position[component];
      }
      const std::array<Real, 3> normal{ normals_[static_cast<std::size_t>( 3 * row )],
                                        normals_[static_cast<std::size_t>( 3 * row + 1 )],
                                        normals_[static_cast<std::size_t>( 3 * row + 2 )] };
      gaps_[static_cast<std::size_t>( row )] = normalDifference( mortar_position, nonmortar_position, normal );
      if ( has_velocity_ ) {
        const auto mortar_velocity =
            interpolateRowField( row, mortar_dof_offsets_, mortar_dofs_, mortar_basis_values_, fields.mortar_velocity );
        const auto nonmortar_velocity = interpolateRowField( row, nonmortar_dof_offsets_, nonmortar_dofs_,
                                                             nonmortar_basis_values_, fields.nonmortar_velocity );
        normal_rates_[static_cast<std::size_t>( row )] =
            normalDifference( mortar_velocity, nonmortar_velocity, normal );
      }
    }
  }

  [[nodiscard]] ConstraintRowBatchView view() const
  {
    const Index rows = numberOfRows();
    const Index mortar_entries = static_cast<Index>( mortar_dofs_.size() );
    const Index nonmortar_entries = static_cast<Index>( nonmortar_dofs_.size() );
    return {
        .dimension = dimension_,
        .cell_row_offsets = { cell_row_offsets_.data(), static_cast<Index>( cell_row_offsets_.size() ) },
        .contact_pair_ids = { contact_pair_ids_.data(), rows },
        .cell_ids = { cell_ids_.data(), rows },
        .row_ids = { row_ids_.data(), rows },
        .owner_ranks = { owner_ranks_.data(), rows },
        .overlap_cells = { overlap_cells_.data(), rows },
        .mortar_parent_faces = { mortar_parent_faces_.data(), rows },
        .nonmortar_parent_faces = { nonmortar_parent_faces_.data(), rows },
        .normals = { { normals_.data(), static_cast<Index>( normals_.size() ) }, rows, 3, FieldLayout::Interleaved },
        .integration_weights = { integration_weights_.data(), rows },
        .mortar_parent_reference_coordinates = { { mortar_parent_reference_coordinates_.data(),
                                                   static_cast<Index>( mortar_parent_reference_coordinates_.size() ) },
                                                 rows,
                                                 dimension_ - 1,
                                                 FieldLayout::Interleaved },
        .nonmortar_parent_reference_coordinates = { { nonmortar_parent_reference_coordinates_.data(),
                                                      static_cast<Index>(
                                                          nonmortar_parent_reference_coordinates_.size() ) },
                                                    rows,
                                                    dimension_ - 1,
                                                    FieldLayout::Interleaved },
        .mortar_positions = { { mortar_positions_.data(), static_cast<Index>( mortar_positions_.size() ) },
                              rows,
                              3,
                              FieldLayout::Interleaved },
        .nonmortar_positions = { { nonmortar_positions_.data(), static_cast<Index>( nonmortar_positions_.size() ) },
                                 rows,
                                 3,
                                 FieldLayout::Interleaved },
        .gaps = { gaps_.data(), rows },
        .normal_rates = { normal_rates_.data(), has_velocity_ ? rows : 0 },
        .mortar_dof_offsets = { mortar_dof_offsets_.data(), static_cast<Index>( mortar_dof_offsets_.size() ) },
        .mortar_dofs = { mortar_dofs_.data(), mortar_entries },
        .mortar_basis_values = { mortar_basis_values_.data(), mortar_entries },
        .nonmortar_dof_offsets = { nonmortar_dof_offsets_.data(), static_cast<Index>( nonmortar_dof_offsets_.size() ) },
        .nonmortar_dofs = { nonmortar_dofs_.data(), nonmortar_entries },
        .nonmortar_basis_values = { nonmortar_basis_values_.data(), nonmortar_entries },
    };
  }

  void reserve( Index rows, Index mortar_entries, Index nonmortar_entries )
  {
    contact_pair_ids_.reserve( rows );
    cell_ids_.reserve( rows );
    row_ids_.reserve( rows );
    owner_ranks_.reserve( rows );
    overlap_cells_.reserve( rows );
    mortar_parent_faces_.reserve( rows );
    nonmortar_parent_faces_.reserve( rows );
    normals_.reserve( 3 * rows );
    integration_weights_.reserve( rows );
    mortar_parent_reference_coordinates_.reserve( ( dimension_ - 1 ) * rows );
    nonmortar_parent_reference_coordinates_.reserve( ( dimension_ - 1 ) * rows );
    mortar_positions_.reserve( 3 * rows );
    nonmortar_positions_.reserve( 3 * rows );
    gaps_.reserve( rows );
    if ( has_velocity_ ) {
      normal_rates_.reserve( rows );
    }
    mortar_dofs_.reserve( mortar_entries );
    mortar_basis_values_.reserve( mortar_entries );
    nonmortar_dofs_.reserve( nonmortar_entries );
    nonmortar_basis_values_.reserve( nonmortar_entries );
  }

  void append( const OverlapCellBatchView& cells, Index cell, std::uint64_t row_id,
               const integration::OverlapQuadraturePoint& point, const ParentTracePairView& traces,
               const ParentTraceFieldPairView& fields )
  {
    contact_pair_ids_.push_back( cells.contact_pair_ids[cell] );
    cell_ids_.push_back( cells.cell_ids[cell] );
    row_ids_.push_back( row_id );
    owner_ranks_.push_back( cells.owner_ranks[cell] );
    overlap_cells_.push_back( cell );
    const Index mortar_face = cells.mortar_parent_faces[cell];
    const Index nonmortar_face = cells.nonmortar_parent_faces[cell];
    mortar_parent_faces_.push_back( mortar_face );
    nonmortar_parent_faces_.push_back( nonmortar_face );
    normals_.insert( normals_.end(), point.normal.begin(), point.normal.end() );
    integration_weights_.push_back( point.weight );
    appendReference( mortar_parent_reference_coordinates_, point.mortar_reference );
    appendReference( nonmortar_parent_reference_coordinates_, point.nonmortar_reference );
    const auto mortar_position =
        basis::interpolateParentField( traces.mortar, mortar_face, fields.mortar_coordinates, point.mortar_reference );
    const auto nonmortar_position = basis::interpolateParentField(
        traces.nonmortar, nonmortar_face, fields.nonmortar_coordinates, point.nonmortar_reference );
    mortar_positions_.insert( mortar_positions_.end(), mortar_position.begin(), mortar_position.end() );
    nonmortar_positions_.insert( nonmortar_positions_.end(), nonmortar_position.begin(), nonmortar_position.end() );
    gaps_.push_back( normalDifference( mortar_position, nonmortar_position, point.normal ) );
    if ( has_velocity_ ) {
      if ( fields.hasVelocity() ) {
        const auto mortar_velocity =
            basis::interpolateParentField( traces.mortar, mortar_face, fields.mortar_velocity, point.mortar_reference );
        const auto nonmortar_velocity = basis::interpolateParentField(
            traces.nonmortar, nonmortar_face, fields.nonmortar_velocity, point.nonmortar_reference );
        normal_rates_.push_back( normalDifference( mortar_velocity, nonmortar_velocity, point.normal ) );
      } else {
        normal_rates_.push_back( 0.0 );
      }
    }
    appendBasis( traces.mortar, mortar_face, point.mortar_reference, mortar_dofs_, mortar_basis_values_,
                 mortar_dof_offsets_ );
    appendBasis( traces.nonmortar, nonmortar_face, point.nonmortar_reference, nonmortar_dofs_, nonmortar_basis_values_,
                 nonmortar_dof_offsets_ );
  }

 private:
  friend class ConstraintRowAssembler;

  struct FieldShapeValidator {
    static void require( FieldView<const Real> field, int dimension, const std::vector<Index>& dofs,
                         const char* message )
    {
      if ( !field.isStructurallyValid() || field.components != dimension ) {
        throw std::invalid_argument( message );
      }
      for ( Index dof : dofs ) {
        if ( dof < 0 || dof >= field.entities ) {
          throw std::invalid_argument( message );
        }
      }
    }
  };

  [[nodiscard]] std::array<Real, 3> interpolateRowField( Index row, const std::vector<Index>& offsets,
                                                         const std::vector<Index>& dofs,
                                                         const std::vector<Real>& basis_values,
                                                         FieldView<const Real> field ) const
  {
    std::array<Real, 3> result{};
    for ( Index entry = offsets[static_cast<std::size_t>( row )]; entry < offsets[static_cast<std::size_t>( row + 1 )];
          ++entry ) {
      for ( int component = 0; component < dimension_; ++component ) {
        result[component] += basis_values[static_cast<std::size_t>( entry )] *
                             field( dofs[static_cast<std::size_t>( entry )], component );
      }
    }
    return result;
  }

  void appendMappedRow( ConstraintRowBatchView source, Index source_row, Index coupled_cell,
                        ArrayView<const Index> mortar_dof_map, ArrayView<const Index> nonmortar_dof_map )
  {
    contact_pair_ids_.push_back( source.contact_pair_ids[source_row] );
    cell_ids_.push_back( source.cell_ids[source_row] );
    row_ids_.push_back( source.row_ids[source_row] );
    owner_ranks_.push_back( source.owner_ranks[source_row] );
    overlap_cells_.push_back( coupled_cell );
    mortar_parent_faces_.push_back( source.mortar_parent_faces[source_row] );
    nonmortar_parent_faces_.push_back( source.nonmortar_parent_faces[source_row] );
    for ( int component = 0; component < 3; ++component ) {
      normals_.push_back( source.normals( source_row, component ) );
      mortar_positions_.push_back( source.mortar_positions( source_row, component ) );
      nonmortar_positions_.push_back( source.nonmortar_positions( source_row, component ) );
    }
    integration_weights_.push_back( source.integration_weights[source_row] );
    for ( int component = 0; component < dimension_ - 1; ++component ) {
      mortar_parent_reference_coordinates_.push_back(
          source.mortar_parent_reference_coordinates( source_row, component ) );
      nonmortar_parent_reference_coordinates_.push_back(
          source.nonmortar_parent_reference_coordinates( source_row, component ) );
    }
    gaps_.push_back( source.gaps[source_row] );
    if ( has_velocity_ ) {
      normal_rates_.push_back( source.normal_rates[source_row] );
    }
    appendMappedBasis( source_row, source.mortar_dof_offsets, source.mortar_dofs, source.mortar_basis_values,
                       mortar_dof_map, mortar_dofs_, mortar_basis_values_, mortar_dof_offsets_ );
    appendMappedBasis( source_row, source.nonmortar_dof_offsets, source.nonmortar_dofs, source.nonmortar_basis_values,
                       nonmortar_dof_map, nonmortar_dofs_, nonmortar_basis_values_, nonmortar_dof_offsets_ );
  }

  static void appendMappedBasis( Index row, ArrayView<const Index> source_offsets, ArrayView<const Index> source_dofs,
                                 ArrayView<const Real> source_basis, ArrayView<const Index> dof_map,
                                 std::vector<Index>& dofs, std::vector<Real>& basis, std::vector<Index>& offsets )
  {
    for ( Index entry = source_offsets[row]; entry < source_offsets[row + 1]; ++entry ) {
      dofs.push_back( dof_map[source_dofs[entry]] );
      basis.push_back( source_basis[entry] );
    }
    offsets.push_back( static_cast<Index>( dofs.size() ) );
  }

  void appendReference( std::vector<Real>& values, basis::ReferencePoint point ) const
  {
    values.push_back( point.first );
    if ( dimension_ == 3 ) {
      values.push_back( point.second );
    }
  }

  [[nodiscard]] Real normalDifference( const std::array<Real, 3>& mortar, const std::array<Real, 3>& nonmortar,
                                       const std::array<Real, 3>& normal ) const
  {
    Real result{};
    for ( int component = 0; component < dimension_; ++component ) {
      result += ( mortar[component] - nonmortar[component] ) * normal[component];
    }
    return result;
  }

  static void appendBasis( const ParentTraceView& trace, Index face, basis::ReferencePoint reference,
                           std::vector<Index>& dofs, std::vector<Real>& values, std::vector<Index>& offsets )
  {
    const auto shape = basis::parentShape( trace, face, reference );
    const Index begin = trace.parent_dof_offsets[face];
    for ( int local_dof = 0; local_dof < shape.size; ++local_dof ) {
      dofs.push_back( trace.parent_dofs[begin + local_dof] );
      values.push_back( trace.parent_dof_signs[begin + local_dof] * shape[local_dof] );
    }
    offsets.push_back( static_cast<Index>( dofs.size() ) );
  }

  int dimension_{};
  std::vector<Index> cell_row_offsets_{};
  bool has_velocity_{};
  std::vector<std::uint64_t> contact_pair_ids_{};
  std::vector<std::uint64_t> cell_ids_{};
  std::vector<std::uint64_t> row_ids_{};
  std::vector<int> owner_ranks_{};
  std::vector<Index> overlap_cells_{};
  std::vector<Index> mortar_parent_faces_{};
  std::vector<Index> nonmortar_parent_faces_{};
  std::vector<Real> normals_{};
  std::vector<Real> integration_weights_{};
  std::vector<Real> mortar_parent_reference_coordinates_{};
  std::vector<Real> nonmortar_parent_reference_coordinates_{};
  std::vector<Real> mortar_positions_{};
  std::vector<Real> nonmortar_positions_{};
  std::vector<Real> gaps_{};
  std::vector<Real> normal_rates_{};
  std::vector<Index> mortar_dof_offsets_{};
  std::vector<Index> mortar_dofs_{};
  std::vector<Real> mortar_basis_values_{};
  std::vector<Index> nonmortar_dof_offsets_{};
  std::vector<Index> nonmortar_dofs_{};
  std::vector<Real> nonmortar_basis_values_{};
};

namespace constraint_row_detail {

inline void requireField( FieldView<const Real> field, const ParentTraceView& trace, const char* message )
{
  if ( !field.isStructurallyValid() || field.components != trace.geometry.dimension ) {
    throw std::invalid_argument( message );
  }
  for ( Index dof = 0; dof < trace.parent_dofs.size(); ++dof ) {
    if ( trace.parent_dofs[dof] < 0 || trace.parent_dofs[dof] >= field.entities ) {
      throw std::invalid_argument( message );
    }
  }
}

}  // namespace constraint_row_detail

[[nodiscard]] inline ConstraintRowBatch buildConstraintRowBatch( ParentTracePairView traces, OverlapCellBatchView cells,
                                                                 ParentTraceFieldPairView fields,
                                                                 ConstraintRowBuildOptions options = {} )
{
  if ( !traces.isStructurallyValid() || !cells.isStructurallyValid() ||
       cells.dimension != traces.mortar.geometry.dimension ) {
    throw std::invalid_argument( "Constraint rows require compatible parent traces and overlap cells." );
  }
  if ( options.quadrature.minimum_exactness < 0 || options.quadrature.traction_order < -1 ) {
    throw std::invalid_argument( "Automatic quadrature orders cannot be negative." );
  }
  constraint_row_detail::requireField( fields.mortar_coordinates, traces.mortar,
                                       "Constraint rows require valid mortar parent coordinates." );
  constraint_row_detail::requireField( fields.nonmortar_coordinates, traces.nonmortar,
                                       "Constraint rows require valid nonmortar parent coordinates." );
  const bool any_velocity = options.include_normal_rates || !fields.mortar_velocity.values.empty() ||
                            !fields.nonmortar_velocity.values.empty();
  if ( any_velocity ) {
    if ( fields.mortar_velocity.values.empty() != fields.nonmortar_velocity.values.empty() ) {
      throw std::invalid_argument( "Constraint rows require velocity fields on both parent traces." );
    }
    if ( !fields.mortar_velocity.values.empty() ) {
      constraint_row_detail::requireField( fields.mortar_velocity, traces.mortar,
                                           "Constraint rows require valid mortar parent velocity." );
      constraint_row_detail::requireField( fields.nonmortar_velocity, traces.nonmortar,
                                           "Constraint rows require valid nonmortar parent velocity." );
    }
  }

  std::vector<Index> cell_row_offsets( static_cast<std::size_t>( cells.numberOfAttempts() + 1 ), 0 );
  for ( Index cell = 0; cell < cells.numberOfAttempts(); ++cell ) {
    if ( cells.isAccepted( cell ) &&
         integration::automaticQuadratureExactness( traces, cells, cell, options.quadrature ) >
             integration::maximumAutomaticExactness ) {
      throw std::invalid_argument( "Requested automatic quadrature exceeds the built-in exactness." );
    }
    cell_row_offsets[cell + 1] = integration::automaticQuadraturePointCount( traces, cells, cell, options.quadrature );
  }
  for ( Index cell = 0; cell < cells.numberOfAttempts(); ++cell ) {
    cell_row_offsets[cell + 1] += cell_row_offsets[cell];
  }
  const Index rows = cell_row_offsets.back();
  Index mortar_entries{};
  Index nonmortar_entries{};
  for ( Index cell = 0; cell < cells.numberOfAttempts(); ++cell ) {
    const Index count = cell_row_offsets[cell + 1] - cell_row_offsets[cell];
    if ( count > 0 ) {
      mortar_entries +=
          count * parentFaceDofCount( traces.mortar.parent_face_topologies[cells.mortar_parent_faces[cell]],
                                      traces.mortar.parent_face_orders[cells.mortar_parent_faces[cell]] );
      nonmortar_entries +=
          count * parentFaceDofCount( traces.nonmortar.parent_face_topologies[cells.nonmortar_parent_faces[cell]],
                                      traces.nonmortar.parent_face_orders[cells.nonmortar_parent_faces[cell]] );
    }
  }
  ConstraintRowBatch result( cells.dimension, std::move( cell_row_offsets ), any_velocity );
  result.reserve( rows, mortar_entries, nonmortar_entries );
  const auto offsets = result.view().cell_row_offsets;
  for ( Index cell = 0; cell < cells.numberOfAttempts(); ++cell ) {
    const Index count = offsets[cell + 1] - offsets[cell];
    for ( Index local_row = 0; local_row < count; ++local_row ) {
      const auto point = integration::automaticQuadraturePoint( traces, cells, cell, local_row, options.quadrature );
      const auto row_id = options.first_row_id + static_cast<std::uint64_t>( offsets[cell] + local_row );
      result.append( cells, cell, row_id, point, traces, fields );
    }
  }
  return result;
}

}  // namespace tribol

#endif
