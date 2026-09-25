#ifndef TRIBOL_CORE_CONSTRAINTROWVIEW_HPP_
#define TRIBOL_CORE_CONSTRAINTROWVIEW_HPP_

#include "tribol/core/ParentTraceView.hpp"
#include "tribol/method/Method.hpp"

#include <array>
#include <cstdint>
#include <type_traits>

namespace tribol {

template <typename Scalar>
struct ParentTraceFieldPairViewT {
  FieldView<const Scalar> mortar_coordinates{};
  FieldView<const Scalar> nonmortar_coordinates{};
  FieldView<const Scalar> mortar_velocity{};
  FieldView<const Scalar> nonmortar_velocity{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool hasVelocity() const
  {
    return !mortar_velocity.values.empty() && !nonmortar_velocity.values.empty();
  }
};

using ParentTraceFieldPairView = ParentTraceFieldPairViewT<Real>;

template <typename Scalar>
struct ConstraintRowBatchViewT {
  int dimension{};
  ArrayView<const Index> cell_row_offsets{};
  ArrayView<const std::uint64_t> contact_pair_ids{};
  ArrayView<const std::uint64_t> cell_ids{};
  ArrayView<const std::uint64_t> row_ids{};
  ArrayView<const int> owner_ranks{};
  ArrayView<const Index> overlap_cells{};
  ArrayView<const Index> mortar_parent_faces{};
  ArrayView<const Index> nonmortar_parent_faces{};
  FieldView<const Scalar> normals{};
  ArrayView<const Scalar> integration_weights{};
  FieldView<const Scalar> mortar_parent_reference_coordinates{};
  FieldView<const Scalar> nonmortar_parent_reference_coordinates{};
  FieldView<const Scalar> mortar_positions{};
  FieldView<const Scalar> nonmortar_positions{};
  ArrayView<const Scalar> gaps{};
  ArrayView<const Scalar> normal_rates{};
  ArrayView<const Index> mortar_dof_offsets{};
  ArrayView<const Index> mortar_dofs{};
  ArrayView<const Scalar> mortar_basis_values{};
  ArrayView<const Index> nonmortar_dof_offsets{};
  ArrayView<const Index> nonmortar_dofs{};
  ArrayView<const Scalar> nonmortar_basis_values{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index numberOfCells() const { return cell_row_offsets.size() - 1; }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index numberOfRows() const { return row_ids.size(); }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool hasNormalRates() const { return !normal_rates.empty(); }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool isStructurallyValid() const
  {
    const Index rows = numberOfRows();
    const int reference_dimension = dimension - 1;
    const bool valid_arrays =
        cell_row_offsets.isStructurallyValid() && contact_pair_ids.isStructurallyValid() &&
        cell_ids.isStructurallyValid() && row_ids.isStructurallyValid() && owner_ranks.isStructurallyValid() &&
        overlap_cells.isStructurallyValid() && mortar_parent_faces.isStructurallyValid() &&
        nonmortar_parent_faces.isStructurallyValid() && integration_weights.isStructurallyValid() &&
        gaps.isStructurallyValid() && normal_rates.isStructurallyValid() && mortar_dof_offsets.isStructurallyValid() &&
        mortar_dofs.isStructurallyValid() && mortar_basis_values.isStructurallyValid() &&
        nonmortar_dof_offsets.isStructurallyValid() && nonmortar_dofs.isStructurallyValid() &&
        nonmortar_basis_values.isStructurallyValid();
    const bool row_sizes = contact_pair_ids.size() == rows && cell_ids.size() == rows && owner_ranks.size() == rows &&
                           overlap_cells.size() == rows && mortar_parent_faces.size() == rows &&
                           nonmortar_parent_faces.size() == rows && integration_weights.size() == rows &&
                           gaps.size() == rows && ( normal_rates.empty() || normal_rates.size() == rows );
    const bool field_shapes = normals.isStructurallyValid() && normals.entities == rows && normals.components == 3 &&
                              mortar_parent_reference_coordinates.isStructurallyValid() &&
                              mortar_parent_reference_coordinates.entities == rows &&
                              mortar_parent_reference_coordinates.components == reference_dimension &&
                              nonmortar_parent_reference_coordinates.isStructurallyValid() &&
                              nonmortar_parent_reference_coordinates.entities == rows &&
                              nonmortar_parent_reference_coordinates.components == reference_dimension &&
                              mortar_positions.isStructurallyValid() && mortar_positions.entities == rows &&
                              mortar_positions.components == 3 && nonmortar_positions.isStructurallyValid() &&
                              nonmortar_positions.entities == rows && nonmortar_positions.components == 3;
    const bool dof_shapes = mortar_dof_offsets.size() == rows + 1 && nonmortar_dof_offsets.size() == rows + 1 &&
                            mortar_dofs.size() == mortar_basis_values.size() &&
                            nonmortar_dofs.size() == nonmortar_basis_values.size();
    if ( ( dimension != 2 && dimension != 3 ) || numberOfCells() < 0 || !valid_arrays || !row_sizes || !field_shapes ||
         !dof_shapes || cell_row_offsets[0] != 0 || cell_row_offsets[numberOfCells()] != rows ||
         mortar_dof_offsets[0] != 0 || nonmortar_dof_offsets[0] != 0 ||
         mortar_dof_offsets[rows] != mortar_dofs.size() || nonmortar_dof_offsets[rows] != nonmortar_dofs.size() ) {
      return false;
    }
    for ( Index row = 0; row < rows; ++row ) {
      if ( mortar_dof_offsets[row] > mortar_dof_offsets[row + 1] ||
           nonmortar_dof_offsets[row] > nonmortar_dof_offsets[row + 1] || integration_weights[row] < 0.0 ) {
        return false;
      }
    }
    return true;
  }
};

using ConstraintRowBatchView = ConstraintRowBatchViewT<Real>;

template <typename BasisScalar, typename FieldScalar>
[[nodiscard]] TRIBOL_HOST_DEVICE inline auto interpolateConstraintRowField( Index row, ArrayView<const Index> offsets,
                                                                            ArrayView<const Index> dofs,
                                                                            ArrayView<const BasisScalar> basis_values,
                                                                            FieldView<FieldScalar> field,
                                                                            int dimension )
{
  using ResultScalar = decltype( basis_values[Index{}] * field( Index{}, 0 ) );
  std::array<ResultScalar, 3> result{};
  for ( Index entry = offsets[row]; entry < offsets[row + 1]; ++entry ) {
    for ( int component = 0; component < dimension; ++component ) {
      result[component] += basis_values[entry] * field( dofs[entry], component );
    }
  }
  return result;
}

template <typename Scalar>
struct PenaltyRowInputViewT {
  ArrayView<const Scalar> pressure_stiffness{};
  ArrayView<const Scalar> rate_coefficients{};
  constraint::GapActivationParameters activation{};
};

using PenaltyRowInputView = PenaltyRowInputViewT<Real>;

template <typename Scalar>
struct PenaltyRowOutputViewT {
  ArrayView<Scalar> effective_gap{};
  ArrayView<Scalar> pressure{};
  ArrayView<Scalar> row_force{};
  ArrayView<Scalar> row_stiffness{};
  ArrayView<Scalar> row_rate_coefficient{};
};

using PenaltyRowOutputView = PenaltyRowOutputViewT<Real>;

static_assert( std::is_trivially_copyable_v<ParentTraceFieldPairView> );
static_assert( std::is_trivially_copyable_v<ConstraintRowBatchView> );
static_assert( std::is_trivially_copyable_v<PenaltyRowInputView> );
static_assert( std::is_trivially_copyable_v<PenaltyRowOutputView> );

}  // namespace tribol

#endif
