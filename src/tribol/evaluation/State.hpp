#ifndef TRIBOL_EVALUATION_STATE_HPP_
#define TRIBOL_EVALUATION_STATE_HPP_

#include "tribol/core/MeshView.hpp"
#include "tribol/core/Version.hpp"

#include <limits>

namespace tribol {

template <typename Scalar>
struct ContactStateViewT {
  FieldView<const Scalar> mortar_velocity{};
  FieldView<const Scalar> nonmortar_velocity{};
  FieldView<const Scalar> mortar_reference_coordinates{};
  FieldView<const Scalar> nonmortar_reference_coordinates{};
  ArrayView<const Scalar> mortar_element_thickness{};
  ArrayView<const Scalar> nonmortar_element_thickness{};
  ArrayView<const Scalar> mortar_material_modulus{};
  ArrayView<const Scalar> nonmortar_material_modulus{};
  ArrayView<const Scalar> multiplier{};
  ArrayView<const Scalar> external_potential_density{};
  ArrayView<const Scalar> external_pressure{};
  ArrayView<const Scalar> external_pressure_tangent{};
};

using ContactStateView = ContactStateViewT<Real>;

struct ContactStateDirectionView {
  FieldView<const Real> mortar_velocity{};
  FieldView<const Real> nonmortar_velocity{};
  FieldView<const Real> mortar_reference_coordinates{};
  FieldView<const Real> nonmortar_reference_coordinates{};
  ArrayView<const Real> mortar_element_thickness{};
  ArrayView<const Real> nonmortar_element_thickness{};
  ArrayView<const Real> mortar_material_modulus{};
  ArrayView<const Real> nonmortar_material_modulus{};
  ArrayView<const Real> multiplier{};
  ArrayView<const Real> external_potential_density{};
  ArrayView<const Real> external_pressure{};
  ArrayView<const Real> external_pressure_tangent{};
};

template <typename Scalar>
struct ContactResidualViewT {
  FieldView<Scalar> mortar{};
  FieldView<Scalar> nonmortar{};
  ArrayView<Scalar> constraint{};
};

using ContactResidualView = ContactResidualViewT<Real>;

template <typename Scalar>
struct ContactOutputViewT {
  ContactResidualViewT<Scalar> residual{};
  ArrayView<Scalar> gap{};
  ArrayView<Scalar> weighted_gap{};
  ArrayView<Scalar> tributary_area{};
  ArrayView<Scalar> mortar_weights{};
  ArrayView<Scalar> mortar_mass_weights{};
  ArrayView<Scalar> quadrature_gap{};
  ArrayView<Scalar> quadrature_pressure{};
  ArrayView<Scalar> pressure{};
};

using ContactOutputView = ContactOutputViewT<Real>;

struct ContactDirectionView {
  FieldView<const Real> mortar{};
  FieldView<const Real> nonmortar{};
};

struct ContactLinearizationDirectionView {
  ContactDirectionView coordinates{};
  ContactStateDirectionView state{};
};

struct DenseMatrixView {
  ArrayView<const Real> values{};
  Index rows{};
  Index columns{};

  [[nodiscard]] constexpr Real operator()( Index row, Index column ) const { return values[row * columns + column]; }
};

struct CsrMatrixView {
  ArrayView<const Index> row_offsets{};
  ArrayView<const Index> column_indices{};
  ArrayView<const Real> values{};
  Index rows{};
  Index columns{};

  [[nodiscard]] constexpr Index numberOfNonzeros() const { return values.size(); }
};

struct EvaluationSummary {
  Real energy{};
  Real timestep_vote{ std::numeric_limits<Real>::infinity() };
  Index active_interactions{};
  Index quadrature_points{};
};

struct NodalKinematicsView {
  ArrayView<const Real> gap{};
  ArrayView<const Real> weighted_gap{};
  ArrayView<const Real> tributary_area{};
  GeometryVersion geometry_version{};
  InteractionVersion interaction_version{};
};

struct ContactResultView {
  FieldView<const Real> mortar_force{};
  FieldView<const Real> nonmortar_force{};
  ArrayView<const Real> constraint_residual{};
  ArrayView<const Real> gap{};
  ArrayView<const Real> weighted_gap{};
  ArrayView<const Real> tributary_area{};
  ArrayView<const Real> mortar_weights{};
  ArrayView<const Real> mortar_mass_weights{};
  ArrayView<const Real> quadrature_gap{};
  ArrayView<const Real> quadrature_pressure{};
  ArrayView<const Real> pressure{};
  EvaluationSummary summary{};
  GeometryVersion geometry_version{};
  InteractionVersion interaction_version{};
};

}  // namespace tribol

#endif
