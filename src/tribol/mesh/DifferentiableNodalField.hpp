// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Apply sparse coordinate chain rules to energies of assembled nodal fields.
 */
#ifndef SRC_TRIBOL_MESH_DIFFERENTIABLENODALFIELD_HPP_
#define SRC_TRIBOL_MESH_DIFFERENTIABLENODALFIELD_HPP_

#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <functional>
#include <map>
#include <vector>
#include "tribol/common/BasicTypes.hpp"
#include "shared/math/ParSparseMat.hpp"

namespace tribol {

/**
 * @brief Borrow a nodal field and its sparse coordinate derivative without copying either.
 *
 * Values use scalar field true DOFs; Jacobian rows use those same DOFs and columns use coordinate true DOFs.
 * Both objects must outlive every calculation using this view and retain their MPI partitions.
 */
struct DifferentiableNodalField {
  const shared::ParVector& values;       ///< Borrowed assembled values q(x).
  const shared::ParSparseMat& jacobian;  ///< Borrowed coordinate Jacobian dq/dx.
};

/** @brief First energy partials keyed by field index; absent entries mean zero partial derivatives. */
using NodalEnergyGradient = std::map<IndexT, shared::ParVector>;

/** @brief Energy and nodewise partial derivatives with respect to assembled field values. */
struct NodalEnergyDerivatives {
  RealT energy = 0.0;                                              ///< Globally summed energy.
  NodalEnergyGradient gradient;                                    ///< First partials dE/dq_i on field true DOFs.
  std::map<std::pair<IndexT, IndexT>, shared::ParVector> hessian;  ///< Upper triangle of nodewise d²E/(dq_i dq_j).
};

/**
 * @brief Bind fields calculated together and assemble both terms of the coordinate chain rule.
 *
 * Each energy is a sum of nodewise expressions on one scalar field space. Coordinate Hessians are requested as a
 * weighted sum, never stored separately for every field entry. The callback can evaluate all fields in one pair pass.
 * Field values, Jacobians, spaces, and any callback references must remain valid until assembly finishes. This
 * host-only object retains no geometry. Construct a new binding after replacing mesh data or assembled fields. All
 * operations are collective on the supplied spaces' communicator; their partitions must match the bound matrices.
 */
class DifferentiableNodalFields {
 public:
  /**
   * @brief Calculate sum_i weights_i * d²q_i/dx²; missing field weights mean zero.
   * The operation must return a fresh, owned matrix on coordinate true DOFs each time it is called.
   */
  using WeightedHessian = std::function<shared::ParSparseMat( const NodalEnergyGradient& )>;

  /**
   * @brief Borrow assembled fields and their coordinate derivative operation.
   * @param field_space Scalar space shared by the fields.
   * @param coordinate_space Space defining the coordinate true DOFs.
   * @param fields Value and Jacobian views, indexed in expression-variable order.
   * @param weighted_hessian Optional operation required to assemble a coordinate Jacobian.
   * @throws std::invalid_argument Local dimensions do not match the supplied spaces.
   */
  DifferentiableNodalFields( const mfem::ParFiniteElementSpace& field_space,
                             const mfem::ParFiniteElementSpace& coordinate_space,
                             std::vector<DifferentiableNodalField> fields, WeightedHessian weighted_hessian = {} );

  /** @brief Return the number of bound fields. @return Field count. */
  IndexT size() const { return fields_.size(); }
  /** @brief Read a bound field. @param index Valid field index. @return Borrowed field data. */
  const DifferentiableNodalField& operator[]( IndexT index ) const { return fields_.at( index ); }
  /** @brief Read field-space metadata. @return Borrowed scalar field space. */
  const mfem::ParFiniteElementSpace& fieldSpace() const { return field_space_; }
  /** @brief Allocate a zero field true-DOF vector. @return Owned zero vector. */
  shared::ParVector makeVector() const;

  /**
   * @brief Calculate sum_i (dq_i/dx)^T * dE/dq_i, including an allocated zero for a constant energy.
   * @param derivatives Nodewise energy partials on the field space.
   * @return Owned residual on coordinate true DOFs.
   * @throws std::invalid_argument A partial has an invalid field index or local vector length.
   */
  shared::ParVector assembleForce( const NodalEnergyDerivatives& derivatives ) const;

  /**
   * @brief Combine the weighted field Hessians with sum_ij J_i^T * diag(d²E/(dq_i dq_j)) * J_j.
   * @param derivatives Nodewise partials; only the upper triangle of the symmetric outer Hessian is supplied.
   * @return Owned coordinate Jacobian, including both chain-rule terms.
   * @throws std::logic_error No weighted-Hessian operation was supplied; curvature is never silently omitted.
   * @throws std::invalid_argument Partial indices, vector lengths, or callback matrix dimensions are invalid.
   */
  shared::ParSparseMat assembleJacobian( const NodalEnergyDerivatives& derivatives ) const;

 private:
  /**
   * @brief Check a partial before using its entries or field index.
   * @param index Field index.
   * @param partial Nodal partial derivative vector.
   * @throws std::invalid_argument Index or vector length is invalid.
   */
  void validatePartial( IndexT index, const shared::ParVector& partial ) const;

  const mfem::ParFiniteElementSpace& field_space_;       ///< Borrowed scalar field space.
  const mfem::ParFiniteElementSpace& coordinate_space_;  ///< Borrowed coordinate space.
  std::vector<DifferentiableNodalField> fields_;         ///< Borrowed value/Jacobian pairs.
  WeightedHessian weighted_hessian_;  ///< Shared operation for coordinate curvature of all bound fields.
};

}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_MESH_DIFFERENTIABLENODALFIELD_HPP_
