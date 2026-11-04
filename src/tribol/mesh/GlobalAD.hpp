// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Differentiate distributed vector and sparse matrix arithmetic with respect to coordinate true DOFs.
 */
#ifndef SRC_TRIBOL_MESH_GLOBALAD_HPP_
#define SRC_TRIBOL_MESH_GLOBALAD_HPP_

#include "tribol/config.hpp"
#ifdef BUILD_REDECOMP
#include <concepts>
#include <functional>
#include <memory>
#include <unordered_map>
#include "tribol/mesh/DifferentiableNodalField.hpp"

namespace tribol {
namespace detail {
struct GlobalADVectorNode;
struct GlobalADMatrixData;
struct GlobalADScalarData;
struct GlobalADAccess;
}  // namespace detail
class ADScalar;
class ADMatrix;

/**
 * @brief Distributed vector values with sparse coordinate derivatives composed by arithmetic operators.
 *
 * Operators form an owning graph of intermediate Hypre vectors. Coordinate Jacobians are formed only when requested;
 * weighted Hessians combine operator curvature with supplied field curvature. No Hessian is stored for each vector
 * entry. Spaces must outlive all calculations. Operands share the same coordinate space and pointwise operations use
 * the same value-space object. Calculations are collective, host-only, and require matching graphs on all MPI ranks.
 */
class ADVector {
 public:
  /** @brief Operation returning sum_i weights_i d²q_i/dx² as a fresh owned coordinate matrix. */
  using WeightedHessian = std::function<shared::ParSparseMat( const shared::ParVector& )>;
  /**
   * @brief Own supplied values and derivatives, including an optional custom weighted coordinate Hessian.
   * @param value_space Space defining vector rows.
   * @param coordinate_space Space defining differentiation columns.
   * @param values Owned value snapshot.
   * @param jacobian Owned coordinate Jacobian.
   * @param hessian Weighted curvature operation; required for second derivatives, but not first derivatives.
   * @return Differentiable vector. Callback captures and spaces must remain valid throughout differentiation.
   * @throws std::invalid_argument Value or matrix dimensions do not match the spaces.
   */
  static ADVector fromDerivatives( const mfem::ParFiniteElementSpace& value_space,
                                   const mfem::ParFiniteElementSpace& coordinate_space, shared::ParVector values,
                                   shared::ParSparseMat jacobian, WeightedHessian hessian = {} );
  /**
   * @brief Own an affine vector snapshot with a constant Jacobian and explicitly zero coordinate curvature.
   * @param value_space Vector row space. @param coordinate_space Differentiation space.
   * @param values Owned values. @param jacobian Owned constant coordinate derivative.
   * @return Affine vector. @throws std::invalid_argument Dimensions do not match the supplied spaces.
   */
  static ADVector affine( const mfem::ParFiniteElementSpace& value_space,
                          const mfem::ParFiniteElementSpace& coordinate_space, shared::ParVector values,
                          shared::ParSparseMat jacobian );
  /**
   * @brief Seed coordinate values with the distributed identity Jacobian and zero curvature.
   * @param coordinate_space Independent coordinate space. @param values Owned coordinate snapshot.
   * @return Independent variable. @throws std::invalid_argument Vector length does not match the space.
   */
  static ADVector variable( const mfem::ParFiniteElementSpace& coordinate_space, shared::ParVector values );
  /**
   * @brief Own coefficients held fixed during coordinate differentiation.
   * @param value_space Coefficient space. @param coordinate_space Differentiation space.
   * @param values Owned coefficients. @return Coordinate-independent vector.
   * @throws std::invalid_argument Vector length does not match the space.
   */
  static ADVector constant( const mfem::ParFiniteElementSpace& value_space,
                            const mfem::ParFiniteElementSpace& coordinate_space, shared::ParVector values );
  /** @brief Read current distributed values. @return Borrowed immutable Hypre vector wrapper. */
  const shared::ParVector& values() const;
  /** @brief Read the value space. @return Borrowed row-space metadata. */
  const mfem::ParFiniteElementSpace& space() const;
  /** @brief Read the differentiation space. @return Borrowed coordinate-space metadata. */
  const mfem::ParFiniteElementSpace& coordinateSpace() const;
  /** @brief Calculate dq/dx. @return Owned sparse coordinate Jacobian. */
  shared::ParSparseMat jacobian() const;
  /**
   * @brief Calculate coordinate curvature of a weighted vector sum.
   * @param weights Fixed vector weights on the value space.
   * @return Owned sum_i weights_i d²q_i/dx².
   * @throws std::logic_error A required field curvature operation was not supplied.
   */
  shared::ParSparseMat weightedHessian( const shared::ParVector& weights ) const;

 private:
  /** @brief Wrap an immutable vector calculation. @param node Shared calculation. */
  explicit ADVector( std::shared_ptr<const detail::GlobalADVectorNode> node );
  std::shared_ptr<const detail::GlobalADVectorNode> node_;  ///< Shared immutable vector calculation.
  friend struct detail::GlobalADAccess;
};

/**
 * @brief Sparse matrix values with derivative actions for multiplication by distributed vectors.
 *
 * Active matrix providers return A(x)v and A(x)^T w as ADVectors for fixed v and w. This represents matrix derivatives
 * without materializing third- or fourth-order tensors. Providers and borrowed spaces must remain valid during use.
 * Matrix multiplication includes derivatives of both operands, including mixed second derivatives. All calls are host
 * collective operations. Constant matrices may be used with any compatible coordinate space.
 */
class ADMatrix {
 public:
  /** @brief Return a differentiated matrix action, owning its calculated values and derivatives.
   * Copy any fixed input data needed by later derivative requests rather than retaining argument references. */
  using Action = std::function<ADVector( const shared::ParVector& )>;
  /**
   * @brief Own an active matrix snapshot and derivative actions.
   * @param rows Matrix row space. @param columns Matrix column space. @param coordinates Differentiation space.
   * @param values Owned current matrix. @param apply Differentiated A(x)v with v held fixed.
   * @param transpose_apply Differentiated A(x)^T w with w held fixed.
   * @return Active matrix. Both providers must return matching primal values and compatible ADVector spaces.
   * @throws std::invalid_argument Dimensions or either action are invalid.
   */
  static ADMatrix fromActions( const mfem::ParFiniteElementSpace& rows, const mfem::ParFiniteElementSpace& columns,
                               const mfem::ParFiniteElementSpace& coordinates, shared::ParSparseMat values,
                               Action apply, Action transpose_apply );
  /**
   * @brief Own a coordinate-independent matrix.
   * @param rows Matrix row space. @param columns Matrix column space. @param values Owned matrix.
   * @return Constant matrix. @throws std::invalid_argument Matrix dimensions do not match the spaces.
   */
  static ADMatrix constant( const mfem::ParFiniteElementSpace& rows, const mfem::ParFiniteElementSpace& columns,
                            shared::ParSparseMat values );
  /** @brief Read current matrix values. @return Borrowed immutable Hypre matrix wrapper. */
  const shared::ParSparseMat& values() const;
  /** @brief Read row-space metadata. @return Borrowed row space. */
  const mfem::ParFiniteElementSpace& rowSpace() const;
  /** @brief Read column-space metadata. @return Borrowed column space. */
  const mfem::ParFiniteElementSpace& columnSpace() const;
  /** @brief Transpose values and derivative actions. @return Owned transpose calculation. */
  ADMatrix transpose() const;

 private:
  /** @brief Wrap immutable matrix values and actions. @param data Shared calculation. */
  explicit ADMatrix( std::shared_ptr<const detail::GlobalADMatrixData> data );
  std::shared_ptr<const detail::GlobalADMatrixData> data_;  ///< Shared immutable matrix values and actions.
  friend struct detail::GlobalADAccess;
};

/** @brief Owned derivatives of a globally reduced energy, with first partials for participating vector objects. */
class ADScalarDerivatives {
 public:
  RealT value = 0.0;             ///< Globally reduced scalar value.
  shared::ParVector gradient;    ///< Coordinate gradient on distributed true DOFs.
  shared::ParSparseMat hessian;  ///< Coordinate Hessian; absent for gradient-only evaluation.
  /** @brief Check whether a vector participated in the expression. @param input Vector to query.
   * @return Whether its first-partial vector is available. */
  bool hasPartial( const ADVector& input ) const;
  /** @brief Read a participating vector's first partial. @param input Participating vector.
   * @return Borrowed partial on the input value space. @throws std::out_of_range The input did not participate. */
  const shared::ParVector& partial( const ADVector& input ) const;
  /** @brief Transfer ownership of a first partial. @param input Participating vector.
   * @return Owned partial. @throws std::out_of_range The partial is absent or has already been released. */
  shared::ParVector releasePartial( const ADVector& input );

 private:
  std::unordered_map<std::size_t, shared::ParVector> partials_;  ///< Owned reverse weights keyed by vector identity.
  friend struct detail::GlobalADAccess;
};

/**
 * @brief Globally reduced scalar with distributed coordinate gradient and Hessian calculations.
 *
 * Vector sum/dot create reductions; scalar results support addition, subtraction, and constant scaling. Form nonlinear
 * expressions before reduction to preserve sparse coordinate Hessians. Reduced-scalar coefficients must agree on all
 * MPI ranks. Multiplication/division of two reduced scalars
 * is not provided because it generally creates dense global outer products. Referenced spaces and borrowed field data
 * must remain valid through the last differentiation request. Each request uses fresh scratch storage and retains no
 * calculated derivatives. Use differentiate() when both derivative orders are needed to share one reverse pass.
 */
class ADScalar {
 public:
  /**
   * @brief Own a custom scalar value and its supplied global coordinate derivatives.
   * @param coordinates Independent coordinate space. @param value Globally agreed scalar value.
   * @param gradient Owned coordinate gradient. @param hessian Owned coordinate Hessian.
   * @return Scalar primitive usable in linear combinations; no vector-object partials are supplied by this primitive.
   * @throws std::invalid_argument Derivative dimensions do not match the coordinate space.
   */
  static ADScalar fromDerivatives( const mfem::ParFiniteElementSpace& coordinates, RealT value,
                                   shared::ParVector gradient, shared::ParSparseMat hessian );
  /** @brief Read the global scalar value. @return Globally reduced value. */
  RealT value() const;
  /** @brief Calculate the coordinate gradient. @return Owned distributed gradient; no field Hessians are requested. */
  shared::ParVector gradient() const;
  /** @brief Calculate the coordinate Hessian. @return Owned sparse Hessian. */
  shared::ParSparseMat hessian() const;
  /** @brief Calculate both derivative orders and vector partials in one request.
   * @return Owned global derivatives and reverse weights. @throws std::logic_error Required field curvature is absent.
   */
  ADScalarDerivatives differentiate() const;

 private:
  /** @brief Wrap a scalar reduction. @param data Shared calculation. */
  explicit ADScalar( std::shared_ptr<const detail::GlobalADScalarData> data );
  std::shared_ptr<const detail::GlobalADScalarData> data_;  ///< Shared scalar value and weighted reduction roots.
  friend struct detail::GlobalADAccess;
};

/**
 * @brief Borrow assembled fields as ADVectors, retaining their shared weighted-Hessian calculation.
 * @param fields Bound distributed values, coordinate Jacobians, and grouped curvature operation.
 * @return Vectors sharing one field source; input values, Jacobians, callback captures, and spaces must outlive use.
 */
std::vector<ADVector> makeADFields( const DifferentiableNodalFields& fields );
/** @brief Form a sum of compatible distributed calculations. @param left First operand.
 * @param right Second operand. @return Differentiated sum. */
ADVector operator+( const ADVector& left, const ADVector& right );
/** @brief Form a difference of compatible distributed calculations. @param left First operand.
 * @param right Second operand. @return Differentiated difference. */
ADVector operator-( const ADVector& left, const ADVector& right );
/** @brief Negate a distributed calculation. @param value Operand. @return Negated calculation. */
ADVector operator-( const ADVector& value );
/** @brief Scale a distributed calculation. @param scale Fixed coefficient. @param value Operand. @return Scaled result.
 */
ADVector operator*( RealT scale, const ADVector& value );
/** @brief Scale on the right. @param value Operand. @param scale Fixed coefficient. @return Scaled result. */
ADVector operator*( const ADVector& value, RealT scale );
/** @brief Divide by a nonzero constant. @param value Numerator. @param scale Nonzero denominator.
 * @return Scaled result. @throws std::domain_error The denominator is zero or nonfinite. */
ADVector operator/( const ADVector& value, RealT scale );
/** @brief Form a sum of compatible distributed calculations. @param left First operand.
 * @param right Second operand. @return Differentiated sum. */
ADMatrix operator+( const ADMatrix& left, const ADMatrix& right );
/** @brief Form a difference of compatible distributed calculations. @param left First operand.
 * @param right Second operand. @return Differentiated difference. */
ADMatrix operator-( const ADMatrix& left, const ADMatrix& right );
/** @brief Negate a distributed calculation. @param value Operand. @return Negated calculation. */
ADMatrix operator-( const ADMatrix& value );
/** @brief Scale a distributed calculation. @param scale Fixed coefficient. @param value Operand. @return Scaled result.
 */
ADMatrix operator*( RealT scale, const ADMatrix& value );
/** @brief Scale on the right. @param value Operand. @param scale Fixed coefficient. @return Scaled result. */
ADMatrix operator*( const ADMatrix& value, RealT scale );
/** @brief Divide by a nonzero constant. @param value Numerator. @param scale Nonzero denominator.
 * @return Scaled result. @throws std::domain_error The denominator is zero or nonfinite. */
ADMatrix operator/( const ADMatrix& value, RealT scale );
/** @brief Form a sum of compatible distributed calculations. @param left First operand.
 * @param right Second operand. @return Differentiated sum. */
ADScalar operator+( const ADScalar& left, const ADScalar& right );
/** @brief Form a difference of compatible distributed calculations. @param left First operand.
 * @param right Second operand. @return Differentiated difference. */
ADScalar operator-( const ADScalar& left, const ADScalar& right );
/** @brief Negate a distributed calculation. @param value Operand. @return Negated calculation. */
ADScalar operator-( const ADScalar& value );
/** @brief Scale a distributed calculation. @param scale Fixed coefficient. @param value Operand. @return Scaled result.
 */
ADScalar operator*( RealT scale, const ADScalar& value );
/** @brief Scale on the right. @param value Operand. @param scale Fixed coefficient. @return Scaled result. */
ADScalar operator*( const ADScalar& value, RealT scale );
/** @brief Divide by a nonzero constant. @param value Numerator. @param scale Nonzero denominator.
 * @return Scaled result. @throws std::domain_error The denominator is zero or nonfinite. */
ADScalar operator/( const ADScalar& value, RealT scale );
/** @brief Add or subtract a constant. @param value First operand. @param scalar Second operand. @return sum. */
ADVector operator+( const ADVector& value, RealT scalar );
/** @brief Add or subtract from a constant. @param scalar First operand. @param value Second operand. @return sum. */
ADVector operator+( RealT scalar, const ADVector& value );
/** @brief Add or subtract a constant. @param value First operand. @param scalar Second operand. @return sum. */
ADScalar operator+( const ADScalar& value, RealT scalar );
/** @brief Add or subtract from a constant. @param scalar First operand. @param value Second operand. @return sum. */
ADScalar operator+( RealT scalar, const ADScalar& value );
/** @brief Add or subtract a constant. @param value First operand. @param scalar Second operand. @return difference. */
ADVector operator-( const ADVector& value, RealT scalar );
/** @brief Add or subtract from a constant. @param scalar First operand. @param value Second operand. @return
 * difference. */
ADVector operator-( RealT scalar, const ADVector& value );
/** @brief Add or subtract a constant. @param value First operand. @param scalar Second operand. @return difference. */
ADScalar operator-( const ADScalar& value, RealT scalar );
/** @brief Add or subtract from a constant. @param scalar First operand. @param value Second operand. @return
 * difference. */
ADScalar operator-( RealT scalar, const ADScalar& value );
/** @brief Form a pointwise product. @param left First field. @param right Second field. @return Product field. */
ADVector operator*( const ADVector& left, const ADVector& right );
/** @brief Form a pointwise quotient. @param left Numerator. @param right Nonzero denominator field.
 * @return Quotient. @throws std::domain_error A denominator entry is zero or nonfinite on any rank. */
ADVector operator/( const ADVector& left, const ADVector& right );
/** @brief Divide a constant by a field. @param scalar Numerator. @param value Nonzero denominator field.
 * @return Quotient. @throws std::domain_error A denominator entry is zero or nonfinite on any rank. */
ADVector operator/( RealT scalar, const ADVector& value );
/** @brief Form a sparse matrix product. @param left First matrix. @param right Second matrix. @return Product matrix.
 */
ADMatrix operator*( const ADMatrix& left, const ADMatrix& right );
/** @brief Apply a matrix to an AD vector, including both operands' derivatives.
 * @param matrix Matrix operand. @param vector Vector operand. @return Differentiated matrix-vector product. */
ADVector operator*( const ADMatrix& matrix, const ADVector& vector );
/** @brief Form a matrix with differentiated diagonal entries. @param vector Diagonal values.
 * @return Diagonal matrix, with actions rather than matrix-entry derivative tensors. */
ADMatrix diagonal( const ADVector& vector );
/** @brief Invert supported entries, returning zero value and derivatives elsewhere.
 * @param vector Denominator field. @param tolerance Finite nonnegative unsupported magnitude.
 * @return Reciprocal field. @throws std::invalid_argument The tolerance is invalid.
 * @throws std::domain_error A denominator entry is nonfinite on any rank. */
ADVector reciprocal( const ADVector& vector, RealT tolerance );
/** @brief Select the smaller entry and its derivatives, retaining the first operand at equality.
 * @param left First field. @param right Second field. @return Selected field. */
ADVector min( const ADVector& left, const ADVector& right );
/** @brief Select against a constant, retaining field derivatives at equality.
 * @param vector Field. @param bound Constant. @return Selected field. */
ADVector min( const ADVector& vector, RealT bound );
/** @brief Select against a field, retaining constant derivatives at equality.
 * @param bound Constant. @param vector Field. @return Selected field. */
ADVector min( RealT bound, const ADVector& vector );
/** @brief Select the larger entry and its derivatives, retaining the first operand at equality.
 * @param left First field. @param right Second field. @return Selected field. */
ADVector max( const ADVector& left, const ADVector& right );
/** @brief Select against a constant, retaining field derivatives at equality.
 * @param vector Field. @param bound Constant. @return Selected field. */
ADVector max( const ADVector& vector, RealT bound );
/** @brief Select against a field, retaining constant derivatives at equality.
 * @param bound Constant. @param vector Field. @return Selected field. */
ADVector max( RealT bound, const ADVector& vector );
/** @brief Sum vector entries across MPI ranks. @param vector Distributed values. @return Global scalar calculation. */
ADScalar sum( const ADVector& vector );
/** @brief Sum a pointwise product across ranks. @param left First field. @param right Second field.
 * @return Global dot product and its derivative calculation. */
ADScalar dot( const ADVector& left, const ADVector& right );
/** @brief Reduce a product of multiple fields. @tparam Rest Remaining field types.
 * @param first First field. @param second Second field. @param rest Remaining fields. @return Global product sum. */
template <typename... Rest>
  requires( std::same_as<Rest, ADVector> && ... )
ADScalar sum( const ADVector& first, const ADVector& second, const Rest&... rest )
{
  return sum( ( ( first * second ) * ... * rest ) );
}

}  // namespace tribol
#endif  // BUILD_REDECOMP
#endif  // SRC_TRIBOL_MESH_GLOBALAD_HPP_
