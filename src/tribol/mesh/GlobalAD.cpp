// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/GlobalAD.hpp"
#ifdef BUILD_REDECOMP
#include <atomic>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace tribol {
namespace detail {
struct GlobalADEvaluation;
/** @brief Share assembled field curvature across vector leaves. */
struct GlobalADSource {
  const mfem::ParFiniteElementSpace* coordinates;                      ///< Borrowed differentiation space.
  std::vector<std::shared_ptr<const shared::ParSparseMat>> jacobians;  ///< Owned or borrowed field Jacobians.
  DifferentiableNodalFields::WeightedHessian hessian;  ///< One weighted Hessian operation for these fields.
};
/** @brief Describe a first derivative from an operator output to one vector operand. */
struct GlobalADEdge {
  std::shared_ptr<const GlobalADVectorNode> parent;  ///< Operand calculation.
  RealT scale = 1.0;  ///< Scalar derivative, used when neither diagonal nor matrix is supplied.
  std::shared_ptr<const shared::ParVector> diagonal;   ///< Pointwise derivative coefficients, if present.
  std::shared_ptr<const shared::ParSparseMat> matrix;  ///< General sparse linear derivative, if present.
};
/** @brief Own an intermediate vector and its immediate derivative rules. */
struct GlobalADVectorNode {
  std::size_t id;                                   ///< Process-local identity used for derivatives of shared operands.
  const mfem::ParFiniteElementSpace* space;         ///< Borrowed vector-value space.
  const mfem::ParFiniteElementSpace* coordinates;   ///< Borrowed independent coordinate space.
  std::shared_ptr<const shared::ParVector> values;  ///< Owned or borrowed primal values.
  bool active = false;                              ///< Whether coordinate derivatives may be nonzero.
  std::shared_ptr<const GlobalADSource> source;     ///< Shared supplied field derivatives for a primitive input.
  IndexT source_index = 0;                          ///< Field index within source.
  std::vector<GlobalADEdge> edges;                  ///< Immediate first derivative rules in operand order.
  std::function<void( const shared::ParVector&, GlobalADEvaluation&, shared::ParSparseMat& )> curvature;
  ///< Add this operator's second-derivative terms with operands' first coordinate derivatives.
};
/** @brief Own matrix values and derivative actions on fixed vectors. */
struct GlobalADMatrixData {
  const mfem::ParFiniteElementSpace* rows;                   ///< Borrowed row space.
  const mfem::ParFiniteElementSpace* columns;                ///< Borrowed column space.
  const mfem::ParFiniteElementSpace* coordinates = nullptr;  ///< Active coordinate space, null for a constant matrix.
  std::shared_ptr<const shared::ParSparseMat> values;        ///< Owned primal matrix.
  ADMatrix::Action apply;                                    ///< A(x)v with v held fixed.
  ADMatrix::Action transpose_apply;                          ///< A(x)^T w with w held fixed.
};
/** @brief Associate a globally summed vector with its scalar coefficient. */
struct GlobalADReduction {
  std::shared_ptr<const GlobalADVectorNode> root;  ///< Vector whose entries are summed.
  RealT scale;                                     ///< Fixed multiplier of the reduction.
};
/** @brief Hold supplied global scalar derivatives without a vector expression. */
struct GlobalADScalarSource {
  std::shared_ptr<const shared::ParVector> gradient;    ///< Owned custom coordinate gradient.
  std::shared_ptr<const shared::ParSparseMat> hessian;  ///< Owned custom coordinate Hessian.
  RealT scale;                                          ///< Fixed coefficient in a scalar linear combination.
};
/** @brief Own a scalar value and the reductions needed for its derivatives. */
struct GlobalADScalarData {
  RealT value;                                     ///< Globally reduced value, including constant offsets.
  const mfem::ParFiniteElementSpace* coordinates;  ///< Borrowed differentiation space.
  std::vector<GlobalADScalarSource> sources;       ///< Supplied scalar derivatives in deterministic operand order.
  std::vector<GlobalADReduction> roots;            ///< Weighted reductions, preserving operand order.
};
/** @brief Access opaque AD storage only within this implementation. */
struct GlobalADAccess {
  /** @brief Read vector storage. @param value Public vector. @return Shared calculation. */
  static const auto& node( const ADVector& value ) { return value.node_; }
  /** @brief Read matrix storage. @param value Public matrix. @return Shared calculation. */
  static const auto& data( const ADMatrix& value ) { return value.data_; }
  /** @brief Read scalar storage. @param value Public scalar. @return Shared calculation. */
  static const auto& data( const ADScalar& value ) { return value.data_; }
  /** @brief Wrap a vector calculation. @param value Internal node. @return Public vector. */
  static ADVector vector( std::shared_ptr<const GlobalADVectorNode> value ) { return ADVector( std::move( value ) ); }
  /** @brief Wrap matrix storage. @param value Internal matrix. @return Public matrix. */
  static ADMatrix matrix( std::shared_ptr<const GlobalADMatrixData> value ) { return ADMatrix( std::move( value ) ); }
  /** @brief Wrap scalar storage. @param value Internal scalar. @return Public scalar. */
  static ADScalar scalar( std::shared_ptr<const GlobalADScalarData> value ) { return ADScalar( std::move( value ) ); }
  /** @brief Access owned vector partials. @param value Derivative result. @return Mutable partial map. */
  static auto& partials( ADScalarDerivatives& value ) { return value.partials_; }
};
}  // namespace detail

namespace {
using Node = detail::GlobalADVectorNode;
using NodePtr = std::shared_ptr<const Node>;
using Access = detail::GlobalADAccess;
using Space = mfem::ParFiniteElementSpace;
using Vector = shared::ParVector;
using Matrix = shared::ParSparseMat;
/** @brief Supply process-local graph identities. */
std::atomic<std::size_t> next_identity{ 0 };

/** @brief Allocate a filled true-DOF vector. @param space Vector space. @param value Fill value. @return Owned vector.
 */
Vector filled( const Space& space, RealT value )
{
  Vector result( const_cast<Space*>( &space ) );
  result.fill( value );
  return result;
}
/** @brief Construct a distributed diagonal. @param space Row/column space. @param values Local diagonal.
 * @return Owned matrix. */
Matrix makeDiagonal( const Space& space, const Vector& values )
{
  return Matrix::diagonalMatrix( space.GetComm(), space.GlobalTrueVSize(), space.GetTrueDofOffsets(), values.get() );
}
/** @brief Allocate an empty rectangular derivative. @param rows Row space. @param columns Column space.
 * @return Owned zero matrix with valid MPI partitions. */
Matrix zeroMatrix( const Space& rows, const Space& columns )
{
  mfem::SparseMatrix local( rows.GetTrueVSize(), columns.GetTrueVSize() );
  local.Finalize();
  return Matrix( rows.GetComm(), rows.GlobalTrueVSize(), columns.GlobalTrueVSize(), rows.GetTrueDofOffsets(),
                 columns.GetTrueDofOffsets(), std::move( local ) );
}
/** @brief Copy a matrix into independent storage. @param matrix Matrix snapshot. @return Owned deep copy. */
Matrix copyMatrix( const Matrix& matrix ) { return Matrix( std::make_unique<mfem::HypreParMatrix>( matrix.get() ) ); }
/** @brief Check communicator rank ordering before distributed algebra.
 * @param left First communicator. @param right Second communicator.
 * @throws std::invalid_argument Communicators do not have the same ranks in the same order. */
void requireCommunicators( MPI_Comm left, MPI_Comm right )
{
  int relation = MPI_UNEQUAL;
  MPI_Comm_compare( left, right, &relation );
  if ( relation != MPI_IDENT && relation != MPI_CONGRUENT )
    throw std::invalid_argument( "Global AD spaces must use matching MPI communicators." );
}
/** @brief Check vector dimensions. @param values Vector. @param space Expected space.
 * @throws std::invalid_argument Local/global sizes differ. */
void validateVector( const Vector& values, const Space& space )
{
  requireCommunicators( values.get().GetComm(), space.GetComm() );
  if ( values.size() != space.GetTrueVSize() || values.get().GlobalSize() != space.GlobalTrueVSize() )
    throw std::invalid_argument( "Global AD vector dimensions must match its value space." );
}
/** @brief Check matrix dimensions. @param values Matrix. @param rows Row space. @param columns Column space.
 * @throws std::invalid_argument Local/global dimensions differ. */
void validateMatrix( const Matrix& values, const Space& rows, const Space& columns )
{
  requireCommunicators( rows.GetComm(), columns.GetComm() );
  requireCommunicators( values.get().GetComm(), rows.GetComm() );
  if ( values.height() != rows.GetTrueVSize() || values.width() != columns.GetTrueVSize() ||
       values.get().M() != rows.GlobalTrueVSize() || values.get().N() != columns.GlobalTrueVSize() )
    throw std::invalid_argument( "Global AD matrix dimensions must match its row and column spaces." );
}
/** @brief Require identical independent-coordinate ordering. @param left First space. @param right Second space.
 * @throws std::invalid_argument The spaces are distinct. */
void requireCoordinates( const Space& left, const Space& right )
{
  if ( &left != &right ) throw std::invalid_argument( "Global AD operands must use the same coordinate-space object." );
}
/** @brief Require compatible pointwise operands. @param left First vector. @param right Second vector.
 * @throws std::invalid_argument Value or coordinate spaces differ. */
void requireVectors( const ADVector& left, const ADVector& right )
{
  requireCoordinates( left.coordinateSpace(), right.coordinateSpace() );
  if ( &left.space() != &right.space() )
    throw std::invalid_argument( "Pointwise AD operands must use the same value-space object." );
}
/** @brief Create a vector calculation. @param space Value space. @param coordinates Coordinate space.
 * @param values Immutable primal storage. @return Mutable node ready for derivative rules. */
std::shared_ptr<Node> node( const Space& space, const Space& coordinates, std::shared_ptr<const Vector> values )
{
  validateVector( *values, space );
  requireCommunicators( space.GetComm(), coordinates.GetComm() );
  auto result = std::make_shared<Node>();
  result->id = next_identity.fetch_add( 1, std::memory_order_relaxed );
  result->space = &space;
  result->coordinates = &coordinates;
  result->values = std::move( values );
  return result;
}
/** @brief Own a vector calculation's values. @param space Value space. @param coordinates Coordinate space.
 * @param values Owned values. @return Mutable node. */
std::shared_ptr<Node> node( const Space& space, const Space& coordinates, Vector values )
{
  return node( space, coordinates, std::make_shared<Vector>( std::move( values ) ) );
}
/** @brief Make a constant on a vector's space. @param like Space provider. @param value Constant value.
 * @return Coordinate-independent broadcast. */
ADVector broadcast( const ADVector& like, RealT value )
{
  return ADVector::constant( like.space(), like.coordinateSpace(), filled( like.space(), value ) );
}
/** @brief Apply an edge transpose to reverse weights. @param edge First derivative rule. @param weights Output weights.
 * @return Owned operand weights. */
Vector transposeWeights( const detail::GlobalADEdge& edge, const Vector& weights )
{
  if ( edge.matrix ) return weights * *edge.matrix;
  if ( edge.diagonal ) return weights.multiply( *edge.diagonal );
  return weights * edge.scale;
}
/** @brief Apply an edge to coordinate derivatives. @param edge First derivative rule. @param jacobian Operand
 * derivative.
 * @return Owned output derivative contribution. */
Matrix applyJacobian( const detail::GlobalADEdge& edge, const Matrix& jacobian )
{
  if ( edge.matrix ) return *edge.matrix * jacobian;
  if ( edge.diagonal ) return makeDiagonal( *edge.parent->space, *edge.diagonal ) * jacobian;
  return jacobian * edge.scale;
}
/** @brief Collect each shared operand before its users. @param current Current calculation.
 * @param[in,out] seen Visited identities. @param[in,out] order Deterministic postorder across matching MPI graphs. */
void visit( const NodePtr& current, std::unordered_set<std::size_t>& seen, std::vector<NodePtr>& order )
{
  if ( !seen.insert( current->id ).second ) return;
  for ( const auto& edge : current->edges ) visit( edge.parent, seen, order );
  order.push_back( current );
}
/** @brief Reject invalid fixed scalar denominators. @param value Scalar denominator.
 * @throws std::domain_error The denominator is zero or nonfinite. */
void requireDivisor( RealT value )
{
  if ( value == 0.0 || !std::isfinite( value ) )
    throw std::domain_error( "AD division requires a finite nonzero denominator." );
}
}  // namespace

namespace detail {
/** @brief Represent a coordinate Jacobian as a row scaling of another sparse Jacobian. */
struct GlobalADRowJacobian {
  NodePtr owner;                ///< Retain a provider while its Jacobian is borrowed.
  const Matrix* base;           ///< Matrix borrowed from supplied inputs or evaluation scratch.
  std::optional<Vector> scale;  ///< Fused pointwise derivative factors; absent means identity scaling.
};
/** @brief Share derivative scratch during one global calculation. */
struct GlobalADEvaluation {
  std::unordered_map<std::size_t, Matrix> jacobians;  ///< Owned coordinate Jacobians, calculated on demand.
  std::unordered_map<std::size_t, GlobalADRowJacobian>
      row_jacobians;  ///< Fused row scalings for pointwise operator chains.
  /** @brief Read or calculate a coordinate Jacobian. @param value Vector node. @return Borrowed matrix valid in this
   * request. */
  const Matrix& jacobian( const NodePtr& value )
  {
    if ( value->source ) return *value->source->jacobians.at( value->source_index );
    if ( auto found = jacobians.find( value->id ); found != jacobians.end() ) return found->second;
    std::optional<Matrix> result;
    for ( const auto& edge : value->edges ) {
      if ( !edge.parent->active ) continue;
      auto part = applyJacobian( edge, jacobian( edge.parent ) );
      if ( result )
        *result += part;
      else
        result = std::move( part );
    }
    if ( !result ) result = zeroMatrix( *value->space, *value->coordinates );
    return jacobians.emplace( value->id, std::move( *result ) ).first->second;
  }
  /** @brief Fuse diagonal derivatives without forming intermediate sparse products. @param value Vector node.
   * @return Row-scaled Jacobian, valid for this request. */
  const GlobalADRowJacobian& rows( const NodePtr& value )
  {
    if ( auto found = row_jacobians.find( value->id ); found != row_jacobians.end() ) return found->second;
    GlobalADRowJacobian result{};
    result.owner = value;
    const GlobalADEdge* single = nullptr;
    for ( const auto& edge : value->edges ) {
      if ( !edge.parent->active ) continue;
      if ( single ) {
        single = nullptr;
        break;
      }
      single = &edge;
    }
    if ( single && !single->matrix ) {
      const auto& edge = *single;
      const auto& parent = rows( edge.parent );
      result.base = parent.base;
      if ( edge.diagonal )
        result.scale = parent.scale ? parent.scale->multiply( *edge.diagonal ) : Vector( *edge.diagonal );
      else if ( parent.scale )
        result.scale = *parent.scale * edge.scale;
      else if ( edge.scale != 1.0 )
        result.scale = filled( *value->space, edge.scale );
    } else
      result.base = &jacobian( value );
    return row_jacobians.emplace( value->id, std::move( result ) ).first->second;
  }
  /** @brief Add J_left^T diag(weights) J_right and optionally its transpose, fusing pointwise derivative factors.
   * @param[in,out] result Coordinate Hessian. @param left First operand. @param right Second operand.
   * @param weights Local second-partial weights. @param symmetric_pair Also add the transposed mixed product. */
  void addOuter( Matrix& result, const NodePtr& left, const NodePtr& right, const Vector& weights, bool symmetric_pair )
  {
    if ( !left->active || !right->active ) return;
    const auto& a = rows( left );
    const auto& b = rows( right );
    auto coefficients = Vector( weights );
    if ( a.scale ) coefficients.multiplyInPlace( *a.scale );
    if ( b.scale ) coefficients.multiplyInPlace( *b.scale );
    const auto diagonal = makeDiagonal( *left->space, coefficients );
    if ( a.base == b.base )
      result += Matrix::rap( diagonal, *a.base ) * ( symmetric_pair ? 2.0 : 1.0 );
    else {
      auto contribution = Matrix::rap( *a.base, diagonal, *b.base );
      if ( symmetric_pair ) {
        auto transpose = contribution.transpose();
        result += transpose;
      }
      result += contribution;
    }
  }
};
}  // namespace detail

namespace {
/** @brief Reverse all reductions once and assemble coordinate derivatives.
 * @param scalar Reduction roots. @param second Whether to assemble a Hessian.
 * @return Owned gradient, optional Hessian, and vector partials. */
ADScalarDerivatives evaluateScalar( const detail::GlobalADScalarData& scalar, bool second )
{
  std::vector<NodePtr> order;
  std::unordered_set<std::size_t> seen;
  std::unordered_map<std::size_t, Vector> weights;
  for ( const auto& root : scalar.roots ) {
    visit( root.root, seen, order );
    auto seed = filled( *root.root->space, root.scale );
    auto found = weights.find( root.root->id );
    if ( found == weights.end() )
      weights.emplace( root.root->id, std::move( seed ) );
    else
      found->second += seed;
  }
  for ( auto i = order.rbegin(); i != order.rend(); ++i ) {
    const auto& value = *i;
    const auto& weight = weights.at( value->id );
    for ( const auto& edge : value->edges ) {
      auto contribution = transposeWeights( edge, weight );
      auto found = weights.find( edge.parent->id );
      if ( found == weights.end() )
        weights.emplace( edge.parent->id, std::move( contribution ) );
      else
        found->second += contribution;
    }
  }
  ADScalarDerivatives result;
  result.value = scalar.value;
  std::optional<Vector> gradient;
  const auto add_gradient = [&]( Vector contribution ) {
    if ( gradient )
      *gradient += contribution;
    else
      gradient = std::move( contribution );
  };
  for ( const auto& source : scalar.sources ) add_gradient( *source.gradient * source.scale );
  struct SourceWeights {
    std::shared_ptr<const detail::GlobalADSource> source;  ///< Shared derivative provider.
    NodalEnergyGradient weights;  ///< Accumulated weights for all participating fields in this provider.
  };
  std::vector<SourceWeights> sources;
  std::unordered_map<const detail::GlobalADSource*, std::size_t> source_indices;
  for ( const auto& value : order ) {
    if ( !value->source ) continue;
    const auto& weight = weights.at( value->id );
    add_gradient( weight * *value->source->jacobians.at( value->source_index ) );
    if ( !second ) continue;
    auto [entry, inserted] = source_indices.emplace( value->source.get(), sources.size() );
    if ( inserted ) sources.push_back( { value->source, {} } );
    auto& partials = sources[entry->second].weights;
    auto found = partials.find( value->source_index );
    if ( found == partials.end() )
      partials.emplace( value->source_index, weight );
    else
      found->second += weight;
  }
  result.gradient = gradient ? std::move( *gradient ) : filled( *scalar.coordinates, 0.0 );
  if ( second ) {
    std::optional<Matrix> hessian;
    const auto add_hessian = [&]( Matrix contribution ) {
      if ( hessian )
        *hessian += contribution;
      else
        hessian = std::move( contribution );
    };
    for ( const auto& source : scalar.sources ) add_hessian( *source.hessian * source.scale );
    for ( const auto& source : sources ) {
      if ( !source.source->hessian )
        throw std::logic_error( "Global AD requires the supplied fields' weighted Hessian." );
      auto contribution = source.source->hessian( source.weights );
      validateMatrix( contribution, *scalar.coordinates, *scalar.coordinates );
      add_hessian( std::move( contribution ) );
    }
    detail::GlobalADEvaluation evaluation;
    for ( const auto& value : order )
      if ( value->curvature ) {
        if ( !hessian ) hessian = zeroMatrix( *scalar.coordinates, *scalar.coordinates );
        value->curvature( weights.at( value->id ), evaluation, *hessian );
      }
    result.hessian = hessian ? std::move( *hessian ) : zeroMatrix( *scalar.coordinates, *scalar.coordinates );
  }
  Access::partials( result ) = std::move( weights );
  return result;
}
/** @brief Attach a scalar linear combination. @param left First scalar. @param right Second scalar.
 * @param right_scale Coefficient of the second scalar. @return Combined reduction graph. */
ADScalar combine( const ADScalar& left, const ADScalar& right, RealT right_scale )
{
  const auto& a = Access::data( left );
  const auto& b = Access::data( right );
  requireCoordinates( *a->coordinates, *b->coordinates );
  auto result = std::make_shared<detail::GlobalADScalarData>( *a );
  result->value += right_scale * b->value;
  for ( auto root : b->roots ) {
    root.scale *= right_scale;
    result->roots.push_back( std::move( root ) );
  }
  for ( auto source : b->sources ) {
    source.scale *= right_scale;
    result->sources.push_back( std::move( source ) );
  }
  return Access::scalar( std::move( result ) );
}
/** @brief Apply a constant value shift to a vector. @param vector Input. @param shift Fixed shift.
 * @return Shifted vector with unchanged coordinate derivatives. */
ADVector shift( const ADVector& vector, RealT shift )
{
  Vector values( vector.values() );
  auto* data = values.get().HostReadWrite();
  for ( int i = 0; i < values.size(); ++i ) data[i] += shift;
  auto output = node( vector.space(), vector.coordinateSpace(), std::move( values ) );
  output->active = Access::node( vector )->active;
  output->edges.push_back( { Access::node( vector ), 1.0, {}, {} } );
  return Access::vector( std::move( output ) );
}
/** @brief Form a reciprocal with strict or supported-denominator behavior.
 * @param vector Denominator values. @param tolerance Unsupported magnitude.
 * @param guarded Return zero on unsupported finite values when true; otherwise reject exact zero.
 * @return Reciprocal calculation. @throws std::domain_error Invalid denominator data occurs on any rank. */
ADVector inverse( const ADVector& vector, RealT tolerance, bool guarded )
{
  if ( !std::isfinite( tolerance ) || tolerance < 0.0 )
    throw std::invalid_argument( "AD reciprocal tolerance must be finite and nonnegative." );
  const auto* input = vector.values().get().HostRead();
  int invalid = 0, global_invalid = 0;
  for ( int i = 0; i < vector.values().size(); ++i )
    invalid = invalid || !std::isfinite( input[i] ) || ( !guarded && input[i] == 0.0 );
  MPI_Allreduce( &invalid, &global_invalid, 1, MPI_INT, MPI_MAX, vector.space().GetComm() );
  if ( global_invalid )
    throw std::domain_error( "AD reciprocal requires finite, supported denominators on every rank." );
  auto values = filled( vector.space(), 0.0 ), first = filled( vector.space(), 0.0 ),
       second = filled( vector.space(), 0.0 );
  auto* v = values.get().HostReadWrite();
  auto* f = first.get().HostReadWrite();
  auto* s = second.get().HostReadWrite();
  for ( int i = 0; i < values.size(); ++i )
    if ( !guarded || std::abs( input[i] ) > tolerance ) {
      v[i] = 1.0 / input[i];
      f[i] = -v[i] * v[i];
      s[i] = -2.0 * f[i] * v[i];
    }
  auto parent = Access::node( vector );
  auto output = node( vector.space(), vector.coordinateSpace(), std::move( values ) );
  output->active = parent->active;
  output->edges.push_back( { parent, 1.0, std::make_shared<Vector>( std::move( first ) ), {} } );
  if ( parent->active )
    output->curvature = [parent, second = std::make_shared<Vector>( std::move( second ) )](
                            const Vector& weights, detail::GlobalADEvaluation& evaluation, Matrix& result ) {
      evaluation.addOuter( result, parent, parent, weights.multiply( *second ), false );
    };
  return Access::vector( std::move( output ) );
}
}  // namespace

ADVector::ADVector( std::shared_ptr<const detail::GlobalADVectorNode> value ) : node_( std::move( value ) ) {}
ADMatrix::ADMatrix( std::shared_ptr<const detail::GlobalADMatrixData> value ) : data_( std::move( value ) ) {}
ADScalar::ADScalar( std::shared_ptr<const detail::GlobalADScalarData> value ) : data_( std::move( value ) ) {}
const Vector& ADVector::values() const { return *node_->values; }
const Space& ADVector::space() const { return *node_->space; }
const Space& ADVector::coordinateSpace() const { return *node_->coordinates; }
const Matrix& ADMatrix::values() const { return *data_->values; }
const Space& ADMatrix::rowSpace() const { return *data_->rows; }
const Space& ADMatrix::columnSpace() const { return *data_->columns; }
ADScalar ADScalar::fromDerivatives( const Space& coordinates, RealT value, Vector gradient, Matrix hessian )
{
  validateVector( gradient, coordinates );
  validateMatrix( hessian, coordinates, coordinates );
  auto data = std::make_shared<detail::GlobalADScalarData>();
  data->value = value;
  data->coordinates = &coordinates;
  data->sources.push_back(
      { std::make_shared<Vector>( std::move( gradient ) ), std::make_shared<Matrix>( std::move( hessian ) ), 1.0 } );
  return Access::scalar( std::move( data ) );
}
RealT ADScalar::value() const { return data_->value; }
Vector ADScalar::gradient() const { return std::move( evaluateScalar( *data_, false ).gradient ); }
Matrix ADScalar::hessian() const { return std::move( evaluateScalar( *data_, true ).hessian ); }
ADScalarDerivatives ADScalar::differentiate() const { return evaluateScalar( *data_, true ); }
bool ADScalarDerivatives::hasPartial( const ADVector& input ) const
{
  return partials_.contains( Access::node( input )->id );
}
const Vector& ADScalarDerivatives::partial( const ADVector& input ) const
{
  return partials_.at( Access::node( input )->id );
}
Vector ADScalarDerivatives::releasePartial( const ADVector& input )
{
  auto found = partials_.find( Access::node( input )->id );
  if ( found == partials_.end() ) throw std::out_of_range( "The AD vector partial is absent or already released." );
  Vector result = std::move( found->second );
  partials_.erase( found );
  return result;
}
Matrix ADVector::jacobian() const
{
  detail::GlobalADEvaluation evaluation;
  const auto& result = evaluation.jacobian( node_ );
  if ( node_->source ) return copyMatrix( result );
  return std::move( evaluation.jacobians.at( node_->id ) );
}
Matrix ADVector::weightedHessian( const Vector& weights ) const
{
  validateVector( weights, space() );
  // A fixed weighted sum has the same Hessian as the corresponding dot product.
  return dot( *this, constant( space(), coordinateSpace(), weights ) ).hessian();
}
ADVector ADVector::fromDerivatives( const Space& value_space, const Space& coordinates, Vector values, Matrix jacobian,
                                    WeightedHessian hessian )
{
  validateMatrix( jacobian, value_space, coordinates );
  auto source = std::make_shared<detail::GlobalADSource>();
  source->coordinates = &coordinates;
  source->jacobians.push_back( std::make_shared<Matrix>( std::move( jacobian ) ) );
  if ( hessian )
    source->hessian = [hessian = std::move( hessian )]( const NodalEnergyGradient& weights ) {
      return hessian( weights.at( 0 ) );
    };
  auto output = node( value_space, coordinates, std::move( values ) );
  output->active = true;
  output->source = std::move( source );
  return Access::vector( std::move( output ) );
}
ADVector ADVector::affine( const Space& value_space, const Space& coordinates, Vector values, Matrix jacobian )
{
  return fromDerivatives( value_space, coordinates, std::move( values ), std::move( jacobian ),
                          [&coordinates]( const Vector& ) { return zeroMatrix( coordinates, coordinates ); } );
}
ADVector ADVector::variable( const Space& coordinates, Vector values )
{
  const auto ones = filled( coordinates, 1.0 );
  return affine( coordinates, coordinates, std::move( values ), makeDiagonal( coordinates, ones ) );
}
ADVector ADVector::constant( const Space& value_space, const Space& coordinates, Vector values )
{
  return Access::vector( node( value_space, coordinates, std::move( values ) ) );
}
std::vector<ADVector> makeADFields( const DifferentiableNodalFields& fields )
{
  auto source = std::make_shared<detail::GlobalADSource>();
  source->coordinates = &fields.coordinateSpace();
  source->hessian = [fields]( const NodalEnergyGradient& weights ) {
    return fields.weightedCoordinateHessian( weights );
  };
  std::vector<ADVector> result;
  result.reserve( fields.size() );
  for ( IndexT i = 0; i < fields.size(); ++i ) {
    validateMatrix( fields[i].jacobian, fields.fieldSpace(), fields.coordinateSpace() );
    source->jacobians.emplace_back( &fields[i].jacobian, []( const Matrix* ) {} );
    auto values = std::shared_ptr<const Vector>( &fields[i].values, []( const Vector* ) {} );
    auto leaf = node( fields.fieldSpace(), fields.coordinateSpace(), std::move( values ) );
    leaf->active = true;
    leaf->source = source;
    leaf->source_index = i;
    result.push_back( Access::vector( std::move( leaf ) ) );
  }
  return result;
}
ADVector operator+( const ADVector& left, const ADVector& right )
{
  requireVectors( left, right );
  auto result = node( left.space(), left.coordinateSpace(), left.values() + right.values() );
  auto a = Access::node( left ), b = Access::node( right );
  result->active = a->active || b->active;
  result->edges = { { a, 1.0, {}, {} }, { b, 1.0, {}, {} } };
  return Access::vector( std::move( result ) );
}
ADVector operator*( RealT scale, const ADVector& value )
{
  auto result = node( value.space(), value.coordinateSpace(), value.values() * scale );
  result->active = Access::node( value )->active;
  result->edges = { { Access::node( value ), scale, {}, {} } };
  return Access::vector( std::move( result ) );
}
ADVector operator*( const ADVector& value, RealT scale ) { return scale * value; }
ADVector operator-( const ADVector& value ) { return -1.0 * value; }
ADVector operator-( const ADVector& left, const ADVector& right ) { return left + -right; }
ADVector operator/( const ADVector& value, RealT scale )
{
  requireDivisor( scale );
  return ( 1.0 / scale ) * value;
}
ADVector operator+( const ADVector& value, RealT scalar ) { return shift( value, scalar ); }
ADVector operator+( RealT scalar, const ADVector& value ) { return shift( value, scalar ); }
ADVector operator-( const ADVector& value, RealT scalar ) { return shift( value, -scalar ); }
ADVector operator-( RealT scalar, const ADVector& value ) { return shift( -value, scalar ); }
ADVector operator*( const ADVector& left, const ADVector& right )
{
  requireVectors( left, right );
  auto a = Access::node( left ), b = Access::node( right );
  auto result = node( left.space(), left.coordinateSpace(), left.values().multiply( right.values() ) );
  result->active = a->active || b->active;
  if ( a == b ) {
    result->edges = { { a, 1.0, std::make_shared<Vector>( left.values() * 2.0 ), {} } };
    if ( a->active )
      result->curvature = [a]( const Vector& weights, detail::GlobalADEvaluation& evaluation, Matrix& result ) {
        evaluation.addOuter( result, a, a, weights * 2.0, false );
      };
  } else {
    result->edges = { { a, 1.0, b->values, {} }, { b, 1.0, a->values, {} } };
    if ( a->active && b->active )
      result->curvature = [a, b]( const Vector& weights, detail::GlobalADEvaluation& evaluation, Matrix& result ) {
        evaluation.addOuter( result, a, b, weights, true );
      };
  }
  return Access::vector( std::move( result ) );
}
ADVector reciprocal( const ADVector& value, RealT tolerance ) { return inverse( value, tolerance, true ); }
ADVector operator/( const ADVector& left, const ADVector& right )
{
  requireVectors( left, right );
  return left * inverse( right, 0.0, false );
}
ADVector operator/( RealT scalar, const ADVector& value ) { return scalar * inverse( value, 0.0, false ); }

namespace {
/** @brief Select one branch per entry, with zero branch curvature away from switches.
 * @param left First field. @param right Second field. @param minimum Choose the minimum when true.
 * @return Selected field with the first branch retained at equality. */
ADVector select( const ADVector& left, const ADVector& right, bool minimum )
{
  requireVectors( left, right );
  auto values = filled( left.space(), 0.0 ), first = filled( left.space(), 0.0 ), second = filled( left.space(), 0.0 );
  const auto* a = left.values().get().HostRead();
  const auto* b = right.values().get().HostRead();
  auto* v = values.get().HostReadWrite();
  auto* f = first.get().HostReadWrite();
  auto* s = second.get().HostReadWrite();
  for ( int i = 0; i < values.size(); ++i ) {
    const bool choose = minimum ? !( b[i] < a[i] ) : !( a[i] < b[i] );
    v[i] = choose ? a[i] : b[i];
    f[i] = choose ? 1.0 : 0.0;
    s[i] = choose ? 0.0 : 1.0;
  }
  auto result = node( left.space(), left.coordinateSpace(), std::move( values ) );
  auto pa = Access::node( left ), pb = Access::node( right );
  result->active = pa->active || pb->active;
  result->edges = { { pa, 1.0, std::make_shared<Vector>( std::move( first ) ), {} },
                    { pb, 1.0, std::make_shared<Vector>( std::move( second ) ), {} } };
  return Access::vector( std::move( result ) );
}
}  // namespace
ADVector min( const ADVector& left, const ADVector& right ) { return select( left, right, true ); }
ADVector max( const ADVector& left, const ADVector& right ) { return select( left, right, false ); }
ADVector min( const ADVector& value, RealT bound ) { return min( value, broadcast( value, bound ) ); }
ADVector min( RealT bound, const ADVector& value ) { return min( broadcast( value, bound ), value ); }
ADVector max( const ADVector& value, RealT bound ) { return max( value, broadcast( value, bound ) ); }
ADVector max( RealT bound, const ADVector& value ) { return max( broadcast( value, bound ), value ); }
ADScalar sum( const ADVector& value )
{
  RealT local = 0.0, global = 0.0;
  const auto* data = value.values().get().HostRead();
  for ( int i = 0; i < value.values().size(); ++i ) local += data[i];
  const auto type = std::is_same_v<RealT, double> ? MPI_DOUBLE : MPI_FLOAT;
  MPI_Allreduce( &local, &global, 1, type, MPI_SUM, value.space().GetComm() );
  auto result = std::make_shared<detail::GlobalADScalarData>();
  result->value = global;
  result->coordinates = &value.coordinateSpace();
  result->roots = { { Access::node( value ), 1.0 } };
  return Access::scalar( std::move( result ) );
}
ADScalar dot( const ADVector& left, const ADVector& right ) { return sum( left * right ); }
ADScalar operator+( const ADScalar& left, const ADScalar& right ) { return combine( left, right, 1.0 ); }
ADScalar operator-( const ADScalar& left, const ADScalar& right ) { return combine( left, right, -1.0 ); }
ADScalar operator*( RealT scale, const ADScalar& value )
{
  auto result = std::make_shared<detail::GlobalADScalarData>( *Access::data( value ) );
  result->value *= scale;
  for ( auto& root : result->roots ) root.scale *= scale;
  for ( auto& source : result->sources ) source.scale *= scale;
  return Access::scalar( std::move( result ) );
}
ADScalar operator*( const ADScalar& value, RealT scale ) { return scale * value; }
ADScalar operator-( const ADScalar& value ) { return -1.0 * value; }
ADScalar operator/( const ADScalar& value, RealT scale )
{
  requireDivisor( scale );
  return value * ( 1.0 / scale );
}
ADScalar operator+( const ADScalar& value, RealT scalar )
{
  auto result = std::make_shared<detail::GlobalADScalarData>( *Access::data( value ) );
  result->value += scalar;
  return Access::scalar( std::move( result ) );
}
ADScalar operator+( RealT scalar, const ADScalar& value ) { return value + scalar; }
ADScalar operator-( const ADScalar& value, RealT scalar ) { return value + ( -scalar ); }
ADScalar operator-( RealT scalar, const ADScalar& value ) { return -value + scalar; }

ADMatrix ADMatrix::constant( const Space& rows, const Space& columns, Matrix values )
{
  validateMatrix( values, rows, columns );
  auto result = std::make_shared<detail::GlobalADMatrixData>();
  result->rows = &rows;
  result->columns = &columns;
  result->values = std::make_shared<Matrix>( std::move( values ) );
  return Access::matrix( std::move( result ) );
}
ADMatrix ADMatrix::fromActions( const Space& rows, const Space& columns, const Space& coordinates, Matrix values,
                                Action apply, Action transpose_apply )
{
  validateMatrix( values, rows, columns );
  if ( !apply || !transpose_apply )
    throw std::invalid_argument( "Active AD matrices require both derivative actions." );
  requireCommunicators( rows.GetComm(), coordinates.GetComm() );
  auto result = std::make_shared<detail::GlobalADMatrixData>();
  result->rows = &rows;
  result->columns = &columns;
  result->coordinates = &coordinates;
  result->values = std::make_shared<Matrix>( std::move( values ) );
  result->apply = std::move( apply );
  result->transpose_apply = std::move( transpose_apply );
  return Access::matrix( std::move( result ) );
}
ADMatrix ADMatrix::transpose() const
{
  if ( !data_->coordinates ) return constant( *data_->columns, *data_->rows, data_->values->transpose() );
  return fromActions( *data_->columns, *data_->rows, *data_->coordinates, data_->values->transpose(),
                      data_->transpose_apply, data_->apply );
}
namespace {
/** @brief Apply a matrix to fixed values in a specified coordinate context.
 * @param matrix Matrix. @param values Fixed input values. @param coordinates Differentiation space.
 * @param transpose Use the transposed matrix. @return Owned differentiated action.
 * @throws std::invalid_argument Provider spaces do not match the declared matrix operation. */
ADVector fixedAction( const ADMatrix& matrix, const Vector& values, const Space& coordinates, bool transpose )
{
  const auto& data = Access::data( matrix );
  const auto& input = transpose ? *data->rows : *data->columns;
  const auto& output = transpose ? *data->columns : *data->rows;
  validateVector( values, input );
  if ( !data->coordinates )
    return ADVector::constant( output, coordinates, transpose ? values * *data->values : *data->values * values );
  requireCoordinates( coordinates, *data->coordinates );
  auto result = transpose ? data->transpose_apply( values ) : data->apply( values );
  if ( &result.space() != &output )
    throw std::invalid_argument( "AD matrix action returned the wrong value-space object." );
  requireCoordinates( result.coordinateSpace(), coordinates );
  return result;
}
/** @brief Select compatible coordinate metadata for matrix arithmetic.
 * @param left First matrix. @param right Second matrix. @return Active coordinate space or null for two constants.
 * @throws std::invalid_argument Active coordinate spaces differ. */
const Space* matrixCoordinates( const ADMatrix& left, const ADMatrix& right )
{
  const auto& a = Access::data( left );
  const auto& b = Access::data( right );
  if ( a->coordinates && b->coordinates ) requireCoordinates( *a->coordinates, *b->coordinates );
  return a->coordinates ? a->coordinates : b->coordinates;
}
}  // namespace
ADVector operator*( const ADMatrix& matrix, const ADVector& vector )
{
  const auto data = Access::data( matrix );
  if ( &matrix.columnSpace() != &vector.space() )
    throw std::invalid_argument( "AD matrix columns must match the vector-space object." );
  if ( data->coordinates ) requireCoordinates( *data->coordinates, vector.coordinateSpace() );
  auto parent = Access::node( vector );
  auto result = node( matrix.rowSpace(), vector.coordinateSpace(), matrix.values() * vector.values() );
  result->active = parent->active || data->coordinates;
  if ( !data->coordinates )
    result->edges = { { parent, 1.0, {}, data->values } };
  else {
    const auto fixed = fixedAction( matrix, vector.values(), vector.coordinateSpace(), false );
    result->edges = { { Access::node( fixed ), 1.0, {}, {} }, { parent, 1.0, {}, data->values } };
    if ( parent->active )
      result->curvature = [matrix, parent]( const Vector& weights, detail::GlobalADEvaluation& evaluation,
                                            Matrix& output ) {
        const auto transpose = fixedAction( matrix, weights, *parent->coordinates, true );
        evaluation.addOuter( output, Access::node( transpose ), parent, filled( *parent->space, 1.0 ), true );
      };
  }
  return Access::vector( std::move( result ) );
}
ADMatrix operator+( const ADMatrix& left, const ADMatrix& right )
{
  if ( &left.rowSpace() != &right.rowSpace() || &left.columnSpace() != &right.columnSpace() )
    throw std::invalid_argument( "AD matrix addition requires identical row- and column-space objects." );
  const auto* coordinates = matrixCoordinates( left, right );
  auto values = left.values() + right.values();
  if ( !coordinates ) return ADMatrix::constant( left.rowSpace(), left.columnSpace(), std::move( values ) );
  return ADMatrix::fromActions(
      left.rowSpace(), left.columnSpace(), *coordinates, std::move( values ),
      [left, right, coordinates]( const Vector& input ) {
        return fixedAction( left, input, *coordinates, false ) + fixedAction( right, input, *coordinates, false );
      },
      [left, right, coordinates]( const Vector& input ) {
        return fixedAction( left, input, *coordinates, true ) + fixedAction( right, input, *coordinates, true );
      } );
}
ADMatrix operator*( RealT scale, const ADMatrix& matrix )
{
  const auto data = Access::data( matrix );
  auto values = matrix.values() * scale;
  if ( !data->coordinates ) return ADMatrix::constant( matrix.rowSpace(), matrix.columnSpace(), std::move( values ) );
  return ADMatrix::fromActions(
      matrix.rowSpace(), matrix.columnSpace(), *data->coordinates, std::move( values ),
      [matrix, scale, data]( const Vector& input ) {
        return scale * fixedAction( matrix, input, *data->coordinates, false );
      },
      [matrix, scale, data]( const Vector& input ) {
        return scale * fixedAction( matrix, input, *data->coordinates, true );
      } );
}
ADMatrix operator*( const ADMatrix& value, RealT scale ) { return scale * value; }
ADMatrix operator-( const ADMatrix& value ) { return -1.0 * value; }
ADMatrix operator-( const ADMatrix& left, const ADMatrix& right ) { return left + -right; }
ADMatrix operator/( const ADMatrix& value, RealT scale )
{
  requireDivisor( scale );
  return ( 1.0 / scale ) * value;
}
ADMatrix operator*( const ADMatrix& left, const ADMatrix& right )
{
  if ( &left.columnSpace() != &right.rowSpace() )
    throw std::invalid_argument( "AD matrix product spaces do not match." );
  const auto* coordinates = matrixCoordinates( left, right );
  auto values = left.values() * right.values();
  if ( !coordinates ) return ADMatrix::constant( left.rowSpace(), right.columnSpace(), std::move( values ) );
  return ADMatrix::fromActions(
      left.rowSpace(), right.columnSpace(), *coordinates, std::move( values ),
      [left, right, coordinates]( const Vector& input ) {
        return left * fixedAction( right, input, *coordinates, false );
      },
      [left, right, coordinates]( const Vector& input ) {
        return right.transpose() * fixedAction( left, input, *coordinates, true );
      } );
}
ADMatrix diagonal( const ADVector& vector )
{
  auto values = makeDiagonal( vector.space(), vector.values() );
  if ( !Access::node( vector )->active )
    return ADMatrix::constant( vector.space(), vector.space(), std::move( values ) );
  const auto action = [vector]( const Vector& input ) {
    return vector * ADVector::constant( vector.space(), vector.coordinateSpace(), input );
  };
  return ADMatrix::fromActions( vector.space(), vector.space(), vector.coordinateSpace(), std::move( values ), action,
                                action );
}

}  // namespace tribol
#endif  // BUILD_REDECOMP
