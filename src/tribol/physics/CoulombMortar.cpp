// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
// SPDX-License-Identifier: (MIT)
#include "tribol/physics/CoulombMortar.hpp"
#ifdef BUILD_REDECOMP
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/Mortar.hpp"
#include "tribol/search/CoarseSearchFactory.hpp"
#include "shared/math/ParSparseMat.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tribol {
namespace detail {
#ifdef TRIBOL_USE_ENZYME
namespace {
/** @brief Report local failures on every rank before entering subsequent transfer collectives.
 * @param comm Parent communicator.
 * @param error Local failure message, empty on success.
 * @throws std::invalid_argument Any rank reported a failure.
 */
void checkRanks( MPI_Comm comm, const std::string& error )
{
  int failed = error.empty() ? 0 : 1, any = 0;
  MPI_Allreduce( &failed, &any, 1, MPI_INT, MPI_MAX, comm );
  if ( any )
    throw std::invalid_argument( error.empty() ? "Coulomb mortar calculation failed on another rank." : error );
}

/** @brief Assemble a local-DOF matrix using both spaces' true-DOF prolongations.
 * @param matrix Local sparse contributions; consumed.
 * @param rows Row finite-element space.
 * @param cols Column finite-element space.
 * @return Owned distributed true-DOF matrix.
 */
shared::ParSparseMat assembleLocal( mfem::SparseMatrix&& matrix, const mfem::ParFiniteElementSpace& rows,
                                    const mfem::ParFiniteElementSpace& cols )
{
  matrix.Finalize();
  shared::ParSparseMat local( rows.GetComm(), rows.GlobalVSize(), cols.GlobalVSize(), rows.GetDofOffsets(),
                              cols.GetDofOffsets(), std::move( matrix ) );
  return shared::ParSparseMat::rap( shared::ParSparseMatView( rows.Dof_TrueDof_Matrix() ), local,
                                    shared::ParSparseMatView( cols.Dof_TrueDof_Matrix() ) );
}

/** @brief Store an on-rank true-DOF block as a distributed matrix.
 * @param matrix Locally owned true-DOF rows and columns; consumed.
 * @param space Common row/column space.
 * @return Owned matrix with copied partition arrays.
 */
shared::ParSparseMat trueMatrix( mfem::SparseMatrix&& matrix, const mfem::ParFiniteElementSpace& space )
{
  matrix.Finalize();
  return shared::ParSparseMat( space.GetComm(), space.GlobalTrueVSize(), space.GetTrueDofOffsets(),
                               std::move( matrix ) );
}

/** @brief Sum Redecomp nodal integrals onto submesh true DOFs.
 * @param sub Current multiplier field transfers.
 * @param field Redecomp vector-valued nodal integrals.
 * @return Owned distributed sums.
 */
std::unique_ptr<mfem::HypreParVector> sumSubmesh( const MfemSubmeshData& sub, const mfem::GridFunction& field )
{
  const auto& fes = sub.GetSubmeshFESpace();
  mfem::Vector local( fes.GetVSize() );
  local = 0.0;
  sub.GetPressureTransfer().RedecompToSubmesh( field, local );
  shared::ParVector result( const_cast<mfem::ParFiniteElementSpace*>( &fes ) );
  result.fill( 0.0 );
  fes.GetProlongationMatrix()->MultTranspose( local, result.get() );
  return std::unique_ptr<mfem::HypreParVector>( result.release() );
}

/** @brief Pack only visited element blocks, using the paper's nonmortar/mortar element order. */
struct PairBlocks {
  std::vector<PackedPairJacobianContribs> data;  ///< Blocks and copied element/value arrays.
  /** @brief Select displacement or multiplier row/column spaces.
   * @param mesh Displacement geometry and transfers.
   * @param sub Multiplier field and transfers.
   * @param dual_rows Use only nonmortar multiplier rows.
   * @param dual_cols Use only nonmortar multiplier columns.
   */
  PairBlocks( const MfemMeshData& mesh, const MfemSubmeshData& sub, bool dual_rows, bool dual_cols )
  {
    const auto& primal_surface = mesh.GetSubmeshFESpace();
    const auto& primal_redecomp = *mesh.GetRedecompResponse().FESpace();
    const auto& dual_surface = sub.GetSubmeshFESpace();
    const auto& dual_redecomp = *sub.GetRedecompGap().FESpace();
    const Array1D<int>* maps[2] = { &mesh.GetElemMap2(), &mesh.GetElemMap1() };
    for ( int r = 0; r < ( dual_rows ? 1 : 2 ); ++r )
      for ( int c = 0; c < ( dual_cols ? 1 : 2 ); ++c )
        data.emplace_back( dual_rows ? dual_surface : primal_surface, dual_cols ? dual_surface : primal_surface,
                           dual_rows ? dual_redecomp : primal_redecomp, dual_cols ? dual_redecomp : primal_redecomp,
                           *maps[r], *maps[c] );
  }
  /** @brief Copy one dense element block.
   * @param index Packed block index.
   * @param row_element Row element ID.
   * @param col_element Column element ID.
   * @param block Column-major element matrix.
   */
  void append( int index, int row_element, int col_element, const mfem::DenseMatrix& block )
  {
    data[index].append( row_element, col_element, block.Data(), block.Height() * block.Width() );
  }
};

/** @brief Distributed corner-normal sums and their parent-coordinate derivatives. */
struct NormalData {
  std::unique_ptr<mfem::HypreParVector> sum;  ///< Unit corner normals summed over attached faces and ranks.
  shared::ParSparseMat jacobian;              ///< Derivative of the sum on submesh rows and parent columns.
};

/** @brief Calculate corner-normal sums on the fixed parent-linked surface partition.
 * @param cs Current Coulomb coupling scheme.
 * @return Owned normal sums and derivatives, before normalization of the sums.
 */
NormalData normalData( CouplingScheme& cs )
{
  auto& mesh = *cs.getMfemMeshData();
  const auto& dual = cs.getMfemSubmeshData()->GetSubmeshFESpace();
  const auto& parent = *mesh.GetParentCoords().ParFESpace();
  auto& surface = mesh.GetSubmesh();
  mfem::Vector local( dual.GetVSize() );
  local = 0.0;
  mfem::SparseMatrix jacobian( dual.GetVSize(), parent.GetVSize() );
  std::string error;
  try {
    for ( int e = 0; e < surface.GetNE(); ++e ) {
      mfem::Array<int> vertices, nodes, pdofs;
      surface.GetElementVertices( e, vertices );
      dual.GetElementDofs( e, nodes );
      int n = vertices.Size();
      MdArray<RealT, 12> x{};
      MdArray<int, 4> parent_nodes{};
      for ( int i = 0; i < n; ++i ) {
        parent.GetVertexDofs( surface.GetParentVertexIDMap()[vertices[i]], pdofs );
        parent_nodes[i] = pdofs[0];
        for ( int d = 0; d < 3; ++d ) x[d * n + i] = mesh.GetParentCoords()[parent.DofToVDof( pdofs[0], d )];
      }
      auto normals = computeCornerNormals( x.data(), n );
      if ( !mesh.GetBoundaryAttribs2().count( surface.GetAttribute( e ) ) ) continue;
      for ( int i = 0; i < n; ++i )
        for ( int d = 0; d < 3; ++d ) {
          int row = dual.DofToVDof( nodes[i], d );
          local[row] += normals.values[d * n + i];
          for ( int j = 0; j < n; ++j )
            for ( int k = 0; k < 3; ++k )
              jacobian.Add( row, parent.DofToVDof( parent_nodes[j], k ), normals.derivative( d * n + i, k * n + j ) );
        }
    }
  } catch ( const std::exception& e ) {
    error = e.what();
  }
  checkRanks( parent.GetComm(), error );
  shared::ParVector sum( const_cast<mfem::ParFiniteElementSpace*>( &dual ) );
  sum.fill( 0.0 );
  dual.GetProlongationMatrix()->MultTranspose( local, sum.get() );
  return { std::unique_ptr<mfem::HypreParVector>( sum.release() ),
           assembleLocal( std::move( jacobian ), dual, parent ) };
}

/** @brief Check whether element centroid normals oppose each other.
 * @param x1 Nonmortar component-major coordinates.
 * @param x2 Mortar component-major coordinates.
 * @param n Nodes per face.
 * @return True when the normal dot product is nonpositive.
 * @throws std::invalid_argument A centroid normal is degenerate or nonfinite.
 */
bool opposingFaces( const RealT* x1, const RealT* x2, int n )
{
  MdArray<RealT, 2, 3> normals{};
  const RealT* x[2] = { x1, x2 };
  for ( int face = 0; face < 2; ++face ) {
    MdArray<RealT, 3> a{}, b{};
    for ( int d = 0; d < 3; ++d ) {
      if ( n == 4 ) {
        a[d] = -x[face][d * n] + x[face][d * n + 1] + x[face][d * n + 2] - x[face][d * n + 3];
        b[d] = -x[face][d * n] - x[face][d * n + 1] + x[face][d * n + 2] + x[face][d * n + 3];
      } else {
        a[d] = x[face][d * n + 1] - x[face][d * n];
        b[d] = x[face][d * n + n - 1] - x[face][d * n];
      }
    }
    normals( face, 0 ) = a[1] * b[2] - a[2] * b[1];
    normals( face, 1 ) = a[2] * b[0] - a[0] * b[2];
    normals( face, 2 ) = a[0] * b[1] - a[1] * b[0];
    RealT mag = 0.0;
    for ( int d = 0; d < 3; ++d ) mag += normals( face, d ) * normals( face, d );
    if ( !std::isfinite( mag ) || mag <= 0.0 ) throw std::invalid_argument( "Degenerate mortar face." );
  }
  RealT dot = 0.0;
  for ( int d = 0; d < 3; ++d ) dot += normals( 0, d ) * normals( 1, d );
  return dot <= 0.0;
}

/** @brief Trial outputs and the exact input values used to calculate them. */
struct Evaluation {
  ContactResidual residual;                              ///< Owned solver residual and diagnostics.
  std::unique_ptr<mfem::HypreParVector> gap_vector;      ///< Unshifted vector for accepted-step history.
  std::unique_ptr<mfem::HypreParMatrix> ff, fl, rf, rl;  ///< Coordinate/multiplier derivatives of both residuals.
  mfem::Vector coordinates, traction;                    ///< Parent local coordinates and multiplier true DOFs.
  RealT residual_gap{ 0.0 };                             ///< Normal offset used in this calculation.
  RealT collapse_ratio{ 0.0 };                           ///< Clipping tolerance used in this calculation.
  RealT proximity_scale{ 0.0 };                          ///< Effective search scale used in this calculation.
};
}  // namespace

/** @brief Own configuration and history on spaces that survive Redecomp rebuilds. */
struct CoulombMortar::Impl {
  CoulombFrictionOptions options;                               ///< Fixed numerical parameters for a load step.
  bool configured{ false }, active{ false };                    ///< Whether options are valid and a load step is open.
  long parent_sequence{ -1 }, space_sequence{ -1 };             ///< Parent topology/space versions at registration.
  const mfem::ParFiniteElementSpace* parent_space{ nullptr };   ///< Parent space identity at registration.
  std::unique_ptr<mfem::ParGridFunction> accepted_coordinates;  ///< Owned accepted parent coordinates.
  mfem::Vector accepted_traction, accepted_gap;  ///< Accepted local traction input and true-DOF gap vector.
  std::unique_ptr<Evaluation> latest;            ///< Successful trial, invalidated before each calculation.

  /** @brief Record the supported parent spaces. @param scheme Borrowed scheme. */
  explicit Impl( CouplingScheme& scheme )
  {
    parent_space = scheme.getMfemMeshData()->GetParentCoords().ParFESpace();
    parent_sequence = parent_space->GetParMesh()->GetSequence();
    space_sequence = parent_space->GetSequence();
  }

  /** @brief Reject unsupported configuration or parent-space changes. @param cs Current scheme. */
  void validate( CouplingScheme& cs ) const
  {
    if ( !configured ) throw std::logic_error( "Call setCoulombFrictionOptions() before beginning friction contact." );
    if ( !cs.hasMfemData() || !cs.hasMfemSubmeshData() || !cs.hasMfemJacobianData() ||
         cs.getContactMethod() != SINGLE_MORTAR || cs.getContactModel() != FRICTION_COULOMB ||
         cs.getEnforcementMethod() != LAGRANGE_MULTIPLIER || !cs.isEnzymeEnabled() ||
         isOnDevice( cs.getExecutionMode() ) || cs.getContactMode() != SURFACE_TO_SURFACE ||
         ( cs.getContactCase() != NO_CASE && cs.getContactCase() != NO_SLIDING ) )
      throw std::invalid_argument( "Coulomb mortar requires host MFEM Enzyme single mortar with direct multipliers." );
    bool host_execution = cs.getExecutionMode() == ExecutionMode::Sequential;
#ifdef TRIBOL_USE_OPENMP
    host_execution = host_execution || cs.getExecutionMode() == ExecutionMode::OpenMP;
#endif
    if ( !host_execution || ( cs.getBinningMethod() != BINNING_GRID && cs.getBinningMethod() != BINNING_BVH ) ||
         cs.hasFixedBinning() || !std::isfinite( cs.getEffectiveBinningProximityScale() ) ||
         cs.getEffectiveBinningProximityScale() <= 0.0 )
      throw std::invalid_argument( "Coulomb mortar requires an enabled host backend and current GRID or BVH search." );
    const auto& mesh = *cs.getMfemMeshData();
    const auto& parent = *mesh.GetParentCoords().ParFESpace();
    const auto& dual = cs.getMfemSubmeshData()->GetSubmeshFESpace();
    if ( parent.GetVDim() != 3 || parent.FEColl()->GetOrder() != 1 || parent.GetParMesh()->Dimension() != 3 ||
         dynamic_cast<const mfem::H1_FECollection*>( parent.FEColl() ) == nullptr || mesh.GetLORMesh() != nullptr ||
         parent.GetParMesh()->Nonconforming() || dual.GetOrdering() != mfem::Ordering::byNODES ||
         !std::isfinite( cs.getParameters().residual_gap ) || cs.getParameters().residual_gap < 0.0 ||
         !std::isfinite( cs.getParameters().len_collapse_ratio ) || cs.getParameters().len_collapse_ratio <= 0.0 )
      throw std::invalid_argument( "Coulomb mortar supports conforming linear 3D parent meshes without LOR." );
    for ( int i = 0; i < mesh.GetParentCoords().Size(); ++i )
      if ( !std::isfinite( mesh.GetParentCoords()[i] ) )
        throw std::invalid_argument( "Nonfinite current coordinates." );
    if ( &parent != parent_space || parent.GetParMesh()->GetSequence() != parent_sequence ||
         parent.GetSequence() != space_sequence ||
         ( active && accepted_coordinates->Size() != mesh.GetParentCoords().Size() ) )
      throw std::invalid_argument( "Parent topology, spaces, or partition changed; register a new Coulomb scheme." );
  }

  /** @brief Assemble current integrals, nodal constraints, and all four Jacobian blocks.
   * @param cs Current scheme and input fields; not retained.
   * @return Owned trial outputs; failures leave accepted-step data untouched.
   */
  std::unique_ptr<Evaluation> calculate( CouplingScheme& cs )
  {
    auto& mesh = *cs.getMfemMeshData();
    auto& sub = *cs.getMfemSubmeshData();
    const auto& parent = *mesh.GetParentCoords().ParFESpace();
    const auto& dual = sub.GetSubmeshFESpace();
    std::string error;
    try {
      validate( cs );
      if ( !cs.setMeshPointers() ) throw std::invalid_argument( "Missing registered contact meshes." );
    } catch ( const std::exception& e ) {
      error = e.what();
    }
    checkRanks( parent.GetComm(), error );
    auto geometry1 = cs.getMesh2().getView(), geometry2 = cs.getMesh1().getView();
    const int n = geometry1.numberOfNodesPerElement(), nd = 3 * n;
    error.clear();
    if ( geometry1.spatialDimension() != 3 || geometry2.spatialDimension() != 3 || ( n != 3 && n != 4 ) ||
         geometry2.numberOfNodesPerElement() != n || geometry1.getFaceOrder() != LINEAR_LAGRANGE ||
         geometry2.getFaceOrder() != LINEAR_LAGRANGE )
      error = "Coulomb mortar requires matching TRI3/TRI3 or QUAD4/QUAD4 element shapes.";
    checkRanks( parent.GetComm(), error );
    auto normals = normalData( cs );
    auto search = createCoarseSearch( 3, cs.getBinningMethod(), cs.getExecutionMode(),
                                      cs.getEffectiveBinningProximityScale(), cs.getParameters().residual_gap );
    auto pairs = search->findCandidatePairs( geometry1, geometry2 );
    auto* rfes = const_cast<mfem::FiniteElementSpace*>( sub.GetRedecompGap().FESpace() );
    auto* pfes = const_cast<mfem::FiniteElementSpace*>( mesh.GetRedecompResponse().FESpace() );
    mfem::GridFunction w( rfes ), q( rfes ), area( rfes ), force( pfes ), old( pfes ), traction( rfes );
    w = 0.0;
    q = 0.0;
    area = 0.0;
    force = 0.0;
    old = 0.0;
    traction = 0.0;
    mesh.GetParentRedecompTransfer().ParentToRedecomp( active ? *accepted_coordinates : mesh.GetParentCoords(), old );
    sub.GetPressureTransfer().SubmeshToRedecomp( sub.GetSubmeshPressure(), traction );
    const int rnodes = rfes->GetNDofs(), pnodes = pfes->GetNDofs();
    PairBlocks fx( mesh, sub, false, false ), fl( mesh, sub, false, true ), wx( mesh, sub, true, false ),
        qx( mesh, sub, true, false ), ax( mesh, sub, true, false );
    error.clear();
    try {
      for ( const auto& pair : pairs ) {
        MdArray<int, 2> elements{ static_cast<int>( pair.element_id1_ ), static_cast<int>( pair.element_id2_ ) };
        MdArray<int, 2, 4> nodes{};
        MdArray<RealT, 2, 12> x{}, previous{};
        MdArray<RealT, 12> lambda{};
        for ( int side = 0; side < 2; ++side )
          for ( int i = 0; i < n; ++i ) {
            nodes( side, i ) = ( side == 0 ? geometry1 : geometry2 ).getGlobalNodeId( elements[side], i );
            for ( int d = 0; d < 3; ++d ) {
              x( side, d * n + i ) = ( side == 0 ? geometry1 : geometry2 ).getPosition()[d][nodes( side, i )];
              previous( side, d * n + i ) = old[nodes( side, i ) + d * pnodes];
              if ( side == 0 ) lambda[d * n + i] = traction[nodes( side, i ) + d * rnodes];
            }
          }
        if ( !opposingFaces( x.data(), x.data() + 12, n ) ) continue;
        auto weights =
            computeMortarPairWeightJacobian( x.data(), n, x.data() + 12, n, cs.getParameters().len_collapse_ratio );
        if ( weights.weights.area <= 0.0 ) continue;
        const auto& D = weights.weights.nonmortar;
        const auto& M = weights.weights.mortar;
        for ( int i = 0; i < n; ++i )
          for ( int j = 0; j < n; ++j )
            for ( int d = 0; d < 3; ++d ) {
              w[nodes( 0, i ) + d * rnodes] += D[i * n + j] * x( 0, d * n + j ) - M[i * n + j] * x( 1, d * n + j );
              q[nodes( 0, i ) + d * rnodes] +=
                  D[i * n + j] * previous( 0, d * n + j ) - M[i * n + j] * previous( 1, d * n + j );
              area[nodes( 0, i ) + d * rnodes] += D[i * n + j];
              force[nodes( 0, j ) + d * pnodes] += D[i * n + j] * lambda[d * n + i];
              force[nodes( 1, j ) + d * pnodes] -= M[i * n + j] * lambda[d * n + i];
            }
        for ( int side = 0; side < 2; ++side ) {
          mfem::DenseMatrix f_lambda( nd, nd );
          f_lambda = 0.0;
          for ( int i = 0; i < n; ++i )
            for ( int j = 0; j < n; ++j )
              for ( int d = 0; d < 3; ++d )
                f_lambda( d * n + j, d * n + i ) = ( side == 0 ? D[i * n + j] : -M[i * n + j] );
          fl.append( side, elements[side], elements[0], f_lambda );
          for ( int col_side = 0; col_side < 2; ++col_side ) {
            mfem::DenseMatrix f_x( nd, nd );
            f_x = 0.0;
            for ( int col = 0; col < nd; ++col )
              for ( int i = 0; i < n; ++i )
                for ( int j = 0; j < n; ++j )
                  for ( int d = 0; d < 3; ++d )
                    f_x( d * n + j, col ) +=
                        ( side == 0 ? 1.0 : -1.0 ) *
                        weights.derivative( ( side == 0 ? 0 : 16 ) + i * n + j, col_side * nd + col ) *
                        lambda[d * n + i];
            fx.append( side * 2 + col_side, elements[side], elements[col_side], f_x );
          }
        }
        for ( int side = 0; side < 2; ++side ) {
          mfem::DenseMatrix w_x( nd, nd ), q_x( nd, nd ), a_x( nd, nd );
          w_x = 0.0;
          q_x = 0.0;
          a_x = 0.0;
          for ( int col = 0; col < nd; ++col )
            for ( int i = 0; i < n; ++i )
              for ( int j = 0; j < n; ++j )
                for ( int d = 0; d < 3; ++d ) {
                  RealT dD = weights.derivative( i * n + j, side * nd + col ),
                        dM = weights.derivative( 16 + i * n + j, side * nd + col );
                  w_x( d * n + i, col ) += dD * x( 0, d * n + j ) - dM * x( 1, d * n + j );
                  q_x( d * n + i, col ) += dD * previous( 0, d * n + j ) - dM * previous( 1, d * n + j );
                  a_x( d * n + i, col ) += dD;
                  if ( col == d * n + j ) w_x( d * n + i, col ) += ( side == 0 ? D[i * n + j] : -M[i * n + j] );
                }
          wx.append( side, elements[0], elements[side], w_x );
          qx.append( side, elements[0], elements[side], q_x );
          ax.append( side, elements[0], elements[side], a_x );
        }
      }
    } catch ( const std::exception& e ) {
      error = e.what();
    }
    checkRanks( parent.GetComm(), error );
    auto result = std::make_unique<Evaluation>();
    result->gap_vector = sumSubmesh( sub, w );
    auto previous_weights = sumSubmesh( sub, q ), areas = sumSubmesh( sub, area );
    shared::ParVector force_true( const_cast<mfem::ParFiniteElementSpace*>( &parent ) );
    force_true.fill( 0.0 );
    mfem::Vector parent_force( parent.GetVSize() );
    parent_force = 0.0;
    mesh.GetParentRedecompTransfer().RedecompToParent( force, parent_force );
    parent.GetProlongationMatrix()->MultTranspose( parent_force, force_true.get() );
    result->residual.force.reset( force_true.release() );
    shared::ParVector constraint( const_cast<mfem::ParFiniteElementSpace*>( &dual ) );
    constraint.fill( 0.0 );
    mfem::Vector multiplier;
    sub.GetSubmeshPressure().GetTrueDofs( multiplier );
    int owned_nodes = dual.GetTrueVSize() / 3;
    auto& residual = result->residual;
    residual.gap.SetSize( owned_nodes );
    residual.pressure.SetSize( owned_nodes );
    residual.slip.SetSize( dual.GetTrueVSize() );
    residual.normal.SetSize( dual.GetTrueVSize() );
    residual.state.SetSize( owned_nodes );
    mfem::SparseMatrix kn( dual.GetTrueVSize() ), kw( dual.GetTrueVSize() ), kq( dual.GetTrueVSize() ),
        ka( dual.GetTrueVSize() ), kl( dual.GetTrueVSize() );
    error.clear();
    try {
      for ( int i = 0; i < owned_nodes; ++i ) {
        CoulombNodeInput input;
        for ( int d = 0; d < 3; ++d ) {
          int row = i + d * owned_nodes;
          input.normal_sum[d] = ( *normals.sum )[row];
          input.gap_vector[d] = ( *result->gap_vector )[row];
          input.slip_vector[d] = active ? ( *previous_weights )[row] - accepted_gap[row] : 0.0;
          input.traction[d] = multiplier[row];
        }
        input.area = ( *areas )[i];
        auto node = evaluateCoulombNode( input, options, cs.getParameters().residual_gap );
        residual.gap[i] = node.gap;
        residual.pressure[i] = node.pressure;
        residual.state[i] = static_cast<int>( node.state );
        for ( int d = 0; d < 3; ++d ) {
          int row = i + d * owned_nodes;
          constraint[row] = node.constraint[d];
          residual.slip[row] = node.slip[d];
          residual.normal[row] = node.normal[d];
          for ( int k = 0; k < 3; ++k ) {
            int col = i + k * owned_nodes;
            kn.Add( row, col, node.derivative( d, k ) );
            kw.Add( row, col, node.derivative( d, 3 + k ) );
            // During begin(), no slip history is used; only its assembled gap vector is retained.
            kq.Add( row, col, active ? node.derivative( d, 6 + k ) : 0.0 );
            kl.Add( row, col, node.derivative( d, 9 + k ) );
          }
          ka.Add( row, i, node.derivative( d, 12 ) );
        }
      }
    } catch ( const std::exception& e ) {
      error = e.what();
    }
    checkRanks( parent.GetComm(), error );
    residual.constraint.reset( constraint.release() );
    auto Jw = cs.getMfemJacobianData()->GetMfemJacobian( &dual, &parent, wx.data );
    auto Jq = cs.getMfemJacobianData()->GetMfemJacobian( &dual, &parent, qx.data );
    auto Ja = cs.getMfemJacobianData()->GetMfemJacobian( &dual, &parent, ax.data );
    auto chain = trueMatrix( std::move( kn ), dual ) * normals.jacobian;
    chain += trueMatrix( std::move( kw ), dual ) * Jw;
    chain += trueMatrix( std::move( kq ), dual ) * Jq;
    chain += trueMatrix( std::move( ka ), dual ) * Ja;
    result->rf.reset( chain.release() );
    auto multiplier_block = trueMatrix( std::move( kl ), dual );
    result->rl.reset( multiplier_block.release() );
    auto force_x = cs.getMfemJacobianData()->GetMfemJacobian( &parent, &parent, fx.data );
    auto force_lambda = cs.getMfemJacobianData()->GetMfemJacobian( &parent, &dual, fl.data );
    result->ff.reset( force_x.release() );
    result->fl.reset( force_lambda.release() );
    result->coordinates = mesh.GetParentCoords();
    result->traction = multiplier;
    result->residual_gap = cs.getParameters().residual_gap;
    result->collapse_ratio = cs.getParameters().len_collapse_ratio;
    result->proximity_scale = cs.getEffectiveBinningProximityScale();
    return result;
  }

  /** @brief Require a successful trial at unchanged inputs on every rank. @param cs Current scheme. */
  void requireLatest( CouplingScheme& cs ) const
  {
    std::string error;
    try {
      validate( cs );
      if ( !latest ) throw std::logic_error( "No current friction result; call update() first." );
      if ( latest->residual_gap != cs.getParameters().residual_gap ||
           latest->collapse_ratio != cs.getParameters().len_collapse_ratio ||
           latest->proximity_scale != cs.getEffectiveBinningProximityScale() )
        throw std::logic_error( "Friction settings changed since the last evaluation." );
      const auto& coordinates = cs.getMfemMeshData()->GetParentCoords();
      mfem::Vector traction;
      cs.getMfemSubmeshData()->GetSubmeshPressure().GetTrueDofs( traction );
      if ( coordinates.Size() != latest->coordinates.Size() || traction.Size() != latest->traction.Size() )
        throw std::logic_error( "Friction inputs changed since the last evaluation." );
      for ( int i = 0; i < coordinates.Size(); ++i )
        if ( coordinates[i] != latest->coordinates[i] )
          throw std::logic_error( "Coordinates changed since the last friction evaluation." );
      for ( int i = 0; i < traction.Size(); ++i )
        if ( traction[i] != latest->traction[i] )
          throw std::logic_error( "Traction changed since the last friction evaluation." );
    } catch ( const std::exception& e ) {
      error = e.what();
    }
    checkRanks( cs.getMfemMeshData()->GetParentCoords().ParFESpace()->GetComm(), error );
  }
};

CoulombMortar::CoulombMortar( CouplingScheme& scheme ) : impl_( std::make_unique<Impl>( scheme ) ) {}
CoulombMortar::~CoulombMortar() = default;
void CoulombMortar::setOptions( const CoulombFrictionOptions& options )
{
  if ( impl_->active ) throw std::logic_error( "Change Coulomb options outside an active friction step." );
  CoulombNodeInput empty;
  evaluateCoulombNode( empty, options, 0.0, false );
  impl_->options = options;
  impl_->configured = true;
  impl_->latest.reset();
}
void CoulombMortar::begin( CouplingScheme& cs )
{
  if ( impl_->active ) throw std::logic_error( "A friction step is already active." );
  impl_->latest.reset();
  auto baseline = impl_->calculate( cs );
  const auto& coordinates = cs.getMfemMeshData()->GetParentCoords();
  impl_->accepted_coordinates = std::make_unique<mfem::ParGridFunction>( coordinates );
  impl_->accepted_traction = cs.getMfemSubmeshData()->GetSubmeshPressure();
  impl_->accepted_gap = *baseline->gap_vector;
  impl_->active = true;
}
void CoulombMortar::commit( CouplingScheme& cs )
{
  if ( !impl_->active ) throw std::logic_error( "No active friction step to commit." );
  impl_->requireLatest( cs );
  *impl_->accepted_coordinates = cs.getMfemMeshData()->GetParentCoords();
  impl_->accepted_gap = *impl_->latest->gap_vector;
  impl_->accepted_traction = cs.getMfemSubmeshData()->GetSubmeshPressure();
  impl_->active = false;
}
void CoulombMortar::rollback( CouplingScheme& cs )
{
  if ( !impl_->active ) throw std::logic_error( "No active friction step to roll back." );
  cs.getMfemSubmeshData()->GetSubmeshPressure() = impl_->accepted_traction;
  impl_->latest.reset();
  impl_->active = false;
}
void CoulombMortar::evaluate( CouplingScheme& cs )
{
  impl_->latest.reset();
  if ( !impl_->active ) throw std::logic_error( "Call beginContactStep() before evaluating Coulomb contact." );
  impl_->latest = impl_->calculate( cs );
}
void CoulombMortar::checkParentMesh( CouplingScheme& cs ) const
{
  const auto& parent = *cs.getMfemMeshData()->GetParentCoords().ParFESpace();
  std::string error;
  if ( &parent != impl_->parent_space || parent.GetParMesh()->GetSequence() != impl_->parent_sequence ||
       parent.GetSequence() != impl_->space_sequence ||
       ( impl_->active && impl_->accepted_coordinates->Size() != cs.getMfemMeshData()->GetParentCoords().Size() ) )
    error = "Parent topology, spaces, or partition changed; register a new Coulomb scheme.";
  checkRanks( parent.GetComm(), error );
}
void CoulombMortar::invalidate() { impl_->latest.reset(); }
ContactResidual CoulombMortar::residual( CouplingScheme& cs ) const
{
  impl_->requireLatest( cs );
  const auto& source = impl_->latest->residual;
  ContactResidual result;
  result.force = std::make_unique<mfem::HypreParVector>( *source.force );
  result.constraint = std::make_unique<mfem::HypreParVector>( *source.constraint );
  result.gap = source.gap;
  result.pressure = source.pressure;
  result.slip = source.slip;
  result.normal = source.normal;
  result.state = source.state;
  return result;
}
mfem::HypreParVector CoulombMortar::force( CouplingScheme& cs ) const
{
  impl_->requireLatest( cs );
  return *impl_->latest->residual.force;
}
std::unique_ptr<mfem::BlockOperator> CoulombMortar::jacobian( CouplingScheme& cs ) const
{
  impl_->requireLatest( cs );
  const auto& result = *impl_->latest;
  mfem::Array<int> offsets( 3 );
  offsets[0] = 0;
  offsets[1] = result.ff->Height();
  offsets[2] = offsets[1] + result.rl->Height();
  auto op = std::make_unique<mfem::BlockOperator>( offsets );
  op->owns_blocks = 1;
  op->SetBlock( 0, 0, new mfem::HypreParMatrix( *result.ff ) );
  op->SetBlock( 0, 1, new mfem::HypreParMatrix( *result.fl ) );
  op->SetBlock( 1, 0, new mfem::HypreParMatrix( *result.rf ) );
  op->SetBlock( 1, 1, new mfem::HypreParMatrix( *result.rl ) );
  return op;
}
#else
struct CoulombMortar::Impl {};
CoulombMortar::CoulombMortar( CouplingScheme& ) : impl_( std::make_unique<Impl>() )
{
  throw std::runtime_error( "Coulomb mortar requires Enzyme support." );
}
CoulombMortar::~CoulombMortar() = default;
void CoulombMortar::setOptions( const CoulombFrictionOptions& )
{
  throw std::runtime_error( "Enzyme is unavailable." );
}
void CoulombMortar::begin( CouplingScheme& ) { throw std::runtime_error( "Enzyme is unavailable." ); }
void CoulombMortar::commit( CouplingScheme& ) { throw std::runtime_error( "Enzyme is unavailable." ); }
void CoulombMortar::rollback( CouplingScheme& ) { throw std::runtime_error( "Enzyme is unavailable." ); }
void CoulombMortar::evaluate( CouplingScheme& ) { throw std::runtime_error( "Enzyme is unavailable." ); }
void CoulombMortar::checkParentMesh( CouplingScheme& ) const {}
void CoulombMortar::invalidate() {}
ContactResidual CoulombMortar::residual( CouplingScheme& ) const
{
  throw std::runtime_error( "Enzyme is unavailable." );
}
mfem::HypreParVector CoulombMortar::force( CouplingScheme& ) const
{
  throw std::runtime_error( "Enzyme is unavailable." );
}
std::unique_ptr<mfem::BlockOperator> CoulombMortar::jacobian( CouplingScheme& ) const
{
  throw std::runtime_error( "Enzyme is unavailable." );
}
#endif
}  // namespace detail
}  // namespace tribol
#endif
