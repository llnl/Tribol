// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MfemData.hpp"

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP

#include <cmath>

#include "axom/slic/interface/slic_macros.hpp"

#include "shared/infrastructure/Profiling.hpp"
#include "tribol/common/LoopExec.hpp"

#include "redecomp/utils/ArrayUtility.hpp"

#include "tribol/common/LoopExec.hpp"
#include "tribol/common/Atomics.hpp"

namespace tribol {

namespace {

std::unique_ptr<shared::ParSparseMat> TryGetMfemTrueRestrictionMatrix(
    const mfem::ParFiniteElementSpace& ho_scalar_fes, const mfem::ParFiniteElementSpace& lor_scalar_fes )
{
  // MFEM builds the true-dof restriction operator inside its L2 projection transfer
  // helper, but does not expose it publicly. We use a tiny derived class to read
  // out the internal `R` pointer and clone it when it is an assembled HypreParMatrix.
  class MfemTrueRestrictionAccessor : public mfem::L2ProjectionGridTransfer::L2ProjectionH1Space {
   public:
    MfemTrueRestrictionAccessor( const mfem::ParFiniteElementSpace& pfes_ho,
                                 const mfem::ParFiniteElementSpace& pfes_lor, bool use_ea )
        : mfem::L2ProjectionGridTransfer::L2ProjectionH1Space( pfes_ho, pfes_lor, use_ea )
    {
    }

    const mfem::Operator* TrueRestriction() const { return R.get(); }
  };

  // Use MFEM's non-EA path because we only reuse this branch when MFEM exposes
  // the true-space restriction as an assembled HypreParMatrix.
  MfemTrueRestrictionAccessor mfem_h1( ho_scalar_fes, lor_scalar_fes, false );
  const mfem::Operator* R_op = mfem_h1.TrueRestriction();
  const auto* R_hypre = dynamic_cast<const mfem::HypreParMatrix*>( R_op );
  if ( R_hypre == nullptr ) {
    return nullptr;
  }

  return std::make_unique<shared::ParSparseMat>( std::make_unique<mfem::HypreParMatrix>( *R_hypre ) );
}

void TDofsListByVDim( const mfem::ParFiniteElementSpace& fes, int vdim, mfem::Array<int>& tdofs_list )
{
  // Build the subset of true dofs that correspond to one vector component. When a
  // restriction matrix is available, use it to correctly handle constrained dofs;
  // otherwise, fall back to the raw vdof list.
  const mfem::SparseMatrix* R_mat = fes.GetRestrictionMatrix();
  if ( R_mat ) {
    mfem::Array<int> x_vdofs_list( fes.GetNDofs() );
    mfem::Array<int> x_vdofs_marker( fes.GetVSize() );
    mfem::Array<int> X_vdofs_marker( fes.GetTrueVSize() );
    fes.GetVDofs( vdim, x_vdofs_list );
    mfem::FiniteElementSpace::ListToMarker( x_vdofs_list, fes.GetVSize(), x_vdofs_marker );
    R_mat->BooleanMult( x_vdofs_marker, X_vdofs_marker );
    mfem::FiniteElementSpace::MarkerToList( X_vdofs_marker, tdofs_list );
  } else {
    tdofs_list.SetSize( fes.GetNDofs() );
    fes.GetVDofs( vdim, tdofs_list );
  }
}

shared::ParSparseMat BuildTrueDofExtract( const mfem::ParFiniteElementSpace& vec_fes,
                                          const mfem::ParFiniteElementSpace& scalar_fes, int vdim )
{
  // Construct a sparse selection operator E such that x_scalar_true = E * x_vec_true,
  // pulling out one component from a vector-valued true-dof vector.
  mfem::Array<int> tdofs_list;
  TDofsListByVDim( vec_fes, vdim, tdofs_list );
  SLIC_ERROR_ROOT_IF( tdofs_list.Size() != scalar_fes.GetTrueVSize(),
                      "Unexpected TDofs list size when building LOR transfer." );

  const int nrows = scalar_fes.GetTrueVSize();
  const int ncols = vec_fes.GetTrueVSize();

  auto* I = new int[nrows + 1];
  auto* J = new int[nrows];
  auto* data = new double[nrows];
  for ( int i = 0; i < nrows + 1; ++i ) I[i] = i;
  for ( int r = 0; r < nrows; ++r ) {
    J[r] = tdofs_list[r];
    data[r] = 1.0;
  }
  mfem::SparseMatrix diag( I, J, data, nrows, ncols, true, true, true );
  return shared::ParSparseMat( scalar_fes.GetComm(), scalar_fes.GlobalTrueVSize(), vec_fes.GlobalTrueVSize(),
                               scalar_fes.GetTrueDofOffsets(), vec_fes.GetTrueDofOffsets(), std::move( diag ) );
}

shared::ParSparseMat BuildTrueDofInject( const mfem::ParFiniteElementSpace& vec_fes,
                                         const mfem::ParFiniteElementSpace& scalar_fes, int vdim )
{
  // Construct a sparse injection operator S such that y_vec_true += S * y_scalar_true,
  // inserting one component back into a vector-valued true-dof vector.
  mfem::Array<int> tdofs_list;
  TDofsListByVDim( vec_fes, vdim, tdofs_list );
  SLIC_ERROR_ROOT_IF( tdofs_list.Size() != scalar_fes.GetTrueVSize(),
                      "Unexpected TDofs list size when building LOR transfer." );

  const int nrows = vec_fes.GetTrueVSize();
  const int ncols = scalar_fes.GetTrueVSize();

  std::vector<int> row_counts( static_cast<size_t>( nrows ), 0 );
  for ( int c = 0; c < ncols; ++c ) {
    const int r = tdofs_list[c];
    SLIC_ERROR_ROOT_IF( r < 0 || r >= nrows, "TDof index out of bounds in LOR transfer." );
    row_counts[static_cast<size_t>( r )] += 1;
  }

  auto* I = new int[nrows + 1];
  I[0] = 0;
  for ( int r = 0; r < nrows; ++r ) {
    I[r + 1] = I[r] + row_counts[static_cast<size_t>( r )];
  }
  const int nnz = I[nrows];
  auto* J = new int[nnz];
  auto* data = new double[nnz];
  for ( int k = 0; k < nnz; ++k ) data[k] = 1.0;

  std::vector<int> next( static_cast<size_t>( nrows ), 0 );
  for ( int r = 0; r < nrows; ++r ) next[static_cast<size_t>( r )] = I[r];

  for ( int c = 0; c < ncols; ++c ) {
    const int r = tdofs_list[c];
    const int pos = next[static_cast<size_t>( r )]++;
    J[pos] = c;
  }

  mfem::SparseMatrix diag( I, J, data, nrows, ncols, true, true, true );
  return shared::ParSparseMat( vec_fes.GetComm(), vec_fes.GlobalTrueVSize(), scalar_fes.GlobalTrueVSize(),
                               vec_fes.GetTrueDofOffsets(), scalar_fes.GetTrueDofOffsets(), std::move( diag ) );
}

}  // namespace

HoToLorTransferAction::HoToLorTransferAction( const mfem::ParFiniteElementSpace& ho_fes,
                                              const mfem::ParFiniteElementSpace& lor_fes, bool use_ea )
    : mfem::Operator( lor_fes.GetVSize(), ho_fes.GetVSize() ), ho_fes_( ho_fes ), lor_fes_( lor_fes )
{
  transfer_ = std::make_unique<mfem::L2ProjectionGridTransfer>( const_cast<mfem::ParFiniteElementSpace&>( ho_fes_ ),
                                                                const_cast<mfem::ParFiniteElementSpace&>( lor_fes_ ) );
  transfer_->UseEA( use_ea );
}

void HoToLorTransferAction::Mult( const mfem::Vector& x, mfem::Vector& y ) const
{
  transfer_->ForwardOperator().Mult( x, y );
}

HoToLorTransferMat::HoToLorTransferMat( const mfem::ParFiniteElementSpace& ho_fes,
                                        const mfem::ParFiniteElementSpace& lor_fes,
                                        const mfem::ParFiniteElementSpace& ho_scalar_fes,
                                        const mfem::ParFiniteElementSpace& lor_scalar_fes )
    : ho_fes_( ho_fes ), lor_fes_( lor_fes ), ho_scalar_fes_( ho_scalar_fes ), lor_scalar_fes_( lor_scalar_fes )
{
}

shared::ParSparseMat HoToLorTransferMat::Assemble() const
{
  // Assemble an explicit HO->LOR transfer matrix that matches MFEM's runtime L2
  // projection transfer operator. This is used for tests and for assembled Jacobians.
  SLIC_ERROR_ROOT_IF(
      ho_fes_.GetOrdering() != mfem::Ordering::byNODES || lor_fes_.GetOrdering() != mfem::Ordering::byNODES,
      "LOR transfer matrix build only supports mfem::Ordering::byNODES." );
  SLIC_ERROR_ROOT_IF( ho_fes_.GetVDim() != lor_fes_.GetVDim(), "HO/LOR vdim mismatch in LOR transfer." );
  SLIC_ERROR_ROOT_IF( ho_scalar_fes_.GetVDim() != 1 || lor_scalar_fes_.GetVDim() != 1,
                      "Scalar FE spaces must be vdim=1." );

  shared::ParSparseMatView P_lor( const_cast<mfem::HypreParMatrix*>( lor_scalar_fes_.Dof_TrueDof_Matrix() ) );
  std::unique_ptr<shared::ParSparseMat> R_true_owned =
      TryGetMfemTrueRestrictionMatrix( ho_scalar_fes_, lor_scalar_fes_ );
  SLIC_ERROR_ROOT_IF(
      !R_true_owned,
      "Failed to access MFEM's assembled true-dof restriction operator for HO->LOR transfer. "
      "This build expects mfem::L2ProjectionGridTransfer::L2ProjectionH1Space(use_ea=false) to store R as a "
      "mfem::HypreParMatrix." );

  const mfem::HypreParMatrix* P_ho_scalar = ho_scalar_fes_.Dof_TrueDof_Matrix();
  SLIC_ERROR_ROOT_IF( P_ho_scalar == nullptr, "Null HO scalar Dof_TrueDof_Matrix() in LOR transfer build." );
  std::unique_ptr<mfem::HypreParMatrix> P_ho_scalar_T( P_ho_scalar->Transpose() );
  SLIC_ERROR_ROOT_IF( P_ho_scalar_T == nullptr, "Failed to transpose HO scalar Dof_TrueDof_Matrix()." );

  if ( ho_fes_.GetVDim() == 1 ) {
    shared::ParSparseMatView P_ho_T_view( P_ho_scalar_T.get() );
    return P_lor * ( *R_true_owned ) * P_ho_T_view;
  }

  const int vdim = ho_fes_.GetVDim();
  shared::ParSparseMatView P_lor_vec( const_cast<mfem::HypreParMatrix*>( lor_fes_.Dof_TrueDof_Matrix() ) );

  const mfem::HypreParMatrix* P_ho_vec = ho_fes_.Dof_TrueDof_Matrix();
  SLIC_ERROR_ROOT_IF( P_ho_vec == nullptr, "Null HO vector Dof_TrueDof_Matrix() in LOR transfer build." );
  std::unique_ptr<mfem::HypreParMatrix> P_ho_vec_T( P_ho_vec->Transpose() );
  SLIC_ERROR_ROOT_IF( P_ho_vec_T == nullptr, "Failed to transpose HO vector Dof_TrueDof_Matrix()." );
  shared::ParSparseMatView R_ho_vec( P_ho_vec_T.get() );

  std::unique_ptr<shared::ParSparseMat> R_vec_true;
  for ( int d = 0; d < vdim; ++d ) {
    // MFEM's vector-valued transfer is assembled component-by-component in true-dof
    // space, then composed back to DOF space through the vector prolongation.
    auto E_ho_d = BuildTrueDofExtract( ho_fes_, ho_scalar_fes_, d );
    auto S_lor_d = BuildTrueDofInject( lor_fes_, lor_scalar_fes_, d );

    auto block = S_lor_d * ( *R_true_owned ) * E_ho_d;

    if ( !R_vec_true ) {
      R_vec_true = std::make_unique<shared::ParSparseMat>( std::move( block ) );
    } else {
      ( *R_vec_true ) += block;
    }
  }

  return P_lor_vec * ( *R_vec_true ) * R_ho_vec;
}

SubmeshParentTransferMat::SubmeshParentTransferMat( const mfem::ParFiniteElementSpace& submesh_fes,
                                                    const mfem::ParFiniteElementSpace& parent_fes,
                                                    const mfem::Array<HYPRE_BigInt>& submesh2parent_vdof_list )
    : submesh_fes_( submesh_fes ), parent_fes_( parent_fes ), submesh2parent_vdof_list_( submesh2parent_vdof_list )
{
}

shared::ParSparseMat SubmeshParentTransferMat::Assemble() const
{
  // Build a pure injection matrix that maps submesh vdofs into the parent space.
  // Each submesh vdof contributes directly to exactly one parent vdof with a unit entry.
  auto submesh_parent_I = redecomp::ArrayUtility::IndexArray<int>( submesh2parent_vdof_list_.Size() + 1 );
  mfem::Vector submesh_parent_data( submesh2parent_vdof_list_.Size() );
  submesh_parent_data = 1.0;
  return shared::ParSparseMat(
      parent_fes_.GetComm(), submesh_fes_.GetVSize(), submesh_fes_.GlobalVSize(), parent_fes_.GlobalVSize(),
      submesh_parent_I.data(), const_cast<HYPRE_BigInt*>( submesh2parent_vdof_list_.GetData() ),
      submesh_parent_data.GetData(), submesh_fes_.GetDofOffsets(), parent_fes_.GetDofOffsets() );
}

RedecompJacobianAssembler::RedecompJacobianAssembler( const redecomp::MatrixTransfer& transfer,
                                                      std::vector<PackedPairJacobianContribs> contributions )
    : transfer_( transfer ), contributions_( std::move( contributions ) )
{
}

shared::ParSparseMat RedecompJacobianAssembler::Assemble() const
{
  axom::Array<int> row_redecomp_ids;
  axom::Array<int> col_redecomp_ids;
  axom::Array<double> jacobian_data;
  axom::Array<int> value_offsets;

  for ( const auto& contrib : contributions_ ) {
    SLIC_ERROR_ROOT_IF( contrib.row_elem_ids.size() != contrib.col_elem_ids.size() ||
                            contrib.row_elem_ids.size() != contrib.value_offsets.size(),
                        "PackedPairJacobianContribs arrays must have matching sizes." );
    SLIC_ERROR_ROOT_IF( contrib.row_elem_map == nullptr || contrib.col_elem_map == nullptr,
                        "PackedPairJacobianContribs must provide row/col element maps for redecomp transfer." );

    const int current_offset = jacobian_data.size();
    row_redecomp_ids.reserve( row_redecomp_ids.size() + contrib.numEntries() );
    for ( auto id : contrib.row_elem_ids ) {
      row_redecomp_ids.push_back( ( *contrib.row_elem_map )[static_cast<size_t>( id )] );
    }

    col_redecomp_ids.reserve( col_redecomp_ids.size() + contrib.numEntries() );
    for ( auto id : contrib.col_elem_ids ) {
      col_redecomp_ids.push_back( ( *contrib.col_elem_map )[static_cast<size_t>( id )] );
    }

    if ( contrib.jacobian_data.size() > 0 ) {
      jacobian_data.append( axom::ArrayView<const double>( contrib.jacobian_data ) );
    }

    // Offsets are stored relative to each contribution chunk, so remap them into the
    // flattened array passed to redecomp::MatrixTransfer.
    value_offsets.reserve( value_offsets.size() + contrib.value_offsets.size() );
    for ( auto offset : contrib.value_offsets ) {
      value_offsets.push_back( current_offset + offset );
    }
  }

  return transfer_.TransferToParallel( row_redecomp_ids, col_redecomp_ids, jacobian_data, value_offsets, false );
}

MfemJacobianTransfer::MfemJacobianTransfer( const MfemJacobianPath& row_path, const MfemJacobianPath& col_path )
    : row_path_( row_path ), col_path_( col_path )
{
  SLIC_ERROR_ROOT_IF( row_path_.final_fes == nullptr || row_path_.surface_fes == nullptr,
                      "MfemJacobianTransfer: row_path must provide non-null final_fes and surface_fes." );
  SLIC_ERROR_ROOT_IF( col_path_.final_fes == nullptr || col_path_.surface_fes == nullptr,
                      "MfemJacobianTransfer: col_path must provide non-null final_fes and surface_fes." );
  SLIC_ERROR_ROOT_IF( row_path_.ops.empty() || col_path_.ops.empty(),
                      "MfemJacobianTransfer: row/col operator chains must be non-empty." );
}

shared::ParSparseMat MfemJacobianTransfer::Assemble(
    const std::vector<PackedPairJacobianContribs>& contributions ) const
{
  SLIC_ERROR_ROOT_IF(
      contributions.empty(),
      "MfemJacobianTransfer::Assemble: requires a non-empty contributions vector so FE-space metadata is "
      "available on all ranks." );

  const auto* row_surface_fes = contributions.front().row_surface_fes;
  const auto* col_surface_fes = contributions.front().col_surface_fes;
  const auto* row_redecomp_fes = contributions.front().row_redecomp_fes;
  const auto* col_redecomp_fes = contributions.front().col_redecomp_fes;
  // "Surface" FE spaces are the parent FE spaces of the redecomp FE spaces used for the transfer. In this MFEM path
  // those surface FE spaces are either on the LOR surface mesh (when LOR is active) or on the boundary submesh
  // (otherwise), and they define the DOF layout of the intermediate redecomp-to-surface Jacobian.

  SLIC_ERROR_ROOT_IF( row_surface_fes == nullptr || col_surface_fes == nullptr,
                      "MfemJacobianTransfer::Assemble: contributions must provide row/col surface FE spaces." );
  SLIC_ERROR_ROOT_IF( row_redecomp_fes == nullptr || col_redecomp_fes == nullptr,
                      "MfemJacobianTransfer::Assemble: contributions must provide row/col redecomp FE spaces." );

  for ( const auto& contrib : contributions ) {
    SLIC_ERROR_ROOT_IF( contrib.row_surface_fes != row_surface_fes || contrib.col_surface_fes != col_surface_fes,
                        "MfemJacobianTransfer::Assemble: all contributions must belong to the same surface row/column "
                        "FE-space pairing." );
    SLIC_ERROR_ROOT_IF( contrib.row_redecomp_fes != row_redecomp_fes || contrib.col_redecomp_fes != col_redecomp_fes,
                        "MfemJacobianTransfer::Assemble: all contributions must belong to the same redecomp row/column "
                        "FE-space pairing." );
  }

  SLIC_ERROR_ROOT_IF( row_surface_fes != row_path_.surface_fes,
                      "MfemJacobianTransfer::Assemble: row contributions surface FE space does not match the selected "
                      "row transfer pathway." );
  SLIC_ERROR_ROOT_IF( col_surface_fes != col_path_.surface_fes,
                      "MfemJacobianTransfer::Assemble: column contributions surface FE space does not match the "
                      "selected column transfer pathway." );

  int local_pairs = 0;
  for ( const auto& contrib : contributions ) {
    local_pairs += contrib.numEntries();
  }
  int global_pairs = 0;
  MPI_Allreduce( &local_pairs, &global_pairs, 1, MPI_INT, MPI_SUM, row_surface_fes->GetComm() );

  // If there are no element pairs anywhere, build an explicit empty matrix on the appropriate surface-space DOF layout.
  shared::ParSparseMat surface_jacobian = [&]() -> shared::ParSparseMat {
    if ( global_pairs == 0 ) {
      mfem::SparseMatrix empty_diag( row_surface_fes->GetVSize(), col_surface_fes->GetVSize() );
      empty_diag.Finalize();
      return shared::ParSparseMat( row_surface_fes->GetComm(), row_surface_fes->GlobalVSize(),
                                   col_surface_fes->GlobalVSize(), row_surface_fes->GetDofOffsets(),
                                   col_surface_fes->GetDofOffsets(), std::move( empty_diag ) );
    }

    redecomp::MatrixTransfer transfer( *row_surface_fes, *col_surface_fes, *row_redecomp_fes, *col_redecomp_fes );
    RedecompJacobianAssembler redecomp_to_surface( transfer, contributions );
    return redecomp_to_surface.Assemble();
  }();

  JacobianAssembler assembler( row_path_.ops, col_path_.ops );
  return assembler.Assemble( std::move( surface_jacobian ) );
}

JacobianAssembler::JacobianAssembler( std::vector<shared::ParSparseMatView> row_transfer_ops,
                                      std::vector<shared::ParSparseMatView> col_transfer_ops )
    : row_transfer_ops_( std::move( row_transfer_ops ) ), col_transfer_ops_( std::move( col_transfer_ops ) )
{
}

shared::ParSparseMat JacobianAssembler::Assemble( shared::ParSparseMat lor_or_submesh_jacobian ) const
{
  // Apply column transfers on the right (reverse order) to build A * M_col.
  // Then apply row transfers on the left (reverse order, transposed) to build M_row^T * ( ... ).
  shared::ParSparseMat assembled = std::move( lor_or_submesh_jacobian );

  for ( size_t i = col_transfer_ops_.size(); i-- > 0; ) {
    shared::ParSparseMatView lhs( &assembled.get() );
    assembled = lhs * col_transfer_ops_[i];
  }

  for ( size_t i = row_transfer_ops_.size(); i-- > 0; ) {
    auto op_t = row_transfer_ops_[i].transpose();
    shared::ParSparseMatView lhs( &op_t.get() );
    shared::ParSparseMatView rhs( &assembled.get() );
    assembled = lhs * rhs;
  }

  return assembled;
}

SubmeshLORTransfer::SubmeshLORTransfer( mfem::ParFiniteElementSpace& submesh_fes, mfem::ParMesh& lor_mesh, bool use_ea )
    : lor_gridfn_{ CreateLORGridFunction(
          lor_mesh, std::make_unique<mfem::H1_FECollection>( 1, lor_mesh.SpaceDimension() ), submesh_fes.GetVDim() ) },
      lor_xfer_{ submesh_fes, *lor_gridfn_->ParFESpace() }
{
  lor_xfer_.UseEA( use_ea );
}

void SubmeshLORTransfer::TransferToLORGridFn( const mfem::ParGridFunction& submesh_src )
{
  SubmeshToLOR( submesh_src, *lor_gridfn_ );
}

void SubmeshLORTransfer::TransferFromLORVector( mfem::Vector& submesh_dst ) const
{
  lor_xfer_.ForwardOperator().MultTranspose( *lor_gridfn_, submesh_dst );
}

void SubmeshLORTransfer::SubmeshToLOR( const mfem::ParGridFunction& submesh_src, mfem::ParGridFunction& lor_dst )
{
  TRIBOL_MARK_FUNCTION;
  lor_xfer_.ForwardOperator().Mult( submesh_src, lor_dst );
}

std::unique_ptr<mfem::ParGridFunction> SubmeshLORTransfer::CreateLORGridFunction(
    mfem::ParMesh& lor_mesh, std::unique_ptr<mfem::FiniteElementCollection> lor_fec, int vdim )
{
  auto lor_gridfn = std::make_unique<mfem::ParGridFunction>(
      new mfem::ParFiniteElementSpace( &lor_mesh, lor_fec.get(), vdim, mfem::Ordering::byNODES ) );
  lor_gridfn->MakeOwner( lor_fec.release() );
  return lor_gridfn;
}

SubmeshRedecompTransfer::SubmeshRedecompTransfer( mfem::ParFiniteElementSpace& submesh_fes,
                                                  SubmeshLORTransfer* submesh_lor_xfer,
                                                  redecomp::RedecompMesh& redecomp_mesh )
    : submesh_fes_{ submesh_fes },
      redecomp_fes_{ submesh_lor_xfer
                         ? CreateRedecompFESpace( redecomp_mesh, *submesh_lor_xfer->GetLORGridFn().ParFESpace() )
                         : CreateRedecompFESpace( redecomp_mesh, submesh_fes_ ) },
      submesh_lor_xfer_{ submesh_lor_xfer },
      redecomp_xfer_{}  // default (element transfer) constructor
{
  // make sure submesh_fes is a submesh and redecomp's parent is submesh_fes's
  // submesh
  SLIC_ERROR_ROOT_IF( !mfem::ParSubMesh::IsParSubMesh( submesh_fes_.GetParMesh() ),
                      "submesh_fes must be on a ParSubMesh." );
  SLIC_ERROR_ROOT_IF( !submesh_lor_xfer && &redecomp_mesh.getParent() != submesh_fes_.GetParMesh(),
                      "redecomp's parent must match the submesh_fes ParMesh." );
  SLIC_ERROR_ROOT_IF(
      submesh_lor_xfer && &redecomp_mesh.getParent() != submesh_lor_xfer->GetLORGridFn().ParFESpace()->GetParMesh(),
      "redecomp's parent must match the submesh_fes ParMesh." );
}

void SubmeshRedecompTransfer::SubmeshToRedecomp( const mfem::ParGridFunction& submesh_src,
                                                 mfem::GridFunction& redecomp_dst ) const
{
  auto src_ptr = &submesh_src;
  if ( submesh_lor_xfer_ ) {
    submesh_lor_xfer_->GetLORGridFn() = 0.0;
    submesh_lor_xfer_->TransferToLORGridFn( submesh_src );
    src_ptr = &submesh_lor_xfer_->GetLORGridFn();
  }
  redecomp_xfer_.TransferToSerial( *src_ptr, redecomp_dst );
}

void SubmeshRedecompTransfer::RedecompToSubmesh( const mfem::GridFunction& redecomp_src,
                                                 mfem::Vector& submesh_dst ) const
{
  auto dst_ptr = &submesh_dst;
  auto dst_fespace_ptr = &submesh_fes_;
  // first initialize LOR grid function (if using LOR)
  if ( submesh_lor_xfer_ ) {
    submesh_lor_xfer_->GetLORVector() = 0.0;
    dst_ptr = &submesh_lor_xfer_->GetLORVector();
    dst_fespace_ptr = submesh_lor_xfer_->GetLORGridFn().ParFESpace();
  }
  // transfer data from redecomp mesh
  mfem::ParGridFunction dst_gridfn( dst_fespace_ptr, *dst_ptr );
  redecomp_xfer_.TransferToParallel( redecomp_src, dst_gridfn );
  dst_ptr->SyncMemory( dst_gridfn );

  // using redecomp, shared dof values are set equal (i.e. a ParGridFunction), but we want the sum of shared dof values
  // to equal the actual dof value when transferring dual fields (i.e. force and gap) back to the parallel mesh
  // following MFEMs convention.  set non-owned DOF values to zero.

  // P_I is the row index vector on the MFEM prolongation matrix. If there are no column entries for the row, then the
  // DOF is owned by another rank.
  auto dst_data = dst_ptr->ReadWrite( dst_ptr->UseDevice() );
  auto P_I = mfem::Read( dst_fespace_ptr->Dof_TrueDof_Matrix()->GetDiagMemoryI(), dst_fespace_ptr->GetVSize() + 1,
                         dst_ptr->UseDevice() );
  // set non-owned DOF values to zero.
  // P_I[i+1] == P_I[i] implies no diagonal entry, so the DOF is not owned.
  mfem::forall_switch( dst_ptr->UseDevice(), dst_fespace_ptr->GetVSize(), [=] MFEM_HOST_DEVICE( int i ) {
    if ( P_I[i + 1] == P_I[i] ) {
      dst_data[i] = 0.0;
    }
  } );
  // if using LOR, transfer data from LOR mesh to submesh
  if ( submesh_lor_xfer_ ) {
    submesh_lor_xfer_->TransferFromLORVector( submesh_dst );
  }
}

std::unique_ptr<mfem::FiniteElementSpace> SubmeshRedecompTransfer::CreateRedecompFESpace(
    redecomp::RedecompMesh& redecomp_mesh, mfem::ParFiniteElementSpace& submesh_fes )
{
  return std::make_unique<mfem::FiniteElementSpace>( &redecomp_mesh, submesh_fes.FEColl(), submesh_fes.GetVDim(),
                                                     mfem::Ordering::byNODES );
}

ParentRedecompTransfer::ParentRedecompTransfer( const mfem::ParFiniteElementSpace& parent_fes,
                                                mfem::ParGridFunction& submesh_gridfn,
                                                SubmeshLORTransfer* submesh_lor_xfer,
                                                redecomp::RedecompMesh& redecomp_mesh )
    : parent_fes_{ parent_fes },
      submesh_gridfn_{ submesh_gridfn },
      submesh_redecomp_xfer_{ *submesh_gridfn_.ParFESpace(), submesh_lor_xfer, redecomp_mesh }
{
  // Note: this is checked in the SubmeshRedecompTransfer constructor
  // SLIC_ERROR_ROOT_IF(
  //   !mfem::ParSubMesh::IsParSubMesh(submesh_gridfn_.ParFESpace()->GetParMesh()),
  //   "submesh_gridfn_ must be associated with an mfem::ParSubMesh."
  // );
  SLIC_ERROR_ROOT_IF( submesh_redecomp_xfer_.GetSubmesh().GetParent() != parent_fes_.GetParMesh(),
                      "submesh_gridfn's parent mesh must match the parent_fes ParMesh." );
}

void ParentRedecompTransfer::ParentToRedecomp( const mfem::ParGridFunction& parent_src,
                                               mfem::GridFunction& redecomp_dst ) const
{
  submesh_gridfn_ = 0.0;
  submesh_redecomp_xfer_.GetSubmesh().Transfer( parent_src, submesh_gridfn_ );
  submesh_redecomp_xfer_.SubmeshToRedecomp( submesh_gridfn_, redecomp_dst );
}

void ParentRedecompTransfer::RedecompToParent( const mfem::GridFunction& redecomp_src, mfem::Vector& parent_dst ) const
{
  submesh_gridfn_ = 0.0;
  submesh_redecomp_xfer_.RedecompToSubmesh( redecomp_src, submesh_gridfn_ );
  // submesh transfer requires a grid function.  create one using parent_dst's data
  mfem::ParGridFunction parent_gridfn( &parent_fes_, parent_dst );
  submesh_redecomp_xfer_.GetSubmesh().Transfer( submesh_gridfn_, parent_gridfn );
  parent_dst.SyncMemory( parent_gridfn );
}

void ParentRedecompTransfer::RedecompToSubmesh( const mfem::GridFunction& redecomp_src,
                                                mfem::ParGridFunction& submesh_dst ) const
{
  submesh_dst = 0.0;
  submesh_redecomp_xfer_.RedecompToSubmesh( redecomp_src, submesh_dst );
}

ParentField::ParentField( const mfem::ParGridFunction& parent_gridfn ) : parent_gridfn_{ parent_gridfn } {}

void ParentField::SetParentGridFn( const mfem::ParGridFunction& parent_gridfn )
{
  parent_gridfn_ = parent_gridfn;
  update_data_.reset( nullptr );
}

void ParentField::UpdateField( ParentRedecompTransfer& parent_redecomp_xfer, bool use_device )
{
  update_data_ = std::make_unique<UpdateData>( parent_redecomp_xfer, parent_gridfn_, use_device );
}

std::vector<const RealT*> ParentField::GetRedecompFieldPtrs() const
{
  auto data_ptrs = std::vector<const RealT*>( 3, nullptr );
  if ( GetRedecompGridFn().FESpace()->GetNDofs() > 0 ) {
    auto data = GetRedecompGridFn().Read( GetRedecompGridFn().UseDevice() );
    for ( size_t i{}; i < static_cast<size_t>( GetRedecompGridFn().FESpace()->GetVDim() ); ++i ) {
      data_ptrs[i] = data + GetRedecompGridFn().FESpace()->DofToVDof( 0, i );
    }
  }
  return data_ptrs;
}

std::vector<RealT*> ParentField::GetRedecompFieldPtrs( mfem::GridFunction& redecomp_gridfn )
{
  auto data_ptrs = std::vector<RealT*>( 3, nullptr );
  if ( redecomp_gridfn.FESpace()->GetNDofs() > 0 ) {
    auto data = redecomp_gridfn.ReadWrite( redecomp_gridfn.UseDevice() );
    for ( size_t i{}; i < static_cast<size_t>( redecomp_gridfn.FESpace()->GetVDim() ); ++i ) {
      data_ptrs[i] = data + redecomp_gridfn.FESpace()->DofToVDof( 0, i );
    }
  }
  return data_ptrs;
}

ParentField::UpdateData& ParentField::GetUpdateData()
{
  SLIC_ERROR_ROOT_IF( update_data_ == nullptr, "UpdateField() must be called to generate UpdateData." );
  return *update_data_;
}

const ParentField::UpdateData& ParentField::GetUpdateData() const
{
  SLIC_ERROR_ROOT_IF( update_data_ == nullptr, "UpdateField() must be called to generate UpdateData." );
  return *update_data_;
}

ParentField::UpdateData::UpdateData( ParentRedecompTransfer& parent_redecomp_xfer,
                                     const mfem::ParGridFunction& parent_gridfn, bool use_device )
    : parent_redecomp_xfer_{ parent_redecomp_xfer }, redecomp_gridfn_{ &parent_redecomp_xfer.GetRedecompFESpace() }
{
  TRIBOL_MARK_FUNCTION;
  redecomp_gridfn_.UseDevice( use_device );
  redecomp_gridfn_ = 0.0;
  parent_redecomp_xfer_.ParentToRedecomp( parent_gridfn, redecomp_gridfn_ );
}

PressureField::PressureField( const mfem::ParGridFunction& submesh_gridfn ) : submesh_gridfn_{ submesh_gridfn } {}

void PressureField::SetSubmeshField( const mfem::ParGridFunction& submesh_gridfn )
{
  submesh_gridfn_ = submesh_gridfn;
  update_data_.reset( nullptr );
}

void PressureField::UpdateField( SubmeshRedecompTransfer& submesh_redecomp_xfer )
{
  update_data_ = std::make_unique<UpdateData>( submesh_redecomp_xfer, submesh_gridfn_ );
}

std::vector<const RealT*> PressureField::GetRedecompFieldPtrs() const
{
  auto data_ptrs = std::vector<const RealT*>( 3, nullptr );
  if ( GetRedecompGridFn().FESpace()->GetNDofs() > 0 ) {
    auto data = GetRedecompGridFn().Read( GetRedecompGridFn().UseDevice() );
    for ( size_t i{}; i < static_cast<size_t>( GetRedecompGridFn().FESpace()->GetVDim() ); ++i ) {
      data_ptrs[i] = data + GetRedecompGridFn().FESpace()->DofToVDof( 0, i );
    }
  }
  return data_ptrs;
}

std::vector<RealT*> PressureField::GetRedecompFieldPtrs( mfem::GridFunction& redecomp_gridfn )
{
  auto data_ptrs = std::vector<RealT*>( 3, nullptr );
  if ( redecomp_gridfn.FESpace()->GetNDofs() > 0 ) {
    auto data = redecomp_gridfn.ReadWrite( redecomp_gridfn.UseDevice() );
    for ( size_t i{}; i < static_cast<size_t>( redecomp_gridfn.FESpace()->GetVDim() ); ++i ) {
      data_ptrs[i] = data + redecomp_gridfn.FESpace()->DofToVDof( 0, i );
    }
  }
  return data_ptrs;
}

PressureField::UpdateData& PressureField::GetUpdateData()
{
  SLIC_ERROR_ROOT_IF( update_data_ == nullptr, "UpdateField() must be called to generate UpdateData." );
  return *update_data_;
}

const PressureField::UpdateData& PressureField::GetUpdateData() const
{
  SLIC_ERROR_ROOT_IF( update_data_ == nullptr, "UpdateField() must be called to generate UpdateData." );
  return *update_data_;
}

PressureField::UpdateData::UpdateData( SubmeshRedecompTransfer& submesh_redecomp_xfer,
                                       const mfem::ParGridFunction& submesh_gridfn )
    : submesh_redecomp_xfer_{ submesh_redecomp_xfer }, redecomp_gridfn_{ &submesh_redecomp_xfer.GetRedecompFESpace() }
{
  // keep on host since tribol always does mortar computations there (update when mortar is on gpu)
  redecomp_gridfn_.UseDevice( false );
  redecomp_gridfn_ = 0.0;
  submesh_redecomp_xfer_.SubmeshToRedecomp( submesh_gridfn, redecomp_gridfn_ );
}

MfemMeshData::MfemMeshData( IndexT mesh_id_1, IndexT mesh_id_2, const mfem::ParMesh& parent_mesh,
                            const mfem::ParGridFunction& current_coords, std::set<int>&& attributes_1,
                            std::set<int>&& attributes_2, ExecutionMode exec_mode, MemorySpace mem_space )
    : mesh_id_1_{ mesh_id_1 },
      mesh_id_2_{ mesh_id_2 },
      parent_mesh_{ parent_mesh },
      attributes_1_{ std::move( attributes_1 ) },
      attributes_2_{ std::move( attributes_2 ) },
      submesh_{ CreateSubmesh( parent_mesh_, attributes_1_, attributes_2_ ) },
      coords_{ current_coords },
      lor_factor_{ 0 },
      exec_mode_{ exec_mode },
      mem_space_{ mem_space },
      use_device_{ isOnDevice( exec_mode ) },
      allocator_id_{ getResourceAllocatorID( mem_space ) }
{
  // make sure a grid function exists on the submesh
  submesh_.EnsureNodes();

  // create submesh grid function
  std::unique_ptr<mfem::FiniteElementCollection> submesh_fec{
      current_coords.ParFESpace()->FEColl()->Clone( current_coords.ParFESpace()->FEColl()->GetOrder() ) };
  submesh_xfer_gridfn_.SetSpace( new mfem::ParFiniteElementSpace(
      &submesh_, submesh_fec.get(), current_coords.ParFESpace()->GetVDim(), mfem::Ordering::byNODES ) );
  submesh_xfer_gridfn_.MakeOwner( submesh_fec.release() );

  // build LOR submesh
  if ( current_coords.FESpace()->FEColl()->GetOrder() > 1 ) {
    SetLORFactor( current_coords.FESpace()->FEColl()->GetOrder() );
  }

  // set default redecomp trigger displacement
  auto mpi = redecomp::MPIUtility( parent_mesh_.GetComm() );
  redecomp_trigger_displacement_ = redecomp::RedecompMesh::MaxElementSize( parent_mesh_, mpi );
}

void MfemMeshData::SetParentCoords( const mfem::ParGridFunction& current_coords )
{
  coords_.SetParentGridFn( current_coords );
}

void MfemMeshData::SetParentReferenceCoords( const mfem::ParGridFunction& reference_coords )
{
  if ( reference_coords_ ) {
    reference_coords_->SetParentGridFn( reference_coords );
  } else {
    reference_coords_ = std::make_unique<ParentField>( reference_coords );
  }
}

bool MfemMeshData::UpdateMfemMeshData( RealT binning_proximity_scale, int n_ranks, bool force_new_redecomp )
{
  TRIBOL_MARK_FUNCTION;

  // check if redecomp mesh needs to be updated
  TRIBOL_MARK_BEGIN( "Check if new Redecomp mesh is needed" );
  if ( !force_new_redecomp && update_data_ ) {
    // compute max displacement change
    auto& current_coords_gf = coords_.GetParentGridFn();
    // Use inf-norm of coordinate differences as a proxy for max displacement change.
    const RealT* d_curr = current_coords_gf.Read( use_device_ );
    const RealT* d_last = coords_at_last_redecomp_.Read( use_device_ );
    mfem::Vector max_diff( 1 );
    max_diff.UseDevice( use_device_ );
    max_diff = 0.0;
    RealT* d_max_diff = max_diff.Write( use_device_ );
    // Note: A true reduction would be more efficient here.
    // However, it is not currently implemented because forAllExec would need
    // to know the execution mode at compile time to instantiate the correct reducer.
    forAllExec( exec_mode_, current_coords_gf.Size(), [d_curr, d_last, d_max_diff] TRIBOL_HOST_DEVICE( int i ) {
      tribol::atomicMax( d_max_diff, std::abs( d_curr[i] - d_last[i] ) );
    } );
    RealT* h_max_diff = max_diff.HostReadWrite();
    // Allreduce to get global max
    MPI_Allreduce( MPI_IN_PLACE, h_max_diff, 1, MPI_DOUBLE, MPI_MAX, parent_mesh_.GetComm() );

    // If max change is greater than threshold, make a new RedecompMesh
    // NOTE: max_diff is the max component diff, i.e. x, y, z components are considered separately
    if ( *h_max_diff > redecomp_trigger_displacement_ ) {
      force_new_redecomp = true;
    }
  }
  TRIBOL_MARK_END( "Check if new Redecomp mesh is needed" );

  TRIBOL_MARK_BEGIN( "Build new Redecomp mesh" );
  bool rebuilt = false;
  if ( force_new_redecomp || !update_data_ ) {
    // update coordinates of submesh and LOR mesh
    auto submesh_nodes = dynamic_cast<mfem::ParGridFunction*>( submesh_.GetNodes() );
    SLIC_ERROR_ROOT_IF( !submesh_nodes, "submesh_ Nodes is not a ParGridFunction." );
    TRIBOL_MARK_BEGIN( "Update SubMesh coords" );
    submesh_.Transfer( coords_.GetParentGridFn(), *submesh_nodes );
    TRIBOL_MARK_END( "Update SubMesh coords" );
    if ( lor_mesh_.get() ) {
      TRIBOL_MARK_BEGIN( "Update LOR coords" );
      auto lor_nodes = dynamic_cast<mfem::ParGridFunction*>( lor_mesh_->GetNodes() );
      SLIC_ERROR_ROOT_IF( !lor_nodes, "lor_mesh_ Nodes is not a ParGridFunction." );
      submesh_lor_xfer_->SubmeshToLOR( *submesh_nodes, *lor_nodes );
      TRIBOL_MARK_END( "Update LOR coords" );
    }
    update_data_ =
        std::make_unique<UpdateData>( submesh_, lor_mesh_.get(), *coords_.GetParentGridFn().ParFESpace(),
                                      submesh_xfer_gridfn_, submesh_lor_xfer_.get(), attributes_1_, attributes_2_,
                                      binning_proximity_scale, n_ranks, allocator_id_, redecomp_trigger_displacement_ );
    rebuilt = true;
  }

  // this is done here so the redecomp grid fn is updated before we update redecomp_response_
  coords_.UpdateField( update_data_->vector_xfer_, use_device_ );

  if ( rebuilt ) {
    // NOTE: SetSpace() would be preferrable to call here, but it looks like all memory isn't mapped to
    // mfem::MemoryManager when this is used. TODO: Debug this and switch to SetSpace()
    redecomp_response_ = std::make_unique<mfem::GridFunction>( coords_.GetRedecompGridFn().FESpace() );
    redecomp_response_->UseDevice( use_device_ );

    // Store current coordinates
    coords_at_last_redecomp_.SetSize( coords_.GetParentGridFn().Size() );
    coords_at_last_redecomp_ = coords_.GetParentGridFn();
  }
  TRIBOL_MARK_END( "Build new Redecomp mesh" );

  TRIBOL_MARK_BEGIN( "Copy fields to Redecomp mesh" );
  ( *redecomp_response_ ) = 0.0;

  if ( reference_coords_ ) {
    reference_coords_->UpdateField( update_data_->vector_xfer_, use_device_ );
  }
  if ( velocity_ ) {
    velocity_->UpdateField( update_data_->vector_xfer_, use_device_ );
  }
  TRIBOL_MARK_END( "Copy fields to Redecomp mesh" );

  if ( rebuilt && elem_thickness_ ) {
    if ( !material_modulus_ ) {
      SLIC_ERROR_ROOT(
          "Kinematic element penalty requires material modulus information. "
          "Call registerMfemMaterialModulus() to set this." );
    }
    TRIBOL_MARK_BEGIN( "Copy element thickness to Redecomp mesh" );
    redecomp::RedecompTransfer redecomp_xfer;
    // set element thickness on redecomp mesh
    redecomp_elem_thickness_ =
        std::make_unique<mfem::QuadratureFunction>( new mfem::QuadratureSpace( &GetRedecompMesh(), 0 ) );
    redecomp_elem_thickness_->SetOwnsSpace( true );
    redecomp_elem_thickness_->UseDevice( use_device_ );
    *redecomp_elem_thickness_ = 0.0;
    redecomp_xfer.TransferToSerial( *elem_thickness_, *redecomp_elem_thickness_ );
    // set element thickness on tribol mesh
    tribol_elem_thickness_1_ = std::make_unique<ArrayT<RealT>>(
        GetElemMap1().size(), GetElemMap1().empty() ? 1 : GetElemMap1().size(), allocator_id_ );
    auto redecomp_t_view = redecomp_elem_thickness_->Read( use_device_ );
    ArrayViewT<RealT> tribol_t1_view( *tribol_elem_thickness_1_ );
    ArrayViewT<const int> elem_map1_view( GetElemMap1() );
    // NOTE: this assumes 1 thickness value per element. This is NOT true, in general, for mfem::QuadratureFunction.
    forAllExec( exec_mode_, GetElemMap1().size(),
                [tribol_t1_view, redecomp_t_view, elem_map1_view] TRIBOL_HOST_DEVICE( int i ) {
                  tribol_t1_view[i] = redecomp_t_view[elem_map1_view[i]];
                } );
    tribol_elem_thickness_2_ = std::make_unique<ArrayT<RealT>>(
        GetElemMap2().size(), GetElemMap2().empty() ? 1 : GetElemMap2().size(), allocator_id_ );
    ArrayViewT<RealT> tribol_t2_view( *tribol_elem_thickness_2_ );
    ArrayViewT<const int> elem_map2_view( GetElemMap2() );
    // NOTE: this assumes 1 thickness value per element. This is NOT true, in general, for mfem::QuadratureFunction.
    forAllExec( exec_mode_, GetElemMap2().size(),
                [tribol_t2_view, redecomp_t_view, elem_map2_view] TRIBOL_HOST_DEVICE( int i ) {
                  tribol_t2_view[i] = redecomp_t_view[elem_map2_view[i]];
                } );
    // set material modulus on redecomp mesh
    redecomp_material_modulus_ =
        std::make_unique<mfem::QuadratureFunction>( new mfem::QuadratureSpace( &GetRedecompMesh(), 0 ) );
    redecomp_material_modulus_->SetOwnsSpace( true );
    redecomp_material_modulus_->UseDevice( use_device_ );
    *redecomp_material_modulus_ = 0.0;
    redecomp_xfer.TransferToSerial( *material_modulus_, *redecomp_material_modulus_ );
    // set material modulus on tribol mesh
    tribol_material_modulus_1_ = std::make_unique<ArrayT<RealT>>(
        GetElemMap1().size(), GetElemMap1().empty() ? 1 : GetElemMap1().size(), allocator_id_ );
    auto redecomp_m_view = redecomp_material_modulus_->Read( use_device_ );
    ArrayViewT<RealT> tribol_m1_view( *tribol_material_modulus_1_ );
    // NOTE: this assumes 1 thickness value per element. This is NOT true, in general, for mfem::QuadratureFunction.
    forAllExec( exec_mode_, GetElemMap1().size(),
                [tribol_m1_view, redecomp_m_view, elem_map1_view] TRIBOL_HOST_DEVICE( int i ) {
                  tribol_m1_view[i] = redecomp_m_view[elem_map1_view[i]];
                } );
    tribol_material_modulus_2_ = std::make_unique<ArrayT<RealT>>(
        GetElemMap2().size(), GetElemMap2().empty() ? 1 : GetElemMap2().size(), allocator_id_ );
    ArrayViewT<RealT> tribol_m2_view( *tribol_material_modulus_2_ );
    // NOTE: this assumes 1 thickness value per element. This is NOT true, in general, for mfem::QuadratureFunction.
    forAllExec( exec_mode_, GetElemMap2().size(),
                [tribol_m2_view, redecomp_m_view, elem_map2_view] TRIBOL_HOST_DEVICE( int i ) {
                  tribol_m2_view[i] = redecomp_m_view[elem_map2_view[i]];
                } );
    TRIBOL_MARK_END( "Copy element thickness to Redecomp mesh" );
  }

  return rebuilt;
}

void MfemMeshData::GetParentResponse( mfem::Vector& r ) const
{
  GetParentRedecompTransfer().RedecompToParent( *redecomp_response_, r );
}

void MfemMeshData::SetParentVelocity( const mfem::ParGridFunction& velocity )
{
  if ( velocity_ ) {
    velocity_->SetParentGridFn( velocity );
  } else {
    velocity_ = std::make_unique<ParentField>( velocity );
  }
}

void MfemMeshData::ClearAllPenaltyData()
{
  ClearRatePenaltyData();
  kinematic_constant_penalty_1_.reset( nullptr );
  kinematic_constant_penalty_2_.reset( nullptr );
  kinematic_penalty_scale_1_.reset( nullptr );
  kinematic_penalty_scale_2_.reset( nullptr );
  viscous_damping_coeff_1_.reset( nullptr );
  viscous_damping_coeff_2_.reset( nullptr );
  elem_thickness_.reset( nullptr );
  redecomp_elem_thickness_.reset( nullptr );
  tribol_elem_thickness_1_.reset( nullptr );
  tribol_elem_thickness_2_.reset( nullptr );
  material_modulus_.reset( nullptr );
  redecomp_material_modulus_.reset( nullptr );
  tribol_material_modulus_1_.reset( nullptr );
  tribol_material_modulus_2_.reset( nullptr );
}

void MfemMeshData::ClearRatePenaltyData()
{
  rate_constant_penalty_1_.reset( nullptr );
  rate_constant_penalty_2_.reset( nullptr );
  rate_percent_ratio_1_.reset( nullptr );
  rate_percent_ratio_2_.reset( nullptr );
}

void MfemMeshData::SetLORFactor( int lor_factor )
{
  if ( lor_factor <= 1 ) {
    SLIC_WARNING_ROOT( "lor_factor must be an integer > 1.  LOR factor not changed." );
    return;
  }
  if ( coords_.GetParentGridFn().FESpace()->FEColl()->GetOrder() <= 1 ) {
    SLIC_WARNING_ROOT(
        "lor_factor is only applicable to higher order geometry.  "
        "LOR factor not changed." );
    return;
  }
  lor_factor_ = lor_factor;
  // note: calls ParMesh's move ctor
  lor_mesh_ = std::make_unique<mfem::ParMesh>(
      mfem::ParMesh::MakeRefined( submesh_, lor_factor, mfem::BasisType::ClosedUniform ) );
  lor_mesh_->EnsureNodes();
  submesh_lor_xfer_ =
      std::make_unique<SubmeshLORTransfer>( *submesh_xfer_gridfn_.ParFESpace(), *lor_mesh_, use_device_ );
}

void MfemMeshData::ComputeElementThicknesses()
{
  const bool has_reference_coords = reference_coords_ != nullptr;
  const auto& thickness_coords =
      has_reference_coords ? reference_coords_->GetParentGridFn() : coords_.GetParentGridFn();
  SLIC_WARNING_ROOT_IF( !has_reference_coords,
                        "tribol::MfemMeshData::ComputeElementThicknesses(): no MFEM reference coordinates "
                        "registered; calculating element thickness from current coordinates." );

  auto submesh_thickness = std::make_unique<mfem::QuadratureFunction>( new mfem::QuadratureSpace( &submesh_, 0 ) );
  submesh_thickness->SetOwnsSpace( true );
  // All the elements in the submesh are on the contact surface. The algorithm
  // works as follows:
  // 1) For each submesh element, find the corresponding parent volume element
  // 2) Compute the thickness of the parent volume element (det J at element
  //    centroid)
  // 3) If no LOR mesh, store this on a quadrature function on the submesh
  // 4) If there is an LOR mesh, use the CoarseFineTransformation to find the
  //    LOR elements linked to the HO mesh and store the thickness of the HO
  //    element on all of its linked LOR elements.
  for ( int submesh_e{ 0 }; submesh_e < submesh_.GetNE(); ++submesh_e ) {
    // Step 1
    auto parent_bdr_e = submesh_.GetParentElementIDMap()[submesh_e];
    auto& parent_mesh = const_cast<mfem::ParMesh&>( parent_mesh_ );
    auto& face_el_tr = *parent_mesh.GetBdrFaceTransformations( parent_bdr_e );
    auto mask = face_el_tr.GetConfigurationMask();
    auto parent_e = ( mask & mfem::FaceElementTransformations::HAVE_ELEM1 ) ? face_el_tr.Elem1No : face_el_tr.Elem2No;

    // Step 2
    // normal = (dx/dxi x dx/deta) / || dx/dxi x dx/deta || on parent volume boundary element centroid
    auto& parent_fes = *thickness_coords.ParFESpace();
    mfem::Array<int> be_dofs;
    parent_fes.GetBdrElementDofs( parent_bdr_e, be_dofs );
    mfem::DenseMatrix elem_coords( parent_mesh_.Dimension(), be_dofs.Size() );
    for ( int d{ 0 }; d < parent_mesh_.Dimension(); ++d ) {
      mfem::Array<int> be_vdofs( be_dofs );
      parent_fes.DofsToVDofs( d, be_vdofs );
      mfem::Vector elemvect( be_dofs.Size() );
      thickness_coords.GetSubVector( be_vdofs, elemvect );
      elem_coords.SetRow( d, elemvect );
    }
    auto& be = *parent_fes.GetBE( parent_bdr_e );
    // create an integration point at the element centroid
    mfem::IntegrationPoint ip;
    ip.Init( 0 );
    mfem::DenseMatrix dshape( be_dofs.Size(), submesh_.Dimension() );
    // calculate shape function derivatives at the surface element centroid
    be.CalcDShape( ip, dshape );
    mfem::DenseMatrix dxdxi_mat( parent_mesh_.Dimension(), submesh_.Dimension() );
    mfem::Mult( elem_coords, dshape, dxdxi_mat );
    mfem::Vector norm( parent_mesh_.Dimension() );
    mfem::CalcOrtho( dxdxi_mat, norm );

    // This mirrors mfem::Mesh::GetElementSize(i, dir), but builds the element Jacobian from thickness_coords so it also
    // works when the parent mesh does not store coordinates in a Nodes GridFunction.
    mfem::Array<int> elem_dofs;
    parent_fes.GetElementDofs( parent_e, elem_dofs );
    mfem::DenseMatrix elem_coords_vol( parent_mesh_.Dimension(), elem_dofs.Size() );
    for ( int d{ 0 }; d < parent_mesh_.Dimension(); ++d ) {
      mfem::Array<int> elem_vdofs( elem_dofs );
      parent_fes.DofsToVDofs( d, elem_vdofs );
      mfem::Vector elemvect( elem_dofs.Size() );
      thickness_coords.GetSubVector( elem_vdofs, elemvect );
      elem_coords_vol.SetRow( d, elemvect );
    }
    auto& parent_fe = *parent_fes.GetFE( parent_e );
    mfem::IntegrationPoint ip_vol;
    ip_vol.Init( 0 );
    mfem::DenseMatrix dshape_vol( elem_dofs.Size(), parent_mesh_.Dimension() );
    parent_fe.CalcDShape( ip_vol, dshape_vol );
    mfem::DenseMatrix J( parent_mesh_.Dimension(), parent_mesh_.Dimension() );
    mfem::Mult( elem_coords_vol, dshape_vol, J );

    mfem::Vector d_hat( parent_mesh_.Dimension() );
    J.MultTranspose( norm, d_hat );
    double h = std::sqrt( ( d_hat * d_hat ) / ( norm * norm ) );

    // Step 3
    mfem::Vector quad_val;
    submesh_thickness->GetValues( submesh_e, quad_val );
    quad_val[0] = h;
  }

  // Step 4
  if ( GetLORMesh() ) {
    elem_thickness_ = std::make_unique<mfem::QuadratureFunction>( new mfem::QuadratureSpace( GetLORMesh(), 0 ) );
    elem_thickness_->SetOwnsSpace( true );
    for ( int lor_e{ 0 }; lor_e < GetLORMesh()->GetNE(); ++lor_e ) {
      auto submesh_e = GetLORMesh()->GetRefinementTransforms().embeddings[lor_e].parent;
      mfem::Vector submesh_val;
      submesh_thickness->GetValues( submesh_e, submesh_val );
      mfem::Vector lor_val;
      elem_thickness_->GetValues( lor_e, lor_val );
      lor_val[0] = submesh_val[0];
    }
  } else {
    elem_thickness_ = std::move( submesh_thickness );
  }
}

void MfemMeshData::SetMaterialModulus( mfem::Coefficient& modulus_field )
{
  material_modulus_ = std::make_unique<mfem::QuadratureFunction>(
      new mfem::QuadratureSpace( GetLORMesh() ? GetLORMesh() : &submesh_, 0 ) );
  material_modulus_->SetOwnsSpace( true );
  // TODO: why isn't Project() const?
  modulus_field.Project( *material_modulus_ );
}

MfemMeshData::UpdateData::UpdateData( mfem::ParSubMesh& submesh, mfem::ParMesh* lor_mesh,
                                      const mfem::ParFiniteElementSpace& parent_fes,
                                      mfem::ParGridFunction& submesh_gridfn, SubmeshLORTransfer* submesh_lor_xfer,
                                      const std::set<int>& attributes_1, const std::set<int>& attributes_2,
                                      RealT binning_proximity_scale, int n_ranks, int allocator_id,
                                      RealT redecomp_trigger_displacement )
    : redecomp_mesh_{ lor_mesh
                          ? redecomp::RedecompMesh(
                                *lor_mesh,
                                binning_proximity_scale * redecomp::RedecompMesh::MaxElementSize(
                                                              *lor_mesh, redecomp::MPIUtility( lor_mesh->GetComm() ) ) +
                                    redecomp_trigger_displacement,
                                redecomp::RedecompMesh::RCB, n_ranks )
                          : redecomp::RedecompMesh(
                                submesh,
                                binning_proximity_scale * redecomp::RedecompMesh::MaxElementSize(
                                                              submesh, redecomp::MPIUtility( submesh.GetComm() ) ) +
                                    redecomp_trigger_displacement,
                                redecomp::RedecompMesh::RCB, n_ranks ) },
      vector_xfer_{ parent_fes, submesh_gridfn, submesh_lor_xfer, redecomp_mesh_ },
      allocator_id_{ allocator_id }
{
  TRIBOL_MARK_FUNCTION;
  // set element type based on redecomp mesh
  SetElementData();
  // updates the connectivity of the tribol surface mesh
  UpdateConnectivity( attributes_1, attributes_2 );
}

void MfemMeshData::UpdateData::UpdateConnectivity( const std::set<int>& attributes_1,
                                                   const std::set<int>& attributes_2 )
{
  // create this on host since MFEM connectivity data is stored there
  Array2D<IndexT, MemorySpace::Host> conn_1_host;
  Array2D<IndexT, MemorySpace::Host> conn_2_host;
  Array1D<int, MemorySpace::Host> elem_map_1_host;
  Array1D<int, MemorySpace::Host> elem_map_2_host;
  conn_1_host.reserve( redecomp_mesh_.GetNE() * num_verts_per_elem_ );
  conn_2_host.reserve( redecomp_mesh_.GetNE() * num_verts_per_elem_ );
  elem_map_1_host.reserve( static_cast<size_t>( redecomp_mesh_.GetNE() ) );
  elem_map_2_host.reserve( static_cast<size_t>( redecomp_mesh_.GetNE() ) );
  for ( int e{}; e < redecomp_mesh_.GetNE(); ++e ) {
    auto elem_attrib = redecomp_mesh_.GetAttribute( e );
    auto elem_conn = mfem::Array<int>();
    redecomp_mesh_.GetElementVertices( e, elem_conn );
    for ( auto attribute_1 : attributes_1 ) {
      if ( attribute_1 == elem_attrib ) {
        elem_map_1_host.push_back( e );
        conn_1_host.resize( elem_map_1_host.size(), num_verts_per_elem_ );
        for ( int v{}; v < num_verts_per_elem_; ++v ) {
          conn_1_host( elem_map_1_host.size() - 1, v ) = elem_conn[v];
        }
        break;
      }
    }
    for ( auto attribute_2 : attributes_2 ) {
      if ( attribute_2 == elem_attrib ) {
        elem_map_2_host.push_back( e );
        conn_2_host.resize( elem_map_2_host.size(), num_verts_per_elem_ );
        for ( int v{}; v < num_verts_per_elem_; ++v ) {
          conn_2_host( elem_map_2_host.size() - 1, v ) = elem_conn[v];
        }
        break;
      }
    }
  }
  if ( allocator_id_ == conn_1_host.getAllocatorID() ) {
    // same memory space, just move
    conn_1_ = std::move( conn_1_host );
    conn_2_ = std::move( conn_2_host );
    elem_map_1_ = std::move( elem_map_1_host );
    elem_map_2_ = std::move( elem_map_2_host );
  } else {
    // copy to new memory space
    conn_1_ = Array2D<IndexT>( conn_1_host, allocator_id_ );
    conn_2_ = Array2D<IndexT>( conn_2_host, allocator_id_ );
    elem_map_1_ = Array1D<int>( elem_map_1_host, allocator_id_ );
    elem_map_2_ = Array1D<int>( elem_map_2_host, allocator_id_ );
  }
}

MfemMeshData::UpdateData& MfemMeshData::GetUpdateData()
{
  SLIC_ERROR_ROOT_IF( update_data_ == nullptr, "UpdateField() must be called to generate UpdateData." );
  return *update_data_;
}

const MfemMeshData::UpdateData& MfemMeshData::GetUpdateData() const
{
  SLIC_ERROR_ROOT_IF( update_data_ == nullptr, "UpdateField() must be called to generate UpdateData." );
  return *update_data_;
}

mfem::ParSubMesh MfemMeshData::CreateSubmesh( const mfem::ParMesh& parent_mesh, const std::set<int>& attributes_1,
                                              const std::set<int>& attributes_2 )
{
  // TODO: Create PR for mfem::ParSubMesh::CreateFromBoundary taking a const
  // reference to attributes. Then we can construct submesh_ in the initializer
  // list without this function (because CreateFromBoundary will be willing to
  // take an rvalue for attributes)
  // NOTE (EBC): This has been updated in the latest MFEM. Make the change when MFEM is updated.
  auto attributes_array = arrayFromSet( mergeContainers( attributes_1, attributes_2 ) );
  // NOTE (EBC): The Nodes ParGridFunction is created on host. No support for creating this on device yet.
  return mfem::ParSubMesh::CreateFromBoundary( parent_mesh, attributes_array );
}

void MfemMeshData::UpdateData::SetElementData()
{
  if ( redecomp_mesh_.GetNE() > 0 ) {
    auto element_type = redecomp_mesh_.GetElementType( 0 );

    switch ( element_type ) {
      case mfem::Element::SEGMENT:
        elem_type_ = LINEAR_EDGE;
        break;
      case mfem::Element::TRIANGLE:
        elem_type_ = LINEAR_TRIANGLE;
        break;
      case mfem::Element::QUADRILATERAL:
        elem_type_ = LINEAR_QUAD;
        break;
      case mfem::Element::TETRAHEDRON:
        elem_type_ = LINEAR_TET;
        break;
      case mfem::Element::HEXAHEDRON:
        elem_type_ = LINEAR_HEX;
        break;

      case mfem::Element::POINT:
        SLIC_ERROR_ROOT( "Unsupported element type!" );
        break;

      default:
        SLIC_ERROR_ROOT( "Unknown element type!" );
        break;
    }

    num_verts_per_elem_ = mfem::Geometry::NumVerts[element_type];
  } else {
    // just put something here so Tribol will not give a warning for zero element meshes.  use a 2d element so arrays
    // are sized for 3d (max supported dimension) in case they are accessed later on.
    elem_type_ = LINEAR_QUAD;
    num_verts_per_elem_ = 2;
  }
}

MfemSubmeshData::MfemSubmeshData( mfem::ParSubMesh& submesh, mfem::ParMesh* lor_mesh,
                                  std::unique_ptr<mfem::FiniteElementCollection> pressure_fec, int pressure_vdim,
                                  bool use_device )
    : submesh_pressure_{ new mfem::ParFiniteElementSpace( &submesh, pressure_fec.get(), pressure_vdim ) },
      pressure_{ submesh_pressure_ },
      submesh_lor_xfer_{ lor_mesh ? std::make_unique<SubmeshLORTransfer>( *submesh_pressure_.ParFESpace(), *lor_mesh )
                                  : nullptr },
      use_device_{ use_device }
{
  submesh_pressure_.MakeOwner( pressure_fec.release() );
  submesh_pressure_ = 0.0;
}

void MfemSubmeshData::SetLORMesh( mfem::ParMesh* lor_mesh )
{
  submesh_lor_xfer_ =
      lor_mesh ? std::make_unique<SubmeshLORTransfer>( *submesh_pressure_.ParFESpace(), *lor_mesh ) : nullptr;
  update_data_.reset( nullptr );
}

void MfemSubmeshData::UpdateMfemSubmeshData( redecomp::RedecompMesh& redecomp_mesh, bool new_redecomp )
{
  if ( new_redecomp || !update_data_ ) {
    update_data_ =
        std::make_unique<UpdateData>( *submesh_pressure_.ParFESpace(), submesh_lor_xfer_.get(), redecomp_mesh );
  }
  pressure_.UpdateField( update_data_->pressure_xfer_ );
  redecomp_gap_.SetSpace( pressure_.GetRedecompGridFn().FESpace() );
  redecomp_gap_.UseDevice( use_device_ );
  redecomp_gap_ = 0.0;
}

void MfemSubmeshData::GetSubmeshGap( mfem::Vector& g ) const
{
  g.SetSize( submesh_pressure_.ParFESpace()->GetVSize() );
  g = 0.0;
  GetPressureTransfer().RedecompToSubmesh( redecomp_gap_, g );
}

MfemSubmeshData::UpdateData::UpdateData( mfem::ParFiniteElementSpace& submesh_fes, SubmeshLORTransfer* submesh_lor_xfer,
                                         redecomp::RedecompMesh& redecomp_mesh )
    : pressure_xfer_{ submesh_fes, submesh_lor_xfer, redecomp_mesh }
{
}

MfemSubmeshData::UpdateData& MfemSubmeshData::GetUpdateData()
{
  SLIC_ERROR_ROOT_IF( update_data_ == nullptr, "UpdateField() must be called to generate UpdateData." );
  return *update_data_;
}

const MfemSubmeshData::UpdateData& MfemSubmeshData::GetUpdateData() const
{
  SLIC_ERROR_ROOT_IF( update_data_ == nullptr, "UpdateField() must be called to generate UpdateData." );
  return *update_data_;
}

MfemJacobianData::MfemJacobianData( const MfemMeshData& parent_data, const MfemSubmeshData& submesh_data )
    : parent_data_{ parent_data }, submesh_data_{ submesh_data }
{
  if ( parent_data.GetParentCoords().ParFESpace()->FEColl()->GetOrder() > 1 ) {
    SLIC_ERROR_ROOT_IF( parent_data.GetLORMesh() == nullptr,
                        "Higher order Jacobian support requires an LOR mesh (call MfemMeshData::SetLORFactor)." );
    SLIC_ERROR_ROOT_IF( parent_data.GetLORMeshFESpace() == nullptr || submesh_data.GetLORMeshFESpace() == nullptr,
                        "Higher order Jacobian support requires LOR finite element spaces." );
  }

  mfem::Array<int> vdof_list_int;

  mfem::SubMeshUtils::BuildVdofToVdofMap( parent_data_.GetSubmeshFESpace(), *parent_data_.GetParentCoords().FESpace(),
                                          parent_data_.GetSubmesh().GetFrom(),
                                          parent_data_.GetSubmesh().GetParentElementIDMap(), vdof_list_int );

  auto dof_offset = parent_data_.GetParentCoords().ParFESpace()->GetMyDofOffset();
  submesh2parent_vdof_list_.SetSize( vdof_list_int.Size() );
  for ( int i{ 0 }; i < vdof_list_int.Size(); ++i ) {
    submesh2parent_vdof_list_[i] = dof_offset + static_cast<HYPRE_BigInt>( vdof_list_int[i] );
  }
}

void MfemJacobianData::UpdateJacobianXfer()
{
  update_data_ = std::make_unique<UpdateData>( parent_data_, submesh_data_, submesh2parent_vdof_list_ );
}

MfemJacobianData::UpdateData::UpdateData( const MfemMeshData& parent_data, const MfemSubmeshData& submesh_data,
                                          const mfem::Array<HYPRE_BigInt>& submesh2parent_vdof_list )
{
  // Build cached pathways that map final true dofs into the selected surface DOF layouts.
  // Here, "surface" refers to the parent FE spaces of the redecomp FE spaces used during redecomp transfer.
  // In this MFEM setup, those parent (surface) FE spaces live on either:
  // - the LOR surface mesh (when LOR is active), or
  // - the boundary submesh (otherwise).
  parent_path_.final_fes = parent_data.GetParentCoords().ParFESpace();
  parent_path_.owned_ops.clear();
  parent_path_.ops.clear();
  parent_path_.ops.emplace_back( parent_path_.final_fes->Dof_TrueDof_Matrix() );

  // Always include the parent->submesh restriction (primary variables live on the boundary submesh).
  {
    SubmeshParentTransferMat submesh_parent_xfer(
        parent_data.GetSubmeshFESpace(), *parent_data.GetParentCoords().ParFESpace(), submesh2parent_vdof_list );
    auto submesh_parent_mat = submesh_parent_xfer.Assemble();
    parent_path_.owned_ops.push_back( std::make_unique<shared::ParSparseMat>( std::move( submesh_parent_mat ) ) );
    parent_path_.ops.emplace_back( &parent_path_.owned_ops.back()->get() );
  }

  if ( parent_data.GetLORMesh() ) {
    parent_path_.surface_fes = parent_data.GetLORMeshFESpace();
    auto primary_ho_scalar_fes = std::make_unique<mfem::ParFiniteElementSpace>(
        parent_data.GetSubmeshFESpace().GetParMesh(), parent_data.GetSubmeshFESpace().FEColl(), 1,
        parent_data.GetSubmeshFESpace().GetOrdering() );
    auto primary_lor_scalar_fes = std::make_unique<mfem::ParFiniteElementSpace>(
        parent_data.GetLORMeshFESpace()->GetParMesh(), parent_data.GetLORMeshFESpace()->FEColl(), 1,
        parent_data.GetLORMeshFESpace()->GetOrdering() );
    HoToLorTransferMat primary_ho_to_lor( parent_data.GetSubmeshFESpace(), *parent_data.GetLORMeshFESpace(),
                                          *primary_ho_scalar_fes, *primary_lor_scalar_fes );
    auto ho_to_lor_mat = primary_ho_to_lor.Assemble();
    parent_path_.owned_ops.push_back( std::make_unique<shared::ParSparseMat>( std::move( ho_to_lor_mat ) ) );
    parent_path_.ops.emplace_back( &parent_path_.owned_ops.back()->get() );
  } else {
    parent_path_.surface_fes = &parent_data.GetSubmeshFESpace();
  }

  submesh_path_.final_fes = &submesh_data.GetSubmeshFESpace();
  submesh_path_.owned_ops.clear();
  submesh_path_.ops.clear();
  submesh_path_.ops.emplace_back( submesh_path_.final_fes->Dof_TrueDof_Matrix() );

  if ( parent_data.GetLORMesh() ) {
    submesh_path_.surface_fes = submesh_data.GetLORMeshFESpace();
    auto lm_ho_scalar_fes = std::make_unique<mfem::ParFiniteElementSpace>(
        submesh_data.GetSubmeshFESpace().GetParMesh(), submesh_data.GetSubmeshFESpace().FEColl(), 1,
        submesh_data.GetSubmeshFESpace().GetOrdering() );
    auto lm_lor_scalar_fes = std::make_unique<mfem::ParFiniteElementSpace>(
        submesh_data.GetLORMeshFESpace()->GetParMesh(), submesh_data.GetLORMeshFESpace()->FEColl(), 1,
        submesh_data.GetLORMeshFESpace()->GetOrdering() );
    HoToLorTransferMat dual_ho_to_lor( submesh_data.GetSubmeshFESpace(), *submesh_data.GetLORMeshFESpace(),
                                       *lm_ho_scalar_fes, *lm_lor_scalar_fes );
    auto ho_to_lor_mat = dual_ho_to_lor.Assemble();
    submesh_path_.owned_ops.push_back( std::make_unique<shared::ParSparseMat>( std::move( ho_to_lor_mat ) ) );
    submesh_path_.ops.emplace_back( &submesh_path_.owned_ops.back()->get() );
  } else {
    submesh_path_.surface_fes = &submesh_data.GetSubmeshFESpace();
  }
}

MfemJacobianData::UpdateData& MfemJacobianData::GetUpdateData()
{
  SLIC_ERROR_ROOT_IF( update_data_ == nullptr, "UpdateField() must be called to generate UpdateData." );
  return *update_data_;
}

const MfemJacobianData::UpdateData& MfemJacobianData::GetUpdateData() const
{
  SLIC_ERROR_ROOT_IF( update_data_ == nullptr, "UpdateField() must be called to generate UpdateData." );
  return *update_data_;
}

const MfemJacobianPath& MfemJacobianData::ParentPath() const
{
  SLIC_ERROR_ROOT_IF( !update_data_, "ParentPath() requires UpdateJacobianXfer() to have been called." );
  return update_data_->parent_path_;
}

const MfemJacobianPath& MfemJacobianData::SubmeshPath() const
{
  SLIC_ERROR_ROOT_IF( !update_data_, "SubmeshPath() requires UpdateJacobianXfer() to have been called." );
  return update_data_->submesh_path_;
}

shared::ParSparseMat MfemJacobianData::GetMfemJacobian(
    const mfem::ParFiniteElementSpace* row_final_fes, const mfem::ParFiniteElementSpace* col_final_fes,
    const std::vector<PackedPairJacobianContribs>& contributions ) const
{
  SLIC_ERROR_ROOT_IF( row_final_fes == nullptr || col_final_fes == nullptr,
                      "GetMfemJacobian() requires non-null final FE-space pointers." );
  SLIC_ERROR_ROOT_IF( !update_data_, "GetMfemJacobian() requires UpdateJacobianXfer() to have been called." );

  const mfem::ParFiniteElementSpace* parent_fes = parent_data_.GetParentCoords().ParFESpace();
  const mfem::ParFiniteElementSpace* multiplier_submesh_fes = &submesh_data_.GetSubmeshFESpace();

  const MfemJacobianPath* row_path = nullptr;
  const MfemJacobianPath* col_path = nullptr;

  if ( row_final_fes == parent_fes ) {
    row_path = &ParentPath();
  } else if ( row_final_fes == multiplier_submesh_fes ) {
    row_path = &SubmeshPath();
  } else {
    SLIC_ERROR_ROOT( "GetMfemJacobian(): unsupported row_final_fes pointer." );
  }

  if ( col_final_fes == parent_fes ) {
    col_path = &ParentPath();
  } else if ( col_final_fes == multiplier_submesh_fes ) {
    col_path = &SubmeshPath();
  } else {
    SLIC_ERROR_ROOT( "GetMfemJacobian(): unsupported col_final_fes pointer." );
  }

  MfemJacobianTransfer transfer( *row_path, *col_path );
  return transfer.Assemble( contributions );
}

}  // namespace tribol

#endif /* BUILD_REDECOMP */
