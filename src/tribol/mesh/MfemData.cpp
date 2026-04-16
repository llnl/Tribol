// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MfemData.hpp"

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP

#include "axom/slic/interface/slic_macros.hpp"

#include "shared/infrastructure/Profiling.hpp"
#include "tribol/common/LoopExec.hpp"

#include "redecomp/utils/ArrayUtility.hpp"

#include "tribol/common/LoopExec.hpp"

namespace tribol {

namespace {

class L2ProjectionH1SpaceHack : public mfem::L2ProjectionGridTransfer::L2ProjectionH1Space {
 public:
  L2ProjectionH1SpaceHack( const mfem::ParFiniteElementSpace& pfes_ho, const mfem::ParFiniteElementSpace& pfes_lor,
                           bool use_ea )
      : mfem::L2ProjectionGridTransfer::L2ProjectionH1Space( pfes_ho, pfes_lor, use_ea )
  {
  }

  const mfem::Operator* GetTrueRestriction() const { return R.get(); }
};

bool GetEnvBool( const char* name )
{
  const char* v = std::getenv( name );
  if ( v == nullptr ) return false;
  // Treat explicit "0" as false; anything else means enabled.
  return !( v[0] == '0' && v[1] == '\0' );
}

void TDofsListByVDim( const mfem::ParFiniteElementSpace& fes, int vdim, mfem::Array<int>& tdofs_list )
{
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

void BuildHo2Lor( mfem::Table& ho2lor, int nel_ho, int nel_lor, const mfem::CoarseFineTransformations& cf_tr )
{
  ho2lor.MakeI( nel_ho );
  for ( int ilor = 0; ilor < nel_lor; ++ilor ) {
    const int iho = cf_tr.embeddings[ilor].parent;
    ho2lor.AddAColumnInRow( iho );
  }
  ho2lor.MakeJ();
  for ( int ilor = 0; ilor < nel_lor; ++ilor ) {
    const int iho = cf_tr.embeddings[ilor].parent;
    ho2lor.AddConnection( iho, ilor );
  }
  ho2lor.ShiftUpI();
}

void ElemMixedMass( mfem::Geometry::Type geom, const mfem::FiniteElement& fe_ho, const mfem::FiniteElement& fe_lor,
                    mfem::ElementTransformation* tr_ho, mfem::ElementTransformation* tr_lor,
                    mfem::IntegrationPointTransformation& ip_tr, mfem::DenseMatrix& M_mixed_el )
{
  const int order = fe_lor.GetOrder() + fe_ho.GetOrder() + tr_lor->OrderW();
  const mfem::IntegrationRule* ir = &mfem::IntRules.Get( geom, order );
  M_mixed_el = 0.0;
  for ( int i = 0; i < ir->GetNPoints(); ++i ) {
    const mfem::IntegrationPoint& ip_lor = ir->IntPoint( i );
    mfem::IntegrationPoint ip_ho;
    ip_tr.Transform( ip_lor, ip_ho );

    mfem::Vector shape_lor( fe_lor.GetDof() );
    fe_lor.CalcShape( ip_lor, shape_lor );

    mfem::Vector shape_ho( fe_ho.GetDof() );
    tr_ho->SetIntPoint( &ip_ho );
    fe_ho.CalcPhysShape( *tr_ho, shape_ho );

    tr_lor->SetIntPoint( &ip_lor );
    double w = ip_lor.weight;
    if ( fe_lor.GetMapType() == mfem::FiniteElement::VALUE ) {
      w *= tr_lor->Weight();
    }
    shape_lor *= w;
    mfem::AddMultVWt( shape_lor, shape_ho, M_mixed_el );
  }
}

void LumpedMassInverse( const mfem::ParFiniteElementSpace& fes_lor_scalar, mfem::Vector& ML_inv_ldof )
{
  mfem::Vector ML_inv_true( fes_lor_scalar.GetTrueVSize() );
  const mfem::Operator* P = fes_lor_scalar.GetProlongationMatrix();
  if ( P ) {
    P->MultTranspose( ML_inv_ldof, ML_inv_true );
  } else {
    ML_inv_true = ML_inv_ldof;
  }

  ML_inv_true.Reciprocal();

  if ( P ) {
    P->Mult( ML_inv_true, ML_inv_ldof );
  } else {
    ML_inv_ldof = ML_inv_true;
  }
}

shared::ParSparseMat BuildH1TrueRestriction( const mfem::ParFiniteElementSpace& ho_scalar,
                                             const mfem::ParFiniteElementSpace& lor_scalar )
{
  SLIC_ERROR_ROOT_IF( ho_scalar.FEColl()->GetContType() != mfem::FiniteElementCollection::CONTINUOUS ||
                          lor_scalar.FEColl()->GetContType() != mfem::FiniteElementCollection::CONTINUOUS,
                      "LOR transfer only supports continuous (H1) spaces." );

  auto* mesh_ho = ho_scalar.GetParMesh();
  auto* mesh_lor = lor_scalar.GetParMesh();
  SLIC_ERROR_ROOT_IF( mesh_ho == nullptr || mesh_lor == nullptr, "Null ParMesh in LOR transfer." );

  const int nel_ho = mesh_ho->GetNE();
  const int nel_lor = mesh_lor->GetNE();
  const int ndof_ho = ho_scalar.GetNDofs();
  const int ndof_lor = lor_scalar.GetNDofs();

  mfem::SparseMatrix R_dof( ndof_lor, ndof_ho );
  mfem::Vector ML_inv;
  mfem::Table ho2lor;
  mfem::IntegrationPointTransformation ip_tr;
  mfem::IsoparametricTransformation& emb_tr = ip_tr.Transf;

  // Even if a rank has no local elements/DOFs (e.g. small meshes on many MPI
  // ranks), all ranks must still participate in the subsequent parallel
  // multiplies. In that case we simply build an empty local operator here.
  if ( nel_ho > 0 && nel_lor > 0 && ndof_ho > 0 && ndof_lor > 0 ) {
    const mfem::CoarseFineTransformations& cf_tr = mesh_lor->GetRefinementTransforms();
    BuildHo2Lor( ho2lor, nel_ho, nel_lor, cf_tr );

    ML_inv.SetSize( ndof_lor );
    ML_inv = 0.0;

    // Assemble lumped LOR mass row-sums on ldofs
    for ( int ilor = 0; ilor < nel_lor; ++ilor ) {
      const mfem::Geometry::Type geom = mesh_lor->GetElementBaseGeometry( ilor );
      const mfem::FiniteElement& fe_lor = *lor_scalar.GetFE( ilor );
      mfem::ElementTransformation* el_tr = lor_scalar.GetElementTransformation( ilor );

      const int order = 2 * fe_lor.GetOrder() + el_tr->OrderW();
      const mfem::IntegrationRule* ir = &mfem::IntRules.Get( geom, order );

      const int nedof_lor = fe_lor.GetDof();
      mfem::Vector ML_el( nedof_lor );
      mfem::Vector shape_lor( nedof_lor );
      ML_el = 0.0;

      for ( int i = 0; i < ir->GetNPoints(); ++i ) {
        const mfem::IntegrationPoint& ip_lor = ir->IntPoint( i );
        fe_lor.CalcShape( ip_lor, shape_lor );
        el_tr->SetIntPoint( &ip_lor );
        shape_lor *= ( el_tr->Weight() * ip_lor.weight );
        ML_el += shape_lor;
      }

      mfem::Array<int> dofs_lor( nedof_lor );
      lor_scalar.GetElementDofs( ilor, dofs_lor );
      ML_inv.AddElementVector( dofs_lor, ML_el );
    }

    LumpedMassInverse( lor_scalar, ML_inv );

    for ( int iho = 0; iho < nel_ho; ++iho ) {
      mfem::Array<int> lor_els;
      ho2lor.GetRow( iho, lor_els );
      if ( lor_els.Size() == 0 ) continue;

      const mfem::Geometry::Type geom = mesh_ho->GetElementBaseGeometry( iho );
      const mfem::FiniteElement& fe_ho = *ho_scalar.GetFE( iho );
      const mfem::FiniteElement& fe_lor = *lor_scalar.GetFE( lor_els[0] );
      mfem::ElementTransformation* tr_ho = ho_scalar.GetElementTransformation( iho );

      emb_tr.SetIdentityTransformation( geom );
      const mfem::DenseTensor& pmats = cf_tr.point_matrices[geom];

      const int nedof_ho = fe_ho.GetDof();
      const int nedof_lor = fe_lor.GetDof();
      mfem::DenseMatrix M_LH_el( nedof_lor, nedof_ho );
      mfem::DenseMatrix R_el( nedof_lor, nedof_ho );

      mfem::Array<int> dofs_ho( nedof_ho );
      ho_scalar.GetElementDofs( iho, dofs_ho );

      for ( int iref = 0; iref < lor_els.Size(); ++iref ) {
        const int ilor = lor_els[iref];
        mfem::ElementTransformation* tr_lor = lor_scalar.GetElementTransformation( ilor );

        emb_tr.SetPointMat( pmats( cf_tr.embeddings[ilor].matrix ) );

        ElemMixedMass( geom, fe_ho, fe_lor, tr_ho, tr_lor, ip_tr, M_LH_el );

        mfem::Array<int> dofs_lor( nedof_lor );
        lor_scalar.GetElementDofs( ilor, dofs_lor );

        R_el = M_LH_el;
        mfem::Vector row;
        for ( int r = 0; r < nedof_lor; ++r ) {
          R_el.GetRow( r, row );
          row *= ML_inv[dofs_lor[r]];
          R_el.SetRow( r, row );
        }

        R_dof.AddSubMatrix( dofs_lor, dofs_ho, R_el );
      }
    }
  }

  R_dof.Finalize();

  auto comm = mesh_ho->GetComm();
  shared::ParSparseMat R_local( comm, lor_scalar.GlobalVSize(), ho_scalar.GlobalVSize(), lor_scalar.GetDofOffsets(),
                                ho_scalar.GetDofOffsets(), std::move( R_dof ) );

  return shared::ParSparseMat::rap( lor_scalar.Dof_TrueDof_Matrix(), R_local, ho_scalar.Dof_TrueDof_Matrix() );
}

[[maybe_unused]] shared::ParSparseMat ExpandScalarDofTransferByNodes( const shared::ParSparseMat& T_scalar_dof,
                                                                      const mfem::ParFiniteElementSpace& ho_fes,
                                                                      const mfem::ParFiniteElementSpace& lor_fes,
                                                                      HYPRE_BigInt ho_scalar_dof_offset )
{
  (void)ho_scalar_dof_offset;
  SLIC_ERROR_ROOT_IF( ho_fes.GetVDim() != lor_fes.GetVDim(), "HO/LOR vdim mismatch in LOR transfer." );
  const int vdim = ho_fes.GetVDim();
  SLIC_ERROR_ROOT_IF(
      ho_fes.GetOrdering() != mfem::Ordering::byNODES || lor_fes.GetOrdering() != mfem::Ordering::byNODES,
      "LOR transfer expansion only supports mfem::Ordering::byNODES." );

  auto& A = T_scalar_dof.get();
  const int nrows_scalar = A.Height();
  SLIC_ERROR_ROOT_IF( nrows_scalar * vdim != lor_fes.GetVSize(), "Unexpected LOR scalar row size in transfer." );

  mfem::SparseMatrix diag;
  A.GetDiag( diag );

  mfem::SparseMatrix offd;
  HYPRE_BigInt* col_map_offd = nullptr;
  A.GetOffd( offd, col_map_offd );

  const int* diagI = diag.GetI();
  const int* diagJ = diag.GetJ();
  const double* diagData = diag.GetData();
  SLIC_ERROR_ROOT_IF( diagI == nullptr || diagJ == nullptr || diagData == nullptr, "Invalid diag block in transfer." );

  const int* offdI = offd.GetI();
  const int* offdJ = offd.GetJ();
  const double* offdData = offd.GetData();
  const bool has_offd = ( offdI != nullptr && offdJ != nullptr && offdData != nullptr );

  const int nrows_vec = nrows_scalar * vdim;
  const int ncols_diag_scalar = diag.Width();
  const int ncols_offd_scalar = has_offd ? offd.Width() : 0;

  SLIC_ERROR_ROOT_IF( ncols_diag_scalar * vdim != ho_fes.GetVSize(),
                      "Unexpected HO scalar column size in transfer expansion." );

  // Build expanded diag block (local columns) and expanded offd block (off-rank columns).
  std::vector<int> I_diag_vec( static_cast<size_t>( nrows_vec + 1 ), 0 );
  std::vector<int> I_offd_vec( static_cast<size_t>( nrows_vec + 1 ), 0 );

  for ( int i = 0; i < nrows_scalar; ++i ) {
    const int diag_row_nnz = diagI[i + 1] - diagI[i];
    const int offd_row_nnz = has_offd ? ( offdI[i + 1] - offdI[i] ) : 0;
    for ( int d = 0; d < vdim; ++d ) {
      const int vr = i * vdim + d;
      I_diag_vec[static_cast<size_t>( vr + 1 )] = I_diag_vec[static_cast<size_t>( vr )] + diag_row_nnz;
      I_offd_vec[static_cast<size_t>( vr + 1 )] = I_offd_vec[static_cast<size_t>( vr )] + offd_row_nnz;
    }
  }

  const int nnz_diag_vec = I_diag_vec.back();
  const int nnz_offd_vec = I_offd_vec.back();

  auto* J_diag_vec = new int[nnz_diag_vec];
  auto* data_diag_vec = new double[nnz_diag_vec];

  int* J_offd_vec = nullptr;
  double* data_offd_vec = nullptr;
  HYPRE_BigInt* col_map_offd_vec = nullptr;

  if ( nnz_offd_vec > 0 ) {
    J_offd_vec = new int[nnz_offd_vec];
    data_offd_vec = new double[nnz_offd_vec];
    SLIC_ERROR_ROOT_IF( col_map_offd == nullptr, "Null col_map_offd in scalar transfer matrix." );
    col_map_offd_vec = new HYPRE_BigInt[static_cast<size_t>( ncols_offd_scalar * vdim )];
    for ( int j = 0; j < ncols_offd_scalar; ++j ) {
      const HYPRE_BigInt g_scalar = col_map_offd[j];
      for ( int d = 0; d < vdim; ++d ) {
        col_map_offd_vec[static_cast<size_t>( j * vdim + d )] = g_scalar * static_cast<HYPRE_BigInt>( vdim ) + d;
      }
    }
  }

  for ( int i = 0; i < nrows_scalar; ++i ) {
    const int diag_begin = diagI[i];
    const int diag_end = diagI[i + 1];
    const int offd_begin = has_offd ? offdI[i] : 0;
    const int offd_end = has_offd ? offdI[i + 1] : 0;

    for ( int d = 0; d < vdim; ++d ) {
      const int vr = i * vdim + d;

      int out_diag = I_diag_vec[static_cast<size_t>( vr )];
      for ( int k = diag_begin; k < diag_end; ++k, ++out_diag ) {
        J_diag_vec[out_diag] = diagJ[k] * vdim + d;
        data_diag_vec[out_diag] = diagData[k];
      }

      if ( nnz_offd_vec > 0 ) {
        int out_offd = I_offd_vec[static_cast<size_t>( vr )];
        for ( int k = offd_begin; k < offd_end; ++k, ++out_offd ) {
          J_offd_vec[out_offd] = offdJ[k] * vdim + d;
          data_offd_vec[out_offd] = offdData[k];
        }
      }
    }
  }

  auto* I_diag_raw = new int[nrows_vec + 1];
  for ( int i = 0; i <= nrows_vec; ++i ) I_diag_raw[i] = I_diag_vec[static_cast<size_t>( i )];
  auto* I_offd_raw = new int[nrows_vec + 1];
  for ( int i = 0; i <= nrows_vec; ++i ) I_offd_raw[i] = I_offd_vec[static_cast<size_t>( i )];

  mfem::SparseMatrix diag_vec( I_diag_raw, J_diag_vec, data_diag_vec, nrows_vec, ncols_diag_scalar * vdim, true, true,
                               true );
  mfem::SparseMatrix offd_vec;
  if ( nnz_offd_vec > 0 ) {
    offd_vec = mfem::SparseMatrix( I_offd_raw, J_offd_vec, data_offd_vec, nrows_vec, ncols_offd_scalar * vdim, true,
                                   true, true );
  } else {
    delete[] I_offd_raw;
    offd_vec = mfem::SparseMatrix( nrows_vec, 0 );
    offd_vec.Finalize();
  }

  auto* hypre =
      new mfem::HypreParMatrix( lor_fes.GetComm(), lor_fes.GlobalVSize(), ho_fes.GlobalVSize(), lor_fes.GetDofOffsets(),
                                ho_fes.GetDofOffsets(), &diag_vec, &offd_vec, col_map_offd_vec );
  // HypreParMatrix steals the CSR arrays from diag_vec/offd_vec; prevent the stack
  // SparseMatrix objects from freeing them on scope exit.
  diag_vec.GetMemoryI().ClearOwnerFlags();
  diag_vec.GetMemoryJ().ClearOwnerFlags();
  diag_vec.GetMemoryData().ClearOwnerFlags();
  offd_vec.GetMemoryI().ClearOwnerFlags();
  offd_vec.GetMemoryJ().ClearOwnerFlags();
  offd_vec.GetMemoryData().ClearOwnerFlags();

  constexpr int mfem_owned_host_flag = 3;
  hypre->SetOwnerFlags( mfem_owned_host_flag, mfem_owned_host_flag, mfem_owned_host_flag );

  return shared::ParSparseMat( std::unique_ptr<mfem::HypreParMatrix>( hypre ) );
}

}  // namespace

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
    forAllExec( exec_mode_, current_coords_gf.Size(), [d_curr, d_last, d_max_diff] TRIBOL_HOST_DEVICE( int i ) {
#ifdef TRIBOL_USE_RAJA
      RAJA::atomicMax<RAJA::auto_atomic>( d_max_diff, std::abs( d_curr[i] - d_last[i] ) );
#else
      d_max_diff[0] = std::max( d_max_diff[0], std::abs( d_curr[i] - d_last[i] ) );
#endif
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
    auto& parent_fes = *coords_.GetParentGridFn().ParFESpace();
    mfem::Array<int> be_dofs;
    parent_fes.GetBdrElementDofs( parent_bdr_e, be_dofs );
    mfem::DenseMatrix elem_coords( parent_mesh_.Dimension(), be_dofs.Size() );
    for ( int d{ 0 }; d < parent_mesh_.Dimension(); ++d ) {
      mfem::Array<int> be_vdofs( be_dofs );
      parent_fes.DofsToVDofs( d, be_vdofs );
      mfem::Vector elemvect( be_dofs.Size() );
      coords_.GetParentGridFn().GetSubVector( be_vdofs, elemvect );
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
    double h = parent_mesh.GetElementSize( parent_e, norm );

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

MfemJacobianData::MfemJacobianData( const MfemMeshData& parent_data, const MfemSubmeshData& submesh_data,
                                    ContactMethod contact_method )
    : parent_data_{ parent_data }, submesh_data_{ submesh_data }, block_offsets_( 3 ), disp_offsets_( 2 )
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

  auto& parent_fes = *parent_data_.GetParentCoords().ParFESpace();
  auto& submesh_fes = parent_data_.GetSubmeshFESpace();
  auto submesh_parent_I = redecomp::ArrayUtility::IndexArray<int>( submesh2parent_vdof_list_.Size() + 1 );
  mfem::Vector submesh_parent_data( submesh2parent_vdof_list_.Size() );
  submesh_parent_data = 1.0;
  // This constructor copies all of the data, so don't worry about ownership of the CSR data
  submesh_parent_vdof_xfer_ = std::make_unique<shared::ParSparseMat>(
      parent_fes.GetComm(), submesh_fes.GetVSize(), submesh_fes.GlobalVSize(), parent_fes.GlobalVSize(),
      submesh_parent_I.data(), submesh2parent_vdof_list_.GetData(), submesh_parent_data.GetData(),
      submesh_fes.GetDofOffsets(), parent_fes.GetDofOffsets() );

  auto disp_size = parent_data_.GetParentCoords().ParFESpace()->GetTrueVSize();
  auto lm_size = submesh_data_.GetSubmeshPressure().ParFESpace()->GetTrueVSize();
  // this is used to size Jacobian contributions that are dependent on the pressure
  block_offsets_[0] = 0;
  block_offsets_[1] = disp_size;
  block_offsets_[2] = disp_size + lm_size;
  // this is used to size Jacobian contributions that are not dependent on the pressure (e.g. normal)
  disp_offsets_[0] = 0;
  disp_offsets_[1] = disp_size;

  // Rows/columns of pressure/gap DOFs only on the mortar surface need to be eliminated from the Jacobian when using
  // single mortar. The code in this block creates a list of the true DOFs only on the mortar surface.
  if ( contact_method == SINGLE_MORTAR ) {
    // Get submesh
    auto& submesh_fe_space = submesh_data_.GetSubmeshFESpace();
    auto& submesh = parent_data_.GetSubmesh();
    // Create marker of attributes for faster querying
    mfem::Array<int> attr_marker( submesh.attributes.Max() );
    attr_marker = 0;
    for ( auto nonmortar_attr : parent_data_.GetBoundaryAttribs2() ) {
      attr_marker[nonmortar_attr - 1] = 1;
    }
    // Create marker of dofs only on mortar surface
    mfem::Array<int> mortar_dof_marker( submesh_fe_space.GetVSize() );
    mortar_dof_marker = 1;
    for ( int e{ 0 }; e < submesh.GetNE(); ++e ) {
      if ( attr_marker[submesh_fe_space.GetAttribute( e ) - 1] ) {
        mfem::Array<int> vdofs;
        submesh_fe_space.GetElementVDofs( e, vdofs );
        for ( int d{ 0 }; d < vdofs.Size(); ++d ) {
          int k = vdofs[d];
          if ( k < 0 ) {
            k = -1 - k;
          }
          mortar_dof_marker[k] = 0;
        }
      }
    }
    // Convert marker of dofs to marker of tdofs
    mfem::Array<int> mortar_tdof_marker( submesh_fe_space.GetTrueVSize() );
    submesh_fe_space.GetRestrictionMatrix()->BooleanMult( mortar_dof_marker, mortar_tdof_marker );
    // Convert markers of tdofs only on mortar surface to a list
    mfem::FiniteElementSpace::MarkerToList( mortar_tdof_marker, mortar_tdof_list_ );
  }
}

void MfemJacobianData::UpdateJacobianXfer()
{
  update_data_ = std::make_unique<UpdateData>( parent_data_, submesh_data_ );
}

std::unique_ptr<mfem::BlockOperator> MfemJacobianData::GetMfemBlockJacobian(
    const MethodData& method_data, const std::vector<std::pair<int, BlockSpace>>& row_info,
    const std::vector<std::pair<int, BlockSpace>>& col_info ) const
{
  // Determine block structure
  int max_row_block = 0;
  for ( auto info : row_info ) {
    if ( info.first > max_row_block ) max_row_block = info.first;
  }
  int max_col_block = 0;
  for ( auto info : col_info ) {
    if ( info.first > max_col_block ) max_col_block = info.first;
  }

  SLIC_ERROR_ROOT_IF( max_row_block > GetUpdateData().submesh_redecomp_xfer_.shape()[0] ||
                          max_col_block > GetUpdateData().submesh_redecomp_xfer_.shape()[1],
                      axom::fmt::format( "No transfer object for row {0} and col {1}", max_row_block, max_col_block ) );

  const mfem::Array<int>& row_offsets = ( max_row_block == 0 ) ? disp_offsets_ : block_offsets_;
  const mfem::Array<int>& col_offsets = ( max_col_block == 0 ) ? disp_offsets_ : block_offsets_;

  auto block_J = std::make_unique<mfem::BlockOperator>( row_offsets, col_offsets );
  block_J->owns_blocks = 1;

  // Map unique (r_blk, c_blk) -> list of (row_space, col_space) pairs
  std::map<std::pair<int, int>, std::vector<std::pair<BlockSpace, BlockSpace>>> block_contribs;

  for ( const auto& r_pair : row_info ) {
    for ( const auto& c_pair : col_info ) {
      block_contribs[{ r_pair.first, c_pair.first }].push_back( { r_pair.second, c_pair.second } );
    }
  }

  // Maps BlockSpaces (MORTAR, NONMORTAR, LAGRANGE_MULTIPLIER) to a tribol element map
  const std::vector<const Array1D<int>*> elem_map_by_space{ &parent_data_.GetElemMap1(), &parent_data_.GetElemMap2(),
                                                            &parent_data_.GetElemMap2() };

  // Iterate over unique blocks
  for ( const auto& entry : block_contribs ) {
    int r_blk = entry.first.first;
    int c_blk = entry.first.second;
    const auto& contribs = entry.second;

    std::unique_ptr<mfem::SparseMatrix> submesh_J;

    for ( const auto& pair : contribs ) {
      BlockSpace rs = pair.first;
      BlockSpace cs = pair.second;

      // Get block from method_data
      const auto& J_block = method_data.getBlockJ()( static_cast<int>( rs ), static_cast<int>( cs ) );

      // Map element IDs to redecomp IDs
      const auto& row_elem_ids_tribol = method_data.getBlockJElementIds()[static_cast<int>( rs )];
      ArrayT<int> row_redecomp_ids;
      row_redecomp_ids.reserve( row_elem_ids_tribol.size() );
      for ( auto id : row_elem_ids_tribol ) {
        row_redecomp_ids.push_back( ( *elem_map_by_space[static_cast<size_t>( rs )] )[static_cast<size_t>( id )] );
      }

      const auto& col_elem_ids_tribol = method_data.getBlockJElementIds()[static_cast<int>( cs )];
      ArrayT<int> col_redecomp_ids;
      col_redecomp_ids.reserve( col_elem_ids_tribol.size() );
      for ( auto id : col_elem_ids_tribol ) {
        col_redecomp_ids.push_back( ( *elem_map_by_space[static_cast<size_t>( cs )] )[static_cast<size_t>( id )] );
      }

      // Pick transfer object
      redecomp::MatrixTransfer* xfer = GetUpdateData().submesh_redecomp_xfer_( r_blk, c_blk ).get();

      // No transfer object for LAGRANGE_MULTIPLER, LAGRANGE_MULTIPLIER block
      if ( xfer != nullptr ) {
        auto J_contrib = xfer->TransferToParallelSparse( row_redecomp_ids, col_redecomp_ids, J_block );
        if ( !submesh_J ) {
          submesh_J = std::make_unique<mfem::SparseMatrix>( std::move( J_contrib ) );
        } else {
          ( *submesh_J ) += J_contrib;
        }
      }
    }

    if ( submesh_J ) {
      submesh_J->Finalize();

      // Pick xfer again for conversion
      redecomp::MatrixTransfer* xfer = GetUpdateData().submesh_redecomp_xfer_( r_blk, c_blk ).get();

      auto submesh_J_hypre = xfer->ConvertToParSparseMat( std::move( *submesh_J ), false );
      std::unique_ptr<shared::ParSparseMat> mapped_submesh_J;
      shared::ParSparseMatView submesh_J_view( &submesh_J_hypre.get() );
      if ( parent_data_.GetLORMesh() ) {
        SLIC_ERROR_ROOT_IF( GetUpdateData().T_disp_ho_to_lor_ == nullptr || GetUpdateData().T_lm_ho_to_lor_ == nullptr,
                            "LOR Jacobian mapping requires transfer matrices. Call UpdateJacobianXfer()." );
        const auto& T_row_mat = ( r_blk == 0 ) ? *GetUpdateData().T_disp_ho_to_lor_ : *GetUpdateData().T_lm_ho_to_lor_;
        const auto& T_col_mat = ( c_blk == 0 ) ? *GetUpdateData().T_disp_ho_to_lor_ : *GetUpdateData().T_lm_ho_to_lor_;
        shared::ParSparseMatView T_row( const_cast<mfem::HypreParMatrix*>( &T_row_mat.get() ) );
        shared::ParSparseMatView T_col( const_cast<mfem::HypreParMatrix*>( &T_col_mat.get() ) );
        mapped_submesh_J =
            std::make_unique<shared::ParSparseMat>( shared::ParSparseMat::rap( T_row, submesh_J_view, T_col ) );
        submesh_J_view = shared::ParSparseMatView( &mapped_submesh_J->get() );
      }

      mfem::HypreParMatrix* block_mat = nullptr;

      if ( r_blk == 0 && c_blk == 0 ) {
        auto parent_J = submesh_J_view.rap( *submesh_parent_vdof_xfer_ );
        shared::ParSparseMatView parent_P( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() );
        block_mat = parent_J.rap( parent_P ).release();
      } else if ( r_blk == 0 && c_blk == 1 ) {
        auto parent_J = submesh_parent_vdof_xfer_->transpose() * submesh_J_view;
        block_mat = shared::ParSparseMat::rap( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix(),
                                               parent_J, submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix() )
                        .release();
      } else if ( r_blk == 1 && c_blk == 0 ) {
        auto parent_J = submesh_J_view * ( *submesh_parent_vdof_xfer_ );
        shared::ParSparseMatView submesh_P( submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix() );
        shared::ParSparseMatView parent_P( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() );
        block_mat = shared::ParSparseMat::rap( submesh_P, parent_J, parent_P ).release();
      }

      block_J->SetBlock( r_blk, c_blk, block_mat );
    }
  }

  // Handle Inactive DOFs for (1, 1)
  bool has_11 = false;
  for ( auto rb : row_info )
    if ( rb.first == 1 ) has_11 = true;
  bool col_has_1 = false;
  for ( auto cb : col_info )
    if ( cb.first == 1 ) col_has_1 = true;
  has_11 = has_11 && col_has_1;

  if ( has_11 ) {
    auto& submesh_fes_full = submesh_data_.GetSubmeshFESpace();
    auto comm = parent_data_.GetParentCoords().ParFESpace()->GetComm();
    shared::ParSparseMat inactive_hpm_full = shared::ParSparseMat::diagonalMatrix(
        comm, submesh_fes_full.GlobalTrueVSize(), submesh_fes_full.GetTrueDofOffsets(), 1.0, mortar_tdof_list_, false );

    if ( block_J->IsZeroBlock( 1, 1 ) ) {
      block_J->SetBlock( 1, 1, inactive_hpm_full.release() );
    }
  }

  return block_J;
}

MfemJacobianData::UpdateData::UpdateData( const MfemMeshData& parent_data, const MfemSubmeshData& submesh_data )
    : submesh_redecomp_xfer_( 2, 2 )
{
  auto dual_submesh_fes = &submesh_data.GetSubmeshFESpace();
  auto primal_submesh_fes = &parent_data.GetSubmeshFESpace();
  if ( parent_data.GetLORMesh() ) {
    dual_submesh_fes = submesh_data.GetLORMeshFESpace();
    primal_submesh_fes = parent_data.GetLORMeshFESpace();
  }
  // create a matrix transfer operator for moving data from redecomp to the submesh
  submesh_redecomp_xfer_( 0, 0 ) = std::make_unique<redecomp::MatrixTransfer>(
      *primal_submesh_fes, *primal_submesh_fes, *parent_data.GetRedecompResponse().FESpace(),
      *parent_data.GetRedecompResponse().FESpace() );
  submesh_redecomp_xfer_( 0, 1 ) = std::make_unique<redecomp::MatrixTransfer>(
      *primal_submesh_fes, *dual_submesh_fes, *parent_data.GetRedecompResponse().FESpace(),
      *submesh_data.GetRedecompGap().FESpace() );
  submesh_redecomp_xfer_( 1, 0 ) = std::make_unique<redecomp::MatrixTransfer>(
      *dual_submesh_fes, *primal_submesh_fes, *submesh_data.GetRedecompGap().FESpace(),
      *parent_data.GetRedecompResponse().FESpace() );

  if ( parent_data.GetLORMesh() ) {
    disp_ho_scalar_fes_ = std::make_unique<mfem::ParFiniteElementSpace>(
        parent_data.GetSubmeshFESpace().GetParMesh(), parent_data.GetSubmeshFESpace().FEColl(), 1,
        parent_data.GetSubmeshFESpace().GetOrdering() );
    disp_lor_scalar_fes_ = std::make_unique<mfem::ParFiniteElementSpace>(
        parent_data.GetLORMeshFESpace()->GetParMesh(), parent_data.GetLORMeshFESpace()->FEColl(), 1,
        parent_data.GetLORMeshFESpace()->GetOrdering() );
    lm_ho_scalar_fes_ = std::make_unique<mfem::ParFiniteElementSpace>( submesh_data.GetSubmeshFESpace().GetParMesh(),
                                                                       submesh_data.GetSubmeshFESpace().FEColl(), 1,
                                                                       submesh_data.GetSubmeshFESpace().GetOrdering() );
    lm_lor_scalar_fes_ = std::make_unique<mfem::ParFiniteElementSpace>(
        submesh_data.GetLORMeshFESpace()->GetParMesh(), submesh_data.GetLORMeshFESpace()->FEColl(), 1,
        submesh_data.GetLORMeshFESpace()->GetOrdering() );

    T_disp_ho_to_lor_ = std::make_unique<shared::ParSparseMat>(
        MfemJacobianData::BuildLORTransferMatrix( parent_data.GetSubmeshFESpace(), *parent_data.GetLORMeshFESpace(),
                                                  *disp_ho_scalar_fes_, *disp_lor_scalar_fes_ ) );
    T_lm_ho_to_lor_ = std::make_unique<shared::ParSparseMat>(
        MfemJacobianData::BuildLORTransferMatrix( submesh_data.GetSubmeshFESpace(), *submesh_data.GetLORMeshFESpace(),
                                                  *lm_ho_scalar_fes_, *lm_lor_scalar_fes_ ) );
  }
}

shared::ParSparseMat MfemJacobianData::BuildLORTransferMatrix( const mfem::ParFiniteElementSpace& ho_fes,
                                                               const mfem::ParFiniteElementSpace& lor_fes,
                                                               const mfem::ParFiniteElementSpace& ho_scalar_fes,
                                                               const mfem::ParFiniteElementSpace& lor_scalar_fes )
{
  SLIC_ERROR_ROOT_IF(
      ho_fes.GetOrdering() != mfem::Ordering::byNODES || lor_fes.GetOrdering() != mfem::Ordering::byNODES,
      "LOR transfer matrix build only supports mfem::Ordering::byNODES." );
  SLIC_ERROR_ROOT_IF( ho_fes.GetVDim() != lor_fes.GetVDim(), "HO/LOR vdim mismatch in LOR transfer." );
  SLIC_ERROR_ROOT_IF( ho_scalar_fes.GetVDim() != 1 || lor_scalar_fes.GetVDim() != 1,
                      "Scalar FE spaces must be vdim=1." );

  shared::ParSparseMatView P_lor( const_cast<mfem::HypreParMatrix*>( lor_scalar_fes.Dof_TrueDof_Matrix() ) );
  std::unique_ptr<L2ProjectionH1SpaceHack> mfem_h1;
  std::unique_ptr<shared::ParSparseMat> R_true_owned;

  const bool force_fallback = GetEnvBool( "TRIBOL_MFEM_FORCE_LOR_FALLBACK" );

  const mfem::HypreParMatrix* R_hypre = nullptr;
  if ( !force_fallback ) {
    mfem_h1 = std::make_unique<L2ProjectionH1SpaceHack>( ho_scalar_fes, lor_scalar_fes, false );
    const mfem::Operator* R_op = mfem_h1->GetTrueRestriction();
    R_hypre = dynamic_cast<const mfem::HypreParMatrix*>( R_op );
  }

  if ( R_hypre == nullptr ) {
    mfem_h1.reset();
    R_true_owned = std::make_unique<shared::ParSparseMat>( BuildH1TrueRestriction( ho_scalar_fes, lor_scalar_fes ) );
    R_hypre = &R_true_owned->get();
  }

  shared::ParSparseMatView R_true_view( const_cast<mfem::HypreParMatrix*>( R_hypre ) );

  const mfem::HypreParMatrix* P_ho_scalar = ho_scalar_fes.Dof_TrueDof_Matrix();
  SLIC_ERROR_ROOT_IF( P_ho_scalar == nullptr, "Null HO scalar Dof_TrueDof_Matrix() in LOR transfer build." );
  std::unique_ptr<mfem::HypreParMatrix> P_ho_scalar_T( P_ho_scalar->Transpose() );
  SLIC_ERROR_ROOT_IF( P_ho_scalar_T == nullptr, "Failed to transpose HO scalar Dof_TrueDof_Matrix()." );

  if ( ho_fes.GetVDim() == 1 ) {
    shared::ParSparseMatView P_ho_T_view( P_ho_scalar_T.get() );
    auto tmp = R_true_view * P_ho_T_view;
    shared::ParSparseMatView tmp_view( &tmp.get() );
    return P_lor * tmp_view;
  }

  // Build the vector-valued operator by mimicking MFEM's Mult():
  // y = P_lor_vec * (sum_d S_lor(d) * R_true_scalar * E_ho(d)) * R_ho_vec * x
  //
  // where E_ho(d) extracts component-d true dofs into scalar true dofs, and
  // S_lor(d) injects scalar true dofs into component-d true dofs.
  const int vdim = ho_fes.GetVDim();

  // P_lor_vec: dof <- true
  shared::ParSparseMatView P_lor_vec( const_cast<mfem::HypreParMatrix*>( lor_fes.Dof_TrueDof_Matrix() ) );

  // R_ho_vec: true <- dof (use P^T, consistent with MFEM restriction on dof vectors)
  const mfem::HypreParMatrix* P_ho_vec = ho_fes.Dof_TrueDof_Matrix();
  SLIC_ERROR_ROOT_IF( P_ho_vec == nullptr, "Null HO vector Dof_TrueDof_Matrix() in LOR transfer build." );
  std::unique_ptr<mfem::HypreParMatrix> P_ho_vec_T( P_ho_vec->Transpose() );
  SLIC_ERROR_ROOT_IF( P_ho_vec_T == nullptr, "Failed to transpose HO vector Dof_TrueDof_Matrix()." );
  shared::ParSparseMatView R_ho_vec( P_ho_vec_T.get() );

  std::unique_ptr<shared::ParSparseMat> R_vec_true;
  for ( int d = 0; d < vdim; ++d ) {
    auto E_ho_d = BuildTrueDofExtract( ho_fes, ho_scalar_fes, d );
    auto S_lor_d = BuildTrueDofInject( lor_fes, lor_scalar_fes, d );

    shared::ParSparseMatView E_view( &E_ho_d.get() );
    shared::ParSparseMatView S_view( &S_lor_d.get() );

    auto tmp = R_true_view * E_view;  // (lor_true_scalar x ho_true_vec)
    shared::ParSparseMatView tmp_view( &tmp.get() );
    auto block = S_view * tmp_view;  // (lor_true_vec x ho_true_vec)

    if ( !R_vec_true ) {
      R_vec_true = std::make_unique<shared::ParSparseMat>( std::move( block ) );
    } else {
      ( *R_vec_true ) += shared::ParSparseMatView( &block.get() );
    }
  }

  shared::ParSparseMatView R_vec_true_view( &R_vec_true->get() );
  auto tmp_true = R_vec_true_view * R_ho_vec;  // (lor_true_vec x ho_dof_vec)
  shared::ParSparseMatView tmp_true_view( &tmp_true.get() );
  return P_lor_vec * tmp_true_view;  // (lor_dof_vec x ho_dof_vec)
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

const shared::ParSparseMat* MfemJacobianData::GetDisplacementHoToLorTransfer() const
{
  if ( !update_data_ ) return nullptr;
  return update_data_->T_disp_ho_to_lor_.get();
}

const shared::ParSparseMat* MfemJacobianData::GetLagrangeMultiplierHoToLorTransfer() const
{
  if ( !update_data_ ) return nullptr;
  return update_data_->T_lm_ho_to_lor_.get();
}

shared::ParSparseMat MfemJacobianData::GetMfemJacobian( const std::vector<ComputedElementData>& contributions ) const
{
  std::unique_ptr<shared::ParSparseMat> par_J;

  // Maps BlockSpaces (MORTAR, NONMORTAR, LAGRANGE_MULTIPLIER) to a tribol element map
  const std::vector<const Array1D<int>*> elem_map_by_space{ &parent_data_.GetElemMap1(), &parent_data_.GetElemMap2(),
                                                            &parent_data_.GetElemMap2() };

  auto comm = parent_data_.GetParentCoords().ParFESpace()->GetComm();

  // Iterate over all possible blocks: (0,0), (0,1), (1,0), (1,1)
  // 0: Displacement (MORTAR/NONMORTAR)
  // 1: Pressure/Gap (LAGRANGE_MULTIPLIER)
  for ( int r_blk = 0; r_blk < 2; ++r_blk ) {
    for ( int c_blk = 0; c_blk < 2; ++c_blk ) {
      // Check if we have a transfer operator for this block
      if ( GetUpdateData().submesh_redecomp_xfer_.shape()[0] <= r_blk ||
           GetUpdateData().submesh_redecomp_xfer_.shape()[1] <= c_blk ||
           !GetUpdateData().submesh_redecomp_xfer_( r_blk, c_blk ) ) {
        continue;
      }

      axom::Array<int> row_redecomp_ids;
      axom::Array<int> col_redecomp_ids;
      axom::Array<double> jacobian_data;
      axom::Array<int> jacobian_offsets;

      // Aggregate data for this block pair
      for ( const auto& contrib : contributions ) {
        int contrib_r_blk = ( contrib.row_space == BlockSpace::LAGRANGE_MULTIPLIER ) ? 1 : 0;
        int contrib_c_blk = ( contrib.col_space == BlockSpace::LAGRANGE_MULTIPLIER ) ? 1 : 0;

        if ( contrib_r_blk == r_blk && contrib_c_blk == c_blk ) {
          int current_offset = jacobian_data.size();
          row_redecomp_ids.reserve( row_redecomp_ids.size() + contrib.row_elem_ids.size() );
          for ( auto id : contrib.row_elem_ids ) {
            row_redecomp_ids.push_back(
                ( *elem_map_by_space[static_cast<size_t>( contrib.row_space )] )[static_cast<size_t>( id )] );
          }
          col_redecomp_ids.reserve( col_redecomp_ids.size() + contrib.col_elem_ids.size() );
          for ( auto id : contrib.col_elem_ids ) {
            col_redecomp_ids.push_back(
                ( *elem_map_by_space[static_cast<size_t>( contrib.col_space )] )[static_cast<size_t>( id )] );
          }
          // NOTE (EBC): This can be removed when Axom PR 1819 goes in
          if ( contrib.jacobian_data.size() > 0 ) {
            jacobian_data.append( axom::ArrayView<const double>( contrib.jacobian_data ) );
          }
          jacobian_offsets.reserve( jacobian_offsets.size() + contrib.jacobian_offsets.size() );
          for ( auto offset : contrib.jacobian_offsets ) {
            jacobian_offsets.push_back( current_offset + offset );
          }
        }
      }

      // Check globally if any rank has data for this block
      int local_has_data = row_redecomp_ids.empty() ? 0 : 1;
      int global_has_data = 0;
      MPI_Allreduce( &local_has_data, &global_has_data, 1, MPI_INT, MPI_MAX, comm );

      if ( global_has_data ) {
        redecomp::MatrixTransfer* xfer = GetUpdateData().submesh_redecomp_xfer_( r_blk, c_blk ).get();
        auto submesh_J =
            xfer->TransferToParallel( row_redecomp_ids, col_redecomp_ids, jacobian_data, jacobian_offsets, false );

        std::unique_ptr<shared::ParSparseMat> mapped_submesh_J;
        shared::ParSparseMatView submesh_J_view( &submesh_J.get() );
        if ( parent_data_.GetLORMesh() ) {
          SLIC_ERROR_ROOT_IF(
              GetUpdateData().T_disp_ho_to_lor_ == nullptr || GetUpdateData().T_lm_ho_to_lor_ == nullptr,
              "LOR Jacobian mapping requires transfer matrices. Call UpdateJacobianXfer()." );
          const auto& T_row_mat =
              ( r_blk == 0 ) ? *GetUpdateData().T_disp_ho_to_lor_ : *GetUpdateData().T_lm_ho_to_lor_;
          const auto& T_col_mat =
              ( c_blk == 0 ) ? *GetUpdateData().T_disp_ho_to_lor_ : *GetUpdateData().T_lm_ho_to_lor_;
          shared::ParSparseMatView T_row( const_cast<mfem::HypreParMatrix*>( &T_row_mat.get() ) );
          shared::ParSparseMatView T_col( const_cast<mfem::HypreParMatrix*>( &T_col_mat.get() ) );
          mapped_submesh_J =
              std::make_unique<shared::ParSparseMat>( shared::ParSparseMat::rap( T_row, submesh_J_view, T_col ) );
          submesh_J_view = shared::ParSparseMatView( &mapped_submesh_J->get() );
        }
        std::unique_ptr<shared::ParSparseMat> contrib_J;

        if ( r_blk == 0 && c_blk == 0 ) {
          auto parent_J = submesh_J_view.rap( *submesh_parent_vdof_xfer_ );
          shared::ParSparseMatView parent_P( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() );
          contrib_J = std::make_unique<shared::ParSparseMat>( parent_J.rap( parent_P ) );
        } else if ( r_blk == 0 && c_blk == 1 ) {
          auto parent_J = submesh_parent_vdof_xfer_->transpose() * submesh_J_view;
          contrib_J = std::make_unique<shared::ParSparseMat>(
              shared::ParSparseMat::rap( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix(), parent_J,
                                         submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix() ) );
        } else if ( r_blk == 1 && c_blk == 0 ) {
          auto parent_J = submesh_J_view * ( *submesh_parent_vdof_xfer_ );
          shared::ParSparseMatView submesh_P( submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix() );
          shared::ParSparseMatView parent_P( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() );
          contrib_J =
              std::make_unique<shared::ParSparseMat>( shared::ParSparseMat::rap( submesh_P, parent_J, parent_P ) );
        } else {
          // (1, 1) block
          shared::ParSparseMatView submesh_P( submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix() );
          contrib_J = std::make_unique<shared::ParSparseMat>( submesh_J_view.rap( submesh_P ) );
        }

        if ( !par_J ) {
          par_J = std::move( contrib_J );
        } else {
          ( *par_J ) += *contrib_J;
        }
      }
    }
  }

  if ( !par_J ) {
    int target_r_blk = 0;
    int target_c_blk = 0;
    if ( !contributions.empty() ) {
      target_r_blk = ( contributions[0].row_space == BlockSpace::LAGRANGE_MULTIPLIER ) ? 1 : 0;
      target_c_blk = ( contributions[0].col_space == BlockSpace::LAGRANGE_MULTIPLIER ) ? 1 : 0;
    }

    auto& row_fes =
        ( target_r_blk == 0 ) ? *parent_data_.GetParentCoords().ParFESpace() : submesh_data_.GetSubmeshFESpace();
    auto& col_fes =
        ( target_c_blk == 0 ) ? *parent_data_.GetParentCoords().ParFESpace() : submesh_data_.GetSubmeshFESpace();

    if ( target_r_blk == target_c_blk ) {
      return shared::ParSparseMat::diagonalMatrix( comm, row_fes.GlobalTrueVSize(), row_fes.GetTrueDofOffsets(), 0.0,
                                                   mfem::Array<int>(), true );
    } else {
      mfem::SparseMatrix empty_diag( row_fes.GetTrueVSize(), col_fes.GetTrueVSize() );
      empty_diag.Finalize();
      return shared::ParSparseMat( comm, row_fes.GlobalTrueVSize(), col_fes.GlobalTrueVSize(),
                                   row_fes.GetTrueDofOffsets(), col_fes.GetTrueDofOffsets(), std::move( empty_diag ) );
    }
  }

  return std::move( *par_J );
}

JacobianContributions::JacobianContributions( std::initializer_list<std::pair<BlockSpace, BlockSpace>> blocks )
{
  for ( const auto& block : blocks ) {
    ComputedElementData data;
    data.row_space = block.first;
    data.col_space = block.second;
    contributions_.push_back( std::move( data ) );
  }
}

void JacobianContributions::reserve( int n_pairs, int n_entries_per_pair )
{
  for ( auto& contrib : contributions_ ) {
    contrib.row_elem_ids.reserve( n_pairs );
    contrib.col_elem_ids.reserve( n_pairs );
    contrib.jacobian_data.reserve( n_pairs * n_entries_per_pair );
    contrib.jacobian_offsets.reserve( n_pairs );
  }
}

void JacobianContributions::push_back( int block_idx, int row_elem_id, int col_elem_id, const double* data, int size )
{
  auto& contrib = contributions_[block_idx];
  contrib.row_elem_ids.push_back( row_elem_id );
  contrib.col_elem_ids.push_back( col_elem_id );
  contrib.jacobian_offsets.push_back( contrib.jacobian_data.size() );
  contrib.jacobian_data.append( axom::ArrayView<const double>( data, size ) );
}

}  // namespace tribol

#endif /* BUILD_REDECOMP */
