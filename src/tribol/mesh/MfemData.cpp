// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MfemData.hpp"

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP

#include "axom/slic.hpp"

#include "shared/infrastructure/Profiling.hpp"

#include "redecomp/utils/ArrayUtility.hpp"

namespace tribol {

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
  // make sure host data is up to date.  this transfer needs to be on the host until submesh supports device transfer
  lor_gridfn_->HostRead();
  submesh_dst.HostWrite();
  lor_xfer_.ForwardOperator().MultTranspose( *lor_gridfn_, submesh_dst );
}

void SubmeshLORTransfer::SubmeshToLOR( const mfem::ParGridFunction& submesh_src, mfem::ParGridFunction& lor_dst )
{
  TRIBOL_MARK_FUNCTION;
  // make sure host data is up to date.  this transfer needs to be on the host until submesh supports device transfer
  submesh_src.HostRead();
  lor_dst.HostWrite();
  lor_xfer_.ForwardOperator().Mult( submesh_src, lor_dst );
}

std::unique_ptr<mfem::ParGridFunction> SubmeshLORTransfer::CreateLORGridFunction(
    mfem::ParMesh& lor_mesh, std::unique_ptr<mfem::FiniteElementCollection> lor_fec, int vdim )
{
  auto lor_gridfn = std::make_unique<mfem::ParGridFunction>(
      new mfem::ParFiniteElementSpace( &lor_mesh, lor_fec.get(), vdim, mfem::Ordering::byNODES ) );
  lor_gridfn->MakeOwner( lor_fec.release() );
  // NOTE: This needs to be false until submesh supports device transfer. Otherwise, there will be extra copies to/from
  // device.
  lor_gridfn->UseDevice( false );
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
  auto dst_data = dst_ptr->HostWrite();
  auto P_I =
      mfem::Read( dst_fespace_ptr->Dof_TrueDof_Matrix()->GetDiagMemoryI(), dst_fespace_ptr->GetVSize() + 1, false );
  HYPRE_Int tdof_ct{ 0 };
  // TODO: Convert to mfem::forall() once submesh transfers on device and once GPU-enabled MPI is in redecomp (dst_data
  // is always on host now so not needed yet)
  for ( int i{ 0 }; i < dst_fespace_ptr->GetVSize(); ++i ) {
    if ( P_I[i + 1] != tdof_ct ) {
      ++tdof_ct;
    } else {
      dst_data[i] = 0.0;
    }
  }
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
  // keep on host since tribol does computations there
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
  // NOTE: This needs to be on host until the submesh transfer supports device.  Otherwise, there will be extra
  // transfers to/from device.
  submesh_xfer_gridfn_.UseDevice( false );

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
    const RealT* d_curr = current_coords_gf.Read();
    const RealT* d_last = coords_at_last_redecomp_.Read();
    mfem::Vector max_diff( 1 );
    max_diff.UseDevice( use_device_ );
    max_diff = 0.0;
    RealT* d_max_diff = max_diff.Write();
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
                                  std::unique_ptr<mfem::FiniteElementCollection> pressure_fec, int pressure_vdim )
    : submesh_pressure_{ new mfem::ParFiniteElementSpace( &submesh, pressure_fec.get(), pressure_vdim ) },
      pressure_{ submesh_pressure_ },
      submesh_lor_xfer_{ lor_mesh ? std::make_unique<SubmeshLORTransfer>( *submesh_pressure_.ParFESpace(), *lor_mesh )
                                  : nullptr }
{
  submesh_pressure_.MakeOwner( pressure_fec.release() );
  submesh_pressure_ = 0.0;
}

void MfemSubmeshData::UpdateMfemSubmeshData( redecomp::RedecompMesh& redecomp_mesh, bool new_redecomp )
{
  if ( new_redecomp || !update_data_ ) {
    update_data_ =
        std::make_unique<UpdateData>( *submesh_pressure_.ParFESpace(), submesh_lor_xfer_.get(), redecomp_mesh );
  }
  pressure_.UpdateField( update_data_->pressure_xfer_ );
  redecomp_gap_.SetSpace( pressure_.GetRedecompGridFn().FESpace() );
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
  SLIC_ERROR_ROOT_IF( parent_data.GetParentCoords().ParFESpace()->FEColl()->GetOrder() > 1,
                      "Higher order meshes not yet supported for Jacobian matrices." );

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
  submesh_parent_vdof_xfer_ = std::make_unique<mfem::HypreParMatrix>(
      TRIBOL_COMM_WORLD, submesh_fes.GetVSize(), submesh_fes.GlobalVSize(), parent_fes.GlobalVSize(),
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

std::unique_ptr<mfem::BlockOperator> MfemJacobianData::GetMfemBlockJacobian( const MethodData* method_data ) const
{
  // 0 = displacement DOFs, 1 = lagrange multiplier DOFs
  // (0,0) block is empty (for now using SINGLE_MORTAR with approximate tangent)
  // (1,1) block is a diagonal matrix with ones on the diagonal of submesh nodes without a Lagrange multiplier DOF
  // (0,1) and (1,0) are symmetric (for now using SINGLE_MORTAR with approximate tangent)
  const auto& elem_map_1 = parent_data_.GetElemMap1();
  const auto& elem_map_2 = parent_data_.GetElemMap2();
  // empty data structures are needed even when no meshes are on rank since TransferToParallelSparse() needs to be
  // called on all ranks (even those without data)
  auto mortar_elems = ArrayT<int>( 0, 0 );
  auto nonmortar_elems = ArrayT<int>( 0, 0 );
  auto lm_elems = ArrayT<int>( 0, 0 );
  auto elem_J_1_ptr = std::make_unique<ArrayT<mfem::DenseMatrix>>( 0, 0 );
  auto elem_J_2_ptr = std::make_unique<ArrayT<mfem::DenseMatrix>>( 0, 0 );
  const ArrayT<mfem::DenseMatrix>* elem_J_1 = elem_J_1_ptr.get();
  const ArrayT<mfem::DenseMatrix>* elem_J_2 = elem_J_2_ptr.get();
  // this means both of the meshes exist
  if ( method_data != nullptr && !elem_map_1.empty() && !elem_map_2.empty() ) {
    mortar_elems = method_data->getBlockJElementIds()[static_cast<int>( BlockSpace::MORTAR )];
    for ( auto& mortar_elem : mortar_elems ) {
      mortar_elem = elem_map_1[static_cast<size_t>( mortar_elem )];
    }
    nonmortar_elems = method_data->getBlockJElementIds()[static_cast<int>( BlockSpace::NONMORTAR )];
    for ( auto& nonmortar_elem : nonmortar_elems ) {
      nonmortar_elem = elem_map_2[static_cast<size_t>( nonmortar_elem )];
    }
    lm_elems = method_data->getBlockJElementIds()[static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER )];
    for ( auto& lm_elem : lm_elems ) {
      lm_elem = elem_map_2[static_cast<size_t>( lm_elem )];
    }
    // get (1,0) block
    elem_J_1 = &method_data->getBlockJ()( static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ),
                                          static_cast<int>( BlockSpace::MORTAR ) );
    elem_J_2 = &method_data->getBlockJ()( static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ),
                                          static_cast<int>( BlockSpace::NONMORTAR ) );
  }
  // move to submesh level
  auto submesh_J =
      GetUpdateData().submesh_redecomp_xfer_10_->TransferToParallelSparse( lm_elems, mortar_elems, *elem_J_1 );
  submesh_J +=
      GetUpdateData().submesh_redecomp_xfer_10_->TransferToParallelSparse( lm_elems, nonmortar_elems, *elem_J_2 );
  submesh_J.Finalize();

  // transform J values from submesh to (global) parent mesh
  mfem::Array<HYPRE_BigInt> J( submesh_J.NumNonZeroElems() );
  // This copy is needed to convert mfem::SparseMatrix int J values to the HYPRE_BigInt values the mfem::HypreParMatrix
  // constructor needs
  auto* J_int = submesh_J.GetJ();
  for ( int i{ 0 }; i < J.Size(); ++i ) {
    J[i] = J_int[i];
  }
  auto submesh_vector_fes = parent_data_.GetSubmeshFESpace();
  auto mpi = redecomp::MPIUtility( submesh_vector_fes.GetComm() );
  auto submesh_dof_offsets = ArrayT<int>( mpi.NRanks() + 1, mpi.NRanks() + 1 );
  // we need the dof offsets of each rank.  check if mfem stores this or if we
  // need to create it.
  if ( HYPRE_AssumedPartitionCheck() ) {
    submesh_dof_offsets[mpi.MyRank() + 1] = submesh_vector_fes.GetDofOffsets()[1];
    mpi.Allreduce( &submesh_dof_offsets, MPI_SUM );
  } else {
    for ( int i{ 0 }; i < mpi.NRanks(); ++i ) {
      submesh_dof_offsets[i] = submesh_vector_fes.GetDofOffsets()[i];
    }
  }
  // the submesh to parent vdof map only exists for vdofs on rank, so J values
  // not on rank will need to be transferred to the rank that the vdof exists on
  // to query the map. the steps are laid out below.

  // step 1) query J values on rank for their parent vdof and package J values
  // not on rank to send
  auto send_J_by_rank = redecomp::MPIArray<int>( &mpi );
  auto J_idx = redecomp::MPIArray<int>( &mpi );
  auto est_num_J = submesh_J.NumNonZeroElems() / mpi.NRanks();
  for ( int r{}; r < mpi.NRanks(); ++r ) {
    if ( r == mpi.MyRank() ) {
      send_J_by_rank[r].shrink();
      J_idx[r].shrink();
    } else {
      send_J_by_rank[r].reserve( est_num_J );
      J_idx[r].reserve( est_num_J );
    }
  }
  for ( int j{}; j < submesh_J.NumNonZeroElems(); ++j ) {
    if ( J[j] >= submesh_dof_offsets[mpi.MyRank()] && J[j] < submesh_dof_offsets[mpi.MyRank() + 1] ) {
      J[j] = submesh2parent_vdof_list_[J[j] - submesh_dof_offsets[mpi.MyRank()]];
    } else {
      for ( int r{}; r < mpi.NRanks(); ++r ) {
        if ( J[j] >= submesh_dof_offsets[r] && J[j] < submesh_dof_offsets[r + 1] ) {
          send_J_by_rank[r].push_back( J[j] - submesh_dof_offsets[r] );
          J_idx[r].push_back( j );
          break;
        }
      }
    }
  }
  // step 2) sends the J values to the ranks that own them
  auto recv_J_by_rank = redecomp::MPIArray<int>( &mpi );
  recv_J_by_rank.SendRecvArrayEach( send_J_by_rank );
  // step 3) query the on-rank map to recover J values
  for ( int r{}; r < mpi.NRanks(); ++r ) {
    for ( auto& recv_J : recv_J_by_rank[r] ) {
      recv_J = submesh2parent_vdof_list_[recv_J];
    }
  }
  // step 4) send the updated parent J values back and update the J vector
  send_J_by_rank.SendRecvArrayEach( recv_J_by_rank );
  for ( int r{}; r < mpi.NRanks(); ++r ) {
    for ( int j{}; j < send_J_by_rank[r].size(); ++j ) {
      J[J_idx[r][j]] = send_J_by_rank[r][j];
    }
  }

  // create block operator
  auto block_J = std::make_unique<mfem::BlockOperator>( block_offsets_ );
  block_J->owns_blocks = 1;

  // fill block operator
  auto& submesh_fes = submesh_data_.GetSubmeshFESpace();
  auto& parent_trial_fes = *parent_data_.GetParentCoords().ParFESpace();
  // NOTE: we don't call MatrixTransfer::ConvertToHypreParMatrix() because the
  // trial space is on the parent mesh, not the submesh
  auto J_full = std::make_unique<mfem::HypreParMatrix>( mpi.MPIComm(), submesh_fes.GetVSize(),
                                                        submesh_fes.GlobalVSize(), parent_trial_fes.GlobalVSize(),
                                                        submesh_J.GetI(), J.GetData(), submesh_J.GetData(),
                                                        submesh_fes.GetDofOffsets(), parent_trial_fes.GetDofOffsets() );
  auto J_true = std::unique_ptr<mfem::HypreParMatrix>(
      mfem::RAP( submesh_fes.Dof_TrueDof_Matrix(), J_full.get(), parent_trial_fes.Dof_TrueDof_Matrix() ) );

  // Create ones on diagonal of eliminated mortar tdofs, i.e. inactive dofs (CSR sparse matrix -> HypreParMatrix)
  // I vector
  mfem::Array<int> rows( submesh_fes.GetTrueVSize() + 1 );
  rows = 0;
  auto mortar_tdofs_ct = 0;
  for ( int i{ 0 }; i < submesh_fes.GetTrueVSize(); ++i ) {
    if ( mortar_tdofs_ct < mortar_tdof_list_.Size() && mortar_tdof_list_[mortar_tdofs_ct] == i ) {
      ++mortar_tdofs_ct;
    }
    rows[i + 1] = mortar_tdofs_ct;
  }
  // J vector
  mfem::Array<int> mortar_tdofs( mortar_tdof_list_ );
  // data vector
  mfem::Vector ones( mortar_tdofs_ct );
  ones = 1.0;
  mfem::SparseMatrix inactive_sm( rows.GetData(), mortar_tdofs.GetData(), ones.GetData(), submesh_fes.GetTrueVSize(),
                                  submesh_fes.GetTrueVSize(), false, false, true );
  auto inactive_hpm = std::make_unique<mfem::HypreParMatrix>( J_true->GetComm(), J_true->GetGlobalNumRows(),
                                                              J_true->GetRowStarts(), &inactive_sm );
  // Have the mfem::HypreParMatrix manage the data pointers
  rows.GetMemory().ClearOwnerFlags();
  mortar_tdofs.GetMemory().ClearOwnerFlags();
  ones.GetMemory().ClearOwnerFlags();
  inactive_sm.GetMemoryI().ClearOwnerFlags();
  inactive_sm.GetMemoryJ().ClearOwnerFlags();
  inactive_sm.GetMemoryData().ClearOwnerFlags();

  block_J->SetBlock( 0, 1, J_true->Transpose() );
  block_J->SetBlock( 1, 0, J_true.release() );
  block_J->SetBlock( 1, 1, inactive_hpm.release() );

  return block_J;
}

// TODO: Merge with GetMfemBlockJacobian() to avoid code duplication
std::unique_ptr<mfem::BlockOperator> MfemJacobianData::GetMfemDfDxFullJacobian( const MethodData& method_data ) const
{
  // create block operator
  auto block_J = std::make_unique<mfem::BlockOperator>( block_offsets_ );
  block_J->owns_blocks = 1;

  // these are Tribol element ids
  auto mortar_elems = method_data.getBlockJElementIds()[static_cast<int>( BlockSpace::MORTAR )];
  // convert them to redecomp element ids
  const auto& elem_map_1 = parent_data_.GetElemMap1();
  for ( auto& mortar_elem : mortar_elems ) {
    mortar_elem = elem_map_1[static_cast<size_t>( mortar_elem )];
  }

  // these are Tribol element ids
  auto nonmortar_elems = method_data.getBlockJElementIds()[static_cast<int>( BlockSpace::NONMORTAR )];
  // convert them to redecomp element ids
  const auto& elem_map_2 = parent_data_.GetElemMap2();
  for ( auto& nonmortar_elem : nonmortar_elems ) {
    nonmortar_elem = elem_map_2[static_cast<size_t>( nonmortar_elem )];
  }

  // these are Tribol element ids
  auto lm_elems = method_data.getBlockJElementIds()[static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER )];
  // convert them to redecomp element ids
  for ( auto& lm_elem : lm_elems ) {
    lm_elem = elem_map_2[static_cast<size_t>( lm_elem )];
  }

  // transfer (0, 0) block (residual dof rows, displacement dof cols)
  auto submesh_J = GetUpdateData().submesh_redecomp_xfer_00_->TransferToParallelSparse(
      mortar_elems, mortar_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::MORTAR ), static_cast<int>( BlockSpace::MORTAR ) ) );
  submesh_J += GetUpdateData().submesh_redecomp_xfer_00_->TransferToParallelSparse(
      mortar_elems, nonmortar_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::MORTAR ), static_cast<int>( BlockSpace::NONMORTAR ) ) );
  submesh_J += GetUpdateData().submesh_redecomp_xfer_00_->TransferToParallelSparse(
      nonmortar_elems, mortar_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ), static_cast<int>( BlockSpace::MORTAR ) ) );
  submesh_J += GetUpdateData().submesh_redecomp_xfer_00_->TransferToParallelSparse(
      nonmortar_elems, nonmortar_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ), static_cast<int>( BlockSpace::NONMORTAR ) ) );
  submesh_J.Finalize();
  auto submesh_J_hypre = GetUpdateData().submesh_redecomp_xfer_00_->ConvertToHypreParMatrix( submesh_J, false );
  // Matrix returned by mfem::RAP copies all existing data and owns its data
  auto parent_J_hypre =
      std::unique_ptr<mfem::HypreParMatrix>( mfem::RAP( submesh_J_hypre.get(), submesh_parent_vdof_xfer_.get() ) );
  block_J->SetBlock(
      0, 0, mfem::RAP( parent_J_hypre.get(), parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() ) );

  // transfer (0, 1) block (residual dof rows, lagrange multiplier dof cols)
  submesh_J = GetUpdateData().submesh_redecomp_xfer_01_->TransferToParallelSparse(
      mortar_elems, lm_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::MORTAR ),
                               static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ) ) );
  submesh_J += GetUpdateData().submesh_redecomp_xfer_01_->TransferToParallelSparse(
      nonmortar_elems, lm_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ),
                               static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ) ) );
  submesh_J.Finalize();
  submesh_J_hypre = GetUpdateData().submesh_redecomp_xfer_01_->ConvertToHypreParMatrix( submesh_J, false );
  // Matrix returned by mfem::ParMult copies row and column starts since last arg is true. All other data is copied and
  // owned by the new matrix.
  parent_J_hypre = std::unique_ptr<mfem::HypreParMatrix>(
      mfem::ParMult( std::unique_ptr<mfem::HypreParMatrix>( submesh_parent_vdof_xfer_->Transpose() ).get(),
                     submesh_J_hypre.get(), true ) );
  block_J->SetBlock( 0, 1,
                     mfem::RAP( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix(), parent_J_hypre.get(),
                                submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix() ) );

  // transfer (1, 0) block (gap dof rows, displacement dof cols)
  submesh_J = GetUpdateData().submesh_redecomp_xfer_10_->TransferToParallelSparse(
      lm_elems, mortar_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ),
                               static_cast<int>( BlockSpace::MORTAR ) ) );
  submesh_J += GetUpdateData().submesh_redecomp_xfer_10_->TransferToParallelSparse(
      lm_elems, nonmortar_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ),
                               static_cast<int>( BlockSpace::NONMORTAR ) ) );
  submesh_J.Finalize();
  submesh_J_hypre = GetUpdateData().submesh_redecomp_xfer_10_->ConvertToHypreParMatrix( submesh_J, false );
  // Matrix returned by mfem::ParMult copies row and column starts since last arg is true. All other data is copied and
  // owned by the new matrix.
  parent_J_hypre = std::unique_ptr<mfem::HypreParMatrix>( std::unique_ptr<mfem::HypreParMatrix>(
      mfem::ParMult( submesh_J_hypre.get(), submesh_parent_vdof_xfer_.get(), true ) ) );
  block_J->SetBlock( 1, 0,
                     mfem::RAP( submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix(), parent_J_hypre.get(),
                                parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() ) );

  // Create ones on diagonal of eliminated mortar tdofs, i.e. inactive dofs (CSR sparse matrix -> HypreParMatrix)
  // I vector
  auto& submesh_fes = submesh_data_.GetSubmeshFESpace();
  mfem::Array<int> rows( submesh_fes.GetTrueVSize() + 1 );
  rows = 0;
  auto mortar_tdofs_ct = 0;
  for ( int i{ 0 }; i < submesh_fes.GetTrueVSize(); ++i ) {
    if ( mortar_tdofs_ct < mortar_tdof_list_.Size() && mortar_tdof_list_[mortar_tdofs_ct] == i ) {
      ++mortar_tdofs_ct;
    }
    rows[i + 1] = mortar_tdofs_ct;
  }
  // J vector
  mfem::Array<int> mortar_tdofs( mortar_tdof_list_ );
  // data vector
  mfem::Vector ones( mortar_tdofs_ct );
  ones = 1.0;
  mfem::SparseMatrix inactive_sm( rows.GetData(), mortar_tdofs.GetData(), ones.GetData(), submesh_fes.GetTrueVSize(),
                                  submesh_fes.GetTrueVSize(), false, false, true );
  auto inactive_hpm = std::make_unique<mfem::HypreParMatrix>( TRIBOL_COMM_WORLD, submesh_fes.GlobalTrueVSize(),
                                                              submesh_fes.GetTrueDofOffsets(), &inactive_sm );
  // Have the mfem::HypreParMatrix manage the data pointers
  rows.GetMemory().SetHostPtrOwner( false );
  mortar_tdofs.GetMemory().SetHostPtrOwner( false );
  ones.GetMemory().SetHostPtrOwner( false );
  inactive_sm.SetDataOwner( false );
  inactive_hpm->SetOwnerFlags( 3, 3, 1 );
  block_J->SetBlock( 1, 1, inactive_hpm.release() );

  return block_J;
}

// TODO: Merge with GetMfemBlockJacobian() to avoid code duplication
std::unique_ptr<mfem::BlockOperator> MfemJacobianData::GetMfemDfDnJacobian( const MethodData& method_data ) const
{
  // create block operator
  auto block_J = std::make_unique<mfem::BlockOperator>( block_offsets_, disp_offsets_ );
  block_J->owns_blocks = 1;

  // these are Tribol element ids
  auto mortar_elems = method_data.getBlockJElementIds()[static_cast<int>( BlockSpace::MORTAR )];
  // convert them to redecomp element ids
  const auto& elem_map_1 = parent_data_.GetElemMap1();
  for ( auto& mortar_elem : mortar_elems ) {
    mortar_elem = elem_map_1[static_cast<size_t>( mortar_elem )];
  }

  // these are Tribol element ids
  auto nonmortar_elems = method_data.getBlockJElementIds()[static_cast<int>( BlockSpace::NONMORTAR )];
  // convert them to redecomp element ids
  const auto& elem_map_2 = parent_data_.GetElemMap2();
  for ( auto& nonmortar_elem : nonmortar_elems ) {
    nonmortar_elem = elem_map_2[static_cast<size_t>( nonmortar_elem )];
  }

  // these are Tribol element ids
  auto lm_elems = method_data.getBlockJElementIds()[static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER )];
  // convert them to redecomp element ids
  for ( auto& lm_elem : lm_elems ) {
    lm_elem = elem_map_2[static_cast<size_t>( lm_elem )];
  }

  // transfer (0, 0) block (residual dof rows, displacement dof cols)
  auto submesh_J = GetUpdateData().submesh_redecomp_xfer_00_->TransferToParallelSparse(
      mortar_elems, nonmortar_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::MORTAR ), static_cast<int>( BlockSpace::NONMORTAR ) ) );
  submesh_J += GetUpdateData().submesh_redecomp_xfer_00_->TransferToParallelSparse(
      nonmortar_elems, nonmortar_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ), static_cast<int>( BlockSpace::NONMORTAR ) ) );
  submesh_J.Finalize();
  auto submesh_J_hypre = GetUpdateData().submesh_redecomp_xfer_00_->ConvertToHypreParMatrix( submesh_J, false );
  // Matrix returned by mfem::RAP copies all existing data and owns its data
  auto parent_J_hypre =
      std::unique_ptr<mfem::HypreParMatrix>( mfem::RAP( submesh_J_hypre.get(), submesh_parent_vdof_xfer_.get() ) );
  block_J->SetBlock(
      0, 0, mfem::RAP( parent_J_hypre.get(), parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() ) );

  // transfer (1, 0) block (gap dof rows, displacement dof cols)
  submesh_J = GetUpdateData().submesh_redecomp_xfer_10_->TransferToParallelSparse(
      lm_elems, nonmortar_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ),
                               static_cast<int>( BlockSpace::NONMORTAR ) ) );
  submesh_J.Finalize();
  submesh_J_hypre = GetUpdateData().submesh_redecomp_xfer_10_->ConvertToHypreParMatrix( submesh_J, false );
  // Matrix returned by mfem::ParMult copies row and column starts since last arg is true. All other data is copied and
  // owned by the new matrix.
  parent_J_hypre = std::unique_ptr<mfem::HypreParMatrix>(
      mfem::ParMult( submesh_J_hypre.get(), submesh_parent_vdof_xfer_.get(), true ) );
  block_J->SetBlock( 1, 0,
                     mfem::RAP( submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix(), parent_J_hypre.get(),
                                parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() ) );

  return block_J;
}

// TODO: Merge with GetMfemBlockJacobian() to avoid code duplication
std::unique_ptr<mfem::BlockOperator> MfemJacobianData::GetMfemDnDxJacobian( const MethodData& method_data ) const
{
  // create block operator
  auto block_J = std::make_unique<mfem::BlockOperator>( disp_offsets_, disp_offsets_ );
  block_J->owns_blocks = 1;

  // these are Tribol element ids
  auto nonmortar_elems = method_data.getBlockJElementIds()[static_cast<int>( BlockSpace::NONMORTAR )];
  // convert them to redecomp element ids
  const auto& elem_map_2 = parent_data_.GetElemMap2();
  for ( auto& nonmortar_elem : nonmortar_elems ) {
    nonmortar_elem = elem_map_2[static_cast<size_t>( nonmortar_elem )];
  }

  // transfer (0, 0) block (residual dof rows, displacement dof cols)
  auto submesh_J = GetUpdateData().submesh_redecomp_xfer_00_->TransferToParallelSparse(
      nonmortar_elems, nonmortar_elems,
      method_data.getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ), static_cast<int>( BlockSpace::NONMORTAR ) ) );
  submesh_J.Finalize();
  auto submesh_J_hypre = GetUpdateData().submesh_redecomp_xfer_00_->ConvertToHypreParMatrix( submesh_J, false );
  // Matrix returned by mfem::RAP copies all existing data and owns its data
  auto parent_J_hypre =
      std::unique_ptr<mfem::HypreParMatrix>( mfem::RAP( submesh_J_hypre.get(), submesh_parent_vdof_xfer_.get() ) );
  block_J->SetBlock(
      0, 0, mfem::RAP( parent_J_hypre.get(), parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() ) );

  return block_J;
}

MfemJacobianData::UpdateData::UpdateData( const MfemMeshData& parent_data, const MfemSubmeshData& submesh_data )
{
  auto dual_submesh_fes = &submesh_data.GetSubmeshFESpace();
  auto primal_submesh_fes = &parent_data.GetSubmeshFESpace();
  if ( parent_data.GetLORMesh() ) {
    dual_submesh_fes = submesh_data.GetLORMeshFESpace();
    primal_submesh_fes = parent_data.GetLORMeshFESpace();
  }
  // create a matrix transfer operator for moving data from redecomp to the submesh
  submesh_redecomp_xfer_00_ = std::make_unique<redecomp::MatrixTransfer>(
      *primal_submesh_fes, *primal_submesh_fes, *parent_data.GetRedecompResponse().FESpace(),
      *parent_data.GetRedecompResponse().FESpace() );
  submesh_redecomp_xfer_01_ = std::make_unique<redecomp::MatrixTransfer>( *primal_submesh_fes, *dual_submesh_fes,
                                                                          *parent_data.GetRedecompResponse().FESpace(),
                                                                          *submesh_data.GetRedecompGap().FESpace() );
  submesh_redecomp_xfer_10_ = std::make_unique<redecomp::MatrixTransfer>(
      *dual_submesh_fes, *primal_submesh_fes, *submesh_data.GetRedecompGap().FESpace(),
      *parent_data.GetRedecompResponse().FESpace() );
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

}  // namespace tribol

#endif /* BUILD_REDECOMP */
