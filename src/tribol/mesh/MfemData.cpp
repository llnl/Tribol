// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MfemData.hpp"

#include "tribol/config.hpp"

#ifdef BUILD_REDECOMP

#include <map>

#include "axom/slic.hpp"

#include "shared/infrastructure/Profiling.hpp"

#include "redecomp/utils/ArrayUtility.hpp"

#include "tribol/common/LoopExec.hpp"

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
  submesh_parent_vdof_xfer_ = std::make_unique<ParSparseMat>(
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

      auto submesh_J_hypre = xfer->ConvertToHypreParMatrix( *submesh_J, false );

      mfem::HypreParMatrix* block_mat = nullptr;

      if ( r_blk == 0 && c_blk == 0 ) {
        ParSparseMatView submesh_J_view( submesh_J_hypre.get() );
        auto parent_J = submesh_J_view.RAP( *submesh_parent_vdof_xfer_ );
        ParSparseMatView parent_P( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() );
        block_mat = parent_J.RAP( parent_P ).release();
      } else if ( r_blk == 0 && c_blk == 1 ) {
        auto parent_J = submesh_parent_vdof_xfer_->transpose() * submesh_J_hypre.get();
        block_mat = ParSparseMat::RAP( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix(), parent_J,
                                       submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix() )
                        .release();
      } else if ( r_blk == 1 && c_blk == 0 ) {
        ParSparseMatView submesh_J_view( submesh_J_hypre.get() );
        auto parent_J = submesh_J_view * ( *submesh_parent_vdof_xfer_ );
        ParSparseMatView submesh_P( submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix() );
        ParSparseMatView parent_P( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() );
        block_mat = ParSparseMat::RAP( submesh_P, parent_J, parent_P ).release();
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
    ParSparseMat inactive_hpm_full =
        ParSparseMat::diagonalMatrix( TRIBOL_COMM_WORLD, submesh_fes_full.GlobalTrueVSize(),
                                      submesh_fes_full.GetTrueDofOffsets(), 1.0, mortar_tdof_list_, false );

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

ParSparseMat MfemJacobianData::GetMfemJacobian( const std::vector<ComputedElementData>& contributions ) const
{
  std::unique_ptr<ParSparseMat> par_J;

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
          jacobian_data.append( axom::ArrayView<const double>( contrib.jacobian_data ) );
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
        auto submesh_J_hypre = xfer->TransferToParallelSparse( row_redecomp_ids, col_redecomp_ids, jacobian_data,
                                                               jacobian_offsets );

        ParSparseMatView submesh_J_view( submesh_J_hypre.get() );
        std::unique_ptr<ParSparseMat> contrib_J;

        if ( r_blk == 0 && c_blk == 0 ) {
          auto parent_J = submesh_J_view.RAP( *submesh_parent_vdof_xfer_ );
          ParSparseMatView parent_P( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() );
          contrib_J = std::make_unique<ParSparseMat>( parent_J.RAP( parent_P ) );
        } else if ( r_blk == 0 && c_blk == 1 ) {
          auto parent_J = submesh_parent_vdof_xfer_->transpose() * submesh_J_hypre.get();
          contrib_J = std::make_unique<ParSparseMat>(
              ParSparseMat::RAP( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix(), parent_J,
                                 submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix() ) );
        } else if ( r_blk == 1 && c_blk == 0 ) {
          auto parent_J = submesh_J_view * ( *submesh_parent_vdof_xfer_ );
          ParSparseMatView submesh_P( submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix() );
          ParSparseMatView parent_P( parent_data_.GetParentCoords().ParFESpace()->Dof_TrueDof_Matrix() );
          contrib_J = std::make_unique<ParSparseMat>( ParSparseMat::RAP( submesh_P, parent_J, parent_P ) );
        } else {
          // (1, 1) block
          ParSparseMatView submesh_P( submesh_data_.GetSubmeshFESpace().Dof_TrueDof_Matrix() );
          contrib_J = std::make_unique<ParSparseMat>( submesh_J_view.RAP( submesh_P ) );
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
    auto& fes = *parent_data_.GetParentCoords().ParFESpace();
    return ParSparseMat::diagonalMatrix( TRIBOL_COMM_WORLD, fes.GetTrueVSize(), fes.GetTrueDofOffsets(), 0.0,
                                         mfem::Array<int>(), true );
  }

  return std::move( *par_J );
}

}  // namespace tribol

#endif /* BUILD_REDECOMP */
