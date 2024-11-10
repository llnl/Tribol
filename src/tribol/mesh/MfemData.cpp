// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)


#include "tribol/mesh/MfemData.hpp"

#ifdef BUILD_REDECOMP

#include "axom/slic.hpp"

namespace tribol
{

SubmeshLORTransfer::SubmeshLORTransfer(
  mfem::ParFiniteElementSpace& submesh_fes,
  mfem::ParMesh& lor_mesh
)
: lor_gridfn_ { CreateLORGridFunction(
    lor_mesh,
    std::make_unique<mfem::H1_FECollection>(1, lor_mesh.SpaceDimension()),
    submesh_fes.GetVDim()
  ) },
  lor_xfer_ { submesh_fes, *lor_gridfn_->ParFESpace() }
{}

void SubmeshLORTransfer::TransferToLORGridFn(
  const mfem::ParGridFunction& submesh_src
)
{
  SubmeshToLOR(submesh_src, *lor_gridfn_);
}

void SubmeshLORTransfer::TransferFromLORVector(
  mfem::Vector& submesh_dst
) const
{
  lor_xfer_.ForwardOperator().MultTranspose(*lor_gridfn_, submesh_dst);
}

void SubmeshLORTransfer::SubmeshToLOR(
  const mfem::ParGridFunction& submesh_src,
  mfem::ParGridFunction& lor_dst
)
{
  lor_xfer_.ForwardOperator().Mult(submesh_src, lor_dst);
}

std::unique_ptr<mfem::ParGridFunction> SubmeshLORTransfer::CreateLORGridFunction(
  mfem::ParMesh& lor_mesh,
  std::unique_ptr<mfem::FiniteElementCollection> lor_fec,
  int vdim
)
{
  auto lor_gridfn = std::make_unique<mfem::ParGridFunction>( 
    new mfem::ParFiniteElementSpace(
      &lor_mesh,
      lor_fec.get(),
      vdim,
      mfem::Ordering::byNODES
    )
  );
  lor_gridfn->MakeOwner(lor_fec.release());
  return lor_gridfn;
}

SubmeshRedecompTransfer::SubmeshRedecompTransfer(
  mfem::ParFiniteElementSpace& submesh_fes,
  SubmeshLORTransfer* submesh_lor_xfer,
  redecomp::RedecompMesh& redecomp_mesh
)
: submesh_fes_ { submesh_fes },
  redecomp_fes_ { submesh_lor_xfer ?
    CreateRedecompFESpace(redecomp_mesh, *submesh_lor_xfer->GetLORGridFn().ParFESpace()) :
    CreateRedecompFESpace(redecomp_mesh, submesh_fes_) },
  submesh_lor_xfer_ { submesh_lor_xfer },
  redecomp_xfer_ { } // default (element transfer) constructor
{
  // make sure submesh_fes is a submesh and redecomp's parent is submesh_fes's
  // submesh
  SLIC_ERROR_ROOT_IF(
    !mfem::ParSubMesh::IsParSubMesh(submesh_fes_.GetParMesh()),
    "submesh_fes must be on a ParSubMesh."
  );
  SLIC_ERROR_ROOT_IF(
    !submesh_lor_xfer &&
      &redecomp_mesh.getParent() != submesh_fes_.GetParMesh(),
    "redecomp's parent must match the submesh_fes ParMesh."
  );
  SLIC_ERROR_ROOT_IF(
    submesh_lor_xfer &&
      &redecomp_mesh.getParent() != submesh_lor_xfer->GetLORGridFn().ParFESpace()->GetParMesh(),
    "redecomp's parent must match the submesh_fes ParMesh."
  );
}

void SubmeshRedecompTransfer::SubmeshToRedecomp(
  const mfem::ParGridFunction& submesh_src,
  mfem::GridFunction& redecomp_dst
) const
{
  auto src_ptr = &submesh_src;
  if (submesh_lor_xfer_)
  {
    submesh_lor_xfer_->GetLORGridFn() = 0.0;
    submesh_lor_xfer_->TransferToLORGridFn(submesh_src);
    src_ptr = &submesh_lor_xfer_->GetLORGridFn();
  }
  redecomp_xfer_.TransferToSerial(*src_ptr, redecomp_dst);
}

void SubmeshRedecompTransfer::RedecompToSubmesh(
  const mfem::GridFunction& redecomp_src,
  mfem::Vector& submesh_dst
) const
{
  auto dst_ptr = &submesh_dst;
  auto dst_fespace_ptr = &submesh_fes_;
  // first initialize LOR grid function (if using LOR)
  if (submesh_lor_xfer_)
  {
    submesh_lor_xfer_->GetLORVector() = 0.0;
    dst_ptr = &submesh_lor_xfer_->GetLORVector();
    dst_fespace_ptr = submesh_lor_xfer_->GetLORGridFn().ParFESpace();
  }
  // transfer data from redecomp mesh
  mfem::ParGridFunction dst_gridfn(dst_fespace_ptr, *dst_ptr);
  redecomp_xfer_.TransferToParallel(redecomp_src, dst_gridfn);

  // using redecomp, shared dof values are set equal (i.e. a ParGridFunction), but we want the sum of shared dof values
  // to equal the actual dof value when transferring dual fields (i.e. force and gap) back to the parallel mesh
  // following MFEMs convention.  set non-owned DOF values to zero.
  
  // P_I is the row index vector on the MFEM prolongation matrix. If there are no column entries for the row, then the
  // DOF is owned by another rank.
  auto P_I = dst_fespace_ptr->Dof_TrueDof_Matrix()->GetDiagMemoryI();
  HYPRE_Int tdof_ct {0};
  for (int i{0}; i < dst_fespace_ptr->GetVSize(); ++i)
  {
    if (P_I[i+1] != tdof_ct)
    {
      ++tdof_ct;
    }
    else
    {
      (*dst_ptr)[i] = 0.0;
    }
  }
  // if using LOR, transfer data from LOR mesh to submesh
  if (submesh_lor_xfer_)
  {
    submesh_lor_xfer_->TransferFromLORVector(submesh_dst);
  }
}

std::unique_ptr<mfem::FiniteElementSpace> SubmeshRedecompTransfer::CreateRedecompFESpace(
  redecomp::RedecompMesh& redecomp_mesh,
  mfem::ParFiniteElementSpace& submesh_fes
)
{
  return std::make_unique<mfem::FiniteElementSpace>(
    &redecomp_mesh,
    submesh_fes.FEColl(),
    submesh_fes.GetVDim(),
    mfem::Ordering::byNODES
  );
}

ParentRedecompTransfer::ParentRedecompTransfer(
  const mfem::ParFiniteElementSpace& parent_fes,
  mfem::ParGridFunction& submesh_gridfn,
  SubmeshLORTransfer* submesh_lor_xfer,
  redecomp::RedecompMesh& redecomp_mesh
)
: parent_fes_ { parent_fes },
  submesh_gridfn_ { submesh_gridfn },
  submesh_redecomp_xfer_ { 
    *submesh_gridfn_.ParFESpace(),
    submesh_lor_xfer, 
    redecomp_mesh
  }
{
  // Note: this is checked in the SubmeshRedecompTransfer constructor
  // SLIC_ERROR_ROOT_IF(
  //   !mfem::ParSubMesh::IsParSubMesh(submesh_gridfn_.ParFESpace()->GetParMesh()),
  //   "submesh_gridfn_ must be associated with an mfem::ParSubMesh."
  // );
  SLIC_ERROR_ROOT_IF(
    submesh_redecomp_xfer_.GetSubmesh().GetParent() != parent_fes_.GetParMesh(),
    "submesh_gridfn's parent mesh must match the parent_fes ParMesh."
  );
}

void ParentRedecompTransfer::ParentToRedecomp(
  const mfem::ParGridFunction& parent_src,
  mfem::GridFunction& redecomp_dst
) const
{
  submesh_gridfn_ = 0.0;
  submesh_redecomp_xfer_.GetSubmesh().Transfer(parent_src, submesh_gridfn_);
  submesh_redecomp_xfer_.SubmeshToRedecomp(submesh_gridfn_, redecomp_dst);
}

void ParentRedecompTransfer::RedecompToParent(
  const mfem::GridFunction& redecomp_src,
  mfem::Vector& parent_dst
) const
{
  submesh_gridfn_ = 0.0;
  submesh_redecomp_xfer_.RedecompToSubmesh(redecomp_src, submesh_gridfn_);
  // submesh transfer requires a grid function.  create one using parent_dst's data
  mfem::ParGridFunction parent_gridfn(&parent_fes_, parent_dst);
  submesh_redecomp_xfer_.GetSubmesh().Transfer(submesh_gridfn_, parent_gridfn);
}

ParentField::ParentField(
  const mfem::ParGridFunction& parent_gridfn
)
: parent_gridfn_ { parent_gridfn }
{}

void ParentField::SetParentGridFn(const mfem::ParGridFunction& parent_gridfn)
{
  parent_gridfn_ = parent_gridfn;
  update_data_.reset(nullptr);
}

void ParentField::UpdateField(ParentRedecompTransfer& parent_redecomp_xfer)
{
  update_data_ = std::make_unique<UpdateData>(parent_redecomp_xfer, parent_gridfn_);
}

std::vector<const RealT*> ParentField::GetRedecompFieldPtrs() const
{
  auto data_ptrs = std::vector<const RealT*>(3, nullptr);
  if (GetRedecompGridFn().FESpace()->GetNDofs() > 0)
  {
    for (size_t i{}; i < static_cast<size_t>(GetRedecompGridFn().FESpace()->GetVDim()); ++i)
    {
      data_ptrs[i] = &GetRedecompGridFn()(GetRedecompGridFn().FESpace()->DofToVDof(0, i));
    }
  }
  return data_ptrs;
}

std::vector<RealT*> ParentField::GetRedecompFieldPtrs(mfem::GridFunction& redecomp_gridfn)
{
  auto data_ptrs = std::vector<RealT*>(3, nullptr);
  if (redecomp_gridfn.FESpace()->GetNDofs() > 0)
  {
    for (size_t i{}; i < static_cast<size_t>(redecomp_gridfn.FESpace()->GetVDim()); ++i)
    {
      data_ptrs[i] = &redecomp_gridfn(redecomp_gridfn.FESpace()->DofToVDof(0, i));
    }
  }
  return data_ptrs;
}

ParentField::UpdateData& ParentField::GetUpdateData()
{
  SLIC_ERROR_ROOT_IF(
    update_data_ == nullptr,
    "UpdateField() must be called to generate UpdateData."
  );
  return *update_data_;
}

const ParentField::UpdateData& ParentField::GetUpdateData() const
{
  SLIC_ERROR_ROOT_IF(
    update_data_ == nullptr,
    "UpdateField() must be called to generate UpdateData."
  );
  return *update_data_;
}

ParentField::UpdateData::UpdateData(
  ParentRedecompTransfer& parent_redecomp_xfer,
  const mfem::ParGridFunction& parent_gridfn
)
: parent_redecomp_xfer_ { parent_redecomp_xfer },
  redecomp_gridfn_ { &parent_redecomp_xfer.GetRedecompFESpace() }
{
  redecomp_gridfn_ = 0.0;
  parent_redecomp_xfer_.ParentToRedecomp(parent_gridfn, redecomp_gridfn_);
}

PressureField::PressureField(
  const mfem::ParGridFunction& submesh_gridfn
)
: submesh_gridfn_ { submesh_gridfn }
{}

void PressureField::SetSubmeshField(const mfem::ParGridFunction& submesh_gridfn)
{
  submesh_gridfn_ = submesh_gridfn;
  update_data_.reset(nullptr);
}

void PressureField::UpdateField(SubmeshRedecompTransfer& submesh_redecomp_xfer)
{
  update_data_ = std::make_unique<UpdateData>(submesh_redecomp_xfer, submesh_gridfn_);
}

std::vector<const RealT*> PressureField::GetRedecompFieldPtrs() const
{
  auto data_ptrs = std::vector<const RealT*>(3, nullptr);
  if (GetRedecompGridFn().FESpace()->GetNDofs() > 0)
  {
    for (size_t i{}; i < static_cast<size_t>(GetRedecompGridFn().FESpace()->GetVDim()); ++i)
    {
      data_ptrs[i] = &GetRedecompGridFn()(GetRedecompGridFn().FESpace()->DofToVDof(0, i));
    }
  }
  return data_ptrs;
}

std::vector<RealT*> PressureField::GetRedecompFieldPtrs(mfem::GridFunction& redecomp_gridfn)
{
  auto data_ptrs = std::vector<RealT*>(3, nullptr);
  if (redecomp_gridfn.FESpace()->GetNDofs() > 0)
  {
    for (size_t i{}; i < static_cast<size_t>(redecomp_gridfn.FESpace()->GetVDim()); ++i)
    {
      data_ptrs[i] = &redecomp_gridfn(redecomp_gridfn.FESpace()->DofToVDof(0, i));
    }
  }
  return data_ptrs;
}

PressureField::UpdateData& PressureField::GetUpdateData()
{
  SLIC_ERROR_ROOT_IF(
    update_data_ == nullptr,
    "UpdateField() must be called to generate UpdateData."
  );
  return *update_data_;
}

const PressureField::UpdateData& PressureField::GetUpdateData() const
{
  SLIC_ERROR_ROOT_IF(
    update_data_ == nullptr,
    "UpdateField() must be called to generate UpdateData."
  );
  return *update_data_;
}

PressureField::UpdateData::UpdateData(
  SubmeshRedecompTransfer& submesh_redecomp_xfer,
  const mfem::ParGridFunction& submesh_gridfn
)
: submesh_redecomp_xfer_ { submesh_redecomp_xfer },
  redecomp_gridfn_ { &submesh_redecomp_xfer.GetRedecompFESpace() }
{
  redecomp_gridfn_ = 0.0;
  submesh_redecomp_xfer_.SubmeshToRedecomp(submesh_gridfn, redecomp_gridfn_);
}

MfemMeshData::MfemMeshData(
  IndexT mesh_id_1,
  IndexT mesh_id_2,
  const mfem::ParMesh& parent_mesh,
  const mfem::ParGridFunction& current_coords,
  std::set<int>&& attributes_1,
  std::set<int>&& attributes_2
)
: mesh_id_1_ { mesh_id_1 },
  mesh_id_2_ { mesh_id_2 },
  parent_mesh_ { parent_mesh },
  attributes_1_ { std::move(attributes_1) },
  attributes_2_ { std::move(attributes_2) },
  submesh_ { CreateSubmesh(parent_mesh_, attributes_1_, attributes_2_) },
  coords_ { current_coords },
  lor_factor_ { 0 }
{
  // make sure a grid function exists on the submesh
  submesh_.EnsureNodes();

  // create submesh grid function
  std::unique_ptr<mfem::FiniteElementCollection> submesh_fec {
    current_coords.ParFESpace()->FEColl()->Clone(
      current_coords.ParFESpace()->FEColl()->GetOrder()
    )
  };
  submesh_xfer_gridfn_.SetSpace(
    new mfem::ParFiniteElementSpace(
      &submesh_,
      submesh_fec.get(),
      current_coords.ParFESpace()->GetVDim(),
      mfem::Ordering::byNODES
    )
  );
  submesh_xfer_gridfn_.MakeOwner(submesh_fec.release());

  // build LOR submesh
  if (current_coords.FESpace()->FEColl()->GetOrder() > 1)
  {
    SetLORFactor(current_coords.FESpace()->FEColl()->GetOrder());
  }
}

void MfemMeshData::SetParentCoords(const mfem::ParGridFunction& current_coords)
{
  coords_.SetParentGridFn(current_coords);
}

void MfemMeshData::UpdateMfemMeshData()
{
  // update coordinates of submesh and LOR mesh
  auto submesh_nodes = dynamic_cast<mfem::ParGridFunction*>(submesh_.GetNodes());
  SLIC_ERROR_ROOT_IF(!submesh_nodes, "submesh_ Nodes is not a ParGridFunction.");
  submesh_.Transfer(coords_.GetParentGridFn(), *submesh_nodes);
  if (lor_mesh_.get())
  {
    auto lor_nodes = dynamic_cast<mfem::ParGridFunction*>(lor_mesh_->GetNodes());
    SLIC_ERROR_ROOT_IF(!lor_nodes, "lor_mesh_ Nodes is not a ParGridFunction.");
    submesh_lor_xfer_->SubmeshToLOR(*submesh_nodes, *lor_nodes);
  }
  update_data_ = std::make_unique<UpdateData>(
    submesh_,
    lor_mesh_.get(),
    *coords_.GetParentGridFn().ParFESpace(),
    submesh_xfer_gridfn_,
    submesh_lor_xfer_.get(),
    attributes_1_, 
    attributes_2_
  );
  coords_.UpdateField(update_data_->vector_xfer_);
  redecomp_response_.SetSpace(coords_.GetRedecompGridFn().FESpace());
  redecomp_response_ = 0.0;
  if (velocity_)
  {
    velocity_->UpdateField(update_data_->vector_xfer_);
  }
  if (elem_thickness_)
  {
    if (!material_modulus_)
    {
      SLIC_ERROR_ROOT("Kinematic element penalty requires material modulus information. "
                      "Call registerMfemMaterialModulus() to set this.");
    }
    redecomp::RedecompTransfer redecomp_xfer;
    // set element thickness on redecomp mesh
    redecomp_elem_thickness_ = std::make_unique<mfem::QuadratureFunction>(
      new mfem::QuadratureSpace(&GetRedecompMesh(), 0)
    );
    redecomp_elem_thickness_->SetOwnsSpace(true);
    *redecomp_elem_thickness_ = 0.0;
    redecomp_xfer.TransferToSerial(*elem_thickness_, *redecomp_elem_thickness_);
    // set element thickness on tribol mesh
    tribol_elem_thickness_1_ = std::make_unique<ArrayT<RealT>>(
      0, GetElemMap1().empty() ? 1 : GetElemMap1().size());
    for (auto redecomp_e : GetElemMap1())
    {
      mfem::Vector quad_val;
      redecomp_elem_thickness_->GetValues(redecomp_e, quad_val);
      tribol_elem_thickness_1_->push_back(quad_val[0]);
    }
    tribol_elem_thickness_2_ = std::make_unique<ArrayT<RealT>>(
      0, GetElemMap2().empty() ? 1 : GetElemMap2().size());
    for (auto redecomp_e : GetElemMap2())
    {
      mfem::Vector quad_val;
      redecomp_elem_thickness_->GetValues(redecomp_e, quad_val);
      tribol_elem_thickness_2_->push_back(quad_val[0]);
    }
    // set material modulus on redecomp mesh
    redecomp_material_modulus_ = std::make_unique<mfem::QuadratureFunction>(
      new mfem::QuadratureSpace(&GetRedecompMesh(), 0)
    );
    redecomp_material_modulus_->SetOwnsSpace(true);
    *redecomp_material_modulus_ = 0.0;
    redecomp_xfer.TransferToSerial(*material_modulus_, *redecomp_material_modulus_);
    // set material modulus on tribol mesh
    tribol_material_modulus_1_ = std::make_unique<ArrayT<RealT>>(
      0, GetElemMap1().empty() ? 1 : GetElemMap1().size());
    for (auto redecomp_e : GetElemMap1())
    {
      mfem::Vector quad_val;
      redecomp_material_modulus_->GetValues(redecomp_e, quad_val);
      tribol_material_modulus_1_->push_back(quad_val[0]);
    }
    tribol_material_modulus_2_ = std::make_unique<ArrayT<RealT>>(
      0, GetElemMap2().empty() ? 1 : GetElemMap2().size());
    for (auto redecomp_e : GetElemMap2())
    {
      mfem::Vector quad_val;
      redecomp_material_modulus_->GetValues(redecomp_e, quad_val);
      tribol_material_modulus_2_->push_back(quad_val[0]);
    }
  }
}

void MfemMeshData::GetParentResponse(mfem::Vector& r) const
{
  GetParentRedecompTransfer().RedecompToParent(redecomp_response_, r);
}

void MfemMeshData::SetParentVelocity(const mfem::ParGridFunction& velocity)
{
  if (velocity_)
  {
    velocity_->SetParentGridFn(velocity);
  }
  else
  {
    velocity_ = std::make_unique<ParentField>(velocity);
  }
}

void MfemMeshData::ClearAllPenaltyData()
{
  ClearRatePenaltyData();
  kinematic_constant_penalty_1_.reset(nullptr);
  kinematic_constant_penalty_2_.reset(nullptr);
  kinematic_penalty_scale_1_.reset(nullptr);
  kinematic_penalty_scale_2_.reset(nullptr);
  elem_thickness_.reset(nullptr);
  redecomp_elem_thickness_.reset(nullptr);
  tribol_elem_thickness_1_.reset(nullptr);
  tribol_elem_thickness_2_.reset(nullptr);
  material_modulus_.reset(nullptr);
  redecomp_material_modulus_.reset(nullptr);
  tribol_material_modulus_1_.reset(nullptr);
  tribol_material_modulus_2_.reset(nullptr);
}

void MfemMeshData::ClearRatePenaltyData()
{
  rate_constant_penalty_1_.reset(nullptr);
  rate_constant_penalty_2_.reset(nullptr);
  rate_percent_ratio_1_.reset(nullptr);
  rate_percent_ratio_2_.reset(nullptr);
}

void MfemMeshData::SetLORFactor(int lor_factor)
{
  if (lor_factor <= 1)
  {
    SLIC_WARNING_ROOT("lor_factor must be an integer > 1.  LOR factor not changed.");
    return;
  }
  if (coords_.GetParentGridFn().FESpace()->FEColl()->GetOrder() <= 1)
  {
    SLIC_WARNING_ROOT("lor_factor is only applicable to higher order geometry.  "
      "LOR factor not changed.");
    return;
  }
  lor_factor_ = lor_factor;
  // note: calls ParMesh's move ctor
  lor_mesh_ = std::make_unique<mfem::ParMesh>(mfem::ParMesh::MakeRefined(
    submesh_, lor_factor, mfem::BasisType::ClosedUniform
  ));
  lor_mesh_->EnsureNodes();
  submesh_lor_xfer_ = std::make_unique<SubmeshLORTransfer>(
    *submesh_xfer_gridfn_.ParFESpace(),
    *lor_mesh_
  );
}

void MfemMeshData::ComputeElementThicknesses()
{
  auto submesh_thickness = std::make_unique<mfem::QuadratureFunction>(
    new mfem::QuadratureSpace(&submesh_, 0)
  );
  submesh_thickness->SetOwnsSpace(true);
  // All the elements in the submesh are on the contact surface. The algorithm
  // works as follows:
  // 1) For each submesh element, find the corresponding parent volume element
  // 2) Compute the thickness of the parent volume element (det J at element
  //    centroid)
  // 3) If no LOR mesh, store this on a quadrature function on the submesh
  // 4) If there is an LOR mesh, use the CoarseFineTransformation to find the
  //    LOR elements linked to the HO mesh and store the thickness of the HO
  //    element on all of its linked LOR elements.
  for (int submesh_e{0}; submesh_e < submesh_.GetNE(); ++submesh_e)
  {
    // Step 1
    auto parent_bdr_e = submesh_.GetParentElementIDMap()[submesh_e];
    auto& parent_mesh = const_cast<mfem::ParMesh&>(parent_mesh_);
    auto& face_el_tr = *parent_mesh.GetBdrFaceTransformations(parent_bdr_e);
    auto mask = face_el_tr.GetConfigurationMask();
    auto parent_e = (mask & mfem::FaceElementTransformations::HAVE_ELEM1) ?
      face_el_tr.Elem1No :
      face_el_tr.Elem2No;
    
    // Step 2 
    // normal = (dx/dxi x dx/deta) / || dx/dxi x dx/deta || on parent volume boundary element centroid
    auto& parent_fes = *coords_.GetParentGridFn().ParFESpace();
    mfem::Array<int> be_dofs;
    parent_fes.GetBdrElementDofs(parent_bdr_e, be_dofs);
    mfem::DenseMatrix elem_coords(parent_mesh_.Dimension(), be_dofs.Size());
    for (int d{0}; d < parent_mesh_.Dimension(); ++d)
    {
      mfem::Array<int> be_vdofs(be_dofs);
      parent_fes.DofsToVDofs(d, be_vdofs);
      mfem::Vector elemvect(be_dofs.Size());
      coords_.GetParentGridFn().GetSubVector(be_vdofs, elemvect);
      elem_coords.SetRow(d, elemvect);
    }
    auto& be = *parent_fes.GetBE(parent_bdr_e);
    // create an integration point at the element centroid
    mfem::IntegrationPoint ip;
    ip.Init(0);
    mfem::DenseMatrix dshape(be_dofs.Size(), submesh_.Dimension());
    // calculate shape function derivatives at the surface element centroid
    be.CalcDShape(ip, dshape);
    mfem::DenseMatrix dxdxi_mat(parent_mesh_.Dimension(), submesh_.Dimension());
    mfem::Mult(elem_coords, dshape, dxdxi_mat);
    mfem::Vector norm(parent_mesh_.Dimension());
    mfem::CalcOrtho(dxdxi_mat, norm);
    double h = parent_mesh.GetElementSize(parent_e, norm);

    // Step 3
    mfem::Vector quad_val;
    submesh_thickness->GetValues(submesh_e, quad_val);
    quad_val[0] = h;
  }

  // Step 4
  if (GetLORMesh())
  {
    elem_thickness_ = std::make_unique<mfem::QuadratureFunction>(
      new mfem::QuadratureSpace(GetLORMesh(), 0)
    );
    elem_thickness_->SetOwnsSpace(true);
    for (int lor_e{0}; lor_e < GetLORMesh()->GetNE(); ++lor_e)
    {
      auto submesh_e = GetLORMesh()->GetRefinementTransforms()
        .embeddings[lor_e].parent;
      mfem::Vector submesh_val;
      submesh_thickness->GetValues(submesh_e, submesh_val);
      mfem::Vector lor_val;
      elem_thickness_->GetValues(lor_e, lor_val);
      lor_val[0] = submesh_val[0];
    }
  }
  else
  {
    elem_thickness_ = std::move(submesh_thickness);
  }
}

void MfemMeshData::SetMaterialModulus(mfem::Coefficient& modulus_field)
{
  material_modulus_ = std::make_unique<mfem::QuadratureFunction>(
    new mfem::QuadratureSpace(GetLORMesh() ? GetLORMesh() : &submesh_, 0)
  );
  material_modulus_->SetOwnsSpace(true);
  // TODO: why isn't Project() const?
  modulus_field.Project(*material_modulus_);
}

MfemMeshData::UpdateData::UpdateData(
  mfem::ParSubMesh& submesh,
  mfem::ParMesh* lor_mesh,
  const mfem::ParFiniteElementSpace& parent_fes,
  mfem::ParGridFunction& submesh_gridfn,
  SubmeshLORTransfer* submesh_lor_xfer,
  const std::set<int>& attributes_1,
  const std::set<int>& attributes_2
)
: redecomp_mesh_ { lor_mesh ? 
    redecomp::RedecompMesh(*lor_mesh) :
    redecomp::RedecompMesh(submesh)
  },
  vector_xfer_ { parent_fes, submesh_gridfn, submesh_lor_xfer, redecomp_mesh_ }
{
  // set element type based on redecomp mesh
  SetElementData();
  // updates the connectivity of the tribol surface mesh
  UpdateConnectivity(attributes_1, attributes_2);
}

void MfemMeshData::UpdateData::UpdateConnectivity(
  const std::set<int>& attributes_1,
  const std::set<int>& attributes_2
)
{
  conn_1_.reserve(redecomp_mesh_.GetNE() * num_verts_per_elem_);
  conn_2_.reserve(redecomp_mesh_.GetNE() * num_verts_per_elem_);
  elem_map_1_.reserve(static_cast<size_t>(redecomp_mesh_.GetNE()));
  elem_map_2_.reserve(static_cast<size_t>(redecomp_mesh_.GetNE()));
  for (int e{}; e < redecomp_mesh_.GetNE(); ++e)
  {
    auto elem_attrib = redecomp_mesh_.GetAttribute(e);
    auto elem_conn = mfem::Array<int>();
    redecomp_mesh_.GetElementVertices(e, elem_conn);
    for (auto attribute_1 : attributes_1)
    {
      if (attribute_1 == elem_attrib)
      {
        elem_map_1_.push_back(e);
        conn_1_.resize(elem_map_1_.size(), num_verts_per_elem_);
        for (int v{}; v < num_verts_per_elem_; ++v)
        {
          conn_1_(elem_map_1_.size() - 1, v) = elem_conn[v];
        }
        break;
      }
    }
    for (auto attribute_2 : attributes_2)
    {
      if (attribute_2 == elem_attrib)
      {
        elem_map_2_.push_back(e);
        conn_2_.resize(elem_map_2_.size(), num_verts_per_elem_);
        for (int v{}; v < num_verts_per_elem_; ++v)
        {
          conn_2_(elem_map_2_.size() - 1, v) = elem_conn[v];
        }
        break;
      }
    }
  }
  conn_1_.shrink();
  conn_2_.shrink();
  elem_map_1_.shrink_to_fit();
  elem_map_2_.shrink_to_fit();
}

MfemMeshData::UpdateData& MfemMeshData::GetUpdateData()
{
  SLIC_ERROR_ROOT_IF(
    update_data_ == nullptr,
    "UpdateField() must be called to generate UpdateData."
  );
  return *update_data_;
}

const MfemMeshData::UpdateData& MfemMeshData::GetUpdateData() const
{
  SLIC_ERROR_ROOT_IF(
    update_data_ == nullptr,
    "UpdateField() must be called to generate UpdateData."
  );
  return *update_data_;
}

mfem::ParSubMesh MfemMeshData::CreateSubmesh(
  const mfem::ParMesh& parent_mesh,
  const std::set<int>& attributes_1,
  const std::set<int>& attributes_2
)
{
  // TODO: Create PR for mfem::ParSubMesh::CreateFromBoundary taking a const
  // reference to attributes. Then we can construct submesh_ in the initializer
  // list without this function (because CreateFromBoundary will be willing to
  // take an rvalue for attributes)
  auto attributes_array 
    = arrayFromSet(mergeContainers(attributes_1, attributes_2));
  return mfem::ParSubMesh::CreateFromBoundary(
    parent_mesh,
    attributes_array
  );
}

void MfemMeshData::UpdateData::SetElementData()
{
  if (redecomp_mesh_.GetNE() > 0)
  {
    auto element_type = redecomp_mesh_.GetElementType(0);

    switch (element_type) 
    {
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
        SLIC_ERROR_ROOT("Unsupported element type!");
        break;

      default:
        SLIC_ERROR_ROOT("Unknown element type!");
        break;
    }

    num_verts_per_elem_ = mfem::Geometry::NumVerts[element_type];
  }
  else
  {
    // just put something here so Tribol will not give a warning for zero element meshes
    elem_type_ = LINEAR_EDGE;
    num_verts_per_elem_ = 2;
  }
}

MfemSubmeshData::MfemSubmeshData(
  mfem::ParSubMesh& submesh,
  mfem::ParMesh* lor_mesh,
  std::unique_ptr<mfem::FiniteElementCollection> pressure_fec,
  int pressure_vdim
)
: submesh_pressure_ {
    new mfem::ParFiniteElementSpace(
      &submesh,
      pressure_fec.get(),
      pressure_vdim
    )
  },
  pressure_ { submesh_pressure_ },
  submesh_lor_xfer_ { lor_mesh ?
    std::make_unique<SubmeshLORTransfer>(
      *submesh_pressure_.ParFESpace(), 
      *lor_mesh
    ) :
    nullptr
  }
{
  submesh_pressure_.MakeOwner(pressure_fec.release());
  submesh_pressure_ = 0.0;
}

void MfemSubmeshData::UpdateMfemSubmeshData(redecomp::RedecompMesh& redecomp_mesh)
{
  update_data_ = std::make_unique<UpdateData>(
    *submesh_pressure_.ParFESpace(),
    submesh_lor_xfer_.get(),
    redecomp_mesh
  );
  pressure_.UpdateField(update_data_->pressure_xfer_);
  redecomp_gap_.SetSpace(pressure_.GetRedecompGridFn().FESpace());
  redecomp_gap_ = 0.0;
}

void MfemSubmeshData::GetSubmeshGap(mfem::Vector& g) const
{
  g.SetSize(submesh_pressure_.ParFESpace()->GetVSize());
  g = 0.0;
  GetPressureTransfer().RedecompToSubmesh(redecomp_gap_, g);
}

MfemSubmeshData::UpdateData::UpdateData(
  mfem::ParFiniteElementSpace& submesh_fes,
  SubmeshLORTransfer* submesh_lor_xfer,
  redecomp::RedecompMesh& redecomp_mesh
)
: pressure_xfer_ { submesh_fes, submesh_lor_xfer, redecomp_mesh }
{}

MfemSubmeshData::UpdateData& MfemSubmeshData::GetUpdateData()
{
  SLIC_ERROR_ROOT_IF(
    update_data_ == nullptr,
    "UpdateField() must be called to generate UpdateData."
  );
  return *update_data_;
}

const MfemSubmeshData::UpdateData& MfemSubmeshData::GetUpdateData() const
{
  SLIC_ERROR_ROOT_IF(
    update_data_ == nullptr,
    "UpdateField() must be called to generate UpdateData."
  );
  return *update_data_;
}

MfemJacobianData::MfemJacobianData(
  const MfemMeshData& parent_data,
  const MfemSubmeshData& submesh_data,
  ContactMethod contact_method
)
: parent_data_ { parent_data },
  submesh_data_ { submesh_data },
  block_offsets_ { 3 }
{
  SLIC_ERROR_ROOT_IF(
    parent_data.GetParentCoords().ParFESpace()->FEColl()->GetOrder() > 1,
    "Higher order meshes not yet supported for Jacobian matrices."
  );

  // NOTE: Looks like GetFrom() should be const in MFEM
  mfem::SubMeshUtils::BuildVdofToVdofMap(
    parent_data_.GetSubmeshFESpace(),
    *parent_data_.GetParentCoords().FESpace(),
    const_cast<mfem::ParSubMesh&>(parent_data_.GetSubmesh()).GetFrom(),
    parent_data_.GetSubmesh().GetParentElementIDMap(),
    submesh2parent_vdof_list_
  );

  int my_rank {0};
  MPI_Comm_rank(TRIBOL_COMM_WORLD, &my_rank);
  auto offset_idx = HYPRE_AssumedPartitionCheck() ? 0 : my_rank;
  auto dof_offset = parent_data_.GetParentCoords().ParFESpace()->
    GetDofOffsets()[offset_idx];
  for (auto& vdof : submesh2parent_vdof_list_)
  {
    vdof = vdof + dof_offset;
  }

  auto disp_size = parent_data_.GetParentCoords().ParFESpace()
    ->GetTrueVSize();
  auto lm_size = submesh_data_.GetSubmeshPressure().ParFESpace()
    ->GetTrueVSize();
  block_offsets_[0] = 0;
  block_offsets_[1] = disp_size;
  block_offsets_[2] = disp_size + lm_size;

  // Rows/columns of pressure/gap DOFs only on the mortar surface need to be
  // eliminated from the Jacobian when using single mortar. The code in this
  // block creates a list of the true DOFs only on the mortar surface.
  if (contact_method == SINGLE_MORTAR)
  {
    // Get submesh
    auto& submesh_fe_space = submesh_data_.GetSubmeshFESpace();
    auto& submesh = parent_data_.GetSubmesh();
    // Create marker of attributes for faster querying
    mfem::Array<int> attr_marker(submesh.attributes.Max());
    attr_marker = 0;
    for (auto nonmortar_attr : parent_data_.GetBoundaryAttribs2())
    {
      attr_marker[nonmortar_attr - 1] = 1;
    }
    // Create marker of dofs only on mortar surface
    mfem::Array<int> mortar_dof_marker(submesh_fe_space.GetVSize());
    mortar_dof_marker = 1;
    for (int e{0}; e < submesh.GetNE(); ++e)
    {
      if (attr_marker[submesh_fe_space.GetAttribute(e) - 1])
      {
        mfem::Array<int> vdofs;
        submesh_fe_space.GetElementVDofs(e, vdofs);
        for (int d{0}; d < vdofs.Size(); ++d)
        {
          int k = vdofs[d];
          if (k < 0) { k = -1 - k; }
          mortar_dof_marker[k] = 0;
        }
      }
    }
    // Convert marker of dofs to marker of tdofs
    mfem::Array<int> mortar_tdof_marker(submesh_fe_space.GetTrueVSize());
    submesh_fe_space.GetRestrictionMatrix()->BooleanMult(mortar_dof_marker, mortar_tdof_marker);
    // Convert markers of tdofs only on mortar surface to a list
    mfem::FiniteElementSpace::MarkerToList(mortar_tdof_marker, mortar_tdof_list_);
  }
}

void MfemJacobianData::UpdateJacobianXfer()
{
  update_data_ = std::make_unique<UpdateData>(parent_data_, submesh_data_);
}

std::unique_ptr<mfem::BlockOperator> MfemJacobianData::GetMfemBlockJacobian(
  const MethodData* method_data
) const
{
  // 0 = displacement DOFs, 1 = lagrange multiplier DOFs
  // (0,0) block is empty (for now using SINGLE_MORTAR with approximate tangent)
  // (1,1) block is a diagonal matrix with ones on the diagonal of submesh nodes without a Lagrange multiplier DOF
  // (0,1) and (1,0) are symmetric (for now using SINGLE_MORTAR with approximate tangent)
  const auto& elem_map_1 = parent_data_.GetElemMap1();
  const auto& elem_map_2 = parent_data_.GetElemMap2();
  // empty data structures are needed even when no meshes are on rank since TransferToParallelSparse() needs to be
  // called on all ranks (even those without data)
  auto mortar_elems = ArrayT<int>(0, 0);
  auto nonmortar_elems = ArrayT<int>(0, 0);
  auto lm_elems = ArrayT<int>(0, 0);
  auto elem_J_1_ptr = std::make_unique<ArrayT<mfem::DenseMatrix>>(0, 0);
  auto elem_J_2_ptr = std::make_unique<ArrayT<mfem::DenseMatrix>>(0, 0);
  const ArrayT<mfem::DenseMatrix>* elem_J_1 = elem_J_1_ptr.get();
  const ArrayT<mfem::DenseMatrix>* elem_J_2 = elem_J_2_ptr.get();
  // this means both of the meshes exist
  if (method_data != nullptr && !elem_map_1.empty() && !elem_map_2.empty())
  {
    mortar_elems = method_data
      ->getBlockJElementIds()[static_cast<int>(BlockSpace::MORTAR)];
    for (auto& mortar_elem : mortar_elems)
    {
      mortar_elem = elem_map_1[static_cast<size_t>(mortar_elem)];
    }
    nonmortar_elems = method_data
      ->getBlockJElementIds()[static_cast<int>(BlockSpace::NONMORTAR)];
    for (auto& nonmortar_elem : nonmortar_elems)
    {
      nonmortar_elem = elem_map_2[static_cast<size_t>(nonmortar_elem)];
    }
    lm_elems = method_data
      ->getBlockJElementIds()[static_cast<int>(BlockSpace::LAGRANGE_MULTIPLIER)];
    for (auto& lm_elem : lm_elems)
    {
      lm_elem = elem_map_2[static_cast<size_t>(lm_elem)];
    }
    // get (1,0) block
    elem_J_1 = &method_data->getBlockJ()(
      static_cast<int>(BlockSpace::LAGRANGE_MULTIPLIER),
      static_cast<int>(BlockSpace::MORTAR)
    );
    elem_J_2 = &method_data->getBlockJ()(
      static_cast<int>(BlockSpace::LAGRANGE_MULTIPLIER),
      static_cast<int>(BlockSpace::NONMORTAR)
    );
  }
  // move to submesh level
  auto submesh_J = GetUpdateData().submesh_redecomp_xfer_->TransferToParallelSparse(
    lm_elems, 
    mortar_elems, 
    *elem_J_1
  );
  submesh_J += GetUpdateData().submesh_redecomp_xfer_->TransferToParallelSparse(
    lm_elems, 
    nonmortar_elems, 
    *elem_J_2
  );
  submesh_J.Finalize();

  // transform J values from submesh to parent mesh
  auto J = submesh_J.GetJ();
  auto submesh_vector_fes = parent_data_.GetSubmeshFESpace();
  auto mpi = redecomp::MPIUtility(submesh_vector_fes.GetComm());
  auto submesh_dof_offsets = ArrayT<int>(mpi.NRanks() + 1, mpi.NRanks() + 1);
  // we need the dof offsets of each rank.  check if mfem stores this or if we
  // need to create it.
  if (HYPRE_AssumedPartitionCheck())
  {
    submesh_dof_offsets[mpi.MyRank()+1] = submesh_vector_fes.GetDofOffsets()[1];
    mpi.Allreduce(&submesh_dof_offsets, MPI_SUM);
  }
  else
  {
    for (int i{0}; i < mpi.NRanks(); ++i)
    {
        submesh_dof_offsets[i] = submesh_vector_fes.GetDofOffsets()[i];
    }
    
  }
  // the submesh to parent vdof map only exists for vdofs on rank, so J values
  // not on rank will need to be transferred to the rank that the vdof exists on
  // to query the map. the steps are laid out below.
  
  // step 1) query J values on rank for their parent vdof and package J values
  // not on rank to send
  auto send_J_by_rank = redecomp::MPIArray<int>(&mpi);
  auto J_idx = redecomp::MPIArray<int>(&mpi);
  auto est_num_J = submesh_J.NumNonZeroElems() / mpi.NRanks();
  for (int r{}; r < mpi.NRanks(); ++r)
  {
    if (r == mpi.MyRank())
    {
        send_J_by_rank[r].shrink();
        J_idx[r].shrink();
    }
    else
    {
        send_J_by_rank[r].reserve(est_num_J);
        J_idx[r].reserve(est_num_J);
    }
    
  }
  for (int j{}; j < submesh_J.NumNonZeroElems(); ++j)
  {
    if (J[j] >= submesh_dof_offsets[mpi.MyRank()] 
        && J[j] < submesh_dof_offsets[mpi.MyRank() + 1])
    {
        J[j] = submesh2parent_vdof_list_[J[j] - submesh_dof_offsets[mpi.MyRank()]];
    }
    else
    {
        for (int r{}; r < mpi.NRanks(); ++r)
        {
        if (J[j] >= submesh_dof_offsets[r] && J[j] < submesh_dof_offsets[r + 1])
        {
          send_J_by_rank[r].push_back(J[j] - submesh_dof_offsets[r]);
          J_idx[r].push_back(j);
          break;
        }
        }
    }
  }
  // step 2) sends the J values to the ranks that own them
  auto recv_J_by_rank = redecomp::MPIArray<int>(&mpi);
  recv_J_by_rank.SendRecvArrayEach(send_J_by_rank);
  // step 3) query the on-rank map to recover J values
  for (int r{}; r < mpi.NRanks(); ++r)
  {
    for (auto& recv_J : recv_J_by_rank[r])
    {
        recv_J = submesh2parent_vdof_list_[recv_J];
    }
  }
  // step 4) send the updated parent J values back and update the J vector
  send_J_by_rank.SendRecvArrayEach(recv_J_by_rank);
  for (int r{}; r < mpi.NRanks(); ++r)
  {
    for (int j{}; j < send_J_by_rank[r].size(); ++j)
    {
        J[J_idx[r][j]] = send_J_by_rank[r][j];
    }
  }

  // create block operator
  auto block_J = std::make_unique<mfem::BlockOperator>(block_offsets_);
  block_J->owns_blocks = 1;

  // fill block operator
  auto& submesh_fes = submesh_data_.GetSubmeshFESpace();
  auto& parent_trial_fes = *parent_data_.GetParentCoords().ParFESpace();
  auto J_full = std::make_unique<mfem::HypreParMatrix>(
    mpi.MPIComm(), submesh_fes.GetVSize(), 
    submesh_fes.GlobalVSize(), parent_trial_fes.GlobalVSize(),
    submesh_J.GetI(), submesh_J.GetJ(), submesh_J.GetData(),
    submesh_fes.GetDofOffsets(), parent_trial_fes.GetDofOffsets()
  );
  auto J_true = std::unique_ptr<mfem::HypreParMatrix>(mfem::RAP(
    submesh_fes.Dof_TrueDof_Matrix(),
    J_full.get(),
    parent_trial_fes.Dof_TrueDof_Matrix()
  ));
  
  // Create ones on diagonal of eliminated mortar tdofs (CSR sparse matrix -> HypreParMatrix)
  // I vector
  mfem::Array<int> rows(submesh_fes.GetTrueVSize() + 1);
  rows = 0;
  auto mortar_tdofs_ct = 0;
  for (int i{0}; i < submesh_fes.GetTrueVSize(); ++i)
  {
    if (mortar_tdofs_ct < mortar_tdof_list_.Size() && mortar_tdof_list_[mortar_tdofs_ct] == i)
    {
      ++mortar_tdofs_ct;
    }
    rows[i + 1] = mortar_tdofs_ct;
  }
  // J vector
  mfem::Array<int> mortar_tdofs(mortar_tdof_list_);
  // data vector
  mfem::Vector ones(mortar_tdofs_ct);
  ones = 1.0;
  mfem::SparseMatrix inactive_sm(
    rows.GetData(), mortar_tdofs.GetData(), ones.GetData(),
    submesh_fes.GetTrueVSize(), submesh_fes.GetTrueVSize(),
    false, false, true
  );
  auto inactive_hpm = std::make_unique<mfem::HypreParMatrix>(
    J_true->GetComm(), J_true->GetGlobalNumRows(), J_true->GetRowStarts(), &inactive_sm
  );
  // Have the mfem::HypreParMatrix manage the data pointers
  rows.GetMemory().SetHostPtrOwner(false);
  mortar_tdofs.GetMemory().SetHostPtrOwner(false);
  ones.GetMemory().SetHostPtrOwner(false);
  inactive_sm.SetDataOwner(false);
  inactive_hpm->SetOwnerFlags(3, 3, 1);

  block_J->SetBlock(0, 1, J_true->Transpose());
  block_J->SetBlock(1, 0, J_true.release());
  block_J->SetBlock(1, 1, inactive_hpm.release());

  return block_J;
}

MfemJacobianData::UpdateData::UpdateData(
  const MfemMeshData& parent_data,
  const MfemSubmeshData& submesh_data
)
{
  auto dual_submesh_fes = &submesh_data.GetSubmeshFESpace();
  auto primal_submesh_fes = &parent_data.GetSubmeshFESpace();
  if (parent_data.GetLORMesh())
  {
    dual_submesh_fes = submesh_data.GetLORMeshFESpace();
    primal_submesh_fes = parent_data.GetLORMeshFESpace();
  }
  // create a matrix transfer operator for moving data from redecomp to the
  // submesh
  submesh_redecomp_xfer_ = std::make_unique<redecomp::MatrixTransfer>(
    *dual_submesh_fes,
    *primal_submesh_fes,
    *submesh_data.GetRedecompGap().FESpace(),
    *parent_data.GetRedecompResponse().FESpace()
  );
}

MfemJacobianData::UpdateData& MfemJacobianData::GetUpdateData()
{
  SLIC_ERROR_ROOT_IF(
    update_data_ == nullptr,
    "UpdateField() must be called to generate UpdateData."
  );
  return *update_data_;
}

const MfemJacobianData::UpdateData& MfemJacobianData::GetUpdateData() const
{
  SLIC_ERROR_ROOT_IF(
    update_data_ == nullptr,
    "UpdateField() must be called to generate UpdateData."
  );
  return *update_data_;
}

} // end tribol namespace

#endif /* BUILD_REDECOMP */
