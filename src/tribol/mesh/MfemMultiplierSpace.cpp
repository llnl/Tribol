// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MfemMultiplierSpace.hpp"

#ifdef BUILD_REDECOMP

#include <map>
#include <utility>

#include "axom/slic.hpp"

namespace tribol {

namespace {

mfem::Array<int> BuildAttributeArray( const std::set<int>& attributes )
{
  mfem::Array<int> attribute_array( static_cast<int>( attributes.size() ) );
  int index = 0;
  for ( int attribute : attributes ) attribute_array[index++] = attribute;
  return attribute_array;
}

}  // namespace

MfemMultiplierSpace::MfemMultiplierSpace( const MfemContactSurface& dual_surface )
{
  auto attributes = BuildAttributeArray( dual_surface.BoundaryAttributes() );
  submesh_ =
      std::make_unique<mfem::ParSubMesh>( mfem::ParSubMesh::CreateFromBoundary( dual_surface.Mesh(), attributes ) );
  const int order = dual_surface.Coordinates().ParFESpace()->FEColl()->GetOrder();
  const auto* source_collection =
      dynamic_cast<const mfem::H1_FECollection*>( dual_surface.Coordinates().ParFESpace()->FEColl() );
  SLIC_ERROR_ROOT_IF( source_collection == nullptr,
                      "The multiplier surface requires a conforming nodal H1 coordinate space." );
  collection_ =
      std::make_unique<mfem::H1_FECollection>( order, submesh_->Dimension(), source_collection->GetBasisType() );
  space_ = std::make_unique<mfem::ParFiniteElementSpace>( submesh_.get(), collection_.get() );
  pressure_true_dofs_.SetSize( space_->GetTrueVSize() );
  gap_true_dofs_.SetSize( space_->GetTrueVSize() );
  pressure_true_dofs_ = 0.0;
  gap_true_dofs_ = 0.0;
}

void MfemMultiplierSpace::BuildPatchOperator( mfem::Array<int> selected_parent_boundary_elements,
                                              std::vector<mfem::Array<int>> elements_by_destination,
                                              std::vector<redecomp::VirtualFaceSpecification> destination_faces,
                                              mfem::Array<int> occurrence_to_node, int num_patch_nodes )
{
  std::map<int, int> submesh_element_for_parent;
  const auto& parent_elements = submesh_->GetParentElementIDMap();
  for ( int submesh_element = 0; submesh_element < parent_elements.Size(); ++submesh_element ) {
    submesh_element_for_parent[parent_elements[submesh_element]] = submesh_element;
  }
  mfem::Array<int> selected_submesh_elements( selected_parent_boundary_elements.Size() );
  for ( int selected = 0; selected < selected_parent_boundary_elements.Size(); ++selected ) {
    const auto submesh_element = submesh_element_for_parent.find( selected_parent_boundary_elements[selected] );
    SLIC_ERROR_ROOT_IF( submesh_element == submesh_element_for_parent.end(),
                        "The multiplier submesh is missing a requested dual-surface element." );
    selected_submesh_elements[selected] = submesh_element->second;
  }
  patch_operator_ = std::make_unique<redecomp::ElementPatchOperator>(
      *space_, std::move( selected_submesh_elements ), std::move( elements_by_destination ),
      std::move( destination_faces ), std::move( occurrence_to_node ), num_patch_nodes );
  assembled_patch_operator_.reset();
  patch_pressure_.SetSize( patch_operator_->Height() );
  patch_gap_.SetSize( patch_operator_->Height() );
  patch_pressure_ = 0.0;
  patch_gap_ = 0.0;
}

void MfemMultiplierSpace::GatherPressure()
{
  SLIC_ERROR_ROOT_IF( !patch_operator_, "The multiplier patch operator has not been built." );
  patch_operator_->Mult( pressure_true_dofs_, patch_pressure_ );
  patch_gap_ = 0.0;
}

void MfemMultiplierSpace::ReturnGap()
{
  SLIC_ERROR_ROOT_IF( !patch_operator_, "The multiplier patch operator has not been built." );
  patch_operator_->MultTranspose( patch_gap_, gap_true_dofs_ );
}

const shared::ParSparseMat& MfemMultiplierSpace::AssembledPatchOperator()
{
  SLIC_ERROR_ROOT_IF( !patch_operator_, "The multiplier patch operator has not been built." );
  if ( !assembled_patch_operator_ ) {
    assembled_patch_operator_ = std::make_unique<shared::ParSparseMat>( patch_operator_->Assemble() );
  }
  return *assembled_patch_operator_;
}

}  // namespace tribol

#endif  // BUILD_REDECOMP
