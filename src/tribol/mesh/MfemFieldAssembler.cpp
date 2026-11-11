// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MfemFieldAssembler.hpp"
#ifdef BUILD_REDECOMP
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tribol {
namespace {
/**
 * @brief Check a nonnegative array-size product before allocation.
 * @param count Number of entries or blocks.
 * @param size Values per entry or block.
 * @return Product representable by IndexT.
 * @throws std::length_error The product exceeds the supported index range.
 */
IndexT checkedProduct( IndexT count, IndexT size )
{
  if ( size != 0 && count > std::numeric_limits<IndexT>::max() / size ) {
    throw std::length_error( "Scalar field assembly storage exceeds the supported index range." );
  }
  return count * size;
}
}  // namespace

MfemFieldAssembler::MfemFieldAssembler( const MfemMeshData& mesh_data, const MfemSubmeshData& submesh_data,
                                        const MfemJacobianData& jacobian_data, MeshData::Viewer mesh1,
                                        MeshData::Viewer mesh2, IndexT field_count, IndexT pair_capacity )
    : mesh_data_( mesh_data ),
      submesh_data_( submesh_data ),
      jacobian_data_( jacobian_data ),
      connectivity_{ mesh1.getConnectivity(), mesh2.getConnectivity() },
      dimension_( mesh1.spatialDimension() ),
      nodes1_( mesh1.numberOfNodesPerElement() )
{
  const auto& field_fes = submesh_data.GetRedecompFESpace();
  if ( field_count <= 0 || pair_capacity < 0 || dimension_ <= 0 || mesh2.spatialDimension() != dimension_ ||
       mesh_data.GetRedecompResponse().FESpace()->GetVDim() != dimension_ || nodes1_ <= 0 ||
       mesh2.numberOfNodesPerElement() <= 0 || field_fes.GetVDim() != 1 ||
       submesh_data.GetSubmeshFESpace().GetVDim() != 1 ) {
    throw std::invalid_argument(
        "Field assembly requires matching dimensions, positive field/node counts, and a scalar space." );
  }
  const IndexT dofs1 = checkedProduct( nodes1_, dimension_ );
  const IndexT dofs2 = checkedProduct( mesh2.numberOfNodesPerElement(), dimension_ );
  if ( dofs1 > std::numeric_limits<IndexT>::max() - dofs2 ) {
    throw std::length_error( "Pair coordinate count exceeds the supported index range." );
  }
  dof_offsets_ = MdArray<IndexT, 3>{ 0, dofs1, dofs1 + dofs2 };
  const bool use_lor = mesh_data.GetLORMesh() != nullptr;
  const auto& coordinate_surface_fes = use_lor ? *mesh_data.GetLORMeshFESpace() : mesh_data.GetSubmeshFESpace();
  const auto& field_surface_fes = use_lor ? *submesh_data.GetLORMeshFESpace() : submesh_data.GetSubmeshFESpace();
  const auto& coordinate_redecomp_fes = *mesh_data.GetRedecompResponse().FESpace();
  const std::array<const Array1D<int>*, 2> element_maps{ &mesh_data.GetElemMap1(), &mesh_data.GetElemMap2() };
  fields_.reserve( field_count );
  gradients_.resize( field_count );
  for ( IndexT field = 0; field < field_count; ++field ) {
    fields_.emplace_back( const_cast<mfem::FiniteElementSpace*>( &field_fes ) );
    fields_.back() = 0.0;
    gradients_[field].reserve( 2 );
    for ( IndexT side = 0; side < 2; ++side ) {
      const IndexT block_size = checkedProduct( nodes1_, dof_offsets_[side + 1] - dof_offsets_[side] );
      gradients_[field].emplace_back( field_surface_fes, coordinate_surface_fes, field_fes, coordinate_redecomp_fes,
                                      *element_maps[0], *element_maps[side] );
      gradients_[field].back().reserve( pair_capacity, checkedProduct( pair_capacity, block_size ) );
    }
  }
  gradient_block_.resize( checkedProduct( nodes1_, std::max( dofs1, dofs2 ) ) );
}

void MfemFieldAssembler::addPair( InterfacePair pair, ArrayViewT<const RealT, 2> values,
                                  ArrayViewT<const RealT, 3> gradients )
{
  const IndexT field_count = fields_.size();
  if ( values.shape()[0] != field_count || values.shape()[1] != nodes1_ || gradients.shape()[0] != field_count ||
       gradients.shape()[1] != nodes1_ || gradients.shape()[2] != dof_offsets_[2] ) {
    throw std::invalid_argument( "Scalar field inputs must match the configured fields, nodes, and pair coordinates." );
  }
  const MdArray<IndexT, 2> elements{ pair.element_id1_, pair.element_id2_ };
  for ( IndexT field = 0; field < field_count; ++field ) {
    for ( IndexT node = 0; node < nodes1_; ++node ) {
      fields_[field]( connectivity_[0]( pair.element_id1_, node ) ) += values( field, node );
    }
    for ( IndexT side = 0; side < 2; ++side ) {
      const IndexT column_nodes = connectivity_[side].shape()[1];
      const IndexT column_dofs = dof_offsets_[side + 1] - dof_offsets_[side];
      for ( IndexT column = 0; column < column_dofs; ++column ) {
        const IndexT coordinate = dof_offsets_[side] + ( column % column_nodes ) * dimension_ + column / column_nodes;
        for ( IndexT node = 0; node < nodes1_; ++node ) {
          gradient_block_[node + column * nodes1_] = gradients( field, node, coordinate );
        }
      }
      gradients_[field][side].append( elements[0], elements[side], gradient_block_.data(),
                                      checkedProduct( nodes1_, column_dofs ) );
    }
  }
}

std::vector<AssembledField> MfemFieldAssembler::assemble() const
{
  auto* field_fes = const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() );
  const auto* parent_fes = mesh_data_.GetParentCoords().ParFESpace();
  std::vector<AssembledField> result;
  result.reserve( fields_.size() );
  for ( IndexT field = 0; field < static_cast<IndexT>( fields_.size() ); ++field ) {
    mfem::ParLinearForm linear_form( field_fes );
    submesh_data_.GetPressureTransfer().RedecompToSubmesh( fields_[field], linear_form );
    shared::ParVector values( field_fes );
    values.fill( 0.0 );
    field_fes->GetProlongationMatrix()->MultTranspose( linear_form, values.get() );
    auto jacobian = jacobian_data_.GetMfemJacobian( field_fes, parent_fes, gradients_[field] );
    result.push_back( { std::move( values ), std::move( jacobian ) } );
  }
  return result;
}
}  // namespace tribol
#endif  // BUILD_REDECOMP
