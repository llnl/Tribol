// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/assembly/MfemFieldAssembler.hpp"
#ifdef BUILD_REDECOMP
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
    : submesh_data_( submesh_data ),
      connectivity_( mesh1.getConnectivity() ),
      nodes1_( mesh1.numberOfNodesPerElement() )
{
  const auto& field_fes = submesh_data.GetRedecompFESpace();
  const int dimension = mesh1.spatialDimension();
  if ( field_count <= 0 || pair_capacity < 0 || dimension <= 0 || mesh2.spatialDimension() != dimension ||
       mesh_data.GetRedecompResponse().FESpace()->GetVDim() != dimension || nodes1_ <= 0 ||
       mesh2.numberOfNodesPerElement() <= 0 || field_fes.GetVDim() != 1 ||
       submesh_data.GetSubmeshFESpace().GetVDim() != 1 ) {
    throw std::invalid_argument(
        "Field assembly requires matching dimensions, positive field/node counts, and a scalar space." );
  }
  const IndexT dofs1 = checkedProduct( nodes1_, dimension );
  const IndexT dofs2 = checkedProduct( mesh2.numberOfNodesPerElement(), dimension );
  if ( dofs1 > std::numeric_limits<IndexT>::max() - dofs2 ) {
    throw std::length_error( "Pair coordinate count exceeds the supported index range." );
  }
  dof_offsets_ = MdArray<IndexT, 3>{ 0, dofs1, dofs1 + dofs2 };
  const auto& coordinate_redecomp_fes = *mesh_data.GetRedecompResponse().FESpace();
  const std::array<const Array1D<int>*, 2> element_maps{ &mesh_data.GetElemMap1(), &mesh_data.GetElemMap2() };
  fields_.reserve( field_count );
  matrices_.reserve( field_count );
  for ( IndexT field = 0; field < field_count; ++field ) {
    fields_.emplace_back( const_cast<mfem::FiniteElementSpace*>( &field_fes ) );
    fields_.back() = 0.0;
    matrices_.emplace_back(
        MfemMatrixSpace{ jacobian_data.SubmeshPath(), field_fes },
        MfemMatrixSpace{ jacobian_data.ParentPath(), coordinate_redecomp_fes },
        std::vector<MfemMatrixBlock>{ { 0, 0, nodes1_, nodes1_, 0, 0, *element_maps[0], *element_maps[0] },
                                      { 0, 1, nodes1_, mesh2.numberOfNodesPerElement(), 0, dof_offsets_[1],
                                        *element_maps[0], *element_maps[1] } },
        pair_capacity );
  }
}

void MfemFieldAssembler::addPair( InterfacePair pair, ArrayViewT<const RealT, 2> values,
                                  ArrayViewT<const RealT, 3> gradients )
{
  const IndexT field_count = fields_.size();
  if ( values.shape()[0] != field_count || values.shape()[1] != nodes1_ || gradients.shape()[0] != field_count ||
       gradients.shape()[1] != nodes1_ || gradients.shape()[2] != dof_offsets_[2] ) {
    throw std::invalid_argument( "Scalar field inputs must match the configured fields, nodes, and pair coordinates." );
  }
  for ( IndexT field = 0; field < field_count; ++field ) {
    for ( IndexT node = 0; node < nodes1_; ++node ) {
      fields_[field]( connectivity_( pair.element_id1_, node ) ) += values( field, node );
    }
    matrices_[field].addPairFromReader(
        pair, [&]( IndexT node, IndexT coordinate ) { return gradients( field, node, coordinate ); } );
  }
}

std::vector<AssembledField> MfemFieldAssembler::assemble() const
{
  auto* field_fes = const_cast<mfem::ParFiniteElementSpace*>( &submesh_data_.GetSubmeshFESpace() );
  std::vector<AssembledField> result;
  result.reserve( fields_.size() );
  for ( IndexT field = 0; field < static_cast<IndexT>( fields_.size() ); ++field ) {
    mfem::ParLinearForm linear_form( field_fes );
    submesh_data_.GetPressureTransfer().RedecompToSubmesh( fields_[field], linear_form );
    shared::ParVector values( field_fes );
    values.fill( 0.0 );
    field_fes->GetProlongationMatrix()->MultTranspose( linear_form, values.get() );
    auto jacobian = matrices_[field].assemble();
    result.push_back( { std::move( values ), std::move( jacobian ) } );
  }
  return result;
}
}  // namespace tribol
#endif  // BUILD_REDECOMP
