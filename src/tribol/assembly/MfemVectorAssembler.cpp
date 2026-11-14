// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/assembly/MfemVectorAssembler.hpp"
#ifdef BUILD_REDECOMP
#include <algorithm>
#include <limits>
#include <utility>

namespace tribol {
MfemVectorAssembler::MfemVectorAssembler( MfemVectorSpace space, std::vector<MfemVectorBlock> blocks )
    : space_( std::move( space ) ), blocks_( std::move( blocks ) )
{
  const int num_values_per_node = space_.redecomp_fes.GetVDim();
  if ( num_values_per_node <= 0 || num_values_per_node != space_.final_fes.GetVDim() || !space_.transfer ||
       blocks_.empty() ) {
    throw std::invalid_argument(
        "Vector assembly requires matching positive numbers of values per node, a transfer, and blocks." );
  }
  for ( const auto& block : blocks_ ) {
    const IndexT num_nodes_per_element = block.connectivity.shape()[1];
    if ( block.geometry < 0 || block.geometry > 1 || block.offset < 0 || num_nodes_per_element <= 0 ) {
      throw std::invalid_argument( "Vector geometry selectors, offsets, or node counts are invalid." );
    }
    const IndexT max_size = std::numeric_limits<IndexT>::max();
    if ( num_nodes_per_element > ( max_size - block.offset ) / num_values_per_node ) {
      throw std::length_error( "Vector input extent exceeds the supported index range." );
    }
    size_values_ = std::max( size_values_, block.offset + num_nodes_per_element * num_values_per_node );
  }
  redecomp_values_.SetSpace( const_cast<mfem::FiniteElementSpace*>( &space_.redecomp_fes ) );
  redecomp_values_ = 0.0;
}

shared::ParVector MfemVectorAssembler::assemble() const
{
  auto* final_fes = const_cast<mfem::ParFiniteElementSpace*>( &space_.final_fes );
  shared::ParVector result( final_fes );
  result.fill( 0.0 );
  mfem::Vector local( final_fes->GetVSize() );
  local = 0.0;
  space_.transfer( redecomp_values_, local );
  final_fes->GetProlongationMatrix()->MultTranspose( local, result.get() );
  return result;
}

namespace detail {
MfemVectorAssembler makeForceVectorAssembler( const MfemMeshData& data, MeshData::Viewer mesh1, MeshData::Viewer mesh2 )
{
  const auto& redecomp_fes = *data.GetRedecompResponse().FESpace();
  const int num_values_per_node = redecomp_fes.GetVDim();
  if ( num_values_per_node <= 0 || mesh1.spatialDim() != num_values_per_node ||
       mesh2.spatialDim() != num_values_per_node || mesh1.numberOfNodesPerElement() <= 0 ||
       mesh2.numberOfNodesPerElement() <= 0 ) {
    throw std::invalid_argument( "Force vector assembly requires matching geometry and FE-space dimensions." );
  }
  const IndexT num_geom1_nodes = mesh1.numberOfNodesPerElement();
  if ( num_geom1_nodes > std::numeric_limits<IndexT>::max() / num_values_per_node ) {
    throw std::length_error( "Force vector input extent exceeds the supported index range." );
  }
  return {
      { *data.GetParentCoords().ParFESpace(), redecomp_fes,
        [&data]( const mfem::GridFunction& values, mfem::Vector& destination ) {
          data.GetParentRedecompTransfer().RedecompToParent( values, destination );
        } },
      { { 0, mesh1.getConnectivity(), 0 }, { 1, mesh2.getConnectivity(), num_geom1_nodes * num_values_per_node } } };
}

MfemVectorAssembler makeScalarFieldVectorAssembler( const MfemSubmeshData& data, MeshData::Viewer mesh1 )
{
  if ( data.GetRedecompFESpace().GetVDim() != 1 || data.GetSubmeshFESpace().GetVDim() != 1 ) {
    throw std::invalid_argument( "Scalar field vector assembly requires scalar FE spaces." );
  }
  return { { data.GetSubmeshFESpace(), data.GetRedecompFESpace(),
             [&data]( const mfem::GridFunction& values, mfem::Vector& destination ) {
               data.GetPressureTransfer().RedecompToSubmesh( values, destination );
             } },
           { { 0, mesh1.getConnectivity(), 0 } } };
}
}  // namespace detail
}  // namespace tribol
#endif  // BUILD_REDECOMP
