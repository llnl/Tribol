// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/assembly/MfemVectorAssembler.hpp"
#include "axom/slic/interface/slic.hpp"
#include "axom/slic/interface/slic_macros.hpp"
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
    SLIC_ERROR( "Vector assembly requires matching positive numbers of values per node, a transfer, and blocks." );
    return;
  }
  for ( const auto& block : blocks_ ) {
    const IndexT num_nodes_per_element = block.connectivity.shape()[1];
    if ( block.geometry < 0 || block.geometry > 1 || block.offset < 0 || num_nodes_per_element <= 0 ) {
      SLIC_ERROR( "Vector geometry selectors, offsets, or node counts are invalid." );
      return;
    }
    const IndexT max_size = std::numeric_limits<IndexT>::max();
    if ( num_nodes_per_element > ( max_size - block.offset ) / num_values_per_node ) {
      SLIC_ERROR( "Vector input extent exceeds the supported index range." );
      return;
    }
    size_values_ = std::max( size_values_, block.offset + num_nodes_per_element * num_values_per_node );
  }
  redecomp_values_.SetSpace( const_cast<mfem::FiniteElementSpace*>( &space_.redecomp_fes ) );
  redecomp_values_ = 0.0;
}

shared::ParVector MfemVectorAssembler::assemble( ExecutionMode execution_mode ) const
{
  const bool use_device = isOnDevice( execution_mode );
  const auto memory_space = use_device ? shared::ParVectorMemory::Device : shared::ParVectorMemory::Host;
  auto* final_fes = const_cast<mfem::ParFiniteElementSpace*>( &space_.final_fes );
  shared::ParVector result( memory_space, final_fes );
  result.fill( 0.0 );
  mfem::Vector local;
  local.UseDevice( use_device );
  local.SetSize( final_fes->GetVSize() );
  local = 0.0;
  space_.transfer( redecomp_values_, local );
  if ( use_device ) {
    shared::detail::HypreMemoryLocationGuard memory_guard( shared::ParVectorMemory::Device );
    final_fes->GetProlongationMatrix()->MultTranspose( local, result.get() );
  } else {
    // A GPU-enabled MFEM build still selects its device conforming operator for a sequential Tribol formulation. Use
    // the concrete HYPRE matrix so host execution updates the host-backed result.
    shared::detail::HypreMemoryLocationGuard memory_guard( shared::ParVectorMemory::Host );
    auto* prolongation = final_fes->Dof_TrueDof_Matrix();
    prolongation->HostRead();
    prolongation->MultTranspose( local, result.get() );
  }
  return result;
}

namespace detail {
MfemVectorAssembler makeResidualVectorAssembler( const MfemMeshData& data, MeshData::Viewer mesh1,
                                                 MeshData::Viewer mesh2 )
{
  const auto& redecomp_fes = *data.GetRedecompResponse().FESpace();
  const int num_values_per_node = redecomp_fes.GetVDim();
  if ( num_values_per_node <= 0 || mesh1.spatialDim() != num_values_per_node ||
       mesh2.spatialDim() != num_values_per_node || mesh1.numberOfNodesPerElement() <= 0 ||
       mesh2.numberOfNodesPerElement() <= 0 ) {
    SLIC_ERROR( "Residual vector assembly requires matching geometry and FE-space dimensions." );
  }
  const IndexT num_geom1_nodes = mesh1.numberOfNodesPerElement();
  if ( num_geom1_nodes > std::numeric_limits<IndexT>::max() / num_values_per_node ) {
    SLIC_ERROR( "Residual vector input extent exceeds the supported index range." );
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
    SLIC_ERROR( "Scalar field vector assembly requires scalar FE spaces." );
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
