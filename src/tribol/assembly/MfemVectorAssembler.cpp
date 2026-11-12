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
  const int components = space_.redecomp_fes.GetVDim();
  if ( components <= 0 || components != space_.final_fes.GetVDim() || !space_.transfer || blocks_.empty() ) {
    throw std::invalid_argument(
        "Vector assembly requires matching positive component counts, a transfer, and blocks." );
  }
  for ( const auto& block : blocks_ ) {
    const IndexT nodes = block.connectivity.shape()[1];
    if ( block.geometry < 0 || block.geometry > 1 || block.offset < 0 || nodes <= 0 ) {
      throw std::invalid_argument( "Vector geometry selectors, offsets, or node counts are invalid." );
    }
    const IndexT max_size = std::numeric_limits<IndexT>::max();
    if ( nodes > ( max_size - block.offset ) / components ) {
      throw std::length_error( "Vector input extent exceeds the supported index range." );
    }
    value_ct_ = std::max( value_ct_, block.offset + nodes * components );
  }
  redecomp_values_.SetSpace( const_cast<mfem::FiniteElementSpace*>( &space_.redecomp_fes ) );
  redecomp_values_ = 0.0;
}

void MfemVectorAssembler::addPair( InterfacePair pair, ArrayViewT<const RealT> values )
{
  if ( values.size() != value_ct_ ) {
    throw std::invalid_argument( "Vector input size does not match the configured extent." );
  }
  addPairFromReader( pair, [=]( IndexT row ) { return values[row]; } );
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
MfemVectorAssembler makeDisplacementVectorAssembler( const MfemMeshData& data, MeshData::Viewer mesh1,
                                                     MeshData::Viewer mesh2 )
{
  const auto& redecomp_fes = *data.GetRedecompResponse().FESpace();
  const int components = redecomp_fes.GetVDim();
  if ( components <= 0 || mesh1.spatialDimension() != components || mesh2.spatialDimension() != components ||
       mesh1.numberOfNodesPerElement() <= 0 || mesh2.numberOfNodesPerElement() <= 0 ) {
    throw std::invalid_argument( "Displacement vector assembly requires matching geometry and FE-space dimensions." );
  }
  const IndexT nodes1 = mesh1.numberOfNodesPerElement();
  if ( nodes1 > std::numeric_limits<IndexT>::max() / components ) {
    throw std::length_error( "Displacement vector input extent exceeds the supported index range." );
  }
  return { { *data.GetParentCoords().ParFESpace(), redecomp_fes,
             [&data]( const mfem::GridFunction& values, mfem::Vector& destination ) {
               data.GetParentRedecompTransfer().RedecompToParent( values, destination );
             } },
           { { 0, mesh1.getConnectivity(), 0 }, { 1, mesh2.getConnectivity(), nodes1 * components } } };
}

MfemVectorAssembler makeNodalVectorAssembler( const MfemSubmeshData& data, MeshData::Viewer mesh1 )
{
  if ( data.GetRedecompFESpace().GetVDim() != 1 || data.GetSubmeshFESpace().GetVDim() != 1 ) {
    throw std::invalid_argument( "Nodal vector assembly requires scalar FE spaces." );
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
