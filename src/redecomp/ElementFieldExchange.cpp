// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "redecomp/ElementFieldExchange.hpp"

#include "axom/slic.hpp"

namespace redecomp {

namespace {

MPI_Comm GetQuadratureCommunicator( const mfem::QuadratureFunction& source )
{
  const auto* parallel_mesh = dynamic_cast<const mfem::ParMesh*>( source.GetSpace()->GetMesh() );
  SLIC_ERROR_ROOT_IF( parallel_mesh == nullptr,
                      "ElementFieldExchange requires a QuadratureFunction on an mfem::ParMesh." );
  return parallel_mesh->GetComm();
}

mfem::Array<int> BuildQuadratureElementOffsets( const mfem::QuadratureFunction& source )
{
  const auto* space = source.GetSpace();
  SLIC_ERROR_ROOT_IF( space == nullptr, "ElementFieldExchange requires a QuadratureFunction space." );
  mfem::Array<int> offsets( space->GetNE() + 1 );
  for ( int element = 0; element <= space->GetNE(); ++element ) {
    offsets[element] = space->Offset( element ) * source.GetVDim();
  }
  return offsets;
}

}  // namespace

ElementFieldExchange::ElementFieldExchange( const mfem::QuadratureFunction& source_layout,
                                            std::vector<mfem::Array<int>> source_elements_by_destination )
    : source_space_{ source_layout.GetSpace() },
      vector_dimension_{ source_layout.GetVDim() },
      plan_{ std::make_shared<ElementExchangePlan>( GetQuadratureCommunicator( source_layout ),
                                                    BuildQuadratureElementOffsets( source_layout ),
                                                    std::move( source_elements_by_destination ) ) },
      exchange_{ plan_ },
      returned_source_dual_( exchange_.Width() )
{
}

ExchangedElementField ElementFieldExchange::Gather( const mfem::QuadratureFunction& source ) const
{
  SLIC_ERROR_ROOT_IF( source.GetSpace() != source_space_ || source.GetVDim() != vector_dimension_,
                      "ElementFieldExchange source does not match the layout used to build the exchange plan." );
  ExchangedElementField destination;
  destination.values.SetSize( exchange_.Height() );
  exchange_.Mult( source, destination.values );
  destination.element_offsets = plan_->DestinationElementOffsets();
  destination.source_ranks = plan_->DestinationSourceRanks();
  destination.source_elements = plan_->DestinationSourceElements();
  return destination;
}

void ElementFieldExchange::AddToSource( const ExchangedElementField& destination_dual,
                                        mfem::QuadratureFunction& source_dual, mfem::real_t scale ) const
{
  SLIC_ERROR_ROOT_IF( source_dual.GetSpace() != source_space_ || source_dual.GetVDim() != vector_dimension_,
                      "ElementFieldExchange source dual does not match the exchange plan." );
  SLIC_ERROR_ROOT_IF( destination_dual.values.Size() != exchange_.Height(),
                      "ElementFieldExchange destination dual size mismatch." );
  exchange_.MultTranspose( destination_dual.values, returned_source_dual_ );
  source_dual.Add( scale, returned_source_dual_ );
}

}  // namespace redecomp
