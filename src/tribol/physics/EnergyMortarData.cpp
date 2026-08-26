// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortarData.hpp"

#include "axom/slic.hpp"

#include <algorithm>
#include <map>
#include <utility>

namespace tribol {

CsrMatrix::CsrMatrix( IndexT rows, IndexT columns, std::vector<IndexT> row_offsets,
                      std::vector<IndexT> column_indices, std::vector<RealT> values )
    : rows_( rows ),
      columns_( columns ),
      row_offsets_( std::move( row_offsets ) ),
      column_indices_( std::move( column_indices ) ),
      values_( std::move( values ) )
{
  SLIC_ERROR_ROOT_IF( rows_ < 0 || columns_ < 0, "CSR matrix dimensions must be nonnegative." );
  SLIC_ERROR_ROOT_IF( row_offsets_.size() != static_cast<std::size_t>( rows_ + 1 ),
                      "CSR row-offset array has the wrong size." );
  SLIC_ERROR_ROOT_IF( column_indices_.size() != values_.size(), "CSR column and value arrays have different sizes." );
  SLIC_ERROR_ROOT_IF( !row_offsets_.empty() && row_offsets_.back() != static_cast<IndexT>( values_.size() ),
                      "CSR row offsets do not match the number of entries." );
}

CsrMatrix CsrMatrix::fromEntries( IndexT rows, IndexT columns, std::vector<Entry> entries )
{
  entries.erase( std::remove_if( entries.begin(), entries.end(), []( const Entry& entry ) {
                   return entry.value == 0.0;
                 } ),
                 entries.end() );
  std::sort( entries.begin(), entries.end(), []( const Entry& left, const Entry& right ) {
    return left.row < right.row || ( left.row == right.row && left.column < right.column );
  } );

  std::vector<IndexT> row_offsets( static_cast<std::size_t>( rows ) + 1, 0 );
  std::vector<IndexT> column_indices;
  std::vector<RealT> values;
  column_indices.reserve( entries.size() );
  values.reserve( entries.size() );

  IndexT current_row = 0;
  for ( std::size_t i = 0; i < entries.size(); ) {
    const auto row = entries[i].row;
    const auto column = entries[i].column;
    SLIC_ERROR_ROOT_IF( row < 0 || row >= rows || column < 0 || column >= columns,
                        "Sparse entry lies outside the matrix dimensions." );
    while ( current_row < row ) {
      row_offsets[static_cast<std::size_t>( ++current_row )] = static_cast<IndexT>( values.size() );
    }
    RealT value = 0.0;
    do {
      value += entries[i].value;
      ++i;
    } while ( i < entries.size() && entries[i].row == row && entries[i].column == column );
    if ( value != 0.0 ) {
      column_indices.push_back( column );
      values.push_back( value );
    }
  }
  while ( current_row < rows ) {
    row_offsets[static_cast<std::size_t>( ++current_row )] = static_cast<IndexT>( values.size() );
  }
  return CsrMatrix( rows, columns, std::move( row_offsets ), std::move( column_indices ), std::move( values ) );
}

CsrMatrix CsrMatrix::diagonal( const std::vector<RealT>& values )
{
  std::vector<Entry> entries;
  entries.reserve( values.size() );
  for ( IndexT i = 0; i < static_cast<IndexT>( values.size() ); ++i ) {
    entries.push_back( { i, i, values[static_cast<std::size_t>( i )] } );
  }
  return fromEntries( static_cast<IndexT>( values.size() ), static_cast<IndexT>( values.size() ),
                      std::move( entries ) );
}

std::vector<RealT> CsrMatrix::multiply( const std::vector<RealT>& vector ) const
{
  SLIC_ERROR_ROOT_IF( vector.size() != static_cast<std::size_t>( columns_ ), "CSR matrix-vector size mismatch." );
  std::vector<RealT> result( static_cast<std::size_t>( rows_ ), 0.0 );
  for ( IndexT row = 0; row < rows_; ++row ) {
    for ( IndexT entry = row_offsets_[row]; entry < row_offsets_[row + 1]; ++entry ) {
      result[row] += values_[entry] * vector[column_indices_[entry]];
    }
  }
  return result;
}

std::vector<RealT> CsrMatrix::transposeMultiply( const std::vector<RealT>& vector ) const
{
  SLIC_ERROR_ROOT_IF( vector.size() != static_cast<std::size_t>( rows_ ), "CSR transpose-vector size mismatch." );
  std::vector<RealT> result( static_cast<std::size_t>( columns_ ), 0.0 );
  for ( IndexT row = 0; row < rows_; ++row ) {
    for ( IndexT entry = row_offsets_[row]; entry < row_offsets_[row + 1]; ++entry ) {
      result[column_indices_[entry]] += values_[entry] * vector[row];
    }
  }
  return result;
}

CsrMatrix CsrMatrix::transpose() const
{
  std::vector<Entry> entries;
  entries.reserve( values_.size() );
  for ( IndexT row = 0; row < rows_; ++row ) {
    for ( IndexT entry = row_offsets_[row]; entry < row_offsets_[row + 1]; ++entry ) {
      entries.push_back( { column_indices_[entry], row, values_[entry] } );
    }
  }
  return fromEntries( columns_, rows_, std::move( entries ) );
}

CsrMatrix CsrMatrix::multiply( const CsrMatrix& right ) const
{
  SLIC_ERROR_ROOT_IF( columns_ != right.rows_, "CSR matrix-matrix size mismatch." );
  std::vector<Entry> entries;
  for ( IndexT row = 0; row < rows_; ++row ) {
    std::map<IndexT, RealT> row_values;
    for ( IndexT left_entry = row_offsets_[row]; left_entry < row_offsets_[row + 1]; ++left_entry ) {
      const auto inner = column_indices_[left_entry];
      for ( IndexT right_entry = right.row_offsets_[inner]; right_entry < right.row_offsets_[inner + 1];
            ++right_entry ) {
        row_values[right.column_indices_[right_entry]] += values_[left_entry] * right.values_[right_entry];
      }
    }
    for ( const auto& entry : row_values ) {
      entries.push_back( { row, entry.first, entry.second } );
    }
  }
  return fromEntries( rows_, right.columns_, std::move( entries ) );
}

CsrMatrix CsrMatrix::add( const CsrMatrix& right, RealT right_scale ) const
{
  SLIC_ERROR_ROOT_IF( rows_ != right.rows_ || columns_ != right.columns_, "CSR matrix addition size mismatch." );
  std::vector<Entry> entries;
  entries.reserve( values_.size() + right.values_.size() );
  for ( IndexT row = 0; row < rows_; ++row ) {
    for ( IndexT entry = row_offsets_[row]; entry < row_offsets_[row + 1]; ++entry ) {
      entries.push_back( { row, column_indices_[entry], values_[entry] } );
    }
    for ( IndexT entry = right.row_offsets_[row]; entry < right.row_offsets_[row + 1]; ++entry ) {
      entries.push_back( { row, right.column_indices_[entry], right_scale * right.values_[entry] } );
    }
  }
  return fromEntries( rows_, columns_, std::move( entries ) );
}

CsrMatrix CsrMatrix::scaledRows( const std::vector<RealT>& scales ) const
{
  SLIC_ERROR_ROOT_IF( scales.size() != static_cast<std::size_t>( rows_ ), "CSR row scaling size mismatch." );
  auto values = values_;
  for ( IndexT row = 0; row < rows_; ++row ) {
    for ( IndexT entry = row_offsets_[row]; entry < row_offsets_[row + 1]; ++entry ) {
      values[entry] *= scales[row];
    }
  }
  return CsrMatrix( rows_, columns_, row_offsets_, column_indices_, std::move( values ) );
}

void PrimitivePairContributions::append( PrimitiveBlockRole role, ContactVariable row_variable,
                                         ContactVariable column_variable, IndexT row_element_id,
                                         IndexT column_element_id, IndexT rows, IndexT columns, const RealT* data )
{
  const auto offset = values.size();
  values.insert( values.end(), data, data + rows * columns );
  contributions.push_back(
      { role, row_variable, column_variable, row_element_id, column_element_id, rows, columns, offset } );
}

}  // namespace tribol
