#ifndef TRIBOL_ADAPTERS_MFEM_MFEMPARENTTRACE_HPP_
#define TRIBOL_ADAPTERS_MFEM_MFEMPARENTTRACE_HPP_

#include "tribol/basis/ParentBasis.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tribol::mfem::detail {

inline int decodeDof( int dof ) { return dof >= 0 ? dof : -1 - dof; }
inline Real dofSign( int dof ) { return dof >= 0 ? 1.0 : -1.0; }

inline ElementTopology topology( ::mfem::Element::Type type )
{
  switch ( type ) {
    case ::mfem::Element::SEGMENT:
      return ElementTopology::Segment;
    case ::mfem::Element::TRIANGLE:
      return ElementTopology::Triangle;
    case ::mfem::Element::QUADRILATERAL:
      return ElementTopology::Quadrilateral;
    default:
      throw std::invalid_argument( "MFEM contact supports segment, triangle, and quadrilateral boundaries." );
  }
}

struct SurfacePacketHeader {
  ElementTopology child_topology{};
  int attribute{};
  int boundary_element{};
  ElementTopology parent_topology{};
  int parent_order{};
  int parent_dof_count{};
  std::size_t parent_dof_offset{};
  int node_count{};
};

inline SurfacePacketHeader readSurfacePacketHeader( const std::vector<std::int64_t>& metadata, std::size_t& offset )
{
  SurfacePacketHeader header;
  header.child_topology = static_cast<ElementTopology>( metadata[offset++] );
  header.attribute = static_cast<int>( metadata[offset++] );
  header.boundary_element = static_cast<int>( metadata[offset++] );
  header.parent_topology = static_cast<ElementTopology>( metadata[offset++] );
  header.parent_order = static_cast<int>( metadata[offset++] );
  header.parent_dof_count = static_cast<int>( metadata[offset++] );
  header.parent_dof_offset = offset;
  offset += static_cast<std::size_t>( header.parent_dof_count );
  header.node_count = static_cast<int>( metadata[offset++] );
  return header;
}

inline std::vector<Real> parentBasisCoefficients( const ::mfem::FiniteElement& element, ElementTopology topology,
                                                  int order )
{
  const int size = element.GetDof();
  if ( size != basis::parentDofCount( topology, order ) || order < 1 || order > maximumParentTraceOrder ) {
    throw std::invalid_argument( "MFEM native parent traces support complete Q1-Q4 boundary bases." );
  }
  const auto& nodes = element.GetNodes();
  if ( nodes.GetNPoints() != size ) {
    throw std::invalid_argument( "MFEM parent basis must provide one reference node per degree of freedom." );
  }
  const int columns = 2 * size;
  std::vector<Real> augmented( static_cast<std::size_t>( size * columns ) );
  ::mfem::Vector shape( size );
  for ( int row = 0; row < size; ++row ) {
    const auto& point = nodes.IntPoint( row );
    const auto monomials = basis::parentMonomials( topology, order, basis::ReferencePoint{ point.x, point.y } );
    element.CalcShape( point, shape );
    for ( int column = 0; column < size; ++column ) {
      augmented[static_cast<std::size_t>( row * columns + column )] = monomials[column];
      augmented[static_cast<std::size_t>( row * columns + size + column )] = shape[column];
    }
  }
  for ( int pivot = 0; pivot < size; ++pivot ) {
    int selected = pivot;
    for ( int row = pivot + 1; row < size; ++row ) {
      if ( std::abs( augmented[static_cast<std::size_t>( row * columns + pivot )] ) >
           std::abs( augmented[static_cast<std::size_t>( selected * columns + pivot )] ) ) {
        selected = row;
      }
    }
    const Real diagonal = augmented[static_cast<std::size_t>( selected * columns + pivot )];
    if ( std::abs( diagonal ) <= 1.0e-14 ) {
      throw std::invalid_argument( "MFEM parent basis cannot be represented by the supported polynomial basis." );
    }
    if ( selected != pivot ) {
      for ( int column = 0; column < columns; ++column ) {
        std::swap( augmented[static_cast<std::size_t>( pivot * columns + column )],
                   augmented[static_cast<std::size_t>( selected * columns + column )] );
      }
    }
    const Real inverse = 1.0 / augmented[static_cast<std::size_t>( pivot * columns + pivot )];
    for ( int column = 0; column < columns; ++column ) {
      augmented[static_cast<std::size_t>( pivot * columns + column )] *= inverse;
    }
    for ( int row = 0; row < size; ++row ) {
      if ( row == pivot ) {
        continue;
      }
      const Real factor = augmented[static_cast<std::size_t>( row * columns + pivot )];
      for ( int column = 0; column < columns; ++column ) {
        augmented[static_cast<std::size_t>( row * columns + column )] -=
            factor * augmented[static_cast<std::size_t>( pivot * columns + column )];
      }
    }
  }
  std::vector<Real> coefficients( static_cast<std::size_t>( size * size ) );
  for ( int dof = 0; dof < size; ++dof ) {
    for ( int term = 0; term < size; ++term ) {
      coefficients[static_cast<std::size_t>( dof * size + term )] =
          augmented[static_cast<std::size_t>( term * columns + size + dof )];
    }
  }
  return coefficients;
}

class ParentTraceStorage {
 public:
  void clear()
  {
    parent_faces_.clear();
    face_ids_.clear();
    face_owners_.clear();
    face_topologies_.clear();
    face_orders_.clear();
    dof_offsets_.assign( 1, 0 );
    dofs_.clear();
    dof_signs_.clear();
    basis_offsets_.assign( 1, 0 );
    basis_coefficients_.clear();
    subfacet_faces_.clear();
    reference_coordinates_.clear();
  }

  Index findOrAppendFace( int owner, const SurfacePacketHeader& header, const std::vector<std::int64_t>& metadata,
                          ArrayView<const Real> coefficients )
  {
    const auto [iterator, inserted] = parent_faces_.emplace( std::make_pair( owner, header.boundary_element ),
                                                             static_cast<Index>( face_ids_.size() ) );
    if ( !inserted ) {
      return iterator->second;
    }
    face_ids_.push_back( ( static_cast<std::uint64_t>( static_cast<std::uint32_t>( owner ) ) << 32 ) |
                         static_cast<std::uint32_t>( header.boundary_element ) );
    face_owners_.push_back( owner );
    face_topologies_.push_back( header.parent_topology );
    face_orders_.push_back( header.parent_order );
    for ( int dof = 0; dof < header.parent_dof_count; ++dof ) {
      const int encoded = static_cast<int>( metadata[header.parent_dof_offset + static_cast<std::size_t>( dof )] );
      dofs_.push_back( decodeDof( encoded ) );
      dof_signs_.push_back( dofSign( encoded ) );
    }
    dof_offsets_.push_back( static_cast<Index>( dofs_.size() ) );
    basis_coefficients_.insert( basis_coefficients_.end(), coefficients.begin(), coefficients.end() );
    basis_offsets_.push_back( static_cast<Index>( basis_coefficients_.size() ) );
    return iterator->second;
  }

  void appendSubfacet( Index parent_face ) { subfacet_faces_.push_back( parent_face ); }

  void appendReferenceCoordinate( Real value ) { reference_coordinates_.push_back( value ); }

  [[nodiscard]] ParentTraceView view( SurfaceMeshView geometry ) const
  {
    return {
        .geometry = geometry,
        .parent_face_ids = { face_ids_.data(), static_cast<Index>( face_ids_.size() ) },
        .parent_face_owners = { face_owners_.data(), static_cast<Index>( face_owners_.size() ) },
        .parent_face_topologies = { face_topologies_.data(), static_cast<Index>( face_topologies_.size() ) },
        .parent_face_orders = { face_orders_.data(), static_cast<Index>( face_orders_.size() ) },
        .parent_dof_offsets = { dof_offsets_.data(), static_cast<Index>( dof_offsets_.size() ) },
        .parent_dofs = { dofs_.data(), static_cast<Index>( dofs_.size() ) },
        .parent_dof_signs = { dof_signs_.data(), static_cast<Index>( dof_signs_.size() ) },
        .parent_basis_offsets = { basis_offsets_.data(), static_cast<Index>( basis_offsets_.size() ) },
        .parent_basis_coefficients = { basis_coefficients_.data(), static_cast<Index>( basis_coefficients_.size() ) },
        .subfacet_parent_faces = { subfacet_faces_.data(), static_cast<Index>( subfacet_faces_.size() ) },
        .subfacet_parent_reference_coordinates = { { reference_coordinates_.data(),
                                                     static_cast<Index>( reference_coordinates_.size() ) },
                                                   geometry.connectivity.size(),
                                                   geometry.dimension - 1,
                                                   FieldLayout::Interleaved },
    };
  }

 private:
  std::map<std::pair<int, int>, Index> parent_faces_;
  std::vector<std::uint64_t> face_ids_;
  std::vector<int> face_owners_;
  std::vector<ElementTopology> face_topologies_;
  std::vector<int> face_orders_;
  std::vector<Index> dof_offsets_{ 0 };
  std::vector<Index> dofs_;
  std::vector<Real> dof_signs_;
  std::vector<Index> basis_offsets_{ 0 };
  std::vector<Real> basis_coefficients_;
  std::vector<Index> subfacet_faces_;
  std::vector<Real> reference_coordinates_;
};

}  // namespace tribol::mfem::detail

#endif
