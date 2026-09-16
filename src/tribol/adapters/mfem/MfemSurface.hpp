#ifndef TRIBOL_ADAPTERS_MFEM_MFEMSURFACE_HPP_
#define TRIBOL_ADAPTERS_MFEM_MFEMSURFACE_HPP_

#include "tribol/core/MeshView.hpp"

#include "mfem.hpp"
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tribol::mfem {

struct PairedBoundaryAttributes {
  std::vector<int> mortar;
  std::vector<int> nonmortar;

  PairedBoundaryAttributes( std::initializer_list<int> mortar_attributes,
                            std::initializer_list<int> nonmortar_attributes )
      : mortar( mortar_attributes ), nonmortar( nonmortar_attributes )
  {
    if ( mortar.empty() || nonmortar.empty() ) {
      throw std::invalid_argument( "MFEM contact requires nonempty mortar and nonmortar boundary attributes." );
    }
  }
};

namespace detail {

inline int decodeDof( int dof ) { return dof >= 0 ? dof : -1 - dof; }
inline Real dofSign( int dof ) { return dof >= 0 ? 1.0 : -1.0; }
inline MPI_Datatype realMpiType() { return std::is_same_v<Real, float> ? MPI_FLOAT : MPI_DOUBLE; }

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

inline bool contains( const std::vector<int>& values, int value )
{
  return std::find( values.begin(), values.end(), value ) != values.end();
}

class SurfaceStorage {
 public:
  struct RestrictionNode {
    std::vector<int> scalar_dofs;
    std::vector<Real> weights;
  };

  SurfaceStorage( const ::mfem::ParMesh& mesh, const ::mfem::ParGridFunction& coordinates,
                  const std::vector<int>& selected_attributes )
      : mesh_( mesh ), selected_attributes_( selected_attributes ), dimension_( mesh.SpaceDimension() )
  {
    MPI_Comm_rank( mesh.GetComm(), &rank_ );
    rebuild( coordinates );
  }

  void updateCoordinates( const ::mfem::ParGridFunction& coordinates ) { rebuild( coordinates ); }

  [[nodiscard]] SurfaceMeshView view() const
  {
    return {
        .dimension = dimension_,
        .coordinates =
            FieldView<const Real>{
                ArrayView<const Real>{ coordinates_.data(), static_cast<Index>( coordinates_.size() ) },
                static_cast<Index>( owner_local_nodes_.size() ), dimension_, FieldLayout::Interleaved },
        .element_offsets = ArrayView<const Index>{ offsets_.data(), static_cast<Index>( offsets_.size() ) },
        .connectivity = ArrayView<const Index>{ connectivity_.data(), static_cast<Index>( connectivity_.size() ) },
        .topologies = ArrayView<const ElementTopology>{ topologies_.data(), static_cast<Index>( topologies_.size() ) },
        .attributes = ArrayView<const int>{ attributes_.data(), static_cast<Index>( attributes_.size() ) },
    };
  }

  [[nodiscard]] bool isLocallyOwned( std::size_t node ) const { return owner_ranks_[node] == rank_; }

  [[nodiscard]] const RestrictionNode& localRestriction( std::size_t node ) const
  {
    return local_restrictions_[static_cast<std::size_t>( owner_local_nodes_[node] )];
  }

 private:
  template <typename T>
  static MPI_Datatype mpiType()
  {
    if constexpr ( std::is_same_v<T, Real> ) {
      return realMpiType();
    } else {
      static_assert( std::is_same_v<T, std::int64_t> );
      return MPI_INT64_T;
    }
  }

  template <typename T>
  static std::vector<T> allGather( MPI_Comm communicator, const std::vector<T>& local )
  {
    int size{};
    MPI_Comm_size( communicator, &size );
    const int local_count = static_cast<int>( local.size() );
    std::vector<int> counts( static_cast<std::size_t>( size ) );
    MPI_Allgather( &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, communicator );
    std::vector<int> displacements( static_cast<std::size_t>( size ) );
    int total{};
    for ( int rank = 0; rank < size; ++rank ) {
      displacements[static_cast<std::size_t>( rank )] = total;
      total += counts[static_cast<std::size_t>( rank )];
    }
    std::vector<T> gathered( static_cast<std::size_t>( total ) );
    MPI_Allgatherv( local.data(), local_count, mpiType<T>(), gathered.data(), counts.data(), displacements.data(),
                    mpiType<T>(), communicator );
    return gathered;
  }

  void rebuild( const ::mfem::ParGridFunction& coordinate_field )
  {
    const auto* space = coordinate_field.ParFESpace();
    if ( space == nullptr || space->GetParMesh() != &mesh_ || space->GetVDim() != dimension_ ) {
      throw std::invalid_argument( "The MFEM adapter requires a vector coordinate field on the supplied mesh." );
    }
    std::vector<std::int64_t> local_metadata;
    std::vector<Real> local_coordinates;
    local_restrictions_.clear();
    for ( int boundary_element = 0; boundary_element < mesh_.GetNBE(); ++boundary_element ) {
      const int attribute = mesh_.GetBdrAttribute( boundary_element );
      if ( contains( selected_attributes_, attribute ) ) {
        appendRefinedBoundaryElement( *space, coordinate_field, boundary_element, attribute, local_metadata,
                                      local_coordinates );
      }
    }
    const auto metadata = allGather( mesh_.GetComm(), local_metadata );
    coordinates_ = allGather( mesh_.GetComm(), local_coordinates );
    rebuildTopology( metadata, local_metadata.size() );
  }

  void rebuildTopology( const std::vector<std::int64_t>& metadata, std::size_t local_metadata_size )
  {
    offsets_.assign( 1, 0 );
    connectivity_.clear();
    topologies_.clear();
    attributes_.clear();
    owner_local_nodes_.clear();
    owner_ranks_.clear();
    std::size_t metadata_offset{};
    std::size_t coordinate_offset{};
    int communicator_size{};
    MPI_Comm_size( mesh_.GetComm(), &communicator_size );
    std::vector<int> local_metadata_counts( static_cast<std::size_t>( communicator_size ) );
    const int local_metadata_count = static_cast<int>( local_metadata_size );
    MPI_Allgather( &local_metadata_count, 1, MPI_INT, local_metadata_counts.data(), 1, MPI_INT, mesh_.GetComm() );
    for ( int owner = 0; owner < communicator_size; ++owner ) {
      const std::size_t owner_end = metadata_offset + local_metadata_counts[static_cast<std::size_t>( owner )];
      while ( metadata_offset < owner_end ) {
        const auto element_topology = static_cast<ElementTopology>( metadata[metadata_offset++] );
        const int attribute = static_cast<int>( metadata[metadata_offset++] );
        const int node_count = static_cast<int>( metadata[metadata_offset++] );
        topologies_.push_back( element_topology );
        attributes_.push_back( attribute );
        for ( int local_node = 0; local_node < node_count; ++local_node ) {
          owner_local_nodes_.push_back( static_cast<int>( metadata[metadata_offset++] ) );
          owner_ranks_.push_back( owner );
          connectivity_.push_back( static_cast<Index>( connectivity_.size() ) );
          coordinate_offset += static_cast<std::size_t>( dimension_ );
        }
        offsets_.push_back( static_cast<Index>( connectivity_.size() ) );
      }
    }
    if ( topologies_.empty() ) {
      throw std::invalid_argument( "Selected MFEM boundary attributes contain no boundary elements on any rank." );
    }
    if ( coordinate_offset != coordinates_.size() ) {
      throw std::logic_error( "MFEM distributed surface metadata and coordinates are inconsistent." );
    }
  }

  void appendNode( const ::mfem::ParFiniteElementSpace& space, const ::mfem::ParGridFunction& coordinate_field,
                   int boundary_element, const ::mfem::IntegrationPoint& point, std::vector<std::int64_t>& metadata,
                   std::vector<Real>& coordinates )
  {
    const auto* element = space.GetBE( boundary_element );
    ::mfem::Vector shape( element->GetDof() );
    ::mfem::Array<int> dofs;
    element->CalcShape( point, shape );
    space.GetBdrElementDofs( boundary_element, dofs );
    if ( shape.Size() != dofs.Size() ) {
      throw std::invalid_argument( "MFEM boundary element shape and degree-of-freedom counts do not match." );
    }
    RestrictionNode restriction;
    restriction.scalar_dofs.assign( dofs.begin(), dofs.end() );
    restriction.weights.assign( shape.GetData(), shape.GetData() + shape.Size() );
    metadata.push_back( static_cast<std::int64_t>( local_restrictions_.size() ) );
    local_restrictions_.push_back( std::move( restriction ) );
    for ( int component = 0; component < dimension_; ++component ) {
      Real value{};
      for ( int dof = 0; dof < shape.Size(); ++dof ) {
        const int scalar_dof = dofs[dof];
        const int vector_dof = space.DofToVDof( decodeDof( scalar_dof ), component );
        value += shape[dof] * dofSign( scalar_dof ) * dofSign( vector_dof ) * coordinate_field[decodeDof( vector_dof )];
      }
      coordinates.push_back( value );
    }
  }

  void appendSubelement( const ::mfem::ParFiniteElementSpace& space, const ::mfem::ParGridFunction& coordinate_field,
                         int boundary_element, int attribute, ElementTopology child_topology,
                         const ::mfem::IntegrationPoint* points, int point_count, std::vector<std::int64_t>& metadata,
                         std::vector<Real>& coordinates )
  {
    metadata.push_back( static_cast<std::int64_t>( child_topology ) );
    metadata.push_back( attribute );
    metadata.push_back( point_count );
    for ( int point = 0; point < point_count; ++point ) {
      appendNode( space, coordinate_field, boundary_element, points[point], metadata, coordinates );
    }
  }

  void appendRefinedBoundaryElement( const ::mfem::ParFiniteElementSpace& space,
                                     const ::mfem::ParGridFunction& coordinate_field, int boundary_element,
                                     int attribute, std::vector<std::int64_t>& metadata,
                                     std::vector<Real>& coordinates )
  {
    const auto* finite_element = space.GetBE( boundary_element );
    const int order = std::max( finite_element->GetOrder(), 1 );
    const auto element_topology = topology( mesh_.GetBdrElementType( boundary_element ) );
    if ( element_topology == ElementTopology::Segment ) {
      for ( int segment = 0; segment < order; ++segment ) {
        ::mfem::IntegrationPoint points[2];
        points[0].x = static_cast<Real>( segment ) / order;
        points[1].x = static_cast<Real>( segment + 1 ) / order;
        appendSubelement( space, coordinate_field, boundary_element, attribute, ElementTopology::Segment, points, 2,
                          metadata, coordinates );
      }
      return;
    }
    if ( element_topology == ElementTopology::Quadrilateral ) {
      for ( int second = 0; second < order; ++second ) {
        for ( int first = 0; first < order; ++first ) {
          ::mfem::IntegrationPoint points[4];
          points[0].Set2( static_cast<Real>( first ) / order, static_cast<Real>( second ) / order );
          points[1].Set2( static_cast<Real>( first + 1 ) / order, static_cast<Real>( second ) / order );
          points[2].Set2( static_cast<Real>( first + 1 ) / order, static_cast<Real>( second + 1 ) / order );
          points[3].Set2( static_cast<Real>( first ) / order, static_cast<Real>( second + 1 ) / order );
          appendSubelement( space, coordinate_field, boundary_element, attribute, ElementTopology::Quadrilateral,
                            points, 4, metadata, coordinates );
        }
      }
      return;
    }
    for ( int first = 0; first < order; ++first ) {
      for ( int second = 0; second < order - first; ++second ) {
        ::mfem::IntegrationPoint lower[3];
        lower[0].Set2( static_cast<Real>( first ) / order, static_cast<Real>( second ) / order );
        lower[1].Set2( static_cast<Real>( first + 1 ) / order, static_cast<Real>( second ) / order );
        lower[2].Set2( static_cast<Real>( first ) / order, static_cast<Real>( second + 1 ) / order );
        appendSubelement( space, coordinate_field, boundary_element, attribute, ElementTopology::Triangle, lower, 3,
                          metadata, coordinates );
        if ( first + second < order - 1 ) {
          ::mfem::IntegrationPoint upper[3];
          upper[0].Set2( static_cast<Real>( first + 1 ) / order, static_cast<Real>( second ) / order );
          upper[1].Set2( static_cast<Real>( first + 1 ) / order, static_cast<Real>( second + 1 ) / order );
          upper[2].Set2( static_cast<Real>( first ) / order, static_cast<Real>( second + 1 ) / order );
          appendSubelement( space, coordinate_field, boundary_element, attribute, ElementTopology::Triangle, upper, 3,
                            metadata, coordinates );
        }
      }
    }
  }

  const ::mfem::ParMesh& mesh_;
  std::vector<int> selected_attributes_;
  int dimension_{};
  int rank_{};
  std::vector<Real> coordinates_;
  std::vector<Index> offsets_;
  std::vector<Index> connectivity_;
  std::vector<ElementTopology> topologies_;
  std::vector<int> attributes_;
  std::vector<int> owner_local_nodes_;
  std::vector<int> owner_ranks_;
  std::vector<RestrictionNode> local_restrictions_;
};

class SurfacePairStorage {
 public:
  SurfacePairStorage( const ::mfem::ParMesh& mesh, const ::mfem::ParGridFunction& coordinates,
                      const PairedBoundaryAttributes& attributes )
      : mortar_( mesh, coordinates, attributes.mortar ), nonmortar_( mesh, coordinates, attributes.nonmortar )
  {
  }

  void updateCoordinates( const ::mfem::ParGridFunction& coordinates )
  {
    mortar_.updateCoordinates( coordinates );
    nonmortar_.updateCoordinates( coordinates );
  }

  [[nodiscard]] SurfacePairView view() const { return { mortar_.view(), nonmortar_.view() }; }
  [[nodiscard]] const SurfaceStorage& mortar() const { return mortar_; }
  [[nodiscard]] const SurfaceStorage& nonmortar() const { return nonmortar_; }

 private:
  SurfaceStorage mortar_;
  SurfaceStorage nonmortar_;
};

}  // namespace detail
}  // namespace tribol::mfem

#endif
