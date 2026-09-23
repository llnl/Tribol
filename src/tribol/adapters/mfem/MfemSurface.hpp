#ifndef TRIBOL_ADAPTERS_MFEM_MFEMSURFACE_HPP_
#define TRIBOL_ADAPTERS_MFEM_MFEMSURFACE_HPP_

#include "tribol/adapters/mfem/MfemDistribution.hpp"
#include "tribol/adapters/mfem/MfemParentTrace.hpp"

#include "mfem.hpp"
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
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

struct SurfaceDiscretization {
  int subdivision_factor{};
};

namespace detail {

inline MPI_Datatype realMpiType() { return std::is_same_v<Real, float> ? MPI_FLOAT : MPI_DOUBLE; }

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
                  const std::vector<int>& selected_attributes, SurfaceDiscretization discretization = {} )
      : mesh_( mesh ),
        selected_attributes_( selected_attributes ),
        dimension_( mesh.SpaceDimension() ),
        subdivision_factor_( discretization.subdivision_factor )
  {
    if ( subdivision_factor_ < 0 ) {
      throw std::invalid_argument( "MFEM surface subdivision factor cannot be negative." );
    }
    MPI_Comm_rank( mesh.GetComm(), &rank_ );
    destinations_ = localDestination( mesh_.GetComm() );
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

  [[nodiscard]] ParentTraceView parentTraceView() const { return parent_trace_.view( view() ); }

  [[nodiscard]] bool isLocallyOwned( std::size_t node ) const { return owner_ranks_[node] == rank_; }

  [[nodiscard]] const RestrictionNode& localRestriction( std::size_t node ) const
  {
    return local_restrictions_[static_cast<std::size_t>( owner_local_nodes_[node] )];
  }

  [[nodiscard]] const RestrictionNode& localRestrictionByIndex( std::size_t node ) const
  {
    return local_restrictions_[node];
  }

  [[nodiscard]] std::size_t localNodeCount() const { return local_restrictions_.size(); }
  [[nodiscard]] std::size_t localElementCount() const { return local_boundary_elements_.size(); }
  [[nodiscard]] const ::mfem::ParMesh& mesh() const { return mesh_; }

  [[nodiscard]] search::detail::BoundingBox localExpandedBounds( DistributionParameters parameters, bool& valid ) const
  {
    search::detail::BoundingBox result;
    valid = !local_boundary_elements_.empty();
    if ( !valid ) {
      return result;
    }
    std::size_t metadata_offset{};
    std::size_t coordinate_offset{};
    bool first_element{ true };
    while ( metadata_offset < local_metadata_.size() ) {
      const auto header = readSurfacePacketHeader( local_metadata_, metadata_offset );
      search::detail::BoundingBox element;
      Real longest_extent{};
      for ( int component = 0; component < dimension_; ++component ) {
        element.minimum[component] = local_coordinates_[coordinate_offset + component];
        element.maximum[component] = element.minimum[component];
        for ( int node = 1; node < header.node_count; ++node ) {
          const Real value =
              local_coordinates_[coordinate_offset + static_cast<std::size_t>( node * dimension_ + component )];
          element.minimum[component] = std::min( element.minimum[component], value );
          element.maximum[component] = std::max( element.maximum[component], value );
        }
        longest_extent = std::max( longest_extent, element.maximum[component] - element.minimum[component] );
      }
      const Real expansion = parameters.expansion + parameters.proximity_scale * longest_extent;
      for ( int component = 0; component < dimension_; ++component ) {
        element.minimum[component] -= expansion;
        element.maximum[component] += expansion;
        if ( first_element ) {
          result.minimum[component] = element.minimum[component];
          result.maximum[component] = element.maximum[component];
        } else {
          result.minimum[component] = std::min( result.minimum[component], element.minimum[component] );
          result.maximum[component] = std::max( result.maximum[component], element.maximum[component] );
        }
      }
      first_element = false;
      metadata_offset += static_cast<std::size_t>( header.node_count );
      coordinate_offset += static_cast<std::size_t>( header.node_count * dimension_ );
    }
    return result;
  }

  void redistribute( std::vector<unsigned char> destinations )
  {
    destinations_ = std::move( destinations );
    auto metadata = exchangeToDestinations( mesh_.GetComm(), local_metadata_, destinations_ );
    auto coordinates = exchangeToDestinations( mesh_.GetComm(), local_coordinates_, destinations_ );
    auto trace_data = exchangeToDestinations( mesh_.GetComm(), local_trace_data_, destinations_ );
    coordinates_ = std::move( coordinates.values );
    received_node_counts_.resize( coordinates.receive_counts.size() );
    for ( std::size_t owner = 0; owner < coordinates.receive_counts.size(); ++owner ) {
      if ( coordinates.receive_counts[owner] % dimension_ != 0 ) {
        throw std::logic_error( "MFEM distributed coordinate packet has an invalid size." );
      }
      received_node_counts_[owner] = coordinates.receive_counts[owner] / dimension_;
    }
    rebuildTopology( metadata.values, metadata.receive_counts, trace_data.values, trace_data.receive_counts );
  }

  [[nodiscard]] std::vector<Real> distributeNodeValues( const std::vector<Real>& local, int components ) const
  {
    if ( local.size() != localNodeCount() * static_cast<std::size_t>( components ) ) {
      throw std::logic_error( "MFEM local nodal packet has an invalid size." );
    }
    return exchangeToDestinations( mesh_.GetComm(), local, destinations_ ).values;
  }

  [[nodiscard]] std::vector<Real> distributeElementValues( const std::vector<Real>& local ) const
  {
    if ( local.size() != localElementCount() ) {
      throw std::logic_error( "MFEM local element packet has an invalid size." );
    }
    return exchangeToDestinations( mesh_.GetComm(), local, destinations_ ).values;
  }

  [[nodiscard]] std::vector<Real> reduceNodeValues( ArrayView<const Real> distributed, int components ) const
  {
    const auto local_values = localNodeCount() * static_cast<std::size_t>( components );
    if ( distributed.size() != static_cast<Index>( coordinates_.size() / dimension_ * components ) ) {
      throw std::invalid_argument( "MFEM distributed nodal field has an invalid size." );
    }
    int communicator_size{};
    MPI_Comm_size( mesh_.GetComm(), &communicator_size );
    std::vector<int> send_counts( static_cast<std::size_t>( communicator_size ) );
    std::vector<int> receive_counts( static_cast<std::size_t>( communicator_size ) );
    std::vector<int> send_displacements( static_cast<std::size_t>( communicator_size ) );
    int send_total{};
    for ( int owner = 0; owner < communicator_size; ++owner ) {
      send_displacements[static_cast<std::size_t>( owner )] = send_total;
      send_counts[static_cast<std::size_t>( owner )] =
          received_node_counts_[static_cast<std::size_t>( owner )] * components;
      send_total += send_counts[static_cast<std::size_t>( owner )];
    }
    MPI_Alltoall( send_counts.data(), 1, MPI_INT, receive_counts.data(), 1, MPI_INT, mesh_.GetComm() );
    std::vector<int> receive_displacements( static_cast<std::size_t>( communicator_size ) );
    int receive_total{};
    for ( int source = 0; source < communicator_size; ++source ) {
      receive_displacements[static_cast<std::size_t>( source )] = receive_total;
      receive_total += receive_counts[static_cast<std::size_t>( source )];
    }
    std::vector<Real> received( static_cast<std::size_t>( receive_total ) );
    MPI_Alltoallv( distributed.data(), send_counts.data(), send_displacements.data(), realMpiType(), received.data(),
                   receive_counts.data(), receive_displacements.data(), realMpiType(), mesh_.GetComm() );
    std::vector<Real> result( local_values );
    for ( int source = 0; source < communicator_size; ++source ) {
      const int count = receive_counts[static_cast<std::size_t>( source )];
      if ( count != 0 && count != static_cast<int>( local_values ) ) {
        throw std::logic_error( "MFEM reverse ghost packet has an invalid size." );
      }
      const int offset = receive_displacements[static_cast<std::size_t>( source )];
      for ( int value = 0; value < count; ++value ) {
        result[static_cast<std::size_t>( value )] += received[static_cast<std::size_t>( offset + value )];
      }
    }
    return result;
  }

  void sampleElementCoefficient( ::mfem::Coefficient& coefficient, std::vector<Real>& values ) const
  {
    std::vector<Real> local( localElementCount() );
    for ( std::size_t element = 0; element < localElementCount(); ++element ) {
      const int boundary_element = local_boundary_elements_[element];
      auto* transformation = const_cast<::mfem::ParMesh&>( mesh_ ).GetBdrElementTransformation( boundary_element );
      const auto& point = ::mfem::Geometries.GetCenter( mesh_.GetBdrElementBaseGeometry( boundary_element ) );
      local[element] = coefficient.Eval( *transformation, point );
    }
    values = distributeElementValues( local );
  }

  void computeElementThickness( std::vector<Real>& values ) const
  {
    std::vector<Real> local( localElementCount() );
    auto& mutable_mesh = const_cast<::mfem::ParMesh&>( mesh_ );
    for ( std::size_t element = 0; element < localElementCount(); ++element ) {
      int volume_element{};
      int orientation{};
      mutable_mesh.GetBdrElementAdjacentElement( local_boundary_elements_[element], volume_element, orientation );
      local[element] = mutable_mesh.GetElementSize( volume_element, 1 );
    }
    values = distributeElementValues( local );
  }

 private:
  void rebuild( const ::mfem::ParGridFunction& coordinate_field )
  {
    const auto* space = coordinate_field.ParFESpace();
    if ( space == nullptr || space->GetParMesh() != &mesh_ || space->GetVDim() != dimension_ ) {
      throw std::invalid_argument( "The MFEM adapter requires a vector coordinate field on the supplied mesh." );
    }
    std::vector<std::int64_t> local_metadata;
    std::vector<Real> local_coordinates;
    std::vector<Real> local_trace_data;
    local_restrictions_.clear();
    local_boundary_elements_.clear();
    for ( int boundary_element = 0; boundary_element < mesh_.GetNBE(); ++boundary_element ) {
      const int attribute = mesh_.GetBdrAttribute( boundary_element );
      if ( contains( selected_attributes_, attribute ) ) {
        appendRefinedBoundaryElement( *space, coordinate_field, boundary_element, attribute, local_metadata,
                                      local_coordinates, local_trace_data );
      }
    }
    local_metadata_ = std::move( local_metadata );
    local_coordinates_ = std::move( local_coordinates );
    local_trace_data_ = std::move( local_trace_data );
    redistribute( destinations_ );
  }

  void rebuildTopology( const std::vector<std::int64_t>& metadata, const std::vector<int>& metadata_counts,
                        const std::vector<Real>& trace_data, const std::vector<int>& trace_data_counts )
  {
    offsets_.assign( 1, 0 );
    connectivity_.clear();
    topologies_.clear();
    attributes_.clear();
    owner_local_nodes_.clear();
    owner_ranks_.clear();
    parent_trace_.clear();
    std::size_t metadata_offset{};
    std::size_t coordinate_offset{};
    std::size_t trace_data_offset{};
    const int communicator_size = static_cast<int>( metadata_counts.size() );
    if ( trace_data_counts.size() != metadata_counts.size() ) {
      throw std::logic_error( "MFEM distributed parent-trace packet counts are inconsistent." );
    }
    received_element_counts_.assign( metadata_counts.size(), 0 );
    for ( int owner = 0; owner < communicator_size; ++owner ) {
      const std::size_t owner_end = metadata_offset + metadata_counts[static_cast<std::size_t>( owner )];
      const std::size_t trace_data_end = trace_data_offset + trace_data_counts[static_cast<std::size_t>( owner )];
      while ( metadata_offset < owner_end ) {
        const auto header = readSurfacePacketHeader( metadata, metadata_offset );
        topologies_.push_back( header.child_topology );
        attributes_.push_back( header.attribute );
        ++received_element_counts_[static_cast<std::size_t>( owner )];
        const std::size_t coefficient_count =
            static_cast<std::size_t>( header.parent_dof_count * header.parent_dof_count );
        const Index parent_face = parent_trace_.findOrAppendFace(
            owner, header, metadata,
            { trace_data.data() + trace_data_offset, static_cast<Index>( coefficient_count ) } );
        trace_data_offset += coefficient_count;
        parent_trace_.appendSubfacet( parent_face );
        for ( int local_node = 0; local_node < header.node_count; ++local_node ) {
          owner_local_nodes_.push_back( static_cast<int>( metadata[metadata_offset++] ) );
          owner_ranks_.push_back( owner );
          connectivity_.push_back( static_cast<Index>( connectivity_.size() ) );
          coordinate_offset += static_cast<std::size_t>( dimension_ );
          for ( int component = 0; component < dimension_ - 1; ++component ) {
            parent_trace_.appendReferenceCoordinate( trace_data[trace_data_offset++] );
          }
        }
        offsets_.push_back( static_cast<Index>( connectivity_.size() ) );
      }
      if ( trace_data_offset != trace_data_end ) {
        throw std::logic_error( "MFEM distributed parent-trace packet has an invalid size." );
      }
    }
    if ( coordinate_offset != coordinates_.size() ) {
      throw std::logic_error( "MFEM distributed surface metadata and coordinates are inconsistent." );
    }
  }

#include "tribol/adapters/mfem/MfemSurfaceConstruction.inl"

  const ::mfem::ParMesh& mesh_;
  std::vector<int> selected_attributes_;
  int dimension_{};
  int rank_{};
  int subdivision_factor_{};
  std::vector<unsigned char> destinations_;
  std::vector<std::int64_t> local_metadata_;
  std::vector<Real> local_coordinates_;
  std::vector<Real> local_trace_data_;
  std::vector<int> local_boundary_elements_;
  std::vector<int> received_node_counts_;
  std::vector<int> received_element_counts_;
  std::vector<Real> coordinates_;
  std::vector<Index> offsets_;
  std::vector<Index> connectivity_;
  std::vector<ElementTopology> topologies_;
  std::vector<int> attributes_;
  std::vector<int> owner_local_nodes_;
  std::vector<int> owner_ranks_;
  std::vector<RestrictionNode> local_restrictions_;
  ParentTraceStorage parent_trace_;
};

class SurfacePairStorage {
 public:
  SurfacePairStorage( const ::mfem::ParMesh& mesh, const ::mfem::ParGridFunction& coordinates,
                      const PairedBoundaryAttributes& attributes, SurfaceDiscretization discretization = {},
                      DistributionParameters distribution = {} )
      : mortar_( mesh, coordinates, attributes.mortar, discretization ),
        nonmortar_( mesh, coordinates, attributes.nonmortar, discretization ),
        distribution_( distribution )
  {
    validateGlobalSurface( mortar_ );
    validateGlobalSurface( nonmortar_ );
    redistributeNonmortar();
  }

  void updateCoordinates( const ::mfem::ParGridFunction& coordinates )
  {
    mortar_.updateCoordinates( coordinates );
    nonmortar_.updateCoordinates( coordinates );
  }

  void rebuildCoordinates( const ::mfem::ParGridFunction& coordinates )
  {
    mortar_.updateCoordinates( coordinates );
    nonmortar_.updateCoordinates( coordinates );
    redistributeNonmortar();
  }

  [[nodiscard]] SurfacePairView view() const { return { mortar_.view(), nonmortar_.view() }; }
  [[nodiscard]] ParentTracePairView parentTraceView() const
  {
    return { mortar_.parentTraceView(), nonmortar_.parentTraceView() };
  }
  [[nodiscard]] const SurfaceStorage& mortar() const { return mortar_; }
  [[nodiscard]] const SurfaceStorage& nonmortar() const { return nonmortar_; }

 private:
  void validateGlobalSurface( const SurfaceStorage& surface ) const
  {
    std::uint64_t local = surface.localElementCount();
    std::uint64_t global{};
    MPI_Allreduce( &local, &global, 1, MPI_UINT64_T, MPI_SUM, mesh().GetComm() );
    if ( global == 0 ) {
      throw std::invalid_argument( "Selected MFEM boundary attributes contain no boundary elements." );
    }
  }

  [[nodiscard]] const ::mfem::ParMesh& mesh() const { return mortar_.mesh(); }

  void redistributeNonmortar()
  {
    bool nonmortar_valid{};
    const auto nonmortar_bounds = nonmortar_.localExpandedBounds( distribution_, nonmortar_valid );
    nonmortar_.redistribute(
        ghostDestinations( mesh().GetComm(), mortar_.view(), nonmortar_bounds, nonmortar_valid, distribution_ ) );
  }

  SurfaceStorage mortar_;
  SurfaceStorage nonmortar_;
  DistributionParameters distribution_;
};

}  // namespace detail
}  // namespace tribol::mfem

#endif
