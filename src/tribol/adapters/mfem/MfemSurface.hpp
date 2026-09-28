#ifndef TRIBOL_ADAPTERS_MFEM_MFEMSURFACE_HPP_
#define TRIBOL_ADAPTERS_MFEM_MFEMSURFACE_HPP_

#include "tribol/adapters/mfem/MfemDistribution.hpp"
#include "tribol/adapters/mfem/MfemParentTrace.hpp"
#include "tribol/core/ConstraintRowView.hpp"

#include "mfem.hpp"
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <map>
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

  [[nodiscard]] FieldView<const Real> parentCoordinateView() const
  {
    return { { parent_coordinates_.data(), static_cast<Index>( parent_coordinates_.size() ) },
             parent_trace_.numberOfDofs(),
             dimension_,
             FieldLayout::Interleaved };
  }

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
  [[nodiscard]] std::size_t localParentDofCount() const { return local_parent_dofs_.size(); }
  [[nodiscard]] int localParentDof( std::size_t parent_dof ) const { return local_parent_dofs_[parent_dof]; }
  [[nodiscard]] Index parentDofCount() const { return parent_trace_.numberOfDofs(); }
  [[nodiscard]] std::size_t localElementCount() const { return local_boundary_elements_.size(); }
  [[nodiscard]] const ::mfem::ParMesh& mesh() const { return mesh_; }

  void distributeParentDofField( const ::mfem::ParFiniteElementSpace& space, const ::mfem::Vector& local_field,
                                 int components, bool apply_vector_dof_sign, std::vector<Real>& values ) const;

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
    auto parent_coordinates = exchangeToDestinations( mesh_.GetComm(), local_parent_coordinates_, destinations_ );
    coordinates_ = std::move( coordinates.values );
    parent_coordinates_ = std::move( parent_coordinates.values );
    received_node_counts_.resize( coordinates.receive_counts.size() );
    for ( std::size_t owner = 0; owner < coordinates.receive_counts.size(); ++owner ) {
      if ( coordinates.receive_counts[owner] % dimension_ != 0 ) {
        throw std::logic_error( "MFEM distributed coordinate packet has an invalid size." );
      }
      received_node_counts_[owner] = coordinates.receive_counts[owner] / dimension_;
    }
    received_parent_dof_counts_.resize( parent_coordinates.receive_counts.size() );
    for ( std::size_t owner = 0; owner < parent_coordinates.receive_counts.size(); ++owner ) {
      if ( parent_coordinates.receive_counts[owner] % dimension_ != 0 ) {
        throw std::logic_error( "MFEM distributed parent-coordinate packet has an invalid size." );
      }
      received_parent_dof_counts_[owner] = parent_coordinates.receive_counts[owner] / dimension_;
    }
    rebuildTopology( metadata.values, metadata.receive_counts, trace_data.values, trace_data.receive_counts );
    if ( parent_coordinates_.size() != static_cast<std::size_t>( parent_trace_.numberOfDofs() * dimension_ ) ) {
      throw std::logic_error( "MFEM distributed parent coordinates do not match parent-DOF identities." );
    }
    prepareExchangeWorkspaces();
  }

  [[nodiscard]] ArrayView<Real> localNodeValues( int components ) const
  {
    if ( components == 1 ) {
      return node_scalar_exchange_.localValues();
    }
    if ( components == dimension_ ) {
      return node_vector_exchange_.localValues();
    }
    throw std::invalid_argument( "MFEM nodal exchange requires a scalar or spatial vector field." );
  }

  void distributeNodeValues( int components, std::vector<Real>& values ) const
  {
    if ( components == 1 ) {
      node_scalar_exchange_.distribute( mesh_.GetComm(), values );
      return;
    }
    if ( components == dimension_ ) {
      node_vector_exchange_.distribute( mesh_.GetComm(), values );
      return;
    }
    throw std::invalid_argument( "MFEM nodal exchange requires a scalar or spatial vector field." );
  }

  [[nodiscard]] ArrayView<const Real> reduceNodeValues( ArrayView<const Real> distributed, int components ) const
  {
    if ( components == 1 ) {
      return node_scalar_exchange_.reduce( mesh_.GetComm(), distributed );
    }
    if ( components == dimension_ ) {
      return node_vector_exchange_.reduce( mesh_.GetComm(), distributed );
    }
    throw std::invalid_argument( "MFEM nodal exchange requires a scalar or spatial vector field." );
  }

  [[nodiscard]] ArrayView<const Real> reduceParentDofValues( ArrayView<const Real> distributed, int components ) const;

  void sampleElementCoefficient( ::mfem::Coefficient& coefficient, std::vector<Real>& values ) const
  {
    auto local = element_scalar_exchange_.localValues();
    for ( std::size_t element = 0; element < localElementCount(); ++element ) {
      const int boundary_element = local_boundary_elements_[element];
      const auto& point = ::mfem::Geometries.GetCenter( mesh_.GetBdrElementBaseGeometry( boundary_element ) );
      local[static_cast<Index>( element )] = coefficient.Eval( local_element_transformations_[element], point );
    }
    element_scalar_exchange_.distribute( mesh_.GetComm(), values );
  }

  void computeElementThickness( std::vector<Real>& values ) const
  {
    auto local = element_scalar_exchange_.localValues();
    for ( std::size_t element = 0; element < localElementCount(); ++element ) {
      local[static_cast<Index>( element )] = local_element_thickness_[element];
    }
    element_scalar_exchange_.distribute( mesh_.GetComm(), values );
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
    local_parent_dof_indices_.clear();
    local_parent_dofs_.clear();
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
    collectLocalParentField( *space, coordinate_field, dimension_, true, local_parent_coordinates_ );
    updateLocalElementData();
    redistribute( destinations_ );
  }

  void updateLocalElementData()
  {
    local_element_thickness_.resize( localElementCount() );
    local_element_transformations_.resize( localElementCount() );
    auto& mutable_mesh = const_cast<::mfem::ParMesh&>( mesh_ );
    for ( std::size_t element = 0; element < localElementCount(); ++element ) {
      int volume_element{};
      int orientation{};
      const int boundary_element = local_boundary_elements_[element];
      mutable_mesh.GetBdrElementAdjacentElement( boundary_element, volume_element, orientation );
      local_element_thickness_[element] = mutable_mesh.GetElementSize( volume_element, 1 );
      mesh_.GetBdrElementTransformation( boundary_element, &local_element_transformations_[element] );
    }
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

  void prepareExchangeWorkspaces()
  {
    node_scalar_exchange_.configure( destinations_, localNodeCount(), received_node_counts_, 1 );
    node_vector_exchange_.configure( destinations_, localNodeCount(), received_node_counts_, dimension_ );
    parent_vector_exchange_.configure( destinations_, localParentDofCount(), received_parent_dof_counts_, dimension_ );
    element_scalar_exchange_.configure( destinations_, localElementCount(), received_element_counts_, 1 );
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
  std::vector<Real> local_parent_coordinates_;
  std::vector<Real> local_element_thickness_;
  mutable std::vector<::mfem::IsoparametricTransformation> local_element_transformations_;
  std::vector<int> local_boundary_elements_;
  std::map<int, Index> local_parent_dof_indices_;
  std::vector<int> local_parent_dofs_;
  std::vector<int> received_node_counts_;
  std::vector<int> received_parent_dof_counts_;
  std::vector<int> received_element_counts_;
  std::vector<Real> coordinates_;
  std::vector<Real> parent_coordinates_;
  std::vector<Index> offsets_;
  std::vector<Index> connectivity_;
  std::vector<ElementTopology> topologies_;
  std::vector<int> attributes_;
  std::vector<int> owner_local_nodes_;
  std::vector<int> owner_ranks_;
  std::vector<RestrictionNode> local_restrictions_;
  ParentTraceStorage parent_trace_;
  mutable ExchangeWorkspace<Real> node_scalar_exchange_;
  mutable ExchangeWorkspace<Real> node_vector_exchange_;
  mutable ExchangeWorkspace<Real> parent_vector_exchange_;
  mutable ExchangeWorkspace<Real> element_scalar_exchange_;
};

#include "tribol/adapters/mfem/MfemParentFieldCommunication.inl"

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
  [[nodiscard]] ParentTraceFieldPairView parentFields() const
  {
    return { .mortar_coordinates = mortar_.parentCoordinateView(),
             .nonmortar_coordinates = nonmortar_.parentCoordinateView() };
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
