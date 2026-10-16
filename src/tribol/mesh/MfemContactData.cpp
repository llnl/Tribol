// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MfemContactData.hpp"

#ifdef BUILD_REDECOMP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "axom/slic.hpp"
#include "axom/primal.hpp"
#include "axom/spin.hpp"
#include "shared/math/ParSparseMat.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"

namespace tribol {

namespace {

struct FaceKey {
  int rank;
  int element;

  auto Tie() const { return std::tie( rank, element ); }
  bool operator<( const FaceKey& other ) const { return Tie() < other.Tie(); }
  bool operator==( const FaceKey& other ) const { return Tie() == other.Tie(); }
};

struct FaceBoundsRecord {
  int surface;
  FaceKey key;
  long long global_face;
  std::array<double, 3> minimum;
  std::array<double, 3> maximum;
};

struct FacePairRecord {
  FaceKey first;
  FaceKey second;
  int spatial_rank;
};

struct FaceMetadata {
  FaceKey key;
  long long global_face;
  int geometry;
  int order;
  int refinement_factor;
  int basis_type;
  int num_vertices;
  static constexpr int max_vertices = 4;
  std::array<long long, max_vertices> global_vertices;
};

enum class VirtualNodeKind
{
  VERTEX,
  EDGE_INTERIOR,
  FACE_INTERIOR
};

struct NodeKey {
  VirtualNodeKind kind;
  std::array<long long, 5> values;

  auto Tie() const { return std::tie( kind, values ); }
  bool operator<( const NodeKey& other ) const { return Tie() < other.Tie(); }
};

bool UsesNonmortarAnchor( ContactMethod contact_method )
{
  switch ( contact_method ) {
    case SINGLE_MORTAR:
    case ALIGNED_MORTAR:
    case MORTAR_WEIGHTS:
    case ENERGY_MORTAR:
      return true;
    default:
      return false;
  }
}

bool UsesMultiplierSpace( ContactMethod contact_method, EnforcementMethod enforcement_method )
{
  if ( enforcement_method != LAGRANGE_MULTIPLIER ) return false;
  return contact_method == SINGLE_MORTAR || contact_method == ALIGNED_MORTAR || contact_method == ENERGY_MORTAR;
}

template <typename T>
std::vector<std::vector<T>> ExchangeRecordsByRank( MPI_Comm communicator,
                                                   const std::vector<std::vector<T>>& records_by_destination )
{
  int num_ranks = 1;
  MPI_Comm_size( communicator, &num_ranks );
  SLIC_ERROR_ROOT_IF( static_cast<int>( records_by_destination.size() ) != num_ranks,
                      "Distributed record exchange requires one destination list per rank." );

  std::vector<int> send_counts( static_cast<size_t>( num_ranks ) );
  std::vector<int> receive_counts( static_cast<size_t>( num_ranks ) );
  for ( int rank = 0; rank < num_ranks; ++rank ) {
    send_counts[rank] = static_cast<int>( records_by_destination[rank].size() );
  }
  MPI_Alltoall( send_counts.data(), 1, MPI_INT, receive_counts.data(), 1, MPI_INT, communicator );

  std::vector<int> send_displacements( static_cast<size_t>( num_ranks ), 0 );
  std::vector<int> receive_displacements( static_cast<size_t>( num_ranks ), 0 );
  for ( int rank = 1; rank < num_ranks; ++rank ) {
    send_displacements[rank] = send_displacements[rank - 1] + send_counts[rank - 1];
    receive_displacements[rank] = receive_displacements[rank - 1] + receive_counts[rank - 1];
  }
  const int send_size = send_displacements.back() + send_counts.back();
  const int receive_size = receive_displacements.back() + receive_counts.back();
  std::vector<T> send_buffer( static_cast<size_t>( send_size ) );
  std::vector<T> receive_buffer( static_cast<size_t>( receive_size ) );
  for ( int rank = 0; rank < num_ranks; ++rank ) {
    std::copy( records_by_destination[rank].begin(), records_by_destination[rank].end(),
               send_buffer.begin() + send_displacements[rank] );
  }

  MPI_Datatype record_type;
  MPI_Type_contiguous( static_cast<int>( sizeof( T ) ), MPI_BYTE, &record_type );
  MPI_Type_commit( &record_type );
  MPI_Alltoallv( send_buffer.data(), send_counts.data(), send_displacements.data(), record_type, receive_buffer.data(),
                 receive_counts.data(), receive_displacements.data(), record_type, communicator );
  MPI_Type_free( &record_type );

  std::vector<std::vector<T>> records_by_source( static_cast<size_t>( num_ranks ) );
  for ( int rank = 0; rank < num_ranks; ++rank ) {
    const auto begin = receive_buffer.begin() + receive_displacements[rank];
    records_by_source[rank].assign( begin, begin + receive_counts[rank] );
  }
  return records_by_source;
}

template <typename T>
std::vector<T> FlattenByRank( const std::vector<std::vector<T>>& records_by_rank )
{
  std::vector<T> records;
  for ( const auto& rank_records : records_by_rank ) {
    records.insert( records.end(), rank_records.begin(), rank_records.end() );
  }
  return records;
}

template <int Dimension>
std::vector<std::pair<int, int>> FindIntersectingBoundsWithBvh( const std::vector<FaceBoundsRecord>& first_bounds,
                                                                const std::vector<FaceBoundsRecord>& second_bounds )
{
  using Bvh = axom::spin::BVH<Dimension, axom::SEQ_EXEC, double>;
  using Box = typename Bvh::BoxType;
  using Point = typename Bvh::PointType;
  if ( first_bounds.empty() || second_bounds.empty() ) return {};

  axom::Array<Box> first_boxes( first_bounds.size(), first_bounds.size() );
  axom::Array<Box> second_boxes( second_bounds.size(), second_bounds.size() );
  const auto fill_boxes = []( const auto& records, auto& boxes ) {
    for ( int record = 0; record < static_cast<int>( records.size() ); ++record ) {
      Point minimum;
      Point maximum;
      for ( int dimension = 0; dimension < Dimension; ++dimension ) {
        minimum[dimension] = records[record].minimum[dimension];
        maximum[dimension] = records[record].maximum[dimension];
      }
      boxes[record] = Box( minimum, maximum );
    }
  };
  fill_boxes( first_bounds, first_boxes );
  fill_boxes( second_bounds, second_boxes );

  Bvh bvh;
  bvh.initialize( first_boxes.view(), first_boxes.size() );
  axom::Array<axom::IndexType> offsets( second_boxes.size(), second_boxes.size() );
  axom::Array<axom::IndexType> counts( second_boxes.size(), second_boxes.size() );
  axom::Array<axom::IndexType> candidates;
  bvh.findBoundingBoxes( offsets.view(), counts.view(), candidates, second_boxes.size(), second_boxes.view() );

  std::vector<std::pair<int, int>> intersections;
  for ( int second = 0; second < second_boxes.size(); ++second ) {
    for ( axom::IndexType candidate = offsets[second]; candidate < offsets[second] + counts[second]; ++candidate ) {
      intersections.emplace_back( candidates[candidate], second );
    }
  }
  return intersections;
}

int ChooseSpatialAxis( const std::array<double, 3>& global_minimum, const std::array<double, 3>& global_maximum,
                       int space_dimension )
{
  int axis = 0;
  for ( int dimension = 1; dimension < space_dimension; ++dimension ) {
    if ( global_maximum[dimension] - global_minimum[dimension] > global_maximum[axis] - global_minimum[axis] ) {
      axis = dimension;
    }
  }
  return axis;
}

int DomainForCoordinate( double coordinate, double minimum, double maximum, int domain_count )
{
  if ( domain_count <= 1 || maximum <= minimum ) {
    return 0;
  }
  const double normalized = ( coordinate - minimum ) / ( maximum - minimum );
  return std::clamp( static_cast<int>( normalized * domain_count ), 0, domain_count - 1 );
}

int VerticesPerElement( InterfaceElementType element_type )
{
  switch ( element_type ) {
    case LINEAR_EDGE:
      return 2;
    case LINEAR_TRIANGLE:
      return 3;
    case LINEAR_QUAD:
      return 4;
    default:
      SLIC_ERROR_ROOT( "The MFEM contact patch contains an unsupported Tribol element type." );
      return 0;
  }
}

InterfaceElementType ToTribolElementType( mfem::Geometry::Type geometry )
{
  switch ( geometry ) {
    case mfem::Geometry::SEGMENT:
      return LINEAR_EDGE;
    case mfem::Geometry::TRIANGLE:
      return LINEAR_TRIANGLE;
    case mfem::Geometry::SQUARE:
      return LINEAR_QUAD;
    default:
      SLIC_ERROR_ROOT( "MFEM contact supports segment, triangle, and quadrilateral boundary elements." );
      return LINEAR_EDGE;
  }
}

int RefinementFactor( const MfemContactOptions& options, const mfem::FiniteElement& boundary_element )
{
  return options.refinement_factor > 0 ? options.refinement_factor : std::max( 1, boundary_element.GetOrder() );
}

const mfem::H1_FECollection& GetH1Collection( const mfem::ParFiniteElementSpace& space, const char* field_name )
{
  const auto* collection = dynamic_cast<const mfem::H1_FECollection*>( space.FEColl() );
  SLIC_ERROR_ROOT_IF( collection == nullptr, field_name << " must use a conforming nodal H1 finite element space." );
  return *collection;
}

void ExtractBoundaryElementValues( const mfem::ParGridFunction& field, int boundary_element,
                                   mfem::Vector& element_values )
{
  mfem::Array<int> boundary_vdofs;
  mfem::DofTransformation orientation;
  field.ParFESpace()->GetBdrElementVDofs( boundary_element, boundary_vdofs, orientation );
  element_values.SetSize( boundary_vdofs.Size() );
  field.GetSubVector( boundary_vdofs, element_values );
  orientation.InvTransformPrimal( element_values );
}

mfem::Vector EvaluateVirtualFace( const mfem::ParGridFunction& field, int boundary_element, int refinement_factor )
{
  const auto& space = *field.ParFESpace();
  const auto& element = *space.GetBE( boundary_element );
  redecomp::VirtualRefinementOperator refinement( element, space.GetVDim(), refinement_factor );
  mfem::Vector element_values;
  ExtractBoundaryElementValues( field, boundary_element, element_values );
  mfem::Vector virtual_values( refinement.Height() );
  refinement.Mult( element_values, virtual_values );
  return virtual_values;
}

FaceBoundsRecord ComputeFaceBounds( const MfemContactSurface& surface, int local_surface_element, int boundary_element,
                                    int refinement_factor, RealT inflation, long long global_face, int rank )
{
  const auto virtual_coordinates = EvaluateVirtualFace( surface.Coordinates(), boundary_element, refinement_factor );
  const int space_dimension = surface.Mesh().SpaceDimension();
  const int num_occurrences = virtual_coordinates.Size() / space_dimension;
  FaceBoundsRecord bounds{};
  bounds.key = { rank, local_surface_element };
  bounds.global_face = global_face;
  for ( int dimension = 0; dimension < 3; ++dimension ) {
    bounds.minimum[dimension] = std::numeric_limits<double>::max();
    bounds.maximum[dimension] = std::numeric_limits<double>::lowest();
  }
  for ( int dimension = 0; dimension < space_dimension; ++dimension ) {
    for ( int occurrence = 0; occurrence < num_occurrences; ++occurrence ) {
      const double coordinate = virtual_coordinates[occurrence + dimension * num_occurrences];
      bounds.minimum[dimension] = std::min( bounds.minimum[dimension], coordinate - inflation );
      bounds.maximum[dimension] = std::max( bounds.maximum[dimension], coordinate + inflation );
    }
  }
  return bounds;
}

double ComputeMaximumChildDiameter( const mfem::ParGridFunction& coordinates, int boundary_element,
                                    int refinement_factor )
{
  const auto virtual_coordinates = EvaluateVirtualFace( coordinates, boundary_element, refinement_factor );
  const auto& boundary_fe = *coordinates.ParFESpace()->GetBE( boundary_element );
  redecomp::VirtualRefinementOperator refinement( boundary_fe, coordinates.VectorDim(), refinement_factor );
  const int space_dimension = coordinates.ParFESpace()->GetParMesh()->SpaceDimension();
  const int num_occurrences = virtual_coordinates.Size() / space_dimension;
  double maximum_diameter = 0.0;
  for ( int child = 0; child < refinement.NumChildren(); ++child ) {
    for ( int first = 0; first < refinement.VerticesPerChild(); ++first ) {
      for ( int second = first + 1; second < refinement.VerticesPerChild(); ++second ) {
        double squared_distance = 0.0;
        for ( int dimension = 0; dimension < space_dimension; ++dimension ) {
          const int first_occurrence = child * refinement.VerticesPerChild() + first;
          const int second_occurrence = child * refinement.VerticesPerChild() + second;
          const double difference = virtual_coordinates[first_occurrence + dimension * num_occurrences] -
                                    virtual_coordinates[second_occurrence + dimension * num_occurrences];
          squared_distance += difference * difference;
        }
        maximum_diameter = std::max( maximum_diameter, std::sqrt( squared_distance ) );
      }
    }
  }
  return maximum_diameter;
}

NodeKey VertexNodeKey( long long vertex ) { return { VirtualNodeKind::VERTEX, { vertex, 0, 0, 0, 0 } }; }

NodeKey EdgeNodeKey( long long first_vertex, long long second_vertex, int numerator, int denominator )
{
  if ( second_vertex < first_vertex ) {
    std::swap( first_vertex, second_vertex );
    numerator = denominator - numerator;
  }
  return { VirtualNodeKind::EDGE_INTERIOR, { first_vertex, second_vertex, numerator, denominator, 0 } };
}

NodeKey FaceInteriorNodeKey( long long global_face, int first_index, int second_index, int denominator )
{
  return { VirtualNodeKind::FACE_INTERIOR, { global_face, first_index, second_index, denominator, 0 } };
}

NodeKey BuildNodeKey( const FaceMetadata& face, const mfem::IntegrationPoint& point )
{
  const int factor = face.refinement_factor;
  const int first = static_cast<int>( std::llround( point.x * factor ) );
  const int second = static_cast<int>( std::llround( point.y * factor ) );
  const auto& vertices = face.global_vertices;
  const auto geometry = static_cast<mfem::Geometry::Type>( face.geometry );
  if ( geometry == mfem::Geometry::SEGMENT ) {
    if ( first == 0 ) return VertexNodeKey( vertices[0] );
    if ( first == factor ) return VertexNodeKey( vertices[1] );
    return EdgeNodeKey( vertices[0], vertices[1], first, factor );
  }
  if ( geometry == mfem::Geometry::TRIANGLE ) {
    if ( first == 0 && second == 0 ) return VertexNodeKey( vertices[0] );
    if ( first == factor && second == 0 ) return VertexNodeKey( vertices[1] );
    if ( first == 0 && second == factor ) return VertexNodeKey( vertices[2] );
    if ( second == 0 ) return EdgeNodeKey( vertices[0], vertices[1], first, factor );
    if ( first == 0 ) return EdgeNodeKey( vertices[0], vertices[2], second, factor );
    if ( first + second == factor ) return EdgeNodeKey( vertices[1], vertices[2], second, factor );
    return FaceInteriorNodeKey( face.global_face, first, second, factor );
  }
  if ( first == 0 && second == 0 ) return VertexNodeKey( vertices[0] );
  if ( first == factor && second == 0 ) return VertexNodeKey( vertices[1] );
  if ( first == factor && second == factor ) return VertexNodeKey( vertices[2] );
  if ( first == 0 && second == factor ) return VertexNodeKey( vertices[3] );
  if ( second == 0 ) return EdgeNodeKey( vertices[0], vertices[1], first, factor );
  if ( first == factor ) return EdgeNodeKey( vertices[1], vertices[2], second, factor );
  if ( second == factor ) return EdgeNodeKey( vertices[3], vertices[2], first, factor );
  if ( first == 0 ) return EdgeNodeKey( vertices[0], vertices[3], second, factor );
  return FaceInteriorNodeKey( face.global_face, first, second, factor );
}

std::vector<IndexT> BuildConnectivity( const std::vector<FaceMetadata>& faces,
                                       const redecomp::VirtualRefinementCollectionOperator& refinement,
                                       mfem::Array<int>& occurrence_to_node, int& num_nodes )
{
  occurrence_to_node.SetSize( refinement.NumOccurrences() );
  std::map<NodeKey, int> node_ids;
  std::vector<IndexT> connectivity;
  connectivity.reserve( static_cast<size_t>( refinement.NumOccurrences() ) );
  for ( int face = 0; face < refinement.NumFaces(); ++face ) {
    const auto& face_refinement = refinement.FaceRefinement( face );
    const auto& reference_points = face_refinement.ChildVertexReferencePoints();
    for ( int local_occurrence = 0; local_occurrence < reference_points.GetNPoints(); ++local_occurrence ) {
      const NodeKey key = BuildNodeKey( faces[face], reference_points.IntPoint( local_occurrence ) );
      const auto iterator = node_ids.emplace( key, static_cast<int>( node_ids.size() ) ).first;
      const int occurrence = refinement.OccurrenceOffsets()[face] + local_occurrence;
      occurrence_to_node[occurrence] = iterator->second;
      connectivity.push_back( iterator->second );
    }
  }
  num_nodes = static_cast<int>( node_ids.size() );
  return connectivity;
}

struct PatchBlockView {
  const MfemContactPatchView& patch;
  int vector_dimension;

  int NumDofs() const { return patch.NumNodes() * vector_dimension; }
};

shared::ParSparseMat BuildPatchJacobianMatrix( MPI_Comm communicator, const PatchBlockView& row_block,
                                               const PatchBlockView& column_block, const MethodData& method_data,
                                               BlockSpace row_space, BlockSpace column_space )
{
  const int row_space_index = static_cast<int>( row_space );
  const int column_space_index = static_cast<int>( column_space );
  const auto& element_matrices = method_data.getBlockJ()( row_space_index, column_space_index );
  const auto& row_elements = method_data.getBlockJElementIds()[row_space_index];
  const auto& column_elements = method_data.getBlockJElementIds()[column_space_index];
  SLIC_ERROR_ROOT_IF(
      element_matrices.size() != row_elements.size() || element_matrices.size() != column_elements.size(),
      "Patch Jacobian matrices and element IDs must have matching lengths." );

  const auto& row_patch = row_block.patch;
  const auto& column_patch = column_block.patch;
  const int local_rows = row_block.NumDofs();
  const int local_columns = column_block.NumDofs();
  int rank = 0;
  int num_ranks = 1;
  MPI_Comm_rank( communicator, &rank );
  MPI_Comm_size( communicator, &num_ranks );
  mfem::Array<int> row_sizes( num_ranks );
  mfem::Array<int> column_sizes( num_ranks );
  MPI_Allgather( &local_rows, 1, MPI_INT, row_sizes.GetData(), 1, MPI_INT, communicator );
  MPI_Allgather( &local_columns, 1, MPI_INT, column_sizes.GetData(), 1, MPI_INT, communicator );
  mfem::Array<HYPRE_BigInt> row_starts( num_ranks + 1 );
  mfem::Array<HYPRE_BigInt> column_starts( num_ranks + 1 );
  row_starts[0] = 0;
  column_starts[0] = 0;
  for ( int owner = 0; owner < num_ranks; ++owner ) {
    row_starts[owner + 1] = row_starts[owner] + row_sizes[owner];
    column_starts[owner + 1] = column_starts[owner] + column_sizes[owner];
  }

  std::vector<std::map<int, mfem::real_t>> rows( static_cast<size_t>( local_rows ) );
  const int row_vertices = row_patch.VerticesPerElement();
  const int column_vertices = column_patch.VerticesPerElement();
  for ( int contribution = 0; contribution < element_matrices.size(); ++contribution ) {
    const int row_element = row_elements[contribution];
    const int column_element = column_elements[contribution];
    SLIC_ERROR_ROOT_IF( row_element < 0 || row_element >= row_patch.NumElements() || column_element < 0 ||
                            column_element >= column_patch.NumElements(),
                        "Patch Jacobian contribution references an out-of-range child element." );
    const auto& element_matrix = element_matrices[contribution];
    SLIC_ERROR_ROOT_IF( element_matrix.Height() != row_vertices * row_block.vector_dimension ||
                            element_matrix.Width() != column_vertices * column_block.vector_dimension,
                        "Patch Jacobian contribution has an unexpected element block size." );
    for ( int element_row = 0; element_row < element_matrix.Height(); ++element_row ) {
      const int row_node_in_element = element_row / row_block.vector_dimension;
      const int row_component = element_row % row_block.vector_dimension;
      const int row_node = row_patch.connectivity[row_element * row_vertices + row_node_in_element];
      const int patch_row = row_node + row_component * row_patch.NumNodes();
      for ( int element_column = 0; element_column < element_matrix.Width(); ++element_column ) {
        const mfem::real_t value = element_matrix( element_row, element_column );
        if ( value == 0.0 ) continue;
        const int column_node_in_element = element_column / column_block.vector_dimension;
        const int column_component = element_column % column_block.vector_dimension;
        const int column_node = column_patch.connectivity[column_element * column_vertices + column_node_in_element];
        const int patch_column = column_node + column_component * column_patch.NumNodes();
        rows[patch_row][patch_column] += value;
      }
    }
  }

  HYPRE_Int nonzeros = 0;
  for ( const auto& row : rows ) nonzeros += static_cast<HYPRE_Int>( row.size() );
  auto* row_offsets = new HYPRE_Int[local_rows + 1];
  auto* global_columns = nonzeros > 0 ? new HYPRE_BigInt[nonzeros] : nullptr;
  auto* values = nonzeros > 0 ? new mfem::real_t[nonzeros] : nullptr;
  row_offsets[0] = 0;
  HYPRE_Int entry = 0;
  for ( int row = 0; row < local_rows; ++row ) {
    for ( const auto& [local_column, value] : rows[row] ) {
      global_columns[entry] = column_starts[rank] + local_column;
      values[entry] = value;
      ++entry;
    }
    row_offsets[row + 1] = entry;
  }
  HYPRE_BigInt local_row_partition[2]{ row_starts[rank], row_starts[rank + 1] };
  HYPRE_BigInt local_column_partition[2]{ column_starts[rank], column_starts[rank + 1] };
  return shared::ParSparseMat( communicator, local_rows, row_starts.Last(), column_starts.Last(), row_offsets,
                               global_columns, values, local_row_partition, local_column_partition );
}

}  // namespace

struct MfemContactData::SurfaceState {
  std::vector<int> boundary_elements;
  std::vector<FaceMetadata> local_metadata;
  std::vector<FaceMetadata> destination_metadata;
  std::vector<int> destination_face_child_offsets;
  std::unique_ptr<redecomp::ContactPatchOperator> patch_operator;
  std::unique_ptr<redecomp::ContactPatchOperator> velocity_operator;
  std::unique_ptr<redecomp::ContactPatchOperator> reference_coordinate_operator;
  std::unique_ptr<redecomp::ElementFieldExchange> thickness_exchange;
  std::unique_ptr<redecomp::ElementFieldExchange> modulus_exchange;
  std::unique_ptr<shared::ParSparseMat> assembled_patch_operator;
  mfem::Array<int> selected_parent_boundary_elements;
  std::vector<mfem::Array<int>> elements_by_destination;
  std::vector<redecomp::VirtualFaceSpecification> destination_face_specifications;
  mfem::Array<int> occurrence_to_node;
  int num_patch_nodes{ 0 };
  std::unique_ptr<redecomp::SelectedBoundaryRestriction> motion_restriction;
  std::unique_ptr<redecomp::VirtualRefinementCollectionOperator> motion_refinement;
  mfem::Vector baseline_boundary_coordinates;
  mutable mfem::Vector current_true_coordinates;
  mutable mfem::Vector current_boundary_element_coordinates;
  mutable mfem::Vector current_boundary_coordinates;
  const mfem::ParFiniteElementSpace* coordinate_space{ nullptr };
  long mesh_sequence{ -1 };
  long space_sequence{ -1 };
  long velocity_space_sequence{ -1 };
  long reference_coordinate_space_sequence{ -1 };
};

MfemContactData::~MfemContactData() = default;

int MfemContactPatchView::VerticesPerElement() const { return tribol::VerticesPerElement( element_type ); }

int MfemContactPatchView::NumElements() const
{
  const int vertices_per_element = VerticesPerElement();
  return vertices_per_element > 0 ? static_cast<int>( connectivity.size() ) / vertices_per_element : 0;
}

int MfemContactPatchView::NumNodes() const { return coordinates[0].Size(); }

MfemContactData::MfemContactData( MfemContactSurface surface_1, MfemContactSurface surface_2,
                                  MfemContactOptions options, ContactMethod contact_method,
                                  EnforcementMethod enforcement_method )
    : surfaces_{ std::move( surface_1 ), std::move( surface_2 ) },
      options_{ options },
      contact_method_{ contact_method },
      enforcement_method_{ enforcement_method },
      communicator_{ surfaces_[0].Mesh().GetComm() }
{
  MPI_Comm_rank( communicator_, &rank_ );
  MPI_Comm_size( communicator_, &num_ranks_ );
  evaluation_rank_count_ =
      options_.evaluation_rank_count == 0 ? num_ranks_ : std::clamp( options_.evaluation_rank_count, 1, num_ranks_ );
  surface_patches_[0].mesh_id = surfaces_[0].MeshId();
  surface_patches_[1].mesh_id = surfaces_[1].MeshId();
  ValidateRegistration();
  if ( UsesMultiplierSpace( contact_method_, enforcement_method_ ) ) {
    multiplier_space_ = std::make_unique<MfemMultiplierSpace>( surfaces_[1] );
  }
  InitializeContactResiduals();
}

void MfemContactData::ValidateRegistration() const
{
  SLIC_ERROR_ROOT_IF( options_.memory_space != MemorySpace::Dynamic && options_.memory_space != MemorySpace::Host,
                      "Operator-based MFEM contact currently stores patch arrays in host memory. Set "
                      "MfemContactOptions::memory_space to Host or Dynamic." );
  SLIC_ERROR_ROOT_IF( options_.execution_mode != ExecutionMode::Dynamic && isOnDevice( options_.execution_mode ),
                      "Operator-based MFEM contact currently runs patch search, exchange, and refinement on the "
                      "host. Select Sequential, OpenMP, or Dynamic execution." );
  SLIC_ERROR_ROOT_IF( options_.evaluation_rank_count < 0, "MFEM contact evaluation rank count must be nonnegative." );
  SLIC_ERROR_ROOT_IF( options_.refinement_factor < 0, "MFEM contact refinement factor must be nonnegative." );
  SLIC_ERROR_ROOT_IF( options_.motion_skin < 0.0 && options_.motion_skin != MfemContactOptions::AUTO_MOTION_SKIN,
                      "MFEM contact motion skin must be nonnegative or AUTO_MOTION_SKIN." );
  int communicator_relation = MPI_UNEQUAL;
  MPI_Comm_compare( surfaces_[0].Mesh().GetComm(), surfaces_[1].Mesh().GetComm(), &communicator_relation );
  SLIC_ERROR_ROOT_IF( communicator_relation != MPI_IDENT && communicator_relation != MPI_CONGRUENT,
                      "MFEM contact surfaces require identical or congruent MPI communicators." );
  SLIC_ERROR_ROOT_IF( surfaces_[0].Mesh().SpaceDimension() != surfaces_[1].Mesh().SpaceDimension(),
                      "MFEM contact surfaces require the same physical dimension." );
  SLIC_ERROR_ROOT_IF( contact_method_ == ENERGY_MORTAR && surfaces_[0].Mesh().SpaceDimension() != 2,
                      "Operator-based EnergyMortar currently supports two-dimensional contact only." );
  SLIC_ERROR_ROOT_IF( ( surfaces_[0].ElementThickness() == nullptr ) != ( surfaces_[1].ElementThickness() == nullptr ),
                      "Element-based MFEM contact penalties require fields on both surfaces." );
  for ( const auto& surface : surfaces_ ) {
    SLIC_ERROR_ROOT_IF( surface.Coordinates().ParFESpace() == nullptr,
                        "MFEM contact coordinates require a parallel finite element space." );
    SLIC_ERROR_ROOT_IF( surface.Coordinates().VectorDim() != surface.Mesh().SpaceDimension(),
                        "MFEM contact coordinates must have one component per physical dimension." );
    SLIC_ERROR_ROOT_IF( surface.Coordinates().ParFESpace()->GetParMesh() != &surface.Mesh(),
                        "MFEM contact coordinates must be defined on the registered parent mesh." );
    GetH1Collection( *surface.Coordinates().ParFESpace(), "MFEM contact coordinates" );
    const auto validate_optional_field = [&]( const mfem::ParGridFunction* field, const char* field_name ) {
      if ( field == nullptr ) return;
      SLIC_ERROR_ROOT_IF( field->ParFESpace() == nullptr || field->ParFESpace()->GetParMesh() != &surface.Mesh(),
                          field_name << " must be defined on the registered parent mesh." );
      SLIC_ERROR_ROOT_IF( field->VectorDim() != surface.Mesh().SpaceDimension(),
                          field_name << " must have one component per physical dimension." );
      GetH1Collection( *field->ParFESpace(), field_name );
    };
    validate_optional_field( surface.Velocity(), "MFEM contact velocity" );
    validate_optional_field( surface.ReferenceCoordinates(), "MFEM reference coordinates" );
    const auto validate_quadrature_field = [&]( const mfem::QuadratureFunction* field, const char* field_name ) {
      if ( field == nullptr ) return;
      SLIC_ERROR_ROOT_IF( field->GetSpace() == nullptr || field->GetSpace()->GetMesh() != &surface.Mesh(),
                          field_name << " must be defined on the registered parent mesh." );
      SLIC_ERROR_ROOT_IF( field->GetVDim() != 1, field_name << " must be scalar." );
      for ( int element = 0; element < field->GetSpace()->GetNE(); ++element ) {
        const int block_size = field->GetSpace()->Offset( element + 1 ) - field->GetSpace()->Offset( element );
        SLIC_ERROR_ROOT_IF( block_size != 1, field_name
                                                 << " must provide exactly one value per parent volume element. Use an "
                                                 << "order-zero QuadratureSpace for Tribol element penalties." );
      }
    };
    validate_quadrature_field( surface.ElementThickness(), "MFEM element thickness" );
    validate_quadrature_field( surface.MaterialModulus(), "MFEM material modulus" );
    SLIC_ERROR_ROOT_IF( ( surface.ElementThickness() == nullptr ) != ( surface.MaterialModulus() == nullptr ),
                        "MFEM element penalty data requires both thickness and material modulus fields." );
  }
}

MfemContactPlacement MfemContactData::ResolvePlacement() const
{
  if ( options_.placement != MfemContactPlacement::AUTO ) {
    return options_.placement;
  }
  const bool self_contact = &surfaces_[0].Mesh() == &surfaces_[1].Mesh() &&
                            surfaces_[0].BoundaryAttributes() == surfaces_[1].BoundaryAttributes();
  if ( self_contact ) return MfemContactPlacement::SPATIAL_PARTITION;
  if ( UsesNonmortarAnchor( contact_method_ ) ) {
    return MfemContactPlacement::ANCHOR_OWNER;
  }
  return MfemContactPlacement::BALANCED_ANCHOR;
}

void MfemContactData::InitializeContactResiduals()
{
  contact_residuals_.clear();
  for ( const auto& surface : surfaces_ ) {
    const auto* space = surface.Coordinates().ParFESpace();
    const auto existing = std::find_if( contact_residuals_.begin(), contact_residuals_.end(),
                                        [space]( const auto& result ) { return result.space == space; } );
    if ( existing == contact_residuals_.end() ) {
      MfemContactResidual contact_residual;
      contact_residual.space = space;
      contact_residual.true_dof_values.SetSize( space->GetTrueVSize() );
      contact_residual.true_dof_values = 0.0;
      contact_residuals_.push_back( std::move( contact_residual ) );
    }
  }
  if ( multiplier_space_ ) {
    MfemContactResidual multiplier_residual;
    multiplier_residual.space = &multiplier_space_->Space();
    multiplier_residual.true_dof_values.SetSize( multiplier_space_->Space().GetTrueVSize() );
    multiplier_residual.true_dof_values = 0.0;
    contact_residuals_.push_back( std::move( multiplier_residual ) );
  }
}

bool MfemContactData::ContactPatchNeedsRebuild( RealT contact_reach ) const
{
  if ( patch_rebuild_requested_ || !surface_states_[0] || !surface_states_[1] ||
       contact_reach != rebuild_contact_reach_ ) {
    return true;
  }
  double local_surface_motion[2]{ 0.0, 0.0 };
  for ( int surface = 0; surface < 2; ++surface ) {
    const auto& state = *surface_states_[surface];
    const auto& mesh = surfaces_[surface].Mesh();
    const auto& space = *surfaces_[surface].Coordinates().ParFESpace();
    if ( state.mesh_sequence != mesh.GetSequence() || state.coordinate_space != &space ||
         state.space_sequence != space.GetSequence() ) {
      return true;
    }
    if ( surfaces_[surface].Velocity() != nullptr &&
         state.velocity_space_sequence != surfaces_[surface].Velocity()->ParFESpace()->GetSequence() ) {
      return true;
    }
    if ( surfaces_[surface].ReferenceCoordinates() != nullptr &&
         state.reference_coordinate_space_sequence !=
             surfaces_[surface].ReferenceCoordinates()->ParFESpace()->GetSequence() ) {
      return true;
    }
    surfaces_[surface].Coordinates().GetTrueDofs( state.current_true_coordinates );
    state.motion_restriction->Mult( state.current_true_coordinates, state.current_boundary_element_coordinates );
    state.motion_refinement->Mult( state.current_boundary_element_coordinates, state.current_boundary_coordinates );
    SLIC_ERROR_ROOT_IF( state.current_boundary_coordinates.Size() != state.baseline_boundary_coordinates.Size(),
                        "A contact boundary changed layout without updating its finite element sequence." );
    const int num_occurrences =
        state.current_boundary_coordinates.Size() / surfaces_[surface].Mesh().SpaceDimension();
    for ( int occurrence = 0; occurrence < num_occurrences; ++occurrence ) {
      double squared_motion = 0.0;
      for ( int dimension = 0; dimension < surfaces_[surface].Mesh().SpaceDimension(); ++dimension ) {
        const int boundary_dof = occurrence + dimension * num_occurrences;
        const double component_motion =
            state.current_boundary_coordinates[boundary_dof] - state.baseline_boundary_coordinates[boundary_dof];
        squared_motion += component_motion * component_motion;
      }
      local_surface_motion[surface] = std::max( local_surface_motion[surface], std::sqrt( squared_motion ) );
    }
  }
  double global_surface_motion[2]{ 0.0, 0.0 };
  MPI_Allreduce( local_surface_motion, global_surface_motion, 2, MPI_DOUBLE, MPI_MAX, communicator_ );
  const bool self_contact = &surfaces_[0].Mesh() == &surfaces_[1].Mesh() &&
                            surfaces_[0].BoundaryAttributes() == surfaces_[1].BoundaryAttributes();
  const double motion =
      self_contact ? 2.0 * global_surface_motion[0] : global_surface_motion[0] + global_surface_motion[1];
  return motion >= motion_skin_;
}

bool MfemContactData::Update( RealT contact_reach, bool force_rebuild )
{
  SLIC_ERROR_ROOT_IF( contact_reach < 0.0, "MFEM contact reach is a physical distance and must be nonnegative." );
  const bool rebuild = force_rebuild || ContactPatchNeedsRebuild( contact_reach );
  if ( rebuild ) {
    const auto& dual_surface = surfaces_[1];
    const auto* dual_space = dual_surface.Coordinates().ParFESpace();
    const bool multiplier_topology_changed =
        multiplier_space_ && surface_states_[1] &&
        ( surface_states_[1]->mesh_sequence != dual_surface.Mesh().GetSequence() ||
          surface_states_[1]->coordinate_space != dual_space ||
          surface_states_[1]->space_sequence != dual_space->GetSequence() );
    if ( multiplier_topology_changed ) {
      // Mesh adaptation changes the multiplier true-DOF meaning. Rebuild its
      // solver space rather than retaining pressure values with stale indices.
      multiplier_space_ = std::make_unique<MfemMultiplierSpace>( dual_surface );
    }
    BuildPatch( contact_reach );
    InitializeContactResiduals();
  }
  RefreshPatchFields();
  BuildChildPairs();
  return rebuild;
}

void MfemContactData::BuildPatch( RealT contact_reach )
{
  // Exact virtual-facet bounds make the parent search conservative for the
  // same high-order geometry that Tribol later evaluates.
  std::array<std::vector<FaceBoundsRecord>, 2> local_bounds;
  double local_maximum_child_diameter = 0.0;
  for ( int surface = 0; surface < 2; ++surface ) {
    auto state = std::make_unique<SurfaceState>();
    const auto& definition = surfaces_[surface];
    const auto& mesh = definition.Mesh();
    auto& coordinate_space = *definition.Coordinates().ParFESpace();
    const auto& coordinate_collection = GetH1Collection( coordinate_space, "MFEM contact coordinates" );
    mfem::Array<HYPRE_BigInt> global_face_ids;
    mfem::Array<HYPRE_BigInt> global_vertex_ids;
    mesh.GetGlobalFaceIndices( global_face_ids );
    mesh.GetGlobalVertexIndices( global_vertex_ids );

    for ( int boundary_element = 0; boundary_element < mesh.GetNBE(); ++boundary_element ) {
      const int attribute = mesh.GetBdrAttribute( boundary_element );
      if ( definition.BoundaryAttributes().count( attribute ) == 0 ) continue;
      const int local_surface_element = static_cast<int>( state->boundary_elements.size() );
      state->boundary_elements.push_back( boundary_element );
      const auto& boundary_fe = *coordinate_space.GetBE( boundary_element );
      const int refinement_factor = RefinementFactor( options_, boundary_fe );
      local_maximum_child_diameter =
          std::max( local_maximum_child_diameter,
                    ComputeMaximumChildDiameter( definition.Coordinates(), boundary_element, refinement_factor ) );

      FaceMetadata metadata{};
      metadata.key = { rank_, local_surface_element };
      const int local_face = mesh.GetBdrElementFaceIndex( boundary_element );
      metadata.global_face = global_face_ids[local_face];
      metadata.geometry = mesh.GetBdrElementGeometry( boundary_element );
      metadata.order = boundary_fe.GetOrder();
      metadata.refinement_factor = refinement_factor;
      metadata.basis_type = coordinate_collection.GetBasisType();
      mfem::Array<int> vertices;
      mesh.GetBdrElementVertices( boundary_element, vertices );
      metadata.num_vertices = vertices.Size();
      SLIC_ERROR_ROOT_IF( metadata.num_vertices > FaceMetadata::max_vertices,
                          "MFEM contact supports boundary faces with at most four vertices." );
      for ( int vertex = 0; vertex < vertices.Size(); ++vertex ) {
        metadata.global_vertices[vertex] = global_vertex_ids[vertices[vertex]];
      }
      state->local_metadata.push_back( metadata );
    }
    mfem::Array<int> all_boundary_elements( static_cast<int>( state->boundary_elements.size() ) );
    std::vector<redecomp::VirtualFaceSpecification> all_face_specifications;
    all_face_specifications.reserve( state->local_metadata.size() );
    for ( int face = 0; face < static_cast<int>( state->boundary_elements.size() ); ++face ) {
      all_boundary_elements[face] = state->boundary_elements[face];
      const auto& metadata = state->local_metadata[face];
      all_face_specifications.push_back(
          { static_cast<mfem::Geometry::Type>( metadata.geometry ), metadata.order, metadata.refinement_factor,
            metadata.basis_type } );
    }
    state->motion_restriction =
        std::make_unique<redecomp::SelectedBoundaryRestriction>( coordinate_space, std::move( all_boundary_elements ) );
    state->motion_refinement = std::make_unique<redecomp::VirtualRefinementCollectionOperator>(
        state->motion_restriction->SelectedElementOffsets(), std::move( all_face_specifications ),
        coordinate_space.GetVDim() );
    state->current_true_coordinates.SetSize( coordinate_space.GetTrueVSize() );
    state->current_boundary_element_coordinates.SetSize( state->motion_restriction->Height() );
    state->current_boundary_coordinates.SetSize( state->motion_refinement->Height() );
    state->baseline_boundary_coordinates.SetSize( state->motion_refinement->Height() );
    definition.Coordinates().GetTrueDofs( state->current_true_coordinates );
    state->motion_restriction->Mult( state->current_true_coordinates, state->current_boundary_element_coordinates );
    state->motion_refinement->Mult( state->current_boundary_element_coordinates,
                                    state->baseline_boundary_coordinates );
    state->coordinate_space = &coordinate_space;
    state->mesh_sequence = mesh.GetSequence();
    state->space_sequence = coordinate_space.GetSequence();
    surface_states_[surface] = std::move( state );
  }

  double global_maximum_child_diameter = 0.0;
  MPI_Allreduce( &local_maximum_child_diameter, &global_maximum_child_diameter, 1, MPI_DOUBLE, MPI_MAX, communicator_ );
  motion_skin_ = options_.motion_skin >= 0.0 ? options_.motion_skin : 0.25 * global_maximum_child_diameter;
  const RealT bounds_inflation = contact_reach + 0.5 * motion_skin_;

  std::array<double, 3> local_minimum;
  std::array<double, 3> local_maximum;
  local_minimum.fill( std::numeric_limits<double>::max() );
  local_maximum.fill( std::numeric_limits<double>::lowest() );
  for ( int surface = 0; surface < 2; ++surface ) {
    const auto& state = *surface_states_[surface];
    for ( int local_surface_element = 0; local_surface_element < static_cast<int>( state.boundary_elements.size() );
          ++local_surface_element ) {
      const auto& metadata = state.local_metadata[local_surface_element];
      auto bounds =
          ComputeFaceBounds( surfaces_[surface], local_surface_element, state.boundary_elements[local_surface_element],
                             metadata.refinement_factor, bounds_inflation, metadata.global_face, rank_ );
      bounds.surface = surface;
      for ( int dimension = 0; dimension < surfaces_[surface].Mesh().SpaceDimension(); ++dimension ) {
        local_minimum[dimension] = std::min( local_minimum[dimension], bounds.minimum[dimension] );
        local_maximum[dimension] = std::max( local_maximum[dimension], bounds.maximum[dimension] );
      }
      local_bounds[surface].push_back( bounds );
    }
  }

  std::array<double, 3> global_minimum;
  std::array<double, 3> global_maximum;
  MPI_Allreduce( local_minimum.data(), global_minimum.data(), 3, MPI_DOUBLE, MPI_MIN, communicator_ );
  MPI_Allreduce( local_maximum.data(), global_maximum.data(), 3, MPI_DOUBLE, MPI_MAX, communicator_ );
  const int space_dimension = surfaces_[0].Mesh().SpaceDimension();
  const int spatial_axis = ChooseSpatialAxis( global_minimum, global_maximum, space_dimension );

  // Route each bound only to the spatial slabs it overlaps. This avoids an
  // all-gather while giving every slab all bounds needed for a local BVH.
  std::vector<std::vector<FaceBoundsRecord>> bounds_by_spatial_rank( static_cast<size_t>( num_ranks_ ) );
  for ( int surface = 0; surface < 2; ++surface ) {
    for ( const auto& bounds : local_bounds[surface] ) {
      const int first_domain = DomainForCoordinate( bounds.minimum[spatial_axis], global_minimum[spatial_axis],
                                                    global_maximum[spatial_axis], evaluation_rank_count_ );
      const int last_domain = DomainForCoordinate( bounds.maximum[spatial_axis], global_minimum[spatial_axis],
                                                   global_maximum[spatial_axis], evaluation_rank_count_ );
      for ( int domain = first_domain; domain <= last_domain; ++domain ) {
        bounds_by_spatial_rank[domain].push_back( bounds );
      }
    }
  }
  const auto spatial_bounds = FlattenByRank( ExchangeRecordsByRank( communicator_, bounds_by_spatial_rank ) );
  std::vector<FaceBoundsRecord> first_surface_bounds;
  std::vector<FaceBoundsRecord> second_surface_bounds;
  for ( const auto& bounds : spatial_bounds ) {
    ( bounds.surface == 0 ? first_surface_bounds : second_surface_bounds ).push_back( bounds );
  }

  const bool self_contact = &surfaces_[0].Mesh() == &surfaces_[1].Mesh() &&
                            surfaces_[0].BoundaryAttributes() == surfaces_[1].BoundaryAttributes();
  std::set<std::pair<FaceKey, FaceKey>> unique_spatial_pairs;
  std::vector<FacePairRecord> spatial_pairs;
  const auto intersecting_bounds =
      space_dimension == 2 ? FindIntersectingBoundsWithBvh<2>( first_surface_bounds, second_surface_bounds )
                           : FindIntersectingBoundsWithBvh<3>( first_surface_bounds, second_surface_bounds );
  for ( const auto& [first_index, second_index] : intersecting_bounds ) {
    const auto& first = first_surface_bounds[first_index];
    const auto& second = second_surface_bounds[second_index];
    FaceKey first_key = first.key;
    FaceKey second_key = second.key;
    if ( self_contact ) {
      if ( first.global_face == second.global_face ) continue;
      if ( second_key < first_key ) std::swap( first_key, second_key );
    }
    double overlap_center = 0.5 * ( std::max( first.minimum[spatial_axis], second.minimum[spatial_axis] ) +
                                    std::min( first.maximum[spatial_axis], second.maximum[spatial_axis] ) );
    const int pair_spatial_rank = DomainForCoordinate( overlap_center, global_minimum[spatial_axis],
                                                       global_maximum[spatial_axis], evaluation_rank_count_ );
    if ( pair_spatial_rank != rank_ || !unique_spatial_pairs.emplace( first_key, second_key ).second ) continue;
    spatial_pairs.push_back( { first_key, second_key, pair_spatial_rank } );
  }

  const MfemContactPlacement placement = ResolvePlacement();
  std::vector<std::vector<FacePairRecord>> pairs_by_evaluation_rank( static_cast<size_t>( num_ranks_ ) );
  for ( const auto& pair : spatial_pairs ) {
    const FaceKey anchor = UsesNonmortarAnchor( contact_method_ ) ? pair.second : pair.first;
    int evaluation_rank = pair.spatial_rank;
    if ( placement == MfemContactPlacement::ANCHOR_OWNER ) {
      // A restricted evaluation set contains the lowest ranks. Folding an
      // excluded owner into that set keeps all pairs for its anchor together.
      evaluation_rank = anchor.rank % evaluation_rank_count_;
    } else if ( placement == MfemContactPlacement::BALANCED_ANCHOR ) {
      // Multiplicative hashing spreads anchor groups without communicating a
      // global load table; every pair with this anchor receives the same rank.
      constexpr std::uint64_t anchor_hash_stride = 0x9e3779b97f4a7c15ULL;
      const std::uint64_t hash =
          static_cast<std::uint64_t>( anchor.rank ) * anchor_hash_stride + static_cast<std::uint64_t>( anchor.element );
      evaluation_rank = static_cast<int>( hash % static_cast<std::uint64_t>( evaluation_rank_count_ ) );
    }
    pairs_by_evaluation_rank[evaluation_rank].push_back( pair );
  }
  auto received_pairs = FlattenByRank( ExchangeRecordsByRank( communicator_, pairs_by_evaluation_rank ) );
  std::sort( received_pairs.begin(), received_pairs.end(), []( const auto& first, const auto& second ) {
    return std::tie( first.first.rank, first.first.element, first.second.rank, first.second.element ) <
           std::tie( second.first.rank, second.first.element, second.second.rank, second.second.element );
  } );
  received_pairs.erase( std::unique( received_pairs.begin(), received_pairs.end(),
                                     []( const auto& first, const auto& second ) {
                                       return first.first == second.first && first.second == second.second;
                                     } ),
                        received_pairs.end() );
  parent_pairs_.clear();
  for ( const auto& pair : received_pairs ) {
    parent_pairs_.push_back( { pair.first.rank, pair.first.element, pair.second.rank, pair.second.element } );
  }

  for ( int surface = 0; surface < 2; ++surface ) {
    // Once pair placement is final, request each needed parent face once per
    // evaluation rank and cache that sparse schedule for subsequent updates.
    std::vector<std::set<int>> requested_elements_by_owner( static_cast<size_t>( num_ranks_ ) );
    for ( const auto& pair : parent_pairs_ ) {
      const FaceKey key = surface == 0 ? FaceKey{ pair.first_rank, pair.first_element }
                                       : FaceKey{ pair.second_rank, pair.second_element };
      requested_elements_by_owner[key.rank].insert( key.element );
    }
    std::vector<std::vector<int>> requests_by_owner( static_cast<size_t>( num_ranks_ ) );
    for ( int owner = 0; owner < num_ranks_; ++owner ) {
      requests_by_owner[owner].assign( requested_elements_by_owner[owner].begin(),
                                       requested_elements_by_owner[owner].end() );
    }
    const auto requests_by_destination = ExchangeRecordsByRank( communicator_, requests_by_owner );

    auto& state = *surface_states_[surface];
    std::set<int> requested_local_elements;
    for ( const auto& destination_requests : requests_by_destination ) {
      requested_local_elements.insert( destination_requests.begin(), destination_requests.end() );
    }
    std::vector<int> local_elements( requested_local_elements.begin(), requested_local_elements.end() );
    std::unordered_map<int, int> local_to_compact;
    mfem::Array<int> selected_boundary_elements( static_cast<int>( local_elements.size() ) );
    std::vector<FaceMetadata> compact_metadata( local_elements.size() );
    for ( int compact = 0; compact < static_cast<int>( local_elements.size() ); ++compact ) {
      const int local_element = local_elements[compact];
      local_to_compact[local_element] = compact;
      selected_boundary_elements[compact] = state.boundary_elements[local_element];
      compact_metadata[compact] = state.local_metadata[local_element];
    }

    std::vector<mfem::Array<int>> compact_elements_by_destination( static_cast<size_t>( num_ranks_ ) );
    std::vector<std::vector<FaceMetadata>> metadata_by_destination( static_cast<size_t>( num_ranks_ ) );
    for ( int destination = 0; destination < num_ranks_; ++destination ) {
      for ( int local_element : requests_by_destination[destination] ) {
        const int compact = local_to_compact.at( local_element );
        compact_elements_by_destination[destination].Append( compact );
        metadata_by_destination[destination].push_back( compact_metadata[compact] );
      }
    }
    state.destination_metadata = FlattenByRank( ExchangeRecordsByRank( communicator_, metadata_by_destination ) );
    std::vector<redecomp::VirtualFaceSpecification> face_specifications;
    face_specifications.reserve( state.destination_metadata.size() );
    for ( const auto& metadata : state.destination_metadata ) {
      face_specifications.push_back(
          { static_cast<mfem::Geometry::Type>( metadata.geometry ), metadata.order, metadata.refinement_factor,
            metadata.basis_type } );
    }

    // Build refinement first to derive canonical node IDs, then rebuild it as
    // part of H. Topology construction happens only when the skin is rebuilt.
    redecomp::SelectedBoundaryRestriction local_restriction( *surfaces_[surface].Coordinates().ParFESpace(),
                                                             selected_boundary_elements );
    auto exchange_plan = std::make_shared<redecomp::ElementExchangePlan>(
        communicator_, local_restriction.SelectedElementOffsets(), compact_elements_by_destination );
    redecomp::VirtualRefinementCollectionOperator topology_refinement(
        exchange_plan->DestinationElementOffsets(), face_specifications, surfaces_[surface].Coordinates().VectorDim(),
        mfem::Ordering::byNODES );
    mfem::Array<int> occurrence_to_node;
    int num_nodes = 0;
    surface_patches_[surface].connectivity =
        BuildConnectivity( state.destination_metadata, topology_refinement, occurrence_to_node, num_nodes );
    state.selected_parent_boundary_elements = selected_boundary_elements;
    state.elements_by_destination = compact_elements_by_destination;
    state.destination_face_specifications = face_specifications;
    state.occurrence_to_node = occurrence_to_node;
    state.num_patch_nodes = num_nodes;
    state.patch_operator = std::make_unique<redecomp::ContactPatchOperator>(
        *surfaces_[surface].Coordinates().ParFESpace(), std::move( selected_boundary_elements ),
        std::move( compact_elements_by_destination ), std::move( face_specifications ), std::move( occurrence_to_node ),
        num_nodes );

    const auto build_optional_field_operator = [&]( const mfem::ParGridFunction* field ) {
      if ( field == nullptr || field->ParFESpace() == surfaces_[surface].Coordinates().ParFESpace() ) {
        return std::unique_ptr<redecomp::ContactPatchOperator>{};
      }
      const int field_order = field->ParFESpace()->FEColl()->GetOrder();
      SLIC_ERROR_ROOT_IF( field_order < 0,
                          "Optional MFEM contact fields with variable polynomial order are not supported." );
      auto field_faces = state.destination_face_specifications;
      const auto& field_collection = GetH1Collection( *field->ParFESpace(), "Optional MFEM contact field" );
      for ( auto& face : field_faces ) {
        face.order = field_order;
        face.basis_type = field_collection.GetBasisType();
      }
      return std::make_unique<redecomp::ContactPatchOperator>(
          *field->ParFESpace(), state.selected_parent_boundary_elements, state.elements_by_destination,
          std::move( field_faces ), state.occurrence_to_node, state.num_patch_nodes );
    };
    state.velocity_operator = build_optional_field_operator( surfaces_[surface].Velocity() );
    state.reference_coordinate_operator = build_optional_field_operator( surfaces_[surface].ReferenceCoordinates() );

    const auto build_element_field_exchange = [&]( const mfem::QuadratureFunction* field ) {
      if ( field == nullptr ) return std::unique_ptr<redecomp::ElementFieldExchange>{};
      std::vector<mfem::Array<int>> volume_elements_by_destination( static_cast<size_t>( num_ranks_ ) );
      for ( int destination = 0; destination < num_ranks_; ++destination ) {
        for ( int compact_element : state.elements_by_destination[destination] ) {
          int volume_element = -1;
          int orientation = 0;
          surfaces_[surface].Mesh().GetBdrElementAdjacentElement(
              state.selected_parent_boundary_elements[compact_element], volume_element, orientation );
          volume_elements_by_destination[destination].Append( volume_element );
        }
      }
      return std::make_unique<redecomp::ElementFieldExchange>( *field, std::move( volume_elements_by_destination ) );
    };
    state.thickness_exchange = build_element_field_exchange( surfaces_[surface].ElementThickness() );
    state.modulus_exchange = build_element_field_exchange( surfaces_[surface].MaterialModulus() );
    state.velocity_space_sequence =
        surfaces_[surface].Velocity() == nullptr ? -1 : surfaces_[surface].Velocity()->ParFESpace()->GetSequence();
    state.reference_coordinate_space_sequence =
        surfaces_[surface].ReferenceCoordinates() == nullptr
            ? -1
            : surfaces_[surface].ReferenceCoordinates()->ParFESpace()->GetSequence();
    if ( surface == 1 && multiplier_space_ ) {
      multiplier_space_->BuildPatchOperator( state.selected_parent_boundary_elements, state.elements_by_destination,
                                             state.destination_face_specifications, state.occurrence_to_node,
                                             state.num_patch_nodes );
    }

    state.destination_face_child_offsets.assign( state.destination_metadata.size() + 1, 0 );
    for ( int face = 0; face < static_cast<int>( state.destination_metadata.size() ); ++face ) {
      state.destination_face_child_offsets[face + 1] =
          state.destination_face_child_offsets[face] +
          state.patch_operator->Refinement().FaceRefinement( face ).NumChildren();
    }
    if ( !state.destination_metadata.empty() ) {
      surface_patches_[surface].element_type =
          ToTribolElementType( static_cast<mfem::Geometry::Type>( state.destination_metadata.front().geometry ) );
      for ( const auto& metadata : state.destination_metadata ) {
        SLIC_ERROR_ROOT_IF( ToTribolElementType( static_cast<mfem::Geometry::Type>( metadata.geometry ) ) !=
                                surface_patches_[surface].element_type,
                            "One Tribol contact surface cannot mix linear child element geometries." );
      }
    }
    surface_patches_[surface].space_dimension = space_dimension;
    for ( int dimension = 0; dimension < 3; ++dimension ) {
      surface_patches_[surface].coordinates[dimension].SetSize( num_nodes );
      surface_patches_[surface].response[dimension].SetSize( num_nodes );
      surface_patches_[surface].velocity[dimension].SetSize( num_nodes );
      surface_patches_[surface].reference_coordinates[dimension].SetSize( num_nodes );
    }
    surface_patches_[surface].element_thickness.SetSize( surface_patches_[surface].NumElements() );
    surface_patches_[surface].material_modulus.SetSize( surface_patches_[surface].NumElements() );
  }

  rebuild_contact_reach_ = contact_reach;
  patch_rebuild_requested_ = false;
}

void MfemContactData::RefreshPatchFields()
{
  for ( int surface = 0; surface < 2; ++surface ) {
    auto& state = *surface_states_[surface];
    auto& patch = surface_patches_[surface];
    const int num_nodes = patch.NumNodes();
    const int space_dimension = patch.space_dimension;
    mfem::Vector source_true_dofs;
    surfaces_[surface].Coordinates().GetTrueDofs( source_true_dofs );
    mfem::Vector patch_values( state.patch_operator->Height() );
    state.patch_operator->Mult( source_true_dofs, patch_values );
    for ( int dimension = 0; dimension < space_dimension; ++dimension ) {
      std::copy_n( patch_values.HostRead() + dimension * num_nodes, num_nodes,
                   patch.coordinates[dimension].HostWrite() );
      patch.response[dimension] = 0.0;
    }

    patch.has_velocity = surfaces_[surface].Velocity() != nullptr;
    if ( patch.has_velocity ) {
      surfaces_[surface].Velocity()->GetTrueDofs( source_true_dofs );
      const auto& velocity_operator = state.velocity_operator ? *state.velocity_operator : *state.patch_operator;
      patch_values.SetSize( velocity_operator.Height() );
      velocity_operator.Mult( source_true_dofs, patch_values );
      for ( int dimension = 0; dimension < space_dimension; ++dimension ) {
        std::copy_n( patch_values.HostRead() + dimension * num_nodes, num_nodes,
                     patch.velocity[dimension].HostWrite() );
      }
    }

    patch.has_reference_coordinates = surfaces_[surface].ReferenceCoordinates() != nullptr;
    if ( patch.has_reference_coordinates ) {
      surfaces_[surface].ReferenceCoordinates()->GetTrueDofs( source_true_dofs );
      const auto& reference_operator =
          state.reference_coordinate_operator ? *state.reference_coordinate_operator : *state.patch_operator;
      patch_values.SetSize( reference_operator.Height() );
      reference_operator.Mult( source_true_dofs, patch_values );
      for ( int dimension = 0; dimension < space_dimension; ++dimension ) {
        std::copy_n( patch_values.HostRead() + dimension * num_nodes, num_nodes,
                     patch.reference_coordinates[dimension].HostWrite() );
      }
    }

    const auto gather_element_field = [&]( const mfem::QuadratureFunction* source,
                                           const std::unique_ptr<redecomp::ElementFieldExchange>& exchange,
                                           mfem::Vector& child_values, const char* field_name ) {
      if ( source == nullptr ) {
        child_values.SetSize( 0 );
        return;
      }
      const auto exchanged = exchange->Gather( *source );
      SLIC_ERROR_ROOT_IF( exchanged.element_offsets.Size() - 1 != static_cast<int>( state.destination_metadata.size() ),
                          field_name << " exchange does not match the contact-patch faces." );
      child_values.SetSize( patch.NumElements() );
      for ( int face = 0; face < exchanged.element_offsets.Size() - 1; ++face ) {
        const int block_begin = exchanged.element_offsets[face];
        const int block_size = exchanged.element_offsets[face + 1] - block_begin;
        SLIC_ERROR_ROOT_IF( block_size != 1,
                            field_name << " must use one value per source element when consumed by Tribol." );
        for ( int child = state.destination_face_child_offsets[face];
              child < state.destination_face_child_offsets[face + 1]; ++child ) {
          child_values[child] = exchanged.values[block_begin];
        }
      }
    };
    gather_element_field( surfaces_[surface].ElementThickness(), state.thickness_exchange, patch.element_thickness,
                          "MFEM element thickness" );
    gather_element_field( surfaces_[surface].MaterialModulus(), state.modulus_exchange, patch.material_modulus,
                          "MFEM material modulus" );
  }
  if ( multiplier_space_ ) {
    multiplier_space_->GatherPressure();
  }
}

void MfemContactData::BuildChildPairs()
{
  pair_surface_1_elements_.clear();
  pair_surface_2_elements_.clear();
  std::array<std::map<FaceKey, int>, 2> destination_face_ids;
  for ( int surface = 0; surface < 2; ++surface ) {
    const auto& metadata = surface_states_[surface]->destination_metadata;
    for ( int face = 0; face < static_cast<int>( metadata.size() ); ++face ) {
      destination_face_ids[surface][metadata[face].key] = face;
    }
  }
  for ( const auto& pair : parent_pairs_ ) {
    const auto first_face = destination_face_ids[0].find( { pair.first_rank, pair.first_element } );
    const auto second_face = destination_face_ids[1].find( { pair.second_rank, pair.second_element } );
    SLIC_ERROR_ROOT_IF( first_face == destination_face_ids[0].end() || second_face == destination_face_ids[1].end(),
                        "A placed parent pair is missing a requested contact face." );
    const auto& first_offsets = surface_states_[0]->destination_face_child_offsets;
    const auto& second_offsets = surface_states_[1]->destination_face_child_offsets;
    for ( int first_child = first_offsets[first_face->second]; first_child < first_offsets[first_face->second + 1];
          ++first_child ) {
      for ( int second_child = second_offsets[second_face->second];
            second_child < second_offsets[second_face->second + 1]; ++second_child ) {
        const RealT child_search_reach = rebuild_contact_reach_ + 0.5 * motion_skin_;
        bool child_bounds_intersect = true;
        for ( int dimension = 0; dimension < surface_patches_[0].space_dimension; ++dimension ) {
          double first_minimum = std::numeric_limits<double>::max();
          double first_maximum = std::numeric_limits<double>::lowest();
          double second_minimum = std::numeric_limits<double>::max();
          double second_maximum = std::numeric_limits<double>::lowest();
          const int first_vertices = surface_patches_[0].VerticesPerElement();
          const int second_vertices = surface_patches_[1].VerticesPerElement();
          for ( int vertex = 0; vertex < first_vertices; ++vertex ) {
            const int node = surface_patches_[0].connectivity[first_child * first_vertices + vertex];
            first_minimum = std::min( first_minimum, surface_patches_[0].coordinates[dimension][node] );
            first_maximum = std::max( first_maximum, surface_patches_[0].coordinates[dimension][node] );
          }
          for ( int vertex = 0; vertex < second_vertices; ++vertex ) {
            const int node = surface_patches_[1].connectivity[second_child * second_vertices + vertex];
            second_minimum = std::min( second_minimum, surface_patches_[1].coordinates[dimension][node] );
            second_maximum = std::max( second_maximum, surface_patches_[1].coordinates[dimension][node] );
          }
          if ( first_maximum + child_search_reach < second_minimum ||
               second_maximum + child_search_reach < first_minimum ) {
            child_bounds_intersect = false;
            break;
          }
        }
        if ( !child_bounds_intersect ) continue;
        pair_surface_1_elements_.push_back( first_child );
        pair_surface_2_elements_.push_back( second_child );
      }
    }
  }
}

void MfemContactData::CollectResiduals()
{
  for ( auto& contact_residual : contact_residuals_ ) {
    contact_residual.true_dof_values = 0.0;
  }
  for ( int surface = 0; surface < 2; ++surface ) {
    const auto& patch = surface_patches_[surface];
    const int num_nodes = patch.NumNodes();
    mfem::Vector patch_dual( surface_states_[surface]->patch_operator->Height() );
    for ( int dimension = 0; dimension < patch.space_dimension; ++dimension ) {
      std::copy_n( patch.response[dimension].HostRead(), num_nodes, patch_dual.HostWrite() + dimension * num_nodes );
    }
    mfem::Vector source_dual( surface_states_[surface]->patch_operator->Width() );
    surface_states_[surface]->patch_operator->MultTranspose( patch_dual, source_dual );
    const auto* space = surfaces_[surface].Coordinates().ParFESpace();
    auto contact_residual = std::find_if( contact_residuals_.begin(), contact_residuals_.end(),
                                          [space]( const auto& result ) { return result.space == space; } );
    contact_residual->true_dof_values += source_dual;
  }
  if ( multiplier_space_ ) {
    multiplier_space_->ReturnGap();
    contact_residuals_.back().true_dof_values = multiplier_space_->Gap();
  }
}

std::unique_ptr<mfem::BlockOperator> MfemContactData::BuildJacobian( const MethodData& method_data )
{
  std::array<int, 2> residual_block_for_surface{};
  for ( int surface = 0; surface < 2; ++surface ) {
    const auto* space = surfaces_[surface].Coordinates().ParFESpace();
    residual_block_for_surface[surface] = static_cast<int>(
        std::distance( contact_residuals_.begin(),
                       std::find_if( contact_residuals_.begin(), contact_residuals_.end(),
                                     [space]( const auto& result ) { return result.space == space; } ) ) );
  }

  mfem::Array<int> offsets( static_cast<int>( contact_residuals_.size() ) + 1 );
  offsets[0] = 0;
  for ( int block = 0; block < static_cast<int>( contact_residuals_.size() ); ++block ) {
    offsets[block + 1] = offsets[block] + contact_residuals_[block].space->GetTrueVSize();
  }
  auto jacobian = std::make_unique<mfem::BlockOperator>( offsets );
  jacobian->owns_blocks = 1;
  std::map<std::pair<int, int>, std::unique_ptr<shared::ParSparseMat>> residual_blocks;
  const auto patch_block_for = [&]( BlockSpace space ) -> PatchBlockView {
    switch ( space ) {
      case BlockSpace::MORTAR:
        return { surface_patches_[0], surface_patches_[0].space_dimension };
      case BlockSpace::NONMORTAR:
        return { surface_patches_[1], surface_patches_[1].space_dimension };
      case BlockSpace::LAGRANGE_MULTIPLIER:
        SLIC_ERROR_ROOT_IF( !multiplier_space_,
                            "A multiplier Jacobian block requires a solver-visible multiplier space." );
        return { surface_patches_[1], 1 };
      default:
        SLIC_ERROR_ROOT( "Unsupported contact Jacobian block space." );
        return { surface_patches_[0], surface_patches_[0].space_dimension };
    }
  };
  const auto residual_block_for_space = [&]( BlockSpace space ) {
    if ( space == BlockSpace::MORTAR ) return residual_block_for_surface[0];
    if ( space == BlockSpace::NONMORTAR ) return residual_block_for_surface[1];
    return static_cast<int>( contact_residuals_.size() ) - 1;
  };
  const auto assembled_operator_for = [&]( BlockSpace space ) -> const shared::ParSparseMat& {
    if ( space == BlockSpace::LAGRANGE_MULTIPLIER ) {
      SLIC_ERROR_ROOT_IF( !multiplier_space_,
                          "A multiplier Jacobian block requires a solver-visible multiplier space." );
      return multiplier_space_->AssembledPatchOperator();
    }
    const int surface = space == BlockSpace::MORTAR ? 0 : 1;
    auto& state = *surface_states_[surface];
    if ( !state.assembled_patch_operator ) {
      state.assembled_patch_operator = std::make_unique<shared::ParSparseMat>( state.patch_operator->Assemble() );
    }
    return *state.assembled_patch_operator;
  };

  constexpr std::array<BlockSpace, 3> block_spaces{ BlockSpace::MORTAR, BlockSpace::NONMORTAR,
                                                    BlockSpace::LAGRANGE_MULTIPLIER };
  for ( BlockSpace row_space : block_spaces ) {
    for ( BlockSpace column_space : block_spaces ) {
      const auto& element_matrices =
          method_data.getBlockJ()( static_cast<int>( row_space ), static_cast<int>( column_space ) );
      const int local_block_is_present = element_matrices.empty() ? 0 : 1;
      int global_block_is_present = 0;
      MPI_Allreduce( &local_block_is_present, &global_block_is_present, 1, MPI_INT, MPI_MAX, communicator_ );
      if ( global_block_is_present == 0 ) continue;
      auto patch_jacobian =
          BuildPatchJacobianMatrix( communicator_, patch_block_for( row_space ), patch_block_for( column_space ),
                                    method_data, row_space, column_space );
      auto contribution = shared::ParSparseMat::rap( assembled_operator_for( row_space ), patch_jacobian,
                                                     assembled_operator_for( column_space ) );
      const std::pair<int, int> residual_block{ residual_block_for_space( row_space ),
                                                residual_block_for_space( column_space ) };
      const auto existing = residual_blocks.find( residual_block );
      if ( existing == residual_blocks.end() ) {
        residual_blocks.emplace( residual_block,
                                 std::make_unique<shared::ParSparseMat>( std::move( contribution ) ) );
      } else {
        *existing->second += contribution;
      }
    }
  }
  for ( auto& [block, matrix] : residual_blocks ) {
    jacobian->SetBlock( block.first, block.second, matrix->release() );
  }
  return jacobian;
}

std::array<int, 2> MfemContactData::SentFaceCounts() const
{
  std::array<int, 2> counts{};
  for ( int surface = 0; surface < 2; ++surface ) {
    counts[surface] =
        surface_states_[surface] ? surface_states_[surface]->patch_operator->ExchangePlan().NumSentElements() : 0;
  }
  return counts;
}

std::array<int, 2> MfemContactData::ReceivedFaceCounts() const
{
  std::array<int, 2> counts{};
  for ( int surface = 0; surface < 2; ++surface ) {
    counts[surface] = surface_states_[surface]
                          ? surface_states_[surface]->patch_operator->ExchangePlan().NumDestinationElements()
                          : 0;
  }
  return counts;
}

}  // namespace tribol

#endif  // BUILD_REDECOMP
