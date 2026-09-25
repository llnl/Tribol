#ifndef TRIBOL_GEOM_OVERLAPCELL_HPP_
#define TRIBOL_GEOM_OVERLAPCELL_HPP_

#include "tribol/basis/LinearBasis.hpp"
#include "tribol/core/OverlapCellView.hpp"
#include "tribol/core/ParentTraceView.hpp"
#include "tribol/geom/ProjectedOverlap.hpp"
#include "tribol/search/Search.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tribol {

struct OverlapCellBuildOptions {
  std::uint64_t contact_pair_id{};
  std::uint64_t first_cell_id{};
  Real mapping_tolerance{ 1.0e-10 };
  Real orientation_tolerance{ 1.0e-12 };
};

class OverlapCellBatch {
 public:
  explicit OverlapCellBatch( int dimension ) : dimension_( dimension ), vertex_offsets_{ 0 } {}

  [[nodiscard]] Index numberOfAttempts() const { return static_cast<Index>( statuses_.size() ); }

  [[nodiscard]] Index numberOfAcceptedCells() const
  {
    return static_cast<Index>( std::count( statuses_.begin(), statuses_.end(), GeometryStatus::Accepted ) );
  }

  [[nodiscard]] OverlapCellBatchView view() const
  {
    const Index attempts = numberOfAttempts();
    const Index vertices = static_cast<Index>( integration_vertices_.size() ) / 3;
    return {
        .dimension = dimension_,
        .contact_pair_ids = { contact_pair_ids_.data(), attempts },
        .cell_ids = { cell_ids_.data(), attempts },
        .owner_ranks = { owner_ranks_.data(), attempts },
        .mortar_subfacets = { mortar_subfacets_.data(), attempts },
        .nonmortar_subfacets = { nonmortar_subfacets_.data(), attempts },
        .mortar_parent_faces = { mortar_parent_faces_.data(), attempts },
        .nonmortar_parent_faces = { nonmortar_parent_faces_.data(), attempts },
        .statuses = { statuses_.data(), attempts },
        .normals = { { normals_.data(), static_cast<Index>( normals_.size() ) },
                     attempts,
                     3,
                     FieldLayout::Interleaved },
        .measures = { measures_.data(), attempts },
        .vertex_offsets = { vertex_offsets_.data(), static_cast<Index>( vertex_offsets_.size() ) },
        .integration_vertices = { { integration_vertices_.data(), static_cast<Index>( integration_vertices_.size() ) },
                                  vertices,
                                  3,
                                  FieldLayout::Interleaved },
        .mortar_parent_reference_coordinates = { { mortar_parent_reference_coordinates_.data(),
                                                   static_cast<Index>( mortar_parent_reference_coordinates_.size() ) },
                                                 vertices,
                                                 dimension_ - 1,
                                                 FieldLayout::Interleaved },
        .nonmortar_parent_reference_coordinates = { { nonmortar_parent_reference_coordinates_.data(),
                                                      static_cast<Index>(
                                                          nonmortar_parent_reference_coordinates_.size() ) },
                                                    vertices,
                                                    dimension_ - 1,
                                                    FieldLayout::Interleaved },
    };
  }

  void append( std::uint64_t contact_pair_id, std::uint64_t cell_id, int owner_rank, ElementPair subfacets,
               Index mortar_parent_face, Index nonmortar_parent_face, GeometryStatus status,
               const InteractionPatch& patch, const std::array<basis::ReferencePoint, maximumOverlapVertices>& mortar,
               const std::array<basis::ReferencePoint, maximumOverlapVertices>& nonmortar )
  {
    contact_pair_ids_.push_back( contact_pair_id );
    cell_ids_.push_back( cell_id );
    owner_ranks_.push_back( owner_rank );
    mortar_subfacets_.push_back( subfacets.mortar_element );
    nonmortar_subfacets_.push_back( subfacets.nonmortar_element );
    mortar_parent_faces_.push_back( mortar_parent_face );
    nonmortar_parent_faces_.push_back( nonmortar_parent_face );
    statuses_.push_back( status );
    measures_.push_back( status == GeometryStatus::Accepted ? patch.measure : 0.0 );
    normals_.insert( normals_.end(), patch.normal.begin(), patch.normal.end() );
    const int vertex_count = status == GeometryStatus::Accepted ? patch.vertex_count : 0;
    for ( int vertex = 0; vertex < vertex_count; ++vertex ) {
      integration_vertices_.insert( integration_vertices_.end(), patch.integration_vertices[vertex].begin(),
                                    patch.integration_vertices[vertex].end() );
      appendReference( mortar_parent_reference_coordinates_, mortar[vertex] );
      appendReference( nonmortar_parent_reference_coordinates_, nonmortar[vertex] );
    }
    vertex_offsets_.push_back( vertex_offsets_.back() + vertex_count );
  }

 private:
  void appendReference( std::vector<Real>& values, basis::ReferencePoint point ) const
  {
    values.push_back( point.first );
    if ( dimension_ == 3 ) {
      values.push_back( point.second );
    }
  }

  int dimension_{};
  std::vector<std::uint64_t> contact_pair_ids_{};
  std::vector<std::uint64_t> cell_ids_{};
  std::vector<int> owner_ranks_{};
  std::vector<Index> mortar_subfacets_{};
  std::vector<Index> nonmortar_subfacets_{};
  std::vector<Index> mortar_parent_faces_{};
  std::vector<Index> nonmortar_parent_faces_{};
  std::vector<GeometryStatus> statuses_{};
  std::vector<Real> normals_{};
  std::vector<Real> measures_{};
  std::vector<Index> vertex_offsets_{};
  std::vector<Real> integration_vertices_{};
  std::vector<Real> mortar_parent_reference_coordinates_{};
  std::vector<Real> nonmortar_parent_reference_coordinates_{};
};

namespace overlap_cell_detail {

struct MappedPoint {
  basis::ReferencePoint parent_reference{};
  GeometryStatus status{ GeometryStatus::Accepted };
};

[[nodiscard]] TRIBOL_HOST_DEVICE inline bool insideReferenceElement( ElementTopology topology,
                                                                     basis::ReferencePoint point, Real tolerance,
                                                                     bool parent_domain )
{
  const Real lower = parent_domain ? 0.0 : -1.0;
  const Real upper = 1.0;
  if ( topology == ElementTopology::Triangle ) {
    return point.first >= -tolerance && point.second >= -tolerance && point.first + point.second <= 1.0 + tolerance;
  }
  if ( topology == ElementTopology::Segment ) {
    return point.first >= lower - tolerance && point.first <= upper + tolerance;
  }
  return point.first >= lower - tolerance && point.first <= upper + tolerance && point.second >= lower - tolerance &&
         point.second <= upper + tolerance;
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline Real distance( const std::array<Real, 3>& left,
                                                       const std::array<Real, 3>& right, int dimension )
{
  Real squared{};
  for ( int component = 0; component < dimension; ++component ) {
    const Real difference = left[component] - right[component];
    squared += difference * difference;
  }
  return linearization_detail::squareRoot( squared );
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline Real subfacetScale( const SurfaceMeshView& mesh, Index subfacet )
{
  Real scale{};
  const Index begin = mesh.element_offsets[subfacet];
  const Index end = mesh.element_offsets[subfacet + 1];
  for ( Index first = begin; first < end; ++first ) {
    std::array<Real, 3> first_point{};
    for ( int component = 0; component < mesh.dimension; ++component ) {
      first_point[component] = mesh.coordinates( mesh.connectivity[first], component );
    }
    for ( Index second = first + 1; second < end; ++second ) {
      std::array<Real, 3> second_point{};
      for ( int component = 0; component < mesh.dimension; ++component ) {
        second_point[component] = mesh.coordinates( mesh.connectivity[second], component );
      }
      const Real edge_length = distance( first_point, second_point, mesh.dimension );
      scale = edge_length > scale ? edge_length : scale;
    }
  }
  return scale;
}

[[nodiscard]] TRIBOL_HOST_DEVICE inline MappedPoint mapToParent( const ParentTraceView& trace, Index subfacet,
                                                                 const std::array<Real, 3>& point, Real tolerance )
{
  const auto local = basis::mapToReference( trace.geometry, subfacet, point );
  const auto topology = trace.geometry.topologies[subfacet];
  const auto reconstructed = basis::mapToPhysical( trace.geometry, subfacet, local );
  const Real scale = subfacetScale( trace.geometry, subfacet );
  const Real allowed_error = tolerance * ( scale > 1.0 ? scale : 1.0 );
  if ( distance( reconstructed, point, trace.geometry.dimension ) > allowed_error ) {
    return { {}, GeometryStatus::ExcessiveMappingError };
  }
  if ( !insideReferenceElement( topology, local, tolerance, false ) ) {
    return { {}, GeometryStatus::InvalidMapping };
  }
  const auto shape = basis::primalShape( topology, local );
  const Index begin = trace.geometry.element_offsets[subfacet];
  basis::ReferencePoint parent{};
  for ( int local_node = 0; local_node < shape.size; ++local_node ) {
    const Index geometry_node = trace.geometry.connectivity[begin + local_node];
    parent.first += shape[local_node] * trace.subfacet_parent_reference_coordinates( geometry_node, 0 );
    if ( trace.parentReferenceDimension() == 2 ) {
      parent.second += shape[local_node] * trace.subfacet_parent_reference_coordinates( geometry_node, 1 );
    }
  }
  const Index parent_face = trace.subfacet_parent_faces[subfacet];
  if ( !insideReferenceElement( trace.parent_face_topologies[parent_face], parent, tolerance, true ) ) {
    return { {}, GeometryStatus::InvalidMapping };
  }
  return { parent, GeometryStatus::Accepted };
}

[[nodiscard]] inline bool sameCell( const OverlapCellBatchView& cells, Index previous, const InteractionPatch& patch,
                                    std::uint64_t mortar_parent_id, std::uint64_t nonmortar_parent_id,
                                    const ParentTracePairView& traces, Real tolerance )
{
  if ( !cells.isAccepted( previous ) ||
       traces.mortar.parent_face_ids[cells.mortar_parent_faces[previous]] != mortar_parent_id ||
       traces.nonmortar.parent_face_ids[cells.nonmortar_parent_faces[previous]] != nonmortar_parent_id ||
       std::abs( cells.measures[previous] - patch.measure ) > tolerance ||
       cells.vertex_offsets[previous + 1] - cells.vertex_offsets[previous] != patch.vertex_count ) {
    return false;
  }
  const Index begin = cells.vertex_offsets[previous];
  for ( int vertex = 0; vertex < patch.vertex_count; ++vertex ) {
    bool found{};
    for ( Index candidate = begin; candidate < cells.vertex_offsets[previous + 1]; ++candidate ) {
      std::array<Real, 3> stored{};
      for ( int component = 0; component < cells.dimension; ++component ) {
        stored[component] = cells.integration_vertices( candidate, component );
      }
      found = found || distance( stored, patch.integration_vertices[vertex], cells.dimension ) <= tolerance;
    }
    if ( !found ) {
      return false;
    }
  }
  return true;
}

}  // namespace overlap_cell_detail

template <NormalPolicy Normal>
[[nodiscard]] inline OverlapCellBatch buildOverlapCellBatch(
    ParentTracePairView traces, ArrayView<const ElementPair> candidates,
    const typename geometry::ProjectedOverlap<Normal>::Parameters& parameters = {},
    OverlapCellBuildOptions options = {} )
{
  if ( !traces.isStructurallyValid() ) {
    throw std::invalid_argument( "Overlap cells require structurally valid parent traces." );
  }
  if ( options.mapping_tolerance <= 0.0 || options.orientation_tolerance < 0.0 ) {
    throw std::invalid_argument( "Overlap-cell tolerances must be positive." );
  }
  if ( !candidates.isStructurallyValid() ) {
    throw std::invalid_argument( "Overlap-cell candidates must be structurally valid." );
  }
  std::vector<ElementPair> ordered;
  ordered.reserve( static_cast<std::size_t>( candidates.size() ) );
  for ( Index candidate = 0; candidate < candidates.size(); ++candidate ) {
    ordered.push_back( candidates[candidate] );
  }
  std::sort( ordered.begin(), ordered.end(), []( ElementPair left, ElementPair right ) {
    return left.mortar_element < right.mortar_element ||
           ( left.mortar_element == right.mortar_element && left.nonmortar_element < right.nonmortar_element );
  } );

  OverlapCellBatch result( traces.mortar.geometry.dimension );
  for ( Index attempt = 0; attempt < static_cast<Index>( ordered.size() ); ++attempt ) {
    const ElementPair pair = ordered[attempt];
    InteractionPatch patch;
    std::array<basis::ReferencePoint, maximumOverlapVertices> mortar_reference{};
    std::array<basis::ReferencePoint, maximumOverlapVertices> nonmortar_reference{};
    Index mortar_parent_face{ -1 };
    Index nonmortar_parent_face{ -1 };
    int owner_rank{ -1 };
    GeometryStatus status{ GeometryStatus::InvalidProvenance };
    if ( pair.mortar_element >= 0 && pair.mortar_element < traces.mortar.numberOfSubfacets() &&
         pair.nonmortar_element >= 0 && pair.nonmortar_element < traces.nonmortar.numberOfSubfacets() ) {
      mortar_parent_face = traces.mortar.subfacet_parent_faces[pair.mortar_element];
      nonmortar_parent_face = traces.nonmortar.subfacet_parent_faces[pair.nonmortar_element];
      owner_rank = traces.mortar.parent_face_owners[mortar_parent_face];
      const auto mortar_normal =
          traces.mortar.geometry.dimension == 2
              ? projected_overlap_detail::segmentNormal( traces.mortar.geometry, pair.mortar_element )
              : projected_overlap_detail::faceNormal( traces.mortar.geometry, pair.mortar_element );
      const auto nonmortar_normal =
          traces.nonmortar.geometry.dimension == 2
              ? projected_overlap_detail::segmentNormal( traces.nonmortar.geometry, pair.nonmortar_element )
              : projected_overlap_detail::faceNormal( traces.nonmortar.geometry, pair.nonmortar_element );
      const Real orientation = projected_overlap_detail::dot( mortar_normal, nonmortar_normal );
      if ( projected_overlap_detail::norm( mortar_normal ) <= 1.0e-28 ||
           projected_overlap_detail::norm( nonmortar_normal ) <= 1.0e-28 ) {
        status = GeometryStatus::DegenerateOverlap;
      } else if ( orientation >= -options.orientation_tolerance ) {
        status = GeometryStatus::InconsistentOrientation;
      } else {
        const SurfacePairView surfaces{ traces.mortar.geometry, traces.nonmortar.geometry };
        patch = projectedOverlapPatch<Normal>( surfaces, pair, parameters );
        status = patch.status;
      }
      for ( int vertex = 0; status == GeometryStatus::Accepted && vertex < patch.vertex_count; ++vertex ) {
        const auto mortar = overlap_cell_detail::mapToParent(
            traces.mortar, pair.mortar_element, patch.mortar_vertices[vertex], options.mapping_tolerance );
        const auto nonmortar = overlap_cell_detail::mapToParent(
            traces.nonmortar, pair.nonmortar_element, patch.nonmortar_vertices[vertex], options.mapping_tolerance );
        status = mortar.status != GeometryStatus::Accepted ? mortar.status : nonmortar.status;
        mortar_reference[vertex] = mortar.parent_reference;
        nonmortar_reference[vertex] = nonmortar.parent_reference;
      }
      const auto current = result.view();
      for ( Index previous = 0; status == GeometryStatus::Accepted && previous < current.numberOfAttempts();
            ++previous ) {
        if ( current.contact_pair_ids[previous] == options.contact_pair_id &&
             overlap_cell_detail::sameCell( current, previous, patch, traces.mortar.parent_face_ids[mortar_parent_face],
                                            traces.nonmortar.parent_face_ids[nonmortar_parent_face], traces,
                                            options.mapping_tolerance ) ) {
          status = GeometryStatus::Duplicate;
        }
      }
    }
    result.append( options.contact_pair_id, options.first_cell_id + static_cast<std::uint64_t>( attempt ), owner_rank,
                   pair, mortar_parent_face, nonmortar_parent_face, status, patch, mortar_reference,
                   nonmortar_reference );
  }
  return result;
}

}  // namespace tribol

#endif
