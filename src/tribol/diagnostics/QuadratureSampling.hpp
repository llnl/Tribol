#ifndef TRIBOL_DIAGNOSTICS_QUADRATURESAMPLING_HPP_
#define TRIBOL_DIAGNOSTICS_QUADRATURESAMPLING_HPP_

#include "tribol/basis/ParentBasis.hpp"
#include "tribol/core/ConstraintRowView.hpp"
#include "tribol/core/OverlapCellView.hpp"
#include "tribol/integration/AutomaticQuadrature.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tribol::diagnostics {

struct QuadratureSamplingOptions {
  int exactness{ -1 };
  bool include_normal_rates{};
};

template <typename Scalar>
struct QuadratureSampleViewT {
  int dimension{};
  ArrayView<const Index> cell_sample_offsets{};
  ArrayView<const std::uint64_t> contact_pair_ids{};
  ArrayView<const std::uint64_t> cell_ids{};
  ArrayView<const int> owner_ranks{};
  ArrayView<const Index> overlap_cells{};
  FieldView<const Scalar> normals{};
  ArrayView<const Scalar> integration_weights{};
  FieldView<const Scalar> mortar_parent_reference_coordinates{};
  FieldView<const Scalar> nonmortar_parent_reference_coordinates{};
  ArrayView<const Scalar> physical_gaps{};
  ArrayView<const Scalar> normal_rates{};

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index numberOfCells() const { return cell_sample_offsets.size() - 1; }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index numberOfSamples() const { return physical_gaps.size(); }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool hasNormalRates() const { return !normal_rates.empty(); }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool isStructurallyValid() const
  {
    const Index samples = numberOfSamples();
    const int reference_dimension = dimension - 1;
    const bool arrays_valid = cell_sample_offsets.isStructurallyValid() && contact_pair_ids.isStructurallyValid() &&
                              cell_ids.isStructurallyValid() && owner_ranks.isStructurallyValid() &&
                              overlap_cells.isStructurallyValid() && integration_weights.isStructurallyValid() &&
                              physical_gaps.isStructurallyValid() && normal_rates.isStructurallyValid();
    const bool sample_sizes = contact_pair_ids.size() == samples && cell_ids.size() == samples &&
                              owner_ranks.size() == samples && overlap_cells.size() == samples &&
                              integration_weights.size() == samples &&
                              ( normal_rates.empty() || normal_rates.size() == samples );
    const bool field_shapes = normals.isStructurallyValid() && normals.entities == samples && normals.components == 3 &&
                              mortar_parent_reference_coordinates.isStructurallyValid() &&
                              mortar_parent_reference_coordinates.entities == samples &&
                              mortar_parent_reference_coordinates.components == reference_dimension &&
                              nonmortar_parent_reference_coordinates.isStructurallyValid() &&
                              nonmortar_parent_reference_coordinates.entities == samples &&
                              nonmortar_parent_reference_coordinates.components == reference_dimension;
    if ( ( dimension != 2 && dimension != 3 ) || numberOfCells() < 0 || !arrays_valid || !sample_sizes ||
         !field_shapes || cell_sample_offsets[0] != 0 || cell_sample_offsets[numberOfCells()] != samples ) {
      return false;
    }
    for ( Index cell = 0; cell < numberOfCells(); ++cell ) {
      if ( cell_sample_offsets[cell] > cell_sample_offsets[cell + 1] ) {
        return false;
      }
    }
    for ( Index sample = 0; sample < samples; ++sample ) {
      if ( overlap_cells[sample] < 0 || overlap_cells[sample] >= numberOfCells() ||
           integration_weights[sample] < 0.0 ) {
        return false;
      }
    }
    return true;
  }
};

using QuadratureSampleView = QuadratureSampleViewT<Real>;

struct QuadratureSampleSummary {
  Index samples{};
  Index owned_samples{};
  Real maximum_penetration{};
  Real maximum_closing_rate{};
};

class QuadratureSamples;

[[nodiscard]] QuadratureSamples sampleQuadrature( ParentTracePairView traces, OverlapCellBatchView cells,
                                                  ParentTraceFieldPairView fields,
                                                  QuadratureSamplingOptions options = {} );

class QuadratureSamples {
 public:
  [[nodiscard]] Index numberOfSamples() const { return static_cast<Index>( physical_gaps_.size() ); }

  [[nodiscard]] QuadratureSampleView view() const
  {
    const Index samples = numberOfSamples();
    const Index reference_values = static_cast<Index>( mortar_parent_reference_coordinates_.size() );
    return {
        .dimension = dimension_,
        .cell_sample_offsets = { cell_sample_offsets_.data(), static_cast<Index>( cell_sample_offsets_.size() ) },
        .contact_pair_ids = { contact_pair_ids_.data(), samples },
        .cell_ids = { cell_ids_.data(), samples },
        .owner_ranks = { owner_ranks_.data(), samples },
        .overlap_cells = { overlap_cells_.data(), samples },
        .normals = { { normals_.data(), static_cast<Index>( normals_.size() ) }, samples, 3, FieldLayout::Interleaved },
        .integration_weights = { integration_weights_.data(), samples },
        .mortar_parent_reference_coordinates = { { mortar_parent_reference_coordinates_.data(), reference_values },
                                                 samples,
                                                 dimension_ - 1,
                                                 FieldLayout::Interleaved },
        .nonmortar_parent_reference_coordinates = { { nonmortar_parent_reference_coordinates_.data(),
                                                      static_cast<Index>(
                                                          nonmortar_parent_reference_coordinates_.size() ) },
                                                    samples,
                                                    dimension_ - 1,
                                                    FieldLayout::Interleaved },
        .physical_gaps = { physical_gaps_.data(), samples },
        .normal_rates = { normal_rates_.data(), static_cast<Index>( normal_rates_.size() ) },
    };
  }

 private:
  friend QuadratureSamples sampleQuadrature( ParentTracePairView, OverlapCellBatchView, ParentTraceFieldPairView,
                                             QuadratureSamplingOptions );

  QuadratureSamples( int dimension, std::vector<Index> cell_sample_offsets )
      : dimension_( dimension ), cell_sample_offsets_( std::move( cell_sample_offsets ) )
  {
  }

  int dimension_{};
  std::vector<Index> cell_sample_offsets_{};
  std::vector<std::uint64_t> contact_pair_ids_{};
  std::vector<std::uint64_t> cell_ids_{};
  std::vector<int> owner_ranks_{};
  std::vector<Index> overlap_cells_{};
  std::vector<Real> normals_{};
  std::vector<Real> integration_weights_{};
  std::vector<Real> mortar_parent_reference_coordinates_{};
  std::vector<Real> nonmortar_parent_reference_coordinates_{};
  std::vector<Real> physical_gaps_{};
  std::vector<Real> normal_rates_{};
};

namespace quadrature_sampling_detail {

inline void requireField( FieldView<const Real> field, const ParentTraceView& trace, const char* message )
{
  if ( !field.isStructurallyValid() || field.components != trace.geometry.dimension ) {
    throw std::invalid_argument( message );
  }
  for ( Index entry = 0; entry < trace.parent_dofs.size(); ++entry ) {
    if ( trace.parent_dofs[entry] < 0 || trace.parent_dofs[entry] >= field.entities ) {
      throw std::invalid_argument( message );
    }
  }
}

[[nodiscard]] inline Real normalDifference( const std::array<Real, 3>& mortar, const std::array<Real, 3>& nonmortar,
                                            const std::array<Real, 3>& normal, int dimension )
{
  Real result{};
  for ( int component = 0; component < dimension; ++component ) {
    result += ( mortar[component] - nonmortar[component] ) * normal[component];
  }
  return result;
}

inline void appendReference( std::vector<Real>& values, basis::ReferencePoint reference, int dimension )
{
  values.push_back( reference.first );
  if ( dimension == 3 ) {
    values.push_back( reference.second );
  }
}

}  // namespace quadrature_sampling_detail

[[nodiscard]] inline QuadratureSamples sampleQuadrature( ParentTracePairView traces, OverlapCellBatchView cells,
                                                         ParentTraceFieldPairView fields,
                                                         QuadratureSamplingOptions options )
{
  if ( !traces.isStructurallyValid() || !cells.isStructurallyValid() ||
       cells.dimension != traces.mortar.geometry.dimension ) {
    throw std::invalid_argument( "Diagnostic quadrature requires compatible parent traces and overlap cells." );
  }
  if ( options.exactness < -1 || options.exactness > integration::maximumAutomaticExactness ) {
    throw std::invalid_argument( "Diagnostic quadrature exactness exceeds the built-in range." );
  }
  quadrature_sampling_detail::requireField( fields.mortar_coordinates, traces.mortar,
                                            "Diagnostic quadrature requires valid mortar parent coordinates." );
  quadrature_sampling_detail::requireField( fields.nonmortar_coordinates, traces.nonmortar,
                                            "Diagnostic quadrature requires valid nonmortar parent coordinates." );
  if ( options.include_normal_rates ) {
    quadrature_sampling_detail::requireField( fields.mortar_velocity, traces.mortar,
                                              "Diagnostic quadrature requires valid mortar parent velocity." );
    quadrature_sampling_detail::requireField( fields.nonmortar_velocity, traces.nonmortar,
                                              "Diagnostic quadrature requires valid nonmortar parent velocity." );
  }

  const Index cell_count = cells.numberOfAttempts();
  std::vector<Index> cell_sample_offsets( static_cast<std::size_t>( cell_count + 1 ), 0 );
  std::vector<int> cell_exactness( static_cast<std::size_t>( cell_count ), 0 );
  for ( Index cell = 0; cell < cell_count; ++cell ) {
    if ( !cells.isAccepted( cell ) ) {
      continue;
    }
    const Index mortar_face = cells.mortar_parent_faces[cell];
    const Index nonmortar_face = cells.nonmortar_parent_faces[cell];
    if ( mortar_face < 0 || mortar_face >= traces.mortar.numberOfParentFaces() || nonmortar_face < 0 ||
         nonmortar_face >= traces.nonmortar.numberOfParentFaces() ) {
      throw std::invalid_argument( "Diagnostic quadrature requires valid overlap-cell parent provenance." );
    }
    const int exactness = options.exactness >= 0 ? options.exactness
                                                 : integration::automaticQuadratureExactness(
                                                       traces, cells, cell, integration::Automatic::Parameters{} );
    if ( exactness > integration::maximumAutomaticExactness ) {
      throw std::invalid_argument( "Diagnostic quadrature exactness exceeds the built-in range." );
    }
    cell_exactness[static_cast<std::size_t>( cell )] = exactness;
    cell_sample_offsets[static_cast<std::size_t>( cell + 1 )] =
        integration::automaticQuadraturePointCount( cells, cell, exactness );
  }
  for ( Index cell = 0; cell < cell_count; ++cell ) {
    cell_sample_offsets[static_cast<std::size_t>( cell + 1 )] += cell_sample_offsets[static_cast<std::size_t>( cell )];
  }

  QuadratureSamples result( cells.dimension, std::move( cell_sample_offsets ) );
  const Index sample_count = result.cell_sample_offsets_.back();
  result.contact_pair_ids_.reserve( sample_count );
  result.cell_ids_.reserve( sample_count );
  result.owner_ranks_.reserve( sample_count );
  result.overlap_cells_.reserve( sample_count );
  result.normals_.reserve( 3 * sample_count );
  result.integration_weights_.reserve( sample_count );
  result.mortar_parent_reference_coordinates_.reserve( ( cells.dimension - 1 ) * sample_count );
  result.nonmortar_parent_reference_coordinates_.reserve( ( cells.dimension - 1 ) * sample_count );
  result.physical_gaps_.reserve( sample_count );
  if ( options.include_normal_rates ) {
    result.normal_rates_.reserve( sample_count );
  }

  for ( Index cell = 0; cell < cell_count; ++cell ) {
    const Index count = result.cell_sample_offsets_[static_cast<std::size_t>( cell + 1 )] -
                        result.cell_sample_offsets_[static_cast<std::size_t>( cell )];
    for ( Index local_sample = 0; local_sample < count; ++local_sample ) {
      const auto point = integration::automaticQuadraturePoint( cells, cell, local_sample,
                                                                cell_exactness[static_cast<std::size_t>( cell )] );
      const Index mortar_face = cells.mortar_parent_faces[cell];
      const Index nonmortar_face = cells.nonmortar_parent_faces[cell];
      const auto mortar_position = basis::interpolateParentField( traces.mortar, mortar_face, fields.mortar_coordinates,
                                                                  point.mortar_reference );
      const auto nonmortar_position = basis::interpolateParentField(
          traces.nonmortar, nonmortar_face, fields.nonmortar_coordinates, point.nonmortar_reference );
      result.contact_pair_ids_.push_back( cells.contact_pair_ids[cell] );
      result.cell_ids_.push_back( cells.cell_ids[cell] );
      result.owner_ranks_.push_back( cells.owner_ranks[cell] );
      result.overlap_cells_.push_back( cell );
      result.normals_.insert( result.normals_.end(), point.normal.begin(), point.normal.end() );
      result.integration_weights_.push_back( point.weight );
      quadrature_sampling_detail::appendReference( result.mortar_parent_reference_coordinates_, point.mortar_reference,
                                                   cells.dimension );
      quadrature_sampling_detail::appendReference( result.nonmortar_parent_reference_coordinates_,
                                                   point.nonmortar_reference, cells.dimension );
      result.physical_gaps_.push_back( quadrature_sampling_detail::normalDifference(
          mortar_position, nonmortar_position, point.normal, cells.dimension ) );
      if ( options.include_normal_rates ) {
        const auto mortar_velocity =
            basis::interpolateParentField( traces.mortar, mortar_face, fields.mortar_velocity, point.mortar_reference );
        const auto nonmortar_velocity = basis::interpolateParentField(
            traces.nonmortar, nonmortar_face, fields.nonmortar_velocity, point.nonmortar_reference );
        result.normal_rates_.push_back( quadrature_sampling_detail::normalDifference(
            mortar_velocity, nonmortar_velocity, point.normal, cells.dimension ) );
      }
    }
  }
  return result;
}

[[nodiscard]] inline QuadratureSampleSummary summarizeQuadrature( QuadratureSampleView samples, int local_rank = -1 )
{
  if ( !samples.isStructurallyValid() ) {
    throw std::invalid_argument( "Diagnostic quadrature summary requires valid samples." );
  }
  QuadratureSampleSummary result;
  result.samples = samples.numberOfSamples();
  for ( Index sample = 0; sample < samples.numberOfSamples(); ++sample ) {
    if ( local_rank >= 0 && samples.owner_ranks[sample] != local_rank ) {
      continue;
    }
    ++result.owned_samples;
    const Real penetration = samples.physical_gaps[sample] < 0.0 ? -samples.physical_gaps[sample] : 0.0;
    const Real closing_rate =
        samples.hasNormalRates() && samples.normal_rates[sample] < 0.0 ? -samples.normal_rates[sample] : 0.0;
    result.maximum_penetration = penetration > result.maximum_penetration ? penetration : result.maximum_penetration;
    result.maximum_closing_rate =
        closing_rate > result.maximum_closing_rate ? closing_rate : result.maximum_closing_rate;
  }
  return result;
}

static_assert( std::is_trivially_copyable_v<QuadratureSamplingOptions> );
static_assert( std::is_trivially_copyable_v<QuadratureSampleView> );
static_assert( std::is_trivially_copyable_v<QuadratureSampleSummary> );

}  // namespace tribol::diagnostics

#endif
