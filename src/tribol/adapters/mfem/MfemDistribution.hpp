#ifndef TRIBOL_ADAPTERS_MFEM_MFEMDISTRIBUTION_HPP_
#define TRIBOL_ADAPTERS_MFEM_MFEMDISTRIBUTION_HPP_

#include "tribol/search/Proximity.hpp"

#include <mpi.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tribol::mfem::detail {

struct DistributionParameters {
  Real expansion{};
  Real proximity_scale{};
  bool replicate{};
};

template <typename T>
MPI_Datatype mpiType()
{
  if constexpr ( std::is_same_v<T, Real> ) {
    return std::is_same_v<Real, float> ? MPI_FLOAT : MPI_DOUBLE;
  } else {
    static_assert( std::is_same_v<T, std::int64_t> );
    return MPI_INT64_T;
  }
}

template <typename T>
struct ExchangeResult {
  std::vector<T> values;
  std::vector<int> receive_counts;
};

inline int checkedMpiCount( std::size_t count )
{
  if ( count > static_cast<std::size_t>( std::numeric_limits<int>::max() ) ) {
    throw std::overflow_error( "MFEM surface exchange exceeds the MPI count range." );
  }
  return static_cast<int>( count );
}

inline int checkedMpiCountProduct( int count, int components )
{
  if ( count < 0 || components <= 0 || count > std::numeric_limits<int>::max() / components ) {
    throw std::overflow_error( "MFEM surface exchange exceeds the MPI count range." );
  }
  return count * components;
}

inline int checkedMpiCountSum( int total, int count )
{
  if ( count < 0 || total > std::numeric_limits<int>::max() - count ) {
    throw std::overflow_error( "MFEM surface exchange exceeds the MPI count range." );
  }
  return total + count;
}

template <typename T>
class ExchangeWorkspace {
 public:
  void configure( const std::vector<unsigned char>& destinations, std::size_t local_entities,
                  const std::vector<int>& received_entity_counts, int components )
  {
    if ( destinations.size() != received_entity_counts.size() ) {
      throw std::logic_error( "MFEM surface exchange topology has inconsistent rank counts." );
    }
    const int local_values = checkedMpiCountProduct( checkedMpiCount( local_entities ), components );
    const auto rank_count = destinations.size();
    send_counts_.resize( rank_count );
    receive_counts_.resize( rank_count );
    send_displacements_.resize( rank_count );
    receive_displacements_.resize( rank_count );
    int send_total{};
    int receive_total{};
    for ( std::size_t rank = 0; rank < rank_count; ++rank ) {
      send_displacements_[rank] = send_total;
      receive_displacements_[rank] = receive_total;
      send_counts_[rank] = destinations[rank] ? local_values : 0;
      receive_counts_[rank] = checkedMpiCountProduct( received_entity_counts[rank], components );
      send_total = checkedMpiCountSum( send_total, send_counts_[rank] );
      receive_total = checkedMpiCountSum( receive_total, receive_counts_[rank] );
    }
    local_values_.resize( static_cast<std::size_t>( local_values ) );
    exchange_buffer_.resize( static_cast<std::size_t>( send_total ) );
    reduced_values_.resize( static_cast<std::size_t>( local_values ) );
    distributed_value_count_ = receive_total;
  }

  [[nodiscard]] ArrayView<T> localValues()
  {
    return { local_values_.data(), static_cast<Index>( local_values_.size() ) };
  }

  [[nodiscard]] std::vector<T>& localValueStorage() { return local_values_; }

  void distribute( MPI_Comm communicator, std::vector<T>& distributed )
  {
    for ( std::size_t rank = 0; rank < send_counts_.size(); ++rank ) {
      if ( send_counts_[rank] > 0 ) {
        std::copy( local_values_.begin(), local_values_.end(), exchange_buffer_.begin() + send_displacements_[rank] );
      }
    }
    distributed.resize( static_cast<std::size_t>( distributed_value_count_ ) );
    MPI_Alltoallv( exchange_buffer_.data(), send_counts_.data(), send_displacements_.data(), mpiType<T>(),
                   distributed.data(), receive_counts_.data(), receive_displacements_.data(), mpiType<T>(),
                   communicator );
  }

  [[nodiscard]] ArrayView<const T> reduce( MPI_Comm communicator, ArrayView<const T> distributed )
  {
    if ( distributed.size() != distributed_value_count_ ) {
      throw std::invalid_argument( "MFEM distributed field has an invalid size." );
    }
    MPI_Alltoallv( distributed.data(), receive_counts_.data(), receive_displacements_.data(), mpiType<T>(),
                   exchange_buffer_.data(), send_counts_.data(), send_displacements_.data(), mpiType<T>(),
                   communicator );
    std::fill( reduced_values_.begin(), reduced_values_.end(), T{} );
    for ( std::size_t source = 0; source < send_counts_.size(); ++source ) {
      const int count = send_counts_[source];
      const int offset = send_displacements_[source];
      for ( int value = 0; value < count; ++value ) {
        reduced_values_[static_cast<std::size_t>( value )] +=
            exchange_buffer_[static_cast<std::size_t>( offset + value )];
      }
    }
    return { reduced_values_.data(), static_cast<Index>( reduced_values_.size() ) };
  }

 private:
  std::vector<int> send_counts_;
  std::vector<int> receive_counts_;
  std::vector<int> send_displacements_;
  std::vector<int> receive_displacements_;
  std::vector<T> local_values_;
  std::vector<T> exchange_buffer_;
  std::vector<T> reduced_values_;
  int distributed_value_count_{};
};

template <typename T>
ExchangeResult<T> exchangeToDestinations( MPI_Comm communicator, const std::vector<T>& local,
                                          const std::vector<unsigned char>& destinations )
{
  int communicator_size{};
  MPI_Comm_size( communicator, &communicator_size );
  if ( destinations.size() != static_cast<std::size_t>( communicator_size ) ) {
    throw std::logic_error( "MFEM surface destination map does not match its communicator." );
  }
  std::vector<int> send_counts( static_cast<std::size_t>( communicator_size ) );
  std::vector<int> receive_counts( static_cast<std::size_t>( communicator_size ) );
  for ( int rank = 0; rank < communicator_size; ++rank ) {
    send_counts[static_cast<std::size_t>( rank )] =
        destinations[static_cast<std::size_t>( rank )] ? static_cast<int>( local.size() ) : 0;
  }
  MPI_Alltoall( send_counts.data(), 1, MPI_INT, receive_counts.data(), 1, MPI_INT, communicator );
  std::vector<int> send_displacements( static_cast<std::size_t>( communicator_size ) );
  std::vector<int> receive_displacements( static_cast<std::size_t>( communicator_size ) );
  int send_total{};
  int receive_total{};
  for ( int rank = 0; rank < communicator_size; ++rank ) {
    send_displacements[static_cast<std::size_t>( rank )] = send_total;
    receive_displacements[static_cast<std::size_t>( rank )] = receive_total;
    send_total += send_counts[static_cast<std::size_t>( rank )];
    receive_total += receive_counts[static_cast<std::size_t>( rank )];
  }
  std::vector<T> send_buffer( static_cast<std::size_t>( send_total ) );
  for ( int rank = 0; rank < communicator_size; ++rank ) {
    if ( send_counts[static_cast<std::size_t>( rank )] > 0 ) {
      std::copy( local.begin(), local.end(),
                 send_buffer.begin() + send_displacements[static_cast<std::size_t>( rank )] );
    }
  }
  ExchangeResult<T> result;
  result.receive_counts = receive_counts;
  result.values.resize( static_cast<std::size_t>( receive_total ) );
  MPI_Alltoallv( send_buffer.data(), send_counts.data(), send_displacements.data(), mpiType<T>(), result.values.data(),
                 receive_counts.data(), receive_displacements.data(), mpiType<T>(), communicator );
  return result;
}

inline search::detail::BoundingBox expandedBounds( const SurfaceMeshView& surface, Real expansion, Real proximity_scale,
                                                   bool& valid )
{
  search::detail::BoundingBox result;
  valid = surface.numberOfElements() > 0;
  if ( !valid ) {
    return result;
  }
  result = search::detail::elementBounds( surface, 0, expansion, proximity_scale );
  for ( Index element = 1; element < surface.numberOfElements(); ++element ) {
    const auto bounds = search::detail::elementBounds( surface, element, expansion, proximity_scale );
    for ( int component = 0; component < surface.dimension; ++component ) {
      result.minimum[component] = std::min( result.minimum[component], bounds.minimum[component] );
      result.maximum[component] = std::max( result.maximum[component], bounds.maximum[component] );
    }
  }
  return result;
}

inline std::vector<unsigned char> ghostDestinations( MPI_Comm communicator, const SurfaceMeshView& local_mortar,
                                                     const search::detail::BoundingBox& nonmortar, bool nonmortar_valid,
                                                     DistributionParameters parameters )
{
  int rank{};
  int communicator_size{};
  MPI_Comm_rank( communicator, &rank );
  MPI_Comm_size( communicator, &communicator_size );
  if ( parameters.replicate ) {
    return std::vector<unsigned char>( static_cast<std::size_t>( communicator_size ), 1 );
  }
  bool mortar_valid{};
  const auto mortar = expandedBounds( local_mortar, parameters.expansion, parameters.proximity_scale, mortar_valid );
  std::array<Real, 7> packed{ mortar.minimum[0], mortar.minimum[1], mortar.minimum[2],       mortar.maximum[0],
                              mortar.maximum[1], mortar.maximum[2], mortar_valid ? 1.0 : 0.0 };
  std::vector<Real> gathered( static_cast<std::size_t>( 7 * communicator_size ) );
  MPI_Allgather( packed.data(), 7, mpiType<Real>(), gathered.data(), 7, mpiType<Real>(), communicator );
  std::vector<unsigned char> destinations( static_cast<std::size_t>( communicator_size ) );
  if ( !nonmortar_valid ) {
    return destinations;
  }
  for ( int destination = 0; destination < communicator_size; ++destination ) {
    const auto offset = static_cast<std::size_t>( 7 * destination );
    if ( gathered[offset + 6] == 0.0 ) {
      continue;
    }
    search::detail::BoundingBox remote;
    for ( int component = 0; component < 3; ++component ) {
      remote.minimum[component] = gathered[offset + static_cast<std::size_t>( component )];
      remote.maximum[component] = gathered[offset + static_cast<std::size_t>( component + 3 )];
    }
    destinations[static_cast<std::size_t>( destination )] =
        search::detail::overlaps( remote, nonmortar, local_mortar.dimension );
  }
  return destinations;
}

inline std::vector<unsigned char> ghostDestinations( MPI_Comm communicator, const SurfaceMeshView& local_mortar,
                                                     const SurfaceMeshView& local_nonmortar,
                                                     DistributionParameters parameters )
{
  bool nonmortar_valid{};
  const auto nonmortar =
      expandedBounds( local_nonmortar, parameters.expansion, parameters.proximity_scale, nonmortar_valid );
  return ghostDestinations( communicator, local_mortar, nonmortar, nonmortar_valid, parameters );
}

inline std::vector<unsigned char> localDestination( MPI_Comm communicator )
{
  int rank{};
  int communicator_size{};
  MPI_Comm_rank( communicator, &rank );
  MPI_Comm_size( communicator, &communicator_size );
  std::vector<unsigned char> result( static_cast<std::size_t>( communicator_size ) );
  result[static_cast<std::size_t>( rank )] = 1;
  return result;
}

}  // namespace tribol::mfem::detail

#endif
