#ifndef TRIBOL_ADAPTERS_MFEM_MFEMDISTRIBUTION_HPP_
#define TRIBOL_ADAPTERS_MFEM_MFEMDISTRIBUTION_HPP_

#include "tribol/search/Proximity.hpp"

#include <mpi.h>

#include <array>
#include <cstdint>
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
