inline void SurfaceStorage::distributeParentDofField( const ::mfem::ParFiniteElementSpace& space,
                                                      const ::mfem::Vector& local_field, int components,
                                                      bool apply_vector_dof_sign, std::vector<Real>& values ) const
{
  if ( components <= 0 || space.GetVDim() != components || local_field.Size() != space.GetVSize() ) {
    throw std::invalid_argument( "MFEM parent field must match the supplied local finite-element space." );
  }
  std::vector<Real> local;
  collectLocalParentField( space, local_field, components, apply_vector_dof_sign, local );
  values = exchangeToDestinations( mesh_.GetComm(), local, destinations_ ).values;
  if ( values.size() != static_cast<std::size_t>( parentDofCount() * components ) ) {
    throw std::logic_error( "MFEM distributed parent field has an invalid size." );
  }
}

inline std::vector<Real> SurfaceStorage::reduceParentDofValues( ArrayView<const Real> distributed,
                                                                int components ) const
{
  const auto local_values = localParentDofCount() * static_cast<std::size_t>( components );
  if ( distributed.size() != parentDofCount() * components ) {
    throw std::invalid_argument( "MFEM distributed parent field has an invalid size." );
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
        received_parent_dof_counts_[static_cast<std::size_t>( owner )] * components;
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
      throw std::logic_error( "MFEM reverse parent-DOF packet has an invalid size." );
    }
    const int offset = receive_displacements[static_cast<std::size_t>( source )];
    for ( int value = 0; value < count; ++value ) {
      result[static_cast<std::size_t>( value )] += received[static_cast<std::size_t>( offset + value )];
    }
  }
  return result;
}
