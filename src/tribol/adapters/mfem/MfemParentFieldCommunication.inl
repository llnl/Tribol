inline void SurfaceStorage::distributeParentDofField( const ::mfem::ParFiniteElementSpace& space,
                                                      const ::mfem::Vector& local_field, int components,
                                                      bool apply_vector_dof_sign, std::vector<Real>& values ) const
{
  if ( components <= 0 || space.GetVDim() != components || local_field.Size() != space.GetVSize() ) {
    throw std::invalid_argument( "MFEM parent field must match the supplied local finite-element space." );
  }
  if ( components != dimension_ ) {
    throw std::invalid_argument( "MFEM parent exchange requires a spatial vector field." );
  }
  auto& local = parent_vector_exchange_.localValueStorage();
  collectLocalParentField( space, local_field, components, apply_vector_dof_sign, local );
  parent_vector_exchange_.distribute( mesh_.GetComm(), values );
  if ( values.size() != static_cast<std::size_t>( parentDofCount() * components ) ) {
    throw std::logic_error( "MFEM distributed parent field has an invalid size." );
  }
}

inline ArrayView<const Real> SurfaceStorage::reduceParentDofValues( ArrayView<const Real> distributed,
                                                                    int components ) const
{
  if ( components != dimension_ ) {
    throw std::invalid_argument( "MFEM parent exchange requires a spatial vector field." );
  }
  return parent_vector_exchange_.reduce( mesh_.GetComm(), distributed );
}
