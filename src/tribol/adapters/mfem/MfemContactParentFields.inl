[[nodiscard]] static Index contactEntityCount( const detail::SurfaceStorage& surface )
{
  if constexpr ( tribol::detail::is_native_explicit_method_v<MethodType> ) {
    return surface.parentDofCount();
  } else {
    return surface.view().numberOfNodes();
  }
}

[[nodiscard]] static Index contactValueCount( const detail::SurfaceStorage& surface )
{
  return contactEntityCount( surface ) * surface.view().dimension;
}

[[nodiscard]] static FieldView<const Real> contactField( const std::vector<Real>& values,
                                                         const detail::SurfaceStorage& surface )
{
  return { vectorView( values ), contactEntityCount( surface ), surface.view().dimension, FieldLayout::Interleaved };
}

[[nodiscard]] static FieldView<Real> contactResidualField( std::vector<Real>& values,
                                                           const detail::SurfaceStorage& surface )
{
  return { { values.data(), static_cast<Index>( values.size() ) },
           contactEntityCount( surface ),
           surface.view().dimension,
           FieldLayout::Interleaved };
}

void gatherContactField( const ::mfem::ParGridFunction& field, int components, std::vector<Real>& mortar,
                         std::vector<Real>& nonmortar )
{
  requireFieldSpace( field, components );
  if constexpr ( tribol::detail::is_native_explicit_method_v<MethodType> ) {
    surfaces_.mortar().distributeParentDofField( *field.ParFESpace(), field, components, true, mortar );
    surfaces_.nonmortar().distributeParentDofField( *field.ParFESpace(), field, components, true, nonmortar );
  } else {
    gatherNodalField( field, components, mortar, nonmortar );
  }
}

void distributeParentTrueField( const ::mfem::Vector& true_field, bool apply_vector_dof_sign, std::vector<Real>& mortar,
                                std::vector<Real>& nonmortar )
{
  if ( true_field.Size() != coordinate_space_.GetTrueVSize() ) {
    throw std::invalid_argument( "MFEM parent field must use the coordinate true-DOF space." );
  }
  const auto* prolongation = coordinate_space_.GetProlongationMatrix();
  if ( prolongation == nullptr ) {
    local_parent_residual_ = true_field;
  } else {
    prolongation->Mult( true_field, local_parent_residual_ );
  }
  const int dimension = coordinate_space_.GetVDim();
  surfaces_.mortar().distributeParentDofField( coordinate_space_, local_parent_residual_, dimension,
                                               apply_vector_dof_sign, mortar );
  surfaces_.nonmortar().distributeParentDofField( coordinate_space_, local_parent_residual_, dimension,
                                                  apply_vector_dof_sign, nonmortar );
}

void refreshInverseMass()
{
  if constexpr ( tribol::detail::is_native_explicit_method_v<MethodType> ) {
    if ( has_registered_inverse_mass_ ) {
      distributeParentTrueField( registered_true_inverse_mass_, false, mortar_inverse_mass_, nonmortar_inverse_mass_ );
    }
  }
}

[[nodiscard]] timestep::DiagonalInverseMassView inverseMass() const
{
  if constexpr ( tribol::detail::is_native_explicit_method_v<MethodType> ) {
    if ( has_registered_inverse_mass_ ) {
      return { .mortar = contactField( mortar_inverse_mass_, surfaces_.mortar() ),
               .nonmortar = contactField( nonmortar_inverse_mass_, surfaces_.nonmortar() ),
               .version = inverse_mass_version_ };
    }
  }
  return {};
}

void scatterParentDofField( const detail::SurfaceStorage& surface, ArrayView<const Real> residual )
{
  const int dimension = coordinate_space_.GetVDim();
  const auto local_residual = surface.reduceParentDofValues( residual, dimension );
  for ( std::size_t parent_dof = 0; parent_dof < surface.localParentDofCount(); ++parent_dof ) {
    const int scalar_dof = surface.localParentDof( parent_dof );
    for ( int component = 0; component < dimension; ++component ) {
      const int vector_dof = coordinate_space_.DofToVDof( scalar_dof, component );
      local_parent_residual_[detail::decodeDof( vector_dof )] +=
          detail::dofSign( vector_dof ) *
          local_residual[parent_dof * static_cast<std::size_t>( dimension ) + component];
    }
  }
}
