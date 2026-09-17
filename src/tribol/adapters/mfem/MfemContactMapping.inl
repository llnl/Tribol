ContactStateView restrictState( const ContactState& state )
{
  ContactStateView result;
  if ( state.velocity != nullptr ) {
    gatherNodalField( *state.velocity, coordinate_space_.GetVDim(), mortar_velocity_, nonmortar_velocity_ );
    result.mortar_velocity = surfaceField( mortar_velocity_, surfaces_.view().mortar );
    result.nonmortar_velocity = surfaceField( nonmortar_velocity_, surfaces_.view().nonmortar );
  }
  if ( state.reference_coordinates != nullptr ) {
    gatherNodalField( *state.reference_coordinates, coordinate_space_.GetVDim(), mortar_reference_coordinates_,
                      nonmortar_reference_coordinates_ );
    result.mortar_reference_coordinates = surfaceField( mortar_reference_coordinates_, surfaces_.view().mortar );
    result.nonmortar_reference_coordinates =
        surfaceField( nonmortar_reference_coordinates_, surfaces_.view().nonmortar );
  }
  if ( state.element_thickness != nullptr ) {
    surfaces_.mortar().sampleElementCoefficient( *state.element_thickness, mortar_element_thickness_ );
    surfaces_.nonmortar().sampleElementCoefficient( *state.element_thickness, nonmortar_element_thickness_ );
  } else if ( MethodTraits<MethodType>::capabilities.needs_material_fields || core_.options().timestep.enabled ||
              core_.options().method.constraint.activation.reject_excessive_penetration ) {
    surfaces_.mortar().computeElementThickness( mortar_element_thickness_ );
    surfaces_.nonmortar().computeElementThickness( nonmortar_element_thickness_ );
  }
  if ( state.material_modulus != nullptr ) {
    surfaces_.mortar().sampleElementCoefficient( *state.material_modulus, mortar_material_modulus_ );
    surfaces_.nonmortar().sampleElementCoefficient( *state.material_modulus, nonmortar_material_modulus_ );
  }
  if ( MethodTraits<MethodType>::capabilities.needs_material_fields || core_.options().timestep.enabled ||
       core_.options().method.constraint.activation.reject_excessive_penetration ) {
    result.mortar_element_thickness = vectorView( mortar_element_thickness_ );
    result.nonmortar_element_thickness = vectorView( nonmortar_element_thickness_ );
  }
  if ( MethodTraits<MethodType>::capabilities.needs_material_fields ) {
    if ( state.material_modulus == nullptr ) {
      throw std::invalid_argument( "MFEM material penalty requires a material-modulus coefficient." );
    }
    result.mortar_material_modulus = vectorView( mortar_material_modulus_ );
    result.nonmortar_material_modulus = vectorView( nonmortar_material_modulus_ );
  }
  if ( state.multiplier != nullptr ) {
    gatherScalarField( surfaces_.mortar(), *state.multiplier, multiplier_ );
    result.multiplier = vectorView( multiplier_ );
  }
  if ( state.external_potential_density != nullptr ) {
    gatherScalarField( surfaces_.mortar(), *state.external_potential_density, external_potential_density_ );
    result.external_potential_density = vectorView( external_potential_density_ );
  }
  if ( state.external_pressure != nullptr ) {
    gatherScalarField( surfaces_.mortar(), *state.external_pressure, external_pressure_ );
    result.external_pressure = vectorView( external_pressure_ );
  }
  if ( state.external_pressure_tangent != nullptr ) {
    gatherScalarField( surfaces_.mortar(), *state.external_pressure_tangent, external_pressure_tangent_ );
    result.external_pressure_tangent = vectorView( external_pressure_tangent_ );
  }
  return result;
}

private:
static detail::DistributionParameters distributionParameters( const Options& options )
{
  detail::DistributionParameters result;
  if constexpr ( requires {
                   options.search.expansion;
                   options.search.proximity_scale;
                 } ) {
    result.expansion = std::max( options.search.expansion, options.method.constraint.activation.residual_gap );
    result.proximity_scale = options.search.proximity_scale;
  } else {
    result.replicate = true;
  }
  return result;
}

EvaluationSummary globalSummary( EvaluationSummary local ) const
{
  EvaluationSummary global;
  MPI_Allreduce( &local.energy, &global.energy, 1, detail::realMpiType(), MPI_SUM, mesh_.GetComm() );
  MPI_Allreduce( &local.timestep_vote, &global.timestep_vote, 1, detail::realMpiType(), MPI_MIN, mesh_.GetComm() );
  std::int64_t local_counts[2] = { local.active_interactions, local.quadrature_points };
  std::int64_t global_counts[2]{};
  MPI_Allreduce( local_counts, global_counts, 2, MPI_INT64_T, MPI_SUM, mesh_.GetComm() );
  global.active_interactions = static_cast<Index>( global_counts[0] );
  global.quadrature_points = static_cast<Index>( global_counts[1] );
  return global;
}

void resizeSurfaceWorkspace()
{
  const auto surfaces = surfaces_.view();
  const auto mortar_values = static_cast<std::size_t>( surfaces.mortar.coordinates.values.size() );
  const auto nonmortar_values = static_cast<std::size_t>( surfaces.nonmortar.coordinates.values.size() );
  mortar_residual_.resize( mortar_values );
  nonmortar_residual_.resize( nonmortar_values );
  mortar_direction_.resize( mortar_values );
  nonmortar_direction_.resize( nonmortar_values );
  mortar_derivative_.resize( mortar_values );
  nonmortar_derivative_.resize( nonmortar_values );
  mortar_velocity_.resize( mortar_values );
  nonmortar_velocity_.resize( nonmortar_values );
  mortar_reference_coordinates_.resize( mortar_values );
  nonmortar_reference_coordinates_.resize( nonmortar_values );
  mortar_element_thickness_.resize( static_cast<std::size_t>( surfaces.mortar.numberOfElements() ) );
  nonmortar_element_thickness_.resize( static_cast<std::size_t>( surfaces.nonmortar.numberOfElements() ) );
  mortar_material_modulus_.resize( mortar_element_thickness_.size() );
  nonmortar_material_modulus_.resize( nonmortar_element_thickness_.size() );
  multiplier_.resize( static_cast<std::size_t>( surfaces.mortar.numberOfNodes() ) );
  external_potential_density_.resize( multiplier_.size() );
  external_pressure_.resize( multiplier_.size() );
  external_pressure_tangent_.resize( multiplier_.size() );
  multiplier_direction_.resize( multiplier_.size() );
  constraint_residual_.resize( multiplier_.size() );
  constraint_derivative_.resize( multiplier_.size() );
}

static ::mfem::ParFiniteElementSpace& requireCoordinateSpace( const ::mfem::ParMesh& mesh,
                                                              const ::mfem::ParGridFunction& coordinates )
{
  auto* space = coordinates.ParFESpace();
  if ( space == nullptr || space->GetParMesh() != &mesh || space->GetVDim() != mesh.SpaceDimension() ) {
    throw std::invalid_argument( "MFEM coordinates must be a vector field on the supplied ParMesh." );
  }
  return *space;
}

void requireSameSpace( const ::mfem::ParGridFunction& coordinates ) const
{
  if ( coordinates.ParFESpace() != &coordinate_space_ ) {
    throw std::invalid_argument( "MFEM geometry updates must use the original coordinate finite-element space." );
  }
}

const ::mfem::ParFiniteElementSpace& requireScalarSpace( const ::mfem::ParGridFunction& field ) const
{
  const auto* space = field.ParFESpace();
  if ( space == nullptr || space->GetParMesh() != &mesh_ || space->GetVDim() != 1 ||
       space->GetNDofs() != coordinate_space_.GetNDofs() ) {
    throw std::invalid_argument( "MFEM multiplier and pressure fields must use the coordinate scalar basis." );
  }
  return *space;
}

static ArrayView<const Real> vectorView( const std::vector<Real>& values )
{
  return { values.data(), static_cast<Index>( values.size() ) };
}

static FieldView<const Real> surfaceField( const std::vector<Real>& values, const SurfaceMeshView& surface )
{
  return { vectorView( values ), surface.numberOfNodes(), surface.dimension, FieldLayout::Interleaved };
}

void requireFieldSpace( const ::mfem::ParGridFunction& field, int components ) const
{
  const auto* space = field.ParFESpace();
  if ( space == nullptr || space->GetParMesh() != &mesh_ || space->GetVDim() != components ||
       space->GetNDofs() != coordinate_space_.GetNDofs() ) {
    throw std::invalid_argument( "MFEM state fields must use the coordinate scalar basis on the supplied mesh." );
  }
}

void gatherNodalField( const ::mfem::ParGridFunction& field, int components, std::vector<Real>& mortar,
                       std::vector<Real>& nonmortar )
{
  requireFieldSpace( field, components );
  gatherSurfaceField( surfaces_.mortar(), field, components, mortar );
  gatherSurfaceField( surfaces_.nonmortar(), field, components, nonmortar );
}

void gatherScalarField( const detail::SurfaceStorage& surface, const ::mfem::ParGridFunction& field,
                        std::vector<Real>& values )
{
  requireFieldSpace( field, 1 );
  gatherSurfaceField( surface, field, 1, values );
}

void restrictScalarPrimal( const detail::SurfaceStorage& surface, const ::mfem::ParFiniteElementSpace& space,
                           const ::mfem::Vector& true_field, std::vector<Real>& values )
{
  if ( true_field.Size() != space.GetTrueVSize() ) {
    throw std::invalid_argument( "MFEM scalar restriction requires the scalar true-DOF space." );
  }
  local_scalar_field_.SetSize( space.GetVSize() );
  const auto* prolongation = space.GetProlongationMatrix();
  if ( prolongation == nullptr ) {
    local_scalar_field_ = true_field;
  } else {
    prolongation->Mult( true_field, local_scalar_field_ );
  }
  std::vector<Real> local( surface.localNodeCount() );
  for ( std::size_t local_node = 0; local_node < local.size(); ++local_node ) {
    const auto& restriction = surface.localRestrictionByIndex( local_node );
    for ( std::size_t term = 0; term < restriction.scalar_dofs.size(); ++term ) {
      const int scalar_dof = restriction.scalar_dofs[term];
      local[local_node] += restriction.weights[term] * detail::dofSign( scalar_dof ) *
                           local_scalar_field_[detail::decodeDof( scalar_dof )];
    }
  }
  values = surface.distributeNodeValues( local, 1 );
}

void addScalarDualTranspose( const detail::SurfaceStorage& surface, const ::mfem::ParFiniteElementSpace& space,
                             ArrayView<const Real> surface_residual, ::mfem::Vector& true_residual )
{
  if ( true_residual.Size() != space.GetTrueVSize() || surface_residual.size() != surface.view().numberOfNodes() ) {
    throw std::invalid_argument( "MFEM scalar dual transpose requires matching surface and true-DOF fields." );
  }
  local_scalar_residual_.SetSize( space.GetVSize() );
  local_scalar_residual_ = 0.0;
  const auto local_residual = surface.reduceNodeValues( surface_residual, 1 );
  for ( std::size_t local_node = 0; local_node < local_residual.size(); ++local_node ) {
    const auto& restriction = surface.localRestrictionByIndex( local_node );
    for ( std::size_t term = 0; term < restriction.scalar_dofs.size(); ++term ) {
      const int scalar_dof = restriction.scalar_dofs[term];
      local_scalar_residual_[detail::decodeDof( scalar_dof )] +=
          restriction.weights[term] * detail::dofSign( scalar_dof ) * local_residual[local_node];
    }
  }
  true_scalar_residual_.SetSize( space.GetTrueVSize() );
  true_scalar_residual_ = 0.0;
  const auto* prolongation = space.GetProlongationMatrix();
  if ( prolongation == nullptr ) {
    true_scalar_residual_ = local_scalar_residual_;
  } else {
    prolongation->MultTranspose( local_scalar_residual_, true_scalar_residual_ );
  }
  true_residual += true_scalar_residual_;
}

void projectMortarField( ArrayView<const Real> values, ::mfem::ParGridFunction& destination )
{
  auto& space = const_cast<::mfem::ParFiniteElementSpace&>( requireScalarSpace( destination ) );
  if ( values.size() != surfaces_.view().mortar.numberOfNodes() ) {
    throw std::invalid_argument( "MFEM mortar field projection requires one value per compact mortar node." );
  }
  local_scalar_field_.SetSize( space.GetVSize() );
  local_scalar_residual_.SetSize( space.GetVSize() );
  local_scalar_field_ = 0.0;
  local_scalar_residual_ = 0.0;
  const auto& surface = surfaces_.mortar();
  for ( std::size_t compact_node = 0; compact_node < static_cast<std::size_t>( values.size() ); ++compact_node ) {
    if ( !surface.isLocallyOwned( compact_node ) ) {
      continue;
    }
    const auto& restriction = surface.localRestriction( compact_node );
    for ( std::size_t term = 0; term < restriction.scalar_dofs.size(); ++term ) {
      const int scalar_dof = restriction.scalar_dofs[term];
      const int dof = detail::decodeDof( scalar_dof );
      const Real weight = restriction.weights[term] * detail::dofSign( scalar_dof );
      local_scalar_field_[dof] += weight * values[compact_node];
      local_scalar_residual_[dof] += weight * weight;
    }
  }
  ::mfem::Vector true_values( space.GetTrueVSize() );
  ::mfem::Vector true_weights( space.GetTrueVSize() );
  const auto* prolongation = space.GetProlongationMatrix();
  if ( prolongation == nullptr ) {
    true_values = local_scalar_field_;
    true_weights = local_scalar_residual_;
  } else {
    prolongation->MultTranspose( local_scalar_field_, true_values );
    prolongation->MultTranspose( local_scalar_residual_, true_weights );
  }
  for ( int dof = 0; dof < true_values.Size(); ++dof ) {
    true_values[dof] = true_weights[dof] > 1.0e-28 ? true_values[dof] / true_weights[dof] : 0.0;
  }
  destination.SetFromTrueDofs( true_values );
}

template <typename Action>
std::unique_ptr<::mfem::HypreParMatrix> assembleParallelOperator( const ::mfem::ParFiniteElementSpace& row_space,
                                                                  const ::mfem::ParFiniteElementSpace& column_space,
                                                                  Action action, Real drop_tolerance )
{
  if ( drop_tolerance < 0.0 ) {
    throw std::invalid_argument( "MFEM Jacobian drop tolerance cannot be negative." );
  }
  const int local_rows = row_space.GetTrueVSize();
  const int local_columns = column_space.GetTrueVSize();
  const HYPRE_BigInt global_rows = row_space.GlobalTrueVSize();
  const HYPRE_BigInt global_columns = column_space.GlobalTrueVSize();
  const HYPRE_BigInt local_column_begin = column_space.GetMyTDofOffset();
  std::vector<Real> dense( static_cast<std::size_t>( local_rows ) * static_cast<std::size_t>( global_columns ) );
  ::mfem::Vector direction( local_columns );
  ::mfem::Vector result( local_rows );
  for ( HYPRE_BigInt column = 0; column < global_columns; ++column ) {
    direction = 0.0;
    result = 0.0;
    if ( column >= local_column_begin && column < local_column_begin + local_columns ) {
      direction[static_cast<int>( column - local_column_begin )] = 1.0;
    }
    action( direction, result );
    for ( int row = 0; row < local_rows; ++row ) {
      dense[static_cast<std::size_t>( row ) * static_cast<std::size_t>( global_columns ) +
            static_cast<std::size_t>( column )] = result[row];
    }
  }
  std::vector<int> row_offsets( static_cast<std::size_t>( local_rows ) + 1 );
  std::vector<HYPRE_BigInt> columns;
  std::vector<::mfem::real_t> values;
  for ( int row = 0; row < local_rows; ++row ) {
    for ( HYPRE_BigInt column = 0; column < global_columns; ++column ) {
      const Real value = dense[static_cast<std::size_t>( row ) * static_cast<std::size_t>( global_columns ) +
                               static_cast<std::size_t>( column )];
      if ( std::abs( value ) > drop_tolerance ) {
        columns.push_back( column );
        values.push_back( value );
      }
    }
    row_offsets[static_cast<std::size_t>( row + 1 )] = static_cast<int>( columns.size() );
  }
  return std::make_unique<::mfem::HypreParMatrix>( mesh_.GetComm(), local_rows, global_rows, global_columns,
                                                   row_offsets.data(), columns.data(), values.data(),
                                                   row_space.GetTrueDofOffsets(), column_space.GetTrueDofOffsets() );
}

std::vector<bool> activeMultiplierRows( const ContactState& state,
                                        const ::mfem::ParFiniteElementSpace& multiplier_space )
{
  std::vector<bool> active( static_cast<std::size_t>( multiplier_space.GetTrueVSize() ), false );
  const int local_columns = coordinate_space_.GetTrueVSize();
  const HYPRE_BigInt global_columns = coordinate_space_.GlobalTrueVSize();
  const HYPRE_BigInt local_column_begin = coordinate_space_.GetMyTDofOffset();
  ::mfem::Vector direction( local_columns );
  ::mfem::Vector result( multiplier_space.GetTrueVSize() );
  for ( HYPRE_BigInt column = 0; column < global_columns; ++column ) {
    direction = 0.0;
    result = 0.0;
    if ( column >= local_column_begin && column < local_column_begin + local_columns ) {
      direction[static_cast<int>( column - local_column_begin )] = 1.0;
    }
    addConstraintCoordinateJacobianMult( state, direction, result );
    for ( int row = 0; row < result.Size(); ++row ) {
      active[static_cast<std::size_t>( row )] =
          active[static_cast<std::size_t>( row )] || std::abs( result[row] ) > 1.0e-28;
    }
  }
  return active;
}

std::unique_ptr<::mfem::HypreParMatrix> inactiveIdentity( const ::mfem::ParFiniteElementSpace& space,
                                                          const std::vector<bool>& active ) const
{
  const int local_size = space.GetTrueVSize();
  const HYPRE_BigInt global_size = space.GlobalTrueVSize();
  const HYPRE_BigInt local_begin = space.GetMyTDofOffset();
  std::vector<int> row_offsets( static_cast<std::size_t>( local_size ) + 1 );
  std::vector<HYPRE_BigInt> columns;
  std::vector<::mfem::real_t> values;
  for ( int row = 0; row < local_size; ++row ) {
    if ( !active[static_cast<std::size_t>( row )] ) {
      columns.push_back( local_begin + row );
      values.push_back( 1.0 );
    }
    row_offsets[static_cast<std::size_t>( row + 1 )] = static_cast<int>( columns.size() );
  }
  return std::make_unique<::mfem::HypreParMatrix>( mesh_.GetComm(), local_size, global_size, global_size,
                                                   row_offsets.data(), columns.data(), values.data(),
                                                   space.GetTrueDofOffsets(), space.GetTrueDofOffsets() );
}

void gatherSurfaceField( const detail::SurfaceStorage& surface, const ::mfem::ParGridFunction& field, int components,
                         std::vector<Real>& values )
{
  std::vector<Real> local( surface.localNodeCount() * static_cast<std::size_t>( components ) );
  const auto& space = *field.ParFESpace();
  for ( std::size_t local_node = 0; local_node < surface.localNodeCount(); ++local_node ) {
    const auto& restriction = surface.localRestrictionByIndex( local_node );
    for ( int component = 0; component < components; ++component ) {
      Real value{};
      for ( std::size_t term = 0; term < restriction.scalar_dofs.size(); ++term ) {
        const int scalar_dof = restriction.scalar_dofs[term];
        const int vector_dof = space.DofToVDof( detail::decodeDof( scalar_dof ), component );
        value += restriction.weights[term] * detail::dofSign( scalar_dof ) * detail::dofSign( vector_dof ) *
                 field[detail::decodeDof( vector_dof )];
      }
      local[local_node * static_cast<std::size_t>( components ) + component] = value;
    }
  }
  values = surface.distributeNodeValues( local, components );
}

void gatherSurfaceField( const detail::SurfaceStorage& surface, std::vector<Real>& values )
{
  const int dimension = coordinate_space_.GetVDim();
  std::vector<Real> local( surface.localNodeCount() * static_cast<std::size_t>( dimension ) );
  for ( std::size_t local_node = 0; local_node < surface.localNodeCount(); ++local_node ) {
    const auto& restriction = surface.localRestrictionByIndex( local_node );
    for ( int component = 0; component < dimension; ++component ) {
      Real value{};
      for ( std::size_t term = 0; term < restriction.scalar_dofs.size(); ++term ) {
        const int scalar_dof = restriction.scalar_dofs[term];
        const int vector_dof = coordinate_space_.DofToVDof( detail::decodeDof( scalar_dof ), component );
        value += restriction.weights[term] * detail::dofSign( scalar_dof ) * detail::dofSign( vector_dof ) *
                 local_parent_residual_[detail::decodeDof( vector_dof )];
      }
      local[local_node * static_cast<std::size_t>( dimension ) + component] = value;
    }
  }
  values = surface.distributeNodeValues( local, dimension );
}

void scatterSurface( const detail::SurfaceStorage& surface, ArrayView<const Real> residual )
{
  const int dimension = coordinate_space_.GetVDim();
  const auto local_residual = surface.reduceNodeValues( residual, dimension );
  for ( std::size_t local_node = 0; local_node < surface.localNodeCount(); ++local_node ) {
    const auto& restriction = surface.localRestrictionByIndex( local_node );
    for ( int component = 0; component < dimension; ++component ) {
      for ( std::size_t term = 0; term < restriction.scalar_dofs.size(); ++term ) {
        const int scalar_dof = restriction.scalar_dofs[term];
        const int vector_dof = coordinate_space_.DofToVDof( detail::decodeDof( scalar_dof ), component );
        local_parent_residual_[detail::decodeDof( vector_dof )] +=
            restriction.weights[term] * detail::dofSign( scalar_dof ) * detail::dofSign( vector_dof ) *
            local_residual[local_node * static_cast<std::size_t>( dimension ) + component];
      }
    }
  }
}
