void requireDirection( ContactDirectionView direction ) const
{
  if ( !direction.mortar.isStructurallyValid() || !direction.nonmortar.isStructurallyValid() ||
       direction.mortar.entities != surfaces_.mortar.numberOfNodes() ||
       direction.nonmortar.entities != surfaces_.nonmortar.numberOfNodes() ||
       direction.mortar.components != surfaces_.mortar.dimension ||
       direction.nonmortar.components != surfaces_.nonmortar.dimension ) {
    throw std::invalid_argument( "Coordinate direction must match the two surface coordinate fields." );
  }
}

static void seedCoordinates( const SurfaceMeshView& mesh, const FieldView<const Real>& direction,
                             std::vector<linearization_detail::ExactTangent>& coordinates )
{
  const bool has_direction = !direction.values.empty();
  if ( has_direction && ( !direction.isStructurallyValid() || direction.entities != mesh.numberOfNodes() ||
                          direction.components != mesh.dimension ) ) {
    throw std::invalid_argument( "Coordinate direction must match its surface coordinate field." );
  }
  for ( Index node = 0; node < mesh.numberOfNodes(); ++node ) {
    for ( int component = 0; component < mesh.dimension; ++component ) {
      coordinates[static_cast<std::size_t>( node * mesh.dimension + component )] = {
          mesh.coordinates( node, component ), has_direction ? direction( node, component ) : 0.0 };
    }
  }
}

static FieldView<const linearization_detail::ExactTangent> seedField(
    const FieldView<const Real>& values, const FieldView<const Real>& direction,
    std::vector<linearization_detail::ExactTangent>& storage, const char* message )
{
  if ( values.values.empty() ) {
    if ( !direction.values.empty() ) {
      throw std::invalid_argument( message );
    }
    return {};
  }
  if ( !values.isStructurallyValid() || static_cast<Index>( storage.size() ) != values.values.size() ) {
    throw std::invalid_argument( message );
  }
  const bool has_direction = !direction.values.empty();
  if ( has_direction && ( !direction.isStructurallyValid() || direction.entities != values.entities ||
                          direction.components != values.components ) ) {
    throw std::invalid_argument( message );
  }
  for ( Index entity = 0; entity < values.entities; ++entity ) {
    for ( int component = 0; component < values.components; ++component ) {
      const auto index = static_cast<std::size_t>( entity * values.components + component );
      storage[index] = { values( entity, component ), has_direction ? direction( entity, component ) : 0.0 };
    }
  }
  return { { storage.data(), static_cast<Index>( storage.size() ) },
           values.entities,
           values.components,
           FieldLayout::Interleaved };
}

static ArrayView<const linearization_detail::ExactTangent> seedArray(
    ArrayView<const Real> values, ArrayView<const Real> direction,
    std::vector<linearization_detail::ExactTangent>& storage, const char* message )
{
  if ( values.empty() ) {
    if ( !direction.empty() ) {
      throw std::invalid_argument( message );
    }
    return {};
  }
  if ( !values.isStructurallyValid() || static_cast<Index>( storage.size() ) != values.size() ||
       ( !direction.empty() && ( !direction.isStructurallyValid() || direction.size() != values.size() ) ) ) {
    throw std::invalid_argument( message );
  }
  for ( Index entry = 0; entry < values.size(); ++entry ) {
    storage[static_cast<std::size_t>( entry )] = { values[entry], direction.empty() ? 0.0 : direction[entry] };
  }
  return { storage.data(), static_cast<Index>( storage.size() ) };
}

ContactStateViewT<linearization_detail::ExactTangent> seededState( const ContactStateView& state,
                                                                   const ContactStateDirectionView& direction ) const
{
  return {
      .mortar_velocity = seedField( state.mortar_velocity, direction.mortar_velocity, exact_mortar_velocity_,
                                    "Mortar velocity direction must match the state field." ),
      .nonmortar_velocity =
          seedField( state.nonmortar_velocity, direction.nonmortar_velocity, exact_nonmortar_velocity_,
                     "Nonmortar velocity direction must match the state field." ),
      .mortar_reference_coordinates = seedField(
          state.mortar_reference_coordinates, direction.mortar_reference_coordinates,
          exact_mortar_reference_coordinates_, "Mortar reference-coordinate direction must match the state field." ),
      .nonmortar_reference_coordinates =
          seedField( state.nonmortar_reference_coordinates, direction.nonmortar_reference_coordinates,
                     exact_nonmortar_reference_coordinates_,
                     "Nonmortar reference-coordinate direction must match the state field." ),
      .mortar_element_thickness =
          seedArray( state.mortar_element_thickness, direction.mortar_element_thickness,
                     exact_mortar_element_thickness_, "Mortar thickness direction must match the state field." ),
      .nonmortar_element_thickness =
          seedArray( state.nonmortar_element_thickness, direction.nonmortar_element_thickness,
                     exact_nonmortar_element_thickness_, "Nonmortar thickness direction must match the state field." ),
      .mortar_material_modulus =
          seedArray( state.mortar_material_modulus, direction.mortar_material_modulus, exact_mortar_material_modulus_,
                     "Mortar modulus direction must match the state field." ),
      .nonmortar_material_modulus =
          seedArray( state.nonmortar_material_modulus, direction.nonmortar_material_modulus,
                     exact_nonmortar_material_modulus_, "Nonmortar modulus direction must match the state field." ),
      .multiplier = seedArray( state.multiplier, direction.multiplier, exact_multiplier_,
                               "Multiplier direction must match the state field." ),
      .external_potential_density =
          seedArray( state.external_potential_density, direction.external_potential_density,
                     exact_external_potential_density_, "External-potential direction must match the state field." ),
      .external_pressure = seedArray( state.external_pressure, direction.external_pressure, exact_external_pressure_,
                                      "External-pressure direction must match the state field." ),
      .external_pressure_tangent = seedArray( state.external_pressure_tangent, direction.external_pressure_tangent,
                                              exact_external_pressure_tangent_,
                                              "External-pressure-tangent direction must match the state field." ),
  };
}

static SurfaceMeshViewT<linearization_detail::ExactTangent> withSeededCoordinates(
    const SurfaceMeshView& mesh, const std::vector<linearization_detail::ExactTangent>& coordinates )
{
  return {
      .dimension = mesh.dimension,
      .coordinates = { ArrayView<const linearization_detail::ExactTangent>{ coordinates.data(),
                                                                            static_cast<Index>( coordinates.size() ) },
                       mesh.numberOfNodes(), mesh.dimension, FieldLayout::Interleaved },
      .element_offsets = mesh.element_offsets,
      .connectivity = mesh.connectivity,
      .topologies = mesh.topologies,
      .attributes = mesh.attributes,
  };
}

SurfacePairViewT<linearization_detail::ExactTangent> seededSurfaces() const
{
  return { withSeededCoordinates( surfaces_.mortar, exact_mortar_coordinates_ ),
           withSeededCoordinates( surfaces_.nonmortar, exact_nonmortar_coordinates_ ) };
}

template <typename Scalar>
ContactResidualViewT<Scalar> workspaceResidual( std::vector<Scalar>& mortar, std::vector<Scalar>& nonmortar ) const
{
  return workspaceResidual( mortar, nonmortar, {} );
}

template <typename Scalar>
ContactResidualViewT<Scalar> workspaceResidual( std::vector<Scalar>& mortar, std::vector<Scalar>& nonmortar,
                                                ArrayView<Scalar> constraint ) const
{
  return {
      .mortar =
          FieldView<Scalar>{ ArrayView<Scalar>{ mortar.data(), static_cast<Index>( mortar.size() ) },
                             surfaces_.mortar.numberOfNodes(), surfaces_.mortar.dimension, FieldLayout::Interleaved },
      .nonmortar = FieldView<Scalar>{ ArrayView<Scalar>{ nonmortar.data(), static_cast<Index>( nonmortar.size() ) },
                                      surfaces_.nonmortar.numberOfNodes(), surfaces_.nonmortar.dimension,
                                      FieldLayout::Interleaved },
      .constraint = constraint,
  };
}

template <typename Scalar>
static ArrayView<Scalar> mutableView( std::vector<Scalar>& values )
{
  return { values.data(), static_cast<Index>( values.size() ) };
}

static ArrayView<const Real> constView( const std::vector<Real>& values )
{
  return { values.data(), static_cast<Index>( values.size() ) };
}

static FieldView<const Real> constField( const std::vector<Real>& values, const SurfaceMeshView& mesh )
{
  return { constView( values ), mesh.numberOfNodes(), mesh.dimension, FieldLayout::Interleaved };
}

static void addTangents( const std::vector<linearization_detail::ExactTangent>& values, FieldView<Real> output )
{
  if ( !output.isStructurallyValid() || output.values.size() != static_cast<Index>( values.size() ) ) {
    throw std::invalid_argument( "Coordinate derivative output must match its surface coordinate field." );
  }
  for ( Index entity = 0; entity < output.entities; ++entity ) {
    for ( int component = 0; component < output.components; ++component ) {
      const auto workspace_index = static_cast<std::size_t>( entity * output.components + component );
      output( entity, component ) += linearization_detail::tangent( values[workspace_index] );
    }
  }
}

static void addTangents( const std::vector<linearization_detail::ExactTangent>& values, ArrayView<Real> output )
{
  if ( output.empty() ) {
    return;
  }
  if ( !output.isStructurallyValid() || output.size() != static_cast<Index>( values.size() ) ) {
    throw std::invalid_argument( "Constraint derivative output must match the mortar constraint field." );
  }
  for ( Index entry = 0; entry < output.size(); ++entry ) {
    output[entry] += linearization_detail::tangent( values[static_cast<std::size_t>( entry )] );
  }
}

void clearDerivativeWorkspace() const
{
  std::fill( derivative_mortar_.begin(), derivative_mortar_.end(), 0.0 );
  std::fill( derivative_nonmortar_.begin(), derivative_nonmortar_.end(), 0.0 );
  std::fill( derivative_constraint_.begin(), derivative_constraint_.end(), 0.0 );
}

void writeJacobianColumn( std::vector<Real>& matrix, Index rows, Index columns, Index column ) const
{
  const Index mortar_values = surfaces_.mortar.numberOfNodes() * surfaces_.mortar.dimension;
  const Index nonmortar_values = surfaces_.nonmortar.numberOfNodes() * surfaces_.nonmortar.dimension;
  for ( Index row = 0; row < mortar_values; ++row ) {
    matrix[static_cast<std::size_t>( row * columns + column )] = derivative_mortar_[static_cast<std::size_t>( row )];
  }
  for ( Index row = 0; row < nonmortar_values; ++row ) {
    matrix[static_cast<std::size_t>( ( mortar_values + row ) * columns + column )] =
        derivative_nonmortar_[static_cast<std::size_t>( row )];
  }
  for ( Index row = mortar_values + nonmortar_values; row < rows; ++row ) {
    matrix[static_cast<std::size_t>( row * columns + column )] =
        derivative_constraint_[static_cast<std::size_t>( row - mortar_values - nonmortar_values )];
  }
}

CsrMatrixView compress( DenseMatrixView matrix, Real drop_tolerance ) const
{
  if ( drop_tolerance < 0.0 ) {
    throw std::invalid_argument( "CSR drop tolerance cannot be negative." );
  }
  Index nonzeros{};
  csr_row_offsets_[0] = 0;
  for ( Index row = 0; row < matrix.rows; ++row ) {
    for ( Index column = 0; column < matrix.columns; ++column ) {
      const Real value = matrix( row, column );
      if ( std::abs( value ) > drop_tolerance ) {
        csr_column_indices_[static_cast<std::size_t>( nonzeros )] = column;
        csr_values_[static_cast<std::size_t>( nonzeros )] = value;
        ++nonzeros;
      }
    }
    csr_row_offsets_[static_cast<std::size_t>( row + 1 )] = nonzeros;
  }
  return {
      .row_offsets = { csr_row_offsets_.data(), matrix.rows + 1 },
      .column_indices = { csr_column_indices_.data(), nonzeros },
      .values = { csr_values_.data(), nonzeros },
      .rows = matrix.rows,
      .columns = matrix.columns,
  };
}
