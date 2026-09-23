void appendNode( const ::mfem::ParFiniteElementSpace& space, const ::mfem::ParGridFunction& coordinate_field,
                 int boundary_element, const ::mfem::IntegrationPoint& point, std::vector<std::int64_t>& metadata,
                 std::vector<Real>& coordinates )
{
  const auto* element = space.GetBE( boundary_element );
  ::mfem::Vector shape( element->GetDof() );
  ::mfem::Array<int> dofs;
  element->CalcShape( point, shape );
  space.GetBdrElementDofs( boundary_element, dofs );
  if ( shape.Size() != dofs.Size() ) {
    throw std::invalid_argument( "MFEM boundary element shape and degree-of-freedom counts do not match." );
  }
  RestrictionNode restriction;
  restriction.scalar_dofs.assign( dofs.begin(), dofs.end() );
  restriction.weights.assign( shape.GetData(), shape.GetData() + shape.Size() );
  metadata.push_back( static_cast<std::int64_t>( local_restrictions_.size() ) );
  local_restrictions_.push_back( std::move( restriction ) );
  for ( int component = 0; component < dimension_; ++component ) {
    Real value{};
    for ( int dof = 0; dof < shape.Size(); ++dof ) {
      const int scalar_dof = dofs[dof];
      const int vector_dof = space.DofToVDof( decodeDof( scalar_dof ), component );
      value += shape[dof] * dofSign( scalar_dof ) * dofSign( vector_dof ) * coordinate_field[decodeDof( vector_dof )];
    }
    coordinates.push_back( value );
  }
}

void appendSubelement( const ::mfem::ParFiniteElementSpace& space, const ::mfem::ParGridFunction& coordinate_field,
                       int boundary_element, int attribute, ElementTopology child_topology,
                       const ::mfem::IntegrationPoint* points, int point_count, std::vector<std::int64_t>& metadata,
                       std::vector<Real>& coordinates, std::vector<Real>& trace_data )
{
  const auto* element = space.GetBE( boundary_element );
  const auto parent_topology = topology( mesh_.GetBdrElementType( boundary_element ) );
  const int parent_order = element->GetOrder();
  ::mfem::Array<int> dofs;
  space.GetBdrElementDofs( boundary_element, dofs );
  const auto coefficients = parentBasisCoefficients( *element, parent_topology, parent_order );
  metadata.push_back( static_cast<std::int64_t>( child_topology ) );
  metadata.push_back( attribute );
  metadata.push_back( boundary_element );
  metadata.push_back( static_cast<std::int64_t>( parent_topology ) );
  metadata.push_back( parent_order );
  metadata.push_back( dofs.Size() );
  metadata.insert( metadata.end(), dofs.begin(), dofs.end() );
  metadata.push_back( point_count );
  trace_data.insert( trace_data.end(), coefficients.begin(), coefficients.end() );
  local_boundary_elements_.push_back( boundary_element );
  for ( int point = 0; point < point_count; ++point ) {
    appendNode( space, coordinate_field, boundary_element, points[point], metadata, coordinates );
    trace_data.push_back( points[point].x );
    if ( dimension_ == 3 ) {
      trace_data.push_back( points[point].y );
    }
  }
}

void appendRefinedBoundaryElement( const ::mfem::ParFiniteElementSpace& space,
                                   const ::mfem::ParGridFunction& coordinate_field, int boundary_element, int attribute,
                                   std::vector<std::int64_t>& metadata, std::vector<Real>& coordinates,
                                   std::vector<Real>& trace_data )
{
  const auto* finite_element = space.GetBE( boundary_element );
  const int order = subdivision_factor_ > 0 ? subdivision_factor_ : std::max( finite_element->GetOrder(), 1 );
  const auto element_topology = topology( mesh_.GetBdrElementType( boundary_element ) );
  if ( element_topology == ElementTopology::Segment ) {
    for ( int segment = 0; segment < order; ++segment ) {
      ::mfem::IntegrationPoint points[2];
      points[0].x = static_cast<Real>( segment ) / order;
      points[1].x = static_cast<Real>( segment + 1 ) / order;
      appendSubelement( space, coordinate_field, boundary_element, attribute, ElementTopology::Segment, points, 2,
                        metadata, coordinates, trace_data );
    }
    return;
  }
  if ( element_topology == ElementTopology::Quadrilateral ) {
    for ( int second = 0; second < order; ++second ) {
      for ( int first = 0; first < order; ++first ) {
        ::mfem::IntegrationPoint points[4];
        points[0].Set2( static_cast<Real>( first ) / order, static_cast<Real>( second ) / order );
        points[1].Set2( static_cast<Real>( first + 1 ) / order, static_cast<Real>( second ) / order );
        points[2].Set2( static_cast<Real>( first + 1 ) / order, static_cast<Real>( second + 1 ) / order );
        points[3].Set2( static_cast<Real>( first ) / order, static_cast<Real>( second + 1 ) / order );
        appendSubelement( space, coordinate_field, boundary_element, attribute, ElementTopology::Quadrilateral, points,
                          4, metadata, coordinates, trace_data );
      }
    }
    return;
  }
  for ( int first = 0; first < order; ++first ) {
    for ( int second = 0; second < order - first; ++second ) {
      ::mfem::IntegrationPoint lower[3];
      lower[0].Set2( static_cast<Real>( first ) / order, static_cast<Real>( second ) / order );
      lower[1].Set2( static_cast<Real>( first + 1 ) / order, static_cast<Real>( second ) / order );
      lower[2].Set2( static_cast<Real>( first ) / order, static_cast<Real>( second + 1 ) / order );
      appendSubelement( space, coordinate_field, boundary_element, attribute, ElementTopology::Triangle, lower, 3,
                        metadata, coordinates, trace_data );
      if ( first + second < order - 1 ) {
        ::mfem::IntegrationPoint upper[3];
        upper[0].Set2( static_cast<Real>( first + 1 ) / order, static_cast<Real>( second ) / order );
        upper[1].Set2( static_cast<Real>( first + 1 ) / order, static_cast<Real>( second + 1 ) / order );
        upper[2].Set2( static_cast<Real>( first ) / order, static_cast<Real>( second + 1 ) / order );
        appendSubelement( space, coordinate_field, boundary_element, attribute, ElementTopology::Triangle, upper, 3,
                          metadata, coordinates, trace_data );
      }
    }
  }
}
