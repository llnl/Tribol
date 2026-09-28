struct DeviceConstraintRowStorage {
  int dimension{};
  Index cells{};
  Index rows{};
  bool has_normal_rates{};
  DeviceBuffer<Index> cell_row_offsets;
  DeviceBuffer<std::uint64_t> contact_pair_ids;
  DeviceBuffer<std::uint64_t> cell_ids;
  DeviceBuffer<std::uint64_t> row_ids;
  DeviceBuffer<int> owner_ranks;
  DeviceBuffer<Index> overlap_cells;
  DeviceBuffer<Index> mortar_parent_faces;
  DeviceBuffer<Index> nonmortar_parent_faces;
  DeviceBuffer<Real> normals;
  DeviceBuffer<Real> integration_weights;
  DeviceBuffer<Real> mortar_parent_reference_coordinates;
  DeviceBuffer<Real> nonmortar_parent_reference_coordinates;
  DeviceBuffer<Real> mortar_positions;
  DeviceBuffer<Real> nonmortar_positions;
  DeviceBuffer<Real> gaps;
  DeviceBuffer<Real> normal_rates;
  DeviceBuffer<Index> mortar_dof_offsets;
  DeviceBuffer<Index> mortar_dofs;
  DeviceBuffer<Real> mortar_basis_values;
  DeviceBuffer<Index> nonmortar_dof_offsets;
  DeviceBuffer<Index> nonmortar_dofs;
  DeviceBuffer<Real> nonmortar_basis_values;

  void resize( int row_dimension, Index cell_count, Index row_count, Index mortar_entries, Index nonmortar_entries,
               bool include_normal_rates )
  {
    dimension = row_dimension;
    cells = cell_count;
    rows = row_count;
    has_normal_rates = include_normal_rates;
    cell_row_offsets.resize( static_cast<std::size_t>( cells + 1 ) );
    contact_pair_ids.resize( static_cast<std::size_t>( rows ) );
    cell_ids.resize( static_cast<std::size_t>( rows ) );
    row_ids.resize( static_cast<std::size_t>( rows ) );
    owner_ranks.resize( static_cast<std::size_t>( rows ) );
    overlap_cells.resize( static_cast<std::size_t>( rows ) );
    mortar_parent_faces.resize( static_cast<std::size_t>( rows ) );
    nonmortar_parent_faces.resize( static_cast<std::size_t>( rows ) );
    normals.resize( static_cast<std::size_t>( 3 * rows ) );
    integration_weights.resize( static_cast<std::size_t>( rows ) );
    mortar_parent_reference_coordinates.resize( static_cast<std::size_t>( ( dimension - 1 ) * rows ) );
    nonmortar_parent_reference_coordinates.resize( static_cast<std::size_t>( ( dimension - 1 ) * rows ) );
    mortar_positions.resize( static_cast<std::size_t>( 3 * rows ) );
    nonmortar_positions.resize( static_cast<std::size_t>( 3 * rows ) );
    gaps.resize( static_cast<std::size_t>( rows ) );
    normal_rates.resize( include_normal_rates ? static_cast<std::size_t>( rows ) : 0 );
    mortar_dof_offsets.resize( static_cast<std::size_t>( rows + 1 ) );
    mortar_dofs.resize( static_cast<std::size_t>( mortar_entries ) );
    mortar_basis_values.resize( static_cast<std::size_t>( mortar_entries ) );
    nonmortar_dof_offsets.resize( static_cast<std::size_t>( rows + 1 ) );
    nonmortar_dofs.resize( static_cast<std::size_t>( nonmortar_entries ) );
    nonmortar_basis_values.resize( static_cast<std::size_t>( nonmortar_entries ) );
  }

  void upload( ConstraintRowBatchView source )
  {
    dimension = source.dimension;
    cells = source.numberOfCells();
    rows = source.numberOfRows();
    has_normal_rates = source.hasNormalRates();
    copyToDevice( cell_row_offsets, source.cell_row_offsets, "copy row cell offsets to device" );
    copyToDevice( contact_pair_ids, source.contact_pair_ids, "copy row pair ids to device" );
    copyToDevice( cell_ids, source.cell_ids, "copy row cell ids to device" );
    copyToDevice( row_ids, source.row_ids, "copy row ids to device" );
    copyToDevice( owner_ranks, source.owner_ranks, "copy row owners to device" );
    copyToDevice( overlap_cells, source.overlap_cells, "copy row overlap indices to device" );
    copyToDevice( mortar_parent_faces, source.mortar_parent_faces, "copy mortar parent faces to device" );
    copyToDevice( nonmortar_parent_faces, source.nonmortar_parent_faces, "copy nonmortar parent faces to device" );
    copyToDevice( normals, source.normals.values, "copy row normals to device" );
    copyToDevice( integration_weights, source.integration_weights, "copy row weights to device" );
    copyToDevice( mortar_parent_reference_coordinates, source.mortar_parent_reference_coordinates.values,
                  "copy mortar parent reference coordinates to device" );
    copyToDevice( nonmortar_parent_reference_coordinates, source.nonmortar_parent_reference_coordinates.values,
                  "copy nonmortar parent reference coordinates to device" );
    copyToDevice( mortar_positions, source.mortar_positions.values, "copy mortar row positions to device" );
    copyToDevice( nonmortar_positions, source.nonmortar_positions.values, "copy nonmortar row positions to device" );
    copyToDevice( gaps, source.gaps, "copy row gaps to device" );
    copyToDevice( normal_rates, source.normal_rates, "copy row normal rates to device" );
    copyToDevice( mortar_dof_offsets, source.mortar_dof_offsets, "copy mortar row DOF offsets to device" );
    copyToDevice( mortar_dofs, source.mortar_dofs, "copy mortar row DOFs to device" );
    copyToDevice( mortar_basis_values, source.mortar_basis_values, "copy mortar row basis values to device" );
    copyToDevice( nonmortar_dof_offsets, source.nonmortar_dof_offsets, "copy nonmortar row DOF offsets to device" );
    copyToDevice( nonmortar_dofs, source.nonmortar_dofs, "copy nonmortar row DOFs to device" );
    copyToDevice( nonmortar_basis_values, source.nonmortar_basis_values, "copy nonmortar row basis values to device" );
  }

  [[nodiscard]] ConstraintRowBatchView view() const
  {
    return {
        .dimension = dimension,
        .cell_row_offsets = constDeviceView( cell_row_offsets ),
        .contact_pair_ids = constDeviceView( contact_pair_ids ),
        .cell_ids = constDeviceView( cell_ids ),
        .row_ids = constDeviceView( row_ids ),
        .owner_ranks = constDeviceView( owner_ranks ),
        .overlap_cells = constDeviceView( overlap_cells ),
        .mortar_parent_faces = constDeviceView( mortar_parent_faces ),
        .nonmortar_parent_faces = constDeviceView( nonmortar_parent_faces ),
        .normals = { constDeviceView( normals ), rows, 3, FieldLayout::Interleaved },
        .integration_weights = constDeviceView( integration_weights ),
        .mortar_parent_reference_coordinates = { constDeviceView( mortar_parent_reference_coordinates ), rows,
                                                 dimension - 1, FieldLayout::Interleaved },
        .nonmortar_parent_reference_coordinates = { constDeviceView( nonmortar_parent_reference_coordinates ), rows,
                                                    dimension - 1, FieldLayout::Interleaved },
        .mortar_positions = { constDeviceView( mortar_positions ), rows, 3, FieldLayout::Interleaved },
        .nonmortar_positions = { constDeviceView( nonmortar_positions ), rows, 3, FieldLayout::Interleaved },
        .gaps = constDeviceView( gaps ),
        .normal_rates = has_normal_rates ? constDeviceView( normal_rates ) : ArrayView<const Real>{},
        .mortar_dof_offsets = constDeviceView( mortar_dof_offsets ),
        .mortar_dofs = constDeviceView( mortar_dofs ),
        .mortar_basis_values = constDeviceView( mortar_basis_values ),
        .nonmortar_dof_offsets = constDeviceView( nonmortar_dof_offsets ),
        .nonmortar_dofs = constDeviceView( nonmortar_dofs ),
        .nonmortar_basis_values = constDeviceView( nonmortar_basis_values ),
    };
  }
};

struct DeviceQuadratureSampleStorage {
  int dimension{};
  Index cells{};
  Index samples{};
  bool has_normal_rates{};
  DeviceBuffer<Index> cell_sample_offsets;
  DeviceBuffer<std::uint64_t> contact_pair_ids;
  DeviceBuffer<std::uint64_t> cell_ids;
  DeviceBuffer<int> owner_ranks;
  DeviceBuffer<Index> overlap_cells;
  DeviceBuffer<Real> normals;
  DeviceBuffer<Real> integration_weights;
  DeviceBuffer<Real> mortar_parent_reference_coordinates;
  DeviceBuffer<Real> nonmortar_parent_reference_coordinates;
  DeviceBuffer<Real> physical_gaps;
  DeviceBuffer<Real> normal_rates;

  void reserve( int sample_dimension, Index cell_count, Index sample_capacity )
  {
    dimension = sample_dimension;
    cells = cell_count;
    cell_sample_offsets.resize( static_cast<std::size_t>( cells + 1 ) );
    contact_pair_ids.resize( static_cast<std::size_t>( sample_capacity ) );
    cell_ids.resize( static_cast<std::size_t>( sample_capacity ) );
    owner_ranks.resize( static_cast<std::size_t>( sample_capacity ) );
    overlap_cells.resize( static_cast<std::size_t>( sample_capacity ) );
    normals.resize( static_cast<std::size_t>( 3 * sample_capacity ) );
    integration_weights.resize( static_cast<std::size_t>( sample_capacity ) );
    mortar_parent_reference_coordinates.resize( static_cast<std::size_t>( ( dimension - 1 ) * sample_capacity ) );
    nonmortar_parent_reference_coordinates.resize( static_cast<std::size_t>( ( dimension - 1 ) * sample_capacity ) );
    physical_gaps.resize( static_cast<std::size_t>( sample_capacity ) );
    normal_rates.resize( static_cast<std::size_t>( sample_capacity ) );
    resize( 0, false );
  }

  void resize( Index sample_count, bool include_normal_rates )
  {
    samples = sample_count;
    has_normal_rates = include_normal_rates;
    contact_pair_ids.resize( static_cast<std::size_t>( samples ) );
    cell_ids.resize( static_cast<std::size_t>( samples ) );
    owner_ranks.resize( static_cast<std::size_t>( samples ) );
    overlap_cells.resize( static_cast<std::size_t>( samples ) );
    normals.resize( static_cast<std::size_t>( 3 * samples ) );
    integration_weights.resize( static_cast<std::size_t>( samples ) );
    mortar_parent_reference_coordinates.resize( static_cast<std::size_t>( ( dimension - 1 ) * samples ) );
    nonmortar_parent_reference_coordinates.resize( static_cast<std::size_t>( ( dimension - 1 ) * samples ) );
    physical_gaps.resize( static_cast<std::size_t>( samples ) );
    normal_rates.resize( include_normal_rates ? static_cast<std::size_t>( samples ) : 0 );
  }

  [[nodiscard]] diagnostics::QuadratureSampleView view() const
  {
    return {
        .dimension = dimension,
        .cell_sample_offsets = constDeviceView( cell_sample_offsets ),
        .contact_pair_ids = constDeviceView( contact_pair_ids ),
        .cell_ids = constDeviceView( cell_ids ),
        .owner_ranks = constDeviceView( owner_ranks ),
        .overlap_cells = constDeviceView( overlap_cells ),
        .normals = { constDeviceView( normals ), samples, 3, FieldLayout::Interleaved },
        .integration_weights = constDeviceView( integration_weights ),
        .mortar_parent_reference_coordinates = { constDeviceView( mortar_parent_reference_coordinates ), samples,
                                                 dimension - 1, FieldLayout::Interleaved },
        .nonmortar_parent_reference_coordinates = { constDeviceView( nonmortar_parent_reference_coordinates ), samples,
                                                    dimension - 1, FieldLayout::Interleaved },
        .physical_gaps = constDeviceView( physical_gaps ),
        .normal_rates = has_normal_rates ? constDeviceView( normal_rates ) : ArrayView<const Real>{},
    };
  }
};

struct DeviceParentTraceStorage {
  DeviceSurfaceStorage geometry;
  DeviceBuffer<std::uint64_t> parent_face_ids;
  DeviceBuffer<int> parent_face_owners;
  DeviceBuffer<ElementTopology> parent_face_topologies;
  DeviceBuffer<int> parent_face_orders;
  DeviceBuffer<Index> parent_dof_offsets;
  DeviceBuffer<Index> parent_dofs;
  DeviceBuffer<Real> parent_dof_signs;
  DeviceBuffer<Index> parent_basis_offsets;
  DeviceBuffer<Real> parent_basis_coefficients;
  DeviceBuffer<Index> subfacet_parent_faces;
  DeviceBuffer<Real> subfacet_parent_reference_coordinates;

  void upload( ParentTraceView source )
  {
    geometry.upload( source.geometry );
    copyToDevice( parent_face_ids, source.parent_face_ids, "copy parent face ids to device" );
    copyToDevice( parent_face_owners, source.parent_face_owners, "copy parent face owners to device" );
    copyToDevice( parent_face_topologies, source.parent_face_topologies, "copy parent face topologies to device" );
    copyToDevice( parent_face_orders, source.parent_face_orders, "copy parent face orders to device" );
    copyToDevice( parent_dof_offsets, source.parent_dof_offsets, "copy parent DOF offsets to device" );
    copyToDevice( parent_dofs, source.parent_dofs, "copy parent DOFs to device" );
    copyToDevice( parent_dof_signs, source.parent_dof_signs, "copy parent DOF signs to device" );
    copyToDevice( parent_basis_offsets, source.parent_basis_offsets, "copy parent basis offsets to device" );
    copyToDevice( parent_basis_coefficients, source.parent_basis_coefficients,
                  "copy parent basis coefficients to device" );
    copyToDevice( subfacet_parent_faces, source.subfacet_parent_faces, "copy subfacet parents to device" );
    copyToDevice( subfacet_parent_reference_coordinates, source.subfacet_parent_reference_coordinates.values,
                  "copy subfacet parent coordinates to device" );
  }

  [[nodiscard]] ParentTraceView view() const
  {
    const SurfaceMeshView surface = geometry.view();
    return {
        .geometry = surface,
        .parent_face_ids = constDeviceView( parent_face_ids ),
        .parent_face_owners = constDeviceView( parent_face_owners ),
        .parent_face_topologies = constDeviceView( parent_face_topologies ),
        .parent_face_orders = constDeviceView( parent_face_orders ),
        .parent_dof_offsets = constDeviceView( parent_dof_offsets ),
        .parent_dofs = constDeviceView( parent_dofs ),
        .parent_dof_signs = constDeviceView( parent_dof_signs ),
        .parent_basis_offsets = constDeviceView( parent_basis_offsets ),
        .parent_basis_coefficients = constDeviceView( parent_basis_coefficients ),
        .subfacet_parent_faces = constDeviceView( subfacet_parent_faces ),
        .subfacet_parent_reference_coordinates = {
            constDeviceView( subfacet_parent_reference_coordinates ), surface.connectivity.size(),
            surface.dimension - 1, FieldLayout::Interleaved },
    };
  }
};

struct DeviceFieldStorage {
  DeviceBuffer<Real> values;
  Index entities{};
  int components{};
  FieldLayout layout{ FieldLayout::Interleaved };

  void upload( FieldView<const Real> source, const char* operation )
  {
    entities = source.entities;
    components = source.components;
    layout = source.layout;
    copyToDevice( values, source.values, operation );
  }

  void initializeZero( Index field_entities, int field_components )
  {
    entities = field_entities;
    components = field_components;
    layout = FieldLayout::Interleaved;
    values.resize( static_cast<std::size_t>( entities * components ) );
    ActiveDeviceBackend::clear( values.data(), values.size() * sizeof( Real ) );
  }

  [[nodiscard]] FieldView<const Real> view() const
  {
    return { constDeviceView( values ), entities, components, layout };
  }
};

struct DeviceParentTracePairStorage {
  DeviceParentTraceStorage mortar;
  DeviceParentTraceStorage nonmortar;

  void upload( ParentTracePairView source )
  {
    mortar.upload( source.mortar );
    nonmortar.upload( source.nonmortar );
  }

  [[nodiscard]] ParentTracePairView view() const { return { mortar.view(), nonmortar.view() }; }
};

struct DeviceParentFieldPairStorage {
  DeviceFieldStorage mortar_coordinates;
  DeviceFieldStorage nonmortar_coordinates;
  DeviceFieldStorage mortar_velocity;
  DeviceFieldStorage nonmortar_velocity;

  void uploadCoordinates( ParentTraceFieldPairView source )
  {
    mortar_coordinates.upload( source.mortar_coordinates, "copy mortar parent coordinates to device" );
    nonmortar_coordinates.upload( source.nonmortar_coordinates, "copy nonmortar parent coordinates to device" );
  }

  void reserveVelocity()
  {
    mortar_velocity.initializeZero( mortar_coordinates.entities, mortar_coordinates.components );
    nonmortar_velocity.initializeZero( nonmortar_coordinates.entities, nonmortar_coordinates.components );
  }

  void uploadVelocity( ParentTraceFieldPairView source, bool required )
  {
    if ( source.hasVelocity() ) {
      mortar_velocity.upload( source.mortar_velocity, "copy mortar parent velocity to device" );
      nonmortar_velocity.upload( source.nonmortar_velocity, "copy nonmortar parent velocity to device" );
    } else if ( required ) {
      mortar_velocity.initializeZero( mortar_coordinates.entities, mortar_coordinates.components );
      nonmortar_velocity.initializeZero( nonmortar_coordinates.entities, nonmortar_coordinates.components );
    } else {
      mortar_velocity.values.clear();
      nonmortar_velocity.values.clear();
    }
  }

  [[nodiscard]] ParentTraceFieldPairView view() const
  {
    return { mortar_coordinates.view(), nonmortar_coordinates.view(), mortar_velocity.view(),
             nonmortar_velocity.view() };
  }
};
