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
