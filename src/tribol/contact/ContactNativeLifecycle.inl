explicit Contact( ParentTracePairView traces, ParentTraceFieldPairView fields, Options options = {} )
  requires detail::is_native_explicit_method_v<MethodType>
    : surfaces_{ traces.mortar.geometry, traces.nonmortar.geometry },
      parent_traces_( traces ),
      parent_fields_( fields ),
      options_( effectiveOptions( std::move( options ) ) ),
      search_( effectiveSearchParameters( options_ ) )
{
  requireValidNativeInput( parent_traces_, parent_fields_ );
  validateMethodParameters<MethodType>( options_.method );
  validateMethodDomain<MethodType>( surfaces_ );
  timestep::validate( options_.timestep );
  resizeEvaluationWorkspace();
  if constexpr ( execution::DevicePolicy<Execution> ) {
    execution_workspace_.uploadSurfaces( surfaces_ );
  }
}

void updateGeometry( ParentTracePairView traces, ParentTraceFieldPairView fields )
  requires detail::is_native_explicit_method_v<MethodType>
{
  requireValidNativeInput( traces, fields );
  requireSameTopology( surfaces_, { traces.mortar.geometry, traces.nonmortar.geometry } );
  parent_traces_ = traces;
  parent_fields_ = fields;
  surfaces_ = { traces.mortar.geometry, traces.nonmortar.geometry };
  if constexpr ( execution::DevicePolicy<Execution> ) {
    execution_workspace_.updateGeometry( surfaces_ );
  }
  geometry_version_.advance();
  if ( has_interactions_ ) {
    prepareNativeRows();
  }
}

void rebuildGeometry( ParentTracePairView traces, ParentTraceFieldPairView fields )
  requires detail::is_native_explicit_method_v<MethodType>
{
  requireValidNativeInput( traces, fields );
  parent_traces_ = traces;
  parent_fields_ = fields;
  surfaces_ = { traces.mortar.geometry, traces.nonmortar.geometry };
  candidates_.clear();
  native_cells_.reset();
  native_rows_.reset();
  if constexpr ( execution::DevicePolicy<Execution> ) {
    execution_workspace_.invalidate();
    execution_workspace_.uploadSurfaces( surfaces_ );
  }
  geometry_version_.advance();
  has_interactions_ = false;
  resizeEvaluationWorkspace();
}

[[nodiscard]] OverlapCellBatchView overlapCells() const
  requires detail::is_native_explicit_method_v<MethodType>
{
  if constexpr ( execution::DevicePolicy<Execution> ) {
    return {};
  } else {
    return native_cells_ ? native_cells_->view() : OverlapCellBatchView{};
  }
}

[[nodiscard]] ConstraintRowBatchView constraintRows() const
  requires detail::is_native_explicit_method_v<MethodType>
{
  if constexpr ( execution::DevicePolicy<Execution> ) {
    return has_interactions_ ? native_device_workspace_.deviceResult( last_native_device_summary_ ).rows
                             : ConstraintRowBatchView{};
  } else {
    return native_rows_ ? native_rows_->view() : ConstraintRowBatchView{};
  }
}

[[nodiscard]] timestep::ExplicitStabilityContributionsView explicitStabilityContributions() const
  requires( detail::is_native_explicit_method_v<MethodType> && !execution::DevicePolicy<Execution> )
{
  if ( !native_stability_current_ ) {
    throw std::logic_error( "Explicit stability contributions require a current contact evaluation." );
  }
  return {
      .mortar_stiffness_row_sums = resultField( native_mortar_row_sums_, true ),
      .nonmortar_stiffness_row_sums = resultField( native_nonmortar_row_sums_, false ),
      .mortar_rate_row_sums = resultField( native_mortar_rate_row_sums_, true ),
      .nonmortar_rate_row_sums = resultField( native_nonmortar_rate_row_sums_, false ),
  };
}
