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
}

void updateGeometry( ParentTracePairView traces, ParentTraceFieldPairView fields )
  requires detail::is_native_explicit_method_v<MethodType>
{
  requireValidNativeInput( traces, fields );
  requireSameTopology( surfaces_, { traces.mortar.geometry, traces.nonmortar.geometry } );
  parent_traces_ = traces;
  parent_fields_ = fields;
  surfaces_ = { traces.mortar.geometry, traces.nonmortar.geometry };
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
  geometry_version_.advance();
  has_interactions_ = false;
  resizeEvaluationWorkspace();
}

[[nodiscard]] OverlapCellBatchView overlapCells() const
  requires detail::is_native_explicit_method_v<MethodType>
{
  return native_cells_ ? native_cells_->view() : OverlapCellBatchView{};
}

[[nodiscard]] ConstraintRowBatchView constraintRows() const
  requires detail::is_native_explicit_method_v<MethodType>
{
  return native_rows_ ? native_rows_->view() : ConstraintRowBatchView{};
}
