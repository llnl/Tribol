#include "tribol/execution/DeviceRows.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace tribol::execution {

namespace {

template <typename Body>
void launchRowKernel( Index count, Body&& body )
{
  if ( count <= 0 ) {
    return;
  }
  RAJA::forall<typename ActiveDeviceBackend::ForallPolicy>(
      ActiveDeviceBackend::resource(), RAJA::TypedRangeSegment<Index>( 0, count ), std::forward<Body>( body ) );
  ActiveDeviceBackend::checkLaunch( "launch device row kernel" );
}

#include "tribol/execution/DeviceNativeRowKernels.inl"
#include "tribol/execution/DeviceDiagnosticKernels.inl"

}  // namespace

template <>
class DeviceRowWorkspace<ActiveExecution>::Impl {
 public:
  void prepare( ConstraintRowBatchView rows, DeviceRowDofLayout layout )
  {
    validateRows( rows, layout );
    layout_ = layout;
    rows_.upload( rows );
    native_options_ = {};
    native_candidates_.clear();
    native_mortar_element_thickness_.clear();
    native_nonmortar_element_thickness_.clear();
    native_kinematic_votes_.clear();
    native_kinematic_vote_.clear();
    configureEvaluation();
    evaluated_ = false;
    diagnostics_current_ = false;
    native_prepared_ = false;
    prepared_ = true;
  }

#include "tribol/execution/DeviceRowPreparation.inl"
#include "tribol/execution/DeviceRowDiagnostics.inl"
  DeviceRowEvaluationSummary evaluate( PenaltyRowInputView penalty, timestep::DiagonalInverseMassView inverse_mass,
                                       timestep::IntegratorStabilityLimits integrator, GeometryVersion geometry_version,
                                       InteractionVersion interaction_version, PenaltyVersion penalty_version,
                                       DeviceNativeKinematicInputView kinematic )
  {
    requirePrepared();
    evaluated_ = false;
    diagnostics_current_ = false;
    validateInput( penalty, inverse_mass, integrator );
    uploadInput( penalty, inverse_mass );
    ActiveDeviceBackend::clear( force_.data(), force_.size() * sizeof( Real ) );

    const ConstraintRowBatchView rows = rows_.view();
    const PenaltyRowInputView device_penalty{ .pressure_stiffness = constDeviceView( pressure_stiffness_ ),
                                              .rate_coefficients = constDeviceView( rate_coefficients_ ),
                                              .activation = penalty.activation };
    const PenaltyRowOutputView row_output{ .effective_gap = deviceView( effective_gap_ ),
                                           .pressure = deviceView( pressure_ ),
                                           .row_force = deviceView( row_force_ ),
                                           .row_stiffness = deviceView( row_stiffness_ ),
                                           .row_rate_coefficient = deviceView( row_rate_ ) };
    ExplicitRowSummary* summary_entries = summary_entries_.data();
    std::uint64_t* scatter_keys = scatter_keys_.data();
    Real* scatter_values = scatter_values_.data();
    const DeviceRowDofLayout layout = layout_;
    launchRowKernel( rows.numberOfRows(), [=] RAJA_DEVICE( Index row ) {
      evaluateDeviceRow( row, rows, device_penalty, layout, row_output, summary_entries, scatter_keys, scatter_values );
    } );
    const Index force_entries = rows.dimension * ( rows.mortar_dofs.size() + rows.nonmortar_dofs.size() );
    deterministicScatter( force_entries, force_.data() );

    ExplicitRowSummary explicit_summary{};
    if ( rows.numberOfRows() > 0 ) {
      const std::size_t bytes =
          ActiveDeviceBackend::reduceTemporaryBytes( summary_entries_.data(), summary_.data(), rows.numberOfRows(),
                                                     ExplicitRowSummaryReduction{}, ExplicitRowSummary{} );
      ActiveDeviceBackend::reduce( temporary_storage_.data(), bytes, summary_entries_.data(), summary_.data(),
                                   rows.numberOfRows(), ExplicitRowSummaryReduction{}, ExplicitRowSummary{} );
      ActiveDeviceBackend::copy( &explicit_summary, summary_.data(), sizeof( ExplicitRowSummary ) );
    }

    ActiveDeviceBackend::clear( stability_row_sums_.data(), stability_row_sums_.size() * sizeof( Real ) );
    const Real* device_inverse_mass = inverse_mass_.data();
    const Real* device_stiffness = row_stiffness_.data();
    const Real* device_rate = row_rate_.data();
    const Index force_value_count = layout.forceValues();
    launchRowKernel( rows.numberOfRows(), [=] RAJA_DEVICE( Index row ) {
      emitDeviceStabilityEntries( row, rows, layout, device_inverse_mass, device_stiffness, device_rate,
                                  force_value_count, scatter_keys, scatter_values );
    } );
    deterministicScatter( 2 * force_entries, stability_row_sums_.data() );

    std::array<Real, 2> bounds{};
    if ( force_value_count > 0 ) {
      reduceMaximum( stability_row_sums_.data(), stability_bounds_.data(), force_value_count );
      reduceMaximum( stability_row_sums_.data() + force_value_count, stability_bounds_.data() + 1, force_value_count );
      ActiveDeviceBackend::copy( bounds.data(), stability_bounds_.data(), 2 * sizeof( Real ) );
    }
    timestep::ExplicitStabilityResult stability{
        .stiffness_frequency_squared_bound = bounds[0],
        .rate_frequency_bound = bounds[1],
        .timestep_vote = timestep::explicit_stability_detail::stabilityVote( bounds[0], bounds[1], integrator ),
        .active_stiffness_rows = explicit_summary.active_stiffness_rows,
        .active_rate_rows = explicit_summary.active_rate_rows,
        .geometry_version = geometry_version,
        .interaction_version = interaction_version,
        .mass_version = inverse_mass.version,
        .penalty_version = penalty_version,
        .integrator_version = integrator.version,
    };
    const Real kinematic_vote = evaluateNativeKinematicVote( device_penalty.activation, kinematic );
    last_summary_ = { explicit_summary, stability, kinematic_vote };
    evaluated_ = true;
    return last_summary_;
  }

  [[nodiscard]] Real evaluateNativeKinematicVote( constraint::GapActivationParameters activation,
                                                  DeviceNativeKinematicInputView input )
  {
    const auto& parameters = native_options_.kinematic;
    if ( !parameters.enabled ) {
      return std::numeric_limits<Real>::infinity();
    }
    if ( parameters.current_step < parameters.minimum_current_step ) {
      return parameters.current_step;
    }
    if ( !native_prepared_ || !input.mortar_element_thickness.isStructurallyValid() ||
         !input.nonmortar_element_thickness.isStructurallyValid() ||
         input.mortar_element_thickness.size() != native_traces_.mortar.geometry.elements ||
         input.nonmortar_element_thickness.size() != native_traces_.nonmortar.geometry.elements ) {
      throw std::invalid_argument( "Device native kinematic voting requires compatible subfacet thickness." );
    }
    for ( const Real thickness : input.mortar_element_thickness ) {
      if ( thickness < 0.0 ) {
        throw std::invalid_argument( "Device native kinematic voting requires nonnegative subfacet thickness." );
      }
    }
    for ( const Real thickness : input.nonmortar_element_thickness ) {
      if ( thickness < 0.0 ) {
        throw std::invalid_argument( "Device native kinematic voting requires nonnegative subfacet thickness." );
      }
    }
    ActiveDeviceBackend::copy( native_mortar_element_thickness_.data(), input.mortar_element_thickness.data(),
                               native_mortar_element_thickness_.size() * sizeof( Real ) );
    ActiveDeviceBackend::copy( native_nonmortar_element_thickness_.data(), input.nonmortar_element_thickness.data(),
                               native_nonmortar_element_thickness_.size() * sizeof( Real ) );
    const Index candidates = static_cast<Index>( native_candidates_.size() );
    if ( candidates == 0 ) {
      return parameters.current_step;
    }
    const ParentTracePairView traces = native_traces_.view();
    const ParentTraceFieldPairView fields = native_fields_.view();
    const ElementPair* candidate_data = native_candidates_.data();
    const InteractionPatch* patches = native_patches_.data();
    const Real* mortar_thickness = native_mortar_element_thickness_.data();
    const Real* nonmortar_thickness = native_nonmortar_element_thickness_.data();
    Real* votes = native_kinematic_votes_.data();
    launchRowKernel( candidates, [=] RAJA_DEVICE( Index candidate ) {
      evaluateDeviceNativeKinematicVote( candidate, traces, fields, candidate_data, candidates, patches,
                                         mortar_thickness, nonmortar_thickness, activation, parameters, votes );
    } );
    const std::size_t bytes = ActiveDeviceBackend::reduceTemporaryBytes(
        votes, native_kinematic_vote_.data(), candidates, MinimumReduction{}, parameters.current_step );
    ActiveDeviceBackend::reduce( temporary_storage_.data(), bytes, votes, native_kinematic_vote_.data(), candidates,
                                 MinimumReduction{}, parameters.current_step );
    Real vote{};
    ActiveDeviceBackend::copy( &vote, native_kinematic_vote_.data(), sizeof( Real ) );
    return vote;
  }

  [[nodiscard]] DeviceRowResultView deviceResult( DeviceRowEvaluationSummary summary ) const
  {
    requirePrepared();
    const Index nonmortar_offset = layout_.shared_parent_dofs ? 0 : layout_.mortar_dofs * layout_.dimension;
    return {
        .rows = rows_.view(),
        .mortar_force = { { force_.data(), layout_.mortar_dofs * layout_.dimension },
                          layout_.mortar_dofs,
                          layout_.dimension,
                          FieldLayout::Interleaved },
        .nonmortar_force = { { force_.data() + nonmortar_offset, layout_.nonmortar_dofs * layout_.dimension },
                             layout_.nonmortar_dofs,
                             layout_.dimension,
                             FieldLayout::Interleaved },
        .row_output = { .effective_gap = constDeviceView( effective_gap_ ),
                        .pressure = constDeviceView( pressure_ ),
                        .row_force = constDeviceView( row_force_ ),
                        .row_stiffness = constDeviceView( row_stiffness_ ),
                        .row_rate_coefficient = constDeviceView( row_rate_ ) },
        .summary = summary,
    };
  }

  void download( ArrayView<Real> force, PenaltyRowOutputView row_output ) const
  {
    requirePrepared();
    copyToHost( force, force_, "download device row force" );
    copyOptional( row_output.effective_gap, effective_gap_, "download effective row gaps" );
    copyOptional( row_output.pressure, pressure_, "download row pressures" );
    copyOptional( row_output.row_force, row_force_, "download integrated row forces" );
    copyOptional( row_output.row_stiffness, row_stiffness_, "download row stiffness" );
    copyOptional( row_output.row_rate_coefficient, row_rate_, "download row rate coefficients" );
  }

 public:
  template <typename Body>
  static void launch( Index count, Body&& body )
  {
    launchRowKernel( count, std::forward<Body>( body ) );
  }

  static void validateRows( ConstraintRowBatchView rows, DeviceRowDofLayout layout )
  {
    if ( !rows.isStructurallyValid() || rows.dimension != layout.dimension ||
         ( layout.dimension != 2 && layout.dimension != 3 ) || layout.mortar_dofs < 0 || layout.nonmortar_dofs < 0 ||
         ( layout.shared_parent_dofs && layout.mortar_dofs != layout.nonmortar_dofs ) ) {
      throw std::invalid_argument( "Invalid device row layout or constraint rows." );
    }
    for ( Index dof : rows.mortar_dofs ) {
      if ( dof < 0 || dof >= layout.mortar_dofs ) {
        throw std::out_of_range( "A mortar row references a DOF outside the device layout." );
      }
    }
    for ( Index dof : rows.nonmortar_dofs ) {
      if ( dof < 0 || dof >= layout.nonmortar_dofs ) {
        throw std::out_of_range( "A nonmortar row references a DOF outside the device layout." );
      }
    }
  }

  void validateInput( PenaltyRowInputView penalty, timestep::DiagonalInverseMassView inverse_mass,
                      timestep::IntegratorStabilityLimits integrator ) const
  {
    const Index rows = rows_.rows;
    if ( !penalty.pressure_stiffness.isStructurallyValid() || penalty.pressure_stiffness.size() != rows ||
         !penalty.rate_coefficients.isStructurallyValid() ||
         ( !penalty.rate_coefficients.empty() && penalty.rate_coefficients.size() != rows ) ||
         ( !penalty.rate_coefficients.empty() && !rows_.has_normal_rates ) ) {
      throw std::invalid_argument( "Device row evaluation requires compatible row penalty data." );
    }
    requireMassField( inverse_mass.mortar, layout_.mortar_dofs );
    requireMassField( inverse_mass.nonmortar, layout_.nonmortar_dofs );
    if ( layout_.shared_parent_dofs && inverse_mass.mortar.values.data() != inverse_mass.nonmortar.values.data() ) {
      throw std::invalid_argument( "Coupled device rows require one shared inverse-mass field." );
    }
    if ( !std::isfinite( integrator.maximum_undamped_frequency_step ) ||
         integrator.maximum_undamped_frequency_step <= 0.0 || !std::isfinite( integrator.maximum_decay_step ) ||
         integrator.maximum_decay_step <= 0.0 || !std::isfinite( integrator.safety_factor ) ||
         integrator.safety_factor <= 0.0 || integrator.safety_factor > 1.0 ) {
      throw std::invalid_argument( "Device row evaluation requires valid explicit-integrator stability limits." );
    }
    for ( Index row = 0; row < rows; ++row ) {
      if ( !std::isfinite( penalty.pressure_stiffness[row] ) || penalty.pressure_stiffness[row] < 0.0 ||
           ( !penalty.rate_coefficients.empty() &&
             ( !std::isfinite( penalty.rate_coefficients[row] ) || penalty.rate_coefficients[row] < 0.0 ) ) ) {
        throw std::invalid_argument( "Device row penalty data must be finite and nonnegative." );
      }
    }
  }

  void requireMassField( FieldView<const Real> mass, Index entities ) const
  {
    if ( !mass.isStructurallyValid() || mass.entities != entities || mass.components != layout_.dimension ) {
      throw std::invalid_argument( "Device row inverse mass does not match the prepared DOF layout." );
    }
    for ( Real value : mass.values ) {
      if ( !std::isfinite( value ) || value < 0.0 ) {
        throw std::invalid_argument( "Device row inverse mass must be finite and nonnegative." );
      }
    }
  }

  void uploadInput( PenaltyRowInputView penalty, timestep::DiagonalInverseMassView inverse_mass )
  {
    const std::size_t rows = static_cast<std::size_t>( rows_.rows );
    ActiveDeviceBackend::copy( pressure_stiffness_.data(), penalty.pressure_stiffness.data(), rows * sizeof( Real ) );
    ActiveDeviceBackend::clear( rate_coefficients_.data(), rows * sizeof( Real ) );
    if ( !penalty.rate_coefficients.empty() ) {
      ActiveDeviceBackend::copy( rate_coefficients_.data(), penalty.rate_coefficients.data(), rows * sizeof( Real ) );
    }
    const std::size_t mortar_values = static_cast<std::size_t>( layout_.mortar_dofs * layout_.dimension );
    ActiveDeviceBackend::copy( inverse_mass_.data(), inverse_mass.mortar.values.data(),
                               mortar_values * sizeof( Real ) );
    if ( !layout_.shared_parent_dofs ) {
      const std::size_t nonmortar_values = static_cast<std::size_t>( layout_.nonmortar_dofs * layout_.dimension );
      ActiveDeviceBackend::copy( inverse_mass_.data() + mortar_values, inverse_mass.nonmortar.values.data(),
                                 nonmortar_values * sizeof( Real ) );
    }
  }

  void sizeTemporaryStorage( Index scatter_entries, Index rows, Index force_values )
  {
    std::size_t bytes{};
    if ( scatter_entries > 0 ) {
      bytes = std::max( bytes, ActiveDeviceBackend::sortPairsTemporaryBytes(
                                   scatter_keys_.data(), scatter_keys_alt_.data(), scatter_values_.data(),
                                   scatter_values_alt_.data(), scatter_entries ) );
      bytes = std::max( bytes, ActiveDeviceBackend::runLengthEncodeTemporaryBytes(
                                   scatter_keys_alt_.data(), unique_keys_.data(), run_counts_.data(), run_count_.data(),
                                   scatter_entries ) );
      bytes = std::max( bytes, ActiveDeviceBackend::exclusiveSumTemporaryBytes( run_counts_.data(), run_offsets_.data(),
                                                                                scatter_entries ) );
    }
    if ( rows > 0 ) {
      bytes = std::max(
          bytes, ActiveDeviceBackend::reduceTemporaryBytes( summary_entries_.data(), summary_.data(), rows,
                                                            ExplicitRowSummaryReduction{}, ExplicitRowSummary{} ) );
    }
    if ( force_values > 0 ) {
      bytes = std::max( bytes,
                        ActiveDeviceBackend::reduceTemporaryBytes( stability_row_sums_.data(), stability_bounds_.data(),
                                                                   force_values, MaximumReduction{}, Real{} ) );
    }
    if ( native_kinematic_votes_.size() > 0 ) {
      bytes = std::max( bytes, ActiveDeviceBackend::reduceTemporaryBytes(
                                   native_kinematic_votes_.data(), native_kinematic_vote_.data(),
                                   static_cast<Index>( native_kinematic_votes_.size() ), MinimumReduction{},
                                   native_options_.kinematic.current_step ) );
    }
    temporary_storage_.resize( bytes );
  }

  void deterministicScatter( Index entries, Real* output )
  {
    if ( entries <= 0 ) {
      return;
    }
    std::size_t bytes = ActiveDeviceBackend::sortPairsTemporaryBytes(
        scatter_keys_.data(), scatter_keys_alt_.data(), scatter_values_.data(), scatter_values_alt_.data(), entries );
    ActiveDeviceBackend::sortPairs( temporary_storage_.data(), bytes, scatter_keys_.data(), scatter_keys_alt_.data(),
                                    scatter_values_.data(), scatter_values_alt_.data(), entries );
    bytes = ActiveDeviceBackend::runLengthEncodeTemporaryBytes( scatter_keys_alt_.data(), unique_keys_.data(),
                                                                run_counts_.data(), run_count_.data(), entries );
    ActiveDeviceBackend::clear( run_counts_.data(), run_counts_.size() * sizeof( Index ) );
    ActiveDeviceBackend::runLengthEncode( temporary_storage_.data(), bytes, scatter_keys_alt_.data(),
                                          unique_keys_.data(), run_counts_.data(), run_count_.data(), entries );
    bytes = ActiveDeviceBackend::exclusiveSumTemporaryBytes( run_counts_.data(), run_offsets_.data(), entries );
    ActiveDeviceBackend::exclusiveSum( temporary_storage_.data(), bytes, run_counts_.data(), run_offsets_.data(),
                                       entries );
    const std::uint64_t* unique_keys = unique_keys_.data();
    const Index* run_counts = run_counts_.data();
    const Index* run_offsets = run_offsets_.data();
    const Index* run_count = run_count_.data();
    const Real* sorted_values = scatter_values_alt_.data();
    launch( entries, [=] RAJA_DEVICE( Index run ) {
      reduceScatterRun( run, unique_keys, run_counts, run_offsets, run_count, sorted_values, output );
    } );
  }

  void reduceMaximum( const Real* input, Real* output, Index count )
  {
    const std::size_t bytes =
        ActiveDeviceBackend::reduceTemporaryBytes( input, output, count, MaximumReduction{}, Real{} );
    ActiveDeviceBackend::reduce( temporary_storage_.data(), bytes, input, output, count, MaximumReduction{}, Real{} );
  }

  template <typename T>
  static void copyOptional( ArrayView<T> destination, const DeviceBuffer<std::remove_const_t<T>>& source,
                            const char* operation )
  {
    if ( destination.empty() ) {
      return;
    }
    copyToHost( destination, source, operation );
  }

  void requirePrepared() const
  {
    if ( !prepared_ ) {
      throw std::logic_error( "Device row evaluation requires prepare() first." );
    }
  }

  DeviceConstraintRowStorage rows_;
  DeviceParentTracePairStorage native_traces_;
  DeviceParentFieldPairStorage native_fields_;
  DeviceNativeRowOptions native_options_{};
  DeviceBuffer<ElementPair> native_candidates_;
  DeviceBuffer<Real> native_mortar_element_thickness_;
  DeviceBuffer<Real> native_nonmortar_element_thickness_;
  DeviceBuffer<Real> native_kinematic_votes_;
  DeviceBuffer<Real> native_kinematic_vote_;
  DeviceRowDofLayout layout_{};
  bool prepared_{};
  bool native_prepared_{};
  DeviceBuffer<InteractionPatch> native_patches_;
  DeviceBuffer<Index> native_mortar_parent_faces_;
  DeviceBuffer<Index> native_nonmortar_parent_faces_;
  DeviceBuffer<int> native_owner_ranks_;
  DeviceBuffer<Real> native_mortar_patch_reference_;
  DeviceBuffer<Real> native_nonmortar_patch_reference_;
  DeviceBuffer<Index> native_row_counts_;
  DeviceBuffer<Index> native_row_offsets_;
  DeviceBuffer<Index> native_mortar_entry_counts_;
  DeviceBuffer<Index> native_mortar_entry_offsets_;
  DeviceBuffer<Index> native_nonmortar_entry_counts_;
  DeviceBuffer<Index> native_nonmortar_entry_offsets_;
  DeviceBuffer<Index> diagnostic_sample_counts_;
  DeviceQuadratureSampleStorage diagnostic_samples_;
  DeviceBuffer<DeviceRowDiagnosticPair> row_diagnostic_entries_;
  DeviceBuffer<DeviceRowDiagnosticPair> row_diagnostic_result_;
  DeviceBuffer<DeviceQuadratureDiagnosticPair> quadrature_diagnostic_entries_;
  DeviceBuffer<DeviceQuadratureDiagnosticPair> quadrature_diagnostic_result_;
  DeviceBuffer<Real> pressure_stiffness_;
  DeviceBuffer<Real> rate_coefficients_;
  DeviceBuffer<Real> inverse_mass_;
  DeviceBuffer<Real> force_;
  DeviceBuffer<Real> effective_gap_;
  DeviceBuffer<Real> pressure_;
  DeviceBuffer<Real> row_force_;
  DeviceBuffer<Real> row_stiffness_;
  DeviceBuffer<Real> row_rate_;
  DeviceBuffer<ExplicitRowSummary> summary_entries_;
  DeviceBuffer<ExplicitRowSummary> summary_;
  DeviceBuffer<Real> stability_row_sums_;
  DeviceBuffer<Real> stability_bounds_;
  DeviceBuffer<std::uint64_t> scatter_keys_;
  DeviceBuffer<std::uint64_t> scatter_keys_alt_;
  DeviceBuffer<Real> scatter_values_;
  DeviceBuffer<Real> scatter_values_alt_;
  DeviceBuffer<std::uint64_t> unique_keys_;
  DeviceBuffer<Index> run_counts_;
  DeviceBuffer<Index> run_offsets_;
  DeviceBuffer<Index> run_count_;
  DeviceBuffer<unsigned char> temporary_storage_;
  DeviceRowEvaluationSummary last_summary_{};
  bool evaluated_{};
  bool diagnostics_current_{};
};

#include "tribol/execution/DeviceRowWorkspaceApi.inl"

}  // namespace tribol::execution
