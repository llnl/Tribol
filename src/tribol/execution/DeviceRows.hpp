#ifndef TRIBOL_EXECUTION_DEVICEROWS_HPP_
#define TRIBOL_EXECUTION_DEVICEROWS_HPP_

#include "tribol/basis/ParentBasis.hpp"
#include "tribol/diagnostics/ProductionDiagnostics.hpp"
#include "tribol/evaluation/ExplicitRowEvaluator.hpp"
#include "tribol/execution/Execution.hpp"
#include "tribol/geom/OverlapCell.hpp"
#include "tribol/integration/AutomaticQuadrature.hpp"
#include "tribol/timestep/ExplicitPenaltyStability.hpp"
#include "tribol/timestep/KinematicVote.hpp"

#include <limits>
#include <type_traits>

namespace tribol::execution {

struct DeviceRowDofLayout {
  int dimension{};
  Index mortar_dofs{};
  Index nonmortar_dofs{};
  bool shared_parent_dofs{};

  [[nodiscard]] static constexpr DeviceRowDofLayout separate( int dimension, Index mortar_dofs, Index nonmortar_dofs )
  {
    return { dimension, mortar_dofs, nonmortar_dofs, false };
  }

  [[nodiscard]] static constexpr DeviceRowDofLayout coupled( int dimension, Index parent_dofs )
  {
    return { dimension, parent_dofs, parent_dofs, true };
  }

  [[nodiscard]] constexpr Index forceValues() const
  {
    return dimension * ( shared_parent_dofs ? mortar_dofs : mortar_dofs + nonmortar_dofs );
  }
};

using DevicePenaltyRowOutputView = PenaltyRowOutputViewT<const Real>;

struct DeviceRowEvaluationSummary {
  ExplicitRowSummary explicit_response{};
  timestep::ExplicitStabilityResult stability{};
  Real kinematic_timestep_vote{ std::numeric_limits<Real>::infinity() };
};

struct DeviceRowResultView {
  ConstraintRowBatchView rows{};
  FieldView<const Real> mortar_force{};
  FieldView<const Real> nonmortar_force{};
  DevicePenaltyRowOutputView row_output{};
  DeviceRowEvaluationSummary summary{};
};

struct DeviceNativeRowOptions {
  geometry::ProjectedOverlap<normal::MeanPlane>::Parameters geometry{};
  integration::Automatic::Parameters quadrature{};
  OverlapCellBuildOptions overlap{};
  timestep::Kinematic::Parameters kinematic{};
  std::uint64_t first_row_id{};
  bool include_normal_rates{};
};

struct DeviceNativeKinematicInputView {
  ArrayView<const Real> mortar_element_thickness{};
  ArrayView<const Real> nonmortar_element_thickness{};
};

template <typename Execution>
class DeviceRowWorkspace {
 public:
  class Impl;

  DeviceRowWorkspace();
  ~DeviceRowWorkspace();

  DeviceRowWorkspace( const DeviceRowWorkspace& ) = delete;
  DeviceRowWorkspace& operator=( const DeviceRowWorkspace& ) = delete;
  DeviceRowWorkspace( DeviceRowWorkspace&& ) = delete;
  DeviceRowWorkspace& operator=( DeviceRowWorkspace&& ) = delete;

  void prepare( ConstraintRowBatchView rows, DeviceRowDofLayout layout );
  void prepareNative( ParentTracePairView traces, ParentTraceFieldPairView fields,
                      ArrayView<const ElementPair> device_candidates, DeviceRowDofLayout layout,
                      DeviceNativeRowOptions options = {} );
  void updateNativeFields( ParentTraceFieldPairView fields );
  [[nodiscard]] DeviceRowEvaluationSummary evaluate( PenaltyRowInputView penalty,
                                                     timestep::DiagonalInverseMassView inverse_mass,
                                                     timestep::IntegratorStabilityLimits integrator,
                                                     GeometryVersion geometry_version = {},
                                                     InteractionVersion interaction_version = {},
                                                     PenaltyVersion penalty_version = {},
                                                     DeviceNativeKinematicInputView kinematic = {} );
  [[nodiscard]] DeviceRowResultView deviceResult( DeviceRowEvaluationSummary summary ) const;
  [[nodiscard]] ArrayView<const InteractionPatch> devicePatches() const;
  [[nodiscard]] diagnostics::RankProductionDiagnostics productionDiagnostics(
      ParentTraceFieldPairView fields, Real time_step, int local_rank = 0, int ranks = 1,
      diagnostics::QuadratureSamplingOptions sampling = {} );
  [[nodiscard]] diagnostics::QuadratureSampleView deviceDiagnosticQuadrature() const;
  void download( ArrayView<Real> force, PenaltyRowOutputView row_output ) const;

 private:
  Impl* impl_{};
};

using CudaRowWorkspace = DeviceRowWorkspace<Cuda>;
using HipRowWorkspace = DeviceRowWorkspace<Hip>;

static_assert( std::is_trivially_copyable_v<DeviceRowDofLayout> );
static_assert( std::is_trivially_copyable_v<DeviceNativeRowOptions> );
static_assert( std::is_trivially_copyable_v<DeviceNativeKinematicInputView> );
static_assert( std::is_trivially_copyable_v<DeviceRowEvaluationSummary> );
static_assert( std::is_trivially_copyable_v<DeviceRowResultView> );

}  // namespace tribol::execution

#endif
