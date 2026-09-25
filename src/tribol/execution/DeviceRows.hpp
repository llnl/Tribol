#ifndef TRIBOL_EXECUTION_DEVICEROWS_HPP_
#define TRIBOL_EXECUTION_DEVICEROWS_HPP_

#include "tribol/evaluation/ExplicitRowEvaluator.hpp"
#include "tribol/execution/Execution.hpp"
#include "tribol/timestep/ExplicitPenaltyStability.hpp"

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
};

struct DeviceRowResultView {
  ConstraintRowBatchView rows{};
  FieldView<const Real> mortar_force{};
  FieldView<const Real> nonmortar_force{};
  DevicePenaltyRowOutputView row_output{};
  DeviceRowEvaluationSummary summary{};
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
  [[nodiscard]] DeviceRowEvaluationSummary evaluate( PenaltyRowInputView penalty,
                                                     timestep::DiagonalInverseMassView inverse_mass,
                                                     timestep::IntegratorStabilityLimits integrator,
                                                     GeometryVersion geometry_version = {},
                                                     InteractionVersion interaction_version = {},
                                                     PenaltyVersion penalty_version = {} );
  [[nodiscard]] DeviceRowResultView deviceResult( DeviceRowEvaluationSummary summary ) const;
  void download( ArrayView<Real> force, PenaltyRowOutputView row_output ) const;

 private:
  Impl* impl_{};
};

using CudaRowWorkspace = DeviceRowWorkspace<Cuda>;
using HipRowWorkspace = DeviceRowWorkspace<Hip>;

static_assert( std::is_trivially_copyable_v<DeviceRowDofLayout> );
static_assert( std::is_trivially_copyable_v<DeviceRowEvaluationSummary> );
static_assert( std::is_trivially_copyable_v<DeviceRowResultView> );

}  // namespace tribol::execution

#endif
