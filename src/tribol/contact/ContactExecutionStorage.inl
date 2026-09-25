std::optional<OverlapCellBatch> native_cells_;
mutable std::optional<ConstraintRowBatch> native_rows_;
mutable std::vector<Real> native_pressure_stiffness_;
mutable std::vector<Real> native_rate_coefficient_;
mutable std::vector<Real> native_row_force_;
mutable std::vector<Real> native_row_stiffness_;
mutable std::vector<Real> native_row_rate_;
mutable std::vector<Real> native_mortar_row_sums_;
mutable std::vector<Real> native_nonmortar_row_sums_;
mutable std::vector<Real> native_device_force_;
using ExecutionWorkspace =
    std::conditional_t<execution::DevicePolicy<Execution>, execution::DeviceContactWorkspace<Execution>,
                       std::conditional_t<std::same_as<Execution, execution::OpenMP>, execution::OpenMPContactWorkspace,
                                          execution::HostPenaltyWorkspace>>;
mutable ExecutionWorkspace execution_workspace_;
using NativeDeviceWorkspace =
    std::conditional_t<execution::DevicePolicy<Execution>, execution::DeviceRowWorkspace<Execution>,
                       execution::HostPenaltyWorkspace>;
mutable NativeDeviceWorkspace native_device_workspace_;
mutable EvaluationSummary last_device_summary_{};
mutable execution::DeviceRowEvaluationSummary last_native_device_summary_{};
