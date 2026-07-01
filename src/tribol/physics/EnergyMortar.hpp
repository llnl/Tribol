#pragma once
#include <vector>
#include <array>

#include "tribol/config.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

// EnergyMortar uses a 3-point Gauss-Legendre quad rule
struct QuadPoints {
  std::array<double, 3> qp;  // qp locations
  std::array<double, 3> w;   // weights
};

struct ContactParams {
  double del;                              // Smoothing Parameter
  double k;                                // Penalty
  int N;                                   // Quadrature Points
  bool enzyme_quadrature;                  // Determines how enzyming is performed (default = True)
  EnergyMortarNormalMode normal_mode{ EnergyMortarNormalMode::ELEMENT_NORMAL };  // Normal field used by EnergyMortar
  bool projection_smoothing{ true };       // Apply projection-bound smoothing
  double h1_active_set_smoothing_gap{ 0.0 };  // Active-set smoothing transition gap; disabled when <= 0
  EnergyMortarPenaltyMode penalty_mode{ EnergyMortarPenaltyMode::NODAL_GAP };  // Penalty enforcement mode
  EnergyMortarNodalEnergyBasis nodal_energy_basis{
      EnergyMortarNodalEnergyBasis::CUBIC_SPLINE };  // Basis used by NODAL_ENERGY mode
  bool nodal_energy_angle_smoothing{ true };          // Apply 80-to-90 degree angle smoothing in NODAL_ENERGY mode
  double residual_gap{ 0.0 };                         // User-defined gap offset
};

constexpr int h1_max_stencil_nodes_per_mesh = 16;
constexpr int h1_max_stencil_elems_per_mesh = 16;

// Weighted gap and trib area
struct NodalContactData {
  std::array<double, 2> AI;       // Trib area
  std::array<double, 2> g_tilde;  // Weighted gap
};

/// Stores quadrature-point penalty energy derivatives for one interface pair.
struct QuadraturePointPenaltyData {
  double energy{ 0.0 };
  std::array<double, 8> force{};
  std::array<double, 64> stiffness{};
  int num_mesh1_nodes{ 0 };
  int num_mesh2_nodes{ 0 };
  std::array<int, h1_max_stencil_nodes_per_mesh> mesh1_nodes{};
  std::array<int, h1_max_stencil_nodes_per_mesh> mesh2_nodes{};
  std::array<int, h1_max_stencil_nodes_per_mesh> mesh1_owner_elems{};
  std::array<int, h1_max_stencil_nodes_per_mesh> mesh2_owner_elems{};
  std::vector<double> h1_force;
  std::vector<double> h1_stiffness;
};

/// Stores finite-difference and analytical derivative data for validation tests.
struct FiniteDiffResult {
  /// Finite-difference approximation of the gradient/Hessian for the first
  /// nodal smoothed gap contribution.
  std::vector<double> fd_gradient_g1;

  /// Finite-difference approximation of the gradient/Hessian for the second
  /// nodal smoothed gap contribution.
  std::vector<double> fd_gradient_g2;

  /// Analytical/Enzyme-computed gradient/Hessian for the first nodal smoothed
  /// gap contribution.
  std::vector<double> analytical_gradient_g1;

  /// Analytical/Enzyme-computed gradient/Hessian for the second nodal smoothed
  /// gap contribution.
  std::vector<double> analytical_gradient_g2;

  /// Global node ids associated with the differentiated degrees of freedom.
  /// The expected ordering is the two nodes of edge A followed by the two nodes
  /// of edge B.
  std::vector<int> node_ids;

  /// Baseline value of the first nodal smoothed gap contribution before applying
  /// any finite-difference perturbations.
  double g_tilde1_baseline{ 0.0 };

  /// Baseline value of the second nodal smoothed gap contribution before applying
  /// any finite-difference perturbations.
  double g_tilde2_baseline{ 0.0 };
};

struct Gparams {
  std::array<double, 3> qp;
  std::array<double, 3> w;
};

/// Stores total derivative data for the H1 nodal-normal EnergyMortar path.
struct H1TotalDerivatives {
  int num_mesh1_nodes{ 0 };
  int num_mesh2_nodes{ 0 };
  std::array<int, h1_max_stencil_nodes_per_mesh> mesh1_nodes{};
  std::array<int, h1_max_stencil_nodes_per_mesh> mesh2_nodes{};
  std::array<int, h1_max_stencil_nodes_per_mesh> mesh1_owner_elems{};
  std::array<int, h1_max_stencil_nodes_per_mesh> mesh2_owner_elems{};
  std::array<double, 2> g_tilde{};
  std::array<double, 2> area{};
  std::vector<double> dg1_dx;
  std::vector<double> dg2_dx;
  std::vector<double> dA1_dx;
  std::vector<double> dA2_dx;
  std::vector<double> d2g1_dx2;
  std::vector<double> d2g2_dx2;
  std::vector<double> d2A1_dx2;
  std::vector<double> d2A2_dx2;
};

struct H1KernelData {
  int N{ 3 };
  double del{ 0.1 };
  double k{ 1.0 };
  double residual_gap{ 0.0 };
  double active_set_smoothing_gap{ 0.0 };
  bool projection_smoothing{ false };
  EnergyMortarNodalEnergyBasis nodal_energy_basis{ EnergyMortarNodalEnergyBasis::CUBIC_SPLINE };
  bool nodal_energy_angle_smoothing{ true };
  int num_nodes1{ 0 };
  int num_nodes2{ 0 };
  int num_elems1{ 0 };
  int num_elems2{ 0 };
  int contact_nodes1[2]{ 0, 1 };
  int contact_nodes2[2]{ 0, 1 };
  int elem_nodes1[h1_max_stencil_elems_per_mesh][2]{};
  int elem_nodes2[h1_max_stencil_elems_per_mesh][2]{};
  double xref1[2 * h1_max_stencil_nodes_per_mesh]{};
  double xref2[2 * h1_max_stencil_nodes_per_mesh]{};
};

/// Provides smoothing operations for the Energy Mortar contact formulation.
///
/// This class stores the contact parameters and provides helper routines for
/// constructing smoothed integration bounds from projected overlap intervals.
class ContactSmoothing {
 public:
  /// Clamp the projected overlap interval to the extended smoothing support.
  ///
  /// The input `proj` contains the local projection bounds of edge B onto edge A.
  /// The returned interval is restricted to the extended local range
  /// `[-0.5 - del, 0.5 + del]`.
  static std::array<double, 2> bounds_from_projections( const std::array<double, 2>& proj, double del );

  /// Smooth the integration bounds using the smoothing length `del`.
  ///
  /// The returned bounds are obtained by applying the endpoint smoothing map to
  /// the clamped integration interval. When `del = 0`, the bounds are returned
  /// without smoothing.
  static std::array<double, 2> smooth_bounds( const std::array<double, 2>& bounds, double del );
};

/// Evaluates Energy Mortar contact quantities for a single interface pair.
///
/// This class computes the smoothed mortar gap, tributary areas, contact energy,
/// contact forces, stiffness contributions, and derivative checks used by the
/// Energy Mortar contact formulation. The interface pair is assumed to contain
/// one face from mesh1, treated as edge A/non-mortar/integration side, and one
/// face from mesh2, treated as edge B/mortar/projection side.
class EnergyMortarCalculator {
 public:
  /// Construct a contact evaluator with the supplied contact parameters.
  ///
  /// The parameters define the penalty stiffness, smoothing length, and
  /// derivative path used by the evaluator.
  explicit EnergyMortarCalculator( const ContactParams& p )
      : p_( p ), smoother_() {}  // constructor - copies params into the object

  int get_N() const { return p_.N; }

  std::array<double, 2> compute_projection_bounds( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                   const MeshData::Viewer& mesh2 ) const
  {
    return projections( pair, mesh1, mesh2 );
  }

  /// Construct a three-point Gauss-Legendre quadrature rule over local bounds.
  ///
  /// The input bounds are local coordinates on edge A. The returned quadrature
  /// points and weights are mapped from the reference interval to
  /// `[xi_bounds[0], xi_bounds[1]]`
  static QuadPoints compute_quadrature( const std::array<double, 2>& xi_bounds, int N );

  /// Compute the nodal smoothed gap integrals and tributary areas.
  ///
  /// The output arrays each have length 2, with entries corresponding to the
  /// two nodes of edge A. `gtilde` stores the integrated smoothed gap
  /// contributions, while `area` stores the corresponding tributary area
  /// contributions.
  void compute_gtilde_and_area( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                                double gtilde[2], double area[2] ) const;

  /// Compute first derivatives of the nodal smoothed gap integrals.
  ///
  /// `dgt1_dx` and `dgt2_dx` each have length 8 and store derivatives of the
  /// two nodal gap integrals with respect to the endpoint coordinate vector
  /// described in the class documentation. If `enzyme_quadrature` is false,
  /// the quadrature rule is held fixed during differentiation. If true, the
  /// derivative includes the geometry-dependent quadrature construction.
  void grad_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                    double dgt1_dx[8], double dgt2_dx[8] ) const;

  /// Compute first derivatives of the nodal tributary areas.
  ///
  /// `dA1_dx` and `dA2_dx` each have length 8 and store derivatives of the
  /// two nodal tributary area contributions with respect to the endpoint
  /// coordinate vector described in the class documentation. If
  /// `enzyme_quadrature` is false, the quadrature rule is held fixed during
  /// differentiation. If true, the derivative includes the geometry-dependent
  /// quadrature construction.
  void grad_trib_area( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                       double dA1_dx[8], double dA2_dx[8] ) const;

  /// Compute second derivatives of the nodal smoothed gap integrals.
  ///
  /// `H1` and `H2` each have length 64 and store flattened 8 by 8 Hessian
  /// matrices for the two nodal gap integrals. Entries use row-major indexing:
  /// `H[row * 8 + col]`. If `enzyme_quadrature` is false, the quadrature rule
  /// is held fixed during differentiation. If true, the derivative includes the
  /// geometry-dependent quadrature construction.
  void d2_g2tilde( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                   double dgt1_dx[64], double dgt2_dx[64] ) const;

  /// Compute second derivatives of the nodal tributary areas.
  ///
  /// `d2A1` and `d2A2` each have length 64 and store flattened 8 by 8 Hessian
  /// matrices for the two nodal tributary area contributions. Entries use
  /// row-major indexing: `H[row * 8 + col]`. If `enzyme_quadrature` is false,
  /// the quadrature rule is held fixed during differentiation. If true, the
  /// derivative includes the geometry-dependent quadrature construction.
  void compute_d2A_d2u( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                        double dgt1_dx[64], double dgt2_dx[64] ) const;

  /// Compute values and total coordinate derivatives for the H1 nodal-normal path.
  H1TotalDerivatives compute_h1_total_derivatives( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                   const MeshData::Viewer& mesh2,
                                                   bool compute_second_derivatives = true ) const;

  /// Evaluate and return the two nodal smoothed gap integrals.
  ///
  /// This is a convenience wrapper for obtaining only the gap integral
  /// quantities without also returning the tributary areas for the Finite
  /// difference test.
  std::pair<double, double> eval_gtilde( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                         const MeshData::Viewer& mesh2 ) const;

  /// Validate first derivatives of the smoothed gap integrals using finite differences.
  ///
  /// This routine perturbs the endpoint coordinates of the interface pair and
  /// compares finite-difference approximations against the Enzyme-computed
  /// gradients. The perturbation size is controlled by `epsilon`.
  FiniteDiffResult validate_g_tilde( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
                                     double epsilon = 1e-7 ) const;

  /// Evaluate the two nodal smoothed gap integrals using a fixed quadrature rule.
  ///
  /// This is used for derivative verification when the quadrature points and
  /// weights should remain fixed under coordinate perturbations. Holding the
  /// quadrature fixed isolates derivatives of the gap kernel from derivatives
  /// of the geometry-dependent quadrature construction.
  std::pair<double, double> eval_gtilde_fixed_qp( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                  const MeshData::Viewer& mesh2, const QuadPoints& qp_fixed ) const;

  /// Validate second derivatives of the smoothed gap integrals using finite differences.
  ///
  /// This routine compares Enzyme-computed Hessians against finite-difference
  /// approximations of the first derivatives. The perturbation size is
  /// controlled by `epsilon`.
  FiniteDiffResult validate_hessian( const InterfacePair& pair, MeshData& mesh1, MeshData& mesh2,
                                     double epsilon = 1e-7 ) const;

  /// Compute local energy, force, and stiffness for quadrature-point penalty enforcement.
  QuadraturePointPenaltyData compute_quadrature_point_penalty_data( const InterfacePair& pair,
                                                                    const MeshData::Viewer& mesh1,
                                                                    const MeshData::Viewer& mesh2 ) const;

 private:
  /// Contact parameters controlling penalty stiffness, smoothing, and derivative behavior.
  ContactParams p_;
  /// Helper used to construct smoothed integration bounds
  ContactSmoothing smoother_;

  /// Construct the gap-kernel parameter bundle for the current interface pair.
  ///
  /// This builds the smoothed integration bounds, quadrature points, quadrature
  /// weights, and projected quadrature-point coordinates needed by the lower-level
  /// gap kernel.
  Gparams construct_gparams( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                             const MeshData::Viewer& mesh2 ) const;

  /// Compute the local projection bounds of edge B onto edge A.
  ///
  /// The returned values are local coordinates on edge A and define the interval
  /// used to construct the smoothed integration bounds.
  std::array<double, 2> projections( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                     const MeshData::Viewer& mesh2 ) const;

  /// Compute smoothed gap gradients while holding the quadrature rule fixed.
  ///
  /// This is used by finite-difference verification routines to isolate
  /// derivatives of the gap kernel from derivatives of the quadrature rule.
  void grad_gtilde_with_qp( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                            const QuadPoints& qp_fixed, double dgt1_dx[8], double dgt2_dx[8] ) const;

  /// Evaluate the signed normal gap at a local coordinate on edge A.
  ///
  /// The point on edge A is projected onto edge B, and the gap is evaluated
  /// using the normal from edge B together with the normal-alignment factor.
  double compute_weighted_normal_gap( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                      const MeshData::Viewer& mesh2, double xiA ) const;

  /// Assemble nodal gap and tributary area data for one interface pair.
  ///
  /// This computes the two smoothed nodal gap integrals and the two corresponding
  /// tributary area contributions used to evaluate pressures, forces, energy,
  /// and stiffness terms.
  NodalContactData compute_nodal_contact_data( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                               const MeshData::Viewer& mesh2 ) const;
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
