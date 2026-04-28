#pragma once
#include <vector>
#include <array>

#include "tribol/config.hpp"

#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

#ifdef TRIBOL_USE_ENZYME


//EnergyMortar uses a 3-point Gauss-Legendre quad rule
struct QuadPoints {
  std::array<double, 3> qp; //qp locations
  std::array<double, 3> w;  // weights
};


struct ContactParams {
  double del;               //Smoothing Parameter
  double k;                 //Penalty 
  int N;                    //Quadrature Points
  bool enzyme_quadrature;   //Determines how enzyming is performed (default = True)
};

//Weighted gap and trib area
struct NodalContactData {
  std::array<double, 2> AI; //Trib area
  std::array<double, 2> g_tilde; //Weighted gap
};

///For Finite Difference Test
struct FDResult {
  std::array<double, 2> dgt;
};

///For Finite Difference Test
struct FiniteDiffResult {
  std::vector<double> fd_gradient_g1;
  std::vector<double> fd_gradient_g2;
  std::vector<double> analytical_gradient_g1;
  std::vector<double> analytical_gradient_g2;
  std::vector<int> node_ids;
  double g_tilde1_baseline;
  double g_tilde2_baseline;
};

///Holds data for computations 
struct Gparams {
  int N;              //No. of quad points
  const double* qp;   //Quad points
  const double* w;    //Weights
  const double* x2;   //Coordinates
};

/// Provides smoothing operations for the Energy Mortar contact formulation.
///
/// This class stores the contact parameters and provides helper routines for
/// constructing smoothed integration bounds from projected overlap intervals.
class ContactSmoothing {
 public:
  explicit ContactSmoothing( const ContactParams& p ) : p_( p ) {}  // Constructor

  double get_del() const { return p_.del; }   //getter for delta (smoothing parameter)


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
  static std::array<double, 2> smooth_bounds( const std::array<double, 2>& bounds, double del );        //Smooths bounds 

 private:
  ContactParams p_;
};




/// Evaluates Energy Mortar contact quantities for a single interface pair.
///
/// This class computes the smoothed mortar gap, tributary areas, contact energy,
/// contact forces, stiffness contributions, and derivative checks used by the
/// Energy Mortar contact formulation. The interface pair is assumed to contain
/// one face from mesh1, treated as edge A/non-mortar/integration side, and one
/// face from mesh2, treated as edge B/mortar/projection side.
class ContactEvaluator {
 public:
  /// Construct a contact evaluator with the supplied contact parameters.
  ///
  /// The parameters define the penalty stiffness, smoothing length, and
  /// derivative path used by the evaluator.
  explicit ContactEvaluator( const ContactParams& p )
      : p_( p ), smoother_( p ) {}  // constructor - copies params into the object

  
  /// Compute the scalar contact energy contribution for one interface pair.
  ///
  /// This evaluates the smoothed nodal gap integrals, computes the corresponding
  /// nodal penalty pressures, and contracts the two quantities to form the
  /// pairwise contact energy.
  double compute_contact_energy( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                 const MeshData::Viewer& mesh2 ) const;

  /// Construct a three-point Gauss-Legendre quadrature rule over local bounds.
  ///
  /// The input bounds are local coordinates on edge A. The returned quadrature
  /// points and weights are mapped from the reference interval to
  /// `[xi_bounds[0], xi_bounds[1]]`
  static QuadPoints compute_quadrature( const std::array<double, 2>& xi_bounds );

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

  /// Evaluate and return the two nodal smoothed gap integrals.
  ///
  /// This is a convenience wrapper for obtaining only the gap integral
  /// quantities without also returning the tributary areas for the Finite
  ///difference test.
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

  /// Compute the 8-entry contact force vector for one interface pair.
  ///
  /// The force vector is assembled from derivatives of the nodal smoothed gap
  /// integrals and tributary areas with respect to the endpoint coordinate
  /// vector described in the class documentation.
  std::array<double, 8> compute_contact_forces( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                                const MeshData::Viewer& mesh2 ) const;

  /// Compute the 8 by 8 contact stiffness matrix for one interface pair.
  ///
  /// The stiffness matrix is assembled from first and second derivatives of the
  /// nodal smoothed gap integrals and tributary areas. Matrix entries correspond
  /// to derivatives with respect to the endpoint coordinate vector described in
  /// the class documentation.
  std::array<std::array<double, 8>, 8> compute_stiffness_matrix( const InterfacePair& pair,
                                                                 const MeshData::Viewer& mesh1,
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
  double gap( const InterfacePair& pair, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
              double xiA ) const;

  /// Assemble nodal gap and tributary area data for one interface pair.
  ///
  /// This computes the two smoothed nodal gap integrals and the two corresponding
  /// tributary area contributions used to evaluate pressures, forces, energy,
  /// and stiffness terms.
  NodalContactData compute_nodal_contact_data( const InterfacePair& pair, const MeshData::Viewer& mesh1,
                                               const MeshData::Viewer& mesh2 ) const;

  /// Compute nodal penalty pressures from smoothed gap and area data.
  ///
  /// The integrated nodal gaps are divided by their tributary areas to obtain
  /// area-averaged nodal gaps, and the penalty law is applied only for active
  /// contact.
  std::array<double, 2> compute_penalty_pressures( const NodalContactData& ncd ) const;
};

#endif  // TRIBOL_USE_ENZYME

}  // namespace tribol
