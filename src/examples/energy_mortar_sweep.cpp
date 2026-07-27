// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "axom/slic/core/SimpleLogger.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/physics/EnergyMortar.hpp"

namespace {

using tribol::EnergyMortarCalculator;
using tribol::EnergyMortarNodalEnergyBasis;
using tribol::EnergyMortarNormalMode;
using tribol::EnergyMortarPenaltyMode;
using tribol::EnergyMortarProjectionSmoothingCurve;
using tribol::IndexT;
using tribol::InterfacePair;
using tribol::LINEAR_EDGE;
using tribol::MemorySpace;
using tribol::MeshData;
using tribol::RealT;

struct Options {
  std::string output{ "energy_mortar_sweep.csv" };
  std::string motion{ "slide" };
  std::string normal_mode{ "element" };
  std::string penalty_mode{ "qp_gap" };
  std::string nodal_energy_basis{ "cubic_spline" };
  int steps{ 121 };
  double del{ 0.1 };
  double k{ 3.0 };
  double residual_gap{ 0.0 };
  double h1_active_set_smoothing_gap{ 0.0 };
  double qp_derivative_blend_min_gap{ 0.0 };
  double qp_derivative_blend_max_gap{ 0.0 };
  double qp_derivative_blend_weight{ -1.0 };
  bool qp_derivative_blend_enzyme_gap_weight{ true };
  bool enzyme_quadrature{ true };
  bool fixed_integration_jacobian{ false };
  bool projection_smoothing{ true };
  bool swap_mortar_nonmortar{ false };
  bool eta_gap_scaling{ true };
  bool eta_angle_smoothing{ false };
  double eta_angle_smoothing_start_angle{ 45.0 };
  std::string projection_smoothing_curve{ "quintic" };
  bool nodal_energy_angle_smoothing{ true };
  double fd_step{ 1.0e-5 };
};

struct Geometry {
  std::vector<RealT> x1;
  std::vector<RealT> y1;
  std::vector<RealT> x2;
  std::vector<RealT> y2;
};

struct EvalResult {
  double energy{ 0.0 };
  bool has_energy{ false };
  std::array<double, 8> force{};
  bool has_pair_force{ false };
  std::array<double, 10> global_force{};
  bool has_global_force{ false };
  std::vector<double> h1_force;
  std::vector<int> h1_mesh1_nodes;
  std::vector<int> h1_mesh2_nodes;
  std::array<double, 2> gtilde{};
  std::array<double, 2> area{};
  std::array<double, 2> proj{};
  std::array<double, 2> bounds{};
  std::array<double, 2> smooth_bounds{};
  tribol::QuadPoints qp{};
};

void printUsage( const char* prog )
{
  std::cout << "Usage: " << prog << " [options]\n\n"
            << "Writes a CSV for animating one EnergyMortar element pair.\n\n"
            << "Options:\n"
            << "  --output FILE                  CSV path (default: energy_mortar_sweep.csv)\n"
            << "  --motion slide|slide_kink|slide_flat|slide_corner_down|rotate_perpendicular|approach|skew\n"
            << "                                 Motion path (default: slide)\n"
            << "  --steps N                      Number of frames (default: 121)\n"
            << "  --normal-mode element|h1       Normal mode (default: element)\n"
            << "  --penalty-mode qp_gap|nodal_gap|nodal_energy\n"
            << "                                 Enforcement diagnostic (default: qp_gap)\n"
            << "  --basis fe|cubic_spline        NODAL_ENERGY basis (default: cubic_spline)\n"
            << "  --del VALUE                    Smoothing length (default: 0.1)\n"
            << "  --k VALUE                      Penalty stiffness (default: 3.0)\n"
            << "  --residual-gap VALUE           Residual gap offset (default: 0)\n"
            << "  --h1-active-gap VALUE          H1 active-set smoothing gap (default: 0)\n"
            << "  --qp-blend-gap VALUE           QP residual-gap transition from full to simplified derivative "
               "(default: 0)\n"
            << "  --qp-blend-min-gap VALUE       Residual gap below which the full derivative is used (default: 0)\n"
            << "  --qp-blend-max-gap VALUE       Residual gap above which the simplified derivative is used (default: 0)\n"
            << "  --qp-blend-weight VALUE        Fixed full-path QP derivative blend weight, clamped to [0,1] (default: "
               "disabled)\n"
            << "  --qp-blend-enzyme-gap-weight 0|1\n"
            << "                                 Differentiate gap-based QP blend weight with Enzyme (default: 1)\n"
            << "  --enzyme-quadrature 0|1        Differentiate quadrature construction (default: 1)\n"
            << "  --fixed-jacobian 0|1           Hold integration Jacobian fixed (default: 0)\n"
            << "  --projection-smoothing 0|1     Smooth projection bounds (default: 1)\n"
            << "  --swap-mortar-nonmortar 0|1    Swap edge A and edge B in the EnergyMortar evaluation (default: 0)\n"
            << "  --eta-gap-scaling 0|1          Scale the normal gap by eta, the surface-normal dot product (default: 1)\n"
            << "  --eta-angle-smoothing 0|1      Smooth eta to zero near 90 degrees when eta gap scaling is off "
               "(default: 0)\n"
            << "  --eta-angle-smoothing-start-angle VALUE\n"
            << "                                 Eta smoothing start angle in degrees; smoothing ends at 90 (default: 45)\n"
            << "  --projection-smoothing-curve quadratic|quintic\n"
            << "                                 Projection-bound smoothing curve (default: quintic)\n"
            << "  --angle-smoothing 0|1          NODAL_ENERGY angle smoothing (default: 1)\n"
            << "  --fd-step VALUE                Path finite-difference step (default: 1e-5)\n"
            << "  --help                         Show this message\n";
}

bool parseBool( const std::string& value )
{
  if ( value == "1" || value == "true" || value == "on" ) {
    return true;
  }
  if ( value == "0" || value == "false" || value == "off" ) {
    return false;
  }
  throw std::runtime_error( "Expected boolean 0/1, true/false, or on/off; got '" + value + "'." );
}

Options parseArgs( int argc, char** argv )
{
  Options opts;
  for ( int i = 1; i < argc; ++i ) {
    const std::string arg = argv[i];
    auto needValue = [&]( const std::string& name ) -> std::string {
      if ( i + 1 >= argc ) {
        throw std::runtime_error( "Missing value for " + name );
      }
      return argv[++i];
    };

    if ( arg == "--help" || arg == "-h" ) {
      printUsage( argv[0] );
      std::exit( 0 );
    } else if ( arg == "--output" || arg == "-o" ) {
      opts.output = needValue( arg );
    } else if ( arg == "--motion" ) {
      opts.motion = needValue( arg );
    } else if ( arg == "--steps" ) {
      opts.steps = std::stoi( needValue( arg ) );
    } else if ( arg == "--normal-mode" ) {
      opts.normal_mode = needValue( arg );
    } else if ( arg == "--penalty-mode" ) {
      opts.penalty_mode = needValue( arg );
    } else if ( arg == "--basis" ) {
      opts.nodal_energy_basis = needValue( arg );
    } else if ( arg == "--del" ) {
      opts.del = std::stod( needValue( arg ) );
    } else if ( arg == "--k" ) {
      opts.k = std::stod( needValue( arg ) );
    } else if ( arg == "--residual-gap" ) {
      opts.residual_gap = std::stod( needValue( arg ) );
    } else if ( arg == "--h1-active-gap" ) {
      opts.h1_active_set_smoothing_gap = std::stod( needValue( arg ) );
    } else if ( arg == "--qp-blend-gap" ) {
      opts.qp_derivative_blend_min_gap = 0.0;
      opts.qp_derivative_blend_max_gap = std::stod( needValue( arg ) );
    } else if ( arg == "--qp-blend-min-gap" ) {
      opts.qp_derivative_blend_min_gap = std::stod( needValue( arg ) );
    } else if ( arg == "--qp-blend-max-gap" ) {
      opts.qp_derivative_blend_max_gap = std::stod( needValue( arg ) );
    } else if ( arg == "--qp-blend-weight" ) {
      opts.qp_derivative_blend_weight = std::stod( needValue( arg ) );
    } else if ( arg == "--qp-blend-enzyme-gap-weight" ) {
      opts.qp_derivative_blend_enzyme_gap_weight = parseBool( needValue( arg ) );
    } else if ( arg == "--enzyme-quadrature" ) {
      opts.enzyme_quadrature = parseBool( needValue( arg ) );
    } else if ( arg == "--fixed-jacobian" ) {
      opts.fixed_integration_jacobian = parseBool( needValue( arg ) );
    } else if ( arg == "--projection-smoothing" ) {
      opts.projection_smoothing = parseBool( needValue( arg ) );
    } else if ( arg == "--swap-mortar-nonmortar" ) {
      opts.swap_mortar_nonmortar = parseBool( needValue( arg ) );
    } else if ( arg == "--eta-gap-scaling" ) {
      opts.eta_gap_scaling = parseBool( needValue( arg ) );
    } else if ( arg == "--eta-angle-smoothing" ) {
      opts.eta_angle_smoothing = parseBool( needValue( arg ) );
    } else if ( arg == "--eta-angle-smoothing-start-angle" ) {
      opts.eta_angle_smoothing_start_angle = std::stod( needValue( arg ) );
    } else if ( arg == "--projection-smoothing-curve" ) {
      opts.projection_smoothing_curve = needValue( arg );
    } else if ( arg == "--angle-smoothing" ) {
      opts.nodal_energy_angle_smoothing = parseBool( needValue( arg ) );
    } else if ( arg == "--fd-step" ) {
      opts.fd_step = std::stod( needValue( arg ) );
    } else {
      throw std::runtime_error( "Unknown option '" + arg + "'. Use --help for usage." );
    }
  }

  if ( opts.steps < 2 ) {
    throw std::runtime_error( "--steps must be at least 2." );
  }
  if ( opts.del < 0.0 ) {
    throw std::runtime_error( "--del must be nonnegative." );
  }
  if ( opts.fd_step <= 0.0 ) {
    throw std::runtime_error( "--fd-step must be positive." );
  }
  if ( opts.eta_angle_smoothing_start_angle < 0.0 || opts.eta_angle_smoothing_start_angle >= 90.0 ) {
    throw std::runtime_error( "--eta-angle-smoothing-start-angle must be in [0, 90)." );
  }
  return opts;
}

EnergyMortarProjectionSmoothingCurve parseProjectionSmoothingCurve( const std::string& curve )
{
  if ( curve == "quadratic" ) {
    return EnergyMortarProjectionSmoothingCurve::QUADRATIC;
  }
  if ( curve == "quintic" ) {
    return EnergyMortarProjectionSmoothingCurve::QUINTIC;
  }
  throw std::runtime_error( "Unknown projection smoothing curve '" + curve + "'." );
}

std::string projectionSmoothingCurveName( EnergyMortarProjectionSmoothingCurve curve )
{
  return curve == EnergyMortarProjectionSmoothingCurve::QUADRATIC ? "quadratic" : "quintic";
}

EnergyMortarNormalMode parseNormalMode( const std::string& mode )
{
  if ( mode == "element" || mode == "element_normal" ) {
    return EnergyMortarNormalMode::ELEMENT_NORMAL;
  }
  if ( mode == "h1" || mode == "h1_nodal_normal" ) {
    return EnergyMortarNormalMode::H1_NODAL_NORMAL;
  }
  throw std::runtime_error( "Unknown normal mode '" + mode + "'." );
}

EnergyMortarPenaltyMode parsePenaltyMode( const std::string& mode )
{
  if ( mode == "nodal_gap" ) {
    return EnergyMortarPenaltyMode::NODAL_GAP;
  }
  if ( mode == "qp_gap" || mode == "quadrature_point_gap" ) {
    return EnergyMortarPenaltyMode::QUADRATURE_POINT_GAP;
  }
  if ( mode == "nodal_energy" ) {
    return EnergyMortarPenaltyMode::NODAL_ENERGY;
  }
  throw std::runtime_error( "Unknown penalty mode '" + mode + "'." );
}

EnergyMortarNodalEnergyBasis parseBasis( const std::string& basis )
{
  if ( basis == "fe" ) {
    return EnergyMortarNodalEnergyBasis::FE;
  }
  if ( basis == "cubic_spline" || basis == "spline" ) {
    return EnergyMortarNodalEnergyBasis::CUBIC_SPLINE;
  }
  throw std::runtime_error( "Unknown nodal-energy basis '" + basis + "'." );
}

std::string normalModeName( EnergyMortarNormalMode mode )
{
  return mode == EnergyMortarNormalMode::ELEMENT_NORMAL ? "element" : "h1";
}

std::string penaltyModeName( EnergyMortarPenaltyMode mode )
{
  if ( mode == EnergyMortarPenaltyMode::NODAL_GAP ) {
    return "nodal_gap";
  }
  if ( mode == EnergyMortarPenaltyMode::QUADRATURE_POINT_GAP ) {
    return "qp_gap";
  }
  return "nodal_energy";
}

std::string basisName( EnergyMortarNodalEnergyBasis basis )
{
  return basis == EnergyMortarNodalEnergyBasis::FE ? "fe" : "cubic_spline";
}

tribol::ContactParams makeParams( const Options& opts )
{
  tribol::ContactParams params;
  params.del = opts.del;
  params.k = opts.k;
  params.N = tribol::energy_mortar_num_quad_points;
  params.enzyme_quadrature = opts.enzyme_quadrature;
  params.fixed_integration_jacobian = opts.fixed_integration_jacobian;
  params.normal_mode = parseNormalMode( opts.normal_mode );
  params.projection_smoothing = opts.projection_smoothing;
  params.projection_smoothing_curve = parseProjectionSmoothingCurve( opts.projection_smoothing_curve );
  params.h1_active_set_smoothing_gap = opts.h1_active_set_smoothing_gap;
  params.qp_derivative_blend_min_gap = opts.qp_derivative_blend_min_gap;
  params.qp_derivative_blend_max_gap = opts.qp_derivative_blend_max_gap;
  if ( ( params.qp_derivative_blend_min_gap > 0.0 || params.qp_derivative_blend_max_gap > 0.0 ) &&
       params.qp_derivative_blend_max_gap <= params.qp_derivative_blend_min_gap ) {
    throw std::runtime_error( "QP blend max gap must be greater than min gap." );
  }
  params.qp_derivative_blend_weight = opts.qp_derivative_blend_weight;
  if ( params.qp_derivative_blend_weight > 1.0 ) {
    params.qp_derivative_blend_weight = 1.0;
  }
  params.qp_derivative_blend_enzyme_gap_weight = opts.qp_derivative_blend_enzyme_gap_weight;
  params.penalty_mode = parsePenaltyMode( opts.penalty_mode );
  params.nodal_energy_basis = parseBasis( opts.nodal_energy_basis );
  params.eta_gap_scaling = opts.eta_gap_scaling;
  params.eta_angle_smoothing = opts.eta_angle_smoothing;
  constexpr double pi = 3.14159265358979323846264338327950288;
  params.eta_angle_smoothing_start = opts.eta_angle_smoothing_start_angle * pi / 180.0;
  params.nodal_energy_angle_smoothing = opts.nodal_energy_angle_smoothing;
  params.residual_gap = opts.residual_gap;

  if ( params.penalty_mode == EnergyMortarPenaltyMode::NODAL_ENERGY &&
       params.normal_mode != EnergyMortarNormalMode::H1_NODAL_NORMAL ) {
    throw std::runtime_error( "NODAL_ENERGY requires --normal-mode h1." );
  }
  return params;
}

double pathCoordinate( const Options& opts, int step )
{
  const double t = static_cast<double>( step ) / static_cast<double>( opts.steps - 1 );
  if ( opts.motion == "approach" ) {
    return -1.0 + 2.0 * t;
  }
  if ( opts.motion == "slide" ) {
    return -1.0 + 2.0 * t;
  }
  if ( opts.motion == "slide_kink" ) {
    constexpr double cos30 = 0.86602540378443864676;
    return -0.8 + ( 1.6 + cos30 ) * t;
  }
  if ( opts.motion == "slide_flat" ) {
    return 2.0 * t;
  }
  if ( opts.motion == "slide_corner_down" ) {
    return t;
  }
  if ( opts.motion == "rotate_perpendicular" ) {
    return t;
  }
  if ( opts.motion == "skew" ) {
    return -0.75 + 1.5 * t;
  }
  throw std::runtime_error( "Unknown motion '" + opts.motion + "'." );
}

double stepToPathCoordinate( const Options& opts, int step ) { return pathCoordinate( opts, step ); }

Geometry geometryAt( const Options& opts, const tribol::ContactParams& params, double s )
{
  const bool h1 = params.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL;
  Geometry geom;

  if ( h1 ) {
    constexpr double cos30 = 0.86602540378443864676;
    geom.x1 = { 0.0, 1.0, 2.0 };
    geom.y1 = ( opts.motion == "slide_kink" ) ? std::vector<RealT>{ 0.0, 0.0, 0.5 }
                                               : std::vector<RealT>{ 0.0, 0.0, 0.08 };
    if ( opts.motion == "slide_flat" ) {
      geom.y1 = { 0.0, 0.0, 0.0 };
    }
    if ( opts.motion == "slide_corner_down" ) {
      geom.x1 = { 0.0, 1.0, 1.0 };
      geom.y1 = { 0.0, 0.0, -1.0 };
    }
    if ( opts.motion == "rotate_perpendicular" ) {
      geom.x1 = { 0.0, 1.0, 2.0 };
      geom.y1 = { 0.0, 0.0, 0.0 };
    }
    if ( opts.motion == "slide_kink" ) {
      geom.x1[2] = 1.0 + cos30;
    }
    geom.x2 = ( opts.motion == "slide_flat" || opts.motion == "slide_corner_down" )
                  ? std::vector<RealT>{ 0.2, 0.8 }
              : ( opts.motion == "rotate_perpendicular" ) ? std::vector<RealT>{ 0.2, 0.5, 0.8 }
                                                           : std::vector<RealT>{ 0.2, 0.9, 1.6 };
    geom.y2 = ( opts.motion == "slide_flat" || opts.motion == "slide_corner_down" ||
                opts.motion == "rotate_perpendicular" )
                  ? std::vector<RealT>{ -0.2, -0.2 }
                  : std::vector<RealT>{ -0.14, -0.14, -0.14 };
    if ( opts.motion == "rotate_perpendicular" ) {
      geom.y2 = { -0.2, -0.2, -0.2 };
    }
  } else if ( opts.motion == "slide_kink" || opts.motion == "slide_flat" || opts.motion == "slide_corner_down" ) {
    constexpr double cos30 = 0.86602540378443864676;
    if ( opts.motion == "slide_kink" ) {
      geom.x1 = { 0.0, 1.0, 1.0 + cos30 };
      geom.y1 = { 0.0, 0.0, 0.5 };
    } else if ( opts.motion == "slide_corner_down" ) {
      geom.x1 = { 0.0, 1.0, 1.0 };
      geom.y1 = { 0.0, 0.0, -1.0 };
    } else {
      geom.x1 = { 0.0, 1.0, 2.0 };
      geom.y1 = { 0.0, 0.0, 0.0 };
    }
    geom.x2 = { 0.2, 0.8 };
    geom.y2 = ( opts.motion == "slide_flat" || opts.motion == "slide_corner_down" )
                  ? std::vector<RealT>{ -0.2, -0.2 }
                  : std::vector<RealT>{ -0.1, -0.1 };
  } else {
    geom.x1 = { 0.0, 1.0 };
    geom.y1 = { 0.0, 0.0 };
    geom.x2 = { 0.2, 0.8 };
    geom.y2 = { -0.1, -0.1 };
    if ( opts.motion == "rotate_perpendicular" ) {
      geom.y2 = { -0.2, -0.2 };
    }
  }

  if ( opts.motion == "approach" ) {
    const double gap = 0.18 - 0.32 * ( 0.5 * ( s + 1.0 ) );
    if ( h1 ) {
      geom.y2 = { gap - 0.04, gap, gap + 0.04 };
    } else {
      geom.y2 = { gap, gap };
    }
  } else if ( opts.motion == "slide" || opts.motion == "slide_kink" || opts.motion == "slide_flat" ||
              opts.motion == "slide_corner_down" ) {
    for ( double& x : geom.x2 ) {
      x += s;
    }
  } else if ( opts.motion == "rotate_perpendicular" ) {
    constexpr double max_angle = 110.0 * M_PI / 180.0;
    const double angle = max_angle * s;
    const double cx = 0.5;
    const double cy = -0.2;
    const double c = std::cos( angle );
    const double sn = std::sin( angle );
    for ( std::size_t i = 0; i < geom.x2.size(); ++i ) {
      const double dx = geom.x2[i] - cx;
      const double dy = geom.y2[i] - cy;
      geom.x2[i] = cx + c * dx - sn * dy;
      geom.y2[i] = cy + sn * dx + c * dy;
    }
  } else if ( opts.motion == "skew" ) {
    for ( double& x : geom.x2 ) {
      x += s;
    }
    const double angle = -0.35 + 0.70 * ( ( s + 0.75 ) / 1.5 );
    const double cx = 0.5 + s;
    const double cy = h1 ? -0.14 : -0.1;
    const double c = std::cos( angle );
    const double sn = std::sin( angle );
    for ( std::size_t i = 0; i < geom.x2.size(); ++i ) {
      const double dx = geom.x2[i] - cx;
      const double dy = geom.y2[i] - cy;
      geom.x2[i] = cx + c * dx - sn * dy;
      geom.y2[i] = cy + sn * dx + c * dy;
    }
  } else {
    throw std::runtime_error( "Unknown motion '" + opts.motion + "'." );
  }

  return geom;
}

bool hasTwoElementAEdge( const Options& opts )
{
  return opts.motion == "slide_kink" || opts.motion == "slide_flat" || opts.motion == "slide_corner_down";
}

int activeAElement( const Options& opts, const Geometry& geom )
{
  if ( !hasTwoElementAEdge( opts ) ) {
    return 0;
  }
  const double b_center = 0.5 * ( geom.x2[0] + geom.x2[1] );
  return b_center < geom.x1[1] ? 0 : 1;
}

std::array<double, 8> pairCoordinates( const MeshData& mesh1, const MeshData& mesh2, const InterfacePair& pair )
{
  const auto view1 = const_cast<MeshData&>( mesh1 ).getView();
  const auto view2 = const_cast<MeshData&>( mesh2 ).getView();
  const auto conn1 = view1.getConnectivity()( pair.m_element_id1 );
  const auto conn2 = view2.getConnectivity()( pair.m_element_id2 );
  return { view1.getPosition()[0][conn1[0]], view1.getPosition()[1][conn1[0]], view1.getPosition()[0][conn1[1]],
           view1.getPosition()[1][conn1[1]], view2.getPosition()[0][conn2[0]], view2.getPosition()[1][conn2[0]],
           view2.getPosition()[0][conn2[1]], view2.getPosition()[1][conn2[1]] };
}

double normalForceProxy( const std::array<double, 8>& coord, const std::array<double, 8>& force )
{
  const double tx = coord[2] - coord[0];
  const double ty = coord[3] - coord[1];
  const double len = std::sqrt( tx * tx + ty * ty );
  if ( len == 0.0 ) {
    return 0.0;
  }
  const double nx = ty / len;
  const double ny = -tx / len;
  return ( force[1] + force[3] ) * ny + ( force[0] + force[2] ) * nx;
}

double forceDotPathProxy( const Options& opts, const Geometry& geom, const std::array<double, 8>& force )
{
  if ( opts.motion == "slide" || opts.motion == "slide_kink" || opts.motion == "slide_flat" ||
       opts.motion == "slide_corner_down" || opts.motion == "skew" ) {
    return force[4] + force[6];
  }
  if ( opts.motion == "rotate_perpendicular" ) {
    constexpr double omega = 110.0 * M_PI / 180.0;
    constexpr double cx = 0.5;
    constexpr double cy = -0.2;
    const double vx0 = -omega * ( geom.y2[0] - cy );
    const double vy0 = omega * ( geom.x2[0] - cx );
    const double vx1 = -omega * ( geom.y2[1] - cy );
    const double vy1 = omega * ( geom.x2[1] - cx );
    return force[4] * vx0 + force[5] * vy0 + force[6] * vx1 + force[7] * vy1;
  }
  if ( opts.motion == "approach" ) {
    return force[5] + force[7];
  }
  return 0.0;
}

double normalForceProxyFromB( const Geometry& geom, const std::array<double, 10>& force )
{
  const double tx = geom.x2[1] - geom.x2[0];
  const double ty = geom.y2[1] - geom.y2[0];
  const double len = std::sqrt( tx * tx + ty * ty );
  if ( len == 0.0 ) {
    return 0.0;
  }
  const double nx = ty / len;
  const double ny = -tx / len;
  return force[6] * nx + force[7] * ny + force[8] * nx + force[9] * ny;
}

double addNodalGapPenaltyContribution( double k, double g_tilde, double area, const double* dg_dx, const double* dA_dx,
                                       int ndof, double* dE_dx )
{
  constexpr double area_tol = 1.0e-30;
  if ( area <= area_tol ) {
    return 0.0;
  }

  const double inv_area = 1.0 / area;
  const double energy = k * g_tilde * g_tilde * inv_area;
  const double dE_dg = 2.0 * k * g_tilde * inv_area;
  const double dE_dA = -k * g_tilde * g_tilde * inv_area * inv_area;
  for ( int i = 0; i < ndof; ++i ) {
    dE_dx[i] += dE_dg * dg_dx[i] + dE_dA * dA_dx[i];
  }
  return energy;
}

void addGlobalForce( std::array<double, 10>& global_force, bool side_is_a, int node_id, double fx, double fy )
{
  const int offset = side_is_a ? 0 : 6;
  const int index = offset + 2 * node_id;
  if ( index >= 0 && index + 1 < static_cast<int>( global_force.size() ) ) {
    global_force[index] += fx;
    global_force[index + 1] += fy;
  }
}

template <typename NodeList1, typename NodeList2>
void addH1ForceToGlobal( std::array<double, 10>& global_force, const std::vector<double>& force,
                         const NodeList1& side1_nodes, int num_side1_nodes, const NodeList2& side2_nodes,
                         int num_side2_nodes, bool side1_is_a )
{
  for ( int i = 0; i < num_side1_nodes; ++i ) {
    addGlobalForce( global_force, side1_is_a, side1_nodes[i], force[i], force[num_side1_nodes + i] );
  }
  const int side2_offset = 2 * num_side1_nodes;
  for ( int i = 0; i < num_side2_nodes; ++i ) {
    addGlobalForce( global_force, !side1_is_a, side2_nodes[i], force[side2_offset + i],
                    force[side2_offset + num_side2_nodes + i] );
  }
}

EvalResult evaluate( const Options& opts, const tribol::ContactParams& params, double s )
{
  const Geometry geom = geometryAt( opts, params, s );
  const bool h1 = params.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL;
  const bool two_element_a_edge = hasTwoElementAEdge( opts );

  IndexT conn1_single[2] = { 1, 0 };
  IndexT conn2_single[2] = { 0, 1 };
  IndexT conn1_h1[4] = { 1, 0, 2, 1 };
  IndexT conn2_h1[4] = { 0, 1, 1, 2 };

  const int num_elems1 = ( h1 || two_element_a_edge ) ? 2 : 1;
  const int num_elems2 = ( h1 && !two_element_a_edge ) ? 2 : 1;
  MeshData mesh1( 0, num_elems1, static_cast<int>( geom.x1.size() ), num_elems1 == 2 ? conn1_h1 : conn1_single,
                  LINEAR_EDGE,
                  const_cast<RealT*>( geom.x1.data() ), const_cast<RealT*>( geom.y1.data() ), nullptr,
                  MemorySpace::Host );
  MeshData mesh2( 1, num_elems2, static_cast<int>( geom.x2.size() ), num_elems2 == 2 ? conn2_h1 : conn2_single,
                  LINEAR_EDGE,
                  const_cast<RealT*>( geom.x2.data() ), const_cast<RealT*>( geom.y2.data() ), nullptr,
                  MemorySpace::Host );
  mesh1.setReferencePosition( geom.x1.data(), geom.y1.data(), nullptr );
  mesh2.setReferencePosition( geom.x2.data(), geom.y2.data(), nullptr );

  MeshData& eval_mesh1 = opts.swap_mortar_nonmortar ? mesh2 : mesh1;
  MeshData& eval_mesh2 = opts.swap_mortar_nonmortar ? mesh1 : mesh2;
  const bool eval_side1_is_a = !opts.swap_mortar_nonmortar;

  EnergyMortarCalculator evaluator( params );

  EvalResult result;
  const int num_pairs = two_element_a_edge ? 2 : 1;
  for ( int elem1 = 0; elem1 < num_pairs; ++elem1 ) {
    const int active_a_elem = two_element_a_edge ? elem1 : activeAElement( opts, geom );
    const InterfacePair pair( opts.swap_mortar_nonmortar ? 0 : active_a_elem,
                              opts.swap_mortar_nonmortar ? active_a_elem : 0 );
    std::array<double, 8> pair_force{};
    bool pair_has_energy = false;
    bool pair_has_force = false;
    double pair_energy = 0.0;

    auto diagnostic_params = params;
    diagnostic_params.normal_mode = EnergyMortarNormalMode::ELEMENT_NORMAL;
    EnergyMortarCalculator diagnostic_evaluator( diagnostic_params );
    const auto pair_proj =
        diagnostic_evaluator.compute_projection_bounds( pair, eval_mesh1.getView(), eval_mesh2.getView() );
    const double projection_bound_delta = params.projection_smoothing ? params.del : 0.0;
    const auto pair_bounds = tribol::ContactSmoothing::bounds_from_projections( pair_proj, projection_bound_delta );
    const auto pair_smooth_bounds =
        params.projection_smoothing ? tribol::ContactSmoothing::smooth_bounds( pair_bounds, projection_bound_delta,
                                                                               params.projection_smoothing_curve )
                                    : pair_bounds;
    const auto pair_qp = EnergyMortarCalculator::compute_quadrature( pair_smooth_bounds, params.N );

    if ( elem1 == 0 ) {
      result.proj = pair_proj;
      result.bounds = pair_bounds;
      result.smooth_bounds = pair_smooth_bounds;
      result.qp = pair_qp;
    } else {
      result.proj[0] = std::min( result.proj[0], pair_proj[0] );
      result.proj[1] = std::max( result.proj[1], pair_proj[1] );
      result.bounds[0] = std::min( result.bounds[0], pair_bounds[0] );
      result.bounds[1] = std::max( result.bounds[1], pair_bounds[1] );
      result.smooth_bounds[0] = std::min( result.smooth_bounds[0], pair_smooth_bounds[0] );
      result.smooth_bounds[1] = std::max( result.smooth_bounds[1], pair_smooth_bounds[1] );
    }

    std::array<double, 2> pair_gtilde{};
    std::array<double, 2> pair_area{};
    if ( h1 ) {
      const auto h1_data =
          evaluator.compute_h1_total_derivatives( pair, eval_mesh1.getView(), eval_mesh2.getView(), false );
      pair_gtilde = h1_data.g_tilde;
      pair_area = h1_data.area;
      if ( params.penalty_mode == EnergyMortarPenaltyMode::NODAL_GAP ) {
        const int ndof = static_cast<int>( h1_data.dg1_dx.size() );
        std::vector<double> h1_force( ndof, 0.0 );
        pair_energy += addNodalGapPenaltyContribution( params.k, pair_gtilde[0], pair_area[0],
                                                       h1_data.dg1_dx.data(), h1_data.dA1_dx.data(), ndof,
                                                       h1_force.data() );
        pair_energy += addNodalGapPenaltyContribution( params.k, pair_gtilde[1], pair_area[1],
                                                       h1_data.dg2_dx.data(), h1_data.dA2_dx.data(), ndof,
                                                       h1_force.data() );
        addH1ForceToGlobal( result.global_force, h1_force, h1_data.mesh1_nodes, h1_data.num_mesh1_nodes,
                            h1_data.mesh2_nodes, h1_data.num_mesh2_nodes, eval_side1_is_a );
        pair_has_energy = true;
        pair_has_force = true;
      }
    } else {
      double gtilde[2] = { 0.0, 0.0 };
      double area[2] = { 0.0, 0.0 };
      evaluator.compute_gtilde_and_area( pair, eval_mesh1.getView(), eval_mesh2.getView(), gtilde, area );
      pair_gtilde = { gtilde[0], gtilde[1] };
      pair_area = { area[0], area[1] };
      if ( params.penalty_mode == EnergyMortarPenaltyMode::NODAL_GAP ) {
        double dg1_dx[8] = { 0.0 };
        double dg2_dx[8] = { 0.0 };
        double dA1_dx[8] = { 0.0 };
        double dA2_dx[8] = { 0.0 };
        evaluator.grad_gtilde( pair, eval_mesh1.getView(), eval_mesh2.getView(), dg1_dx, dg2_dx );
        evaluator.grad_trib_area( pair, eval_mesh1.getView(), eval_mesh2.getView(), dA1_dx, dA2_dx );
        pair_energy += addNodalGapPenaltyContribution( params.k, pair_gtilde[0], pair_area[0], dg1_dx, dA1_dx, 8,
                                                       pair_force.data() );
        pair_energy += addNodalGapPenaltyContribution( params.k, pair_gtilde[1], pair_area[1], dg2_dx, dA2_dx, 8,
                                                       pair_force.data() );
        pair_has_energy = true;
        pair_has_force = true;
      }
    }

    if ( params.penalty_mode != EnergyMortarPenaltyMode::NODAL_GAP ) {
      const auto penalty =
          evaluator.compute_quadrature_point_penalty_data( pair, eval_mesh1.getView(), eval_mesh2.getView() );
      pair_energy = penalty.energy;
      pair_has_energy = true;
      if ( h1 ) {
        addH1ForceToGlobal( result.global_force, penalty.h1_force, penalty.mesh1_nodes, penalty.num_mesh1_nodes,
                            penalty.mesh2_nodes, penalty.num_mesh2_nodes, eval_side1_is_a );
      } else {
        pair_force = penalty.force;
      }
      pair_has_force = true;
    }

    result.energy += pair_energy;
    result.has_energy = result.has_energy || pair_has_energy;
    result.has_pair_force = result.has_pair_force || pair_has_force;
    result.gtilde[0] += pair_gtilde[0];
    result.gtilde[1] += pair_gtilde[1];
    result.area[0] += pair_area[0];
    result.area[1] += pair_area[1];

    if ( !h1 ) {
      const auto conn1 = eval_mesh1.getView().getConnectivity()( pair.m_element_id1 );
      const auto conn2 = eval_mesh2.getView().getConnectivity()( pair.m_element_id2 );
      for ( int node = 0; node < 2; ++node ) {
        addGlobalForce( result.global_force, eval_side1_is_a, conn1[node], pair_force[2 * node],
                        pair_force[2 * node + 1] );
      }
      for ( int node = 0; node < 2; ++node ) {
        addGlobalForce( result.global_force, !eval_side1_is_a, conn2[node], pair_force[4 + 2 * node],
                        pair_force[4 + 2 * node + 1] );
      }
    }
  }

  result.has_global_force = result.has_pair_force;
  result.force = { result.global_force[0], result.global_force[1], result.global_force[2], result.global_force[3],
                   result.global_force[6], result.global_force[7], result.global_force[8], result.global_force[9] };

  return result;
}

void writeMaybe( std::ostream& os, bool has_value, double value )
{
  if ( has_value ) {
    os << value;
  }
}

int run( const Options& opts )
{
  const auto params = makeParams( opts );
  std::ofstream out( opts.output );
  if ( !out ) {
    throw std::runtime_error( "Could not open output CSV '" + opts.output + "'." );
  }
  out << std::setprecision( 16 );

  out << "case,step,s,normal_mode,penalty_mode,basis,enzyme_quadrature,fixed_jacobian,projection_smoothing,"
         "swap_mortar_nonmortar,eta_gap_scaling,eta_angle_smoothing,projection_smoothing_curve,"
         "eta_angle_smoothing_start_angle,"
         "active_smoothing_gap,qp_blend_min_gap,qp_blend_max_gap,"
         "qp_blend_weight,"
         "qp_blend_enzyme_gap_weight,"
         "residual_gap,k,del,A0_x,A0_y,A1_x,"
         "A1_y,A2_x,A2_y,B0_x,B0_y,B1_x,B1_y,energy,"
         "normal_force,force_dot_direction,fd_dE_ds,force_error,gap,gtilde0,gtilde1,area0,area1,proj_min,proj_max,"
         "bound_min,bound_max,smooth_min,smooth_max,qp0,qp1,qp2,w0,w1,w2,fA0_x,fA0_y,fA1_x,fA1_y,fA2_x,fA2_y,"
         "fB0_x,fB0_y,fB1_x,fB1_y,fx0,fy0,fx1,fy1,fx2,fy2,fx3,fy3\n";

  for ( int step = 0; step < opts.steps; ++step ) {
    const double s = stepToPathCoordinate( opts, step );
    const auto result = evaluate( opts, params, s );
    const Geometry geom = geometryAt( opts, params, s );
    const bool h1 = params.normal_mode == EnergyMortarNormalMode::H1_NODAL_NORMAL;
    const bool two_element_a_edge = hasTwoElementAEdge( opts );
    IndexT conn1_single[2] = { 1, 0 };
    IndexT conn2_single[2] = { 0, 1 };
    IndexT conn1_h1[4] = { 1, 0, 2, 1 };
    IndexT conn2_h1[4] = { 0, 1, 1, 2 };
    const int num_elems1 = ( h1 || two_element_a_edge ) ? 2 : 1;
    const int num_elems2 = ( h1 && !two_element_a_edge ) ? 2 : 1;
    MeshData mesh1( 0, num_elems1, static_cast<int>( geom.x1.size() ), num_elems1 == 2 ? conn1_h1 : conn1_single,
                    LINEAR_EDGE,
                    const_cast<RealT*>( geom.x1.data() ), const_cast<RealT*>( geom.y1.data() ), nullptr,
                    MemorySpace::Host );
    MeshData mesh2( 1, num_elems2, static_cast<int>( geom.x2.size() ), num_elems2 == 2 ? conn2_h1 : conn2_single,
                    LINEAR_EDGE,
                    const_cast<RealT*>( geom.x2.data() ), const_cast<RealT*>( geom.y2.data() ), nullptr,
                    MemorySpace::Host );
    const InterfacePair pair( activeAElement( opts, geom ), 0 );
    const auto coord = pairCoordinates( mesh1, mesh2, pair );

    bool has_fd = false;
    double fd_dE_ds = 0.0;
    if ( result.has_energy ) {
      const auto plus = evaluate( opts, params, s + opts.fd_step );
      const auto minus = evaluate( opts, params, s - opts.fd_step );
      if ( plus.has_energy && minus.has_energy ) {
        fd_dE_ds = ( plus.energy - minus.energy ) / ( 2.0 * opts.fd_step );
        has_fd = true;
      }
    }

    const double force_dot_path = result.has_pair_force ? forceDotPathProxy( opts, geom, result.force ) : 0.0;
    const double force_error = has_fd ? force_dot_path - fd_dE_ds : 0.0;
    const double gap = 0.5 * ( result.gtilde[0] / std::max( result.area[0], 1.0e-30 ) +
                               result.gtilde[1] / std::max( result.area[1], 1.0e-30 ) );

    out << opts.motion << ',' << step << ',' << s << ',' << normalModeName( params.normal_mode ) << ','
        << penaltyModeName( params.penalty_mode ) << ',' << basisName( params.nodal_energy_basis ) << ','
        << ( params.enzyme_quadrature ? 1 : 0 ) << ',' << ( params.fixed_integration_jacobian ? 1 : 0 ) << ','
        << ( params.projection_smoothing ? 1 : 0 ) << ','
        << ( opts.swap_mortar_nonmortar ? 1 : 0 ) << ','
        << ( params.eta_gap_scaling ? 1 : 0 ) << ','
        << ( params.eta_angle_smoothing ? 1 : 0 ) << ','
        << projectionSmoothingCurveName( params.projection_smoothing_curve ) << ','
        << opts.eta_angle_smoothing_start_angle << ','
        << params.h1_active_set_smoothing_gap << ','
        << params.qp_derivative_blend_min_gap << ',' << params.qp_derivative_blend_max_gap << ','
        << params.qp_derivative_blend_weight << ',' << ( params.qp_derivative_blend_enzyme_gap_weight ? 1 : 0 )
        << ',' << params.residual_gap << ',' << params.k << ',' << params.del
        << ',' << geom.x1[0] << ',' << geom.y1[0] << ',' << geom.x1[1] << ',' << geom.y1[1] << ',';
    writeMaybe( out, geom.x1.size() > 2, geom.x1.size() > 2 ? geom.x1[2] : 0.0 );
    out << ',';
    writeMaybe( out, geom.y1.size() > 2, geom.y1.size() > 2 ? geom.y1[2] : 0.0 );
    out << ',' << geom.x2[0] << ',' << geom.y2[0] << ',' << geom.x2[1] << ',' << geom.y2[1] << ',';
    writeMaybe( out, result.has_energy, result.energy );
    out << ',';
    writeMaybe( out, result.has_pair_force,
                two_element_a_edge ? normalForceProxyFromB( geom, result.global_force )
                                   : normalForceProxy( coord, result.force ) );
    out << ',';
    writeMaybe( out, result.has_pair_force, force_dot_path );
    out << ',';
    writeMaybe( out, has_fd, fd_dE_ds );
    out << ',';
    writeMaybe( out, has_fd && result.has_pair_force, force_error );
    out << ',' << gap << ',' << result.gtilde[0] << ',' << result.gtilde[1] << ',' << result.area[0] << ','
        << result.area[1] << ',' << result.proj[0] << ',' << result.proj[1] << ',' << result.bounds[0] << ','
        << result.bounds[1] << ',' << result.smooth_bounds[0] << ',' << result.smooth_bounds[1] << ','
        << result.qp.qp[0] << ',' << result.qp.qp[1] << ',' << result.qp.qp[2] << ',' << result.qp.w[0] << ','
        << result.qp.w[1] << ',' << result.qp.w[2];
    for ( double value : result.global_force ) {
      out << ',';
      writeMaybe( out, result.has_global_force, value );
    }
    for ( double value : result.force ) {
      out << ',';
      writeMaybe( out, result.has_pair_force, value );
    }
    out << '\n';
  }

  std::cout << "Wrote " << opts.output << " with " << opts.steps << " rows.\n";
  return 0;
}

}  // namespace

int main( int argc, char** argv )
{
  axom::slic::SimpleLogger logger;
  try {
    return run( parseArgs( argc, argv ) );
  } catch ( const std::exception& ex ) {
    std::cerr << "tribol_energy_mortar_sweep: " << ex.what() << '\n';
    return 1;
  }
}
