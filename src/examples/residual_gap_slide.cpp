// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
// SPDX-License-Identifier: (MIT)

#include <fstream>
#include <iomanip>
#include <iostream>

#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/physics/EnergyMortar.hpp"

// Resolve the edge-to-cap handoff as a static sweep so reviewers can inspect energy, force, and tangent continuity.
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  axom::slic::SimpleLogger logger;
  using namespace tribol;
  std::ofstream output( argc > 1 ? argv[1] : "residual_gap_slide.csv" );
  if ( !output ) {
    std::cerr << "Cannot open slide output file\n";
    MPI_Finalize();
    return 1;
  }
  RealT x1[2] = { -1.0, 0.0 };
  RealT y1[2] = { 0.0, 0.0 };
  IndexT conn1[2] = { 0, 1 };
  MeshData source( 0, 1, 2, conn1, LINEAR_EDGE, x1, y1, nullptr, MemorySpace::Host );
  RealT x2[2] = {};
  RealT y2[2] = { -0.1, -0.1 };
  IndexT conn2[2] = { 1, 0 };
  MeshData target( 1, 1, 2, conn2, LINEAR_EDGE, x2, y2, nullptr, MemorySpace::Host );
  ContactParams params;
  params.del = 0.1;
  params.k = 3.0;
  params.residual_gap = 0.2;
  EnergyMortarCalculator calculator( params );
  BallEndpointData endpoints;
  endpoints.weight = { 1.0, 1.0 };
  endpoints.is_open_endpoint = { true, true };
  const double residual[2] = { 0.2, 0.2 };
  const InterfacePair pair( 0, 0 );
  output << "position,edge_energy,cap_energy,total_energy,edge_fx,cap_fx,total_fx,edge_fy,cap_fy,total_fy,kxx,kxy,kyx,"
            "kyy\n";
  output << std::setprecision( 17 );
  for ( int sample = 0; sample <= 2400; ++sample ) {
    const double center = -0.8 + 1.2 * sample / 2400.0;
    x2[0] = center - 0.02;
    x2[1] = center + 0.02;
    target.setPosition( x2, y2, nullptr );
    const auto edge =
        calculator.compute_quadrature_point_penalty_data( pair, source.getView(), target.getView(), residual );
    const auto cap =
        calculator.compute_ball_penalty_data( pair, source.getView(), target.getView(), endpoints, residual );
    double tangent[4] = {};
    for ( int row_component = 0; row_component < 2; ++row_component ) {
      for ( int col_component = 0; col_component < 2; ++col_component ) {
        for ( int row_node : { 2, 3 } ) {
          for ( int col_node : { 2, 3 } ) {
            const int row = 2 * row_node + row_component;
            const int col = 2 * col_node + col_component;
            tangent[2 * row_component + col_component] += edge.stiffness[8 * row + col] + cap.stiffness[12 * row + col];
          }
        }
      }
    }
    const double edge_fx = edge.force[4] + edge.force[6];
    const double cap_fx = cap.force[4] + cap.force[6];
    const double edge_fy = edge.force[5] + edge.force[7];
    const double cap_fy = cap.force[5] + cap.force[7];
    output << center << ',' << edge.energy << ',' << cap.energy << ',' << edge.energy + cap.energy << ',' << edge_fx
           << ',' << cap_fx << ',' << edge_fx + cap_fx << ',' << edge_fy << ',' << cap_fy << ',' << edge_fy + cap_fy;
    for ( double value : tangent ) output << ',' << value;
    output << '\n';
  }
  MPI_Finalize();
}
