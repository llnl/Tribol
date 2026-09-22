#include "BenchmarkMesh.hpp"
#include "BenchmarkProtocol.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/config.hpp"
#include "tribol/interface/simple_tribol.hpp"
#include "tribol/interface/tribol.hpp"

#ifdef TRIBOL_USE_MPI
#include <mpi.h>
#endif

#if defined( TRIBOL_BENCHMARK_USE_CUDA )
#include "LegacyCudaBuffers.hpp"

namespace legacy_device = tribol_benchmark::legacy_cuda;

#elif defined( TRIBOL_BENCHMARK_USE_HIP )
#include "LegacyHipBuffers.hpp"

namespace legacy_device = tribol_benchmark::legacy_hip;
#endif

#if defined( TRIBOL_BENCHMARK_USE_CUDA ) || defined( TRIBOL_BENCHMARK_USE_HIP )
#include "mfem.hpp"
#include "umpire/ResourceManager.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tribol_benchmark::MeshPairData;
using tribol_benchmark::Options;
using tribol_benchmark::Result;

struct Session {
  Session() = default;
  Session( const Session& ) = delete;
  Session& operator=( const Session& ) = delete;
  ~Session() { tribol::finalize(); }
};

struct SimpleSession {
  SimpleSession() { Initialize(); }
  SimpleSession( const SimpleSession& ) = delete;
  SimpleSession& operator=( const SimpleSession& ) = delete;
  ~SimpleSession() { Finalize(); }
};

struct MpiSession {
  MpiSession( int& argc, char**& argv )
  {
#ifdef TRIBOL_USE_MPI
    int initialized{};
    MPI_Initialized( &initialized );
    if ( !initialized ) {
      MPI_Init( &argc, &argv );
      owns_mpi = true;
    }
#else
    (void)argc;
    (void)argv;
#endif
  }

  ~MpiSession()
  {
#ifdef TRIBOL_USE_MPI
    if ( owns_mpi ) {
      MPI_Finalize();
    }
#endif
  }

  bool owns_mpi{};
};

struct Components {
  std::vector<tribol::RealT> x;
  std::vector<tribol::RealT> y;
  std::vector<tribol::RealT> z;
};

Components splitCoordinates( const tribol_benchmark::SurfaceData<tribol::IndexT>& surface )
{
  Components result;
  result.x = surface.component( 0 );
  result.y = surface.component( 1 );
  if ( surface.dimension == 3 ) {
    result.z = surface.component( 2 );
  }
  return result;
}

Components globalCoordinates( const tribol_benchmark::SurfaceData<tribol::IndexT>& surface, int global_nodes,
                              int node_offset )
{
  Components result;
  result.x.resize( static_cast<std::size_t>( global_nodes ) );
  result.y.resize( static_cast<std::size_t>( global_nodes ) );
  result.z.resize( static_cast<std::size_t>( global_nodes ) );
  for ( int node = 0; node < surface.nodes(); ++node ) {
    const auto global_node = static_cast<std::size_t>( node + node_offset );
    result.x[global_node] = surface.coordinates[static_cast<std::size_t>( 3 * node )];
    result.y[global_node] = surface.coordinates[static_cast<std::size_t>( 3 * node + 1 )];
    result.z[global_node] = surface.coordinates[static_cast<std::size_t>( 3 * node + 2 )];
  }
  return result;
}

struct Response {
  explicit Response( int nodes ) : x( nodes ), y( nodes ), z( nodes ) {}
  std::vector<tribol::RealT> x;
  std::vector<tribol::RealT> y;
  std::vector<tribol::RealT> z;

  void clear()
  {
    std::fill( x.begin(), x.end(), 0.0 );
    std::fill( y.begin(), y.end(), 0.0 );
    std::fill( z.begin(), z.end(), 0.0 );
  }
};

void registerSurface( int identifier, const tribol_benchmark::SurfaceData<tribol::IndexT>& surface,
                      const Components& coordinates, int cell_type )
{
  tribol::registerMesh( identifier, surface.elements(), surface.nodes(), surface.connectivity.data(), cell_type,
                        coordinates.x.data(), coordinates.y.data(),
                        surface.dimension == 3 ? coordinates.z.data() : nullptr, tribol::MemorySpace::Host );
}

template <typename ResponseType>
void registerResponse( int identifier, ResponseType& response, int dimension )
{
  tribol::registerNodalResponse( identifier, response.x.data(), response.y.data(),
                                 dimension == 3 ? response.z.data() : nullptr );
}

template <typename ResponseType>
void appendResponse( std::vector<double>& output, const ResponseType& response, int dimension )
{
  for ( std::size_t node = 0; node < response.x.size(); ++node ) {
    output.push_back( response.x[node] );
    output.push_back( response.y[node] );
    if ( dimension == 3 ) {
      output.push_back( response.z[node] );
    }
  }
}

template <typename ResponseType>
void addForceDiagnostics( Result& output, const ResponseType& first, const ResponseType& second, int dimension )
{
  auto& force = output.vectors["nodal_force"];
  appendResponse( force, first, dimension );
  appendResponse( force, second, dimension );
  output.scalars["force_l1"] = tribol_benchmark::vectorL1( force );
  double balance{};
  for ( int component = 0; component < dimension; ++component ) {
    const auto& first_values = component == 0 ? first.x : component == 1 ? first.y : first.z;
    const auto& second_values = component == 0 ? second.x : component == 1 ? second.y : second.z;
    double component_balance{};
    for ( const double value : first_values ) {
      component_balance += value;
    }
    for ( const double value : second_values ) {
      component_balance += value;
    }
    balance = std::max( balance, std::abs( component_balance ) );
  }
  output.scalars["force_balance_linf"] = balance;
}

#if defined( TRIBOL_BENCHMARK_USE_CUDA ) || defined( TRIBOL_BENCHMARK_USE_HIP )

using DeviceResponse = legacy_device::Response;

Response copyResponseToHost( const DeviceResponse& source )
{
  Response result( static_cast<int>( source.x.size() ) );
  legacy_device::copyToHost( result.x, source.x );
  legacy_device::copyToHost( result.y, source.y );
  legacy_device::copyToHost( result.z, source.z );
  return result;
}

Result runPointwiseDevice( const Options& options, int dimension )
{
  Session session;
  const int cell_type = dimension == 2 ? tribol::LINEAR_EDGE : tribol::LINEAR_QUAD;
  auto mesh = tribol_benchmark::makePointwiseMesh<tribol::IndexT>( dimension, options.size );
  auto first_connectivity = legacy_device::copyToDevice( mesh.first.connectivity );
  auto second_connectivity = legacy_device::copyToDevice( mesh.second.connectivity );
  const auto first_coordinates = legacy_device::splitCoordinates( mesh.first );
  const auto second_coordinates = legacy_device::splitCoordinates( mesh.second );
  tribol::registerMesh( 0, mesh.first.elements(), mesh.first.nodes(), first_connectivity.data(), cell_type,
                        first_coordinates.x.data(), first_coordinates.y.data(),
                        dimension == 3 ? first_coordinates.z.data() : nullptr, tribol::MemorySpace::Device );
  tribol::registerMesh( 1, mesh.second.elements(), mesh.second.nodes(), second_connectivity.data(), cell_type,
                        second_coordinates.x.data(), second_coordinates.y.data(),
                        dimension == 3 ? second_coordinates.z.data() : nullptr, tribol::MemorySpace::Device );
  DeviceResponse first_response( mesh.first.nodes() );
  DeviceResponse second_response( mesh.second.nodes() );
  registerResponse( 0, first_response, dimension );
  registerResponse( 1, second_response, dimension );
  tribol::setKinematicConstantPenalty( 0, 1.0 );
  tribol::setKinematicConstantPenalty( 1, 1.0 );
  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE,
                                  tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_BVH,
#if defined( TRIBOL_BENCHMARK_USE_HIP )
                                  tribol::ExecutionMode::Hip );
#else
                                  tribol::ExecutionMode::Cuda );
#endif
  tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  tribol::setContactAreaFrac( 0, 1.0e-12 );

  const auto step = [&] {
    first_response.clear();
    second_response.clear();
    tribol::RealT timestep = 1.0;
    if ( tribol::update( 1, 1.0, timestep ) != 0 ) {
      throw std::runtime_error( "legacy Tribol device update failed" );
    }
    legacy_device::synchronize();
  };
#if defined( TRIBOL_BENCHMARK_USE_HIP )
  Result output( "legacy-hip", options );
#else
  Result output( "legacy-cuda", options );
#endif
  output.step_seconds = tribol_benchmark::measure( options, step );
  addForceDiagnostics( output, copyResponseToHost( first_response ), copyResponseToHost( second_response ), dimension );
  output.scalars["active_interactions"] = options.size;
  return output;
}

#endif

Result runPointwise( const Options& options, int dimension )
{
  Session session;
  const int cell_type = dimension == 2 ? tribol::LINEAR_EDGE : tribol::LINEAR_QUAD;
  auto mesh = tribol_benchmark::makePointwiseMesh<tribol::IndexT>( dimension, options.size );
  const Components first_coordinates = splitCoordinates( mesh.first );
  const Components second_coordinates = splitCoordinates( mesh.second );
  registerSurface( 0, mesh.first, first_coordinates, cell_type );
  registerSurface( 1, mesh.second, second_coordinates, cell_type );
  Response first_response( mesh.first.nodes() );
  Response second_response( mesh.second.nodes() );
  registerResponse( 0, first_response, dimension );
  registerResponse( 1, second_response, dimension );

  std::vector<tribol::RealT> first_velocity_x;
  std::vector<tribol::RealT> first_velocity_y;
  std::vector<tribol::RealT> first_velocity_z;
  std::vector<tribol::RealT> second_velocity_x;
  std::vector<tribol::RealT> second_velocity_y;
  std::vector<tribol::RealT> second_velocity_z;
  if ( options.case_name == "rate-2d" ) {
    first_velocity_x.assign( mesh.first.nodes(), 0.0 );
    first_velocity_y.assign( mesh.first.nodes(), 1.0 );
    second_velocity_x.assign( mesh.second.nodes(), 0.0 );
    second_velocity_y.assign( mesh.second.nodes(), -1.0 );
    tribol::registerNodalVelocities( 0, first_velocity_x.data(), first_velocity_y.data() );
    tribol::registerNodalVelocities( 1, second_velocity_x.data(), second_velocity_y.data() );
  } else if ( options.case_name == "viscous-3d" ) {
    first_velocity_x.assign( mesh.first.nodes(), 2.0 );
    first_velocity_y.assign( mesh.first.nodes(), 2.0 );
    first_velocity_z.assign( mesh.first.nodes(), 2.0 );
    second_velocity_x.assign( mesh.second.nodes(), -2.0 );
    second_velocity_y.assign( mesh.second.nodes(), -2.0 );
    second_velocity_z.assign( mesh.second.nodes(), -2.0 );
    tribol::registerNodalVelocities( 0, first_velocity_x.data(), first_velocity_y.data(), first_velocity_z.data() );
    tribol::registerNodalVelocities( 1, second_velocity_x.data(), second_velocity_y.data(), second_velocity_z.data() );
    tribol::setViscousDampingCoeff( 0, 0.5 );
    tribol::setViscousDampingCoeff( 1, 0.5 );
  }

  tribol::setKinematicConstantPenalty( 0, 1.0 );
  tribol::setKinematicConstantPenalty( 1, 1.0 );
  if ( options.case_name == "rate-2d" ) {
    tribol::setRatePercentPenalty( 0, 0.25 );
    tribol::setRatePercentPenalty( 1, 0.25 );
  }
  const int model = options.case_name == "viscous-3d" ? tribol::VISCOUS_TANGENTIAL : tribol::FRICTIONLESS;
  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::COMMON_PLANE, model,
                                  tribol::PENALTY, tribol::BINNING_GRID, tribol::ExecutionMode::Sequential );
  if ( options.case_name == "rate-2d" ) {
    tribol::setPenaltyOptions( 0, tribol::KINEMATIC_AND_RATE, tribol::KINEMATIC_CONSTANT, tribol::RATE_PERCENT );
  } else {
    tribol::setPenaltyOptions( 0, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT );
  }
  tribol::setContactAreaFrac( 0, 1.0e-12 );

  const auto step = [&] {
    first_response.clear();
    second_response.clear();
    tribol::RealT timestep = 1.0;
    if ( tribol::update( 1, 1.0, timestep ) != 0 ) {
      throw std::runtime_error( "legacy Tribol update failed" );
    }
  };
  Result output( "legacy", options );
  output.step_seconds = tribol_benchmark::measure( options, step );
  addForceDiagnostics( output, first_response, second_response, dimension );
  output.scalars["active_interactions"] = options.size;
  return output;
}

Result runMortar( const Options& options, bool weights_only )
{
  auto mesh = tribol_benchmark::makeMortarMesh<tribol::IndexT>( options.size );
  const Components first_coordinates = splitCoordinates( mesh.first );
  const Components second_coordinates = splitCoordinates( mesh.second );
  if ( weights_only ) {
    SimpleSession session;
    const int global_nodes = mesh.first.nodes() + mesh.second.nodes();
    const Components global_first_coordinates = globalCoordinates( mesh.first, global_nodes, 0 );
    const Components global_second_coordinates = globalCoordinates( mesh.second, global_nodes, mesh.first.nodes() );
    auto global_second_connectivity = mesh.second.connectivity;
    for ( auto& node : global_second_connectivity ) {
      node += mesh.first.nodes();
    }
    std::vector<double> gaps( static_cast<std::size_t>( global_nodes ), 0.0 );
    std::vector<double> pressure( gaps.size(), 1.0 );
    SimpleCouplingSetup( 3, tribol::LINEAR_QUAD, tribol::MORTAR_WEIGHTS, mesh.first.elements(), global_nodes,
                         mesh.first.connectivity.data(), global_first_coordinates.x.data(),
                         global_first_coordinates.y.data(), global_first_coordinates.z.data(), mesh.second.elements(),
                         global_nodes, global_second_connectivity.data(), global_second_coordinates.x.data(),
                         global_second_coordinates.y.data(), global_second_coordinates.z.data(), 1.0e-12, gaps.data(),
                         pressure.data() );
    const auto step = [&] {
      double timestep = 1.0;
      if ( Update( timestep ) != 0 ) {
        throw std::runtime_error( "legacy Tribol mortar-weight update failed" );
      }
    };
    Result output( "legacy", options );
    output.step_seconds = tribol_benchmark::measure( options, step );
    output.scalars["active_interactions"] = options.size;
    int* offsets{};
    int* columns{};
    double* values{};
    int offset_count{};
    int nonzeros{};
    if ( GetSimpleCouplingCSR( &offsets, &columns, &values, &offset_count, &nonzeros ) != 0 ) {
      throw std::runtime_error( "legacy Tribol did not expose mortar weights" );
    }
    double measure{};
    auto& coupling_values = output.vectors["coupling_values"];
    coupling_values.reserve( static_cast<std::size_t>( nonzeros ) );
    for ( int entry = 0; entry < nonzeros; ++entry ) {
      measure += values[entry];
      if ( values[entry] != 0.0 ) {
        coupling_values.push_back( values[entry] );
      }
    }
    std::sort( coupling_values.begin(), coupling_values.end() );
    output.scalars["coupling_measure"] = 0.5 * measure;
    output.scalars["coupling_nonzeros"] = coupling_values.size();
    return output;
  }

  Session session;
  registerSurface( 0, mesh.first, first_coordinates, tribol::LINEAR_QUAD );
  registerSurface( 1, mesh.second, second_coordinates, tribol::LINEAR_QUAD );
  std::vector<tribol::RealT> gaps( static_cast<std::size_t>( mesh.second.nodes() ), 0.0 );
  std::vector<tribol::RealT> pressure( gaps.size(), 1.0 );
  Response first_response( mesh.first.nodes() );
  Response second_response( mesh.second.nodes() );
  registerResponse( 0, first_response, 3 );
  registerResponse( 1, second_response, 3 );
  tribol::registerMortarGaps( 1, gaps.data() );
  tribol::registerMortarPressures( 1, pressure.data() );
  tribol::registerCouplingScheme( 0, 0, 1, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE, tribol::SINGLE_MORTAR,
                                  tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID,
                                  tribol::ExecutionMode::Sequential );
  tribol::setLagrangeMultiplierOptions( 0, tribol::ImplicitEvalMode::MORTAR_RESIDUAL,
                                        tribol::SparseMode::MFEM_LINKED_LIST );

  const auto step = [&] {
    first_response.clear();
    second_response.clear();
    std::fill( gaps.begin(), gaps.end(), 0.0 );
    tribol::RealT timestep = 1.0;
    if ( tribol::update( 1, 1.0, timestep ) != 0 ) {
      throw std::runtime_error( "legacy Tribol mortar update failed" );
    }
  };
  Result output( "legacy", options );
  output.step_seconds = tribol_benchmark::measure( options, step );
  output.scalars["active_interactions"] = options.size;
  addForceDiagnostics( output, first_response, second_response, 3 );
  output.vectors["weighted_gap"] = { gaps.begin(), gaps.end() };
  return output;
}

Result runCase( const Options& options )
{
#if defined( TRIBOL_BENCHMARK_USE_CUDA ) || defined( TRIBOL_BENCHMARK_USE_HIP )
  if ( options.case_name == "penalty-2d" ) {
    return runPointwiseDevice( options, 2 );
  }
  if ( options.case_name == "penalty-3d" ) {
    return runPointwiseDevice( options, 3 );
  }
#else
  if ( options.case_name == "penalty-2d" ) {
    return runPointwise( options, 2 );
  }
  if ( options.case_name == "penalty-3d" ) {
    return runPointwise( options, 3 );
  }
  if ( options.case_name == "rate-2d" ) {
    return runPointwise( options, 2 );
  }
  if ( options.case_name == "viscous-3d" ) {
    return runPointwise( options, 3 );
  }
  if ( options.case_name == "single-mortar-3d" ) {
    return runMortar( options, false );
  }
  if ( options.case_name == "mortar-weights-3d" ) {
    return runMortar( options, true );
  }
#endif
  throw std::invalid_argument( "unsupported benchmark case: " + options.case_name );
}

}  // namespace

int main( int argc, char** argv )
{
  try {
    MpiSession mpi( argc, argv );
#if defined( TRIBOL_BENCHMARK_USE_CUDA ) || defined( TRIBOL_BENCHMARK_USE_HIP )
    umpire::ResourceManager::getInstance();
#if defined( TRIBOL_BENCHMARK_USE_HIP )
    mfem::Device device( "hip" );
#else
    mfem::Device device( "cuda" );
#endif
#endif
    const Options options = tribol_benchmark::parseOptions( argc, argv );
    if ( options.list_cases ) {
#if defined( TRIBOL_BENCHMARK_USE_CUDA ) || defined( TRIBOL_BENCHMARK_USE_HIP )
      std::cout << "penalty-2d\npenalty-3d\n";
#else
      std::cout << "penalty-2d\npenalty-3d\nrate-2d\nviscous-3d\nsingle-mortar-3d\nmortar-weights-3d\n";
#endif
      return 0;
    }
    tribol_benchmark::writeResult( std::cout, runCase( options ) );
    return 0;
  } catch ( const std::exception& error ) {
    std::cerr << "tribol_legacy_benchmark: " << error.what() << '\n';
    return 2;
  }
}
