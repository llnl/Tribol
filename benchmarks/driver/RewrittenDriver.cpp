#include "BenchmarkMesh.hpp"
#include "BenchmarkProtocol.hpp"

#include "tribol/Tribol.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#ifndef TRIBOL_BENCHMARK_IMPLEMENTATION_LABEL
#define TRIBOL_BENCHMARK_IMPLEMENTATION_LABEL "rewritten"
#endif

using namespace tribol;
using tribol_benchmark::MeshPairData;
using tribol_benchmark::Options;
using tribol_benchmark::Result;

template <typename Stiffness = stiffness::Constant, typename Rate = rate::None,
          typename Response = response::Frictionless>
using PointwiseMethod =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Centroid, constraint::Pointwise,
           enforcement::Penalty<Stiffness, Rate>, Response, formulation::PointwiseTraction, linearization::Exact>;

#if defined( TRIBOL_BENCHMARK_USE_CUDA )
using BenchmarkSearch = search::Bvh;
using BenchmarkExecution = execution::Cuda;
#else
using BenchmarkSearch = search::Grid;
using BenchmarkExecution = execution::Sequential;
#endif

using ProjectedMultiplier = Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<4>,
                                   constraint::Nodal<basis::Primal>, enforcement::LagrangeMultiplier,
                                   response::Frictionless, formulation::WeightedWeakForm, linearization::Exact>;

using DiagnosticWeights =
    Method<geometry::ProjectedOverlap<normal::MortarSurface>, integration::Polygon<4>, constraint::Nodal<basis::Primal>,
           enforcement::None, response::Frictionless, formulation::DiagnosticWeights, linearization::Exact>;

struct NewMesh {
  MeshPairData<Index> data;
  std::vector<ElementTopology> first_topologies;
  std::vector<ElementTopology> second_topologies;
  std::vector<int> first_attributes;
  std::vector<int> second_attributes;

  NewMesh( MeshPairData<Index> input, ElementTopology topology ) : data( std::move( input ) )
  {
    first_topologies.assign( static_cast<std::size_t>( data.first.elements() ), topology );
    second_topologies.assign( static_cast<std::size_t>( data.second.elements() ), topology );
    first_attributes.assign( first_topologies.size(), 1 );
    second_attributes.assign( second_topologies.size(), 1 );
  }

  SurfaceMeshView surface( const tribol_benchmark::SurfaceData<Index>& source,
                           const std::vector<ElementTopology>& topologies, const std::vector<int>& attributes ) const
  {
    return { .dimension = source.dimension,
             .coordinates = { { source.coordinates.data(), static_cast<Index>( source.coordinates.size() ) },
                              source.nodes(),
                              source.dimension,
                              FieldLayout::Interleaved },
             .element_offsets = { source.offsets.data(), static_cast<Index>( source.offsets.size() ) },
             .connectivity = { source.connectivity.data(), static_cast<Index>( source.connectivity.size() ) },
             .topologies = { topologies.data(), static_cast<Index>( topologies.size() ) },
             .attributes = { attributes.data(), static_cast<Index>( attributes.size() ) } };
  }

  SurfacePairView view() const
  {
    return { surface( data.first, first_topologies, first_attributes ),
             surface( data.second, second_topologies, second_attributes ) };
  }
};

void appendField( std::vector<double>& values, FieldView<const Real> field, Real scale )
{
  for ( Index node = 0; node < field.entities; ++node ) {
    for ( int component = 0; component < field.components; ++component ) {
      values.push_back( scale * field( node, component ) );
    }
  }
}

void addForceDiagnostics( Result& output, FieldView<const Real> first, FieldView<const Real> second, Real scale = 1.0 )
{
  auto& force = output.vectors["nodal_force"];
  appendField( force, first, scale );
  appendField( force, second, scale );
  output.scalars["force_l1"] = tribol_benchmark::vectorL1( force );
  double balance{};
  for ( int component = 0; component < first.components; ++component ) {
    double component_balance{};
    for ( Index node = 0; node < first.entities; ++node ) {
      component_balance += first( node, component );
    }
    for ( Index node = 0; node < second.entities; ++node ) {
      component_balance += second( node, component );
    }
    balance = std::max( balance, std::abs( component_balance ) );
  }
  output.scalars["force_balance_linf"] = balance;
}

template <typename MethodType>
Result runPointwise( const Options& options, int dimension )
{
  NewMesh mesh( tribol_benchmark::makePointwiseMesh<Index>( dimension, options.size ),
                dimension == 2 ? ElementTopology::Segment : ElementTopology::Quadrilateral );
  using ContactType = Contact<MethodType, BenchmarkSearch, BenchmarkExecution>;
  typename ContactType::Options contact_options;
  contact_options.search.expansion = 0.2;
  contact_options.method.enforcement.stiffness.value = 1.0;
  if constexpr ( std::same_as<typename MethodType::enforcement_policy::rate_policy, rate::Percentage> ) {
    contact_options.method.enforcement.rate.ratio = 0.25;
  }
  if constexpr ( std::same_as<typename MethodType::response_policy, response::ViscousTangential> ) {
    contact_options.method.response.damping = 0.5;
  }
  ContactType contact( mesh.view(), contact_options );

  std::vector<Real> first_velocity;
  std::vector<Real> second_velocity;
  ContactStateView state;
  if ( options.case_name == "rate-2d" ) {
    first_velocity.assign( static_cast<std::size_t>( mesh.data.first.nodes() * dimension ), 0.0 );
    second_velocity.assign( first_velocity.size(), 0.0 );
    for ( int node = 0; node < mesh.data.first.nodes(); ++node ) {
      first_velocity[static_cast<std::size_t>( node * dimension + 1 )] = 1.0;
      second_velocity[static_cast<std::size_t>( node * dimension + 1 )] = -1.0;
    }
  } else if ( options.case_name == "viscous-3d" ) {
    first_velocity.assign( static_cast<std::size_t>( mesh.data.first.nodes() * dimension ), 2.0 );
    second_velocity.assign( first_velocity.size(), -2.0 );
  }
  if ( !first_velocity.empty() ) {
    state.mortar_velocity = { { first_velocity.data(), static_cast<Index>( first_velocity.size() ) },
                              mesh.data.first.nodes(),
                              dimension,
                              FieldLayout::Interleaved };
    state.nonmortar_velocity = { { second_velocity.data(), static_cast<Index>( second_velocity.size() ) },
                                 mesh.data.second.nodes(),
                                 dimension,
                                 FieldLayout::Interleaved };
  }

  ContactResultView result;
  Result output( TRIBOL_BENCHMARK_IMPLEMENTATION_LABEL, options );
#if defined( TRIBOL_BENCHMARK_USE_CUDA )
  output.step_seconds = tribol_benchmark::measure( options, [&] {
    contact.updateInteractions();
    const auto device_result = contact.evaluateDevice( state );
    static_cast<void>( device_result );
  } );
  result = contact.evaluate( state );
#else
  output.step_seconds = tribol_benchmark::measure( options, [&] {
    contact.updateInteractions();
    result = contact.evaluate( state );
  } );
#endif
  addForceDiagnostics( output, result.mortar_force, result.nonmortar_force, -1.0 );
  output.scalars["active_interactions"] = result.summary.active_interactions;
  return output;
}

template <typename MethodType>
Result runMortar( const Options& options )
{
  auto legacy_order = tribol_benchmark::makeMortarMesh<Index>( options.size );
  MeshPairData<Index> new_order{ std::move( legacy_order.second ), std::move( legacy_order.first ) };
  NewMesh mesh( std::move( new_order ), ElementTopology::Quadrilateral );
  typename Contact<MethodType, search::Grid>::Options contact_options;
  contact_options.search.expansion = 0.2;
  Contact<MethodType, search::Grid> contact( mesh.view(), contact_options );
  std::vector<Real> multiplier( static_cast<std::size_t>( mesh.data.first.nodes() ), 1.0 );
  const ContactStateView state{ .multiplier = { multiplier.data(), static_cast<Index>( multiplier.size() ) } };
  ContactResultView result;
  Result output( TRIBOL_BENCHMARK_IMPLEMENTATION_LABEL, options );
  output.step_seconds = tribol_benchmark::measure( options, [&] {
    contact.updateInteractions();
    result = contact.evaluate( state );
  } );
  if constexpr ( std::same_as<MethodType, ProjectedMultiplier> ) {
    addForceDiagnostics( output, result.nonmortar_force, result.mortar_force );
    output.vectors["weighted_gap"] = { result.constraint_residual.begin(), result.constraint_residual.end() };
  } else {
    double coupling_measure{};
    auto& coupling_values = output.vectors["coupling_values"];
    const auto append_weights = [&]( const auto weights ) {
      for ( const Real value : weights ) {
        coupling_measure += value;
        if ( value != 0.0 ) {
          coupling_values.push_back( value );
        }
      }
    };
    append_weights( result.mortar_weights );
    append_weights( result.mortar_mass_weights );
    std::sort( coupling_values.begin(), coupling_values.end() );
    output.scalars["coupling_measure"] = 0.5 * coupling_measure;
    output.scalars["coupling_nonzeros"] = coupling_values.size();
  }
  output.scalars["active_interactions"] = result.summary.active_interactions;
  return output;
}

Result runCase( const Options& options )
{
#if defined( TRIBOL_BENCHMARK_USE_CUDA )
  if ( options.case_name == "penalty-2d" ) {
    return runPointwise<DefaultMethod>( options, 2 );
  }
  if ( options.case_name == "penalty-3d" ) {
    return runPointwise<DefaultMethod>( options, 3 );
  }
#else
  if ( options.case_name == "penalty-2d" ) {
    return runPointwise<PointwiseMethod<> >( options, 2 );
  }
  if ( options.case_name == "penalty-3d" ) {
    return runPointwise<PointwiseMethod<> >( options, 3 );
  }
  if ( options.case_name == "rate-2d" ) {
    return runPointwise<PointwiseMethod<stiffness::Constant, rate::Percentage> >( options, 2 );
  }
  if ( options.case_name == "viscous-3d" ) {
    return runPointwise<PointwiseMethod<stiffness::Constant, rate::None, response::ViscousTangential> >( options, 3 );
  }
  if ( options.case_name == "single-mortar-3d" ) {
    return runMortar<ProjectedMultiplier>( options );
  }
  if ( options.case_name == "mortar-weights-3d" ) {
    return runMortar<DiagnosticWeights>( options );
  }
#endif
  throw std::invalid_argument( "unsupported benchmark case: " + options.case_name );
}

}  // namespace

int main( int argc, char** argv )
{
  try {
    const Options options = tribol_benchmark::parseOptions( argc, argv );
    if ( options.list_cases ) {
#if defined( TRIBOL_BENCHMARK_USE_CUDA )
      std::cout << "penalty-2d\npenalty-3d\n";
#else
      std::cout << "penalty-2d\npenalty-3d\nrate-2d\nviscous-3d\nsingle-mortar-3d\nmortar-weights-3d\n";
#endif
      return 0;
    }
    tribol_benchmark::writeResult( std::cout, runCase( options ) );
    return 0;
  } catch ( const std::exception& error ) {
    std::cerr << "tribol_rewritten_benchmark: " << error.what() << '\n';
    return 2;
  }
}
