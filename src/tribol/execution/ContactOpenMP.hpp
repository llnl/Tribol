#ifndef TRIBOL_EXECUTION_CONTACTOPENMP_HPP_
#define TRIBOL_EXECUTION_CONTACTOPENMP_HPP_

#include "tribol/evaluation/PolicyEvaluator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined( _OPENMP )
#include <omp.h>
#endif

namespace tribol::execution {

namespace openmp_detail {

inline int maximumThreads()
{
#if defined( _OPENMP )
  return omp_get_max_threads();
#else
  return 1;
#endif
}

struct EvaluationStorage {
  std::vector<Real> mortar_force;
  std::vector<Real> nonmortar_force;
  std::vector<Real> constraint;
  std::vector<Real> gap;
  std::vector<Real> weighted_gap;
  std::vector<Real> tributary_area;
  std::vector<Real> mortar_weights;
  std::vector<Real> mortar_mass_weights;
  std::vector<Real> quadrature_gap;
  std::vector<Real> quadrature_pressure;
  std::vector<Real> pressure;
  EvaluationSummary summary{};

  void resize( const SurfacePairView& surfaces, Index maximum_quadrature_points, bool diagnostic_weights )
  {
    const auto mortar_nodes = static_cast<std::size_t>( surfaces.mortar.numberOfNodes() );
    const auto nonmortar_nodes = static_cast<std::size_t>( surfaces.nonmortar.numberOfNodes() );
    mortar_force.resize( mortar_nodes * static_cast<std::size_t>( surfaces.mortar.dimension ) );
    nonmortar_force.resize( nonmortar_nodes * static_cast<std::size_t>( surfaces.nonmortar.dimension ) );
    constraint.resize( mortar_nodes );
    gap.resize( mortar_nodes );
    weighted_gap.resize( mortar_nodes );
    tributary_area.resize( mortar_nodes );
    mortar_weights.resize( diagnostic_weights ? mortar_nodes * nonmortar_nodes : 0 );
    mortar_mass_weights.resize( diagnostic_weights ? mortar_nodes * mortar_nodes : 0 );
    quadrature_gap.resize( static_cast<std::size_t>( maximum_quadrature_points ) );
    quadrature_pressure.resize( static_cast<std::size_t>( maximum_quadrature_points ) );
    pressure.resize( mortar_nodes );
  }

  void clear()
  {
    auto zero = []( std::vector<Real>& values ) { std::fill( values.begin(), values.end(), 0.0 ); };
    zero( mortar_force );
    zero( nonmortar_force );
    zero( constraint );
    zero( gap );
    zero( weighted_gap );
    zero( tributary_area );
    zero( mortar_weights );
    zero( mortar_mass_weights );
    zero( quadrature_gap );
    zero( quadrature_pressure );
    zero( pressure );
    summary = {};
  }

  [[nodiscard]] ContactOutputView view( const SurfacePairView& surfaces )
  {
    return {
        .residual = { .mortar = { { mortar_force.data(), static_cast<Index>( mortar_force.size() ) },
                                  surfaces.mortar.numberOfNodes(),
                                  surfaces.mortar.dimension,
                                  FieldLayout::Interleaved },
                      .nonmortar = { { nonmortar_force.data(), static_cast<Index>( nonmortar_force.size() ) },
                                     surfaces.nonmortar.numberOfNodes(),
                                     surfaces.nonmortar.dimension,
                                     FieldLayout::Interleaved },
                      .constraint = { constraint.data(), static_cast<Index>( constraint.size() ) } },
        .gap = { gap.data(), static_cast<Index>( gap.size() ) },
        .weighted_gap = { weighted_gap.data(), static_cast<Index>( weighted_gap.size() ) },
        .tributary_area = { tributary_area.data(), static_cast<Index>( tributary_area.size() ) },
        .mortar_weights = { mortar_weights.data(), static_cast<Index>( mortar_weights.size() ) },
        .mortar_mass_weights = { mortar_mass_weights.data(), static_cast<Index>( mortar_mass_weights.size() ) },
        .quadrature_gap = { quadrature_gap.data(), static_cast<Index>( quadrature_gap.size() ) },
        .quadrature_pressure = { quadrature_pressure.data(), static_cast<Index>( quadrature_pressure.size() ) },
        .pressure = { pressure.data(), static_cast<Index>( pressure.size() ) },
    };
  }
};

inline void addValues( ArrayView<Real> destination, const std::vector<Real>& source )
{
  if ( destination.empty() ) {
    return;
  }
  for ( Index entry = 0; entry < destination.size(); ++entry ) {
    destination[entry] += source[static_cast<std::size_t>( entry )];
  }
}

inline void addField( FieldView<Real> destination, const std::vector<Real>& source )
{
  addValues( destination.values, source );
}

}  // namespace openmp_detail

class OpenMPContactWorkspace {
 public:
  void reserve( const SurfacePairView& surfaces, Index maximum_quadrature_points, bool diagnostic_weights )
  {
    storage_.resize( static_cast<std::size_t>( openmp_detail::maximumThreads() ) );
    for ( auto& storage : storage_ ) {
      storage.resize( surfaces, maximum_quadrature_points, diagnostic_weights );
    }
  }

  [[nodiscard]] int size() const { return static_cast<int>( storage_.size() ); }
  [[nodiscard]] openmp_detail::EvaluationStorage& operator[]( int thread )
  {
    return storage_[static_cast<std::size_t>( thread )];
  }
  [[nodiscard]] const openmp_detail::EvaluationStorage& operator[]( int thread ) const
  {
    return storage_[static_cast<std::size_t>( thread )];
  }

 private:
  std::vector<openmp_detail::EvaluationStorage> storage_;
};

template <SupportedMethod MethodType>
void evaluateOpenMPNodalKinematics( const SurfacePairView& surfaces, ArrayView<const ElementPair> interactions,
                                    const typename MethodType::Parameters& parameters, ContactOutputView output,
                                    OpenMPContactWorkspace& workspace )
{
  if ( workspace.size() < openmp_detail::maximumThreads() ) {
    throw std::logic_error( "OpenMP contact workspace must be reserved after changing the thread count." );
  }
  for ( int thread = 0; thread < workspace.size(); ++thread ) {
    workspace[thread].clear();
  }

#if defined( _OPENMP )
#pragma omp parallel
#endif
  {
    int thread{};
    int threads{ 1 };
#if defined( _OPENMP )
    thread = omp_get_thread_num();
    threads = omp_get_num_threads();
#endif
    const Index begin = interactions.size() * thread / threads;
    const Index end = interactions.size() * ( thread + 1 ) / threads;
    auto local_output = workspace[thread].view( surfaces );
    local_output.residual = {};
    local_output.gap = {};
    local_output.quadrature_gap = {};
    local_output.quadrature_pressure = {};
    local_output.pressure = {};
    stageVariationalKinematics<MethodType>( surfaces, { interactions.data() + begin, end - begin }, parameters, {},
                                            local_output );
  }

  for ( int thread = 0; thread < workspace.size(); ++thread ) {
    const auto& local = workspace[thread];
    openmp_detail::addValues( output.weighted_gap, local.weighted_gap );
    openmp_detail::addValues( output.tributary_area, local.tributary_area );
  }
  policy_evaluator_detail::finalizeGap( output );
}

template <SupportedMethod MethodType>
EvaluationSummary evaluateOpenMPVariationalContact( const SurfacePairView& surfaces,
                                                    ArrayView<const ElementPair> interactions,
                                                    const typename MethodType::Parameters& parameters,
                                                    const ContactStateView& state, ContactOutputView output,
                                                    OpenMPContactWorkspace& workspace )
{
  validateVariationalEvaluation<MethodType>( surfaces, state, output );
  if ( workspace.size() < openmp_detail::maximumThreads() ) {
    throw std::logic_error( "OpenMP contact workspace must be reserved after changing the thread count." );
  }
  for ( int thread = 0; thread < workspace.size(); ++thread ) {
    workspace[thread].clear();
  }
  if ( interactions.empty() ) {
    return { .timestep_vote = std::numeric_limits<Real>::infinity() };
  }

#if defined( _OPENMP )
#pragma omp parallel
#endif
  {
    int thread{};
    int threads{ 1 };
#if defined( _OPENMP )
    thread = omp_get_thread_num();
    threads = omp_get_num_threads();
#endif
    const Index begin = interactions.size() * thread / threads;
    const Index end = interactions.size() * ( thread + 1 ) / threads;
    workspace[thread].summary = stageVariationalKinematics<MethodType>(
        surfaces, { interactions.data() + begin, end - begin }, parameters, state, workspace[thread].view( surfaces ) );
  }

  EvaluationSummary summary{ .timestep_vote = std::numeric_limits<Real>::infinity() };
  Index quadrature_offset{};
  for ( int thread = 0; thread < workspace.size(); ++thread ) {
    const auto& local = workspace[thread];
    openmp_detail::addValues( output.weighted_gap, local.weighted_gap );
    openmp_detail::addValues( output.tributary_area, local.tributary_area );
    openmp_detail::addValues( output.pressure, local.pressure );
    for ( Index point = 0; point < local.summary.quadrature_points; ++point ) {
      if ( !output.quadrature_gap.empty() ) {
        output.quadrature_gap[quadrature_offset] = local.quadrature_gap[static_cast<std::size_t>( point )];
      }
      if ( !output.quadrature_pressure.empty() ) {
        output.quadrature_pressure[quadrature_offset] = local.quadrature_pressure[static_cast<std::size_t>( point )];
      }
      ++quadrature_offset;
    }
    summary.energy += local.summary.energy;
    summary.active_interactions += local.summary.active_interactions;
    summary.quadrature_points += local.summary.quadrature_points;
  }
  finalizeVariationalKinematics<MethodType>( surfaces, parameters, state, output, summary );

#if defined( _OPENMP )
#pragma omp parallel
#endif
  {
    int thread{};
    int threads{ 1 };
#if defined( _OPENMP )
    thread = omp_get_thread_num();
    threads = omp_get_num_threads();
#endif
    const Index begin = interactions.size() * thread / threads;
    const Index end = interactions.size() * ( thread + 1 ) / threads;
    auto local_output = output;
    local_output.residual = workspace[thread].view( surfaces ).residual;
    addVariationalForces<MethodType>( surfaces, { interactions.data() + begin, end - begin }, parameters, state,
                                      local_output );
  }
  for ( int thread = 0; thread < workspace.size(); ++thread ) {
    openmp_detail::addField( output.residual.mortar, workspace[thread].mortar_force );
    openmp_detail::addField( output.residual.nonmortar, workspace[thread].nonmortar_force );
  }
  finalizeVariationalTimestep<MethodType>( parameters, summary );
  return summary;
}

template <SupportedMethod MethodType>
EvaluationSummary evaluateOpenMPContact( const SurfacePairView& surfaces, ArrayView<const ElementPair> interactions,
                                         const typename MethodType::Parameters& parameters,
                                         const ContactStateView& state, ContactOutputView output,
                                         OpenMPContactWorkspace& workspace )
{
  if constexpr ( std::same_as<typename MethodType::formulation_policy, formulation::Variational> ) {
    return evaluateOpenMPVariationalContact<MethodType>( surfaces, interactions, parameters, state, output, workspace );
  }
  if ( workspace.size() < openmp_detail::maximumThreads() ) {
    throw std::logic_error( "OpenMP contact workspace must be reserved after changing the thread count." );
  }
  for ( int thread = 0; thread < workspace.size(); ++thread ) {
    workspace[thread].clear();
  }
  if ( interactions.empty() ) {
    return { .timestep_vote = std::numeric_limits<Real>::infinity() };
  }

#if defined( _OPENMP )
#pragma omp parallel
#endif
  {
    int thread{};
    int threads{ 1 };
#if defined( _OPENMP )
    thread = omp_get_thread_num();
    threads = omp_get_num_threads();
#endif
    const Index begin = interactions.size() * thread / threads;
    const Index end = interactions.size() * ( thread + 1 ) / threads;
    workspace[thread].summary = evaluateMethod<MethodType>( surfaces, { interactions.data() + begin, end - begin },
                                                            parameters, state, workspace[thread].view( surfaces ) );
  }

  EvaluationSummary summary{ .timestep_vote = std::numeric_limits<Real>::infinity() };
  Index quadrature_offset{};
  for ( int thread = 0; thread < workspace.size(); ++thread ) {
    const auto& local = workspace[thread];
    openmp_detail::addField( output.residual.mortar, local.mortar_force );
    openmp_detail::addField( output.residual.nonmortar, local.nonmortar_force );
    openmp_detail::addValues( output.residual.constraint, local.constraint );
    openmp_detail::addValues( output.weighted_gap, local.weighted_gap );
    openmp_detail::addValues( output.tributary_area, local.tributary_area );
    openmp_detail::addValues( output.mortar_weights, local.mortar_weights );
    openmp_detail::addValues( output.mortar_mass_weights, local.mortar_mass_weights );
    for ( Index point = 0; point < local.summary.quadrature_points; ++point ) {
      if ( !output.quadrature_gap.empty() ) {
        output.quadrature_gap[quadrature_offset] = local.quadrature_gap[static_cast<std::size_t>( point )];
      }
      if ( !output.quadrature_pressure.empty() ) {
        output.quadrature_pressure[quadrature_offset] = local.quadrature_pressure[static_cast<std::size_t>( point )];
      }
      ++quadrature_offset;
    }
    for ( Index node = 0; node < output.pressure.size(); ++node ) {
      output.pressure[node] +=
          local.pressure[static_cast<std::size_t>( node )] * local.tributary_area[static_cast<std::size_t>( node )];
    }
    summary.energy += local.summary.energy;
    summary.timestep_vote = std::min( summary.timestep_vote, local.summary.timestep_vote );
    summary.active_interactions += local.summary.active_interactions;
    summary.quadrature_points += local.summary.quadrature_points;
  }
  for ( Index node = 0; node < output.gap.size(); ++node ) {
    if ( std::abs( output.tributary_area[node] ) > 1.0e-28 ) {
      output.gap[node] = output.weighted_gap[node] / output.tributary_area[node];
    }
  }
  for ( Index node = 0; node < output.pressure.size(); ++node ) {
    if ( std::abs( output.tributary_area[node] ) > 1.0e-28 ) {
      output.pressure[node] /= output.tributary_area[node];
    }
  }
  return summary;
}

}  // namespace tribol::execution

#endif
