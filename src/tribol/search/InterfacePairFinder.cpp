// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/search/InterfacePairFinder.hpp"

#include <memory>
#include <utility>

#include "axom/slic.hpp"

#include "tribol/common/Atomics.hpp"
#include "tribol/common/ExecModel.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/search/BvhSearch.hpp"
#include "tribol/search/GridSearch.hpp"
#include "tribol/utils/Algorithm.hpp"

namespace tribol {

/** @brief Runtime-polymorphic interface used only by the legacy InterfacePairFinder */
class SearchBase {
 public:
  virtual ~SearchBase() = default;
  virtual void findInterfacePairs() = 0;
};

namespace {

/**
 * @brief Apply the legacy geometry checks to candidates returned by a standalone coarse search
 *
 * GridSearch and BvhSearch deliberately know only about spatial bounds. InterfacePairFinder retains existing behavior
 * by applying geomFilter() before it exposes pairs to legacy contact methods.
 */
void FilterCandidatePairs( CouplingScheme& coupling_scheme, ArrayT<InterfacePair>&& candidate_pairs )
{
  const IndexT candidate_count = candidate_pairs.size();
  const int allocator_id = coupling_scheme.getAllocatorId();
  ArrayT<bool> accepted( axom::ArrayOptions::Uninitialized{}, candidate_count, candidate_count, allocator_id );
  ArrayT<IndexT> accepted_count_data( { 0 }, allocator_id );
  auto accepted_view = accepted.view();
  auto accepted_count = accepted_count_data.view();
  auto candidate_view = candidate_pairs.view();
  const auto coupling_scheme_view = coupling_scheme.getView();

  forAllExec(
      coupling_scheme.getExecutionMode(), candidate_count,
      [accepted_view, accepted_count, candidate_view, coupling_scheme_view] TRIBOL_HOST_DEVICE( IndexT candidate ) {
        const auto& pair = candidate_view[candidate];
        accepted_view[candidate] = geomFilter( coupling_scheme_view, pair.m_element_id1, pair.m_element_id2 );
        if ( accepted_view[candidate] ) {
          tribol::atomicInc( accepted_count.data() );
        }
      } );

  ArrayT<IndexT, 1, MemorySpace::Host> accepted_count_host( accepted_count_data );
  auto& interface_pairs = coupling_scheme.getInterfacePairs();
  interface_pairs.resize( accepted_count_host[0] );

  accepted_count_data.fill( 0 );
  auto interface_pairs_view = interface_pairs.view();
  forAllExec(
      coupling_scheme.getExecutionMode(), candidate_count,
      [accepted_view, accepted_count, candidate_view, interface_pairs_view] TRIBOL_HOST_DEVICE( IndexT candidate ) {
        if ( !accepted_view[candidate] ) {
          return;
        }
        const IndexT accepted_index = tribol::atomicInc( accepted_count.data() );
        interface_pairs_view[accepted_index] = candidate_view[candidate];
      } );

  SLIC_INFO( "Coupling scheme has " << interface_pairs.size() << " pairs after filtering " << candidate_count
                                    << " coarse candidates." );
}

/**
 * @brief Adapts a standalone coarse search to InterfacePairFinder's legacy filtered output
 *
 * @tparam Search GridSearch or BvhSearch specialization
 */
template <typename Search>
class FilteredSearch final : public SearchBase {
 public:
  template <typename... Args>
  FilteredSearch( CouplingScheme& coupling_scheme, Args&&... args )
      : coupling_scheme_( coupling_scheme ), search_( std::forward<Args>( args )... )
  {
  }

  void findInterfacePairs() override { FilterCandidatePairs( coupling_scheme_, search_.findCandidatePairs() ); }

 private:
  CouplingScheme& coupling_scheme_;
  Search search_;
};

/**
 * @brief Exhaustively tests every element pair while preserving the legacy memory footprint
 *
 * Unlike GridSearch and BvhSearch, CartesianProduct keeps geomFilter() fused with candidate generation. Materializing
 * every possible InterfacePair before filtering would replace its one-value-per-pair mask with a much larger pair array
 * and can exhaust memory for large meshes.
 */
template <int SpatialDimension>
class CartesianProduct final : public SearchBase {
 public:
  explicit CartesianProduct( CouplingScheme& coupling_scheme ) : coupling_scheme_( coupling_scheme ) {}

  void findInterfacePairs() override
  {
    const auto mesh1 = coupling_scheme_.getMesh1().getView();
    const auto mesh2 = coupling_scheme_.getMesh2().getView();
    const IndexT mesh1_element_count = mesh1.numberOfElements();
    const IndexT mesh2_element_count = mesh2.numberOfElements();

    IndexT maximum_pair_count = mesh1_element_count * mesh2_element_count;
    const bool same_mesh = coupling_scheme_.getMeshId1() == coupling_scheme_.getMeshId2();
    if ( same_mesh ) {
      maximum_pair_count = mesh1_element_count * ( mesh1_element_count + 1 ) / 2;
    }

    ArrayT<bool> accepted( maximum_pair_count, maximum_pair_count, coupling_scheme_.getAllocatorId() );
    ArrayT<IndexT> accepted_count_data( { 0 }, coupling_scheme_.getAllocatorId() );
    bool* accepted_values = accepted.data();
    auto accepted_count = accepted_count_data.view();
    const auto coupling_scheme_view = coupling_scheme_.getView();

    forAllExec( coupling_scheme_.getExecutionMode(), maximum_pair_count,
                [accepted_values, accepted_count, coupling_scheme_view, mesh1_element_count, mesh2_element_count,
                 same_mesh] TRIBOL_HOST_DEVICE( IndexT candidate ) {
                  IndexT mesh1_element = candidate / mesh2_element_count;
                  IndexT mesh2_element = candidate % mesh2_element_count;
                  if ( same_mesh ) {
                    const IndexT row = algorithm::symmMatrixRow( candidate, mesh1_element_count );
                    const IndexT offset = row * ( row + 1 ) / 2;
                    mesh1_element = row;
                    mesh2_element = candidate - offset;
                  }

                  accepted_values[candidate] = geomFilter( coupling_scheme_view, mesh1_element, mesh2_element );
                  if ( accepted_values[candidate] ) {
                    tribol::atomicInc( accepted_count.data() );
                  }
                } );

    ArrayT<IndexT, 1, MemorySpace::Host> accepted_count_host( accepted_count_data );
    auto& interface_pairs = coupling_scheme_.getInterfacePairs();
    interface_pairs.resize( accepted_count_host[0] );

    accepted_count_data.fill( 0 );
    auto interface_pairs_view = interface_pairs.view();
    forAllExec( coupling_scheme_.getExecutionMode(), maximum_pair_count,
                [accepted_values, accepted_count, interface_pairs_view, mesh1_element_count, mesh2_element_count,
                 same_mesh] TRIBOL_HOST_DEVICE( IndexT candidate ) {
                  if ( !accepted_values[candidate] ) {
                    return;
                  }

                  IndexT mesh1_element = candidate / mesh2_element_count;
                  IndexT mesh2_element = candidate % mesh2_element_count;
                  if ( same_mesh ) {
                    const IndexT row = algorithm::symmMatrixRow( candidate, mesh1_element_count );
                    const IndexT offset = row * ( row + 1 ) / 2;
                    mesh1_element = row;
                    mesh2_element = candidate - offset;
                  }

                  const IndexT accepted_index = tribol::atomicInc( accepted_count.data() );
                  interface_pairs_view[accepted_index] = InterfacePair{ mesh1_element, mesh2_element };
                } );

    SLIC_INFO( "Coupling scheme has " << interface_pairs.size() << " pairs out of a maximum possible of "
                                      << maximum_pair_count << "." );
  }

 private:
  CouplingScheme& coupling_scheme_;
};

template <typename Search>
std::unique_ptr<SearchBase> MakeFilteredSearch( CouplingScheme& coupling_scheme )
{
  return std::make_unique<FilteredSearch<Search>>(
      coupling_scheme, coupling_scheme.getMesh1().getView(), coupling_scheme.getMesh2().getView(),
      coupling_scheme.getEffectiveBinningProximityScale(), coupling_scheme.getParameters().residual_gap );
}

}  // namespace

InterfacePairFinder::InterfacePairFinder( CouplingScheme* coupling_scheme ) : m_coupling_scheme( coupling_scheme )
{
  SLIC_ASSERT_MSG( coupling_scheme != nullptr, "Coupling scheme was invalid (null pointer)" );
  const int spatial_dimension = m_coupling_scheme->spatialDimension();
  m_search = nullptr;

  if ( isOnDevice( coupling_scheme->getExecutionMode() ) && coupling_scheme->getBinningMethod() == BINNING_GRID ) {
    SLIC_WARNING_ROOT( "BINNING_GRID is not supported on GPU. Switching to BINNING_BVH." );
    coupling_scheme->setBinningMethod( BINNING_BVH );
  }

  switch ( coupling_scheme->getBinningMethod() ) {
    case BINNING_CARTESIAN_PRODUCT:
      switch ( spatial_dimension ) {
        case 2:
          m_search = std::make_unique<CartesianProduct<2>>( *m_coupling_scheme );
          break;
        case 3:
          m_search = std::make_unique<CartesianProduct<3>>( *m_coupling_scheme );
          break;
        default:
          SLIC_ERROR_ROOT( "Invalid dimension: " << spatial_dimension );
          break;
      }
      break;
    case BINNING_GRID:
      switch ( spatial_dimension ) {
        case 2:
          m_search = MakeFilteredSearch<GridSearch<2>>( *m_coupling_scheme );
          break;
        case 3:
          m_search = MakeFilteredSearch<GridSearch<3>>( *m_coupling_scheme );
          break;
        default:
          SLIC_ERROR_ROOT( "Invalid dimension: " << spatial_dimension );
          break;
      }
      break;
    case BINNING_BVH:
      switch ( spatial_dimension ) {
        case 2:
          switch ( coupling_scheme->getExecutionMode() ) {
            case ExecutionMode::Sequential:
              m_search = MakeFilteredSearch<BvhSearch<2, ExecutionMode::Sequential>>( *m_coupling_scheme );
              break;
#ifdef TRIBOL_USE_OPENMP
            case ExecutionMode::OpenMP:
              m_search = MakeFilteredSearch<BvhSearch<2, ExecutionMode::OpenMP>>( *m_coupling_scheme );
              break;
#endif
#ifdef TRIBOL_USE_CUDA
            case ExecutionMode::Cuda:
              m_search = MakeFilteredSearch<BvhSearch<2, ExecutionMode::Cuda>>( *m_coupling_scheme );
              break;
#endif
#ifdef TRIBOL_USE_HIP
            case ExecutionMode::Hip:
              m_search = MakeFilteredSearch<BvhSearch<2, ExecutionMode::Hip>>( *m_coupling_scheme );
              break;
#endif
            default:
              SLIC_ERROR_ROOT( "Invalid execution mode." );
              break;
          }
          break;
        case 3:
          switch ( coupling_scheme->getExecutionMode() ) {
            case ExecutionMode::Sequential:
              m_search = MakeFilteredSearch<BvhSearch<3, ExecutionMode::Sequential>>( *m_coupling_scheme );
              break;
#ifdef TRIBOL_USE_OPENMP
            case ExecutionMode::OpenMP:
              m_search = MakeFilteredSearch<BvhSearch<3, ExecutionMode::OpenMP>>( *m_coupling_scheme );
              break;
#endif
#ifdef TRIBOL_USE_CUDA
            case ExecutionMode::Cuda:
              m_search = MakeFilteredSearch<BvhSearch<3, ExecutionMode::Cuda>>( *m_coupling_scheme );
              break;
#endif
#ifdef TRIBOL_USE_HIP
            case ExecutionMode::Hip:
              m_search = MakeFilteredSearch<BvhSearch<3, ExecutionMode::Hip>>( *m_coupling_scheme );
              break;
#endif
            default:
              SLIC_ERROR_ROOT( "Invalid execution mode." );
              break;
          }
          break;
        default:
          SLIC_ERROR_ROOT( "Invalid dimension: " << spatial_dimension );
          break;
      }
      break;
    default:
      SLIC_ERROR_ROOT( "Invalid binning method: " << coupling_scheme->getBinningMethod() );
      break;
  }
}

InterfacePairFinder::~InterfacePairFinder() = default;

void InterfacePairFinder::findInterfacePairs()
{
  SLIC_ASSERT( m_search != nullptr );
  SLIC_DEBUG( "Searching for interface pairs" );
  m_search->findInterfacePairs();
  m_coupling_scheme->setBinned( true );
}

}  // namespace tribol
