#ifndef TRIBOL_ADAPTERS_MFEM_MFEMCONTACTCOLLECTION_HPP_
#define TRIBOL_ADAPTERS_MFEM_MFEMCONTACTCOLLECTION_HPP_

#include "tribol/adapters/mfem/MfemContact.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tribol::mfem {

/**
 * Non-owning group of host-native MFEM contacts on one coordinate true-DOF space.
 * Contacts and their shared finite-element space must outlive the collection. All
 * methods that evaluate or update contacts are collective on the mesh communicator.
 */
template <SupportedMethod MethodType, SearchPolicy Search, execution::Policy Execution>
class MfemContactCollection {
 public:
  using ContactType = MfemContact<MethodType, Search, Execution>;

  static_assert( execution::SupportedContactExecution<MethodType, Execution> );
  static_assert( execution::SupportedContactSearch<Search, Execution> );
  static_assert( tribol::detail::is_native_explicit_method_v<MethodType> );
  static_assert( !execution::DevicePolicy<Execution> );

  explicit MfemContactCollection( ContactType& first )
      : coordinate_space_( &first.coordinate_space_ ),
        stiffness_row_sums_( coordinate_space_->GetTrueVSize() ),
        rate_row_sums_( coordinate_space_->GetTrueVSize() )
  {
    contacts_.push_back( &first );
  }

  void add( ContactType& contact )
  {
    if ( &contact.coordinate_space_ != coordinate_space_ ) {
      throw std::invalid_argument( "An MFEM contact collection requires one shared coordinate true-DOF space." );
    }
    if ( std::find( contacts_.begin(), contacts_.end(), &contact ) != contacts_.end() ) {
      throw std::invalid_argument( "An MFEM contact collection cannot contain the same contact object twice." );
    }
    contacts_.push_back( &contact );
  }

  [[nodiscard]] std::size_t size() const { return contacts_.size(); }

  void registerInverseMass( const ::mfem::Vector& parent_true_inverse_mass )
  {
    for ( ContactType* contact : contacts_ ) {
      contact->registerInverseMass( parent_true_inverse_mass );
    }
  }

  void registerInverseMass( const ::mfem::ParGridFunction& inverse_mass )
  {
    if ( inverse_mass.ParFESpace() != coordinate_space_ ) {
      throw std::invalid_argument( "MFEM collection inverse mass must use the shared coordinate space." );
    }
    ::mfem::Vector true_inverse_mass;
    inverse_mass.GetTrueDofs( true_inverse_mass );
    registerInverseMass( true_inverse_mass );
  }

  void registerIntegrator( timestep::IntegratorStabilityLimits integrator )
  {
    for ( ContactType* contact : contacts_ ) {
      contact->registerIntegrator( integrator );
    }
  }

  void registerEssentialTrueDofs( const ::mfem::Array<int>& essential_true_dofs )
  {
    for ( ContactType* contact : contacts_ ) {
      contact->registerEssentialTrueDofs( essential_true_dofs );
    }
  }

  void updateInteractions()
  {
    for ( ContactType* contact : contacts_ ) {
      contact->updateInteractions();
    }
  }

  void updateGeometry( const ::mfem::ParGridFunction& coordinates )
  {
    for ( ContactType* contact : contacts_ ) {
      contact->updateGeometry( coordinates );
    }
  }

  void rebuildGeometry( const ::mfem::ParGridFunction& coordinates )
  {
    for ( ContactType* contact : contacts_ ) {
      contact->rebuildGeometry( coordinates );
    }
  }

  EvaluationSummary addResidual( ::mfem::Vector& parent_true_residual )
  {
    return addResidual( ContactState{}, parent_true_residual );
  }

  EvaluationSummary addResidual( const ContactState& state, ::mfem::Vector& parent_true_residual )
  {
    requireResidual( parent_true_residual );
    requireConsistentStabilityRegistration();
    clearStabilityContributions();
    EvaluationSummary summary;
    for ( ContactType* contact : contacts_ ) {
      accumulateSummary( contact->addResidual( state, parent_true_residual ), summary );
      contact->addNativeStabilityContributions( stiffness_row_sums_, rate_row_sums_ );
    }
    return finalizeStability( summary );
  }

  [[nodiscard]] EvaluationSummary evaluate( const ContactState& state = {} )
  {
    requireConsistentStabilityRegistration();
    clearStabilityContributions();
    EvaluationSummary summary;
    for ( ContactType* contact : contacts_ ) {
      accumulateSummary( contact->evaluate( state ).summary, summary );
      contact->addNativeStabilityContributions( stiffness_row_sums_, rate_row_sums_ );
    }
    return finalizeStability( summary );
  }

  [[nodiscard]] diagnostics::ProductionDiagnosticsPayload productionDiagnostics(
      const ContactState& state = {}, Real time_step = 0.0, diagnostics::QuadratureSamplingOptions sampling = {} )
  {
    const auto communicator = coordinate_space_->GetParMesh()->GetComm();
    const auto communicator_info = detail::communicatorInfo( communicator );
    diagnostics::RankProductionDiagnostics rank_diagnostics{ .rank = communicator_info.rank,
                                                             .ranks = communicator_info.size };
    for ( ContactType* contact : contacts_ ) {
      const auto restricted_state = contact->restrictState( state );
      const auto contribution = contact->core_.productionDiagnostics(
          restricted_state, time_step, communicator_info.rank, communicator_info.size, sampling );
      diagnostics::accumulate( rank_diagnostics.local, contribution.local );
      diagnostics::accumulate( rank_diagnostics.owned, contribution.owned );
    }
    normalizeEmptyMinimumGap( rank_diagnostics.local );
    normalizeEmptyMinimumGap( rank_diagnostics.owned );
    const auto global_diagnostics = detail::reduceOwnedProductionDiagnostics( communicator, rank_diagnostics.owned );
    return diagnostics::makeProductionDiagnosticsPayload( rank_diagnostics, global_diagnostics );
  }

 private:
  static void normalizeEmptyMinimumGap( diagnostics::ProductionDiagnosticSummary& summary )
  {
    if ( summary.rows.rows == 0 ) {
      summary.rows.minimum_gap = 0.0;
    }
  }

  void requireResidual( const ::mfem::Vector& parent_true_residual ) const
  {
    if ( parent_true_residual.Size() != coordinate_space_->GetTrueVSize() ) {
      throw std::invalid_argument( "MFEM collection residual must use the shared coordinate true-DOF space." );
    }
  }

  void clearStabilityContributions()
  {
    stiffness_row_sums_ = 0.0;
    rate_row_sums_ = 0.0;
  }

  static void accumulateSummary( const EvaluationSummary& contribution, EvaluationSummary& summary )
  {
    summary.energy += contribution.energy;
    summary.timestep_vote = std::min( summary.timestep_vote, contribution.timestep_vote );
    summary.active_interactions += contribution.active_interactions;
    summary.quadrature_points += contribution.quadrature_points;
  }

  void requireConsistentStabilityRegistration() const
  {
    const auto& reference = *contacts_.front();
    int local_mismatch{};
    for ( const ContactType* contact : contacts_ ) {
      local_mismatch = local_mismatch || !reference.hasMatchingStabilityRegistration( *contact );
    }
    int global_mismatch{};
    MPI_Allreduce( &local_mismatch, &global_mismatch, 1, MPI_INT, MPI_MAX, coordinate_space_->GetParMesh()->GetComm() );
    if ( global_mismatch != 0 ) {
      throw std::invalid_argument(
          "An MFEM contact collection requires one shared inverse-mass field and matching integrator limits." );
    }
  }

  [[nodiscard]] EvaluationSummary finalizeStability( EvaluationSummary summary ) const
  {
    const auto& reference = *contacts_.front();
    Real local_bounds[2]{};
    for ( int dof = 0; dof < stiffness_row_sums_.Size(); ++dof ) {
      local_bounds[0] = std::max( local_bounds[0], stiffness_row_sums_[dof] );
      local_bounds[1] = std::max( local_bounds[1], rate_row_sums_[dof] );
    }
    Real global_bounds[2]{};
    MPI_Allreduce( local_bounds, global_bounds, 2, detail::realMpiType(), MPI_MAX,
                   coordinate_space_->GetParMesh()->GetComm() );
    summary.timestep_vote =
        std::min( summary.timestep_vote, timestep::explicit_stability_detail::stabilityVote(
                                             global_bounds[0], global_bounds[1], reference.integrator_ ) );
    return summary;
  }

  ::mfem::ParFiniteElementSpace* coordinate_space_{};
  std::vector<ContactType*> contacts_;
  ::mfem::Vector stiffness_row_sums_;
  ::mfem::Vector rate_row_sums_;
};

template <SupportedMethod MethodType, SearchPolicy Search, execution::Policy Execution>
MfemContactCollection( MfemContact<MethodType, Search, Execution>& )
    -> MfemContactCollection<MethodType, Search, Execution>;

}  // namespace tribol::mfem

#endif
