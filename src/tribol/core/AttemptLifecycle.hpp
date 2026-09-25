#ifndef TRIBOL_CORE_ATTEMPTLIFECYCLE_HPP_
#define TRIBOL_CORE_ATTEMPTLIFECYCLE_HPP_

#include "tribol/core/Version.hpp"

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace tribol {

enum class AttemptStatus : std::uint8_t
{
  None,
  Active,
  Accepted,
  Rejected
};

enum class FailureCategory : std::uint8_t
{
  None,
  InvalidInput,
  Search,
  Geometry,
  Kinematic,
  Stability,
  NonfiniteState,
  HostApplication
};

struct ContactGenerations {
  GeometryVersion coordinates{};
  TopologyVersion topology{};
  InteractionVersion interactions{};
  MassVersion mass{};
  BoundaryConditionVersion boundary_conditions{};
  PenaltyVersion penalty{};
  IntegratorVersion integrator{};

  friend constexpr bool operator==( ContactGenerations, ContactGenerations ) = default;
};

enum class ContactInvalidation : std::uint32_t
{
  None = 0,
  Coordinates = 1U << 0U,
  Topology = 1U << 1U,
  Interactions = 1U << 2U,
  Mass = 1U << 3U,
  BoundaryConditions = 1U << 4U,
  Penalty = 1U << 5U,
  Integrator = 1U << 6U
};

[[nodiscard]] constexpr ContactInvalidation operator|( ContactInvalidation left, ContactInvalidation right )
{
  return static_cast<ContactInvalidation>( static_cast<std::uint32_t>( left ) | static_cast<std::uint32_t>( right ) );
}

[[nodiscard]] constexpr bool contains( ContactInvalidation set, ContactInvalidation value )
{
  return ( static_cast<std::uint32_t>( set ) & static_cast<std::uint32_t>( value ) ) != 0U;
}

[[nodiscard]] constexpr ContactInvalidation invalidationsSince( ContactGenerations previous,
                                                                ContactGenerations current )
{
  ContactInvalidation result = ContactInvalidation::None;
  if ( previous.coordinates != current.coordinates ) {
    result = result | ContactInvalidation::Coordinates;
  }
  if ( previous.topology != current.topology ) {
    result = result | ContactInvalidation::Topology;
  }
  if ( previous.interactions != current.interactions ) {
    result = result | ContactInvalidation::Interactions;
  }
  if ( previous.mass != current.mass ) {
    result = result | ContactInvalidation::Mass;
  }
  if ( previous.boundary_conditions != current.boundary_conditions ) {
    result = result | ContactInvalidation::BoundaryConditions;
  }
  if ( previous.penalty != current.penalty ) {
    result = result | ContactInvalidation::Penalty;
  }
  if ( previous.integrator != current.integrator ) {
    result = result | ContactInvalidation::Integrator;
  }
  return result;
}

[[nodiscard]] constexpr bool invalidatesRows( ContactInvalidation changes )
{
  return contains( changes, ContactInvalidation::Coordinates ) || contains( changes, ContactInvalidation::Topology ) ||
         contains( changes, ContactInvalidation::Interactions );
}

[[nodiscard]] constexpr bool invalidatesStability( ContactInvalidation changes )
{
  return invalidatesRows( changes ) || contains( changes, ContactInvalidation::Mass ) ||
         contains( changes, ContactInvalidation::BoundaryConditions ) ||
         contains( changes, ContactInvalidation::Penalty ) || contains( changes, ContactInvalidation::Integrator );
}

struct ContactAttemptRecord {
  std::uint64_t id{};
  std::uint64_t retry{};
  AttemptStatus status{ AttemptStatus::None };
  FailureCategory failure{ FailureCategory::None };
  ContactGenerations generations{};
};

struct ContactAttemptCheckpoint {
  ContactAttemptRecord accepted{};
  std::uint64_t next_attempt_id{ 1 };
  std::uint64_t accepted_attempts{};
  std::uint64_t rejected_attempts{};
};

class ContactAttemptLifecycle {
 public:
  [[nodiscard]] ContactAttemptRecord begin( ContactGenerations generations )
  {
    if ( current_.status == AttemptStatus::Active ) {
      throw std::logic_error( "A contact attempt is already active." );
    }
    current_ = { .id = next_attempt_id_++,
                 .retry = next_retry_,
                 .status = AttemptStatus::Active,
                 .failure = FailureCategory::None,
                 .generations = generations };
    return current_;
  }

  void accept()
  {
    requireActive();
    current_.status = AttemptStatus::Accepted;
    current_.failure = FailureCategory::None;
    accepted_ = current_;
    ++accepted_attempts_;
    next_retry_ = 0;
  }

  void reject( FailureCategory failure )
  {
    requireActive();
    if ( failure == FailureCategory::None ) {
      throw std::invalid_argument( "A rejected contact attempt requires a failure category." );
    }
    current_.status = AttemptStatus::Rejected;
    current_.failure = failure;
    rejected_ = current_;
    ++rejected_attempts_;
    next_retry_ = current_.retry + 1;
  }

  [[nodiscard]] ContactAttemptCheckpoint checkpoint() const
  {
    if ( current_.status == AttemptStatus::Active ) {
      throw std::logic_error( "An active contact attempt cannot be checkpointed." );
    }
    return { .accepted = accepted_,
             .next_attempt_id = next_attempt_id_,
             .accepted_attempts = accepted_attempts_,
             .rejected_attempts = rejected_attempts_ };
  }

  void restore( ContactAttemptCheckpoint checkpoint )
  {
    if ( current_.status == AttemptStatus::Active ) {
      throw std::logic_error( "An active contact attempt cannot be replaced by a checkpoint." );
    }
    if ( checkpoint.accepted.status != AttemptStatus::None && checkpoint.accepted.status != AttemptStatus::Accepted ) {
      throw std::invalid_argument( "A contact checkpoint may contain only accepted state." );
    }
    current_ = checkpoint.accepted;
    accepted_ = checkpoint.accepted;
    rejected_ = {};
    next_attempt_id_ = checkpoint.next_attempt_id;
    accepted_attempts_ = checkpoint.accepted_attempts;
    rejected_attempts_ = checkpoint.rejected_attempts;
    next_retry_ = 0;
  }

  [[nodiscard]] ContactAttemptRecord current() const { return current_; }
  [[nodiscard]] ContactAttemptRecord lastAccepted() const { return accepted_; }
  [[nodiscard]] ContactAttemptRecord lastRejected() const { return rejected_; }
  [[nodiscard]] std::uint64_t acceptedAttempts() const { return accepted_attempts_; }
  [[nodiscard]] std::uint64_t rejectedAttempts() const { return rejected_attempts_; }

 private:
  void requireActive() const
  {
    if ( current_.status != AttemptStatus::Active ) {
      throw std::logic_error( "Contact attempt completion requires an active attempt." );
    }
  }

  ContactAttemptRecord current_{};
  ContactAttemptRecord accepted_{};
  ContactAttemptRecord rejected_{};
  std::uint64_t next_attempt_id_{ 1 };
  std::uint64_t next_retry_{};
  std::uint64_t accepted_attempts_{};
  std::uint64_t rejected_attempts_{};
};

static_assert( std::is_trivially_copyable_v<ContactGenerations> );
static_assert( std::is_trivially_copyable_v<ContactAttemptRecord> );
static_assert( std::is_trivially_copyable_v<ContactAttemptCheckpoint> );

}  // namespace tribol

#endif
