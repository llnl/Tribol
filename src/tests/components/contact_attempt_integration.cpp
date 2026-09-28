#include "tribol/Tribol.hpp"

#include <array>

// Requirements: API-003, API-004, API-006

namespace {

using namespace tribol;

struct SegmentSurface {
  std::array<Real, 4> coordinates{};
  std::array<Index, 2> offsets{ 0, 2 };
  std::array<Index, 2> connectivity{ 0, 1 };
  std::array<ElementTopology, 1> topologies{ ElementTopology::Segment };
  std::array<int, 1> attributes{ 1 };

  [[nodiscard]] SurfaceMeshView view() const
  {
    return {
        .dimension = 2,
        .coordinates = { { coordinates.data(), static_cast<Index>( coordinates.size() ) },
                         2,
                         2,
                         FieldLayout::Interleaved },
        .element_offsets = { offsets.data(), static_cast<Index>( offsets.size() ) },
        .connectivity = { connectivity.data(), static_cast<Index>( connectivity.size() ) },
        .topologies = { topologies.data(), static_cast<Index>( topologies.size() ) },
        .attributes = { attributes.data(), static_cast<Index>( attributes.size() ) },
    };
  }
};

bool hasEveryInvalidation( ContactInvalidation invalidation )
{
  return contains( invalidation, ContactInvalidation::Coordinates ) &&
         contains( invalidation, ContactInvalidation::Topology ) &&
         contains( invalidation, ContactInvalidation::Interactions ) &&
         contains( invalidation, ContactInvalidation::Mass ) &&
         contains( invalidation, ContactInvalidation::BoundaryConditions ) &&
         contains( invalidation, ContactInvalidation::Penalty ) &&
         contains( invalidation, ContactInvalidation::Integrator );
}

bool publicAttemptContract()
{
  SegmentSurface mortar{ .coordinates = { 0.0, 0.0, 1.0, 0.0 } };
  SegmentSurface nonmortar{ .coordinates = { 0.0, 0.1, 1.0, 0.1 } };
  Contact contact( SurfacePairView{ mortar.view(), nonmortar.view() } );

  const auto first = contact.beginAttempt();
  contact.acceptAttempt();
  const auto accepted_checkpoint = contact.checkpoint();
  if ( first.id != 1 || first.retry != 0 || first.status != AttemptStatus::Active ||
       contact.lastAcceptedAttempt().id != first.id || contact.acceptedAttempts() != 1 ||
       contact.rejectedAttempts() != 0 ) {
    return false;
  }

  nonmortar.coordinates[1] = 0.2;
  nonmortar.coordinates[3] = 0.2;
  contact.updateGeometry( { mortar.view(), nonmortar.view() } );
  constexpr std::array<ElementPair, 1> pair{ ElementPair{ 0, 0 } };
  contact.setInteractions( { pair.data(), static_cast<Index>( pair.size() ) } );
  MassVersion mass;
  BoundaryConditionVersion boundary_conditions;
  PenaltyVersion penalty;
  IntegratorVersion integrator;
  mass.advance();
  boundary_conditions.advance();
  penalty.advance();
  integrator.advance();
  contact.setMassGeneration( mass );
  contact.setBoundaryConditionGeneration( boundary_conditions );
  contact.setPenaltyGeneration( penalty );
  contact.setIntegratorGeneration( integrator );

  const auto rejected = contact.beginAttempt();
  const auto changed = contact.invalidationsSinceAcceptedAttempt();
  if ( rejected.id != 2 || rejected.retry != 0 || !invalidatesRows( changed ) || !invalidatesStability( changed ) ||
       contains( changed, ContactInvalidation::Topology ) ) {
    return false;
  }
  contact.rejectAttempt( FailureCategory::Geometry );
  if ( contact.lastAcceptedAttempt().id != first.id || contact.lastRejectedAttempt().id != rejected.id ||
       contact.currentAttempt().failure != FailureCategory::Geometry ) {
    return false;
  }

  contact.rebuildGeometry( { mortar.view(), nonmortar.view() } );
  contact.setInteractions( { pair.data(), static_cast<Index>( pair.size() ) } );
  const auto retry = contact.beginAttempt();
  if ( retry.id != 3 || retry.retry != 1 || !hasEveryInvalidation( contact.invalidationsSinceAcceptedAttempt() ) ) {
    return false;
  }
  contact.rejectAttempt( FailureCategory::Stability );

  contact.restore( accepted_checkpoint );
  if ( contact.currentAttempt().id != first.id || contact.lastRejectedAttempt().status != AttemptStatus::None ||
       contact.acceptedAttempts() != 1 || contact.rejectedAttempts() != 0 ||
       !hasEveryInvalidation( contact.invalidationsSinceAcceptedAttempt() ) ) {
    return false;
  }
  const auto restarted = contact.beginAttempt();
  contact.acceptAttempt();
  return restarted.id == 2 && restarted.retry == 0 &&
         contact.lastAcceptedAttempt().generations == contact.generations();
}

using NativeMethod =
    Method<geometry::ProjectedOverlap<normal::MeanPlane>, integration::Automatic, constraint::QuadraturePoint,
           enforcement::Penalty<>, response::Frictionless, formulation::PointwiseTraction, linearization::Exact>;

bool nativeTopologyGenerationContract()
{
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, 0.1, 1.0, 0.1 };
  constexpr std::array<Real, 4> updated_nonmortar_coordinates{ 0.0, 0.2, 1.0, 0.2 };
  constexpr std::array<Index, 2> connectivity{ 0, 1 };
  array::UniformSurface mortar( 2, ElementTopology::Segment,
                                { mortar_coordinates.data(), static_cast<Index>( mortar_coordinates.size() ) },
                                { connectivity.data(), static_cast<Index>( connectivity.size() ) } );
  array::UniformSurface nonmortar( 2, ElementTopology::Segment,
                                   { nonmortar_coordinates.data(), static_cast<Index>( nonmortar_coordinates.size() ) },
                                   { connectivity.data(), static_cast<Index>( connectivity.size() ) } );
  Contact<NativeMethod> contact(
      { mortar.parentTraceView(), nonmortar.parentTraceView() },
      { .mortar_coordinates = mortar.view().coordinates, .nonmortar_coordinates = nonmortar.view().coordinates } );
  const auto initial = contact.generations();
  nonmortar.updateCoordinates(
      { updated_nonmortar_coordinates.data(), static_cast<Index>( updated_nonmortar_coordinates.size() ) } );
  const ParentTracePairView traces{ mortar.parentTraceView(), nonmortar.parentTraceView() };
  const ParentTraceFieldPairView fields{ .mortar_coordinates = mortar.view().coordinates,
                                         .nonmortar_coordinates = nonmortar.view().coordinates };
  contact.updateGeometry( traces, fields );
  const auto updated = contact.generations();
  contact.rebuildGeometry( traces, fields );
  const auto rebuilt = contact.generations();
  return updated.coordinates != initial.coordinates && updated.topology == initial.topology &&
         rebuilt.coordinates != updated.coordinates && rebuilt.topology != updated.topology;
}

}  // namespace

int main() { return publicAttemptContract() && nativeTopologyGenerationContract() ? 0 : 1; }
