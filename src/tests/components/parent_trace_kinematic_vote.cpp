#include "tribol/timestep/KinematicVote.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

// Requirements: TIME-001

namespace {

using namespace tribol;

struct QuadraticSegmentTrace {
  std::array<Real, 4> coordinates{};
  std::array<Index, 2> element_offsets{ 0, 2 };
  std::array<Index, 2> connectivity{};
  std::array<ElementTopology, 1> geometry_topologies{ ElementTopology::Segment };
  std::array<int, 1> geometry_attributes{ 1 };
  std::array<std::uint64_t, 1> parent_face_ids{};
  std::array<int, 1> parent_face_owners{ 0 };
  std::array<ElementTopology, 1> parent_face_topologies{ ElementTopology::Segment };
  std::array<int, 1> parent_face_orders{ 2 };
  std::array<Index, 2> parent_dof_offsets{ 0, 3 };
  std::array<Index, 3> parent_dofs{ 0, 1, 2 };
  std::array<Real, 3> parent_dof_signs{ 1.0, 1.0, 1.0 };
  std::array<Index, 2> parent_basis_offsets{ 0, 9 };
  std::array<Real, 9> parent_basis_coefficients{ 1.0, -3.0, 2.0, 0.0, 4.0, -4.0, 0.0, -1.0, 2.0 };
  std::array<Index, 1> subfacet_parent_faces{ 0 };
  std::array<Real, 2> subfacet_parent_reference_coordinates{};

  QuadraticSegmentTrace( Real height, bool reverse, std::uint64_t face_id )
      : coordinates{ 0.0, height, 1.0, height },
        connectivity( reverse ? std::array<Index, 2>{ 1, 0 } : std::array<Index, 2>{ 0, 1 } ),
        parent_face_ids{ face_id },
        subfacet_parent_reference_coordinates( reverse ? std::array<Real, 2>{ 1.0, 0.0 }
                                                       : std::array<Real, 2>{ 0.0, 1.0 } )
  {
  }

  [[nodiscard]] ParentTraceView view() const
  {
    return {
        .geometry =
            {
                .dimension = 2,
                .coordinates = { { coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
                .element_offsets = { element_offsets.data(), 2 },
                .connectivity = { connectivity.data(), 2 },
                .topologies = { geometry_topologies.data(), 1 },
                .attributes = { geometry_attributes.data(), 1 },
            },
        .parent_face_ids = { parent_face_ids.data(), 1 },
        .parent_face_owners = { parent_face_owners.data(), 1 },
        .parent_face_topologies = { parent_face_topologies.data(), 1 },
        .parent_face_orders = { parent_face_orders.data(), 1 },
        .parent_dof_offsets = { parent_dof_offsets.data(), 2 },
        .parent_dofs = { parent_dofs.data(), 3 },
        .parent_dof_signs = { parent_dof_signs.data(), 3 },
        .parent_basis_offsets = { parent_basis_offsets.data(), 2 },
        .parent_basis_coefficients = { parent_basis_coefficients.data(), 9 },
        .subfacet_parent_faces = { subfacet_parent_faces.data(), 1 },
        .subfacet_parent_reference_coordinates = { { subfacet_parent_reference_coordinates.data(), 2 },
                                                   2,
                                                   1,
                                                   FieldLayout::Interleaved },
    };
  }
};

[[nodiscard]] FieldView<const Real> vectorField( const std::array<Real, 6>& values )
{
  return { { values.data(), 6 }, 3, 2, FieldLayout::Interleaved };
}

Real parentTraceVote( Real penetration, Real mortar_speed, Real nonmortar_speed )
{
  const QuadraticSegmentTrace mortar( penetration, true, 17 );
  const QuadraticSegmentTrace nonmortar( 0.0, false, 23 );
  const ParentTracePairView traces{ mortar.view(), nonmortar.view() };
  constexpr std::array<ElementPair, 1> interactions{ ElementPair{ 0, 0 } };
  const std::array<Real, 6> mortar_velocity{ 0.0, 0.2 * mortar_speed, 0.0, mortar_speed, 0.0, 0.2 * mortar_speed };
  const std::array<Real, 6> nonmortar_velocity{ 0.0, 0.2 * nonmortar_speed, 0.0, nonmortar_speed,
                                                0.0, 0.2 * nonmortar_speed };
  constexpr std::array<Real, 1> thickness{ 0.1 };
  const ContactStateView state{
      .mortar_velocity = vectorField( mortar_velocity ),
      .nonmortar_velocity = vectorField( nonmortar_velocity ),
      .mortar_element_thickness = { thickness.data(), 1 },
      .nonmortar_element_thickness = { thickness.data(), 1 },
  };
  const DefaultMethod::Parameters method_parameters;
  const timestep::Kinematic::Parameters parameters{
      .enabled = true,
      .current_step = 1.0,
      .penetration_fraction = 0.3,
      .velocity_tolerance = 0.0,
  };
  return timestep::kinematicVote<DefaultMethod>( traces, { interactions.data(), 1 }, method_parameters, state,
                                                 parameters );
}

bool parentInterpolationAndMotionContract()
{
  const Real existing_penetration = parentTraceVote( 0.1, 0.5, -0.5 );
  const Real projected_penetration = parentTraceVote( 0.01, 0.5, -0.5 );
  const Real opening = parentTraceVote( 0.01, -0.5, 0.5 );
  return std::abs( existing_penetration - 0.06 ) < 1.0e-12 && std::abs( projected_penetration - 0.06 ) < 1.0e-12 &&
         std::abs( opening - 1.0 ) < 1.0e-12;
}

bool invalidParentVelocityIsRejected()
{
  const QuadraticSegmentTrace mortar( 0.1, true, 17 );
  const QuadraticSegmentTrace nonmortar( 0.0, false, 23 );
  const ParentTracePairView traces{ mortar.view(), nonmortar.view() };
  constexpr std::array<ElementPair, 1> interactions{ ElementPair{ 0, 0 } };
  constexpr std::array<Real, 4> velocity{};
  constexpr std::array<Real, 1> thickness{ 0.1 };
  const ContactStateView state{
      .mortar_velocity = { { velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .nonmortar_velocity = { { velocity.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .mortar_element_thickness = { thickness.data(), 1 },
      .nonmortar_element_thickness = { thickness.data(), 1 },
  };
  try {
    static_cast<void>(
        timestep::kinematicVote<DefaultMethod>( traces, { interactions.data(), 1 }, DefaultMethod::Parameters{}, state,
                                                { .enabled = true, .current_step = 1.0, .velocity_tolerance = 0.0 } ) );
  } catch ( const std::invalid_argument& ) {
    return true;
  }
  return false;
}

bool disabledVoteIsUnbounded()
{
  return std::isinf(
      timestep::kinematicVote<DefaultMethod>( ParentTracePairView{}, {}, DefaultMethod::Parameters{}, {}, {} ) );
}

}  // namespace

int main()
{
  return parentInterpolationAndMotionContract() && invalidParentVelocityIsRejected() && disabledVoteIsUnbounded() ? 0
                                                                                                                  : 1;
}
