#include "tribol/Tribol.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

// Requirements: CORE-004

namespace {

using namespace tribol;

constexpr Real tolerance = 2.0e-11;

bool close( Real left, Real right ) { return std::abs( left - right ) < tolerance; }

bool nodalAndPartitionContract( ElementTopology topology, int order )
{
  const int count = basis::parentDofCount( topology, order );
  for ( int node = 0; node < count; ++node ) {
    const auto shape = basis::parentShape( topology, order, basis::parentReferenceNode( topology, order, node ) );
    if ( shape.size != count ) {
      return false;
    }
    for ( int other = 0; other < count; ++other ) {
      if ( !close( shape[other], node == other ? 1.0 : 0.0 ) ) {
        return false;
      }
    }
  }
  const basis::ReferencePoint point =
      topology == ElementTopology::Triangle ? basis::ReferencePoint{ 0.23, 0.31 } : basis::ReferencePoint{ 0.23, 0.67 };
  const auto shape = basis::parentShape( topology, order, point );
  Real sum{};
  for ( int node = 0; node < shape.size; ++node ) {
    sum += shape[node];
  }
  return close( sum, 1.0 );
}

Real integerPower( Real value, int power )
{
  Real result{ 1.0 };
  for ( int exponent = 0; exponent < power; ++exponent ) {
    result *= value;
  }
  return result;
}

Real polynomial( ElementTopology topology, int order, basis::ReferencePoint point )
{
  if ( topology == ElementTopology::Segment ) {
    return 1.0 + 0.5 * point.first + integerPower( point.first, order );
  }
  if ( topology == ElementTopology::Triangle ) {
    return 1.0 + integerPower( point.first, order ) - 0.25 * integerPower( point.second, order );
  }
  return 1.0 + integerPower( point.first, order ) - 0.25 * integerPower( point.second, order ) +
         0.5 * point.first * point.second;
}

bool polynomialContract( ElementTopology topology, int order )
{
  const int count = basis::parentDofCount( topology, order );
  std::vector<Real> nodal_values( static_cast<std::size_t>( count ) );
  for ( int node = 0; node < count; ++node ) {
    nodal_values[static_cast<std::size_t>( node )] =
        polynomial( topology, order, basis::parentReferenceNode( topology, order, node ) );
  }
  const basis::ReferencePoint point =
      topology == ElementTopology::Triangle ? basis::ReferencePoint{ 0.17, 0.29 } : basis::ReferencePoint{ 0.17, 0.71 };
  const auto shape = basis::parentShape( topology, order, point );
  Real interpolated{};
  for ( int node = 0; node < count; ++node ) {
    interpolated += shape[node] * nodal_values[static_cast<std::size_t>( node )];
  }
  return close( interpolated, polynomial( topology, order, point ) );
}

bool basisContract()
{
  constexpr std::array<ElementTopology, 3> topologies{ ElementTopology::Segment, ElementTopology::Triangle,
                                                       ElementTopology::Quadrilateral };
  for ( ElementTopology topology : topologies ) {
    for ( int order = 1; order <= maximumParentTraceOrder; ++order ) {
      if ( !nodalAndPartitionContract( topology, order ) || !polynomialContract( topology, order ) ) {
        return false;
      }
    }
  }
  return true;
}

struct SegmentTraceStorage {
  std::array<Real, 4> coordinates{ 0.0, 0.0, 1.0, 0.0 };
  std::array<Index, 2> geometry_offsets{ 0, 2 };
  std::array<Index, 2> geometry_connectivity{ 0, 1 };
  std::array<ElementTopology, 1> geometry_topologies{ ElementTopology::Segment };
  std::array<int, 1> geometry_attributes{ 1 };
  std::array<std::uint64_t, 1> face_ids{ 17 };
  std::array<int, 1> face_owners{ 0 };
  std::array<ElementTopology, 1> face_topologies{ ElementTopology::Segment };
  std::array<int, 1> face_orders{ 2 };
  std::array<Index, 2> dof_offsets{ 0, 3 };
  std::array<Index, 3> dofs{ 0, 1, 2 };
  std::array<Index, 1> subfacet_faces{ 0 };
  std::array<Real, 2> parent_reference_coordinates{ 0.0, 1.0 };

  ParentTraceView view() const
  {
    return {
        .geometry =
            {
                .dimension = 2,
                .coordinates = { { coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
                .element_offsets = { geometry_offsets.data(), 2 },
                .connectivity = { geometry_connectivity.data(), 2 },
                .topologies = { geometry_topologies.data(), 1 },
                .attributes = { geometry_attributes.data(), 1 },
            },
        .parent_face_ids = { face_ids.data(), 1 },
        .parent_face_owners = { face_owners.data(), 1 },
        .parent_face_topologies = { face_topologies.data(), 1 },
        .parent_face_orders = { face_orders.data(), 1 },
        .parent_dof_offsets = { dof_offsets.data(), 2 },
        .parent_dofs = { dofs.data(), 3 },
        .subfacet_parent_faces = { subfacet_faces.data(), 1 },
        .subfacet_parent_reference_coordinates = { { parent_reference_coordinates.data(), 2 },
                                                   2,
                                                   1,
                                                   FieldLayout::Interleaved },
    };
  }
};

bool traceAndInterpolationContract()
{
  SegmentTraceStorage storage;
  const auto trace = storage.view();
  if ( !trace.isStructurallyValid() || trace.numberOfParentFaces() != 1 || trace.numberOfSubfacets() != 1 ||
       trace.parentReferenceDimension() != 1 ) {
    return false;
  }

  constexpr std::array<Real, 6> field_values{ 0.0, 1.0, 0.25, 1.5, 1.0, 2.0 };
  const FieldView<const Real> field{ { field_values.data(), 6 }, 3, 2, FieldLayout::Interleaved };
  const auto value = basis::interpolateParentField( trace, 0, field, basis::ReferencePoint{ 0.4, 0.0 } );
  if ( !close( value[0], 0.16 ) || !close( value[1], 1.4 ) ) {
    return false;
  }

  storage.face_orders[0] = 5;
  return !storage.view().isStructurallyValid();
}

static_assert( std::is_trivially_copyable_v<ParentTraceView> );
static_assert( std::is_trivially_copyable_v<ParentTracePairView> );
static_assert( std::is_trivially_copyable_v<basis::ParentShapeValues> );

}  // namespace

int main() { return basisContract() && traceAndInterpolationContract() ? 0 : 1; }
