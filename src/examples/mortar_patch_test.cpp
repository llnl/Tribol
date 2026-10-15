#include "tribol/Tribol.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <utility>

namespace {

using MortarPatchMethod =
    tribol::Method<tribol::geometry::ProjectedOverlap<tribol::normal::MortarSurface>, tribol::integration::Polygon<4>,
                   tribol::constraint::Nodal<tribol::basis::Primal>, tribol::enforcement::LagrangeMultiplier,
                   tribol::response::Frictionless, tribol::formulation::WeightedWeakForm, tribol::linearization::Exact>;

constexpr tribol::Real gap = 0.1;
constexpr tribol::Real pressure = 2.0;
constexpr tribol::Real tolerance = 1000.0 * std::numeric_limits<tribol::Real>::epsilon();

template <typename T, std::size_t Size>
tribol::ArrayView<const T> view( const std::array<T, Size>& values )
{
  return { values.data(), static_cast<tribol::Index>( values.size() ) };
}

tribol::Real totalComponent( tribol::FieldView<const tribol::Real> field, int component )
{
  tribol::Real total{};
  for ( tribol::Index node = 0; node < field.entities; ++node ) {
    total += field( node, component );
  }
  return total;
}

bool checkNear( std::string_view name, tribol::Real actual, tribol::Real expected )
{
  const tribol::Real error = std::abs( actual - expected );
  const bool passed = error <= tolerance;
  std::cout << "  " << std::left << std::setw( 34 ) << name << ( passed ? "PASS" : "FAIL" ) << "  actual=" << actual
            << " expected=" << expected << '\n';
  return passed;
}

bool checkEqual( std::string_view name, tribol::Index actual, tribol::Index expected )
{
  const bool passed = actual == expected;
  std::cout << "  " << std::left << std::setw( 34 ) << name << ( passed ? "PASS" : "FAIL" ) << "  actual=" << actual
            << " expected=" << expected << '\n';
  return passed;
}

bool checkNodalComponent( std::string_view name, tribol::FieldView<const tribol::Real> field,
                          tribol::ArrayView<const tribol::Real> expected, int component )
{
  bool passed = field.entities == expected.size();
  for ( tribol::Index node = 0; node < field.entities && node < expected.size(); ++node ) {
    passed = std::abs( field( node, component ) - expected[node] ) <= tolerance && passed;
  }
  std::cout << "  " << std::left << std::setw( 34 ) << name << ( passed ? "PASS" : "FAIL" ) << '\n';
  return passed;
}

}  // namespace

int main()
{
  using tribol::ElementTopology;
  using tribol::Index;
  using tribol::Real;

  constexpr std::array<Real, 12> mortar_coordinates{
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
  };
  constexpr std::array<Index, 4> mortar_connectivity{ 0, 1, 2, 3 };

  constexpr std::array<Real, 27> nonmortar_coordinates{
      0.0,  0.0, -gap, 0.5,  0.0, -gap, 1.0,  0.0, -gap, 0.0,  0.5, -gap, 0.5,  0.5,
      -gap, 1.0, 0.5,  -gap, 0.0, 1.0,  -gap, 0.5, 1.0,  -gap, 1.0, 1.0,  -gap,
  };
  // Clockwise nonmortar faces point toward the mortar face, so the two surface normals oppose each other.
  constexpr std::array<Index, 16> nonmortar_connectivity{
      0, 3, 4, 1, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5,
  };

  tribol::array::SingleTopologySurface mortar( 3, ElementTopology::Quadrilateral, view( mortar_coordinates ),
                                               view( mortar_connectivity ) );
  tribol::array::SingleTopologySurface nonmortar( 3, ElementTopology::Quadrilateral, view( nonmortar_coordinates ),
                                                  view( nonmortar_connectivity ) );

  tribol::array::ArrayContact<MortarPatchMethod>::Options options;
  options.search.expansion = 2.0 * gap;
  auto contact = tribol::array::makeContact<MortarPatchMethod>( std::move( mortar ), std::move( nonmortar ), options );
  contact.updateInteractions();

  // A negative multiplier produces compression for these opposing surface normals.
  constexpr std::array<Real, 4> multiplier{ -pressure, -pressure, -pressure, -pressure };
  const auto result = contact.evaluate( { .multiplier = view( multiplier ) } );

  constexpr std::array<Real, 4> expected_mortar_force{ 0.5, 0.5, 0.5, 0.5 };
  constexpr std::array<Real, 9> expected_nonmortar_force{
      -0.125, -0.25, -0.125, -0.25, -0.5, -0.25, -0.125, -0.25, -0.125,
  };

  Real constraint_sum{};
  for ( Real value : result.constraint_residual ) {
    constraint_sum += value;
  }

  std::cout << std::setprecision( 12 );
  std::cout << "Mortar patch test: one coarse face against four fine faces\n";
  bool passed = true;
  passed = checkEqual( "active interactions", result.summary.active_interactions, 4 ) && passed;
  passed = checkNear( "mortar integrated normal force", totalComponent( result.mortar_force, 2 ), pressure ) && passed;
  passed = checkNear( "nonmortar integrated normal force", totalComponent( result.nonmortar_force, 2 ), -pressure ) &&
           passed;
  passed = checkNear( "normal force balance",
                      totalComponent( result.mortar_force, 2 ) + totalComponent( result.nonmortar_force, 2 ), 0.0 ) &&
           passed;
  passed = checkNear( "mortar tangential force x", totalComponent( result.mortar_force, 0 ), 0.0 ) && passed;
  passed = checkNear( "mortar tangential force y", totalComponent( result.mortar_force, 1 ), 0.0 ) && passed;
  passed = checkNodalComponent( "mortar nodal normal forces", result.mortar_force, view( expected_mortar_force ), 2 ) &&
           passed;
  passed = checkNodalComponent( "nonmortar nodal normal forces", result.nonmortar_force,
                                view( expected_nonmortar_force ), 2 ) &&
           passed;
  passed = checkNear( "integrated gap constraint", constraint_sum, -gap ) && passed;

  std::cout << "  quadrature points                 " << result.summary.quadrature_points << '\n';
  std::cout << ( passed ? "Patch test passed.\n" : "Patch test failed.\n" );
  return passed ? 0 : 1;
}
