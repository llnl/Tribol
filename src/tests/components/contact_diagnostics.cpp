#include "tribol/Tribol.hpp"

#include "ConstraintRowFixture.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

// Requirements: OUTPUT-001, OUTPUT-002

namespace {

using namespace tribol;

SurfaceMeshView makeSegment( const std::array<Real, 4>& coordinates )
{
  static constexpr std::array<Index, 2> offsets{ 0, 2 };
  static constexpr std::array<Index, 2> connectivity{ 0, 1 };
  static constexpr std::array<ElementTopology, 1> topologies{ ElementTopology::Segment };
  static constexpr std::array<int, 1> attributes{ 3 };
  return {
      .dimension = 2,
      .coordinates = { { coordinates.data(), 4 }, 2, 2, FieldLayout::Interleaved },
      .element_offsets = { offsets.data(), 2 },
      .connectivity = { connectivity.data(), 2 },
      .topologies = { topologies.data(), 1 },
      .attributes = { attributes.data(), 1 },
  };
}

bool diagnosticsContract()
{
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  const SurfacePairView surfaces{ makeSegment( mortar_coordinates ), makeSegment( nonmortar_coordinates ) };
  Contact<> contact( surfaces, { .search = { .expansion = 0.2 } } );
  contact.updateInteractions();
  const auto result = contact.evaluate();
  std::ostringstream json;
  diagnostics::writeJson( json, surfaces, contact.interactions(), result );
  auto unbounded_result = result;
  unbounded_result.summary.timestep_vote = std::numeric_limits<Real>::infinity();
  std::ostringstream unbounded_json;
  diagnostics::writeJson( unbounded_json, surfaces, contact.interactions(), unbounded_result );
  std::ostringstream vtk;
  diagnostics::writeVtk( vtk, surfaces, result );
  return json.str().find( "\"active_interactions\":1" ) != std::string::npos &&
         json.str().find( "\"mortar_force\"" ) != std::string::npos &&
         json.str().find( "\"pressure\"" ) != std::string::npos &&
         unbounded_json.str().find( "\"timestep_vote\":null" ) != std::string::npos &&
         unbounded_json.str().find( "inf" ) == std::string::npos &&
         vtk.str().find( "DATASET UNSTRUCTURED_GRID" ) != std::string::npos &&
         vtk.str().find( "POINT_DATA 4" ) != std::string::npos &&
         vtk.str().find( "SCALARS contact_gap" ) != std::string::npos &&
         vtk.str().find( "SCALARS contact_pressure" ) != std::string::npos;
}

bool scheduledDiagnosticsContract()
{
  constexpr std::array<Real, 4> mortar_coordinates{ 0.0, 0.0, 1.0, 0.0 };
  constexpr std::array<Real, 4> nonmortar_coordinates{ 0.0, -0.1, 1.0, -0.1 };
  const SurfacePairView surfaces{ makeSegment( mortar_coordinates ), makeSegment( nonmortar_coordinates ) };
  Contact<> contact( surfaces, { .search = { .expansion = 0.2 } } );
  contact.updateInteractions();
  const auto result = contact.evaluate();
  const std::filesystem::path directory = "tribol-scheduled-diagnostics";
  std::filesystem::remove_all( directory );
  const diagnostics::OutputSchedule schedule{
      .directory = directory, .prefix = "cycle", .cycle_interval = 2, .rank = 3 };
  const auto skipped = diagnostics::writeScheduled<DefaultMethod>( schedule, 3, 0.3, surfaces, contact.interactions(),
                                                                   result, contact.options().method );
  const auto written = diagnostics::writeScheduled<DefaultMethod>( schedule, 4, 0.4, surfaces, contact.interactions(),
                                                                   result, contact.options().method );
  std::ifstream json( written.json );
  std::ifstream overlap( written.overlap_vtk );
  const std::string json_text{ std::istreambuf_iterator<char>( json ), std::istreambuf_iterator<char>() };
  const std::string overlap_text{ std::istreambuf_iterator<char>( overlap ), std::istreambuf_iterator<char>() };
  json.close();
  overlap.close();
  const bool valid = skipped.json.empty() && std::filesystem::exists( written.surface_vtk ) &&
                     json_text.find( "\"cycle\":4" ) != std::string::npos &&
                     json_text.find( "\"time\":0.40000000000000002" ) != std::string::npos &&
                     overlap_text.find( "Tribol overlap snapshot" ) != std::string::npos &&
                     overlap_text.find( "SCALARS mortar_element" ) != std::string::npos;
  std::filesystem::remove_all( directory );
  return valid;
}

bool rowDiagnosticsContract_OUTPUT_001()
{
  test::TwoRowFixture fixture;
  const auto rows = fixture.view();
  constexpr std::array<Real, 2> stiffness{ 10.0, 10.0 };
  constexpr std::array<Real, 2> damping{ 0.25, 0.25 };
  std::array<Real, 4> mortar_force{};
  std::array<Real, 4> nonmortar_force{};
  std::array<Real, 2> effective_gap{};
  std::array<Real, 2> pressure{};
  std::array<Real, 2> row_force{};
  std::array<Real, 2> row_stiffness{};
  std::array<Real, 2> row_rate{};
  (void)evaluateExplicitPenaltyRows( rows, { { stiffness.data(), 2 }, { damping.data(), 2 }, {} },
                                     { { { mortar_force.data(), 4 }, 2, 2, FieldLayout::Interleaved },
                                       { { nonmortar_force.data(), 4 }, 2, 2, FieldLayout::Interleaved } },
                                     { { effective_gap.data(), 2 },
                                       { pressure.data(), 2 },
                                       { row_force.data(), 2 },
                                       { row_stiffness.data(), 2 },
                                       { row_rate.data(), 2 } } );
  const auto diagnostics = diagnostics::summarizeExplicitRows( rows,
                                                               { { effective_gap.data(), 2 },
                                                                 { pressure.data(), 2 },
                                                                 { row_force.data(), 2 },
                                                                 { row_stiffness.data(), 2 },
                                                                 { row_rate.data(), 2 } },
                                                               0.2, 0 );
  return diagnostics.rows == 2 && diagnostics.owned_rows == 2 && diagnostics.active_rows == 2 &&
         diagnostics.nonfinite_rows == 0 && std::abs( diagnostics.minimum_gap + 0.1 ) < 1.0e-12 &&
         std::abs( diagnostics.maximum_penetration - 0.1 ) < 1.0e-12 &&
         std::abs( diagnostics.maximum_closing_rate - 0.2 ) < 1.0e-12 &&
         std::abs( diagnostics.mortar_force[1] + 1.025 ) < 1.0e-12 &&
         std::abs( diagnostics.mortar_moment[2] + 0.50625 ) < 1.0e-12 &&
         std::abs( diagnostics.impulse[1] + 0.205 ) < 1.0e-12 &&
         std::abs( diagnostics.maximum_traction - 1.05 ) < 1.0e-12 &&
         std::abs( diagnostics.spring_energy - 0.05 ) < 1.0e-12 &&
         std::abs( diagnostics.dashpot_dissipation - 0.001 ) < 1.0e-12 &&
         std::abs( diagnostics.contact_work - 0.011 ) < 1.0e-12;
}

}  // namespace

int main()
{
  return diagnosticsContract() && scheduledDiagnosticsContract() && rowDiagnosticsContract_OUTPUT_001() ? 0 : 1;
}
