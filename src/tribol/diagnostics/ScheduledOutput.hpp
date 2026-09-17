#ifndef TRIBOL_DIAGNOSTICS_SCHEDULEDOUTPUT_HPP_
#define TRIBOL_DIAGNOSTICS_SCHEDULEDOUTPUT_HPP_

#include "tribol/diagnostics/ContactOutput.hpp"
#include "tribol/geom/ConformingOverlap.hpp"
#include "tribol/geom/ProjectedOverlap.hpp"
#include "tribol/geom/SmoothedSegmentOverlap.hpp"
#include "tribol/method/Method.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tribol::diagnostics {

struct OutputSchedule {
  std::filesystem::path directory{ "." };
  std::string prefix{ "tribol-contact" };
  int cycle_interval{ 100 };
  int rank{ -1 };
  bool json{ true };
  bool surface_vtk{ true };
  bool overlap_vtk{ true };
};

struct SnapshotPaths {
  std::filesystem::path json;
  std::filesystem::path surface_vtk;
  std::filesystem::path overlap_vtk;
};

inline bool shouldWrite( const OutputSchedule& schedule, int cycle )
{
  return schedule.cycle_interval > 0 && cycle >= 0 && cycle % schedule.cycle_interval == 0;
}

namespace detail {

inline std::filesystem::path snapshotPath( const OutputSchedule& schedule, int cycle, const char* suffix )
{
  std::ostringstream name;
  name << schedule.prefix << '-' << std::setfill( '0' ) << std::setw( 8 ) << cycle;
  if ( schedule.rank >= 0 ) {
    name << "-rank" << std::setfill( '0' ) << std::setw( 6 ) << schedule.rank;
  }
  name << suffix;
  return schedule.directory / name.str();
}

template <SupportedMethod MethodType>
InteractionPatch makeDiagnosticPatch( const SurfacePairView& surfaces, ElementPair pair,
                                      const typename MethodType::Parameters& parameters )
{
  using Geometry = typename MethodType::geometry_policy;
  using Integration = typename MethodType::integration_policy;
  if constexpr ( tribol::detail::is_projected_overlap_v<Geometry> ) {
    if constexpr ( tribol::detail::is_smoothed_segment_integration_v<Integration> ) {
      return smoothedProjectedSegmentPatch<typename Geometry::normal_policy, Integration>(
          surfaces, pair, parameters.geometry, parameters.integration );
    } else {
      return projectedOverlapPatch<typename Geometry::normal_policy>( surfaces, pair, parameters.geometry );
    }
  } else {
    return conformingOverlapPatch( surfaces, pair, parameters.geometry );
  }
}

}  // namespace detail

template <SupportedMethod MethodType>
void writeOverlapVtk( std::ostream& stream, const SurfacePairView& surfaces, ArrayView<const ElementPair> interactions,
                      const typename MethodType::Parameters& parameters )
{
  std::vector<InteractionPatch> patches;
  std::vector<ElementPair> active_pairs;
  Index point_count{};
  for ( ElementPair pair : interactions ) {
    auto patch = detail::makeDiagnosticPatch<MethodType>( surfaces, pair, parameters );
    if ( patch.valid ) {
      point_count += patch.vertex_count;
      patches.push_back( patch );
      active_pairs.push_back( pair );
    }
  }
  stream << std::setprecision( 17 ) << "# vtk DataFile Version 3.0\nTribol overlap snapshot\nASCII\n"
         << "DATASET UNSTRUCTURED_GRID\nPOINTS " << point_count << " double\n";
  for ( const auto& patch : patches ) {
    for ( int vertex = 0; vertex < patch.vertex_count; ++vertex ) {
      stream << patch.integration_vertices[vertex][0] << ' ' << patch.integration_vertices[vertex][1] << ' '
             << patch.integration_vertices[vertex][2] << '\n';
    }
  }
  stream << "CELLS " << patches.size() << ' ' << point_count + static_cast<Index>( patches.size() ) << '\n';
  Index point_offset{};
  for ( const auto& patch : patches ) {
    stream << patch.vertex_count;
    for ( int vertex = 0; vertex < patch.vertex_count; ++vertex ) {
      stream << ' ' << point_offset++;
    }
    stream << '\n';
  }
  stream << "CELL_TYPES " << patches.size() << '\n';
  for ( const auto& patch : patches ) {
    stream << ( patch.vertex_count == 2 ? 3 : patch.vertex_count == 3 ? 5 : 7 ) << '\n';
  }
  stream << "CELL_DATA " << patches.size() << "\nSCALARS contact_gap double 1\nLOOKUP_TABLE default\n";
  for ( const auto& patch : patches ) {
    Real gap{};
    for ( int component = 0; component < surfaces.mortar.dimension; ++component ) {
      gap += ( patch.mortar_centroid[component] - patch.nonmortar_centroid[component] ) * patch.normal[component];
    }
    stream << gap << '\n';
  }
  stream << "SCALARS mortar_element int 1\nLOOKUP_TABLE default\n";
  for ( ElementPair pair : active_pairs ) {
    stream << pair.mortar_element << '\n';
  }
  stream << "SCALARS nonmortar_element int 1\nLOOKUP_TABLE default\n";
  for ( ElementPair pair : active_pairs ) {
    stream << pair.nonmortar_element << '\n';
  }
  detail::requireStream( stream );
}

template <SupportedMethod MethodType>
SnapshotPaths writeScheduled( const OutputSchedule& schedule, int cycle, Real time, const SurfacePairView& surfaces,
                              ArrayView<const ElementPair> interactions, const ContactResultView& result,
                              const typename MethodType::Parameters& parameters )
{
  if ( !shouldWrite( schedule, cycle ) ) {
    return {};
  }
  if ( schedule.prefix.empty() ) {
    throw std::invalid_argument( "Diagnostics output prefix cannot be empty." );
  }
  std::filesystem::create_directories( schedule.directory );
  SnapshotPaths paths;
  if ( schedule.json ) {
    paths.json = detail::snapshotPath( schedule, cycle, ".json" );
    std::ofstream stream( paths.json );
    stream << "{\"cycle\":" << cycle << ",\"time\":" << std::setprecision( 17 ) << time << ",\"contact\":";
    writeJson( stream, surfaces, interactions, result );
    stream.seekp( -1, std::ios_base::end );
    stream << "}\n";
    detail::requireStream( stream );
  }
  if ( schedule.surface_vtk ) {
    paths.surface_vtk = detail::snapshotPath( schedule, cycle, "-surfaces.vtk" );
    writeVtk( paths.surface_vtk.string(), surfaces, result );
  }
  if ( schedule.overlap_vtk ) {
    paths.overlap_vtk = detail::snapshotPath( schedule, cycle, "-overlaps.vtk" );
    std::ofstream stream( paths.overlap_vtk );
    if ( !stream ) {
      throw std::runtime_error( "Unable to open Tribol overlap diagnostics file: " + paths.overlap_vtk.string() );
    }
    writeOverlapVtk<MethodType>( stream, surfaces, interactions, parameters );
  }
  return paths;
}

}  // namespace tribol::diagnostics

#endif
