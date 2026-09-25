#ifndef TRIBOL_DIAGNOSTICS_CONTACTOUTPUT_HPP_
#define TRIBOL_DIAGNOSTICS_CONTACTOUTPUT_HPP_

#include "tribol/core/MeshView.hpp"
#include "tribol/diagnostics/ProductionDiagnostics.hpp"
#include "tribol/evaluation/State.hpp"
#include "tribol/search/Search.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tribol::diagnostics {

namespace detail {

inline void writeNumber( std::ostream& stream, Real value )
{
  if ( std::isfinite( value ) ) {
    stream << ( value == 0.0 ? 0.0 : value );
  } else {
    stream << "null";
  }
}

inline void writeArray( std::ostream& stream, ArrayView<const Real> values )
{
  stream << '[';
  for ( Index entry = 0; entry < values.size(); ++entry ) {
    if ( entry != 0 ) {
      stream << ',';
    }
    writeNumber( stream, values[entry] );
  }
  stream << ']';
}

inline void writeVector( std::ostream& stream, const std::array<Real, 3>& values )
{
  stream << '[';
  for ( int component = 0; component < 3; ++component ) {
    if ( component != 0 ) {
      stream << ',';
    }
    writeNumber( stream, values[component] );
  }
  stream << ']';
}

inline void writeField( std::ostream& stream, FieldView<const Real> field )
{
  stream << '[';
  for ( Index entity = 0; entity < field.entities; ++entity ) {
    if ( entity != 0 ) {
      stream << ',';
    }
    stream << '[';
    for ( int component = 0; component < field.components; ++component ) {
      if ( component != 0 ) {
        stream << ',';
      }
      writeNumber( stream, field( entity, component ) );
    }
    stream << ']';
  }
  stream << ']';
}

inline void writeSurface( std::ostream& stream, const char* name, const SurfaceMeshView& surface )
{
  stream << '\"' << name << "\":{" << "\"dimension\":" << surface.dimension << ",\"coordinates\":";
  writeField( stream, surface.coordinates );
  stream << ",\"elements\":[";
  for ( Index element = 0; element < surface.numberOfElements(); ++element ) {
    if ( element != 0 ) {
      stream << ',';
    }
    stream << "{\"topology\":" << static_cast<int>( surface.topologies[element] )
           << ",\"attribute\":" << surface.attributes[element] << ",\"nodes\":[";
    for ( Index entry = surface.element_offsets[element]; entry < surface.element_offsets[element + 1]; ++entry ) {
      if ( entry != surface.element_offsets[element] ) {
        stream << ',';
      }
      stream << surface.connectivity[entry];
    }
    stream << "]}";
  }
  stream << "]}";
}

inline void requireStream( const std::ostream& stream )
{
  if ( !stream ) {
    throw std::runtime_error( "Unable to write Tribol contact diagnostics." );
  }
}

inline const char* attemptStatusName( AttemptStatus status )
{
  switch ( status ) {
    case AttemptStatus::None:
      return "none";
    case AttemptStatus::Active:
      return "active";
    case AttemptStatus::Accepted:
      return "accepted";
    case AttemptStatus::Rejected:
      return "rejected";
  }
  return "unknown";
}

inline const char* failureCategoryName( FailureCategory failure )
{
  switch ( failure ) {
    case FailureCategory::None:
      return "none";
    case FailureCategory::InvalidInput:
      return "invalid_input";
    case FailureCategory::Search:
      return "search";
    case FailureCategory::Geometry:
      return "geometry";
    case FailureCategory::Kinematic:
      return "kinematic";
    case FailureCategory::Stability:
      return "stability";
    case FailureCategory::NonfiniteState:
      return "nonfinite_state";
    case FailureCategory::HostApplication:
      return "host_application";
  }
  return "unknown";
}

inline void writeAttemptRecord( std::ostream& stream, const ContactAttemptRecord& attempt )
{
  const auto& generations = attempt.generations;
  stream << "{\"id\":" << attempt.id << ",\"retry\":" << attempt.retry << ",\"status\":\""
         << attemptStatusName( attempt.status ) << "\",\"failure\":\"" << failureCategoryName( attempt.failure )
         << "\",\"generations\":{\"coordinates\":" << generations.coordinates.value()
         << ",\"topology\":" << generations.topology.value() << ",\"interactions\":" << generations.interactions.value()
         << ",\"mass\":" << generations.mass.value()
         << ",\"boundary_conditions\":" << generations.boundary_conditions.value()
         << ",\"penalty\":" << generations.penalty.value() << ",\"integrator\":" << generations.integrator.value()
         << "}}";
}

inline void writeProductionSummary( std::ostream& stream, const char* semantics,
                                    const ProductionDiagnosticSummary& summary )
{
  stream << "{\"semantics\":\"" << semantics << "\",\"counts\":{\"rows\":" << summary.rows.rows
         << ",\"owned_rows\":" << summary.rows.owned_rows << ",\"active_rows\":" << summary.rows.active_rows
         << ",\"nonfinite_rows\":" << summary.rows.nonfinite_rows
         << ",\"diagnostic_samples\":" << summary.quadrature.samples
         << ",\"owned_diagnostic_samples\":" << summary.quadrature.owned_samples << "},\"sum\":{";
  stream << "\"mortar_force\":";
  writeVector( stream, summary.rows.mortar_force );
  stream << ",\"mortar_moment\":";
  writeVector( stream, summary.rows.mortar_moment );
  stream << ",\"impulse\":";
  writeVector( stream, summary.rows.impulse );
  stream << ",\"spring_energy\":";
  writeNumber( stream, summary.rows.spring_energy );
  stream << ",\"dashpot_dissipation\":";
  writeNumber( stream, summary.rows.dashpot_dissipation );
  stream << ",\"contact_work\":";
  writeNumber( stream, summary.rows.contact_work );
  stream << "},\"maximum\":{\"traction\":";
  writeNumber( stream, summary.rows.maximum_traction );
  stream << ",\"penetration\":";
  writeNumber( stream, summary.quadrature.maximum_penetration );
  stream << ",\"closing_rate\":";
  writeNumber( stream, summary.quadrature.maximum_closing_rate );
  stream << "}}";
}

}  // namespace detail

inline void writeJson( std::ostream& stream, const ProductionDiagnosticsPayload& payload )
{
  if ( !isStructurallyValid( payload ) ) {
    throw std::invalid_argument( "Cannot write an inconsistent production diagnostics payload." );
  }
  std::ostringstream json;
  json.imbue( std::locale::classic() );
  json << std::setprecision( std::numeric_limits<Real>::max_digits10 )
       << "{\"schema\":\"tribol.milestone-1.production-diagnostics\",\"version\":" << payload.schema_version
       << ",\"rank\":" << payload.rank.rank << ",\"ranks\":" << payload.rank.ranks << ",\"attempts\":{\"counts\":{"
       << "\"accepted\":" << payload.attempts.accepted_attempts
       << ",\"rejected\":" << payload.attempts.rejected_attempts << "},\"current\":";
  detail::writeAttemptRecord( json, payload.attempts.current );
  json << ",\"last_accepted\":";
  detail::writeAttemptRecord( json, payload.attempts.last_accepted );
  json << ",\"last_rejected\":";
  detail::writeAttemptRecord( json, payload.attempts.last_rejected );
  json << "},\"local\":";
  detail::writeProductionSummary( json, "all_rows_and_diagnostic_samples_resident_on_rank", payload.rank.local );
  json << ",\"owned\":";
  detail::writeProductionSummary( json, "owner_filtered_rows_and_diagnostic_samples_on_rank", payload.rank.owned );
  json << ",\"global\":";
  detail::writeProductionSummary( json, "rank_ordered_sum_and_maximum_of_owned_values", payload.global );
  json << "}\n";
  stream << json.str();
  detail::requireStream( stream );
}

inline void writeJson( const std::string& path, const ProductionDiagnosticsPayload& payload )
{
  std::ofstream stream( path, std::ios::binary );
  if ( !stream ) {
    throw std::runtime_error( "Unable to open Tribol production diagnostics JSON file: " + path );
  }
  writeJson( stream, payload );
}

inline void writeJson( std::ostream& stream, const SurfacePairView& surfaces, ArrayView<const ElementPair> interactions,
                       const ContactResultView& result )
{
  stream << std::setprecision( 17 ) << '{';
  detail::writeSurface( stream, "mortar", surfaces.mortar );
  stream << ',';
  detail::writeSurface( stream, "nonmortar", surfaces.nonmortar );
  stream << ",\"interactions\":[";
  for ( Index interaction = 0; interaction < interactions.size(); ++interaction ) {
    if ( interaction != 0 ) {
      stream << ',';
    }
    stream << '[' << interactions[interaction].mortar_element << ',' << interactions[interaction].nonmortar_element
           << ']';
  }
  stream << "],\"result\":{";
  stream << "\"mortar_force\":";
  detail::writeField( stream, result.mortar_force );
  stream << ",\"nonmortar_force\":";
  detail::writeField( stream, result.nonmortar_force );
  stream << ",\"constraint_residual\":";
  detail::writeArray( stream, result.constraint_residual );
  stream << ",\"gap\":";
  detail::writeArray( stream, result.gap );
  stream << ",\"weighted_gap\":";
  detail::writeArray( stream, result.weighted_gap );
  stream << ",\"tributary_area\":";
  detail::writeArray( stream, result.tributary_area );
  stream << ",\"mortar_weights\":";
  detail::writeArray( stream, result.mortar_weights );
  stream << ",\"mortar_mass_weights\":";
  detail::writeArray( stream, result.mortar_mass_weights );
  stream << ",\"quadrature_gap\":";
  detail::writeArray( stream, result.quadrature_gap );
  stream << ",\"quadrature_pressure\":";
  detail::writeArray( stream, result.quadrature_pressure );
  stream << ",\"pressure\":";
  detail::writeArray( stream, result.pressure );
  stream << ",\"energy\":";
  detail::writeNumber( stream, result.summary.energy );
  stream << ",\"timestep_vote\":";
  detail::writeNumber( stream, result.summary.timestep_vote );
  stream << ",\"active_interactions\":" << result.summary.active_interactions
         << ",\"quadrature_points\":" << result.summary.quadrature_points << "}}\n";
  detail::requireStream( stream );
}

inline void writeJson( const std::string& path, const SurfacePairView& surfaces,
                       ArrayView<const ElementPair> interactions, const ContactResultView& result )
{
  std::ofstream stream( path );
  if ( !stream ) {
    throw std::runtime_error( "Unable to open Tribol JSON diagnostics file: " + path );
  }
  writeJson( stream, surfaces, interactions, result );
}

inline void writeVtk( std::ostream& stream, const SurfacePairView& surfaces, const ContactResultView& result )
{
  const Index mortar_nodes = surfaces.mortar.numberOfNodes();
  const Index total_nodes = mortar_nodes + surfaces.nonmortar.numberOfNodes();
  const Index total_elements = surfaces.mortar.numberOfElements() + surfaces.nonmortar.numberOfElements();
  Index connectivity_size{};
  for ( const auto& surface : { surfaces.mortar, surfaces.nonmortar } ) {
    connectivity_size += surface.connectivity.size() + surface.numberOfElements();
  }
  stream << std::setprecision( 17 )
         << "# vtk DataFile Version 3.0\nTribol contact snapshot\nASCII\nDATASET UNSTRUCTURED_GRID\n";
  stream << "POINTS " << total_nodes << " double\n";
  for ( const auto& surface : { surfaces.mortar, surfaces.nonmortar } ) {
    for ( Index node = 0; node < surface.numberOfNodes(); ++node ) {
      for ( int component = 0; component < 3; ++component ) {
        stream << ( component < surface.dimension ? surface.coordinates( node, component ) : 0.0 ) << ' ';
      }
      stream << '\n';
    }
  }
  stream << "CELLS " << total_elements << ' ' << connectivity_size << '\n';
  Index node_offset{};
  for ( const auto& surface : { surfaces.mortar, surfaces.nonmortar } ) {
    for ( Index element = 0; element < surface.numberOfElements(); ++element ) {
      const Index begin = surface.element_offsets[element];
      const Index end = surface.element_offsets[element + 1];
      stream << end - begin;
      for ( Index entry = begin; entry < end; ++entry ) {
        stream << ' ' << node_offset + surface.connectivity[entry];
      }
      stream << '\n';
    }
    node_offset += surface.numberOfNodes();
  }
  stream << "CELL_TYPES " << total_elements << '\n';
  for ( const auto& surface : { surfaces.mortar, surfaces.nonmortar } ) {
    for ( ElementTopology topology : surface.topologies ) {
      stream << ( topology == ElementTopology::Segment ? 3 : topology == ElementTopology::Triangle ? 5 : 9 ) << '\n';
    }
  }
  stream << "POINT_DATA " << total_nodes << "\nVECTORS contact_force double\n";
  for ( const auto field : { result.mortar_force, result.nonmortar_force } ) {
    for ( Index node = 0; node < field.entities; ++node ) {
      for ( int component = 0; component < 3; ++component ) {
        stream << ( component < field.components ? field( node, component ) : 0.0 ) << ' ';
      }
      stream << '\n';
    }
  }
  stream << "SCALARS contact_gap double 1\nLOOKUP_TABLE default\n";
  for ( Index node = 0; node < total_nodes; ++node ) {
    stream << ( node < result.gap.size() ? result.gap[node] : 0.0 ) << '\n';
  }
  stream << "SCALARS contact_pressure double 1\nLOOKUP_TABLE default\n";
  for ( Index node = 0; node < total_nodes; ++node ) {
    stream << ( node < result.pressure.size() ? result.pressure[node] : 0.0 ) << '\n';
  }
  detail::requireStream( stream );
}

inline void writeVtk( const std::string& path, const SurfacePairView& surfaces, const ContactResultView& result )
{
  std::ofstream stream( path );
  if ( !stream ) {
    throw std::runtime_error( "Unable to open Tribol VTK diagnostics file: " + path );
  }
  writeVtk( stream, surfaces, result );
}

}  // namespace tribol::diagnostics

#endif
