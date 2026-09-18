#ifndef TRIBOL_BENCHMARKS_DRIVER_BENCHMARKPROTOCOL_HPP_
#define TRIBOL_BENCHMARKS_DRIVER_BENCHMARKPROTOCOL_HPP_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tribol_benchmark {

struct Options {
  std::string case_name{ "penalty-2d" };
  int size{ 1 };
  int warmups{ 2 };
  int iterations{ 10 };
  bool list_cases{};
};

struct Result {
  std::string implementation;
  Options options;
  std::vector<double> step_seconds;
  std::map<std::string, double> scalars;
  std::map<std::string, std::vector<double>> vectors;

  Result( std::string implementation_name, Options run_options )
      : implementation( std::move( implementation_name ) ), options( std::move( run_options ) )
  {
  }
};

inline int parsePositiveInteger( const char* text, const char* option )
{
  char* end{};
  const long value = std::strtol( text, &end, 10 );
  if ( end == text || *end != '\0' || value <= 0 ) {
    throw std::invalid_argument( std::string( option ) + " expects a positive integer" );
  }
  return static_cast<int>( value );
}

inline int parseNonnegativeInteger( const char* text, const char* option )
{
  char* end{};
  const long value = std::strtol( text, &end, 10 );
  if ( end == text || *end != '\0' || value < 0 ) {
    throw std::invalid_argument( std::string( option ) + " expects a nonnegative integer" );
  }
  return static_cast<int>( value );
}

inline Options parseOptions( int argc, char** argv )
{
  Options options;
  for ( int index = 1; index < argc; ++index ) {
    const std::string_view argument{ argv[index] };
    if ( argument == "--help" ) {
      std::cout << "Usage: " << argv[0] << " [--case NAME] [--size N] [--warmups N] [--iterations N] [--list-cases]\n";
      std::exit( 0 );
    }
    if ( argument == "--list-cases" ) {
      options.list_cases = true;
      continue;
    }
    if ( index + 1 >= argc ) {
      throw std::invalid_argument( std::string( argument ) + " expects a value" );
    }
    const char* value = argv[++index];
    if ( argument == "--case" ) {
      options.case_name = value;
    } else if ( argument == "--size" ) {
      options.size = parsePositiveInteger( value, "--size" );
    } else if ( argument == "--warmups" ) {
      options.warmups = parseNonnegativeInteger( value, "--warmups" );
    } else if ( argument == "--iterations" ) {
      options.iterations = parsePositiveInteger( value, "--iterations" );
    } else {
      throw std::invalid_argument( "unknown option: " + std::string( argument ) );
    }
  }
  return options;
}

template <typename Function>
std::vector<double> measure( const Options& options, Function&& function )
{
  for ( int iteration = 0; iteration < options.warmups; ++iteration ) {
    function();
  }
  std::vector<double> samples;
  samples.reserve( static_cast<std::size_t>( options.iterations ) );
  for ( int iteration = 0; iteration < options.iterations; ++iteration ) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto stop = std::chrono::steady_clock::now();
    samples.push_back( std::chrono::duration<double>( stop - start ).count() );
  }
  return samples;
}

inline void writeEscaped( std::ostream& stream, const std::string& value )
{
  stream << '"';
  for ( const char character : value ) {
    if ( character == '"' || character == '\\' ) {
      stream << '\\';
    }
    stream << character;
  }
  stream << '"';
}

inline void writeNumber( std::ostream& stream, double value )
{
  if ( !std::isfinite( value ) ) {
    stream << "null";
  } else {
    stream << std::setprecision( 17 ) << value;
  }
}

inline void writeVector( std::ostream& stream, const std::vector<double>& values )
{
  stream << '[';
  for ( std::size_t index = 0; index < values.size(); ++index ) {
    if ( index != 0 ) {
      stream << ',';
    }
    writeNumber( stream, values[index] );
  }
  stream << ']';
}

inline void writeResult( std::ostream& stream, const Result& result )
{
  stream << "{\"schema_version\":1,\"implementation\":";
  writeEscaped( stream, result.implementation );
  stream << ",\"case\":";
  writeEscaped( stream, result.options.case_name );
  stream << ",\"size\":" << result.options.size << ",\"warmups\":" << result.options.warmups
         << ",\"iterations\":" << result.options.iterations << ",\"timings\":{\"contact_step_seconds\":";
  writeVector( stream, result.step_seconds );
  stream << "},\"exactness\":{\"scalars\":{";
  bool first = true;
  for ( const auto& [name, value] : result.scalars ) {
    stream << ( first ? "" : "," );
    writeEscaped( stream, name );
    stream << ':';
    writeNumber( stream, value );
    first = false;
  }
  stream << "},\"vectors\":{";
  first = true;
  for ( const auto& [name, values] : result.vectors ) {
    stream << ( first ? "" : "," );
    writeEscaped( stream, name );
    stream << ':';
    writeVector( stream, values );
    first = false;
  }
  stream << "}}}\n";
}

inline double vectorL1( const std::vector<double>& values )
{
  double result{};
  for ( const double value : values ) {
    result += std::abs( value );
  }
  return result;
}

}  // namespace tribol_benchmark

#endif
