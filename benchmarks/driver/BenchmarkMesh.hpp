#ifndef TRIBOL_BENCHMARKS_DRIVER_BENCHMARKMESH_HPP_
#define TRIBOL_BENCHMARKS_DRIVER_BENCHMARKMESH_HPP_

#include <array>
#include <cstddef>
#include <vector>

namespace tribol_benchmark {

template <typename Index>
struct SurfaceData {
  int dimension{};
  int nodes_per_element{};
  std::vector<double> coordinates;
  std::vector<Index> offsets;
  std::vector<Index> connectivity;

  SurfaceData( int surface_dimension, int element_nodes )
      : dimension( surface_dimension ), nodes_per_element( element_nodes )
  {
  }

  [[nodiscard]] int nodes() const { return static_cast<int>( coordinates.size() ) / dimension; }
  [[nodiscard]] int elements() const { return static_cast<int>( offsets.size() ) - 1; }

  [[nodiscard]] std::vector<double> component( int component_index ) const
  {
    std::vector<double> values( static_cast<std::size_t>( nodes() ) );
    for ( int node = 0; node < nodes(); ++node ) {
      values[static_cast<std::size_t>( node )] =
          coordinates[static_cast<std::size_t>( node * dimension + component_index )];
    }
    return values;
  }
};

template <typename Index>
struct MeshPairData {
  SurfaceData<Index> first;
  SurfaceData<Index> second;
};

template <typename Index>
void appendElement( SurfaceData<Index>& surface, const std::vector<std::array<double, 3>>& points )
{
  const Index first_node = static_cast<Index>( surface.nodes() );
  for ( const auto& point : points ) {
    for ( int component = 0; component < surface.dimension; ++component ) {
      surface.coordinates.push_back( point[static_cast<std::size_t>( component )] );
    }
    surface.connectivity.push_back( first_node + static_cast<Index>( surface.connectivity.size() % points.size() ) );
  }
  surface.offsets.push_back( static_cast<Index>( surface.connectivity.size() ) );
}

template <typename Index>
MeshPairData<Index> makePointwiseMesh( int dimension, int element_count )
{
  const int nodes_per_element = dimension == 2 ? 2 : 4;
  MeshPairData<Index> mesh{ SurfaceData<Index>( dimension, nodes_per_element ),
                            SurfaceData<Index>( dimension, nodes_per_element ) };
  mesh.first.offsets.push_back( 0 );
  mesh.second.offsets.push_back( 0 );
  for ( int element = 0; element < element_count; ++element ) {
    const double shift = 2.0 * element;
    if ( dimension == 2 ) {
      appendElement( mesh.first, { { shift + 1.0, 0.0, 0.0 }, { shift, 0.0, 0.0 } } );
      appendElement( mesh.second, { { shift, -0.1, 0.0 }, { shift + 1.0, -0.1, 0.0 } } );
    } else {
      appendElement(
          mesh.first,
          { { shift, 0.0, 0.0 }, { shift + 1.0, 0.0, 0.0 }, { shift + 1.0, 1.0, 0.0 }, { shift, 1.0, 0.0 } } );
      appendElement(
          mesh.second,
          { { shift, 0.0, -0.1 }, { shift, 1.0, -0.1 }, { shift + 1.0, 1.0, -0.1 }, { shift + 1.0, 0.0, -0.1 } } );
    }
  }
  return mesh;
}

template <typename Index>
MeshPairData<Index> makeMortarMesh( int element_count )
{
  MeshPairData<Index> mesh{ SurfaceData<Index>( 3, 4 ), SurfaceData<Index>( 3, 4 ) };
  mesh.first.offsets.push_back( 0 );
  mesh.second.offsets.push_back( 0 );
  for ( int element = 0; element < element_count; ++element ) {
    const double shift = 3.0 * element;
    appendElement( mesh.first, { { shift - 1.0, 1.0, 0.1 },
                                 { shift - 1.0, -1.0, 0.1 },
                                 { shift + 1.0, -1.0, 0.1 },
                                 { shift + 1.0, 1.0, 0.1 } } );
    appendElement(
        mesh.second,
        { { shift, 0.0, 0.0 }, { shift + 2.0, 0.0, 0.0 }, { shift + 2.0, -2.0, 0.0 }, { shift, -2.0, 0.0 } } );
  }
  return mesh;
}

}  // namespace tribol_benchmark

#endif
