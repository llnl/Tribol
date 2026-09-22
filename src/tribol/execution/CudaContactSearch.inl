struct DeviceBounds {
  Real minimum[3];
  Real maximum[3];
};

struct DeviceBvhNode {
  DeviceBounds bounds{};
  Index left{ -1 };
  Index right{ -1 };
  Index element{ -1 };
};

struct BoundsUnion {
  TRIBOL_HOST_DEVICE DeviceBounds operator()( const DeviceBounds& left, const DeviceBounds& right ) const
  {
    DeviceBounds result{};
    for ( int component = 0; component < 3; ++component ) {
      result.minimum[component] =
          left.minimum[component] < right.minimum[component] ? left.minimum[component] : right.minimum[component];
      result.maximum[component] =
          left.maximum[component] > right.maximum[component] ? left.maximum[component] : right.maximum[component];
    }
    return result;
  }
};

TRIBOL_HOST_DEVICE DeviceBounds emptyBounds()
{
  DeviceBounds bounds{};
  for ( int component = 0; component < 3; ++component ) {
    bounds.minimum[component] = std::numeric_limits<Real>::max();
    bounds.maximum[component] = -std::numeric_limits<Real>::max();
  }
  return bounds;
}

__device__ DeviceBounds elementBounds( const SurfaceMeshView& mesh, Index element, Real expansion,
                                       Real proximity_scale )
{
  DeviceBounds bounds = emptyBounds();
  const Index begin = mesh.element_offsets[element];
  const Index end = mesh.element_offsets[element + 1];
  Real longest_extent{};
  for ( int component = 0; component < mesh.dimension; ++component ) {
    for ( Index local_node = begin; local_node < end; ++local_node ) {
      const Real value = mesh.coordinates( mesh.connectivity[local_node], component );
      bounds.minimum[component] = value < bounds.minimum[component] ? value : bounds.minimum[component];
      bounds.maximum[component] = value > bounds.maximum[component] ? value : bounds.maximum[component];
    }
    const Real extent = bounds.maximum[component] - bounds.minimum[component];
    longest_extent = extent > longest_extent ? extent : longest_extent;
  }
  for ( int component = mesh.dimension; component < 3; ++component ) {
    bounds.minimum[component] = 0.0;
    bounds.maximum[component] = 0.0;
  }
  const Real total_expansion = expansion + proximity_scale * longest_extent;
  for ( int component = 0; component < mesh.dimension; ++component ) {
    bounds.minimum[component] -= total_expansion;
    bounds.maximum[component] += total_expansion;
  }
  return bounds;
}

__device__ bool overlaps( const DeviceBounds& left, const DeviceBounds& right, int dimension )
{
  for ( int component = 0; component < dimension; ++component ) {
    if ( left.maximum[component] < right.minimum[component] || right.maximum[component] < left.minimum[component] ) {
      return false;
    }
  }
  return true;
}

__global__ void buildBoundsKernel( SurfaceMeshView mesh, Real expansion, Real proximity_scale, DeviceBounds* bounds )
{
  const Index element = static_cast<Index>( blockIdx.x * blockDim.x + threadIdx.x );
  if ( element < mesh.numberOfElements() ) {
    bounds[element] = elementBounds( mesh, element, expansion, proximity_scale );
  }
}

__device__ std::uint32_t expandMortonBits( std::uint32_t value )
{
  value = ( value * 0x00010001u ) & 0xFF0000FFu;
  value = ( value * 0x00000101u ) & 0x0F00F00Fu;
  value = ( value * 0x00000011u ) & 0xC30C30C3u;
  value = ( value * 0x00000005u ) & 0x49249249u;
  return value;
}

__global__ void mortonKernel( const DeviceBounds* bounds, Index count, const DeviceBounds* global_bounds,
                              std::uint32_t* keys, Index* elements )
{
  const Index element = static_cast<Index>( blockIdx.x * blockDim.x + threadIdx.x );
  if ( element >= count ) {
    return;
  }
  std::uint32_t coordinate[3]{};
  for ( int component = 0; component < 3; ++component ) {
    const Real center = 0.5 * ( bounds[element].minimum[component] + bounds[element].maximum[component] );
    const Real extent = global_bounds->maximum[component] - global_bounds->minimum[component];
    Real normalized = extent > 0.0 ? ( center - global_bounds->minimum[component] ) / extent : 0.5;
    normalized = normalized < 0.0 ? 0.0 : normalized > 1.0 ? 1.0 : normalized;
    coordinate[component] = static_cast<std::uint32_t>( normalized * 1023.0 );
  }
  keys[element] = expandMortonBits( coordinate[0] ) | ( expandMortonBits( coordinate[1] ) << 1U ) |
                  ( expandMortonBits( coordinate[2] ) << 2U );
  elements[element] = element;
}

__global__ void initializeLeavesKernel( const DeviceBounds* element_bounds, const Index* order, Index count,
                                        DeviceBvhNode* nodes )
{
  const Index leaf = static_cast<Index>( blockIdx.x * blockDim.x + threadIdx.x );
  if ( leaf < count ) {
    const Index element = order[leaf];
    nodes[leaf] = { .bounds = element_bounds[element], .element = element };
  }
}

__global__ void buildBvhLevelKernel( DeviceBvhNode* nodes, Index child_offset, Index child_count, Index parent_offset )
{
  const Index parent = static_cast<Index>( blockIdx.x * blockDim.x + threadIdx.x );
  const Index first = 2 * parent;
  if ( first >= child_count ) {
    return;
  }
  DeviceBvhNode node;
  node.left = child_offset + first;
  node.bounds = nodes[node.left].bounds;
  if ( first + 1 < child_count ) {
    node.right = node.left + 1;
    node.bounds = BoundsUnion{}( node.bounds, nodes[node.right].bounds );
  }
  nodes[parent_offset + parent] = node;
}

__device__ bool sharesNode( const SurfaceMeshView& mesh, Index left_element, Index right_element )
{
  const Index left_begin = mesh.element_offsets[left_element];
  const Index left_end = mesh.element_offsets[left_element + 1];
  const Index right_begin = mesh.element_offsets[right_element];
  const Index right_end = mesh.element_offsets[right_element + 1];
  for ( Index left = left_begin; left < left_end; ++left ) {
    for ( Index right = right_begin; right < right_end; ++right ) {
      if ( mesh.connectivity[left] == mesh.connectivity[right] ) {
        return true;
      }
    }
  }
  return false;
}

template <bool Fill>
__device__ Index traverseBvh( const DeviceBvhNode* nodes, Index root, const DeviceBounds& query,
                              const SurfacePairView& surfaces, Index nonmortar, bool self_contact,
                              bool exclude_adjacent, ElementPair* output )
{
  Index count{};
  Index stack[64];
  int stack_size = 1;
  stack[0] = root;
  while ( stack_size > 0 ) {
    const DeviceBvhNode& node = nodes[stack[--stack_size]];
    if ( !overlaps( node.bounds, query, surfaces.mortar.dimension ) ) {
      continue;
    }
    if ( node.element >= 0 ) {
      const Index mortar = node.element;
      if ( self_contact &&
           ( mortar >= nonmortar || ( exclude_adjacent && sharesNode( surfaces.mortar, mortar, nonmortar ) ) ) ) {
        continue;
      }
      if constexpr ( Fill ) {
        output[count] = { mortar, nonmortar };
      }
      ++count;
      continue;
    }
    if ( node.right >= 0 ) {
      stack[stack_size++] = node.right;
    }
    if ( node.left >= 0 ) {
      stack[stack_size++] = node.left;
    }
  }
  return count;
}

__global__ void countCandidatesKernel( SurfacePairView surfaces, const DeviceBvhNode* nodes, Index root, Real expansion,
                                       Real proximity_scale, bool self_contact, bool exclude_adjacent, Index* counts )
{
  const Index nonmortar = static_cast<Index>( blockIdx.x * blockDim.x + threadIdx.x );
  if ( nonmortar < surfaces.nonmortar.numberOfElements() ) {
    const DeviceBounds query = elementBounds( surfaces.nonmortar, nonmortar, expansion, proximity_scale );
    counts[nonmortar] =
        traverseBvh<false>( nodes, root, query, surfaces, nonmortar, self_contact, exclude_adjacent, nullptr );
  }
}

__global__ void fillCandidatesKernel( SurfacePairView surfaces, const DeviceBvhNode* nodes, Index root, Real expansion,
                                      Real proximity_scale, bool self_contact, bool exclude_adjacent,
                                      const Index* offsets, ElementPair* candidates )
{
  const Index nonmortar = static_cast<Index>( blockIdx.x * blockDim.x + threadIdx.x );
  if ( nonmortar < surfaces.nonmortar.numberOfElements() ) {
    const DeviceBounds query = elementBounds( surfaces.nonmortar, nonmortar, expansion, proximity_scale );
    traverseBvh<true>( nodes, root, query, surfaces, nonmortar, self_contact, exclude_adjacent,
                       candidates + offsets[nonmortar] );
  }
}

__global__ void setZeroKernel( Index* value ) { *value = 0; }

template <bool MortarKey>
__global__ void pairKeysKernel( const ElementPair* pairs, Index count, Index* keys )
{
  const Index index = static_cast<Index>( blockIdx.x * blockDim.x + threadIdx.x );
  if ( index < count ) {
    keys[index] = MortarKey ? pairs[index].mortar_element : pairs[index].nonmortar_element;
  }
}
