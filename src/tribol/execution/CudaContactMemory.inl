template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { cudaFree( data_ ); }

  DeviceBuffer( const DeviceBuffer& ) = delete;
  DeviceBuffer& operator=( const DeviceBuffer& ) = delete;

  void reserve( std::size_t capacity )
  {
    if ( capacity <= capacity_ ) {
      return;
    }
    T* replacement{};
    requireCuda( cudaMalloc( &replacement, capacity * sizeof( T ) ), "cudaMalloc device workspace" );
    cudaFree( data_ );
    data_ = replacement;
    capacity_ = capacity;
  }

  void resize( std::size_t size )
  {
    reserve( size );
    size_ = size;
  }

  void clear() { size_ = 0; }

  void swap( DeviceBuffer& other ) noexcept
  {
    std::swap( data_, other.data_ );
    std::swap( size_, other.size_ );
    std::swap( capacity_, other.capacity_ );
  }

  [[nodiscard]] T* data() { return data_; }
  [[nodiscard]] const T* data() const { return data_; }
  [[nodiscard]] std::size_t size() const { return size_; }

 private:
  T* data_{};
  std::size_t size_{};
  std::size_t capacity_{};
};

template <typename T>
void copyToDevice( DeviceBuffer<T>& destination, ArrayView<const T> source, const char* operation )
{
  destination.resize( static_cast<std::size_t>( source.size() ) );
  if ( !source.empty() ) {
    requireCuda(
        cudaMemcpy( destination.data(), source.data(), destination.size() * sizeof( T ), cudaMemcpyHostToDevice ),
        operation );
  }
}

template <typename T>
void copyToHost( ArrayView<T> destination, const DeviceBuffer<T>& source, const char* operation )
{
  if ( destination.size() != static_cast<Index>( source.size() ) ) {
    throw std::invalid_argument( std::string( operation ) + " has a mismatched destination size." );
  }
  if ( !destination.empty() ) {
    requireCuda( cudaMemcpy( destination.data(), source.data(), source.size() * sizeof( T ), cudaMemcpyDeviceToHost ),
                 operation );
  }
}

struct DeviceSurfaceStorage {
  int dimension{};
  Index nodes{};
  Index elements{};
  FieldLayout coordinate_layout{ FieldLayout::Interleaved };
  DeviceBuffer<Real> coordinates;
  DeviceBuffer<Index> element_offsets;
  DeviceBuffer<Index> connectivity;
  DeviceBuffer<ElementTopology> topologies;
  DeviceBuffer<int> attributes;

  void upload( const SurfaceMeshView& source )
  {
    dimension = source.dimension;
    nodes = source.numberOfNodes();
    elements = source.numberOfElements();
    coordinate_layout = source.coordinates.layout;
    copyToDevice( coordinates, source.coordinates.values, "copy surface coordinates to device" );
    copyToDevice( element_offsets, source.element_offsets, "copy surface offsets to device" );
    copyToDevice( connectivity, source.connectivity, "copy surface connectivity to device" );
    copyToDevice( topologies, source.topologies, "copy surface topologies to device" );
    copyToDevice( attributes, source.attributes, "copy surface attributes to device" );
  }

  void updateCoordinates( const SurfaceMeshView& source )
  {
    if ( source.dimension != dimension || source.numberOfNodes() != nodes || source.numberOfElements() != elements ||
         source.coordinates.values.size() != static_cast<Index>( coordinates.size() ) ) {
      throw std::invalid_argument( "CUDA geometry update does not match the resident surface topology." );
    }
    coordinate_layout = source.coordinates.layout;
    if ( !source.coordinates.values.empty() ) {
      requireCuda( cudaMemcpy( coordinates.data(), source.coordinates.values.data(),
                               coordinates.size() * sizeof( Real ), cudaMemcpyHostToDevice ),
                   "update device surface coordinates" );
    }
  }

  [[nodiscard]] SurfaceMeshView view() const
  {
    return {
        .dimension = dimension,
        .coordinates = { { coordinates.data(), static_cast<Index>( coordinates.size() ) },
                         nodes,
                         dimension,
                         coordinate_layout },
        .element_offsets = { element_offsets.data(), static_cast<Index>( element_offsets.size() ) },
        .connectivity = { connectivity.data(), static_cast<Index>( connectivity.size() ) },
        .topologies = { topologies.data(), static_cast<Index>( topologies.size() ) },
        .attributes = { attributes.data(), static_cast<Index>( attributes.size() ) },
    };
  }
};

struct DeviceStateStorage {
  DeviceBuffer<Real> mortar_velocity;
  DeviceBuffer<Real> nonmortar_velocity;
  DeviceBuffer<Real> mortar_thickness;
  DeviceBuffer<Real> nonmortar_thickness;

  void reserve( const SurfacePairView& surfaces )
  {
    mortar_velocity.resize( static_cast<std::size_t>( surfaces.mortar.coordinates.values.size() ) );
    nonmortar_velocity.resize( static_cast<std::size_t>( surfaces.nonmortar.coordinates.values.size() ) );
    mortar_thickness.resize( static_cast<std::size_t>( surfaces.mortar.numberOfElements() ) );
    nonmortar_thickness.resize( static_cast<std::size_t>( surfaces.nonmortar.numberOfElements() ) );
  }

  [[nodiscard]] ContactStateView upload( const ContactStateView& state, const SurfacePairView& surfaces,
                                         bool need_velocity, bool need_thickness )
  {
    ContactStateView result;
    if ( need_velocity ) {
      requireField( state.mortar_velocity, surfaces.mortar, "CUDA timestep voting requires mortar velocity." );
      requireField( state.nonmortar_velocity, surfaces.nonmortar, "CUDA timestep voting requires nonmortar velocity." );
      requireCuda( cudaMemcpy( mortar_velocity.data(), state.mortar_velocity.values.data(),
                               mortar_velocity.size() * sizeof( Real ), cudaMemcpyHostToDevice ),
                   "copy mortar velocity to device" );
      requireCuda( cudaMemcpy( nonmortar_velocity.data(), state.nonmortar_velocity.values.data(),
                               nonmortar_velocity.size() * sizeof( Real ), cudaMemcpyHostToDevice ),
                   "copy nonmortar velocity to device" );
      result.mortar_velocity = { { mortar_velocity.data(), static_cast<Index>( mortar_velocity.size() ) },
                                 surfaces.mortar.numberOfNodes(),
                                 surfaces.mortar.dimension,
                                 state.mortar_velocity.layout };
      result.nonmortar_velocity = { { nonmortar_velocity.data(), static_cast<Index>( nonmortar_velocity.size() ) },
                                    surfaces.nonmortar.numberOfNodes(),
                                    surfaces.nonmortar.dimension,
                                    state.nonmortar_velocity.layout };
    }
    if ( need_thickness ) {
      requireArray( state.mortar_element_thickness, surfaces.mortar.numberOfElements(),
                    "CUDA contact requires mortar element thickness." );
      requireArray( state.nonmortar_element_thickness, surfaces.nonmortar.numberOfElements(),
                    "CUDA contact requires nonmortar element thickness." );
      requireCuda( cudaMemcpy( mortar_thickness.data(), state.mortar_element_thickness.data(),
                               mortar_thickness.size() * sizeof( Real ), cudaMemcpyHostToDevice ),
                   "copy mortar thickness to device" );
      requireCuda( cudaMemcpy( nonmortar_thickness.data(), state.nonmortar_element_thickness.data(),
                               nonmortar_thickness.size() * sizeof( Real ), cudaMemcpyHostToDevice ),
                   "copy nonmortar thickness to device" );
      result.mortar_element_thickness = { mortar_thickness.data(), static_cast<Index>( mortar_thickness.size() ) };
      result.nonmortar_element_thickness = { nonmortar_thickness.data(),
                                             static_cast<Index>( nonmortar_thickness.size() ) };
    }
    return result;
  }

 private:
  static void requireField( const FieldView<const Real>& field, const SurfaceMeshView& surface, const char* message )
  {
    if ( !field.isStructurallyValid() || field.entities != surface.numberOfNodes() ||
         field.components != surface.dimension ) {
      throw std::invalid_argument( message );
    }
  }

  static void requireArray( ArrayView<const Real> values, Index expected, const char* message )
  {
    if ( !values.isStructurallyValid() || values.size() != expected ) {
      throw std::invalid_argument( message );
    }
  }
};
