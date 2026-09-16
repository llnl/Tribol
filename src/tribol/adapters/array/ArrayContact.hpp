#ifndef TRIBOL_ADAPTERS_ARRAY_ARRAYCONTACT_HPP_
#define TRIBOL_ADAPTERS_ARRAY_ARRAYCONTACT_HPP_

#include "tribol/contact/Contact.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace tribol::array {

class UniformSurface {
 public:
  UniformSurface( int dimension, ElementTopology topology, ArrayView<const Real> coordinates,
                  ArrayView<const Index> connectivity, FieldLayout layout = FieldLayout::Interleaved,
                  int attribute = 1 )
      : dimension_( dimension ),
        topology_( topology ),
        coordinates_( coordinates ),
        connectivity_( connectivity ),
        layout_( layout )
  {
    const Index nodes_per_element = nodesPerElement( topology_ );
    if ( dimension_ < 2 || dimension_ > 3 || !coordinates_.isStructurallyValid() ||
         coordinates_.size() % dimension_ != 0 || !connectivity_.isStructurallyValid() ||
         connectivity_.size() % nodes_per_element != 0 ) {
      throw std::invalid_argument( "UniformSurface requires valid two- or three-dimensional linear surface arrays." );
    }
    const Index element_count = connectivity_.size() / nodes_per_element;
    offsets_.resize( static_cast<std::size_t>( element_count + 1 ) );
    topologies_.assign( static_cast<std::size_t>( element_count ), topology_ );
    attributes_.assign( static_cast<std::size_t>( element_count ), attribute );
    for ( Index element = 0; element <= element_count; ++element ) {
      offsets_[static_cast<std::size_t>( element )] = element * nodes_per_element;
    }
    if ( !view().isStructurallyValid() ) {
      throw std::invalid_argument( "UniformSurface connectivity references an invalid coordinate node." );
    }
  }

  void updateCoordinates( ArrayView<const Real> coordinates )
  {
    const Index expected = coordinates_.size();
    if ( !coordinates.isStructurallyValid() || coordinates.size() != expected ) {
      throw std::invalid_argument( "Updated UniformSurface coordinates must preserve the original shape." );
    }
    coordinates_ = coordinates;
  }

  [[nodiscard]] SurfaceMeshView view() const
  {
    return {
        .dimension = dimension_,
        .coordinates = { coordinates_, coordinates_.size() / dimension_, dimension_, layout_ },
        .element_offsets = { offsets_.data(), static_cast<Index>( offsets_.size() ) },
        .connectivity = connectivity_,
        .topologies = { topologies_.data(), static_cast<Index>( topologies_.size() ) },
        .attributes = { attributes_.data(), static_cast<Index>( attributes_.size() ) },
    };
  }

 private:
  static constexpr Index nodesPerElement( ElementTopology topology )
  {
    switch ( topology ) {
      case ElementTopology::Segment:
        return 2;
      case ElementTopology::Triangle:
        return 3;
      case ElementTopology::Quadrilateral:
        return 4;
    }
    return 0;
  }

  int dimension_{};
  ElementTopology topology_{};
  ArrayView<const Real> coordinates_{};
  ArrayView<const Index> connectivity_{};
  FieldLayout layout_{};
  std::vector<Index> offsets_;
  std::vector<ElementTopology> topologies_;
  std::vector<int> attributes_;
};

template <SupportedMethod MethodType = DefaultMethod, SearchPolicy Search = search::CartesianProduct,
          execution::Policy Execution = execution::Sequential>
  requires execution::SupportedContactExecution<MethodType, Execution>
class ArrayContact {
 public:
  using CoreContact = Contact<MethodType, Search, Execution>;
  using Options = typename CoreContact::Options;

  ArrayContact( UniformSurface mortar, UniformSurface nonmortar, Options options = {} )
      : mortar_( std::move( mortar ) ),
        nonmortar_( std::move( nonmortar ) ),
        core_( SurfacePairView{ mortar_.view(), nonmortar_.view() }, std::move( options ) )
  {
  }

  void updateInteractions() { core_.updateInteractions(); }

  void updateGeometry( ArrayView<const Real> mortar_coordinates, ArrayView<const Real> nonmortar_coordinates )
  {
    mortar_.updateCoordinates( mortar_coordinates );
    nonmortar_.updateCoordinates( nonmortar_coordinates );
    core_.updateGeometry( { mortar_.view(), nonmortar_.view() } );
  }

  [[nodiscard]] ContactResultView evaluate( const ContactStateView& state = {} ) const
  {
    return core_.evaluate( state );
  }

  [[nodiscard]] CoreContact& core() { return core_; }
  [[nodiscard]] const CoreContact& core() const { return core_; }

 private:
  UniformSurface mortar_;
  UniformSurface nonmortar_;
  CoreContact core_;
};

template <SupportedMethod MethodType = DefaultMethod, SearchPolicy Search = search::CartesianProduct,
          execution::Policy Execution = execution::Sequential>
  requires execution::SupportedContactExecution<MethodType, Execution>
ArrayContact<MethodType, Search, Execution> makeContact(
    UniformSurface mortar, UniformSurface nonmortar,
    typename ArrayContact<MethodType, Search, Execution>::Options options = {} )
{
  return ArrayContact<MethodType, Search, Execution>( std::move( mortar ), std::move( nonmortar ),
                                                      std::move( options ) );
}

}  // namespace tribol::array

#endif
