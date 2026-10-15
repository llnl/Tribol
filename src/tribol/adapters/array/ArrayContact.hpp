#ifndef TRIBOL_ADAPTERS_ARRAY_ARRAYCONTACT_HPP_
#define TRIBOL_ADAPTERS_ARRAY_ARRAYCONTACT_HPP_

#include "tribol/contact/Contact.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tribol::array {

class SingleTopologySurface {
 public:
  SingleTopologySurface( int dimension, ElementTopology topology, ArrayView<const Real> coordinates,
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
      throw std::invalid_argument(
          "SingleTopologySurface requires valid two- or three-dimensional linear surface arrays." );
    }
    const Index element_count = connectivity_.size() / nodes_per_element;
    offsets_.resize( static_cast<std::size_t>( element_count + 1 ) );
    topologies_.assign( static_cast<std::size_t>( element_count ), topology_ );
    attributes_.assign( static_cast<std::size_t>( element_count ), attribute );
    parent_face_ids_.resize( static_cast<std::size_t>( element_count ) );
    parent_face_owners_.assign( static_cast<std::size_t>( element_count ), 0 );
    parent_face_orders_.assign( static_cast<std::size_t>( element_count ), 1 );
    parent_dof_signs_.assign( static_cast<std::size_t>( connectivity_.size() ), 1.0 );
    parent_basis_offsets_.resize( static_cast<std::size_t>( element_count + 1 ) );
    parent_basis_coefficients_.reserve(
        static_cast<std::size_t>( element_count * nodes_per_element * nodes_per_element ) );
    subfacet_parent_faces_.resize( static_cast<std::size_t>( element_count ) );
    parent_reference_coordinates_.reserve( static_cast<std::size_t>( connectivity_.size() * ( dimension_ - 1 ) ) );
    for ( Index element = 0; element <= element_count; ++element ) {
      offsets_[static_cast<std::size_t>( element )] = element * nodes_per_element;
      parent_basis_offsets_[static_cast<std::size_t>( element )] = element * nodes_per_element * nodes_per_element;
      if ( element == element_count ) {
        continue;
      }
      parent_face_ids_[static_cast<std::size_t>( element )] = static_cast<std::uint64_t>( element );
      subfacet_parent_faces_[static_cast<std::size_t>( element )] = element;
      appendBasisCoefficients();
      appendReferenceCoordinates();
    }
    if ( !view().isStructurallyValid() ) {
      throw std::invalid_argument( "SingleTopologySurface connectivity references an invalid coordinate node." );
    }
  }

  void updateCoordinates( ArrayView<const Real> coordinates )
  {
    const Index expected = coordinates_.size();
    if ( !coordinates.isStructurallyValid() || coordinates.size() != expected ) {
      throw std::invalid_argument( "Updated SingleTopologySurface coordinates must preserve the original shape." );
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

  [[nodiscard]] ParentTraceView parentTraceView() const
  {
    return {
        .geometry = view(),
        .parent_face_ids = { parent_face_ids_.data(), static_cast<Index>( parent_face_ids_.size() ) },
        .parent_face_owners = { parent_face_owners_.data(), static_cast<Index>( parent_face_owners_.size() ) },
        .parent_face_topologies = { topologies_.data(), static_cast<Index>( topologies_.size() ) },
        .parent_face_orders = { parent_face_orders_.data(), static_cast<Index>( parent_face_orders_.size() ) },
        .parent_dof_offsets = { offsets_.data(), static_cast<Index>( offsets_.size() ) },
        .parent_dofs = connectivity_,
        .parent_dof_signs = { parent_dof_signs_.data(), static_cast<Index>( parent_dof_signs_.size() ) },
        .parent_basis_offsets = { parent_basis_offsets_.data(), static_cast<Index>( parent_basis_offsets_.size() ) },
        .parent_basis_coefficients = { parent_basis_coefficients_.data(),
                                       static_cast<Index>( parent_basis_coefficients_.size() ) },
        .subfacet_parent_faces = { subfacet_parent_faces_.data(), static_cast<Index>( subfacet_parent_faces_.size() ) },
        .subfacet_parent_reference_coordinates = { { parent_reference_coordinates_.data(),
                                                     static_cast<Index>( parent_reference_coordinates_.size() ) },
                                                   connectivity_.size(),
                                                   dimension_ - 1,
                                                   FieldLayout::Interleaved },
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

  void appendBasisCoefficients()
  {
    switch ( topology_ ) {
      case ElementTopology::Segment:
        parent_basis_coefficients_.insert( parent_basis_coefficients_.end(), { 1.0, -1.0, 0.0, 1.0 } );
        return;
      case ElementTopology::Triangle:
        parent_basis_coefficients_.insert( parent_basis_coefficients_.end(),
                                           { 1.0, -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 } );
        return;
      case ElementTopology::Quadrilateral:
        parent_basis_coefficients_.insert(
            parent_basis_coefficients_.end(),
            { 1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0 } );
        return;
    }
  }

  void appendReferenceCoordinates()
  {
    switch ( topology_ ) {
      case ElementTopology::Segment:
        parent_reference_coordinates_.insert( parent_reference_coordinates_.end(), { 0.0, 1.0 } );
        return;
      case ElementTopology::Triangle:
        parent_reference_coordinates_.insert( parent_reference_coordinates_.end(), { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 } );
        return;
      case ElementTopology::Quadrilateral:
        parent_reference_coordinates_.insert( parent_reference_coordinates_.end(),
                                              { 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 } );
        return;
    }
  }

  int dimension_{};
  ElementTopology topology_{};
  ArrayView<const Real> coordinates_{};
  ArrayView<const Index> connectivity_{};
  FieldLayout layout_{};
  std::vector<Index> offsets_;
  std::vector<ElementTopology> topologies_;
  std::vector<int> attributes_;
  std::vector<std::uint64_t> parent_face_ids_;
  std::vector<int> parent_face_owners_;
  std::vector<int> parent_face_orders_;
  std::vector<Real> parent_dof_signs_;
  std::vector<Index> parent_basis_offsets_;
  std::vector<Real> parent_basis_coefficients_;
  std::vector<Index> subfacet_parent_faces_;
  std::vector<Real> parent_reference_coordinates_;
};

template <SupportedMethod MethodType = DefaultMethod, SearchPolicy Search = search::CartesianProduct,
          execution::Policy Execution = execution::Sequential>
  requires execution::SupportedContactExecution<MethodType, Execution> &&
           execution::SupportedContactSearch<Search, Execution>
class ArrayContact {
 public:
  using CoreContact = Contact<MethodType, Search, Execution>;
  using Options = typename CoreContact::Options;

  ArrayContact( SingleTopologySurface mortar, SingleTopologySurface nonmortar, Options options = {} )
      : mortar_( std::move( mortar ) ),
        nonmortar_( std::move( nonmortar ) ),
        core_( makeCoreContact( mortar_, nonmortar_, std::move( options ) ) )
  {
  }

  void updateInteractions() { core_.updateInteractions(); }

  void updateGeometry( ArrayView<const Real> mortar_coordinates, ArrayView<const Real> nonmortar_coordinates )
  {
    mortar_.updateCoordinates( mortar_coordinates );
    nonmortar_.updateCoordinates( nonmortar_coordinates );
    if constexpr ( detail::is_native_explicit_method_v<MethodType> ) {
      core_.updateGeometry( { mortar_.parentTraceView(), nonmortar_.parentTraceView() }, parentFields() );
    } else {
      core_.updateGeometry( { mortar_.view(), nonmortar_.view() } );
    }
  }

  [[nodiscard]] ContactResultView evaluate( const ContactStateView& state = {} ) const
  {
    return core_.evaluate( state );
  }

  [[nodiscard]] ContactResultView evaluate( const ContactStateView& state,
                                            timestep::DiagonalInverseMassView inverse_mass,
                                            timestep::IntegratorStabilityLimits integrator ) const
  {
    return core_.evaluate( state, inverse_mass, integrator );
  }

  [[nodiscard]] CoreContact& core() { return core_; }
  [[nodiscard]] const CoreContact& core() const { return core_; }

 private:
  static CoreContact makeCoreContact( const SingleTopologySurface& mortar, const SingleTopologySurface& nonmortar,
                                      Options options )
  {
    if constexpr ( detail::is_native_explicit_method_v<MethodType> ) {
      return CoreContact(
          { mortar.parentTraceView(), nonmortar.parentTraceView() },
          { .mortar_coordinates = mortar.view().coordinates, .nonmortar_coordinates = nonmortar.view().coordinates },
          std::move( options ) );
    } else {
      return CoreContact( { mortar.view(), nonmortar.view() }, std::move( options ) );
    }
  }

  [[nodiscard]] ParentTraceFieldPairView parentFields() const
  {
    return {
        .mortar_coordinates = mortar_.view().coordinates,
        .nonmortar_coordinates = nonmortar_.view().coordinates,
    };
  }

  SingleTopologySurface mortar_;
  SingleTopologySurface nonmortar_;
  CoreContact core_;
};

template <SupportedMethod MethodType = DefaultMethod, SearchPolicy Search = search::CartesianProduct,
          execution::Policy Execution = execution::Sequential>
  requires execution::SupportedContactExecution<MethodType, Execution> &&
           execution::SupportedContactSearch<Search, Execution>
ArrayContact<MethodType, Search, Execution> makeContact(
    SingleTopologySurface mortar, SingleTopologySurface nonmortar,
    typename ArrayContact<MethodType, Search, Execution>::Options options = {} )
{
  return ArrayContact<MethodType, Search, Execution>( std::move( mortar ), std::move( nonmortar ),
                                                      std::move( options ) );
}

}  // namespace tribol::array

#endif
