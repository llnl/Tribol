#ifndef TRIBOL_CONTACT_CONTACT_HPP_
#define TRIBOL_CONTACT_CONTACT_HPP_

#include "tribol/core/ExactTangent.hpp"
#include "tribol/core/MeshView.hpp"
#include "tribol/core/Version.hpp"
#include "tribol/evaluation/PolicyEvaluator.hpp"
#include "tribol/evaluation/State.hpp"
#include "tribol/execution/ContactCuda.hpp"
#include "tribol/execution/Execution.hpp"
#include "tribol/method/Traits.hpp"
#include "tribol/method/Validation.hpp"
#include "tribol/search/Search.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tribol {

template <SupportedMethod MethodType = DefaultMethod, SearchPolicy Search = search::CartesianProduct,
          execution::Policy Execution = execution::Sequential>
  requires execution::SupportedContactExecution<MethodType, Execution>
class Contact {
 public:
  using method_type = MethodType;
  using search_type = Search;
  using execution_type = Execution;

  struct Options {
    typename MethodType::Parameters method{};
    typename Search::Parameters search{};
    bool self_contact{};
    bool exclude_adjacent_self_contact{ true };
  };

  explicit Contact( SurfacePairView surfaces, Options options = {} )
      : surfaces_( surfaces ), options_( std::move( options ) ), search_( options_.search )
  {
    requireValidSurfaces( surfaces_ );
    validateMethodParameters<MethodType>( options_.method );
    resizeLinearizationWorkspace();
    resizeEvaluationWorkspace();
  }

  explicit Contact( SurfaceMeshView surface, Options options = {} )
      : Contact( SurfacePairView{ surface, surface }, withSelfContact( std::move( options ) ) )
  {
  }

  Contact( const Contact& ) = delete;
  Contact& operator=( const Contact& ) = delete;
  Contact( Contact&& ) = delete;
  Contact& operator=( Contact&& ) = delete;

  void updateInteractions()
  {
    search_.findCandidates( surfaces_, candidates_ );
    if ( options_.self_contact ) {
      candidates_.filterSelfContact( surfaces_.mortar, options_.exclude_adjacent_self_contact );
    }
    interaction_geometry_version_ = geometry_version_;
    interaction_version_.advance();
    has_interactions_ = true;
    resizeInteractionWorkspace();
  }

  void updateGeometry( SurfacePairView surfaces )
  {
    requireValidSurfaces( surfaces );
    requireSameTopology( surfaces_, surfaces );
    surfaces_ = surfaces;
    geometry_version_.advance();
  }

  void rebuildGeometry( SurfacePairView surfaces )
  {
    requireValidSurfaces( surfaces );
    surfaces_ = surfaces;
    candidates_.clear();
    geometry_version_.advance();
    has_interactions_ = false;
    resizeLinearizationWorkspace();
    resizeEvaluationWorkspace();
  }

  EvaluationSummary addResidual( const ContactStateView& state, ContactResidualView residual ) const
  {
    if ( !has_interactions_ ) {
      throw std::logic_error( "updateInteractions() must be called before contact evaluation." );
    }
    return evaluateConfigured( surfaces_, state, ContactOutputView{ .residual = residual } );
  }

  [[nodiscard]] ContactResultView evaluate( const ContactStateView& state = {} ) const
  {
    if ( !has_interactions_ ) {
      throw std::logic_error( "updateInteractions() must be called before contact evaluation." );
    }
    clearEvaluationWorkspace();
    const ContactOutputView output{
        .residual =
            workspaceResidual( result_mortar_force_, result_nonmortar_force_, mutableView( result_constraint_ ) ),
        .gap = mutableView( result_gap_ ),
        .weighted_gap = mutableView( result_weighted_gap_ ),
        .tributary_area = mutableView( result_tributary_area_ ),
        .mortar_weights = mutableView( result_mortar_weights_ ),
        .quadrature_gap = mutableView( result_quadrature_gap_ ),
        .quadrature_pressure = mutableView( result_quadrature_pressure_ ),
    };
    const auto summary = evaluateConfigured( surfaces_, state, output );
    return {
        .mortar_force = constField( result_mortar_force_, surfaces_.mortar ),
        .nonmortar_force = constField( result_nonmortar_force_, surfaces_.nonmortar ),
        .constraint_residual = constView( result_constraint_ ),
        .gap = constView( result_gap_ ),
        .weighted_gap = constView( result_weighted_gap_ ),
        .tributary_area = constView( result_tributary_area_ ),
        .mortar_weights = constView( result_mortar_weights_ ),
        .quadrature_gap = { result_quadrature_gap_.data(), summary.quadrature_points },
        .quadrature_pressure = { result_quadrature_pressure_.data(), summary.quadrature_points },
        .summary = summary,
        .geometry_version = geometry_version_,
        .interaction_version = interaction_version_,
    };
  }

  void applyCoordinateDerivative( const ContactStateView& state, ContactDirectionView direction,
                                  ContactResidualView derivative ) const
  {
    if ( !has_interactions_ ) {
      throw std::logic_error( "updateInteractions() must be called before contact linearization." );
    }
    requireDirection( direction );
    seedCoordinates( surfaces_.mortar, direction.mortar, exact_mortar_coordinates_ );
    seedCoordinates( surfaces_.nonmortar, direction.nonmortar, exact_nonmortar_coordinates_ );
    std::fill( exact_mortar_residual_.begin(), exact_mortar_residual_.end(), linearization_detail::ExactTangent{} );
    std::fill( exact_nonmortar_residual_.begin(), exact_nonmortar_residual_.end(),
               linearization_detail::ExactTangent{} );
    std::fill( exact_constraint_residual_.begin(), exact_constraint_residual_.end(),
               linearization_detail::ExactTangent{} );

    const auto exact_surfaces = seededSurfaces();
    evaluateMethod<MethodType>( exact_surfaces, candidates_.view(), options_.method, state,
                                ContactOutputViewT<linearization_detail::ExactTangent>{
                                    .residual = workspaceResidual( exact_mortar_residual_, exact_nonmortar_residual_,
                                                                   mutableView( exact_constraint_residual_ ) ) } );
    addTangents( exact_mortar_residual_, derivative.mortar );
    addTangents( exact_nonmortar_residual_, derivative.nonmortar );
    addTangents( exact_constraint_residual_, derivative.constraint );
  }

  [[nodiscard]] DenseMatrixView assembleCoordinateJacobian( const ContactStateView& state ) const
  {
    if ( !has_interactions_ ) {
      throw std::logic_error( "updateInteractions() must be called before contact linearization." );
    }
    const Index mortar_values = surfaces_.mortar.numberOfNodes() * surfaces_.mortar.dimension;
    const Index nonmortar_values = surfaces_.nonmortar.numberOfNodes() * surfaces_.nonmortar.dimension;
    const Index constraint_values =
        MethodTraits<MethodType>::capabilities.produces_constraint_residual ? surfaces_.mortar.numberOfNodes() : 0;
    const Index columns = mortar_values + nonmortar_values;
    const Index rows = columns + constraint_values;
    std::fill( assembled_jacobian_.begin(), assembled_jacobian_.end(), 0.0 );
    for ( Index column = 0; column < columns; ++column ) {
      std::fill( direction_mortar_.begin(), direction_mortar_.end(), 0.0 );
      std::fill( direction_nonmortar_.begin(), direction_nonmortar_.end(), 0.0 );
      std::fill( derivative_mortar_.begin(), derivative_mortar_.end(), 0.0 );
      std::fill( derivative_nonmortar_.begin(), derivative_nonmortar_.end(), 0.0 );
      std::fill( derivative_constraint_.begin(), derivative_constraint_.end(), 0.0 );
      if ( column < mortar_values ) {
        direction_mortar_[static_cast<std::size_t>( column )] = 1.0;
      } else {
        direction_nonmortar_[static_cast<std::size_t>( column - mortar_values )] = 1.0;
      }
      applyCoordinateDerivative(
          state,
          ContactDirectionView{ .mortar = constField( direction_mortar_, surfaces_.mortar ),
                                .nonmortar = constField( direction_nonmortar_, surfaces_.nonmortar ) },
          workspaceResidual( derivative_mortar_, derivative_nonmortar_, mutableView( derivative_constraint_ ) ) );
      for ( Index row = 0; row < mortar_values; ++row ) {
        assembled_jacobian_[static_cast<std::size_t>( row * columns + column )] =
            derivative_mortar_[static_cast<std::size_t>( row )];
      }
      for ( Index row = 0; row < nonmortar_values; ++row ) {
        assembled_jacobian_[static_cast<std::size_t>( ( mortar_values + row ) * columns + column )] =
            derivative_nonmortar_[static_cast<std::size_t>( row )];
      }
      for ( Index row = 0; row < constraint_values; ++row ) {
        assembled_jacobian_[static_cast<std::size_t>( ( columns + row ) * columns + column )] =
            derivative_constraint_[static_cast<std::size_t>( row )];
      }
    }
    return { constView( assembled_jacobian_ ), rows, columns };
  }

  [[nodiscard]] const SurfacePairView& surfaces() const { return surfaces_; }
  [[nodiscard]] const Options& options() const { return options_; }
  [[nodiscard]] ArrayView<const ElementPair> interactions() const { return candidates_.view(); }
  [[nodiscard]] bool hasInteractions() const { return has_interactions_; }
  [[nodiscard]] GeometryVersion geometryVersion() const { return geometry_version_; }
  [[nodiscard]] GeometryVersion interactionGeometryVersion() const { return interaction_geometry_version_; }
  [[nodiscard]] InteractionVersion interactionVersion() const { return interaction_version_; }

 private:
  static Options withSelfContact( Options options )
  {
    options.self_contact = true;
    return options;
  }

  static void requireValidSurfaces( const SurfacePairView& surfaces )
  {
    if ( !surfaces.isStructurallyValid() ) {
      throw std::invalid_argument( "Contact requires two structurally valid surface meshes with equal dimensions." );
    }
  }

  static bool equalArray( ArrayView<const Index> left, ArrayView<const Index> right )
  {
    if ( left.size() != right.size() ) {
      return false;
    }
    for ( Index index = 0; index < left.size(); ++index ) {
      if ( left[index] != right[index] ) {
        return false;
      }
    }
    return true;
  }

  static bool equalArray( ArrayView<const ElementTopology> left, ArrayView<const ElementTopology> right )
  {
    if ( left.size() != right.size() ) {
      return false;
    }
    for ( Index index = 0; index < left.size(); ++index ) {
      if ( left[index] != right[index] ) {
        return false;
      }
    }
    return true;
  }

  static bool sameTopology( const SurfaceMeshView& left, const SurfaceMeshView& right )
  {
    return left.dimension == right.dimension && left.numberOfNodes() == right.numberOfNodes() &&
           equalArray( left.element_offsets, right.element_offsets ) &&
           equalArray( left.connectivity, right.connectivity ) && equalArray( left.topologies, right.topologies );
  }

  static void requireSameTopology( const SurfacePairView& left, const SurfacePairView& right )
  {
    if ( !sameTopology( left.mortar, right.mortar ) || !sameTopology( left.nonmortar, right.nonmortar ) ) {
      throw std::invalid_argument( "updateGeometry() cannot change mesh topology; use rebuildGeometry()." );
    }
  }

  void resizeLinearizationWorkspace()
  {
    const auto mortar_values =
        static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() * surfaces_.mortar.dimension );
    const auto nonmortar_values =
        static_cast<std::size_t>( surfaces_.nonmortar.numberOfNodes() * surfaces_.nonmortar.dimension );
    exact_mortar_coordinates_.resize( mortar_values );
    exact_nonmortar_coordinates_.resize( nonmortar_values );
    exact_mortar_residual_.resize( mortar_values );
    exact_nonmortar_residual_.resize( nonmortar_values );
    exact_constraint_residual_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    direction_mortar_.resize( mortar_values );
    direction_nonmortar_.resize( nonmortar_values );
    derivative_mortar_.resize( mortar_values );
    derivative_nonmortar_.resize( nonmortar_values );
    derivative_constraint_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    const auto columns = mortar_values + nonmortar_values;
    const auto maximum_rows = columns + static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() );
    assembled_jacobian_.resize( maximum_rows * columns );
  }

  void resizeEvaluationWorkspace()
  {
    const auto mortar_nodes = static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() );
    const auto nonmortar_nodes = static_cast<std::size_t>( surfaces_.nonmortar.numberOfNodes() );
    result_mortar_force_.resize( mortar_nodes * static_cast<std::size_t>( surfaces_.mortar.dimension ) );
    result_nonmortar_force_.resize( nonmortar_nodes * static_cast<std::size_t>( surfaces_.nonmortar.dimension ) );
    result_constraint_.resize( mortar_nodes );
    result_gap_.resize( mortar_nodes );
    result_weighted_gap_.resize( mortar_nodes );
    result_tributary_area_.resize( mortar_nodes );
    result_mortar_weights_.resize( mortar_nodes * nonmortar_nodes );
    result_quadrature_gap_.clear();
    result_quadrature_pressure_.clear();
  }

  void resizeInteractionWorkspace()
  {
    const auto maximum_points = static_cast<std::size_t>( candidates_.size() ) * integration::maximumQuadraturePoints;
    result_quadrature_gap_.resize( maximum_points );
    result_quadrature_pressure_.resize( maximum_points );
    if constexpr ( std::same_as<Execution, execution::Cuda> ) {
      execution_patches_.resize( static_cast<std::size_t>( candidates_.size() ) );
      execution_contributions_.resize( static_cast<std::size_t>( candidates_.size() ) );
      execution_workspace_.reserve( candidates_.size() );
    }
  }

  EvaluationSummary evaluateConfigured( const SurfacePairView& surfaces, const ContactStateView& state,
                                        ContactOutputView output ) const
  {
    if constexpr ( std::same_as<Execution, execution::Cuda> ) {
      (void)state;
      return execution::evaluateDefaultContact( surfaces, candidates_.view(), options_.method, output,
                                                execution_workspace_, execution_patches_, execution_contributions_ );
    } else {
      return evaluateMethod<MethodType>( surfaces, candidates_.view(), options_.method, state, output );
    }
  }

  void clearEvaluationWorkspace() const
  {
    auto clear = []( std::vector<Real>& values ) { std::fill( values.begin(), values.end(), 0.0 ); };
    clear( result_mortar_force_ );
    clear( result_nonmortar_force_ );
    clear( result_constraint_ );
    clear( result_gap_ );
    clear( result_weighted_gap_ );
    clear( result_tributary_area_ );
    clear( result_mortar_weights_ );
    clear( result_quadrature_gap_ );
    clear( result_quadrature_pressure_ );
  }

  void requireDirection( ContactDirectionView direction ) const
  {
    if ( !direction.mortar.isStructurallyValid() || !direction.nonmortar.isStructurallyValid() ||
         direction.mortar.entities != surfaces_.mortar.numberOfNodes() ||
         direction.nonmortar.entities != surfaces_.nonmortar.numberOfNodes() ||
         direction.mortar.components != surfaces_.mortar.dimension ||
         direction.nonmortar.components != surfaces_.nonmortar.dimension ) {
      throw std::invalid_argument( "Coordinate direction must match the two surface coordinate fields." );
    }
  }

  static void seedCoordinates( const SurfaceMeshView& mesh, const FieldView<const Real>& direction,
                               std::vector<linearization_detail::ExactTangent>& coordinates )
  {
    for ( Index node = 0; node < mesh.numberOfNodes(); ++node ) {
      for ( int component = 0; component < mesh.dimension; ++component ) {
        coordinates[static_cast<std::size_t>( node * mesh.dimension + component )] = {
            mesh.coordinates( node, component ), direction( node, component ) };
      }
    }
  }

  static SurfaceMeshViewT<linearization_detail::ExactTangent> withSeededCoordinates(
      const SurfaceMeshView& mesh, const std::vector<linearization_detail::ExactTangent>& coordinates )
  {
    return {
        .dimension = mesh.dimension,
        .coordinates = { ArrayView<const linearization_detail::ExactTangent>{
                             coordinates.data(), static_cast<Index>( coordinates.size() ) },
                         mesh.numberOfNodes(), mesh.dimension, FieldLayout::Interleaved },
        .element_offsets = mesh.element_offsets,
        .connectivity = mesh.connectivity,
        .topologies = mesh.topologies,
        .attributes = mesh.attributes,
    };
  }

  SurfacePairViewT<linearization_detail::ExactTangent> seededSurfaces() const
  {
    return { withSeededCoordinates( surfaces_.mortar, exact_mortar_coordinates_ ),
             withSeededCoordinates( surfaces_.nonmortar, exact_nonmortar_coordinates_ ) };
  }

  template <typename Scalar>
  ContactResidualViewT<Scalar> workspaceResidual( std::vector<Scalar>& mortar, std::vector<Scalar>& nonmortar ) const
  {
    return workspaceResidual( mortar, nonmortar, {} );
  }

  template <typename Scalar>
  ContactResidualViewT<Scalar> workspaceResidual( std::vector<Scalar>& mortar, std::vector<Scalar>& nonmortar,
                                                  ArrayView<Scalar> constraint ) const
  {
    return {
        .mortar =
            FieldView<Scalar>{ ArrayView<Scalar>{ mortar.data(), static_cast<Index>( mortar.size() ) },
                               surfaces_.mortar.numberOfNodes(), surfaces_.mortar.dimension, FieldLayout::Interleaved },
        .nonmortar = FieldView<Scalar>{ ArrayView<Scalar>{ nonmortar.data(), static_cast<Index>( nonmortar.size() ) },
                                        surfaces_.nonmortar.numberOfNodes(), surfaces_.nonmortar.dimension,
                                        FieldLayout::Interleaved },
        .constraint = constraint,
    };
  }

  template <typename Scalar>
  static ArrayView<Scalar> mutableView( std::vector<Scalar>& values )
  {
    return { values.data(), static_cast<Index>( values.size() ) };
  }

  static ArrayView<const Real> constView( const std::vector<Real>& values )
  {
    return { values.data(), static_cast<Index>( values.size() ) };
  }

  static FieldView<const Real> constField( const std::vector<Real>& values, const SurfaceMeshView& mesh )
  {
    return { constView( values ), mesh.numberOfNodes(), mesh.dimension, FieldLayout::Interleaved };
  }

  static void addTangents( const std::vector<linearization_detail::ExactTangent>& values, FieldView<Real> output )
  {
    if ( !output.isStructurallyValid() || output.values.size() != static_cast<Index>( values.size() ) ) {
      throw std::invalid_argument( "Coordinate derivative output must match its surface coordinate field." );
    }
    for ( Index entity = 0; entity < output.entities; ++entity ) {
      for ( int component = 0; component < output.components; ++component ) {
        const auto workspace_index = static_cast<std::size_t>( entity * output.components + component );
        output( entity, component ) += linearization_detail::tangent( values[workspace_index] );
      }
    }
  }

  static void addTangents( const std::vector<linearization_detail::ExactTangent>& values, ArrayView<Real> output )
  {
    if ( output.empty() ) {
      return;
    }
    if ( !output.isStructurallyValid() || output.size() != static_cast<Index>( values.size() ) ) {
      throw std::invalid_argument( "Constraint derivative output must match the mortar constraint field." );
    }
    for ( Index entry = 0; entry < output.size(); ++entry ) {
      output[entry] += linearization_detail::tangent( values[static_cast<std::size_t>( entry )] );
    }
  }

  SurfacePairView surfaces_;
  Options options_;
  Search search_;
  CandidatePairs candidates_;
  GeometryVersion geometry_version_{};
  GeometryVersion interaction_geometry_version_{};
  InteractionVersion interaction_version_{};
  bool has_interactions_{};
  mutable std::vector<linearization_detail::ExactTangent> exact_mortar_coordinates_;
  mutable std::vector<linearization_detail::ExactTangent> exact_nonmortar_coordinates_;
  mutable std::vector<linearization_detail::ExactTangent> exact_mortar_residual_;
  mutable std::vector<linearization_detail::ExactTangent> exact_nonmortar_residual_;
  mutable std::vector<linearization_detail::ExactTangent> exact_constraint_residual_;
  mutable std::vector<Real> direction_mortar_;
  mutable std::vector<Real> direction_nonmortar_;
  mutable std::vector<Real> derivative_mortar_;
  mutable std::vector<Real> derivative_nonmortar_;
  mutable std::vector<Real> derivative_constraint_;
  mutable std::vector<Real> assembled_jacobian_;
  mutable std::vector<Real> result_mortar_force_;
  mutable std::vector<Real> result_nonmortar_force_;
  mutable std::vector<Real> result_constraint_;
  mutable std::vector<Real> result_gap_;
  mutable std::vector<Real> result_weighted_gap_;
  mutable std::vector<Real> result_tributary_area_;
  mutable std::vector<Real> result_mortar_weights_;
  mutable std::vector<Real> result_quadrature_gap_;
  mutable std::vector<Real> result_quadrature_pressure_;
  using ExecutionWorkspace = std::conditional_t<std::same_as<Execution, execution::Cuda>,
                                                execution::CudaPenaltyWorkspace, execution::HostPenaltyWorkspace>;
  mutable ExecutionWorkspace execution_workspace_;
  mutable std::vector<InteractionPatch> execution_patches_;
  mutable std::vector<execution::PenaltyContribution> execution_contributions_;
};

}  // namespace tribol

#endif
