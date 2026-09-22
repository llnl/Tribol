#ifndef TRIBOL_CONTACT_CONTACT_HPP_
#define TRIBOL_CONTACT_CONTACT_HPP_

#include "tribol/core/ExactTangent.hpp"
#include "tribol/core/MeshView.hpp"
#include "tribol/core/Version.hpp"
#include "tribol/evaluation/PolicyEvaluator.hpp"
#include "tribol/evaluation/State.hpp"
#include "tribol/execution/ContactCuda.hpp"
#include "tribol/execution/ContactOpenMP.hpp"
#include "tribol/execution/Execution.hpp"
#include "tribol/method/Traits.hpp"
#include "tribol/method/Validation.hpp"
#include "tribol/search/Search.hpp"
#include "tribol/timestep/KinematicVote.hpp"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tribol {

template <SupportedMethod MethodType = DefaultMethod, SearchPolicy Search = search::CartesianProduct,
          execution::Policy Execution = execution::Sequential>
  requires execution::SupportedContactExecution<MethodType, Execution> &&
           execution::SupportedContactSearch<Search, Execution>
class Contact {
 public:
  using method_type = MethodType;
  using search_type = Search;
  using execution_type = Execution;

  struct Options {
    typename MethodType::Parameters method{};
    typename Search::Parameters search{};
    timestep::Kinematic::Parameters timestep{};
    bool self_contact{};
    bool exclude_adjacent_self_contact{ true };
    bool reject_excessive_self_penetration{ true };
  };

  explicit Contact( SurfacePairView surfaces, Options options = {} )
      : surfaces_( surfaces ),
        options_( effectiveOptions( std::move( options ) ) ),
        search_( effectiveSearchParameters( options_ ) )
  {
    requireValidSurfaces( surfaces_ );
    validateMethodParameters<MethodType>( options_.method );
    validateMethodDomain<MethodType>( surfaces_ );
    timestep::validate( options_.timestep );
    resizeLinearizationWorkspace();
    resizeEvaluationWorkspace();
    if constexpr ( std::same_as<Execution, execution::Cuda> ) {
      execution_workspace_.uploadSurfaces( surfaces_ );
    }
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
    if constexpr ( std::same_as<Execution, execution::Cuda> ) {
      execution_workspace_.findCandidates( effectiveSearchParameters( options_ ), options_.self_contact,
                                           options_.exclude_adjacent_self_contact );
      cuda_candidate_mirror_.clear();
    } else {
      search_.findCandidates( surfaces_, candidates_ );
      if ( options_.self_contact ) {
        candidates_.filterSelfContact( surfaces_.mortar, options_.exclude_adjacent_self_contact );
      }
    }
    interaction_geometry_version_ = geometry_version_;
    interaction_version_.advance();
    has_interactions_ = true;
    resizeInteractionWorkspace();
  }

  void setInteractions( ArrayView<const ElementPair> interactions )
  {
    candidates_.clear();
    candidates_.reserve( interactions.size() );
    for ( ElementPair pair : interactions ) {
      if ( pair.mortar_element < 0 || pair.mortar_element >= surfaces_.mortar.numberOfElements() ||
           pair.nonmortar_element < 0 || pair.nonmortar_element >= surfaces_.nonmortar.numberOfElements() ) {
        throw std::out_of_range( "A contact pair references an element outside its surface." );
      }
      candidates_.append( pair );
    }
    candidates_.canonicalize();
    if constexpr ( std::same_as<Execution, execution::Cuda> ) {
      execution_workspace_.setCandidates( candidates_.view() );
      cuda_candidate_mirror_.clear();
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
    if constexpr ( std::same_as<Execution, execution::Cuda> ) {
      execution_workspace_.updateGeometry( surfaces_ );
    }
    geometry_version_.advance();
  }

  void rebuildGeometry( SurfacePairView surfaces )
  {
    requireValidSurfaces( surfaces );
    surfaces_ = surfaces;
    candidates_.clear();
    cuda_candidate_mirror_.clear();
    if constexpr ( std::same_as<Execution, execution::Cuda> ) {
      execution_workspace_.invalidate();
      execution_workspace_.uploadSurfaces( surfaces_ );
    }
    geometry_version_.advance();
    has_interactions_ = false;
    resizeLinearizationWorkspace();
    resizeEvaluationWorkspace();
  }

#include "tribol/contact/ContactEvaluation.inl"

  [[nodiscard]] NodalKinematicsView evaluateNodalKinematics() const
    requires( std::same_as<typename MethodType::formulation_policy, formulation::Variational> &&
              detail::is_nodal_constraint_v<typename MethodType::constraint_policy> )
  {
    if ( !has_interactions_ ) {
      throw std::logic_error( "updateInteractions() must be called before evaluating nodal kinematics." );
    }
    clearEvaluationWorkspace();
    ContactOutputView output{
        .gap = mutableView( result_gap_ ),
        .weighted_gap = mutableView( result_weighted_gap_ ),
        .tributary_area = mutableView( result_tributary_area_ ),
    };
    if constexpr ( std::same_as<Execution, execution::OpenMP> ) {
      execution::evaluateOpenMPNodalKinematics<MethodType>( surfaces_, candidates_.view(), options_.method, output,
                                                            execution_workspace_ );
    } else {
      stageVariationalKinematics<MethodType>( surfaces_, candidates_.view(), options_.method, {}, output );
      policy_evaluator_detail::finalizeGap( output );
    }
    return {
        .gap = constView( result_gap_ ),
        .weighted_gap = constView( result_weighted_gap_ ),
        .tributary_area = constView( result_tributary_area_ ),
        .geometry_version = geometry_version_,
        .interaction_version = interaction_version_,
    };
  }

#include "tribol/contact/ContactLinearization.inl"
  [[nodiscard]] GeometryVersion geometryVersion() const { return geometry_version_; }
  [[nodiscard]] GeometryVersion interactionGeometryVersion() const { return interaction_geometry_version_; }
  [[nodiscard]] InteractionVersion interactionVersion() const { return interaction_version_; }

 private:
  static typename Search::Parameters effectiveSearchParameters( const Options& options )
  {
    auto parameters = options.search;
    if constexpr ( requires { parameters.expansion; } ) {
      parameters.expansion = std::max( parameters.expansion, options.method.constraint.activation.residual_gap );
    }
    return parameters;
  }

  static Options effectiveOptions( Options options )
  {
    if ( options.self_contact && options.reject_excessive_self_penetration ) {
      options.method.constraint.activation.reject_excessive_penetration = true;
    }
    return options;
  }

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

  ContactOutputView resultOutput() const
  {
    return {
        .residual =
            workspaceResidual( result_mortar_force_, result_nonmortar_force_, mutableView( result_constraint_ ) ),
        .gap = mutableView( result_gap_ ),
        .weighted_gap = mutableView( result_weighted_gap_ ),
        .tributary_area = mutableView( result_tributary_area_ ),
        .mortar_weights = mutableView( result_mortar_weights_ ),
        .mortar_mass_weights = mutableView( result_mortar_mass_weights_ ),
        .quadrature_gap = mutableView( result_quadrature_gap_ ),
        .quadrature_pressure = mutableView( result_quadrature_pressure_ ),
        .pressure = mutableView( result_pressure_ ),
    };
  }

  static void addResultField( const std::vector<Real>& values, const SurfaceMeshView& surface, FieldView<Real> output )
  {
    pointwise_penalty_detail::requireVectorField( output, surface, "Contact residual field does not match surface." );
    for ( Index node = 0; node < surface.numberOfNodes(); ++node ) {
      for ( int component = 0; component < surface.dimension; ++component ) {
        output( node, component ) += values[static_cast<std::size_t>( node * surface.dimension + component )];
      }
    }
  }

  [[nodiscard]] Index interactionCount() const
  {
    if constexpr ( std::same_as<Execution, execution::Cuda> ) {
      return execution_workspace_.interactionCount();
    } else {
      return candidates_.size();
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
    exact_gap_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    exact_weighted_gap_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    exact_tributary_area_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    exact_pressure_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    exact_mortar_velocity_.resize( mortar_values );
    exact_nonmortar_velocity_.resize( nonmortar_values );
    exact_mortar_reference_coordinates_.resize( mortar_values );
    exact_nonmortar_reference_coordinates_.resize( nonmortar_values );
    exact_mortar_element_thickness_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfElements() ) );
    exact_nonmortar_element_thickness_.resize( static_cast<std::size_t>( surfaces_.nonmortar.numberOfElements() ) );
    exact_mortar_material_modulus_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfElements() ) );
    exact_nonmortar_material_modulus_.resize( static_cast<std::size_t>( surfaces_.nonmortar.numberOfElements() ) );
    exact_multiplier_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    exact_external_potential_density_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    exact_external_pressure_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    exact_external_pressure_tangent_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    direction_mortar_.resize( mortar_values );
    direction_nonmortar_.resize( nonmortar_values );
    direction_multiplier_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    derivative_mortar_.resize( mortar_values );
    derivative_nonmortar_.resize( nonmortar_values );
    derivative_constraint_.resize( static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) );
    const auto columns = mortar_values + nonmortar_values;
    const auto maximum_rows = columns + static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() );
    coordinate_jacobian_.resize( maximum_rows * columns );
    system_jacobian_.resize( maximum_rows *
                             ( columns + static_cast<std::size_t>( surfaces_.mortar.numberOfNodes() ) ) );
    csr_row_offsets_.resize( maximum_rows + 1 );
    csr_column_indices_.resize( system_jacobian_.size() );
    csr_values_.resize( system_jacobian_.size() );
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
    if constexpr ( MethodTraits<MethodType>::capabilities.produces_diagnostic_weights ) {
      result_mortar_weights_.resize( mortar_nodes * nonmortar_nodes );
      result_mortar_mass_weights_.resize( mortar_nodes * mortar_nodes );
    } else {
      result_mortar_weights_.clear();
      result_mortar_mass_weights_.clear();
    }
    result_pressure_.resize( mortar_nodes );
    result_quadrature_gap_.clear();
    result_quadrature_pressure_.clear();
  }

  void resizeInteractionWorkspace()
  {
    const auto maximum_points = static_cast<std::size_t>( interactionCount() ) * integration::maximumQuadraturePoints;
    result_quadrature_gap_.resize( maximum_points );
    result_quadrature_pressure_.resize( maximum_points );
    if constexpr ( std::same_as<Execution, execution::OpenMP> ) {
      execution_workspace_.reserve( surfaces_, static_cast<Index>( maximum_points ),
                                    MethodTraits<MethodType>::capabilities.produces_diagnostic_weights );
    }
  }

  EvaluationSummary evaluateConfigured( const SurfacePairView& surfaces, const ContactStateView& state,
                                        ContactOutputView output ) const
  {
    EvaluationSummary summary;
    if constexpr ( std::same_as<Execution, execution::Cuda> ) {
      summary = execution_workspace_.evaluateDefault( options_.method, state, options_.timestep );
      execution_workspace_.downloadResult( output );
      last_cuda_summary_ = summary;
      cuda_result_geometry_version_ = geometry_version_;
      cuda_result_interaction_version_ = interaction_version_;
    } else if constexpr ( std::same_as<Execution, execution::OpenMP> ) {
      summary = execution::evaluateOpenMPContact<MethodType>( surfaces, candidates_.view(), options_.method, state,
                                                              output, execution_workspace_ );
    } else {
      summary = evaluateMethod<MethodType>( surfaces, candidates_.view(), options_.method, state, output );
    }
    if constexpr ( !std::same_as<Execution, execution::Cuda> ) {
      summary.timestep_vote = timestep::kinematicVote<MethodType>( surfaces, candidates_.view(), options_.method, state,
                                                                   options_.timestep );
    }
    return summary;
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
    clear( result_mortar_mass_weights_ );
    clear( result_quadrature_gap_ );
    clear( result_quadrature_pressure_ );
    clear( result_pressure_ );
  }

#include "tribol/contact/ContactExactHelpers.inl"

  SurfacePairView surfaces_;
  Options options_;
  Search search_;
  mutable CandidatePairs candidates_;
  mutable std::vector<ElementPair> cuda_candidate_mirror_;
  GeometryVersion geometry_version_{};
  GeometryVersion interaction_geometry_version_{};
  InteractionVersion interaction_version_{};
  bool has_interactions_{};
  mutable std::vector<linearization_detail::ExactTangent> exact_mortar_coordinates_;
  mutable std::vector<linearization_detail::ExactTangent> exact_nonmortar_coordinates_;
  mutable std::vector<linearization_detail::ExactTangent> exact_mortar_residual_;
  mutable std::vector<linearization_detail::ExactTangent> exact_nonmortar_residual_;
  mutable std::vector<linearization_detail::ExactTangent> exact_constraint_residual_;
  mutable std::vector<linearization_detail::ExactTangent> exact_gap_;
  mutable std::vector<linearization_detail::ExactTangent> exact_weighted_gap_;
  mutable std::vector<linearization_detail::ExactTangent> exact_tributary_area_;
  mutable std::vector<linearization_detail::ExactTangent> exact_pressure_;
  mutable std::vector<linearization_detail::ExactTangent> exact_mortar_velocity_;
  mutable std::vector<linearization_detail::ExactTangent> exact_nonmortar_velocity_;
  mutable std::vector<linearization_detail::ExactTangent> exact_mortar_reference_coordinates_;
  mutable std::vector<linearization_detail::ExactTangent> exact_nonmortar_reference_coordinates_;
  mutable std::vector<linearization_detail::ExactTangent> exact_mortar_element_thickness_;
  mutable std::vector<linearization_detail::ExactTangent> exact_nonmortar_element_thickness_;
  mutable std::vector<linearization_detail::ExactTangent> exact_mortar_material_modulus_;
  mutable std::vector<linearization_detail::ExactTangent> exact_nonmortar_material_modulus_;
  mutable std::vector<linearization_detail::ExactTangent> exact_multiplier_;
  mutable std::vector<linearization_detail::ExactTangent> exact_external_potential_density_;
  mutable std::vector<linearization_detail::ExactTangent> exact_external_pressure_;
  mutable std::vector<linearization_detail::ExactTangent> exact_external_pressure_tangent_;
  mutable std::vector<Real> direction_mortar_;
  mutable std::vector<Real> direction_nonmortar_;
  mutable std::vector<Real> direction_multiplier_;
  mutable std::vector<Real> derivative_mortar_;
  mutable std::vector<Real> derivative_nonmortar_;
  mutable std::vector<Real> derivative_constraint_;
  mutable std::vector<Real> coordinate_jacobian_;
  mutable std::vector<Real> system_jacobian_;
  mutable std::vector<Index> csr_row_offsets_;
  mutable std::vector<Index> csr_column_indices_;
  mutable std::vector<Real> csr_values_;
  mutable std::vector<Real> result_mortar_force_;
  mutable std::vector<Real> result_nonmortar_force_;
  mutable std::vector<Real> result_constraint_;
  mutable std::vector<Real> result_gap_;
  mutable std::vector<Real> result_weighted_gap_;
  mutable std::vector<Real> result_tributary_area_;
  mutable std::vector<Real> result_mortar_weights_;
  mutable std::vector<Real> result_mortar_mass_weights_;
  mutable std::vector<Real> result_quadrature_gap_;
  mutable std::vector<Real> result_quadrature_pressure_;
  mutable std::vector<Real> result_pressure_;
  using ExecutionWorkspace =
      std::conditional_t<std::same_as<Execution, execution::Cuda>, execution::CudaContactWorkspace,
                         std::conditional_t<std::same_as<Execution, execution::OpenMP>,
                                            execution::OpenMPContactWorkspace, execution::HostPenaltyWorkspace>>;
  mutable ExecutionWorkspace execution_workspace_;
  mutable EvaluationSummary last_cuda_summary_{};
  mutable GeometryVersion cuda_result_geometry_version_{};
  mutable InteractionVersion cuda_result_interaction_version_{};
};

}  // namespace tribol

#endif
