#ifndef TRIBOL_ADAPTERS_MFEM_MFEMCONTACT_HPP_
#define TRIBOL_ADAPTERS_MFEM_MFEMCONTACT_HPP_

#include "tribol/adapters/mfem/MfemSurface.hpp"
#include "tribol/contact/Contact.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tribol::mfem {

struct ContactState {
  const ::mfem::ParGridFunction* velocity{};
  const ::mfem::ParGridFunction* reference_coordinates{};
  const ::mfem::ParGridFunction* multiplier{};
  const ::mfem::ParGridFunction* external_potential_density{};
  const ::mfem::ParGridFunction* external_pressure{};
  const ::mfem::ParGridFunction* external_pressure_tangent{};
  ::mfem::Coefficient* element_thickness{};
  ::mfem::Coefficient* material_modulus{};
};

class JacobianBlocks {
 public:
  JacobianBlocks( int coordinate_size, int multiplier_size, std::unique_ptr<::mfem::HypreParMatrix> force_coordinates,
                  std::unique_ptr<::mfem::HypreParMatrix> force_multiplier,
                  std::unique_ptr<::mfem::HypreParMatrix> constraint_coordinates,
                  std::unique_ptr<::mfem::HypreParMatrix> constraint_multiplier )
      : force_coordinates_( std::move( force_coordinates ) ),
        force_multiplier_( std::move( force_multiplier ) ),
        constraint_coordinates_( std::move( constraint_coordinates ) ),
        constraint_multiplier_( std::move( constraint_multiplier ) )
  {
    offsets_.SetSize( 3 );
    offsets_[0] = 0;
    offsets_[1] = coordinate_size;
    offsets_[2] = coordinate_size + multiplier_size;
    operator_ = std::make_unique<::mfem::BlockOperator>( offsets_ );
    operator_->SetBlock( 0, 0, force_coordinates_.get() );
    operator_->SetBlock( 0, 1, force_multiplier_.get() );
    operator_->SetBlock( 1, 0, constraint_coordinates_.get() );
    operator_->SetBlock( 1, 1, constraint_multiplier_.get() );
  }

  [[nodiscard]] const ::mfem::BlockOperator& operator*() const { return *operator_; }
  [[nodiscard]] const ::mfem::BlockOperator* operator->() const { return operator_.get(); }
  [[nodiscard]] const ::mfem::BlockOperator& blockOperator() const { return *operator_; }
  [[nodiscard]] const ::mfem::HypreParMatrix& forceCoordinates() const { return *force_coordinates_; }
  [[nodiscard]] const ::mfem::HypreParMatrix& forceMultiplier() const { return *force_multiplier_; }
  [[nodiscard]] const ::mfem::HypreParMatrix& constraintCoordinates() const { return *constraint_coordinates_; }
  [[nodiscard]] const ::mfem::HypreParMatrix& constraintMultiplier() const { return *constraint_multiplier_; }

 private:
  ::mfem::Array<int> offsets_;
  std::unique_ptr<::mfem::HypreParMatrix> force_coordinates_;
  std::unique_ptr<::mfem::HypreParMatrix> force_multiplier_;
  std::unique_ptr<::mfem::HypreParMatrix> constraint_coordinates_;
  std::unique_ptr<::mfem::HypreParMatrix> constraint_multiplier_;
  std::unique_ptr<::mfem::BlockOperator> operator_;
};

template <SupportedMethod MethodType = DefaultMethod, SearchPolicy Search = search::CartesianProduct,
          execution::Policy Execution = execution::Sequential>
  requires execution::SupportedContactExecution<MethodType, Execution> &&
           execution::SupportedContactSearch<Search, Execution>
class MfemContact {
 public:
  using CoreContact = Contact<MethodType, Search, Execution>;
  using Options = typename CoreContact::Options;

  MfemContact( const ::mfem::ParMesh& mesh, const ::mfem::ParGridFunction& coordinates,
               const PairedBoundaryAttributes& attributes, Options options = {},
               SurfaceDiscretization discretization = {} )
      : mesh_( mesh ),
        coordinate_space_( requireCoordinateSpace( mesh, coordinates ) ),
        surfaces_( mesh, coordinates, attributes, discretization, distributionParameters( options ) ),
        core_( surfaces_.view(), std::move( options ) ),
        local_parent_residual_( coordinate_space_.GetVSize() ),
        true_residual_( coordinate_space_.GetTrueVSize() ),
        true_nodal_gap_( coordinate_space_.GetTrueVSize() / coordinate_space_.GetVDim() ),
        true_nodal_weighted_gap_( true_nodal_gap_.Size() ),
        true_nodal_area_( true_nodal_gap_.Size() ),
        mortar_residual_( surfaces_.view().mortar.coordinates.values.size() ),
        nonmortar_residual_( surfaces_.view().nonmortar.coordinates.values.size() ),
        mortar_direction_( mortar_residual_.size() ),
        nonmortar_direction_( nonmortar_residual_.size() ),
        mortar_derivative_( mortar_residual_.size() ),
        nonmortar_derivative_( nonmortar_residual_.size() ),
        mortar_velocity_( mortar_residual_.size() ),
        nonmortar_velocity_( nonmortar_residual_.size() ),
        mortar_reference_coordinates_( mortar_residual_.size() ),
        nonmortar_reference_coordinates_( nonmortar_residual_.size() ),
        mortar_element_thickness_( static_cast<std::size_t>( surfaces_.view().mortar.numberOfElements() ) ),
        nonmortar_element_thickness_( static_cast<std::size_t>( surfaces_.view().nonmortar.numberOfElements() ) ),
        mortar_material_modulus_( mortar_element_thickness_.size() ),
        nonmortar_material_modulus_( nonmortar_element_thickness_.size() ),
        multiplier_( static_cast<std::size_t>( surfaces_.view().mortar.numberOfNodes() ) ),
        external_potential_density_( multiplier_.size() ),
        external_pressure_( multiplier_.size() ),
        external_pressure_tangent_( multiplier_.size() ),
        multiplier_direction_( multiplier_.size() ),
        constraint_residual_( multiplier_.size() ),
        constraint_derivative_( multiplier_.size() )
  {
  }

  void updateInteractions() { core_.updateInteractions(); }

  void updateGeometry( const ::mfem::ParGridFunction& coordinates )
  {
    requireSameSpace( coordinates );
    surfaces_.updateCoordinates( coordinates );
    core_.updateGeometry( surfaces_.view() );
  }

  void rebuildGeometry( const ::mfem::ParGridFunction& coordinates )
  {
    requireSameSpace( coordinates );
    surfaces_.rebuildCoordinates( coordinates );
    core_.rebuildGeometry( surfaces_.view() );
    resizeSurfaceWorkspace();
  }

  EvaluationSummary addResidual( ::mfem::Vector& parent_true_residual )
  {
    return addResidual( ContactStateView{}, parent_true_residual );
  }

  EvaluationSummary addResidual( const ContactState& state, ::mfem::Vector& parent_true_residual )
  {
    return addResidual( restrictState( state ), parent_true_residual );
  }

  EvaluationSummary addResidual( const ContactState& state, ::mfem::Vector& parent_true_residual,
                                 ::mfem::Vector& multiplier_true_residual )
  {
    if ( state.multiplier == nullptr ) {
      throw std::invalid_argument( "A multiplier field is required for a two-block MFEM residual." );
    }
    const auto& multiplier_space = requireScalarSpace( *state.multiplier );
    if ( multiplier_true_residual.Size() != multiplier_space.GetTrueVSize() ) {
      throw std::invalid_argument( "MFEM multiplier residual must use the multiplier true-DOF space." );
    }
    if ( parent_true_residual.Size() != coordinate_space_.GetTrueVSize() ) {
      throw std::invalid_argument( "MFEM contact residual must use the coordinate true-DOF space." );
    }
    std::fill( mortar_residual_.begin(), mortar_residual_.end(), 0.0 );
    std::fill( nonmortar_residual_.begin(), nonmortar_residual_.end(), 0.0 );
    std::fill( constraint_residual_.begin(), constraint_residual_.end(), 0.0 );
    const auto surface_views = surfaces_.view();
    const auto summary = core_.addResidual(
        restrictState( state ),
        { .mortar = { { mortar_residual_.data(), static_cast<Index>( mortar_residual_.size() ) },
                      surface_views.mortar.numberOfNodes(),
                      surface_views.mortar.dimension,
                      FieldLayout::Interleaved },
          .nonmortar = { { nonmortar_residual_.data(), static_cast<Index>( nonmortar_residual_.size() ) },
                         surface_views.nonmortar.numberOfNodes(),
                         surface_views.nonmortar.dimension,
                         FieldLayout::Interleaved },
          .constraint = { constraint_residual_.data(), static_cast<Index>( constraint_residual_.size() ) } } );
    addDualTranspose( { mortar_residual_.data(), static_cast<Index>( mortar_residual_.size() ) },
                      { nonmortar_residual_.data(), static_cast<Index>( nonmortar_residual_.size() ) },
                      parent_true_residual );
    addScalarDualTranspose( surfaces_.mortar(), multiplier_space,
                            { constraint_residual_.data(), static_cast<Index>( constraint_residual_.size() ) },
                            multiplier_true_residual );
    return globalSummary( summary );
  }

  [[nodiscard]] ContactResultView evaluate( const ContactState& state = {} )
  {
    auto result = core_.evaluate( restrictState( state ) );
    result.summary = globalSummary( result.summary );
    return result;
  }

  void evaluateNodalKinematics( ::mfem::ParGridFunction& gap, ::mfem::ParGridFunction& weighted_gap,
                                ::mfem::ParGridFunction& tributary_area )
    requires( std::same_as<typename MethodType::formulation_policy, formulation::Variational> &&
              tribol::detail::is_nodal_constraint_v<typename MethodType::constraint_policy> )
  {
    auto& scalar_space = const_cast<::mfem::ParFiniteElementSpace&>( requireScalarSpace( gap ) );
    if ( &requireScalarSpace( weighted_gap ) != &scalar_space ||
         &requireScalarSpace( tributary_area ) != &scalar_space ) {
      throw std::invalid_argument( "MFEM nodal kinematics outputs must share one scalar finite-element space." );
    }
    const auto nodal = core_.evaluateNodalKinematics();
    true_nodal_weighted_gap_ = 0.0;
    true_nodal_area_ = 0.0;
    addScalarDualTranspose( surfaces_.mortar(), scalar_space, nodal.weighted_gap, true_nodal_weighted_gap_ );
    addScalarDualTranspose( surfaces_.mortar(), scalar_space, nodal.tributary_area, true_nodal_area_ );
    for ( int dof = 0; dof < true_nodal_gap_.Size(); ++dof ) {
      true_nodal_gap_[dof] =
          std::abs( true_nodal_area_[dof] ) > 1.0e-28 ? true_nodal_weighted_gap_[dof] / true_nodal_area_[dof] : 0.0;
    }
    gap.SetFromTrueDofs( true_nodal_gap_ );
    weighted_gap.SetFromTrueDofs( true_nodal_weighted_gap_ );
    tributary_area.SetFromTrueDofs( true_nodal_area_ );
  }

  EvaluationSummary addResidual( const ContactStateView& state, ::mfem::Vector& parent_true_residual )
  {
    if ( parent_true_residual.Size() != coordinate_space_.GetTrueVSize() ) {
      throw std::invalid_argument( "MFEM contact residual must use the coordinate true-DOF space." );
    }
    std::fill( mortar_residual_.begin(), mortar_residual_.end(), 0.0 );
    std::fill( nonmortar_residual_.begin(), nonmortar_residual_.end(), 0.0 );
    const auto surface_views = surfaces_.view();
    const ContactResidualView residual{
        .mortar =
            FieldView<Real>{ ArrayView<Real>{ mortar_residual_.data(), static_cast<Index>( mortar_residual_.size() ) },
                             surface_views.mortar.numberOfNodes(), surface_views.mortar.dimension,
                             FieldLayout::Interleaved },
        .nonmortar =
            FieldView<Real>{
                ArrayView<Real>{ nonmortar_residual_.data(), static_cast<Index>( nonmortar_residual_.size() ) },
                surface_views.nonmortar.numberOfNodes(), surface_views.nonmortar.dimension, FieldLayout::Interleaved },
    };
    const auto summary = core_.addResidual( state, residual );
    addDualTranspose( { mortar_residual_.data(), static_cast<Index>( mortar_residual_.size() ) },
                      { nonmortar_residual_.data(), static_cast<Index>( nonmortar_residual_.size() ) },
                      parent_true_residual );
    return globalSummary( summary );
  }

  void addCoordinateJacobianMult( const ContactStateView& state, const ::mfem::Vector& parent_true_direction,
                                  ::mfem::Vector& parent_true_derivative )
  {
    if ( parent_true_direction.Size() != coordinate_space_.GetTrueVSize() ||
         parent_true_derivative.Size() != coordinate_space_.GetTrueVSize() ) {
      throw std::invalid_argument( "MFEM Jacobian action requires vectors in the coordinate true-DOF space." );
    }
    restrictPrimal( parent_true_direction, mortar_direction_, nonmortar_direction_ );
    std::fill( mortar_derivative_.begin(), mortar_derivative_.end(), 0.0 );
    std::fill( nonmortar_derivative_.begin(), nonmortar_derivative_.end(), 0.0 );
    const auto surface_views = surfaces_.view();
    core_.applyCoordinateDerivative(
        state,
        { .mortar = { { mortar_direction_.data(), static_cast<Index>( mortar_direction_.size() ) },
                      surface_views.mortar.numberOfNodes(),
                      surface_views.mortar.dimension,
                      FieldLayout::Interleaved },
          .nonmortar = { { nonmortar_direction_.data(), static_cast<Index>( nonmortar_direction_.size() ) },
                         surface_views.nonmortar.numberOfNodes(),
                         surface_views.nonmortar.dimension,
                         FieldLayout::Interleaved } },
        { .mortar = { { mortar_derivative_.data(), static_cast<Index>( mortar_derivative_.size() ) },
                      surface_views.mortar.numberOfNodes(),
                      surface_views.mortar.dimension,
                      FieldLayout::Interleaved },
          .nonmortar = { { nonmortar_derivative_.data(), static_cast<Index>( nonmortar_derivative_.size() ) },
                         surface_views.nonmortar.numberOfNodes(),
                         surface_views.nonmortar.dimension,
                         FieldLayout::Interleaved } } );
    addDualTranspose( { mortar_derivative_.data(), static_cast<Index>( mortar_derivative_.size() ) },
                      { nonmortar_derivative_.data(), static_cast<Index>( nonmortar_derivative_.size() ) },
                      parent_true_derivative );
  }

  void addCoordinateJacobianMult( const ContactState& state, const ::mfem::Vector& parent_true_direction,
                                  ::mfem::Vector& parent_true_derivative )
  {
    addCoordinateJacobianMult( restrictState( state ), parent_true_direction, parent_true_derivative );
  }

  void addMultiplierJacobianMult( const ContactState& state, const ::mfem::Vector& multiplier_true_direction,
                                  ::mfem::Vector& parent_true_derivative )
  {
    if ( state.multiplier == nullptr ) {
      throw std::invalid_argument( "Multiplier Jacobian action requires an MFEM multiplier field." );
    }
    const auto& multiplier_space = requireScalarSpace( *state.multiplier );
    restrictScalarPrimal( surfaces_.mortar(), multiplier_space, multiplier_true_direction, multiplier_direction_ );
    std::fill( mortar_derivative_.begin(), mortar_derivative_.end(), 0.0 );
    std::fill( nonmortar_derivative_.begin(), nonmortar_derivative_.end(), 0.0 );
    std::fill( constraint_derivative_.begin(), constraint_derivative_.end(), 0.0 );
    const auto surface_views = surfaces_.view();
    core_.applyMultiplierDerivative(
        restrictState( state ), { multiplier_direction_.data(), static_cast<Index>( multiplier_direction_.size() ) },
        { .mortar = { { mortar_derivative_.data(), static_cast<Index>( mortar_derivative_.size() ) },
                      surface_views.mortar.numberOfNodes(),
                      surface_views.mortar.dimension,
                      FieldLayout::Interleaved },
          .nonmortar = { { nonmortar_derivative_.data(), static_cast<Index>( nonmortar_derivative_.size() ) },
                         surface_views.nonmortar.numberOfNodes(),
                         surface_views.nonmortar.dimension,
                         FieldLayout::Interleaved },
          .constraint = { constraint_derivative_.data(), static_cast<Index>( constraint_derivative_.size() ) } } );
    addDualTranspose( { mortar_derivative_.data(), static_cast<Index>( mortar_derivative_.size() ) },
                      { nonmortar_derivative_.data(), static_cast<Index>( nonmortar_derivative_.size() ) },
                      parent_true_derivative );
  }

  void addConstraintCoordinateJacobianMult( const ContactState& state, const ::mfem::Vector& parent_true_direction,
                                            ::mfem::Vector& multiplier_true_derivative )
  {
    if ( state.multiplier == nullptr ) {
      throw std::invalid_argument( "Constraint Jacobian action requires an MFEM multiplier field." );
    }
    const auto& multiplier_space = requireScalarSpace( *state.multiplier );
    restrictPrimal( parent_true_direction, mortar_direction_, nonmortar_direction_ );
    std::fill( mortar_derivative_.begin(), mortar_derivative_.end(), 0.0 );
    std::fill( nonmortar_derivative_.begin(), nonmortar_derivative_.end(), 0.0 );
    std::fill( constraint_derivative_.begin(), constraint_derivative_.end(), 0.0 );
    const auto surface_views = surfaces_.view();
    core_.applyCoordinateDerivative(
        restrictState( state ),
        { .mortar = { { mortar_direction_.data(), static_cast<Index>( mortar_direction_.size() ) },
                      surface_views.mortar.numberOfNodes(),
                      surface_views.mortar.dimension,
                      FieldLayout::Interleaved },
          .nonmortar = { { nonmortar_direction_.data(), static_cast<Index>( nonmortar_direction_.size() ) },
                         surface_views.nonmortar.numberOfNodes(),
                         surface_views.nonmortar.dimension,
                         FieldLayout::Interleaved } },
        { .mortar = { { mortar_derivative_.data(), static_cast<Index>( mortar_derivative_.size() ) },
                      surface_views.mortar.numberOfNodes(),
                      surface_views.mortar.dimension,
                      FieldLayout::Interleaved },
          .nonmortar = { { nonmortar_derivative_.data(), static_cast<Index>( nonmortar_derivative_.size() ) },
                         surface_views.nonmortar.numberOfNodes(),
                         surface_views.nonmortar.dimension,
                         FieldLayout::Interleaved },
          .constraint = { constraint_derivative_.data(), static_cast<Index>( constraint_derivative_.size() ) } } );
    addScalarDualTranspose( surfaces_.mortar(), multiplier_space,
                            { constraint_derivative_.data(), static_cast<Index>( constraint_derivative_.size() ) },
                            multiplier_true_derivative );
  }

  void addExternalPressureJacobianMult( const ContactState& state, const ::mfem::Vector& pressure_true_direction,
                                        ::mfem::Vector& parent_true_derivative )
  {
    if ( state.external_pressure == nullptr ) {
      throw std::invalid_argument( "External-pressure Jacobian action requires an MFEM pressure field." );
    }
    const auto& pressure_space = requireScalarSpace( *state.external_pressure );
    restrictScalarPrimal( surfaces_.mortar(), pressure_space, pressure_true_direction, multiplier_direction_ );
    std::fill( mortar_derivative_.begin(), mortar_derivative_.end(), 0.0 );
    std::fill( nonmortar_derivative_.begin(), nonmortar_derivative_.end(), 0.0 );
    std::fill( constraint_derivative_.begin(), constraint_derivative_.end(), 0.0 );
    const auto surface_views = surfaces_.view();
    core_.applyExternalPressureDerivative(
        restrictState( state ), { multiplier_direction_.data(), static_cast<Index>( multiplier_direction_.size() ) },
        { .mortar = { { mortar_derivative_.data(), static_cast<Index>( mortar_derivative_.size() ) },
                      surface_views.mortar.numberOfNodes(),
                      surface_views.mortar.dimension,
                      FieldLayout::Interleaved },
          .nonmortar = { { nonmortar_derivative_.data(), static_cast<Index>( nonmortar_derivative_.size() ) },
                         surface_views.nonmortar.numberOfNodes(),
                         surface_views.nonmortar.dimension,
                         FieldLayout::Interleaved } } );
    addDualTranspose( { mortar_derivative_.data(), static_cast<Index>( mortar_derivative_.size() ) },
                      { nonmortar_derivative_.data(), static_cast<Index>( nonmortar_derivative_.size() ) },
                      parent_true_derivative );
  }

#include "tribol/adapters/mfem/MfemContactAssembly.inl"
  void writeGap( const ContactResultView& result, ::mfem::ParGridFunction& gap )
  {
    projectMortarField( result.gap, gap );
  }

  void writePressure( const ContactResultView& result, ::mfem::ParGridFunction& pressure )
  {
    projectMortarField( result.pressure, pressure );
  }

  void restrictPrimal( const ::mfem::Vector& parent_true_field, std::vector<Real>& mortar,
                       std::vector<Real>& nonmortar )
  {
    if ( parent_true_field.Size() != coordinate_space_.GetTrueVSize() ) {
      throw std::invalid_argument( "MFEM primal restriction requires the coordinate true-DOF space." );
    }
    mortar.assign( mortar_residual_.size(), 0.0 );
    nonmortar.assign( nonmortar_residual_.size(), 0.0 );
    const auto* prolongation = coordinate_space_.GetProlongationMatrix();
    if ( prolongation == nullptr ) {
      local_parent_residual_ = parent_true_field;
    } else {
      prolongation->Mult( parent_true_field, local_parent_residual_ );
    }
    gatherSurfaceField( surfaces_.mortar(), mortar );
    gatherSurfaceField( surfaces_.nonmortar(), nonmortar );
  }

  void addDualTranspose( ArrayView<const Real> mortar, ArrayView<const Real> nonmortar,
                         ::mfem::Vector& parent_true_residual )
  {
    if ( parent_true_residual.Size() != coordinate_space_.GetTrueVSize() ||
         mortar.size() != static_cast<Index>( mortar_residual_.size() ) ||
         nonmortar.size() != static_cast<Index>( nonmortar_residual_.size() ) ) {
      throw std::invalid_argument( "MFEM dual transpose requires matching surface fields and true-DOF output." );
    }
    local_parent_residual_ = 0.0;
    scatterSurface( surfaces_.mortar(), mortar );
    scatterSurface( surfaces_.nonmortar(), nonmortar );
    true_residual_ = 0.0;
    const auto* prolongation = coordinate_space_.GetProlongationMatrix();
    if ( prolongation == nullptr ) {
      true_residual_ = local_parent_residual_;
    } else {
      prolongation->MultTranspose( local_parent_residual_, true_residual_ );
    }
    parent_true_residual += true_residual_;
  }

  [[nodiscard]] const CoreContact& core() const { return core_; }

#include "tribol/adapters/mfem/MfemContactMapping.inl"

  const ::mfem::ParMesh& mesh_;
  ::mfem::ParFiniteElementSpace& coordinate_space_;
  detail::SurfacePairStorage surfaces_;
  CoreContact core_;
  ::mfem::Vector local_parent_residual_;
  ::mfem::Vector true_residual_;
  ::mfem::Vector true_nodal_gap_;
  ::mfem::Vector true_nodal_weighted_gap_;
  ::mfem::Vector true_nodal_area_;
  std::vector<Real> mortar_residual_;
  std::vector<Real> nonmortar_residual_;
  std::vector<Real> mortar_direction_;
  std::vector<Real> nonmortar_direction_;
  std::vector<Real> mortar_derivative_;
  std::vector<Real> nonmortar_derivative_;
  std::vector<Real> mortar_velocity_;
  std::vector<Real> nonmortar_velocity_;
  std::vector<Real> mortar_reference_coordinates_;
  std::vector<Real> nonmortar_reference_coordinates_;
  std::vector<Real> mortar_element_thickness_;
  std::vector<Real> nonmortar_element_thickness_;
  std::vector<Real> mortar_material_modulus_;
  std::vector<Real> nonmortar_material_modulus_;
  std::vector<Real> multiplier_;
  std::vector<Real> external_potential_density_;
  std::vector<Real> external_pressure_;
  std::vector<Real> external_pressure_tangent_;
  std::vector<Real> multiplier_direction_;
  std::vector<Real> constraint_residual_;
  std::vector<Real> constraint_derivative_;
  ::mfem::Vector local_scalar_field_;
  ::mfem::Vector local_scalar_residual_;
  ::mfem::Vector true_scalar_residual_;
};

template <SupportedMethod MethodType = DefaultMethod, SearchPolicy Search = search::CartesianProduct,
          execution::Policy Execution = execution::Sequential>
  requires execution::SupportedContactExecution<MethodType, Execution> &&
           execution::SupportedContactSearch<Search, Execution>
MfemContact<MethodType, Search, Execution> makeContact(
    const ::mfem::ParMesh& mesh, const ::mfem::ParGridFunction& coordinates, const PairedBoundaryAttributes& attributes,
    typename MfemContact<MethodType, Search, Execution>::Options options = {},
    SurfaceDiscretization discretization = {} )
{
  return MfemContact<MethodType, Search, Execution>( mesh, coordinates, attributes, std::move( options ),
                                                     discretization );
}

}  // namespace tribol::mfem

#endif
