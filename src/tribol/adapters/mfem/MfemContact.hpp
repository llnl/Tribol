#ifndef TRIBOL_ADAPTERS_MFEM_MFEMCONTACT_HPP_
#define TRIBOL_ADAPTERS_MFEM_MFEMCONTACT_HPP_

#include "tribol/adapters/mfem/MfemSurface.hpp"
#include "tribol/contact/Contact.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tribol::mfem {

template <SupportedMethod MethodType = DefaultMethod, SearchPolicy Search = search::CartesianProduct,
          execution::Policy Execution = execution::Sequential>
  requires execution::SupportedContactExecution<MethodType, Execution>
class MfemContact {
 public:
  using CoreContact = Contact<MethodType, Search, Execution>;
  using Options = typename CoreContact::Options;

  MfemContact( const ::mfem::ParMesh& mesh, const ::mfem::ParGridFunction& coordinates,
               const PairedBoundaryAttributes& attributes, Options options = {} )
      : mesh_( mesh ),
        coordinate_space_( requireCoordinateSpace( mesh, coordinates ) ),
        surfaces_( mesh, coordinates, attributes ),
        core_( surfaces_.view(), std::move( options ) ),
        local_parent_residual_( coordinate_space_.GetVSize() ),
        true_residual_( coordinate_space_.GetTrueVSize() ),
        mortar_residual_( surfaces_.view().mortar.coordinates.values.size() ),
        nonmortar_residual_( surfaces_.view().nonmortar.coordinates.values.size() )
  {
  }

  void updateInteractions() { core_.updateInteractions(); }

  void updateGeometry( const ::mfem::ParGridFunction& coordinates )
  {
    requireSameSpace( coordinates );
    surfaces_.updateCoordinates( coordinates );
    core_.updateGeometry( surfaces_.view() );
  }

  EvaluationSummary addResidual( ::mfem::Vector& parent_true_residual )
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
    const auto summary = core_.addResidual( {}, residual );
    addDualTranspose( { mortar_residual_.data(), static_cast<Index>( mortar_residual_.size() ) },
                      { nonmortar_residual_.data(), static_cast<Index>( nonmortar_residual_.size() ) },
                      parent_true_residual );
    return summary;
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

 private:
  static ::mfem::ParFiniteElementSpace& requireCoordinateSpace( const ::mfem::ParMesh& mesh,
                                                                const ::mfem::ParGridFunction& coordinates )
  {
    auto* space = coordinates.ParFESpace();
    if ( space == nullptr || space->GetParMesh() != &mesh || space->GetVDim() != mesh.SpaceDimension() ) {
      throw std::invalid_argument( "MFEM coordinates must be a vector field on the supplied ParMesh." );
    }
    return *space;
  }

  void requireSameSpace( const ::mfem::ParGridFunction& coordinates ) const
  {
    if ( coordinates.ParFESpace() != &coordinate_space_ ) {
      throw std::invalid_argument( "MFEM geometry updates must use the original coordinate finite-element space." );
    }
  }

  void gatherSurfaceField( const detail::SurfaceStorage& surface, std::vector<Real>& values )
  {
    const int dimension = coordinate_space_.GetVDim();
    const auto node_count = static_cast<std::size_t>( surface.view().numberOfNodes() );
    for ( std::size_t compact_node = 0; compact_node < node_count; ++compact_node ) {
      if ( !surface.isLocallyOwned( compact_node ) ) {
        continue;
      }
      const auto& restriction = surface.localRestriction( compact_node );
      for ( int component = 0; component < dimension; ++component ) {
        Real value{};
        for ( std::size_t term = 0; term < restriction.scalar_dofs.size(); ++term ) {
          const int scalar_dof = restriction.scalar_dofs[term];
          const int vector_dof = coordinate_space_.DofToVDof( detail::decodeDof( scalar_dof ), component );
          value += restriction.weights[term] * detail::dofSign( scalar_dof ) * detail::dofSign( vector_dof ) *
                   local_parent_residual_[detail::decodeDof( vector_dof )];
        }
        values[compact_node * static_cast<std::size_t>( dimension ) + component] = value;
      }
    }
    MPI_Allreduce( MPI_IN_PLACE, values.data(), static_cast<int>( values.size() ), detail::realMpiType(), MPI_SUM,
                   mesh_.GetComm() );
  }

  void scatterSurface( const detail::SurfaceStorage& surface, ArrayView<const Real> residual )
  {
    const int dimension = coordinate_space_.GetVDim();
    const auto node_count = static_cast<std::size_t>( surface.view().numberOfNodes() );
    for ( std::size_t compact_node = 0; compact_node < node_count; ++compact_node ) {
      if ( !surface.isLocallyOwned( compact_node ) ) {
        continue;
      }
      const auto& restriction = surface.localRestriction( compact_node );
      for ( int component = 0; component < dimension; ++component ) {
        for ( std::size_t term = 0; term < restriction.scalar_dofs.size(); ++term ) {
          const int scalar_dof = restriction.scalar_dofs[term];
          const int vector_dof = coordinate_space_.DofToVDof( detail::decodeDof( scalar_dof ), component );
          local_parent_residual_[detail::decodeDof( vector_dof )] +=
              restriction.weights[term] * detail::dofSign( scalar_dof ) * detail::dofSign( vector_dof ) *
              residual[static_cast<Index>( compact_node ) * dimension + component];
        }
      }
    }
  }

  const ::mfem::ParMesh& mesh_;
  ::mfem::ParFiniteElementSpace& coordinate_space_;
  detail::SurfacePairStorage surfaces_;
  CoreContact core_;
  ::mfem::Vector local_parent_residual_;
  ::mfem::Vector true_residual_;
  std::vector<Real> mortar_residual_;
  std::vector<Real> nonmortar_residual_;
};

template <SupportedMethod MethodType = DefaultMethod, SearchPolicy Search = search::CartesianProduct,
          execution::Policy Execution = execution::Sequential>
  requires execution::SupportedContactExecution<MethodType, Execution>
MfemContact<MethodType, Search, Execution> makeContact(
    const ::mfem::ParMesh& mesh, const ::mfem::ParGridFunction& coordinates, const PairedBoundaryAttributes& attributes,
    typename MfemContact<MethodType, Search, Execution>::Options options = {} )
{
  return MfemContact<MethodType, Search, Execution>( mesh, coordinates, attributes, std::move( options ) );
}

}  // namespace tribol::mfem

#endif
