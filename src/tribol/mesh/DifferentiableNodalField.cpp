// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/DifferentiableNodalField.hpp"
#ifdef BUILD_REDECOMP
#include <optional>
#include <stdexcept>
#include <utility>

namespace tribol {
DifferentiableNodalFields::DifferentiableNodalFields( const mfem::ParFiniteElementSpace& field_space,
                                                      const mfem::ParFiniteElementSpace& coordinate_space,
                                                      std::vector<DifferentiableNodalField> fields,
                                                      WeightedHessian weighted_hessian )
    : field_space_( field_space ),
      coordinate_space_( coordinate_space ),
      fields_( std::move( fields ) ),
      weighted_hessian_( std::move( weighted_hessian ) )
{
  for ( const auto& field : fields_ ) {
    if ( field.values.size() != field_space_.GetTrueVSize() || field.jacobian.height() != field_space_.GetTrueVSize() ||
         field.jacobian.width() != coordinate_space_.GetTrueVSize() ) {
      throw std::invalid_argument( "Nodal field and Jacobian dimensions must match the supplied spaces." );
    }
  }
}

shared::ParVector DifferentiableNodalFields::makeVector() const
{
  shared::ParVector vector( const_cast<mfem::ParFiniteElementSpace*>( &field_space_ ) );
  vector.fill( 0.0 );
  return vector;
}

void DifferentiableNodalFields::validatePartial( IndexT index, const shared::ParVector& partial ) const
{
  if ( index < 0 || index >= size() || partial.size() != field_space_.GetTrueVSize() ) {
    throw std::invalid_argument( "Nodal energy partials must reference a bound field and match its true-DOF count." );
  }
}

shared::ParVector DifferentiableNodalFields::assembleForce( const NodalEnergyDerivatives& derivatives ) const
{
  std::optional<shared::ParVector> force;
  for ( const auto& [index, partial] : derivatives.gradient ) {
    validatePartial( index, partial );
    auto contribution = partial * fields_[index].jacobian;
    if ( force )
      *force += contribution;
    else
      force = std::move( contribution );
  }
  if ( force ) return std::move( *force );
  shared::ParVector zero( const_cast<mfem::ParFiniteElementSpace*>( &coordinate_space_ ) );
  zero.fill( 0.0 );
  return zero;
}

shared::ParSparseMat DifferentiableNodalFields::weightedCoordinateHessian( const NodalEnergyGradient& weights ) const
{
  if ( !weighted_hessian_ )
    throw std::logic_error( "Coordinate assembly requires a weighted field Hessian operation." );
  for ( const auto& [index, partial] : weights ) validatePartial( index, partial );
  auto matrix = weighted_hessian_( weights );
  if ( matrix.height() != coordinate_space_.GetTrueVSize() || matrix.width() != coordinate_space_.GetTrueVSize() )
    throw std::invalid_argument( "Weighted field Hessian must use coordinate true DOFs." );
  return matrix;
}

shared::ParSparseMat DifferentiableNodalFields::assembleJacobian( const NodalEnergyDerivatives& derivatives ) const
{
  for ( const auto& [indices, partial] : derivatives.hessian ) {
    validatePartial( indices.first, partial );
    validatePartial( indices.second, partial );
    if ( indices.first > indices.second )
      throw std::invalid_argument( "Supply only the outer Hessian's upper triangle." );
  }
  auto jacobian = weightedCoordinateHessian( derivatives.gradient );
  for ( const auto& [indices, partial] : derivatives.hessian ) {
    const auto diagonal = shared::ParSparseMat::diagonalMatrix( field_space_.GetComm(), field_space_.GlobalTrueVSize(),
                                                                field_space_.GetTrueDofOffsets(), partial.get() );
    const auto& left = fields_[indices.first].jacobian;
    const auto& right = fields_[indices.second].jacobian;
    if ( indices.first == indices.second ) {
      jacobian += shared::ParSparseMat::rap( diagonal, left );
    } else {
      jacobian += shared::ParSparseMat::rap( left, diagonal, right );
      jacobian += shared::ParSparseMat::rap( right, diagonal, left );
    }
  }
  return jacobian;
}
}  // namespace tribol
#endif  // BUILD_REDECOMP
