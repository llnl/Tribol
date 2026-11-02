// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check sparse nodal-field chain rules on one and two MPI ranks, without contact geometry.
 * Three nonlinear/linear fields give an analytical coordinate residual and Hessian requiring both chain-rule terms.
 * Enzyme builds also differentiate the outer expression; other builds supply the same analytical partials.
 * Repeated assembly checks that weighted curvature does not accumulate. Contact tests cover MFEM field transfers.
 */
#include <cmath>
#include "gtest/gtest.h"
#include "axom/slic/core/SimpleLogger.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/mesh/DifferentiableNodalField.hpp"
#include "tribol/mesh/MfemNodalEnergy.hpp"

namespace {
using namespace tribol;
constexpr RealT tolerance = 1.0e-10;  ///< Roundoff allowance for distributed sparse products.

/** @brief Provide scalar field and coordinate spaces with matching MPI partitions. */
class NodalFieldTest : public testing::Test {
 protected:
  /** @brief Assign three fields whose product is 6u^4 - u^3 - 2u^2 at each true DOF. */
  void SetUp() override
  {
    mesh_ = std::make_unique<mfem::ParMesh>( static_cast<const mfem::ParMesh&>(
        shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( 2, 1 ) ) ) );
    fec_ = std::make_unique<mfem::H1_FECollection>( 1, mesh_->Dimension() );
    fes_ = std::make_unique<mfem::ParFiniteElementSpace>( mesh_.get(), fec_.get() );
    for ( IndexT field = 0; field < field_count; ++field ) {
      values_.emplace_back( fes_.get() );
      jacobian_diagonals_.emplace_back( fes_.get() );
    }
    u_ = shared::ParVector( fes_.get() );
    for ( int node = 0; node < u_.size(); ++node ) {
      const RealT u = 0.5 + 0.05 * ( fes_->GetMyTDofOffset() + node + 1 );
      u_[node] = u;
      values_[0][node] = u * u;
      values_[1][node] = 2.0 * u + 1.0;
      values_[2][node] = 3.0 * u - 2.0;
      jacobian_diagonals_[0][node] = 2.0 * u;
      jacobian_diagonals_[1][node] = 2.0;
      jacobian_diagonals_[2][node] = 3.0;
    }
    for ( const auto& diagonal : jacobian_diagonals_ ) jacobians_.push_back( makeDiagonal( diagonal.get() ) );
  }

  /** @brief Form a matrix on the scalar space. @param values Local diagonal entries. @return Owned diagonal matrix. */
  shared::ParSparseMat makeDiagonal( const mfem::Vector& values ) const
  {
    return shared::ParSparseMat::diagonalMatrix( fes_->GetComm(), fes_->GlobalTrueVSize(), fes_->GetTrueDofOffsets(),
                                                 values );
  }
  /** @brief Bind all three fields. @return Views valid for the fixture's lifetime. */
  std::vector<DifferentiableNodalField> fields() const
  {
    return { { values_[0], jacobians_[0] }, { values_[1], jacobians_[1] }, { values_[2], jacobians_[2] } };
  }

  static constexpr IndexT field_count = 3;             ///< Independent assembled variables in the outer energy.
  std::unique_ptr<mfem::ParMesh> mesh_;                ///< Distributed mesh defining true-DOF partitions.
  std::unique_ptr<mfem::H1_FECollection> fec_;         ///< Scalar linear basis.
  std::unique_ptr<mfem::ParFiniteElementSpace> fes_;   ///< Scalar field and coordinate space for this algebra test.
  shared::ParVector u_;                                ///< Independent coordinate variable.
  std::vector<shared::ParVector> values_;              ///< Owned q_i(u) vectors.
  std::vector<shared::ParVector> jacobian_diagonals_;  ///< Diagonal dq_i/du entries.
  std::vector<shared::ParSparseMat> jacobians_;        ///< Owned sparse dq_i/du matrices.
};

// Neither field curvature nor the outer Hessian alone gives the derivative of the composed residual.
TEST_F( NodalFieldTest, assembles_both_chain_rule_terms_for_three_fields )
{
  int hessian_calls = 0;
  const DifferentiableNodalFields fields_with_derivatives(
      *fes_, *fes_, fields(), [&]( const NodalEnergyGradient& weights ) {
        ++hessian_calls;
        shared::ParVector diagonal( fes_.get() );
        diagonal.fill( 0.0 );
        if ( const auto weight = weights.find( 0 ); weight != weights.end() ) {
          for ( int node = 0; node < diagonal.size(); ++node ) diagonal[node] = 2.0 * weight->second[node];
        }
        return makeDiagonal( diagonal.get() );
      } );
#ifdef TRIBOL_USE_ENZYME
  const auto energy = sum( NodalFieldVariable<0>{}, NodalFieldVariable<1>{}, NodalFieldVariable<2>{} );
  auto derivatives = evaluateNodalEnergy( energy, fields_with_derivatives, 0.0 );
#else
  NodalEnergyDerivatives derivatives;
  for ( IndexT i = 0; i < field_count; ++i ) {
    auto partial = fields_with_derivatives.makeVector();
    for ( int node = 0; node < partial.size(); ++node )
      partial[node] = values_[( i + 1 ) % field_count][node] * values_[( i + 2 ) % field_count][node];
    derivatives.gradient.emplace( i, std::move( partial ) );
    for ( IndexT j = i; j < field_count; ++j ) {
      auto second = fields_with_derivatives.makeVector();
      if ( i != j ) second.get() = values_[field_count - i - j].get();
      derivatives.hessian.emplace( std::pair{ i, j }, std::move( second ) );
    }
  }
#endif
  const auto force = fields_with_derivatives.assembleForce( derivatives );
  const auto jacobian = fields_with_derivatives.assembleJacobian( derivatives );
  const auto repeated = fields_with_derivatives.assembleJacobian( derivatives );
  EXPECT_EQ( hessian_calls, 2 );
  shared::ParVector direction( fes_.get() );
  direction.fill( 1.0 );
  const auto action = jacobian * direction;
  const auto repeated_action = repeated * direction;
  for ( int node = 0; node < u_.size(); ++node ) {
    const RealT u = u_[node];
    EXPECT_NEAR( force[node], 24.0 * u * u * u - 3.0 * u * u - 4.0 * u, tolerance );
    EXPECT_NEAR( action[node], 72.0 * u * u - 6.0 * u - 4.0, tolerance );
    EXPECT_NEAR( repeated_action[node], action[node], tolerance );
  }

  // Changing weights must request a new weighted Hessian rather than reuse the previous matrix.
  derivatives.hessian.clear();
  for ( auto& [index, partial] : derivatives.gradient ) partial.fill( index == 0 ? 1.0 : 0.0 );
  const auto changed = fields_with_derivatives.assembleJacobian( derivatives );
  const auto changed_action = changed * direction;
  for ( int node = 0; node < u_.size(); ++node ) EXPECT_NEAR( changed_action[node], 2.0, tolerance );
  EXPECT_EQ( hessian_calls, 3 );
}

// A Jacobian request must fail if field curvature is unavailable; a value binding can still assemble a residual.
TEST_F( NodalFieldTest, rejects_missing_curvature_and_invalid_partials )
{
  const DifferentiableNodalFields bound( *fes_, *fes_, fields() );
  NodalEnergyDerivatives derivatives;
  EXPECT_THROW( bound.assembleJacobian( derivatives ), std::logic_error );
  const auto zero = bound.assembleForce( derivatives );
  for ( int node = 0; node < zero.size(); ++node ) EXPECT_EQ( zero[node], 0.0 );
  derivatives.gradient.emplace( field_count, bound.makeVector() );
  EXPECT_THROW( bound.assembleForce( derivatives ), std::invalid_argument );
}

#ifdef TRIBOL_USE_ENZYME
// A field's index identifies its variable even when earlier fields are unused; independent inputs need no partials.
TEST_F( NodalFieldTest, allocates_partials_only_for_referenced_fields )
{
  const DifferentiableNodalFields bound( *fes_, *fes_, fields() );
  const auto energy = sum( NodalFieldVariable<field_count - 1>{} );
  const auto derivatives = evaluateNodalEnergy( energy, bound, 0.0 );
  EXPECT_EQ( derivatives.gradient.size(), 1 );
  EXPECT_TRUE( derivatives.gradient.contains( field_count - 1 ) );
  EXPECT_TRUE( derivatives.hessian.empty() );
  const auto force = bound.assembleForce( derivatives );
  for ( int node = 0; node < force.size(); ++node ) EXPECT_NEAR( force[node], 3.0, tolerance );

  const IndependentNodalField coefficients{ { values_[0].get().GetData(), values_[0].size() } };
  const DifferentiableNodalFields no_active_fields( *fes_, *fes_, {} );
  const auto constant = evaluateNodalEnergy( sum( coefficients ), no_active_fields, 0.0 );
  EXPECT_TRUE( constant.gradient.empty() );
  EXPECT_TRUE( constant.hessian.empty() );
  RealT local_energy = 0.0, expected_energy = 0.0;
  for ( int node = 0; node < u_.size(); ++node ) local_energy += u_[node] * u_[node];
  MPI_Allreduce( &local_energy, &expected_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  EXPECT_NEAR( constant.energy, expected_energy, tolerance );
}
#endif
}  // namespace

/** @brief Run sparse field tests with MPI initialized. @param argc Argument count. @param argv Arguments.
 * @return Test status. */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
