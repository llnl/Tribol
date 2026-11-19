// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check explicit MFEM Common Plane selection, nonmatching constant-traction patches, and parent Jacobians.
 * The same tests run on one and two ranks; decomposition onto rank zero exercises a rank without surface elements.
 * Direct calls check output availability and validity after destroying the formulation.
 * Fixed penalties and MFEM setters that invalidate formulations are checked without refreshing the decomposition.
 * Direct residual and Jacobian scaling checks use Sequential and OpenMP when enabled, with host mesh storage.
 * Tilted nonmatching surfaces check agreement with legacy forces for one point and a different load distribution with
 * higher-order rules on the clipped overlap.
 * Local quadrature and full pair derivatives are checked in tribol_common_plane_formulation.
 * Residual-only selection and mode changes run without Enzyme as well, checking that no Jacobian transfers or outputs
 * are created. Nonmatching legacy forces provide an independent reference for residual-only assembly.
 * Formulation updates leave legacy face geometry and method storage unallocated.
 * Higher-order 2D meshes with LOR subdivision check residual agreement with legacy Common Plane forces and
 * finite-difference Jacobian accuracy across LOR transfers. Higher-order 2D meshes without LOR and 3D parents remain
 * rejected.
 */
#include <cmath>
#include <limits>
#include <stdexcept>
#include "gtest/gtest.h"
#include "SlicErrorTest.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/CommonPlaneFormulation.hpp"
#include "tribol/enforcement/PenaltyStiffness.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/integ/QuadratureRuleGeneratorFactory.hpp"
#include "tribol/search/CoarseSearchFactory.hpp"
#include "tribol/search/ProjectedEdgeSearch.hpp"

namespace {
using namespace tribol;
/** @brief Register a two-versus-three-element contact patch for each supported rule. */
class MfemCommonPlaneTest : public testing::TestWithParam<int> {
 protected:
  /** @brief Create separated bodies with a uniform 0.01 penetration along nonmatching boundary meshes. */
  void SetUp() override
  {
    mesh_ = std::make_unique<mfem::ParMesh>( static_cast<const mfem::ParMesh&>( shared::ParMeshBuilder(
        MPI_COMM_WORLD,
        shared::MeshBuilder::Unify( { shared::MeshBuilder::SquareMesh( 2, 1 ), shared::MeshBuilder::SquareMesh( 3, 1 )
                                                                                   .translate( { 0.0, 0.99 } )
                                                                                   .updateBdrAttrib( 1, 5 )
                                                                                   .updateBdrAttrib( 3, 6 )
                                                                                   .updateAttrib( 1, 2 ) } ) ) ) );
    fec_ = std::make_unique<mfem::H1_FECollection>( 1, 2 );
    fes_ = std::make_unique<mfem::ParFiniteElementSpace>( mesh_.get(), fec_.get(), 2, mfem::Ordering::byVDIM );
    coords_ = std::make_unique<mfem::ParGridFunction>( fes_.get() );
    mesh_->GetNodes( *coords_ );
    registerScheme( ExecutionMode::Sequential );
  }
  /** @brief Register the existing patch and fixed penalties. @param execution_mode Requested backend. */
  void registerScheme( ExecutionMode execution_mode )
  {
    registerMfemCouplingScheme( 0, 0, 1, *mesh_, *coords_, { 3 }, { 5 }, SURFACE_TO_SURFACE, NO_CASE, COMMON_PLANE,
                                FRICTIONLESS, PENALTY, BINNING_GRID, execution_mode );
    setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
  }
  /** @brief Release coupling data before destroying the parent mesh and coordinate field it references. */
  void TearDown() override { finalize(); }
  /** @brief Enable the formulation and select the requested reference degree.
   * @param compute_jacobian Whether to calculate df_dx as well as the residual. */
  void enable( bool compute_jacobian = true )
  {
    setMfemUseCommonPlaneFormulation( 0, true, compute_jacobian );
    setQuadratureOrder( 0, GetParam() );
  }
  /** @brief Refresh geometry on one contact rank and compute the current residual. */
  void evaluate()
  {
    updateMfemParallelDecomposition( 1, true );
    RealT dt = 1.0;
    ASSERT_EQ( update( 0, 0.0, dt ), 0 );
  }
  /** @brief Return the registered coupling scheme. @return Scheme zero, valid for this fixture's lifetime. */
  CouplingScheme& scheme() { return CouplingSchemeManager::getInstance().at( 0 ); }
  /** @brief Calculate a distributed vector norm. @param v Distributed true-DOF vector. @return Global Euclidean norm.
   */
  RealT norm( const mfem::Vector& v ) { return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, v, v ) ); }
  /** @brief Mark the upper body's finite-element nodes. @return One at upper-body nodes and zero elsewhere. */
  mfem::Array<int> upperBodyNodes() const
  {
    mfem::Array<int> upper_nodes( fes_->GetNDofs() );
    upper_nodes = 0;
    for ( int element = 0; element < mesh_->GetNE(); ++element ) {
      if ( mesh_->GetAttribute( element ) != 2 ) {
        continue;
      }
      mfem::Array<int> nodes;
      fes_->GetElementDofs( element, nodes );
      for ( const auto node : nodes ) {
        upper_nodes[node] = 1;
      }
    }
    return upper_nodes;
  }
  std::unique_ptr<mfem::ParMesh> mesh_;               ///< Parent mesh referenced by the registered coupling scheme.
  std::unique_ptr<mfem::H1_FECollection> fec_;        ///< Linear coordinate basis.
  std::unique_ptr<mfem::ParFiniteElementSpace> fes_;  ///< Parent vector-coordinate space.
  std::unique_ptr<mfem::ParGridFunction> coords_;     ///< Current parent geometry.
};

// The composed calculation builds pair geometry directly and must not allocate legacy face or method data.
TEST_P( MfemCommonPlaneTest, update_does_not_allocate_legacy_geometry )
{
  enable( false );
  evaluate();
  EXPECT_EQ( scheme().getMethodData(), nullptr );
  EXPECT_EQ( scheme().getNumTotalNodes(), scheme().getMesh1().numberOfNodes() );
  for ( auto* mesh : { &scheme().getMesh1(), &scheme().getMesh2() } ) {
    EXPECT_TRUE( mesh->getView().getElementAreas().empty() );
    EXPECT_TRUE( mesh->getView().getElementCentroids().empty() );
    EXPECT_TRUE( mesh->getView().getElementNormals().empty() );
  }
}

// Residual-only assembly must preserve nonmatching nodal loads and MPI transfers without Jacobian resources.
TEST_P( MfemCommonPlaneTest, residual_only_matches_legacy_without_matrix_transfers )
{
  evaluate();
  mfem::Vector legacy( fes_->GetVSize() );
  legacy = 0.0;
  getMfemResponse( 0, legacy );
  mfem::Vector reference( fes_->GetTrueVSize() );
  fes_->GetProlongationMatrix()->MultTranspose( legacy, reference );

  enable( false );
  EXPECT_FALSE( scheme().hasMfemJacobianData() );
  evaluate();
  ASSERT_NE( scheme().getContactFormulationResult().residual, nullptr );
  EXPECT_EQ( scheme().getContactFormulationResult().df_dx, nullptr );
  EXPECT_EQ( getMfemDfDx( 0 ), nullptr );
  auto residual = getMfemContactResidual( 0 );
  EXPECT_GT( norm( residual ), 0.0 );
  residual += reference;
  EXPECT_LT( norm( residual ), 1.0e-11 );

  setMfemKinematicPenaltyScale( 0, 2.0, 2.0 );
  evaluate();
  EXPECT_FALSE( scheme().hasMfemJacobianData() );
  auto scaled = getMfemContactResidual( 0 );
  scaled.Add( 2.0, reference );
  EXPECT_LT( norm( scaled ), 1.0e-11 );
}

// Changing evaluation mode must rebuild the formulation and release matrix data; unavailable Enzyme must be rejected
// before changing an active residual-only formulation or clearing its outputs.
TEST_P( MfemCommonPlaneTest, changes_jacobian_mode_without_changing_residuals )
{
  enable( false );
  evaluate();
  const auto reference = getMfemContactResidual( 0 );
#ifdef TRIBOL_USE_ENZYME
  setMfemUseCommonPlaneFormulation( 0, true );
  EXPECT_TRUE( scheme().hasMfemJacobianData() );
  evaluate();
  ASSERT_NE( scheme().getContactFormulationResult().df_dx, nullptr );
  auto difference = getMfemContactResidual( 0 );
  difference -= reference;
  EXPECT_LT( norm( difference ), 1.0e-11 );
  setMfemUseCommonPlaneFormulation( 0, true, false );
#else
  auto* formulation = scheme().getContactFormulation();
  EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( 0, true ) );
  EXPECT_EQ( scheme().getContactFormulation(), formulation );
  EXPECT_FALSE( scheme().getParameters().common_plane_compute_jacobian );
#endif
  EXPECT_FALSE( scheme().hasMfemJacobianData() );
  evaluate();
  EXPECT_EQ( scheme().getContactFormulationResult().df_dx, nullptr );
  auto unchanged = getMfemContactResidual( 0 );
  unchanged -= reference;
  EXPECT_LT( norm( unchanged ), 1.0e-11 );
  setMfemUseCommonPlaneFormulation( 0, false );
  EXPECT_FALSE( scheme().hasContactFormulation() );
  EXPECT_FALSE( scheme().hasMfemJacobianData() );
}

// Open contact must return an allocated zero residual with no matrix, including ranks without surface elements.
TEST_P( MfemCommonPlaneTest, residual_only_returns_zero_for_open_contact )
{
  enable( false );
  evaluate();
  const auto upper_nodes = upperBodyNodes();
  for ( int node = 0; node < upper_nodes.Size(); ++node ) {
    if ( upper_nodes[node] ) {
      ( *coords_ )[fes_->DofToVDof( node, 1 )] += 0.1;
    }
  }
  evaluate();
  ASSERT_NE( scheme().getContactFormulationResult().residual, nullptr );
  EXPECT_EQ( scheme().getContactFormulationResult().df_dx, nullptr );
  EXPECT_FALSE( scheme().hasMfemJacobianData() );
  EXPECT_LT( norm( getMfemContactResidual( 0 ) ), 1.0e-11 );
}

#ifdef TRIBOL_USE_ENZYME
// Nonmatching pair assembly must reproduce uniform analytic boundary traction without any pressure or gap field.
TEST_P( MfemCommonPlaneTest, matches_nonmatching_constant_traction_patch )
{
  EXPECT_FALSE( scheme().hasContactFormulation() );
  enable();
  evaluate();
  EXPECT_EQ( scheme().getMfemSubmeshData(), nullptr );
  ASSERT_TRUE( scheme().hasMfemJacobianData() );
  EXPECT_EQ( scheme().getMfemJacobianData()->ParentPath().final_fes, fes_.get() );
  for ( auto* mesh : { &scheme().getMesh1(), &scheme().getMesh2() } ) {
    EXPECT_FALSE( mesh->getNodalFields().m_is_node_gap_set );
    EXPECT_FALSE( mesh->getNodalFields().m_is_node_pressure_set );
  }
  const auto residual = getMfemContactResidual( 0 );
  mfem::Vector traction1( 2 ), traction2( 2 );
  traction1 = 0.0;
  traction2 = 0.0;
  constexpr RealT pressure = 0.015;  // Series stiffness 1.5 times uniform penetration 0.01.
  traction1[1] = pressure;
  traction2[1] = -pressure;
  mfem::VectorConstantCoefficient coefficient1( traction1 ), coefficient2( traction2 );
  mfem::Array<int> marker1( mesh_->bdr_attributes.Max() ), marker2( mesh_->bdr_attributes.Max() );
  marker1 = 0;
  marker2 = 0;
  marker1[3 - 1] = 1;
  marker2[5 - 1] = 1;
  mfem::ParLinearForm expected( fes_.get() );
  expected.AddBoundaryIntegrator( new mfem::VectorBoundaryLFIntegrator( coefficient1 ), marker1 );
  expected.AddBoundaryIntegrator( new mfem::VectorBoundaryLFIntegrator( coefficient2 ), marker2 );
  expected.Assemble();
  std::unique_ptr<mfem::HypreParVector> exact( expected.ParallelAssemble() );
  auto error = residual;
  error -= *exact;
  EXPECT_LT( norm( error ), 1.0e-11 );
  int rank = 0, size = 0;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  if ( size == 2 && rank == 1 ) {
    EXPECT_EQ( scheme().getMesh1().numberOfElements() + scheme().getMesh2().numberOfElements(), 0 );
  }
  EXPECT_FALSE( scheme().getContactFormulationResult().energy.has_value() );
  EXPECT_SLIC_ERROR( scheme().getContactFormulation()->computeNodalGaps( scheme().getMesh1().getView(),
                                                                         scheme().getMesh2().getView() ) );
}

// Jacobian getters return independent copies, so repeated calls must not consume the cached result.
TEST_P( MfemCommonPlaneTest, jacobian_getters_preserve_cached_result )
{
  enable();
  evaluate();
  const auto first = getMfemDfDx( 0 );
  const auto second = getMfemDfDx( 0 );
  ASSERT_NE( first, nullptr );
  ASSERT_NE( second, nullptr );
  ASSERT_NE( scheme().getContactFormulationResult().df_dx, nullptr );
  EXPECT_NE( first.get(), second.get() );
  EXPECT_NE( first.get(), scheme().getContactFormulationResult().df_dx.get() );
  mfem::Vector direction( first->Width() ), first_action( first->Height() ), second_action( second->Height() );
  for ( int dof = 0; dof < direction.Size(); ++dof ) {
    direction[dof] = 1.0 + 0.1 * dof;
  }
  first->Mult( direction, first_action );
  second->Mult( direction, second_action );
  second_action -= first_action;
  EXPECT_EQ( norm( second_action ), 0.0 );
}

// Arithmetic averaging must scale residuals and Jacobians on each host backend for each Gauss count.
TEST_P( MfemCommonPlaneTest, accepts_average_penalty_stiffness )
{
  enable();
  evaluate();
  const auto baseline = getMfemContactResidual( 0 );
  const auto jacobian = getMfemDfDx( 0 );
  mfem::Vector direction( jacobian->Width() ), reference_action( jacobian->Height() );
  for ( int dof = 0; dof < direction.Size(); ++dof ) {
    direction[dof] = 1.0 + 0.1 * dof;
  }
  jacobian->Mult( direction, reference_action );
  for ( const auto execution_mode : { ExecutionMode::Sequential
#ifdef TRIBOL_USE_OPENMP
                                      ,
                                      ExecutionMode::OpenMP
#endif
        } ) {
    SCOPED_TRACE( static_cast<int>( execution_mode ) );
    auto alternate = dispatchExecutionMode( execution_mode, [&]<ExecutionMode Exec>() {
      return createWithQuadratureRuleGenerator(
          GetParam(), CommonPlaneEdgeMapping{}, [&]( auto rule_generator ) -> std::unique_ptr<ContactFormulation> {
            auto coarse = createCoarseSearch( scheme().spatialDim(), scheme().getBinningMethod(), execution_mode,
                                              scheme().getEffectiveBinningProximityScale(), 0.0 );
            auto refined = std::make_unique<ProjectedEdgeSearch>( execution_mode, false );
            return std::make_unique<
                CommonPlaneFormulation<Exec, decltype( rule_generator ), CommonPlaneGap, ResidualAndJacobian>>(
                std::move( rule_generator ), AveragePenaltyStiffness{}( 2.0, 6.0 ), CommonPlaneGap{},
                ResidualAndJacobian{ *scheme().getMfemMeshData(), *scheme().getMfemJacobianData() },
                PairListBuilder{ std::move( coarse ), std::move( refined ) } );
          } );
    } );
    auto result = alternate->computeNodalResiduals( scheme().getMesh1().getView(), scheme().getMesh2().getView(), {} );
    ASSERT_NE( result.residual, nullptr );
    ASSERT_NE( result.df_dx, nullptr );
    EXPECT_FALSE( result.energy.has_value() );
    // The configured stiffnesses 2 and 6 have arithmetic mean 4 and series stiffness 1.5.
    constexpr RealT stiffness_ratio = 4.0 / 1.5;
    result.residual->Add( -stiffness_ratio, baseline );
    EXPECT_LT( norm( *result.residual ), 1.0e-11 );
    mfem::Vector action( result.df_dx->Height() );
    result.df_dx->Mult( direction, action );
    action.Add( -stiffness_ratio, reference_action );
    EXPECT_LT( norm( action ), 1.0e-11 );
  }
}

// Each direct call returns its own residual and matrix without changing earlier results or the coupling-scheme cache.
TEST_P( MfemCommonPlaneTest, returns_owned_outputs_without_retaining_results )
{
  enable();
  updateMfemParallelDecomposition( 1, true );
  ASSERT_TRUE( scheme().init() );
  const auto& formulation = *scheme().getContactFormulation();
  auto first = formulation.computeNodalResiduals( scheme().getMesh1().getView(), scheme().getMesh2().getView(), {} );
  ASSERT_NE( first.residual, nullptr );
  ASSERT_NE( first.df_dx, nullptr );
  EXPECT_FALSE( first.energy.has_value() );
  EXPECT_FALSE( first.time_step.has_value() );
  EXPECT_EQ( first.gap, nullptr );
  EXPECT_EQ( first.pressure, nullptr );
  EXPECT_EQ( first.weighted_gap_jacobian, nullptr );
  EXPECT_EQ( first.df_dp, nullptr );
  EXPECT_EQ( scheme().getContactFormulationResult().residual, nullptr );
  const mfem::HypreParVector original( *first.residual );
  ASSERT_GT( norm( original ), 0.0 );

  auto second = scheme().getContactFormulation()->computeNodalResiduals( scheme().getMesh1().getView(),
                                                                         scheme().getMesh2().getView(), {} );
  auto error = *second.residual;
  error -= original;
  EXPECT_LT( norm( error ), 1.0e-11 );
  scheme().setContactFormulation( nullptr );
  error = *first.residual;
  error -= original;
  EXPECT_EQ( norm( error ), 0.0 );
  mfem::Vector direction( first.df_dx->Width() ), action( first.df_dx->Height() );
  direction = 1.0;
  first.df_dx->Mult( direction, action );
  EXPECT_TRUE( std::isfinite( norm( action ) ) );
}

// A formulation keeps its stiffness when registered constants and mesh scales change outside the MFEM setters.
TEST_P( MfemCommonPlaneTest, keeps_penalty_fixed_until_formulation_is_rebuilt )
{
  enable();
  evaluate();
  const auto initial = getMfemContactResidual( 0 );
  auto& data = *scheme().getMfemMeshData();
  data.SetMesh1KinematicConstantPenalty( 4.0 );
  data.SetMesh2KinematicConstantPenalty( 12.0 );
  data.SetMesh1KinematicPenaltyScale( 2.0 );
  data.SetMesh2KinematicPenaltyScale( 2.0 );
  setPenaltyScale( 0, 2.0 );
  setPenaltyScale( 1, 2.0 );
  auto unchanged = scheme().getContactFormulation()->computeNodalResiduals( scheme().getMesh1().getView(),
                                                                            scheme().getMesh2().getView(), {} );
  unchanged.residual->Add( -1.0, initial );
  EXPECT_LT( norm( *unchanged.residual ), 1.0e-11 );

  scheme().invalidateContactFormulation();
  ASSERT_TRUE( scheme().init() );
  auto rebuilt = scheme().getContactFormulation()->computeNodalResiduals( scheme().getMesh1().getView(),
                                                                          scheme().getMesh2().getView(), {} );
  rebuilt.residual->Add( -4.0, initial );
  EXPECT_LT( norm( *rebuilt.residual ), 1.0e-11 );
}

// Constants and scales set through the MFEM interface rebuild the formulation without changing the decomposition.
TEST_P( MfemCommonPlaneTest, uses_changed_constants_without_decomposition_refresh )
{
  enable();
  evaluate();
  const auto initial = getMfemContactResidual( 0 );
  setMfemKinematicConstantPenalty( 0, 4.0, 12.0 );
  EXPECT_FALSE( scheme().hasContactFormulation() );
  ASSERT_TRUE( scheme().init() );
  auto changed_constants = scheme().getContactFormulation()->computeNodalResiduals( scheme().getMesh1().getView(),
                                                                                    scheme().getMesh2().getView(), {} );
  changed_constants.residual->Add( -2.0, initial );
  EXPECT_LT( norm( *changed_constants.residual ), 1.0e-11 );
  setMfemKinematicPenaltyScale( 0, 2.0, 2.0 );
  EXPECT_FALSE( scheme().hasContactFormulation() );
  ASSERT_TRUE( scheme().init() );
  auto direct = scheme().getContactFormulation()->computeNodalResiduals( scheme().getMesh1().getView(),
                                                                         scheme().getMesh2().getView(), {} );
  // Doubling both constants and both scales multiplies the effective series stiffness by four.
  direct.residual->Add( -4.0, initial );
  EXPECT_LT( norm( *direct.residual ), 1.0e-11 );
  RealT dt = 1.0;
  ASSERT_EQ( update( 0, 0.0, dt ), 0 );
  auto updated = getMfemContactResidual( 0 );
  updated.Add( -4.0, initial );
  EXPECT_LT( norm( updated ), 1.0e-11 );
}

// Rejected constants must not clear the active formulation's registered constants or scales.
TEST_P( MfemCommonPlaneTest, rejects_invalid_constants_without_changing_scheme )
{
  enable( false );
  setMfemKinematicPenaltyScale( 0, 2.0, 3.0 );
  ASSERT_TRUE( scheme().init() );
  auto* formulation = scheme().getContactFormulation();
  const auto& mesh_data = *scheme().getMfemMeshData();
  const RealT mesh1_penalty = *mesh_data.GetMesh1KinematicConstantPenalty();
  const RealT mesh2_penalty = *mesh_data.GetMesh2KinematicConstantPenalty();
  const RealT mesh1_scale = *mesh_data.GetMesh1KinematicPenaltyScale();
  const RealT mesh2_scale = *mesh_data.GetMesh2KinematicPenaltyScale();
  const auto check_unchanged = [&] {
    EXPECT_EQ( scheme().getContactFormulation(), formulation );
    EXPECT_EQ( *mesh_data.GetMesh1KinematicConstantPenalty(), mesh1_penalty );
    EXPECT_EQ( *mesh_data.GetMesh2KinematicConstantPenalty(), mesh2_penalty );
    EXPECT_EQ( *mesh_data.GetMesh1KinematicPenaltyScale(), mesh1_scale );
    EXPECT_EQ( *mesh_data.GetMesh2KinematicPenaltyScale(), mesh2_scale );
  };
  for ( const RealT invalid :
        { -1.0, 0.0, std::numeric_limits<RealT>::infinity(), std::numeric_limits<RealT>::quiet_NaN() } ) {
    SCOPED_TRACE( invalid );
    EXPECT_SLIC_ERROR( setMfemKinematicConstantPenalty( 0, invalid, 6.0 ) );
    check_unchanged();
    EXPECT_SLIC_ERROR( setMfemKinematicConstantPenalty( 0, 2.0, invalid ) );
    check_unchanged();
  }
  const RealT largest_penalty = std::numeric_limits<RealT>::max();
  EXPECT_SLIC_ERROR( setMfemKinematicConstantPenalty( 0, largest_penalty, largest_penalty ) );
  check_unchanged();
}

// Unequal scales configured before decomposition must multiply each constant before combining springs in series.
TEST_P( MfemCommonPlaneTest, constructs_with_scales_before_decomposition )
{
  setMfemKinematicPenaltyScale( 0, 2.0, 3.0 );
  enable();
  evaluate();
  const auto scaled = getMfemContactResidual( 0 );
  // Resetting constants clears registered scales even while the decomposed meshes still contain earlier scales.
  setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
  EXPECT_FALSE( scheme().hasContactFormulation() );
  ASSERT_TRUE( scheme().init() );
  auto unscaled = scheme().getContactFormulation()->computeNodalResiduals( scheme().getMesh1().getView(),
                                                                           scheme().getMesh2().getView(), {} );
  // Scaled constants 4 and 18 give stiffness 36/11; unscaled constants 2 and 6 give stiffness 1.5.
  unscaled.residual->Add( -11.0 / 24.0, scaled );
  EXPECT_LT( norm( *unscaled.residual ), 1.0e-11 );
}

// Each scaled mesh penalty must be positive and finite before a new formulation can be constructed.
TEST_P( MfemCommonPlaneTest, rejects_invalid_scales_at_construction )
{
  enable();
  EXPECT_SLIC_ERROR( setMfemKinematicPenaltyScale( 0, 0.0, 1.0 ) );
  EXPECT_SLIC_ERROR( setMfemKinematicPenaltyScale( 0, 1.0, -1.0 ) );
  EXPECT_SLIC_ERROR( setMfemKinematicPenaltyScale( 0, std::numeric_limits<RealT>::infinity(), 1.0 ) );
  setMfemKinematicPenaltyScale( 0, 1.0, 1.0 );
  evaluate();
}

// Crossing nonmatching surfaces match legacy centroid pressure with one point. Higher-order rules integrate varying
// pressure on the same penetrating interval and must produce a different nodal load distribution.
TEST_P( MfemCommonPlaneTest, uses_configured_pressure_rule_for_interpenetrating_overlap )
{
  const auto upper_nodes = upperBodyNodes();
  for ( int node = 0; node < upper_nodes.Size(); ++node ) {
    if ( upper_nodes[node] ) {
      const RealT x = ( *coords_ )[fes_->DofToVDof( node, 0 )];
      // The upper boundary crosses y=1.0 at x=0.55, inside a nonmatching element pair whose full midpoint is open.
      ( *coords_ )[fes_->DofToVDof( node, 1 )] += 0.04 * ( x - 0.3 );
    }
  }
  evaluate();
  mfem::Vector legacy( fes_->GetVSize() );
  legacy = 0.0;
  getMfemResponse( 0, legacy );
  mfem::Vector true_legacy( fes_->GetTrueVSize() );
  fes_->GetProlongationMatrix()->MultTranspose( legacy, true_legacy );
  ASSERT_GT( norm( true_legacy ), 0.0 );
  enable();
  evaluate();
  auto error = getMfemContactResidual( 0 );
  error += true_legacy;
  if ( GetParam() == 1 ) {
    EXPECT_LT( norm( error ), 1.0e-11 );
  } else {
    EXPECT_GT( norm( error ), 1.0e-11 );
  }
}

// Parent Jacobians differentiate assembled residuals through nonmatching pair maps and MPI transfers.
TEST_P( MfemCommonPlaneTest, assembled_jacobian_matches_finite_difference )
{
  // Shift only the upper body so projection endpoints do not coincide. Rotating either edge under perturbation
  // then tests normal and overlap derivatives without switching which endpoints limit the overlap.
  const auto upper_nodes = upperBodyNodes();
  for ( int node = 0; node < upper_nodes.Size(); ++node ) {
    if ( upper_nodes[node] ) {
      ( *coords_ )[fes_->DofToVDof( node, 0 )] += 0.13;
    }
  }
  enable();
  evaluate();
  auto jacobian = getMfemDfDx( 0 );
  ASSERT_NE( jacobian, nullptr );
  const mfem::ParGridFunction original( *coords_ );
  mfem::VectorFunctionCoefficient coefficient( 2, []( const mfem::Vector& x, mfem::Vector& v ) {
    v.SetSize( 2 );
    v[0] = 0.03 * x[0] + 0.01 * x[1];
    v[1] = 0.02 * x[0] - 0.04 * x[1];
  } );
  mfem::ParGridFunction direction( fes_.get() );
  direction.ProjectCoefficient( coefficient );
  mfem::Vector true_direction;
  direction.GetTrueDofs( true_direction );
  mfem::Vector predicted( true_direction.Size() );
  jacobian->Mult( true_direction, predicted );
  constexpr RealT step = 1.0e-6;
  *coords_ = original;
  coords_->Add( step, direction );
  evaluate();
  const auto plus = getMfemContactResidual( 0 );
  *coords_ = original;
  coords_->Add( -step, direction );
  evaluate();
  auto difference = getMfemContactResidual( 0 );
  difference.Neg();
  difference += plus;
  difference /= 2.0 * step;
  difference -= predicted;
  EXPECT_LT( norm( difference ), 1.0e-7 * ( 1.0 + norm( predicted ) ) );
  *coords_ = original;
}

// Enabling after legacy evaluation, changing penalties, rebuilding, and disabling must preserve transfers and signs.
TEST_P( MfemCommonPlaneTest, rebuilds_and_restores_legacy_response )
{
  evaluate();
  scheme().setFixedBinning( true );  // Re-enabling legacy contact must discard a previously cached pair list.

  enable();
  evaluate();
  const auto initial = getMfemContactResidual( 0 );
  setMfemKinematicConstantPenalty( 0, 4.0, 12.0 );
  evaluate();
  auto scaled = getMfemContactResidual( 0 );
  auto error = scaled;
  error.Add( -2.0, initial );
  EXPECT_LT( norm( error ), 1.0e-11 );
  EXPECT_SLIC_ERROR( setEnforcementLocation( 0, EnforcementLocation::Nodal ) );
  setMfemLORFactor( 0, 2 );
  EXPECT_EQ( scheme().getMfemMeshData()->GetLORMesh(), nullptr );
  EXPECT_TRUE( scheme().getParameters().use_common_plane_formulation );
  EXPECT_EQ( scheme().getParameters().enforcement_location, EnforcementLocation::QuadraturePoint );
  setMfemUseCommonPlaneFormulation( 0, false );
  EXPECT_FALSE( scheme().hasContactFormulation() );
  EXPECT_FALSE( scheme().hasMfemJacobianData() );
  EXPECT_FALSE( scheme().hasFixedBinning() );
  EXPECT_FALSE( scheme().isBinned() );
  evaluate();
  mfem::Vector legacy( fes_->GetVSize() );
  legacy = 0.0;
  getMfemResponse( 0, legacy );
  mfem::Vector true_legacy( fes_->GetTrueVSize() );
  fes_->GetProlongationMatrix()->MultTranspose( legacy, true_legacy );
  scaled += true_legacy;
  EXPECT_LT( norm( scaled ), 1.0e-11 );
  setMfemUseCommonPlaneFormulation( 0, true );
  evaluate();
  EXPECT_GT( norm( getMfemContactResidual( 0 ) ), 0.0 );
}

#endif  // TRIBOL_USE_ENZYME

// Factory-selected execution modes must preserve residuals, and evaluator selection must control matrix availability.
TEST_P( MfemCommonPlaneTest, execution_modes_preserve_residuals_and_jacobian_selection )
{
  mfem::Vector reference;
  for ( const auto mode : { ExecutionMode::Sequential
#ifdef TRIBOL_USE_OPENMP
                            ,
                            ExecutionMode::OpenMP
#endif
        } ) {
    SCOPED_TRACE( static_cast<int>( mode ) );
    finalize();
    registerScheme( mode );
    enable( false );
    evaluate();
    ASSERT_TRUE( scheme().hasContactFormulation() );
    EXPECT_FALSE( scheme().hasMfemJacobianData() );
    EXPECT_EQ( scheme().getContactFormulationResult().df_dx, nullptr );
    auto residual = getMfemContactResidual( 0 );
    if ( mode == ExecutionMode::Sequential ) {
      reference.SetSize( residual.Size() );
      reference = residual;
    }
    residual -= reference;
    EXPECT_LT( norm( residual ), 1.0e-11 );
#ifdef TRIBOL_USE_ENZYME
    enable();
    evaluate();
    ASSERT_NE( scheme().getContactFormulationResult().df_dx, nullptr );
    residual = getMfemContactResidual( 0 );
    residual -= reference;
    EXPECT_LT( norm( residual ), 1.0e-11 );
#endif
  }
}

// A rejected switch must keep the active formulation, its Jacobian setting, and its transfer data unchanged.
TEST_P( MfemCommonPlaneTest, rejected_switch_keeps_active_formulation )
{
  enable( false );
  evaluate();
  const auto reference = getMfemContactResidual( 0 );
  const auto* formulation = scheme().getContactFormulation();
  ASSERT_NE( formulation, nullptr );

  // An unsupported rate penalty makes any new selection invalid, including a switch to Jacobian calculation.
  auto& penalty = scheme().getEnforcementOptions().penalty_options;
  penalty.rate_calc_set = true;
  penalty.rate_calculation = RATE_CONSTANT;
  EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( 0, true, true ) );
  EXPECT_TRUE( scheme().getParameters().use_common_plane_formulation );
  EXPECT_FALSE( scheme().getParameters().common_plane_compute_jacobian );
  EXPECT_EQ( scheme().getContactFormulation(), formulation );
  EXPECT_FALSE( scheme().hasMfemJacobianData() );

  // Restoring the configuration must reuse the retained formulation and reproduce the earlier residual.
  penalty.rate_calc_set = false;
  evaluate();
  EXPECT_EQ( scheme().getContactFormulation(), formulation );
  auto residual = getMfemContactResidual( 0 );
  residual -= reference;
  EXPECT_LT( norm( residual ), 1.0e-11 );
}

// Rejected configuration must leave the legacy selection and its resources intact.
TEST_P( MfemCommonPlaneTest, rejects_unsupported_settings_before_selection )
{
  const auto check = [&]() {
    EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( 0, true, false ) );
    EXPECT_FALSE( scheme().getParameters().use_common_plane_formulation );
    EXPECT_FALSE( scheme().hasContactFormulation() );
    EXPECT_FALSE( scheme().hasMfemJacobianData() );
  };
  EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( 1, true, false ) );
  setMfemKinematicConstantPenalty( 0, 0.0, 6.0 );
  check();
  setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
  scheme().getParameters().enforcement_location = EnforcementLocation::Nodal;
  check();
  scheme().getParameters().enforcement_location = EnforcementLocation::QuadraturePoint;
  auto& penalty = scheme().getEnforcementOptions().penalty_options;
  penalty.kinematic_calculation = KINEMATIC_ELEMENT;
  check();
  penalty.kinematic_calculation = KINEMATIC_CONSTANT;
  penalty.rate_calc_set = true;
  penalty.rate_calculation = RATE_CONSTANT;
  check();
  penalty.rate_calc_set = false;
  setResidualGap( 0, 0.1 );
  check();
  setResidualGap( 0, 0.0 );
  EXPECT_SLIC_ERROR( setQuadratureOrder( 0, 3 ) );
  enable( false );
  for ( int order = 0; order <= 5; ++order ) {
    setQuadratureOrder( 0, order );
    EXPECT_FALSE( scheme().hasContactFormulation() );
    EXPECT_EQ( scheme().getParameters().quadrature_order, order );
    ASSERT_TRUE( scheme().init() );
    EXPECT_TRUE( scheme().hasContactFormulation() );
  }
  EXPECT_SLIC_ERROR( setQuadratureOrder( 0, 6 ) );
  EXPECT_EQ( scheme().getParameters().quadrature_order, 5 );
}
// The opt-in setter must reject unsupported contact models, enforcement, and tied cases before constructing resources.
TEST( CommonPlaneSelection, rejects_unsupported_methods_models_and_cases )
{
  struct Configuration {
    ContactMethod method;           ///< Contact method selected during registration.
    ContactModel model;             ///< Traction model selected during registration.
    EnforcementMethod enforcement;  ///< Constraint enforcement selected during registration.
    ContactCase contact_case;       ///< Pair interaction case.
  };
  const Configuration cases[] = { { COMMON_PLANE, VISCOUS_TANGENTIAL, PENALTY, NO_CASE },
                                  { COMMON_PLANE, FRICTIONLESS, LAGRANGE_MULTIPLIER, NO_CASE },
                                  { COMMON_PLANE, FRICTIONLESS, PENALTY, TIED_NORMAL },
                                  { COMMON_PLANE, FRICTIONLESS, PENALTY, TIED_FULL },
                                  { SINGLE_MORTAR, FRICTIONLESS, PENALTY, NO_CASE } };
  for ( const auto test : cases ) {
    mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( 1, 1 ) );
    mfem::H1_FECollection fec( 1, 2 );
    mfem::ParFiniteElementSpace fes( &mesh, &fec, 2 );
    mfem::ParGridFunction coordinates( &fes );
    mesh.GetNodes( coordinates );
    registerMfemCouplingScheme( 0, 0, 1, mesh, coordinates, { 1 }, { 3 }, SURFACE_TO_SURFACE, test.contact_case,
                                test.method, test.model, test.enforcement, BINNING_GRID, ExecutionMode::Sequential );
    setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
    EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( 0, true, false ) );
    const auto& cs = CouplingSchemeManager::getInstance().at( 0 );
    EXPECT_FALSE( cs.hasContactFormulation() );
    EXPECT_FALSE( cs.hasMfemJacobianData() );
    finalize();
  }
}

// Curved 2D coordinate spaces without LOR subdivision and three-dimensional parents are rejected before any selection.
TEST( CommonPlaneSelection, rejects_higher_order_without_lor_and_three_dimensional_geometry )
{
  for ( const int dim : { 2, 3 } ) {
    const int order = dim == 2 ? 2 : 1;
    mfem::ParMesh mesh = shared::ParMeshBuilder(
        MPI_COMM_WORLD, dim == 2 ? shared::MeshBuilder::SquareMesh( 1, 1 ) : shared::MeshBuilder::CubeMesh( 1, 1, 1 ) );
    mfem::H1_FECollection fec( order, dim );
    mfem::ParFiniteElementSpace fes( &mesh, &fec, dim );
    mfem::ParGridFunction coordinates( &fes );
    mesh.GetNodes( coordinates );
    registerMfemCouplingScheme( 0, 0, 1, mesh, coordinates, { 1 }, { 3 }, SURFACE_TO_SURFACE, NO_CASE, COMMON_PLANE,
                                FRICTIONLESS, PENALTY, BINNING_GRID, ExecutionMode::Sequential );
    setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
    auto& cs = CouplingSchemeManager::getInstance().at( 0 );
    if ( dim == 2 ) {
      // Disable LOR transfer on the quadratic 2D mesh so Common Plane sees an unrefined higher-order surface.
      cs.setMfemMeshData( std::make_unique<MfemMeshData>( 0, 1, mesh, coordinates, std::set<int>{ 1 },
                                                          std::set<int>{ 3 }, ExecutionMode::Sequential,
                                                          MemorySpace::Host, false ) );
    }
    EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( 0, true, false ) );
    EXPECT_FALSE( cs.hasContactFormulation() );
    EXPECT_FALSE( cs.hasMfemJacobianData() );
    finalize();
  }
}

// Higher-order 2D coordinates with LOR subdivision match legacy 1-point forces and parent Jacobian differences.
TEST( CommonPlaneSelection, evaluates_residuals_and_jacobians_with_lor_subdivision )
{
  // Shift the upper nonmatching body horizontally so projection endpoints do not coincide under coordinate
  // perturbations.
  mfem::ParMesh mesh = shared::ParMeshBuilder(
      MPI_COMM_WORLD,
      shared::MeshBuilder::Unify( { shared::MeshBuilder::SquareMesh( 2, 1 ), shared::MeshBuilder::SquareMesh( 3, 1 )
                                                                                 .translate( { 0.13, 0.99 } )
                                                                                 .updateBdrAttrib( 1, 5 )
                                                                                 .updateBdrAttrib( 3, 6 )
                                                                                 .updateAttrib( 1, 2 ) } ) );
  constexpr int order = 2;
  mesh.SetCurvature( order, /*discont=*/false, /*space_dim=*/2, mfem::Ordering::byVDIM );
  auto* mesh_nodes = dynamic_cast<mfem::ParGridFunction*>( mesh.GetNodes() );
  ASSERT_NE( mesh_nodes, nullptr );
  auto& fes = *mesh_nodes->ParFESpace();
  mfem::ParGridFunction coordinates( &fes );
  coordinates = *mesh_nodes;

  registerMfemCouplingScheme( 0, 0, 1, mesh, coordinates, { 3 }, { 5 }, SURFACE_TO_SURFACE, NO_CASE, COMMON_PLANE,
                              FRICTIONLESS, PENALTY, BINNING_GRID, ExecutionMode::Sequential );
  setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );

  const auto& cs = CouplingSchemeManager::getInstance().at( 0 );
  ASSERT_NE( cs.getMfemMeshData()->GetLORMesh(), nullptr );
  EXPECT_EQ( cs.getMfemMeshData()->GetLORFactor(), order );

  const auto norm = []( const mfem::Vector& v ) { return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, v, v ) ); };
  const auto evaluate = [&]() {
    updateMfemParallelDecomposition( 1, true );
    RealT dt = 1.0;
    ASSERT_EQ( update( 0, 0.0, dt ), 0 );
  };

  evaluate();
  mfem::Vector legacy( fes.GetVSize() );
  legacy = 0.0;
  getMfemResponse( 0, legacy );
  mfem::Vector true_legacy( fes.GetTrueVSize() );
  fes.GetProlongationMatrix()->MultTranspose( legacy, true_legacy );
  ASSERT_GT( norm( true_legacy ), 0.0 );

  setMfemUseCommonPlaneFormulation( 0, true, false );
  setQuadratureOrder( 0, 1 );
  evaluate();
  EXPECT_EQ( getMfemDfDx( 0 ), nullptr );
  auto residual_error = getMfemContactResidual( 0 );
  residual_error += true_legacy;
  EXPECT_LT( norm( residual_error ), 1.0e-11 );

#ifdef TRIBOL_USE_ENZYME
  // Enable the Jacobian, then override the LOR factor while the formulation is active and verify the parent-space
  // Jacobian via finite differences.
  setMfemUseCommonPlaneFormulation( 0, true, true );
  setMfemLORFactor( 0, 3 );
  EXPECT_FALSE( cs.hasContactFormulation() );
  EXPECT_TRUE( cs.hasMfemJacobianData() );
  EXPECT_EQ( cs.getMfemMeshData()->GetLORFactor(), 3 );
  evaluate();
  auto jacobian = getMfemDfDx( 0 );
  ASSERT_NE( jacobian, nullptr );

  const mfem::ParGridFunction original( coordinates );
  mfem::VectorFunctionCoefficient coefficient( 2, []( const mfem::Vector& x, mfem::Vector& v ) {
    v.SetSize( 2 );
    v[0] = 0.03 * x[0] + 0.01 * x[1];
    v[1] = 0.02 * x[0] - 0.04 * x[1];
  } );
  mfem::ParGridFunction direction( &fes );
  direction.ProjectCoefficient( coefficient );
  mfem::Vector true_direction;
  direction.GetTrueDofs( true_direction );
  mfem::Vector predicted( true_direction.Size() );
  jacobian->Mult( true_direction, predicted );
  ASSERT_GT( norm( predicted ), 0.0 );
  const RealT predicted_norm = norm( predicted );

  constexpr RealT step = 1.0e-6;
  coordinates = original;
  coordinates.Add( step, direction );
  evaluate();
  const auto plus = getMfemContactResidual( 0 );
  coordinates = original;
  coordinates.Add( -step, direction );
  evaluate();
  auto difference = getMfemContactResidual( 0 );
  difference.Neg();
  difference += plus;
  difference /= 2.0 * step;
  difference -= predicted;
  EXPECT_LT( norm( difference ), 1.0e-7 * ( 1.0 + predicted_norm ) );
#endif

  finalize();
}

INSTANTIATE_TEST_SUITE_P( quadrature, MfemCommonPlaneTest, testing::Values( 1, 3, 5 ) );
}  // namespace

/** @brief Run MFEM integration tests with MPI and SLIC initialized. @param argc Argument count. @param argv Arguments.
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
