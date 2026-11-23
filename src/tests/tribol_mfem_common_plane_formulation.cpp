// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Verify public Common Plane formulation configuration and assembly with MFEM meshes.
 *
 * Tests cover formulation selection and rebuilding, residual and Jacobian modes, quadrature and penalty settings,
 * quadratic 2D meshes evaluated without low-order refinement (and restoring LOR when disabled), and MPI ranks without
 * contact elements. Pair-level quadrature and derivatives are tested separately.
 */
#include <cmath>
#include <limits>
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

constexpr int one_point_polynomial_order = EdgeQuadratureRuleGenerator<1>::polynomial_order;
constexpr int two_point_polynomial_order = EdgeQuadratureRuleGenerator<2>::polynomial_order;
constexpr int three_point_polynomial_order = EdgeQuadratureRuleGenerator<3>::polynomial_order;

#ifdef TRIBOL_USE_ENZYME
/**
 * @brief Check whether a factory result contains the requested Common Plane Gauss rule.
 * @tparam NumPoints Number of Gauss points in the expected concrete formulation.
 * @param formulation Factory-created formulation.
 * @return True when the concrete formulation contains the expected rule generator.
 */
template <int NumPoints>
bool usesCommonPlaneRule( const ContactFormulation* formulation )
{
  using RuleGenerator = PairQuadratureRuleGenerator<CommonPlaneEdgeMapping, EdgeQuadratureRuleGenerator<NumPoints>>;
  using Expected =
      CommonPlaneFormulation<ExecutionMode::Sequential, RuleGenerator, CommonPlaneGap, ResidualAndJacobian>;
  return dynamic_cast<const Expected*>( formulation ) != nullptr;
}
#endif

/** @brief Register contact between boundaries divided into two and three edges for each supported rule. */
class MfemCommonPlaneTest : public testing::TestWithParam<int> {
 protected:
  static constexpr IndexT coupling_scheme_id = 0;

  /** @brief Create two bodies whose opposing boundaries use two and three edges and overlap by 0.01. */
  void SetUp() override
  {
    constexpr int lower_boundary_edge_count = 2;
    constexpr int upper_boundary_edge_count = 3;
    constexpr int vertical_element_count = 1;
    constexpr int upper_contact_boundary_attribute = 5;
    constexpr int upper_noncontact_boundary_attribute = 6;
    constexpr int upper_body_attribute = 2;
    // The lower unit square has two elements across its top contact boundary. The upper unit square has three elements
    // across its bottom boundary and starts at y=0.99, so it penetrates the lower body's top at y=1.0 by 0.01. Renaming
    // the upper boundary and volume attributes keeps its contact edge, noncontact top edge, and body distinct after the
    // two meshes are unified.
    parent_mesh_ = std::make_unique<mfem::ParMesh>( static_cast<const mfem::ParMesh&>( shared::ParMeshBuilder(
        MPI_COMM_WORLD, shared::MeshBuilder::Unify(
                            { shared::MeshBuilder::SquareMesh( lower_boundary_edge_count, vertical_element_count ),
                              shared::MeshBuilder::SquareMesh( upper_boundary_edge_count, vertical_element_count )
                                  .translate( { 0.0, 0.99 } )
                                  .updateBdrAttrib( 1, upper_contact_boundary_attribute )
                                  .updateBdrAttrib( 3, upper_noncontact_boundary_attribute )
                                  .updateAttrib( 1, upper_body_attribute ) } ) ) ) );
    constexpr int spatial_dim = 2;
    constexpr int linear_order = 1;
    coordinate_collection_ = std::make_unique<mfem::H1_FECollection>( linear_order, spatial_dim );
    coordinate_space_ = std::make_unique<mfem::ParFiniteElementSpace>( parent_mesh_.get(), coordinate_collection_.get(),
                                                                       spatial_dim, mfem::Ordering::byVDIM );
    coordinates_ = std::make_unique<mfem::ParGridFunction>( coordinate_space_.get() );
    parent_mesh_->GetNodes( *coordinates_ );
    registerScheme( ExecutionMode::Sequential );
  }
  /** @brief Register the two-body contact configuration and fixed penalties. @param execution_mode Requested backend.
   */
  void registerScheme( ExecutionMode execution_mode )
  {
    constexpr IndexT lower_mesh_id = 0;
    constexpr IndexT upper_mesh_id = 1;
    constexpr int lower_contact_boundary_attribute = 3;
    constexpr int upper_contact_boundary_attribute = 5;
    registerMfemCouplingScheme( coupling_scheme_id, lower_mesh_id, upper_mesh_id, *parent_mesh_, *coordinates_,
                                { lower_contact_boundary_attribute }, { upper_contact_boundary_attribute },
                                SURFACE_TO_SURFACE, NO_CASE, COMMON_PLANE, FRICTIONLESS, PENALTY, BINNING_GRID,
                                execution_mode );
    constexpr RealT lower_mesh_penalty = 2.0;
    constexpr RealT upper_mesh_penalty = 6.0;
    setMfemKinematicConstantPenalty( coupling_scheme_id, lower_mesh_penalty, upper_mesh_penalty );
  }
  /** @brief Release coupling data before destroying the parent mesh and coordinate field it references. */
  void TearDown() override { finalize(); }
  /** @brief Enable the formulation and select the requested reference degree.
   * @param compute_jacobian Whether to calculate df_dx as well as the residual. */
  void enableCommonPlaneFormulation( bool compute_jacobian = true )
  {
    setMfemUseCommonPlaneFormulation( coupling_scheme_id, true, compute_jacobian );
    setQuadratureOrder( coupling_scheme_id, GetParam() );
  }
  /** @brief Refresh geometry on one contact rank and compute the current residual. */
  void redecomposeAndUpdate()
  {
    constexpr int contact_rank_count = 1;
    updateMfemParallelDecomposition( contact_rank_count, true );
    RealT dt = 1.0;
    ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
  }
  /** @brief Return the registered coupling scheme. @return Scheme zero, valid for this fixture's lifetime. */
  CouplingScheme& couplingScheme() { return CouplingSchemeManager::getInstance().at( coupling_scheme_id ); }
  /** @brief Calculate a distributed vector norm. @param v Distributed true-DOF vector. @return Global Euclidean norm.
   */
  RealT parallelL2Norm( const mfem::Vector& v ) { return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, v, v ) ); }
  /** @brief Mark the upper body's finite-element nodes. @return One at upper-body nodes and zero elsewhere. */
  mfem::Array<int> upperBodyNodes() const
  {
    constexpr int upper_body_attribute = 2;
    mfem::Array<int> upper_nodes( coordinate_space_->GetNDofs() );
    upper_nodes = 0;
    for ( int element = 0; element < parent_mesh_->GetNE(); ++element ) {
      if ( parent_mesh_->GetAttribute( element ) != upper_body_attribute ) {
        continue;
      }
      mfem::Array<int> nodes;
      coordinate_space_->GetElementDofs( element, nodes );
      for ( const auto node : nodes ) {
        upper_nodes[node] = 1;
      }
    }
    return upper_nodes;
  }
  std::unique_ptr<mfem::ParMesh> parent_mesh_;  ///< Parent mesh referenced by the registered coupling scheme.
  std::unique_ptr<mfem::H1_FECollection> coordinate_collection_;   ///< Linear coordinate basis.
  std::unique_ptr<mfem::ParFiniteElementSpace> coordinate_space_;  ///< Vector coordinate FE space on the parent mesh.
  std::unique_ptr<mfem::ParGridFunction> coordinates_;             ///< Current coordinates on the parent mesh.
};

// The composed calculation builds pair geometry directly and must not allocate legacy face or method data.
TEST_P( MfemCommonPlaneTest, update_does_not_allocate_legacy_geometry )
{
  enableCommonPlaneFormulation( false );
  redecomposeAndUpdate();
  EXPECT_EQ( couplingScheme().getMethodData(), nullptr );
  EXPECT_EQ( couplingScheme().getNumTotalNodes(), couplingScheme().getMesh1().numberOfNodes() );
  for ( auto* mesh : { &couplingScheme().getMesh1(), &couplingScheme().getMesh2() } ) {
    EXPECT_TRUE( mesh->getView().getElementAreas().empty() );
    EXPECT_TRUE( mesh->getView().getElementCentroids().empty() );
    EXPECT_TRUE( mesh->getView().getElementNormals().empty() );
  }
}

// Residual-only assembly must preserve nonmatching nodal loads and MPI transfers without Jacobian resources.
TEST_P( MfemCommonPlaneTest, residual_only_matches_legacy_without_matrix_transfers )
{
  constexpr RealT residual_tolerance = 1.0e-11;
  redecomposeAndUpdate();
  mfem::Vector legacy_local_residual( coordinate_space_->GetVSize() );
  legacy_local_residual = 0.0;
  getMfemResponse( coupling_scheme_id, legacy_local_residual );
  mfem::Vector legacy_true_residual( coordinate_space_->GetTrueVSize() );
  coordinate_space_->GetProlongationMatrix()->MultTranspose( legacy_local_residual, legacy_true_residual );

  enableCommonPlaneFormulation( false );
  EXPECT_FALSE( couplingScheme().hasMfemJacobianData() );
  redecomposeAndUpdate();
  ASSERT_NE( couplingScheme().getContactFormulationResult().residual, nullptr );
  EXPECT_EQ( couplingScheme().getContactFormulationResult().df_dx, nullptr );
  EXPECT_EQ( getMfemDfDx( coupling_scheme_id ), nullptr );
  auto residual_difference = getMfemContactResidual( coupling_scheme_id );
  EXPECT_GT( parallelL2Norm( residual_difference ), 0.0 );
  residual_difference += legacy_true_residual;
  EXPECT_LT( parallelL2Norm( residual_difference ), residual_tolerance );

  setMfemKinematicPenaltyScale( coupling_scheme_id, 2.0, 2.0 );
  redecomposeAndUpdate();
  EXPECT_FALSE( couplingScheme().hasMfemJacobianData() );
  auto scaled_residual_difference = getMfemContactResidual( coupling_scheme_id );
  scaled_residual_difference.Add( 2.0, legacy_true_residual );
  EXPECT_LT( parallelL2Norm( scaled_residual_difference ), residual_tolerance );
}

// Changing evaluation mode must rebuild the formulation and release matrix data; unavailable Enzyme must be rejected
// before changing an active residual-only formulation or clearing its outputs.
TEST_P( MfemCommonPlaneTest, changes_jacobian_mode_without_changing_residuals )
{
  constexpr RealT residual_tolerance = 1.0e-11;
  enableCommonPlaneFormulation( false );
  redecomposeAndUpdate();
  const auto residual_only = getMfemContactResidual( coupling_scheme_id );
#ifdef TRIBOL_USE_ENZYME
  setMfemUseCommonPlaneFormulation( coupling_scheme_id, true );
  EXPECT_TRUE( couplingScheme().hasMfemJacobianData() );
  redecomposeAndUpdate();
  ASSERT_NE( couplingScheme().getContactFormulationResult().df_dx, nullptr );
  auto jacobian_mode_residual_difference = getMfemContactResidual( coupling_scheme_id );
  jacobian_mode_residual_difference -= residual_only;
  EXPECT_LT( parallelL2Norm( jacobian_mode_residual_difference ), residual_tolerance );
  setMfemUseCommonPlaneFormulation( coupling_scheme_id, true, false );
#else
  auto* residual_only_formulation = couplingScheme().getContactFormulation();
  EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( coupling_scheme_id, true ) );
  EXPECT_EQ( couplingScheme().getContactFormulation(), residual_only_formulation );
  EXPECT_FALSE( couplingScheme().getParameters().common_plane_compute_jacobian );
#endif
  EXPECT_FALSE( couplingScheme().hasMfemJacobianData() );
  redecomposeAndUpdate();
  EXPECT_EQ( couplingScheme().getContactFormulationResult().df_dx, nullptr );
  auto restored_residual_difference = getMfemContactResidual( coupling_scheme_id );
  restored_residual_difference -= residual_only;
  EXPECT_LT( parallelL2Norm( restored_residual_difference ), residual_tolerance );
  setMfemUseCommonPlaneFormulation( coupling_scheme_id, false );
  EXPECT_FALSE( couplingScheme().hasContactFormulation() );
  EXPECT_FALSE( couplingScheme().hasMfemJacobianData() );
}

// Open contact must return an allocated zero residual with no matrix, including ranks without surface elements.
TEST_P( MfemCommonPlaneTest, residual_only_returns_zero_for_open_contact )
{
  constexpr RealT residual_tolerance = 1.0e-11;
  enableCommonPlaneFormulation( false );
  redecomposeAndUpdate();
  const auto upper_nodes = upperBodyNodes();
  for ( int node = 0; node < upper_nodes.Size(); ++node ) {
    if ( upper_nodes[node] ) {
      ( *coordinates_ )[coordinate_space_->DofToVDof( node, 1 )] += 0.1;
    }
  }
  redecomposeAndUpdate();
  ASSERT_NE( couplingScheme().getContactFormulationResult().residual, nullptr );
  EXPECT_EQ( couplingScheme().getContactFormulationResult().df_dx, nullptr );
  EXPECT_FALSE( couplingScheme().hasMfemJacobianData() );
  EXPECT_LT( parallelL2Norm( getMfemContactResidual( coupling_scheme_id ) ), residual_tolerance );
}

#ifdef TRIBOL_USE_ENZYME
// Uniform penetration produces constant pressure. Assembly on the two-edge and three-edge boundaries must match the
// corresponding MFEM boundary loads without requiring a registered pressure or gap field.
TEST_P( MfemCommonPlaneTest, matches_analytic_boundary_loads_for_uniform_penetration )
{
  constexpr int spatial_dim = 2;
  constexpr int lower_contact_boundary_attribute = 3;
  constexpr int upper_contact_boundary_attribute = 5;
  constexpr RealT residual_tolerance = 1.0e-11;
  EXPECT_FALSE( couplingScheme().hasContactFormulation() );
  enableCommonPlaneFormulation();
  redecomposeAndUpdate();
  EXPECT_EQ( couplingScheme().getMfemSubmeshData(), nullptr );
  ASSERT_TRUE( couplingScheme().hasMfemJacobianData() );
  EXPECT_EQ( couplingScheme().getMfemJacobianData()->ParentPath().final_fes, coordinate_space_.get() );
  for ( auto* mesh : { &couplingScheme().getMesh1(), &couplingScheme().getMesh2() } ) {
    EXPECT_FALSE( mesh->getNodalFields().m_is_node_gap_set );
    EXPECT_FALSE( mesh->getNodalFields().m_is_node_pressure_set );
  }
  const auto assembled_residual = getMfemContactResidual( coupling_scheme_id );
  mfem::Vector lower_boundary_traction( spatial_dim ), upper_boundary_traction( spatial_dim );
  lower_boundary_traction = 0.0;
  upper_boundary_traction = 0.0;
  constexpr RealT pressure = 0.015;  // Series stiffness 1.5 times uniform penetration 0.01.
  lower_boundary_traction[1] = pressure;
  upper_boundary_traction[1] = -pressure;
  mfem::VectorConstantCoefficient lower_traction_coefficient( lower_boundary_traction );
  mfem::VectorConstantCoefficient upper_traction_coefficient( upper_boundary_traction );
  mfem::Array<int> lower_boundary_marker( parent_mesh_->bdr_attributes.Max() );
  mfem::Array<int> upper_boundary_marker( parent_mesh_->bdr_attributes.Max() );
  lower_boundary_marker = 0;
  upper_boundary_marker = 0;
  lower_boundary_marker[lower_contact_boundary_attribute - 1] = 1;
  upper_boundary_marker[upper_contact_boundary_attribute - 1] = 1;
  mfem::ParLinearForm analytic_boundary_loads( coordinate_space_.get() );
  analytic_boundary_loads.AddBoundaryIntegrator( new mfem::VectorBoundaryLFIntegrator( lower_traction_coefficient ),
                                                 lower_boundary_marker );
  analytic_boundary_loads.AddBoundaryIntegrator( new mfem::VectorBoundaryLFIntegrator( upper_traction_coefficient ),
                                                 upper_boundary_marker );
  analytic_boundary_loads.Assemble();
  std::unique_ptr<mfem::HypreParVector> analytic_true_residual( analytic_boundary_loads.ParallelAssemble() );
  auto residual_difference = assembled_residual;
  residual_difference -= *analytic_true_residual;
  EXPECT_LT( parallelL2Norm( residual_difference ), residual_tolerance );
  int rank = 0;
  int rank_count = 0;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &rank_count );
  constexpr int two_rank_run = 2;
  constexpr int rank_without_contact_elements = 1;
  if ( rank_count == two_rank_run && rank == rank_without_contact_elements ) {
    EXPECT_EQ( couplingScheme().getMesh1().numberOfElements() + couplingScheme().getMesh2().numberOfElements(), 0 );
  }
  EXPECT_FALSE( couplingScheme().getContactFormulationResult().energy.has_value() );
  EXPECT_SLIC_ERROR( couplingScheme().getContactFormulation()->computeNodalGaps(
      couplingScheme().getMesh1().getView(), couplingScheme().getMesh2().getView() ) );
}

// Jacobian getters return independent copies, so repeated calls must not consume the cached result.
TEST_P( MfemCommonPlaneTest, jacobian_getters_preserve_cached_result )
{
  enableCommonPlaneFormulation();
  redecomposeAndUpdate();
  const auto first_jacobian_copy = getMfemDfDx( coupling_scheme_id );
  const auto second_jacobian_copy = getMfemDfDx( coupling_scheme_id );
  ASSERT_NE( first_jacobian_copy, nullptr );
  ASSERT_NE( second_jacobian_copy, nullptr );
  ASSERT_NE( couplingScheme().getContactFormulationResult().df_dx, nullptr );
  EXPECT_NE( first_jacobian_copy.get(), second_jacobian_copy.get() );
  EXPECT_NE( first_jacobian_copy.get(), couplingScheme().getContactFormulationResult().df_dx.get() );
  mfem::Vector displacement_direction( first_jacobian_copy->Width() );
  mfem::Vector first_jacobian_action( first_jacobian_copy->Height() );
  mfem::Vector second_jacobian_action( second_jacobian_copy->Height() );
  for ( int dof = 0; dof < displacement_direction.Size(); ++dof ) {
    displacement_direction[dof] = 1.0 + 0.1 * dof;
  }
  first_jacobian_copy->Mult( displacement_direction, first_jacobian_action );
  second_jacobian_copy->Mult( displacement_direction, second_jacobian_action );
  second_jacobian_action -= first_jacobian_action;
  EXPECT_EQ( parallelL2Norm( second_jacobian_action ), 0.0 );
}

// Arithmetic averaging must scale residuals and Jacobians on each host backend for each Gauss count.
TEST_P( MfemCommonPlaneTest, accepts_average_penalty_stiffness )
{
  constexpr RealT lower_mesh_penalty = 2.0;
  constexpr RealT upper_mesh_penalty = 6.0;
  constexpr RealT residual_tolerance = 1.0e-11;
  enableCommonPlaneFormulation();
  redecomposeAndUpdate();
  const auto series_penalty_residual = getMfemContactResidual( coupling_scheme_id );
  const auto series_penalty_jacobian = getMfemDfDx( coupling_scheme_id );
  mfem::Vector displacement_direction( series_penalty_jacobian->Width() );
  mfem::Vector series_penalty_jacobian_action( series_penalty_jacobian->Height() );
  for ( int dof = 0; dof < displacement_direction.Size(); ++dof ) {
    displacement_direction[dof] = 1.0 + 0.1 * dof;
  }
  series_penalty_jacobian->Mult( displacement_direction, series_penalty_jacobian_action );
  for ( const auto execution_mode : { ExecutionMode::Sequential
#ifdef TRIBOL_USE_OPENMP
                                      ,
                                      ExecutionMode::OpenMP
#endif
        } ) {
    SCOPED_TRACE( static_cast<int>( execution_mode ) );
    auto average_penalty_formulation = dispatchExecutionMode( execution_mode, [&]<ExecutionMode Exec>() {
      return createWithQuadratureRuleGenerator(
          GetParam(), CommonPlaneEdgeMapping{}, [&]( auto rule_generator ) -> std::unique_ptr<ContactFormulation> {
            auto coarse_search =
                createCoarseSearch( couplingScheme().spatialDim(), couplingScheme().getBinningMethod(), execution_mode,
                                    couplingScheme().getEffectiveBinningProximityScale(), 0.0 );
            auto projected_edge_search = std::make_unique<ProjectedEdgeSearch>( execution_mode, false );
            return std::make_unique<
                CommonPlaneFormulation<Exec, decltype( rule_generator ), CommonPlaneGap, ResidualAndJacobian>>(
                std::move( rule_generator ),
                AveragePenaltyStiffness{}.computeEffectiveStiffness( lower_mesh_penalty, upper_mesh_penalty ),
                CommonPlaneGap{},
                ResidualAndJacobian{ *couplingScheme().getMfemMeshData(), *couplingScheme().getMfemJacobianData() },
                PairListBuilder{ std::move( coarse_search ), std::move( projected_edge_search ) } );
          } );
    } );
    auto average_penalty_result = average_penalty_formulation->computeNodalResiduals(
        couplingScheme().getMesh1().getView(), couplingScheme().getMesh2().getView(), {} );
    ASSERT_NE( average_penalty_result.residual, nullptr );
    ASSERT_NE( average_penalty_result.df_dx, nullptr );
    EXPECT_FALSE( average_penalty_result.energy.has_value() );
    // The configured stiffnesses 2 and 6 have arithmetic mean 4 and series stiffness 1.5.
    constexpr RealT stiffness_ratio = 4.0 / 1.5;
    average_penalty_result.residual->Add( -stiffness_ratio, series_penalty_residual );
    EXPECT_LT( parallelL2Norm( *average_penalty_result.residual ), residual_tolerance );
    mfem::Vector average_penalty_jacobian_action( average_penalty_result.df_dx->Height() );
    average_penalty_result.df_dx->Mult( displacement_direction, average_penalty_jacobian_action );
    average_penalty_jacobian_action.Add( -stiffness_ratio, series_penalty_jacobian_action );
    EXPECT_LT( parallelL2Norm( average_penalty_jacobian_action ), residual_tolerance );
  }
}

// Each direct call returns its own residual and matrix without changing earlier results or the coupling-scheme cache.
TEST_P( MfemCommonPlaneTest, returns_owned_outputs_without_retaining_results )
{
  constexpr RealT residual_tolerance = 1.0e-11;
  enableCommonPlaneFormulation();
  constexpr int contact_rank_count = 1;
  updateMfemParallelDecomposition( contact_rank_count, true );
  ASSERT_TRUE( couplingScheme().init() );
  const auto& formulation = *couplingScheme().getContactFormulation();
  auto retained_result = formulation.computeNodalResiduals( couplingScheme().getMesh1().getView(),
                                                            couplingScheme().getMesh2().getView(), {} );
  ASSERT_NE( retained_result.residual, nullptr );
  ASSERT_NE( retained_result.df_dx, nullptr );
  EXPECT_FALSE( retained_result.energy.has_value() );
  EXPECT_FALSE( retained_result.time_step.has_value() );
  EXPECT_EQ( retained_result.gap, nullptr );
  EXPECT_EQ( retained_result.pressure, nullptr );
  EXPECT_EQ( retained_result.weighted_gap_jacobian, nullptr );
  EXPECT_EQ( retained_result.df_dp, nullptr );
  EXPECT_EQ( couplingScheme().getContactFormulationResult().residual, nullptr );
  const mfem::HypreParVector retained_residual_copy( *retained_result.residual );
  ASSERT_GT( parallelL2Norm( retained_residual_copy ), 0.0 );

  auto subsequent_result = couplingScheme().getContactFormulation()->computeNodalResiduals(
      couplingScheme().getMesh1().getView(), couplingScheme().getMesh2().getView(), {} );
  auto subsequent_residual_difference = *subsequent_result.residual;
  subsequent_residual_difference -= retained_residual_copy;
  EXPECT_LT( parallelL2Norm( subsequent_residual_difference ), residual_tolerance );
  couplingScheme().setContactFormulation( nullptr );
  auto retained_residual_difference = *retained_result.residual;
  retained_residual_difference -= retained_residual_copy;
  EXPECT_EQ( parallelL2Norm( retained_residual_difference ), 0.0 );
  mfem::Vector displacement_direction( retained_result.df_dx->Width() );
  mfem::Vector retained_jacobian_action( retained_result.df_dx->Height() );
  displacement_direction = 1.0;
  retained_result.df_dx->Mult( displacement_direction, retained_jacobian_action );
  EXPECT_TRUE( std::isfinite( parallelL2Norm( retained_jacobian_action ) ) );
}

// A formulation keeps its stiffness when registered constants and mesh scales change outside the MFEM setters.
TEST_P( MfemCommonPlaneTest, keeps_penalty_fixed_until_formulation_is_rebuilt )
{
  constexpr IndexT lower_mesh_id = 0;
  constexpr IndexT upper_mesh_id = 1;
  constexpr RealT residual_tolerance = 1.0e-11;
  enableCommonPlaneFormulation();
  redecomposeAndUpdate();
  const auto baseline_residual = getMfemContactResidual( coupling_scheme_id );
  auto& mfem_mesh_data = *couplingScheme().getMfemMeshData();
  mfem_mesh_data.SetMesh1KinematicConstantPenalty( 4.0 );
  mfem_mesh_data.SetMesh2KinematicConstantPenalty( 12.0 );
  mfem_mesh_data.SetMesh1KinematicPenaltyScale( 2.0 );
  mfem_mesh_data.SetMesh2KinematicPenaltyScale( 2.0 );
  setPenaltyScale( lower_mesh_id, 2.0 );
  setPenaltyScale( upper_mesh_id, 2.0 );
  auto retained_penalty_result = couplingScheme().getContactFormulation()->computeNodalResiduals(
      couplingScheme().getMesh1().getView(), couplingScheme().getMesh2().getView(), {} );
  retained_penalty_result.residual->Add( -1.0, baseline_residual );
  EXPECT_LT( parallelL2Norm( *retained_penalty_result.residual ), residual_tolerance );

  couplingScheme().invalidateContactFormulation();
  ASSERT_TRUE( couplingScheme().init() );
  auto rebuilt_penalty_result = couplingScheme().getContactFormulation()->computeNodalResiduals(
      couplingScheme().getMesh1().getView(), couplingScheme().getMesh2().getView(), {} );
  rebuilt_penalty_result.residual->Add( -4.0, baseline_residual );
  EXPECT_LT( parallelL2Norm( *rebuilt_penalty_result.residual ), residual_tolerance );
}

// Constants and scales set through the MFEM interface rebuild the formulation without changing the decomposition.
TEST_P( MfemCommonPlaneTest, uses_changed_constants_without_decomposition_refresh )
{
  constexpr RealT residual_tolerance = 1.0e-11;
  enableCommonPlaneFormulation();
  redecomposeAndUpdate();
  const auto baseline_residual = getMfemContactResidual( coupling_scheme_id );
  setMfemKinematicConstantPenalty( coupling_scheme_id, 4.0, 12.0 );
  EXPECT_FALSE( couplingScheme().hasContactFormulation() );
  ASSERT_TRUE( couplingScheme().init() );
  auto changed_constants = couplingScheme().getContactFormulation()->computeNodalResiduals(
      couplingScheme().getMesh1().getView(), couplingScheme().getMesh2().getView(), {} );
  changed_constants.residual->Add( -2.0, baseline_residual );
  EXPECT_LT( parallelL2Norm( *changed_constants.residual ), residual_tolerance );
  setMfemKinematicPenaltyScale( coupling_scheme_id, 2.0, 2.0 );
  EXPECT_FALSE( couplingScheme().hasContactFormulation() );
  ASSERT_TRUE( couplingScheme().init() );
  auto changed_constants_and_scales_result = couplingScheme().getContactFormulation()->computeNodalResiduals(
      couplingScheme().getMesh1().getView(), couplingScheme().getMesh2().getView(), {} );
  // Doubling both constants and both scales multiplies the effective series stiffness by four.
  changed_constants_and_scales_result.residual->Add( -4.0, baseline_residual );
  EXPECT_LT( parallelL2Norm( *changed_constants_and_scales_result.residual ), residual_tolerance );
  RealT dt = 1.0;
  ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
  auto updated_residual_difference = getMfemContactResidual( coupling_scheme_id );
  updated_residual_difference.Add( -4.0, baseline_residual );
  EXPECT_LT( parallelL2Norm( updated_residual_difference ), residual_tolerance );
}

// Rejected constants must not clear the active formulation's registered constants or scales.
TEST_P( MfemCommonPlaneTest, rejects_invalid_constants_without_changing_scheme )
{
  constexpr RealT lower_mesh_penalty = 2.0;
  constexpr RealT upper_mesh_penalty = 6.0;
  enableCommonPlaneFormulation( false );
  setMfemKinematicPenaltyScale( coupling_scheme_id, 2.0, 3.0 );
  ASSERT_TRUE( couplingScheme().init() );
  auto* formulation = couplingScheme().getContactFormulation();
  const auto& mfem_mesh_data = *couplingScheme().getMfemMeshData();
  const RealT registered_lower_penalty = *mfem_mesh_data.GetMesh1KinematicConstantPenalty();
  const RealT registered_upper_penalty = *mfem_mesh_data.GetMesh2KinematicConstantPenalty();
  const RealT registered_lower_scale = *mfem_mesh_data.GetMesh1KinematicPenaltyScale();
  const RealT registered_upper_scale = *mfem_mesh_data.GetMesh2KinematicPenaltyScale();
  const auto check_unchanged = [&] {
    EXPECT_EQ( couplingScheme().getContactFormulation(), formulation );
    EXPECT_EQ( *mfem_mesh_data.GetMesh1KinematicConstantPenalty(), registered_lower_penalty );
    EXPECT_EQ( *mfem_mesh_data.GetMesh2KinematicConstantPenalty(), registered_upper_penalty );
    EXPECT_EQ( *mfem_mesh_data.GetMesh1KinematicPenaltyScale(), registered_lower_scale );
    EXPECT_EQ( *mfem_mesh_data.GetMesh2KinematicPenaltyScale(), registered_upper_scale );
  };
  for ( const RealT invalid :
        { -1.0, 0.0, std::numeric_limits<RealT>::infinity(), std::numeric_limits<RealT>::quiet_NaN() } ) {
    SCOPED_TRACE( invalid );
    EXPECT_SLIC_ERROR( setMfemKinematicConstantPenalty( coupling_scheme_id, invalid, upper_mesh_penalty ) );
    check_unchanged();
    EXPECT_SLIC_ERROR( setMfemKinematicConstantPenalty( coupling_scheme_id, lower_mesh_penalty, invalid ) );
    check_unchanged();
  }
  const RealT largest_penalty = std::numeric_limits<RealT>::max();
  EXPECT_SLIC_ERROR( setMfemKinematicConstantPenalty( coupling_scheme_id, largest_penalty, largest_penalty ) );
  check_unchanged();
}

// Unequal scales configured before decomposition must multiply each constant before combining springs in series.
TEST_P( MfemCommonPlaneTest, constructs_with_scales_before_decomposition )
{
  constexpr RealT lower_mesh_penalty = 2.0;
  constexpr RealT upper_mesh_penalty = 6.0;
  constexpr RealT residual_tolerance = 1.0e-11;
  setMfemKinematicPenaltyScale( coupling_scheme_id, 2.0, 3.0 );
  enableCommonPlaneFormulation();
  redecomposeAndUpdate();
  const auto scaled_penalty_residual = getMfemContactResidual( coupling_scheme_id );
  // Resetting constants clears registered scales even while the decomposed meshes still contain earlier scales.
  setMfemKinematicConstantPenalty( coupling_scheme_id, lower_mesh_penalty, upper_mesh_penalty );
  EXPECT_FALSE( couplingScheme().hasContactFormulation() );
  ASSERT_TRUE( couplingScheme().init() );
  auto unscaled_penalty_result = couplingScheme().getContactFormulation()->computeNodalResiduals(
      couplingScheme().getMesh1().getView(), couplingScheme().getMesh2().getView(), {} );
  // Scaled constants 4 and 18 give stiffness 36/11; unscaled constants 2 and 6 give stiffness 1.5.
  unscaled_penalty_result.residual->Add( -11.0 / 24.0, scaled_penalty_residual );
  EXPECT_LT( parallelL2Norm( *unscaled_penalty_result.residual ), residual_tolerance );
}

// Each scaled mesh penalty must be positive and finite before a new formulation can be constructed.
TEST_P( MfemCommonPlaneTest, rejects_invalid_scales_at_construction )
{
  enableCommonPlaneFormulation();
  EXPECT_SLIC_ERROR( setMfemKinematicPenaltyScale( coupling_scheme_id, 0.0, 1.0 ) );
  EXPECT_SLIC_ERROR( setMfemKinematicPenaltyScale( coupling_scheme_id, 1.0, -1.0 ) );
  EXPECT_SLIC_ERROR( setMfemKinematicPenaltyScale( coupling_scheme_id, std::numeric_limits<RealT>::infinity(), 1.0 ) );
  setMfemKinematicPenaltyScale( coupling_scheme_id, 1.0, 1.0 );
  redecomposeAndUpdate();
}

// The factory must instantiate the exact Gauss rule requested by each supported representative polynomial order.
TEST_P( MfemCommonPlaneTest, factory_selects_configured_rule_generator )
{
  enableCommonPlaneFormulation();
  ASSERT_TRUE( couplingScheme().init() );
  const auto* formulation = couplingScheme().getContactFormulation();
  ASSERT_NE( formulation, nullptr );
  switch ( GetParam() ) {
    case one_point_polynomial_order:
      EXPECT_TRUE( usesCommonPlaneRule<1>( formulation ) );
      break;
    case two_point_polynomial_order:
      EXPECT_TRUE( usesCommonPlaneRule<2>( formulation ) );
      break;
    case three_point_polynomial_order:
      EXPECT_TRUE( usesCommonPlaneRule<3>( formulation ) );
      break;
    default:
      FAIL() << "Unexpected quadrature order " << GetParam();
  }
}

// Crossing nonmatching surfaces match legacy centroid pressure with one point. Higher-order rules integrate varying
// pressure on the same penetrating interval and must produce a different nodal load distribution.
TEST_P( MfemCommonPlaneTest, uses_configured_pressure_rule_for_interpenetrating_overlap )
{
  constexpr RealT residual_tolerance = 1.0e-11;
  const auto upper_nodes = upperBodyNodes();
  for ( int node = 0; node < upper_nodes.Size(); ++node ) {
    if ( upper_nodes[node] ) {
      const RealT x = ( *coordinates_ )[coordinate_space_->DofToVDof( node, 0 )];
      // The upper boundary crosses y=1.0 at x=0.55, inside a nonmatching element pair whose full midpoint is open.
      ( *coordinates_ )[coordinate_space_->DofToVDof( node, 1 )] += 0.04 * ( x - 0.3 );
    }
  }
  redecomposeAndUpdate();
  mfem::Vector legacy_local_residual( coordinate_space_->GetVSize() );
  legacy_local_residual = 0.0;
  getMfemResponse( coupling_scheme_id, legacy_local_residual );
  mfem::Vector legacy_true_residual( coordinate_space_->GetTrueVSize() );
  coordinate_space_->GetProlongationMatrix()->MultTranspose( legacy_local_residual, legacy_true_residual );
  ASSERT_GT( parallelL2Norm( legacy_true_residual ), 0.0 );
  enableCommonPlaneFormulation();
  redecomposeAndUpdate();
  auto residual_difference = getMfemContactResidual( coupling_scheme_id );
  residual_difference += legacy_true_residual;
  if ( GetParam() == one_point_polynomial_order ) {
    EXPECT_LT( parallelL2Norm( residual_difference ), residual_tolerance );
  } else {
    EXPECT_GT( parallelL2Norm( residual_difference ), residual_tolerance );
  }
}

// Residual Jacobians on parent-mesh displacement true DOFs differentiate through pair maps and MPI transfers.
TEST_P( MfemCommonPlaneTest, assembled_jacobian_matches_finite_difference )
{
  constexpr int spatial_dim = 2;
  // Shift only the upper body so projection endpoints do not coincide. Rotating either edge under perturbation
  // then tests normal and overlap derivatives without switching which endpoints limit the overlap.
  const auto upper_nodes = upperBodyNodes();
  for ( int node = 0; node < upper_nodes.Size(); ++node ) {
    if ( upper_nodes[node] ) {
      ( *coordinates_ )[coordinate_space_->DofToVDof( node, 0 )] += 0.13;
    }
  }
  enableCommonPlaneFormulation();
  redecomposeAndUpdate();
  auto assembled_jacobian = getMfemDfDx( coupling_scheme_id );
  ASSERT_NE( assembled_jacobian, nullptr );
  const mfem::ParGridFunction original_coordinates( *coordinates_ );
  mfem::VectorFunctionCoefficient coordinate_direction_coefficient( spatial_dim,
                                                                    []( const mfem::Vector& x, mfem::Vector& v ) {
                                                                      v.SetSize( spatial_dim );
                                                                      v[0] = 0.03 * x[0] + 0.01 * x[1];
                                                                      v[1] = 0.02 * x[0] - 0.04 * x[1];
                                                                    } );
  mfem::ParGridFunction coordinate_direction( coordinate_space_.get() );
  coordinate_direction.ProjectCoefficient( coordinate_direction_coefficient );
  mfem::Vector coordinate_true_direction;
  coordinate_direction.GetTrueDofs( coordinate_true_direction );
  mfem::Vector jacobian_action( coordinate_true_direction.Size() );
  assembled_jacobian->Mult( coordinate_true_direction, jacobian_action );
  constexpr RealT finite_difference_step = 1.0e-6;
  constexpr RealT finite_difference_tolerance = 1.0e-7;
  *coordinates_ = original_coordinates;
  coordinates_->Add( finite_difference_step, coordinate_direction );
  redecomposeAndUpdate();
  const auto increased_coordinate_residual = getMfemContactResidual( coupling_scheme_id );
  *coordinates_ = original_coordinates;
  coordinates_->Add( -finite_difference_step, coordinate_direction );
  redecomposeAndUpdate();
  auto finite_difference_error = getMfemContactResidual( coupling_scheme_id );
  finite_difference_error.Neg();
  finite_difference_error += increased_coordinate_residual;
  finite_difference_error /= 2.0 * finite_difference_step;
  finite_difference_error -= jacobian_action;
  EXPECT_LT( parallelL2Norm( finite_difference_error ),
             finite_difference_tolerance * ( 1.0 + parallelL2Norm( jacobian_action ) ) );
  *coordinates_ = original_coordinates;
}

// Enabling after legacy evaluation, changing penalties, rebuilding, and disabling must preserve transfers and signs.
TEST_P( MfemCommonPlaneTest, rebuilds_and_restores_legacy_response )
{
  constexpr RealT residual_tolerance = 1.0e-11;
  redecomposeAndUpdate();
  couplingScheme().setFixedBinning( true );  // Re-enabling legacy contact must discard a previously cached pair list.

  enableCommonPlaneFormulation();
  redecomposeAndUpdate();
  const auto baseline_residual = getMfemContactResidual( coupling_scheme_id );
  setMfemKinematicConstantPenalty( coupling_scheme_id, 4.0, 12.0 );
  redecomposeAndUpdate();
  auto doubled_penalty_residual = getMfemContactResidual( coupling_scheme_id );
  auto doubled_penalty_residual_difference = doubled_penalty_residual;
  doubled_penalty_residual_difference.Add( -2.0, baseline_residual );
  EXPECT_LT( parallelL2Norm( doubled_penalty_residual_difference ), residual_tolerance );
  EXPECT_SLIC_ERROR( setEnforcementLocation( coupling_scheme_id, EnforcementLocation::Nodal ) );
  setMfemLORFactor( coupling_scheme_id, 2 );
  EXPECT_EQ( couplingScheme().getMfemMeshData()->GetLORMesh(), nullptr );
  EXPECT_TRUE( couplingScheme().getParameters().use_common_plane_formulation );
  EXPECT_EQ( couplingScheme().getParameters().enforcement_location, EnforcementLocation::QuadraturePoint );
  setMfemUseCommonPlaneFormulation( coupling_scheme_id, false );
  EXPECT_FALSE( couplingScheme().hasContactFormulation() );
  EXPECT_FALSE( couplingScheme().hasMfemJacobianData() );
  EXPECT_FALSE( couplingScheme().hasFixedBinning() );
  EXPECT_FALSE( couplingScheme().isBinned() );
  redecomposeAndUpdate();
  mfem::Vector legacy_local_residual( coordinate_space_->GetVSize() );
  legacy_local_residual = 0.0;
  getMfemResponse( coupling_scheme_id, legacy_local_residual );
  mfem::Vector legacy_true_residual( coordinate_space_->GetTrueVSize() );
  coordinate_space_->GetProlongationMatrix()->MultTranspose( legacy_local_residual, legacy_true_residual );
  doubled_penalty_residual += legacy_true_residual;
  EXPECT_LT( parallelL2Norm( doubled_penalty_residual ), residual_tolerance );
  setMfemUseCommonPlaneFormulation( coupling_scheme_id, true );
  redecomposeAndUpdate();
  EXPECT_GT( parallelL2Norm( getMfemContactResidual( coupling_scheme_id ) ), 0.0 );
}

#endif  // TRIBOL_USE_ENZYME

// Factory-selected execution modes must preserve residuals, and evaluator selection must control matrix availability.
TEST_P( MfemCommonPlaneTest, execution_modes_preserve_residuals_and_jacobian_selection )
{
  constexpr RealT residual_tolerance = 1.0e-11;
  mfem::Vector sequential_residual;
  for ( const auto execution_mode : { ExecutionMode::Sequential
#ifdef TRIBOL_USE_OPENMP
                                      ,
                                      ExecutionMode::OpenMP
#endif
        } ) {
    SCOPED_TRACE( static_cast<int>( execution_mode ) );
    finalize();
    registerScheme( execution_mode );
    enableCommonPlaneFormulation( false );
    redecomposeAndUpdate();
    ASSERT_TRUE( couplingScheme().hasContactFormulation() );
    EXPECT_FALSE( couplingScheme().hasMfemJacobianData() );
    EXPECT_EQ( couplingScheme().getContactFormulationResult().df_dx, nullptr );
    auto residual_difference = getMfemContactResidual( coupling_scheme_id );
    if ( execution_mode == ExecutionMode::Sequential ) {
      sequential_residual.SetSize( residual_difference.Size() );
      sequential_residual = residual_difference;
    }
    residual_difference -= sequential_residual;
    EXPECT_LT( parallelL2Norm( residual_difference ), residual_tolerance );
#ifdef TRIBOL_USE_ENZYME
    enableCommonPlaneFormulation();
    redecomposeAndUpdate();
    ASSERT_NE( couplingScheme().getContactFormulationResult().df_dx, nullptr );
    residual_difference = getMfemContactResidual( coupling_scheme_id );
    residual_difference -= sequential_residual;
    EXPECT_LT( parallelL2Norm( residual_difference ), residual_tolerance );
#endif
  }
}

// A rejected switch must keep the active formulation, its Jacobian setting, and its transfer data unchanged.
TEST_P( MfemCommonPlaneTest, rejected_switch_keeps_active_formulation )
{
  constexpr RealT residual_tolerance = 1.0e-11;
  enableCommonPlaneFormulation( false );
  redecomposeAndUpdate();
  const auto retained_residual = getMfemContactResidual( coupling_scheme_id );
  const auto* retained_formulation = couplingScheme().getContactFormulation();
  ASSERT_NE( retained_formulation, nullptr );

  // An unsupported rate penalty makes any new selection invalid, including a switch to Jacobian calculation.
  auto& penalty = couplingScheme().getEnforcementOptions().penalty_options;
  penalty.rate_calc_set = true;
  penalty.rate_calculation = RATE_CONSTANT;
  EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( coupling_scheme_id, true, true ) );
  EXPECT_TRUE( couplingScheme().getParameters().use_common_plane_formulation );
  EXPECT_FALSE( couplingScheme().getParameters().common_plane_compute_jacobian );
  EXPECT_EQ( couplingScheme().getContactFormulation(), retained_formulation );
  EXPECT_FALSE( couplingScheme().hasMfemJacobianData() );

  // Restoring the configuration must reuse the retained formulation and reproduce the earlier residual.
  penalty.rate_calc_set = false;
  redecomposeAndUpdate();
  EXPECT_EQ( couplingScheme().getContactFormulation(), retained_formulation );
  auto restored_residual_difference = getMfemContactResidual( coupling_scheme_id );
  restored_residual_difference -= retained_residual;
  EXPECT_LT( parallelL2Norm( restored_residual_difference ), residual_tolerance );
}

// Rejected configuration must leave the legacy selection and its resources intact.
TEST_P( MfemCommonPlaneTest, rejects_unsupported_settings_before_selection )
{
  constexpr RealT lower_mesh_penalty = 2.0;
  constexpr RealT upper_mesh_penalty = 6.0;
  const auto check_selection_rejected = [&]() {
    EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( coupling_scheme_id, true, false ) );
    EXPECT_FALSE( couplingScheme().getParameters().use_common_plane_formulation );
    EXPECT_FALSE( couplingScheme().hasContactFormulation() );
    EXPECT_FALSE( couplingScheme().hasMfemJacobianData() );
  };
  constexpr IndexT unregistered_coupling_scheme_id = 1;
  EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( unregistered_coupling_scheme_id, true, false ) );
  setMfemKinematicConstantPenalty( coupling_scheme_id, 0.0, upper_mesh_penalty );
  check_selection_rejected();
  setMfemKinematicConstantPenalty( coupling_scheme_id, lower_mesh_penalty, upper_mesh_penalty );
  couplingScheme().getParameters().enforcement_location = EnforcementLocation::Nodal;
  check_selection_rejected();
  couplingScheme().getParameters().enforcement_location = EnforcementLocation::QuadraturePoint;
  auto& penalty = couplingScheme().getEnforcementOptions().penalty_options;
  penalty.kinematic_calculation = KINEMATIC_ELEMENT;
  check_selection_rejected();
  penalty.kinematic_calculation = KINEMATIC_CONSTANT;
  penalty.rate_calc_set = true;
  penalty.rate_calculation = RATE_CONSTANT;
  check_selection_rejected();
  penalty.rate_calc_set = false;
  setResidualGap( coupling_scheme_id, 0.1 );
  check_selection_rejected();
  setResidualGap( coupling_scheme_id, 0.0 );
  constexpr int unsupported_quadrature_order = 3;
  EXPECT_SLIC_ERROR( setQuadratureOrder( coupling_scheme_id, unsupported_quadrature_order ) );
  EXPECT_SLIC_ERROR( enableQuadratureSubdivision( coupling_scheme_id, true, default_num_quadrature_subdivisions ) );
  enableCommonPlaneFormulation( false );
  EXPECT_SLIC_ERROR( enableQuadratureSubdivision( coupling_scheme_id, true, default_num_quadrature_subdivisions ) );
  for ( int order = minimum_supported_quadrature_order; order <= maximum_supported_quadrature_order; ++order ) {
    setQuadratureOrder( coupling_scheme_id, order );
    EXPECT_FALSE( couplingScheme().hasContactFormulation() );
    EXPECT_EQ( couplingScheme().getParameters().quadrature_order, order );
    ASSERT_TRUE( couplingScheme().init() );
    EXPECT_TRUE( couplingScheme().hasContactFormulation() );
  }
  EXPECT_SLIC_ERROR( setQuadratureOrder( coupling_scheme_id, maximum_supported_quadrature_order + 1 ) );
  EXPECT_EQ( couplingScheme().getParameters().quadrature_order, maximum_supported_quadrature_order );
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
  const Configuration unsupported_configurations[] = { { COMMON_PLANE, VISCOUS_TANGENTIAL, PENALTY, NO_CASE },
                                                       { COMMON_PLANE, FRICTIONLESS, LAGRANGE_MULTIPLIER, NO_CASE },
                                                       { COMMON_PLANE, FRICTIONLESS, PENALTY, TIED_NORMAL },
                                                       { COMMON_PLANE, FRICTIONLESS, PENALTY, TIED_FULL },
                                                       { SINGLE_MORTAR, FRICTIONLESS, PENALTY, NO_CASE } };
  for ( const auto configuration : unsupported_configurations ) {
    constexpr int horizontal_element_count = 1;
    constexpr int vertical_element_count = 1;
    // A single unit-square element supplies distinct bottom and top boundaries. Contact geometry is intentionally
    // unimportant because each configuration must be rejected before formulation resources are constructed.
    mfem::ParMesh parent_mesh = shared::ParMeshBuilder(
        MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( horizontal_element_count, vertical_element_count ) );
    constexpr int spatial_dim = 2;
    constexpr int linear_order = 1;
    mfem::H1_FECollection coordinate_collection( linear_order, spatial_dim );
    mfem::ParFiniteElementSpace coordinate_space( &parent_mesh, &coordinate_collection, spatial_dim );
    mfem::ParGridFunction coordinates( &coordinate_space );
    parent_mesh.GetNodes( coordinates );
    constexpr IndexT coupling_scheme_id = 0;
    constexpr IndexT lower_mesh_id = 0;
    constexpr IndexT upper_mesh_id = 1;
    constexpr int mesh_boundary_attribute_1 = 1;
    constexpr int mesh_boundary_attribute_2 = 3;
    registerMfemCouplingScheme( coupling_scheme_id, lower_mesh_id, upper_mesh_id, parent_mesh, coordinates,
                                { mesh_boundary_attribute_1 }, { mesh_boundary_attribute_2 }, SURFACE_TO_SURFACE,
                                configuration.contact_case, configuration.method, configuration.model,
                                configuration.enforcement, BINNING_GRID, ExecutionMode::Sequential );
    constexpr RealT lower_mesh_penalty = 2.0;
    constexpr RealT upper_mesh_penalty = 6.0;
    setMfemKinematicConstantPenalty( coupling_scheme_id, lower_mesh_penalty, upper_mesh_penalty );
    EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( coupling_scheme_id, true, false ) );
    const auto& coupling_scheme = CouplingSchemeManager::getInstance().at( coupling_scheme_id );
    EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
    EXPECT_FALSE( coupling_scheme.hasMfemJacobianData() );
    finalize();
  }
}

// Curved 2D coordinate spaces without LOR subdivision and three-dimensional parent meshes are rejected before
// selection.
TEST( CommonPlaneSelection, rejects_higher_order_without_lor_and_three_dimensional_geometry )
{
  constexpr int spatial_dim = 2;
  constexpr int three_dimensional = 3;
  for ( const int geometry_dim : { spatial_dim, three_dimensional } ) {
    constexpr int linear_order = 1;
    constexpr int quadratic_order = 2;
    const int coordinate_order = geometry_dim == spatial_dim ? quadratic_order : linear_order;
    constexpr int horizontal_element_count = 1;
    constexpr int vertical_element_count = 1;
    mfem::ParMesh parent_mesh = shared::ParMeshBuilder(
        MPI_COMM_WORLD, geometry_dim == spatial_dim
                            ? shared::MeshBuilder::SquareMesh( horizontal_element_count, vertical_element_count )
                            : shared::MeshBuilder::CubeMesh( 1, 1, 1 ) );
    mfem::H1_FECollection coordinate_collection( coordinate_order, geometry_dim );
    mfem::ParFiniteElementSpace coordinate_space( &parent_mesh, &coordinate_collection, geometry_dim );
    mfem::ParGridFunction coordinates( &coordinate_space );
    parent_mesh.GetNodes( coordinates );
    constexpr IndexT coupling_scheme_id = 0;
    constexpr IndexT lower_mesh_id = 0;
    constexpr IndexT upper_mesh_id = 1;
    constexpr int mesh_boundary_attribute_1 = 1;
    constexpr int mesh_boundary_attribute_2 = 3;
    registerMfemCouplingScheme( coupling_scheme_id, lower_mesh_id, upper_mesh_id, parent_mesh, coordinates,
                                { mesh_boundary_attribute_1 }, { mesh_boundary_attribute_2 }, SURFACE_TO_SURFACE,
                                NO_CASE, COMMON_PLANE, FRICTIONLESS, PENALTY, BINNING_GRID, ExecutionMode::Sequential );
    constexpr RealT lower_mesh_penalty = 2.0;
    constexpr RealT upper_mesh_penalty = 6.0;
    setMfemKinematicConstantPenalty( coupling_scheme_id, lower_mesh_penalty, upper_mesh_penalty );
    auto& coupling_scheme = CouplingSchemeManager::getInstance().at( coupling_scheme_id );
    if ( geometry_dim == spatial_dim ) {
      // Disable LOR transfer on the quadratic 2D mesh so Common Plane sees an unrefined higher-order surface.
      coupling_scheme.setMfemMeshData( std::make_unique<MfemMeshData>(
          lower_mesh_id, upper_mesh_id, parent_mesh, coordinates, std::set<int>{ mesh_boundary_attribute_1 },
          std::set<int>{ mesh_boundary_attribute_2 }, ExecutionMode::Sequential, MemorySpace::Host, false ) );
    }
    EXPECT_SLIC_ERROR( setMfemUseCommonPlaneFormulation( coupling_scheme_id, true, false ) );
    EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
    EXPECT_FALSE( coupling_scheme.hasMfemJacobianData() );
    finalize();
  }
}

// Higher-order 2D coordinates transferred through an LOR mesh match legacy one-point residuals and finite-difference
// checks of the residual Jacobian on parent-mesh displacement true DOFs.
TEST( CommonPlaneSelection, evaluates_residuals_and_jacobians_with_lor_subdivision )
{
  constexpr int lower_boundary_edge_count = 2;
  constexpr int upper_boundary_edge_count = 3;
  constexpr int vertical_element_count = 1;
  constexpr int upper_contact_boundary_attribute = 5;
  constexpr int upper_noncontact_boundary_attribute = 6;
  constexpr int upper_body_attribute = 2;
  // The lower body's top boundary has two edges. The upper body's bottom boundary has three edges, starts 0.01 below
  // the lower top boundary, and is shifted right by 0.13 so projection endpoints do not coincide under perturbations.
  mfem::ParMesh parent_mesh = shared::ParMeshBuilder(
      MPI_COMM_WORLD, shared::MeshBuilder::Unify(
                          { shared::MeshBuilder::SquareMesh( lower_boundary_edge_count, vertical_element_count ),
                            shared::MeshBuilder::SquareMesh( upper_boundary_edge_count, vertical_element_count )
                                .translate( { 0.13, 0.99 } )
                                .updateBdrAttrib( 1, upper_contact_boundary_attribute )
                                .updateBdrAttrib( 3, upper_noncontact_boundary_attribute )
                                .updateAttrib( 1, upper_body_attribute ) } ) );
  constexpr int spatial_dim = 2;
  constexpr int quadratic_order = 2;
  parent_mesh.SetCurvature( quadratic_order, /*discont=*/false, /*space_dim=*/spatial_dim, mfem::Ordering::byVDIM );
  auto* parent_mesh_nodes = dynamic_cast<mfem::ParGridFunction*>( parent_mesh.GetNodes() );
  ASSERT_NE( parent_mesh_nodes, nullptr );
  auto& coordinate_space = *parent_mesh_nodes->ParFESpace();
  mfem::ParGridFunction coordinates( &coordinate_space );
  coordinates = *parent_mesh_nodes;

  constexpr IndexT coupling_scheme_id = 0;
  constexpr IndexT lower_mesh_id = 0;
  constexpr IndexT upper_mesh_id = 1;
  constexpr int lower_contact_boundary_attribute = 3;
  registerMfemCouplingScheme( coupling_scheme_id, lower_mesh_id, upper_mesh_id, parent_mesh, coordinates,
                              { lower_contact_boundary_attribute }, { upper_contact_boundary_attribute },
                              SURFACE_TO_SURFACE, NO_CASE, COMMON_PLANE, FRICTIONLESS, PENALTY, BINNING_GRID,
                              ExecutionMode::Sequential );
  constexpr RealT lower_mesh_penalty = 2.0;
  constexpr RealT upper_mesh_penalty = 6.0;
  setMfemKinematicConstantPenalty( coupling_scheme_id, lower_mesh_penalty, upper_mesh_penalty );

  const auto& coupling_scheme = CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  ASSERT_NE( coupling_scheme.getMfemMeshData()->GetLORMesh(), nullptr );
  EXPECT_EQ( coupling_scheme.getMfemMeshData()->GetLORFactor(), quadratic_order );
  EXPECT_TRUE( coupling_scheme.getMfemMeshData()->IsLORFieldTransferEnabled() );

  const auto parallelL2Norm = []( const mfem::Vector& v ) {
    return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, v, v ) );
  };
  const auto redecomposeAndUpdate = [&]() {
    constexpr int contact_rank_count = 1;
    updateMfemParallelDecomposition( contact_rank_count, true );
    RealT dt = 1.0;
    ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
  };

  redecomposeAndUpdate();
  mfem::Vector legacy_local_residual( coordinate_space.GetVSize() );
  legacy_local_residual = 0.0;
  getMfemResponse( coupling_scheme_id, legacy_local_residual );
  mfem::Vector legacy_true_residual( coordinate_space.GetTrueVSize() );
  coordinate_space.GetProlongationMatrix()->MultTranspose( legacy_local_residual, legacy_true_residual );
  ASSERT_GT( parallelL2Norm( legacy_true_residual ), 0.0 );

  setMfemUseCommonPlaneFormulation( coupling_scheme_id, true, false );
  EXPECT_NE( coupling_scheme.getMfemMeshData()->GetLORMesh(), nullptr );
  EXPECT_EQ( coupling_scheme.getMfemMeshData()->GetLORFactor(), quadratic_order );
  EXPECT_TRUE( coupling_scheme.getMfemMeshData()->IsLORFieldTransferEnabled() );
  setQuadratureOrder( coupling_scheme_id, one_point_polynomial_order );
  redecomposeAndUpdate();
  EXPECT_EQ( getMfemDfDx( coupling_scheme_id ), nullptr );
  auto residual_difference = getMfemContactResidual( coupling_scheme_id );
  residual_difference += legacy_true_residual;
  constexpr RealT residual_tolerance = 1.0e-11;
  EXPECT_LT( parallelL2Norm( residual_difference ), residual_tolerance );

#ifdef TRIBOL_USE_ENZYME
  // Enable the Jacobian, then override the LOR factor while the formulation is active and verify the residual Jacobian
  // on parent-mesh displacement true DOFs via finite differences.
  constexpr int overridden_lor_factor = 3;
  setMfemUseCommonPlaneFormulation( coupling_scheme_id, true, true );
  setMfemLORFactor( coupling_scheme_id, overridden_lor_factor );
  EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
  EXPECT_TRUE( coupling_scheme.hasMfemJacobianData() );
  EXPECT_EQ( coupling_scheme.getMfemMeshData()->GetLORFactor(), overridden_lor_factor );
  redecomposeAndUpdate();
  auto assembled_jacobian = getMfemDfDx( coupling_scheme_id );
  ASSERT_NE( assembled_jacobian, nullptr );

  const mfem::ParGridFunction original_coordinates( coordinates );
  mfem::VectorFunctionCoefficient coordinate_direction_coefficient( spatial_dim,
                                                                    []( const mfem::Vector& x, mfem::Vector& v ) {
                                                                      v.SetSize( spatial_dim );
                                                                      v[0] = 0.03 * x[0] + 0.01 * x[1];
                                                                      v[1] = 0.02 * x[0] - 0.04 * x[1];
                                                                    } );
  mfem::ParGridFunction coordinate_direction( &coordinate_space );
  coordinate_direction.ProjectCoefficient( coordinate_direction_coefficient );
  mfem::Vector coordinate_true_direction;
  coordinate_direction.GetTrueDofs( coordinate_true_direction );
  mfem::Vector jacobian_action( coordinate_true_direction.Size() );
  assembled_jacobian->Mult( coordinate_true_direction, jacobian_action );
  ASSERT_GT( parallelL2Norm( jacobian_action ), 0.0 );
  const RealT jacobian_action_norm = parallelL2Norm( jacobian_action );

  constexpr RealT finite_difference_step = 1.0e-6;
  constexpr RealT finite_difference_tolerance = 1.0e-7;
  coordinates = original_coordinates;
  coordinates.Add( finite_difference_step, coordinate_direction );
  redecomposeAndUpdate();
  const auto increased_coordinate_residual = getMfemContactResidual( coupling_scheme_id );
  coordinates = original_coordinates;
  coordinates.Add( -finite_difference_step, coordinate_direction );
  redecomposeAndUpdate();
  auto finite_difference_error = getMfemContactResidual( coupling_scheme_id );
  finite_difference_error.Neg();
  finite_difference_error += increased_coordinate_residual;
  finite_difference_error /= 2.0 * finite_difference_step;
  finite_difference_error -= jacobian_action;
  EXPECT_LT( parallelL2Norm( finite_difference_error ), finite_difference_tolerance * ( 1.0 + jacobian_action_norm ) );
#endif

  finalize();
}

// Opting into quadrature subdivision on a quadratic 2D mesh disables LOR, evaluates direct quadratic residuals and
// finite-difference-verified Jacobians on parent-mesh true DOFs, and restores LOR when disabled.
TEST( CommonPlaneSelection, evaluates_quadratic_residuals_and_jacobians_with_subdivision )
{
  constexpr int lower_boundary_edge_count = 2;
  constexpr int upper_boundary_edge_count = 3;
  constexpr int vertical_element_count = 1;
  constexpr int upper_contact_boundary_attribute = 5;
  constexpr int upper_noncontact_boundary_attribute = 6;
  constexpr int upper_body_attribute = 2;
  // Two quadratic unit-square bodies: lower has 2 contact edges on y=1.0, upper has 3 contact edges on y=0.99 shifted
  // right by 0.13 so overlap endpoints do not coincide under coordinate perturbations.
  mfem::ParMesh parent_mesh = shared::ParMeshBuilder(
      MPI_COMM_WORLD, shared::MeshBuilder::Unify(
                          { shared::MeshBuilder::SquareMesh( lower_boundary_edge_count, vertical_element_count ),
                            shared::MeshBuilder::SquareMesh( upper_boundary_edge_count, vertical_element_count )
                                .translate( { 0.13, 0.99 } )
                                .updateBdrAttrib( 1, upper_contact_boundary_attribute )
                                .updateBdrAttrib( 3, upper_noncontact_boundary_attribute )
                                .updateAttrib( 1, upper_body_attribute ) } ) );
  constexpr int spatial_dim = 2;
  constexpr int quadratic_order = 2;
  parent_mesh.SetCurvature( quadratic_order, /*discont=*/false, /*space_dim=*/spatial_dim, mfem::Ordering::byVDIM );
  auto* parent_mesh_nodes = dynamic_cast<mfem::ParGridFunction*>( parent_mesh.GetNodes() );
  ASSERT_NE( parent_mesh_nodes, nullptr );
  auto& coordinate_space = *parent_mesh_nodes->ParFESpace();
  mfem::ParGridFunction coordinates( &coordinate_space );
  coordinates = *parent_mesh_nodes;

  constexpr IndexT coupling_scheme_id = 0;
  constexpr IndexT lower_mesh_id = 0;
  constexpr IndexT upper_mesh_id = 1;
  constexpr int lower_contact_boundary_attribute = 3;
  registerMfemCouplingScheme( coupling_scheme_id, lower_mesh_id, upper_mesh_id, parent_mesh, coordinates,
                              { lower_contact_boundary_attribute }, { upper_contact_boundary_attribute },
                              SURFACE_TO_SURFACE, NO_CASE, COMMON_PLANE, FRICTIONLESS, PENALTY, BINNING_GRID,
                              ExecutionMode::Sequential );
  constexpr RealT lower_mesh_penalty = 2.0;
  constexpr RealT upper_mesh_penalty = 6.0;
  setMfemKinematicConstantPenalty( coupling_scheme_id, lower_mesh_penalty, upper_mesh_penalty );

  auto& coupling_scheme = CouplingSchemeManager::getInstance().at( coupling_scheme_id );
  // Subdivision requires opted-in Common Plane; calls before selection must fail without disabling LOR.
  EXPECT_SLIC_ERROR( enableQuadratureSubdivision( coupling_scheme_id, true, default_num_quadrature_subdivisions ) );
  ASSERT_NE( coupling_scheme.getMfemMeshData()->GetLORMesh(), nullptr );
  EXPECT_EQ( coupling_scheme.getMfemMeshData()->GetLORFactor(), quadratic_order );
  EXPECT_TRUE( coupling_scheme.getMfemMeshData()->IsLORFieldTransferEnabled() );

  setMfemUseCommonPlaneFormulation( coupling_scheme_id, true, false );
  enableQuadratureSubdivision( coupling_scheme_id, true, min_num_quadrature_subdivisions );
  EXPECT_TRUE( coupling_scheme.getParameters().use_quadrature_subdivision );
  EXPECT_EQ( coupling_scheme.getParameters().num_quadrature_subdivisions, min_num_quadrature_subdivisions );
  enableQuadratureSubdivision( coupling_scheme_id, true, max_num_quadrature_subdivisions );
  EXPECT_EQ( coupling_scheme.getParameters().num_quadrature_subdivisions, max_num_quadrature_subdivisions );
  EXPECT_SLIC_ERROR( enableQuadratureSubdivision( coupling_scheme_id, true, min_num_quadrature_subdivisions - 1 ) );
  EXPECT_SLIC_ERROR( enableQuadratureSubdivision( coupling_scheme_id, true, max_num_quadrature_subdivisions + 1 ) );
  enableQuadratureSubdivision( coupling_scheme_id, true, default_num_quadrature_subdivisions );
  EXPECT_EQ( coupling_scheme.getMfemMeshData()->GetLORMesh(), nullptr );
  EXPECT_EQ( coupling_scheme.getMfemMeshData()->GetLORFactor(), 0 );
  EXPECT_FALSE( coupling_scheme.getMfemMeshData()->IsLORFieldTransferEnabled() );

  const auto parallelL2Norm = []( const mfem::Vector& v ) {
    return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, v, v ) );
  };
  const auto redecomposeAndUpdate = [&]() {
    constexpr int contact_rank_count = 1;
    updateMfemParallelDecomposition( contact_rank_count, true );
    RealT dt = 1.0;
    ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
  };

  setQuadratureOrder( coupling_scheme_id, one_point_polynomial_order );
  redecomposeAndUpdate();
  EXPECT_EQ( getMfemDfDx( coupling_scheme_id ), nullptr );
  EXPECT_EQ( coupling_scheme.getMfemMeshData()->GetFaceOrder(), QUADRATIC_LAGRANGE );
  EXPECT_GT( parallelL2Norm( getMfemContactResidual( coupling_scheme_id ) ), 0.0 );

#ifdef TRIBOL_USE_ENZYME
  // Enable the Jacobian, then change pair-local subdivision and verify the residual Jacobian on parent-mesh
  // displacement true DOFs via finite differences.
  constexpr int overridden_num_subdivisions = 3;
  setMfemUseCommonPlaneFormulation( coupling_scheme_id, true, true );
  enableQuadratureSubdivision( coupling_scheme_id, true, overridden_num_subdivisions );
  EXPECT_FALSE( coupling_scheme.hasContactFormulation() );
  EXPECT_TRUE( coupling_scheme.hasMfemJacobianData() );
  EXPECT_EQ( coupling_scheme.getParameters().num_quadrature_subdivisions, overridden_num_subdivisions );
  EXPECT_EQ( coupling_scheme.getMfemMeshData()->GetLORMesh(), nullptr );
  redecomposeAndUpdate();
  auto assembled_jacobian = getMfemDfDx( coupling_scheme_id );
  ASSERT_NE( assembled_jacobian, nullptr );

  const mfem::ParGridFunction original_coordinates( coordinates );
  mfem::VectorFunctionCoefficient coordinate_direction_coefficient( spatial_dim,
                                                                    []( const mfem::Vector& x, mfem::Vector& v ) {
                                                                      v.SetSize( spatial_dim );
                                                                      v[0] = 0.03 * x[0] + 0.01 * x[1];
                                                                      v[1] = 0.02 * x[0] - 0.04 * x[1];
                                                                    } );
  mfem::ParGridFunction coordinate_direction( &coordinate_space );
  coordinate_direction.ProjectCoefficient( coordinate_direction_coefficient );
  mfem::Vector coordinate_true_direction;
  coordinate_direction.GetTrueDofs( coordinate_true_direction );
  mfem::Vector jacobian_action( coordinate_true_direction.Size() );
  assembled_jacobian->Mult( coordinate_true_direction, jacobian_action );
  ASSERT_GT( parallelL2Norm( jacobian_action ), 0.0 );
  const RealT jacobian_action_norm = parallelL2Norm( jacobian_action );

  constexpr RealT finite_difference_step = 1.0e-6;
  constexpr RealT finite_difference_tolerance = 1.0e-7;
  coordinates = original_coordinates;
  coordinates.Add( finite_difference_step, coordinate_direction );
  redecomposeAndUpdate();
  const auto increased_coordinate_residual = getMfemContactResidual( coupling_scheme_id );
  coordinates = original_coordinates;
  coordinates.Add( -finite_difference_step, coordinate_direction );
  redecomposeAndUpdate();
  auto finite_difference_error = getMfemContactResidual( coupling_scheme_id );
  finite_difference_error.Neg();
  finite_difference_error += increased_coordinate_residual;
  finite_difference_error /= 2.0 * finite_difference_step;
  finite_difference_error -= jacobian_action;
  EXPECT_LT( parallelL2Norm( finite_difference_error ), finite_difference_tolerance * ( 1.0 + jacobian_action_norm ) );
#endif

  // Disabling quadrature subdivision restores LOR while keeping Common Plane active.
  enableQuadratureSubdivision( coupling_scheme_id, false );
  EXPECT_FALSE( coupling_scheme.getParameters().use_quadrature_subdivision );
  EXPECT_NE( coupling_scheme.getMfemMeshData()->GetLORMesh(), nullptr );
  EXPECT_EQ( coupling_scheme.getMfemMeshData()->GetLORFactor(), quadratic_order );
  EXPECT_TRUE( coupling_scheme.getMfemMeshData()->IsLORFieldTransferEnabled() );

  // Re-enabling subdivision and then disabling Common Plane must also reset subdivision and restore LOR.
  enableQuadratureSubdivision( coupling_scheme_id, true, default_num_quadrature_subdivisions );
  EXPECT_EQ( coupling_scheme.getMfemMeshData()->GetLORMesh(), nullptr );
  setMfemUseCommonPlaneFormulation( coupling_scheme_id, false );
  EXPECT_FALSE( coupling_scheme.getParameters().use_quadrature_subdivision );
  EXPECT_NE( coupling_scheme.getMfemMeshData()->GetLORMesh(), nullptr );
  EXPECT_EQ( coupling_scheme.getMfemMeshData()->GetLORFactor(), quadratic_order );
  EXPECT_TRUE( coupling_scheme.getMfemMeshData()->IsLORFieldTransferEnabled() );

  finalize();
}

// CommonPlaneFormulation computes allowable timesteps for nodal penalty CFL, pair spring-mass CFL, velocity projection,
// and combined criteria, and reports a SLIC error when an enabled criterion is missing its required fields.
TEST( MfemCommonPlaneTimestepTest, computes_composable_cfl_and_velocity_timesteps )
{
  constexpr IndexT coupling_scheme_id = 0;
  constexpr int lower_mesh_id = 0;
  constexpr int upper_mesh_id = 1;
  constexpr int lower_contact_attribute = 1;
  constexpr int upper_contact_attribute = 2;
  constexpr int spatial_dim = 2;
  constexpr int linear_order = 1;

  // Build two 1x1 2D quads overlapping by 0.02 along y (lower y in [0, 1.01], upper y in [0.99, 2.0]).
  mfem::Mesh serial_mesh( spatial_dim, 8, 2, 2 );
  serial_mesh.AddVertex( 0.0, 0.0 );
  serial_mesh.AddVertex( 1.0, 0.0 );
  serial_mesh.AddVertex( 1.0, 1.01 );
  serial_mesh.AddVertex( 0.0, 1.01 );
  serial_mesh.AddVertex( 0.0, 0.99 );
  serial_mesh.AddVertex( 1.0, 0.99 );
  serial_mesh.AddVertex( 1.0, 2.0 );
  serial_mesh.AddVertex( 0.0, 2.0 );
  const int lower_quad[4] = { 0, 1, 2, 3 };
  const int upper_quad[4] = { 4, 5, 6, 7 };
  serial_mesh.AddQuad( lower_quad, 1 );
  serial_mesh.AddQuad( upper_quad, 2 );
  const int lower_edge[2] = { 3, 2 };
  const int upper_edge[2] = { 4, 5 };
  serial_mesh.AddBdrSegment( lower_edge, lower_contact_attribute );
  serial_mesh.AddBdrSegment( upper_edge, upper_contact_attribute );
  serial_mesh.FinalizeQuadMesh( 1, 0, true );

  mfem::ParMesh parallel_mesh( MPI_COMM_WORLD, serial_mesh );
  mfem::H1_FECollection fe_collection( linear_order, spatial_dim );
  mfem::ParFiniteElementSpace coord_space( &parallel_mesh, &fe_collection, spatial_dim, mfem::Ordering::byNODES );
  mfem::ParGridFunction coords( &coord_space );
  parallel_mesh.GetNodes( coords );

  // Uniform inverse diagonal mass = 0.25 on both blocks (sqrt(inv_m) = 0.5).
  mfem::ParGridFunction inv_mass( &coord_space );
  inv_mass = 0.25;

  // Approaching velocities: lower block (attribute 1) +vy = 10.0, upper block (attribute 2) -vy = -10.0.
  mfem::ParGridFunction velocity( &coord_space );
  mfem::Vector lower_vel( { 0.0, 10.0 } );
  mfem::Vector upper_vel( { 0.0, -10.0 } );
  mfem::VectorConstantCoefficient lower_vel_coeff( lower_vel );
  mfem::VectorConstantCoefficient upper_vel_coeff( upper_vel );
  mfem::PWVectorCoefficient vel_coeff( spatial_dim );
  vel_coeff.UpdateCoefficient( 1, lower_vel_coeff );
  vel_coeff.UpdateCoefficient( 2, upper_vel_coeff );
  velocity.ProjectCoefficient( vel_coeff );

  registerMfemCouplingScheme( coupling_scheme_id, lower_mesh_id, upper_mesh_id, parallel_mesh, coords,
                              { lower_contact_attribute }, { upper_contact_attribute }, SURFACE_TO_SURFACE, NO_CASE,
                              COMMON_PLANE, FRICTIONLESS, PENALTY, BINNING_GRID, ExecutionMode::Sequential );
  // Series penalty = 8 * 8 / (8 + 8) = 4.0.
  setMfemKinematicConstantPenalty( coupling_scheme_id, 8.0, 8.0 );
  setMfemUseCommonPlaneFormulation( coupling_scheme_id, true, false );
  setQuadratureOrder( coupling_scheme_id, 1 );  // Single midpoint quadrature point (phi = 0.5 on both endpoints).

  // Enabling nodal_penalty_cfl without registering inverse mass must report a SLIC error.
  enableTimestepVote( coupling_scheme_id, true );
  setTimestepCriteria( coupling_scheme_id, TimestepCriteria{ false, false, true, false } );
  updateMfemParallelDecomposition( 1, true );
  RealT dt = 10.0;
  EXPECT_SLIC_ERROR( update( coupling_scheme_id, 0.0, dt ) );

  // Register inverse mass and verify exact analytical nodal penalty CFL timestep (returned unclamped even when dt_in <
  // dt_vote).
  registerMfemInverseMass( coupling_scheme_id, inv_mass );
  setExplicitIntegratorStabilityFactor( coupling_scheme_id, 2.0 );
  updateMfemParallelDecomposition( 1, true );
  dt = 0.5;
  ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
  // Single midpoint quadrature on 2-node edges: w = 1.0, k_pen = 4.0, phi = 0.5 at all 4 nodes, sqrt(inv_m) = 0.5.
  // normal_abs_sum = 4 * (0.5 * 0.5) = 1.0. Each y-DOF row sum = (1.0 * 4.0) * (0.5 * 0.5) * 1.0 = 1.0.
  // dt_cfl = 2.0 / sqrt(1.0) = 2.0.
  EXPECT_NEAR( dt, 2.0, 1.0e-10 );

  // Switch to pair_spring_mass_cfl only: each 2-node edge has total mass = 2 * (1/0.25) = 8.0 -> inv_m = 0.125.
  // stiffness_bound = 4.0 * 1.0 * (0.125 + 0.125) = 1.0 -> dt = 2.0 / sqrt(1.0) = 2.0.
  setTimestepCriteria( coupling_scheme_id, TimestepCriteria{ false, false, false, true } );
  dt = 0.5;
  ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
  EXPECT_NEAR( dt, 2.0, 1.0e-10 );

  // When voting is enabled but no criteria are active, dt must be set to +infinity.
  setTimestepCriteria( coupling_scheme_id, TimestepCriteria{ false, false, false, false } );
  dt = 1.0;
  ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
  EXPECT_TRUE( std::isinf( dt ) );

  // Register velocity and enable current_gap + velocity_projection + nodal_penalty_cfl; velocity projection (dt ~
  // 0.015) is tighter than nodal penalty CFL (dt = 2.0) and must govern.
  registerMfemVelocity( coupling_scheme_id, velocity );
  setTimestepPenFrac( coupling_scheme_id, 0.3 );
  setTimestepCriteria( coupling_scheme_id, TimestepCriteria{ true, true, true, false } );
  updateMfemParallelDecomposition( 1, true );
  dt = 10.0;
  ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
  // Element thickness is ~1.01 on both blocks; relative closing speed is 20.0 -> dt_vel = 0.3 * 1.01 / 20.0 = 0.01515.
  EXPECT_NEAR( dt, 0.3 * 1.01 / 20.0, 1.0e-3 );

  // Verify disabling timestep voting leaves dt unchanged.
  enableTimestepVote( coupling_scheme_id, false );
  dt = 10.0;
  ASSERT_EQ( update( coupling_scheme_id, 0.0, dt ), 0 );
  EXPECT_EQ( dt, 10.0 );

  finalize();
}

INSTANTIATE_TEST_SUITE_P( quadrature, MfemCommonPlaneTest,
                          testing::Values( one_point_polynomial_order, two_point_polynomial_order,
                                           three_point_polynomial_order ) );
}  // namespace

/** @brief Run MFEM integration tests with MPI and SLIC initialized. @param argc Argument count. @param argv Arguments.
 * @return Test status. */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int test_status = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_status;
}
