// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Check generic pair preparation independently of MFEM assembly and of the default derivative implementation.
 * Host tests read current registered and scaled penalties, check validation and fallback behavior, and prepare energy
 * and force definitions with either input provider. Preparation also runs without Enzyme; derivative calls run when
 * Enzyme is enabled. Tests run on one and two ranks, with contact data on rank zero.
 */
#include <limits>
#include <array>
#include <gtest/gtest.h>
#include "axom/slic/core/SimpleLogger.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/PairMethod.hpp"
#include "tribol/physics/PairMethodInputs.hpp"
#include "tribol/physics/CommonPlane.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"
#include "tribol/geom/SmoothedProjectedEdgeMapping.hpp"

namespace {
using namespace tribol;
using BaseRule = EdgeQuadratureRuleGenerator<3>;
using EnergyRule = PairQuadratureRule<SmoothedProjectedEdgeMapping, BaseRule>;
constexpr RealT residual_gap = 0.05;  ///< Fixed offset used to check parameter forwarding.
constexpr RealT tolerance = 1.0e-12;  ///< Roundoff allowance for local integrals.

/** @brief Supply registered constants and current mesh fields without selecting a contact formulation. */
class PairMethodTest : public testing::Test {
 protected:
  /** @brief Register two-element linear coordinate geometry and penalties 2 and 6. */
  void SetUp() override
  {
    mesh_ = std::make_unique<mfem::ParMesh>( static_cast<const mfem::ParMesh&>(
        shared::ParMeshBuilder( MPI_COMM_WORLD, shared::MeshBuilder::SquareMesh( 2, 1 ) ) ) );
    fec_ = std::make_unique<mfem::H1_FECollection>( 1, 2 );
    fes_ = std::make_unique<mfem::ParFiniteElementSpace>( mesh_.get(), fec_.get(), 2 );
    coords_ = std::make_unique<mfem::ParGridFunction>( fes_.get() );
    mesh_->GetNodes( *coords_ );
    registerMfemCouplingScheme( 0, 0, 1, *mesh_, *coords_, { 1 }, { 3 }, SURFACE_TO_SURFACE, NO_CASE, COMMON_PLANE,
                                FRICTIONLESS, PENALTY, BINNING_GRID, ExecutionMode::Sequential );
    setMfemKinematicConstantPenalty( 0, 2.0, 6.0 );
    updateMfemParallelDecomposition( 1, true );
  }
  /** @brief Release borrowed Tribol data before parent coordinates and spaces. */
  void TearDown() override { finalize(); }
  /** @brief Access the fixture's scheme. @return Registered scheme zero. */
  CouplingScheme& scheme() { return CouplingSchemeManager::getInstance().at( 0 ); }
  /**
   * @brief Read parameters from current data, without retaining mesh views between requests.
   * @tparam Provider Registered or scaled-mesh penalty input provider.
   * @param provider Configured reader.
   * @return Current combined penalty stiffness.
   */
  template <typename Provider>
  RealT read( const Provider& provider )
  {
    return provider.prepare( *scheme().getMfemMeshData(), scheme().getMesh1().getView(),
                             scheme().getMesh2().getView() );
  }
  std::unique_ptr<mfem::ParMesh> mesh_;               ///< Parent mesh borrowed by the coupling scheme.
  std::unique_ptr<mfem::H1_FECollection> fec_;        ///< Linear coordinate basis.
  std::unique_ptr<mfem::ParFiniteElementSpace> fes_;  ///< Parent coordinate space.
  std::unique_ptr<mfem::ParGridFunction> coords_;     ///< Current parent coordinates.
};

// Registered inputs follow replaced MFEM constants; scaled inputs independently follow current mesh fields.
TEST_F( PairMethodTest, reads_current_registered_and_scaled_penalties )
{
  const RegisteredPenaltyInputs<> registered{ true };
  const ScaledMeshPenaltyInputs<> scaled{};
  EXPECT_EQ( read( registered ), 4.0 );
  EXPECT_EQ( read( scaled ), 1.5 );
  setPenaltyScale( 0, 2.0 );
  setPenaltyScale( 1, 3.0 );
  EXPECT_EQ( read( registered ), 4.0 );
  // Mesh stiffnesses become 4 and 18 after scaling; their series stiffness is 72/22.
  EXPECT_NEAR( read( scaled ), 72.0 / 22.0, tolerance );
  scheme().getMfemMeshData()->SetMesh1KinematicConstantPenalty( 4.0 );
  scheme().getMfemMeshData()->SetMesh2KinematicConstantPenalty( 12.0 );
  EXPECT_EQ( read( registered ), 8.0 );
  EXPECT_NEAR( read( scaled ), 72.0 / 22.0, tolerance );
}

// Removing registered constants must use the fallback rather than retaining the previous combined stiffness.
TEST_F( PairMethodTest, uses_fallback_and_rejects_unconfigured_penalty )
{
  const RegisteredPenaltyInputs<> registered{ true };
  EXPECT_EQ( read( registered ), 4.0 );
  scheme().getMfemMeshData()->ClearAllPenaltyData();
  constexpr RealT fallback_penalty = 1000.0;
  EXPECT_EQ( read( registered ), fallback_penalty );
  const RegisteredPenaltyInputs<> disabled{ false };
  EXPECT_THROW( read( disabled ), std::invalid_argument );
}

// Scaled mesh inputs must reject nonpositive and nonfinite penalties.
TEST_F( PairMethodTest, rejects_invalid_scaled_penalties )
{
  const ScaledMeshPenaltyInputs<> scaled{};
  constexpr RealT infinity = std::numeric_limits<RealT>::infinity();
  constexpr RealT nan = std::numeric_limits<RealT>::quiet_NaN();
  for ( const RealT penalty : std::array<RealT, 4>{ 0.0, -1.0, infinity, nan } ) {
    scheme().getMesh1().getElementData().m_penalty_stiffness = penalty;
    EXPECT_THROW( read( scaled ), std::invalid_argument );
  }
  scheme().getMesh1().getElementData().m_penalty_stiffness = 2.0;
}

// Either physics definition must accept either input source, with explicit quadrature and a forwarded gap offset.
TEST_F( PairMethodTest, prepares_energy_and_force_independently_of_the_penalty_input_source )
{
  setPenaltyScale( 0, 2.0 );
  setPenaltyScale( 1, 3.0 );
  using Energy = EnergyPairMethod<EnergyMortarEnergy<EnergyRule, OffsetGap<OpposingNormalGap>>,
                                  ScaledMeshPenaltyInputs<AveragePenaltyStiffness>>;
  using Force = ForcePairMethod<CommonPlaneForce<BaseRule, OffsetGap<CommonPlaneGap>>,
                                RegisteredPenaltyInputs<SeriesPenaltyStiffness>>;
  const Energy energy{ EnergyRule{ SmoothedProjectedEdgeMapping{ 0.0 } },
                       EnergyMortarPenalty{ OffsetGap{ OpposingNormalGap{}, residual_gap } },
                       ScaledMeshPenaltyInputs<AveragePenaltyStiffness>{} };
  const Force force{ CommonPlaneQuadratureRule<BaseRule>{},
                     CommonPlaneResidual{ OffsetGap{ CommonPlaneGap{}, residual_gap } },
                     RegisteredPenaltyInputs<SeriesPenaltyStiffness>{ true } };
  const auto definition1 =
      energy.prepare( *scheme().getMfemMeshData(), scheme().getMesh1().getView(), scheme().getMesh2().getView() );
  const auto definition2 =
      force.prepare( *scheme().getMfemMeshData(), scheme().getMesh1().getView(), scheme().getMesh2().getView() );
  // Two parallel opposing edges have length 2 and penetration 0.1; the residual offset makes the gap -0.15.
  const MdArray<RealT, Energy::Definition::dof_count> coordinates{ 0.0, 0.0, 2.0, 0.0, 2.0, 0.1, 0.0, 0.1 };
  const auto energy_value = definition1.evaluate( coordinates.view() );
  const auto force_value = definition2.evaluate( coordinates.view() );
  ASSERT_EQ( energy_value.quadrature_result.status, QuadratureStatus::Success );
  ASSERT_EQ( force_value.quadrature_result.status, QuadratureStatus::Success );
  EXPECT_TRUE( energy_value.has_active_points );
  EXPECT_TRUE( force_value.has_active_points );
  // The energy uses scaled arithmetic stiffness 11; the force uses unscaled series stiffness 1.5.
  EXPECT_NEAR( energy_value.energy, 0.2475, tolerance );
  EXPECT_NEAR( force_value.residual[1] + force_value.residual[3], -0.45, tolerance );
  EXPECT_NEAR( force_value.residual[5] + force_value.residual[7], 0.45, tolerance );
  EXPECT_TRUE( Energy::has_energy );
  EXPECT_FALSE( Force::has_energy );
#ifdef TRIBOL_USE_ENZYME
  const auto energy_derivatives = Energy::differentiate( coordinates.view(), definition1, energy_value );
  const auto force_derivatives = Force::differentiate( coordinates.view(), definition2, force_value );
  EXPECT_EQ( energy_derivatives.energy, energy_value.energy );
  for ( IndexT row = 0; row < force_value.residual.size(); ++row ) {
    EXPECT_EQ( force_derivatives.residual[row], force_value.residual[row] );
  }
#endif
}

// Shared preparation must reject a mesh with more coordinate nodes than its definition can consume.
TEST_F( PairMethodTest, rejects_incompatible_coordinate_layouts )
{
  const ForcePairMethod<CommonPlaneForce<BaseRule>, ScaledMeshPenaltyInputs<>> method{
      CommonPlaneQuadratureRule<BaseRule>{}, CommonPlaneResidual{}, ScaledMeshPenaltyInputs<>{} };
  const IndexT connectivity[]{ 0, 1, 2 };
  const RealT x[]{ 0.0, 0.5, 1.0 }, y[]{ 0.0, 0.0, 0.0 };
  MeshData quadratic( 2, 1, 3, connectivity, EDGE, QUADRATIC_LAGRANGE, x, y, nullptr, MemorySpace::Host );
  EXPECT_THROW( method.prepare( *scheme().getMfemMeshData(), quadratic.getView(), scheme().getMesh2().getView() ),
                std::invalid_argument );
}
}  // namespace

/**
 * @brief Run host pair preparation tests with MPI and logging initialized.
 * @param argc Argument count.
 * @param argv Arguments.
 * @return Test status.
 */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  testing::InitGoogleTest( &argc, argv );
  axom::slic::SimpleLogger logger;
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
