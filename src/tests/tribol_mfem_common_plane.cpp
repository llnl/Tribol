// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <set>
#include <string>
#include <utility>

#include <gtest/gtest.h>

// Tribol includes
#include "tribol/config.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/utils/TestUtils.hpp"

// Shared includes
#include "shared/mesh/MeshBuilder.hpp"

// Redecomp includes
#include "redecomp/redecomp.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// MFEM includes
#include "mfem.hpp"

// Axom includes
#include "axom/CLI11.hpp"
#include "axom/slic.hpp"

/**
 * @brief Test legacy and ContactFormulation-based Common Plane in a small explicit contact calculation.
 *
 * The legacy cases retain the original 3D block problem and also run the same 2D constant-penalty cases as the
 * formulation. CommonPlaneFormulation supports the 2D cases but not the 3D or element-penalty cases.
 */
enum class CommonPlaneImplementation
{
  Legacy,
  ContactFormulation
};

enum class CommonPlaneProblem
{
  Blocks2D,
  Blocks3D
};

struct MfemCommonPlaneParameters {
  int order;
  tribol::KinematicPenaltyCalculation penalty_calculation;
  CommonPlaneImplementation implementation;
  CommonPlaneProblem problem;
};

std::string commonPlaneParameterName( const testing::TestParamInfo<MfemCommonPlaneParameters>& parameter_info )
{
  const auto& parameters = parameter_info.param;
  const std::string penalty_name =
      parameters.penalty_calculation == tribol::KINEMATIC_CONSTANT ? "Constant" : "Element";
  const std::string implementation_name =
      parameters.implementation == CommonPlaneImplementation::Legacy ? "Legacy" : "Formulation";
  const std::string problem_name = parameters.problem == CommonPlaneProblem::Blocks2D ? "2D" : "3D";
  return "Order" + std::to_string( parameters.order ) + penalty_name + implementation_name + problem_name;
}

class MfemCommonPlaneTest : public testing::TestWithParam<MfemCommonPlaneParameters> {
 protected:
  tribol::RealT max_displacement_ = 0.0;
  void SetUp() override
  {
    const auto& parameters = GetParam();
    const bool uses_contact_formulation = parameters.implementation == CommonPlaneImplementation::ContactFormulation;
    const bool uses_two_dimensional_problem = parameters.problem == CommonPlaneProblem::Blocks2D;
    // number of times to uniformly refine the serial mesh before constructing the
    // parallel mesh
    int ref_levels = 2;
    // polynomial order of the finite element discretization
    int order = parameters.order;
    // initial separation between the two blocks
    tribol::RealT initial_sep = -0.001;
    // initial velocity
    tribol::RealT initial_v = 0.0;
    // timestep size
    tribol::RealT dt = 0.001;
    // end time
    tribol::RealT t_end = 0.01;
    // material density
    tribol::RealT rho = 1000.0;
    // lame parameter
    tribol::RealT lambda = 100000.0;
    // lame parameter (shear modulus)
    tribol::RealT mu = 100000.0;
    // kinematic constant penalty stiffness equivalent to the element-wise calculation,
    // which is bulk-modulus over element thickness.
    tribol::RealT p_kine = ( lambda + 2.0 / 3.0 * mu ) / ( 1.0 / std::pow( 2.0, ref_levels ) );

    auto contact_surf_1 = uses_two_dimensional_problem ? std::set<int>{ 3 } : std::set<int>{ 6 };
    auto contact_surf_2 = uses_two_dimensional_problem ? std::set<int>{ 5 } : std::set<int>{ 7 };
    // Fix the lower body's bottom boundary for both the 2D and 3D problems.
    auto fixed_attrs = std::set<int>( { 1 } );
    // element attribute corresponding to volume elements where an initial
    // velocity will be applied
    auto moving_attrs = std::set<int>( { 2 } );

#if defined( TRIBOL_USE_CUDA )
    tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Cuda;
#elif defined( TRIBOL_USE_HIP )
    tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Hip;
#else
    tribol::ExecutionMode exec_mode = tribol::ExecutionMode::Sequential;
#endif

    auto mesh_builder = [&]() {
      if ( uses_two_dimensional_problem ) {
        // The lower square occupies [0, 1]^2. The upper square starts at y=0.999, so its bottom edge penetrates the
        // lower square's top edge by 0.001. Attributes 5 and 6 distinguish the upper contact and exterior edges after
        // unifying.
        return shared::MeshBuilder::Unify(
                   { shared::MeshBuilder::SquareMesh( 1, 1 ), shared::MeshBuilder::SquareMesh( 1, 1 )
                                                                  .translate( { 0.0, 1.0 + initial_sep } )
                                                                  .updateAttrib( 1, 2 )
                                                                  .updateBdrAttrib( 1, 5 )
                                                                  .updateBdrAttrib( 3, 6 ) } )
            .refine( ref_levels );
      }
      return shared::MeshBuilder::Unify(
                 { shared::MeshBuilder::CubeMesh( 1, 1, 1 ), shared::MeshBuilder::CubeMesh( 1, 1, 1 )
                                                                 .translate( { 0.0, 0.0, 1.0 + initial_sep } )
                                                                 .updateAttrib( 1, 2 )
                                                                 .updateBdrAttrib( 1, 7 )
                                                                 .updateBdrAttrib( 6, 8 ) } )
          .refine( ref_levels );
    }();
    mfem::ParMesh mesh = shared::ParMeshBuilder( MPI_COMM_WORLD, std::move( mesh_builder ) );

    // grid function for higher-order nodes
    auto fe_coll = mfem::H1_FECollection( order, mesh.SpaceDimension() );
    auto par_fe_space = mfem::ParFiniteElementSpace( &mesh, &fe_coll, mesh.SpaceDimension() );
    auto coords = mfem::ParGridFunction( &par_fe_space );
    if ( order > 1 ) {
      mesh.SetNodalGridFunction( &coords, false );
    } else {
      mesh.GetNodes( coords );
    }

    mfem::ParGridFunction ref_coords{ coords };

    // grid function for displacement
    mfem::ParGridFunction displacement{ &par_fe_space };
    displacement = 0.0;

    // grid function for velocity
    mfem::ParGridFunction velocity{ &par_fe_space };
    velocity = 0.0;

    // set initial velocity
    mfem::Vector init_velocity_vector( mesh.SpaceDimension() );
    init_velocity_vector = 0.0;
    init_velocity_vector[mesh.SpaceDimension() - 1] = -std::abs( initial_v );
    mfem::VectorConstantCoefficient init_velocity_coeff( init_velocity_vector );
    mfem::Array<int> moving_attrs_array;
    mfem::Array<mfem::VectorCoefficient*> init_velocity_coeff_array;
    moving_attrs_array.Reserve( moving_attrs.size() );
    init_velocity_coeff_array.Reserve( moving_attrs.size() );
    for ( auto moving_attr : moving_attrs ) {
      moving_attrs_array.Append( moving_attr );
      init_velocity_coeff_array.Append( &init_velocity_coeff );
    }
    mfem::PWVectorCoefficient initial_v_coeff( mesh.SpaceDimension(), moving_attrs_array, init_velocity_coeff_array );
    velocity.ProjectCoefficient( initial_v_coeff );

    // recover dirichlet bc tdof list
    mfem::Array<int> ess_vdof_list;
    {
      mfem::Array<int> ess_vdof_marker;
      mfem::Array<int> ess_bdr( mesh.bdr_attributes.Max() );
      ess_bdr = 0;
      for ( auto fixed_attr : fixed_attrs ) {
        ess_bdr[fixed_attr - 1] = 1;
      }
      par_fe_space.GetEssentialVDofs( ess_bdr, ess_vdof_marker );
      mfem::FiniteElementSpace::MarkerToList( ess_vdof_marker, ess_vdof_list );
    }

    // set up mfem elasticity bilinear form
    mfem::ConstantCoefficient rho_coeff{ rho };
    mfem::ConstantCoefficient lambda_coeff{ lambda };
    mfem::ConstantCoefficient mu_coeff{ mu };
    mfem_ext::ExplicitMechanics op{ par_fe_space, rho_coeff, lambda_coeff, mu_coeff };

    // set up time integrator
    mfem_ext::CentralDiffSolver solver{ ess_vdof_list };
    solver.Init( op );

    // set up tribol
    int coupling_scheme_id = 0;
    int mesh1_id = 0;
    int mesh2_id = 1;
    tribol::registerMfemCouplingScheme( coupling_scheme_id, mesh1_id, mesh2_id, mesh, coords, contact_surf_1,
                                        contact_surf_2, tribol::SURFACE_TO_SURFACE, tribol::NO_CASE,
                                        tribol::COMMON_PLANE, tribol::FRICTIONLESS, tribol::PENALTY,
                                        tribol::BINNING_BVH, exec_mode );
    tribol::registerMfemVelocity( coupling_scheme_id, velocity );
    if ( parameters.penalty_calculation == tribol::KINEMATIC_CONSTANT ) {
      tribol::setMfemKinematicConstantPenalty( coupling_scheme_id, p_kine, p_kine );
    } else {
      mfem::Vector bulk_moduli_by_bdry_attrib( mesh.bdr_attributes.Max() );
      bulk_moduli_by_bdry_attrib = lambda + 2.0 / 3.0 * mu;
      mfem::PWConstCoefficient mat_coeff( bulk_moduli_by_bdry_attrib );
      tribol::setMfemKinematicElementPenalty( coupling_scheme_id, mat_coeff );
    }
    if ( uses_contact_formulation ) {
      constexpr bool enable_formulation = true;
      constexpr bool compute_jacobian = false;
      tribol::setMfemUseCommonPlaneFormulation( coupling_scheme_id, enable_formulation, compute_jacobian );
      // One point matches the legacy constant-traction integration for this flat overlap.
      constexpr int one_point_quadrature_order = 1;
      tribol::setQuadratureOrder( coupling_scheme_id, one_point_quadrature_order );
    }

    int cycle{ 0 };
    for ( tribol::RealT t{ 0.0 }; t < t_end; t += dt ) {
      // build new parallel decomposed redecomp mesh and update grid functions
      // on each mesh
      tribol::updateMfemParallelDecomposition();
      tribol::update( cycle, t, dt );
      op.f_ext = 0.0;
      if ( uses_contact_formulation ) {
        auto contact_residual = tribol::getMfemContactResidual( coupling_scheme_id );
        par_fe_space.GetProlongationMatrix()->Mult( contact_residual, op.f_ext );
        // The formulation returns a residual contribution, while ExplicitMechanics expects the physical contact force.
        op.f_ext *= -1.0;
      } else {
        tribol::getMfemResponse( coupling_scheme_id, op.f_ext );
      }

      op.SetTime( t );
      solver.Step( displacement, velocity, t, dt );

      coords.Set( 1.0, ref_coords );
      coords += displacement;
      if ( order == 1 ) {
        coords.HostRead();
        mesh.SetVertices( coords );
      }

      ++cycle;
    }

    max_displacement_ = displacement.Max();
    MPI_Allreduce( MPI_IN_PLACE, &max_displacement_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

    tribol::finalize();
  }
};

TEST_P( MfemCommonPlaneTest, common_plane )
{
  // make sure there is some contact response after 10 cycles
  EXPECT_GT( max_displacement_, 1.0e-4 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P(
    tribol, MfemCommonPlaneTest,
    testing::Values( MfemCommonPlaneParameters{ 1, tribol::KINEMATIC_CONSTANT, CommonPlaneImplementation::Legacy,
                                                CommonPlaneProblem::Blocks3D },
                     MfemCommonPlaneParameters{ 1, tribol::KINEMATIC_ELEMENT, CommonPlaneImplementation::Legacy,
                                                CommonPlaneProblem::Blocks3D },
                     MfemCommonPlaneParameters{ 2, tribol::KINEMATIC_CONSTANT, CommonPlaneImplementation::Legacy,
                                                CommonPlaneProblem::Blocks3D },
                     MfemCommonPlaneParameters{ 2, tribol::KINEMATIC_ELEMENT, CommonPlaneImplementation::Legacy,
                                                CommonPlaneProblem::Blocks3D },
                     MfemCommonPlaneParameters{ 1, tribol::KINEMATIC_CONSTANT, CommonPlaneImplementation::Legacy,
                                                CommonPlaneProblem::Blocks2D },
                     MfemCommonPlaneParameters{ 1, tribol::KINEMATIC_CONSTANT,
                                                CommonPlaneImplementation::ContactFormulation,
                                                CommonPlaneProblem::Blocks2D },
                     MfemCommonPlaneParameters{ 2, tribol::KINEMATIC_CONSTANT, CommonPlaneImplementation::Legacy,
                                                CommonPlaneProblem::Blocks2D },
                     MfemCommonPlaneParameters{ 2, tribol::KINEMATIC_CONSTANT,
                                                CommonPlaneImplementation::ContactFormulation,
                                                CommonPlaneProblem::Blocks2D } ),
    commonPlaneParameterName );

//------------------------------------------------------------------------------
int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

#if defined( TRIBOL_USE_CUDA )
  std::string device_str( "cuda" );
#elif defined( TRIBOL_USE_HIP )
  std::string device_str( "hip" );
#elif defined( TRIBOL_USE_OPENMP ) && defined( MFEM_USE_OPENMP )
  std::string device_str( "omp" );
#else
  std::string device_str( "cpu" );
#endif

  mfem::Device device( device_str );
  device.Print();

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
