// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <cmath>
#include <set>

#include <gtest/gtest.h>

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

#include "mfem.hpp"

#include "axom/CLI11.hpp"
#include "axom/slic.hpp"

#include "shared/math/ParSparseMat.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "redecomp/redecomp.hpp"

#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

/**
 * @brief Contact patch test using ENERGY_MORTAR with zero initial gap
 *        and prescribed displacement applied incrementally over timesteps.
 *
 * Two unit squares [0,1]x[0,1] and [0,1]x[1,2] with zero gap.
 * Linear elasticity with lambda = mu = 50.
 *
 * Analytical solution (plane strain, uniaxial stress with sigma_xx = 0):
 *   eps_yy = applied_disp / total_height
 *   eps_xx = -lambda / (lambda + 2*mu) * eps_yy
 *   u_y(x,y) = eps_yy * y
 *   u_x(x,y) = eps_xx * x
 */
class MfemMortarEnergyPatchTest : public testing::TestWithParam<std::tuple<int>> {
 protected:
  tribol::RealT max_disp_;
  double l2_err_vec_;
  double l2_err_x_;
  double l2_err_y_;

  // --- User-configurable parameters ---
  static constexpr int num_timesteps_ = 10;
  static constexpr double total_prescribed_disp_ = -0.01;
  static constexpr double lam_ = 50.0;
  static constexpr double mu_ = 50.0;
  // ------------------------------------

  void SetUp() override
  {
    int ref_levels = std::get<0>( GetParam() );
    int order = 1;

    auto mortar_attrs = std::set<int>( { 5 } );
    auto nonmortar_attrs = std::set<int>( { 3 } );
    auto xfixed_attrs = std::set<int>( { 4 } );
    auto yfixed_bottom_attrs = std::set<int>( { 1 } );
    auto prescribed_attrs = std::set<int>( { 6 } );

    int nel_per_dir = std::pow( 2, ref_levels );

    // clang-format off
    mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
      shared::MeshBuilder::SquareMesh(nel_per_dir, nel_per_dir)
        .updateBdrAttrib(1, 1)   // bottom (Fixed Y)
        .updateBdrAttrib(2, 2)   // right 
        .updateBdrAttrib(3, 3)   // top  (NonMortar)
        .updateBdrAttrib(4, 4),  // left (X-fixed)
      shared::MeshBuilder::SquareMesh(nel_per_dir, nel_per_dir)
        .translate({0.0, 1.0})
        .updateBdrAttrib(1, 5)   // bottom (Mortar)
        .updateBdrAttrib(2, 2)   // right 
        .updateBdrAttrib(3, 6)   // top  (prescribed displacement)
        .updateBdrAttrib(4, 4)   // left  (Fixed x)
    }));
    // clang-format on

    // FE space and grid functions
    auto fe_coll = mfem::H1_FECollection( order, mesh.SpaceDimension() );
    auto par_fe_space = mfem::ParFiniteElementSpace( &mesh, &fe_coll, mesh.SpaceDimension(), mfem::Ordering::byVDIM );
    auto coords = mfem::ParGridFunction( &par_fe_space );
    if ( order > 1 ) {
      mesh.SetNodalGridFunction( &coords, false );
    } else {
      mesh.GetNodes( coords );
    }

    // Grid fucntion for displacement
    mfem::ParGridFunction displacement( &par_fe_space );
    displacement = 0.0;

    mfem::ParGridFunction ref_coords( &par_fe_space );
    mesh.GetNodes( ref_coords );

    // recover dirchlet bd tdof list
    mfem::Array<int> ess_vdof_marker( par_fe_space.GetVSize() );
    ess_vdof_marker = 0;

    // x-fixed on left
    {
      mfem::Array<int> tmp;
      mfem::Array<int> bdr( mesh.bdr_attributes.Max() );
      bdr = 0;
      for ( auto a : xfixed_attrs ) bdr[a - 1] = 1;
      par_fe_space.GetEssentialVDofs( bdr, tmp, 0 );
      for ( int i = 0; i < tmp.Size(); ++i ) ess_vdof_marker[i] = ess_vdof_marker[i] || tmp[i];
    }

    // y-fixed on bottom
    {
      mfem::Array<int> tmp;
      mfem::Array<int> bdr( mesh.bdr_attributes.Max() );
      bdr = 0;
      for ( auto a : yfixed_bottom_attrs ) bdr[a - 1] = 1;
      par_fe_space.GetEssentialVDofs( bdr, tmp, 1 );
      for ( int i = 0; i < tmp.Size(); ++i ) ess_vdof_marker[i] = ess_vdof_marker[i] || tmp[i];
    }

    // y-prescribed on top
    mfem::Array<int> prescribed_vdof_marker( par_fe_space.GetVSize() );
    prescribed_vdof_marker = 0;
    {
      mfem::Array<int> tmp;
      mfem::Array<int> bdr( mesh.bdr_attributes.Max() );
      bdr = 0;
      for ( auto a : prescribed_attrs ) bdr[a - 1] = 1;
      par_fe_space.GetEssentialVDofs( bdr, tmp, 1 );
      prescribed_vdof_marker = tmp;
      for ( int i = 0; i < tmp.Size(); ++i ) ess_vdof_marker[i] = ess_vdof_marker[i] || tmp[i];
    }

    mfem::Array<int> ess_tdof_list;
    {
      mfem::Array<int> ess_tdof_marker;
      par_fe_space.GetRestrictionMatrix()->BooleanMult( ess_vdof_marker, ess_tdof_marker );
      mfem::FiniteElementSpace::MarkerToList( ess_tdof_marker, ess_tdof_list );
    }

    mfem::Array<int> prescribed_tdof_list;
    {
      mfem::Array<int> marker;
      par_fe_space.GetRestrictionMatrix()->BooleanMult( prescribed_vdof_marker, marker );
      mfem::FiniteElementSpace::MarkerToList( marker, prescribed_tdof_list );
    }

    // set up mfem elasticity bilinear form
    mfem::ParBilinearForm a( &par_fe_space );
    mfem::ConstantCoefficient lambda_coeff( lam_ );
    mfem::ConstantCoefficient mu_coeff( mu_ );
    a.AddDomainIntegrator( new mfem::ElasticityIntegrator( lambda_coeff, mu_coeff ) );
    a.Assemble();
    a.Finalize();
    shared::ParSparseMat A_elastic( a.ParallelAssemble() );

    // Visit Output
    mfem::VisItDataCollection visit_dc( "energy_patch_test", &mesh );
    visit_dc.SetPrecision( 8 );
    visit_dc.RegisterField( "displacement", &displacement );
    visit_dc.SetCycle( 0 );
    visit_dc.SetTime( 0.0 );
    visit_dc.Save();

    // timestepping loop for displacement
    double disp_increment = total_prescribed_disp_ / num_timesteps_;
    tribol::RealT dt = 1.0 / num_timesteps_;
    int cs_id = 0, mesh1_id = 0, mesh2_id = 1;

    tribol::registerMfemCouplingScheme( cs_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs, nonmortar_attrs,
                                        tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::ENERGY_MORTAR,
                                        tribol::FRICTIONLESS, tribol::PENALTY, tribol::BINNING_GRID );
    tribol::setEnergyMortarPenaltyMode( cs_id, tribol::EnergyMortarPenaltyMode::QUADRATURE_POINT_GAP );
    tribol::setMfemKinematicConstantPenalty( cs_id, 100.0, 100.0 );

    mfem::Vector X( par_fe_space.GetTrueVSize() );
    X = 0.0;

    for ( int step = 1; step <= num_timesteps_; ++step ) {
      double current_prescribed_disp = disp_increment * step;

      // Prescribed displacement vector
      mfem::Vector X_prescribed( par_fe_space.GetTrueVSize() );
      X_prescribed = 0.0;
      for ( int i = 0; i < prescribed_tdof_list.Size(); ++i ) {
        X_prescribed( prescribed_tdof_list[i] ) = current_prescribed_disp;
      }

      // Update coordinates for contact detection
      {
        mfem::Vector X_temp( X );
        for ( int i = 0; i < prescribed_tdof_list.Size(); ++i ) {
          X_temp( prescribed_tdof_list[i] ) = current_prescribed_disp;
        }
        auto& P = *par_fe_space.GetProlongationMatrix();
        P.Mult( X_temp, displacement );
      }
      coords = ref_coords;
      coords += displacement;

      tribol::updateMfemParallelDecomposition();
      tribol::update( step, step * dt, dt );

      auto A_cont_ptr = tribol::getMfemDfDx( cs_id );
      ASSERT_TRUE( A_cont_ptr != nullptr );
      shared::ParSparseMat A_cont( std::move( A_cont_ptr ) );

      auto f_contact = tribol::getMfemContactForce( cs_id );
      f_contact.Neg();

      // Inhomogeneous Dirichlet: rhs = f_contact - K * u_prescribed
      // Form A_total on host; mfem::Add may dispatch through Hypre device paths in HIP builds.
      shared::ParSparseMat A_total =
          shared::ParSparseMatView( &A_elastic.get() ) + shared::ParSparseMatView( &A_cont.get() );

      mfem::Vector rhs( par_fe_space.GetTrueVSize() );
      A_total.get().Mult( X_prescribed, rhs );
      rhs.Neg();
      rhs += f_contact;

      for ( int i = 0; i < ess_tdof_list.Size(); ++i ) {
        rhs( ess_tdof_list[i] ) = 0.0;
      }

      A_total.eliminateRowsCols( ess_tdof_list );

      mfem::Vector X_free( par_fe_space.GetTrueVSize() );
      X_free = 0.0;

      // Preconditioner: BoomerAMG on CPU builds; use MFEM's built-in diagonal/Jacobi smoother on GPU builds where
      // BoomerAMG may assert/crash with GPU-enabled Hypre.
#if defined( MFEM_USE_CUDA ) || defined( MFEM_USE_HIP ) || defined( MFEM_USE_RAJA )
      // DSmoother operates on SparseMatrix and fails when MINRES passes a HypreParMatrix
      // into prec.SetOperator(). Use HypreSmoother Jacobi, which accepts HypreParMatrix.
      mfem::HypreSmoother prec;
      prec.SetType( mfem::HypreSmoother::Jacobi );
#else
      mfem::HypreBoomerAMG prec( A_total.get() );
      prec.SetElasticityOptions( &par_fe_space );
      prec.SetPrintLevel( 0 );
#endif

      mfem::MINRESSolver solver( MPI_COMM_WORLD );
      solver.SetRelTol( 1.0e-8 );
      solver.SetAbsTol( 1.0e-12 );
      solver.SetMaxIter( 5000 );
      solver.SetPrintLevel( step == num_timesteps_ ? 3 : 1 );
      solver.SetPreconditioner( prec );
      solver.SetOperator( A_total.get() );
      solver.Mult( rhs, X_free );

      X = X_free;
      X += X_prescribed;

      SLIC_INFO( "Timestep " << step << "/" << num_timesteps_ << " | prescribed disp = " << current_prescribed_disp );

      // Save VisIt output
      {
        auto& P = *par_fe_space.GetProlongationMatrix();
        P.Mult( X, displacement );
      }
      visit_dc.SetCycle( step );
      visit_dc.SetTime( step * dt );
      visit_dc.Save();
    }

    // Get final disaplacent
    {
      auto& P = *par_fe_space.GetProlongationMatrix();
      P.Mult( X, displacement );
    }

    auto local_max = displacement.Max();
    max_disp_ = 0.0;
    MPI_Allreduce( &local_max, &max_disp_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

    auto local_min = displacement.Min();
    double min_disp = 0.0;
    MPI_Allreduce( &local_min, &min_disp, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );

    SLIC_INFO( "Max displacement: " << max_disp_ );
    SLIC_INFO( "Min displacement: " << min_disp );

    // -----------------------------------------------------------------
    // Analytical solution comparison
    //
    // Plane strain, uniaxial stress (sigma_xx = 0, free right side):
    //   eps_yy = applied_disp / total_height = -0.01 / 2.0 = -0.005
    //   eps_xx = -lambda/(lambda + 2*mu) * eps_yy
    //   u_y = eps_yy * y
    //   u_x = eps_xx * x
    // -----------------------------------------------------------------
    double total_height = 2.0;
    double eps_yy = total_prescribed_disp_ / total_height;
    double eps_xx = -lam_ / ( lam_ + 2.0 * mu_ ) * eps_yy;

    SLIC_INFO( "Analytical: eps_yy = " << eps_yy << ", eps_xx = " << eps_xx );

    mfem::VectorFunctionCoefficient exact_sol_coeff( 2, [eps_xx, eps_yy]( const mfem::Vector& x, mfem::Vector& u ) {
      u[0] = eps_xx * x[0];
      u[1] = eps_yy * x[1];
    } );

    mfem::ParGridFunction exact_disp( &par_fe_space );
    exact_disp.ProjectCoefficient( exact_sol_coeff );

    // Vector error
    mfem::ParGridFunction error_vec( exact_disp );
    error_vec -= displacement;
    l2_err_vec_ = mfem::ParNormlp( error_vec, 2, MPI_COMM_WORLD );

    // Component-wise errors
    const mfem::FiniteElementCollection* fec = par_fe_space.FEColl();
    mfem::ParFiniteElementSpace scalar_fes( &mesh, fec, 1, par_fe_space.GetOrdering() );
    const int n = scalar_fes.GetNDofs();

    mfem::ParGridFunction ux_exact( &scalar_fes ), ux_num( &scalar_fes );
    mfem::ParGridFunction uy_exact( &scalar_fes ), uy_num( &scalar_fes );

    for ( int i = 0; i < n; ++i ) {
      const int ux_vdof = par_fe_space.DofToVDof( i, 0 );
      const int uy_vdof = par_fe_space.DofToVDof( i, 1 );
      ux_exact( i ) = exact_disp( ux_vdof );
      ux_num( i ) = displacement( ux_vdof );
      uy_exact( i ) = exact_disp( uy_vdof );
      uy_num( i ) = displacement( uy_vdof );
    }

    mfem::ParGridFunction ux_err( ux_exact );
    ux_err -= ux_num;
    l2_err_x_ = mfem::ParNormlp( ux_err, 2, MPI_COMM_WORLD );

    mfem::ParGridFunction uy_err( uy_exact );
    uy_err -= uy_num;
    l2_err_y_ = mfem::ParNormlp( uy_err, 2, MPI_COMM_WORLD );

    SLIC_INFO( "L2 error (vector): " << l2_err_vec_ );
    SLIC_INFO( "L2 error (x):      " << l2_err_x_ );
    SLIC_INFO( "L2 error (y):      " << l2_err_y_ );
    SLIC_INFO( "Consistency check |err_vec^2 - (err_x^2 + err_y^2)| = "
               << std::abs( l2_err_vec_ * l2_err_vec_ - ( l2_err_x_ * l2_err_x_ + l2_err_y_ * l2_err_y_ ) ) );
  }
};

TEST_P( MfemMortarEnergyPatchTest, check_patch_test )
{
  EXPECT_GT( max_disp_, 0.0 );
  EXPECT_NEAR( 0.0, l2_err_vec_, 1.0e-2 );
  EXPECT_NEAR( 0.0, l2_err_x_, 1.0e-2 );
  EXPECT_NEAR( 0.0, l2_err_y_, 1.0e-2 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( tribol, MfemMortarEnergyPatchTest, testing::Values( std::make_tuple( 2 ) ) );

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );
  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();
#endif

  axom::slic::SimpleLogger logger;
  result = RUN_ALL_TESTS();

  tribol::finalize();
  MPI_Finalize();

  return result;
}
