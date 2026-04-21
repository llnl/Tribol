

// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.

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

namespace {

template <typename F>
void WithHypreHostMemory( F&& f )
{
  HYPRE_MemoryLocation old_loc;
  HYPRE_GetMemoryLocation( &old_loc );
  HYPRE_SetMemoryLocation( HYPRE_MEMORY_HOST );
  f();
  HYPRE_SetMemoryLocation( old_loc );
}

}  // namespace

/**
 * @brief Contact patch test using ENERGY_MORTAR with Lagrange multiplier
 *        enforcement and prescribed displacement applied incrementally.
 *
 * Two unit squares [0,1]x[0,1] and [0,1]x[1,2] with zero initial gap.
 * Linear elasticity with lambda = mu = 5.
 *
 *
 * Analytical solution (plane strain, uniaxial stress with sigma_xx = 0):
 *   eps_yy = applied_disp / total_height
 *   eps_xx = -lambda / (lambda + 2*mu) * eps_yy
 *   u_y(x,y) = eps_yy * y
 *   u_x(x,y) = eps_xx * x
 */
class MfemMortarEnergyLagrangePatchTest : public testing::TestWithParam<std::tuple<int>> {
 protected:
  tribol::RealT max_disp_;
  double l2_err_vec_;
  double l2_err_x_;
  double l2_err_y_;

  // --- User-configurable parameters ---
  static constexpr int num_timesteps_ = 1;
  static constexpr double total_prescribed_disp_ = -0.01;
  static constexpr double lam_ = 50.0;
  static constexpr double mu_ = 50.0;
  static constexpr int max_newton_iter_ = 10;
  static constexpr double newton_rtol_ = 1.0e-10;
  static constexpr double newton_atol_ = 1.0e-12;
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

    int nel_per_dir_top = 10;
    int nel_per_dir_bottom = 10;

    // clang-format off
    mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
      shared::MeshBuilder::SquareMesh(nel_per_dir_top, nel_per_dir_top)
        .updateBdrAttrib(1, 1)   // bottom (Fixed Y)
        .updateBdrAttrib(2, 2)   // right
        .updateBdrAttrib(3, 3)   // top  (NonMortar)
        .updateBdrAttrib(4, 4),  // left (X-fixed)
      shared::MeshBuilder::SquareMesh(nel_per_dir_bottom, nel_per_dir_bottom)
        .translate({0.0, 1.0})
        .updateBdrAttrib(1, 5)   // bottom (Mortar)
        .updateBdrAttrib(2, 2)   // right
        .updateBdrAttrib(3, 6)   // top  (prescribed displacement)
        .updateBdrAttrib(4, 4)   // left  (Fixed x)
    }));
    // clang-format on

    // FE space and grid functions
    auto fe_coll = mfem::H1_FECollection( order, mesh.SpaceDimension() );
    auto par_fe_space = mfem::ParFiniteElementSpace( &mesh, &fe_coll, mesh.SpaceDimension() );
    auto coords = mfem::ParGridFunction( &par_fe_space );
    if ( order > 1 ) {
      mesh.SetNodalGridFunction( &coords, false );
    } else {
      mesh.GetNodes( coords );
    }

    // Grid function for displacement
    mfem::ParGridFunction displacement( &par_fe_space );
    displacement = 0.0;

    mfem::ParGridFunction ref_coords( &par_fe_space );
    mesh.GetNodes( ref_coords );

    // ---- Essential boundary conditions ----

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

    // ---- Elastic stiffness matrix ----

    mfem::ParBilinearForm a( &par_fe_space );
    mfem::ConstantCoefficient lambda_coeff( lam_ );
    mfem::ConstantCoefficient mu_coeff( mu_ );
    a.AddDomainIntegrator( new mfem::ElasticityIntegrator( lambda_coeff, mu_coeff ) );
    a.Assemble();
    a.Finalize();
    auto K_elastic = std::unique_ptr<mfem::HypreParMatrix>( a.ParallelAssemble() );

    // ---- VisIt output ----

    mfem::VisItDataCollection visit_dc( "energy_lagrange_patch_test", &mesh );
    visit_dc.SetPrecision( 8 );
    visit_dc.RegisterField( "displacement", &displacement );

    mfem::ParGridFunction exact_disp( &par_fe_space );
    exact_disp = 0.0;
    visit_dc.RegisterField( "Exact Replacement", &exact_disp );
    visit_dc.SetCycle( 0 );
    visit_dc.SetTime( 0.0 );
    visit_dc.Save();

    // ---- Time-stepping loop ----

    double disp_increment = total_prescribed_disp_ / num_timesteps_;
    tribol::RealT dt = 1.0 / num_timesteps_;
    int cs_id = 0, mesh1_id = 0, mesh2_id = 1;

    const int disp_size = par_fe_space.GetTrueVSize();

    mfem::Vector U( disp_size );  // total displacement true-dof vector
    U = 0.0;

    // Lambda persists across timesteps (warm start)
    // NOTE: sized after first tribol registration when contact FE space is known
    mfem::HypreParVector* lambda = nullptr;
    int contact_size = 0;

    for ( int step = 1; step <= num_timesteps_; ++step ) {
      double current_prescribed_disp = disp_increment * step;

      // Build prescribed displacement vector
      mfem::Vector U_prescribed( disp_size );
      U_prescribed = 0.0;
      for ( int i = 0; i < prescribed_tdof_list.Size(); ++i ) {
        U_prescribed( prescribed_tdof_list[i] ) = current_prescribed_disp;
      }

      // Set initial guess for this step: use previous converged displacement
      // with updated prescribed DOFs
      for ( int i = 0; i < prescribed_tdof_list.Size(); ++i ) {
        U( prescribed_tdof_list[i] ) = current_prescribed_disp;
      }

      // ---- Newton iteration ----
      for ( int newton = 0; newton < max_newton_iter_; ++newton ) {
        // Update coordinates with current displacement
        {
          auto& P = *par_fe_space.GetProlongationMatrix();
          P.Mult( U, displacement );
        }
        coords = ref_coords;
        coords += displacement;

        // Register tribol and update contact data
        coords.ReadWrite();
        tribol::registerMfemCouplingScheme( cs_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs, nonmortar_attrs,
                                            tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::ENERGY_MORTAR,
                                            tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );
        tribol::setLagrangeMultiplierOptions( cs_id, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN );

        tribol::updateMfemParallelDecomposition();
        tribol::update( step, step * dt, dt );

        // ---- Get contact surface FE space and initialize lambda on first pass ----
        auto& contact_fes = tribol::getMfemContactFESpace( cs_id );
        contact_size = contact_fes.GetTrueVSize();

        if ( lambda == nullptr ) {
          lambda = new mfem::HypreParVector( &contact_fes );
          *lambda = 0.0;
        }

        // ---- Evaluate contact residual ----
        mfem::HypreParVector r_contact_force( &par_fe_space );  // G^T * lambda (disp-sized)
        r_contact_force = 0.0;
        mfem::HypreParVector r_gap( &contact_fes );  // g_tilde (contact-sized)
        r_gap = 0.0;

        tribol::evaluateContactResidual( cs_id, *lambda, r_contact_force, r_gap );

        // ---- Evaluate contact Jacobian blocks ----
        std::unique_ptr<mfem::HypreParMatrix> H;  // lambda * d2g/du2 (disp x disp)
        std::unique_ptr<mfem::HypreParMatrix> G;  // dg/du (contact x disp)

        tribol::evaluateContactJacobian( cs_id, *lambda, H, G );

        mfem::Vector R_u( disp_size );
        K_elastic->Mult( U, R_u );  // R_u = K * U
        R_u += r_contact_force;     // R_u += G^T * lambda

        mfem::Vector R_lambda( contact_size );
        R_lambda = r_gap;  // R_lambda = g_tilde

        // Compute residual norms for convergence check
        double norm_R_u = mfem::InnerProduct( MPI_COMM_WORLD, R_u, R_u );
        double norm_R_lambda = mfem::InnerProduct( MPI_COMM_WORLD, R_lambda, R_lambda );
        // Zero out essential DOF contributions before computing norm
        for ( int i = 0; i < ess_tdof_list.Size(); ++i ) {
          norm_R_u -= R_u( ess_tdof_list[i] ) * R_u( ess_tdof_list[i] );
        }
        double residual_norm = std::sqrt( std::abs( norm_R_u ) + norm_R_lambda );

        SLIC_INFO( "  Step " << step << " Newton " << newton << " | residual = " << residual_norm );

        if ( newton > 0 && residual_norm < newton_atol_ ) {
          SLIC_INFO( "  Newton converged (abs tol) at iteration " << newton );
          break;
        }

        // ---- Assemble block Jacobian ----
        // (0,0) block: K + H
        // NOTE: H may be null on the first Newton iteration when lambda = 0
        shared::ParSparseMat J_uu =
            ( H && H->NumRows() > 0 )
                ? ( shared::ParSparseMatView( K_elastic.get() ) + shared::ParSparseMatView( H.get() ) )
                : ( shared::ParSparseMatView( K_elastic.get() ) * 1.0 );

        // G^T for the (0,1) block
        shared::ParSparseMat G_T = shared::ParSparseMatView( G.get() ).transpose();

        // ---- Apply essential BCs ----
        // Zero out essential DOF rows/cols in J_uu
        for ( int i = 0; i < ess_tdof_list.Size(); ++i ) {
          R_u( ess_tdof_list[i] ) = 0.0;
        }
        WithHypreHostMemory( [&]() {
          J_uu.get().HostReadWrite();
          J_uu.get().EliminateRowsCols( ess_tdof_list );
        } );

        // Zero out essential DOF rows in G^T (cols in G)
        // Use EliminateRows on G^T which is simpler than EliminateCols on G
        G_T.eliminateRows( ess_tdof_list );
        shared::ParSparseMat G_mod = G_T.transpose();

        // ---- Set up block system ----

        mfem::Array<int> block_offsets( 3 );
        block_offsets[0] = 0;
        block_offsets[1] = disp_size;
        block_offsets[2] = disp_size + contact_size;

        mfem::BlockOperator J_block( block_offsets );
        J_block.SetBlock( 0, 0, &J_uu.get() );
        J_block.SetBlock( 0, 1, &G_T.get() );
        J_block.SetBlock( 1, 0, &G_mod.get() );

        // Block RHS = -[R_u; R_lambda]
        mfem::BlockVector rhs( block_offsets );
        rhs.GetBlock( 0 ) = R_u;
        rhs.GetBlock( 0 ).Neg();
        rhs.GetBlock( 1 ) = R_lambda;
        rhs.GetBlock( 1 ).Neg();

        // ---- Solve with unpreconditioned MINRES ----
        // (keep it simple for debugging; add preconditioner once this works)

        mfem::BlockVector delta( block_offsets );
        delta = 0.0;

        mfem::MINRESSolver solver( MPI_COMM_WORLD );
        solver.SetRelTol( 1.0e-10 );
        solver.SetAbsTol( 1.0e-14 );
        solver.SetMaxIter( 5000 );
        solver.SetPrintLevel( 3 );
        solver.SetOperator( J_block );
        solver.Mult( rhs, delta );

        SLIC_INFO( "    Solver converged: " << solver.GetConverged() << " in " << solver.GetNumIterations()
                                            << " iterations" );

        // ---- Update solution ----

        mfem::Vector& delta_u = delta.GetBlock( 0 );
        mfem::Vector& delta_lambda = delta.GetBlock( 1 );

        U += delta_u;
        *lambda += delta_lambda;

        // Re-enforce prescribed DOFs exactly (guard against solver drift)
        for ( int i = 0; i < prescribed_tdof_list.Size(); ++i ) {
          U( prescribed_tdof_list[i] ) = current_prescribed_disp;
        }

      }  // end Newton loop

      SLIC_INFO( "Timestep " << step << "/" << num_timesteps_ << " | prescribed disp = " << current_prescribed_disp );

      // Save VisIt output
      {
        auto& P = *par_fe_space.GetProlongationMatrix();
        P.Mult( U, displacement );
      }

    }  // end timestep loop

    // Clean up
    delete lambda;

    // ---- Get final displacement ----
    {
      auto& P = *par_fe_space.GetProlongationMatrix();
      P.Mult( U, displacement );
    }

    auto local_max = displacement.Max();
    max_disp_ = 0.0;
    MPI_Allreduce( &local_max, &max_disp_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    SLIC_INFO( "Max displacement: " << max_disp_ );

    // -----------------------------------------------------------------
    // Analytical solution comparison
    // -----------------------------------------------------------------
    double total_height = 2.0;
    double eps_yy = total_prescribed_disp_ / total_height;
    double eps_xx = -lam_ / ( lam_ + 2.0 * mu_ ) * eps_yy;

    SLIC_INFO( "Analytical: eps_yy = " << eps_yy << ", eps_xx = " << eps_xx );

    mfem::VectorFunctionCoefficient exact_sol_coeff( 2, [eps_xx, eps_yy]( const mfem::Vector& x, mfem::Vector& u ) {
      u[0] = eps_xx * x[0];
      u[1] = eps_yy * x[1];
    } );

    exact_disp.ProjectCoefficient( exact_sol_coeff );

    visit_dc.SetCycle( 1 );
    visit_dc.SetTime( 1.0 );
    visit_dc.Save();

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
      ux_exact( i ) = exact_disp( i );
      ux_num( i ) = displacement( i );
      uy_exact( i ) = exact_disp( n + i );
      uy_num( i ) = displacement( n + i );
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

TEST_P( MfemMortarEnergyLagrangePatchTest, check_patch_test )
{
  EXPECT_GT( max_disp_, 0.0 );
  EXPECT_NEAR( 0.0, l2_err_vec_, 1.0e-2 );
  EXPECT_NEAR( 0.0, l2_err_x_, 1.0e-2 );
  EXPECT_NEAR( 0.0, l2_err_y_, 1.0e-2 );

  MPI_Barrier( MPI_COMM_WORLD );
}

INSTANTIATE_TEST_SUITE_P( tribol, MfemMortarEnergyLagrangePatchTest, testing::Values( std::make_tuple( 2 ) ) );

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
