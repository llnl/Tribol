// // Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// // other Tribol Project Developers. See the top-level LICENSE file for details.
// //
// // SPDX-License-Identifier: (MIT)

// #include <cmath>
// #include <set>

// #include <gtest/gtest.h>

// #ifdef TRIBOL_USE_UMPIRE
// #include "umpire/ResourceManager.hpp"
// #endif

// #include "mfem.hpp"

// #include "axom/CLI11.hpp"
// #include "axom/slic.hpp"

// #include "shared/mesh/MeshBuilder.hpp"
// #include "redecomp/redecomp.hpp"

// #include "tribol/config.hpp"
// #include "tribol/common/Parameters.hpp"
// #include "tribol/interface/tribol.hpp"
// #include "tribol/interface/mfem_tribol.hpp"

// /**
//  * @brief Contact patch test using ENERGY_MORTAR with zero initial gap
//  *        and prescribed displacement applied incrementally over timesteps.
//  *
//  * Two unit squares [0,1]x[0,1] and [0,1]x[1,2] with zero gap.
//  * Linear elasticity with lambda = mu = 50.
//  *
//  * Analytical solution (plane strain, uniaxial stress with sigma_xx = 0):
//  *   eps_yy = applied_disp / total_height
//  *   eps_xx = -lambda / (lambda + 2*mu) * eps_yy
//  *   u_y(x,y) = eps_yy * y
//  *   u_x(x,y) = eps_xx * x
//  */
// class MfemMortarEnergyPatchTest : public testing::TestWithParam<std::tuple<int>> {
//  protected:
//   tribol::RealT max_disp_;
//   double l2_err_vec_;
//   double l2_err_x_;
//   double l2_err_y_;

//   // --- User-configurable parameters ---
//   static constexpr int    num_timesteps_ = 10;
//   static constexpr double total_prescribed_disp_ = -0.01;
//   static constexpr double lam_ = 5.0;
//   static constexpr double mu_  = 5.0;
//   // ------------------------------------

//   void SetUp() override
//   {
//     int ref_levels = std::get<0>( GetParam() );
//     int order = 1;

//     auto mortar_attrs     = std::set<int>( { 5 } );
//     auto nonmortar_attrs  = std::set<int>( { 3 } );
//     auto xfixed_attrs     = std::set<int>( { 4 } );
//     auto yfixed_bottom_attrs = std::set<int>( { 1 } );
//     auto prescribed_attrs = std::set<int>( { 6 } );

//     int nel_per_dir = std::pow( 2, ref_levels );

//     // clang-format off
//     mfem::ParMesh mesh = shared::ParMeshBuilder(MPI_COMM_WORLD, shared::MeshBuilder::Unify({
//       shared::MeshBuilder::SquareMesh(nel_per_dir, nel_per_dir)
//         .updateBdrAttrib(1, 1)   // bottom (Fixed Y)
//         .updateBdrAttrib(2, 2)   // right 
//         .updateBdrAttrib(3, 3)   // top  (NonMortar)
//         .updateBdrAttrib(4, 4),  // left (X-fixed)
//       shared::MeshBuilder::SquareMesh(nel_per_dir, nel_per_dir)
//         .translate({0.0, 1.0})
//         .updateBdrAttrib(1, 5)   // bottom (Mortar)
//         .updateBdrAttrib(2, 2)   // right 
//         .updateBdrAttrib(3, 6)   // top  (prescribed displacement)
//         .updateBdrAttrib(4, 4)   // left  (Fixed x)
//     }));
//     // clang-format on

//     // FE space and grid functions
//     auto fe_coll = mfem::H1_FECollection( order, mesh.SpaceDimension() );
//     auto par_fe_space = mfem::ParFiniteElementSpace( &mesh, &fe_coll, mesh.SpaceDimension() );
//     auto coords = mfem::ParGridFunction( &par_fe_space );
//     if ( order > 1 ) {
//       mesh.SetNodalGridFunction( &coords, false );
//     } else {
//       mesh.GetNodes( coords );
//     }


//     //Grid fucntion for displacement
//     mfem::ParGridFunction displacement( &par_fe_space );
//     displacement = 0.0;

//     mfem::ParGridFunction ref_coords( &par_fe_space );
//     mesh.GetNodes( ref_coords );

//     //recover dirchlet bd tdof list
//     mfem::Array<int> ess_vdof_marker( par_fe_space.GetVSize() );
//     ess_vdof_marker = 0;

//     // x-fixed on left
//     {
//       mfem::Array<int> tmp;
//       mfem::Array<int> bdr( mesh.bdr_attributes.Max() );
//       bdr = 0;
//       for ( auto a : xfixed_attrs ) bdr[a - 1] = 1;
//       par_fe_space.GetEssentialVDofs( bdr, tmp, 0 );
//       for ( int i = 0; i < tmp.Size(); ++i )
//         ess_vdof_marker[i] = ess_vdof_marker[i] || tmp[i];
//     }

//     // y-fixed on bottom
//     {
//       mfem::Array<int> tmp;
//       mfem::Array<int> bdr( mesh.bdr_attributes.Max() );
//       bdr = 0;
//       for ( auto a : yfixed_bottom_attrs ) bdr[a - 1] = 1;
//       par_fe_space.GetEssentialVDofs( bdr, tmp, 1 );
//       for ( int i = 0; i < tmp.Size(); ++i )
//         ess_vdof_marker[i] = ess_vdof_marker[i] || tmp[i];
//     }

//     // y-prescribed on top
//     mfem::Array<int> prescribed_vdof_marker( par_fe_space.GetVSize() );
//     prescribed_vdof_marker = 0;
//     {
//       mfem::Array<int> tmp;
//       mfem::Array<int> bdr( mesh.bdr_attributes.Max() );
//       bdr = 0;
//       for ( auto a : prescribed_attrs ) bdr[a - 1] = 1;
//       par_fe_space.GetEssentialVDofs( bdr, tmp, 1 );
//       prescribed_vdof_marker = tmp;
//       for ( int i = 0; i < tmp.Size(); ++i )
//         ess_vdof_marker[i] = ess_vdof_marker[i] || tmp[i];
//     }

//     mfem::Array<int> ess_tdof_list;
//     {
//       mfem::Array<int> ess_tdof_marker;
//       par_fe_space.GetRestrictionMatrix()->BooleanMult( ess_vdof_marker, ess_tdof_marker );
//       mfem::FiniteElementSpace::MarkerToList( ess_tdof_marker, ess_tdof_list );
//     }

//     mfem::Array<int> prescribed_tdof_list;
//     {
//       mfem::Array<int> marker;
//       par_fe_space.GetRestrictionMatrix()->BooleanMult( prescribed_vdof_marker, marker );
//       mfem::FiniteElementSpace::MarkerToList( marker, prescribed_tdof_list );
//     }

//     // set up mfem elasticity bilinear form
//     mfem::ParBilinearForm a( &par_fe_space );
//     mfem::ConstantCoefficient lambda_coeff( lam_ );
//     mfem::ConstantCoefficient mu_coeff( mu_ );
//     a.AddDomainIntegrator( new mfem::ElasticityIntegrator( lambda_coeff, mu_coeff ) );
//     a.Assemble();
//     a.Finalize();
//     auto A_elastic_raw = std::unique_ptr<mfem::HypreParMatrix>( a.ParallelAssemble() );

//     //Visit Output
//     mfem::VisItDataCollection visit_dc( "energy_patch_test", &mesh );
//     visit_dc.SetPrecision( 8 );
//     visit_dc.RegisterField( "displacement", &displacement );
//     visit_dc.SetCycle( 0 );
//     visit_dc.SetTime( 0.0 );
//     visit_dc.Save();

//     // timestepping loop for displacement
//     double disp_increment = total_prescribed_disp_ / num_timesteps_;
//     tribol::RealT dt = 1.0 / num_timesteps_;
//     int cs_id = 0, mesh1_id = 0, mesh2_id = 1;

//     mfem::Vector X( par_fe_space.GetTrueVSize() );
//     X = 0.0;

//     for ( int step = 1; step <= num_timesteps_; ++step )
//     {
//       double current_prescribed_disp = disp_increment * step;

//       // Prescribed displacement vector
//       mfem::Vector X_prescribed( par_fe_space.GetTrueVSize() );
//       X_prescribed = 0.0;
//       for ( int i = 0; i < prescribed_tdof_list.Size(); ++i ) {
//         X_prescribed( prescribed_tdof_list[i] ) = current_prescribed_disp;
//       }

//       // Update coordinates for contact detection
//       {
//         mfem::Vector X_temp( X );
//         for ( int i = 0; i < prescribed_tdof_list.Size(); ++i ) {
//           X_temp( prescribed_tdof_list[i] ) = current_prescribed_disp;
//         }
//         auto& P = *par_fe_space.GetProlongationMatrix();
//         P.Mult( X_temp, displacement );
//       }
//       coords = ref_coords;
//       coords += displacement;

//       // Re-register tribol each step (internal arrays need fresh allocation
//       // when contact pairs change between steps)
//       coords.ReadWrite();
//       tribol::registerMfemCouplingScheme( cs_id, mesh1_id, mesh2_id, mesh, coords,
//                                           mortar_attrs, nonmortar_attrs,
//                                           tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING,
//                                           tribol::ENERGY_MORTAR, tribol::FRICTIONLESS,
//                                           tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );
//       tribol::setLagrangeMultiplierOptions( cs_id, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN );
//       tribol::setMfemKinematicConstantPenalty( cs_id, 10000.0, 10000.0 );

//       tribol::updateMfemParallelDecomposition();
//       tribol::update( step, step * dt, dt );

//       auto A_cont = tribol::getMfemDfDx( cs_id );

//       mfem::Vector f_contact( par_fe_space.GetTrueVSize() );
//       f_contact = 0.0;
//       tribol::getMfemResponse( cs_id, f_contact );
//       f_contact.Neg();

//       // Inhomogeneous Dirichlet: rhs = f_contact - K * u_prescribed
//       auto A_total = std::unique_ptr<mfem::HypreParMatrix>(
//         mfem::Add( 1.0, *A_elastic_raw, 1.0, *A_cont ) );

//       mfem::Vector rhs( par_fe_space.GetTrueVSize() );
//       A_total->Mult( X_prescribed, rhs );
//       rhs.Neg();
//       rhs += f_contact;

//       for ( int i = 0; i < ess_tdof_list.Size(); ++i ) {
//         rhs( ess_tdof_list[i] ) = 0.0;
//       }

//       A_total->EliminateRowsCols( ess_tdof_list );

//       mfem::Vector X_free( par_fe_space.GetTrueVSize() );
//       X_free = 0.0;

//       mfem::HypreBoomerAMG amg( *A_total );
//       amg.SetElasticityOptions( &par_fe_space );
//       amg.SetPrintLevel( 0 );

//       mfem::MINRESSolver solver( MPI_COMM_WORLD );
//       solver.SetRelTol( 1.0e-8 );
//       solver.SetAbsTol( 1.0e-12 );
//       solver.SetMaxIter( 5000 );
//       solver.SetPrintLevel( step == num_timesteps_ ? 3 : 1 );
//       solver.SetPreconditioner( amg );
//       solver.SetOperator( *A_total );
//       solver.Mult( rhs, X_free );

//       X = X_free;
//       X += X_prescribed;

//       SLIC_INFO( "Timestep " << step << "/" << num_timesteps_
//                  << " | prescribed disp = " << current_prescribed_disp );

//       // Save VisIt output
//       {
//         auto& P = *par_fe_space.GetProlongationMatrix();
//         P.Mult( X, displacement );
//       }
//       visit_dc.SetCycle( step );
//       visit_dc.SetTime( step * dt );
//       visit_dc.Save();
//     }

//     //Get final disaplacent
//     {
//       auto& P = *par_fe_space.GetProlongationMatrix();
//       P.Mult( X, displacement );
//     }

//     auto local_max = displacement.Max();
//     max_disp_ = 0.0;
//     MPI_Allreduce( &local_max, &max_disp_, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
//     SLIC_INFO( "Max displacement: " << max_disp_ );

//     // -----------------------------------------------------------------
//     // Analytical solution comparison
//     //
//     // Plane strain, uniaxial stress (sigma_xx = 0, free right side):
//     //   eps_yy = applied_disp / total_height = -0.01 / 2.0 = -0.005
//     //   eps_xx = -lambda/(lambda + 2*mu) * eps_yy
//     //   u_y = eps_yy * y
//     //   u_x = eps_xx * x
//     // -----------------------------------------------------------------
//     double total_height = 2.0;
//     double eps_yy = total_prescribed_disp_ / total_height;
//     double eps_xx = -lam_ / ( lam_ + 2.0 * mu_ ) * eps_yy;

//     SLIC_INFO( "Analytical: eps_yy = " << eps_yy << ", eps_xx = " << eps_xx );

//     mfem::VectorFunctionCoefficient exact_sol_coeff( 2,
//       [eps_xx, eps_yy]( const mfem::Vector& x, mfem::Vector& u ) {
//         u[0] = eps_xx * x[0];
//         u[1] = eps_yy * x[1];
//       } );

//     mfem::ParGridFunction exact_disp( &par_fe_space );
//     exact_disp.ProjectCoefficient( exact_sol_coeff );

//     // Vector error
//     mfem::ParGridFunction error_vec( exact_disp );
//     error_vec -= displacement;
//     l2_err_vec_ = mfem::ParNormlp( error_vec, 2, MPI_COMM_WORLD );

//     // Component-wise errors
//     const mfem::FiniteElementCollection* fec = par_fe_space.FEColl();
//     mfem::ParFiniteElementSpace scalar_fes( &mesh, fec, 1, par_fe_space.GetOrdering() );
//     const int n = scalar_fes.GetNDofs();

//     mfem::ParGridFunction ux_exact( &scalar_fes ), ux_num( &scalar_fes );
//     mfem::ParGridFunction uy_exact( &scalar_fes ), uy_num( &scalar_fes );

//     for ( int i = 0; i < n; ++i ) {
//       ux_exact( i ) = exact_disp( i );
//       ux_num( i )   = displacement( i );
//       uy_exact( i ) = exact_disp( n + i );
//       uy_num( i )   = displacement( n + i );
//     }

//     mfem::ParGridFunction ux_err( ux_exact );
//     ux_err -= ux_num;
//     l2_err_x_ = mfem::ParNormlp( ux_err, 2, MPI_COMM_WORLD );

//     mfem::ParGridFunction uy_err( uy_exact );
//     uy_err -= uy_num;
//     l2_err_y_ = mfem::ParNormlp( uy_err, 2, MPI_COMM_WORLD );

//     SLIC_INFO( "L2 error (vector): " << l2_err_vec_ );
//     SLIC_INFO( "L2 error (x):      " << l2_err_x_ );
//     SLIC_INFO( "L2 error (y):      " << l2_err_y_ );
//     SLIC_INFO( "Consistency check |err_vec^2 - (err_x^2 + err_y^2)| = "
//                << std::abs( l2_err_vec_ * l2_err_vec_
//                             - ( l2_err_x_ * l2_err_x_ + l2_err_y_ * l2_err_y_ ) ) );
//   }
// };

// TEST_P( MfemMortarEnergyPatchTest, check_patch_test )
// {
//   EXPECT_GT( max_disp_, 0.0 );
//   EXPECT_NEAR( 0.0, l2_err_vec_, 1.0e-2 );
//   EXPECT_NEAR( 0.0, l2_err_x_,  1.0e-2 );
//   EXPECT_NEAR( 0.0, l2_err_y_,  1.0e-2 );

//   MPI_Barrier( MPI_COMM_WORLD );
// }

// INSTANTIATE_TEST_SUITE_P( tribol, MfemMortarEnergyPatchTest, testing::Values( std::make_tuple( 2 ) ) );

// //------------------------------------------------------------------------------
// #include "axom/slic/core/SimpleLogger.hpp"

// int main( int argc, char* argv[] )
// {
//   int result = 0;

//   MPI_Init( &argc, &argv );
//   ::testing::InitGoogleTest( &argc, argv );

// #ifdef TRIBOL_USE_UMPIRE
//   umpire::ResourceManager::getInstance();
// #endif

//   axom::slic::SimpleLogger logger;
//   result = RUN_ALL_TESTS();

//   tribol::finalize();
//   MPI_Finalize();

//   return result;
// }



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

#include "shared/mesh/MeshBuilder.hpp"
#include "redecomp/redecomp.hpp"

#include "tribol/config.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

/**
 * @brief Contact patch test using ENERGY_MORTAR with Lagrange multiplier
 *        enforcement and prescribed displacement applied incrementally.
 *
 * Two unit squares [0,1]x[0,1] and [0,1]x[1,2] with zero initial gap.
 * Linear elasticity with lambda = mu = 5.
 *
 * Saddle point system solved each Newton iteration:
 *
 *   [ K + H    G^T ] [ δu ] = -[ R_u ]
 *   [ G         0  ] [ δλ ]    [ R_λ ]
 *
 * where:
 *   K   = elastic stiffness
 *   H   = λ · d²g̃/du²  (contact Hessian contribution)
 *   G   = dg̃/du         (constraint Jacobian)
 *   R_u = K·u + G^T·λ - f_ext  (force residual)
 *   R_λ = g̃(u)                  (gap constraint residual)
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
  static constexpr int    num_timesteps_ = 1;
  static constexpr double total_prescribed_disp_ = -0.01;
  static constexpr double lam_ = 50.0;
  static constexpr double mu_  = 50.0;
  static constexpr int    max_newton_iter_ = 10;
  static constexpr double newton_rtol_ = 1.0e-10;
  static constexpr double newton_atol_ = 1.0e-12;
  // ------------------------------------

  void SetUp() override
  {
    int ref_levels = std::get<0>( GetParam() );
    int order = 1;

    auto mortar_attrs     = std::set<int>( { 5 } );
    auto nonmortar_attrs  = std::set<int>( { 3 } );
    auto xfixed_attrs     = std::set<int>( { 4 } );
    auto yfixed_bottom_attrs = std::set<int>( { 1 } );
    auto prescribed_attrs = std::set<int>( { 6 } );

    // int nel_per_dir_top = std::pow( 2, ref_levels );
    // int nel_per_dir_bottom = std::pow(3, ref_levels);
    int nel_per_dir_top = 10;
    int nel_per_dir_bottom = 3;

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
      for ( int i = 0; i < tmp.Size(); ++i )
        ess_vdof_marker[i] = ess_vdof_marker[i] || tmp[i];
    }

// Pin corners: bottom-left of bottom block (0,0) and top-left of top block (0,2)
// {
//   const double tol = 1.0e-10;
//   const std::vector<std::pair<double, double>> pin_pts = { {0.0, 0.0}, {0.0, 2.0} };

//   for ( int v = 0; v < mesh.GetNV(); ++v ) {
//     const double* vc = mesh.GetVertex(v);
//     for ( auto& [px, py] : pin_pts ) {
//       if ( std::abs(vc[0] - px) < tol && std::abs(vc[1] - py) < tol ) {
//         mfem::Array<int> vdofs;
//         par_fe_space.GetVertexVDofs( v, vdofs );
//         for ( int i = 0; i < vdofs.Size(); ++i )
//           ess_vdof_marker[ vdofs[i] ] = 1;
//         break;
//       }
//     }
//   }
// }



    // y-fixed on bottom
    {
      mfem::Array<int> tmp;
      mfem::Array<int> bdr( mesh.bdr_attributes.Max() );
      bdr = 0;
      for ( auto a : yfixed_bottom_attrs ) bdr[a - 1] = 1;
      par_fe_space.GetEssentialVDofs( bdr, tmp, 1 );
      for ( int i = 0; i < tmp.Size(); ++i )
        ess_vdof_marker[i] = ess_vdof_marker[i] || tmp[i];
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
      for ( int i = 0; i < tmp.Size(); ++i )
        ess_vdof_marker[i] = ess_vdof_marker[i] || tmp[i];
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
    visit_dc.RegisterField( "Exact Replacement", &exact_disp);
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

    for ( int step = 1; step <= num_timesteps_; ++step )
    {
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
      for ( int newton = 0; newton < max_newton_iter_; ++newton )
      {
        // Update coordinates with current displacement
        {
          auto& P = *par_fe_space.GetProlongationMatrix();
          P.Mult( U, displacement );
        }
        coords = ref_coords;
        coords += displacement;

        // Register tribol and update contact data
        coords.ReadWrite();
        tribol::registerMfemCouplingScheme( cs_id, mesh1_id, mesh2_id, mesh, coords,
                                            mortar_attrs, nonmortar_attrs,
                                            tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING,
                                            tribol::ENERGY_MORTAR, tribol::FRICTIONLESS,
                                            tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID );
        tribol::setLagrangeMultiplierOptions( cs_id, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN );

        tribol::updateMfemParallelDecomposition();
        tribol::update( step, step * dt, dt );

        // ---- Get contact surface FE space and initialize lambda on first pass ----
        // TODO: adapt to actual tribol API for accessing contact FE space
        auto& contact_fes = tribol::getMfemContactFESpace( cs_id );
        contact_size = contact_fes.GetTrueVSize();

        if ( lambda == nullptr ) {
          lambda = new mfem::HypreParVector( &contact_fes );
          *lambda = 0.0;
        }

        // ---- Evaluate contact residual ----
        mfem::HypreParVector r_contact_force( &par_fe_space );  // G^T * lambda (disp-sized)
        r_contact_force = 0.0;
        mfem::HypreParVector r_gap( &contact_fes );             // g_tilde (contact-sized)
        r_gap = 0.0;

        // TODO: adapt to actual tribol API
        // NOTE: verify that evaluateContactResidual computes r_force = G^T * lambda
        //       (check Mult vs MultTranspose in the adapter -- see note below)
        tribol::evaluateContactResidual( cs_id, *lambda, r_contact_force, r_gap );

        // ---- Evaluate contact Jacobian blocks ----
        std::unique_ptr<mfem::HypreParMatrix> H;   // lambda * d2g/du2 (disp x disp)
        std::unique_ptr<mfem::HypreParMatrix> G;   // dg/du (contact x disp)
        // TODO: adapt to actual tribol API
        tribol::evaluateContactJacobian( cs_id, *lambda, H, G );

        // ---- Assemble block residual ----
        //
        //   R_u = K*U + G^T*lambda    (elastic force + contact force)
        //         (no external body forces in this test, only prescribed disp)
        //   R_λ = g̃(U)
        //
        // Note: the prescribed displacement is handled by eliminating those DOFs
        // from the Newton system and keeping them fixed at the prescribed values.

        mfem::Vector R_u( disp_size );
        K_elastic->Mult( U, R_u );       // R_u = K * U
        R_u += r_contact_force;           // R_u += G^T * lambda

        mfem::Vector R_lambda( contact_size );
        R_lambda = r_gap;                 // R_lambda = g_tilde

        // Compute residual norms for convergence check
        double norm_R_u = mfem::InnerProduct( MPI_COMM_WORLD, R_u, R_u );
        double norm_R_lambda = mfem::InnerProduct( MPI_COMM_WORLD, R_lambda, R_lambda );
        // Zero out essential DOF contributions before computing norm
        for ( int i = 0; i < ess_tdof_list.Size(); ++i ) {
          norm_R_u -= R_u( ess_tdof_list[i] ) * R_u( ess_tdof_list[i] );
        }
        double residual_norm = std::sqrt( std::abs( norm_R_u ) + norm_R_lambda );

        SLIC_INFO( "  Step " << step << " Newton " << newton
                   << " | residual = " << residual_norm );

        if ( newton > 0 && residual_norm < newton_atol_ ) {
          SLIC_INFO( "  Newton converged (abs tol) at iteration " << newton );
          break;
        }

        // ---- Assemble block Jacobian ----
        //
        //   J = [ K + H    G^T ]
        //       [ G         0  ]

        // SLIC_INFO( "    Building J_uu..." );

        // (0,0) block: K + H
        // NOTE: H may be null on the first Newton iteration when lambda = 0
        std::unique_ptr<mfem::HypreParMatrix> J_uu;
        if ( H && H->NumRows() > 0 ) {
          J_uu.reset( mfem::Add( 1.0, *K_elastic, 1.0, *H ) );
        } else {
          J_uu.reset( new mfem::HypreParMatrix( *K_elastic ) );
        }

        // SLIC_INFO( "    J_uu: " << J_uu->NumRows() << " x " << J_uu->NumCols() );
        // SLIC_INFO( "    G:    " << G->NumRows() << " x " << G->NumCols() );

        // G^T for the (0,1) block
        auto G_T = std::unique_ptr<mfem::HypreParMatrix>( G->Transpose() );

        // SLIC_INFO( "    G^T:  " << G_T->NumRows() << " x " << G_T->NumCols() );

        // ---- Apply essential BCs ----
        // Zero out essential DOF rows/cols in J_uu
        for ( int i = 0; i < ess_tdof_list.Size(); ++i ) {
          R_u( ess_tdof_list[i] ) = 0.0;
        }
        J_uu->EliminateRowsCols( ess_tdof_list );

        // Zero out essential DOF rows in G^T (cols in G)
        // Use EliminateRows on G^T which is simpler than EliminateCols on G
        G_T->EliminateRows( ess_tdof_list );

        // Rebuild G from the modified G^T to stay consistent
        G = std::unique_ptr<mfem::HypreParMatrix>( G_T->Transpose() );

        // SLIC_INFO( "    After BC elim - J_uu: " << J_uu->NumRows() << " x " << J_uu->NumCols() );
        // SLIC_INFO( "    After BC elim - G:    " << G->NumRows() << " x " << G->NumCols() );

        // ---- Set up block system ----

        mfem::Array<int> block_offsets( 3 );
        block_offsets[0] = 0;
        block_offsets[1] = disp_size;
        block_offsets[2] = disp_size + contact_size;

        // SLIC_INFO( "    Block offsets: [0, " << disp_size << ", " << disp_size + contact_size << "]" );

        mfem::BlockOperator J_block( block_offsets );
        J_block.SetBlock( 0, 0, J_uu.get() );
        J_block.SetBlock( 0, 1, G_T.get() );
        J_block.SetBlock( 1, 0, G.get() );

        // Block RHS = -[R_u; R_lambda]
        mfem::BlockVector rhs( block_offsets );
        rhs.GetBlock( 0 ) = R_u;
        rhs.GetBlock( 0 ).Neg();
        rhs.GetBlock( 1 ) = R_lambda;
        rhs.GetBlock( 1 ).Neg();

        // SLIC_INFO( "    Solving saddle point system..." );

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

        SLIC_INFO( "    Solver converged: " << solver.GetConverged()
                   << " in " << solver.GetNumIterations() << " iterations" );

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

      SLIC_INFO( "Timestep " << step << "/" << num_timesteps_
                 << " | prescribed disp = " << current_prescribed_disp );

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

    mfem::VectorFunctionCoefficient exact_sol_coeff( 2,
      [eps_xx, eps_yy]( const mfem::Vector& x, mfem::Vector& u ) {
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
      ux_num( i )   = displacement( i );
      uy_exact( i ) = exact_disp( n + i );
      uy_num( i )   = displacement( n + i );
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
               << std::abs( l2_err_vec_ * l2_err_vec_
                            - ( l2_err_x_ * l2_err_x_ + l2_err_y_ * l2_err_y_ ) ) );
  }
};

TEST_P( MfemMortarEnergyLagrangePatchTest, check_patch_test )
{
  EXPECT_GT( max_disp_, 0.0 );
  EXPECT_NEAR( 0.0, l2_err_vec_, 1.0e-2 );
  EXPECT_NEAR( 0.0, l2_err_x_,  1.0e-2 );
  EXPECT_NEAR( 0.0, l2_err_y_,  1.0e-2 );

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
