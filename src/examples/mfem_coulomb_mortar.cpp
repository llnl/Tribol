// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
// SPDX-License-Identifier: (MIT)
/**
 * @file
 * @brief Compress two nonmatching elastic blocks and load, unload, and reverse their tangential displacement.
 * A damped semismooth Newton solve couples displacement and Cartesian contact multipliers. Each load step begins
 * from accepted geometry; failed solves roll back friction data. The example uses linear elastic bulk response
 * and finite-sliding contact geometry, with a nonsymmetric GMRES solve and no multiplier stabilization.
 */
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "axom/slic/core/SimpleLogger.hpp"
#include "shared/mesh/MeshBuilder.hpp"
#include "shared/math/ParSparseMat.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

namespace {
using namespace tribol;

/** @brief Build and optionally refine the two reference blocks. @param refinements Uniform refinement count.
 * @return Parallel reference mesh with nonmatching contact faces. */
mfem::ParMesh makeMesh( int refinements )
{
  mfem::ParMesh mesh( static_cast<const mfem::ParMesh&>( shared::ParMeshBuilder(
      MPI_COMM_WORLD, shared::MeshBuilder::Unify( { shared::MeshBuilder::CubeMesh( 2, 2, 1 ).updateBdrAttrib( 6, 7 ),
                                                    shared::MeshBuilder::CubeMesh( 3, 3, 1 )
                                                        .translate( { 0.0, 0.0, 1.0 } )
                                                        .updateBdrAttrib( 1, 8 )
                                                        .updateAttrib( 1, 2 ) } ) ) ) );
  for ( int i = 0; i < refinements; ++i ) mesh.UniformRefinement();
  // Keep the cube boundary fixed while separating interior grid lines on the two contact surfaces after refinement.
  mfem::Array<int> upper( mesh.GetNV() );
  upper = 0;
  for ( int e = 0; e < mesh.GetNE(); ++e )
    if ( mesh.GetAttribute( e ) == 2 ) {
      mfem::Array<int> vertices;
      mesh.GetElementVertices( e, vertices );
      for ( int vertex : vertices ) upper[vertex] = 1;
    }
  constexpr RealT pi = 3.14159265358979323846;
  constexpr RealT x_grid_shift = 0.02, y_grid_shift = 0.03;
  for ( int vertex = 0; vertex < mesh.GetNV(); ++vertex )
    if ( upper[vertex] ) {
      auto* x = mesh.GetVertex( vertex );
      x[0] += x_grid_shift * std::sin( pi * x[0] );
      x[1] += y_grid_shift * std::sin( pi * x[1] );
    }
  if ( auto* coordinates = mesh.GetNodes() ) {
    auto& space = *coordinates->FESpace();
    for ( int vertex = 0; vertex < mesh.GetNV(); ++vertex ) {
      mfem::Array<int> dofs;
      space.GetVertexDofs( vertex, dofs );
      for ( int d = 0; d < 3; ++d ) ( *coordinates )[space.DofToVDof( dofs[0], d )] = mesh.GetVertex( vertex )[d];
    }
    mesh.NodesUpdated();
  }
  return mfem::ParMesh( mesh );
}

/** @brief Solve the prescribed loading of two linear-elastic blocks on fixed parent meshes. */
class FrictionPatch {
 public:
  /** @brief Create nonmatching contact faces, linear bulk stiffness, and Coulomb fields.
   * @param refinements Uniform refinement count for the reference mesh. */
  explicit FrictionPatch( int refinements = 0 )
      : mesh_( makeMesh( refinements ) ),
        fec_( 1, 3 ),
        fes_( &mesh_, &fec_, 3 ),
        coordinates_( &fes_ ),
        displacement_( &fes_ )
  {
    mesh_.GetNodes( coordinates_ );
    reference_ = coordinates_;
    displacement_ = 0.0;
    mfem::ConstantCoefficient lambda( 0.0 ), mu( 1.0 );
    mfem::ParBilinearForm form( &fes_ );
    form.AddDomainIntegrator( new mfem::ElasticityIntegrator( lambda, mu ) );
    form.Assemble();
    form.Finalize();
    stiffness_.reset( form.ParallelAssemble() );
    mfem::Array<int> marker( mesh_.bdr_attributes.Max() );
    marker = 0;
    marker[0] = 1;
    marker[5] = 1;
    fes_.GetEssentialTrueDofs( marker, essential_ );
    marker = 0;
    marker[5] = 1;
    fes_.GetEssentialTrueDofs( marker, top_ );
    top_marker_.SetSize( fes_.GetTrueVSize() );
    top_marker_ = 0;
    for ( int i : top_ ) top_marker_[i] = 1;
    registerMfemCouplingScheme( 0, 0, 1, mesh_, coordinates_, { 7 }, { 8 }, SURFACE_TO_SURFACE, NO_CASE, SINGLE_MORTAR,
                                FRICTION_COULOMB, LAGRANGE_MULTIPLIER, BINNING_GRID, ExecutionMode::Sequential );
    setCoulombFrictionOptions( 0, { friction_coefficient_, 2.0, 2.0, 1.0e-14 } );
    updateMfemParallelDecomposition( 1, true );
    displacement_true_.SetSize( fes_.GetTrueVSize() );
    displacement_true_ = 0.0;
  }
  /** @brief Release scheme data while its borrowed parent mesh and spaces remain alive. */
  ~FrictionPatch() { finalize(); }

  /** @brief Run compression, tangential loading, unloading, and reversal. */
  void run()
  {
    const MdArray<RealT, 11> shear{ 0.0, 0.001, 0.002, 0.004, 0.007, 0.01, 0.007, 0.004, 0.001, -0.002, -0.005 };
    RealT previous_shear = 0.0;
    bool saw_stick = false, saw_slip = false, first_step = true;
    constexpr RealT sliding_prediction_increment = 0.0025;  // Smaller increments retain the accepted traction guess.
    for ( int load = 0; load < shear.size(); ++load ) {
      const RealT target = shear[load];
      beginContactStep( 0 );
      mfem::Vector accepted( displacement_true_ );
      bool converged = false;
      try {
        prescribe( target, -0.02 );
        if ( first_step ) {
          // A positive initial guess selects closed normal constraints for the initial compression solve.
          auto& traction = getMfemContactTraction( 0 );
          auto& space = *traction.ParFESpace();
          constexpr RealT initial_pressure_guess = 0.01;
          for ( int e = 0; e < space.GetNE(); ++e )
            if ( space.GetAttribute( e ) == 8 ) {
              mfem::Array<int> nodes;
              space.GetElementDofs( e, nodes );
              for ( int node : nodes ) traction[space.DofToVDof( node, 2 )] = -initial_pressure_guess;
            }
          mfem::Vector values;
          traction.GetTrueDofs( values );
          traction.SetFromTrueDofs( values );
        }
        if ( std::abs( target - previous_shear ) >= sliding_prediction_increment ) {
          predictSliding( target - previous_shear );
          predictEquilibrium();
        }
        for ( int iteration = 0; iteration < 30; ++iteration ) {
          auto residual = evaluate();
          RealT merit = norm( residual );
          int rank = 0;
          MPI_Comm_rank( MPI_COMM_WORLD, &rank );
          if ( rank == 0 )
            std::cout << "shear=" << target << " iteration=" << iteration << " residual=" << merit << '\n';
          if ( merit < 1.0e-9 ) {
            converged = true;
            break;
          }
          auto jacobian = getMfemBlockJacobian( 0 );
          auto& j00 = static_cast<mfem::HypreParMatrix&>( jacobian->GetBlock( 0, 0 ) );
          auto combined = shared::ParSparseMatView( stiffness_.get() ) + shared::ParSparseMatView( &j00 );
          combined.eliminateRowsCols( essential_ );
          jacobian->SetBlock( 0, 0, combined.release() );
          shared::ParSparseMatView( &static_cast<mfem::HypreParMatrix&>( jacobian->GetBlock( 0, 1 ) ) )
              .eliminateRows( essential_ );
          shared::ParSparseMatView( &static_cast<mfem::HypreParMatrix&>( jacobian->GetBlock( 1, 0 ) ) )
              .eliminateCols( essential_ );
          mfem::GMRESSolver solver( MPI_COMM_WORLD );
          solver.SetRelTol( 1.0e-11 );
          solver.SetAbsTol( 1.0e-13 );
          solver.SetMaxIter( 3000 );
          solver.SetKDim( 1000 );
          solver.SetPrintLevel( 0 );
          solver.SetOperator( *jacobian );
          mfem::Vector correction( residual.Size() );
          correction = 0.0;
          residual *= -1.0;
          solver.Mult( residual, correction );
          if ( !solver.GetConverged() ) throw std::runtime_error( "GMRES did not converge." );
          auto& traction = getMfemContactTraction( 0 );
          mfem::Vector base_multiplier;
          traction.GetTrueDofs( base_multiplier );
          mfem::Vector base_displacement( displacement_true_ );
          bool accepted_trial = false;
          for ( RealT alpha = 1.0; alpha >= 1.0 / 1024.0; alpha *= 0.5 ) {
            for ( int i = 0; i < displacement_true_.Size(); ++i )
              displacement_true_[i] = base_displacement[i] + alpha * correction[i];
            mfem::Vector trial( base_multiplier );
            for ( int i = 0; i < trial.Size(); ++i ) trial[i] += alpha * correction[displacement_true_.Size() + i];
            traction.SetFromTrueDofs( trial );
            RealT trial_merit = norm( evaluate() );
            if ( trial_merit <= ( 1.0 - 1.0e-4 * alpha ) * merit ) {
              accepted_trial = true;
              break;
            }
          }
          if ( !accepted_trial ) throw std::runtime_error( "Newton backtracking did not reduce the residual." );
        }
        if ( !converged ) throw std::runtime_error( "Newton did not converge." );
        auto states = checkContact( target == 0.0 && previous_shear == 0.0 );
        saw_stick = saw_stick || states.first;
        saw_slip = saw_slip || states.second;
        commitContactStep( 0 );
        previous_shear = target;
        first_step = false;
      } catch ( ... ) {
        rollbackContactStep( 0 );
        displacement_true_ = accepted;
        updateCoordinates();
        throw;
      }
    }
    if ( !saw_stick || !saw_slip ) throw std::runtime_error( "Loading did not exercise both stick and slip." );
  }

 private:
  static constexpr RealT friction_coefficient_{ 0.1 };  ///< Isotropic friction coefficient for this example.
  /** @brief Initialize tangential traction along the equation-20 slip predicted from a large imposed increment.
   * @param increment Change in top x displacement; the current normal-pressure guess is retained. */
  void predictSliding( RealT increment )
  {
    auto& traction = getMfemContactTraction( 0 );
    mfem::Vector values;
    traction.GetTrueDofs( values );
    const int nodes = accepted_pressure_.Size();
    constexpr RealT trial_factor =
        1.1;  // Select the sliding branch with a guess just outside the current tangent cone.
    for ( int i = 0; i < nodes; ++i ) {
      MdArray<RealT, 3> direction{};
      RealT norm_squared = 0.0;
      for ( int d = 0; d < 3; ++d ) {
        direction[d] = ( increment > 0.0 ? -1.0 : 1.0 ) *
                       ( ( d == 0 ? 1.0 : 0.0 ) - accepted_normal_[i + d * nodes] * accepted_normal_[i] );
        norm_squared += direction[d] * direction[d];
      }
      if ( norm_squared <= 0.0 || accepted_pressure_[i] <= 0.0 ) continue;
      for ( int d = 0; d < 3; ++d )
        values[i + d * nodes] =
            accepted_pressure_[i] * accepted_normal_[i + d * nodes] +
            trial_factor * friction_coefficient_ * accepted_pressure_[i] * direction[d] / std::sqrt( norm_squared );
    }
    traction.SetFromTrueDofs( values );
  }
  /** @brief Balance the linear bulk residual once for the predicted tractions before coupled Newton iteration. */
  void predictEquilibrium()
  {
    auto residual = evaluate();
    mfem::Vector rhs( residual, 0, displacement_true_.Size() );
    rhs *= -1.0;
    shared::ParSparseMat bulk( new mfem::HypreParMatrix( *stiffness_ ) );
    bulk.eliminateRowsCols( essential_ );
    mfem::GMRESSolver solver( MPI_COMM_WORLD );
    solver.SetRelTol( 1.0e-12 );
    solver.SetAbsTol( 1.0e-13 );
    solver.SetMaxIter( 3000 );
    solver.SetKDim( 1000 );
    solver.SetPrintLevel( 0 );
    solver.SetOperator( bulk.get() );
    mfem::Vector correction( displacement_true_.Size() );
    correction = 0.0;
    solver.Mult( rhs, correction );
    if ( !solver.GetConverged() ) throw std::runtime_error( "Bulk displacement prediction did not converge." );
    displacement_true_ += correction;
  }
  /** @brief Set current coordinates from reference plus the current displacement unknown. */
  void updateCoordinates()
  {
    displacement_.SetFromTrueDofs( displacement_true_ );
    coordinates_ = reference_;
    coordinates_ += displacement_;
  }
  /** @brief Set top displacement and retain zero displacement at the bottom.
   * @param shear Prescribed top x displacement. @param compression Prescribed top z displacement. */
  void prescribe( RealT shear, RealT compression )
  {
    mfem::Vector values( 3 );
    values = 0.0;
    values[0] = shear;
    values[2] = compression;
    mfem::VectorConstantCoefficient coefficient( values );
    mfem::ParGridFunction prescribed( &fes_ );
    prescribed.ProjectCoefficient( coefficient );
    mfem::Vector target;
    prescribed.GetTrueDofs( target );
    for ( int i : essential_ ) displacement_true_[i] = top_marker_[i] ? target[i] : 0.0;
  }
  /** @brief Refresh contact and return the coupled bulk/contact residual. @return Locally owned block entries. */
  mfem::Vector evaluate()
  {
    updateCoordinates();
    updateMfemParallelDecomposition( 1, true );
    RealT dt = 1.0;
    update( 0, 0.0, dt );
    auto contact = getMfemContactResidual( 0 );
    mfem::Vector residual( displacement_true_.Size() + contact.constraint->Size() );
    mfem::Vector bulk( residual, 0, displacement_true_.Size() );
    stiffness_->Mult( displacement_true_, bulk );
    bulk += *contact.force;
    for ( int i : essential_ ) bulk[i] = 0.0;
    for ( int i = 0; i < contact.constraint->Size(); ++i )
      residual[displacement_true_.Size() + i] = ( *contact.constraint )[i];
    return residual;
  }
  /** @brief Compute the global residual norm. @param vector Locally owned entries. @return Euclidean norm. */
  RealT norm( const mfem::Vector& vector ) const
  {
    return std::sqrt( mfem::InnerProduct( MPI_COMM_WORLD, vector, vector ) );
  }
  /** @brief Check equilibrium acceptance and Coulomb admissibility; report contact-state counts.
   * @param compression_patch Whether to check the analytic uniaxial pressure for the first load step.
   * @return Whether any nodes stick and whether any nodes slip. */
  std::pair<bool, bool> checkContact( bool compression_patch )
  {
    auto contact = getMfemContactResidual( 0 );
    if ( norm( *contact.constraint ) > 1.0e-8 ) throw std::runtime_error( "Contact constraints are unconverged." );
    mfem::Vector lambda;
    getMfemContactTraction( 0 ).GetTrueDofs( lambda );
    MdArray<int, 3> counts{}, global{};
    int invalid = 0, any_invalid = 0;
    int nodes = contact.state.Size();
    for ( int i = 0; i < nodes; ++i ) {
      ++counts[contact.state[i]];
      RealT tangential_squared = -contact.pressure[i] * contact.pressure[i], dissipation = 0.0;
      for ( int d = 0; d < 3; ++d ) {
        tangential_squared += lambda[i + d * nodes] * lambda[i + d * nodes];
        dissipation += lambda[i + d * nodes] * contact.slip[i + d * nodes];
      }
      RealT limit = friction_coefficient_ * contact.pressure[i];
      if ( contact.pressure[i] < -1.0e-8 || contact.gap[i] > 1.0e-8 || dissipation < -1.0e-8 ||
           tangential_squared > limit * limit + 1.0e-8 || std::abs( contact.pressure[i] * contact.gap[i] ) > 1.0e-8 )
        invalid = 1;
      // With lambda_bulk=0 and mu_bulk=1, E=2. A 0.02 shortening over total height 2 gives pressure 0.02.
      if ( compression_patch && contact.state[i] != static_cast<int>( FrictionState::Open ) &&
           std::abs( contact.pressure[i] - 0.02 ) > 1.0e-6 )
        invalid = 1;
    }
    MPI_Allreduce( &invalid, &any_invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
    if ( any_invalid ) throw std::runtime_error( "Mortar patch acceptance checks failed." );
    MPI_Allreduce( counts.data(), global.data(), 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    int rank = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank == 0 ) std::cout << "open=" << global[0] << " stick=" << global[1] << " slip=" << global[2] << '\n';
    accepted_normal_ = contact.normal;
    accepted_pressure_ = contact.pressure;
    return { global[1] > 0, global[2] > 0 };
  }
  mfem::ParMesh mesh_;                                ///< Fixed reference volume mesh.
  mfem::H1_FECollection fec_;                         ///< Linear bulk basis.
  mfem::ParFiniteElementSpace fes_;                   ///< Displacement and current-coordinate space.
  mfem::ParGridFunction coordinates_, displacement_;  ///< Caller-owned current inputs.
  mfem::Vector accepted_normal_, accepted_pressure_;  ///< Accepted diagnostics used only for the next Newton guess.
  mfem::Vector reference_, displacement_true_;        ///< Reference geometry and current displacement unknown.
  mfem::Array<int> essential_, top_, top_marker_;     ///< Bottom/top boundary conditions on true DOFs.
  std::unique_ptr<mfem::HypreParMatrix> stiffness_;   ///< Linear-elastic reference stiffness.
};
}  // namespace
/** @brief Run the MPI friction example. @param argc Argument count. @param argv Arguments. @return Zero on success. */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  int status = 0;
  {
    axom::slic::SimpleLogger logger;
    try {
      int refinements = 0;
      mfem::OptionsParser options( argc, argv );
      options.AddOption( &refinements, "-r", "--refinements", "Uniform reference-mesh refinement count (0 or 1)." );
      options.Parse();
      if ( !options.Good() || refinements < 0 || refinements > 1 )
        throw std::invalid_argument( "Use --refinements 0 or 1." );
      FrictionPatch patch( refinements );
      patch.run();
    } catch ( const std::exception& error ) {
      std::cerr << error.what() << '\n';
      status = 1;
    }
  }
  MPI_Finalize();
  return status;
}
