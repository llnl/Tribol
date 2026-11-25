// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Parallel timestep calculator for contact formulations built on PairTractionIntegral.
 */
#ifndef SRC_TRIBOL_PHYSICS_TIMESTEP_TIMESTEPCALCULATOR_HPP_
#define SRC_TRIBOL_PHYSICS_TIMESTEP_TIMESTEPCALCULATOR_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>

#include "axom/slic/interface/slic_macros.hpp"
#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Atomics.hpp"
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/LoopExec.hpp"
#include "tribol/common/Reductions.hpp"
#include "tribol/mesh/InterfacePairs.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/physics/integrals/PairIntegrals.hpp"
#include "tribol/physics/timestep/TimestepCriteria.hpp"

#ifdef TRIBOL_USE_MPI
#include <mpi.h>
#endif

#ifdef BUILD_REDECOMP
#include "tribol/assembly/MfemMatrixAssembler.hpp"
#endif

namespace tribol {

class CouplingScheme;

namespace detail {

#ifdef BUILD_REDECOMP
/** @brief MFEM data needed to calculate a nodal penalty CFL limit on parent true DOFs. */
struct MfemParentPenaltyCflData {
  const MfemMeshData& mesh_data;          ///< Supplies the registered parent inverse mass and Redecomp element maps.
  const MfemJacobianData& transfer_data;  ///< Transfers the frozen penalty matrix to the parent true DOFs.
};

/**
 * @brief Prepare the MFEM matrix transfer for a nodal penalty CFL vote.
 * @param scheme MFEM coupling scheme that owns the mesh and caches the matrix transfer.
 * @return References to the scheme's mesh and transfer data. Both must remain alive while calculating a vote.
 */
MfemParentPenaltyCflData prepareMfemParentPenaltyCfl( CouplingScheme& scheme );

/**
 * @brief Bound the mass-normalized frozen penalty matrix by its maximum absolute row sum across all ranks.
 * @param penalty_matrix Signed penalty matrix assembled on parent true DOFs.
 * @param mesh_data Supplies inverse diagonal mass on the parent mesh.
 * @return The row-sum bound, including zero for a zero operator, or std::nullopt after reporting invalid inputs.
 */
std::optional<RealT> computeParentPenaltyRowSumBound( const mfem::HypreParMatrix& penalty_matrix,
                                                      const MfemMeshData& mesh_data );
#else
struct MfemParentPenaltyCflData;
#endif

}  // namespace detail

/**
 * @brief Contact normal, contact activation, and penalty stiffness at one quadrature point for timestep voting.
 */
struct QuadraturePointTimestepData {
  bool valid{ false };                              ///< True when local normals and orientation are usable.
  bool in_contact{ false };                         ///< True when the point is actively in contact.
  axom::primal::Vector<RealT, 3> contact_normal{};  ///< Contact unit normal pointing from geometry1 toward geometry2.
  RealT penalty_stiffness{ 0.0 };                   ///< Normal penalty stiffness per unit area.
};

/**
 * @brief Composable timestep calculator dispatching active criteria selected in TimestepSettings.
 */
class TimestepCalculator {
 public:
  /**
   * @brief Construct a calculator with the given runtime timestep settings.
   * @param settings Criterion toggles and parameters.
   */
  TRIBOL_HOST_DEVICE explicit TimestepCalculator( const TimestepSettings& settings ) : settings_( settings ) {}

  /** @brief Return the stored timestep settings. */
  TRIBOL_HOST_DEVICE const TimestepSettings& settings() const { return settings_; }

  /** @brief Return true if any timestep criterion is enabled. */
  TRIBOL_HOST_DEVICE bool anyEnabled() const { return settings_.active_criteria.anyEnabled(); }

  /**
   * @brief Evaluate active point-level penetration criteria (current_gap and/or velocity_projection).
   * @param point Geometric and kinematic state at one contact point.
   * @param current_dt Incoming simulation timestep for velocity projection (<= 0 disables velocity projection).
   * @return Smallest allowable timestep from active point criteria, or +infinity if none vote.
   */
  TRIBOL_HOST_DEVICE RealT evaluatePenetrationCriteria( const TimestepPointData& point, RealT current_dt = 0.0 ) const
  {
    RealT dt = axom::numeric_limits<RealT>::infinity();
    if ( settings_.active_criteria.current_gap ) {
      dt = axom::utilities::min( dt, CurrentGapTimestepCriterion::evaluate( point, settings_ ) );
    }
    if ( settings_.active_criteria.velocity_projection ) {
      dt = axom::utilities::min( dt, VelocityProjectionTimestepCriterion::evaluate( point, settings_, current_dt ) );
    }
    return dt;
  }

  /**
   * @brief Evaluate the pair-local spring-mass CFL criterion when enabled.
   * @param pair Pair-level inverse masses, penalty stiffness per area, and overlap measure.
   * @return Allowable timestep, or +infinity if pair_spring_mass_cfl is disabled or does not vote.
   */
  TRIBOL_HOST_DEVICE RealT evaluatePairSpringMassCfl( const TimestepPairSpringMassData& pair ) const
  {
    if ( !settings_.active_criteria.pair_spring_mass_cfl ) {
      return axom::numeric_limits<RealT>::infinity();
    }
    return PairSpringMassCflTimestepCriterion::evaluate( pair, settings_ );
  }

  /**
   * @brief Accumulate one integration point into nodal penalty CFL row sums when nodal_penalty_cfl is enabled.
   * @param point Integration point data.
   * @param stiffness_row_sums Per-DOF stiffness row-sum accumulator.
   * @param damping_row_sums Optional per-DOF damping row-sum accumulator.
   * @param invalid_mass Single-element error flag.
   */
  TRIBOL_HOST_DEVICE void accumulateNodalPenaltyPoint( const TimestepNodalPenaltyPointData& point,
                                                       ArrayViewT<RealT> stiffness_row_sums,
                                                       ArrayViewT<RealT> damping_row_sums,
                                                       ArrayViewT<int> invalid_mass ) const
  {
    if ( !settings_.active_criteria.nodal_penalty_cfl ) {
      return;
    }
    NodalPenaltyCflTimestepCriterion::accumulatePoint( point, stiffness_row_sums, damping_row_sums, invalid_mass );
  }

  /**
   * @brief Finalize the nodal penalty CFL stability timestep from global stiffness and damping bounds.
   * @param stiffness_bound Conservative upper bound on lambda_max(M^{-1/2} K M^{-1/2}).
   * @param damping_bound Conservative upper bound on lambda_max(M^{-1/2} C M^{-1/2}).
   * @return Allowable timestep, or +infinity if nodal_penalty_cfl is disabled or no stiffness/damping was accumulated.
   */
  TRIBOL_HOST_DEVICE RealT finalizeNodalPenaltyCfl( RealT stiffness_bound, RealT damping_bound ) const
  {
    if ( !settings_.active_criteria.nodal_penalty_cfl ) {
      return axom::numeric_limits<RealT>::infinity();
    }
    return NodalPenaltyCflTimestepCriterion::finalizeFromBounds( stiffness_bound, damping_bound, settings_ );
  }

  /**
   * @brief Collectively validate that all fields required by active criteria are registered on non-empty meshes.
   * @param mesh1 First surface mesh view.
   * @param mesh2 Second surface mesh view.
   * @param use_parent_mass True when the nodal penalty criterion reads parent inverse mass separately.
   * @return True if all required fields are present; false after reporting a SLIC_ERROR listing every missing field.
   */
  bool validateRequiredFields( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
                               bool use_parent_mass = false ) const
  {
    const bool has_elements = mesh1.numberOfElements() > 0 && mesh2.numberOfElements() > 0;
    const auto criteria = settings_.active_criteria;
    const bool needs_nodal_surface_mass = criteria.nodal_penalty_cfl && !use_parent_mass;
    const bool needs_surface_mass = criteria.pair_spring_mass_cfl || needs_nodal_surface_mass;
    const RealT stability_factor = settings_.explicit_integrator_stability_factor;
    const bool invalid_stability_factor =
        criteria.requiresInverseMass() && ( stability_factor <= 0.0 || !std::isfinite( stability_factor ) );
    int missing[4]{
        ( has_elements && isMissingVelocity( mesh1, mesh2 ) ) ? 1 : 0,
        ( has_elements && isMissingElementThickness( mesh1, mesh2 ) ) ? 1 : 0,
        ( has_elements && needs_surface_mass && ( !mesh1.hasInverseMass() || !mesh2.hasInverseMass() ) ) ? 1 : 0,
        invalid_stability_factor ? 1 : 0 };
#ifdef TRIBOL_USE_MPI
    if ( settings_.use_mpi ) {
      MPI_Allreduce( MPI_IN_PLACE, missing, 4, MPI_INT, MPI_MAX, settings_.communicator );
    }
#endif
    if ( missing[0] == 0 && missing[1] == 0 && missing[2] == 0 && missing[3] == 0 ) {
      return true;
    }

    std::string missing_list;
    const auto append_missing = [&]( const char* field_name ) {
      if ( !missing_list.empty() ) {
        missing_list += ", ";
      }
      missing_list += field_name;
    };
    if ( missing[0] != 0 ) {
      append_missing( "nodal velocities" );
    }
    if ( missing[1] != 0 ) {
      append_missing( "element thicknesses" );
    }
    std::string error;
    if ( !missing_list.empty() ) {
      error = "Enabled penetration timestep criteria require " + missing_list +
              ". Register the missing fields on both meshes or disable the corresponding criteria via "
              "setTimestepCriteria().";
    }
    if ( missing[2] != 0 ) {
      if ( !error.empty() ) {
        error += " ";
      }
      error += "Register component-wise inverse diagonal masses on both meshes for ";
      error += cflCriteriaNames( needs_nodal_surface_mass, criteria.pair_spring_mass_cfl );
      error +=
          " via registerNodalInverseMass() or registerMfemInverseMass(), or disable the named criteria via "
          "setTimestepCriteria().";
    }
    if ( missing[3] != 0 ) {
      if ( !error.empty() ) {
        error += " ";
      }
      error +=
          "Set a finite, positive explicit integrator stability factor via setExplicitIntegratorStabilityFactor() "
          "or disable ";
      error += cflCriteriaNames( criteria.nodal_penalty_cfl, criteria.pair_spring_mass_cfl );
      error += " via setTimestepCriteria().";
    }
    SLIC_ERROR( error );
    return false;
  }

  /**
   * @brief Check the parent transfer and inverse mass collectively before assembling a nodal penalty matrix.
   * @param parent_cfl_data MFEM mesh and transfer data, possibly null.
   * @return True if the inputs are ready, or false after reporting a SLIC error with the missing input.
   */
  bool validateParentPenaltyCflData( [[maybe_unused]] const detail::MfemParentPenaltyCflData* parent_cfl_data ) const
  {
#ifdef BUILD_REDECOMP
    int missing[2]{ !parent_cfl_data || !parent_cfl_data->transfer_data.HasParentPath() ||
                            !parent_cfl_data->mesh_data.HasRedecompMesh()
                        ? 1
                        : 0,
                    parent_cfl_data && !parent_cfl_data->mesh_data.HasInverseMass() ? 1 : 0 };
    if ( settings_.use_mpi ) {
      MPI_Allreduce( MPI_IN_PLACE, missing, 2, MPI_INT, MPI_MAX, settings_.communicator );
    }
    if ( missing[0] == 0 && missing[1] == 0 ) {
      return true;
    }
    std::string error;
    if ( missing[0] != 0 ) {
      error +=
          "The parent matrix transfer is missing for nodal_penalty_cfl. Set up the MFEM coupling scheme and call "
          "updateMfemParallelDecomposition(). ";
    }
    if ( missing[1] != 0 ) {
      error += "Parent inverse diagonal mass is missing for nodal_penalty_cfl. Call registerMfemInverseMass(). ";
    }
    SLIC_ERROR( error << "Set nodal_penalty_cfl = false via setTimestepCriteria() to disable this criterion." );
#else
    SLIC_ERROR(
        "nodal_penalty_cfl on MFEM parent true DOFs requires a build with Redecomp. "
        "Set nodal_penalty_cfl = false via setTimestepCriteria() to disable this criterion." );
#endif
    return false;
  }

  /**
   * @brief Report invalid surface CFL data after the flags are reduced across MPI ranks.
   * @param invalid_nodal_mass True if the nodal calculation encountered an invalid inverse mass or DOF index.
   * @param invalid_pair_mass True if the pair spring-mass calculation encountered an invalid inverse mass.
   * @return True if neither criterion reported an error, or false after reporting a SLIC error.
   */
  bool validateSurfaceCflData( bool invalid_nodal_mass, bool invalid_pair_mass ) const
  {
    if ( !invalid_nodal_mass && !invalid_pair_mass ) {
      return true;
    }
    SLIC_ERROR( cflCriteriaNames( invalid_nodal_mass, invalid_pair_mass )
                << " encountered invalid surface CFL data. Check that registered inverse diagonal masses are finite "
                   "and nonnegative"
                << ( invalid_nodal_mass ? " and that the surface DOF indices are valid" : "" )
                << ". Correct the inputs or disable the named criteria via setTimestepCriteria()." );
    return false;
  }

  /**
   * @brief Compute the allowable timestep for a normal kinematic penalty formulation built on PairTractionIntegral.
   *
   * Evaluates each pair's quadrature points via pair_integral, queries
   * TractionFunction::computeTimestepDataAtQuadraturePoint() for local normals, contact activation, and penalty
   * stiffness, and runs all active criteria in TimestepSettings.
   *
   * @tparam Exec Concrete execution mode for the pair and row-sum reduction kernels.
   * @tparam RuleGenerator Pair quadrature rule generator.
   * @tparam TractionFunction Point traction function providing computePairGeometryData() and
   * computeTimestepDataAtQuadraturePoint().
   * @tparam Basis1 Residual basis on geometry1.
   * @tparam Basis2 Residual basis on geometry2.
   * @param pairs Refined candidate pairs in execution-accessible memory.
   * @param mesh1 First surface mesh view.
   * @param mesh2 Second surface mesh view.
   * @param pair_integral Configured pair traction integral supplying the rule generator and traction function.
   * @param current_dt Optional incoming simulation timestep used by velocity-projection voting.
   * @param parent_cfl_data MFEM transfer and parent mass data required when nodal_penalty_cfl is enabled.
   * @return Unclamped global allowable timestep, or std::nullopt when no active criterion votes or inputs are invalid.
   */
  template <ExecutionMode Exec, PairRuleGenerator RuleGenerator, PairTractionFunction<RuleGenerator> TractionFunction,
            FiniteElementBasis Basis1, FiniteElementBasis Basis2>
  std::optional<RealT> compute(
      const ArrayT<InterfacePair>& pairs, const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2,
      const PairTractionIntegral<RuleGenerator, TractionFunction, Basis1, Basis2>& pair_integral,
      std::optional<RealT> current_dt = std::nullopt,
      [[maybe_unused]] const detail::MfemParentPenaltyCflData* parent_cfl_data = nullptr ) const
  {
    if ( !settings_.active_criteria.anyEnabled() ) {
      return std::nullopt;
    }

    const bool use_parent_penalty_cfl = settings_.active_criteria.nodal_penalty_cfl;
    if ( use_parent_penalty_cfl && !validateParentPenaltyCflData( parent_cfl_data ) ) {
      return std::nullopt;
    }
    if ( !validateRequiredFields( mesh1, mesh2, use_parent_penalty_cfl ) ) {
      return std::nullopt;
    }

    using Geometry1 = typename RuleGenerator::Geometry1;
    using Geometry2 = typename RuleGenerator::Geometry2;
    using Coords1 = typename Geometry1::Coords;
    using Coords2 = typename Geometry2::Coords;
    constexpr int spatial_dim = Geometry1::spatial_dim;
    constexpr IndexT size_basis1 = Basis1::size_basis;
    constexpr IndexT size_basis2 = Basis2::size_basis;
    static_assert( Geometry1::spatial_dim == Geometry2::spatial_dim );

    const IndexT num_pairs = pairs.size();
    const int allocator_id = pairs.getAllocatorID();
    const RealT dt_in = ( current_dt.has_value() && *current_dt > 0.0 ) ? *current_dt : 0.0;

    ArrayT<int> invalid_pair_mass_data( { 0 }, allocator_id );
    ArrayViewT<int> invalid_pair_mass = invalid_pair_mass_data.view();

#ifdef BUILD_REDECOMP
    constexpr IndexT size_geom1_dofs = size_basis1 * spatial_dim;
    constexpr IndexT size_geom2_dofs = size_basis2 * spatial_dim;
    constexpr IndexT size_pair_dofs = size_geom1_dofs + size_geom2_dofs;
    using PairPenaltyMatrix = MdArray<RealT, size_pair_dofs, size_pair_dofs>;
    const IndexT num_pair_matrices = use_parent_penalty_cfl ? num_pairs : 0;
    ArrayT<PairPenaltyMatrix> pair_penalty_matrix( axom::ArrayOptions::Uninitialized{}, num_pair_matrices,
                                                   num_pair_matrices, allocator_id );
    const auto pair_penalty_matrix_view = pair_penalty_matrix.view();
#endif

    // RAJA's minimum reduction starts each worker at the largest finite value, including workers with no vote.
    constexpr RealT no_timestep_vote = std::numeric_limits<RealT>::max();
    ReduceMin<Exec, RealT> min_dt( no_timestep_vote );

    const auto pairs_view = pairs.view();
    const auto rule_generator = pair_integral.ruleGenerator();
    const auto traction_function = pair_integral.tractionFunction();
    const TimestepCalculator calculator( settings_ );
    const TimestepCriteria criteria = settings_.active_criteria;

    forAllExec<Exec>( num_pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair_index ) {
      const InterfacePair pair = pairs_view[pair_index];
      const IndexT elem1 = pair.element_id1_;
      const IndexT elem2 = pair.element_id2_;

      const auto coords1 = mesh1.template getFaceCoords<Coords1>( elem1 );
      const auto coords2 = mesh2.template getFaceCoords<Coords2>( elem2 );
      const Geometry1 geometry1( coords1.view() );
      const Geometry2 geometry2( coords2.view() );

      const auto points = rule_generator.generate( geometry1, geometry2 );
#ifdef BUILD_REDECOMP
      if ( use_parent_penalty_cfl ) {
        pair_penalty_matrix_view[pair_index] = PairPenaltyMatrix{};
      }
#endif
      if ( points.metadata.status != QuadratureStatus::Success || points.metadata.num_required_points == 0 ) {
        return;
      }

      const auto pair_geometry_data = traction_function.computePairGeometryData( geometry1, geometry2 );

      RealT vels1[size_basis1 * spatial_dim]{};
      RealT vels2[size_basis2 * spatial_dim]{};
      RealT thickness1 = 0.0;
      RealT thickness2 = 0.0;
      if ( criteria.requiresVelocity() ) {
        mesh1.getFaceVelocities( elem1, vels1 );
        mesh2.getFaceVelocities( elem2, vels2 );
        thickness1 = mesh1.getElementData().m_thickness[elem1];
        thickness2 = mesh2.getElementData().m_thickness[elem2];
      }

      RealT active_overlap_measure = 0.0;
      RealT weighted_penalty_stiffness = 0.0;
      axom::primal::Vector<RealT, 3> weighted_normal{};

      for ( IndexT point = 0; point < points.metadata.num_required_points; ++point ) {
        const MdSpan<const RealT, Geometry1::parametric_dim> parametric_coords1{
            &points.geom1_parametric_coords( point, 0 ), Geometry1::parametric_dim };
        const MdSpan<const RealT, Geometry2::parametric_dim> parametric_coords2{
            &points.geom2_parametric_coords( point, 0 ), Geometry2::parametric_dim };

        const auto point1 = geometry1.position( parametric_coords1 );
        const auto point2 = geometry2.position( parametric_coords2 );
        const QuadraturePointTimestepData qp_data = traction_function.computeTimestepDataAtQuadraturePoint(
            point1, point2, parametric_coords1, parametric_coords2, pair_geometry_data );
        if ( !qp_data.valid ) {
          continue;
        }

        MdArray<RealT, size_basis1> basis1{};
        MdArray<RealT, size_basis2> basis2{};
        Basis1::eval( parametric_coords1, basis1.view() );
        Basis2::eval( parametric_coords2, basis2.view() );

        if ( criteria.requiresVelocity() ) {
          TimestepPointData point_data;
          point_data.contact_normal = qp_data.contact_normal;
          for ( int d = 0; d < spatial_dim; ++d ) {
            point_data.point1[d] = point1[d];
            point_data.point2[d] = point2[d];
            RealT vel1_d = 0.0;
            for ( IndexT basis_index = 0; basis_index < size_basis1; ++basis_index ) {
              vel1_d += basis1[basis_index] * vels1[basis_index * spatial_dim + d];
            }
            RealT vel2_d = 0.0;
            for ( IndexT basis_index = 0; basis_index < size_basis2; ++basis_index ) {
              vel2_d += basis2[basis_index] * vels2[basis_index * spatial_dim + d];
            }
            point_data.vel1[d] = vel1_d;
            point_data.vel2[d] = vel2_d;
          }
          point_data.thickness1 = thickness1;
          point_data.thickness2 = thickness2;
          point_data.in_contact = qp_data.in_contact;

          const RealT dt_point = calculator.evaluatePenetrationCriteria( point_data, dt_in );
          if ( std::isfinite( dt_point ) && dt_point > 0.0 ) {
            min_dt.min( dt_point );
          }
        }

        if ( qp_data.in_contact && points.weights[point] > 0.0 ) {
          active_overlap_measure += points.weights[point];
          weighted_penalty_stiffness += points.weights[point] * qp_data.penalty_stiffness;
          weighted_normal += points.weights[point] * qp_data.contact_normal;
#ifdef BUILD_REDECOMP
          if ( use_parent_penalty_cfl ) {
            // Keep the gap coefficients signed so the MFEM transfer can combine contributions to each parent DOF.
            MdArray<RealT, size_pair_dofs> gap_row{};
            for ( IndexT d = 0; d < spatial_dim; ++d ) {
              for ( IndexT basis_index = 0; basis_index < size_basis1; ++basis_index ) {
                gap_row[basis_index * spatial_dim + d] = basis1[basis_index] * qp_data.contact_normal[d];
              }
              for ( IndexT basis_index = 0; basis_index < size_basis2; ++basis_index ) {
                gap_row[size_geom1_dofs + basis_index * spatial_dim + d] =
                    -basis2[basis_index] * qp_data.contact_normal[d];
              }
            }
            auto& penalty_matrix = pair_penalty_matrix_view[pair_index];
            const RealT weighted_stiffness = points.weights[point] * qp_data.penalty_stiffness;
            for ( IndexT row = 0; row < size_pair_dofs; ++row ) {
              for ( IndexT column = 0; column < size_pair_dofs; ++column ) {
                penalty_matrix( row, column ) += weighted_stiffness * gap_row[row] * gap_row[column];
              }
            }
          }
#endif
        }
      }

      if ( criteria.pair_spring_mass_cfl && active_overlap_measure > 0.0 ) {
        const RealT norm_sq = weighted_normal.squared_norm();
        if ( norm_sq > 0.0 ) {
          weighted_normal /= std::sqrt( norm_sq );
          bool local_invalid_mass = false;
          TimestepPairSpringMassData pair_data;
          pair_data.inv_mass1 =
              computeSurfaceElementNormalInverseMass( mesh1, elem1, weighted_normal, spatial_dim, local_invalid_mass );
          pair_data.inv_mass2 =
              computeSurfaceElementNormalInverseMass( mesh2, elem2, weighted_normal, spatial_dim, local_invalid_mass );
          if ( local_invalid_mass ) {
            tribol::atomicMax( &invalid_pair_mass[0], 1 );
          } else {
            pair_data.penalty_stiffness_per_area = weighted_penalty_stiffness / active_overlap_measure;
            pair_data.overlap_measure = active_overlap_measure;
            pair_data.in_contact = true;
            const RealT dt_pair = calculator.evaluatePairSpringMassCfl( pair_data );
            if ( std::isfinite( dt_pair ) && dt_pair > 0.0 ) {
              min_dt.min( dt_pair );
            }
          }
        }
      }
    } );

    const ArrayT<int, 1, MemorySpace::Host> invalid_pair_mass_host( invalid_pair_mass_data );
    int global_invalid_pair_mass = invalid_pair_mass_host[0];
    RealT global_dt = min_dt.get();

#ifdef TRIBOL_USE_MPI
    if ( settings_.use_mpi ) {
      MPI_Allreduce( MPI_IN_PLACE, &global_invalid_pair_mass, 1, MPI_INT, MPI_MAX, settings_.communicator );
      MPI_Allreduce( MPI_IN_PLACE, &global_dt, 1, MPI_DOUBLE, MPI_MIN, settings_.communicator );
    }
#endif

    if ( !validateSurfaceCflData( false, global_invalid_pair_mass != 0 ) ) {
      return std::nullopt;
    }

#ifdef BUILD_REDECOMP
    if ( use_parent_penalty_cfl ) {
      // The same two-sided transfer as the residual Jacobian gives T^T K_LOR T on parent true DOFs.
      auto assembler = detail::makeResidualJacobianAssembler(
          parent_cfl_data->mesh_data, parent_cfl_data->transfer_data, mesh1, mesh2, num_pairs, allocator_id );
      assembler.template addPairs<Exec, size_geom1_dofs, size_geom2_dofs, 4>(
          pairs, [=] TRIBOL_HOST_DEVICE( IndexT pair_index, IndexT row, IndexT column ) {
            return pair_penalty_matrix_view[pair_index]( row, column );
          } );
      const auto penalty_matrix = assembler.assemble();
      const auto bound = detail::computeParentPenaltyRowSumBound( penalty_matrix.get(), parent_cfl_data->mesh_data );
      if ( !bound ) {
        return std::nullopt;
      }
      const RealT nodal_penalty_dt = finalizeNodalPenaltyCfl( *bound, 0.0 );
      if ( std::isfinite( nodal_penalty_dt ) && nodal_penalty_dt > 0.0 ) {
        global_dt = std::min( global_dt, nodal_penalty_dt );
      }
    }

#endif
    if ( global_dt == no_timestep_vote || !std::isfinite( global_dt ) ) {
      return std::nullopt;
    }
    return global_dt;
  }

 private:
  /**
   * @brief Name the CFL criteria that require a field or reported an error.
   * @param nodal True when nodal_penalty_cfl is affected.
   * @param pair True when pair_spring_mass_cfl is affected.
   * @return Names of the affected criteria. At least one argument must be true.
   */
  static const char* cflCriteriaNames( bool nodal, bool pair )
  {
    if ( nodal && pair ) {
      return "nodal_penalty_cfl and pair_spring_mass_cfl";
    }
    return nodal ? "nodal_penalty_cfl" : "pair_spring_mass_cfl";
  }

  /**
   * @brief Check whether enabled criteria require nodal velocities that are missing on either mesh.
   * @param mesh1 First surface mesh view.
   * @param mesh2 Second surface mesh view.
   * @return True when nodal velocities are required and missing on mesh1 or mesh2.
   */
  bool isMissingVelocity( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const
  {
    return settings_.active_criteria.requiresVelocity() && ( !mesh1.hasVelocity() || !mesh2.hasVelocity() );
  }

  /**
   * @brief Check whether enabled criteria require element thicknesses that are missing on either mesh.
   * @param mesh1 First surface mesh view.
   * @param mesh2 Second surface mesh view.
   * @return True when element thicknesses are required and missing on mesh1 or mesh2.
   */
  bool isMissingElementThickness( const MeshData::Viewer& mesh1, const MeshData::Viewer& mesh2 ) const
  {
    return settings_.active_criteria.requiresElementThickness() &&
           ( !mesh1.getElementData().m_is_element_thickness_set || !mesh2.getElementData().m_is_element_thickness_set );
  }

  TimestepSettings settings_{};  ///< Configured timestep settings and active criterion flags.
};

}  // namespace tribol

#endif  // SRC_TRIBOL_PHYSICS_TIMESTEP_TIMESTEPCALCULATOR_HPP_
