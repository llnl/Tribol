// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file
 * @brief Compare indexed nodal-field differentiation and assembly with explicit gap/area calculations.
 *
 * Manufactured gap/area values and sparse coordinate Jacobians isolate field-interface costs from search, quadrature,
 * and MFEM mesh transfers. A distributed-operator path is compared against the same explicit reference.
 * Jacobians couple neighboring DOFs through mass/diffusion matrices; a supplied diagonal
 * coordinate curvature exercises the inner Hessian term. Both implementations use Enzyme, the same weighted curvature,
 * and the same sparse matrix operations. Penalty reference inputs are clamped before timing, including Jacobian rows.
 * Numerical agreement is required before timing. Timings include allocations and destruction, use the slowest rank,
 * alternate implementation order, and report medians and ranges without imposing a performance pass/fail threshold.
 */
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "axom/slic/core/SimpleLogger.hpp"
#include "tribol/mesh/MfemNodalEnergy.hpp"
#include "tribol/mesh/GlobalAD.hpp"
#include "tribol/physics/EnergyMortarEnergy.hpp"

namespace {
using namespace tribol;
constexpr RealT area_tolerance = 1.0e-14;  ///< Support threshold used by the contact formulation.
constexpr RealT penalty = 4.0;             ///< Fixed stiffness used by both penalty expressions.
RealT checksum = 0.0;                      ///< Keep calculated energy observable across timed calls.

/** @brief Command-line controls for input size and repeated measurements. */
struct Options {
  int elements_per_side = 64;   ///< Elements along each side of the square scalar-field mesh.
  int iterations = 10;          ///< Calculations per timed sample.
  int repetitions = 7;          ///< Timed samples per implementation and calculation.
  int warmups = 2;              ///< Untimed calculations per implementation before each comparison.
  RealT active_fraction = 0.5;  ///< Fraction of a repeating nodal pattern with negative weighted gap.
};

/** @brief Own manufactured nodal fields and sparse derivatives on one distributed scalar space. */
struct Fields {
  mfem::ParFiniteElementSpace& space;         ///< Borrowed scalar field and coordinate space.
  shared::ParVector gap;                      ///< Raw weighted gap with mixed active and open nodes.
  shared::ParVector area;                     ///< Strictly positive tributary area.
  shared::ParVector clamped_gap;              ///< Weighted gap clamped for explicit penalty assembly.
  shared::ParVector multipliers;              ///< Independent, fixed LM coefficients.
  shared::ParVector direction;                ///< Nonconstant direction used to check Jacobian actions.
  shared::ParSparseMat gap_jacobian;          ///< Prescribed sparse coordinate derivative of weighted gap.
  shared::ParSparseMat area_jacobian;         ///< Prescribed sparse coordinate derivative of area.
  shared::ParSparseMat clamped_gap_jacobian;  ///< Gap derivative with open rows eliminated for the penalty reference.

  /**
   * @brief Populate fields and assemble their derivative matrices outside the timed calculations.
   * @param fes Scalar space, borrowed for the benchmark lifetime.
   * @param active_fraction Fraction of the repeating nodal pattern in compression, in [0, 1].
   */
  Fields( mfem::ParFiniteElementSpace& fes, RealT active_fraction )
      : space( fes ),
        gap( &fes ),
        area( &fes ),
        clamped_gap( &fes ),
        multipliers( &fes ),
        direction( &fes ),
        gap_jacobian( makeJacobian( 1.0, 0.05 ) ),
        area_jacobian( makeJacobian( 0.75, 0.02 ) ),
        clamped_gap_jacobian( std::make_unique<mfem::HypreParMatrix>( gap_jacobian.get() ) )
  {
    // Different mass/diffusion weights keep the two field Jacobians distinct while retaining neighboring-node coupling.
    constexpr int pattern_size = 100;
    mfem::Array<int> open_rows;
    for ( int node = 0; node < gap.size(); ++node ) {
      const auto global_node = space.GetMyTDofOffset() + node;
      const RealT phase = static_cast<RealT>( global_node % pattern_size ) / pattern_size;
      const bool active = phase < active_fraction;
      const RealT magnitude = 0.1 + 0.02 * phase;
      gap[node] = active ? -magnitude : magnitude;
      clamped_gap[node] = active ? gap[node] : 0.0;
      area[node] = 1.0 + 0.1 * phase;
      multipliers[node] = -2.0;
      direction[node] = 1.0 + 0.05 * phase;
      if ( !active ) open_rows.Append( node );
    }
    clamped_gap_jacobian.eliminateRows( open_rows );
  }

  /** @brief Allocate a zero field vector. @return Owned zero vector on the scalar space. */
  shared::ParVector makeVector() const
  {
    shared::ParVector result( &space );
    result.fill( 0.0 );
    return result;
  }

  /** @brief Form a distributed diagonal matrix. @param diagonal Local diagonal entries. @return Owned matrix. */
  shared::ParSparseMat makeDiagonal( const mfem::Vector& diagonal ) const
  {
    return shared::ParSparseMat::diagonalMatrix( space.GetComm(), space.GlobalTrueVSize(), space.GetTrueDofOffsets(),
                                                 diagonal );
  }

  /**
   * @brief Contract prescribed coordinate curvature with gap weights.
   * @param gap_weights First energy partials for weighted gap.
   * @return Owned weighted gap Hessian, with no stored Hessian for each field entry.
   */
  shared::ParSparseMat weightedHessian( const shared::ParVector& gap_weights ) const
  {
    mfem::Vector diagonal( gap_weights.size() );
    for ( int node = 0; node < diagonal.Size(); ++node ) diagonal[node] = gap_curvature_ * gap_weights[node];
    return makeDiagonal( diagonal );
  }

  /**
   * @brief Contract both fields' coordinate curvature in one pass over nodal weights.
   * @param gap_weights Weighted-gap energy partials.
   * @param area_weights Area energy partials.
   * @return Owned weighted gap/area Hessian.
   */
  shared::ParSparseMat weightedHessian( const shared::ParVector& gap_weights,
                                        const shared::ParVector& area_weights ) const
  {
    mfem::Vector diagonal( gap_weights.size() );
    for ( int node = 0; node < diagonal.Size(); ++node )
      diagonal[node] = gap_curvature_ * gap_weights[node] + area_curvature_ * area_weights[node];
    return makeDiagonal( diagonal );
  }

  /** @brief Bind raw fields and their shared curvature operation. @return Binding borrowing this object's data. */
  DifferentiableNodalFields bind() const
  {
    return {
        space, space, { { gap, gap_jacobian }, { area, area_jacobian } }, [this]( const NodalEnergyGradient& weights ) {
          return weights.contains( 1 ) ? weightedHessian( weights.at( 0 ), weights.at( 1 ) )
                                       : weightedHessian( weights.at( 0 ) );
        } };
  }

 private:
  /**
   * @brief Form a sparse Jacobian with neighboring-node coupling.
   * @param mass_weight Coefficient of the scalar mass matrix.
   * @param diffusion_weight Coefficient of the scalar diffusion matrix.
   * @return Owned distributed Jacobian on scalar true DOFs.
   */
  shared::ParSparseMat makeJacobian( RealT mass_weight, RealT diffusion_weight )
  {
    mfem::ConstantCoefficient mass( mass_weight ), diffusion( diffusion_weight );
    mfem::ParBilinearForm form( &space );
    form.AddDomainIntegrator( new mfem::MassIntegrator( mass ) );
    form.AddDomainIntegrator( new mfem::DiffusionIntegrator( diffusion ) );
    form.Assemble();
    form.Finalize();
    return shared::ParSparseMat( form.ParallelAssemble() );
  }
  static constexpr RealT gap_curvature_ = 0.2;   ///< Nonzero gap curvature requiring the inner chain-rule term.
  static constexpr RealT area_curvature_ = 0.1;  ///< Distinct area curvature contributing to penalty assembly.
};

/** @brief Explicit gap/area partials, including a separately allocated pressure output. */
struct ExplicitPartials {
  RealT energy = 0.0;           ///< Global energy.
  shared::ParVector pressure;   ///< Separate nodal pressure copy.
  shared::ParVector gap;        ///< Gap first partial.
  shared::ParVector area;       ///< Area first partial, allocated only for penalty.
  shared::ParVector gap_gap;    ///< Gap second partial, allocated only for penalty.
  shared::ParVector gap_area;   ///< Mixed second partial, allocated only for penalty.
  shared::ParVector area_area;  ///< Area second partial, allocated only for penalty.
};

/**
 * @brief Differentiate the same nodal expression with a fixed two-field state for the explicit reference.
 * @tparam NodeEnergy Bound energy expression, with one or two active state entries.
 */
template <typename NodeEnergy>
struct TwoFieldEnergy {
  static constexpr IndexT dof_count = nodal_integral_count;  ///< Fixed gap/area state width, including for LM.
  NodeEnergy expression;  ///< Expression borrowing the same independent coefficients as the indexed calculation.
  /** @brief Evaluate a node. @param state Weighted gap and area. @return Node energy. */
  TRIBOL_ENZYME_INLINE RealT energy( MdSpan<const RealT, dof_count> state ) const
  {
    return expression.energy( MdSpan<const RealT, NodeEnergy::dof_count>{ state.data(), NodeEnergy::dof_count } );
  }
};

/**
 * @brief Evaluate Enzyme partials directly into explicit gap/area vectors.
 * @tparam UsePenalty Whether area partials and an outer Hessian are needed.
 * @tparam Expression Deferred penalty or LM energy.
 * @param expression Same expression evaluated by the generic field path.
 * @param fields Supplied nodal values and scalar space.
 * @return Owned explicit partials and globally summed energy.
 */
template <bool UsePenalty, typename Expression>
ExplicitPartials evaluateExplicit( const Expression& expression, const Fields& fields )
{
  ExplicitPartials result;
  result.pressure = fields.makeVector();
  result.gap = fields.makeVector();
  if constexpr ( UsePenalty ) {
    result.area = fields.makeVector();
    result.gap_gap = fields.makeVector();
    result.gap_area = fields.makeVector();
    result.area_area = fields.makeVector();
  }
  RealT local_energy = 0.0;
  for ( int node = 0; node < fields.gap.size(); ++node ) {
    const MdArray<RealT, nodal_integral_count> state{ UsePenalty ? fields.clamped_gap[node] : fields.gap[node],
                                                      fields.area[node] };
    const auto node_energy = expression.atNode( node, area_tolerance );
    const TwoFieldEnergy definition{ node_energy };
    const auto evaluated = EnergyPairEvaluator<decltype( definition )>::evaluate( state.view(), definition );
    local_energy += evaluated.energy;
    result.pressure[node] = result.gap[node] = evaluated.residual[0];
    if constexpr ( UsePenalty ) {
      result.area[node] = evaluated.residual[1];
      result.gap_gap[node] = evaluated.jacobian( 0, 0 );
      result.gap_area[node] = evaluated.jacobian( 0, 1 );
      result.area_area[node] = evaluated.jacobian( 1, 1 );
    }
  }
  MPI_Allreduce( &local_energy, &result.energy, 1, MPI_DOUBLE, MPI_SUM, fields.space.GetComm() );
  return result;
}

/** @brief Owned coordinate outputs used for numerical checks and timing. */
struct CoordinateResult {
  RealT energy;                   ///< Global energy.
  shared::ParVector residual;     ///< Coordinate residual.
  shared::ParSparseMat jacobian;  ///< Coordinate Jacobian.
};

/**
 * @brief Assemble the explicit gap/area coordinate chain rule.
 * @tparam UsePenalty Whether area partials and outer Hessian products are present.
 * @param fields Supplied field Jacobians and weighted curvature.
 * @param partials Explicit outer energy partials.
 * @return Owned coordinate outputs.
 */
template <bool UsePenalty>
CoordinateResult assembleExplicit( const Fields& fields, const ExplicitPartials& partials )
{
  const auto& gap_jacobian = UsePenalty ? fields.clamped_gap_jacobian : fields.gap_jacobian;
  auto residual = partials.gap * gap_jacobian;
  auto jacobian = [&] {
    if constexpr ( UsePenalty )
      return fields.weightedHessian( partials.gap, partials.area );
    else
      return fields.weightedHessian( partials.gap );
  }();
  if constexpr ( UsePenalty ) {
    residual += partials.area * fields.area_jacobian;
    const auto gg = fields.makeDiagonal( partials.gap_gap.get() );
    const auto ga = fields.makeDiagonal( partials.gap_area.get() );
    const auto aa = fields.makeDiagonal( partials.area_area.get() );
    jacobian += shared::ParSparseMat::rap( gg, gap_jacobian );
    jacobian += shared::ParSparseMat::rap( gap_jacobian, ga, fields.area_jacobian );
    jacobian += shared::ParSparseMat::rap( fields.area_jacobian, ga, gap_jacobian );
    jacobian += shared::ParSparseMat::rap( aa, fields.area_jacobian );
  }
  return { partials.energy, std::move( residual ), std::move( jacobian ) };
}

/**
 * @brief Require agreement of energy, residual, and Jacobian action before measuring performance.
 * @param explicit_result Outputs from the explicit reference.
 * @param generic_result Outputs from indexed fields.
 * @param fields Scalar-space metadata and a nonconstant coordinate direction.
 * @throws std::runtime_error A relative difference exceeds the numerical tolerance.
 */
void checkAgreement( const CoordinateResult& explicit_result, const CoordinateResult& generic_result,
                     const Fields& fields )
{
  constexpr RealT tolerance = 1.0e-10;
  const auto check_vector = [&]( const shared::ParVector& reference, const shared::ParVector& actual ) {
    const auto difference = reference - actual;
    return std::sqrt( difference.dot( difference ) ) <=
           tolerance * std::max( 1.0, std::sqrt( reference.dot( reference ) ) );
  };
  const auto reference_action = explicit_result.jacobian * fields.direction;
  const auto generic_action = generic_result.jacobian * fields.direction;
  if ( std::abs( explicit_result.energy - generic_result.energy ) >
           tolerance * std::max( 1.0, std::abs( explicit_result.energy ) ) ||
       !check_vector( explicit_result.residual, generic_result.residual ) ||
       !check_vector( reference_action, generic_action ) ) {
    throw std::runtime_error( "Explicit and generic nodal calculations disagree; timings were not collected." );
  }
}

/**
 * @brief Time a block of calculations and report its slowest rank's average duration.
 * @tparam Function Timed calculation; owns and destroys each result within the call.
 * @param function Calculation to repeat.
 * @param iterations Number of calls in the timed block.
 * @return Maximum elapsed seconds across ranks, divided by iteration count; synchronization is outside timing.
 */
template <typename Function>
RealT measure( const Function& function, int iterations )
{
  MPI_Barrier( MPI_COMM_WORLD );
  const auto start = std::chrono::steady_clock::now();
  for ( int iteration = 0; iteration < iterations; ++iteration ) function();
  const double elapsed = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();
  double maximum = 0.0;
  MPI_Allreduce( &elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
  return maximum / iterations;
}

/**
 * @brief Print median and range for both implementations, alternating their timing order to reduce order effects.
 * @tparam Explicit Explicit calculation callable.
 * @tparam Generic Indexed-field calculation callable.
 * @param method Penalty or LM label.
 * @param calculation Partials, chain_rule, or combined label.
 * @param options Measurement controls.
 * @param fields Field-space metadata used to label results.
 * @param explicit_calculation Explicit calculation.
 * @param generic_calculation Indexed-field calculation.
 */
template <typename Explicit, typename Generic>
void compare( const char* method, const char* calculation, const Options& options, const Fields& fields,
              const Explicit& explicit_calculation, const Generic& generic_calculation )
{
  for ( int warmup = 0; warmup < options.warmups; ++warmup ) {
    explicit_calculation();
    generic_calculation();
  }
  std::vector<RealT> explicit_times( options.repetitions ), generic_times( options.repetitions );
  for ( int repetition = 0; repetition < options.repetitions; ++repetition ) {
    if ( repetition % 2 == 0 ) {
      explicit_times[repetition] = measure( explicit_calculation, options.iterations );
      generic_times[repetition] = measure( generic_calculation, options.iterations );
    } else {
      generic_times[repetition] = measure( generic_calculation, options.iterations );
      explicit_times[repetition] = measure( explicit_calculation, options.iterations );
    }
  }
  std::sort( explicit_times.begin(), explicit_times.end() );
  std::sort( generic_times.begin(), generic_times.end() );
  int rank = 0, ranks = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &ranks );
  if ( rank == 0 ) {
    constexpr RealT microseconds_per_second = 1.0e6;
    const auto median = options.repetitions / 2;
    std::cout << method << ',' << calculation << ',' << fields.space.GlobalTrueVSize() << ',' << ranks << ','
              << options.active_fraction << ',' << options.iterations << ',' << options.repetitions << ','
              << microseconds_per_second * explicit_times[median] << ','
              << microseconds_per_second * generic_times[median] << ','
              << generic_times[median] / explicit_times[median] << ','
              << microseconds_per_second * explicit_times.front() << ','
              << microseconds_per_second * explicit_times.back() << ','
              << microseconds_per_second * generic_times.front() << ','
              << microseconds_per_second * generic_times.back() << '\n';
  }
}

/**
 * @brief Check and time outer partials, sparse chain rules, and their combination for one energy.
 * @tparam UsePenalty Whether the energy has area derivatives and an outer Hessian.
 * @tparam Expression Deferred nodal energy.
 * @param method Method label.
 * @param energy Same expression for both calculations.
 * @param fields Supplied fields and derivative operations.
 * @param options Measurement controls.
 */
template <bool UsePenalty, typename Expression>
void benchmarkEnergy( const char* method, const Expression& energy, const Fields& fields, const Options& options )
{
  const auto bound = fields.bind();
  const auto explicit_partials = evaluateExplicit<UsePenalty>( energy, fields );
  const auto generic_partials = evaluateNodalEnergy( energy, bound, area_tolerance );
  const auto assemble_generic = [&]( const DifferentiableNodalFields& inputs, const NodalEnergyDerivatives& partials ) {
    return CoordinateResult{ partials.energy, inputs.assembleForce( partials ), inputs.assembleJacobian( partials ) };
  };
  checkAgreement( assembleExplicit<UsePenalty>( fields, explicit_partials ),
                  assemble_generic( bound, generic_partials ), fields );
  const auto global_calculation = [&] {
    const auto inputs = fields.bind();
    const auto variables = makeADFields( inputs );
    const auto energy = [&] {
      if constexpr ( UsePenalty )
        return EnergyMortarNodal{}.penaltyEnergy( variables[0], variables[1], penalty, area_tolerance );
      else {
        const auto pressure = ADVector::constant( fields.space, fields.space, fields.multipliers );
        return EnergyMortarNodal{}.lagrangeMultiplierEnergy( variables[0], pressure );
      }
    }();
    auto result = energy.differentiate();
    return CoordinateResult{ result.value, std::move( result.gradient ), std::move( result.hessian ) };
  };
  checkAgreement( assembleExplicit<UsePenalty>( fields, explicit_partials ), global_calculation(), fields );
  compare(
      method, "partials", options, fields, [&] { checksum += evaluateExplicit<UsePenalty>( energy, fields ).energy; },
      [&] { checksum += evaluateNodalEnergy( energy, bound, area_tolerance ).energy; } );
  compare(
      method, "chain_rule", options, fields,
      [&] { checksum += assembleExplicit<UsePenalty>( fields, explicit_partials ).energy; },
      [&] { checksum += assemble_generic( bound, generic_partials ).energy; } );
  compare(
      method, "combined", options, fields,
      [&] {
        checksum += assembleExplicit<UsePenalty>( fields, evaluateExplicit<UsePenalty>( energy, fields ) ).energy;
      },
      [&] {
        const auto inputs = fields.bind();
        checksum += assemble_generic( inputs, evaluateNodalEnergy( energy, inputs, area_tolerance ) ).energy;
      } );
  compare(
      method, "global_combined", options, fields,
      [&] {
        checksum += assembleExplicit<UsePenalty>( fields, evaluateExplicit<UsePenalty>( energy, fields ) ).energy;
      },
      [&] { checksum += global_calculation().energy; } );
}
}  // namespace

/**
 * @brief Run the host nodal-field benchmark with MPI initialized.
 * @param argc Argument count.
 * @param argv Arguments; --help prints input-size and measurement controls.
 * @return Zero on success or help, nonzero for invalid arguments or numerical disagreement.
 */
int main( int argc, char** argv )
{
  MPI_Init( &argc, &argv );
  axom::slic::SimpleLogger logger;
  int rank = 0, ranks = 1;
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &ranks );
  int result = 0;
  {
    Options options;
    mfem::OptionsParser parser( argc, argv );
    parser.AddOption( &options.elements_per_side, "-n", "--elements-per-side",
                      "Elements along each side of the square mesh." );
    parser.AddOption( &options.iterations, "-i", "--iterations", "Calculations per timed sample." );
    parser.AddOption( &options.repetitions, "-r", "--repetitions", "Odd number of timed samples per implementation." );
    parser.AddOption( &options.warmups, "-w", "--warmups", "Untimed calculations before each comparison." );
    parser.AddOption( &options.active_fraction, "-a", "--active-fraction",
                      "Fraction of the nodal pattern in compression [0, 1]." );
    parser.Parse();
    if ( !parser.Good() ) {
      if ( rank == 0 ) parser.PrintUsage( std::cerr );
      result = parser.Help() ? 0 : 1;
    } else if ( options.elements_per_side <= 0 ||
                options.elements_per_side > std::sqrt( std::numeric_limits<int>::max() ) - 1.0 ||
                options.elements_per_side * options.elements_per_side < ranks || options.iterations <= 0 ||
                options.repetitions <= 0 || options.repetitions % 2 == 0 || options.warmups < 0 ||
                !std::isfinite( options.active_fraction ) || options.active_fraction < 0.0 ||
                options.active_fraction > 1.0 ) {
      if ( rank == 0 )
        std::cerr << "Use positive mesh/iteration counts, an odd positive repetition count, nonnegative "
                     "warmups, and an active fraction in [0, 1]. The mesh needs at least one element per rank.\n";
      result = 1;
    } else {
      try {
        auto serial = mfem::Mesh::MakeCartesian2D( options.elements_per_side, options.elements_per_side,
                                                   mfem::Element::QUADRILATERAL, true, 1.0, 1.0 );
        mfem::ParMesh mesh( MPI_COMM_WORLD, serial );
        mfem::H1_FECollection basis( 1, mesh.Dimension() );
        mfem::ParFiniteElementSpace space( &mesh, &basis );
        const Fields fields( space, options.active_fraction );
        if ( rank == 0 ) {
          std::cout << std::setprecision( 8 )
                    << "method,calculation,global_dofs,ranks,active_fraction,iterations,repetitions,explicit_median_us,"
                       "generic_median_us,generic_over_explicit,explicit_min_us,explicit_max_us,generic_min_us,generic_"
                       "max_us\n";
        }
        benchmarkEnergy<true>( "penalty", EnergyMortarNodal{}.penaltyEnergy( nodal_weighted_gap, nodal_area, penalty ),
                               fields, options );
        const IndependentNodalField pressure{ { fields.multipliers.get().GetData(), fields.multipliers.size() } };
        benchmarkEnergy<false>( "lm", EnergyMortarNodal{}.lagrangeMultiplierEnergy( nodal_weighted_gap, pressure ),
                                fields, options );
        if ( rank == 0 ) std::cerr << "Numerical agreement passed; output checksum: " << checksum << '\n';
      } catch ( const std::exception& error ) {
        if ( rank == 0 ) std::cerr << error.what() << '\n';
        result = 1;
      }
    }
  }
  MPI_Finalize();
  return result;
}
