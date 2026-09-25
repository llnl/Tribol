#include "tribol/Tribol.hpp"

#include "../components/ConstraintRowFixture.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

// Requirements: PAR-002, PAR-003, PAR-007, TIME-002

namespace {

using namespace tribol;
using test::TwoRowFixture;

constexpr std::array<Index, 2> first_mortar_map{ 0, 1 };
constexpr std::array<Index, 2> first_nonmortar_map{ 2, 3 };
constexpr std::array<Index, 2> second_mortar_map{ 2, 3 };
constexpr std::array<Index, 2> second_nonmortar_map{ 4, 5 };

FieldView<Real> field( std::vector<Real>& values )
{
  return { { values.data(), static_cast<Index>( values.size() ) },
           static_cast<Index>( values.size() / 2 ),
           2,
           FieldLayout::Interleaved };
}

FieldView<const Real> constField( const std::vector<Real>& values )
{
  return { { values.data(), static_cast<Index>( values.size() ) },
           static_cast<Index>( values.size() / 2 ),
           2,
           FieldLayout::Interleaved };
}

ConstraintRowBatch assemble( TwoRowFixture& first, TwoRowFixture& second, int owner_rank )
{
  const std::array<ConstraintRowSourceView, 3> sources{
      ConstraintRowSourceView{ first.view(), { first_mortar_map.data(), 2 }, { first_nonmortar_map.data(), 2 } },
      ConstraintRowSourceView{ second.view(), { second_mortar_map.data(), 2 }, { second_nonmortar_map.data(), 2 } },
      ConstraintRowSourceView{ first.view(), { first_mortar_map.data(), 2 }, { first_nonmortar_map.data(), 2 } },
  };
  return assembleOwnedConstraintRows( { sources.data(), 3 },
                                      { .dimension = 2, .number_of_parent_dofs = 6, .owner_rank = owner_rank } );
}

ConstraintRowBatch assemblePermuted( TwoRowFixture& first, TwoRowFixture& second, int owner_rank )
{
  const std::array<ConstraintRowSourceView, 3> sources{
      ConstraintRowSourceView{ second.view(), { second_mortar_map.data(), 2 }, { second_nonmortar_map.data(), 2 } },
      ConstraintRowSourceView{ first.view(), { first_mortar_map.data(), 2 }, { first_nonmortar_map.data(), 2 } },
      ConstraintRowSourceView{ first.view(), { first_mortar_map.data(), 2 }, { first_nonmortar_map.data(), 2 } },
  };
  return assembleOwnedConstraintRows( { sources.data(), 3 },
                                      { .dimension = 2, .number_of_parent_dofs = 6, .owner_rank = owner_rank } );
}

std::vector<Real> evaluate( ConstraintRowBatchView rows )
{
  std::vector<Real> force( 12 );
  std::vector<Real> stiffness( static_cast<std::size_t>( rows.numberOfRows() ), 10.0 );
  (void)evaluateExplicitPenaltyRows( rows, { .pressure_stiffness = { stiffness.data(), rows.numberOfRows() } },
                                     { .mortar = field( force ), .nonmortar = field( force ) } );
  return force;
}

bool deterministicAssemblyContract( TwoRowFixture& first, TwoRowFixture& second )
{
  const auto coupled = assemble( first, second, -1 );
  const auto permuted = assemblePermuted( first, second, -1 );
  const auto rows = coupled.view();
  const auto permuted_rows = permuted.view();
  if ( rows.numberOfRows() != 4 || permuted_rows.numberOfRows() != rows.numberOfRows() ||
       rows.contact_pair_ids[0] != 7 || rows.contact_pair_ids[1] != 7 || rows.contact_pair_ids[2] != 8 ||
       rows.contact_pair_ids[3] != 8 || rows.mortar_dofs[0] != 0 || rows.nonmortar_dofs[0] != 2 ||
       rows.mortar_dofs[4] != 2 || rows.nonmortar_dofs[4] != 4 ) {
    return false;
  }
  for ( Index row = 0; row < rows.numberOfRows(); ++row ) {
    if ( rows.contact_pair_ids[row] != permuted_rows.contact_pair_ids[row] ||
         rows.row_ids[row] != permuted_rows.row_ids[row] ) {
      return false;
    }
  }
  const auto force = evaluate( rows );
  const auto permuted_force = evaluate( permuted_rows );
  return std::equal( force.begin(), force.end(), permuted_force.begin(),
                     []( Real left, Real right ) { return std::abs( left - right ) < 1.0e-13; } );
}

bool coupledStabilityContract( TwoRowFixture& first, TwoRowFixture& second )
{
  auto coupled = assemble( first, second, -1 );
  const std::array<ConstraintRowSourceView, 1> first_source{
      ConstraintRowSourceView{ first.view(), { first_mortar_map.data(), 2 }, { first_nonmortar_map.data(), 2 } } };
  auto first_only =
      assembleOwnedConstraintRows( { first_source.data(), 1 }, { .dimension = 2, .number_of_parent_dofs = 6 } );
  std::vector<Real> inverse_mass( 12, 1.0 );
  for ( std::size_t value = 0; value < inverse_mass.size(); value += 2 ) {
    inverse_mass[value] = 0.0;
  }
  std::vector<Real> coupled_stiffness( 4, 5.0 );
  std::vector<Real> first_stiffness( 2, 5.0 );
  std::vector<Real> coupled_workspace( 12 );
  std::vector<Real> first_workspace( 12 );
  const auto inverse_mass_field = constField( inverse_mass );
  const auto integrator = timestep::IntegratorStabilityLimits::centralDifference();
  const auto coupled_result = timestep::explicitPenaltyStabilityVote(
      coupled.view(),
      { .row_stiffness = { coupled_stiffness.data(), 4 },
        .inverse_mass = { .mortar = inverse_mass_field, .nonmortar = inverse_mass_field },
        .integrator = integrator },
      { .mortar_row_sums = field( coupled_workspace ), .nonmortar_row_sums = field( coupled_workspace ) } );
  const auto first_result = timestep::explicitPenaltyStabilityVote(
      first_only.view(),
      { .row_stiffness = { first_stiffness.data(), 2 },
        .inverse_mass = { .mortar = inverse_mass_field, .nonmortar = inverse_mass_field },
        .integrator = integrator },
      { .mortar_row_sums = field( first_workspace ), .nonmortar_row_sums = field( first_workspace ) } );
  return std::abs( coupled_result.stiffness_frequency_squared_bound - 20.0 ) < 1.0e-13 &&
         std::abs( first_result.stiffness_frequency_squared_bound - 10.0 ) < 1.0e-13 &&
         coupled_result.timestep_vote < first_result.timestep_vote;
}

bool distributedOwnershipContract( TwoRowFixture& first, TwoRowFixture& second, int rank, int ranks )
{
  for ( Index row = 0; row < 2; ++row ) {
    first.owner_ranks[row] = static_cast<int>( row ) % ranks;
    second.owner_ranks[row] = static_cast<int>( row ) % ranks;
  }
  const auto reference_rows = assemble( first, second, -1 );
  const auto local_rows = assemble( first, second, rank );
  const auto reference_force = evaluate( reference_rows.view() );
  auto distributed_force = evaluate( local_rows.view() );
  MPI_Allreduce( MPI_IN_PLACE, distributed_force.data(), static_cast<int>( distributed_force.size() ), MPI_DOUBLE,
                 MPI_SUM, MPI_COMM_WORLD );
  long long local_rows_count = local_rows.numberOfRows();
  long long global_rows_count{};
  MPI_Allreduce( &local_rows_count, &global_rows_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
  Real maximum_error{};
  for ( std::size_t value = 0; value < reference_force.size(); ++value ) {
    maximum_error = std::max( maximum_error, std::abs( distributed_force[value] - reference_force[value] ) );
  }
  return global_rows_count == reference_rows.numberOfRows() && maximum_error < 1.0e-13;
}

}  // namespace

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  int rank{};
  int ranks{};
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &ranks );
  TwoRowFixture first;
  TwoRowFixture second;
  second.contact_pair_ids = { 8, 8 };
  second.cell_ids = { 12, 12 };
  second.row_ids = { 30, 31 };
  const bool passed = deterministicAssemblyContract( first, second ) && coupledStabilityContract( first, second ) &&
                      distributedOwnershipContract( first, second, rank, ranks );
  int local_failure = passed ? 0 : 1;
  int global_failure{};
  MPI_Allreduce( &local_failure, &global_failure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
  MPI_Finalize();
  return global_failure;
}
